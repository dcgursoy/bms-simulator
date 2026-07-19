"""Second-order Thevenin equivalent-circuit cell model.

Electrical structure per cell::

    OCV(soc) + h  --[R0]--+--[ R1 || C1 ]--+--[ R2 || C2 ]--o  terminal
                            fast (~30 s)      slow (~600 s)

Sign convention: current i > 0 discharges the cell. Terminal voltage:

    v_t = OCV(soc) + h - i*R0 - v_rc1 - v_rc2

Dynamic state per cell: soc, the two RC branch voltages, the hysteresis
voltage h, and cumulative Ah throughput (which drives aging).

Physics captured beyond the basic 2-RC ladder:

- Temperature: all resistances follow an Arrhenius law (a cold cell is
  more resistive), and usable capacity shrinks below the reference
  temperature. Cell temperature is an *input* here; Phase 5's thermal
  model closes the electro-thermal loop by computing it from losses.
- Hysteresis: Plett's one-state model. h relaxes toward -M while
  discharging and +M while charging at a rate proportional to |i|/Q,
  and holds its value at rest — so the OCV seen after a discharge sits
  below the OCV seen after a charge, as in real graphite cells.
- Aging: capacity fades and R0 grows as power laws in equivalent full
  cycles (EFC = Ah throughput / 2*Q_nom). A per-cell ``aging_accel``
  multiplier lets fault scenarios age one cell faster than its
  neighbors (Phase 6's accelerated-degradation fault).

The implementation is vectorized over n cells (``CellArray``) because
the pack model, the estimator benchmarks, and the live dashboard all
step many cells at once; a single physical cell is just CellArray(n=1).
All per-step math is exact zero-order-hold discretization (closed-form
solution for constant current over the step), so it is stable and
accurate for any step size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.ocv import ocv as _ocv

#: CellParams fields that become per-cell arrays in CellArray, so that
#: manufacturing variation and differential aging can act cell-by-cell.
PER_CELL_PARAMS = (
    "q_nom_ah", "r0", "r1", "c1", "r2", "c2", "m_hyst",
    "eta_chg", "self_discharge_a",
)


@dataclass
class CellParams:
    """Nominal fresh-cell parameters for a ~2.5 Ah graphite/NMC 18650 at 25 degC."""

    q_nom_ah: float = 2.5       # nameplate capacity [Ah]
    r0: float = 0.025           # ohmic (instantaneous) resistance [ohm]
    r1: float = 0.015           # fast RC branch resistance [ohm]
    c1: float = 2.0e3           # fast RC branch capacitance [F]   -> tau1 ~ 30 s
    r2: float = 0.010           # slow RC branch resistance [ohm]
    c2: float = 6.0e4           # slow RC branch capacitance [F]   -> tau2 ~ 600 s
    m_hyst: float = 0.012       # hysteresis voltage magnitude M [V]
    gamma_hyst: float = 50.0    # hysteresis convergence rate [1 / unit SOC]
    eta_chg: float = 0.995      # coulombic efficiency while charging
    self_discharge_a: float = 70e-6  # internal leakage current [A] (~2%/month)
    v_max: float = 4.2          # charge voltage limit [V]
    v_min: float = 3.0          # discharge cutoff voltage [V]
    t_ref_c: float = 25.0       # reference temperature [degC]
    docv_dt_v_per_k: float = -0.15e-3  # entropy coefficient dOCV/dT [V/K]
    ea_over_r: float = 3000.0   # Arrhenius activation Ea/R for resistances [K]
    cap_temp_coeff: float = 0.006    # fractional capacity loss per degC below ref
    fade_at_1000_efc: float = 0.20   # fractional capacity fade after 1000 EFC
    fade_exponent: float = 0.8       # power-law exponent of fade vs EFC
    r_growth_at_1000_efc: float = 0.35  # fractional R0 growth after 1000 EFC


class CellArray:
    """A bank of n independent cells stepped together (vectorized over n).

    ``multipliers`` maps a PER_CELL_PARAMS field name to an (n,) array of
    per-cell scale factors — the hook Phase 2 uses for manufacturing
    variation without touching the nominal parameter set.
    """

    def __init__(
        self,
        n: int,
        params: CellParams | None = None,
        multipliers: dict[str, np.ndarray] | None = None,
        soc0: float | np.ndarray = 1.0,
    ):
        self.n = int(n)
        self.p = params if params is not None else CellParams()

        mult = multipliers or {}
        unknown = set(mult) - set(PER_CELL_PARAMS)
        if unknown:
            raise ValueError(f"multipliers for non-per-cell params: {sorted(unknown)}")
        for name in PER_CELL_PARAMS:
            base = np.full(self.n, getattr(self.p, name), dtype=float)
            setattr(self, name, base * np.asarray(mult.get(name, 1.0), dtype=float))

        # Dynamic state
        self.soc = np.broadcast_to(np.asarray(soc0, dtype=float), (self.n,)).copy()
        self.v_rc1 = np.zeros(self.n)
        self.v_rc2 = np.zeros(self.n)
        self.hyst = np.zeros(self.n)
        self.ah_throughput = np.zeros(self.n)
        self.aging_accel = np.ones(self.n)

    # ------------------------------------------------------------------ aging

    @property
    def efc(self) -> np.ndarray:
        """Equivalent full cycles: one EFC = one full discharge + full charge."""
        return self.ah_throughput / (2.0 * self.q_nom_ah)

    def fade_fraction(self) -> np.ndarray:
        """Fraction of nameplate capacity lost to cycling (capped at 40%)."""
        efc_eff = self.efc * self.aging_accel
        fade = self.p.fade_at_1000_efc * (efc_eff / 1000.0) ** self.p.fade_exponent
        return np.minimum(fade, 0.40)

    def r_aging_factor(self) -> np.ndarray:
        """Multiplier on R0 from cycling-induced impedance growth."""
        efc_eff = self.efc * self.aging_accel
        return 1.0 + self.p.r_growth_at_1000_efc * (efc_eff / 1000.0)

    def fast_forward_aging(self, efc: float | np.ndarray) -> None:
        """Jump the aging state to a given EFC count without simulating
        the cycles (scenario setup and validation)."""
        efc = np.broadcast_to(np.asarray(efc, dtype=float), (self.n,))
        self.ah_throughput = efc * 2.0 * self.q_nom_ah

    # ------------------------------------------------------- effective params

    def _arrhenius(self, temp_c: np.ndarray) -> np.ndarray:
        t_k = np.asarray(temp_c, dtype=float) + 273.15
        t_ref_k = self.p.t_ref_c + 273.15
        return np.exp(self.p.ea_over_r * (1.0 / t_k - 1.0 / t_ref_k))

    def capacity_ah(self, temp_c: float | np.ndarray | None = None) -> np.ndarray:
        """Effective capacity [Ah]: nameplate reduced by fade and by cold."""
        temp_c = self.p.t_ref_c if temp_c is None else temp_c
        cold_loss = self.p.cap_temp_coeff * np.maximum(
            self.p.t_ref_c - np.asarray(temp_c, dtype=float), 0.0
        )
        cold_factor = np.clip(1.0 - cold_loss, 0.6, 1.0)
        return self.q_nom_ah * (1.0 - self.fade_fraction()) * cold_factor

    def resistances(
        self, temp_c: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Effective (temperature- and age-adjusted) R0, R1, R2 [ohm]."""
        arrh = self._arrhenius(temp_c)
        return (
            self.r0 * arrh * self.r_aging_factor(),
            self.r1 * arrh,
            self.r2 * arrh,
        )

    # -------------------------------------------------------------- dynamics

    def terminal_voltage(
        self, current_a: float | np.ndarray, temp_c: float | np.ndarray = 25.0
    ) -> np.ndarray:
        """Terminal voltage [V] at the current state under the given load,
        without advancing the state (the estimators' measurement model)."""
        i = np.broadcast_to(np.asarray(current_a, dtype=float), (self.n,))
        r0_eff, _, _ = self.resistances(temp_c)
        return _ocv(self.soc) + self.hyst - i * r0_eff - self.v_rc1 - self.v_rc2

    def step(
        self,
        current_a: float | np.ndarray,
        dt_s: float,
        temp_c: float | np.ndarray = 25.0,
    ) -> np.ndarray:
        """Advance every cell by dt_s under per-cell current [A] (positive =
        discharge) and per-cell temperature [degC]. Returns the terminal
        voltage [V] at the end of the step."""
        i = np.broadcast_to(np.asarray(current_a, dtype=float), (self.n,)).astype(float)
        temp_c = np.broadcast_to(np.asarray(temp_c, dtype=float), (self.n,))

        r0_eff, r1_eff, r2_eff = self.resistances(temp_c)
        q_as = self.capacity_ah(temp_c) * 3600.0

        # RC branches: exact response to constant current over the step
        a1 = np.exp(-dt_s / (r1_eff * self.c1))
        a2 = np.exp(-dt_s / (r2_eff * self.c2))
        self.v_rc1 = a1 * self.v_rc1 + r1_eff * (1.0 - a1) * i
        self.v_rc2 = a2 * self.v_rc2 + r2_eff * (1.0 - a2) * i

        # Coulomb counting; charging pays the coulombic-efficiency tax and
        # internal leakage always drains (per-cell differences in both are
        # what makes series strings drift out of balance over cycles)
        eta = np.where(i < 0.0, self.eta_chg, 1.0)
        self.soc = np.clip(
            self.soc - (eta * i + self.self_discharge_a) * dt_s / q_as, 0.0, 1.0
        )

        # Hysteresis: relax toward -M (discharge) / +M (charge); f = 1 at
        # rest, so h holds between pulses like a real cell
        f = np.exp(-np.abs(i) * self.p.gamma_hyst * dt_s / q_as)
        self.hyst = f * self.hyst - (1.0 - f) * np.sign(i) * self.m_hyst

        self.ah_throughput = self.ah_throughput + np.abs(i) * dt_s / 3600.0

        return _ocv(self.soc) + self.hyst - i * r0_eff - self.v_rc1 - self.v_rc2

    def heat_generation_w(
        self, current_a: float | np.ndarray, temp_c: float | np.ndarray = 25.0
    ) -> np.ndarray:
        """Per-cell heat generation [W] at the current state: irreversible
        I2R dissipation in all three resistances plus reversible entropic
        heat -i*T*(dOCV/dT) (exothermic on discharge, endothermic on
        charge, with the negative entropy coefficient)."""
        i = np.broadcast_to(np.asarray(current_a, dtype=float), (self.n,))
        temp_c = np.asarray(temp_c, dtype=float)
        r0_eff, r1_eff, r2_eff = self.resistances(temp_c)
        q_irr = i**2 * r0_eff + self.v_rc1**2 / r1_eff + self.v_rc2**2 / r2_eff
        q_rev = -i * (temp_c + 273.15) * self.p.docv_dt_v_per_k
        return q_irr + q_rev

    # --------------------------------------------------------------- logging

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copies of the dynamic state (ground-truth logging)."""
        return {
            "soc": self.soc.copy(),
            "v_rc1": self.v_rc1.copy(),
            "v_rc2": self.v_rc2.copy(),
            "hyst": self.hyst.copy(),
            "ah_throughput": self.ah_throughput.copy(),
            "efc": self.efc.copy(),
        }
