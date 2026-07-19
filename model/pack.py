"""Series-string battery pack: 6 modules x 8 cells = 48s1p.

All 48 cells carry the same string current (plus any per-cell auxiliary
current injected later by the active balancer or fault models). Pack
voltage is the sum of cell terminal voltages; usable pack capacity is
set by whichever cell hits a voltage limit first — which is exactly why
imbalance matters and Phase 4's balancer exists.

Manufacturing variation: every cell gets per-parameter multipliers drawn
from a seeded RNG (normal, clipped at +/-3 sigma; lognormal for strictly
positive skewed quantities like self-discharge and aging rate), so packs
are reproducible per seed and cells drift apart naturally over cycling:

- capacity spread     -> different SOC swing per cycle, weakest cell
                         clips the usable pack capacity
- R0 spread           -> different sag, different I2R heat (Phase 5)
- coulombic-efficiency and self-discharge spread
                      -> monotonic SOC divergence cycle over cycle
- aging_accel spread  -> differential SOH fade over the pack's life

Geometry: cells are arranged in a 6 x 8 grid (module = row, position =
column) — the adjacency map Phase 5's thermal conduction model uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.cell import CellArray, CellParams


@dataclass
class PackConfig:
    n_modules: int = 6
    cells_per_module: int = 8
    seed: int = 2026
    # 1-sigma fractional manufacturing spreads
    sigma_capacity: float = 0.015
    sigma_r0: float = 0.04
    sigma_rc: float = 0.06        # R1, C1, R2, C2
    sigma_m_hyst: float = 0.05
    sigma_eta_loss: float = 0.30  # spread of the charge *loss* (1 - eta)
    sigma_self_discharge: float = 0.50  # lognormal sigma
    sigma_aging_accel: float = 0.08     # lognormal sigma

    @property
    def n_cells(self) -> int:
        return self.n_modules * self.cells_per_module


class Pack:
    def __init__(
        self,
        config: PackConfig | None = None,
        params: CellParams | None = None,
        soc0: float | np.ndarray = 1.0,
    ):
        self.config = config if config is not None else PackConfig()
        cfg = self.config
        n = cfg.n_cells
        rng = np.random.default_rng(cfg.seed)

        def spread(sigma: float) -> np.ndarray:
            return 1.0 + np.clip(rng.normal(0.0, sigma, n), -3.0 * sigma, 3.0 * sigma)

        # Coulombic efficiency: vary the loss (1 - eta), not eta itself, so
        # no cell ends up with eta >= 1
        base = params if params is not None else CellParams()
        eta_loss = (1.0 - base.eta_chg) * np.exp(
            rng.normal(0.0, cfg.sigma_eta_loss, n)
        )
        eta_mult = (1.0 - eta_loss) / base.eta_chg

        multipliers = {
            "q_nom_ah": spread(cfg.sigma_capacity),
            "r0": spread(cfg.sigma_r0),
            "r1": spread(cfg.sigma_rc),
            "c1": spread(cfg.sigma_rc),
            "r2": spread(cfg.sigma_rc),
            "c2": spread(cfg.sigma_rc),
            "m_hyst": spread(cfg.sigma_m_hyst),
            "eta_chg": eta_mult,
            "self_discharge_a": np.exp(rng.normal(0.0, cfg.sigma_self_discharge, n)),
        }
        self.cells = CellArray(n, params=params, multipliers=multipliers, soc0=soc0)
        self.cells.aging_accel = np.exp(rng.normal(0.0, cfg.sigma_aging_accel, n))

        self.module_of = np.repeat(np.arange(cfg.n_modules), cfg.cells_per_module)
        self.last_v = self.cells.terminal_voltage(0.0)
        self.last_current_a = 0.0

    # ------------------------------------------------------------- topology

    @property
    def n_cells(self) -> int:
        return self.config.n_cells

    def module_slice(self, m: int) -> slice:
        cpm = self.config.cells_per_module
        return slice(m * cpm, (m + 1) * cpm)

    def grid_position(self, cell: int) -> tuple[int, int]:
        """(row, col) of a cell in the physical 6 x 8 layout (row = module)."""
        cpm = self.config.cells_per_module
        return divmod(cell, cpm)

    # ------------------------------------------------------------- dynamics

    def step(
        self,
        pack_current_a: float,
        dt_s: float,
        temp_c: float | np.ndarray = 25.0,
        aux_current_a: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance the pack one step. pack_current_a > 0 discharges (series
        current through every cell); aux_current_a is an optional per-cell
        add-on used by the active balancer and fault models. Returns and
        caches per-cell terminal voltages."""
        i = np.full(self.n_cells, float(pack_current_a))
        if aux_current_a is not None:
            i = i + aux_current_a
        self.last_v = self.cells.step(i, dt_s, temp_c)
        self.last_current_a = float(pack_current_a)
        return self.last_v

    # ----------------------------------------------------------- convenience

    @property
    def pack_voltage(self) -> float:
        return float(np.sum(self.last_v))

    @property
    def v_min_cell(self) -> float:
        return float(np.min(self.last_v))

    @property
    def v_max_cell(self) -> float:
        return float(np.max(self.last_v))

    @property
    def soc_spread(self) -> float:
        return float(np.max(self.cells.soc) - np.min(self.cells.soc))
