"""Pack-level estimator: runs a bank of per-cell filters on telemetry.

This is the piece that lives on the (simulated) pack controller. It sees
exactly what a real one would:

- the pack current sensor at tick rate (10 Hz),
- per-cell voltage reports arriving asynchronously over the
  bandwidth-limited bus (each up to a full round-robin period stale),
- module NTC temperatures (no per-cell temperature).

It never touches ground truth. The filter bank runs at a 1 Hz cadence
(a realistic BMS MCU budget): between filter steps the controller only
accumulates current samples; at each step it predicts with the mean
current, then applies measurement updates for exactly those cells whose
telemetry timestamp advanced since the last step. Because a report is
sampled at bus-grant time, the update pairs each reported voltage with
the current *at its sample time*, reconstructed from a short ring buffer
of current-sensor readings — pairing a stale voltage with the live
current would inject i(t) - i(t_sample) transients straight into the R0
estimate.

Hysteresis is not estimated: it is propagated open-loop from the
measured current with nominal parameters and fed to the measurement
model, removing an otherwise ~12 mV (≈2% SOC on the OCV plateau) bias.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from estimation.filters import FilterTuning, make_filter_bank
from model.cell import CellParams
from model.comms import Telemetry


class PackEstimator:
    def __init__(
        self,
        n_cells: int,
        module_of: np.ndarray,
        kind: str = "ukf",
        soc0: float = 0.5,
        params: CellParams | None = None,
        tuning: FilterTuning | None = None,
        dt_filter_s: float = 1.0,
    ):
        self.n = n_cells
        self.module_of = module_of
        self.kind = kind
        self.p = params if params is not None else CellParams()
        self.dt_filter = dt_filter_s

        x0 = np.tile(
            [soc0, 0.0, 0.0, self.p.r0, self.p.q_nom_ah], (n_cells, 1)
        )
        self.bank = make_filter_bank(kind, n_cells, x0, self.p, tuning)

        self.h = np.zeros(n_cells)  # open-loop hysteresis tracker
        self._last_seen = np.full(n_cells, -np.inf)
        self._i_hist: deque[tuple[float, float]] = deque(maxlen=40)
        self._i_sum = 0.0
        self._aux_sum = np.zeros(n_cells)
        self._i_n = 0
        self._t_next: float | None = None
        self.compute_s = 0.0  # cumulative filter wall time
        self.n_steps = 0

    # ------------------------------------------------------------------ run

    def tick(
        self,
        t: float,
        tel: Telemetry,
        dt_tick: float,
        aux_cmd_a: np.ndarray | None = None,
    ) -> None:
        """Call once per simulation tick with the current telemetry.

        aux_cmd_a: per-cell currents the controller itself commands (the
        active balancer) — known to the BMS even though they bypass the
        pack current sensor, so the filters must account for them.
        """
        self._i_hist.append((t, tel.i))
        self._i_sum += tel.i
        if aux_cmd_a is not None:
            self._aux_sum += aux_cmd_a
        self._i_n += 1
        if self._t_next is None:
            self._t_next = t + self.dt_filter
        if t + 1e-9 < self._t_next:
            return

        i_mean = self._i_sum / max(self._i_n, 1)
        aux_mean = self._aux_sum / max(self._i_n, 1)
        self._i_sum, self._i_n = 0.0, 0
        self._aux_sum = np.zeros(self.n)
        self._t_next += self.dt_filter

        i_cells = i_mean + aux_mean  # per-cell current seen by each cell
        temp = self._cell_temps(tel)
        t0 = time.perf_counter()
        self.bank.predict(i_cells, self.dt_filter, temp)
        self._propagate_hysteresis(i_cells)

        fresh = tel.v_time > self._last_seen
        if np.any(fresh):
            i_at_sample = self._current_at(tel.v_time, fallback=i_mean) + aux_mean
            self.bank.update(fresh, tel.v, i_at_sample, self.h)
            self._last_seen = np.maximum(self._last_seen, tel.v_time)
        self.compute_s += time.perf_counter() - t0
        self.n_steps += 1

    # -------------------------------------------------------------- helpers

    def _cell_temps(self, tel: Telemetry) -> np.ndarray:
        """Best available per-cell temperature: mean of the module's NTCs,
        falling back to the reference temperature before first reports."""
        finite = np.isfinite(tel.t_mod)
        cnt = finite.sum(axis=1)
        summed = np.where(finite, tel.t_mod, 0.0).sum(axis=1)
        t_mod = np.where(cnt > 0, summed / np.maximum(cnt, 1), self.p.t_ref_c)
        return t_mod[self.module_of]

    def _propagate_hysteresis(self, i_cells: float | np.ndarray) -> None:
        f = np.exp(
            -np.abs(i_cells) * self.p.gamma_hyst * self.dt_filter
            / (3600.0 * self.p.q_nom_ah)
        )
        self.h = f * self.h - (1.0 - f) * np.sign(i_cells) * self.p.m_hyst

    def _current_at(self, t_sample: np.ndarray, fallback: float) -> np.ndarray:
        """Current-sensor reading nearest each (per-cell) sample time."""
        if not self._i_hist:
            return np.full(self.n, fallback)
        hist_t = np.array([h[0] for h in self._i_hist])
        hist_i = np.array([h[1] for h in self._i_hist])
        idx = np.clip(np.searchsorted(hist_t, t_sample), 0, len(hist_t) - 1)
        out = hist_i[idx]
        return np.where(np.isfinite(t_sample), out, fallback)

    # -------------------------------------------------------------- reports

    @property
    def soc(self) -> np.ndarray:
        return self.bank.soc

    @property
    def r0(self) -> np.ndarray:
        return self.bank.r0

    @property
    def q_ah(self) -> np.ndarray:
        return self.bank.q_ah

    def soc_std(self) -> np.ndarray:
        return self.bank.std()[:, 0]

    def us_per_cell_step(self) -> float:
        if self.n_steps == 0:
            return float("nan")
        return 1e6 * self.compute_s / (self.n_steps * self.n)
