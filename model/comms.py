"""Bandwidth-limited BMS telemetry bus between module monitors and the
pack controller.

Architecture mirrors a real distributed BMS:

- Each module carries a cell-monitor board (LTC68xx-class AFE) that
  digitizes its 8 cell voltages and 2 NTC temperatures with gaussian
  noise + ADC quantization. NTC A senses the thermal neighborhood of the
  module's first 4 cells, NTC B the last 4 — per-cell temperature is NOT
  directly observable, only these coarse module probes.
- Monitors share one bandwidth-limited serial bus (CAN-class) to the
  pack controller: a fixed frame budget per second is granted
  round-robin. One module snapshot costs 3 frames — cell voltages 0-3,
  cell voltages 4-7, temperatures/status — so with the default 20
  frames/s and 6 modules the full pack refreshes every
  6*3/20 = 0.9 s, and each datum the controller holds is up to that
  stale. Frames sample the *live* value at grant time, so staleness
  comes from the schedule, not an extra queue delay.
- The pack current sensor is wired directly to the controller and
  sampled every simulation tick (no bus cost), as in real packs.

The controller-side picture lives in ``Telemetry``: last reported value
+ report timestamp per channel. Estimators (Phase 3) consume ONLY this
object — never ground truth — which is what makes the bandwidth limit
meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.pack import Pack


@dataclass
class SensorSpec:
    v_noise_sigma: float = 0.0015   # cell voltage sense noise [V]
    v_quant: float = 0.001          # cell voltage ADC LSB [V]
    t_noise_sigma: float = 0.3      # NTC noise [degC]
    t_quant: float = 0.25           # NTC quantization [degC]
    i_noise_sigma: float = 0.025    # pack current sense noise [A]
    i_quant: float = 0.01           # pack current quantization [A]


@dataclass
class BusConfig:
    frames_per_s: float = 20.0
    frames_per_module: int = 3      # 2 voltage frames + 1 temp/status frame

    def refresh_period_s(self, n_modules: int) -> float:
        return n_modules * self.frames_per_module / self.frames_per_s


class Telemetry:
    """What the pack controller actually knows, and how old it is."""

    def __init__(self, n_cells: int, n_modules: int):
        self.v = np.full(n_cells, np.nan)          # last reported voltage [V]
        self.v_time = np.full(n_cells, -np.inf)    # report timestamp [s]
        self.t_mod = np.full((n_modules, 2), np.nan)  # NTC A/B per module [degC]
        self.t_mod_time = np.full(n_modules, -np.inf)
        self.i = np.nan                            # pack current [A]
        self.i_time = -np.inf

    def v_age(self, t: float) -> np.ndarray:
        return t - self.v_time

    def t_age(self, t: float) -> np.ndarray:
        return t - self.t_mod_time


class BmsBus:
    def __init__(
        self,
        pack: Pack,
        sensors: SensorSpec | None = None,
        config: BusConfig | None = None,
        seed: int = 7,
    ):
        self.pack = pack
        self.sensors = sensors if sensors is not None else SensorSpec()
        self.config = config if config is not None else BusConfig()
        self.rng = np.random.default_rng(seed)
        self.telemetry = Telemetry(pack.n_cells, pack.config.n_modules)
        # Round-robin schedule of (module, frame_kind); kind 0/1 = voltage
        # halves, kind 2 = temperatures
        self._schedule = [
            (m, k)
            for m in range(pack.config.n_modules)
            for k in range(self.config.frames_per_module)
        ]
        self._next_frame = 0
        self._frame_credit = 0.0

    def _quantize(self, x, lsb: float):
        return np.round(np.asarray(x, dtype=float) / lsb) * lsb

    def _send_frame(self, t: float, temp_c_percell: np.ndarray) -> None:
        m, kind = self._schedule[self._next_frame]
        self._next_frame = (self._next_frame + 1) % len(self._schedule)
        tel, s = self.telemetry, self.sensors
        sl = self.pack.module_slice(m)
        if kind in (0, 1):  # one half of the module's cell voltages
            half = slice(sl.start + 4 * kind, sl.start + 4 * (kind + 1))
            true_v = self.pack.last_v[half]
            meas = true_v + self.rng.normal(0.0, s.v_noise_sigma, true_v.shape)
            tel.v[half] = self._quantize(meas, s.v_quant)
            tel.v_time[half] = t
        else:  # temperatures: NTC A = first 4 cells' area, NTC B = last 4
            temps = temp_c_percell[sl]
            for probe, quarter in enumerate((temps[:4], temps[4:])):
                meas = float(np.mean(quarter)) + self.rng.normal(0.0, s.t_noise_sigma)
                tel.t_mod[m, probe] = float(self._quantize(meas, s.t_quant))
            tel.t_mod_time[m] = t

    def step(self, t: float, dt_s: float, temp_c_percell: np.ndarray) -> None:
        """Deliver whatever frames the bandwidth budget allows this tick,
        and sample the (directly wired) pack current sensor."""
        s = self.sensors
        i_meas = self.pack.last_current_a + self.rng.normal(0.0, s.i_noise_sigma)
        self.telemetry.i = float(self._quantize(i_meas, s.i_quant))
        self.telemetry.i_time = t

        self._frame_credit += self.config.frames_per_s * dt_s
        while self._frame_credit >= 1.0:
            self._send_frame(t, np.broadcast_to(
                np.asarray(temp_c_percell, dtype=float), (self.pack.n_cells,)
            ))
            self._frame_credit -= 1.0
