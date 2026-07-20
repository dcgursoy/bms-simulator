"""Fault injection into the truth simulation.

Three physically distinct fault mechanisms, injectable on any cell at
any time (this is the interface the live dashboard's fault panel
drives):

- internal short: a parallel fault resistance across the cell. The cell
  discharges through it at v/R_short (tens of amps for a hard short),
  invisible to the pack current sensor, and the *entire* electrochemical
  power dissipates as heat inside the cell casing.
- sensor fault: the module AFE channel for one cell goes bad — either
  frozen (reports hold the value at fault onset while timestamps keep
  updating, like a stuck ADC/mux) or offset (a constant bias on every
  subsequent report). Corruption happens on the telemetry the controller
  sees; the cell itself is healthy.
- accelerated degradation: the cell's aging_accel multiplier is raised,
  so its capacity fades and R0 grows far faster than the fleet under
  the same throughput (e.g. a cell developing lithium plating).

The injector sits between the bus and the controller for sensor faults,
and feeds aux currents / extra heat into the truth step for shorts.
"""

from __future__ import annotations

import numpy as np

from model.comms import Telemetry
from model.pack import Pack

FAULT_KINDS = ("short", "sensor_freeze", "sensor_offset", "degradation")


class FaultInjector:
    def __init__(self, pack: Pack):
        self.pack = pack
        n = pack.n_cells
        self.short_r_ohm = np.full(n, np.inf)
        self.sensor_offset_v = np.zeros(n)
        self.sensor_frozen = np.zeros(n, dtype=bool)
        self._frozen_value = np.full(n, np.nan)
        self._offset_applied_t: dict[int, float] = {}
        self._nominal_accel = pack.cells.aging_accel.copy()
        self.log: list[tuple[float, str]] = []

    # ----------------------------------------------------------- injection

    def inject_short(self, cell: int, r_ohm: float = 0.2, t: float = 0.0) -> None:
        self.short_r_ohm[cell] = r_ohm
        self.log.append((t, f"short {r_ohm} ohm on cell {cell}"))

    def inject_sensor_freeze(self, cell: int, t: float = 0.0) -> None:
        self.sensor_frozen[cell] = True
        self._frozen_value[cell] = np.nan  # captured at next report
        self.log.append((t, f"sensor freeze on cell {cell}"))

    def inject_sensor_offset(
        self, cell: int, offset_v: float = 0.06, t: float = 0.0
    ) -> None:
        self.sensor_offset_v[cell] = offset_v
        self.log.append((t, f"sensor offset {1e3 * offset_v:+.0f} mV on cell {cell}"))

    def inject_degradation(
        self, cell: int, accel: float = 1000.0, t: float = 0.0
    ) -> None:
        self.pack.cells.aging_accel[cell] = accel
        self.log.append((t, f"degradation x{accel:.0f} on cell {cell}"))

    def clear(self, t: float = 0.0) -> None:
        self.short_r_ohm[:] = np.inf
        self.sensor_offset_v[:] = 0.0
        self.sensor_frozen[:] = False
        self._frozen_value[:] = np.nan
        self.pack.cells.aging_accel = self._nominal_accel.copy()
        self.log.append((t, "all faults cleared"))

    def active_faults(self) -> list[tuple[int, str]]:
        out = [(int(c), "short") for c in np.flatnonzero(np.isfinite(self.short_r_ohm))]
        out += [(int(c), "sensor_freeze") for c in np.flatnonzero(self.sensor_frozen)]
        out += [(int(c), "sensor_offset")
                for c in np.flatnonzero(self.sensor_offset_v != 0.0)]
        out += [(int(c), "degradation") for c in np.flatnonzero(
            self.pack.cells.aging_accel != self._nominal_accel)]
        return out

    # ------------------------------------------------------- truth coupling

    def short_currents_a(self) -> np.ndarray:
        """Per-cell internal short currents [A] at the present cell
        voltages — add to pack.step's aux current."""
        v = np.maximum(self.pack.last_v, 0.0)
        return np.where(np.isfinite(self.short_r_ohm), v / self.short_r_ohm, 0.0)

    def short_heat_w(self, i_short: np.ndarray) -> np.ndarray:
        """Heat [W] dumped inside shorted cells (the short resistance is
        internal, so the delivered power never leaves the casing)."""
        return np.maximum(self.pack.last_v, 0.0) * i_short

    # --------------------------------------------------- telemetry coupling

    def corrupt_telemetry(self, tel: Telemetry) -> None:
        """Apply sensor faults to fresh reports. Call once per tick, after
        bus.step. Offsets are applied exactly once per new report (tracked
        via the report timestamps); frozen channels overwrite every tick."""
        for cell in np.flatnonzero(self.sensor_offset_v != 0.0):
            if tel.v_time[cell] > self._offset_applied_t.get(cell, -np.inf):
                tel.v[cell] += self.sensor_offset_v[cell]
                self._offset_applied_t[cell] = tel.v_time[cell]
        frozen = self.sensor_frozen & np.isfinite(tel.v_time)
        for cell in np.flatnonzero(frozen):
            if np.isnan(self._frozen_value[cell]):
                self._frozen_value[cell] = tel.v[cell]
            tel.v[cell] = self._frozen_value[cell]
