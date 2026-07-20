"""Severity-tiered safety response policy.

Maps diagnoses to actions a real pack controller has at its disposal.
Actions are one-way ratchets within a run (a safety state never relaxes
just because the symptom faded):

- internal short (critical): open the main contactor — simulated
  shutdown, pack current forced to zero. The short itself keeps burning
  inside the cell (a contactor cannot stop an internal fault), but the
  load/charger is isolated and the event is alarmed.
- sensor fault (warning): quarantine the channel — the estimator stops
  applying that cell's voltage updates and propagates it open-loop —
  and derate pack current to 50% (reduced observability means reduced
  confidence in enforcing that cell's limits).
- accelerated degradation (maintenance): flag for service and derate
  pack current to 75% to slow further damage.
"""

from __future__ import annotations

import numpy as np

from fault_detection.detector import Diagnosis


class SafetyPolicy:
    def __init__(self, n_cells: int):
        self.n = n_cells
        self.i_limit_frac = 1.0
        self.contactor_open = False
        self.flagged = np.zeros(n_cells, dtype=bool)
        self.excluded = np.zeros(n_cells, dtype=bool)  # sensor quarantine
        self.events: list[tuple[float, str]] = []

    def apply(self, t: float, diagnoses: list[Diagnosis]) -> None:
        for d in diagnoses:
            self.flagged[d.cell] = True
            if d.kind == "internal_short":
                self.contactor_open = True
                self.i_limit_frac = 0.0
                self.events.append((t, (
                    f"SHUTDOWN: contactor opened — {d.subtype} on cell "
                    f"{d.cell} ({d.detail})")))
            elif d.kind == "sensor_fault":
                self.excluded[d.cell] = True
                self.i_limit_frac = min(self.i_limit_frac, 0.5)
                self.events.append((t, (
                    f"DERATE 50% + quarantine cell {d.cell} telemetry — "
                    f"sensor {d.subtype} ({d.detail})")))
            elif d.kind == "degradation":
                self.i_limit_frac = min(self.i_limit_frac, 0.75)
                self.events.append((t, (
                    f"DERATE 75% + maintenance flag on cell {d.cell} — "
                    f"{d.subtype} ({d.detail})")))

    def limit_current(self, i_request_a: float) -> float:
        """The current the pack will actually carry for a request."""
        if self.contactor_open:
            return 0.0
        return i_request_a * self.i_limit_frac
