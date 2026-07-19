"""Active-balancing hardware model: bidirectional cell-to-pack DC-DC.

Topology: every cell has a small bidirectional flyback converter onto a
shared isolated balancing rail. A converter can *drain* its cell
(positive balancing current) feeding the rail, or *charge* it (negative)
drawing from the rail. Each conversion leg has efficiency eta_leg, so a
cell-to-cell transfer round-trips at eta_leg^2 (~88%).

The rail stores no energy, so drained power must fund injected power at
every instant:

    eta_leg * sum(v_k * i_k+)  >=  (1/eta_leg) * sum(v_k * i_k-)

The plant enforces this physically: if a commanded schedule would
overdraw the rail, all injections are scaled down to the available
power (and the event is counted — a correct optimizer never triggers
it). Converter losses are integrated in Wh for the energy comparison.

A passive-bleed mode models the classic baseline: per-cell resistive
bleed (drain only, much smaller current, no rail) where every bled
joule is simply lost as heat.
"""

from __future__ import annotations

import numpy as np


class BalancerPlant:
    def __init__(
        self,
        n_cells: int,
        i_max_a: float = 1.0,
        eta_leg: float = 0.938,  # per conversion leg; round trip ~0.88
        passive: bool = False,
        i_bleed_a: float = 0.15,
    ):
        self.n = n_cells
        self.i_max = i_max_a
        self.eta = eta_leg
        self.passive = passive
        self.i_bleed = i_bleed_a
        self.loss_wh = 0.0
        self.moved_ah = 0.0
        self.rail_scale_events = 0
        self.last_applied = np.zeros(n_cells)

    def apply(self, cmd_a: np.ndarray, cell_v: np.ndarray, dt_s: float) -> np.ndarray:
        """Turn commanded balancing currents into physically consistent
        per-cell auxiliary currents (positive = extra discharge), and
        account the losses. Returns the applied currents."""
        if self.passive:
            i = np.clip(cmd_a, 0.0, self.i_bleed)  # bleed resistors only drain
            self.loss_wh += float(np.sum(cell_v * i)) * dt_s / 3600.0
            self.moved_ah += float(np.sum(i)) * dt_s / 3600.0
            self.last_applied = i
            return i

        i = np.clip(cmd_a, -self.i_max, self.i_max)
        drain = np.maximum(i, 0.0)
        inject = np.maximum(-i, 0.0)
        p_avail = self.eta * float(np.sum(cell_v * drain))
        p_need = float(np.sum(cell_v * inject)) / self.eta
        if p_need > p_avail + 1e-12:
            scale = p_avail / p_need if p_need > 0 else 0.0
            inject *= scale
            self.rail_scale_events += 1
        i = drain - inject
        # Loss = power leaving cells minus power arriving back in cells
        # (surplus drained power that funds nothing is counted as lost)
        p_out = float(np.sum(cell_v * drain))
        p_in = float(np.sum(cell_v * inject))
        self.loss_wh += (p_out - p_in) * dt_s / 3600.0
        self.moved_ah += float(np.sum(drain)) * dt_s / 3600.0
        self.last_applied = i
        return i
