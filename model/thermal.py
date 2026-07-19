"""Per-cell lumped thermal model on the pack's 6 x 8 physical grid.

Each cell is one thermal mass (C_th ~ 40 J/K for an 18650-class cell)
with three heat paths:

- generation: irreversible I2R losses in R0/R1/R2 plus reversible
  entropic heat (computed by CellArray.heat_generation_w),
- conduction to orthogonally adjacent cells through holders/busbars
  (graph-Laplacian coupling on the 6 x 8 grid, module = row),
- convection to ambient, with MORE cooling for cells that expose sides
  to the enclosure airflow: interior cells run hottest, corners coolest
  — which is what makes the pack thermal map spatially interesting.

The electro-thermal loop closes outside this class: the simulation
feeds cell temperatures back into the electrical model's Arrhenius
resistances and cold-capacity derating every step, so heating lowers
impedance (less sag, less heat — negative feedback) and cold does the
opposite. Explicit Euler is fine: the fastest thermal time constant
(C_th / k_total ~ 45 s) is orders above the 0.1 s tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalParams:
    c_th_j_per_k: float = 40.0        # per-cell heat capacity [J/K]
    k_cond_w_per_k: float = 0.2       # cell-to-cell conduction [W/K]
    k_amb_interior: float = 0.06      # convection, fully surrounded cell [W/K]
    k_amb_per_exposed_side: float = 0.03  # extra convection per open side [W/K]
    t_amb_c: float = 25.0


class ThermalModel:
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        params: ThermalParams | None = None,
        t0_c: float | np.ndarray | None = None,
    ):
        self.p = params if params is not None else ThermalParams()
        self.rows, self.cols = n_rows, n_cols
        n = n_rows * n_cols
        self.n = n

        # Graph Laplacian of the orthogonal-neighbor grid (row-major
        # ordering matches the pack's cell index: module = row)
        lap = np.zeros((n, n))
        for r in range(n_rows):
            for c in range(n_cols):
                k = r * n_cols + c
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n_rows and 0 <= cc < n_cols:
                        j = rr * n_cols + cc
                        lap[k, k] += 1.0
                        lap[k, j] -= 1.0
        self.laplacian = lap
        degree = np.diag(lap)
        self.k_amb = (
            self.p.k_amb_interior
            + self.p.k_amb_per_exposed_side * (4.0 - degree)
        )
        t0 = self.p.t_amb_c if t0_c is None else t0_c
        self.temps = np.broadcast_to(np.asarray(t0, dtype=float), (n,)).copy()

    def step(self, q_gen_w: np.ndarray, dt_s: float) -> np.ndarray:
        """Advance temperatures one step under per-cell heat input [W]."""
        cond = -self.p.k_cond_w_per_k * (self.laplacian @ self.temps)
        conv = -self.k_amb * (self.temps - self.p.t_amb_c)
        self.temps = self.temps + dt_s * (q_gen_w + cond + conv) / self.p.c_th_j_per_k
        return self.temps
