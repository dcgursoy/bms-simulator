"""Optimization-based balancing controller (receding-horizon LP).

Every replan interval the controller reads the *estimator's* SOC and
capacity (never ground truth) and solves a linear program for the next
window's per-cell balancing currents:

    variables   d_k >= 0 (drain), c_k >= 0 (charge), s_k (deviation slack)
    dynamics    soc'_k = soc_k - (d_k - c_k) * dt / (3600 q_k)
    objective   min  sum_k s_k  +  lambda_loss * (1 - eta^2) * sum_k v_k d_k
    s.t.        s_k >= |soc'_k - m*|            (via two inequalities)
                d_k + c_k <= i_max_k            (converter rating; the
                                                 per-cell vector is the
                                                 Phase 5 thermal-derate hook)
                eta * sum(v_k d_k) = (1/eta) * sum(v_k c_k)   (rail balance)

where m* is the charge-weighted mean SOC (the quantity a lossless
shuttle conserves). Greedily minimizing next-window deviation under the
true converter constraints is the max-descent approximation of the
time-optimal policy for this linear system; lambda_loss prices converter
losses so the LP stops churning charge when the remaining deviation is
cheaper to keep than to move. Solved with scipy's HiGHS in ~1 ms for 48
cells (144 variables) — this runs every 60 s of sim time.

The passive baseline (PassiveBleeder) is the classic alternative: bleed
every cell whose estimated SOC sits above the pack minimum, full stop.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from model.ocv import ocv


class LPBalancer:
    def __init__(
        self,
        n_cells: int,
        i_max_a: float = 1.0,
        eta_leg: float = 0.938,
        window_s: float = 60.0,
        deadband: float = 0.003,
        # Must stay below the marginal deviation gain per amp-window
        # (~2*window/(3600*q) ≈ 0.014) or the LP prefers doing nothing;
        # any positive value already kills useless drain+charge churn
        lambda_loss: float = 0.002,
    ):
        self.n = n_cells
        self.i_max = i_max_a
        self.eta = eta_leg
        self.window = window_s
        self.deadband = deadband
        self.lambda_loss = lambda_loss
        self.last_cmd = np.zeros(n_cells)
        self.solves = 0
        self.solve_failures = 0

    def replan(
        self,
        soc_est: np.ndarray,
        q_est_ah: np.ndarray,
        i_max_percell: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve the window LP; returns commanded currents (+ = drain)."""
        n = self.n
        i_lim = np.full(n, self.i_max) if i_max_percell is None else np.minimum(
            i_max_percell, self.i_max
        )
        m_star = float(np.sum(q_est_ah * soc_est) / np.sum(q_est_ah))
        if np.max(np.abs(soc_est - m_star)) < self.deadband:
            self.last_cmd = np.zeros(n)
            return self.last_cmd

        v_hat = ocv(soc_est)
        alpha = self.window / (3600.0 * q_est_ah)  # dSOC per amp-window

        # Variable layout: [d(0:n), c(n:2n), s(2n:3n)]
        cost = np.concatenate([
            self.lambda_loss * (1.0 - self.eta**2) * v_hat, np.zeros(n), np.ones(n)
        ])
        # |soc' - m*| <= s as two inequality blocks
        dev0 = soc_est - m_star
        a_ub = np.zeros((2 * n + n, 3 * n))
        b_ub = np.zeros(2 * n + n)
        rows = np.arange(n)
        #  +dev: -alpha d + alpha c - s <= -(dev0)  * (-1) ... expanded:
        #  dev = dev0 - alpha(d - c);  dev - s <= 0;  -dev - s <= 0
        a_ub[rows, rows] = -alpha
        a_ub[rows, n + rows] = alpha
        a_ub[rows, 2 * n + rows] = -1.0
        b_ub[rows] = -dev0
        a_ub[n + rows, rows] = alpha
        a_ub[n + rows, n + rows] = -alpha
        a_ub[n + rows, 2 * n + rows] = -1.0
        b_ub[n + rows] = dev0
        # converter rating: d + c <= i_lim
        a_ub[2 * n + rows, rows] = 1.0
        a_ub[2 * n + rows, n + rows] = 1.0
        b_ub[2 * n + rows] = i_lim
        # Rail power balance with a planning margin on the drain side. The
        # LP prices cells at estimated OCV, but the plant balances real
        # terminal voltages: a cell drained at full current sags ~2% below
        # OCV and an injected cell rides ~2% above (i*(R0+R1+R2)/v), so
        # schedules need >4% headroom or the rail intermittently overdraws
        # and the hardware must scale injections mid-window
        a_eq = np.concatenate(
            [0.94 * self.eta * v_hat, -v_hat / self.eta, np.zeros(n)]
        )
        res = linprog(
            cost,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq[None, :],
            b_eq=[0.0],
            bounds=[(0.0, None)] * (2 * n) + [(0.0, None)] * n,
            method="highs",
        )
        self.solves += 1
        if not res.success:
            self.solve_failures += 1
            self.last_cmd = np.zeros(n)
            return self.last_cmd
        d, c = res.x[:n], res.x[n : 2 * n]
        self.last_cmd = d - c
        return self.last_cmd


def thermal_derate(
    i_max_a: float,
    temp_c: np.ndarray,
    t_full_c: float = 40.0,
    t_zero_c: float = 55.0,
) -> np.ndarray:
    """Per-cell balancing-current limit vs temperature: full rating below
    t_full_c, linearly derated to zero at t_zero_c. Feed the result to
    LPBalancer.replan(i_max_percell=...) — the LP then plans around hot
    cells instead of pushing current through them."""
    frac = np.clip((t_zero_c - np.asarray(temp_c, dtype=float))
                   / (t_zero_c - t_full_c), 0.0, 1.0)
    return i_max_a * frac


class PassiveBleeder:
    """Baseline: resistive bleed toward the estimated minimum SOC."""

    def __init__(self, n_cells: int, i_bleed_a: float = 0.15, deadband: float = 0.003):
        self.n = n_cells
        self.i_bleed = i_bleed_a
        self.deadband = deadband
        self.last_cmd = np.zeros(n_cells)

    def replan(self, soc_est: np.ndarray, q_est_ah: np.ndarray, _=None) -> np.ndarray:
        target = float(np.min(soc_est))
        self.last_cmd = np.where(
            soc_est > target + self.deadband, self.i_bleed, 0.0
        )
        return self.last_cmd
