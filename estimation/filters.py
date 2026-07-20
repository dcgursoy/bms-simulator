"""Vectorized UKF and EKF banks for per-cell joint SOC + SOH estimation.

Per-cell state vector (5 states):

    x = [ soc, v_rc1, v_rc2, r0, q_ah ]

soc + RC voltages track fast dynamics; r0 and q_ah are slowly-varying
parameters (random walks in the process model) whose online adaptation
IS the SOH estimate — impedance growth and capacity fade are the two
canonical aging signatures. Joint (single-filter) estimation is used
rather than a dual-filter scheme: with only 5 states the extra coupling
costs little and the cross-covariances between SOC and capacity are
exactly what makes capacity converge from coulomb/OCV consistency.

Both filters share the same process/measurement model and are batched
over all n cells with einsum linear algebra (no per-cell Python loop).
The measurement is scalar (one terminal voltage per cell), so the
innovation covariance is a scalar per cell and no matrix inversion is
ever needed.

UKF vs EKF (why the UKF is the primary):
- The measurement model's nonlinearity is the OCV curve: nearly flat at
  mid-SOC, steep at the knees. The EKF linearizes at the mean, so it
  systematically mis-weights updates when the estimate sits near a
  curvature change; the UKF's sigma points sample the curve across the
  uncertainty spread and capture that structure.
- The process model is nonlinear in q (soc drift ~ i/q); sigma points
  propagate that interaction exactly where the EKF needs a Jacobian
  approximation.
- Cost: the UKF evaluates 2n+1 = 11 sigma points and one Cholesky per
  cell per step vs the EKF's single model call — measured, not assumed,
  in the Phase 3 benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.cell import CellParams
from model.ocv import docv_dsoc, ocv

NX = 5
STATE_NAMES = ("soc", "v_rc1", "v_rc2", "r0", "q_ah")


@dataclass
class FilterTuning:
    # Initial standard deviations per state
    p0_std: tuple = (0.40, 0.010, 0.010, 0.005, 0.20)
    # Continuous process-noise variance rates [unit^2 / s]:
    # soc absorbs current-sensor error + hysteresis/efficiency mismatch;
    # r0 and q_ah random-walk rates set how fast SOH can be tracked. The
    # r0 rate is sized to follow abnormally fast impedance growth (the
    # accelerated-degradation fault, ~4 mOhm/h) — costs only ~0.2 mOhm
    # of extra wander between excitations
    q_rate: tuple = (7e-9, 1e-6, 5e-7, 1e-8, 2.8e-8)
    # Voltage measurement variance: sensor noise + quantization + the
    # residual from up-to-1 s report staleness
    r_var: float = (2.5e-3) ** 2
    # Physical projection bounds for the parameter states
    r0_bounds: tuple = (0.008, 0.080)
    q_bounds: tuple = (1.5, 3.2)
    # R0 is unobservable near zero current; below this excitation its
    # update gain is zeroed so large innovations (e.g. a cold boot at
    # rest) cannot shove it through weak cross-covariance
    min_exc_a: float = 0.5
    # Unscented-transform scaling
    alpha: float = 0.5
    beta: float = 2.0
    kappa: float = 0.0


class _FilterBank:
    """Shared model + storage for a bank of n per-cell filters."""

    def __init__(
        self,
        n: int,
        x0: np.ndarray,
        params: CellParams | None = None,
        tuning: FilterTuning | None = None,
    ):
        self.n = n
        self.p = params if params is not None else CellParams()
        self.tun = tuning if tuning is not None else FilterTuning()
        self.x = np.array(x0, dtype=float).reshape(n, NX).copy()
        p0 = np.square(self.tun.p0_std)
        self.P = np.tile(np.diag(p0), (n, 1, 1))
        self._q_rate = np.diag(self.tun.q_rate)
        self._jitter = 1e-9 * np.diag(p0)
        # Innovation record of the most recent filter step — what the
        # residual-based fault detector consumes. NaN = no update.
        self.last_innov = np.full(n, np.nan)
        self.last_innov_var = np.full(n, np.nan)
        self.last_update_mask = np.zeros(n, dtype=bool)

    def clear_innovations(self) -> None:
        """Reset the per-step innovation record (call before each filter
        step so stale innovations never masquerade as fresh ones)."""
        self.last_innov[:] = np.nan
        self.last_innov_var[:] = np.nan
        self.last_update_mask[:] = False

    def _record_innovations(self, mask, innov, innov_var) -> None:
        self.last_innov = np.where(mask, innov, np.nan)
        self.last_innov_var = np.where(mask, innov_var, np.nan)
        self.last_update_mask = mask.copy()

    # -------------------------------------------------------- shared model

    def _propagate(self, x, i, dt, temp_c):
        """Nominal-parameter process model; works on (..., NX) stacks."""
        soc, v1, v2, r0, q = np.moveaxis(x, -1, 0)
        q = np.clip(q, 0.5, None)
        arrh = np.exp(
            self.p.ea_over_r * (1.0 / (temp_c + 273.15) - 1.0 / (self.p.t_ref_c + 273.15))
        )
        r1e, r2e = self.p.r1 * arrh, self.p.r2 * arrh
        a1 = np.exp(-dt / (r1e * self.p.c1))
        a2 = np.exp(-dt / (r2e * self.p.c2))
        eta = np.where(i < 0.0, self.p.eta_chg, 1.0)
        soc2 = np.clip(soc - eta * i * dt / (3600.0 * q), 0.0, 1.0)
        v1_2 = a1 * v1 + r1e * (1.0 - a1) * i
        v2_2 = a2 * v2 + r2e * (1.0 - a2) * i
        return np.stack(
            [soc2, v1_2, v2_2, np.broadcast_to(r0, soc2.shape),
             np.broadcast_to(q, soc2.shape)],
            axis=-1,
        )

    def _measure(self, x, i, h):
        """Terminal-voltage measurement model on (..., NX) stacks."""
        soc, v1, v2, r0, _ = np.moveaxis(x, -1, 0)
        return ocv(soc) + h - i * r0 - v1 - v2

    def _finalize(self):
        """Symmetrize P, floor its diagonal, project parameters into
        physical bounds."""
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2)) + self._jitter
        self.x[:, 3] = np.clip(self.x[:, 3], *self.tun.r0_bounds)
        self.x[:, 4] = np.clip(self.x[:, 4], *self.tun.q_bounds)
        self.x[:, 0] = np.clip(self.x[:, 0], 0.0, 1.0)

    # ------------------------------------------------------------- reports

    @property
    def soc(self) -> np.ndarray:
        return self.x[:, 0]

    @property
    def r0(self) -> np.ndarray:
        return self.x[:, 3]

    @property
    def q_ah(self) -> np.ndarray:
        return self.x[:, 4]

    def std(self) -> np.ndarray:
        return np.sqrt(np.diagonal(self.P, axis1=1, axis2=2))


class UKFBank(_FilterBank):
    """Scaled unscented Kalman filter, batched over cells."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t = self.tun
        lam = t.alpha**2 * (NX + t.kappa) - NX
        self._gamma = np.sqrt(NX + lam)
        self.wm = np.full(2 * NX + 1, 1.0 / (2.0 * (NX + lam)))
        self.wc = self.wm.copy()
        self.wm[0] = lam / (NX + lam)
        self.wc[0] = lam / (NX + lam) + (1.0 - t.alpha**2 + t.beta)

    def _sigma_points(self) -> np.ndarray:
        L = np.linalg.cholesky(self.P + self._jitter)  # (n, NX, NX), lower
        X = np.empty((self.n, 2 * NX + 1, NX))
        X[:, 0] = self.x
        offsets = self._gamma * np.swapaxes(L, 1, 2)  # rows = scaled columns of L
        X[:, 1 : NX + 1] = self.x[:, None, :] + offsets
        X[:, NX + 1 :] = self.x[:, None, :] - offsets
        return X

    def predict(self, i, dt: float, temp_c) -> None:
        i = np.broadcast_to(np.asarray(i, dtype=float), (self.n,))
        temp_c = np.broadcast_to(np.asarray(temp_c, dtype=float), (self.n,))
        X = self._sigma_points()
        Xp = self._propagate(X, i[:, None], dt, temp_c[:, None])
        self.x = np.einsum("s,nsx->nx", self.wm, Xp)
        d = Xp - self.x[:, None, :]
        self.P = np.einsum("s,nsx,nsy->nxy", self.wc, d, d) + self._q_rate * dt
        self._finalize()

    def update(self, mask: np.ndarray, v_meas: np.ndarray, i_meas, h) -> None:
        if not np.any(mask):
            return
        i_meas = np.broadcast_to(np.asarray(i_meas, dtype=float), (self.n,))
        h = np.broadcast_to(np.asarray(h, dtype=float), (self.n,))
        X = self._sigma_points()
        Y = self._measure(X, i_meas[:, None], h[:, None])  # (n, 2NX+1)
        ybar = np.einsum("s,ns->n", self.wm, Y)
        dy = Y - ybar[:, None]
        pyy = np.einsum("s,ns,ns->n", self.wc, dy, dy) + self.tun.r_var
        dx = X - self.x[:, None, :]
        pxy = np.einsum("s,nsx,ns->nx", self.wc, dx, dy)
        gain = pxy / pyy[:, None]
        gain[np.abs(i_meas) < self.tun.min_exc_a, 3] = 0.0
        innov = v_meas - ybar
        m = mask
        self._record_innovations(m, innov, pyy)
        self.x[m] += gain[m] * innov[m, None]
        # Arbitrary-gain covariance update (valid with the gated gain,
        # where the optimal-gain shortcut P -= pyy*K*K^T is not):
        # P <- P - K*Pxy^T - Pxy*K^T + pyy*K*K^T
        kp = gain[m, :, None] * pxy[m, None, :]
        self.P[m] -= (
            kp + np.swapaxes(kp, 1, 2)
            - pyy[m, None, None] * gain[m, :, None] * gain[m, None, :]
        )
        self._finalize()


class EKFBank(_FilterBank):
    """Extended Kalman filter, batched over cells."""

    def predict(self, i, dt: float, temp_c) -> None:
        i = np.broadcast_to(np.asarray(i, dtype=float), (self.n,))
        temp_c = np.broadcast_to(np.asarray(temp_c, dtype=float), (self.n,))
        q = np.clip(self.x[:, 4], 0.5, None)
        arrh = np.exp(
            self.p.ea_over_r * (1.0 / (temp_c + 273.15) - 1.0 / (self.p.t_ref_c + 273.15))
        )
        a1 = np.exp(-dt / (self.p.r1 * arrh * self.p.c1))
        a2 = np.exp(-dt / (self.p.r2 * arrh * self.p.c2))
        eta = np.where(i < 0.0, self.p.eta_chg, 1.0)

        self.x = self._propagate(self.x, i, dt, temp_c)

        F = np.tile(np.eye(NX), (self.n, 1, 1))
        F[:, 0, 4] = eta * i * dt / (3600.0 * q**2)
        F[:, 1, 1] = a1
        F[:, 2, 2] = a2
        self.P = (
            np.einsum("nij,njk,nlk->nil", F, self.P, F) + self._q_rate * dt
        )
        self._finalize()

    def update(self, mask: np.ndarray, v_meas: np.ndarray, i_meas, h) -> None:
        if not np.any(mask):
            return
        i_meas = np.broadcast_to(np.asarray(i_meas, dtype=float), (self.n,))
        h = np.broadcast_to(np.asarray(h, dtype=float), (self.n,))
        H = np.zeros((self.n, NX))
        H[:, 0] = docv_dsoc(self.x[:, 0])
        H[:, 1] = -1.0
        H[:, 2] = -1.0
        H[:, 3] = -i_meas
        ybar = self._measure(self.x, i_meas, h)
        PH = np.einsum("nxy,ny->nx", self.P, H)
        S = np.einsum("nx,nx->n", H, PH) + self.tun.r_var
        gain = PH / S[:, None]
        gain[np.abs(i_meas) < self.tun.min_exc_a, 3] = 0.0
        innov = v_meas - ybar
        m = mask
        self._record_innovations(m, innov, S)
        self.x[m] += gain[m] * innov[m, None]
        # Arbitrary-gain covariance update (see UKF note): with PH = P*H^T,
        # P <- P - K*PH^T - PH*K^T + S*K*K^T
        kp = gain[m, :, None] * PH[m, None, :]
        self.P[m] -= (
            kp + np.swapaxes(kp, 1, 2)
            - S[m, None, None] * gain[m, :, None] * gain[m, None, :]
        )
        self._finalize()


def make_filter_bank(kind: str, *args, **kwargs) -> _FilterBank:
    banks = {"ukf": UKFBank, "ekf": EKFBank}
    return banks[kind.lower()](*args, **kwargs)
