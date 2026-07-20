"""Model-based fault detection: residuals between what the sensors
report and what the estimator's cell model predicts.

The detector runs at the filter cadence (1 Hz) on the pack controller
and consumes exactly three streams, all of them things a real BMS has:

- the UKF's voltage innovations (measurement minus model prediction)
  and their predicted variances,
- the estimator's state outputs (SOC, R0) per cell,
- raw telemetry (report values/timestamps, module NTC temps).

From these it maintains per-cell features (all fleet-relative, so
pack-wide effects like temperature or load cancel):

- normalized-innovation EWMA and instantaneous innovation jumps,
- SOC slew anomaly: d(soc)/dt vs the fleet median — an internal short
  drains one cell's SOC while its siblings stay put,
- module temperature-rise anomaly (corroborates real heat),
- report-freeze counters (a stuck ADC repeats one value while the
  fleet's reports move),
- R0 growth-rate anomaly (accelerated aging, minutes-scale).

Fault signatures (first match wins; a diagnosis is sticky per cell):

| fault              | voltage residual      | SOC slew | temp | reports |
|--------------------|-----------------------|----------|------|---------|
| internal short     | negative jump at onset| draining | rise | live    |
| sensor freeze      | grows as truth drifts | normal   | flat | frozen  |
| sensor offset      | sustained, no physics | normal   | flat | live    |
| accel. degradation | none (filter adapts)  | normal   | flat | live    |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimation.pack_estimator import PackEstimator
from model.cell import CellParams
from model.comms import Telemetry

CRITICAL, WARNING, MAINTENANCE = "critical", "warning", "maintenance"


@dataclass
class Diagnosis:
    cell: int
    kind: str        # "internal_short" | "sensor_fault" | "degradation"
    subtype: str
    severity: str
    t_detect: float
    detail: str


@dataclass
class DetectorTuning:
    # innovation jump: instantaneous |innovation| above this many sigmas
    # (and above an absolute floor) opens a classification window
    jump_nsigma: float = 6.0
    jump_floor_v: float = 0.040
    jump_window_s: float = 90.0
    # SOC slew anomaly (vs fleet median) flagging unexplained drain [1/s]
    drain_slew: float = -3.0e-4
    # module temperature rise anomaly corroborating real heat [K / 45 s]
    hot_rise_k: float = 2.0
    # consecutive suspicious steps before declaring (debounce)
    streak_n: int = 3
    # sensor-freeze: identical reports in a row while the fleet moves.
    # Healthy channels never freeze — sensor noise (1.5 mV) exceeds the
    # 1 mV LSB, so consecutive healthy reports repeat exactly only ~25%
    # of the time; 12 in a row is ~1e-8. The fleet guard (median report
    # delta, driven by that same noise, ~1.4 mV) only blocks degenerate
    # dead-bus cases.
    freeze_reports: int = 12
    freeze_fleet_mv: float = 0.5
    # sensor-offset: the filter absorbs an offset within seconds, so the
    # surviving signature is a STEP in fleet-relative SOC across the
    # jump window that then settles (no continuing drain, no heat) —
    # unlike a short, whose drain persists. Only part of the bias lands
    # in SOC (the RC states and R0 soak up the rest), so the step
    # threshold is ~1/4 of offset/dOCV — still ~4x the healthy
    # fleet-relative SOC fluctuation, and the jump gate does the heavy
    # lifting against false positives
    offset_confirm_s: float = 45.0
    offset_soc_step: float = 0.015
    # degradation: fleet-relative growth of the (EWMA-smoothed,
    # temperature-normalized) R0 estimate over a 10 min window. Healthy
    # estimates wander a few % on minutes timescales (noise-driven gain
    # kicks + RC-mismatch excursions during load shifts) but those
    # excursions MEAN-REVERT; true aging is monotone — so the rule
    # requires growth at both the half-window and full-window horizons.
    # Temperature normalization (Arrhenius at the module NTC) stops a
    # warming pack's uneven cooling from masquerading as aging, and the
    # estimator-confidence gate keeps the cold-boot convergence fan-out
    # (estimates walking from nominal to per-cell truth) from tripping it
    # Thresholds sized empirically: the filter tracks a genuine aging
    # ramp at ~70% of its true rate (q_rate-limited lag); healthy cells'
    # own mismatch-driven excursions reach ~5% per window but are
    # transient peaks, while a real fault sustains its growth rate in
    # every window — hence the high threshold AND the long streak.
    # Converged R0 std sits at 1.4-1.9 mOhm vs 4+ mOhm during cold boot,
    # so 2.5 mOhm separates the regimes cleanly
    degradation_rel: float = 0.05
    degradation_rel_mid: float = 0.018
    degradation_streak: int = 75
    r0_confident_std: float = 2.5e-3
    r0_smooth_alpha: float = 0.05
    # SOH trending is invalid during thermal events: the module NTC no
    # longer represents individual cells, and at rest the excitation-
    # gated R0 estimate cannot follow the real Arrhenius shift — the
    # normalization then fabricates apparent growth
    degradation_max_temp_c: float = 45.0

_SOC_RING = 46    # 45 s slew window at 1 Hz
_R0_RING = 601    # 600 s R0-trend window (300 s half-window checkpoint)


class FaultDetector:
    def __init__(
        self,
        n_cells: int,
        module_of: np.ndarray,
        tuning: DetectorTuning | None = None,
        dt_s: float = 1.0,
        params: CellParams | None = None,
    ):
        self.n = n_cells
        self.module_of = module_of
        self.tun = tuning if tuning is not None else DetectorTuning()
        self.dt = dt_s
        self.p = params if params is not None else CellParams()
        self.diagnoses: dict[int, Diagnosis] = {}

        self.ewma_z = np.zeros(n_cells)
        self._soc_ring = np.full((_SOC_RING, n_cells), np.nan)
        self._r0_ring = np.full((_R0_RING, n_cells), np.nan)
        self._temp_ring = np.full((_SOC_RING, n_cells), np.nan)
        self._ring_i = 0
        self._last_report_v = np.full(n_cells, np.nan)
        self._last_report_t = np.full(n_cells, -np.inf)
        self._consec_same = np.zeros(n_cells, dtype=int)
        self._fleet_delta_ewma = 0.0
        self._pending_t = np.full(n_cells, np.nan)
        self._pending_sign = np.zeros(n_cells)
        self._soc_rel_at_jump = np.full(n_cells, np.nan)
        self._soc_rel_prev = np.zeros(n_cells)
        self._r0_smooth: np.ndarray | None = None
        self._short_streak = np.zeros(n_cells, dtype=int)
        self._degr_streak = np.zeros(n_cells, dtype=int)
        self.n_steps = 0

    # ------------------------------------------------------------------ run

    def step(
        self,
        t: float,
        tel: Telemetry,
        est: PackEstimator,
        alarm: bool = False,
    ) -> list[Diagnosis]:
        """Call once per filter step; returns diagnoses newly made now.

        alarm=True (pack already in a critical safety state) suppresses
        maintenance-tier trending — post-shutdown thermal transients and
        the absence of load make SOH trends meaningless.
        """
        tun = self.tun
        quarantined = est.excluded

        # -- innovation features ------------------------------------------
        upd = est.bank.last_update_mask & ~quarantined
        z = np.zeros(self.n)
        innov = est.bank.last_innov
        with np.errstate(invalid="ignore"):
            z[upd] = innov[upd] / np.sqrt(est.bank.last_innov_var[upd])
        self.ewma_z[upd] = 0.75 * self.ewma_z[upd] + 0.25 * z[upd]

        jump = upd & (
            np.abs(innov) > np.maximum(
                tun.jump_nsigma * np.sqrt(est.bank.last_innov_var),
                tun.jump_floor_v,
            )
        )
        soc_rel = est.soc - np.median(est.soc)
        opens = jump & np.isnan(self._pending_t)
        self._pending_t[opens] = t
        self._pending_sign[opens] = np.sign(innov[opens])
        # Snapshot the PRE-update fleet-relative SOC: by the time the
        # detector runs, this step's update has already absorbed part of
        # the anomaly into the estimate
        self._soc_rel_at_jump[opens] = self._soc_rel_prev[opens]
        self._soc_rel_prev = soc_rel
        expired = (t - self._pending_t) > tun.jump_window_s
        self._pending_t[expired] = np.nan
        self._soc_rel_at_jump[expired] = np.nan

        # -- report tracking (freeze + fleet activity) --------------------
        fresh = tel.v_time > self._last_report_t
        had_prev = fresh & np.isfinite(self._last_report_v)
        delta = np.where(had_prev, tel.v - self._last_report_v, np.nan)
        same = had_prev & (np.abs(delta) < 0.5e-3)
        self._consec_same[same] += 1
        self._consec_same[fresh & ~same] = 0
        if had_prev.any():
            fleet_delta = float(np.median(np.abs(delta[had_prev])))
            self._fleet_delta_ewma = (
                0.8 * self._fleet_delta_ewma + 0.2 * fleet_delta
            )
        self._last_report_v = np.where(fresh, tel.v, self._last_report_v)
        self._last_report_t = np.maximum(self._last_report_t, tel.v_time)

        # -- slow rings: SOC slew, R0 trend, module temperature ------------
        oldest = (self._ring_i + 1) % _SOC_RING
        soc_old = self._soc_ring[oldest]
        temp_old = self._temp_ring[oldest]
        window = (_SOC_RING - 1) * self.dt
        cell_temp = self._module_temps(tel)
        slew = np.where(
            np.isfinite(soc_old), (est.soc - soc_old) / window, 0.0
        )
        slew_anom = slew - np.median(slew)
        temp_rise = np.where(
            np.isfinite(temp_old), cell_temp - temp_old, 0.0
        )
        rise_anom = temp_rise - np.median(temp_rise)
        self._soc_ring[self._ring_i % _SOC_RING] = est.soc
        self._temp_ring[self._ring_i % _SOC_RING] = cell_temp

        # R0 trend: normalize out temperature (Arrhenius at the module
        # NTC) so only genuine impedance change remains
        arrh = np.exp(self.p.ea_over_r * (
            1.0 / (np.nan_to_num(cell_temp, nan=self.p.t_ref_c) + 273.15)
            - 1.0 / (self.p.t_ref_c + 273.15)
        ))
        r0_norm = est.r0 / arrh
        if self._r0_smooth is None:
            self._r0_smooth = r0_norm.copy()
        a = self.tun.r0_smooth_alpha
        self._r0_smooth = (1.0 - a) * self._r0_smooth + a * r0_norm
        # Values enter the trend ring only once the filter is confident
        # about that cell's R0 — a ratio must never straddle the cold-boot
        # convergence, where the estimate's walk from nominal to per-cell
        # truth is monotone "growth" indistinguishable from aging
        r0_std = est.bank.std()[:, 3]
        temp_valid = np.nan_to_num(cell_temp, nan=self.p.t_ref_c) \
            < self.tun.degradation_max_temp_c
        confident = (r0_std < self.tun.r0_confident_std) & temp_valid
        r0_oldest = self._r0_ring[(self._ring_i + 1) % _R0_RING]
        r0_mid = self._r0_ring[(self._ring_i + 1 + _R0_RING // 2) % _R0_RING]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_end = np.where(
                np.isfinite(r0_oldest), self._r0_smooth / r0_oldest, 1.0
            )
            ratio_mid = np.where(
                np.isfinite(r0_oldest) & np.isfinite(r0_mid),
                r0_mid / r0_oldest, 1.0,
            )
        r0_rel = ratio_end - np.median(ratio_end)
        r0_rel_mid = ratio_mid - np.median(ratio_mid)
        self._r0_ring[self._ring_i % _R0_RING] = np.where(
            confident, self._r0_smooth, np.nan
        )
        self._ring_i += 1
        self.n_steps += 1

        # ------------------------------------------------------- classify
        new: list[Diagnosis] = []
        ready = self.n_steps > _SOC_RING  # rings warm

        # Internal short: unexplained SOC drain + (negative-jump onset or
        # real local heating), debounced
        pending_neg = np.isfinite(self._pending_t) & (self._pending_sign < 0)
        drain = slew_anom < tun.drain_slew
        hot = rise_anom > tun.hot_rise_k
        cond = ready & drain & (pending_neg | hot) & ~quarantined
        self._short_streak[cond] += 1
        self._short_streak[~cond] = 0
        for cell in np.flatnonzero(self._short_streak >= tun.streak_n):
            self._diagnose(new, cell, "internal_short", "hard_short", CRITICAL, t, (
                f"SOC draining {-slew_anom[cell] * 3600 * 100:.1f} %/h vs fleet, "
                f"module +{rise_anom[cell]:.1f} K/45s, "
                f"innovation jump {'yes' if pending_neg[cell] else 'no'}"
            ))

        # Sensor freeze: reports stuck while the fleet's reports move
        frozen = ready & (
            (self._consec_same >= tun.freeze_reports)
            & (self._fleet_delta_ewma > tun.freeze_fleet_mv * 1e-3)
            & ~quarantined
        )
        for cell in np.flatnonzero(frozen):
            self._diagnose(new, cell, "sensor_fault", "frozen", WARNING, t, (
                f"{self._consec_same[cell]} identical reports while fleet "
                f"moves {1e3 * self._fleet_delta_ewma:.1f} mV/report"
            ))

        # Sensor offset: a jump opened a window; by confirm time the
        # filter has absorbed the bias as a STEP in this cell's
        # fleet-relative SOC that then settled — no continuing drain, no
        # heat. (A short reaches the drain rule long before this fires.)
        soc_step = np.abs(soc_rel - self._soc_rel_at_jump)
        offset = ready & np.isfinite(self._pending_t) & ~quarantined
        offset &= (t - np.nan_to_num(self._pending_t, nan=np.inf)
                   ) > tun.offset_confirm_s
        offset &= np.nan_to_num(soc_step) > tun.offset_soc_step
        offset &= np.abs(slew_anom) < (-tun.drain_slew / 2)
        offset &= rise_anom < tun.hot_rise_k
        for cell in np.flatnonzero(offset):
            self._diagnose(new, cell, "sensor_fault", "offset", WARNING, t, (
                f"reported voltage stepped this cell's fleet-relative SOC "
                f"by {100 * soc_step[cell]:.1f}% in "
                f"{t - self._pending_t[cell]:.0f} s, then settled — no "
                "coulomb or thermal explanation"
            ))

        # Accelerated degradation: temperature-normalized R0 growing over
        # minutes vs the fleet (ring entries are already confidence-gated;
        # suppressed entirely while the pack is in a critical alarm state)
        degr = ready & (self.n_steps > _R0_RING) & ~quarantined & confident
        degr &= not alarm
        degr &= (r0_rel > tun.degradation_rel) & (
            r0_rel_mid > tun.degradation_rel_mid
        )
        self._degr_streak[degr] += 1
        self._degr_streak[~degr] = 0
        for cell in np.flatnonzero(self._degr_streak >= tun.degradation_streak):
            self._diagnose(new, cell, "degradation", "impedance_growth",
                           MAINTENANCE, t, (
                f"R0 estimate grew {100 * (ratio_end[cell] - 1):.1f}% in 10 "
                f"min vs fleet median {100 * (np.median(ratio_end) - 1):.1f}%,"
                " monotone across the window"
            ))

        return new

    # -------------------------------------------------------------- helpers

    def _diagnose(self, new, cell, kind, subtype, severity, t, detail) -> None:
        cell = int(cell)
        if cell in self.diagnoses:
            return
        d = Diagnosis(cell, kind, subtype, severity, t, detail)
        self.diagnoses[cell] = d
        new.append(d)

    def _module_temps(self, tel: Telemetry) -> np.ndarray:
        finite = np.isfinite(tel.t_mod)
        cnt = finite.sum(axis=1)
        summed = np.where(finite, tel.t_mod, 0.0).sum(axis=1)
        t_mod = np.where(cnt > 0, summed / np.maximum(cnt, 1), np.nan)
        return t_mod[self.module_of]
