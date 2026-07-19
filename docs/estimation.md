# Adaptive state estimation — joint SOC + SOH (UKF primary, EKF comparison)

## Problem

The pack controller must know every cell's SOC and SOH, but it only
sees ([docs/pack_comms.md](pack_comms.md)): per-cell voltage reports that
are noisy (1.5 mV), quantized (1 mV) and up to 0.9 s stale; a 10 Hz pack
current sensor; and 2 NTC temperatures per module. Cells have unknown
per-cell capacity fade and impedance growth (aging) on top of unknown
manufacturing variation.

## State vector and models

Per cell (5 states, one filter per cell, batched over all 48):

```
x = [ soc, v_rc1, v_rc2, r0, q_ah ]
```

`r0` and `q_ah` are random walks whose online adaptation *is* the SOH
estimate — impedance growth and capacity fade are the two canonical
aging signatures. Joint (single-filter) estimation is used instead of a
dual/cascaded scheme: with 5 states the extra cost is trivial, and the
SOC–capacity cross-covariance the joint filter maintains is exactly the
mechanism that lets coulomb/OCV consistency pull the capacity estimate.

Process model = the nominal 2-RC cell model (Arrhenius-scaled at the
module NTC temperature); measurement model = terminal voltage
`OCV(soc) + h − i·r0 − v_rc1 − v_rc2`. Hysteresis `h` is not estimated:
it is propagated open-loop from measured current with nominal
parameters, removing an otherwise ~12 mV (≈2% SOC on the OCV plateau)
bias.

## Handling the bandwidth-limited bus

The filter bank runs at 1 Hz (a realistic BMS MCU budget). Between
steps the controller accumulates 10 Hz current samples; each step
predicts with the mean current, then applies measurement updates for
exactly those cells whose telemetry timestamp advanced. Each reported
voltage is paired with the current *at its sample time* (reconstructed
from a short current ring buffer) — pairing a stale voltage with live
current would inject `i(t) − i(t_sample)` transients into the R0
estimate.

Two robustness details that mattered:

- **Excitation gating**: R0 is unobservable at rest (`∂v/∂r0 = −i ≈ 0`),
  but a large innovation (e.g. a cold boot) can still shove it through
  weak cross-covariances. Below 0.2C the R0 gain row is zeroed.
- **Arbitrary-gain covariance update**: with a gated (suboptimal) gain,
  the optimal-gain shortcut `P −= p_yy K Kᵀ` is inconsistent and can
  break positive-definiteness. Both filters use the general form
  `P ← P − K P_xyᵀ − P_xy Kᵀ + p_yy K Kᵀ`, valid for any gain.

The measurement is scalar per cell, so no matrix inversion appears
anywhere; all linear algebra is batched `einsum` over the 48 cells.

## UKF vs EKF — the tradeoff, measured

The measurement nonlinearity is the OCV curve (flat plateau, steep
knees); the process nonlinearity is `soc ∝ i/q`. The EKF linearizes both
at the mean; the UKF (scaled unscented transform, α=0.5, β=2, κ=0)
samples 11 sigma points per cell through the full models — 2n+1 model
evaluations plus a Cholesky per step vs the EKF's single evaluation.

Measured on the Phase 3 scenario (pre-aged pack, 150–400 EFC/cell, ~2 h
urban drive cycle, cold boot at 50% SOC vs true 95%):

| metric | UKF | EKF |
|---|---|---|
| cold-boot convergence to <2% | **2 s** | 58 s |
| SOC RMSE (steady, all cells) | **0.44%** | 0.71% |
| SOC max abs error (steady) | **1.45%** | 2.10% |
| R0 error init → final | 9.1% → 0.8% | 9.1% → 0.8% |
| capacity error init → final | 7.8% → **1.7%** | 7.8% → 3.8% |
| compute | 23 µs/cell/step | 17 µs/cell/step |

(The capacity init error of 7.8% reflects initializing at nameplate
while true cells are faded; both filters must *learn* the fade.)

Interpretation: the EKF's cold-boot lag is its linearized update
overshooting through the OCV curve (the estimate rails at 100% SOC and
bounces); the UKF's sigma points span the uncertainty and land the
first update near the right SOC. Capacity converges ~2× tighter for the
UKF because sigma points carry the `1/q` process nonlinearity into the
cross-covariance. At 48 cells the UKF's extra cost is ~6 µs/cell/step —
irrelevant — so the UKF is the primary estimator; the EKF remains as
the cheap fallback and comparison baseline.

A coulomb-counting baseline (same wrong init, nominal capacity, noisy
current integration, no correction) lands at **41.7% RMSE** — the gap
between open-loop integration and model-based fusion.

## Validation

`scripts/validate_estimation.py`, 6/6 checks
(plots + `validation_summary.md` in `results/phase3/`):
cold-boot convergence, steady-state SOC accuracy for both filters, R0
and capacity adaptation, and the coulomb-counting comparison.
