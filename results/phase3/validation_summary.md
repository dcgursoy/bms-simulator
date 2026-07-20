# Phase 3 — SOC/SOH estimation validation

Pre-aged 48-cell pack, ~2 h urban drive cycle, cold boot at 50% SOC vs true 95%. Estimators see only bandwidth-limited telemetry (0.9 s refresh, noisy, quantized).

| metric | UKF | EKF |
|---|---|---|
| convergence to <2% [s] | 2 | 58 |
| SOC RMSE (steady) | 0.44% | 0.71% |
| SOC max abs err (steady) | 1.44% | 2.08% |
| R0 error init -> final | 9.1% -> 0.8% | 9.1% -> 0.8% |
| capacity error init -> final | 7.8% -> 1.7% | 7.8% -> 3.8% |
| compute [us/cell/step] | 25 | 18 |

Coulomb-counting baseline (wrong init, nominal capacity): 41.71% RMSE.

```
[PASS] UKF cold-boot convergence: 45% initial error -> <2% in 2 s (< 900 s)
[PASS] UKF SOC RMSE 0.44% < 1% (steady state, all 48 cells)
[PASS] EKF SOC RMSE 0.71% < 2%
[PASS] R0 (SOH) adaptation: UKF error 9.1% -> 0.8% (< 5%)
[PASS] Capacity (SOH) adaptation: UKF error 7.8% -> 1.7% (at least halved)
[PASS] Coulomb-counting baseline 41.71% RMSE is >= 5x worse than UKF 0.44%
```

Result: 6/6 checks passed.