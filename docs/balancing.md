# Optimization-based active balancing

## Topology

Bidirectional cell-to-pack DC-DC (flyback per cell onto a shared
isolated balancing rail), 1 A converter rating, ~94% per conversion leg
(≈88% round trip). Any cell can be drained or charged independently —
the topology that makes optimization meaningful. A 150 mA passive
bleed-to-minimum baseline provides the comparison.

The plant model ([balancing/plant.py](../balancing/plant.py)) enforces
the physics the schedule must respect: converter current limits and
instantaneous rail power balance (the rail stores nothing, so injected
power must be funded by drained power at every moment; infeasible
schedules get their injections scaled and the event counted). Converter
losses are integrated in Wh.

## Controller: receding-horizon LP

Every 60 s the controller ([balancing/optimizer.py](../balancing/optimizer.py))
reads the **UKF estimator's** SOC and capacity — never ground truth —
and solves, with per-cell drain/charge variables `d_k, c_k ≥ 0`:

```
min   Σ_k s_k  +  λ_loss (1-η²) Σ_k v̂_k d_k        (deviation + losses)
s.t.  soc'_k = soc_k − (d_k − c_k)·Δt/(3600 q̂_k)    (per-cell dynamics)
      s_k ≥ |soc'_k − m*|                            (slack pair)
      d_k + c_k ≤ i_max,k                            (converter rating —
                                                      per-cell vector =
                                                      thermal-derate hook)
      0.94·η·Σ v̂_k d_k = (1/η)·Σ v̂_k c_k            (rail balance + margin)
```

`m*` is the charge-weighted mean SOC (what a lossless shuttle
conserves). Greedy per-window deviation minimization under the true
hardware constraints is the max-descent approximation of time-optimal
balancing for this linear system; `λ_loss` prices converter losses so
the LP never churns charge pointlessly (it must stay below the marginal
deviation gain per amp-window, ~0.014, or the LP refuses to act).

Two implementation notes that came from validation:

- **Rail planning margin (6%)**: the LP prices cells at estimated OCV
  but the hardware balances real terminal voltages; a cell drained at
  1 A sags ~2% below OCV and an injected cell rides ~2% above, so
  without margin the rail intermittently overdraws (3300+ scaling
  events in the first run; 0 after).
- **Estimator awareness**: balancing currents bypass the pack current
  sensor, so the controller feeds its own commands to `PackEstimator`
  (as a real BMS does) — otherwise the filters mis-attribute the
  balancing Ah and the LP chases corrupted estimates.

Solved with scipy HiGHS: 144 variables, ~1 ms per solve.

## Phase 4 validation (results/phase4/)

Aged pack, initial true SOC spread 10.5%, balanced at rest, controller
cold-boots its estimator (50% flat) and runs purely on telemetry.
6/6 checks:

| metric | active (LP, 1 A DC-DC) | passive (150 mA bleed) |
|---|---|---|
| time to <1% spread | **473 s (~8 min)** | 5667 s (~94 min) — 12× slower |
| energy lost | **1.03 Wh** | 21.75 Wh — 21× more |
| final spread | 0.97% | 0.90% |

1.50 Ah of charge moved; estimator SOC RMSE 0.54% throughout (the LP
never saw truth); 0 rail violations across 9 solves; hardware limits
respected. The schedule heatmap (`balancing_schedule.png`) shows the
optimizer saturating the extreme cells at ±1 A early, then tapering as
the pack converges.
