# Model-based fault detection, diagnosis, and safety response

## Approach

The detector ([fault_detection/detector.py](../fault_detection/detector.py))
runs at the filter cadence (1 Hz) on the pack controller and consumes
only what a real BMS has: the UKF's voltage innovations (measurement −
model prediction) with their predicted variances, the estimator's
SOC/R0 outputs, and raw telemetry. Every feature is **fleet-relative**
(vs the 48-cell median), so pack-wide effects — load, ambient, uniform
heating — cancel by construction.

### Fault signature table

| fault | voltage residual | SOC (fleet-rel.) | temp | reports | severity |
|---|---|---|---|---|---|
| internal short | negative jump at onset | **drains persistently** | rises | live | critical |
| sensor freeze | grows as truth drifts | normal | flat | **frozen** | warning |
| sensor offset | jump, then absorbed | **steps, then settles** | flat | live | warning |
| accel. degradation | none (filter adapts) | normal | flat | live | maintenance |

Key discriminator: a short and a sensor offset both open an
innovation-jump window, but the short's SOC drain *persists* while the
offset's absorbed SOC shift *settles* — and only the short eventually
heats the module.

### Safety policy ([fault_detection/policy.py](../fault_detection/policy.py))

One-way ratchet: **critical** → contactor opened (simulated shutdown;
the internal short keeps burning — a contactor cannot stop it — but the
load is isolated); **warning** → sensor channel quarantined (the
estimator propagates that cell open-loop) + pack derated to 50%;
**maintenance** → service flag + 75% derate.

## Non-obvious lessons the validation forced

These came from actual false-positive hunts, not theory:

1. **The filter eats the evidence.** The UKF absorbs a 60 mV sensor
   offset in seconds, so "sustained residual" rules never fire. Detect
   the *step* it leaves in fleet-relative SOC instead — and snapshot the
   pre-update SOC, because the update that records the jump has already
   absorbed part of the bias.
2. **Trend windows must not straddle cold boot.** R0 estimates walking
   from nominal init to per-cell truth look exactly like monotone aging;
   trend-ring entries are gated on the filter's own R0 confidence
   (converged std 1.4–1.9 mΩ vs 4+ mΩ during convergence).
3. **Temperature-normalize SOH trends** (Arrhenius at the module NTC) —
   a warming pack's spatially uneven cooling otherwise masquerades as
   impedance change. But **suspend trending during thermal events**: post-
   shutdown, the burning cell heats its neighbors' NTCs while the
   excitation-gated R0 estimates cannot follow, and the normalization
   fabricates 40–100% apparent growth.
4. **Healthy estimates wander — and short scenarios understate it.**
   Model-mismatch excursions reach ~5–7% per 10 min window (hour-long
   dashboard soaks found what 15-minute validation scenarios missed)
   but they mean-revert, while true aging sustains its rate
   indefinitely. The decisive discriminator is *persistence*: growth
   above threshold for ≥80% of a sliding 15 min window. A consecutive-
   streak test that long is too brittle (one flickering second resets
   it) — it must be a windowed duty-cycle test.

## Validation (results/phase6/, 17/17 checks)

Five closed-loop scenarios (truth + thermal + bus + UKF + detector +
policy). Detection performance:

| scenario | injected | detected as | latency | response | FPs |
|---|---|---|---|---|---|
| clean (15 min) | — | — | — | — | **0** |
| short | 0.2 Ω on cell 20 | internal_short | **7 s** | shutdown | 0 |
| sensor freeze | stuck AFE, cell 35 | sensor_fault/frozen | **12 s** | quarantine + 50% | 0 |
| sensor offset | +60 mV, cell 8 | sensor_fault/offset | **59 s** | quarantine + 50% | 0 |
| degradation | 2000× aging, cell 42 | degradation | **1302 s** | 75% + service flag | 0 |

**4/4 fault types correctly classified, 0 false positives** across all
scenarios (~1.4 cell-hours × 48 cells). The short is caught by the
voltage/SOC path minutes before meaningful heat develops — early
detection is exactly the value of model-based residuals over plain
temperature alarms.
