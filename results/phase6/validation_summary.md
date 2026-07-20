# Phase 6 — fault detection validation

Closed loop: truth + thermal + bus + UKF estimator + residual detector + safety policy; the controller side sees only telemetry.

| scenario | injected fault | detected | latency | response |
|---|---|---|---|---|
| clean | — | 0 diagnoses | — | — |
| short | internal_short on cell 20 | yes | 7 s | shutdown |
| sensor_freeze | sensor_fault on cell 35 | yes | 12 s | derate to 50% |
| sensor_offset | sensor_fault on cell 8 | yes | 59 s | derate to 50% |
| degradation | degradation on cell 42 | yes | 563 s | derate to 75% |

**True positives 4/4, false positives 0 across 80 cell-hours.**

```
[PASS] [clean] no false positives over 15 min x 48 cells (0 diagnoses)
[PASS] [short] diagnosed internal_short on cell 20 (hard_short)
[PASS] [short] detection latency 7 s < 90 s bound
[PASS] [short] no other cell flagged (0 false positives)
[PASS] [short] safety response correct (contactor open, current limit 0%)
[PASS] [sensor_freeze] diagnosed sensor_fault on cell 35 (frozen)
[PASS] [sensor_freeze] detection latency 12 s < 60 s bound
[PASS] [sensor_freeze] no other cell flagged (0 false positives)
[PASS] [sensor_freeze] safety response correct (contactor closed, current limit 50%)
[PASS] [sensor_offset] diagnosed sensor_fault on cell 8 (offset)
[PASS] [sensor_offset] detection latency 59 s < 150 s bound
[PASS] [sensor_offset] no other cell flagged (0 false positives)
[PASS] [sensor_offset] safety response correct (contactor closed, current limit 50%)
[PASS] [degradation] diagnosed degradation on cell 42 (impedance_growth)
[PASS] [degradation] detection latency 563 s < 1200 s bound
[PASS] [degradation] no other cell flagged (0 false positives)
[PASS] [degradation] safety response correct (contactor closed, current limit 75%)
```

Result: 17/17 checks passed.