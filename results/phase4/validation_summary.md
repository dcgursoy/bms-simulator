# Phase 4 — active balancing validation

Aged pack, initial true SOC spread 10.5%, balanced at rest by controllers running purely on UKF estimates.

| metric | active (LP, 1 A DC-DC) | passive (150 mA bleed) |
|---|---|---|
| time to <1% spread | 473 s | 5667 s |
| energy lost | 1.03 Wh | 21.75 Wh |
| final spread | 0.97% | 0.90% |

```
[PASS] Active balancing: true spread 10.5% -> <1% in 473 s (8 min)
[PASS] LP consumed accurate estimates: estimator SOC RMSE 0.54% < 1% during balancing
[PASS] Passive is 12.0x slower (5667 s vs 473 s)
[PASS] Active loses 1.03 Wh vs passive 21.75 Wh burned (5%)
[PASS] Rail never overdrawn (0 scale events; 9 LP solves, 0 failures)
[PASS] Hardware ratings respected (max |i| active 1.00 A <= 1 A, passive 0.15 A <= 0.15 A)
```

Result: 6/6 checks passed.