# Phase 2 — pack + comms validation

6 modules x 8s = 48 cells (~178 V nominal); bus 20 frames/s -> full-pack refresh 0.90 s.

Per-cycle results:

| cycle | delivered [Ah] | SOC spread bottom | SOC spread top |
|---|---|---|---|
| 1 | 2.399 | 7.03% | 0.69% |
| 2 | 2.359 | 7.24% | 1.34% |
| 3 | 2.352 | 7.45% | 2.00% |

```
[PASS] Capacity spread sigma 1.73% ~= configured 1.5%, all 48 cells unique
[PASS] Usable capacity clipped by weakest cell: delivered 2.399 Ah vs weakest 2.427 Ah (mean cell 2.501)
[PASS] Voltage spread amplifies at the knee: 303 mV at cutoff vs 21 mV mid-discharge (x14.2)
[PASS] Bottom SOC spread reflects capacity spread (7.03% > 3%)
[PASS] Top-of-charge SOC spread grows with cycling (0.69% -> 2.00%)
[PASS] Telemetry staleness bounded by round-robin period (max age 0.80 s <= 0.90 s + tick)
[PASS] Reported-vs-true RMS error 1.6 mV consistent with noise + quantization + staleness (< 8 mV)
[PASS] Cell 0 updated ~29 times in 30 s (expected ~33 at 0.9 s refresh)
```

Result: 8/8 checks passed.