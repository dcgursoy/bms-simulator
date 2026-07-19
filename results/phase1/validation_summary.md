# Phase 1 — cell model validation

Nominal cell: 2.5 Ah NMC, R0=25 mOhm, tau1=30 s, tau2=600 s, M_hyst=12 mV

```
[PASS] Delivered capacity falls with C-rate (0.5C 2.484 > 1C 2.469 > 2C 2.426 Ah)
[PASS] Cold cuts delivered capacity (1C@0degC 2.038 < 1C@25degC 2.469 Ah)
[PASS] Pulse edge recovers R0 (25.07 mOhm vs true 25.00 mOhm)
[PASS] 10 s resistance 30.9 mOhm exceeds R0 (RC branches charging)
[PASS] Charge branch sits above discharge branch everywhere (min gap 33.0 mV)
[PASS] Mid-SOC gap 34.0 mV ~= expected 2M + 2iR = 34.0 mV
[PASS] Relative capacity at 800 EFC: measured 0.831 vs law 0.833
[PASS] Fade is monotonic in EFC
```

Result: 8/8 checks passed.