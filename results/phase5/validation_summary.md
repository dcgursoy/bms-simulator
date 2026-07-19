# Phase 5 — electro-thermal validation

3C stress: peak 48.1 degC, interior-corner gradient 4.3 K, delivered 2.135 Ah closed-loop vs 2.044 Ah open-loop. Internal 0.2-ohm short: cell peaks 120 degC, neighbors +21.7 K, far corner +0.0 K.

```
[PASS] 3C peak temperature 48.1 degC is plausible for an air-cooled pack (40-70 degC)
[PASS] Spatial gradient at end of discharge: interior max - corner min = 4.3 K (> 3 K)
[PASS] Electro-thermal feedback: every hot cell ends less resistive than it started (mean R0 25.0 -> 12.8 mOhm)
[PASS] Self-heating recovers capacity at 3C: 2.135 Ah closed-loop > 2.044 Ah at fixed 25 degC
[PASS] Shorted cell reaches 120 degC (> 70 degC thermal alarm territory)
[PASS] Heat propagates to neighbors (+21.7 K > 5 K) but stays local (far corner +0.0 K < half the neighbor rise)
[PASS] Thermal derate: shorted cell limited to 0.00 A, 43 cool cells keep the full 1.00 A
```

Result: 7/7 checks passed.