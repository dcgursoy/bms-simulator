# Pack architecture and BMS telemetry bus

## Pack: 6 modules × 8s = 48s1p (~178 V nominal, 2.5 Ah)

All 48 cells sit in one series string ([model/pack.py](../model/pack.py)):
every cell carries the pack current (plus per-cell auxiliary current
injected later by the active balancer and fault models). Pack voltage is
the sum of cell voltages, and usable pack capacity is set by whichever
cell hits a voltage limit first — the core reason balancing exists.

Cells are physically arranged in a 6 × 8 grid (module = row), which is
the adjacency map Phase 5's thermal conduction model uses.

### Manufacturing variation (seeded, reproducible)

| Parameter | 1σ spread | Consequence |
|---|---|---|
| capacity | 1.5% | weakest cell clips usable pack capacity; SOC swing differs per cell |
| R0 | 4% | sag spread, I²R heat spread (Phase 5) |
| R1/C1/R2/C2 | 6% | dynamic response spread |
| coulombic *loss* (1−η) | 30% (lognormal) | monotonic top-of-charge SOC divergence, cycle over cycle |
| self-discharge | 50% (lognormal) | slow calendar divergence |
| aging rate | 8% (lognormal) | differential SOH fade over pack life |

## Telemetry bus ([model/comms.py](../model/comms.py))

Mirrors a real distributed BMS:

- **Module monitors** (LTC68xx-class AFE per module) digitize 8 cell
  voltages (σ = 1.5 mV noise, 1 mV LSB) and 2 NTC temperatures
  (σ = 0.3 °C, 0.25 °C LSB). Per-cell temperature is *not* observable —
  NTC A covers the module's first 4 cells, NTC B the last 4.
- **Bandwidth-limited bus** (CAN-class): 20 frames/s granted round-robin;
  a module snapshot costs 3 frames (voltages 0–3, voltages 4–7,
  temps/status). Full-pack refresh = 6·3/20 = **0.9 s**, so every datum
  the pack controller holds is up to 0.9 s stale. Frames sample live
  values at grant time — staleness comes from the schedule itself.
- **Pack current sensor** is wired directly to the controller (σ = 25 mA,
  10 mA LSB) and sampled every tick, as in real packs.

The controller-side state lives in `Telemetry` (last value + timestamp
per channel). **Estimators consume only this object, never ground
truth** — that is what makes the bandwidth constraint meaningful for
Phase 3.

## Phase 2 validation (results/phase2/)

3 CC-CV cycles (1C discharge → rest → C/2 CC-CV charge → rest) with the
protocol controller running purely on bus telemetry. 8/8 checks passed:

- Delivered capacity **2.399 Ah** vs weakest cell 2.427 Ah / mean cell
  2.501 Ah — weakest-cell clipping, ~4% of pack capacity already lost to
  variation with zero balancing.
- Voltage spread amplifies **×14** at the discharge knee (303 mV at
  cutoff vs 21 mV mid-discharge).
- Bottom-of-discharge SOC spread ≈ **7%** (capacity spread); top-of-charge
  spread grows **0.69% → 2.00%** over 3 cycles (coulombic-efficiency
  divergence) and delivered capacity falls cycle-over-cycle
  (2.399 → 2.352 Ah) — the imbalance Phase 4's balancer must fix.
- Telemetry: max staleness 0.80 s ≤ 0.9 s round-robin bound; 1.6 mV RMS
  reported-vs-true error, consistent with noise + quantization.
