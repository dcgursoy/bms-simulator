# Coupled electro-thermal model

## Thermal model ([model/thermal.py](../model/thermal.py))

Each cell is a lumped thermal mass (C_th = 40 J/K, 18650-class) on the
pack's physical 6 × 8 grid (module = row) with three heat paths:

- **Generation** ([model/cell.py](../model/cell.py) `heat_generation_w`):
  irreversible I²R dissipation in R0/R1/R2 (using the RC branch states,
  so pulse heating is captured correctly) plus reversible entropic heat
  `−i·T·(dOCV/dT)` with dOCV/dT = −0.15 mV/K — exothermic on discharge,
  endothermic on charge.
- **Conduction** between orthogonal neighbors (0.2 W/K) via the grid's
  graph Laplacian — one dense 48×48 matrix-vector product per step.
- **Convection** to ambient, with more cooling per exposed enclosure
  side (interior 0.06 W/K, +0.03 per open side): interior cells run
  hottest, corners coolest, which produces the pack's characteristic
  center-hot thermal map.

Explicit Euler at the 0.1 s tick is comfortably stable (fastest thermal
time constant ≈ 45 s).

## Closing the loop

The simulation feeds `ThermalModel.temps` back into the electrical
model every step: Arrhenius resistances drop as cells heat, cold
capacity derating lifts. This is real feedback, not decoration —
measured at 3C discharge:

- mean R0 falls 25.0 → 12.8 mΩ as the pack warms 25 → 48 °C;
- the closed-loop pack delivers **2.135 Ah vs 2.044 Ah** with
  temperatures pinned at 25 °C (+4.5%): self-heating reduces sag and
  postpones the weakest cell's cutoff.

## Stress scenarios (results/phase5/, 7/7 checks)

**3C sustained discharge** (air-cooled, 25 °C ambient): peak 48.1 °C,
interior-corner gradient 4.3 K at cutoff, thermal map shows the
expected center-hot ring structure.

**Internal soft short** (0.2 Ω across one interior cell of a resting
pack, injected at t = 60 s): the cell discharges internally at ~18 A
(~7C), dumping ~60 W into its own thermal mass — it passes 70 °C within
a minute and reaches the 120 °C model-validity/alarm ceiling at
~3.6 min. Adjacent cells rise +21.7 K by conduction while the far
corner stays at ambient: thermal risk propagates, but locally on these
timescales — exactly the signature Phase 6's detection must catch
*early* (the voltage anomaly appears seconds after injection, the
neighbor heating tens of seconds later). No runaway chemistry is
modeled; this is a pure simulation exercise.

**Thermal derating** (`balancing.thermal_derate`): the Phase 4 LP's
per-cell current-limit vector, full 1 A below 40 °C linearly derated to
0 A at 55 °C. Applied to the short scenario's temperature field: the
shorted cell's balancing limit collapses to 0 A while all cool cells
keep their full rating — the LP plans around hot cells instead of
pushing current through them.
