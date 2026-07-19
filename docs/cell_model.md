# Cell model — second-order Thevenin equivalent circuit

## Structure

Each cell is a second-order Thevenin (2-RC) equivalent-circuit model:

```
 OCV(soc) + h ──[R0]──┬──[ R1 ∥ C1 ]──┬──[ R2 ∥ C2 ]──○ terminal
                        fast (~30 s)     slow (~600 s)
```

Sign convention: current `i > 0` discharges the cell.

**Terminal voltage**

$$v_t = \mathrm{OCV}(z) + h - i R_0 - v_{RC1} - v_{RC2}$$

**Continuous-time state equations** (per cell)

$$\dot z = -\frac{\eta\, i}{3600\, Q_\mathrm{eff}(T, \mathrm{age})}$$

$$\dot v_{RC1} = -\frac{v_{RC1}}{R_1 C_1} + \frac{i}{C_1},
\qquad
\dot v_{RC2} = -\frac{v_{RC2}}{R_2 C_2} + \frac{i}{C_2}$$

with coulombic efficiency $\eta = \eta_\mathrm{chg} < 1$ while charging and
$\eta = 1$ while discharging.

The implementation advances all states with the **exact zero-order-hold
discretization** (closed-form response to constant current over the step),
so the simulation is stable and accurate for any step size — important
later when the dashboard runs the pack faster than real time.

## OCV curve

`model/ocv.py` defines OCV(SOC) for a 3.0–4.2 V graphite/NMC cell as a
shape-preserving PCHIP interpolant through anchor points styled after
published NMC data. PCHIP (vs a polynomial fit) preserves monotonicity
exactly and adds no ringing on the flat mid-SOC plateau — which matters
because the estimators differentiate this curve, and spurious slope sign
changes would corrupt the SOC innovation.

## Hysteresis (Plett one-state model)

Graphite cells relax to a different open-circuit voltage after charging
vs after discharging. The hysteresis state $h$ obeys

$$h_{k+1} = f\, h_k - (1 - f)\,\mathrm{sgn}(i)\, M,
\qquad
f = \exp\!\left(-\frac{\gamma\, |i|\, \Delta t}{3600\, Q_\mathrm{eff}}\right)$$

so $h \to -M$ while discharging, $h \to +M$ while charging, converging
over roughly $1/\gamma \approx 2\%$ SOC of throughput, and $f = 1$ at
rest — hysteresis *holds* between pulses, as in real cells. Nominal
$M = 12$ mV (NMC-scale; LFP would be several times larger).

## Temperature dependence

Cell temperature is an input to the electrical model (Phase 5's thermal
model closes the loop):

- **Resistances** follow an Arrhenius law,
  $R(T) = R\,\exp\!\left[\frac{E_a}{R_g}\left(\frac{1}{T} - \frac{1}{T_\mathrm{ref}}\right)\right]$
  with $E_a/R_g = 3000\,\mathrm{K}$ — a cell at 0 °C is ~2.5× more
  resistive than at 25 °C.
- **Usable capacity** shrinks by 0.6%/°C below 25 °C (floored at 60%),
  capturing slower diffusion in the cold.

## Aging / capacity fade

Aging is driven by cumulative Ah throughput, expressed as equivalent
full cycles $\mathrm{EFC} = \frac{\text{Ah throughput}}{2 Q_\mathrm{nom}}$:

- **Capacity fade** (power law): $Q$ loses
  $20\% \cdot (\mathrm{EFC}/1000)^{0.8}$ of nameplate (capped at 40%).
- **Impedance growth**: $R_0$ grows $35\%$ per 1000 EFC (linear).

A per-cell `aging_accel` multiplier scales the *effective* EFC, which is
how Phase 6 injects an accelerated-degradation fault into a single cell.

## Nominal parameters (~2.5 Ah NMC 18650 @ 25 °C)

| Parameter | Value | Notes |
|---|---|---|
| $Q_\mathrm{nom}$ | 2.5 Ah | 1C = 2.5 A |
| $R_0$ | 25 mΩ | ohmic |
| $R_1, C_1$ | 15 mΩ, 2 kF | $\tau_1 = 30$ s (charge transfer / fast diffusion) |
| $R_2, C_2$ | 10 mΩ, 60 kF | $\tau_2 = 600$ s (slow solid-phase diffusion) |
| $M, \gamma$ | 12 mV, 50 | hysteresis magnitude / convergence rate |
| $\eta_\mathrm{chg}$ | 0.995 | coulombic efficiency (charge) |
| $v_\mathrm{min}, v_\mathrm{max}$ | 3.0 V, 4.2 V | operating window |

## Phase 1 validation

`scripts/validate_cell.py` exercises one cell in isolation
(plots + `validation_summary.md` in `results/phase1/`):

1. **Discharge family** — 0.5C/1C/2C @ 25 °C and 1C @ 0 °C: delivered
   capacity decreases with rate and with cold; sag grows with rate.
2. **HPPC pulse** — the instantaneous voltage step across the pulse edge
   recovers $R_0$ to <5%; relaxation shows both time constants.
3. **C/25 quasi-static loop** — charge/discharge branches bracket the
   true OCV curve with a mid-SOC gap of $2M + 2iR_\mathrm{tot}$.
4. **Capacity fade** — measured 1C delivered capacity at fast-forwarded
   aged states tracks the fade law (relative fade within 2%).
