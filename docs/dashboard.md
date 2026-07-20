# Live dashboard (FastAPI + WebSocket)

## Run it

```bash
.venv/Scripts/python dashboard/server.py     # Windows
# open http://127.0.0.1:8420
```

## What it is

A mission-console view of the full simulator running live: the complete
Phase 1–6 stack (48-cell pack + electro-thermal loop + bandwidth-limited
bus + UKF estimator + LP balancer + residual fault detector + safety
policy) steps in a background thread at a selectable multiple of real
time (1× / 5× / 20×), streaming ~5 Hz JSON snapshots over a WebSocket to
a fully self-contained canvas frontend (no external JS/CSS — works
offline).

## Architecture

- [dashboard/engine.py](../dashboard/engine.py) — `SimEngine`: the
  closed loop per 0.1 s tick mirrors the Phase 6 validation runner plus
  the balancer. The policy limits requested pack current; the balancer
  replans every 60 s from estimates only and stands down when the
  contactor opens; fault aux currents/heat feed the truth; sensor
  faults corrupt telemetry after the bus; the estimator knows balancer
  commands but never faults. Thread-safe snapshots + a command queue.
- [dashboard/server.py](../dashboard/server.py) — FastAPI: static
  frontend + one WebSocket streaming snapshots and accepting commands.
- [dashboard/static/](../dashboard/static/) — hand-rolled canvas UI:
  heatmaps, SOC/balancing bars, rolling line charts, hover readouts.
  The categorical series palette (estimate `#4493f8` / reported
  `#cd750d` on the dark surface) passes all six colorblind-safety and
  contrast checks; sequential single-hue ramps for magnitude maps, a
  blue↔orange diverging ramp for balancing polarity, and status colors
  reserved for severity (always with text, never color-alone).

## Panels

- **Pack grid** (6 modules × 8s): per-cell voltage / SOC / SOH-R0 /
  SOH-capacity / balancing current, with flag borders from the safety
  policy. Click a cell to select it everywhere.
- **Thermal map**: true cell temperatures (the truth the estimator
  can't see — module NTCs are its only window).
- **SOC & balancing bars**: 48 estimated SOCs converging under the LP
  balancer, bar color = drain/charge/idle, dashed charge-weighted mean.
- **Traces** (selected cell): truth vs bus-reported voltage (the
  quantized 0.9 s staircase is clearly visible at 1×), truth vs UKF
  SOC, and pack current (applied vs requested — derates show as gaps).
- **Fault injection panel**: internal short / rapid aging / freeze
  sensor / offset sensor on the selected cell, live diagnosis cards
  (severity-colored), and the safety event log.
- **Header**: pack voltage/current, mean estimated SOC, true spread,
  current limit, balancing energy loss, contactor state; load mode
  (drive cycle / rest / CC-CV charge / 2C discharge), speed, balancer
  toggle, reset.

## A note from live soak testing

Hour-long unattended soaks surfaced false-positive degradation flags
that the 15-minute validation scenarios never hit, and drove two
hardenings: (1) the estimator's R0 excitation gate was raised to 0.5C
so balancing-scale currents — weak R0 signal, full model mismatch in
the innovations — stop random-walking r̂0 into the trend detector; and
(2) the degradation rule became a persistence test (fleet-relative
growth above threshold for ≥80% of a sliding 15 min window), because
healthy-cell excursions can spike past any single-window threshold but
mean-revert, while true aging sustains. All Phase 3/6 validation suites
still pass (17/17, 6/6) after both changes.
