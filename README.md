# bms-simulator

A multi-cell Battery Management System (BMS) simulator: electrochemical
cell modeling, adaptive nonlinear state estimation (SOC + SOH),
optimization-based active balancing, coupled electro-thermal simulation,
and model-based fault detection with a live interactive dashboard.

**Status: work in progress** — built in phases; this README is finalized
in Phase 8.

| Phase | Scope | Status |
|---|---|---|
| 1 | Second-order Thevenin cell model (temperature, hysteresis, aging) | ✅ |
| 2 | 48-cell pack (6 modules × 8s) + bandwidth-limited BMS bus | ✅ |
| 3 | UKF (primary) + EKF joint SOC/SOH estimation | ✅ |
| 4 | Optimization-based active balancing | ✅ |
| 5 | Coupled electro-thermal model | — |
| 6 | Model-based fault detection + safety policy | — |
| 7 | Live dashboard (FastAPI + WebSocket) | — |
| 8 | Results, analysis, portfolio README | — |

## Layout

```
model/            cell + (later) pack and thermal models
estimation/       UKF/EKF SOC+SOH estimators
balancing/        optimization-based active balancing
fault_detection/  residual-based fault detection + safety policy
dashboard/        FastAPI + WebSocket live dashboard
scripts/          per-phase validation and demo runners
docs/             model and design notes
results/          generated plots and metrics (committed evidence)
```

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
.venv/Scripts/python scripts/validate_cell.py   # Phase 1 validation
```

See [docs/cell_model.md](docs/cell_model.md) for the cell model math.
