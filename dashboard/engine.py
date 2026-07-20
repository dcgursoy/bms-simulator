"""Live simulation engine behind the dashboard.

Runs the complete Phase 1-6 stack — pack + electro-thermal loop +
bandwidth-limited bus + fault injector + UKF estimator + LP balancer +
residual fault detector + safety policy — in a background thread at a
configurable sim-time speed multiple, exposing:

- thread-safe JSON-able snapshots for the WebSocket streamer,
- a command queue for the UI: fault injection on any cell, load mode
  (drive / rest / CC-CV charge / 2C discharge), speed, balancer toggle,
  and full reset.

The closed loop per 0.1 s tick mirrors scripts/validate_faults.py plus
the balancer: the policy limits the requested pack current, the
balancer runs on estimates only (and is stood down while the contactor
is open), fault aux currents and heat feed the truth, sensor faults
corrupt telemetry after the bus, and the estimator knows the balancer's
commands but never the faults.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from balancing import BalancerPlant, LPBalancer, thermal_derate
from estimation import PackEstimator
from fault_detection import FaultDetector, FaultInjector, SafetyPolicy
from model.cell import CellParams
from model.comms import BmsBus
from model.drive_cycle import synth_drive_cycle
from model.pack import Pack
from model.thermal import ThermalModel

DT = 0.1
WALL_SLICE_S = 0.05
MAX_TICKS_PER_SLICE = 400  # bound the catch-up burst so the lock breathes

MODES = ("drive", "rest", "charge", "discharge")


class SimEngine(threading.Thread):
    def __init__(self, speed: float = 5.0, seed: int = 2026):
        super().__init__(daemon=True, name="sim-engine")
        self.lock = threading.RLock()
        self.cmds: queue.Queue[dict] = queue.Queue()
        self.speed = float(speed)
        self.seed = int(seed)
        self.running = True
        self._resets = 0
        self._build()

    # ------------------------------------------------------------ lifecycle

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed + self._resets)
        self.params = CellParams()
        soc0 = np.clip(0.55 + rng.normal(0.0, 0.04, 48), 0.40, 0.70)
        self.pack = Pack(soc0=soc0)
        self.pack.cells.fast_forward_aging(rng.uniform(150.0, 400.0, 48))
        self.thermal = ThermalModel(
            self.pack.config.n_modules, self.pack.config.cells_per_module
        )
        self.bus = BmsBus(self.pack)
        self.injector = FaultInjector(self.pack)
        self.est = PackEstimator(48, self.pack.module_of, "ukf", soc0=0.5)
        self.balancer = LPBalancer(48)
        self.plant = BalancerPlant(48)
        self.detector = FaultDetector(48, self.pack.module_of)
        self.policy = SafetyPolicy(48)

        self.t = 0.0
        self.mode = "drive"
        self.balancer_on = True
        self.i_request = 0.0
        self.i_applied = 0.0
        self._seen_steps = 0
        self._bal_cmd = np.zeros(48)
        self._next_replan = 90.0  # give the estimator a warmup
        self._profile_rng = rng
        self._refill_profile()
        self._charge_finished = False
        self._auto_cycle = False  # set when drive auto-switches to charge
        self.ui_events: list[tuple[float, str]] = [(0.0, "simulation started")]

    def _refill_profile(self) -> None:
        self._profile = synth_drive_cycle(
            1200.0, DT, 0.7 * self.params.q_nom_ah,
            seed=int(self._profile_rng.integers(1 << 31)),
            rest_s=(20.0, 40.0, 20.0),
        )
        self._profile_i = 0

    # ------------------------------------------------------------- commands

    def command(self, msg: dict) -> None:
        self.cmds.put(msg)

    def _apply_command(self, m: dict) -> None:
        cmd = m.get("cmd")
        cell = int(m.get("cell", 0)) % 48
        if cmd == "inject":
            kind = m.get("kind")
            if kind == "short":
                self.injector.inject_short(cell, 0.2, self.t)
            elif kind == "freeze":
                self.injector.inject_sensor_freeze(cell, self.t)
            elif kind == "offset":
                self.injector.inject_sensor_offset(cell, 0.06, self.t)
            elif kind == "degradation":
                self.injector.inject_degradation(cell, 2000.0, self.t)
            self.ui_events.append(
                (self.t, f"INJECTED {kind} on cell {cell}"))
        elif cmd == "mode" and m.get("value") in MODES:
            self.mode = m["value"]
            self._charge_finished = False
            self.ui_events.append((self.t, f"load mode -> {self.mode}"))
        elif cmd == "speed":
            self.speed = float(np.clip(float(m.get("value", 5.0)), 0.5, 50.0))
        elif cmd == "balancer":
            self.balancer_on = bool(m.get("value", True))
            if not self.balancer_on:
                self._bal_cmd = np.zeros(48)
            self.ui_events.append(
                (self.t, f"balancer {'enabled' if self.balancer_on else 'disabled'}"))
        elif cmd == "reset":
            self._resets += 1
            self._build()

    # ------------------------------------------------------------ sim loop

    def _reported_extreme(self, fn) -> float:
        v = self.bus.telemetry.v
        return float(fn(v[np.isfinite(v)])) if np.isfinite(v).any() else float("nan")

    def _current_request(self) -> float:
        p = self.params
        if self.mode == "rest":
            return 0.0
        if self.mode == "drive":
            # Depleted pack: hand over to CC-CV charge so an unattended
            # demo cycles indefinitely instead of parking at cutoff
            vmin = self._reported_extreme(np.min)
            if np.isfinite(vmin) and vmin <= p.v_min + 0.02:
                self.mode = "charge"
                self._charge_finished = False
                self._auto_cycle = True
                self.ui_events.append(
                    (self.t, "pack depleted -> auto CC-CV charge"))
                return 0.0
            if self._profile_i >= len(self._profile):
                self._refill_profile()
            i = float(self._profile[self._profile_i])
            self._profile_i += 1
            return i
        if self.mode == "discharge":
            vmin = self._reported_extreme(np.min)
            if np.isnan(vmin) or vmin > p.v_min + 0.02:
                return 2.0 * p.q_nom_ah
            return 0.0
        # CC-CV charge on the reported max cell, then hold
        vmax = self._reported_extreme(np.max)
        if self._charge_finished or np.isnan(vmax):
            return 0.0
        err = p.v_max - vmax
        mag = float(np.clip(0.5 * p.q_nom_ah * err / 0.05, 0.0, 0.5 * p.q_nom_ah))
        if mag < p.q_nom_ah / 50.0:
            self._charge_finished = True
            self.ui_events.append((self.t, "CC-CV charge complete"))
            if self._auto_cycle:
                self._auto_cycle = False
                self.mode = "drive"
                self.ui_events.append((self.t, "auto-resuming drive cycle"))
            return 0.0
        return -mag

    def _tick(self) -> None:
        self.i_request = self._current_request()
        i_cmd = self.policy.limit_current(self.i_request)

        if self.balancer_on and not self.policy.contactor_open:
            if self.t + 1e-9 >= self._next_replan:
                temps_est = self.detector._module_temps(self.bus.telemetry)
                limits = thermal_derate(
                    self.balancer.i_max,
                    np.nan_to_num(temps_est, nan=self.params.t_ref_c),
                )
                self._bal_cmd = self.balancer.replan(
                    self.est.soc, self.est.q_ah, i_max_percell=limits
                )
                self._next_replan += self.balancer.window
            bal_applied = self.plant.apply(self._bal_cmd, self.pack.last_v, DT)
        else:
            bal_applied = np.zeros(48)

        aux_fault = self.injector.short_currents_a()
        aux = aux_fault + bal_applied
        self.pack.step(i_cmd, DT, self.thermal.temps, aux_current_a=aux)
        q = self.pack.cells.heat_generation_w(i_cmd + aux, self.thermal.temps)
        q += self.injector.short_heat_w(aux_fault)
        self.thermal.step(q, DT)
        self.t += DT
        self.i_applied = i_cmd
        self.bus.step(self.t, DT, self.thermal.temps)
        self.injector.corrupt_telemetry(self.bus.telemetry)
        self.est.tick(self.t, self.bus.telemetry, DT, aux_cmd_a=bal_applied)
        if self.est.n_steps > self._seen_steps:
            self._seen_steps = self.est.n_steps
            new = self.detector.step(
                self.t, self.bus.telemetry, self.est,
                alarm=self.policy.contactor_open,
            )
            if new:
                self.policy.apply(self.t, new)
                self.est.excluded = self.policy.excluded.copy()

    def run(self) -> None:
        credit = 0.0
        last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            credit += self.speed * (now - last) / DT
            last = now
            n = min(int(credit), MAX_TICKS_PER_SLICE)
            credit -= n
            with self.lock:
                while not self.cmds.empty():
                    self._apply_command(self.cmds.get_nowait())
                for _ in range(n):
                    self._tick()
            time.sleep(WALL_SLICE_S)

    # ------------------------------------------------------------ snapshot

    def snapshot(self) -> dict:
        def arr(a, nd=4):
            return [round(float(x), nd) for x in a]

        with self.lock:
            tel = self.bus.telemetry
            v_rep = [
                round(float(x), 4) if np.isfinite(x) else None for x in tel.v
            ]
            diagnoses = [
                {"cell": d.cell, "kind": d.kind, "subtype": d.subtype,
                 "severity": d.severity, "t": round(d.t_detect, 1),
                 "detail": d.detail}
                for d in self.detector.diagnoses.values()
            ]
            events = sorted(
                self.policy.events + self.ui_events, key=lambda e: e[0]
            )[-30:]
            return {
                "type": "state",
                "t": round(self.t, 1),
                "mode": self.mode,
                "speed": self.speed,
                "balancer_on": self.balancer_on,
                "pack": {
                    "v": round(self.pack.pack_voltage, 2),
                    "i": round(self.i_applied, 2),
                    "i_request": round(self.i_request, 2),
                    "limit": self.policy.i_limit_frac,
                    "contactor_open": self.policy.contactor_open,
                    "soc_mean_est": round(float(np.mean(self.est.soc)), 4),
                    "soc_spread_true": round(float(np.ptp(self.pack.cells.soc)), 4),
                    "bal_loss_wh": round(self.plant.loss_wh, 3),
                },
                "cells": {
                    "v_true": arr(self.pack.last_v),
                    "v_rep": v_rep,
                    "soc_true": arr(self.pack.cells.soc),
                    "soc_est": arr(self.est.soc),
                    "temp": arr(self.thermal.temps, 2),
                    "r0_mohm": arr(1e3 * self.est.r0, 3),
                    "q_est_ah": arr(self.est.q_ah, 3),
                    "bal_a": arr(self.plant.last_applied, 3),
                    "flagged": [bool(x) for x in self.policy.flagged],
                    "excluded": [bool(x) for x in self.policy.excluded],
                },
                "diagnoses": diagnoses,
                "events": [
                    {"t": round(t_, 1), "msg": m_} for t_, m_ in events
                ],
            }
