"""Phase 6 validation: fault detection, diagnosis, and safety response.

Five closed-loop scenarios (truth + thermal + bus + UKF estimator +
detector + safety policy, controller side seeing only telemetry):

- clean:          15 min drive, no fault        -> zero diagnoses
- short:          0.2-ohm internal short        -> critical + shutdown
- sensor_freeze:  stuck AFE channel             -> warning + quarantine
- sensor_offset:  +60 mV reporting bias         -> warning + quarantine
- degradation:    2000x aging on one cell       -> maintenance + derate

Each faulted scenario must diagnose the right fault type on the right
cell within its latency bound, trigger the right tier of safety
response, and flag NO other cell. The clean run establishes the
false-positive floor.

Usage: validate_faults.py [--scenario clean|short|sensor_freeze|
                            sensor_offset|degradation|all]

Outputs: PNGs + validation_summary.md in results/phase6/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from estimation import PackEstimator  # noqa: E402
from fault_detection import FaultDetector, FaultInjector, SafetyPolicy  # noqa: E402
from model.cell import CellParams  # noqa: E402
from model.comms import BmsBus  # noqa: E402
from model.drive_cycle import synth_drive_cycle  # noqa: E402
from model.pack import Pack  # noqa: E402
from model.thermal import ThermalModel  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "phase6"
DT = 0.1
I_1C = CellParams().q_nom_ah


def drive_profile(duration_s: float) -> np.ndarray:
    return synth_drive_cycle(duration_s, DT, 0.7 * I_1C, seed=5,
                             rest_s=(60.0, 120.0, 60.0))


def cycle_profile(duration_s: float) -> np.ndarray:
    """+/-1C square wave, 60 s blocks: steady throughput so aging
    accumulates and R0 stays excited, with blocks short enough that
    charge/discharge model-mismatch drift averages out inside the
    detector's 10 min trend window (long blocks make the whole fleet's
    R0 estimates wander block-correlated by several % — indistinguishable
    from aging at trend timescales)."""
    n = int(duration_s / DT)
    block = int(60 / DT)
    i = np.empty(n)
    for b in range(0, n, block):
        i[b : b + block] = I_1C if (b // block) % 2 == 0 else -I_1C
    return i


SCENARIOS: dict[str, dict] = {
    "clean": dict(duration=900.0, profile=drive_profile, t_inject=None,
                  inject=None, expect=None),
    "short": dict(duration=900.0, profile=drive_profile, t_inject=300.0,
                  inject=lambda inj, t: inj.inject_short(20, 0.2, t),
                  expect=("internal_short", 20), latency_max=90.0,
                  shutdown=True, i_limit=0.0),
    "sensor_freeze": dict(duration=900.0, profile=drive_profile,
                          t_inject=300.0,
                          inject=lambda inj, t: inj.inject_sensor_freeze(35, t),
                          expect=("sensor_fault", 35), latency_max=60.0,
                          shutdown=False, i_limit=0.5),
    "sensor_offset": dict(duration=900.0, profile=drive_profile,
                          t_inject=300.0,
                          inject=lambda inj, t: inj.inject_sensor_offset(8, 0.06, t),
                          expect=("sensor_fault", 8), latency_max=150.0,
                          shutdown=False, i_limit=0.5),
    "degradation": dict(duration=2400.0, profile=cycle_profile,
                        t_inject=200.0,
                        inject=lambda inj, t: inj.inject_degradation(42, 2000.0, t),
                        expect=("degradation", 42), latency_max=1200.0,
                        shutdown=False, i_limit=0.75),
}


def run_scenario(name: str, spec: dict) -> dict:
    pack = Pack(soc0=0.80)
    thermal = ThermalModel(pack.config.n_modules, pack.config.cells_per_module)
    bus = BmsBus(pack)
    injector = FaultInjector(pack)
    est = PackEstimator(pack.n_cells, pack.module_of, "ukf", soc0=0.5)
    detector = FaultDetector(pack.n_cells, pack.module_of)
    policy = SafetyPolicy(pack.n_cells)
    profile = spec["profile"](spec["duration"])

    log: dict[str, list] = {k: [] for k in (
        "t", "true_v", "tel_v", "soc_est", "soc_true", "r0_est",
        "t_mod", "i_applied", "limit_frac", "contactor")}
    t, injected, seen_steps = 0.0, False, 0
    for k in range(len(profile)):
        if (spec["t_inject"] is not None and not injected
                and t >= spec["t_inject"]):
            spec["inject"](injector, t)
            injected = True
        i_cmd = policy.limit_current(float(profile[k]))
        aux = injector.short_currents_a()
        pack.step(i_cmd, DT, thermal.temps, aux_current_a=aux)
        q = pack.cells.heat_generation_w(i_cmd + aux, thermal.temps)
        q += injector.short_heat_w(aux)
        thermal.step(q, DT)
        t += DT
        bus.step(t, DT, thermal.temps)
        injector.corrupt_telemetry(bus.telemetry)
        est.tick(t, bus.telemetry, DT)
        if est.n_steps > seen_steps:
            seen_steps = est.n_steps
            new = detector.step(t, bus.telemetry, est)
            if new:
                policy.apply(t, new)
                est.excluded = policy.excluded.copy()
        if (k + 1) % 10 == 0:
            log["t"].append(t)
            log["true_v"].append(pack.last_v.copy())
            log["tel_v"].append(bus.telemetry.v.copy())
            log["soc_est"].append(est.soc.copy())
            log["soc_true"].append(pack.cells.soc.copy())
            log["r0_est"].append(est.r0.copy())
            log["t_mod"].append(thermal.temps.reshape(6, 8).mean(axis=1))
            log["i_applied"].append(i_cmd)
            log["limit_frac"].append(policy.i_limit_frac)
            log["contactor"].append(policy.contactor_open)
    return {
        "name": name,
        "log": {k_: np.asarray(v_) for k_, v_ in log.items()},
        "diagnoses": list(detector.diagnoses.values()),
        "policy": policy,
        "spec": spec,
    }


def evaluate(res: dict) -> tuple[list[str], dict]:
    checks: list[str] = []
    spec, name = res["spec"], res["name"]
    diags = res["diagnoses"]
    policy = res["policy"]
    stats: dict = {"name": name, "latency": None, "fp": 0, "tp": 0}

    if spec["expect"] is None:
        stats["fp"] = len(diags)
        checks.append(_check(
            f"[{name}] no false positives over 15 min x 48 cells "
            f"({len(diags)} diagnoses)",
            len(diags) == 0,
        ))
        return checks, stats

    kind, cell = spec["expect"]
    hits = [d for d in diags if d.cell == cell and d.kind == kind]
    others = [d for d in diags if d.cell != cell]
    misclass = [d for d in diags if d.cell == cell and d.kind != kind]
    stats["tp"] = int(bool(hits))
    stats["fp"] = len(others)
    checks.append(_check(
        f"[{name}] diagnosed {kind} on cell {cell} "
        f"({hits[0].subtype if hits else 'MISSED'}"
        + (f'; misclassified as {misclass[0].kind}' if misclass else '') + ")",
        bool(hits) and not misclass,
    ))
    if hits:
        latency = hits[0].t_detect - spec["t_inject"]
        stats["latency"] = latency
        checks.append(_check(
            f"[{name}] detection latency {latency:.0f} s < "
            f"{spec['latency_max']:.0f} s bound",
            latency < spec["latency_max"],
        ))
    checks.append(_check(
        f"[{name}] no other cell flagged ({len(others)} false positives)",
        len(others) == 0,
    ))
    action_ok = (policy.contactor_open == spec["shutdown"]
                 and abs(policy.i_limit_frac - spec["i_limit"]) < 1e-9)
    checks.append(_check(
        f"[{name}] safety response correct (contactor "
        f"{'open' if policy.contactor_open else 'closed'}, current limit "
        f"{100 * policy.i_limit_frac:.0f}%)",
        action_ok,
    ))
    return checks, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=[*SCENARIOS.keys(), "all"])
    args = ap.parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== Phase 6 fault detection validation ==")
    all_checks: list[str] = []
    all_stats: list[dict] = []
    wall = time.time()
    for name in names:
        res = run_scenario(name, SCENARIOS[name])
        checks, stats = evaluate(res)
        all_checks.extend(checks)
        all_stats.append(stats)
        _plot_scenario(res)
        for ts, ev in res["policy"].events:
            print(f"    t={ts:7.1f}s  {ev}")
    print(f"{len(names)} scenarios in {time.time() - wall:.1f} s wall")

    n_fail = sum(1 for c in all_checks if c.startswith("[FAIL]"))
    if args.scenario == "all":
        tp = sum(s["tp"] for s in all_stats)
        fp = sum(s["fp"] for s in all_stats)
        lines = [
            "# Phase 6 — fault detection validation",
            "",
            "Closed loop: truth + thermal + bus + UKF estimator + residual "
            "detector + safety policy; the controller side sees only "
            "telemetry.",
            "",
            "| scenario | injected fault | detected | latency | response |",
            "|---|---|---|---|---|",
        ]
        for s, name in zip(all_stats, names):
            spec = SCENARIOS[name]
            if spec["expect"] is None:
                lines.append(f"| {name} | — | {s['fp']} diagnoses | — | — |")
            else:
                lat = f"{s['latency']:.0f} s" if s["latency"] else "missed"
                resp = ("shutdown" if spec["shutdown"]
                        else f"derate to {100 * spec['i_limit']:.0f}%")
                lines.append(f"| {name} | {spec['expect'][0]} on cell "
                             f"{spec['expect'][1]} | "
                             f"{'yes' if s['tp'] else 'NO'} | {lat} | {resp} |")
        lines += [
            "",
            f"**True positives {tp}/4, false positives {fp} across "
            f"{sum(SCENARIOS[n]['duration'] for n in names) / 3600 * 48:.0f} "
            "cell-hours.**",
            "",
            "```", *all_checks, "```", "",
            f"Result: {len(all_checks) - n_fail}/{len(all_checks)} checks passed.",
        ]
        (OUT_DIR / "validation_summary.md").write_text(
            "\n".join(lines), encoding="utf-8")
    print(f"\n{len(all_checks) - n_fail}/{len(all_checks)} checks passed; "
          f"plots + summary in {OUT_DIR.relative_to(REPO_ROOT)}")
    return 1 if n_fail else 0


def _plot_scenario(res: dict) -> None:
    name, L, spec = res["name"], res["log"], res["spec"]
    if spec["expect"] is None:
        return
    cell = spec["expect"][1]
    t_inj = spec["t_inject"]
    t_det = res["diagnoses"][0].t_detect if res["diagnoses"] else None
    tm = L["t"] / 60.0
    others = [c for c in range(48) if c != cell]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.plot(tm, L["true_v"][:, others], lw=0.4, color="0.8")
    ax.plot(tm, L["true_v"][:, cell], "C3", lw=1.4, label=f"cell {cell} true")
    ax.plot(tm, L["tel_v"][:, cell], "C0", lw=0.9, alpha=0.8,
            label=f"cell {cell} reported")
    ax.set_ylabel("voltage [V]")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    ax.plot(tm, L["soc_est"][:, others], lw=0.4, color="0.8")
    ax.plot(tm, L["soc_true"][:, cell], "k", lw=1.4, label="true SOC")
    ax.plot(tm, L["soc_est"][:, cell], "C3", lw=1.2, label="estimated SOC")
    ax.set_ylabel("SOC")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[2]
    if name == "degradation":
        ax.plot(tm, 1e3 * L["r0_est"][:, others], lw=0.4, color="0.8")
        ax.plot(tm, 1e3 * L["r0_est"][:, cell], "C3", lw=1.4,
                label=f"cell {cell} R0 estimate")
        ax.set_ylabel("R0 [mOhm]")
    else:
        ax.plot(tm, L["t_mod"], lw=0.7)
        ax.set_ylabel("module temp [degC]")
    ax2 = ax.twinx()
    ax2.plot(tm, 100 * np.asarray(L["limit_frac"]), "C2", ls="--", lw=1.2)
    ax2.set_ylabel("pack current limit [%]", color="C2")
    ax2.set_ylim(-5, 110)
    ax.legend(fontsize=8, loc="center left")
    ax.set_xlabel("time [min]")
    for a in axes:
        a.grid(alpha=0.3)
        a.axvline(t_inj / 60.0, color="k", ls=":", lw=1)
        if t_det:
            a.axvline(t_det / 60.0, color="C3", ls=":", lw=1)
    axes[0].set_title(
        f"{name}: injected t={t_inj:.0f} s (black), "
        f"diagnosed t={t_det:.0f} s (red)" if t_det else f"{name}: MISSED")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fault_{name}.png", dpi=140)
    plt.close(fig)


def _check(desc: str, ok) -> str:
    line = f"[{'PASS' if bool(ok) else 'FAIL'}] {desc}"
    print(line)
    return line


if __name__ == "__main__":
    raise SystemExit(main())
