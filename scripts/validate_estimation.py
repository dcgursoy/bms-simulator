"""Phase 3 validation: UKF/EKF joint SOC + SOH estimation accuracy.

Scenario: a pre-aged 48-cell pack (150-400 equivalent full cycles per
cell, so every cell has real capacity fade and impedance growth the
estimator does NOT know about) runs a ~2 h urban drive cycle. Both
filter banks cold-boot at 50% SOC for every cell — a worst-case wrong
init (true SOC is 95%) — and see only bandwidth-limited telemetry.

A coulomb-counting baseline (same wrong init, nominal capacity, noisy
current integration) shows what the filters add.

Checks
------
1. UKF converges from the 45%-wrong cold boot to <2% max-cell error.
2. UKF SOC RMSE (post-convergence) < 1%; EKF < 2%.
3. R0 (SOH-impedance) error shrinks vs init and lands < 5%.
4. Capacity (SOH-fade) error at least halves from the ~7% nominal-value
   init error.
5. Coulomb counting is >= 5x worse than the UKF.

Outputs: PNGs + validation_summary.md in results/phase3/.
"""

from __future__ import annotations

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
from model.cell import CellParams  # noqa: E402
from model.comms import BmsBus  # noqa: E402
from model.drive_cycle import synth_drive_cycle  # noqa: E402
from model.pack import Pack  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "phase3"
DT = 0.1
DURATION_S = 7500.0  # ends near 15-20% true SOC (no over-discharge)
NOMINAL = CellParams()
AMBIENT_C = 25.0
SOC0_TRUE = 0.95
SOC0_EST = 0.50
CONV_THRESH = 0.02
STEADY_T = 900.0  # metrics window start [s]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []
    print("== Phase 3 estimation validation ==")

    # Truth: pre-aged pack, unknown to the estimators
    pack = Pack(soc0=SOC0_TRUE)
    aging_rng = np.random.default_rng(42)
    pack.cells.fast_forward_aging(aging_rng.uniform(150.0, 400.0, pack.n_cells))
    bus = BmsBus(pack)
    temps = np.full(pack.n_cells, AMBIENT_C)
    i_profile = synth_drive_cycle(DURATION_S, DT, NOMINAL.q_nom_ah, seed=3)

    est = {
        "ukf": PackEstimator(pack.n_cells, pack.module_of, "ukf", soc0=SOC0_EST),
        "ekf": PackEstimator(pack.n_cells, pack.module_of, "ekf", soc0=SOC0_EST),
    }
    soc_cc = SOC0_EST  # coulomb-counting baseline (scalar per-pack SOC drift
    # applied to every cell; same wrong init, nominal capacity)

    log: dict[str, list] = {k: [] for k in (
        "t", "i", "soc_true", "soc_ukf", "soc_ekf", "soc_cc",
        "r0_ukf", "r0_ekf", "q_ukf", "q_ekf", "r0_true", "q_true")}

    wall = time.time()
    t = 0.0
    for k, i_pack in enumerate(i_profile):
        pack.step(i_pack, DT, temps)
        t += DT
        bus.step(t, DT, temps)
        for e in est.values():
            e.tick(t, bus.telemetry, DT)
        soc_cc -= bus.telemetry.i * DT / (3600.0 * NOMINAL.q_nom_ah)
        if (k + 1) % 10 == 0:  # 1 s cadence
            log["t"].append(t)
            log["i"].append(i_pack)
            log["soc_true"].append(pack.cells.soc.copy())
            log["soc_ukf"].append(est["ukf"].soc.copy())
            log["soc_ekf"].append(est["ekf"].soc.copy())
            log["soc_cc"].append(soc_cc)
            log["r0_ukf"].append(est["ukf"].r0.copy())
            log["r0_ekf"].append(est["ekf"].r0.copy())
            log["q_ukf"].append(est["ukf"].q_ah.copy())
            log["q_ekf"].append(est["ekf"].q_ah.copy())
            log["r0_true"].append(pack.cells.resistances(AMBIENT_C)[0].copy())
            log["q_true"].append(pack.cells.capacity_ah(AMBIENT_C).copy())
    print(f"simulated {DURATION_S / 3600:.2f} h in {time.time() - wall:.1f} s wall; "
          f"true SOC now {pack.cells.soc.mean():.3f} "
          f"(span {pack.cells.soc.min():.3f}-{pack.cells.soc.max():.3f})")

    L = {k: np.asarray(v) for k, v in log.items()}
    tt = L["t"]
    steady = tt >= STEADY_T

    # ------------------------------------------------------------- metrics
    results: dict[str, dict] = {}
    for name in ("ukf", "ekf"):
        err = L[f"soc_{name}"] - L["soc_true"]  # (T, 48)
        max_err = np.max(np.abs(err), axis=1)
        conv_t = _sustained_below(tt, max_err, CONV_THRESH, hold_s=60.0)
        r0_rel = np.abs(L[f"r0_{name}"] - L["r0_true"]) / L["r0_true"]
        q_rel = np.abs(L[f"q_{name}"] - L["q_true"]) / L["q_true"]
        results[name] = {
            "conv_t": conv_t,
            "rmse": float(np.sqrt(np.mean(err[steady] ** 2))),
            "max_abs": float(np.max(np.abs(err[steady]))),
            "r0_init": float(np.mean(r0_rel[0])),
            "r0_final": float(np.mean(r0_rel[-600:])),
            "q_init": float(np.mean(q_rel[0])),
            "q_final": float(np.mean(q_rel[-600:])),
            "us_per_cell_step": est[name].us_per_cell_step(),
        }
    cc_err = L["soc_cc"][:, None] - L["soc_true"]
    cc_rmse = float(np.sqrt(np.mean(cc_err[steady] ** 2)))

    for name in ("ukf", "ekf"):
        r = results[name]
        print(f"{name.upper()}: conv {r['conv_t']:.0f} s, SOC RMSE "
              f"{100 * r['rmse']:.2f}% (max {100 * r['max_abs']:.2f}%), R0 err "
              f"{100 * r['r0_init']:.1f}%->{100 * r['r0_final']:.1f}%, Q err "
              f"{100 * r['q_init']:.1f}%->{100 * r['q_final']:.1f}%, "
              f"{r['us_per_cell_step']:.0f} us/cell/step")
    print(f"CC baseline RMSE {100 * cc_rmse:.2f}%")

    # -------------------------------------------------------------- checks
    u, e = results["ukf"], results["ekf"]
    checks.append(_check(
        f"UKF cold-boot convergence: 45% initial error -> <2% in "
        f"{u['conv_t']:.0f} s (< 900 s)",
        u["conv_t"] < 900.0,
    ))
    checks.append(_check(
        f"UKF SOC RMSE {100 * u['rmse']:.2f}% < 1% (steady state, all 48 cells)",
        u["rmse"] < 0.01,
    ))
    checks.append(_check(
        f"EKF SOC RMSE {100 * e['rmse']:.2f}% < 2%",
        e["rmse"] < 0.02,
    ))
    checks.append(_check(
        f"R0 (SOH) adaptation: UKF error {100 * u['r0_init']:.1f}% -> "
        f"{100 * u['r0_final']:.1f}% (< 5%)",
        u["r0_final"] < 0.05 and u["r0_final"] < 0.6 * u["r0_init"],
    ))
    checks.append(_check(
        f"Capacity (SOH) adaptation: UKF error {100 * u['q_init']:.1f}% -> "
        f"{100 * u['q_final']:.1f}% (at least halved)",
        u["q_final"] < 0.5 * u["q_init"],
    ))
    checks.append(_check(
        f"Coulomb-counting baseline {100 * cc_rmse:.2f}% RMSE is >= 5x worse "
        f"than UKF {100 * u['rmse']:.2f}%",
        cc_rmse > 5.0 * u["rmse"],
    ))

    _make_plots(L, tt, results)

    n_fail = sum(1 for c in checks if c.startswith("[FAIL]"))
    lines = [
        "# Phase 3 — SOC/SOH estimation validation",
        "",
        f"Pre-aged 48-cell pack, ~2 h urban drive cycle, cold boot at "
        f"{100 * SOC0_EST:.0f}% SOC vs true {100 * SOC0_TRUE:.0f}%. Estimators "
        "see only bandwidth-limited telemetry (0.9 s refresh, noisy, "
        "quantized).",
        "",
        "| metric | UKF | EKF |",
        "|---|---|---|",
        f"| convergence to <2% [s] | {u['conv_t']:.0f} | {e['conv_t']:.0f} |",
        f"| SOC RMSE (steady) | {100 * u['rmse']:.2f}% | {100 * e['rmse']:.2f}% |",
        f"| SOC max abs err (steady) | {100 * u['max_abs']:.2f}% | {100 * e['max_abs']:.2f}% |",
        f"| R0 error init -> final | {100 * u['r0_init']:.1f}% -> {100 * u['r0_final']:.1f}% "
        f"| {100 * e['r0_init']:.1f}% -> {100 * e['r0_final']:.1f}% |",
        f"| capacity error init -> final | {100 * u['q_init']:.1f}% -> {100 * u['q_final']:.1f}% "
        f"| {100 * e['q_init']:.1f}% -> {100 * e['q_final']:.1f}% |",
        f"| compute [us/cell/step] | {u['us_per_cell_step']:.0f} | {e['us_per_cell_step']:.0f} |",
        "",
        f"Coulomb-counting baseline (wrong init, nominal capacity): "
        f"{100 * cc_rmse:.2f}% RMSE.",
        "",
        "```",
        *checks,
        "```",
        "",
        f"Result: {len(checks) - n_fail}/{len(checks)} checks passed.",
    ]
    (OUT_DIR / "validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed; "
          f"plots + summary in {OUT_DIR.relative_to(REPO_ROOT)}")
    return 1 if n_fail else 0


def _sustained_below(t, series, thresh, hold_s) -> float:
    """First time the series stays below thresh for hold_s seconds."""
    below = series < thresh
    run_start = None
    for k in range(len(t)):
        if below[k]:
            if run_start is None:
                run_start = t[k]
            if t[k] - run_start >= hold_s:
                return float(run_start)
        else:
            run_start = None
    return float("inf")


def _make_plots(L, tt, results) -> None:
    th = tt / 3600.0
    worst = int(np.argmax(np.abs(L["q_true"][0] - NOMINAL.q_nom_ah)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, tmax in ((ax1, None), (ax2, 1500.0)):
        m = slice(None) if tmax is None else tt <= tmax
        x = th[m] * 3600.0 if tmax else th[m]
        ax.plot(x, L["soc_true"][m, worst], "k", lw=2, label="truth")
        ax.plot(x, L["soc_ukf"][m, worst], label="UKF")
        ax.plot(x, L["soc_ekf"][m, worst], label="EKF")
        ax.plot(x, L["soc_cc"][m], ls="--", label="coulomb counting")
        ax.set_ylabel("SOC")
        ax.grid(alpha=0.3)
    ax1.set_xlabel("Time [h]")
    ax1.set_title(f"Most-aged cell #{worst}: full drive cycle")
    ax1.legend()
    ax2.set_xlabel("Time [s]")
    ax2.set_title("Cold-boot convergence (init 50%, truth 95%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soc_tracking.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, color in (("ukf", "C0"), ("ekf", "C1")):
        err = np.abs(L[f"soc_{name}"] - L["soc_true"])
        ax.semilogy(th, 100 * np.median(err, axis=1), color, lw=1.4,
                    label=f"{name.upper()} median")
        ax.semilogy(th, 100 * np.max(err, axis=1), color, lw=0.8, alpha=0.45,
                    label=f"{name.upper()} worst cell")
    ax.axhline(2.0, color="k", ls=":", lw=1, label="2% threshold")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("|SOC error| [%]")
    ax.set_title("SOC estimation error across all 48 cells")
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncols=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "estimation_error.png", dpi=140)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color in (("ukf", "C0"), ("ekf", "C1")):
        r0_rel = np.abs(L[f"r0_{name}"] - L["r0_true"]) / L["r0_true"]
        q_rel = np.abs(L[f"q_{name}"] - L["q_true"]) / L["q_true"]
        ax1.plot(th, 100 * np.mean(r0_rel, axis=1), color, label=name.upper())
        ax2.plot(th, 100 * np.mean(q_rel, axis=1), color, label=name.upper())
    ax1.set_title("R0 estimate (impedance SOH)")
    ax2.set_title("Capacity estimate (fade SOH)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("mean relative error [%]")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soh_convergence.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    steady = tt >= STEADY_T
    for name, color in (("ukf", "C0"), ("ekf", "C1")):
        err = L[f"soc_{name}"] - L["soc_true"]
        per_cell = 100 * np.sqrt(np.mean(err[steady] ** 2, axis=0))
        xs = np.sort(per_cell)
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), color,
                label=f"{name.upper()} (med {np.median(per_cell):.2f}%)")
    ax.set_xlabel("per-cell SOC RMSE [%] (steady state)")
    ax.set_ylabel("fraction of cells")
    ax.set_title("UKF vs EKF: per-cell accuracy distribution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ukf_vs_ekf.png", dpi=140)
    plt.close(fig)


def _check(desc: str, ok) -> str:
    line = f"[{'PASS' if bool(ok) else 'FAIL'}] {desc}"
    print(line)
    return line


if __name__ == "__main__":
    raise SystemExit(main())
