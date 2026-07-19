"""Phase 4 validation: optimization-based active balancing vs passive bleed.

Scenario: an aged pack comes back from unbalanced service with a large
SOC spread (~55% +/- 9%). Two identical copies of that pack are
balanced at rest, each by a controller that sees ONLY the UKF
estimator's SOC/capacity (cold-booted at 50%, fed by the bandwidth-
limited bus):

- active:  receding-horizon LP over per-cell bidirectional DC-DC
           currents (1 A rating, 88% round-trip, shared-rail power
           balance), replanned every 60 s
- passive: classic bleed-to-minimum resistors (150 mA, pure loss)

Checks
------
1. Active balancing converges (true spread < 1%) within the cap.
2. The LP ran on accurate estimates (estimator RMSE < 1% throughout).
3. Passive takes > 2x longer.
4. Active loses < 50% of the energy passive burns.
5. The LP's schedules never overdraw the balancing rail.
6. Applied currents respect hardware ratings.

Outputs: PNGs + validation_summary.md in results/phase4/.
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

from balancing import BalancerPlant, LPBalancer, PassiveBleeder  # noqa: E402
from estimation import PackEstimator  # noqa: E402
from model.comms import BmsBus  # noqa: E402
from model.pack import Pack  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "phase4"
DT = 0.1
AMBIENT_C = 25.0
ENABLE_T = 60.0        # let the estimator settle before balancing
REPLAN_S = 60.0
SPREAD_DONE = 0.01     # "balanced" = true max-min SOC below 1%


def build_pack() -> Pack:
    rng = np.random.default_rng(11)
    soc0 = np.clip(0.55 + rng.normal(0.0, 0.03, 48), 0.44, 0.66)
    pack = Pack(soc0=soc0)
    pack.cells.fast_forward_aging(rng.uniform(150.0, 400.0, pack.n_cells))
    return pack


def run_case(kind: str, cap_s: float) -> dict:
    pack = build_pack()
    bus = BmsBus(pack)
    est = PackEstimator(pack.n_cells, pack.module_of, "ukf", soc0=0.5)
    temps = np.full(pack.n_cells, AMBIENT_C)
    if kind == "active":
        ctrl = LPBalancer(pack.n_cells)
        plant = BalancerPlant(pack.n_cells)
    else:
        ctrl = PassiveBleeder(pack.n_cells)
        plant = BalancerPlant(pack.n_cells, passive=True)

    log: dict[str, list] = {k: [] for k in (
        "t", "soc", "soc_est", "applied", "loss_wh", "spread")}
    t, next_replan = 0.0, ENABLE_T
    cmd = np.zeros(pack.n_cells)
    done_since = None
    n_ticks = int(cap_s / DT)
    for k in range(n_ticks):
        if t + 1e-9 >= next_replan:
            cmd = ctrl.replan(est.soc, est.q_ah)
            next_replan += REPLAN_S
        applied = plant.apply(cmd, pack.last_v, DT)
        pack.step(0.0, DT, temps, aux_current_a=applied)
        t += DT
        bus.step(t, DT, temps)
        est.tick(t, bus.telemetry, DT, aux_cmd_a=applied)
        if (k + 1) % 10 == 0:
            spread = float(np.ptp(pack.cells.soc))
            log["t"].append(t)
            log["soc"].append(pack.cells.soc.copy())
            log["soc_est"].append(est.soc.copy())
            log["applied"].append(applied.copy())
            log["loss_wh"].append(plant.loss_wh)
            log["spread"].append(spread)
            if spread < SPREAD_DONE and t > ENABLE_T:
                if done_since is None:
                    done_since = t
                if t - done_since >= 60.0:
                    break
            else:
                done_since = None

    L = {k_: np.asarray(v) for k_, v in log.items()}
    conv_t = float("inf") if done_since is None else done_since - ENABLE_T
    est_err = L["soc_est"] - L["soc"]
    return {
        "log": L,
        "conv_t": conv_t,
        "loss_wh": plant.loss_wh,
        "moved_ah": plant.moved_ah,
        "rail_scale_events": plant.rail_scale_events,
        "max_applied": float(np.max(np.abs(L["applied"]))),
        "est_rmse": float(np.sqrt(np.mean(est_err[L["t"] > ENABLE_T] ** 2))),
        "spread0": float(L["spread"][0]),
        "spread_end": float(L["spread"][-1]),
        "ctrl": ctrl,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []
    print("== Phase 4 balancing validation ==")

    wall = time.time()
    active = run_case("active", cap_s=5400.0)
    passive = run_case("passive", cap_s=14400.0)
    print(f"both cases simulated in {time.time() - wall:.1f} s wall")
    print(f"initial true SOC spread: {100 * active['spread0']:.1f}%")
    print(f"active:  converged in {active['conv_t']:.0f} s, "
          f"loss {active['loss_wh']:.2f} Wh, moved {active['moved_ah']:.2f} Ah, "
          f"final spread {100 * active['spread_end']:.2f}%")
    print(f"passive: converged in {passive['conv_t']:.0f} s, "
          f"loss {passive['loss_wh']:.2f} Wh, "
          f"final spread {100 * passive['spread_end']:.2f}%")

    checks.append(_check(
        f"Active balancing: true spread {100 * active['spread0']:.1f}% -> "
        f"<1% in {active['conv_t']:.0f} s ({active['conv_t'] / 60:.0f} min)",
        np.isfinite(active["conv_t"]),
    ))
    checks.append(_check(
        f"LP consumed accurate estimates: estimator SOC RMSE "
        f"{100 * active['est_rmse']:.2f}% < 1% during balancing",
        active["est_rmse"] < 0.01,
    ))
    ratio = passive["conv_t"] / active["conv_t"]
    checks.append(_check(
        f"Passive is {ratio:.1f}x slower ({passive['conv_t']:.0f} s vs "
        f"{active['conv_t']:.0f} s)",
        passive["conv_t"] > 2.0 * active["conv_t"],
    ))
    checks.append(_check(
        f"Active loses {active['loss_wh']:.2f} Wh vs passive "
        f"{passive['loss_wh']:.2f} Wh burned "
        f"({100 * active['loss_wh'] / passive['loss_wh']:.0f}%)",
        active["loss_wh"] < 0.5 * passive["loss_wh"],
    ))
    checks.append(_check(
        f"Rail never overdrawn ({active['rail_scale_events']} scale events; "
        f"{active['ctrl'].solves} LP solves, "
        f"{active['ctrl'].solve_failures} failures)",
        active["rail_scale_events"] == 0
        and active["ctrl"].solve_failures == 0,
    ))
    checks.append(_check(
        f"Hardware ratings respected (max |i| active "
        f"{active['max_applied']:.2f} A <= 1 A, passive "
        f"{passive['max_applied']:.2f} A <= 0.15 A)",
        active["max_applied"] <= 1.0 + 1e-9
        and passive["max_applied"] <= 0.15 + 1e-9,
    ))

    _make_plots(active, passive)

    n_fail = sum(1 for c in checks if c.startswith("[FAIL]"))
    lines = [
        "# Phase 4 — active balancing validation",
        "",
        f"Aged pack, initial true SOC spread {100 * active['spread0']:.1f}%, "
        "balanced at rest by controllers running purely on UKF estimates.",
        "",
        "| metric | active (LP, 1 A DC-DC) | passive (150 mA bleed) |",
        "|---|---|---|",
        f"| time to <1% spread | {active['conv_t']:.0f} s | {passive['conv_t']:.0f} s |",
        f"| energy lost | {active['loss_wh']:.2f} Wh | {passive['loss_wh']:.2f} Wh |",
        f"| final spread | {100 * active['spread_end']:.2f}% | {100 * passive['spread_end']:.2f}% |",
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


def _make_plots(active: dict, passive: dict) -> None:
    La, Lp = active["log"], passive["log"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(La["t"] / 60.0, La["soc"], lw=0.7, alpha=0.6)
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("true cell SOC")
    ax1.set_title("Active balancing: 48 cells converge")
    ax1.grid(alpha=0.3)
    ax2.semilogy(La["t"] / 60.0, 100 * La["spread"], label="active (LP, 1 A)")
    ax2.semilogy(Lp["t"] / 60.0, 100 * Lp["spread"], label="passive (150 mA)")
    ax2.axhline(1.0, color="k", ls=":", lw=1, label="1% target")
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("SOC spread max-min [%]")
    ax2.set_title("Convergence: active vs passive")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "balancing_convergence.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    lim = np.max(np.abs(La["applied"]))
    im = ax.imshow(
        La["applied"].T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        extent=[La["t"][0] / 60.0, La["t"][-1] / 60.0, 47.5, -0.5],
        interpolation="nearest",
    )
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("cell index")
    ax.set_title("LP balancing schedule (red = drain, blue = charge)")
    fig.colorbar(im, ax=ax, label="balancing current [A]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "balancing_schedule.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(La["t"] / 60.0, La["loss_wh"], label="active: converter losses")
    ax.plot(Lp["t"] / 60.0, Lp["loss_wh"], label="passive: bled energy")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("cumulative energy lost [Wh]")
    ax.set_title("Energy cost of balancing")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "balancing_energy.png", dpi=140)
    plt.close(fig)


def _check(desc: str, ok) -> str:
    line = f"[{'PASS' if bool(ok) else 'FAIL'}] {desc}"
    print(line)
    return line


if __name__ == "__main__":
    raise SystemExit(main())
