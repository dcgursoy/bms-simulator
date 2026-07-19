"""Phase 5 validation: coupled electro-thermal simulation.

Scenario A — 3C sustained discharge stress at 25 degC ambient with the
electro-thermal loop CLOSED (cell temps feed Arrhenius resistances and
capacity every step). Run twice: closed-loop vs electrically identical
open-loop (temps pinned at 25 degC) to prove the coupling matters.

Scenario B — internal soft short (0.2 ohm) on an interior cell of a
resting pack: the short drains the cell at ~7C internally, dumping
~60 W of heat into one thermal mass; neighbors heat by conduction —
thermal risk propagation. (Pure simulation exercise; no runaway
chemistry is modeled — 120 degC is treated as the model-validity /
alarm ceiling.)

Scenario C — thermal derating: the Phase 4 LP's per-cell current-limit
vector computed from Scenario B's temperature field.

Checks
------
A: plausible peak temperature, interior-vs-corner gradient, hot cells
   less resistive (negative feedback), self-heating recovers capacity.
B: shorted cell exceeds 70 degC, neighbors heat >5 K, far cells stay
   much cooler (propagation is local).
C: hot cells derated to (near) zero, cool cells keep full rating.

Outputs: PNGs + validation_summary.md in results/phase5/.
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

from balancing import thermal_derate  # noqa: E402
from model.cell import CellParams  # noqa: E402
from model.pack import Pack  # noqa: E402
from model.thermal import ThermalModel  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "phase5"
DT = 0.1
NOMINAL = CellParams()
I_3C = 3.0 * NOMINAL.q_nom_ah  # 7.5 A
SHORT_R_OHM = 0.2
SHORT_CELL = 20  # interior cell (row 2, col 4)


def run_stress(closed_loop: bool) -> dict:
    """3C discharge from 90% SOC until the weakest cell hits cutoff."""
    pack = Pack(soc0=0.90)
    thermal = ThermalModel(pack.config.n_modules, pack.config.cells_per_module)
    log: dict[str, list] = {k: [] for k in ("t", "temps", "v", "soc", "r0")}
    t = 0.0
    for k in range(int(3600 / DT)):
        temps = thermal.temps if closed_loop else np.full(pack.n_cells, 25.0)
        pack.step(I_3C, DT, temps)
        q = pack.cells.heat_generation_w(I_3C, temps)
        thermal.step(q, DT)  # thermal state advances either way; only the
        # electrical coupling is switched by closed_loop
        t += DT
        if (k + 1) % 10 == 0:
            log["t"].append(t)
            log["temps"].append(thermal.temps.copy())
            log["v"].append(pack.last_v.copy())
            log["soc"].append(pack.cells.soc.copy())
            log["r0"].append(pack.cells.resistances(temps)[0].copy())
        if pack.v_min_cell <= NOMINAL.v_min:
            break
    L = {k_: np.asarray(v_) for k_, v_ in log.items()}
    L["delivered_ah"] = I_3C * L["t"][-1] / 3600.0
    return L


def run_short() -> dict:
    """Resting pack, internal soft short on one interior cell at t=60 s."""
    pack = Pack(soc0=0.90)
    thermal = ThermalModel(pack.config.n_modules, pack.config.cells_per_module)
    log: dict[str, list] = {k: [] for k in ("t", "temps", "v", "soc", "i_short")}
    t = 0.0
    for k in range(int(480 / DT)):
        temps = thermal.temps
        aux = np.zeros(pack.n_cells)
        i_short = 0.0
        if t >= 60.0:
            # Internal short: cell discharges through its own fault
            # resistance; the full electrochemical power dissipates as
            # heat inside the cell casing
            i_short = max(float(pack.last_v[SHORT_CELL]) / SHORT_R_OHM, 0.0)
            aux[SHORT_CELL] = i_short
        pack.step(0.0, DT, temps, aux_current_a=aux)
        q = pack.cells.heat_generation_w(aux, temps)
        q[SHORT_CELL] += float(pack.last_v[SHORT_CELL]) * i_short
        thermal.step(q, DT)
        t += DT
        if (k + 1) % 10 == 0:
            log["t"].append(t)
            log["temps"].append(thermal.temps.copy())
            log["v"].append(pack.last_v.copy())
            log["soc"].append(pack.cells.soc.copy())
            log["i_short"].append(i_short)
        if thermal.temps[SHORT_CELL] >= 120.0:
            break
    return {k_: np.asarray(v_) for k_, v_ in log.items()}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []
    print("== Phase 5 electro-thermal validation ==")
    wall = time.time()
    closed = run_stress(closed_loop=True)
    fixed = run_stress(closed_loop=False)
    short = run_short()
    print(f"scenarios simulated in {time.time() - wall:.1f} s wall")

    grid = (6, 8)
    temps_end = closed["temps"][-1]
    lap_degree = _grid_degree(*grid)
    interior = lap_degree == 4
    corner = lap_degree == 2

    # ------------------------------------------------------------ scenario A
    peak = float(np.max(closed["temps"]))
    gradient = float(np.max(temps_end[interior]) - np.min(temps_end[corner]))
    r0_start = closed["r0"][0]
    r0_end = closed["r0"][-1]
    print(f"3C stress: peak {peak:.1f} degC, end interior-corner gradient "
          f"{gradient:.1f} K, delivered {closed['delivered_ah']:.3f} Ah "
          f"(closed) vs {fixed['delivered_ah']:.3f} Ah (fixed 25 degC)")
    checks.append(_check(
        f"3C peak temperature {peak:.1f} degC is plausible for an "
        "air-cooled pack (40-70 degC)",
        40.0 <= peak <= 70.0,
    ))
    checks.append(_check(
        f"Spatial gradient at end of discharge: interior max - corner min "
        f"= {gradient:.1f} K (> 3 K)",
        gradient > 3.0,
    ))
    checks.append(_check(
        "Electro-thermal feedback: every hot cell ends less resistive "
        f"than it started (mean R0 {1e3 * np.mean(r0_start):.1f} -> "
        f"{1e3 * np.mean(r0_end):.1f} mOhm)",
        bool(np.all(r0_end < r0_start)),
    ))
    checks.append(_check(
        f"Self-heating recovers capacity at 3C: {closed['delivered_ah']:.3f} "
        f"Ah closed-loop > {fixed['delivered_ah']:.3f} Ah at fixed 25 degC",
        closed["delivered_ah"] > fixed["delivered_ah"] * 1.01,
    ))

    # ------------------------------------------------------------ scenario B
    neigh = _neighbors(SHORT_CELL, *grid)
    far = int(np.argmax([abs(k // 8 - 2) + abs(k % 8 - 4) for k in range(48)]))
    t_short = float(np.max(short["temps"][:, SHORT_CELL]))
    rise_neigh = float(np.max(short["temps"][:, neigh])) - 25.0
    rise_far = float(np.max(short["temps"][:, far])) - 25.0
    print(f"short: cell peaks {t_short:.0f} degC, neighbor rise "
          f"{rise_neigh:.1f} K, far-corner rise {rise_far:.1f} K, "
          f"short current start {short['i_short'][np.flatnonzero(short['i_short'])[0]]:.1f} A")
    checks.append(_check(
        f"Shorted cell reaches {t_short:.0f} degC (> 70 degC thermal alarm "
        "territory)",
        t_short > 70.0,
    ))
    checks.append(_check(
        f"Heat propagates to neighbors (+{rise_neigh:.1f} K > 5 K) but "
        f"stays local (far corner +{rise_far:.1f} K < half the neighbor rise)",
        rise_neigh > 5.0 and rise_far < 0.5 * rise_neigh,
    ))

    # ------------------------------------------------------------ scenario C
    temps_b = short["temps"][-1]
    limits = thermal_derate(1.0, temps_b)
    cool = temps_b < 40.0
    checks.append(_check(
        f"Thermal derate: shorted cell limited to {limits[SHORT_CELL]:.2f} A, "
        f"{int(np.sum(cool))} cool cells keep the full 1.00 A",
        limits[SHORT_CELL] < 0.05 and bool(np.all(limits[cool] > 0.999)),
    ))

    _make_plots(closed, fixed, short, grid)

    n_fail = sum(1 for c in checks if c.startswith("[FAIL]"))
    lines = [
        "# Phase 5 — electro-thermal validation",
        "",
        f"3C stress: peak {peak:.1f} degC, interior-corner gradient "
        f"{gradient:.1f} K, delivered {closed['delivered_ah']:.3f} Ah closed-loop "
        f"vs {fixed['delivered_ah']:.3f} Ah open-loop. "
        f"Internal 0.2-ohm short: cell peaks {t_short:.0f} degC, neighbors "
        f"+{rise_neigh:.1f} K, far corner +{rise_far:.1f} K.",
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


def _grid_degree(rows: int, cols: int) -> np.ndarray:
    deg = np.zeros(rows * cols)
    for r in range(rows):
        for c in range(cols):
            deg[r * cols + c] = sum(
                1 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < rows and 0 <= c + dc < cols
            )
    return deg


def _neighbors(k: int, rows: int, cols: int) -> list[int]:
    r, c = divmod(k, cols)
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= r + dr < rows and 0 <= c + dc < cols:
            out.append((r + dr) * cols + (c + dc))
    return out


def _make_plots(closed, fixed, short, grid) -> None:
    rows, cols = grid

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.plot(closed["t"] / 60.0, closed["temps"], lw=0.7, alpha=0.6)
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("cell temperature [degC]")
    ax1.set_title("3C discharge: all 48 cells self-heat")
    ax1.grid(alpha=0.3)
    im = ax2.imshow(closed["temps"][-1].reshape(rows, cols), cmap="inferno")
    ax2.set_title("Pack thermal map at cutoff")
    ax2.set_xlabel("cell position")
    ax2.set_ylabel("module")
    fig.colorbar(im, ax=ax2, label="degC")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "thermal_stress.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ah_c = I_3C * closed["t"] / 3600.0
    ah_f = I_3C * fixed["t"] / 3600.0
    ax.plot(ah_c, np.sum(closed["v"], axis=1), label=(
        f"closed electro-thermal loop ({closed['delivered_ah']:.2f} Ah)"))
    ax.plot(ah_f, np.sum(fixed["v"], axis=1), ls="--", label=(
        f"temps pinned at 25 degC ({fixed['delivered_ah']:.2f} Ah)"))
    ax.set_xlabel("Discharged capacity [Ah]")
    ax.set_ylabel("Pack voltage [V]")
    ax.set_title("Feedback matters: warm cells sag less, deliver more")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "thermal_feedback.png", dpi=140)
    plt.close(fig)

    neigh = _neighbors(SHORT_CELL, rows, cols)
    far = int(np.argmax([abs(k // cols - 2) + abs(k % cols - 4)
                         for k in range(rows * cols)]))
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :])
    tm = short["t"] / 60.0
    ax.plot(tm, short["temps"][:, SHORT_CELL], "r", lw=2,
            label=f"shorted cell #{SHORT_CELL} (0.2 ohm internal)")
    ax.plot(tm, short["temps"][:, neigh], "C1", lw=1, alpha=0.8)
    ax.plot([], [], "C1", label="adjacent cells")
    ax.plot(tm, short["temps"][:, far], "C0", label="far corner cell")
    ax.axvline(1.0, color="k", ls=":", lw=1, label="short injected t=60 s")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("temperature [degC]")
    ax.set_title("Internal short: local heating propagates to neighbors")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    vmax = float(np.max(short["temps"]))
    snaps = 1.0 + np.array([0.2, 0.45, 0.7, 1.0]) * (tm[-1] - 1.0)
    for j, t_snap in enumerate(snaps):
        axs = fig.add_subplot(gs[1, j])
        idx = int(np.argmin(np.abs(tm - t_snap)))
        ims = axs.imshow(short["temps"][idx].reshape(rows, cols),
                         cmap="inferno", vmin=25.0, vmax=vmax)
        axs.set_title(f"t = {tm[idx]:.1f} min", fontsize=9)
        axs.set_xticks([])
        axs.set_yticks([])
    fig.colorbar(ims, ax=fig.axes[-4:], label="degC", shrink=0.85)
    fig.savefig(OUT_DIR / "short_propagation.png", dpi=140)
    plt.close(fig)


def _check(desc: str, ok) -> str:
    line = f"[{'PASS' if bool(ok) else 'FAIL'}] {desc}"
    print(line)
    return line


if __name__ == "__main__":
    raise SystemExit(main())
