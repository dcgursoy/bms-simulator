"""Phase 2 validation: 48-cell series pack + bandwidth-limited BMS bus.

Runs 3 full CC-CV cycles (1C discharge to first-cell cutoff, rest, C/2
CC-CV charge, rest) where the *protocol controller only sees bus
telemetry* — noisy, quantized, up to 0.9 s stale — exactly like a real
pack controller.

Checks
------
1. Manufacturing variation lands at the configured spread and every cell
   is unique.
2. Usable pack capacity is clipped by the weakest cell.
3. Cell voltage spread blows up at the discharge knee (weak cells dive
   first) relative to mid-discharge.
4. Top-of-charge SOC spread grows cycle over cycle (coulombic-efficiency
   divergence) and bottom spread reflects the capacity spread.
5. Telemetry staleness is bounded by the round-robin period and the
   reported-vs-true error is consistent with sensor noise + staleness.

Outputs: PNGs + validation_summary.md in results/phase2/.
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

from model.cell import CellParams  # noqa: E402
from model.comms import BmsBus  # noqa: E402
from model.pack import Pack  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "phase2"
DT = 0.1                       # simulation tick [s]
NOMINAL = CellParams()
I_1C = NOMINAL.q_nom_ah        # 2.5 A
I_CHG = 0.5 * I_1C             # C/2 charge
I_TAPER_STOP = I_1C / 50.0     # CV taper termination current
AMBIENT_C = 25.0


class PackSim:
    """Steps pack + bus together and accumulates the validation logs."""

    def __init__(self):
        self.pack = Pack(soc0=1.0)
        self.bus = BmsBus(self.pack)
        self.t = 0.0
        self.temps = np.full(self.pack.n_cells, AMBIENT_C)
        self.log: dict[str, list] = {k: [] for k in (
            "t", "i", "v_cells", "soc", "tel_err_rms", "tel_age_max")}
        self.hires: dict[str, list] = {k: [] for k in ("t", "true0", "rep0")}
        self.hires_window = (1000.0, 1030.0)
        self._ticks = 0

    # -- telemetry-facing views (what the protocol controller may use) ------

    def reported_vmin(self) -> float:
        v = self.bus.telemetry.v
        return float(np.nanmin(v)) if np.isfinite(v).any() else self.pack.v_min_cell

    def reported_vmax(self) -> float:
        v = self.bus.telemetry.v
        return float(np.nanmax(v)) if np.isfinite(v).any() else self.pack.v_max_cell

    # ----------------------------------------------------------------------

    def tick(self, i_pack: float) -> None:
        self.pack.step(i_pack, DT, self.temps)
        self.t += DT
        self.bus.step(self.t, DT, self.temps)
        self._ticks += 1
        if self._ticks % 10 == 0:  # 1 s logging cadence
            tel = self.bus.telemetry
            mask = np.isfinite(tel.v)
            err = tel.v[mask] - self.pack.last_v[mask]
            self.log["t"].append(self.t)
            self.log["i"].append(i_pack)
            self.log["v_cells"].append(self.pack.last_v.copy())
            self.log["soc"].append(self.pack.cells.soc.copy())
            self.log["tel_err_rms"].append(float(np.sqrt(np.mean(err**2))))
            self.log["tel_age_max"].append(float(np.max(tel.v_age(self.t)[mask])))
        lo, hi = self.hires_window
        if lo <= self.t < hi:
            self.hires["t"].append(self.t)
            self.hires["true0"].append(float(self.pack.last_v[0]))
            self.hires["rep0"].append(float(self.bus.telemetry.v[0]))

    def run_phase(self, kind: str, max_s: float) -> float:
        """Run one protocol phase; returns its duration [s]."""
        t0 = self.t
        while self.t - t0 < max_s:
            if kind == "discharge":
                if self.reported_vmin() <= NOMINAL.v_min:
                    break
                self.tick(I_1C)
            elif kind == "charge":
                # CC-CV via proportional taper on the reported max cell
                err = NOMINAL.v_max - self.reported_vmax()
                mag = float(np.clip(I_CHG * err / 0.05, 0.0, I_CHG))
                if mag < I_TAPER_STOP:
                    break
                self.tick(-mag)
            else:  # rest
                self.tick(0.0)
        return self.t - t0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sim = PackSim()
    pack = sim.pack
    checks: list[str] = []
    print("== Phase 2 pack + comms validation ==")

    cap_mult = pack.cells.q_nom_ah / NOMINAL.q_nom_ah
    r0_mult = pack.cells.r0 / NOMINAL.r0
    print(f"pack: {pack.config.n_modules} modules x "
          f"{pack.config.cells_per_module}s = {pack.n_cells} cells, "
          f"nominal {pack.n_cells * 3.7:.0f} V / {NOMINAL.q_nom_ah} Ah")

    # ---------------------------------------------------------- run 3 cycles
    wall = time.time()
    per_cycle = []
    for cycle in range(3):
        dis_s = sim.run_phase("discharge", 3600 * 2)
        delivered_ah = I_1C * dis_s / 3600.0
        soc_bottom = pack.cells.soc.copy()
        sim.run_phase("rest", 600)
        sim.run_phase("charge", 3600 * 4)
        sim.run_phase("rest", 600)
        soc_top = pack.cells.soc.copy()
        per_cycle.append({
            "delivered_ah": delivered_ah,
            "bottom_spread": float(np.ptp(soc_bottom)),
            "top_spread": float(np.ptp(soc_top)),
        })
        print(f"cycle {cycle + 1}: delivered {delivered_ah:.3f} Ah, "
              f"SOC spread bottom {100 * per_cycle[-1]['bottom_spread']:.2f}% / "
              f"top {100 * per_cycle[-1]['top_spread']:.2f}%")
    print(f"simulated {sim.t / 3600:.2f} h in {time.time() - wall:.1f} s wall")

    log_t = np.asarray(sim.log["t"])
    log_v = np.asarray(sim.log["v_cells"])          # (n_samples, 48)
    log_soc = np.asarray(sim.log["soc"])
    log_i = np.asarray(sim.log["i"])

    # ------------------------------------------------------------ checks 1-2
    checks.append(_check(
        f"Capacity spread sigma {100 * np.std(cap_mult):.2f}% ~= configured "
        f"{100 * pack.config.sigma_capacity:.1f}%, all 48 cells unique",
        0.008 < np.std(cap_mult) < 0.022 and len(np.unique(cap_mult)) == 48,
    ))
    min_cap = float(np.min(pack.cells.q_nom_ah))  # fresh pack @ 25C
    d1 = per_cycle[0]["delivered_ah"]
    checks.append(_check(
        f"Usable capacity clipped by weakest cell: delivered {d1:.3f} Ah vs "
        f"weakest {min_cap:.3f} Ah (mean cell {np.mean(pack.cells.q_nom_ah):.3f})",
        0.90 * min_cap <= d1 <= min_cap,
    ))

    # ------------------------------------------------- check 3: knee spread
    first_dis = np.flatnonzero(log_i > 0)
    end = first_dis[-1]
    mid = first_dis[len(first_dis) // 2]
    spread_end = float(np.ptp(log_v[end]))
    spread_mid = float(np.ptp(log_v[mid]))
    checks.append(_check(
        f"Voltage spread amplifies at the knee: {1e3 * spread_end:.0f} mV at "
        f"cutoff vs {1e3 * spread_mid:.0f} mV mid-discharge (x"
        f"{spread_end / spread_mid:.1f})",
        spread_end > 2.5 * spread_mid,
    ))

    # ----------------------------------------------- check 4: SOC imbalance
    top1, top3 = per_cycle[0]["top_spread"], per_cycle[2]["top_spread"]
    checks.append(_check(
        f"Bottom SOC spread reflects capacity spread "
        f"({100 * per_cycle[0]['bottom_spread']:.2f}% > 3%)",
        per_cycle[0]["bottom_spread"] > 0.03,
    ))
    checks.append(_check(
        f"Top-of-charge SOC spread grows with cycling "
        f"({100 * top1:.2f}% -> {100 * top3:.2f}%)",
        top3 > top1,
    ))

    # --------------------------------------------------- check 5: telemetry
    age_max = float(np.max(sim.log["tel_age_max"]))
    err_rms = float(np.sqrt(np.mean(np.asarray(sim.log["tel_err_rms"])**2)))
    refresh = sim.bus.config.refresh_period_s(pack.config.n_modules)
    rep0 = np.asarray(sim.hires["rep0"])
    n_updates = int(np.sum(np.abs(np.diff(rep0)) > 0)) + 1
    expected_updates = 30.0 / refresh
    checks.append(_check(
        f"Telemetry staleness bounded by round-robin period "
        f"(max age {age_max:.2f} s <= {refresh:.2f} s + tick)",
        age_max <= refresh + DT + 1e-9,
    ))
    checks.append(_check(
        f"Reported-vs-true RMS error {1e3 * err_rms:.1f} mV consistent with "
        "noise + quantization + staleness (< 8 mV)",
        err_rms < 0.008,
    ))
    checks.append(_check(
        f"Cell 0 updated ~{n_updates} times in 30 s (expected ~"
        f"{expected_updates:.0f} at {refresh:.1f} s refresh)",
        abs(n_updates - expected_updates) <= 5,
    ))

    _make_plots(sim, cap_mult, r0_mult, log_t, log_v, log_soc, log_i, first_dis)

    n_fail = sum(1 for c in checks if c.startswith("[FAIL]"))
    lines = [
        "# Phase 2 — pack + comms validation",
        "",
        f"{pack.config.n_modules} modules x {pack.config.cells_per_module}s "
        f"= {pack.n_cells} cells (~{pack.n_cells * 3.7:.0f} V nominal); bus "
        f"{sim.bus.config.frames_per_s:.0f} frames/s -> full-pack refresh "
        f"{refresh:.2f} s.",
        "",
        "Per-cycle results:",
        "",
        "| cycle | delivered [Ah] | SOC spread bottom | SOC spread top |",
        "|---|---|---|---|",
    ]
    for k, c in enumerate(per_cycle):
        lines.append(f"| {k + 1} | {c['delivered_ah']:.3f} | "
                     f"{100 * c['bottom_spread']:.2f}% | "
                     f"{100 * c['top_spread']:.2f}% |")
    lines += ["", "```", *checks, "```", "",
              f"Result: {len(checks) - n_fail}/{len(checks)} checks passed."]
    (OUT_DIR / "validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed; "
          f"plots + summary in {OUT_DIR.relative_to(REPO_ROOT)}")
    return 1 if n_fail else 0


def _make_plots(sim, cap_mult, r0_mult, log_t, log_v, log_soc, log_i, first_dis):
    th = log_t / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mult, name, sigma in (
        (axes[0], cap_mult, "capacity", sim.pack.config.sigma_capacity),
        (axes[1], r0_mult, "R0", sim.pack.config.sigma_r0),
    ):
        ax.hist(100 * (mult - 1.0), bins=16, edgecolor="k", alpha=0.75)
        ax.set_xlabel(f"{name} deviation from nominal [%]")
        ax.set_ylabel("cells")
        ax.set_title(f"{name} spread (sigma target {100 * sigma:.1f}%)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variation_hist.png", dpi=140)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(th, np.sum(log_v, axis=1))
    ax1.set_ylabel("Pack voltage [V]")
    ax1.set_title("3 CC-CV cycles, controller driven by bus telemetry")
    ax1.grid(alpha=0.3)
    ax2.plot(th, log_i)
    ax2.set_ylabel("Pack current [A]\n(+ = discharge)")
    ax2.set_xlabel("Time [h]")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pack_cycling.png", dpi=140)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    tail = first_dis[-450:]  # last ~7.5 min of the first discharge
    ax1.plot(log_t[tail] - log_t[tail[-1]], log_v[tail], lw=0.7, alpha=0.6)
    ax1.set_xlabel("Time before first-cell cutoff [s]")
    ax1.set_ylabel("Cell voltage [V]")
    ax1.set_title("All 48 cells at the discharge knee")
    ax1.grid(alpha=0.3)
    ax2.plot(th, 100 * np.ptp(log_soc, axis=1))
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("SOC spread max-min [%]")
    ax2.set_title("Imbalance develops over cycling (no balancing)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cell_spread.png", dpi=140)
    plt.close(fig)

    ht = np.asarray(sim.hires["t"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ht, sim.hires["true0"], lw=1.2, label="true cell 0 voltage")
    ax.step(ht, sim.hires["rep0"], where="post", lw=1.2,
            label="bus-reported (noisy, quantized, 0.9 s round-robin)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("What the pack controller actually sees (1C discharge)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comms_staleness.png", dpi=140)
    plt.close(fig)


def _check(desc: str, ok) -> str:
    line = f"[{'PASS' if bool(ok) else 'FAIL'}] {desc}"
    print(line)
    return line


if __name__ == "__main__":
    raise SystemExit(main())
