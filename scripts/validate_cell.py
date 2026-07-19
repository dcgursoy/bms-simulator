"""Phase 1 validation: exercise a single cell in isolation and check that
the physics is sane before any pack/estimation work builds on it.

Checks
------
1. Constant-current discharge at 0.5C / 1C / 2C @ 25 degC and 1C @ 0 degC:
   delivered capacity must fall with rate and with cold; voltage sag must
   grow with rate.
2. HPPC-style pulse at 60% SOC: the instantaneous voltage step must
   recover R0, and the relaxation must show the two RC time constants.
3. C/25 quasi-static full charge then discharge: terminal voltage must
   trace a hysteresis loop bracketing the true OCV curve.
4. Capacity fade: measured 1C delivered capacity at fast-forwarded aged
   states must track the configured fade law.

Outputs: PNGs + validation_summary.md in results/phase1/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import CellArray, CellParams, ocv  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "phase1"
NOMINAL = CellParams()
C_RATE_AMPS = NOMINAL.q_nom_ah  # 1C current for the nominal cell [A]


# --------------------------------------------------------------------- runs


def run_constant_current(
    current_a: float,
    temp_c: float,
    soc0: float,
    dt_s: float,
    stop: str,
) -> dict[str, np.ndarray]:
    """Run one cell at constant current until the voltage limit.

    stop='discharge' ends at v_min, stop='charge' ends at v_max.
    Returns time [s], terminal voltage [V], SOC, and throughput [Ah].
    """
    cell = CellArray(1, soc0=soc0)
    t, v, soc, ah = [0.0], [float(cell.terminal_voltage(0.0, temp_c)[0])], [soc0], [0.0]
    max_steps = int(20 * 3600 / dt_s) * 3  # hard guard: 60 h of sim time
    for k in range(1, max_steps):
        vt = float(cell.step(current_a, dt_s, temp_c)[0])
        t.append(k * dt_s)
        v.append(vt)
        soc.append(float(cell.soc[0]))
        ah.append(abs(current_a) * k * dt_s / 3600.0)
        if stop == "discharge" and vt <= cell.p.v_min:
            break
        if stop == "charge" and vt >= cell.p.v_max:
            break
    return {k_: np.asarray(v_) for k_, v_ in
            {"t": t, "v": v, "soc": soc, "ah": ah}.items()}


def test_discharge_family() -> tuple[dict, list[str]]:
    """Rate and temperature dependence of the discharge curve."""
    cases = {
        "0.5C @ 25degC": (0.5 * C_RATE_AMPS, 25.0),
        "1C @ 25degC": (1.0 * C_RATE_AMPS, 25.0),
        "2C @ 25degC": (2.0 * C_RATE_AMPS, 25.0),
        "1C @ 0degC": (1.0 * C_RATE_AMPS, 0.0),
    }
    curves, delivered = {}, {}
    for label, (i_a, temp) in cases.items():
        r = run_constant_current(i_a, temp, soc0=1.0, dt_s=1.0, stop="discharge")
        curves[label] = r
        delivered[label] = r["ah"][-1]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, r in curves.items():
        ax.plot(r["ah"], r["v"], label=f"{label}  ({delivered[label]:.2f} Ah)")
    ax.axhline(NOMINAL.v_min, color="k", ls=":", lw=1, label="cutoff 3.0 V")
    ax.set_xlabel("Discharged capacity [Ah]")
    ax.set_ylabel("Terminal voltage [V]")
    ax.set_title("Constant-current discharge: rate and temperature dependence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "discharge_curves.png", dpi=140)
    plt.close(fig)

    d = delivered
    checks = [
        _check(
            "Delivered capacity falls with C-rate "
            f"(0.5C {d['0.5C @ 25degC']:.3f} > 1C {d['1C @ 25degC']:.3f} "
            f"> 2C {d['2C @ 25degC']:.3f} Ah)",
            d["0.5C @ 25degC"] > d["1C @ 25degC"] > d["2C @ 25degC"],
        ),
        _check(
            "Cold cuts delivered capacity "
            f"(1C@0degC {d['1C @ 0degC']:.3f} < 1C@25degC {d['1C @ 25degC']:.3f} Ah)",
            d["1C @ 0degC"] < 0.95 * d["1C @ 25degC"],
        ),
    ]
    return {"delivered_ah": d}, checks


def test_pulse_response() -> tuple[dict, list[str]]:
    """HPPC-style pulse: rest 60 s -> 1C discharge 10 s -> rest 600 s."""
    dt = 0.1
    i_pulse = 1.0 * C_RATE_AMPS
    cell = CellArray(1, soc0=0.60)
    t, v, i_log = [], [], []
    plan = [(60.0, 0.0), (10.0, i_pulse), (600.0, 0.0)]
    now = 0.0
    for duration, i_a in plan:
        for _ in range(int(round(duration / dt))):
            vt = float(cell.step(i_a, dt, 25.0)[0])
            now += dt
            t.append(now)
            v.append(vt)
            i_log.append(i_a)
    t, v, i_log = np.asarray(t), np.asarray(v), np.asarray(i_log)

    # Instantaneous step across the pulse edge isolates R0 (RC branches
    # barely move in one 0.1 s step: exp(-0.1/30) ~ 0.997)
    edge = np.flatnonzero(np.diff(i_log) > 0)[0]  # last rest sample index
    r0_recovered = (v[edge] - v[edge + 1]) / i_pulse
    r0_true = float(cell.resistances(25.0)[0][0])

    # Total DC-ish resistance after the full 10 s pulse (R0 + partial RC)
    pulse_end = np.flatnonzero(np.diff(i_log) < 0)[0]
    r_10s = (v[edge] - v[pulse_end]) / i_pulse

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(t, v)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Terminal voltage [V]")
    ax1.set_title("1C / 10 s pulse at 60% SOC")
    ax1.grid(alpha=0.3)
    ax1.annotate(
        f"instant step = i*R0\nR0 recovered = {1e3 * r0_recovered:.1f} mOhm",
        xy=(t[edge + 1], v[edge + 1]),
        xytext=(t[edge + 1] + 60, v[edge + 1] - 0.01),
        arrowprops=dict(arrowstyle="->"),
    )
    mask = t > t[pulse_end]
    ax2.semilogy(t[mask] - t[pulse_end], v[-1] - v[mask] + 1e-6)
    ax2.set_xlabel("Time since pulse end [s]")
    ax2.set_ylabel("Voltage recovery deficit [V] (log)")
    ax2.set_title("Two-time-constant relaxation (~30 s and ~600 s)")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pulse_response.png", dpi=140)
    plt.close(fig)

    checks = [
        _check(
            f"Pulse edge recovers R0 ({1e3 * r0_recovered:.2f} mOhm vs "
            f"true {1e3 * r0_true:.2f} mOhm)",
            abs(r0_recovered - r0_true) / r0_true < 0.05,
        ),
        _check(
            f"10 s resistance {1e3 * r_10s:.1f} mOhm exceeds R0 "
            "(RC branches charging)",
            r_10s > r0_recovered * 1.2,
        ),
    ]
    return {"r0_recovered_mohm": 1e3 * r0_recovered, "r_10s_mohm": 1e3 * r_10s}, checks


def test_hysteresis_loop() -> tuple[dict, list[str]]:
    """C/25 full charge then full discharge traces the quasi-OCV loop."""
    i_slow = C_RATE_AMPS / 25.0
    chg = run_constant_current(-i_slow, 25.0, soc0=0.0, dt_s=10.0, stop="charge")
    dis = run_constant_current(+i_slow, 25.0, soc0=1.0, dt_s=10.0, stop="discharge")

    soc_grid = np.linspace(0.05, 0.95, 181)
    # Interpolate both branches onto a common SOC grid (charge SOC ascends,
    # discharge SOC descends)
    v_chg = np.interp(soc_grid, chg["soc"], chg["v"])
    v_dis = np.interp(soc_grid, dis["soc"][::-1], dis["v"][::-1])
    gap = v_chg - v_dis
    gap_mid = float(np.interp(0.5, soc_grid, gap))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(chg["soc"], chg["v"], label="C/25 charge branch")
    ax.plot(dis["soc"], dis["v"], label="C/25 discharge branch")
    ax.plot(soc_grid, ocv(soc_grid), "k--", lw=1.2, label="true OCV")
    ax.set_xlabel("SOC")
    ax.set_ylabel("Terminal voltage [V]")
    ax.set_title(
        f"Quasi-static hysteresis loop (gap at 50% SOC = {1e3 * gap_mid:.1f} mV)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hysteresis_loop.png", dpi=140)
    plt.close(fig)

    # Expected gap: 2*M (hysteresis) + 2*i*(R0+R1+R2) (IR offset at C/25)
    r_tot = NOMINAL.r0 + NOMINAL.r1 + NOMINAL.r2
    gap_expected = 2.0 * NOMINAL.m_hyst + 2.0 * i_slow * r_tot
    checks = [
        _check(
            f"Charge branch sits above discharge branch everywhere "
            f"(min gap {1e3 * gap.min():.1f} mV)",
            bool(np.all(gap > 0)),
        ),
        _check(
            f"Mid-SOC gap {1e3 * gap_mid:.1f} mV ~= expected "
            f"2M + 2iR = {1e3 * gap_expected:.1f} mV",
            abs(gap_mid - gap_expected) < 0.010,
        ),
    ]
    return {"gap_mid_mv": 1e3 * gap_mid}, checks


def test_capacity_fade() -> tuple[dict, list[str]]:
    """Measured 1C delivered capacity at aged states vs the fade law."""
    efc_marks = [0, 200, 400, 600, 800]
    measured = []
    for efc in efc_marks:
        cell = CellArray(1, soc0=1.0)
        cell.fast_forward_aging(efc)
        t, dt = 0.0, 1.0
        while True:
            vt = float(cell.step(C_RATE_AMPS, dt, 25.0)[0])
            t += dt
            if vt <= cell.p.v_min or t > 3600 * 5:
                break
        measured.append(C_RATE_AMPS * t / 3600.0)

    efc_grid = np.linspace(0, 900, 200)
    probe = CellArray(1)
    law = []
    for efc in efc_grid:
        probe.fast_forward_aging(efc)
        law.append(float(probe.capacity_ah()[0]))
    law = np.asarray(law)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(efc_grid, law, label="fade law: effective capacity")
    ax.plot(efc_marks, measured, "o", label="measured 1C delivered capacity")
    ax.set_xlabel("Equivalent full cycles")
    ax.set_ylabel("Capacity [Ah]")
    ax.set_title("Capacity fade: law vs measured delivered capacity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "capacity_fade.png", dpi=140)
    plt.close(fig)

    # Compare *relative* fade so the rate-dependent cutoff offset cancels;
    # aged cells also grow R0 (more sag, earlier cutoff), so allow 2%
    rel_measured = measured[-1] / measured[0]
    probe.fast_forward_aging(efc_marks[-1])
    rel_law = float(probe.capacity_ah()[0]) / NOMINAL.q_nom_ah
    checks = [
        _check(
            f"Relative capacity at 800 EFC: measured {rel_measured:.3f} vs "
            f"law {rel_law:.3f}",
            abs(rel_measured - rel_law) < 0.02,
        ),
        _check(
            "Fade is monotonic in EFC",
            bool(np.all(np.diff(measured) < 0)),
        ),
    ]
    return {"measured_ah": dict(zip(map(str, efc_marks), measured))}, checks


# ------------------------------------------------------------------ plumbing


def _check(desc: str, ok: bool) -> str:
    line = f"[{'PASS' if ok else 'FAIL'}] {desc}"
    print(line)
    return line


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_checks: list[str] = []
    print("== Phase 1 cell-model validation ==")
    for name, fn in [
        ("Discharge family", test_discharge_family),
        ("Pulse response", test_pulse_response),
        ("Hysteresis loop", test_hysteresis_loop),
        ("Capacity fade", test_capacity_fade),
    ]:
        print(f"\n-- {name} --")
        _, checks = fn()
        all_checks.extend(checks)

    n_fail = sum(1 for c in all_checks if c.startswith("[FAIL]"))
    summary = "\n".join(
        [
            "# Phase 1 — cell model validation",
            "",
            f"Nominal cell: {NOMINAL.q_nom_ah} Ah NMC, R0={1e3 * NOMINAL.r0:.0f} mOhm, "
            f"tau1={NOMINAL.r1 * NOMINAL.c1:.0f} s, tau2={NOMINAL.r2 * NOMINAL.c2:.0f} s, "
            f"M_hyst={1e3 * NOMINAL.m_hyst:.0f} mV",
            "",
            "```",
            *all_checks,
            "```",
            "",
            f"Result: {len(all_checks) - n_fail}/{len(all_checks)} checks passed.",
        ]
    )
    (OUT_DIR / "validation_summary.md").write_text(summary, encoding="utf-8")
    print(f"\n{len(all_checks) - n_fail}/{len(all_checks)} checks passed; "
          f"plots + summary in {OUT_DIR.relative_to(REPO_ROOT)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
