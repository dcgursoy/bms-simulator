"""Synthetic drive-cycle current profiles for the pack.

Generates an urban/highway-style pulsed load: idle stretches, hard
acceleration bursts, cruise segments, and regenerative-braking charge
pulses, drawn from a seeded segment grammar so profiles are reproducible
but non-repeating. Long rests are inserted at the start, middle, and end
— they matter for estimation (at rest the terminal voltage relaxes
toward OCV, which is when SOC becomes directly observable) and they are
where real vehicles spend much of their time anyway.

Currents are in amps at pack level (positive = discharge), scaled by the
nominal 1C current so the same grammar works for any cell size.
"""

from __future__ import annotations

import numpy as np

#: (probability, (min_dur_s, max_dur_s), (min_c_rate, max_c_rate))
_SEGMENTS = {
    "idle": (0.25, (5.0, 20.0), (0.0, 0.0)),
    "accel": (0.20, (3.0, 6.0), (1.5, 3.0)),
    "cruise": (0.40, (10.0, 40.0), (0.30, 0.60)),
    "regen": (0.15, (2.0, 6.0), (-1.0, -0.35)),
}


def synth_drive_cycle(
    duration_s: float,
    dt_s: float,
    i_1c_a: float,
    seed: int = 0,
    rest_s: tuple[float, float, float] = (300.0, 600.0, 900.0),
) -> np.ndarray:
    """Return the pack current [A] at every tick of an urban-style drive
    cycle: rest_s[0] of rest, driving, rest_s[1] mid-drive rest, driving,
    rest_s[2] of final rest."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt_s))
    i = np.zeros(n)

    names = list(_SEGMENTS)
    probs = np.array([_SEGMENTS[k][0] for k in names])
    probs = probs / probs.sum()

    drive_total = duration_s - sum(rest_s)
    half = drive_total / 2.0
    # tick spans of the two driving blocks
    blocks = [
        (rest_s[0], rest_s[0] + half),
        (rest_s[0] + half + rest_s[1], rest_s[0] + half + rest_s[1] + half),
    ]
    for t0, t1 in blocks:
        t = t0
        while t < t1:
            name = rng.choice(names, p=probs)
            _, (d_lo, d_hi), (c_lo, c_hi) = _SEGMENTS[name]
            dur = min(rng.uniform(d_lo, d_hi), t1 - t)
            c_rate = rng.uniform(c_lo, c_hi)
            k0, k1 = int(round(t / dt_s)), int(round((t + dur) / dt_s))
            i[k0:k1] = c_rate * i_1c_a
            t += dur
    return i
