"""Open-circuit voltage (OCV) curve for a graphite/NMC lithium-ion cell.

The OCV(SOC) relationship is the static backbone of the whole simulator:
the cell model adds dynamic overpotentials on top of it, and the state
estimators in /estimation invert it to recover SOC from measured voltage,
so its shape (steep knees at the ends, flat plateau mid-range) directly
drives estimator observability.

A shape-preserving PCHIP interpolant through anchor points styled after
published graphite/NMC OCV data is used instead of a polynomial fit:
monotonicity of the anchors is preserved exactly and there is no Runge
oscillation on the flat mid-SOC plateau, which would otherwise corrupt
the dOCV/dSOC term the EKF linearizes through.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

# (SOC, OCV [V]) anchors for a 3.0-4.2 V graphite/NMC cell at 25 degC.
_ANCHOR_SOC = np.array(
    [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
     0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
)
_ANCHOR_OCV = np.array(
    [3.000, 3.200, 3.350, 3.470, 3.530, 3.575, 3.625, 3.665,
     3.710, 3.775, 3.860, 3.950, 4.050, 4.110, 4.185]
)

_OCV = PchipInterpolator(_ANCHOR_SOC, _ANCHOR_OCV)
_DOCV = _OCV.derivative()


def ocv(soc):
    """Open-circuit voltage [V] at the given SOC. SOC is clipped to [0, 1]."""
    return _OCV(np.clip(soc, 0.0, 1.0))


def docv_dsoc(soc):
    """Slope dOCV/dSOC [V per unit SOC], clipped to the valid SOC range."""
    return _DOCV(np.clip(soc, 0.0, 1.0))
