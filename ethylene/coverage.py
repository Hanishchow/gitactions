"""Stimulus-coverage check: is the data too narrow (under-stimulated) or outside what the hardware can give?

`coverage_report` summarises each feature (percentiles, physical-range violations, ADC rail hits) and, given the
scaler fitted on the training data, how much of a dataset lies far outside the training range.
"""

from __future__ import annotations

import numpy as np

from ethylene.data import Scaler
from ethylene.physics import FEATURE_NAMES

# Physical limits of the sensing chain (Mega 10-bit ADC, DHT11 datasheet range, plausible Rs/R0 for an MQ element).
PHYSICAL_RANGE = {
    "rs_r0": (0.05, 3.0),
    "c_comp": (0.0, 5000.0),
    "ldr_raw": (0.0, 1023.0),
    "ldr_blk": (0.0, 1023.0),
    "temp_c": (0.0, 50.0),
    "rh": (0.0, 100.0),
}
Z_FAR = 4.0  # standardised distance regarded as extrapolation


def coverage_report(X: np.ndarray, ref: Scaler | None = None) -> dict:
    """Per-feature stats of `X` (N,6). With `ref`, `frac_far` is the share of rows with |z| > Z_FAR."""
    X = np.asarray(X, dtype=np.float64)
    report = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        finite = col[np.isfinite(col)]
        lo, hi = PHYSICAL_RANGE[name]
        entry = {
            "min": float(finite.min()), "p1": float(np.percentile(finite, 1)), "median": float(np.median(finite)),
            "p99": float(np.percentile(finite, 99)), "max": float(finite.max()),
            "frac_nonfinite": float(1.0 - finite.size / col.size),
            "frac_outside_physical": float(np.mean((finite < lo) | (finite > hi))),
        }
        if ref is not None:
            z = (finite - ref.mean[i]) / ref.std[i]
            entry["frac_far"] = float(np.mean(np.abs(z) > Z_FAR))
        report[name] = entry
    return report
