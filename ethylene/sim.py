"""Synthetic ripening runs and sensor-fault injection.

`simulate` produces the training data for the models: ripeness `r` rises over each
run, the MQ response follows `c = A * (1 + g * r)`, and the temperature / humidity
compensation of `physics` is applied to the DHT11 reading exactly as the firmware does.
Everything is vectorised over (run, step) and driven by one `np.random.default_rng(seed)`.
"""
import numpy as np

from ethylene.data import Dataset
from ethylene.physics import FEATURE_NAMES, KELVIN, RH_COEFF, T_REF_K, SensorConfig

STAGE_MEANS = (0.05, 0.22, 0.42, 0.62, 0.82, 0.93)  # mean ripeness per stage; run ends are the extremes
FAULT_KINDS = ("rs_drift", "ldr_dead", "dht_zero", "c_comp_stuck")

STEP_S = 2.0  # seconds between samples
_COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _f_temp(temp_c):
    """Array form of `physics.f_temp` (same constants)."""
    return np.sqrt(T_REF_K / (temp_c + KELVIN))


def _f_rh(rh):
    """Array form of `physics.f_rh` (same constants)."""
    return 1.0 / (1.0 + RH_COEFF * rh / 100.0)


def _drift(rng, n_runs, t_s, amplitude):
    """Slow sinusoid per run, shape (n_runs, steps): one cycle takes 0.5-1.5 run lengths."""
    period = rng.uniform(0.5, 1.5, (n_runs, 1)) * len(t_s) * STEP_S
    phase = rng.uniform(0.0, 2.0 * np.pi, (n_runs, 1))
    return amplitude * np.sin(2.0 * np.pi * t_s / period + phase)


def simulate(n_runs: int, steps: int, seed: int, cfg: SensorConfig = SensorConfig(), g: float = 3.0) -> Dataset:
    """`n_runs` ripening runs of `steps` samples each, rows ordered run-major then time.

    Ripeness climbs from STAGE_MEANS[0] to STAGE_MEANS[-1] at a run-specific speed
    (power-law time warp) with N(0, 0.01) jitter, clipped to [0, 1]; `y` is that ripeness.
    The DHT11 reading is the rounded true value plus noise (so it is integral and, as on the
    real sensor chain, may stray outside the 0-100 % RH range by a few counts).
    """
    if n_runs < 1 or steps < 1:
        raise ValueError(f"n_runs and steps must be >= 1, got {n_runs}, {steps}")
    rng = np.random.default_rng(seed)
    shape = (n_runs, steps)

    def per_run(lo, hi):
        return rng.uniform(lo, hi, (n_runs, 1))

    t_s = STEP_S * np.arange(steps)

    # Ripeness path.
    lo, hi = STAGE_MEANS[0], STAGE_MEANS[-1]
    progress = np.linspace(0.0, 1.0, steps) ** per_run(0.7, 1.4)
    r = np.clip(lo + (hi - lo) * progress + rng.normal(0.0, 0.01, shape), 0.0, 1.0)

    # True and measured environment.
    temp_true = per_run(12, 28) + _drift(rng, n_runs, t_s, 3.0)
    rh_true = per_run(60, 92) + _drift(rng, n_runs, t_s, 5.0)
    temp_c = np.rint(temp_true + rng.normal(0.0, 0.7, shape))
    rh = np.rint(rh_true + rng.normal(0.0, 2.0, shape))

    # MQ response: raw concentration is what the sensor "sees" (T/RH-skewed), c_comp undoes it
    # with the *measured* T/RH.
    c_true = cfg.A * (1.0 + g * r)
    c_raw = c_true / (_f_temp(temp_true) * _f_rh(rh_true)) * (1.0 + rng.normal(0.0, 0.02, shape))
    rs_r0 = (c_raw / cfg.A) ** (1.0 / cfg.B)
    c_comp = c_raw * _f_temp(temp_c) * _f_rh(rh)

    # LDR: per-run blank level, darkening as the fruit ripens.
    ldr_blk = np.broadcast_to(per_run(500, 700), shape)
    ldr_raw = ldr_blk * (1.0 - 0.35 * r) + rng.normal(0.0, 8.0, shape)

    cols = {"rs_r0": rs_r0, "c_comp": c_comp, "ldr_raw": ldr_raw, "ldr_blk": ldr_blk, "temp_c": temp_c, "rh": rh}
    n = n_runs * steps
    return Dataset(
        X=np.stack([cols[name] for name in FEATURE_NAMES], axis=-1).reshape(n, len(FEATURE_NAMES)).astype(np.float32),
        y=r.reshape(n).astype(np.float32),
        run=np.repeat(np.arange(n_runs, dtype=np.int32), steps),
        t_s=np.tile(t_s, n_runs).astype(np.float32),
        flags=np.zeros(n, dtype=np.int32),
    )


def inject_faults(X: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Copy of `X` (feature axis last, same shape and dtype) with one sensor fault applied.

    rs_drift      rs_r0 scaled by one factor U(1.5, 2.5) drawn per call; c_comp is left alone
    ldr_dead      ldr_raw forced to 0
    dht_zero      temp_c and rh forced to 0
    c_comp_stuck  c_comp frozen at its median over `X`
    """
    if kind not in FAULT_KINDS:
        raise ValueError(f"unknown fault kind {kind!r}, expected one of {FAULT_KINDS}")
    out = np.array(X, copy=True)
    if kind == "rs_drift":
        out[..., _COL["rs_r0"]] *= rng.uniform(1.5, 2.5)
    elif kind == "ldr_dead":
        out[..., _COL["ldr_raw"]] = 0
    elif kind == "dht_zero":
        out[..., _COL["temp_c"]] = 0
        out[..., _COL["rh"]] = 0
    else:  # c_comp_stuck
        out[..., _COL["c_comp"]] = np.nanmedian(out[..., _COL["c_comp"]])
    return out
