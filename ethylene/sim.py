"""Synthetic ripening runs and sensor-fault injection.

`simulate` is the realistic generator (cultivar-to-cultivar variation, per-unit sensor spread, imperfect T/RH
compensation, interferent VOC episodes, drift, ADC quantisation, ambient light). `simulate_idealised` is the
original clean generator, kept only to show why it was too easy: a straight line on `c_comp` already scores
R2 = 0.996 on it. All ranges here are ASSUMPTIONS, not measurements.

`simulate_idealised` produces the training data for the models: ripeness `r` rises over each
run, the MQ response follows `c = A * (1 + g * r)`, and the temperature / humidity
compensation of `physics` is applied to the DHT11 reading exactly as the firmware does.
Everything is vectorised over (run, step) and driven by one `np.random.default_rng(seed)`.
"""
from dataclasses import dataclass
from functools import lru_cache

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


def simulate_idealised(n_runs: int, steps: int, seed: int, cfg: SensorConfig = SensorConfig(), g: float = 3.0) -> Dataset:
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


# ---------------------------------------------------------------------------------------------
# Realistic generator
# ---------------------------------------------------------------------------------------------
N_CULTIVARS = 32


@dataclass(frozen=True)
class SimProfile:
    """Operating conditions of a simulated batch. The defaults are the training distribution."""

    cultivars: tuple[int, int] = (0, 10)           # cultivar ids drawn uniformly from [lo, hi)
    temp_range: tuple[float, float] = (8.0, 30.0)  # run-mean temperature, degC
    rh_range: tuple[float, float] = (55.0, 95.0)   # run-mean relative humidity, %
    sensor_spread: float = 0.10                    # sd of per-unit gain A (relative); exponent B gets half
    comp_mismatch: float = 0.35                    # sd of the per-run error in the sensor's true T/RH exponent
    interferent_rate: float = 0.6                  # expected non-ripening VOC episodes per run
    drift_sd: float = 0.04                         # sd of per-run slow multiplicative sensor drift
    adc_noise: float = 1.5                         # ADC noise, counts


PROFILE_TRAIN = SimProfile()
OOD_PROFILES = {
    "unseen_cultivars": SimProfile(cultivars=(10, 20)),
    "cold_chain": SimProfile(temp_range=(1.0, 10.0), rh_range=(80.0, 99.0)),
    "sensor_shift": SimProfile(sensor_spread=0.25, drift_sd=0.10, comp_mismatch=0.6, interferent_rate=1.5),
}


def cultivar_params(k: int) -> dict:
    """Deterministic traits of cultivar `k`: emission gain, response shape, clean offset, LDR darkening, speed."""
    rng = np.random.default_rng(7_000_003 + int(k))
    return {
        "gain": float(np.exp(rng.normal(0.0, 0.30))),
        "shape": float(rng.uniform(0.8, 1.8)),
        "base": float(rng.uniform(0.0, 0.15)),
        "dark": float(rng.uniform(0.15, 0.5)),
        "speed": float(rng.uniform(0.7, 1.4)),
    }


@lru_cache(maxsize=1)
def _cultivar_table() -> dict:
    rows = [cultivar_params(k) for k in range(N_CULTIVARS)]
    return {name: np.array([row[name] for row in rows]) for name in rows[0]}


def simulate(n_runs: int, steps: int, seed: int, profile: SimProfile = PROFILE_TRAIN,
             cfg: SensorConfig = SensorConfig(), g: float = 3.0) -> Dataset:
    """`n_runs` realistic ripening runs of `steps` samples, rows ordered run-major then time.

    Ripeness `r` (the label) runs from a random start to a random end. What the sensor sees is the volatile
    envelope `v = base + gain * r**shape` of that cultivar plus random interferent episodes; the true sensor
    (unit-specific A, B, R0, drift, T/RH response) turns it into a resistance, which is quantised by a noisy
    10-bit ADC. Features are then computed exactly as the firmware does (R0 estimated with 1% error, nominal
    A and B, the paper's T/RH factors applied to the DHT11 reading). `group` holds the cultivar id.
    """
    if n_runs < 1 or steps < 1:
        raise ValueError(f"n_runs and steps must be >= 1, got {n_runs}, {steps}")
    lo_k, hi_k = profile.cultivars
    if not 0 <= lo_k < hi_k <= N_CULTIVARS:
        raise ValueError(f"cultivars must satisfy 0 <= lo < hi <= {N_CULTIVARS}, got {profile.cultivars}")
    rng = np.random.default_rng(seed)
    shape = (n_runs, steps)

    def per_run(lo, hi):
        return rng.uniform(lo, hi, (n_runs, 1))

    t_s = STEP_S * np.arange(steps)
    t_idx = np.arange(steps)[None, :]
    t_norm = np.linspace(0.0, 1.0, steps)[None, :]

    k = rng.integers(lo_k, hi_k, n_runs)
    table = _cultivar_table()
    gain, shape_p, base, dark, speed = (table[name][k][:, None] for name in ("gain", "shape", "base", "dark", "speed"))

    # Sensor unit.
    a_unit = cfg.A * (1.0 + rng.normal(0.0, profile.sensor_spread, (n_runs, 1)))
    b_unit = cfg.B * (1.0 + rng.normal(0.0, profile.sensor_spread / 2.0, (n_runs, 1)))
    r0_ohm = np.clip(np.exp(rng.normal(np.log(20000.0), 0.35, (n_runs, 1))), 6000.0, 80000.0)
    k_temp = np.clip(1.0 + rng.normal(0.0, profile.comp_mismatch, (n_runs, 1)), 0.1, 3.0)
    k_rh = np.clip(1.0 + rng.normal(0.0, profile.comp_mismatch, (n_runs, 1)), 0.1, 3.0)
    drift = rng.normal(0.0, profile.drift_sd, (n_runs, 1))

    # Ripeness path (label).
    r_start = per_run(0.0, 0.45)
    r_end = np.minimum(np.maximum(per_run(0.6, 1.0), r_start + 0.25), 1.0)
    r = np.clip(r_start + (r_end - r_start) * t_norm ** speed + rng.normal(0.0, 0.015, shape), 0.0, 1.0)

    # Environment: true values drive the sensor, the DHT11 reports a coarse, noisy, integral version.
    temp_true = np.clip(per_run(*profile.temp_range) + _drift(rng, n_runs, t_s, 3.0), 0.0, 50.0)
    rh_true = np.clip(per_run(*profile.rh_range) + _drift(rng, n_runs, t_s, 6.0), 10.0, 100.0)
    temp_c = np.clip(np.rint(temp_true + rng.normal(0.0, 1.0, shape)), 0.0, 50.0)
    rh = np.clip(np.rint(rh_true + rng.normal(0.0, 3.0, shape)), 0.0, 100.0)

    # Volatile envelope (units of the clean-air level) with interferent episodes unrelated to ripeness.
    v = base + gain * r ** shape_p
    for _ in range(3):
        active = rng.random((n_runs, 1)) < profile.interferent_rate / 3.0
        start = per_run(0.0, steps)
        tau = per_run(5.0, 30.0)
        amp = per_run(0.05, 0.6)
        v = v + np.where(active & (t_idx >= start), amp * np.exp(-(t_idx - start) / tau), 0.0)

    # True sensor response -> resistance -> noisy quantised ADC.
    c_raw = (a_unit * (1.0 + g * v) / (_f_temp(temp_true) ** k_temp * _f_rh(rh_true) ** k_rh)
             * (1.0 + drift * t_norm) * (1.0 + rng.normal(0.0, 0.02, shape)))
    ratio = (c_raw / a_unit) ** (1.0 / b_unit)
    adc = np.clip(np.rint(cfg.adc_max * cfg.rl_ohm / (r0_ohm * ratio + cfg.rl_ohm)
                          + rng.normal(0.0, profile.adc_noise, shape)), 1, cfg.adc_max - 1)

    # Firmware-side features.
    rs_meas = cfg.rl_ohm * (cfg.adc_max - adc) / adc
    rs_r0 = rs_meas / (r0_ohm * (1.0 + rng.normal(0.0, 0.01, (n_runs, 1))))
    c_comp = cfg.A * rs_r0 ** cfg.B * _f_temp(temp_c) * _f_rh(rh)

    # LDR: ambient light varies independently of ripeness; fruit darkens with ripeness (cultivar-specific).
    l0 = per_run(300.0, 800.0)
    light = 1.0 + _drift(rng, n_runs, t_s, 0.08) + rng.normal(0.0, 0.04, shape)
    ldr_raw = np.clip(np.rint(l0 * light * (1.0 - dark * r) + rng.normal(0.0, 8.0, shape)), 0, cfg.adc_max)
    ldr_blk = np.clip(np.rint(l0 * light[:, :1] * (1.0 + rng.normal(0.0, 0.02, (n_runs, 1)))), 0, cfg.adc_max)
    ldr_blk = np.broadcast_to(ldr_blk, shape)

    cols = {"rs_r0": rs_r0, "c_comp": c_comp, "ldr_raw": ldr_raw, "ldr_blk": ldr_blk, "temp_c": temp_c, "rh": rh}
    n = n_runs * steps
    return Dataset(
        X=np.stack([cols[name] for name in FEATURE_NAMES], axis=-1).reshape(n, len(FEATURE_NAMES)).astype(np.float32),
        y=r.reshape(n).astype(np.float32),
        run=np.repeat(np.arange(n_runs, dtype=np.int32), steps),
        t_s=np.tile(t_s, n_runs).astype(np.float32),
        flags=np.zeros(n, dtype=np.int32),
        group=np.repeat(k.astype(np.int32), steps),
    )
