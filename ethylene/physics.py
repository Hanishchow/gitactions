"""Sensor formulas: the single source of truth the Arduino firmware mirrors.

    c      = A * (Rs/R0) ** B
    f_T    = sqrt(303.15 / (T + 273.15))
    f_RH   = 1 / (1 + 0.15 * RH / 100)
    c_comp = c * f_T * f_RH

Pure Python `math` throughout; numpy is only used to build the model input
vector. Invalid inputs yield NaN rather than raising or returning inf.
"""
import math
from dataclasses import dataclass

import numpy as np

KELVIN = 273.15
T_REF_K = 303.15  # 30 degC reference temperature
RH_COEFF = 0.15

# Model input order. The firmware and the ML app must both follow it.
FEATURE_NAMES = ("rs_r0", "c_comp", "ldr_raw", "ldr_blk", "temp_c", "rh")


@dataclass(frozen=True)
class SensorConfig:
    A: float = 116.6020682
    B: float = -2.769034857
    rl_ohm: float = 10000.0
    adc_max: int = 1023


def adc_to_rs(adc: float, cfg: SensorConfig = SensorConfig()) -> float:
    """Sensor resistance (ohm) from a raw ADC reading; NaN on a saturated rail."""
    if adc <= 0 or adc >= cfg.adc_max:
        return float("nan")
    return cfg.rl_ohm * (cfg.adc_max - adc) / adc


def concentration(rs_r0: float, cfg: SensorConfig = SensorConfig()) -> float:
    """Concentration from the Rs/R0 ratio via the power law; NaN if ratio <= 0."""
    if rs_r0 <= 0:
        return float("nan")
    return cfg.A * rs_r0**cfg.B


def f_temp(temp_c: float) -> float:
    """Temperature correction factor; NaN at or below absolute zero."""
    if temp_c <= -KELVIN:
        return float("nan")
    return math.sqrt(T_REF_K / (temp_c + KELVIN))


def f_rh(rh: float) -> float:
    """Relative-humidity correction factor; NaN if the denominator is not positive."""
    denom = 1 + RH_COEFF * rh / 100
    if denom <= 0:
        return float("nan")
    return 1 / denom


def compensate(c: float, temp_c: float, rh: float) -> float:
    """Concentration corrected for temperature and humidity."""
    return c * f_temp(temp_c) * f_rh(rh)


def dht_in_range(temp_c: float, rh: float) -> bool:
    """True when a DHT reading is inside the sensor's valid range."""
    return 0 <= temp_c <= 50 and 0 <= rh <= 100


def features(
    rs_r0: float,
    c_comp: float,
    ldr_raw: float,
    ldr_blk: float,
    temp_c: float,
    rh: float,
) -> np.ndarray:
    """Model input vector in FEATURE_NAMES order, shape (6,), float32."""
    return np.array([rs_r0, c_comp, ldr_raw, ldr_blk, temp_c, rh], dtype=np.float32)
