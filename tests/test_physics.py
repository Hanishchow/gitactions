import math

import numpy as np
import pytest
from pytest import approx

from ethylene.physics import (
    FEATURE_NAMES,
    SensorConfig,
    adc_to_rs,
    compensate,
    concentration,
    dht_in_range,
    f_rh,
    f_temp,
    features,
)


def test_f_temp_reference_point():
    assert f_temp(30.0) == approx(1.0)
    assert f_temp(0.0) == approx(math.sqrt(303.15 / 273.15))


def test_f_rh():
    assert f_rh(0) == 1.0
    assert f_rh(100) == approx(1 / 1.15)


def test_compensate():
    assert compensate(10, 30, 0) == approx(10)
    assert compensate(10, 30, 100) == approx(10 / 1.15)


def test_concentration_power_law():
    cfg = SensorConfig()
    assert concentration(1.0, cfg) == approx(cfg.A)
    assert concentration(0.5, cfg) == approx(cfg.A * 0.5**cfg.B)
    assert concentration(0.5, cfg) > concentration(1.0, cfg)


def test_adc_to_rs():
    assert adc_to_rs(511.5) == approx(10000)
    assert adc_to_rs(200) > adc_to_rs(400) > adc_to_rs(800)


def test_adc_rails_are_nan():
    assert math.isnan(adc_to_rs(0))
    assert math.isnan(adc_to_rs(1023))


def test_f_temp_invalid_is_nan():
    assert math.isnan(f_temp(-300))


def test_dht_in_range():
    assert dht_in_range(25, 60) is True
    assert dht_in_range(85, 0) is False
    assert dht_in_range(-5, 50) is False
    assert dht_in_range(25, 120) is False


def test_features_order_and_dtype():
    assert FEATURE_NAMES == ("rs_r0", "c_comp", "ldr_raw", "ldr_blk", "temp_c", "rh")
    vec = features(1.5, 2.5, 300.0, 120.0, 24.0, 55.0)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (6,)
    assert vec.tolist() == [1.5, 2.5, 300.0, 120.0, 24.0, 55.0]


def test_concentration_nonpositive_ratio_is_nan():
    assert math.isnan(concentration(0.0, SensorConfig()))
    assert math.isnan(concentration(-0.5, SensorConfig()))


def test_f_rh_degenerate_denominator_is_nan():
    assert math.isnan(f_rh(-100 / 0.15))
