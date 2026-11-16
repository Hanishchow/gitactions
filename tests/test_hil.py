import dataclasses
import math

import pytest

from ethylene import hil
from ethylene.export import read_header
from ethylene.serial_io import parse_line


@pytest.fixture(scope="module")
def q():
    return read_header(hil.HEADER_PATH)


def chip_line(exp: dict) -> str:
    """What the firmware would print for these reference values (4 decimals, nan, idx -1)."""
    f = lambda v: "nan" if v is None else f"{v:.4f}"
    return f"1000,500,{f(exp['rs_r0'])},{f(exp['c_comp'])},600.0000,650.0000,24.0000,60.0000,{'-1' if exp['idx'] is None else f(exp['idx'])},32"


def test_vectors_cover_rails_and_realistic_range():
    vs = hil.make_vectors(40)
    assert len(vs) == 43
    assert {v["adc_mq"] for v in vs[-3:]} == {0, 1023, 1}
    assert all(0 < v["adc_mq"] < 1023 for v in vs[:40])


def test_expected_flags_rail_readings(q):
    vs = hil.make_vectors(10)
    for v in vs[-3:-1]:  # adc 0 and 1023
        exp = hil.expected(v, q)
        assert exp["rs_r0"] is None and exp["c_comp"] is None and exp["idx"] is None
    assert hil.expected(vs[0], q)["idx"] is not None


def test_chip_output_with_firmware_rounding_agrees(q):
    for v in hil.make_vectors(40):
        exp = hil.expected(v, q)
        assert hil.compare(parse_line(chip_line(exp)), exp) == [], v


def test_compare_detects_wrong_chip_values(q):
    v = hil.make_vectors(10)[3]
    exp = hil.expected(v, q)
    good = parse_line(chip_line(exp))
    assert hil.compare(dataclasses.replace(good, idx_dev=good.idx_dev + 0.05), exp)
    assert hil.compare(dataclasses.replace(good, c_comp=good.c_comp * 1.01), exp)
    assert hil.compare(dataclasses.replace(good, rs_r0=None), exp)
    assert hil.compare(dataclasses.replace(good, flags=0), exp)


def test_vector_line_format():
    line = hil.vector_line({"adc_mq": 341, "ldr_raw": 600.0, "temp_c": 24.0, "rh": 60.0, "r0_ohm": 20000.0, "ldr_blk": 650.0})
    assert line == "T,341,600,24,60,20000,650"
    assert all(math.isfinite(float(x)) for x in line.split(",")[1:])
