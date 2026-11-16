import itertools
import math

import pytest

from ethylene.data import CSV_COLUMNS, FLAG_WARMUP
from ethylene.serial_io import Reading, SerialSource, SimulatedSource, parse_line

GOOD = "12000,512.0,0.8123,250.5,600.0,650.0,24.0,71.0,0.6234,0"


def test_parse_valid_line():
    r = parse_line(GOOD)
    assert r == Reading(t_ms=12000, adc_mq=512.0, rs_r0=0.8123, c_comp=250.5, ldr_raw=600.0,
                        ldr_blk=650.0, temp_c=24.0, rh=71.0, idx_dev=0.6234, flags=0, label=None)


def test_parse_label_column():
    assert parse_line(GOOD + ",0.55").label == 0.55


@pytest.mark.parametrize("line", [",".join(CSV_COLUMNS), "", "   ", "\r\n"])
def test_header_and_blank_return_none(line):
    assert parse_line(line) is None


@pytest.mark.parametrize("line", [
    "hello world", "\x00\x01\x02\xff\xfe", "12000,512.0,0.8", GOOD + ",1,2,3",
    "x,512.0,0.8123,250.5,600.0,650.0,24.0,71.0,0.6234,0",
    "12000,512.0,0.8123,250.5,600.0,650.0,24.0,71.0,0.6234,abc",
    "-5,512.0,0.8123,250.5,600.0,650.0,24.0,71.0,0.6234,0",
    "12000,512.0,0.8123,250.5,600.0,650.0,24.0,71.0,0.6234,1.5",
    "A" * 10_000,
])
def test_garbage_returns_none(line):
    assert parse_line(line) is None


def test_nan_and_idx_minus1_become_none():
    r = parse_line("500,0,nan,nan,600,650,nan,nan,-1,5")
    assert r.rs_r0 is None and r.c_comp is None and r.idx_dev is None
    assert r.temp_c is None and r.rh is None and r.flags == 5


def test_simulated_source_yields_valid_readings():
    rs = list(itertools.islice(SimulatedSource(seed=0), 50))
    assert len(rs) == 50
    assert rs[0].flags == FLAG_WARMUP and rs[0].rs_r0 is None
    for r in rs[5:]:
        assert r.flags == 0
        assert all(v is not None and math.isfinite(v) for v in (r.rs_r0, r.c_comp, r.temp_c, r.rh))
    assert [r.t_ms for r in rs] == sorted(r.t_ms for r in rs)


class FakeSerial:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else b""

    def close(self):
        pass


def test_serial_source_reconnects():
    calls = []

    def factory(port, baud, timeout):
        calls.append(port)
        if len(calls) == 1:
            raise OSError("device not ready")
        return FakeSerial([(GOOD + "\n").encode(), b"\xff\xfe garbage\n", (GOOD + "\n").encode()])

    src = SerialSource("COM9", serial_factory=factory, reconnect_delay_s=0.0)
    got = list(itertools.islice(src.readings(), 2))
    src.close()
    assert len(got) == 2 and all(r.t_ms == 12000 for r in got)
    assert calls == ["COM9", "COM9"]
