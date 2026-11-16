"""Parity between the Mega firmware sources and the Python single source of truth."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ethylene import data
from ethylene.physics import SensorConfig

FW = Path(__file__).resolve().parent.parent / "firmware" / "ethylene_mega"
FQBN = "arduino:avr:mega:cpu=atmega2560"
DEFAULT_CLI = Path("C:/Program Files/Arduino CLI/arduino-cli.exe")
MEGA_FLASH = 253952  # bytes available to sketches (256 KB minus 8 KB bootloader)
MEGA_RAM = 8192


def _define(text: str, name: str) -> str:
    m = re.search(rf"^\s*#define\s+{name}\s+(\S+)", text, re.MULTILINE)
    assert m, f"{name} not defined"
    return m.group(1).rstrip("fFuUlL")


def _cli():
    found = shutil.which("arduino-cli")
    if found:
        return found
    return str(DEFAULT_CLI) if DEFAULT_CLI.exists() else None


def test_config_constants_match_python():
    cfg_h = (FW / "config.h").read_text(encoding="utf-8")
    cfg = SensorConfig()
    assert float(_define(cfg_h, "SENSOR_A")) == pytest.approx(cfg.A, rel=1e-9)
    assert float(_define(cfg_h, "SENSOR_B")) == pytest.approx(cfg.B, rel=1e-9)
    assert float(_define(cfg_h, "RL_OHM")) == cfg.rl_ohm
    assert int(float(_define(cfg_h, "ADC_MAX"))) == cfg.adc_max
    for name in ("FLAG_WARMUP", "FLAG_DHT_FAIL", "FLAG_MQ_FAULT", "FLAG_VENT", "FLAG_SCRUB"):
        assert int(_define(cfg_h, name)) == getattr(data, name), name
    expected = {"WARMUP_MS": 180000, "BASELINE_MS": 30000, "SAMPLE_MS": 2000}
    for name, value in expected.items():
        assert int(_define(cfg_h, name)) == value, name
    for name, value in {"VENT_IDX": 0.90, "SCRUB_IDX": 0.85, "RELAY_HYST": 0.03}.items():
        assert float(_define(cfg_h, name)) == pytest.approx(value), name
    pins = {"PIN_MQ": "A0", "PIN_LDR": "A1", "PIN_DHT": "22", "PIN_VENT": "30", "PIN_SCRUB": "31",
            "PIN_LED": "13", "SERIAL_BAUD": "115200"}
    for name, value in pins.items():
        assert _define(cfg_h, name) == value, name

    ino = (FW / "ethylene_mega.ino").read_text(encoding="utf-8")
    m = re.search(r'#define\s+CSV_HEADER\s+"([^"]*)"', ino)
    assert m, "CSV_HEADER string not found in the sketch"
    assert m.group(1) == ",".join(data.CSV_COLUMNS)


def test_math_header_uses_paper_constants():
    src = (FW / "ethylene_math.h").read_text(encoding="utf-8")
    for const in ("303.15", "273.15", "0.15"):
        assert const in src, const


def test_placeholder_weights_header_layout():
    """Contracts section A: names and sizes model_weights.h must provide to mlp.h."""
    src = (FW / "model_weights.h").read_text(encoding="utf-8")
    assert "#include <avr/pgmspace.h>" in src
    dims = {"MLP_IN": 6, "MLP_L1": 256, "MLP_L2": 128, "MLP_L3": 64, "MLP_L4": 32, "MLP_OUT": 1}
    for name, value in dims.items():
        assert int(_define(src, name)) == value, name
    # W2 (32 KB) is split in two 64-row halves: avr-gcc rejects objects of 32768 bytes or more.
    # Sizes must be literals: 128*256 overflows the AVR's 16-bit int.
    sizes = {"W1": 1536, "W2A": 16384, "W2B": 16384, "W3": 8192, "W4": 2048, "W5": 32}
    for arr, n in sizes.items():
        assert re.search(rf"const int8_t {arr}\[{n}\] PROGMEM", src), arr
    for arr in ("S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4", "B5", "SCALER_MEAN", "SCALER_STD"):
        assert re.search(rf"const float {arr}\[[^\]]+\] PROGMEM", src), arr


def test_compile_if_toolchain(tmp_path):
    cli = _cli()
    if cli is None:
        pytest.skip("arduino-cli not installed")
    # Build outside the repo so no artifacts land in the checkout.
    proc = subprocess.run(
        [cli, "compile", "--fqbn", FQBN, "--build-path", str(tmp_path / "build"), str(FW)],
        capture_output=True, text=True, timeout=600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    flash = re.search(r"Sketch uses (\d+) bytes .*?Maximum is (\d+) bytes", out)
    ram = re.search(r"Global variables use (\d+) bytes .*?Maximum is (\d+) bytes", out)
    assert flash and ram, out
    assert int(flash.group(1)) < 0.90 * int(flash.group(2)), flash.group(0)
    assert int(ram.group(1)) < 0.75 * int(ram.group(2)), ram.group(0)
    assert int(flash.group(2)) == MEGA_FLASH and int(ram.group(2)) == MEGA_RAM
