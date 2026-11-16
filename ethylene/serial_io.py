"""Mega 2560 serial input: line parser, serial reader with reconnect, and a simulated feed.

Firmware line (see ``data.CSV_COLUMNS``): ``t_ms,adc_mq,rs_r0,c_comp,ldr_raw,ldr_blk,temp_c,rh,idx,flags``
(an optional 11th ``label`` column is accepted). ``nan`` fields and ``idx == -1`` mean "no value".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterator

from ethylene import physics, sim
from ethylene.data import CSV_COLUMNS, FLAG_WARMUP

MAX_LINE_CHARS = 512


@dataclass(frozen=True)
class Reading:
    t_ms: int
    adc_mq: float | None
    rs_r0: float | None
    c_comp: float | None
    ldr_raw: float | None
    ldr_blk: float | None
    temp_c: float | None
    rh: float | None
    idx_dev: float | None
    flags: int
    label: float | None = None


def _num(cell: str) -> float | None:
    """Float, or None for ``nan``/empty/inf. Raises ValueError for non-numeric text."""
    cell = cell.strip()
    if not cell:
        return None
    value = float(cell)
    return value if math.isfinite(value) else None


def parse_line(line: str) -> Reading | None:
    """Parse one firmware CSV line. Returns None for anything that is not a data line; never raises."""
    try:
        if not isinstance(line, str) or len(line) > MAX_LINE_CHARS:
            return None
        cells = line.strip().split(",")
        if len(cells) not in (len(CSV_COLUMNS), len(CSV_COLUMNS) + 1):
            return None
        if cells[0].strip() == CSV_COLUMNS[0]:  # header
            return None
        t_ms = _num(cells[0])
        flags = _num(cells[9])
        if t_ms is None or flags is None or t_ms < 0 or flags < 0 or flags != int(flags):
            return None
        idx = _num(cells[8])
        label = _num(cells[10]) if len(cells) == len(CSV_COLUMNS) + 1 else None
        return Reading(
            t_ms=int(t_ms),
            adc_mq=_num(cells[1]),
            rs_r0=_num(cells[2]),
            c_comp=_num(cells[3]),
            ldr_raw=_num(cells[4]),
            ldr_blk=_num(cells[5]),
            temp_c=_num(cells[6]),
            rh=_num(cells[7]),
            idx_dev=None if idx is None or idx == -1 else idx,
            flags=int(flags),
            label=label,
        )
    except (ValueError, OverflowError, IndexError):
        return None


class SimulatedSource:
    """Yields synthetic readings from the ripening simulator, with a short warm-up phase."""

    def __init__(self, seed: int = 0, interval_s: float = 0.0, steps: int = 120, warmup_steps: int = 3):
        self.interval_s = interval_s
        self.warmup_steps = warmup_steps
        self._ds = sim.simulate(1, steps, seed)
        self._cfg = physics.SensorConfig()

    def __iter__(self) -> Iterator[Reading]:
        ds, cfg = self._ds, self._cfg
        for i in range(len(ds.X)):
            rs_r0, c_comp, ldr_raw, ldr_blk, temp_c, rh = (float(v) for v in ds.X[i])
            rs = rs_r0 * cfg.rl_ohm
            warm = i < self.warmup_steps
            yield Reading(
                t_ms=int(round(float(ds.t_s[i]) * 1000)),
                adc_mq=cfg.adc_max * cfg.rl_ohm / (rs + cfg.rl_ohm),
                rs_r0=None if warm else rs_r0,
                c_comp=None if warm else c_comp,
                ldr_raw=ldr_raw, ldr_blk=ldr_blk, temp_c=temp_c, rh=rh,
                idx_dev=None,
                flags=FLAG_WARMUP if warm else 0,
                label=None if ds.y is None else float(ds.y[i]),
            )
            if self.interval_s:
                time.sleep(self.interval_s)


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as lp
        return [p.device for p in lp.comports()]
    except Exception:  # pyserial missing or enumeration failed
        return []


def _open_without_reset(port: str, baud: int, timeout: float):
    """Open the port with DTR/RTS low: pyserial asserts DTR on open, which auto-resets a Mega."""
    import serial

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, timeout
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


class SerialSource:
    """Reads the Mega over USB serial, reconnecting after a serial error."""

    def __init__(self, port: str, baud: int = 115200, serial_factory: Callable | None = None,
                 reconnect_delay_s: float = 1.0):
        self.port, self.baud = port, baud
        self.reconnect_delay_s = reconnect_delay_s
        self._factory = serial_factory
        self._ser = None
        self._closed = False
        self.connected = False

    def _open(self):
        factory = self._factory or _open_without_reset
        return factory(self.port, self.baud, timeout=1)

    def readings(self) -> Iterator[Reading]:
        while not self._closed:
            try:
                self._ser = self._open()
                self.connected = True
                while not self._closed:
                    raw = self._ser.readline()
                    if not raw:
                        continue
                    reading = parse_line(raw.decode("utf-8", errors="replace"))
                    if reading is not None:
                        yield reading
            except Exception:  # serial.SerialException, OSError, unplugged board, ...
                self.connected = False
                self._shutdown()
                if self._closed:
                    return
                time.sleep(self.reconnect_delay_s)
        self._shutdown()

    def send(self, data: bytes) -> bool:
        """Write bytes to the board (for example b'r' to re-baseline). False when not connected."""
        ser = self._ser
        if ser is None or not self.connected:
            return False
        try:
            ser.write(data)
            ser.flush()
            return True
        except Exception:
            return False

    def _shutdown(self) -> None:
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def close(self) -> None:
        self._closed = True
        self.connected = False
        self._shutdown()
