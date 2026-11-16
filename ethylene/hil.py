"""Hardware-in-the-loop check of the on-chip maths, with no sensors wired.

``python -m ethylene.hil --port COM4`` sends known raw readings to the Mega's self-test command
(``T,adc_mq,ldr_raw,temp_c,rh,r0_ohm,ldr_blk``), reads back the values the chip computed (Rs/R0, compensated
concentration, MLP index) and compares them with the Python reference: the paper's formulas in double precision
and the int8 network read back from the generated ``model_weights.h``. Exit code 0 only if every vector agrees.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from ethylene import physics, serial_io, sim
from ethylene.data import FLAG_SELFTEST
from ethylene.export import HEADER_PATH, QuantizedMLP, quantized_forward, read_header

R0_OHM = 20_000.0
# Chip output is printed with 4 decimals and computed in float32, the reference in float64.
TOL_RS_R0_ABS = 2e-4
TOL_C_COMP_ABS = 1e-3  # the 4-decimal print floor
TOL_C_COMP_REL = 1e-3
TOL_IDX_ABS = 2e-3


def make_vectors(n: int = 40, seed: int = 7, cfg: physics.SensorConfig = physics.SensorConfig()) -> list[dict]:
    """`n` realistic readings (as raw ADC counts) plus the edge cases: ADC at both rails and at 1."""
    ds = sim.simulate(max(2, n // 20), 60, seed)
    picks = np.linspace(0, len(ds.X) - 1, n).astype(int)
    vectors = []
    for i in picks:
        rs_r0, _, ldr_raw, ldr_blk, temp_c, rh = (float(v) for v in ds.X[i])
        rs = rs_r0 * R0_OHM
        adc = int(round(cfg.adc_max * cfg.rl_ohm / (rs + cfg.rl_ohm)))
        vectors.append({"adc_mq": adc, "ldr_raw": ldr_raw, "temp_c": temp_c, "rh": rh, "r0_ohm": R0_OHM, "ldr_blk": ldr_blk})
    base = vectors[0]
    for adc in (0, cfg.adc_max, 1):
        vectors.append({**base, "adc_mq": adc})
    return vectors


def vector_line(v: dict) -> str:
    return "T,{adc_mq},{ldr_raw:g},{temp_c:g},{rh:g},{r0_ohm:g},{ldr_blk:g}".format(**v)


def expected(v: dict, q: QuantizedMLP, cfg: physics.SensorConfig = physics.SensorConfig()) -> dict:
    """Reference values for one vector; None where the chip must report nan / no index."""
    rs = physics.adc_to_rs(float(v["adc_mq"]), cfg)
    rs_r0 = rs / v["r0_ohm"]
    c_comp = physics.compensate(physics.concentration(rs_r0, cfg), v["temp_c"], v["rh"])
    ok = math.isfinite(rs_r0) and math.isfinite(c_comp)
    idx = quantized_forward(q, np.array([rs_r0, c_comp, v["ldr_raw"], v["ldr_blk"], v["temp_c"], v["rh"]])) if ok else None
    return {"rs_r0": rs_r0 if math.isfinite(rs_r0) else None, "c_comp": c_comp if math.isfinite(c_comp) else None, "idx": idx}


def compare(reading: serial_io.Reading, exp: dict) -> list[str]:
    """Human-readable mismatches between a chip reading and the reference (empty = agree)."""
    problems = []
    if not reading.flags & FLAG_SELFTEST:
        problems.append("reply is not a self-test line")
    for name, got, want, tol_abs, tol_rel in (("rs_r0", reading.rs_r0, exp["rs_r0"], TOL_RS_R0_ABS, 0.0),
                                              ("c_comp", reading.c_comp, exp["c_comp"], TOL_C_COMP_ABS, TOL_C_COMP_REL),
                                              ("idx", reading.idx_dev, exp["idx"], TOL_IDX_ABS, 0.0)):
        if (got is None) != (want is None):
            problems.append(f"{name}: chip {got} vs reference {want}")
        elif got is not None:
            allowed = tol_abs + tol_rel * abs(want)
            if abs(got - want) > allowed:
                problems.append(f"{name}: chip {got:.5f} vs reference {want:.5f} (error {abs(got - want):.2e} > {allowed:.1e})")
    return problems


def run(port: str, n: int = 40, baud: int = 115200, header=HEADER_PATH, settle_s: float = 3.0) -> int:
    q = read_header(header)
    vectors = make_vectors(n)
    ser = serial_io._open_without_reset(port, baud, 1)
    try:
        time.sleep(settle_s)  # a CH340 may still reset the board on open
        ser.reset_input_buffer()
        failures = 0
        worst = {"rs_r0": 0.0, "c_comp": 0.0, "idx": 0.0}
        for v in vectors:
            ser.write((vector_line(v) + "\n").encode())
            ser.flush()
            reply, deadline = None, time.time() + 5.0
            while time.time() < deadline and reply is None:
                r = serial_io.parse_line(ser.readline().decode("utf-8", "replace"))
                if r is not None and r.flags & FLAG_SELFTEST:
                    reply = r
            exp = expected(v, q)
            problems = ["no reply from the board"] if reply is None else compare(reply, exp)
            if reply is not None:
                for name, got in (("rs_r0", reply.rs_r0), ("c_comp", reply.c_comp), ("idx", reply.idx_dev)):
                    if got is not None and exp[name] is not None:
                        worst[name] = max(worst[name], abs(got - exp[name]))
            if problems:
                failures += 1
                print(f"FAIL {vector_line(v)}\n     " + "\n     ".join(problems))
        total = len(vectors)
        print(f"{total - failures}/{total} vectors agree with the Python reference; "
              f"worst abs error: rs_r0 {worst['rs_r0']:.2e}, c_comp {worst['c_comp']:.2e}, idx {worst['idx']:.2e}")
        return 0 if failures == 0 else 1
    finally:
        ser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    sys.exit(run(args.port, args.n, args.baud))
