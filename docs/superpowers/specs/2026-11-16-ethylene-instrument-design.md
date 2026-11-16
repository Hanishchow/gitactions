# Ethylene Instrument: Mega 2560 Firmware + ML App — Design

Status: approved by the user on 2026-11-16. Revised the same day after the user said the board is an
Arduino **Mega 2560** (not an Uno): the paper's full MLP now fits on-device, so the "tiny distilled MLP"
from the first proposal is dropped.

## Intent
Turn the ETCSP-2026 paper ("Stoichiometry-Constrained Neural Inference of Ethylene from Catalytic
Metal-Oxide Sensing") into a working prototype: Arduino Mega 2560 firmware for the three sensors
(MQ SnO2, DHT11, LDR) with the paper's formulas and its full MLP, plus a desktop app that adds an
autoencoder, a neural-net vs light-transformer comparison, automatic decisions, and Qwen-based analysis.
The author is the paper's lead (TRL 3 -> field-ready prototype).

## Goals
- Mega firmware: warm-up, R0 baseline, Rs, c = A(Rs/R0)^B, f_T, f_RH, c_comp, the paper's
  6-256-128-64-32-1 MLP (int8 weights in PROGMEM, about 46 KB flash), relay decision pins, CSV over serial.
- Formulas exactly as the paper: f_T = sqrt(303.15/(T+273.15)), f_RH = 1/(1+0.15*RH/100),
  c_comp = c*f_T*f_RH; model input x = [Rs/R0, c_comp, LDR_raw, LDR_blk, T, RH].
- Physics-based simulator producing synthetic data; CSV import for real Mega logs (same CSV format).
- Models: stoichiometry-constrained MLP (paper loss: MSE + lambda_phys*R_stoich + L2), autoencoder
  (sensor-fault/anomaly detection), small transformer over time windows. A benchmark decides which is primary.
- Decision engine: ventilate / scrub / hold / alert.
- Qwen3 (`qwen3-fast` via Ollama on omen over Tailscale) for plain-language analysis, with graceful fallback.
- FastAPI app + single-page dashboard; pytest suite run by existing CI.

## Non-goals
- No claim of certified field accuracy: all metrics are on synthetic data and labelled so.
- No quantisation-aware training (post-training int8 only).
- No ethylene ppm conversion: the paper's ethylene output is an uncalibrated index (c_comp); the model outputs
  the 0-1 ripeness index.
- No actuator drivers beyond switching two relay pins.

## Anti-goals
- Presenting synthetic-data R^2 as real-world performance.
- Silent failure when omen/Qwen is unreachable, a sensor is unplugged, or serial data is garbage.

## Constraints
- Board: ATmega2560, 16 MHz, 8 KB SRAM, 256 KB flash, 10-bit ADC, 16 analog inputs, no FPU, 5 V logic.
- Python 3.12, torch 2.14 CPU only, numpy, scikit-learn, fastapi, uvicorn, httpx installed; pyserial to be added;
  no pandas/streamlit.
- No `arduino-cli` / `avr-gcc` / `g++` on this machine: firmware math and quantised forward pass are verified by a
  Python mirror on shared test vectors; a real compile happens in the user's Arduino IDE, or via arduino-cli if the
  user agrees to install it.
- Paper does not name the MQ model/load resistor: defaults A=116.6020682, B=-2.769034857 (common MQ-135 curve),
  RL=10 kOhm in one config; my call, to be recalibrated on the bench.
- `LDR_blk` is my interpretation: LDR baseline captured at the same time as R0 (no second LDR).

## Pins (my call)
MQ analog out A0; LDR divider A1; DHT11 data D22; vent relay D30; scrub relay D31; status LED D13;
USB Serial0 at 115200 baud for CSV; Serial1 reserved for a later Bluetooth/ESP link.

## Approaches considered
1. **Dumb sensor head, all ML on PC** — works but loses the paper's on-device story.
2. **Mega does sensing + compensation + full paper MLP (int8) + relays and streams; PC does AE, transformer, Qwen** — chosen.
   Cost: about 45k MACs per inference, roughly 0.4 s on AVR with float activations; sample period 2 s.

## Architecture
```
firmware/ethylene_mega/   ethylene_mega.ino, config.h, ethylene_math.h, dht11.h, mlp.h, model_weights.h (generated)
ethylene/
  physics.py     formulas (single source of truth, mirrored in firmware)
  data.py        Dataset, CSV load/save, windowing, Scaler, run-level splits
  sim.py         ripening simulator + fault injection
  models.py      ConstrainedMLP, Autoencoder, TinyTransformer
  train.py       training + benchmark (accuracy, params, latency) -> artifacts/
  export.py      int8 quantisation, Python mirror of firmware forward pass, writes model_weights.h
  decide.py      decision engine
  llm.py         Ollama client (omen), fallback
  serial_io.py   Mega serial reader, line parser, simulated feed
  api.py         FastAPI app + static dashboard
tests/           pytest
```

## Left to the builder
Dataset size, hyperparameters, decision thresholds, dashboard layout, DHT11 bit-bang details.
