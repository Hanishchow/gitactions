# Ethylene Instrument: Uno Firmware + ML App — Design

Status: approved by the user on 2026-11-16 (approach and design as proposed in chat).

## Intent
Turn the ETCSP-2026 paper ("Stoichiometry-Constrained Neural Inference of Ethylene from Catalytic
Metal-Oxide Sensing") into a working prototype: Arduino Uno firmware for the three sensors
(MQ SnO2, DHT11, LDR) with the paper's formulas, plus a desktop app that adds an autoencoder,
a neural-net vs light-transformer comparison, automatic decisions, and Qwen-based analysis.
The author is the paper's lead (TRL 3 -> field-ready prototype).

## Goals
- Uno firmware: warm-up, R0 baseline, Rs, c = A(Rs/R0)^B, f_T, f_RH, c_comp, tiny on-device MLP, CSV over serial.
- Formulas exactly as the paper: f_T = sqrt(303.15/(T+273.15)), f_RH = 1/(1+0.15*RH/100), c_comp = c*f_T*f_RH;
  model input x = [Rs/R0, c_comp, LDR_raw, LDR_blk, T, RH].
- Physics-based simulator producing synthetic data; CSV import for real Uno logs.
- Models: stoichiometry-constrained MLP (paper loss: MSE + lambda_phys*R_stoich + L2), autoencoder
  (fault/anomaly + denoise), small transformer over time windows. Benchmark picks the winner.
- Decision engine: ventilate / scrub / hold / alert.
- Qwen3 (`qwen3-fast` via Ollama on omen over Tailscale) for plain-language analysis, with graceful fallback.
- FastAPI app + single-page dashboard; pytest suite run by existing CI.

## Non-goals
- No claim of certified field accuracy: all metrics are on synthetic data and labelled so.
- No full 6-256-128-64-32-1 MLP on the Uno (about 44.6k params; Uno has 2 KB RAM / 32 KB flash / no FPU).
- No actuator hardware drivers beyond emitting a decision signal.

## Anti-goals
- Presenting synthetic-data R^2 as real-world performance.
- Silent failure when omen/Qwen is unreachable.

## Constraints
- Python 3.12, torch 2.14 CPU only, numpy, scikit-learn, fastapi, uvicorn installed; no pandas/streamlit.
- No `arduino-cli` / `avr-gcc` on this machine: firmware math is verified through a Python mirror on shared
  test vectors; a real compile happens in the user's Arduino IDE (or arduino-cli if the user agrees to install it).
- Paper does not name the MQ model/load resistor: default MQ-135 constants in one config (A, B, RL), flagged as my call.

## Approaches considered
1. **Uno as full inference node** — rejected: the paper's network cannot fit.
2. **Uno as dumb sensor head, all ML on PC** — works but loses the paper's "on-device" story.
3. **Uno does sensing + compensation + tiny distilled MLP and streams; PC does AE/MLP/transformer/Qwen** — chosen.

## Architecture
```
firmware/ethylene_uno/   sketch + config.h (A, B, RL, R0 logic, tiny MLP weights header)
ethylene/
  physics.py     formulas (single source of truth, mirrored in firmware)
  sim.py         ripening simulator -> synthetic dataset
  data.py        CSV load/save, windowing, splits
  models.py      ConstrainedMLP, Autoencoder, TinyTransformer
  train.py       training + benchmark (accuracy, params, latency)
  decide.py      decision engine
  llm.py         Ollama client (omen), fallback
  serial_io.py   Uno serial reader (+ simulated feed)
  api.py         FastAPI app + static dashboard
tests/           pytest
```

## Left to the builder
Dataset size, hyperparameters, thresholds for decisions, dashboard layout, tiny-MLP distillation details.
