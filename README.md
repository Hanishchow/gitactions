# Ethylene instrument: Mega 2560 firmware + ML app

Prototype of the ETCSP-2026 paper *Stoichiometry-Constrained Neural Inference of Ethylene from Catalytic
Metal-Oxide Sensing*. An Arduino Mega 2560 reads an MQ gas sensor, a DHT11 and an LDR, applies the paper's
formulas and runs its full 6-256-128-64-32-1 MLP on-device (int8 weights in flash). A Python app adds an
autoencoder (sensor-fault detection), a light-transformer comparison, an operator decision engine and
Qwen-based plain-language analysis.

> **All accuracy numbers below are on synthetic data from a physics-based simulator. They show that the
> models learn the simulated relationship, not field accuracy.** Real validation needs co-located GC/NDIR
> reference data (see the paper's section 7). The "ethylene" output is an uncalibrated index (`c_comp`), not ppm.

## Formulas (`ethylene/physics.py`, mirrored in `firmware/ethylene_mega/ethylene_math.h`)

```
Rs = RL * (1023 - adc) / adc          c = A * (Rs/R0)^B
f_T = sqrt(303.15 / (T + 273.15))     f_RH = 1 / (1 + 0.15 * RH / 100)
c_comp = c * f_T * f_RH               x = [Rs/R0, c_comp, LDR_raw, LDR_blk, T, RH]
```

Defaults `A=116.6020682, B=-2.769034857, RL=10 kOhm` are the common MQ-135 curve and are placeholders:
recalibrate on your bench. `R0` is measured in clean air during warm-up; `LDR_blk` is the LDR baseline
captured at the same time.

## Wiring (Mega 2560, 5 V)

| Part | Pin |
|---|---|
| MQ sensor analog out | A0 |
| LDR divider | A1 |
| DHT11 data (with pull-up) | D22 |
| Vent relay / scrub relay | D30 / D31 |
| Status LED | D13 |
| USB serial | 115200 baud |

Power up in clean air: the firmware warms the heater for 3 minutes, baselines `R0` over the last 30 s, then
prints one CSV line every 2 s (`t_ms,adc_mq,rs_r0,c_comp,ldr_raw,ldr_blk,temp_c,rh,idx,flags`). Send `r`
over serial to re-baseline. Flags: 1 warm-up, 2 DHT failure, 4 MQ fault (ADC at a rail), 8 vent relay, 16 scrub relay.

Flash with `arduino-cli` (compiled size: 23% flash, 27% RAM):

```bash
arduino-cli core install arduino:avr
arduino-cli compile --upload -p COM4 --fqbn arduino:avr:mega:cpu=atmega2560 firmware/ethylene_mega
```

## Run the app

```bash
pip install -r requirements.txt
python -m ethylene.train          # optional: artifacts/ is committed; retrains MLP, AE, transformer (~3 min)
python -m ethylene.export         # optional: regenerates firmware/ethylene_mega/model_weights.h
python app.py                     # http://127.0.0.1:8000  (API docs at /docs)
```

The app starts on a simulated feed. Switch to the board with `POST /api/source {"kind":"serial","port":"COM4"}`.
Upload a logged CSV to `POST /api/analyze-csv`. The front end is a single static page,
`ethylene/static/index.html`, served at `/` (drop your template there). For Qwen, point `OLLAMA_URL` at an
Ollama server that has `qwen3-fast`; when it is unreachable the app returns a built-in summary instead.

## Benchmark (synthetic, held-out runs; `artifacts/metrics.json`)

| Model | R2 | MAE | Params | int8 size | Latency (CPU, batch 1) |
|---|---|---|---|---|---|
| Constrained MLP (primary) | 0.9964 | 0.0119 | 45,057 | 44 KB | 0.09 ms |
| Light transformer (16-step window) | 0.9968 | 0.0102 | 17,857 | 17 KB | 0.43 ms |
| Autoencoder (6-16-5-16-6) | detects 98.1% of injected faults, 0.46% false alarms | | | | |

The transformer is only 14% better in MAE, below the 20% bar set for preferring it, and it needs 16 samples of
history and about 5x the latency, so the MLP is primary. The MLP never decreased when compensated
concentration increased (monotonic violation rate 0.0). int8 quantisation changes outputs by 0.0004 on average.

## Decisions (`ethylene/decide.py`)

Warm-up: hold. MQ fault, autoencoder anomaly or no index: ALERT. Index >= 0.90: VENTILATE. Index >= 0.85: SCRUB.
Otherwise HOLD (below target under 0.55, in target window up to 0.85). A DHT11 failure raises severity only.

## Layout

`ethylene/` physics, data, simulator, models, training, export, serial, decisions, Qwen client, engine, API.
`firmware/ethylene_mega/` sketch. `tests/` pytest (`pytest -q`). `docs/superpowers/` design, plan and handoff notes.

## Known limits

DHT11 is +/-2 degC and +/-5% RH, MQ sensors drift and cross-react, constants need bench calibration, and the
synthetic corpus is in-distribution by construction.
