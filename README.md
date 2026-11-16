# Ethylene instrument: Mega 2560 firmware + ML app

Prototype of the ETCSP-2026 paper *Stoichiometry-Constrained Neural Inference of Ethylene from Catalytic
Metal-Oxide Sensing*. An Arduino Mega 2560 reads an MQ gas sensor, a DHT11 and an LDR, applies the paper's
formulas and runs a compact 6-16-8-1 neural net on-device (257 int8 weights, 5.8 ms per inference). A Python app
adds an autoencoder with a range gate (sensor-fault detection), a light-transformer comparison, an operator decision
engine and Qwen-based plain-language analysis.

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
| Vent relay / scrub relay (disabled by default, see below) | D30 / D31 |
| Status LED | D13 |
| USB serial | 115200 baud |

Power up in clean air: the firmware warms the heater for 3 minutes, baselines `R0` over the last 30 s, then
prints one CSV line every 2 s (`t_ms,adc_mq,rs_r0,c_comp,ldr_raw,ldr_blk,temp_c,rh,idx,flags`). Send `r`
over serial to re-baseline. Flags: 1 warm-up, 2 DHT failure, 4 MQ fault (ADC at a rail), 8 vent relay, 16 scrub relay.

Flash with `arduino-cli` (compiled size: 5% flash, 5% RAM):

```bash
arduino-cli core install arduino:avr
arduino-cli compile --upload -p COM4 --fqbn arduino:avr:mega:cpu=atmega2560 firmware/ethylene_mega
```

## Real-hardware notes

- **Relays are off by default** (`RELAYS_ENABLED 0` in `config.h`). The model was trained on simulated data, so on real
  sensors the index can be arbitrary until you recalibrate or retrain with logged data; flags 8/16 still show what would be
  commanded. Before enabling, bench-test polarity with no load (many blue opto modules are active-low: set
  `RELAY_ACTIVE_HIGH 0`) and add a 10 kOhm resistor from each relay input to its OFF level, because the pins float during
  the ~2 s bootloader.
- **Opening the serial port can reset the Mega** (DTR). The app opens it with DTR/RTS low, but some CH340 drivers still pulse
  it; a 10 uF capacitor from RESET to GND stops auto-reset for good (remove it to upload sketches). A reset restarts the
  3-minute warm-up and takes R0 from whatever air is present, so reconnect only in clean air. Only one program can hold the
  COM port: close the Arduino serial monitor before starting the app.
- **MQ sensors need burn-in.** After a long idle a new MQ element needs 24-48 h powered, and even then R0 taken 3 minutes after
  a cold start is inside the warm-up drift. Let the sensor run 15-30 minutes in clean air, then re-baseline with
  `POST /api/rebaseline` (sends `r`; ignored during the first warm-up).
- **Self-test of the on-chip maths (no sensors needed):** send `T,adc_mq,ldr_raw,temp_c,rh,r0_ohm,ldr_blk` over serial and
  the board computes Rs/R0, compensated concentration and the network output from those numbers (flag 32, nothing else
  touched). `python -m ethylene.hil` sends 43 vectors, including the ADC rails, and compares with the Python reference:
  43/43 agree (worst error: Rs/R0 5e-5, c_comp 3e-4, index 8e-5).
- A `# mlp_us=...` comment line after the header reports one MLP inference time on your board (the app and `load_csv` ignore
  `#` lines; a repeated header in a log marks a board reset and starts a new run).
- The DHT11 needs a 4.7-10 kOhm pull-up to 5 V; with a long cable or weak pull-up reads fail safe (flag 2). After 15 failed
  reads in a row (~30 s) the last good T/RH is dropped and no index is produced.

## Run the app

```bash
pip install -r requirements.txt
python -m ethylene.benchmark      # optional (~8 min): compare candidate nets, baselines and shifted data
python -m ethylene.train          # optional (~2 min): retrain with the benchmark's pick; artifacts/ is committed
python -m ethylene.export         # optional: regenerates firmware/ethylene_mega/model_weights.h
python -m ethylene.hil --port COM4   # with the board flashed and NO sensors: checks the on-chip maths
python app.py                     # http://127.0.0.1:8000  (API docs at /docs)
```

The app starts on a simulated feed. Switch to the board with `POST /api/source {"kind":"serial","port":"COM4"}`.
Upload a logged CSV to `POST /api/analyze-csv`. The front end is a single static page,
`ethylene/static/index.html`, served at `/` (drop your template there). For Qwen, point `OLLAMA_URL` at an
Ollama server that has `qwen3-fast`; when it is unreachable the app returns a built-in summary instead.

## What the numbers mean (read this before trusting any accuracy figure)

**The first version looked "too perfect" because the simulator made it so.** It generated `c_comp` straight from
ripeness, so a straight line on `c_comp` alone scored R2 = 0.996 and the 45k-parameter net added nothing. That
generator is kept as `simulate_idealised` (a test pins the fact) and replaced by a realistic one
(`ethylene/sim.py`): cultivar-to-cultivar emission gain, response shape and clean-air offset; per-sensor-unit gain,
exponent and R0; imperfect T/RH compensation (the sensor's true response differs from the paper's factors);
non-ripening VOC episodes; slow drift; a noisy 10-bit ADC; DHT11 +/-1 degC / +/-3 %RH quantised readings; ambient
light changes on the LDR. **All of these ranges are my assumptions, not measurements.** Real data will differ.

Models are scored on a held-out in-distribution set and on three shifted sets never seen in training (unseen
cultivars, cold-chain 1-10 degC / 80-99 %RH, shifted sensor units). Cells are MAE / R2 on the 0-1 ripeness index
(`python -m ethylene.benchmark`, `artifacts/benchmark.json`):

| Model | Params | Mega time | In-dist | Unseen cultivars | Cold chain | Sensor shift | Mean MAE |
|---|---|---|---|---|---|---|---|
| Paper MLP 6-256-128-64-32-1 | 45,057 | 663 ms (measured) | 0.090 / 0.68 | 0.099 / 0.63 | 0.088 / 0.70 | 0.108 / 0.45 | 0.0960 |
| MLP 64-32 | 2,561 | ~37 ms | 0.093 / 0.66 | 0.100 / 0.62 | 0.091 / 0.68 | 0.109 / 0.44 | 0.0981 |
| MLP 32-16 | 769 | ~11 ms | 0.094 / 0.65 | 0.102 / 0.60 | 0.092 / 0.67 | 0.104 / 0.52 | 0.0979 |
| **MLP 16-8 (shipped)** | 257 | **5.8 ms (measured)** | 0.091 / 0.67 | 0.097 / 0.64 | 0.099 / 0.62 | 0.105 / 0.50 | 0.0980 |
| 5 x MLP 32-16 ensemble | 3,845 | - | 0.089 / 0.68 | 0.098 / 0.63 | 0.088 / 0.69 | 0.104 / 0.51 | 0.0950 |
| Light transformer, 16-step window | 17,857 | - | 0.096 / 0.58 | 0.098 / 0.57 | 0.094 / 0.60 | 0.100 / 0.45 | 0.0969 |
| Linear on `c_comp` only | 2 | - | 0.101 / 0.58 | 0.119 / 0.48 | 0.100 / 0.63 | 0.136 / 0.02 | 0.1139 |
| Ridge on all 6 features | 7 | - | 0.086 / 0.71 | 0.097 / 0.65 | 0.084 / 0.72 | 0.102 / 0.55 | 0.0921 |
| Gradient boosting | - | - | 0.095 / 0.63 | 0.120 / 0.47 | 0.094 / 0.65 | 0.114 / 0.45 | 0.1057 |

What this says, honestly:
- **Size does not help.** Every network from 257 to 45,057 parameters lands within 2% of each other, so the shipped net
  is the smallest one within 5% of the best: 114x faster on the Mega than the paper's net, 0.5 KB instead of 45 KB.
- **On this simulated data a plain ridge regression is as good as or better than every network.** The error is
  dominated by ambiguity the sensor cannot resolve (different cultivars give different `c_comp` for the same ripeness,
  interferent VOCs), not by model capacity. A network only earns its place if real data shows nonlinearity; collect
  logged data with the self-test-verified board and rerun `python -m ethylene.benchmark`.
- Ensemble spread does not predict error (correlation about 0.02), so there is no free uncertainty estimate.
- The physics prior (monotonic in compensated concentration) and noise augmentation change results by less than 1%.
  The shipped net never decreases when compensated concentration increases (violation rate 0.4%).
- Light transformer is worse than the MLP here, so the MLP is primary.
- **Fault gate:** a row is flagged when the autoencoder's reconstruction error exceeds its 99th-percentile threshold
  *or* any feature is more than 4 standard deviations outside the training data (an input the models never saw, such as
  RH = 0 %, which the autoencoder alone passes). Detects 94.9% of injected faults (sensor drift 100%, dead LDR 100%,
  DHT zeros 100%, frozen `c_comp` 79.5%) with 1.3% false alarms. `ethylene/coverage.py` reports how far any dataset
  sits outside the training range.

## Decisions (`ethylene/decide.py`)

Warm-up: hold. MQ fault, autoencoder anomaly or no index: ALERT. Index >= 0.90: VENTILATE. Index >= 0.85: SCRUB.
Otherwise HOLD (below target under 0.55, in target window up to 0.85). A DHT11 failure raises severity only.

## Layout

`ethylene/` physics, data, simulator, models, training, export, serial, decisions, Qwen client, engine, API.
`firmware/ethylene_mega/` sketch. `tests/` pytest (`pytest -q`). `docs/superpowers/` design, plan and handoff notes.

## Known limits

DHT11 is +/-2 degC and +/-5% RH, MQ sensors drift and cross-react, constants need bench calibration, and the
synthetic corpus is in-distribution by construction.
