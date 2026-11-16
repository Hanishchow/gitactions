# Ethylene Instrument Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mega 2560 firmware for MQ + DHT11 + LDR running the paper's full int8 MLP, plus a FastAPI/dashboard app with autoencoder, MLP-vs-transformer benchmark, decision engine and Qwen analysis.

**Architecture:** `ethylene/physics.py` is the single source of formulas; the simulator makes synthetic runs; PyTorch trains MLP/AE/transformer into `artifacts/`; `export.py` quantises the MLP to `firmware/ethylene_mega/model_weights.h`; the app ingests the Mega's CSV stream (or a simulated feed) and serves a dashboard.

**Tech Stack:** Python 3.12, torch 2.14 (CPU), numpy, scikit-learn, fastapi, uvicorn, httpx, pyserial, pytest; Arduino C++ (AVR).

**Spec:** `docs/superpowers/specs/2026-11-16-ethylene-instrument-design.md`

## Global Constraints
- Formulas: `c = A(Rs/R0)^B`; `f_T = sqrt(303.15/(T+273.15))`; `f_RH = 1/(1+0.15*RH/100)`; `c_comp = c*f_T*f_RH`.
- Model input order: `[Rs/R0, c_comp, LDR_raw, LDR_blk, T, RH]`; MLP 6-256-128-64-32-1, ReLU hidden, linear output (45,057 params).
- Loss: `MSE + lambda_phys*R_stoich + lambda2*||W||^2`, `R_stoich = mean(max(0, -dy/dc_comp)^2)`, lambda2 = 1e-4, Adam, early stopping.
- Defaults: A=116.6020682, B=-2.769034857, RL=10000 ohm, ADC max 1023, MQ-135 curve (placeholder to recalibrate).
- Board pins: MQ A0, LDR A1, DHT11 D22, vent relay D30, scrub relay D31, LED D13; Serial0 115200 baud.
- Python 3.12; torch CPU only; no pandas/streamlit; Ollama at `http://OMEN_TAILNET_IP:11434`, model `qwen3-fast`.
- Every metric shown to a user is labelled synthetic-data / not field accuracy. Ethylene output is an uncalibrated index (c_comp), no ppm.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus
1. MQ ADC reads 0 or 1023 (unplugged/shorted): Rs is NaN, flag set, relays off, decision ALERT, never a crash or inf. (Tasks 1, 7, 9)
2. DHT11 checksum/timeout/out-of-range read: last good T/RH reused, `FLAG_DHT_FAIL` set; no good value yet means no index. (Tasks 1, 7, 9)
3. Warm-up window: no index, no decision, no R0 from unstable readings, warm-up rows excluded from training. (Tasks 5, 7, 9, 11)
4. Serial garbage (partial lines, binary noise, wrong column count, `nan`): ignored, app keeps running. (Task 8)
5. omen/Qwen unreachable, slow, or returns `<think>` text: fallback summary within the timeout, UI not blocked. (Tasks 10, 11)

---

### Task 1: Physics formulas and project scaffold

**Files:**
- Create: `ethylene/__init__.py`, `ethylene/physics.py`, `tests/test_physics.py`, `pytest.ini` (`testpaths = tests`, marker `live`), `.gitignore` (`__pycache__/`, `.pytest_cache/`)
- Modify: `requirements.txt` (numpy, torch, scikit-learn, fastapi, uvicorn, httpx, pyserial, python-multipart, pytest)

**Interfaces:**
- Produces: `SensorConfig(A, B, rl_ohm, adc_max)` frozen dataclass with the defaults above; `FEATURE_NAMES = ("rs_r0","c_comp","ldr_raw","ldr_blk","temp_c","rh")`; `adc_to_rs(adc: float, cfg=SensorConfig()) -> float` (`rl*(adc_max-adc)/adc`, NaN when adc<=0 or adc>=adc_max); `concentration(rs_r0: float, cfg) -> float`; `f_temp(temp_c) -> float` (NaN when T<=-273.15); `f_rh(rh) -> float`; `compensate(c, temp_c, rh) -> float`; `dht_in_range(temp_c, rh) -> bool` (0<=T<=50, 0<=RH<=100); `features(rs_r0, c_comp, ldr_raw, ldr_blk, temp_c, rh) -> np.ndarray` shape (6,) float32.

- [ ] **Step 1: Write tests in `tests/test_physics.py`:** `test_f_temp_reference_point` (`f_temp(30.0)==approx(1.0)`, `f_temp(0.0)==approx(sqrt(303.15/273.15))`); `test_f_rh` (`f_rh(0)==1.0`, `f_rh(100)==approx(1/1.15)`); `test_compensate` (`compensate(10,30,0)==approx(10)`, `compensate(10,30,100)==approx(10/1.15)`); `test_concentration_power_law` (ratio 1.0 gives A; ratio 0.5 gives `A*0.5**B` and is larger than at 1.0); `test_adc_to_rs` (`adc_to_rs(511.5)==approx(10000)`, decreasing in adc); `test_adc_rails_are_nan` (adc 0 and 1023 give NaN); `test_f_temp_invalid_is_nan` (`f_temp(-300)` NaN); `test_dht_in_range` (True at (25,60), False at (85,0), (-5,50), (25,120)); `test_features_order_and_dtype`.
- [ ] **Step 2: Run `pytest tests/test_physics.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement `ethylene/physics.py`** per the Interfaces block; pure Python `math`, no numpy except `features`.
- [ ] **Step 4: Run `pytest tests/test_physics.py -v`.** Expected: all PASS.
- [ ] **Step 5: Commit** `feat: physics formulas and scaffold`.

### Task 2: Data layer (dataset, scaler, CSV, windows, splits)

**Files:** Create `ethylene/data.py`, `tests/test_data.py`.

**Interfaces:**
- Consumes: `FEATURE_NAMES`.
- Produces: `CSV_COLUMNS = ("t_ms","adc_mq","rs_r0","c_comp","ldr_raw","ldr_blk","temp_c","rh","idx","flags")` (optional trailing `label`); `FLAG_WARMUP=1, FLAG_DHT_FAIL=2, FLAG_MQ_FAULT=4, FLAG_VENT=8, FLAG_SCRUB=16`; `Dataset(X: (N,6) float32, y: (N,) float32|None, run: (N,) int32, t_s: (N,) float32, flags: (N,) int32)` dataclass; `Scaler` with `fit(X) -> Scaler`, `transform(X)`, `to_dict()`, `from_dict(d)` classmethod, attributes `mean`, `std` (std floored at 1e-6); `save_csv(ds, path)`; `load_csv(path) -> Dataset` (raises `ValueError` naming the missing column or the offending line number; `idx==-1` and `nan` accepted); `make_windows(ds, width) -> (W: (M,width,6), y: (M,), run: (M,))` never crossing runs, y taken at the last step; `split_by_run(ds, val_frac, test_frac, seed) -> (train, val, test)` disjoint by run id.

- [ ] **Step 1: Write tests:** `test_windows_never_cross_runs` (two runs of length 5, width 3 gives 6 windows, none mixing run ids); `test_split_by_run_is_disjoint`; `test_scaler_roundtrip` (`from_dict(to_dict())` transforms identically; transformed train has mean ~0, std ~1); `test_csv_roundtrip`; `test_load_csv_missing_column_names_it`; `test_load_csv_non_numeric_reports_line` (error message contains the line number); `test_load_csv_accepts_idx_minus1_and_nan`.
- [ ] **Step 2: Run `pytest tests/test_data.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `ethylene/data.py`** (csv module, numpy; `load_csv` drops `adc_mq` and `idx`, they are not in `Dataset`; `save_csv` writes `adc_mq` by inverting `adc_to_rs` with R0 = 1 reference and `idx = -1`).
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: dataset, scaler, csv io`.

### Task 3: Ripening simulator and fault injection

**Files:** Create `ethylene/sim.py`, `tests/test_sim.py`.

**Interfaces:**
- Consumes: `physics.*`, `Dataset`.
- Produces: `STAGE_MEANS = (0.05,0.22,0.42,0.62,0.82,0.93)`; `simulate(n_runs: int, steps: int, seed: int, cfg=SensorConfig(), g: float = 3.0) -> Dataset`; `FAULT_KINDS = ("rs_drift","ldr_dead","dht_zero","c_comp_stuck")`; `inject_faults(X: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray` (copy, same shape/dtype).

- [ ] **Step 1: Write tests:** `test_simulate_deterministic` (same seed, identical X); `test_shapes_and_ranges` (X (n_runs*steps,6), y in [0,1]); `test_dht_values_are_integers` (temp_c and rh integral); `test_c_comp_tracks_ripeness` (`corrcoef(X[:,1], y) > 0.95`); `test_ripeness_rises_within_run` (per-run correlation of y with t_s > 0.9); `test_each_fault_changes_input` (every kind changes at least one column on a 100-row sample, shape/dtype preserved); `test_unknown_fault_raises_value_error`.
- [ ] **Step 2: Run `pytest tests/test_sim.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `simulate`.** Per run: ripeness path rising from 0.05 to 0.93 across `steps` (run-specific speed, jitter sd 0.01, clipped [0,1]); T baseline U(12,28) and RH U(60,92) with slow sinusoidal drift (±3 degC, ±5 %RH); DHT11 reading = rounded true value plus noise (T sd 0.7, RH sd 2); `c_true = A*(1+g*r)`; `c_raw = c_true/(f_T(T_true)*f_RH(RH_true))*(1+N(0,0.02))`; `rs_r0 = (c_raw/A)**(1/B)`; `c_comp = c_raw*f_T(T_meas)*f_RH(RH_meas)`; `ldr_blk = L0 ~ U(500,700)` per run; `ldr_raw = L0*(1-0.35*r)+N(0,8)`; `y = r`; `t_s` spaced 2 s. Faults: `rs_drift` multiplies rs_r0 by U(1.5,2.5) without touching c_comp; `ldr_dead` sets ldr_raw to 0; `dht_zero` sets temp_c and rh to 0; `c_comp_stuck` sets c_comp to its column median.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: ripening simulator and fault injection`.

### Task 4: Models and the stoichiometric penalty

**Files:** Create `ethylene/models.py`, `tests/test_models.py`.

**Interfaces:**
- Produces: `ConstrainedMLP()` (`forward(x: (B,6)) -> (B,)`, 45,057 params); `stoich_penalty(f: Callable[[Tensor], Tensor], x: Tensor, c_idx: int = 1) -> Tensor` (mean of `relu(-d f/d x[:,c_idx])**2` via autograd, differentiable); `composite_loss(model, x, y, lam_phys: float = 1.0) -> Tensor` (MSE plus lam_phys times penalty; L2 comes from optimizer weight_decay=1e-4); `Autoencoder()` (6-8-3-8-6, `forward(x)->recon`, `score(x)->(B,)` per-sample MSE); `TinyTransformer(width: int = 16)` (`forward(w: (B,width,6)) -> (B,)`; d_model 32, 2 heads, 2 layers, ff 64, learned positions, mean pool).

- [ ] **Step 1: Write tests:** `test_mlp_param_count` (== 45057); `test_mlp_output_shape`; `test_penalty_zero_for_increasing` (`f = lambda x: x[:,1]` gives 0.0); `test_penalty_one_for_decreasing` (`f = lambda x: -x[:,1]` gives approx 1.0); `test_penalty_is_differentiable` (backward through composite_loss populates grads); `test_autoencoder_score_shape_and_learns` (loss lower after 50 steps on random data); `test_transformer_shape_and_size` (output (B,), params < 50k).
- [ ] **Step 2: Run `pytest tests/test_models.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `ethylene/models.py`.** Penalty uses `torch.autograd.grad(f(x).sum(), x, create_graph=True)` with `x.requires_grad_(True)`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: constrained mlp, autoencoder, tiny transformer`.

### Task 5: Training, benchmark and artifacts

**Files:** Create `ethylene/train.py`, `tests/test_train.py`; generate `artifacts/` (committed, under 1 MB).

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces: `main(out_dir: str = "artifacts", fast: bool = False) -> dict` writing `mlp.pt`, `ae.pt`, `transformer.pt`, `scaler.json`, `metrics.json` and returning the metrics. `metrics.json`: `{"data": {"kind": "synthetic", "n_train", "n_val", "n_test"}, "mlp": {"r2","mae","params","fp32_kb","int8_kb","latency_ms","monotonic_violation_rate"}, "transformer": {same keys minus violation}, "autoencoder": {"threshold","fault_detection_rate","false_positive_rate"}, "primary": "mlp"|"transformer"}`; `choose_primary(metrics: dict) -> str` (transformer only when its MAE <= 0.8 x the MLP's MAE, else mlp); `load_artifacts(dir) -> Artifacts` (dataclass with `mlp, ae, transformer, scaler, threshold, metrics`). Training uses `split_by_run` (no run leakage), drops `FLAG_WARMUP` rows, fits the Scaler on train only, trains the AE on fault-free train rows, sets the AE threshold at the 99th percentile of clean validation scores, measures latency as median batch-1 CPU time over 200 calls, and `monotonic_violation_rate` as the fraction of test rows where a +1% step in standardised c_comp lowers the output.

- [ ] **Step 1: Write tests (fast config):** `test_choose_primary_rule` (transformer mae 0.7x mlp gives "transformer"; 0.9x gives "mlp"); `test_main_fast_writes_artifacts_and_metrics` (all five files exist, `data.kind == "synthetic"`); `test_fast_mlp_quality` (r2 >= 0.9, violation rate <= 0.05); `test_autoencoder_detects_faults` (detection >= 0.8, false positive <= 0.05); `test_warmup_rows_excluded` (a dataset with warm-up flags set on some rows trains on fewer rows than total).
- [ ] **Step 2: Run `pytest tests/test_train.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `ethylene/train.py`.** Defaults: 300 runs x 120 steps, MLP up to 150 epochs (Adam lr 1e-3, batch 256, patience 15, lam_phys 1.0), transformer window 16; `fast=True` uses 40 runs x 60 steps and about 15 epochs. Seeds fixed.
- [ ] **Step 4: Run tests, then the real run:** `python -m ethylene.train` (CPU, a few minutes). Expected: tests PASS; `artifacts/metrics.json` written; record the printed R2/MAE/latency table.
- [ ] **Step 5: Commit** `feat: training, benchmark, artifacts`.

### Task 6: int8 export and firmware-faithful forward pass

**Files:** Create `ethylene/export.py`, `tests/test_export.py`, `firmware/ethylene_mega/model_weights.h` (generated).

**Interfaces:**
- Consumes: `ConstrainedMLP`, `Scaler`.
- Produces: `QuantizedMLP` dataclass (`W: list[np.ndarray int8]`, `S: list[np.ndarray float32]` per-output-row scales, `B: list[np.ndarray float32]`, `mean`, `std`); `quantize_mlp(model, scaler) -> QuantizedMLP` (symmetric per-row int8, `S = max|row|/127`); `quantized_forward(q, x_raw: np.ndarray) -> float` (standardise, then per layer `a_out[j] = (sum_i W[j][i]*a[i]) * S[j] + B[j]`, ReLU on hidden layers, float32 arithmetic, exactly what the sketch does); `write_header(q, path)` (PROGMEM arrays `W1..W5` int8_t, `S1..S5` and `B1..B5` float, `SCALER_MEAN[6]`, `SCALER_STD[6]`); `read_header(path) -> QuantizedMLP`; `flash_bytes(q) -> int`.

- [ ] **Step 1: Write tests:** `test_quantized_close_to_float` (on the held-out synthetic test set, mean abs diff < 0.01 and max < 0.03 against the float model); `test_header_roundtrip` (`quantized_forward(read_header(write_header(q)))` equals the in-memory q output exactly); `test_flash_budget` (`flash_bytes(q) <= 60_000`); `test_header_contains_progmem_arrays` (text contains `PROGMEM` and `const int8_t W1[`).
- [ ] **Step 2: Run `pytest tests/test_export.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `ethylene/export.py`;** add a `main()` that loads `artifacts/` and writes `firmware/ethylene_mega/model_weights.h`; run `python -m ethylene.export`.
- [ ] **Step 4: Run tests.** Expected: PASS. If the 0.03 bound fails, report the measured error and stop for a decision rather than loosening it.
- [ ] **Step 5: Commit** `feat: int8 export and generated weights header`.

### Task 7: Mega 2560 firmware

**Files:** Create `firmware/ethylene_mega/{ethylene_mega.ino, config.h, ethylene_math.h, dht11.h, mlp.h}`, `tests/test_firmware_parity.py`.

**Interfaces:**
- Consumes: `model_weights.h`, `CSV_COLUMNS`, flag values, `SensorConfig`.
- Produces: serial CSV at 115200 baud, one line per `SAMPLE_MS` (2000), columns exactly `CSV_COLUMNS` (header printed once at start; `nan` for invalid floats, `idx = -1` when no valid index). `config.h` holds pins, `SENSOR_A`, `SENSOR_B`, `RL_OHM`, `ADC_MAX`, `WARMUP_MS 180000`, `BASELINE_MS 30000`, `SAMPLE_MS 2000`, `VENT_IDX 0.90`, `SCRUB_IDX 0.85`, `RELAY_HYST 0.03`, and the five `FLAG_*` values. `ethylene_math.h` mirrors the physics functions (NaN on ADC rails and T<=-273.15). `mlp.h`: `float mlp_forward(const float x[6])` reading PROGMEM (`pgm_read_byte`/`pgm_read_float`) with two 256-float ping-pong buffers. `dht11.h`: `bool dht11_read(uint8_t pin, int8_t *temp_c, uint8_t *rh)` bit-banged with timeouts and checksum, no external library.
- Behaviour: during warm-up print flag `FLAG_WARMUP`, no index; R0 = mean Rs and LDR_blk = mean LDR over the last `BASELINE_MS` of warm-up (power up in clean air); on DHT failure reuse last good values and set `FLAG_DHT_FAIL` (no good value yet: no index); on ADC 0 or 1023 set `FLAG_MQ_FAULT`, print `nan`, `idx=-1`, both relays off; relays use hysteresis; serial byte `r` re-baselines R0.

- [ ] **Step 1: Write `tests/test_firmware_parity.py`:** `test_config_constants_match_python` (regex-parse `config.h`: A, B, RL, ADC_MAX equal `SensorConfig()`; FLAG values equal `data.FLAG_*`; the `.ino` header string equals `",".join(CSV_COLUMNS)`); `test_math_header_uses_paper_constants` (contains `303.15`, `273.15`, `0.15`); `test_weights_header_matches_artifacts` (`read_header` forward on three fixed inputs equals `quantized_forward` from a fresh `quantize_mlp` of `artifacts/`); `test_compile_if_toolchain` (skipped unless `shutil.which("arduino-cli")`; runs `arduino-cli compile --fqbn arduino:avr:mega firmware/ethylene_mega` and asserts exit 0 and flash use under 90%).
- [ ] **Step 2: Run `pytest tests/test_firmware_parity.py -v`.** Expected: FAIL.
- [ ] **Step 3: Write the firmware files** per the Interfaces and Behaviour above (no `String`, no dynamic allocation; sketch uses a `WARMUP/RUN` state machine).
- [ ] **Step 4: Run `pytest tests/test_firmware_parity.py -v`.** Expected: PASS (compile test skipped when no toolchain; say so in the report).
- [ ] **Step 5: Commit** `feat: mega 2560 firmware`.

### Task 8: Serial input and simulated feed

**Files:** Create `ethylene/serial_io.py`, `tests/test_serial_io.py`.

**Interfaces:**
- Produces: `Reading` frozen dataclass (`t_ms: int`, `adc_mq, rs_r0, c_comp, idx_dev: float | None`, `ldr_raw, ldr_blk, temp_c, rh: float`, `flags: int`, `label: float | None`); `parse_line(line: str) -> Reading | None` (None for header, blank, wrong column count, non-numeric, binary noise, lines over 512 chars; `nan` or `idx == -1` become None; never raises); `SimulatedSource(seed: int = 0, interval_s: float = 0.0)` iterator yielding `Reading`; `SerialSource(port: str, baud: int = 115200, serial_factory=None)` with `readings() -> Iterator[Reading]` that reconnects after `SerialException` and `close()`; `list_ports() -> list[str]`.

- [ ] **Step 1: Write tests:** `test_parse_valid_line`; `test_header_and_blank_return_none`; `test_garbage_returns_none` (random text, binary bytes decoded with errors replaced, a 10 KB line); `test_wrong_column_count_returns_none`; `test_nan_fields_become_none`; `test_simulated_source_yields_valid_readings` (first 50 are finite); `test_serial_source_reconnects` (fake factory raises once, then yields two valid lines and one garbage line: two `Reading`s come out, no exception).
- [ ] **Step 2: Run `pytest tests/test_serial_io.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `ethylene/serial_io.py`** (`serial_factory` defaults to `serial.Serial`; simulated source built on `sim.simulate`).
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: serial parser and simulated source`.

### Task 9: Decision engine

**Files:** Create `ethylene/decide.py`, `tests/test_decide.py`.

**Interfaces:**
- Produces: `Action` str Enum (`VENTILATE, SCRUB, HOLD, ALERT`); `Decision(action: Action, reason: str, severity: int)` frozen (0 ok, 1 watch, 2 act); `RipeningTarget(lo: float = 0.55, hi: float = 0.85)`; `decide(index: float | None, anomaly: bool, flags: int, target: RipeningTarget = RipeningTarget()) -> Decision`. Rule order: warm-up flag gives HOLD "warming up" (sev 0); `FLAG_MQ_FAULT` or `anomaly` or index None gives ALERT (sev 2); index >= hi+0.05 gives VENTILATE (sev 2); index >= hi gives SCRUB (sev 1); otherwise HOLD (sev 0, reason says "below target" when index < lo, else "in target window"); `FLAG_DHT_FAIL` raises severity to at least 1 and appends a note.

- [ ] **Step 1: Write table-driven tests:** warm-up beats everything; MQ fault gives ALERT even when index is in-window; anomaly gives ALERT; index None gives ALERT; 0.30 gives HOLD "below target"; 0.70 gives HOLD "in target window"; 0.86 gives SCRUB; 0.91 gives VENTILATE; DHT fail with 0.70 gives HOLD severity 1.
- [ ] **Step 2: Run `pytest tests/test_decide.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `ethylene/decide.py`.**
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: decision engine`.

### Task 10: Qwen client with fallback

**Files:** Create `ethylene/llm.py`, `tests/test_llm.py`.

**Interfaces:**
- Produces: `DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://OMEN_TAILNET_IP:11434")`; `DEFAULT_MODEL = "qwen3-fast"`; `LLMResult(text: str, ok: bool, source: str, error: str | None)` frozen (`source` is `"qwen"` or `"fallback"`); `strip_thinking(text: str) -> str` (removes `<think>...</think>`, including an unclosed block); `build_messages(snapshot: dict, question: str | None) -> list[dict]` (system message states the data is synthetic/uncalibrated unless the snapshot says real, and that the model must not invent measurements); `fallback_text(snapshot: dict) -> str` (deterministic summary from index, decision, flags); `analyze(snapshot, question=None, *, client: httpx.Client | None = None, url=DEFAULT_URL, model=DEFAULT_MODEL, timeout_s: float = 45.0) -> LLMResult` posting `{model, messages, stream: false, think: false, options: {temperature: 0.2, num_predict: 400}}` to `{url}/api/chat`; any connect error, timeout, non-200 or malformed JSON returns the fallback with `ok=False` and the error text.

- [ ] **Step 1: Write tests using `httpx.MockTransport`:** `test_success_strips_thinking` (`<think>x</think>Hello` gives "Hello", source "qwen"); `test_unclosed_think_is_stripped`; `test_connect_error_falls_back`; `test_timeout_falls_back`; `test_http_500_falls_back`; `test_malformed_json_falls_back`; `test_system_prompt_mentions_uncalibrated_and_synthetic`; `test_request_body_disables_thinking` (captured request has `think: false`, `stream: false`); `test_live_qwen` marked `live`, skipped unless `RUN_LIVE=1`.
- [ ] **Step 2: Run `pytest tests/test_llm.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `ethylene/llm.py`.**
- [ ] **Step 4: Run tests; then `RUN_LIVE=1 pytest tests/test_llm.py -k live -v`.** Expected: PASS; live test reaches omen and returns non-empty text.
- [ ] **Step 5: Commit** `feat: qwen client with fallback`.

### Task 11: API, engine and dashboard

**Files:** Create `ethylene/api.py`, `ethylene/engine.py`, `ethylene/static/index.html`, `tests/test_api.py`; replace `app.py` with a launcher (`uvicorn.run("ethylene.api:app", host="127.0.0.1", port=8000)`); delete `test_app.py`.

**Interfaces:**
- Consumes: Tasks 2-5, 8-10.
- Produces: `Engine(artifacts: Artifacts)` with `ingest(r: Reading) -> None` (warm-up rows get no index), `snapshot() -> dict` (`reading`, `mlp_index`, `transformer_index | None` until 16 readings, `anomaly_score`, `anomaly`, `decision`, `primary`, `data_kind`, last 300 `history` points); background thread feeds it from the selected source. Endpoints: `GET /` dashboard; `GET /api/ports`; `POST /api/source` `{kind: "sim"|"serial", port?, baud?}`; `GET /api/state` (503 with a message naming `python -m ethylene.train` if artifacts are missing); `GET /api/benchmark`; `POST /api/analyze-csv` (multipart; 400 with a readable message on malformed CSV; returns row count, mean index, anomaly fraction, final decision, per-row first/last 5); `POST /api/ask` `{question?}` (plain `def`, so it runs in the threadpool and never blocks other requests). Dashboard (single file, vanilla JS, canvas chart, follows system light/dark): live index and c_comp, T/RH, decision card, model comparison table, CSV upload, "Ask Qwen" box, and a permanent banner "Metrics are on synthetic data, not field accuracy".

- [ ] **Step 1: Write tests with `fastapi.testclient.TestClient`:** `test_state_after_sim_readings` (pump 30 readings; response has the listed fields and a decision); `test_state_503_without_artifacts`; `test_analyze_csv_ok` (CSV produced by `save_csv` of a simulated run); `test_analyze_csv_malformed_returns_400`; `test_ask_falls_back_when_llm_down` (monkeypatch `llm.analyze` to return a fallback result; endpoint 200 with `source == "fallback"`); `test_warmup_rows_have_no_index`; `test_dashboard_served_with_synthetic_banner`.
- [ ] **Step 2: Run `pytest tests/test_api.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement the engine, API and dashboard; replace `app.py`; delete `test_app.py`.**
- [ ] **Step 4: Run `pytest -q`.** Expected: all tests PASS.
- [ ] **Step 5: Run `python app.py`, open `http://127.0.0.1:8000`, confirm live values update, the decision card changes, CSV upload works, and "Ask Qwen" answers from omen;** capture a screenshot.
- [ ] **Step 6: Commit** `feat: api, engine and dashboard`.

### Task 12: README, CI and final verification

**Files:** Modify `README.md`, `.github/workflows/ci.yml`.

- [ ] **Step 1: Rewrite `README.md`:** wiring table (pins above, 5 V, MQ heater current, DHT11 pull-up), flash/serial steps, `python -m ethylene.train` / `python -m ethylene.export` / `python app.py`, how to log real data to CSV and upload it, the synthetic-data warning, the MQ-135 constants-need-calibration note, and the measured benchmark table from Task 5.
- [ ] **Step 2: Update CI:** install CPU torch with `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt` (torch is already satisfied by the CPU build), run `pytest -q`.
- [ ] **Step 3: Run `pytest -q` and `python -m ethylene.train` once more from a clean checkout state;** confirm `artifacts/metrics.json` values match the README table.
- [ ] **Step 4: If the user agrees to install `arduino-cli`** (`winget install ArduinoSA.CLI`, then `arduino-cli core install arduino:avr`), run the compile test and record flash/RAM use; otherwise state in the final report that the sketch is not compile-verified.
- [ ] **Step 5: Commit** `docs: readme and ci`.
