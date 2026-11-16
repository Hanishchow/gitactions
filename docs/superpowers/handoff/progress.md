# SDD ledger — plan: docs/superpowers/plans/2026-11-16-ethylene-instrument.md

Spec: docs/superpowers/specs/2026-11-16-ethylene-instrument-design.md (reachable)
Worktree: C:/Users/yakka/Downloads/.claude/worktrees/arduino-sensors-ml-app-aed010 (branch claude/arduino-sensors-ml-app-aed010)
arduino-cli: "C:/Program Files/Arduino CLI/arduino-cli.exe" v1.6.0 (installed via winget; AVR core install in progress)
Start BASE: 5569414

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 -> T2 | FEATURE_NAMES | consistent |
| T1 -> T3 | SensorConfig, f_temp, f_rh, concentration | consistent |
| T2 -> T3 | Dataset (flags, run, t_s) | consistent; sim sets flags=0 |
| T2 save_csv | "adc_mq by inverting adc_to_rs with R0 = 1 reference" | ambiguous |
| T3 -> T5 | inject_faults/FAULT_KINDS used by AE metric | consistent; aggregate detection is mean over kinds |
| T4 -> T5 | ConstrainedMLP, stoich_penalty, composite_loss, Autoencoder, TinyTransformer | consistent |
| T5 -> T6 | Artifacts, Scaler; T6 test needs same held-out test set | gap: no shared split helper |
| T5 metrics int8_kb vs T6 flash_bytes | T5 runs before T6 exists | ambiguous |
| T5 monotonic_violation_rate "+1% step" | step size undefined | ambiguous |
| T5 test_warmup_rows_excluded | needs a way to inject a flagged dataset | gap |
| T6 -> T7 | read_header/write_header, model_weights.h | float round-trip needs 9 sig digits |
| T2 -> T7/T8 | CSV_COLUMNS, FLAG_* | consistent |
| T8 -> T11 | Reading, SimulatedSource, SerialSource | consistent |
| T9 -> T11 | decide(index, anomaly, flags) | consistent |
| T10 -> T11 | llm.analyze | consistent |
| T5 -> T11 | Artifacts, load_artifacts; test_state_503 needs app built on a tmp dir | gap: app factory |
| T11 / T1 | T11 deletes test_app.py and rewrites app.py; T1 pytest.ini testpaths=tests | consistent (root test not collected until deleted) |
| T4 | weight_decay=1e-4 vs loss lambda2*||W||^2 | paper says L2 decay 1e-4; keep |
| T12 | CI torch cpu + artifacts committed | consistent |
| Tasks 1-12 self-consistency | tests vs code vs files | each task agrees with itself; T3 DHT "rounded true value plus noise" order ambiguous |

## Rulings
- Ruling: save_csv writes adc_mq using R0 = cfg.rl_ohm (so Rs = rs_r0*RL, adc = adc_max*RL/(Rs+RL)) — gives a valid in-range ADC, only cosmetic — none if wrong.
- Ruling: T5 exposes `make_splits(fast: bool = False) -> tuple[Dataset, Dataset, Dataset]` (train/val/test, deterministic seed) and `main(out_dir, fast, dataset: Dataset | None = None)`; T6 tests and T5 warm-up test use them — shared test set without duplication — trivial rework if wrong.
- Ruling: metrics fp32_kb = params*4/1024 and int8_kb = params/1024 (approximate, weights only); exact flash lives in export.flash_bytes — avoids T5->T6 dependency — none.
- Ruling: monotonic_violation_rate = fraction of test rows where output at (c_comp_std + 0.1) is lower than output at c_comp_std by more than 1e-4 — concrete definition — threshold in test may need tuning.
- Ruling: DHT value = round(true + noise) so values are integral — matches DHT11 — none.
- Ruling: write_header prints floats with 9 significant digits (`%.9g`) + `f` suffix so float32 round-trips exactly — none.
- Ruling: api exposes `create_app(artifacts_dir: str = "artifacts") -> FastAPI` and module-level `app = create_app()`; tests build apps on tmp dirs — makes 503 test possible — none.
- Ruling: AE `fault_detection_rate` = mean over all FAULT_KINDS of detected fraction; per-kind rates also stored in metrics under `per_kind` — richer report — none.
- Ruling: after T7 compiles, controller flashes the board the user connects (`arduino-cli upload`) and reads serial; user authorised control of the board — hardware not wired may print nan/flags, expected.
- Ruling: model tiers — Sonnet for mechanical tasks (T1,T2,T3,T4,T6,T8,T9,T10), Opus for judgment tasks (T5,T7,T11,T12); reviewers Sonnet floor, Opus for T5/T7/T11; final review Fable.

## Progress
- Env: arduino-cli 1.6.0 works, arduino:avr@1.8.8 core installed. `board list` shows COM4 (USB serial, board "Unknown" — likely a CH340 Mega clone). Use fqbn arduino:avr:mega:cpu=atmega2560.
- Task 1: dispatched (sonnet) BASE=5569414 agent a9127886420821774
- Task 1: implemented b01bed0 (DONE, 11 tests; extra guards: concentration NaN rs_r0<=0, f_rh NaN bad denom, pytest.ini pythonpath=.). Reviewer adf9a227951cb527f dispatched (sonnet), diff review-5569414..b01bed0.diff
- Task 1: complete (commits 5569414..b01bed0, review clean). Parked Minors: concentration OverflowError for tiny ratio (physics.py:75); dht_in_range returns np.bool_ for numpy input (physics.py:100); test gaps: dht boundaries/NaN, compensate with both factors off-reference, non-default SensorConfig. Revisit at final review.
- Task 2: dispatched (sonnet) BASE=b01bed0
- Task 2: implemented 20a1181 (33 tests; concerns: run col only when multi-run; load_csv requires all 10 cols). User asked for parallel work from the bottom: dispatched T4 (a8c526b5997897650 sonnet), T10 (a49c34dc6a0db6520 sonnet), T7 (a0f5f32cfaba092fc opus), T11a dashboard (a253928b3734ded44 opus) in shared checkout, path-limited commits; contracts in contracts.md. Reviews use path-limited diffs (review-taskN.diff).
- Running (agent ids): T2 reviewer abf5bd381513b31d9; T3 sim ad790d7af0d5c9a44; T9 decide a5410fa3ceb37c66a; T4 a8c526b5997897650; T10 a49c34dc6a0db6520; T7 a0f5f32cfaba092fc; T11a a253928b3734ded44
- Next: after T3 -> T8 serial_io; after T3+T4 -> T5 train; T5 -> T6 export (+ weights-vs-artifacts parity test moved here from T7); then T11b engine/api; then flash board.
- Task 4: implemented ad24464 (9 tests; params MLP 45057, transformer 17857; dropout 0.1 so eval() needed; stoich_penalty raises if f constant in x). Reviewer dispatched.
- Task 2: complete (20a1181, review clean). Parked Minors: run column dropped for single non-zero run id (data.py:98) and no merge helper (renumber run ids when concatenating files); non-UTF-8 bytes give UnicodeDecodeError without line context (data.py:129); Scaler fully-NaN column warning + to_dict on unfitted; split_by_run with <3 runs silently returns empty splits; BOM/CRLF/adc inverse test gaps.
- Task 4: complete (ad24464, review clean). Parked Minors: differentiability test does not isolate the penalty path (test_models.py:150-156) and ±1-gradient tests do not pin squaring (:133-140) — harden before final; docstring notes (row-independence, .eval() for transformer dropout).
- Task 3: implemented d482371 (20 tests, corrcoef 0.9978). Concerns: 5/36000 rows have rh>100 (not clipped); inject_faults c_comp_stuck uses median over whole array. Task 9: implemented f70425a (241 tests); DHT-fail applies during warm-up too (sev 1); inf index treated as missing. Both need review.
- Ruling: public repo (Hanishchow/gitactions) -> the omen tailnet IP must never be committed; llm.py default URL comes from env OLLAMA_URL (default http://127.0.0.1:11434); plan sanitized; branch pushed via a sanitized export clone (filter-branch) so local history is untouched.
