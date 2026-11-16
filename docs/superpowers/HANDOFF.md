# Handoff: ethylene instrument (resume in another account)

Branch `claude/arduino-sensors-ml-app-aed010`. Read `README.md` first, then the spec
(`specs/2026-11-16-ethylene-instrument-design.md`), the plan (`plans/2026-11-16-ethylene-instrument.md`),
and `handoff/progress.md` (ledger of rulings; its commit SHAs predate a history rewrite and are stale).

## Status
Built, tested (448 pytest cases pass) and committed: physics formulas, data layer, simulator, models
(autoencoder widened to 6-16-5-16-6 because a 3-wide bottleneck detected only 55% of faults), training and
benchmark (`artifacts/`), int8 export and generated `model_weights.h`, serial parser, decision engine, Qwen
client with fallback, engine, FastAPI app, Mega firmware (compiles: 23% flash, 27% RAM).

Not done:
- Flashing and verifying the firmware on the real board and sensors (board was not connected when last
  tried: COM4 CH340 entry was stale). Then read serial output, confirm `rs_r0`, `temp_c`, `idx` look sane and
  that `idx` from the board matches `ethylene.export.quantized_forward` for the same inputs.
- Front end: the user will supply a template; put it at `ethylene/static/index.html` using the JSON contract in
  `handoff/contracts.md` section B. An unreviewed draft dashboard may exist untracked in the original checkout.
- Qwen live check: `OLLAMA_URL` must point at an Ollama server with `qwen3-fast` (the user's "omen" PC,
  which was offline at last attempt). The address is deliberately not committed (public repo).
- Bench calibration of the MQ constants and real-data validation (all metrics are synthetic).

## Local facts (not in git)
Mega 2560 on a CH340 serial port (was COM4). `arduino-cli` 1.6.0 at `C:/Program Files/Arduino CLI/arduino-cli.exe`,
core `arduino:avr@1.8.8`, FQBN `arduino:avr:mega:cpu=atmega2560`.

## Working style the user asked for
Do the work sequentially and economically: no fleets of parallel subagents. At most one Claude agent locally and
one worker on the omen PC (when it is on); prefer direct edits and one final whole-branch review.
