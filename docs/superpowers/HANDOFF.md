# Handoff: ethylene instrument (resume in another account)

Branch `claude/arduino-sensors-ml-app-aed010`. Read `README.md` first, then the spec
(`specs/2026-11-16-ethylene-instrument-design.md`), the plan (`plans/2026-11-16-ethylene-instrument.md`),
and `handoff/progress.md` (ledger of rulings; its commit SHAs predate a history rewrite and are stale).

## Status
Built, tested and committed: physics formulas, data layer, REALISTIC simulator (the first one was trivially easy:
R2 0.996 from one feature), shifted-condition test sets, benchmark of candidate nets and baselines, compact
6-16-8-1 network (257 weights) shipped, autoencoder 6-16-5-16-6 plus a 4-sigma range gate, generic int8 export,
serial parser, decision engine, Qwen client with fallback, engine, FastAPI app, Mega firmware with a serial
self-test (5% flash, 5% RAM, 5.8 ms per inference measured). The board was flashed and `python -m ethylene.hil`
passed 43/43 vectors against the Python reference.

Not done:
- Wiring the real sensors and checking live readings (`rs_r0`, `temp_c`, `idx`); everything on-chip was verified
  with the self-test only. Real logged data is needed to know whether any network beats a ridge regression.
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
