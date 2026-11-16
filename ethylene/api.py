"""FastAPI app: live state, benchmark, CSV analysis, Qwen analysis, source selection.

Run with ``python app.py``. The front end is a single page at ``ethylene/static/index.html``
(drop your own template there); this module only serves it and exposes the JSON API.
"""

from __future__ import annotations

import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ethylene import llm, serial_io
from ethylene.data import load_csv
from ethylene.engine import Engine, batch_analyze
from ethylene.train import load_artifacts

STATIC = Path(__file__).resolve().parent / "static"
BANNER = "Metrics are on synthetic data, not field accuracy"
PLACEHOLDER = f"<!doctype html><title>Ethylene instrument</title><p>{BANNER}.</p><p>API docs: <a href='/docs'>/docs</a></p>"
NOT_TRAINED = "Models are not trained yet: run `python -m ethylene.train`."


class SourceRequest(BaseModel):
    kind: str
    port: str | None = None
    baud: int = 115200


class AskRequest(BaseModel):
    question: str | None = None


class SourceRunner:
    """Feeds the engine from a simulated or serial source on a background thread."""

    def __init__(self, engine: Engine, sim_interval_s: float):
        self.engine, self.sim_interval_s = engine, sim_interval_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._serial: serial_io.SerialSource | None = None

    def start(self, kind: str, port: str | None = None, baud: int = 115200) -> None:
        self.stop()
        self._stop = threading.Event()
        if kind == "sim":
            self.engine.reset("synthetic")
            self.engine.set_source("sim", None, True)
            target = self._run_sim
        elif kind == "serial":
            if not port:
                raise ValueError("a serial port is required, for example COM4")
            self.engine.reset("live")
            self._serial = serial_io.SerialSource(port, baud)
            self.engine.set_source("serial", port, False)
            target = self._run_serial
        else:
            raise ValueError("kind must be 'sim' or 'serial'")
        self._thread = threading.Thread(target=target, daemon=True, name=f"source-{kind}")
        self._thread.start()

    def _run_sim(self) -> None:
        seed = 0
        while not self._stop.is_set():
            self.engine.reset("synthetic")
            for reading in serial_io.SimulatedSource(seed=seed, interval_s=0.0):
                if self._stop.is_set():
                    return
                self.engine.ingest(reading)
                self._stop.wait(self.sim_interval_s)
            seed += 1

    def _run_serial(self) -> None:
        src = self._serial
        for reading in src.readings():
            if self._stop.is_set():
                break
            self.engine.set_source("serial", src.port, True)
            self.engine.ingest(reading)

    def stop(self) -> None:
        self._stop.set()
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self.engine.set_source(None, None, False)


def create_app(artifacts_dir: str = "artifacts", autostart: bool = True, sim_interval_s: float = 1.0) -> FastAPI:
    try:
        engine: Engine | None = Engine(load_artifacts(artifacts_dir))
    except (OSError, ValueError, KeyError):
        engine = None
    runner = SourceRunner(engine, sim_interval_s) if engine else None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if runner and autostart:
            runner.start("sim")
        yield
        if runner:
            runner.stop()

    app = FastAPI(title="Ethylene instrument", lifespan=lifespan)
    app.state.engine, app.state.runner = engine, runner

    def need_engine() -> Engine:
        if engine is None:
            raise HTTPException(status_code=503, detail=NOT_TRAINED)
        return engine

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = STATIC / "index.html"
        return FileResponse(page) if page.exists() else HTMLResponse(PLACEHOLDER)

    @app.get("/api/ports")
    def ports():
        return {"ports": serial_io.list_ports()}

    @app.post("/api/source")
    def set_source(req: SourceRequest):
        need_engine()
        try:
            runner.start(req.kind, req.port, req.baud)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"source": engine.snapshot()["source"]}

    @app.get("/api/state")
    def state():
        return need_engine().snapshot()

    @app.get("/api/benchmark")
    def benchmark():
        return need_engine().art.metrics

    @app.post("/api/analyze-csv")
    async def analyze_csv(file: UploadFile = File(...)):
        eng = need_engine()
        data = await file.read()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.csv"
            path.write_bytes(data)
            try:
                ds = load_csv(path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
        if len(ds.X) == 0:
            raise HTTPException(status_code=400, detail="the CSV has no data rows")
        return batch_analyze(eng.art, ds)

    @app.post("/api/ask")
    def ask(req: AskRequest):
        result = llm.analyze(need_engine().snapshot(), req.question)
        return {"text": result.text, "ok": result.ok, "source": result.source, "error": result.error}

    return app


app = create_app()
