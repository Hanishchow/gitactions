"""Live inference engine: readings in, predictions + anomaly + decision out (contract: handoff/contracts.md B)."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import torch

from ethylene.data import FLAG_WARMUP, Dataset
from ethylene.decide import decide
from ethylene.serial_io import Reading
from ethylene.train import WINDOW, Artifacts, anomaly_mask

HISTORY_LEN = 300
FEATURES = ("rs_r0", "c_comp", "ldr_raw", "ldr_blk", "temp_c", "rh")


def _f(value) -> float | None:
    return None if value is None else float(value)


def _features(r: Reading) -> np.ndarray | None:
    values = [getattr(r, name) for name in FEATURES]
    return None if any(v is None for v in values) else np.asarray(values, dtype=np.float32)


class Engine:
    def __init__(self, art: Artifacts, data_kind: str = "synthetic"):
        self.art = art
        self.data_kind = data_kind
        self.source_info = {"kind": None, "port": None, "connected": False}
        self._lock = threading.Lock()
        self._window: deque[np.ndarray] = deque(maxlen=WINDOW)
        self._history: deque[dict] = deque(maxlen=HISTORY_LEN)
        self._last: dict | None = None

    def reset(self, data_kind: str) -> None:
        with self._lock:
            self.data_kind = data_kind
            self._window.clear()
            self._history.clear()
            self._last = None

    def set_source(self, kind: str | None, port: str | None, connected: bool) -> None:
        with self._lock:
            self.source_info = {"kind": kind, "port": port, "connected": connected}

    def ingest(self, r: Reading) -> None:
        art = self.art
        warm = bool(r.flags & FLAG_WARMUP)
        x = None if warm else _features(r)
        mlp = tf = score = None
        anomaly = False
        with torch.no_grad():
            if x is None:
                self._window.clear()
            else:
                xs = torch.as_tensor(art.scaler.transform(x[None, :]))
                mlp = float(art.mlp(xs)[0])
                flagged, scores = anomaly_mask(art.ae, art.scaler, art.threshold, x[None, :])
                score, anomaly = float(scores[0]), bool(flagged[0])
                self._window.append(xs[0].numpy())
                if len(self._window) == WINDOW:
                    w = torch.as_tensor(np.stack(self._window)[None, ...])
                    tf = float(art.transformer(w)[0])
        primary = art.metrics.get("primary", "mlp")
        index = tf if (primary == "transformer" and tf is not None) else mlp
        decision = decide(index, anomaly, r.flags)
        point = {"t_ms": r.t_ms, "mlp": mlp, "transformer": tf, "idx_dev": r.idx_dev, "c_comp": r.c_comp,
                 "temp_c": r.temp_c, "rh": r.rh, "anomaly_score": score}
        snap = {
            "reading": {"t_ms": r.t_ms, "rs_r0": r.rs_r0, "c_comp": r.c_comp, "ldr_raw": r.ldr_raw,
                        "ldr_blk": r.ldr_blk, "temp_c": r.temp_c, "rh": r.rh, "idx_dev": r.idx_dev,
                        "flags": r.flags},
            "mlp_index": mlp, "transformer_index": tf, "anomaly_score": score, "anomaly": bool(anomaly),
            "threshold": art.threshold, "decision": decision.to_dict(), "primary": primary,
        }
        with self._lock:
            self._history.append(point)
            self._last = snap

    def snapshot(self) -> dict:
        with self._lock:
            last = dict(self._last) if self._last else {
                "reading": None, "mlp_index": None, "transformer_index": None, "anomaly_score": None,
                "anomaly": False, "threshold": self.art.threshold,
                "decision": {"action": "HOLD", "reason": "waiting for data", "severity": 0},
                "primary": self.art.metrics.get("primary", "mlp"),
            }
            return {"source": dict(self.source_info), **last, "data_kind": self.data_kind,
                    "history": list(self._history)}


def batch_analyze(art: Artifacts, ds: Dataset) -> dict:
    """Analyse a whole dataset (uploaded CSV). Warm-up rows and rows with missing features get no index."""
    n = len(ds.X)
    ok = ~np.isnan(ds.X).any(axis=1) & ((ds.flags & FLAG_WARMUP) == 0)
    index = np.full(n, np.nan, dtype=np.float32)
    anomaly = np.zeros(n, dtype=bool)
    if ok.any():
        with torch.no_grad():
            xs = torch.as_tensor(art.scaler.transform(ds.X[ok]))
            index[ok] = art.mlp(xs).numpy()
            anomaly[ok] = anomaly_mask(art.ae, art.scaler, art.threshold, ds.X[ok])[0]
    last_index = None if np.isnan(index[-1]) else float(index[-1])
    decision = decide(last_index, bool(anomaly[-1]), int(ds.flags[-1]))

    def row(i: int) -> dict:
        return {"t_s": float(ds.t_s[i]), "index": None if np.isnan(index[i]) else float(index[i]),
                "anomaly": bool(anomaly[i]), "c_comp": _f(ds.X[i, 1]) if np.isfinite(ds.X[i, 1]) else None,
                "temp_c": _f(ds.X[i, 4]) if np.isfinite(ds.X[i, 4]) else None,
                "rh": _f(ds.X[i, 5]) if np.isfinite(ds.X[i, 5]) else None}

    return {
        "rows": n,
        "mean_index": float(np.nanmean(index)) if ok.any() else None,
        "anomaly_fraction": float(anomaly[ok].mean()) if ok.any() else 0.0,
        "decision": decision.to_dict(),
        "first": [row(i) for i in range(min(5, n))],
        "last": [row(i) for i in range(max(0, n - 5), n)],
    }
