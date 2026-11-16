"""Qwen (Ollama) client with a deterministic offline fallback.

`analyze` never raises: any network, HTTP, JSON or shape problem yields the
fallback summary with ok=False. The fallback only restates values that are in
the snapshot; it never computes or invents measurements.
"""

import json
import math
import os
import re
from dataclasses import dataclass

import httpx

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = "qwen3-fast"

DEFAULT_QUESTION = "Briefly summarise the current state and what the operator should do next."
HISTORY_POINTS = 5  # recent history entries sent to the model

# flags bits from the instrument (see /api/state contract)
FLAG_NAMES = (
    (1, "warm-up"),
    (2, "DHT sensor failure"),
    (4, "MQ sensor fault"),
    (8, "vent relay on"),
    (16, "scrub relay on"),
)

SYNTHETIC_LABEL = "Metrics are on synthetic data, not field accuracy."
LIVE_LABEL = "Live instrument data, but the models were trained on synthetic data, so metrics are not field accuracy."

_SYSTEM_RULES = (
    "You are the assistant for an ethylene-sensing produce-storage instrument. "
    "Answer in plain language, 120 words max. "
    "Refer only to values given in the snapshot; never invent or estimate measurements. "
    "The ethylene output is an uncalibrated index, not a concentration: never output ppm figures. "
    "Field meanings: mlp_index and transformer_index are ripeness indices (about 0 unripe to 1 ripe) from two "
    "models and primary names the one in use; reading.c_comp is the compensated ethylene index (uncalibrated); "
    "reading.idx_dev is the ripeness index computed on the instrument itself; an anomaly is flagged when "
    "anomaly_score exceeds threshold. "
    "Instrument flags bits: 1 warm-up, 2 DHT sensor failure, 4 MQ sensor fault, "
    "8 vent relay on, 16 scrub relay on. "
    "Decision actions: HOLD, SCRUB, VENTILATE, ALERT."
)
_SYSTEM_SYNTHETIC = (
    "data_kind is synthetic, so you must say that the results are simulated, the ethylene index is "
    "uncalibrated and the metrics are not field accuracy."
)
_SYSTEM_LIVE = (
    "data_kind is live, but the ripeness models were trained on synthetic data, so their metrics are not "
    "field accuracy; remind the reader that the ethylene index is uncalibrated."
)

_REMINDER_SYNTHETIC = "Remember: state that the results are simulated and the ethylene index is uncalibrated."
_REMINDER_LIVE = "Remember: the ethylene index is uncalibrated."

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_THINK_STRAY_CLOSE = re.compile(r".*</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class LLMResult:
    text: str
    ok: bool
    source: str  # "qwen" or "fallback"
    error: str | None = None


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks, an unclosed trailing block, and text before a stray </think>."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_UNCLOSED.sub("", text)
    text = _THINK_STRAY_CLOSE.sub("", text)
    return text.strip()


# --- snapshot helpers (every field may be missing, null or the wrong type) ---

def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _text(value) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None


def _flag_names(flags) -> list[str] | None:
    value = _num(flags)
    if value is None or value != int(value):
        return None
    bits = int(value)
    return [name for bit, name in FLAG_NAMES if bits & bit]


def _is_live(snapshot: dict) -> bool:
    return snapshot.get("data_kind") == "live"


def fallback_text(snapshot: dict) -> str:
    """Deterministic summary that only restates values present in the snapshot."""
    snap = _dict(snapshot)
    reading = _dict(snap.get("reading"))
    decision = _dict(snap.get("decision"))
    primary = _text(snap.get("primary"))
    sentences = [LIVE_LABEL if _is_live(snap) else SYNTHETIC_LABEL]

    # ripeness index from each model
    indices = []
    for name, key in (("MLP", "mlp_index"), ("Transformer", "transformer_index")):
        value = _num(snap.get(key))
        if value is not None:
            label = f"{name} (primary)" if primary and primary.lower() == name.lower() else name
            indices.append(f"{label} {value:.3f}")
    sentences.append(f"Ripeness index: {', '.join(indices)}." if indices else "Ripeness index: not available.")

    # sensor reading
    if reading:
        c_comp = _num(reading.get("c_comp"))
        if c_comp is not None:
            sentences.append(f"Ethylene index (uncalibrated, not a ppm value): {c_comp:.3f}.")
        else:
            sentences.append("Ethylene index (uncalibrated): not available.")
        temp, rh = _num(reading.get("temp_c")), _num(reading.get("rh"))
        if temp is not None:
            sentences.append(f"Temperature {temp:.1f} C.")
        if rh is not None:
            sentences.append(f"Relative humidity {rh:.1f} %.")
    else:
        sentences.append("No sensor reading available.")

    # anomaly detector
    score, threshold, anomaly = _num(snap.get("anomaly_score")), _num(snap.get("threshold")), snap.get("anomaly")
    detail = []
    if score is not None:
        detail.append(f"score {score:.3f}")
    if threshold is not None:
        detail.append(f"threshold {threshold:.3f}")
    detail_text = f" ({', '.join(detail)})" if detail else ""
    if anomaly is True:
        sentences.append(f"Anomaly detected{detail_text}.")
    elif anomaly is False:
        sentences.append(f"Anomaly detector: normal{detail_text}.")
    elif detail:
        sentences.append(f"Anomaly detector: no verdict{detail_text}.")
    else:
        sentences.append("Anomaly detector: not available.")

    # controller decision
    action = _text(decision.get("action"))
    if action:
        line = f"Decision: {action}"
        severity = _num(decision.get("severity"))
        if severity is not None:
            line += f" (severity {severity:g})"
        reason = _text(decision.get("reason"))
        if reason:
            line += f" - {reason}"
        sentences.append(line + ".")
    else:
        sentences.append("Decision: not available.")

    # instrument flags
    names = _flag_names(reading.get("flags"))
    if names is not None:
        sentences.append(f"Instrument flags: {', '.join(names)}." if names else "Instrument flags: none set.")

    return " ".join(sentences)


def build_messages(snapshot: dict, question: str | None) -> list[dict]:
    """System + user chat messages for the model. Tolerates missing or null snapshot fields."""
    snap = _dict(snapshot)
    system = f"{_SYSTEM_RULES} {_SYSTEM_LIVE if _is_live(snap) else _SYSTEM_SYNTHETIC}"

    shown = {key: snap.get(key) for key in (
        "reading", "mlp_index", "transformer_index", "anomaly_score", "anomaly",
        "threshold", "decision", "primary", "data_kind",
    )}
    flags = _flag_names(_dict(snap.get("reading")).get("flags"))
    if flags is not None:
        shown["flags_decoded"] = flags or ["none set"]
    history = snap.get("history")
    if isinstance(history, list):
        shown["history_points"] = len(history)
        shown["history_recent"] = history[-HISTORY_POINTS:]
    asked = question.strip() if isinstance(question, str) else ""
    user = (
        "Instrument snapshot (JSON):\n"
        f"{json.dumps(shown, default=str)}\n\n"
        f"Question: {asked or DEFAULT_QUESTION}\n"
        f"{_REMINDER_LIVE if _is_live(snap) else _REMINDER_SYNTHETIC}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _describe(exc: BaseException) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _ask(client: httpx.Client, url: str, model: str, messages: list[dict], timeout_s: float) -> str:
    """POST one chat request and return the cleaned reply text; raises on any failure."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 400},
    }
    response = client.post(f"{url.rstrip('/')}/api/chat", json=body, timeout=timeout_s)
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
    content = response.json()["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("unexpected response shape: message content is not text")
    text = strip_thinking(content)
    if not text:
        raise ValueError("empty reply from model")
    return text


def analyze(
    snapshot: dict,
    question: str | None = None,
    *,
    client: httpx.Client | None = None,
    url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
    timeout_s: float = 45.0,
) -> LLMResult:
    """Ask Qwen about the snapshot. Never raises: any failure returns the offline fallback with ok=False."""
    owned = client is None
    try:
        if owned:
            client = httpx.Client()
        text = _ask(client, url, model, build_messages(snapshot, question), timeout_s)
        return LLMResult(text=text, ok=True, source="qwen", error=None)
    except Exception as exc:  # noqa: BLE001 - contract: unreachable/slow/garbled Qwen must never raise
        return LLMResult(text=fallback_text(snapshot), ok=False, source="fallback", error=_describe(exc))
    finally:
        if owned and client is not None:
            client.close()
