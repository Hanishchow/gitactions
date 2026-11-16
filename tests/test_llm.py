import dataclasses
import json
import os
import re

import httpx
import pytest

from ethylene import llm


def make_snapshot(**overrides):
    snap = {
        "source": {"kind": "sim", "port": None, "connected": True},
        "reading": {
            "t_ms": 12000,
            "rs_r0": 0.81,
            "c_comp": 7.891,
            "ldr_raw": 410.0,
            "ldr_blk": 12.0,
            "temp_c": 21.5,
            "rh": 61.0,
            "idx_dev": 0.2,
            "flags": 9,
        },
        "mlp_index": 0.123,
        "transformer_index": 0.456,
        "anomaly_score": 0.234,
        "anomaly": False,
        "threshold": 0.345,
        "decision": {"action": "SCRUB", "reason": "ethylene rising", "severity": 1},
        "primary": "mlp",
        "data_kind": "synthetic",
        "history": [
            {"t_ms": 1000 * i + 7, "mlp": 0.1, "transformer": 0.2, "idx_dev": None,
             "c_comp": 1.0, "temp_c": 20.0, "rh": 50.0, "anomaly_score": 0.1}
            for i in range(300)
        ],
    }
    snap.update(overrides)
    return snap


def ok_body(content):
    return {"model": llm.DEFAULT_MODEL, "message": {"role": "assistant", "content": content}, "done": True}


def client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def analyze_with(handler, snapshot=None, question=None, **kwargs):
    with client_with(handler) as client:
        return llm.analyze(make_snapshot() if snapshot is None else snapshot, question, client=client, **kwargs)


def system_text(snapshot):
    return llm.build_messages(snapshot, None)[0]["content"]


# --- constants -------------------------------------------------------------

def test_defaults():
    assert llm.DEFAULT_MODEL == "qwen3-fast"
    assert llm.DEFAULT_URL == os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


def test_llmresult_is_frozen():
    res = llm.LLMResult(text="t", ok=True, source="qwen", error=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.text = "other"


# --- strip_thinking --------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("<think>x</think>Hello", "Hello"),
        ("<think>a</think>One <think>b</think>Two", "One Two"),
        ("<think>\nline 1\nline 2\n</think>\n\nAnswer", "Answer"),
        ("<THINK>x</THINK>Hi", "Hi"),
        ("  plain text  ", "plain text"),
        ("", ""),
    ],
)
def test_strip_thinking_closed_blocks(raw, expected):
    assert llm.strip_thinking(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Answer. <think>still thinking", "Answer."),
        ("<think>only thinking, never closed", ""),
        ("<think>a</think>Kept <think>unclosed", "Kept"),
        ("reasoning with no opener</think>Final", "Final"),
    ],
)
def test_unclosed_think_is_stripped(raw, expected):
    assert llm.strip_thinking(raw) == expected


# --- build_messages --------------------------------------------------------

def test_system_prompt_mentions_uncalibrated_and_synthetic():
    messages = llm.build_messages(make_snapshot(), "How is it?")
    assert [m["role"] for m in messages] == ["system", "user"]
    system = messages[0]["content"].lower()
    assert "uncalibrated" in system
    assert "synthetic" in system
    assert "simulated" in system
    assert "ppm" in system  # told never to output ppm figures
    assert "120 words" in system
    assert "plain language" in system
    assert "invent" in system


def test_system_prompt_for_missing_data_kind_assumes_synthetic():
    system = system_text({}).lower()
    assert "simulated" in system
    assert "uncalibrated" in system


def test_system_prompt_for_live_data_is_not_labelled_simulated():
    system = system_text(make_snapshot(data_kind="live")).lower()
    assert "simulated" not in system
    assert "uncalibrated" in system
    assert "ppm" in system


def test_user_message_carries_question_and_values():
    messages = llm.build_messages(make_snapshot(), "Should I vent now?")
    user = messages[1]["content"]
    assert "Should I vent now?" in user
    for value in ("0.123", "0.456", "7.891", "SCRUB", "ethylene rising"):
        assert value in user


@pytest.mark.parametrize("question", [None, "", "   "])
def test_blank_question_gets_default(question):
    user = llm.build_messages(make_snapshot(), question)[1]["content"]
    assert f"Question: {llm.DEFAULT_QUESTION}" in user


def test_user_message_decodes_flags_and_repeats_data_kind_reminder():
    synthetic = llm.build_messages(make_snapshot(), None)[1]["content"]
    assert '"flags_decoded": ["warm-up", "vent relay on"]' in synthetic  # flags = 9
    assert "simulated" in synthetic.splitlines()[-1]
    assert '"flags_decoded": ["none set"]' in llm.build_messages(make_snapshot(reading={"flags": 0}), None)[1]["content"]
    assert "flags_decoded" not in llm.build_messages(make_snapshot(reading=None), None)[1]["content"]
    live = llm.build_messages(make_snapshot(data_kind="live"), None)[1]["content"]
    assert "simulated" not in live
    assert "uncalibrated" in live.splitlines()[-1]


def test_history_is_trimmed_to_recent_points():
    user = llm.build_messages(make_snapshot(), None)[1]["content"]
    assert '"t_ms": 299007' in user  # newest point present
    assert '"t_ms": 7,' not in user  # oldest point dropped


class _Unserializable:
    def __repr__(self):
        return "<unserializable>"


GARBAGE_SNAPSHOTS = [
    {},
    None,
    [],
    "not a dict",
    {"reading": None, "decision": None, "history": None, "data_kind": None, "primary": None},
    {"reading": [1, 2], "decision": "HOLD", "history": "xyz", "mlp_index": "abc", "flags": "no"},
    {"mlp_index": float("nan"), "transformer_index": float("inf"), "anomaly_score": True,
     "threshold": {"a": 1}, "anomaly": "yes", "history": [None, 3, "x", {"mlp": None}]},
    {"reading": {"flags": None, "c_comp": None, "temp_c": "hot", "rh": []}, "decision": {"action": None}},
    {"reading": {"flags": 1.5}, "decision": {"action": "HOLD", "reason": None, "severity": None}},
    {"decision": {"action": 5, "reason": 7, "severity": "high"}, "extra": _Unserializable()},
]


@pytest.mark.parametrize("snapshot", GARBAGE_SNAPSHOTS)
def test_build_messages_tolerates_garbage(snapshot):
    messages = llm.build_messages(snapshot, "q")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(isinstance(m["content"], str) and m["content"] for m in messages)


# --- fallback_text ---------------------------------------------------------

def test_fallback_restates_snapshot_values():
    text = llm.fallback_text(make_snapshot())
    for expected in ("0.123", "0.456", "7.891", "0.234", "0.345", "21.5", "61.0", "SCRUB", "ethylene rising"):
        assert expected in text
    assert "synthetic" in text.lower()
    assert "field accuracy" in text.lower()
    assert "uncalibrated" in text.lower()
    assert "mlp" in text.lower() and "primary" in text.lower()
    assert "warm-up" in text  # flags = 9 -> warm-up (1) + vent relay on (8)
    assert "vent relay on" in text
    assert "scrub relay on" not in text
    assert "DHT" not in text


def test_fallback_is_deterministic():
    assert llm.fallback_text(make_snapshot()) == llm.fallback_text(make_snapshot())


def test_fallback_never_invents_numbers():
    snap = make_snapshot()
    # every number in the text must be a value (as formatted) present in the snapshot
    allowed = {"0.123", "0.456", "7.891", "0.234", "0.345", "21.5", "61.0", "1"}
    numbers = set(re.findall(r"\d+(?:\.\d+)?", llm.fallback_text(snap)))
    assert numbers <= allowed, numbers - allowed
    assert not re.search(r"\d\s*ppm", llm.fallback_text(snap), re.IGNORECASE)


def test_fallback_for_empty_snapshot_has_no_numbers_and_no_verdicts():
    text = llm.fallback_text({})
    assert not re.search(r"\d", text)
    assert "synthetic" in text.lower()
    lowered = text.lower()
    assert "normal" not in lowered
    assert "hold" not in lowered
    assert "not available" in lowered


def test_fallback_does_not_claim_normal_when_anomaly_missing():
    text = llm.fallback_text(make_snapshot(anomaly=None, anomaly_score=None, threshold=None))
    assert "normal" not in text.lower()
    assert "0.234" not in text


def test_fallback_reports_anomaly_flag():
    assert "anomaly detected" in llm.fallback_text(make_snapshot(anomaly=True)).lower()
    assert "normal" in llm.fallback_text(make_snapshot(anomaly=False)).lower()


def test_fallback_live_data_is_labelled():
    text = llm.fallback_text(make_snapshot(data_kind="live"))
    assert "live" in text.lower()
    assert "synthetic" in text.lower()  # models were trained on synthetic data
    assert "not field accuracy" in text.lower()


def test_fallback_decodes_every_flag_bit():
    text = llm.fallback_text(make_snapshot(reading={"flags": 31}))
    for name in ("warm-up", "DHT", "MQ", "vent relay on", "scrub relay on"):
        assert name in text
    assert "none set" in llm.fallback_text(make_snapshot(reading={"flags": 0}))


@pytest.mark.parametrize("snapshot", GARBAGE_SNAPSHOTS)
def test_fallback_tolerates_garbage(snapshot):
    text = llm.fallback_text(snapshot)
    assert isinstance(text, str) and text
    assert "nan" not in text.lower() and "inf" not in text.lower()


# --- analyze ---------------------------------------------------------------

def test_success_strips_thinking():
    result = analyze_with(lambda request: httpx.Response(200, json=ok_body("<think>x</think>Hello")))
    assert result == llm.LLMResult(text="Hello", ok=True, source="qwen", error=None)


def test_success_with_unclosed_think_keeps_answer():
    result = analyze_with(lambda request: httpx.Response(200, json=ok_body("Fine. <think>trailing")))
    assert result.ok and result.source == "qwen" and result.text == "Fine."


def test_request_body_disables_thinking():
    seen = {}

    def handler(request):
        seen["request"] = request
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body("ok"))

    result = analyze_with(handler, question="Is it ripe?")
    assert result.ok
    request, body = seen["request"], seen["body"]
    assert request.method == "POST"
    assert request.url == httpx.URL(llm.DEFAULT_URL.rstrip("/") + "/api/chat")
    assert body["think"] is False
    assert body["stream"] is False
    assert body["model"] == "qwen3-fast"
    assert body["options"] == {"temperature": 0.2, "num_predict": 400}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "Is it ripe?" in body["messages"][1]["content"]


def test_custom_url_and_model_are_used():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=ok_body("ok"))

    analyze_with(handler, url="http://omen.test:1234/", model="other-model")
    assert seen == {"url": "http://omen.test:1234/api/chat", "model": "other-model"}


def test_timeout_is_passed_to_request():
    seen = {}

    def handler(request):
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, json=ok_body("ok"))

    analyze_with(handler, timeout_s=7.5)
    assert seen["timeout"] == {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}


def assert_fallback(result, snapshot, error_contains=None):
    assert result.ok is False
    assert result.source == "fallback"
    assert result.text == llm.fallback_text(snapshot)
    assert isinstance(result.error, str) and result.error
    if error_contains:
        assert error_contains.lower() in result.error.lower()


def test_connect_error_falls_back():
    def handler(request):
        raise httpx.ConnectError("no route to host", request=request)

    assert_fallback(analyze_with(handler), make_snapshot(), "no route to host")


@pytest.mark.parametrize("exc", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout])
def test_timeout_falls_back(exc):
    def handler(request):
        raise exc("slow", request=request)

    result = analyze_with(handler, timeout_s=0.1)
    assert_fallback(result, make_snapshot(), "Timeout")


def test_http_500_falls_back():
    result = analyze_with(lambda request: httpx.Response(500, json={"error": "boom"}))
    assert_fallback(result, make_snapshot(), "500")
    assert "boom" in result.error


def test_http_404_model_missing_falls_back():
    result = analyze_with(lambda request: httpx.Response(404, json={"error": "model 'qwen3-fast' not found"}))
    assert_fallback(result, make_snapshot(), "404")


def test_malformed_json_falls_back():
    assert_fallback(analyze_with(lambda request: httpx.Response(200, content=b"<html>not json")), make_snapshot())


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": 1},
        [],
        "just a string",
        None,
        {"message": None},
        {"message": []},
        {"message": {"content": None}},
        {"message": {"content": 42}},
        {"message": {}},
    ],
)
def test_wrong_json_shape_falls_back(payload):
    assert_fallback(analyze_with(lambda request: httpx.Response(200, json=payload)), make_snapshot())


@pytest.mark.parametrize("content", ["", "   \n", "<think>reasoning that never closes", "<think>a</think>"])
def test_reply_with_no_answer_falls_back(content):
    result = analyze_with(lambda request: httpx.Response(200, json=ok_body(content)))
    assert_fallback(result, make_snapshot(), "empty")


def test_unexpected_transport_error_falls_back():
    def handler(request):
        raise OSError("socket exploded")

    assert_fallback(analyze_with(handler), make_snapshot(), "socket exploded")


def test_bad_url_falls_back():
    with httpx.Client() as client:
        result = llm.analyze(make_snapshot(), None, client=client, url="not a url")
    assert_fallback(result, make_snapshot())


@pytest.mark.parametrize("snapshot", GARBAGE_SNAPSHOTS)
def test_analyze_never_raises_for_garbage_snapshots(snapshot):
    ok = analyze_with(lambda request: httpx.Response(200, json=ok_body("fine")), snapshot=snapshot)
    assert ok.ok and ok.text == "fine"

    def handler(request):
        raise httpx.ConnectError("down", request=request)

    bad = analyze_with(handler, snapshot=snapshot)
    assert bad.ok is False and bad.source == "fallback" and bad.text


def test_injected_client_is_not_closed():
    client = client_with(lambda request: httpx.Response(200, json=ok_body("ok")))
    assert llm.analyze(make_snapshot(), None, client=client).ok
    assert not client.is_closed
    client.close()


def test_owned_client_is_created_and_closed(monkeypatch):
    real_client = httpx.Client
    made = []

    def factory(*args, **kwargs):
        client = real_client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ok_body("ok"))))
        made.append(client)
        return client

    monkeypatch.setattr(llm.httpx, "Client", factory)
    result = llm.analyze(make_snapshot(), None)
    assert result.ok
    assert len(made) == 1 and made[0].is_closed


def test_owned_client_is_closed_on_failure(monkeypatch):
    real_client = httpx.Client
    made = []

    def handler(request):
        raise httpx.ConnectError("down", request=request)

    def factory(*args, **kwargs):
        client = real_client(transport=httpx.MockTransport(handler))
        made.append(client)
        return client

    monkeypatch.setattr(llm.httpx, "Client", factory)
    result = llm.analyze(make_snapshot(), None)
    assert result.ok is False
    assert made[0].is_closed


# --- live (needs omen reachable over Tailscale) ----------------------------

@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real Ollama server")
def test_live_qwen():
    snapshot = make_snapshot()
    result = llm.analyze(snapshot, "Summarise the current state in two sentences.", timeout_s=90)
    assert result.ok, result.error
    assert result.source == "qwen"
    assert result.error is None
    assert result.text.strip()
    assert "<think>" not in result.text.lower()
