import itertools

import pytest
from fastapi.testclient import TestClient

from ethylene import api, sim
from ethylene.data import FLAG_MQ_FAULT, FLAG_WARMUP, save_csv
from ethylene.llm import LLMResult
from ethylene.serial_io import Reading, SimulatedSource


@pytest.fixture(scope="module")
def client():
    app = api.create_app("artifacts", autostart=False)
    with TestClient(app) as c:
        yield c


def pump(client, n):
    for r in itertools.islice(SimulatedSource(seed=3), n):
        client.app.state.engine.ingest(r)


def test_state_after_sim_readings(client):
    client.app.state.engine.reset("synthetic")
    pump(client, 30)
    body = client.get("/api/state").json()
    for key in ("source", "reading", "mlp_index", "transformer_index", "anomaly_score", "anomaly",
                "threshold", "decision", "primary", "data_kind", "history"):
        assert key in body
    assert body["data_kind"] == "synthetic"
    assert 0.0 <= body["mlp_index"] <= 1.0
    assert body["transformer_index"] is not None
    assert body["decision"]["action"] in ("HOLD", "SCRUB", "VENTILATE", "ALERT")
    assert len(body["history"]) == 30


def test_warmup_rows_have_no_index(client):
    eng = client.app.state.engine
    eng.reset("synthetic")
    eng.ingest(next(iter(SimulatedSource(seed=1, warmup_steps=3))))
    body = client.get("/api/state").json()
    assert body["mlp_index"] is None and body["decision"]["action"] == "HOLD"
    assert "warming" in body["decision"]["reason"]


def test_mq_fault_reading_alerts(client):
    eng = client.app.state.engine
    eng.reset("live")
    eng.ingest(Reading(t_ms=1000, adc_mq=1023, rs_r0=None, c_comp=None, ldr_raw=600, ldr_blk=650,
                       temp_c=24, rh=60, idx_dev=None, flags=FLAG_MQ_FAULT))
    body = client.get("/api/state").json()
    assert body["decision"]["action"] == "ALERT" and body["mlp_index"] is None


def test_state_503_without_artifacts(tmp_path):
    with TestClient(api.create_app(str(tmp_path / "missing"), autostart=False)) as c:
        for path in ("/api/state", "/api/benchmark"):
            r = c.get(path)
            assert r.status_code == 503 and "ethylene.train" in r.json()["detail"]


def test_benchmark_returns_metrics(client):
    body = client.get("/api/benchmark").json()
    assert body["data"]["kind"] == "synthetic" and "mlp" in body


def test_analyze_csv_ok(client, tmp_path):
    path = tmp_path / "run.csv"
    save_csv(sim.simulate(1, 40, seed=5), path)
    with open(path, "rb") as f:
        r = client.post("/api/analyze-csv", files={"file": ("run.csv", f, "text/csv")})
    body = r.json()
    assert r.status_code == 200 and body["rows"] == 40
    assert 0.0 <= body["mean_index"] <= 1.0 and len(body["first"]) == 5 and len(body["last"]) == 5
    assert body["decision"]["action"] in ("HOLD", "SCRUB", "VENTILATE", "ALERT")


@pytest.mark.parametrize("content", [b"foo,bar\n1,2\n", b"", b"\xff\xfe\x00garbage"])
def test_analyze_csv_malformed_returns_400(client, content):
    r = client.post("/api/analyze-csv", files={"file": ("bad.csv", content, "text/csv")})
    assert r.status_code == 400 and r.json()["detail"]


def test_ask_falls_back_when_llm_down(client, monkeypatch):
    monkeypatch.setattr(api.llm, "analyze", lambda snap, q=None: LLMResult("offline summary", False, "fallback", "down"))
    r = client.post("/api/ask", json={"question": None})
    assert r.status_code == 200 and r.json()["source"] == "fallback" and r.json()["ok"] is False


def test_source_validation(client):
    assert client.post("/api/source", json={"kind": "serial"}).status_code == 400
    assert client.post("/api/source", json={"kind": "bogus"}).status_code == 400


def test_ports_and_page(client):
    assert isinstance(client.get("/api/ports").json()["ports"], list)
    assert "synthetic" in client.get("/").text
