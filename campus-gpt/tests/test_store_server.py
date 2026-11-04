"""SQLite store, Ollama client (mocked HTTP) and the server's rate limiter. No network."""
import json

import pytest

from campus_gpt import ollama
from campus_gpt.bot import REFUSAL, CampusBot
from campus_gpt.store import SqliteRetriever, connect, load_chunks, load_kb
from server import serve

CHUNKS = [
    {"file": "BCS502-module-4-textbook.pdf", "location": {"page": 3}, "subject": "BCS502", "module": 4,
     "text": "The transport layer provides process to process delivery using TCP and UDP."},
    {"file": "BCS503-mod3.pdf", "location": {"page": 7}, "subject": "BCS503", "module": 3,
     "text": "A pushdown automaton recognises context free languages."},
]


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.db"
    conn = connect(p)
    load_kb(conn)
    f = tmp_path / "c.jsonl"
    f.write_text("\n".join(json.dumps(c) for c in CHUNKS), encoding="utf-8")
    load_chunks(conn, f)
    conn.close()
    return p


def test_kb_counts(db):
    c = connect(db)
    assert c.execute("SELECT count(*) FROM rules").fetchone()[0] == 7
    assert c.execute("SELECT count(*) FROM courses WHERE semester=7").fetchone()[0] == 6


def test_fts_retrieval_and_filters(db):
    r = SqliteRetriever(db)
    top = r.retrieve("what does TCP do at the transport layer", k=3)
    assert top and top[0]["chunk"]["subject"] == "BCS502"
    assert r.retrieve("pushdown automaton", subject="BCS503")[0]["chunk"]["module"] == 3
    assert r.retrieve("pushdown automaton", subject="BCS502") == []


def test_retrieval_works_from_another_thread(db):
    import threading

    r = SqliteRetriever(db)
    out = []
    t = threading.Thread(target=lambda: out.append(r.retrieve("TCP transport layer")))
    t.start()
    t.join()
    assert out and out[0]


def test_no_match_and_punctuation_safe(db):
    r = SqliteRetriever(db)
    assert r.retrieve("zebra quantum bananas") == []
    assert r.retrieve('"; DROP TABLE chunks; --') == []
    assert r.retrieve("???") == []


def test_bot_with_sqlite_and_stub_llm(db):
    bot = CampusBot(retriever=SqliteRetriever(db), generate=lambda m: "TCP gives reliable delivery [1].")
    out = bot.answer("explain TCP at the transport layer")
    assert "[1]" in out and "BCS502-module-4-textbook.pdf" in out
    assert bot.answer("who won the world cup") == REFUSAL


def test_llm_outage_falls_back_to_passages(db):
    def boom(_m):
        raise ollama.LLMUnavailable("HTTP 402")

    out = CampusBot(retriever=SqliteRetriever(db), generate=boom).answer("TCP transport layer")
    assert "unavailable" in out and "TCP" in out


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._b = code, body

    def json(self):
        return self._b


def test_ollama_client_request_shape(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, json=json)
        return _Resp(200, {"message": {"content": "hi"}})

    monkeypatch.setattr(ollama.requests, "post", fake_post)
    gen = ollama.make_generate(api_key="k123", model="gpt-oss:20b", base_url="https://example.test/")
    assert gen([{"role": "user", "content": "x"}]) == "hi"
    assert seen["url"] == "https://example.test/api/chat"
    assert seen["headers"]["Authorization"] == "Bearer k123"
    assert seen["json"]["stream"] is False and seen["json"]["options"]["temperature"] == 0


def test_ollama_errors_are_wrapped(monkeypatch):
    monkeypatch.setattr(ollama.requests, "post", lambda *a, **k: _Resp(402, {}))
    with pytest.raises(ollama.LLMUnavailable, match="402"):
        ollama.make_generate(api_key="k")([{"role": "user", "content": "x"}])


def test_missing_key_is_an_error(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ollama.LLMUnavailable):
        ollama.make_generate()


def test_rate_limiter():
    serve._hits.clear()
    serve._global.clear()
    results = [serve._allow("1.2.3.4", now=1000.0 + i) for i in range(22)]
    assert results[:20] == [True] * 20 and results[20:] == [False, False]
    assert serve._allow("5.6.7.8", now=1030.0)          # other client unaffected
    assert serve._allow("1.2.3.4", now=1000.0 + 700)    # window expired
