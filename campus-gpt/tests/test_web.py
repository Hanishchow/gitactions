"""Web mode with mocked search and model (no network)."""
from campus_gpt import web
from campus_gpt.web import NO_SEARCH_NOTE, WebBot

RESULTS = [{"title": "ML intro", "url": "https://example.edu/ml", "content": "Machine learning learns patterns from data."}]


def _bot(results, gen=None):
    seen = {}

    def generate(msgs):
        seen["msgs"] = msgs
        return "ML learns from data [1]."

    return WebBot(gen or generate, "k", search=lambda q, k: results), seen


def test_answer_cites_sources_and_passes_results_to_model():
    bot, seen = _bot(RESULTS)
    out = bot.answer("ai ml basics?")
    assert "[1]" in out and "https://example.edu/ml" in out
    assert "No official VTU page" in out
    assert "Machine learning learns patterns" in seen["msgs"][0]["content"]
    assert seen["msgs"][-1] == {"role": "user", "content": "ai ml basics?"}


def test_no_results_is_flagged_not_hidden():
    bot, _ = _bot([])
    assert NO_SEARCH_NOTE in bot.answer("what is overfitting")


def test_model_outage_message():
    def boom(_m):
        raise RuntimeError("x")

    bot, _ = _bot(RESULTS, gen=boom)
    assert "unavailable" in bot.answer("anything")


def test_greeting_and_empty():
    bot, _ = _bot(RESULTS)
    assert "search the web" in bot.answer("hi")
    assert "Ask me" in bot.answer("  ")


def test_history_is_forwarded_and_trimmed():
    bot, seen = _bot(RESULTS)
    hist = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    bot.answer("next", hist)
    assert len(seen["msgs"]) == 1 + 6 + 1


class _R:
    def __init__(self, code, body):
        self.status_code, self._b = code, body

    def json(self):
        return self._b


def test_web_search_parses_and_fails_closed(monkeypatch):
    monkeypatch.setattr(web.requests, "post", lambda *a, **k: _R(200, {"results": [
        {"title": "t", "url": "https://a.b", "content": "c"}, {"title": "no url"}]}))
    assert web.web_search("q", "k") == [{"title": "t", "url": "https://a.b", "content": "c"}]
    monkeypatch.setattr(web.requests, "post", lambda *a, **k: _R(401, {}))
    assert web.web_search("q", "k") == []

    def boom(*a, **k):
        raise web.requests.ConnectionError()

    monkeypatch.setattr(web.requests, "post", boom)
    assert web.web_search("q", "k") == []


def test_vtu_results_preferred_and_labelled():
    calls = []

    def search(q, k):
        calls.append(q)
        if "site:vtu.ac.in" in q:
            return [{"title": "Scheme", "url": "https://vtu.ac.in/pdf/x.pdf", "content": "official"},
                    {"title": "Fake", "url": "https://vtu.ac.in.evil.com/a", "content": "spoof"}]
        return RESULTS

    out = WebBot(lambda m: "ok [1]", "k", search=search).answer("scheme")
    assert "Official VTU sources" in out and "vtu.ac.in/pdf/x.pdf" in out
    assert "evil.com" not in out and "example.edu" not in out
    assert len(calls) == 1


def test_is_vtu_rejects_lookalikes():
    assert web.is_vtu("https://vtu.ac.in/a") and web.is_vtu("https://results.vtu.ac.in/x")
    assert not web.is_vtu("https://notvtu.ac.in/a") and not web.is_vtu("https://vtu.ac.in.evil.com")


def test_model_made_source_claims_are_stripped():
    gen = lambda m: "ML learns from data [1]. 【4†L1-L4】\n*Source: VTU syllabus on AI*"
    out = WebBot(gen, "k", search=lambda q, k: RESULTS).answer("ml")
    assert "VTU syllabus" not in out and "【" not in out and "[1]" in out
