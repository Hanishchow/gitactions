"""Campus GPT web service for celureai: SQLite knowledge DB + Ollama Cloud + Gradio.

Env (from the systemd EnvironmentFile): OLLAMA_API_KEY (required for LLM answers),
OLLAMA_MODEL, CAMPUS_DB (default data/campus.db), CAMPUS_PORT (default 7860).
Binds to 127.0.0.1 only; the public tunnel points at that port.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from campus_gpt.bot import CampusBot
from campus_gpt.ollama import LLMUnavailable, make_generate
from campus_gpt.store import SqliteRetriever

MAX_Q_CHARS = 500
PER_IP = (20, 600)      # 20 questions / 10 minutes per client
GLOBAL = (300, 3600)    # 300 questions / hour overall

_hits = defaultdict(deque)
_global = deque()


def _allow(ip: str, now: float | None = None) -> bool:
    """Sliding-window limiter; returns False when the client or the service is over its quota."""
    now = time.time() if now is None else now
    for dq, (limit, window) in ((_hits[ip], PER_IP), (_global, GLOBAL)):
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= limit:
            return False
    _hits[ip].append(now)
    _global.append(now)
    return True


def build_bot() -> CampusBot:
    db = os.environ.get("CAMPUS_DB", "data/campus.db")
    try:
        gen = make_generate()
    except LLMUnavailable:
        gen = None  # no key yet: answers fall back to cited passages
    return CampusBot(retriever=SqliteRetriever(db), generate=gen)


def main() -> None:
    import gradio as gr

    bot = build_bot()

    def respond(message, history, request: gr.Request = None):
        if request is not None:
            ip = (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
                  or (request.client.host if request.client else "?")).split(",")[0].strip()
        else:
            ip = "?"
        if len(message) > MAX_Q_CHARS:
            return f"Please keep questions under {MAX_Q_CHARS} characters."
        if not _allow(ip):
            return "Too many questions right now. Please try again in a few minutes."
        return bot.answer(message, history)

    kwargs = dict(
        title="Campus GPT - VTU AI&ML assistant",
        description=("Answers only from the VTU rules, the AIML 2022 scheme and uploaded course "
                     "material stored in the knowledge database. If it is not there, it says so."),
    )
    try:
        demo = gr.ChatInterface(fn=respond, type="messages", **kwargs)
    except TypeError:
        demo = gr.ChatInterface(fn=respond, **kwargs)
    demo.queue(max_size=20).launch(server_name="127.0.0.1", server_port=int(os.environ.get("CAMPUS_PORT", "7860")))


if __name__ == "__main__":
    main()
