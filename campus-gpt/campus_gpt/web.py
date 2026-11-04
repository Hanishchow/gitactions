"""Web mode: Ollama Cloud web search + chat model, no knowledge database.

``WebBot.answer(question, history)`` has the same interface as ``CampusBot`` so the
server can swap them with ``CAMPUS_MODE=web|db``.
"""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from .bot import GREETING, _GREET_RE
from .ollama import DEFAULT_URL

Generate = Callable[[List[Dict[str, str]]], str]

WEB_PROMPT = (
    "You are Campus GPT, a study assistant for VTU (Visvesvaraya Technological University) "
    "AI & ML engineering students. Use the numbered web search results below as your main source "
    "and cite them like [1]. Explain clearly and concisely for an undergraduate. If the results "
    "do not answer the question, say so and give a short general answer marked "
    "'(general knowledge, not verified)'. For VTU-specific rules, marks, attendance or syllabus, "
    "tell the student to confirm on https://vtu.ac.in. IMPORTANT: cite only with [n] numbers of the "
    "results above. Never write 'Source:' lines, never say information comes from the VTU syllabus "
    "or any organisation unless that exact result is cited, and never invent sources."
)
_JUNK_CITE = re.compile(r"【[^】]*】|^\s*\*?\s*source:.*$", re.IGNORECASE | re.MULTILINE)
VTU_DOMAIN = "vtu.ac.in"
VTU_LABEL = "Official VTU sources"
OTHER_LABEL = "No official VTU page was found for this; these are general web sources, so please confirm on https://vtu.ac.in"


def is_vtu(url: str) -> bool:
    """True only for vtu.ac.in and its subdomains (not look-alike hosts)."""
    host = (urlparse(url).hostname or "").lower()
    return host == VTU_DOMAIN or host.endswith("." + VTU_DOMAIN)


NO_SEARCH_NOTE = "(Web search was unavailable, so this is from the model's general knowledge and may be incomplete.)"


def web_search(query: str, api_key: str, base_url: Optional[str] = None, max_results: int = 5,
               timeout: float = 25.0) -> List[Dict[str, str]]:
    """Return ``[{"title", "url", "content"}]`` from Ollama's web-search API; ``[]`` on any failure."""
    url = (base_url or os.environ.get("OLLAMA_URL", DEFAULT_URL)).rstrip("/") + "/api/web_search"
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}"},
                             json={"query": query, "max_results": max_results}, timeout=timeout)
        if resp.status_code != 200:
            return []
        rows = resp.json().get("results", [])
    except (requests.RequestException, ValueError):
        return []
    out = []
    for r in rows[:max_results]:
        if isinstance(r, dict) and r.get("url"):
            out.append({"title": str(r.get("title", ""))[:120], "url": str(r["url"]),
                        "content": str(r.get("content", ""))[:1200]})
    return out


class WebBot:
    """Answer by searching the web and letting the model summarise with citations."""

    def __init__(self, generate: Generate, api_key: str, search: Optional[Callable[..., List[Dict[str, str]]]] = None) -> None:
        self.generate = generate
        self.api_key = api_key
        self.search = search or web_search

    def answer(self, question: str, history: Optional[Sequence[Any]] = None) -> str:
        text = (question or "").strip()
        if _GREET_RE.match(text):
            return GREETING.replace("I only answer from that database.", "I search the web and explain with sources.")
        if not text:
            return "Ask me anything about your AI & ML subjects."
        vtu = [r for r in self.search(f"{text} site:{VTU_DOMAIN}", self.api_key) if is_vtu(r["url"])]
        results = vtu or self.search(text, self.api_key)
        label = VTU_LABEL if vtu else OTHER_LABEL
        if results:
            ctx = "\n\n".join(f"[{i}] {r['title']} ({r['url']})\n{r['content']}" for i, r in enumerate(results, 1))
            system = WEB_PROMPT + "\n\nSearch results:\n" + ctx
        else:
            system = WEB_PROMPT + "\n\nNo search results are available. " + NO_SEARCH_NOTE
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for m in list(history or [])[-6:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                msgs.append({"role": m["role"], "content": m["content"][:1500]})
        msgs.append({"role": "user", "content": text})
        try:
            answer = _JUNK_CITE.sub("", self.generate(msgs)).strip()
        except Exception:
            return "The language model is unavailable right now. Please try again in a few minutes."
        if results:
            refs = "\n".join(f"[{i}] {r['title'] or r['url']} - {r['url']}" for i, r in enumerate(results, 1))
            if vtu:
                return f"{answer}\n\n{label}:\n{refs}"
            return f"{answer}\n\n{label}.\nSources:\n{refs}"
        return f"{answer}\n\n{NO_SEARCH_NOTE}"
