"""Ollama Cloud chat client (``POST {base}/api/chat``) with the same shape as ``llm.load_generator``."""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import requests

DEFAULT_URL = "https://ollama.com"
DEFAULT_MODEL = "gpt-oss:120b"


class LLMUnavailable(RuntimeError):
    """Raised when the model API cannot be reached or rejects the request."""


def make_generate(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    max_tokens: int = 400,
) -> Callable[[List[Dict[str, str]]], str]:
    """Build ``generate(messages) -> str``. Key/model/url fall back to env vars
    ``OLLAMA_API_KEY``, ``OLLAMA_MODEL``, ``OLLAMA_URL``. The key is never logged."""
    key = api_key or os.environ.get("OLLAMA_API_KEY", "")
    mdl = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    url = (base_url or os.environ.get("OLLAMA_URL", DEFAULT_URL)).rstrip("/") + "/api/chat"
    if not key:
        raise LLMUnavailable("OLLAMA_API_KEY is not set")

    def generate(messages: List[Dict[str, str]]) -> str:
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": mdl, "messages": messages, "stream": False,
                      "options": {"temperature": 0, "num_predict": max_tokens}},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise LLMUnavailable(f"network error: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise LLMUnavailable(f"HTTP {resp.status_code}")
        try:
            return resp.json()["message"]["content"]
        except (KeyError, ValueError) as exc:
            raise LLMUnavailable("unexpected response shape") from exc

    return generate
