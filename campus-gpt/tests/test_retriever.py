"""Tests for BM25 retrieval, filtering, and score thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from campus_gpt.rag import Retriever, tokenize


def _chunk(
    chunk_id: str,
    text: str,
    subject: str | None,
    module: int | None,
) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "text": text,
        "file": f"{subject or 'unknown'}-module-{module or 0}-notes.txt",
        "location": {"page": 1},
        "subject": subject,
        "module": module,
        "char_len": len(text),
    }


@pytest.fixture()
def chunks_file(tmp_path: Path) -> Path:
    chunks: List[Dict[str, Any]] = [
        _chunk(
            "chunk-1",
            "Quiet study rooms and research support are available on the second floor.",
            "TST101",
            1,
        ),
        _chunk(
            "chunk-2",
            "The cafeteria serves breakfast lunch and dinner every weekday.",
            "TST101",
            2,
        ),
        _chunk(
            "chunk-3",
            "The library loans books and journals to registered students.",
            "TST102",
            1,
        ),
    ]
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")
    return path


def test_tokenize_lowercases_and_keeps_subject_codes() -> None:
    tokens = tokenize("TST101 library and THE study rooms")

    assert "tst101" in tokens
    assert "library" in tokens
    assert "study" in tokens
    assert "the" not in tokens
    assert "and" not in tokens


def test_retriever_ranks_best_match_first(chunks_file: Path) -> None:
    retriever = Retriever(str(chunks_file))

    results = retriever.retrieve("quiet study rooms", k=3)

    assert results
    assert results[0]["chunk"]["id"] == "chunk-1"
    assert all(result["score"] >= 0 for result in results)


def test_retriever_honours_subject_filter(chunks_file: Path) -> None:
    retriever = Retriever(str(chunks_file))

    results = retriever.retrieve("library books", subject="TST102")

    assert len(results) == 1
    assert results[0]["chunk"]["id"] == "chunk-3"


def test_retriever_honours_module_filter(chunks_file: Path) -> None:
    retriever = Retriever(str(chunks_file))

    results = retriever.retrieve("cafeteria", module=2)

    assert len(results) == 1
    assert results[0]["chunk"]["id"] == "chunk-2"


def test_retriever_returns_empty_when_filter_excludes_everything(
    chunks_file: Path,
) -> None:
    retriever = Retriever(str(chunks_file))

    assert retriever.retrieve("cafeteria", subject="TST102") == []


def test_retriever_returns_empty_for_low_scores(chunks_file: Path) -> None:
    retriever = Retriever(str(chunks_file))

    assert retriever.retrieve("quantum physics") == []


def test_retriever_respects_custom_threshold(chunks_file: Path) -> None:
    retriever = Retriever(str(chunks_file))
    retriever.min_score = 1_000_000

    assert retriever.retrieve("quiet study rooms") == []


def test_retriever_respects_k(chunks_file: Path) -> None:
    retriever = Retriever(str(chunks_file))

    results = retriever.retrieve("library study cafeteria", k=1)

    assert len(results) == 1


def test_retriever_handles_empty_query(chunks_file: Path) -> None:
    retriever = Retriever(str(chunks_file))

    assert retriever.retrieve("the and of") == []
    assert retriever.retrieve("", k=3) == []


def test_retriever_handles_empty_corpus(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    retriever = Retriever(str(path))

    assert retriever.retrieve("anything") == []


def test_retriever_skips_malformed_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_chunk("good", "Library services", "TST101", 1)),
                "{not valid json",
                json.dumps(["not", "an", "object"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    retriever = Retriever(str(path))

    assert len(retriever.chunks) == 1
    assert retriever.retrieve("library")[0]["chunk"]["id"] == "good"


def test_missing_chunks_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Retriever(str(tmp_path / "missing.jsonl"))
