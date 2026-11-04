"""Tests for chunking and filename metadata inference."""

from __future__ import annotations

import pytest

from campus_gpt.chunking import chunk_documents, infer_subject_module


def _long_text(sentences: int = 60) -> str:
    return " ".join(
        f"Sentence {index} explains a campus topic in enough detail."
        for index in range(sentences)
    )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("TST101-module-2-notes.pdf", ("TST101", 2)),
        ("Module-3_Notes.md", (None, 3)),
        ("TST102_Module_1_slides.pptx", ("TST102", 1)),
        ("random-notes.pdf", (None, None)),
        ("TST205-notes.pdf", ("TST205", None)),
        ("module-4.txt", (None, 4)),
    ],
)
def test_infer_subject_module(filename: str, expected: tuple) -> None:
    assert infer_subject_module(filename) == expected


def test_chunk_documents_creates_expected_metadata() -> None:
    docs = [
        {
            "text": _long_text(),
            "file": "TST101-module-2-notes.pdf",
            "location": {"page": 3},
        }
    ]

    chunks = chunk_documents(docs, overlap=40, chunk_size=200)

    assert len(chunks) > 1
    for chunk in chunks:
        assert set(chunk) == {
            "id",
            "text",
            "file",
            "location",
            "subject",
            "module",
            "char_len",
        }
        assert chunk["file"] == "TST101-module-2-notes.pdf"
        assert chunk["location"] == {"page": 3}
        assert chunk["subject"] == "TST101"
        assert chunk["module"] == 2
        assert chunk["char_len"] == len(chunk["text"])
        assert len(chunk["text"]) <= 200


def test_chunk_documents_overlaps_adjacent_chunks() -> None:
    docs = [{"text": _long_text(), "file": "TST101-module-2-notes.pdf"}]

    chunks = chunk_documents(docs, overlap=40, chunk_size=200)

    for previous, following in zip(chunks, chunks[1:]):
        assert previous["text"][-40:] in following["text"]


def test_chunk_documents_does_not_split_words() -> None:
    text = " ".join(f"word{index}" for index in range(200))
    docs = [{"text": text, "file": "TST101-module-1-notes.txt"}]

    chunks = chunk_documents(docs, overlap=20, chunk_size=120)

    for chunk in chunks:
        words = chunk["text"].split()
        assert words
        assert all(word.startswith("word") and word[4:].isdigit() for word in words)


def test_short_document_produces_one_chunk() -> None:
    docs = [{"text": "A short note.", "file": "Module-1_Notes.md"}]

    chunks = chunk_documents(docs, overlap=120, chunk_size=800)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "A short note."
    assert chunks[0]["subject"] is None
    assert chunks[0]["module"] == 1
    assert isinstance(chunks[0]["id"], str)


def test_empty_documents_are_ignored() -> None:
    docs = [
        {"text": "", "file": "TST101-empty.pdf"},
        {"text": "   \n\t ", "file": "TST101-blank.pdf"},
    ]

    assert chunk_documents(docs) == []


def test_invalid_chunk_settings_raise() -> None:
    docs = [{"text": "Some text.", "file": "TST101-notes.txt"}]

    with pytest.raises(ValueError):
        chunk_documents(docs, chunk_size=0)
    with pytest.raises(ValueError):
        chunk_documents(docs, overlap=-1)
    with pytest.raises(ValueError):
        chunk_documents(docs, chunk_size=100, overlap=100)
