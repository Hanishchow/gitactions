"""Boundary-aware chunking for CampusGPT.

Documents returned by :func:`campus_gpt.ingest.extract_file` are split into
overlapping chunks of roughly ``chunk_size`` characters.  Cuts prefer sentence
and paragraph boundaries and never start or end in the middle of a word (unless
a single word is longer than the requested chunk size).
"""

from __future__ import annotations

import re
import uuid
from pathlib import PurePath
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["chunk_documents", "infer_subject_module"]

# A subject code is three letters followed by three digits, e.g. TST101.
_SUBJECT_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{3}\d{3})(?![A-Za-z0-9])")
_MODULE_RE = re.compile(r"module[\s_-]*(\d+)", re.IGNORECASE)

_SENTENCE_END_RE = re.compile(r"[.!?](?=\s)")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def infer_subject_module(
    filename: str,
) -> Tuple[Optional[str], Optional[int]]:
    """Infer a subject code and module number from a filename.

    Examples
    --------
    >>> infer_subject_module("TST101-module-2-notes.pdf")
    ('TST101', 2)
    >>> infer_subject_module("Module-3_Notes.md")
    (None, 3)
    >>> infer_subject_module("random-notes.pdf")
    (None, None)
    """

    # Work with the basename so directory names cannot accidentally match.
    name = PurePath(str(filename)).name if filename else ""

    subject_match = _SUBJECT_CODE_RE.search(name)
    subject = subject_match.group(1).upper() if subject_match else None

    module_match = _MODULE_RE.search(name)
    module = int(module_match.group(1)) if module_match else None

    return subject, module


def _normalise_text(text: str) -> str:
    """Collapse whitespace while preserving a single paragraph separator."""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_cut_position(text: str, start: int, ideal_end: int) -> int:
    """Choose a word/sentence boundary at or before ``ideal_end``.

    Returning a position that is not the end of the text guarantees that the
    resulting chunk is no longer than ``chunk_size`` characters.
    """

    if ideal_end >= len(text):
        return len(text)

    window = text[start:ideal_end]
    # Prefer a sentence boundary in the second half of the window.  Cutting at
    # the first sentence would produce needlessly small chunks.
    minimum_boundary = len(window) // 2
    last_sentence_end = -1
    for match in _SENTENCE_END_RE.finditer(window):
        if match.end() >= minimum_boundary:
            last_sentence_end = match.end()
    if last_sentence_end != -1:
        return start + last_sentence_end

    # Next best: a paragraph boundary.
    last_paragraph = -1
    for match in _PARAGRAPH_RE.finditer(window):
        last_paragraph = match.end()
    if last_paragraph >= minimum_boundary:
        return start + last_paragraph

    # Otherwise cut at the last whitespace so words stay intact.
    last_space = window.rfind(" ")
    if last_space > 0:
        return start + last_space

    # A single token is longer than the chunk size: hard cut it rather than
    # looping forever.
    return ideal_end


def _rewind_to_word_start(text: str, position: int, minimum: int) -> int:
    """Move ``position`` backwards to the start of a word.

    The returned position is a non-whitespace character immediately following
    whitespace (or ``minimum``) so that stripping a chunk cannot discard the
    requested overlap when it begins with a newline.
    """

    position = min(position, len(text))
    while position > minimum:
        current_is_space = text[position].isspace()
        previous_is_space = text[position - 1].isspace()
        if previous_is_space and not current_is_space:
            break
        position -= 1
    return position


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split normalised text into overlapping, boundary-aware pieces."""

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        ideal_end = min(text_length, start + chunk_size)
        end = _find_cut_position(text, start, ideal_end)

        # Guard against a pathological boundary search returning no progress.
        if end <= start:
            end = ideal_end

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_length:
            break

        # Start the next chunk roughly ``overlap`` characters before the end.
        # Moving backwards to the beginning of a word gives at least the
        # requested overlap, never a partial word.
        next_start = max(start + 1, end - overlap)
        next_start = _rewind_to_word_start(text, next_start, minimum=start + 1)

        # Ensure forward progress even for unusual (very short) chunks.
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def chunk_documents(
    docs: Sequence[Dict[str, Any]],
    overlap: int = 120,
    chunk_size: int = 800,
) -> List[Dict[str, Any]]:
    """Split extracted documents into overlapping chunks.

    Each output chunk has the following keys: ``id`` (a UUID4 string),
    ``text``, ``file``, ``location``, ``subject``, ``module``, and
    ``char_len``.

    Parameters
    ----------
    docs:
        Documents with a ``text`` key.  A ``file`` (or legacy ``path``) key is
        used to infer the subject and module; ``location`` is copied into every
        resulting chunk.
    overlap:
        Number of characters to repeat between adjacent chunks.
    chunk_size:
        Maximum target size in characters for each chunk.

    Raises
    ------
    ValueError
        If ``chunk_size`` is not positive, ``overlap`` is negative, or
        ``overlap`` is not smaller than ``chunk_size``.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict[str, Any]] = []

    for doc in docs:
        raw_text = doc.get("text", "") or ""
        text = _normalise_text(str(raw_text))
        if not text:
            continue

        file_value = doc.get("file", doc.get("path", doc.get("source", ""))) or ""
        file_name = str(file_value)
        subject, module = infer_subject_module(file_name)

        # Respect metadata already attached by the caller, when present.
        if doc.get("subject") is not None:
            subject = str(doc["subject"]).upper()
        if doc.get("module") is not None:
            module = int(doc["module"])

        location = dict(doc.get("location") or {})

        for piece in _split_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": piece,
                    "file": file_name,
                    "location": dict(location),
                    "subject": subject,
                    "module": module,
                    "char_len": len(piece),
                }
            )

    return chunks
