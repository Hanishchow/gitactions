"""Document ingestion for CampusGPT.

Supported formats
-----------------
* PDF  - extracted page by page with :mod:`pypdf`.
* PPTX - extracted slide by slide with :mod:`python-pptx`; speaker notes are
  included in the slide text.
* DOCX - extracted heading by heading with :mod:`python-docx`.
* TXT/MD - read as UTF-8 text.

Every extractor returns a list of dictionaries with exactly two keys::

    {"text": "...", "location": {"page": 1}}
    {"text": "...", "location": {"slide": 2}}
    {"text": "...", "location": {"section": "Library services"}}

Empty pages/slides/sections are skipped with a logged warning instead of
raising an exception.  This mirrors the behaviour of scanned PDF pages, which
contain no machine-readable text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Union

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "extract_file",
    "extract_pdf",
    "extract_pptx",
    "extract_docx",
    "extract_text_file",
]

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, "Path"]

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".pptx", ".docx", ".txt", ".md"})


def _normalise_text(text: str) -> str:
    """Clean extracted text without changing its meaning.

    PDF extractors in particular tend to emit NUL bytes, non-breaking spaces,
    and ragged line endings.  Normalising those characters makes downstream
    chunking deterministic.
    """

    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    # Trim trailing spaces on each line and collapse excessive blank lines.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: PathLike) -> List[Dict[str, Any]]:
    """Extract text page by page from a PDF file.

    Pages that contain no extractable text (for example scanned images) are
    skipped and a warning is logged.
    """

    from pypdf import PdfReader  # imported lazily so the package stays light

    pdf_path = Path(path)
    records: List[Dict[str, Any]] = []

    reader = PdfReader(str(pdf_path))
    if getattr(reader, "is_encrypted", False):
        # Many PDFs are encrypted with an empty user password.  Try that first;
        # if it fails, the caller still gets a clear exception below.
        try:
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover - depends on input PDF
            raise ValueError(f"Could not decrypt PDF: {pdf_path}") from exc

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = _normalise_text(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - malformed page
            LOGGER.warning(
                "Skipping unreadable page %d in %s: %s", page_number, pdf_path, exc
            )
            continue

        if not text:
            LOGGER.warning(
                "Skipping empty/scanned page %d in %s (no extractable text)",
                page_number,
                pdf_path,
            )
            continue

        records.append({"text": text, "location": {"page": page_number}})

    return records


def _shape_text(shape: Any) -> List[str]:
    """Return all text contained in a single PowerPoint shape."""

    parts: List[str] = []
    if getattr(shape, "has_text_frame", False):
        frame_text = shape.text_frame.text
        if frame_text and frame_text.strip():
            parts.append(frame_text)
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return parts


def extract_pptx(path: PathLike) -> List[Dict[str, Any]]:
    """Extract text slide by slide from a PPTX file, including speaker notes."""

    from pptx import Presentation  # lazy import

    pptx_path = Path(path)
    presentation = Presentation(str(pptx_path))
    records: List[Dict[str, Any]] = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        parts: List[str] = []
        for shape in slide.shapes:
            parts.extend(_shape_text(shape))

        notes_text = ""
        if getattr(slide, "has_notes_slide", False):
            notes_frame = slide.notes_slide.notes_text_frame
            notes_text = _normalise_text(notes_frame.text if notes_frame else "")
        if notes_text:
            parts.append(f"Notes: {notes_text}")

        text = _normalise_text("\n".join(parts))
        if not text:
            LOGGER.warning(
                "Skipping empty slide %d in %s (no extractable text)",
                slide_number,
                pptx_path,
            )
            continue

        records.append({"text": text, "location": {"slide": slide_number}})

    return records


def _paragraph_is_heading(paragraph: Any) -> bool:
    """Return ``True`` when a python-docx paragraph uses a heading style."""

    style = getattr(paragraph, "style", None)
    style_name = getattr(style, "name", "") or ""
    return style_name.lower().startswith(("heading", "title"))


def extract_docx(path: PathLike) -> List[Dict[str, Any]]:
    """Extract a DOCX file section by section.

    A new section starts at every heading/title paragraph.  Body text that
    appears before the first heading is placed in an ``Introduction`` section.
    Table cell text is appended to the section that precedes the table.
    """

    from docx import Document  # lazy import

    docx_path = Path(path)
    document = Document(str(docx_path))

    records: List[Dict[str, Any]] = []
    current_title = "Introduction"
    current_parts: List[str] = []

    def flush() -> None:
        nonlocal current_parts
        text = _normalise_text("\n".join(current_parts))
        if text:
            records.append({"text": text, "location": {"section": current_title}})
        else:
            LOGGER.warning(
                "Skipping empty section %r in %s", current_title, docx_path
            )
        current_parts = []

    for paragraph in document.paragraphs:
        paragraph_text = (paragraph.text or "").strip()
        if _paragraph_is_heading(paragraph):
            # Finish the previous section before starting a new one.
            flush()
            current_title = paragraph_text or "Untitled section"
            if paragraph_text:
                current_parts.append(paragraph_text)
        elif paragraph_text:
            current_parts.append(paragraph_text)

    flush()

    # Include table content.  Tables are appended to the most recent section
    # (or a standalone "Tables" section when the document has no body text).
    table_lines: List[str] = []
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                table_lines.append(" | ".join(cells))
    if table_lines:
        table_text = _normalise_text("\n".join(table_lines))
        if records:
            records[-1]["text"] = _normalise_text(records[-1]["text"] + "\n" + table_text)
        else:
            records.append(
                {"text": table_text, "location": {"section": "Tables"}}
            )

    return records


def extract_text_file(path: PathLike) -> List[Dict[str, Any]]:
    """Extract a plain-text or Markdown file as one section."""

    text_path = Path(path)
    try:
        text = text_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        LOGGER.warning("Falling back to latin-1 decoding for %s", text_path)
        text = text_path.read_text(encoding="latin-1", errors="replace")

    text = _normalise_text(text)
    if not text:
        LOGGER.warning("Skipping empty text file %s", text_path)
        return []

    return [{"text": text, "location": {"section": text_path.stem}}]


def extract_file(path: PathLike) -> List[Dict[str, Any]]:
    """Extract text records from ``path`` based on its extension.

    Parameters
    ----------
    path:
        Path to a PDF, PPTX, DOCX, TXT, or Markdown file.

    Returns
    -------
    list of dict
        Each dictionary has a ``text`` string and a ``location`` dictionary.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the extension is not supported.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a regular file: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(file_path)
    if suffix == ".pptx":
        return extract_pptx(file_path)
    if suffix == ".docx":
        return extract_docx(file_path)
    if suffix in {".txt", ".md"}:
        return extract_text_file(file_path)

    raise ValueError(
        f"Unsupported file type {suffix!r} for {file_path}. "
        f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )
