"""Tests for document extraction using synthetic documents only."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from docx import Document
from pptx import Presentation
from pptx.util import Inches
from reportlab.pdfgen import canvas

from campus_gpt.ingest import extract_file


def _write_pdf(path: Path, text: str) -> None:
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, text)
    pdf.showPage()
    pdf.save()


def _write_empty_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path))
    pdf.showPage()
    pdf.save()


def _write_pptx(path: Path) -> None:
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "TST101 Module 1 Overview"
    textbox = slide.shapes.add_textbox(
        Inches(1), Inches(2), Inches(6), Inches(2)
    )
    textbox.text_frame.text = "Campus safety and emergency contacts."
    slide.notes_slide.notes_text_frame.text = "Remind students about the help desk."
    presentation.save(str(path))


def _write_docx(path: Path) -> None:
    document = Document()
    document.add_heading("Campus Services", level=1)
    document.add_paragraph("The library is open from nine to five.")
    document.add_heading("Library Rules", level=2)
    document.add_paragraph("Silent study rooms are available on the second floor.")
    document.save(str(path))


def test_extract_pdf_returns_page_location(tmp_path: Path) -> None:
    pdf_path = tmp_path / "TST101-module-1-notes.pdf"
    _write_pdf(pdf_path, "TST101 campus library and student services.")

    docs = extract_file(pdf_path)

    assert len(docs) == 1
    assert docs[0]["location"] == {"page": 1}
    assert "library" in docs[0]["text"].lower()
    assert set(docs[0]) == {"text", "location"}


def test_extract_pptx_includes_slide_text_and_notes(tmp_path: Path) -> None:
    pptx_path = tmp_path / "TST101-module-1-slides.pptx"
    _write_pptx(pptx_path)

    docs = extract_file(pptx_path)

    assert len(docs) == 1
    assert docs[0]["location"] == {"slide": 1}
    assert "Campus safety" in docs[0]["text"]
    assert "help desk" in docs[0]["text"].lower()


def test_extract_docx_splits_on_headings(tmp_path: Path) -> None:
    docx_path = tmp_path / "TST101-module-2-handout.docx"
    _write_docx(docx_path)

    docs = extract_file(docx_path)

    assert len(docs) == 2
    locations = [doc["location"] for doc in docs]
    assert {"section": "Campus Services"} in locations
    assert {"section": "Library Rules"} in locations

    by_section = {doc["location"]["section"]: doc["text"] for doc in docs}
    assert "nine to five" in by_section["Campus Services"]
    assert "Silent study rooms" in by_section["Library Rules"]


def test_extract_text_and_markdown_files(tmp_path: Path) -> None:
    txt_path = tmp_path / "TST101-module-3-notes.txt"
    txt_path.write_text("Office hours are posted on the department noticeboard.", encoding="utf-8")
    md_path = tmp_path / "Module-3_Notes.md"
    md_path.write_text("# Module 3\n\nExam revision tips.", encoding="utf-8")

    txt_docs = extract_file(txt_path)
    md_docs = extract_file(md_path)

    assert txt_docs[0]["location"] == {"section": "TST101-module-3-notes"}
    assert "Office hours" in txt_docs[0]["text"]
    assert md_docs[0]["location"] == {"section": "Module-3_Notes"}
    assert "revision tips" in md_docs[0]["text"]


def test_empty_pdf_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = tmp_path / "TST101-empty.pdf"
    _write_empty_pdf(pdf_path)

    with caplog.at_level(logging.WARNING, logger="campus_gpt.ingest"):
        docs = extract_file(pdf_path)

    assert docs == []
    assert any("empty" in record.message.lower() for record in caplog.records)


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "TST101-data.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_file(csv_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_file(tmp_path / "does-not-exist.pdf")
