"""Build a chunk corpus from a directory of raw documents.

Usage::

    python -m campus_gpt.ingest.build --input data/raw --out data/chunks.jsonl

The command walks ``--input`` recursively, extracts every supported document,
chunks it, and writes one JSON object per line to ``--out``.  It is safe to run
repeatedly; the output file is replaced each time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import SUPPORTED_EXTENSIONS, extract_file

__all__ = ["build_chunks", "main"]

LOGGER = logging.getLogger(__name__)


def _iter_supported_files(input_dir: Path) -> List[Path]:
    """Return supported files below ``input_dir`` in deterministic order."""

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_chunks(
    input_dir: str | Path,
    output_path: str | Path,
    chunk_size: int = 800,
    overlap: int = 120,
) -> int:
    """Extract and chunk all supported files in ``input_dir``.

    Returns the number of chunks written.
    """

    from campus_gpt.chunking import chunk_documents

    source = Path(input_dir)
    destination = Path(output_path)

    if not source.exists():
        raise FileNotFoundError(f"Input directory does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"Input path is not a directory: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[dict] = []
    files = _iter_supported_files(source)
    if not files:
        LOGGER.warning("No supported documents found in %s", source)

    for file_path in files:
        try:
            docs = extract_file(file_path)
        except Exception:
            LOGGER.exception("Failed to extract %s; skipping", file_path)
            continue

        # Attach the source path so filename-based metadata inference works.
        for doc in docs:
            doc["file"] = str(file_path)

        chunks = chunk_documents(
            docs, overlap=overlap, chunk_size=chunk_size
        )
        all_chunks.extend(chunks)
        LOGGER.info("Extracted %d chunk(s) from %s", len(chunks), file_path)

    with destination.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    LOGGER.info("Wrote %d chunk(s) to %s", len(all_chunks), destination)
    return len(all_chunks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point for building the chunk corpus."""

    parser = argparse.ArgumentParser(
        description="Extract campus documents and write a JSONL chunk corpus."
    )
    parser.add_argument(
        "--input",
        default="data/raw",
        help="Directory containing raw PDF/PPTX/DOCX/TXT/MD files "
        "(default: data/raw)",
    )
    parser.add_argument(
        "--out",
        default="data/chunks.jsonl",
        help="Destination JSONL file (default: data/chunks.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Target chunk size in characters (default: 800)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Overlap between adjacent chunks in characters (default: 120)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable informational logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        count = build_chunks(
            input_dir=args.input,
            output_path=args.out,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {count} chunk(s) to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    sys.exit(main())
