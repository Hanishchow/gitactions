"""Run a small CampusGPT retrieval demo.

Build a corpus first::

    python -m campus_gpt.ingest.build --input data/raw --out data/chunks.jsonl
    python demo_query.py

The demo uses synthetic identifiers and requires no network access.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from campus_gpt.rag import Retriever

DEFAULT_CHUNKS = Path("data/chunks.jsonl")

SAMPLE_QUERIES = [
    "What are the library opening hours?",
    "TST101 module 2 exam guidelines",
]


def _print_results(query: str, results: List[dict]) -> None:
    print(f"\nQuery: {query}")
    if not results:
        print("  (no results above the relevance threshold)")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        location = chunk.get("location") or {}
        where = ", ".join(f"{key}={value}" for key, value in location.items())
        subject = chunk.get("subject") or "n/a"
        module = chunk.get("module")
        module_text = "n/a" if module is None else str(module)
        preview = " ".join(str(chunk.get("text", "")).split())
        if len(preview) > 220:
            preview = preview[:217] + "..."
        print(f"  {rank}. score={result['score']:.3f} [{subject} module {module_text}]")
        print(f"     file={chunk.get('file', 'unknown')} ({where})")
        print(f"     {preview}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CampusGPT retrieval demo")
    parser.add_argument(
        "--chunks",
        default=str(DEFAULT_CHUNKS),
        help="Path to the JSONL chunk corpus (default: data/chunks.jsonl)",
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of results to show (default: 3)"
    )
    args = parser.parse_args(argv)

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"Chunk corpus not found: {chunks_path}")
        print(
            "Build it first with:\n"
            "  python -m campus_gpt.ingest.build --input data/raw "
            "--out data/chunks.jsonl"
        )
        return 1

    retriever = Retriever(str(chunks_path))
    for query in SAMPLE_QUERIES:
        _print_results(query, retriever.retrieve(query, k=args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
