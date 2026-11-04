"""Build/refresh the knowledge database: ``python -m server.build_db --db data/campus.db [--chunks data/chunks.jsonl]``."""
from __future__ import annotations

import argparse
from pathlib import Path

from campus_gpt.store import connect, load_chunks, load_kb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/campus.db")
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    a = ap.parse_args()
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(a.db)
    load_kb(conn)
    n = load_chunks(conn, a.chunks) if Path(a.chunks).exists() else 0
    rules = conn.execute("SELECT count(*) FROM rules").fetchone()[0]
    courses = conn.execute("SELECT count(*) FROM courses").fetchone()[0]
    print(f"{a.db}: {rules} rules, {courses} courses, {n} chunks")


if __name__ == "__main__":
    main()
