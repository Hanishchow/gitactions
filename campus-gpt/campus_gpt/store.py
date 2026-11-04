"""SQLite (FTS5) knowledge database: rules, courses and document chunks.

``SqliteRetriever`` offers the same ``retrieve()`` API as the BM25 ``Retriever``,
so :class:`campus_gpt.bot.CampusBot` works with either.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .kb import load_rules, load_subjects

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules(id TEXT PRIMARY KEY, keywords TEXT, answer TEXT, verified INTEGER);
CREATE TABLE IF NOT EXISTS courses(code TEXT, semester INTEGER, title TEXT, category TEXT,
    credits INTEGER, hours TEXT, cie INTEGER, see INTEGER, options TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, file TEXT, location TEXT,
    subject TEXT, module INTEGER, text TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id',
    tokenize='porter unicode61');
"""

_TOKEN = re.compile(r"[A-Za-z0-9]{2,}")
_STOP = {"the", "is", "are", "of", "and", "to", "in", "a", "an", "for", "what", "how", "does",
         "do", "about", "tell", "me", "explain", "with", "on", "at", "it", "this", "that", "from"}


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the database and ensure the schema exists."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load_kb(conn: sqlite3.Connection) -> None:
    """(Re)load rules and courses from the verified YAML files."""
    conn.execute("DELETE FROM rules")
    conn.execute("DELETE FROM courses")
    for r in load_rules():
        conn.execute("INSERT INTO rules VALUES (?,?,?,?)",
                     (r["id"], json.dumps(r["keywords"]), r["answer"], int(bool(r.get("verified")))))
    for sem, data in load_subjects().items():
        for c in data["courses"]:
            conn.execute("INSERT INTO courses VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (c["code"], sem, c["title"], c.get("category"), c.get("credits"), c.get("hours"),
                          c.get("cie"), c.get("see"), json.dumps(c.get("options", [])), c.get("note")))
    conn.commit()


def load_chunks(conn: sqlite3.Connection, jsonl: str | Path, replace: bool = True) -> int:
    """Load ingested chunks (``data/chunks.jsonl``) and rebuild the FTS index."""
    if replace:
        conn.execute("DELETE FROM chunks")
    n = 0
    for line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        conn.execute("INSERT INTO chunks(file, location, subject, module, text) VALUES (?,?,?,?,?)",
                     (c.get("file"), json.dumps(c.get("location") or {}), c.get("subject"),
                      c.get("module"), c.get("text", "")))
        n += 1
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()
    return n


def _fts_query(question: str) -> Optional[str]:
    terms = [t for t in _TOKEN.findall(question.lower()) if t not in _STOP]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class SqliteRetriever:
    """BM25 over the ``chunks`` table using SQLite FTS5."""

    def __init__(self, db_path: str | Path, min_hits: int = 1) -> None:
        self.db_path = str(db_path)
        self.min_hits = min_hits
        connect(self.db_path).close()  # fail fast / ensure schema

    def retrieve(self, query: str, k: int = 5, subject: Optional[str] = None,
                 module: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return up to ``k`` ``{"chunk", "score"}`` dicts, best first; ``[]`` if nothing matches."""
        match = _fts_query(query)
        if match is None or k <= 0:
            return []
        sql = ("SELECT c.file, c.location, c.subject, c.module, c.text, bm25(chunks_fts) AS r "
               "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ?")
        args: list = [match]
        if subject:
            sql += " AND upper(c.subject) = ?"
            args.append(subject.upper())
        if module is not None:
            sql += " AND c.module = ?"
            args.append(int(module))
        sql += " ORDER BY r LIMIT ?"
        args.append(k)
        conn = connect(self.db_path)  # per call: SQLite connections are thread-bound
        try:
            rows = conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()
        return [{"chunk": {"file": r["file"], "location": json.loads(r["location"] or "{}"),
                           "subject": r["subject"], "module": r["module"], "text": r["text"]},
                 "score": -float(r["r"])} for r in rows]
