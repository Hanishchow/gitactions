# CampusGPT

CampusGPT is a small, fully local retrieval-augmented generation (RAG) toolkit
for campus documents. It:

1. **Ingests** PDF, PPTX, DOCX, TXT, and Markdown files.
2. **Chunks** each document into overlapping, boundary-aware segments.
3. **Retrieves** the most relevant chunks with BM25 (`rank_bm25`).

No hosted LLM or network connection is required. All examples use synthetic
identifiers such as `TST101`; no real university data is included.

## Project layout

```text
campus-gpt/
├── campus_gpt/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py      # extract_file(), format-specific extractors
│   │   └── build.py         # JSONL corpus builder / CLI
│   ├── chunking/
│   │   └── __init__.py      # chunk_documents(), infer_subject_module()
│   └── rag/
│       └── __init__.py      # Retriever (BM25 + optional embeddings)
├── tests/
│   ├── test_ingest.py
│   ├── test_chunking.py
│   └── test_retriever.py
├── demo_query.py
├── requirements.txt
└── README.md
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The project can also be installed in editable mode:

```bash
pip install -e .
```

## Build a chunk corpus

Put your documents under `data/raw/` (or any directory) and run:

```bash
python -m campus_gpt.ingest.build --input data/raw --out data/chunks.jsonl
```

Each supported file is extracted, chunked, and written as one JSON object per
line. Useful options:

```bash
python -m campus_gpt.ingest.build \
  --input data/raw \
  --out data/chunks.jsonl \
  --chunk-size 800 \
  --overlap 120 \
  --verbose
```

Supported extensions are `.pdf`, `.pptx`, `.docx`, `.txt`, and `.md`.
Scanned/empty PDF pages and empty slides are skipped with a logged warning.

## Use the retriever

```python
from campus_gpt.rag import Retriever

retriever = Retriever("data/chunks.jsonl")
results = retriever.retrieve("library opening hours", k=5)

for result in results:
    chunk = result["chunk"]
    print(result["score"], chunk["file"], chunk["location"])
    print(chunk["text"])
```

Filter by subject code and/or module number before ranking:

```python
results = retriever.retrieve(
    "exam guidelines",
    k=3,
    subject="TST101",
    module=2,
)
```

The default relevance threshold is `0.1`. If the best BM25 score is below it,
`retrieve()` returns an empty list. Adjust it when needed:

```python
retriever.min_score = 0.0
```

For an optional semantic fallback, install `sentence-transformers` separately
and construct the retriever with `use_embeddings=True`. The dependency is
imported lazily; when it is unavailable, CampusGPT simply continues with BM25.

## Demo

```bash
python -m campus_gpt.ingest.build --input data/raw --out data/chunks.jsonl
python demo_query.py
```

The demo prints the top three results for two sample queries. Use
`--chunks path/to/chunks.jsonl` to point at a different corpus.

## Metadata inference

Subject codes and module numbers are inferred from filenames:

| Filename | Subject | Module |
| --- | --- | --- |
| `TST101-module-2-notes.pdf` | `TST101` | `2` |
| `TST102_Module_1_slides.pptx` | `TST102` | `1` |
| `Module-3_Notes.md` | `None` | `3` |
| `random-notes.pdf` | `None` | `None` |

## Tests

```bash
pytest -q
```

The tests generate all fixtures at runtime with `reportlab`, `python-pptx`, and
`python-docx`; no external documents are needed.

## Server deployment (celureai)
`server/serve.py` runs Gradio on 127.0.0.1:7860 over a SQLite FTS5 knowledge DB (`server/build_db.py`), answering via
Ollama Cloud (`campus_gpt/ollama.py`, key from `~/.config/campus-gpt.env`). A cloudflared quick tunnel exposes it.
Add documents: copy files to `data/raw/` (names like `BCS502-module-3-textbook.pdf`), then
`python -m campus_gpt.ingest.build --input data/raw --out data/chunks.jsonl && python -m server.build_db`.
