# Campus GPT (VTU-SLM-BOT): build via celure-chain (Hermes drives opencode)

## Context
You want the whole Campus GPT project built: a VTU/AIML syllabus chatbot (Qwen2.5-1.5B 4-bit + Gradio on Colab), answering from course material with RAG, a fine-tuning step run once at the end, and an IEEE-style LaTeX→PDF research paper. The screenshots show the material exists only on your phone (BCS502 Computer Networks textbook modules 1-5, BCS503 Mod1-5 notes, BAI701 Mod 4, BAI613A Mod 1-2, BEC654B Consumer Electronics notes/PPTs). Neither Hermes nor I can read those, so per your answer **Hermes fetches the official VTU syllabus by subject code and finalises the knowledge base**, and real textbook files can be dropped in later.

The current worktree is a toy repo (`app.py` with `add()`, `test_app.py`, pytest CI, `.github/workflows/ci.yml`), so this is greenfield. Everything goes under `campus-gpt/`, reusing the existing CI workflow (extend, don't replace).

Facts that shape the plan (checked read-only): bridge on celureai is reachable and the queue is empty; MiKTeX `pdflatex` is on this PC (so I compile the paper locally); Hermes' sandbox is 2 CPU / 3 GB, no GPU, one task at a time, ≤60 turns / ≤3600 s each, free-tier models only. So Hermes builds and unit-tests everything on CPU; **model loading, training and GPU evaluation run by you on Colab**.

## Target layout (`campus-gpt/`)
- `kb/subjects.yaml`: per subject: code, title, scheme, semester, credits, CIE/SEE marks, modules[] (title + topics), textbooks, `source_url`, `fetched_on`, `verified: true|false`
- `kb/rules.yaml`: marks/CIE/SEE/aggregate, CBCS grade table, CGPA→% formula, attendance + condonation; each with source URL and `verified`
- `ingest/`: PDF/PPTX/DOCX → text (pypdf, python-pptx, python-docx) → chunks with `{file, page|slide, subject, module}` → `data/chunks.jsonl`
- `rag/`: BM25 (pure Python, CPU-testable default) + optional bge-small embeddings behind a flag; top-k retrieval; subject/module/semester filters
- `bot/`: query router (rules fact-table → exact answer, no LLM; syllabus/module → structured KB; free text → RAG), prompt builder, citation formatting, "not found" refusal, `respond()` with proper dict-history handling, Gradio app, single Colab cell (`colab/run_bot.py`)
- `train/`: synthetic QA/instruction dataset builder from KB+chunks (with filtering/dedup), QLoRA script + Colab notebook (Qwen2.5-1.5B-Instruct, 4-bit NF4), merge/export
- `eval/`: golden Q&A set (rules, subject codes, module topics, out-of-scope), metrics: retrieval hit@k, fact exact-match, citation correctness, refusal rate; runner for pre/post fine-tune
- `tests/`: pytest, CPU-only, no network
- `paper/`: `main.tex` (IEEEtran conference), `refs.bib`, figures (architecture, results) generated from `eval/` output
- `.gitignore`: `data/raw/` and model weights (see copyright note)

## Execution order: tasks queued to Hermes as lane `chain`
Each task prompt is a self-contained file (goal, constraints, definition of done, the standards below) sent with `forward.ps1 -Agent chain` (no `-Wait`; check with `bridge wait`). One at a time. Each must `mem_save` its key decisions so the next task inherits them.

1. **T1 Syllabus + rules fetch.** Using the web toolset, fetch the official VTU 2022-scheme PDFs (known: `vtu.ac.in/pdf/2022_3to8/aimlsch.pdf`, `58aimlsch.pdf`, regulations PDF) and the per-course syllabus for the codes seen in the screenshots: BCS502, BCS503, BAI701, BAI613A, BEC654B, plus the AIML Sem 3-7 course list. Produce `subjects.yaml` and `rules.yaml`. Anything it cannot confirm from a fetched source is `verified: false`, never guessed. It must confirm CIE 20/50 vs SEE 18/50, attendance 85% and condonation, and the 18/20 discrepancy from your slide 4, and report conflicts.
2. **T2 Ingestion + RAG core** with unit tests on tiny synthetic PDF/PPTX/DOCX fixtures it generates itself.
3. **T3 Bot** (router, prompt, citations, refusal, Gradio, Colab cell). Replaces the DistilBERT path; fixes the earlier bugs (history dict format, subject-code regexes `22ai\w*`, `ai and ml`, semester coverage warnings for uncovered semesters, no `temperature` with `do_sample=False`).
4. **T4 Eval set + harness + CI extension** (pytest + lint on every push).
5. **T5 Training pipeline** (dataset builder + QLoRA script/notebook). Only authored and dry-run on a 10-example sample in the sandbox; actual training is yours, once, at the end.
6. **T6 Paper skeleton** (IEEEtran): Introduction, Related Work, System Design, Knowledge Base & RAG, Fine-tuning, Evaluation, Limitations/Ethics, Conclusion. Result tables are explicitly marked `TBD` until real numbers exist.

After each task I do not trust the summary: I pull `~/hermes-work/<id>/` over SSH, review the files, and run the tests myself on this PC.

## Standards given to Hermes (go verbatim in every prompt)
- **No fabricated facts or numbers.** Every VTU fact carries a source URL + fetch date; unverifiable → `verified: false` and surfaced in the report. No invented results, citations, or benchmarks in code or paper.
- **Independent verification is mandatory** (chain rule): run the tests/code it wrote, open created files, report the exact commands and outputs, state what was wrong and fixed.
- Python 3.10+, type hints, docstrings on public functions, pinned versions in `requirements.txt`/`requirements-colab.txt`, no network or GPU needed for the test suite, no secrets anywhere.
- Deterministic where possible (fixed seeds, `do_sample=False` for fact answers); pytest must pass with zero failures; deliver a README per package with how to run.
- Small, readable modules; reuse libraries rather than hand-rolling (pypdf, python-pptx, python-docx, rank-bm25, gradio, peft/trl for training).
- Each task ends with a report: files created, commands run, test output, open issues, what the next task needs.
- Never touch the server's other stacks (Nextcloud, Label Studio, CityShield); stay inside the task folder.

## Copyright and honesty notes
- The BCS502 textbook PDFs and PPT/notes are copyrighted/third-party. Raw files stay out of git; only derived chunks stay local and the repo ships with the ingestion code. The paper cites sources and states this.
- Your slides claim "offline", "deterministic and factual" and "syllabus-accurate"; the paper must use the accurate wording (answers from supplied rules/documents, Colab share link isn't offline).
- Fine-tuning on synthetic data generated by free LLMs: the paper will report it as such, with the eval set kept separate from training data (no leakage).

## Finalisation (me, not Hermes)
1. Pull all artifacts from `~/hermes-work/<ids>/` to `campus-gpt/`, review, run `pytest`, fix gaps.
2. You run the Colab notebook once: ingest → RAG eval (pre) → QLoRA train → eval (post). Results JSON comes back to me.
3. I fill the paper tables/figures from that JSON only, compile with MiKTeX (`pdflatex`+`bibtex`, IEEEtran), and fix LaTeX errors.
4. Commit to branch `claude/celure-skills-setup-4dde02` and leave the PDF in `campus-gpt/paper/main.pdf`.

## Verification
- Per task: `bridge status/result`, `scp` the folder, `pytest -v` on this PC, spot-check 5 `subjects.yaml` entries against the cited PDF URLs.
- End-to-end before Colab: ingest the synthetic fixtures → ask 10 golden questions through `respond()` with a stub LLM → all rule facts exact, out-of-scope questions refused with a citation-free "not found".
- Colab: bot answers "CIE minimum?", "BAI701 module 4?", a BCS502 question with citation, and an out-of-scope question correctly.
- Paper: `pdflatex` builds with no errors, no `TBD` left, every number traceable to `eval/results.json`.

## Risks
- Free-tier model quota/402 benching and serial queue make six tasks slow (hours); I will split any task that hits the 3600 s budget.
- VTU PDFs may be unreachable or restructured; those entries stay `verified: false` and I ask you rather than guess.
- Without the real textbooks the RAG demo runs on syllabus + synthetic samples; quality claims in the paper wait for the real documents.
