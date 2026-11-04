"""Build the grounded-answer fine-tuning set (chat format) with a leak-free held-out split.

Each example is what the bot sends the LLM at inference time: a system prompt with
numbered context passages (one relevant + distractors) and a question. The target is
either an answer that cites the relevant passage or the fixed refusal sentence when
no passage supports the question. Passages come from the verified rules/subjects KB
and, if ``data/chunks.jsonl`` exists, from ingested course chunks (answered
extractively so no invented text enters the targets).

Held-out split: whole question *templates* and whole passages are reserved before
any training example is written, so eval questions/passages never appear in training.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from campus_gpt.bot import SYSTEM_PROMPT
from campus_gpt.kb import load_rules, load_subjects

REFUSE = "I don't have that in my knowledge base."

RULE_TEMPLATES = [
    "{k}?", "Tell me about {k}.", "What is the rule for {k}?", "Explain {k} as per VTU.",
    "I want to know about {k}.", "VTU {k} details please.",
]
COURSE_TEMPLATES = [
    "What is {code}?", "Tell me about {code}.", "{code} course details", "Which course is {code} and how many credits?",
    "Explain {code}.", "{title} - semester and credits?",
]
OUT_OF_SCOPE = [
    "Who won the cricket world cup?", "Write a poem about rain.", "What is the capital of France?",
    "How do I cook biryani?", "Explain quantum entanglement.", "What is the stock price of Tesla?",
    "Which hostel has the best food?", "Translate hello to Japanese.", "What is the weather today?",
    "Who is the principal of another college?", "Tell me a joke.", "How do I fix my bike?",
]


def _passages(rules, subjects) -> List[Dict[str, str]]:
    out = [{"id": f"rule:{r['id']}", "text": r["answer"], "kw": r["keywords"]} for r in rules]
    for sem, data in subjects.items():
        for c in data["courses"]:
            text = f"{c['code']} {c['title']}: Semester {sem}, category {c.get('category', '?')}"
            if c.get("credits") is not None:
                text += f", {c['credits']} credits"
            if c.get("options"):
                text += ". Options: " + "; ".join(c["options"])
            out.append({"id": f"course:{c['code']}", "text": text, "code": c["code"], "title": c["title"]})
    return out


def _messages(context: List[str], question: str, answer: str) -> Dict[str, Any]:
    ctx = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(context, 1))
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nContext:\n" + ctx},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def build(seed: int = 7, heldout_frac: float = 0.25, chunks_path: str | None = None):
    rng = random.Random(seed)
    passages = _passages(load_rules(), load_subjects())
    ids = sorted(p["id"] for p in passages)
    rng.shuffle(ids)
    held_ids = set(ids[: max(1, int(len(ids) * heldout_frac))])
    split = {"train": [], "heldout": []}

    def distractors(pid: str, pool: List[Dict[str, str]], n: int) -> List[str]:
        others = [p["text"] for p in pool if p["id"] != pid]
        rng.shuffle(others)
        return others[:n]

    for name, pool in (("train", [p for p in passages if p["id"] not in held_ids]),
                       ("heldout", [p for p in passages if p["id"] in held_ids])):
        templates_rule = RULE_TEMPLATES[:4] if name == "train" else RULE_TEMPLATES[4:]
        templates_course = COURSE_TEMPLATES[:4] if name == "train" else COURSE_TEMPLATES[4:]
        for p in pool:
            if p["id"].startswith("rule:"):
                qs = [t.format(k=k) for t in templates_rule for k in p["kw"][:2]]
            else:
                qs = [t.format(code=p["code"], title=p["title"]) for t in templates_course]
            for q in qs:
                ctx = distractors(p["id"], pool, 2) + [p["text"]]
                rng.shuffle(ctx)
                pos = ctx.index(p["text"]) + 1
                split[name].append(_messages(ctx, q, f"{p['text']} [{pos}]"))
        for q in OUT_OF_SCOPE[: 8 if name == "train" else 4] if name == "train" else OUT_OF_SCOPE[8:]:
            ctx = [p["text"] for p in rng.sample(pool, min(3, len(pool)))]
            split[name].append(_messages(ctx, q, REFUSE))

    if chunks_path and Path(chunks_path).exists():
        chunks = [json.loads(line) for line in Path(chunks_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        rng.shuffle(chunks)
        cut = int(len(chunks) * (1 - heldout_frac))
        for name, part in (("train", chunks[:cut]), ("heldout", chunks[cut:])):
            for ch in part:
                first = ch["text"].split(". ")[0].strip()
                if len(first) < 30:
                    continue
                others = [c["text"] for c in part if c is not ch][:2]
                ctx = others + [ch["text"]]
                rng.shuffle(ctx)
                pos = ctx.index(ch["text"]) + 1
                split[name].append(_messages(ctx, f"Explain: {first[:80]}", f"{ch['text']} [{pos}]"))
    for v in split.values():
        rng.shuffle(v)
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    split = build(a.seed, chunks_path=a.chunks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in split.items():
        with open(out / f"sft_{name}.jsonl", "w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples -> {out / f'sft_{name}.jsonl'}")


if __name__ == "__main__":
    main()
