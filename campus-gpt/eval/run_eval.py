"""Evaluate the bot on the held-out set and a golden question list; writes eval/results.json.

    python -m eval.run_eval                 # router only (CPU)
    python -m eval.run_eval --adapter adapters/campus-gpt   # with the LLM (Colab GPU)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from campus_gpt.bot import REFUSAL, CampusBot

GOLDEN = [  # (question, substring that must appear, or None for refusal)
    ("minimum CIE marks", "20 out of 50"),
    ("what is the minimum SEE pass mark", "35%"),
    ("passing marks per course", "40 out of 100"),
    ("minimum attendance required", "85%"),
    ("attendance condonation limit", "10%"),
    ("grade for 85 percent marks", "80-89 A+ (9)"),
    ("convert 7.5 cgpa to percentage", "75.0%"),
    ("AIML 7th semester subjects", "BAI702"),
    ("8th sem courses", "BAI803"),
    ("what is BAI703", "Data Security"),
    ("BCS503", "Theory of Computation"),
    ("BAI613A", "Human-Centred AI"),
    ("6th sem subjects", "BAI601"),
    ("who won the world cup", None),
    ("write me a poem", None),
]


def run(bot: CampusBot, heldout: str | None) -> dict:
    gold_ok = []
    for q, want in GOLDEN:
        out = bot.answer(q)
        ok = (out == REFUSAL) if want is None else (want in out)
        gold_ok.append({"q": q, "ok": ok})
    res = {
        "golden_total": len(gold_ok),
        "golden_correct": sum(g["ok"] for g in gold_ok),
        "golden_failures": [g["q"] for g in gold_ok if not g["ok"]],
    }
    if heldout and Path(heldout).exists() and bot.generate is not None:
        rows = [json.loads(x) for x in open(heldout, encoding="utf-8") if x.strip()]
        refuse = [r for r in rows if "I don't have that in my knowledge base" in r["messages"][-1]["content"]]
        answer = [r for r in rows if r not in refuse]

        def ask(r):
            return bot.generate(r["messages"][:-1]).strip()

        res["heldout_answerable"] = len(answer)
        res["heldout_answerable_correct"] = sum(
            r["messages"][-1]["content"].split(" [")[0][:60].lower() in ask(r).lower() for r in answer)
        res["heldout_refusal_total"] = len(refuse)
        res["heldout_refusal_correct"] = sum("don't have that" in ask(r).lower() for r in refuse)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--heldout", default="data/sft_heldout.jsonl")
    ap.add_argument("--out", default="eval/results.json")
    a = ap.parse_args()
    gen = None
    if a.adapter:
        from campus_gpt.llm import load_generator
        gen = load_generator(a.adapter)
    res = run(CampusBot(generate=gen), a.heldout)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
