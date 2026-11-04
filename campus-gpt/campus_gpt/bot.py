"""Query router: subject table -> rule table -> BM25 retrieval -> (optional) LLM.

The LLM is injected as ``generate(messages) -> str`` so the router is testable
on CPU with a stub and the same code runs on Colab with a real model.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .kb import load_rules, load_subjects

Generate = Callable[[List[Dict[str, str]]], str]

REFUSAL = (
    "I don't have that in my knowledge base. Please check the official VTU portal "
    "(https://vtu.ac.in/b-e-scheme-syllabus/) or ask your faculty."
)

SYSTEM_PROMPT = (
    "You are Campus GPT, an assistant for VTU AI & ML students. Answer ONLY from the "
    "numbered context passages below. Cite the passage you use like [1]. If the context "
    "does not contain the answer, reply exactly: I don't have that in my knowledge base."
)

_CODE_RE = re.compile(r"\b(B[A-Z]{2,4}L?\d{3}[A-Za-z]?)\b", re.I)
_SEM_RE = re.compile(r"\b(?:(\d)\s*(?:st|nd|rd|th)?\s*sem(?:ester)?|sem(?:ester)?\s*(\d))\b", re.I)
_CGPA_NUM_RE = re.compile(
    r"\b(\d{1,2}(?:\.\d{1,2})?)\s*cgpa\b|\bcgpa\s*(?:of|is|=)?\s*(\d{1,2}(?:\.\d{1,2})?)\b", re.I
)


def _kw_match(text: str, keyword: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)", text) is not None


class CampusBot:
    """Answer a student question from verified sources, or refuse."""

    def __init__(
        self,
        retriever: Any = None,
        generate: Optional[Generate] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        subjects: Optional[Dict[int, Dict[str, Any]]] = None,
        k: int = 4,
    ) -> None:
        self.rules = rules if rules is not None else load_rules()
        self.subjects = subjects if subjects is not None else load_subjects()
        self.retriever = retriever
        self.generate = generate
        self.k = k

    # -- rules -----------------------------------------------------------
    def _rule_answers(self, text: str) -> List[str]:
        m = _CGPA_NUM_RE.search(text)
        if m and re.search(r"percent|%|convert", text, re.I):
            cgpa = float(m.group(1) or m.group(2))
            if 0 <= cgpa <= 10:
                return [
                    f"CGPA {cgpa:g} = {cgpa * 10:.1f}% "
                    "(percentage = CGPA x 10, VTU 2022 regulations 22OB 6.7)."
                ]
        low = text.lower()
        hits = [r for r in self.rules if any(_kw_match(low, k) for k in r["keywords"])]

        def score(r: Dict[str, Any]) -> int:
            return sum(len(k) for k in r["keywords"] if _kw_match(low, k))

        hits.sort(key=score, reverse=True)
        return [h["answer"] for h in hits[:2]]

    # -- subjects --------------------------------------------------------
    def _all_courses(self):
        for sem, data in self.subjects.items():
            for c in data["courses"]:
                yield sem, c

    @staticmethod
    def _fmt_course(sem: int, c: Dict[str, Any]) -> str:
        head = f"{c['code']} {c['title']} (Sem {sem}, {c.get('category', '?')}"
        if c.get("credits") is not None:
            head += f", {c['credits']} credits"
        if c.get("hours"):
            head += f", L:T:P {c['hours']}"
        if c.get("cie") is not None:
            head += f", CIE {c['cie']}/SEE {c['see']}"
        line = head + ")"
        if c.get("options"):
            line += "\n    Options: " + "; ".join(c["options"])
        if c.get("note"):
            line += f"\n    Note: {c['note']}"
        return line

    def _subject_answer(self, text: str, history_text: str = "") -> Optional[str]:
        codes = {m.upper() for m in _CODE_RE.findall(text)}
        if codes:
            lines = []
            for sem, c in self._all_courses():
                base = c["code"].upper()
                opt_codes = {o.split()[0].upper() for o in c.get("options", [])}
                family = {cd[:-1] + "X" for cd in codes if cd[-1] in "ABCDE"}
                if base in codes or base in family or codes & opt_codes:
                    lines.append(self._fmt_course(sem, c))
            if lines:
                return "\n".join(lines) + (
                    "\nModule-wise topics are not in my knowledge base yet; add the course "
                    "textbook or notes to get topic answers."
                )
        sem_match = _SEM_RE.search(text)
        if sem_match is None and history_text and re.search(r"\b(subjects?|courses?|credits?)\b", text, re.I):
            sem_match = _SEM_RE.search(history_text)
        if sem_match and re.search(r"subject|course|code|credit|syllabus|scheme|sem", text, re.I):
            sem = int(sem_match.group(1) or sem_match.group(2))
            if sem in self.subjects:
                data = self.subjects[sem]
                body = "\n".join(self._fmt_course(sem, c) for c in data["courses"])
                return f"AIML 2022 scheme, Semester {sem}:\n{body}\n({data.get('note', '')})".strip()
            return (
                f"I don't have Semester {sem} in my knowledge base "
                f"(covered: {sorted(self.subjects)})."
            )
        return None

    # -- retrieval + generation -----------------------------------------
    def _rag(self, text: str) -> str:
        if self.retriever is None:
            return REFUSAL
        sub = _CODE_RE.search(text)
        results = self.retriever.retrieve(text, k=self.k, subject=sub.group(1).upper() if sub else None)
        if not results and sub:
            results = self.retriever.retrieve(text, k=self.k)
        if not results:
            return REFUSAL
        cites = []
        for i, r in enumerate(results, 1):
            ch = r["chunk"]
            loc = ch.get("location") or {}
            where = ", ".join(f"{k} {v}" for k, v in loc.items()) if isinstance(loc, dict) else str(loc)
            cites.append((i, Path(str(ch.get("file", "?"))).name, where, ch.get("text", "")))
        if self.generate is None:
            return "\n\n".join(f"[{i}] ({f}, {w})\n{t}" for i, f, w, t in cites)
        context = "\n\n".join(f"[{i}] {t}" for i, _f, _w, t in cites)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nContext:\n" + context},
            {"role": "user", "content": text},
        ]
        try:
            answer = self.generate(messages).strip()
        except Exception:  # model API down/rate-limited: still show the sourced passages
            body = "\n\n".join(f"[{i}] ({f}, {w})\n{t}" for i, f, w, t in cites)
            return "The language model is unavailable right now. Relevant passages:\n\n" + body
        if "don't have that in my knowledge base" in answer.lower():
            return REFUSAL
        refs = "\n".join(f"[{i}] {f} ({w})" for i, f, w, _t in cites)
        return f"{answer}\n\nSources:\n{refs}"

    # -- public API ------------------------------------------------------
    def answer(self, question: str, history: Optional[Sequence[Any]] = None) -> str:
        """Answer ``question``; ``history`` is Gradio's list of message dicts."""
        text = (question or "").strip()
        if not text:
            return (
                "Ask me about VTU rules (marks, attendance, grades, CGPA), "
                "AIML semester courses, or your course material."
            )
        prior = " ".join(
            m["content"]
            for m in (history or [])
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
        )
        subject = self._subject_answer(text, prior)
        if subject:
            return subject
        rules = self._rule_answers(text)
        if rules:
            return "\n\n".join(rules)
        return self._rag(text)
