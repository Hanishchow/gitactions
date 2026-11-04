"""CPU-only tests for the router, using the real KB and a stub LLM."""
import json

import pytest

from campus_gpt.bot import REFUSAL, CampusBot
from campus_gpt.rag import Retriever


@pytest.fixture()
def bot():
    return CampusBot()


def test_cie_minimum(bot):
    out = bot.answer("What is the minimum CIE marks to pass?")
    assert "20 out of 50" in out


def test_see_minimum(bot):
    assert "35%" in bot.answer("minimum SEE marks?")


def test_pass_total(bot):
    assert "40 out of 100" in bot.answer("what are the passing marks?")


def test_attendance(bot):
    out = bot.answer("minimum attendance and condonation")
    assert "85%" in out and "10%" in out


def test_grade_table(bot):
    out = bot.answer("show the grading scheme")
    assert "90-100 O (10)" in out and "40-49 P (4)" in out


def test_cgpa_percentage_calculation(bot):
    assert "82.0%" in bot.answer("convert 8.2 cgpa to percentage")


def test_cgpa_formula_text(bot):
    out = bot.answer("what is the cgpa to percentage formula")
    assert "CGPA x 10" in out


def test_semester7_listing(bot):
    out = bot.answer("AIML 7th sem subjects")
    for code in ("BAI701", "BAI702", "BAI703", "BAI786"):
        assert code in out
    assert "24 credits" in out


def test_semester8_listing(bot):
    out = bot.answer("what are the 8th semester courses")
    assert "BAI803" in out and "Internship" in out


def test_uncovered_semester_is_declined(bot):
    out = bot.answer("2nd semester subjects")
    assert "don't have Semester 2" in out


def test_subject_code_lookup(bot):
    out = bot.answer("tell me about BAI701")
    assert "Deep Learning" in out and "4 credits" in out


def test_elective_option_code_resolves_to_slot(bot):
    out = bot.answer("what is BAI613A")
    assert "Human-Centred AI" in out and "Sem 6" in out


def test_bcs502_is_computer_networks(bot):
    assert "Computer Networks" in bot.answer("BCS502")


def test_followup_semester_from_history(bot):
    hist = [{"role": "user", "content": "AIML 7th sem subjects"}, {"role": "assistant", "content": "..."}]
    out = bot.answer("how many credits are the subjects?", hist)
    assert "Semester 7" in out


def test_history_tuples_do_not_crash(bot):
    assert bot.answer("minimum attendance", [("hi", "hello")])


def test_out_of_scope_refused_without_retriever(bot):
    assert bot.answer("who won the world cup") == REFUSAL


def test_empty_question(bot):
    assert "Ask me" in bot.answer("   ")


@pytest.fixture()
def rag_bot(tmp_path):
    chunks = [
        {"id": "1", "text": "The transport layer provides process to process delivery using TCP and UDP.",
         "file": "BCS502-module-4-textbook.pdf", "location": {"page": 3}, "subject": "BCS502", "module": 4},
        {"id": "2", "text": "A pushdown automaton recognises context free languages.",
         "file": "BCS503-mod3.pdf", "location": {"page": 7}, "subject": "BCS503", "module": 3},
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    return Retriever(path)


def test_rag_returns_cited_passages_without_llm(rag_bot):
    out = CampusBot(retriever=rag_bot).answer("what does TCP provide at the transport layer")
    assert "BCS502-module-4-textbook.pdf" in out and "page 3" in out


def test_rag_with_stub_llm_adds_sources(rag_bot):
    out = CampusBot(retriever=rag_bot, generate=lambda m: "It gives process to process delivery [1].").answer(
        "what does TCP provide at the transport layer"
    )
    assert "[1]" in out and "Sources:" in out


def test_llm_refusal_is_normalised(rag_bot):
    bot = CampusBot(retriever=rag_bot, generate=lambda m: "I don't have that in my knowledge base.")
    assert bot.answer("explain pushdown automaton") == REFUSAL


def test_unrelated_query_refused_with_retriever(rag_bot):
    assert CampusBot(retriever=rag_bot).answer("zebra quantum bananas") == REFUSAL


def test_greeting_and_help(bot):
    for q in ("hi", "Hello!", "help", "what can you do?"):
        assert "Campus GPT" in bot.answer(q)


def test_refusal_lists_what_it_can_answer(bot):
    out = bot.answer("ai ml basics?")
    assert out == REFUSAL and "AIML 7th sem subjects" in out
