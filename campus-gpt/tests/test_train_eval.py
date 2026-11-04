"""Dataset builder and eval harness (CPU, no model)."""
from campus_gpt.bot import CampusBot
from eval.run_eval import run
from train.make_dataset import build


def test_split_is_disjoint_and_nonempty():
    s = build()
    assert len(s["train"]) > 50 and len(s["heldout"]) > 10
    tq = {r["messages"][1]["content"] for r in s["train"]}
    hq = {r["messages"][1]["content"] for r in s["heldout"]}
    assert not (tq & hq)


def test_heldout_answers_not_in_train_targets():
    s = build()
    train_targets = {r["messages"][-1]["content"].rsplit(" [", 1)[0] for r in s["train"]}
    held_targets = {r["messages"][-1]["content"].rsplit(" [", 1)[0] for r in s["heldout"]
                    if not r["messages"][-1]["content"].startswith("I don't")}
    assert held_targets and not (train_targets & held_targets)


def test_deterministic():
    assert build(seed=3) == build(seed=3)


def test_targets_cite_the_relevant_passage():
    for r in build()["train"][:40]:
        ans = r["messages"][-1]["content"]
        if ans.startswith("I don't"):
            continue
        pos = int(ans.rsplit("[", 1)[1].rstrip("]"))
        ctx = r["messages"][0]["content"]
        assert f"[{pos}] {ans.rsplit(' [', 1)[0]}" in ctx


def test_golden_all_pass_on_router():
    res = run(CampusBot(), None)
    assert res["golden_correct"] == res["golden_total"], res["golden_failures"]
