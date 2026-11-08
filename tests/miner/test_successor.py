"""
How a paper taking a position is ranked, and how many slots one argued-with paper may take
"""

from __future__ import annotations

from datetime import date

from src.core.schema import Edge, Paper
from src.miner.miner import PER_TARGET, Miner, Successor
from src.miner.recipe import build_recipe


def paper(arxiv_id: str, cosine: float) -> Paper:
    return Paper(
        arxiv_id=arxiv_id, title=arxiv_id, abstract="", date=date(2026, 1, 1), cosine=cosine
    )


def test_reach_counts_what_a_paper_positioned_itself_against():
    one = Successor(paper=paper("a", 0.5), targets={"x": "extends", "y": "critiques"})
    assert one.reach == 2


def test_closeness_breaks_a_tie_on_reach():
    near = Successor(paper=paper("a", 0.8), targets={"x": "extends"})
    far = Successor(paper=paper("b", 0.2), targets={"y": "extends"})
    assert near.pull > far.pull


def test_a_broad_name_check_loses_to_a_close_supersession():
    broad = Successor(paper=paper("a", 0.1), targets={f"x{i}": "compares" for i in range(4)})
    close = Successor(paper=paper("b", 0.9), targets={"y": "extends"})
    assert close.pull > broad.pull


def edge(source: str, target: str, intent: str = "critiques") -> Edge:
    return Edge(source=source, target=target, intent=intent, sentence="x" * 60)


def test_one_paper_cannot_take_every_dispute_slot():
    miner = Miner.__new__(Miner)
    miner.recipe = build_recipe({})
    edges = [edge(f"c{i}", "a") for i in range(6)] + [edge("d", "b"), edge("e", "b")]
    out = miner.disputes(edges, {"a": 1.0, "b": 0.5})
    assert sum(1 for e in out if e.target == "a") == PER_TARGET
    assert {e.target for e in out} == {"a", "b"}
