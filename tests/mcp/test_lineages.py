"""
The poset's pure steps, one citation per pair, the reduction a figure draws and the longest chain
"""

from datetime import date

from src.core.schema import Edge
from src.mcp.lineages import best_edges, longest, reduce
from src.mcp.papers import Record


def paper(arxiv_id: str, year: int) -> Record:
    return Record(
        arxiv_id=arxiv_id, title=arxiv_id, abstract="", authors="A B", date=date(year, 1, 1)
    )


PAPERS = {a: paper(a, y) for a, y in (("a", 2020), ("b", 2021), ("c", 2022), ("d", 2023))}


def test_one_citation_per_pair_keeps_the_strongest_intent_and_drops_backward_ones():
    edges = [
        Edge(source="b", target="a", intent="compares", sentence="b compares a"),
        Edge(source="b", target="a", intent="critiques", sentence="b critiques a"),
        Edge(source="a", target="b", intent="extends", sentence="a cannot cite a later b"),
        Edge(source="c", target="a", intent="background", sentence="kept out"),
    ]
    best = best_edges(edges, PAPERS, ["extends", "critiques", "compares"])
    assert list(best) == [("b", "a")] and best[("b", "a")].intent == "critiques"


def test_the_reduction_drops_a_citation_a_longer_path_implies():
    pairs = {("b", "a"), ("c", "b"), ("c", "a"), ("d", "c")}
    assert reduce(pairs) == {("b", "a"), ("c", "b"), ("d", "c")}
    assert longest(reduce(pairs), PAPERS, {a: 1.0 for a in PAPERS}) == ["a", "b", "c", "d"]


def test_the_chain_starts_where_its_papers_turn_to_the_question():
    pairs = {("b", "a"), ("c", "b"), ("d", "c")}
    weight = {"a": -0.3, "b": -0.2, "c": 0.4, "d": 0.5}
    assert longest(pairs, PAPERS, weight) == ["c", "d"]
