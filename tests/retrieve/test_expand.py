"""
Checks an expander picks the right targets and appends without disturbing the pool

The database round trip is stubbed, since what it returns is decided by the ids asked for
"""

from __future__ import annotations

import contextlib
from datetime import date

import pytest

from src.core.parse_graph_export import GraphExport
from src.core.schema import Paper, Query
from src.retrieve import expand
from src.retrieve.expand import AncestorExpander, FrontierExpander, get_expansion_targets


def _paper(arxiv_id: str, cosine: float) -> Paper:
    """
    One filled Paper

    Args:
        arxiv_id: Its id.
        cosine: Its cosine.

    Returns:
        The paper.
    """
    return Paper(arxiv_id=arxiv_id, title="t", abstract="a", date=date(2020, 1, 1), cosine=cosine)


def test_expansion_targets_are_the_top_by_cosine() -> None:
    """
    The two best cosines are taken and the order is best first
    """
    pool = [_paper("a", 0.1), _paper("b", 0.9), _paper("c", 0.5)]
    assert [p.arxiv_id for p in get_expansion_targets(pool, 2)] == ["b", "c"]
    assert len(get_expansion_targets(pool, 10)) == 3


def test_ancestor_expander_finds_what_the_core_cites(graph: GraphExport, pool: list[Paper]) -> None:
    """
    p0 and p1 are the two best cosines, and between them they cite p4 and p5
    """
    expander = AncestorExpander(graph, "unused", top_k=2)
    assert expander.get_neighbor_ids(pool[:2]) == ["p4", "p5"]


def test_frontier_expander_finds_what_cites_the_core(graph: GraphExport, pool: list[Paper]) -> None:
    """
    Only p3 cites either of the two best papers
    """
    expander = FrontierExpander(graph, "unused", top_k=2)
    assert expander.get_neighbor_ids(pool[:2]) == ["p3"]


def test_expander_never_returns_what_the_pool_already_holds(
    graph: GraphExport, pool: list[Paper]
) -> None:
    """
    p4 and p5 are already in the whole pool, so expanding it finds nothing new
    """
    expander = AncestorExpander(graph, "unused", top_k=2)
    assert expander.get_neighbor_ids(pool) == []


def test_expander_appends_and_keeps_the_input(
    graph: GraphExport, pool: list[Paper], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The pool comes back first and unchanged, with what the fetch returned after it

    The connection and the fetch are stubbed, since what they return is decided by the ids the
    expander asked for and those are checked above.
    """
    asked: list[list[str]] = []

    def fake_fetch(_conn: object, _q: Query, ids: list[str], _cap: date | None) -> list[Paper]:
        asked.append(ids)
        return [_paper(a, 0.0) for a in ids]

    monkeypatch.setattr(expand, "fetch_papers", fake_fetch)
    monkeypatch.setattr(expand.psycopg, "connect", lambda _dsn: contextlib.nullcontext(None))
    got = AncestorExpander(graph, "unused", top_k=2).rerank(
        Query(text="t", emb=[0.0]), pool[:2], None
    )
    assert asked == [["p4", "p5"]]
    assert [p.arxiv_id for p in got] == ["p0", "p1", "p4", "p5"]


def test_expander_respects_the_limit(graph: GraphExport, pool: list[Paper]) -> None:
    """
    A hop limit of one keeps only the most reached neighbor
    """
    expander = AncestorExpander(graph, "unused", top_k=2, hop_limit=1)
    assert expander.get_neighbor_ids(pool[:2]) == ["p4"]


def test_expander_never_returns_a_blocked_paper(graph: GraphExport, pool: list[Paper]) -> None:
    """
    p4 is what the core cites, and a case that blocks it gets only p5
    """
    expander = AncestorExpander(graph, "unused", top_k=2)
    expander.blocked = ["p4"]
    assert expander.get_neighbor_ids(pool[:2]) == ["p5"]
