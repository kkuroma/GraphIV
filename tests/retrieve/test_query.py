"""
Checks the pipeline runs its steps in order, and that the pool and the scoring split cleanly
"""

from __future__ import annotations

from datetime import date

import pytest

from src.core.schema import Paper, Query
from src.retrieve.query import PaperQuery


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


class _Retriever:
    """
    Returns a fixed pool and records the limit it was asked for
    """

    def __init__(self) -> None:
        """
        Starts with nothing recorded and nothing blocked
        """
        self.limit: int | None = None
        self.max_date: date | None = None
        self.blocked: list[str] = []

    def retrieve(self, query: Query, limit: int) -> list[Paper]:
        """
        Args:
            query: The embedded question.
            limit: The most papers to return.

        Returns:
            Three papers, cut to the limit.
        """
        self.limit = limit
        self.max_date = query.max_date
        return [_paper("a", 0.9), _paper("b", 0.5), _paper("c", 0.1)][:limit]


class _Appender:
    """
    Appends one paper and records the limit, standing in for an expander
    """

    def __init__(self, arxiv_id: str) -> None:
        """
        Args:
            arxiv_id: The id this step appends.
        """
        self.arxiv_id = arxiv_id
        self.limit: int | None = None

    def rerank(self, query: Query, papers: list[Paper], limit: int | None) -> list[Paper]:
        """
        Args:
            query: The embedded question.
            papers: The pool.
            limit: The ceiling this step was given.

        Returns:
            The pool plus one paper.
        """
        self.limit = limit
        return papers + [_paper(self.arxiv_id, 0.0)]


class _Reverser:
    """
    Reverses the pool and cuts it, standing in for a scorer
    """

    def rerank(self, query: Query, papers: list[Paper], limit: int | None) -> list[Paper]:
        """
        Args:
            query: The embedded question.
            papers: The pool.
            limit: The most papers to return.

        Returns:
            The pool reversed and cut.
        """
        return list(reversed(papers))[: limit if limit is not None else len(papers)]


def test_pipeline_runs_every_step_in_order() -> None:
    """
    Two appenders then a reverser puts the last appended paper first
    """
    first, second = _Appender("x"), _Appender("y")
    pipeline = PaperQuery(_Retriever(), [first, second], [10, 10], [_Reverser()])
    got = pipeline.do_query("t", [0.0], retrieve_limit=3, return_limit=10)
    assert [p.arxiv_id for p in got] == ["y", "x", "c", "b", "a"]


def test_pipeline_hands_each_expander_its_own_limit() -> None:
    """
    Each expander sees the limit at its own position, not the one before it
    """
    first, second = _Appender("x"), _Appender("y")
    retriever = _Retriever()
    pipeline = PaperQuery(retriever, [first, second], [7, 8], [_Reverser()])
    _ = pipeline.do_query("t", [0.0], retrieve_limit=2, return_limit=10)
    assert retriever.limit == 2
    assert first.limit == 7
    assert second.limit == 8


def test_pipeline_cuts_to_the_return_limit() -> None:
    """
    The return limit applies after every step has run
    """
    pipeline = PaperQuery(_Retriever(), [], [], [_Reverser()])
    got = pipeline.do_query("t", [0.0], retrieve_limit=3, return_limit=2)
    assert [p.arxiv_id for p in got] == ["c", "b"]


def test_pipeline_passes_the_date_cap_down() -> None:
    """
    The cap reaches the retriever on the Query rather than as an argument
    """
    retriever = _Retriever()
    pipeline = PaperQuery(retriever, [], [], [])
    _ = pipeline.do_query("t", [0.0], 3, 3, max_date=date(2021, 1, 1))
    assert retriever.max_date == date(2021, 1, 1)


def test_blocking_reaches_the_retriever() -> None:
    """
    A case's own reviews are set on the pipeline rather than baked in when it is built

    That is what lets one pipeline serve every case instead of being rebuilt per case.
    """
    retriever = _Retriever()
    pipeline = PaperQuery(retriever, [], [], [])
    pipeline.block(["a", "b"])
    assert retriever.blocked == ["a", "b"]
    pipeline.block([])
    assert retriever.blocked == []


def test_blocking_reaches_every_step_that_holds_a_blocklist() -> None:
    """
    An expander or a scorer reaches papers through a hop or a count, so each gets the list
    """

    class _BlockingAppender(_Appender):
        blocked: list[str] = []

    class _BlockingReverser(_Reverser):
        blocked: list[str] = []

    expander, scorer = _BlockingAppender("x"), _BlockingReverser()
    pipeline = PaperQuery(_Retriever(), [expander], [None], [scorer])
    pipeline.block(["a"])
    assert expander.blocked == ["a"] and scorer.blocked == ["a"]


def test_pipeline_refuses_a_limit_per_expander_mismatch() -> None:
    """
    A missing limit is a construction error rather than a surprise at query time
    """
    with pytest.raises(ValueError, match="own limit"):
        _ = PaperQuery(_Retriever(), [_Appender("x")], [], [])


def test_a_pool_built_once_serves_several_scorings() -> None:
    """
    Two recipes sharing a pool see the same candidates and only their scoring differs

    This is what lets the benchmark run six recipes for the cost of one retrieval.
    """
    retriever = _Retriever()
    shared = PaperQuery(retriever, [_Appender("x")], [10], [_Reverser()])
    pool = shared.build_pool("t", [0.0], retrieve_limit=3)
    assert [p.arxiv_id for p in pool] == ["a", "b", "c", "x"]

    reversed_once = shared.score_pool("t", [0.0], pool)
    twice = PaperQuery(retriever, [], [], [_Reverser(), _Reverser()]).score_pool("t", [0.0], pool)
    assert [p.arxiv_id for p in reversed_once] == ["x", "c", "b", "a"]
    assert [p.arxiv_id for p in twice] == ["a", "b", "c", "x"]
    assert [p.arxiv_id for p in pool] == ["a", "b", "c", "x"]
