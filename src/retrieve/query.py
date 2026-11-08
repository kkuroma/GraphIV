"""
Chains one retriever, any number of expanders and the step that scores into the call a caller makes

The split is what lets several recipes share one pool, since only the scoring differs between them
An expander only appends and the scorer only cuts, so the order inside each half does not matter
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from src.core.schema import Paper, Query, Reranker, Retriever


@runtime_checkable
class Blocking(Protocol):
    """
    A step that can reach a paper the retriever never returned, so it holds the case's blocklist
    """

    blocked: list[str]


class PaperQuery:
    """
    The pipeline, which owns the order of the steps and nothing else
    """

    def __init__(
        self,
        retriever: Retriever,
        expanders: list[Reranker],
        limits: list[int | None],
        scorers: list[Reranker],
    ) -> None:
        """
        Holds the steps in the order they run, split at the point the pool stops growing

        Args:
            retriever: The step that picks the first pool.
            expanders: The steps that append candidates.
            limits: One ceiling per expander, `None` for no ceiling.
            scorers: The steps that rank and cut.
        """
        if len(expanders) != len(limits):
            raise ValueError("every expander needs its own limit")
        self.retriever = retriever
        self.expanders = expanders
        self.limits = limits
        self.scorers = scorers

    def block(self, arxiv_ids: list[str]) -> None:
        """
        Sets what this pipeline may never return or count, which is the only thing a case changes

        Every step holding a blocklist gets it, since a hop or a count reaches past the retriever.

        Args:
            arxiv_ids: Papers to hide, usually an eval case's own reviews.
        """
        self.retriever.blocked = list(arxiv_ids)
        for step in (*self.expanders, *self.scorers):
            if isinstance(step, Blocking):
                step.blocked = list(arxiv_ids)

    def build_pool(
        self, text: str, emb: list[float], retrieve_limit: int, max_date: date | None = None
    ) -> list[Paper]:
        """
        Every candidate this recipe considers, before anything ranks them

        Args:
            text: The question.
            emb: Its embedding, made once so no step re-embeds it.
            retrieve_limit: How many papers the retriever may return.
            max_date: The latest date any paper may carry.

        Returns:
            The pool.
        """
        query = Query(text=text, emb=emb, max_date=max_date)
        papers = self.retriever.retrieve(query, retrieve_limit)
        for expander, limit in zip(self.expanders, self.limits, strict=True):
            papers = expander.rerank(query, papers, limit)
        return papers

    def score_pool(
        self,
        text: str,
        emb: list[float],
        papers: list[Paper],
        max_date: date | None = None,
        return_limit: int | None = None,
    ) -> list[Paper]:
        """
        Ranks a pool this recipe did not have to build itself

        Args:
            text: The question.
            emb: Its embedding.
            papers: The pool.
            max_date: The latest date any paper may carry.
            return_limit: How many papers the caller wants, `None` for all of them.

        Returns:
            The ranked pool, best first.
        """
        query = Query(text=text, emb=emb, max_date=max_date)
        out = list(papers)
        for scorer in self.scorers:
            out = scorer.rerank(query, out, None)
        return out[:return_limit] if return_limit is not None else out

    def do_query(
        self,
        text: str,
        emb: list[float],
        retrieve_limit: int,
        return_limit: int,
        max_date: date | None = None,
    ) -> list[Paper]:
        """
        Runs one question through every step and returns the top of what survives

        Args:
            text: The question.
            emb: Its embedding, made once so no step re-embeds it.
            retrieve_limit: How many papers the retriever may return.
            return_limit: How many papers the caller wants.
            max_date: The latest date any paper may carry.

        Returns:
            At most `return_limit` papers, best first by the last step's key.
        """
        pool = self.build_pool(text, emb, retrieve_limit, max_date)
        return self.score_pool(text, emb, pool, max_date, return_limit)
