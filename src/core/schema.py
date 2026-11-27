"""
The shapes that cross a step boundary, and the two protocols every step implements

A step fills what it knows and never clears what another step filled
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """
    Represents a paper as it travels through a query

    Every field but the scores is filled by whichever step first puts the paper in the pool, so a
    later step never sees a half-built paper.
    """

    arxiv_id: str
    title: str
    abstract: str
    date: date
    citations: int = 0
    cosine: float | None = None
    score: float | None = None
    signals: dict[str, float] = Field(default_factory=dict)
    llm_score: int | None = None


class Query(BaseModel):
    """
    Represents a query that is pre-embedded once so that no step downstream re-embeds it

    `max_date` rides here rather than on the pipeline because every step that reads a date needs
    it, and an eval case has to retrieve the world its survey retrieved.
    """

    text: str
    emb: list[float]
    max_date: date | None = None


class Retriever(Protocol):
    """
    A template for a caller that, given a query, retrieves a list of papers that answers it
    """

    blocked: list[str]

    def retrieve(self, query: Query, limit: int) -> list[Paper]:
        """
        Returns the pool, highest first by this retriever's own key

        Args:
            query: The embedded question.
            limit: The most papers the call may return.

        Returns:
            At most `limit` papers.
        """
        ...


class Reranker(Protocol):
    """
    A template for a caller that, given a query and a pool of papers, reorders it by some algorithm
    """

    def rerank(self, query: Query, papers: list[Paper], limit: int | None) -> list[Paper]:
        """
        Returns the pool cut by this reranker's own key, never by cosine

        A step that ranks on something else would otherwise delete what it just added, and a step
        that only appends returns more papers than it was given.

        Args:
            query: The embedded question.
            papers: The pool as the previous step left it.
            limit: The most papers the call may return, `None` for no limit.

        Returns:
            At most `limit` papers.
        """
        ...


class Edge(BaseModel):
    """
    One citation occurrence, carrying why it was made and the sentence that made it

    `intent` is one of the six labels the citation classifier writes, and an unlabeled edge
    arrives as `background`, which weighs nothing.
    """

    source: str
    target: str
    intent: str
    sentence: str = ""


class Grade(BaseModel):
    """
    One reader verdict on one paper, which is the only place a language model touches a ranking
    """

    role: str
    relevance: int
    reason: str = ""

    @property
    def factor(self) -> float:
        """
        What the verdict multiplies a score by, where the lowest relevance zeroes it

        Returns:
            A multiplier on [0, 1].
        """
        return (max(-2, min(2, self.relevance)) + 2) / 4.0
