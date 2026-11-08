"""
Scores a pool by where each paper sits in the citation neighborhood of the query's best matches

Every channel is a walk from the same core, so one adjacency serves all of them
Nothing here opens a connection, since the steps above leave every Paper filled
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from src.core.parse_graph_export import GraphExport, gather_rows
from src.core.schema import Paper, Query
from src.retrieve.keyword import get_bm25

# the multiplier is what makes log1p compress, since normalizing leaves values where it is linear
LOG_SCALE = 100.0

TERMS = (
    "cos",
    "rec",
    "anc",
    "fro",
    "anc2",
    "fro2",
    "anc_decay",
    "fro_decay",
    "cocite",
    "couple",
    "bm25",
)


@dataclass(frozen=True)
class Weights:
    """
    One exponent per term, where zero makes that term exactly 1 and drops it from the product
    """

    cos: float = 0.5
    rec: float = 1.0
    anc: float = 2.0
    fro: float = 1.0
    anc2: float = 0.0
    fro2: float = 0.0
    anc_decay: float = 1.0
    fro_decay: float = 0.0
    cocite: float = 1.0
    couple: float = 1.0
    bm25: float = 0.0
    hub: float = 0.3


def normalize_to_max(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Scales a vector onto [0, 1] after clipping it at zero

    Args:
        values: Any vector.

    Returns:
        The vector divided by its maximum, all zeros when it has no positive entry.
    """
    clipped = np.clip(values, 0, None)
    return clipped / (clipped.max() or 1.0)


def compress_long_tail(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    The transform every walk channel passes through, which keeps zeros at zero

    A channel spans orders of magnitude, so without the log the top paper is the only one that
    moves the score at all.

    Args:
        values: A channel.

    Returns:
        The channel on [0, 1].
    """
    return normalize_to_max(np.log1p(normalize_to_max(values) * LOG_SCALE))


def get_combined_score(
    individual: dict[str, NDArray[np.float64]], weights: Weights
) -> NDArray[np.float64]:
    """
    Multiplies every term by its own exponent, over the hub denominator

    Each factor sits on [1, 2 ** exponent], so a channel a paper scores nothing on is neutral
    rather than fatal.

    Args:
        individual: One vector per term, each already on [0, 1].
        weights: The exponents.

    Returns:
        One score per paper, higher is better.
    """
    out = np.ones(len(individual["cos"]))
    for term in TERMS:
        out = out * (1.0 + individual[term]) ** getattr(weights, term)
    return out / (1.0 + individual["hub"]) ** weights.hub


class GraphScorer:
    """
    The one step that ranks, cutting the pool by a product of bounded graph terms
    """

    def __init__(
        self,
        graph: GraphExport,
        weights: Weights | None = None,
        *,
        core: int = 200,
        depth: int = 4,
        decay: float = 0.5,
    ) -> None:
        """
        Holds the export and the constants, since one scorer serves every query in a process

        Args:
            graph: The citation graph export.
            weights: The exponents, defaulting to the shipped point.
            core: How many of the pool every walk starts from, taken off the top by cosine.
            depth: The furthest distance the decay channel still carries a value.
            decay: What one step of distance costs that channel.
        """
        self.graph = graph
        self.weights = weights or Weights()
        self.core = core
        self.depth = depth
        self.decay = decay
        self.blocked: list[str] = []

    def get_paper_rows(self, papers: list[Paper]) -> NDArray[np.int32]:
        """
        Graph rows for the pool, in the order the pool was handed over

        Args:
            papers: The pool.

        Returns:
            One row per paper.
        """
        return np.array([self.graph.index[p.arxiv_id] for p in papers], np.int32)

    def build_adjacency_matrix(self, rows: NDArray[np.int32]) -> sp.csr_matrix:
        """
        Citations among the pool, where entry `i, j` is 1 when paper j cites paper i

        One multiply by the matrix is therefore a hop backward in time and one by its transpose
        is a hop forward.

        Args:
            rows: Graph rows of the pool.

        Returns:
            A square matrix over the pool.
        """
        size = len(rows)
        seat = np.full(self.graph.n_papers, -1, np.int64)
        seat[rows] = np.arange(size)
        start = self.graph.refs_ptr[rows]
        count = self.graph.refs_ptr[rows + 1] - start
        citing = np.repeat(np.arange(size), count)
        cited = seat[gather_rows(self.graph.refs_idx, start, count)]
        live = cited >= 0
        return sp.csr_matrix(
            (np.ones(int(live.sum()), np.float32), (cited[live], citing[live])),
            shape=(size, size),
        )

    def get_core_weights(self, papers: list[Paper]) -> NDArray[np.float64]:
        """
        Where every walk starts, cosine on the top `core` of the pool and zero elsewhere

        Args:
            papers: The pool.

        Returns:
            A vector over the pool summing to 1.
        """
        cosine = np.array([p.cosine or 0.0 for p in papers])
        out = np.zeros(len(papers))
        top = np.argsort(-cosine)[: self.core]
        out[top] = cosine[top]
        return out / (out.sum() or 1.0)

    def get_distance_decay(
        self, step: sp.csr_matrix, weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        `decay ** distance` from the core, and zero where the walk never arrives

        A one-step channel is exactly zero for every paper the core does not touch, so this is
        what lets a paper further out score at all.

        Args:
            step: The adjacency to walk, already oriented.
            weights: The core, whose nonzeros are the sources.

        Returns:
            One value per paper.
        """
        out = np.zeros(len(weights))
        seen = weights > 0
        out[seen] = 1.0
        current = seen.astype(np.float64)
        for hop in range(1, self.depth + 1):
            fresh = ((step @ current) > 0) & ~seen
            if not fresh.any():
                break
            out[fresh] = self.decay**hop
            seen |= fresh
            current = fresh.astype(np.float64)
        return out

    def get_blocked_rows(self) -> NDArray[np.int32]:
        """
        Graph rows of the blocked papers, which no citation count may include as a citer

        Returns:
            One row per blocked paper the export holds.
        """
        index = self.graph.index
        return np.array([index[a] for a in self.blocked if a in index], np.int32)

    def get_individual_score(
        self, papers: list[Paper], rows: NDArray[np.int32], cutoff: int, text: str = ""
    ) -> dict[str, NDArray[np.float64]]:
        """
        Every term on [0, 1], one vector each, before any exponent is applied

        The walk channels pass through `compress_long_tail` and the decay channels do not,
        since a decay curve already spans [0, 1] and compressing it would rescale it onto its
        own maximum.

        Args:
            papers: The pool.
            rows: Graph rows of the pool.
            cutoff: The latest day a citation may carry.
            text: The question, which the keyword term matches against each title and abstract.

        Returns:
            One normalized vector per term, keyed by the name of its exponent.
        """
        back = self.build_adjacency_matrix(rows)
        fwd = sp.csr_matrix(back.T)
        core = self.get_core_weights(papers)
        age = np.clip((cutoff - self.graph.day[rows]) / 365.25, 0.0, None)
        recency = 1.0 / (1.0 + age)
        ancestors, frontier = back @ core, fwd @ core
        return {
            "cos": normalize_to_max(np.array([p.cosine or 0.0 for p in papers])),
            "rec": normalize_to_max(recency),
            "anc": compress_long_tail(ancestors),
            "fro": compress_long_tail(frontier),
            "anc2": compress_long_tail(back @ ancestors),
            "fro2": compress_long_tail(fwd @ frontier),
            "anc_decay": self.get_distance_decay(back, core),
            "fro_decay": self.get_distance_decay(fwd, core),
            "cocite": compress_long_tail((back @ frontier) * recency),
            "couple": compress_long_tail((fwd @ ancestors) * recency),
            "bm25": normalize_to_max(get_bm25(text, [f"{p.title}\n{p.abstract}" for p in papers])),
            "hub": compress_long_tail(
                self.graph.get_citation_counts_before(rows, cutoff, self.get_blocked_rows())
            ),
        }

    def rerank(self, query: Query, papers: list[Paper], limit: int | None) -> list[Paper]:
        """
        Cuts the pool to `limit` by the graph score, leaving every term on the Paper

        Args:
            query: The embedded question, whose `max_date` caps recency and the citation count.
            papers: The pool as the steps above left it.
            limit: The most papers to return, `None` for no limit.

        Returns:
            At most `limit` papers, best first.
        """
        hidden = set(self.blocked)
        papers = [p for p in papers if p.arxiv_id not in hidden]
        if not papers:
            return []
        rows = self.get_paper_rows(papers)
        cutoff = self.graph.get_day_number(query.max_date or date.today())
        individual = self.get_individual_score(papers, rows, cutoff, query.text)
        combined = get_combined_score(individual, self.weights)
        order = np.argsort(-combined)[: limit if limit is not None else len(papers)]

        out: list[Paper] = []
        for seat in order:
            paper = papers[int(seat)]
            paper.signals = {name: float(v[seat]) for name, v in individual.items()}
            paper.score = float(combined[seat])
            out.append(paper)
        return out
