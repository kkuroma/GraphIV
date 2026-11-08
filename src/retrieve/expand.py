"""
Adds candidates one citation hop off the best matches, in each direction

An expander only appends, so the two commute and neither can delete what the other found
Both fill every field, so no step downstream receives a half-built Paper
"""

from __future__ import annotations

from datetime import date

import numpy as np
import psycopg
from numpy.typing import NDArray

from src.core.parse_graph_export import GraphExport
from src.core.schema import Paper, Query

FETCH = """
SELECT arxiv_id, title, abstract, date, 1 - (embed_full <=> %s::halfvec)
FROM papers WHERE arxiv_id = ANY(%s) AND stage >= 3
"""


def get_expansion_targets(papers: list[Paper], top_k: int) -> list[Paper]:
    """
    The papers an expander hops from, taken off the top by cosine

    Args:
        papers: The pool.
        top_k: How many to hop from.

    Returns:
        At most `top_k` papers, best cosine first.
    """
    ranked = sorted(papers, key=lambda p: -(p.cosine or 0.0))
    return ranked[:top_k]


def fetch_papers(
    conn: psycopg.Connection, query: Query, arxiv_ids: list[str], cutoff: date | None
) -> list[Paper]:
    """
    Reads every column a Paper needs for a list of ids, in one round trip

    Title and abstract ride along free, since the row has to be read for the date anyway.

    Args:
        conn: An open connection.
        query: The embedded question, whose vector fills the cosine.
        arxiv_ids: The papers to read.
        cutoff: The latest date a paper may carry, or None for no cap.

    Returns:
        One Paper per id that exists and passed the cap.
    """
    if not arxiv_ids:
        return []
    vector = "[" + ",".join(f"{x:.6f}" for x in query.emb) + "]"
    rows = conn.execute(FETCH, (vector, arxiv_ids)).fetchall()
    return [
        Paper(
            arxiv_id=str(a),
            title=str(t or ""),
            abstract=str(abstract or ""),
            date=when,
            cosine=float(cosine),
        )
        for a, t, abstract, when, cosine in rows
        if cutoff is None or when <= cutoff
    ]


class _Expander:
    """
    The half of an expander that does not depend on which direction it walks
    """

    forward: bool

    def __init__(
        self, graph: GraphExport, pg_dest: str, *, top_k: int = 200, hop_limit: int = 20000
    ) -> None:
        """
        Holds the export and the connection string, since one expander serves every query

        Args:
            graph: The citation graph export.
            pg_dest: libpq connection string, used once per call.
            top_k: How many of the pool to hop from.
            hop_limit: The most neighbors to keep.
        """
        self.graph = graph
        self.pg_dest = pg_dest
        self.top_k = top_k
        self.hop_limit = hop_limit
        self.blocked: list[str] = []

    def get_neighbor_ids(self, papers: list[Paper]) -> list[str]:
        """
        The arXiv ids one hop away that the pool does not already hold and the case does not block

        Args:
            papers: The pool.

        Returns:
            Ids to fetch, most reached first.
        """
        rows: NDArray[np.int32] = np.array(
            [
                self.graph.index[p.arxiv_id]
                for p in get_expansion_targets(papers, self.top_k)
                if p.arxiv_id in self.graph.index
            ],
            np.int32,
        )
        found = self.graph.get_neighbors(rows, self.forward, self.hop_limit)
        held = {p.arxiv_id for p in papers} | set(self.blocked)
        ids = [bytes(self.graph.ids[i]).decode() for i in found]
        return [a for a in ids if a not in held]

    def rerank(self, query: Query, papers: list[Paper], limit: int | None) -> list[Paper]:
        """
        Returns the pool plus its neighbors, appending and never cutting

        Args:
            query: The embedded question.
            papers: The pool as the previous step left it.
            limit: The most neighbors to append, `None` for the hop limit.

        Returns:
            The input followed by the papers it reached.
        """
        wanted = self.get_neighbor_ids(papers)[: limit if limit is not None else self.hop_limit]
        if not wanted:
            return papers
        with psycopg.connect(self.pg_dest) as conn:
            found = fetch_papers(conn, query, wanted, query.max_date)
        return papers + found


class AncestorExpander(_Expander):
    """
    Adds what the best matches cite, which is where a field came from
    """

    forward: bool = False


class FrontierExpander(_Expander):
    """
    Adds what cites the best matches, which is what the field did next
    """

    forward: bool = True
