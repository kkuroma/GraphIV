"""
Every statement this package sends to the store, so no other module opens a connection

An edge is read induced on one set of papers, since a bridge only counts when both ends are held
"""

from __future__ import annotations

import logging

import psycopg

from src.core.schema import Edge, Paper

LOG = logging.getLogger(__name__)

EDGES = """
SELECT source_id, target_id, intent, sentence
FROM citations
WHERE stage = 1 AND source_id = ANY(%s) AND target_id = ANY(%s) AND target_id IS NOT NULL
  AND source_id <> target_id
"""

COSINES = """
SELECT arxiv_id, 1 - (embed_full <=> %s::halfvec)
FROM papers WHERE arxiv_id = ANY(%s) AND stage >= 3
"""


CITERS = """
WITH c AS (
  SELECT source_id, count(DISTINCT target_id) AS n FROM citations
  WHERE stage = 1 AND target_id = ANY(%s) AND NOT source_id = ANY(%s) AND intent = ANY(%s)
  GROUP BY source_id HAVING count(DISTINCT target_id) >= %s
)
SELECT c.source_id FROM c JOIN papers p ON p.arxiv_id = c.source_id AND p.stage >= 3
ORDER BY p.embed_full <=> %s::halfvec LIMIT %s
"""
PAPERS = """
SELECT arxiv_id, title, abstract, date, 1 - (embed_full <=> %s::halfvec)
FROM papers WHERE arxiv_id = ANY(%s) AND stage >= 3
"""

SEED = """
SELECT arxiv_id, title, abstract, date, embed_full::vector::real[]
FROM papers WHERE arxiv_id = %s AND stage >= 3
"""


def as_vector(values: list[float]) -> str:
    """
    One embedding as the literal pgvector parses

    Args:
        values: The embedding.

    Returns:
        The literal.
    """
    return "[" + ",".join(f"{x:.6f}" for x in values) + "]"


def read_edges(dsn: str, arxiv_ids: list[str]) -> list[Edge]:
    """
    Every labeled citation with both ends inside a named set

    Args:
        dsn: libpq connection string.
        arxiv_ids: The papers the subgraph is induced on.

    Returns:
        One entry per citation occurrence.
    """
    if not arxiv_ids:
        return []
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(EDGES, (arxiv_ids, arxiv_ids)).fetchall()
    return [
        Edge(source=str(s), target=str(t), intent=str(i or "background"), sentence=str(sent or ""))
        for s, t, i, sent in rows
    ]


def read_cosines(dsn: str, vector: list[float], arxiv_ids: list[str]) -> dict[str, float]:
    """
    Cosine of every named paper against one vector, in one round trip

    Args:
        dsn: libpq connection string.
        vector: What to measure against.
        arxiv_ids: The papers.

    Returns:
        One cosine per paper the store holds.
    """
    if not arxiv_ids:
        return {}
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(COSINES, (as_vector(vector), arxiv_ids)).fetchall()
    return {str(a): float(c) for a, c in rows}


def read_papers(dsn: str, vector: list[float], arxiv_ids: list[str]) -> list[Paper]:
    """
    Every column a Paper needs for a list of ids

    Args:
        dsn: libpq connection string.
        vector: The question embedding, which fills the cosine.
        arxiv_ids: The papers.

    Returns:
        One Paper per id the store holds.
    """
    if not arxiv_ids:
        return []
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(PAPERS, (as_vector(vector), arxiv_ids)).fetchall()
    return [
        Paper(
            arxiv_id=str(a),
            title=str(t or ""),
            abstract=str(abstract or ""),
            date=when,
            cosine=float(cosine),
        )
        for a, t, abstract, when, cosine in rows
    ]


def read_citers(
    dsn: str, arxiv_ids: list[str], intents: list[str], least: int, vector: list[float], limit: int
) -> list[str]:
    """
    Papers outside a set that cite at least some of it with an intent, closest to a question first

    Args:
        dsn: libpq connection string.
        arxiv_ids: The set.
        intents: The intents a citation must carry.
        least: How many papers of the set a paper must cite.
        vector: The question embedding, which orders the papers found.
        limit: How many papers to return.

    Returns:
        The ids, closest first.
    """
    if not arxiv_ids:
        return []
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            CITERS, (arxiv_ids, arxiv_ids, intents, least, as_vector(vector), limit)
        ).fetchall()
    return [str(r[0]) for r in rows]


def read_seed(dsn: str, arxiv_id: str) -> tuple[Paper, list[float]] | None:
    """
    One paper and its stored vector, which is what a search seeded on that paper sends

    Args:
        dsn: libpq connection string.
        arxiv_id: The paper.

    Returns:
        The paper at a cosine of 1 and its vector, or None when the store holds no vector for it.
    """
    with psycopg.connect(dsn) as conn:
        row = conn.execute(SEED, (arxiv_id,)).fetchone()
    if row is None:
        return None
    a, t, abstract, when, vector = row
    paper = Paper(
        arxiv_id=str(a), title=str(t or ""), abstract=str(abstract or ""), date=when, cosine=1.0
    )
    return paper, [float(x) for x in vector]
