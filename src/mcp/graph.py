"""
The citation graph among the papers a run was given, which the webui's timeline draws

A paper citing another several times is one edge, under its strongest intent
Nothing here depends on the question, so the graph is the corpus's own and not a ranking
"""

from __future__ import annotations

from typing import Any

import psycopg

from src.mcp.citations import ORDER, SENTENCE_CHARS
from src.mcp.context import Context
from src.mcp.papers import clean

NODES = "SELECT arxiv_id, title, date FROM papers WHERE arxiv_id = ANY(%s) AND date IS NOT NULL"
EDGES = """
SELECT source_id, target_id, intent, sentence
FROM citations WHERE source_id = ANY(%s) AND target_id = ANY(%s) AND stage = 1
"""


def citation_graph(
    ctx: Context, served: set[str], read: set[str], cited: set[str]
) -> dict[str, Any]:
    """
    The run's papers as nodes and the citations among them as edges

    Args:
        ctx: The server context, for the store and the graph export.
        served: Every paper any tool returned.
        read: The papers the reader went through.
        cited: The papers the report cites.

    Returns:
        Nodes with title, date, role and citation count, and edges with intent and sentence.
    """
    ids = sorted(served | read)
    if not ids:
        return {"nodes": [], "edges": []}
    with psycopg.connect(ctx.dsn) as conn:
        rows = conn.execute(NODES, (ids,)).fetchall()
        links = conn.execute(EDGES, (ids, ids)).fetchall()
    ptr, index = ctx.graph.cites_ptr, ctx.graph.index
    nodes = []
    for arxiv_id, title, date in rows:
        row = index.get(arxiv_id)
        role = "cited" if arxiv_id in cited else "read" if arxiv_id in read else "found"
        nodes.append(
            {
                "arxiv_id": arxiv_id,
                "title": clean(title),
                "date": date.isoformat(),
                "role": role,
                "cited_by": int(ptr[row + 1] - ptr[row]) if row is not None else 0,
            }
        )
    best: dict[tuple[str, str], tuple[int, str, str]] = {}
    for source, target, intent, sentence in links:
        intent = intent or "background"
        place = ORDER.index(intent) if intent in ORDER else len(ORDER)
        key = (source, target)
        if source != target and (key not in best or place < best[key][0]):
            best[key] = (place, intent, clean(sentence)[:SENTENCE_CHARS])
    edges = [
        {"source": s, "target": t, "intent": intent, "sentence": sentence}
        for (s, t), (_, intent, sentence) in best.items()
    ]
    return {"nodes": nodes, "edges": edges}
