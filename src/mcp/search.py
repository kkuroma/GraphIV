"""
BoundedField over the whole corpus, from a question or from one paper, held to a filter

The pipeline object is shared and remembers what it blocks, so one lock covers every ranking
"""

from __future__ import annotations

import math
import threading
from datetime import date

from src.core.schema import Paper
from src.mcp.context import Context
from src.mcp.papers import Filter, Record, allowed, by_authors, read_records
from src.mcp.runs import Run
from src.miner.store import read_seed

LOCK = threading.Lock()
MOST = 50
# ranked papers looked at per paper wanted, since a filter can drop most of the head
OVERSHOOT = 20
SEARCH_ABSTRACT = 200
SEARCH_AUTHORS = 3


def why(ctx: Context, mode: str, paper: Paper) -> dict[str, float]:
    """
    Each term's share of one paper's log score, which is what the webui shows as why it ranked

    Args:
        ctx: The open handles.
        mode: The recipe that ranked it.
        paper: The paper, carrying the scorer's terms.

    Returns:
        The log contribution per term, the hub one negative.
    """
    weights = ctx.weights[mode]
    out = {t: weights.get(t, 0.0) * math.log1p(v) for t, v in paper.signals.items() if t != "hub"}
    out["hub"] = -weights.get("hub", 0.0) * math.log1p(paper.signals.get("hub", 0.0))
    return {t: round(v, 4) for t, v in out.items() if v}


def rank(
    ctx: Context, mode: str, query: str, arxiv_id: str, held: Filter, wanted: int
) -> tuple[list[Record], str]:
    """
    The top of one BoundedField ranking that passes the filter

    Args:
        ctx: The open handles.
        mode: Which recipe ranks.
        query: The question, in words.
        arxiv_id: A paper whose vector replaces the question's, empty for none.
        held: The filter.
        wanted: How many papers to return.

    Returns:
        The records, best first, and a note when the seed paper is missing.
    """
    pipe = ctx.pipes[mode]
    cap = held.latest or date.today()
    if arxiv_id:
        seed = read_seed(ctx.dsn, arxiv_id.strip())
        if seed is None:
            return [], f"{arxiv_id} is not in the corpus."
        text, vector = query or seed[0].title, seed[1]
    else:
        text, vector = query, ctx.embed(query)
    with LOCK:
        pipe.block(list(held.exclude))
        pool = pipe.build_pool(text, vector, ctx.seed_limit, cap)
        ranked: list[Paper] = pipe.score_pool(text, vector, pool, cap)
    head = [p.arxiv_id for p in ranked[: max(wanted * OVERSHOOT, 200)]]
    records = read_records(ctx.dsn, head)
    kept = [records[a] for a in head if a in records and allowed(records[a], held)][:wanted]
    terms = {p.arxiv_id: why(ctx, mode, p) for p in ranked[: max(wanted * OVERSHOOT, 200)]}
    for record in kept:
        record.terms = terms.get(record.arxiv_id, {})
    return kept, ""


def search_papers(
    ctx: Context, run: Run, query: str, arxiv_id: str, mode: str, held: Filter, limit: int
) -> str:
    """
    A ranked list of papers for a question or around a paper

    Args:
        ctx: The open handles.
        run: The caller's run, which keeps why each paper ranked.
        query: The question.
        arxiv_id: A reference paper, empty for none.
        mode: focused, survey, frontier or consensus.
        held: The filter.
        limit: How many papers, at most 50.

    Returns:
        One entry per paper, best first.
    """
    if mode not in ctx.pipes:
        return f"Unknown mode {mode}. Use one of: {', '.join(ctx.pipes)}."
    wanted = max(1, min(limit, MOST))
    if not query.strip() and not arxiv_id.strip():
        if not held.authors:
            return "Pass a query, an arxiv_id, or authors in the filter."
        ids = by_authors(ctx.dsn, held, wanted)
        records = read_records(ctx.dsn, ids)
        listed = [records[a] for a in ids if a in records]
        return "\n".join(r.line(SEARCH_ABSTRACT, SEARCH_AUTHORS) for r in listed) or "No papers."
    found, note = rank(ctx, mode, query, arxiv_id, held, wanted)
    if note:
        return note
    if not found:
        return "No papers passed the filter."
    with run.lock:
        run.searches.append(
            {
                "query": query,
                "arxiv_id": arxiv_id,
                "mode": mode,
                "papers": [
                    {"arxiv_id": r.arxiv_id, "title": r.title, "terms": r.terms} for r in found
                ],
            }
        )
    return "\n".join(r.line(SEARCH_ABSTRACT, SEARCH_AUTHORS) for r in found)
