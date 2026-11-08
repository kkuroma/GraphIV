"""
One paper from a vague description: the filter, then candidates, then the model, then a name check

The model rules on every candidate, and a pick it ruled is not the paper is no pick
A name in the description must be carried by the pick, and without one the abstract must back it
"""

from __future__ import annotations

import re

import psycopg

from src.core import llm
from src.mcp.context import Context, prompt
from src.mcp.papers import Filter, Record, allowed, by_authors, read_records
from src.mcp.search import rank
from src.report.check import names_in

CANDIDATES = 5
# title matches on a name come before the ranking, which misses a name it cannot embed
NAMED = 3
NAMES = 2
PICK_TOKENS = 1200
BY_TITLE = "SELECT arxiv_id FROM papers WHERE stage >= 3 AND title ~ %s LIMIT %s"
CALLED = "SELECT 1 FROM citations WHERE target_id = %s AND sentence ~ %s LIMIT 1"
PICK = prompt("recall_pick.md")


def names_of(description: str) -> list[str]:
    """
    The distinctive names a description carries, at most NAMES of them

    Args:
        description: What the caller remembers.

    Returns:
        The names, in order.
    """
    return names_in(description)[:NAMES]


def word(name: str) -> str:
    """
    A whole-word pattern for a name

    Args:
        name: The name.

    Returns:
        The regex.
    """
    return rf"\m{re.escape(name)}\M"


def candidates(ctx: Context, description: str, held: Filter) -> list[Record]:
    """
    The papers the judge chooses among: titles carrying a name first, then the ranking

    Args:
        ctx: The open handles.
        description: What the caller remembers.
        held: The filter.

    Returns:
        The candidates, best first.
    """
    if held.authors and not description.strip():
        ids = by_authors(ctx.dsn, held, CANDIDATES)
        records = read_records(ctx.dsn, ids)
        return [records[a] for a in ids if a in records]
    named: list[str] = []
    with psycopg.connect(ctx.dsn) as conn:
        for name in names_of(description):
            rows = conn.execute(BY_TITLE, (word(name), NAMED * 4)).fetchall()
            named += [a for (a,) in rows]
    records = read_records(ctx.dsn, named)
    kept = [records[a] for a in dict.fromkeys(named) if a in records and allowed(records[a], held)]
    kept = kept[:NAMED]
    ranked, _ = rank(ctx, "focused", description, "", held, CANDIDATES)
    seen = {k.arxiv_id for k in kept}
    return kept + [r for r in ranked if r.arxiv_id not in seen]


def carries(ctx: Context, record: Record, names: list[str]) -> bool:
    """
    Whether a paper carries one of the names, in its title or abstract or in what citers call it

    Args:
        ctx: The open handles.
        record: The paper.
        names: The names from the description.

    Returns:
        True when any name is found, and always when there is no name.
    """
    if not names:
        return True
    text = f"{record.title} {record.abstract}".lower()
    if any(re.search(rf"\b{re.escape(n.lower())}\b", text) for n in names):
        return True
    with psycopg.connect(ctx.dsn) as conn:
        return any(conn.execute(CALLED, (record.arxiv_id, word(n))).fetchone() for n in names)


def pick(
    ctx: Context, description: str, found: list[Record], named: bool
) -> tuple[Record | None, str]:
    """
    The candidate the model says is the paper, kept only when its own ruling on it agrees

    A name the paper carries identifies it, so without one the abstract must state all of what the
    description says.

    Args:
        ctx: The open handles.
        description: What the caller remembers.
        found: The candidates.
        named: Whether the description carries a name, checked against the pick afterwards.

    Returns:
        The match or None, and why there is none.
    """
    listed = "\n\n".join(f"{i}. {r.line(900)}" for i, r in enumerate(found, 1))
    prompt = PICK.format(description=description, candidates=listed)
    parsed = llm.parse_json(ctx.model.ask(prompt, PICK_TOKENS))
    if not isinstance(parsed, dict):
        return None, "the judge returned nothing readable"
    raw = str(parsed.get("match", "0")).strip()
    spot = int(raw) if raw.isdigit() else 0
    given = parsed.get("candidates")
    verdicts: dict = given if isinstance(given, dict) else {}
    if not 1 <= spot <= len(found):
        return None, "none of the candidates is the paper described"
    verdict = verdicts.get(str(spot))
    held: dict = verdict if isinstance(verdict, dict) else {}
    if held.get("same_paper") is False:
        return None, f"the closest is not it: {held.get('why', '')}"
    if not named and str(held.get("stated", "")).lower() != "all":
        return None, f"the closest states too little of the description: {held.get('why', '')}"
    return found[spot - 1], ""


def recall(ctx: Context, description: str, held: Filter) -> tuple[list[Record], Record | None, str]:
    """
    The candidates for a description and the one that survived the judge and the name check

    Args:
        ctx: The open handles.
        description: What the caller remembers of the paper.
        held: The filter.

    Returns:
        The candidates, the match or None, and why there is none.
    """
    found = candidates(ctx, description, held)
    if not found:
        return [], None, "no paper passed the filter"
    names = names_of(description)
    match, why = pick(ctx, description, found, bool(names))
    if match is not None and not carries(ctx, match, names):
        return found, None, f"the closest does not carry the name {', '.join(names)}"
    return found, match, why


def recall_paper(ctx: Context, description: str, held: Filter) -> str:
    """
    The paper a description points at, or a plain statement that none matches

    Args:
        ctx: The open handles.
        description: What the caller remembers of the paper.
        held: The filter.

    Returns:
        The paper with its full abstract, or the no-match line.
    """
    if not description.strip() and not held.authors:
        return "Pass a description, or authors in the filter."
    found, match, why = recall(ctx, description, held)
    if not found:
        return "No paper passed the filter."
    if match is None:
        return f"No match among the {len(found)} closest papers: {why}."
    return match.line(10_000)
