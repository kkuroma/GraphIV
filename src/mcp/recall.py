"""
One paper from a description: an arXiv id in it first, else candidates, the model and a name check

The model rules on every candidate, and a pick it ruled is not the paper is no pick
A name in the description must be carried by the pick, and without one the abstract must back it
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import psycopg

from src.core import llm
from src.mcp.context import Context, prompt
from src.mcp.papers import Filter, Record, allowed, read_records
from src.mcp.search import rank
from src.report.check import ARXIV, names_in

CANDIDATES = 5
# title matches on a name come before the ranking, which misses a name it cannot embed
NAMED = 3
NAMES = 2
PICK_TOKENS = 1200
# a title this long, carried whole by the description, is the paper without asking the judge
TITLE_WORDS = 4
BY_TITLE = "SELECT arxiv_id FROM papers WHERE stage >= 3 AND title ~ %s LIMIT %s"
CALLED = "SELECT 1 FROM citations WHERE target_id = %s AND sentence ~ %s LIMIT 1"
PICK = prompt("recall_pick.md")
# the word arXiv beside an id would otherwise count as a name to look up in titles
MENTION = re.compile(r"\barxiv\b:?", re.IGNORECASE)


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


def titled(ctx: Context, description: str, held: Filter) -> list[Record]:
    """
    The papers whose titles carry a name from the description, at most NAMED

    Args:
        ctx: The open handles.
        description: What the caller remembers.
        held: The filter.

    Returns:
        The papers in the order the store returned them.
    """
    named: list[str] = []
    with psycopg.connect(ctx.dsn) as conn:
        for name in names_of(description):
            rows = conn.execute(BY_TITLE, (word(name), NAMED * 4)).fetchall()
            named += [a for (a,) in rows]
    records = read_records(ctx.dsn, named)
    kept = [records[a] for a in dict.fromkeys(named) if a in records and allowed(records[a], held)]
    return kept[:NAMED]


def candidates(ctx: Context, description: str, held: Filter) -> list[Record]:
    """
    The papers the judge chooses among: titles carrying a name first, then the ranking

    The title scan and the ranking run at the same time, since neither needs the other.

    Args:
        ctx: The open handles.
        description: What the caller remembers.
        held: The filter.

    Returns:
        The candidates, best first.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        by_title = pool.submit(titled, ctx, description, held)
        by_rank = pool.submit(rank, ctx, "focused", description, "", held, CANDIDATES)
        kept, (ranked, _) = by_title.result(), by_rank.result()
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


def plain(text: str) -> str:
    """
    Text as lowercase words and digits, for comparing a title written with other punctuation

    Args:
        text: Any text.

    Returns:
        The words joined by single spaces.
    """
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def whole_title(description: str, found: list[Record]) -> Record | None:
    """
    The one candidate whose whole title the description carries, which needs no judge

    Args:
        description: What the caller remembers.
        found: The candidates.

    Returns:
        The paper, or None when no title or several are carried.
    """
    said = f" {plain(description)} "
    hits = [
        r for r in found if len(r.title.split()) >= TITLE_WORDS and f" {plain(r.title)} " in said
    ]
    return hits[0] if len(hits) == 1 else None


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
    titled_one = whole_title(description, found)
    if titled_one is not None:
        return found, titled_one, ""
    names = names_of(description)
    match, why = pick(ctx, description, found, bool(names))
    if match is not None and not carries(ctx, match, names):
        return found, None, f"the closest does not carry the name {', '.join(names)}"
    return found, match, why


def by_id(ctx: Context, description: str, held: Filter) -> tuple[Record | None, list[str]]:
    """
    The paper an arXiv id in the description names, which outranks whatever the words say

    Args:
        ctx: The open handles.
        description: What the caller remembers.
        held: The filter.

    Returns:
        The first id the store holds that passes the filter, and the ids it does not hold.
    """
    ids = list(dict.fromkeys(ARXIV.findall(description)))
    records = read_records(ctx.dsn, ids)
    found = next((records[a] for a in ids if a in records and allowed(records[a], held)), None)
    return found, [a for a in ids if a not in records]


def recall_paper(ctx: Context, description: str, held: Filter) -> str:
    """
    The paper a description points at, or a plain statement that none matches with the candidates

    Args:
        ctx: The open handles.
        description: What the caller remembers of the paper.
        held: The filter.

    Returns:
        The paper with its full abstract, or the no-match line and one line per candidate.
    """
    if not description.strip():
        return "Pass a description. search_papers lists an author's papers."
    named, missing = by_id(ctx, description, held)
    if named is not None:
        return named.line(10_000)
    note = "".join(f"{a} is not in the corpus. " for a in missing)
    words = MENTION.sub(" ", ARXIV.sub(" ", description))
    if not words.strip():
        return f"{note}Pass words describing the paper as well."
    found, match, why = recall(ctx, words, held)
    if not found:
        return f"{note}No paper passed the filter."
    if match is None:
        listed = "\n".join(r.line(0) for r in found)
        return (
            f"{note}No match among the {len(found)} closest papers: {why}.\nCandidates:\n{listed}"
        )
    return note + match.line(10_000)
