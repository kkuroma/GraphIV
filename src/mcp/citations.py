"""
What one paper cites, or what cites it, each with its classified intent and the sentence itself

A paper citing another several times appears once, under its strongest intent
"""

from __future__ import annotations

import psycopg

from src.mcp.papers import clean, read_records
from src.mcp.runs import Run

SENTENCE_CHARS = 300
MOST = 100
ORDER = ("critiques", "extends", "compares", "uses", "motivates", "background")
CITED_BY = """
SELECT c.source_id, p.date, p.title, c.intent, c.intent_score, c.sentence
FROM citations c JOIN papers p ON p.arxiv_id = c.source_id
WHERE c.target_id = %s AND c.stage = 1
"""
CITES = """
SELECT c.target_id, p.date, p.title, c.intent, c.intent_score, c.sentence
FROM citations c JOIN papers p ON p.arxiv_id = c.target_id
WHERE c.source_id = %s AND c.target_id IS NOT NULL AND c.stage = 1
"""


def list_citations(
    dsn: str, run: Run, arxiv_id: str, direction: str, intents: list[str], limit: int
) -> str:
    """
    One paper's citations in one direction, strongest intent first and then newest

    Args:
        dsn: libpq connection string.
        run: The caller's run, which labels every sentence shown so a report can cite it.
        arxiv_id: The paper.
        direction: cited_by for the papers citing it, cites for the papers it cites.
        intents: Intents to keep, every intent when empty.
        limit: How many papers, at most 100.

    Returns:
        One line per paper: id, date, title, intent and the citing sentence.
    """
    if direction not in ("cited_by", "cites"):
        return "direction is cited_by or cites."
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            CITED_BY if direction == "cited_by" else CITES, (arxiv_id.strip(),)
        ).fetchall()
    wanted = {i.strip().lower() for i in intents if i.strip()}
    best: dict[str, tuple] = {}
    for other, when, title, intent, score, sentence in rows:
        intent = intent or "background"
        if wanted and intent not in wanted:
            continue
        place = ORDER.index(intent) if intent in ORDER else len(ORDER)
        held = best.get(other)
        if held is None or (place, -(score or 0)) < (held[0], -(held[5] or 0)):
            best[other] = (place, when, clean(title), intent, clean(sentence), score)
    ordered = sorted(best.items(), key=lambda kv: (kv[1][0], -kv[1][1].toordinal()))
    total = len(ordered)
    shown = ordered[: max(1, min(limit, MOST))]
    me = arxiv_id.strip()
    citers = {other if direction == "cited_by" else me for other, _ in shown}
    keys = {a: r.key for a, r in read_records(dsn, list(citers)).items()}
    lines = []
    for other, (_, when, title, intent, sentence, _) in shown:
        citer, cited = (other, me) if direction == "cited_by" else (me, other)
        label = run.cite(citer, cited, sentence, keys.get(citer, "Paper"))
        lines.append(
            f"{other} | {when} | {title} | {intent}\n  {sentence[:SENTENCE_CHARS]} [{label}]"
        )
    if not lines:
        side = "of" if direction == "cited_by" else "from"
        return f"No labeled citations {side} {arxiv_id} in the store."
    return f"{total} papers in all, {len(lines)} shown\n" + "\n".join(lines)
