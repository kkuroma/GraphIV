"""
The one shape every tool hands a paper back in, and the filter every lookup takes

A filter is applied in the store before ranking, so an excluded paper can never come back
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

import psycopg
from pydantic import BaseModel, Field

# a TeX accent on one letter, as in O\u{g}uz
ACCENT = re.compile(r"\\[`'^\"~=.uvHtcdbrk]\{?\s*([A-Za-z])\}?")
AUTHOR_NAMES = 6
ABSTRACT_CHARS = 600
RECORD = """
SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.date, n.name
FROM papers p
LEFT JOIN topic_tree t ON t.arxiv_id = p.arxiv_id
LEFT JOIN topic_names n ON n.level_0 = t.level_0 AND n.level_1 = t.level_1 AND n.level_2 = t.level_2
WHERE p.arxiv_id = ANY(%s)
"""


class Filter(BaseModel):
    """
    What a lookup is held to: papers left out, authors any of whom must be listed, and a date range
    """

    exclude: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    date_range: list[str | None] = Field(default_factory=lambda: [None, None])

    @property
    def earliest(self) -> date | None:
        """
        The first day allowed

        Returns:
            The date, or None when open.
        """
        low = (self.date_range + [None, None])[0]
        return date.fromisoformat(low) if low else None

    @property
    def latest(self) -> date | None:
        """
        The last day allowed

        Returns:
            The date, or None when open.
        """
        high = (self.date_range + [None, None])[1]
        return date.fromisoformat(high) if high else None


class Record(BaseModel):
    """
    One paper as a tool returns it
    """

    arxiv_id: str
    title: str
    abstract: str
    authors: str
    date: date
    topic: str = ""
    terms: dict[str, float] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """
        The first author's surname in plain letters, which labels this paper's quotes

        Returns:
            The surname, or Paper when the store lists no author.
        """
        first = next((n.strip() for n in re.split(r",|\band\b", self.authors) if n.strip()), "")
        last = first.split()[-1] if first else ""
        decomposed = unicodedata.normalize("NFKD", last)
        plain = "".join(c for c in decomposed if c.isascii() and c.isalpha())
        return plain[:1].upper() + plain[1:] if plain else "Paper"

    def line(self, abstract_chars: int = ABSTRACT_CHARS, names: int = AUTHOR_NAMES) -> str:
        """
        The paper as two lines of text, which is what a caller reads

        Args:
            abstract_chars: How much of the abstract to show, none at zero.
            names: How many authors to list before the count of the rest.

        Returns:
            The text.
        """
        listed = [n.strip() for n in re.split(r",|\band\b", self.authors) if n.strip()]
        shown = ", ".join(listed[:names]) + (
            f" and {len(listed) - names} more" if len(listed) > names else ""
        )
        head = f"{self.arxiv_id} | {self.date} | {self.title} | {shown}" + (
            f" | topic: {self.topic}" if self.topic else ""
        )
        if abstract_chars <= 0:
            return head
        body = (
            self.abstract
            if len(self.abstract) <= abstract_chars
            else self.abstract[:abstract_chars].rsplit(" ", 1)[0] + " ..."
        )
        return f"{head}\n  {body}"


def clean(text: str | None) -> str:
    """
    Text with its whitespace collapsed and TeX accent commands reduced to their letter

    Args:
        text: Raw store text.

    Returns:
        One line.
    """
    plain = ACCENT.sub(r"\1", text or "").replace("{", "").replace("}", "")
    return " ".join(plain.split())


def read_records(dsn: str, arxiv_ids: list[str]) -> dict[str, Record]:
    """
    The records of a list of papers, with each one's topic leaf

    Args:
        dsn: libpq connection string.
        arxiv_ids: The papers.

    Returns:
        One record per id the store holds.
    """
    if not arxiv_ids:
        return {}
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(RECORD, (list(arxiv_ids),)).fetchall()
    return {
        a: Record(
            arxiv_id=a,
            title=clean(t),
            abstract=clean(b),
            authors=clean(n),
            date=d,
            topic=clean(leaf),
        )
        for a, t, b, n, d, leaf in rows
        if d is not None
    }


def allowed(record: Record, held: Filter) -> bool:
    """
    Whether one paper passes a filter

    Args:
        record: The paper.
        held: The filter.

    Returns:
        True when it passes.
    """
    if record.arxiv_id in set(held.exclude):
        return False
    if held.earliest and record.date < held.earliest:
        return False
    if held.latest and record.date > held.latest:
        return False
    if held.authors:
        listed = record.authors.lower()
        return any(name.strip().lower() in listed for name in held.authors if name.strip())
    return True


def by_authors(dsn: str, held: Filter, limit: int) -> list[str]:
    """
    Papers listing any of the filter's authors, inside its dates, newest first

    Args:
        dsn: libpq connection string.
        held: The filter, whose authors are required.
        limit: The most to return.

    Returns:
        The ids.
    """
    names = [f"%{n.strip()}%" for n in held.authors if n.strip()]
    low, high = held.earliest or date.min, held.latest or date.max
    query = (
        "SELECT arxiv_id FROM papers WHERE stage >= 3 AND authors ILIKE ANY(%s) "
        "AND date BETWEEN %s AND %s ORDER BY date DESC LIMIT %s"
    )
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(query, (names, low, high, limit + len(held.exclude))).fetchall()
    out = [a for (a,) in rows if a not in set(held.exclude)]
    return out[:limit]
