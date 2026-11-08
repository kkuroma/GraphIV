"""
One paper as numbered paragraphs, and the paragraphs a question needs, by cosine and BM25 fused

Math keeps its LaTeX alttext and a table keeps its caption and rows, since results live there
A quote is checked against the one paragraph it came from, so a paragraph id is the unit of evidence
"""

from __future__ import annotations

import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from selectolax.lexbor import LexborHTMLParser, LexborNode

# a paragraph shorter than this joins the next one in its section
MIN_CHARS = 120
# pdf text has no paragraph markup, so it is cut into windows of about this size
PDF_WINDOW = 1200
TABLE_ROWS = 40
HEADING_CHARS = 80
# rank fusion constant, the usual value for reciprocal rank fusion
FUSE = 60
BM25_K = 1.2
BM25_B = 0.75
WORD = re.compile(r"[a-z0-9]+")
SKIPPED = "section.ltx_bibliography, nav, header, footer"
READABLE = "p.ltx_p, figure.ltx_table, figure.ltx_figure > figcaption"


@dataclass
class Chunk:
    """
    One paragraph, table or caption, under the heading of the section it sits in
    """

    pid: str
    section: str
    text: str


def heading(node: LexborNode | None) -> str:
    """
    The title of the nearest section around a node

    Args:
        node: Where to start looking.

    Returns:
        The heading text, empty when the node sits outside every section.
    """
    while node is not None:
        if node.tag == "section":
            head = node.css_first("h1, h2, h3, h4, h5, h6")
            if head is not None:
                return " ".join(head.text().split())[:HEADING_CHARS]
        node = node.parent
    return ""


def in_float(node: LexborNode) -> bool:
    """
    Whether a paragraph belongs to a figure or table, which is read as a whole instead

    Args:
        node: The paragraph.

    Returns:
        True inside a figure or caption.
    """
    above = node.parent
    while above is not None:
        if above.tag in ("figure", "figcaption"):
            return True
        above = above.parent
    return False


def table_text(node: LexborNode) -> str:
    """
    A table as its caption over its rows, one row a line and cells split by a bar

    Args:
        node: The table's figure.

    Returns:
        The text.
    """
    caption = node.css_first("figcaption")
    rows = []
    for row in node.css("tr"):
        cells = [" ".join(cell.text().split()) for cell in row.css("td, th")]
        if any(cells):
            rows.append(" | ".join(cells))
    title = " ".join(caption.text().split()) if caption is not None else ""
    return "\n".join([title] + rows[:TABLE_ROWS]).strip()


def chunk_html(path: Path) -> list[Chunk]:
    """
    A LaTeXML render as paragraphs, tables and figure captions in reading order

    Args:
        path: The html file.

    Returns:
        The chunks, numbered p1 onward.
    """
    tree = LexborHTMLParser(path.read_text(encoding="utf-8", errors="ignore"))
    for node in tree.css("math"):
        node.replace_with(f" ${node.attributes.get('alttext') or ''}$ ")
    for node in tree.css(SKIPPED):
        node.decompose()
    parts = []
    for node in tree.css(READABLE):
        if node.tag == "p" and in_float(node):
            continue
        text = table_text(node) if node.tag == "figure" else " ".join(node.text().split())
        if text:
            parts.append((heading(node), text))
    return number(parts)


def chunk_pdf(path: Path, pdftotext: str) -> list[Chunk]:
    """
    A pdf's text before its references, cut into windows at blank lines

    Args:
        path: The pdf file.
        pdftotext: The poppler binary.

    Returns:
        The chunks, with no section headings since the text carries none.
    """
    done = subprocess.run(
        [pdftotext, str(path), "-"], capture_output=True, text=True, timeout=120, check=False
    )
    body = re.split(r"\n(?:References|REFERENCES|Bibliography)\s*\n", done.stdout)[0]
    parts, held = [], ""
    body = re.sub(r"(\w)-\n(\w)", r"\1\2", body)
    for block in re.split(r"\n\s*\n", body):
        held = f"{held} {' '.join(block.split())}".strip()
        if len(held) >= PDF_WINDOW:
            parts.append(("", held))
            held = ""
    if held:
        parts.append(("", held))
    return number(parts)


def number(parts: list[tuple[str, str]]) -> list[Chunk]:
    """
    Joins short paragraphs onto their neighbor and numbers the rest

    Args:
        parts: Section and text pairs in reading order.

    Returns:
        The chunks.
    """
    chunks: list[Chunk] = []
    for section, text in parts:
        if chunks and len(chunks[-1].text) < MIN_CHARS and chunks[-1].section == section:
            chunks[-1].text += " " + text
            continue
        chunks.append(Chunk("", section, text))
    for spot, chunk in enumerate(chunks, 1):
        chunk.pid = f"p{spot}"
    return chunks


def bm25(question: str, chunks: list[Chunk]) -> list[float]:
    """
    BM25 of the question's words against every chunk, with the chunks as the collection

    Args:
        question: What is asked.
        chunks: The paper.

    Returns:
        One score per chunk.
    """
    docs = [Counter(WORD.findall(c.text.lower())) for c in chunks]
    terms = set(WORD.findall(question.lower()))
    size = len(docs)
    mean = sum(sum(d.values()) for d in docs) / max(size, 1)
    freq = {t: sum(1 for d in docs if t in d) for t in terms}
    out = []
    for doc in docs:
        length, score = sum(doc.values()), 0.0
        for term in terms:
            if doc[term]:
                idf = math.log(1 + (size - freq[term] + 0.5) / (freq[term] + 0.5))
                score += (
                    idf
                    * doc[term]
                    * (BM25_K + 1)
                    / (doc[term] + BM25_K * (1 - BM25_B + BM25_B * length / mean))
                )
        out.append(score)
    return out


def rank(
    question: str, chunks: list[Chunk], vectors: list[list[float]], query: list[float]
) -> list[int]:
    """
    Chunk positions in order of reciprocal rank fusion of cosine and BM25

    Args:
        question: What is asked.
        chunks: The paper.
        vectors: One unit vector per chunk.
        query: The question's unit vector.

    Returns:
        Every chunk position, best first.
    """
    cosine = [sum(a * b for a, b in zip(v, query, strict=True)) for v in vectors]
    lexical = bm25(question, chunks)
    order = range(len(chunks))
    by_cosine = {i: r for r, i in enumerate(sorted(order, key=lambda i: -cosine[i]))}
    by_words = {i: r for r, i in enumerate(sorted(order, key=lambda i: -lexical[i]))}
    return sorted(order, key=lambda i: -(1 / (FUSE + by_cosine[i]) + 1 / (FUSE + by_words[i])))


def around(hits: list[int], count: int, reach: int) -> list[int]:
    """
    The hits with their neighbors on each side, in reading order

    A claim and its number often span two paragraphs, which is why a hit brings its neighbors.

    Args:
        hits: Chunk positions, best first.
        count: How many chunks there are.
        reach: Neighbors kept on each side.

    Returns:
        The positions to show.
    """
    return sorted({j for i in hits for j in range(i - reach, i + reach + 1) if 0 <= j < count})
