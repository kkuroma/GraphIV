"""
Reads papers through the paragraphs a question needs, so the reader sees evidence, not a paper

A quote is kept only when it sits in the paragraph it names, which is what a citation points at
With a schema, the same retrieval fills one table row per method, each cell carrying its quote
"""

from __future__ import annotations

import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.core import llm
from src.core.fetch import html_file, pdf_file
from src.mcp.context import Context, prompt
from src.mcp.papers import Record, read_records
from src.mcp.runs import Evidence, Run
from src.report.check import numbers_in
from src.report.chunks import Chunk, around, chunk_html, chunk_pdf, rank

READERS = 4
HITS = 8
REACH = 1
COLUMN_HITS = 3
ANSWER_TOKENS = 3000
TABLE_TOKENS = 2500
KEPT_PAPERS = 300
ANSWER = prompt("read_answer.md")
TABLE = prompt("read_table.md")


class Column(BaseModel):
    """
    One column of a comparison table the caller designed
    """

    name: str
    description: str = ""
    kind: str = "text"
    options: list[str] = Field(default_factory=list)


class Schema(BaseModel):
    """
    The columns a table compares methods on
    """

    columns: list[Column]


@dataclass
class Paper:
    """
    One paper's chunks and their vectors, kept for the life of the process
    """

    record: Record
    chunks: list[Chunk]
    vectors: list[list[float]]
    source: str


def fold(text: str) -> str:
    """
    Text compared for quoting, with typography and case folded away

    Args:
        text: Any text.

    Returns:
        The folded text.
    """
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "\xa0": " "})
    return " ".join(unicodedata.normalize("NFKC", text).translate(table).split()).lower()


def open_paper(ctx: Context, arxiv_id: str) -> Paper | None:
    """
    One paper split into paragraphs and embedded, from the cache when it was opened before

    Args:
        ctx: The open handles.
        arxiv_id: The paper.

    Returns:
        The paper, or None when the store does not hold it.
    """
    if arxiv_id in ctx.chunks:
        return ctx.chunks[arxiv_id]
    record = read_records(ctx.dsn, [arxiv_id]).get(arxiv_id)
    if record is None:
        return None
    chunks, source = [], "abstract only"
    html = html_file(ctx.html, arxiv_id, ctx.fetch)
    if html is not None:
        chunks, source = chunk_html(html), "html"
    if len(chunks) < 3:
        pdf = pdf_file(ctx.pdfs, arxiv_id, ctx.fetch)
        if pdf is not None:
            chunks, source = chunk_pdf(pdf, ctx.pdftotext), "pdf"
    if not chunks:
        chunks, source = [Chunk("p1", "Abstract", record.abstract)], "abstract only"
    paper = Paper(record, chunks, ctx.embed_documents([c.text for c in chunks]), source)
    if len(ctx.chunks) >= KEPT_PAPERS:
        ctx.chunks.pop(next(iter(ctx.chunks)))
    ctx.chunks[arxiv_id] = paper
    return paper


def passages(paper: Paper, spots: list[int]) -> str:
    """
    The chunks shown to the reader, each under its id and section

    Args:
        paper: The paper.
        spots: Chunk positions.

    Returns:
        The text.
    """
    return "\n\n".join(
        f"[{paper.chunks[i].pid}] ({paper.chunks[i].section or 'body'}) {paper.chunks[i].text}"
        for i in spots
    )


def ground(paper: Paper, spots: list[int], quote: str, pid: str) -> str:
    """
    The id of the shown paragraph a quote really sits in

    Args:
        paper: The paper.
        spots: The chunk positions the reader saw.
        quote: The quoted words.
        pid: The paragraph the reader named.

    Returns:
        The named id when it holds the quote, another shown one that does, or empty.
    """
    shown = {paper.chunks[i].pid: fold(paper.chunks[i].text) for i in spots}
    words = fold(quote)
    if words in shown.get(pid, ""):
        return pid
    return next((p for p, text in shown.items() if words in text), "")


def answer(ctx: Context, paper: Paper, question: str, run: Run) -> str:
    """
    The reader's grounded answer to one question about one paper

    A claim whose quote is not in the passages it names, or in any passage shown, is dropped.

    Args:
        ctx: The open handles.
        paper: The paper.
        question: What the caller needs from it.
        run: Where the evidence is recorded.

    Returns:
        One claim a line, each with its quote and marker.
    """
    query = ctx.embed(question)
    spots = around(
        rank(question, paper.chunks, paper.vectors, query)[:HITS], len(paper.chunks), REACH
    )
    prompt = ANSWER.format(
        title=paper.record.title,
        abstract=paper.record.abstract,
        passages=passages(paper, spots),
        question=question,
    )
    parsed = llm.parse_json(ctx.model.ask(prompt, ANSWER_TOKENS))
    if not isinstance(parsed, dict):
        return "The reader returned nothing readable."
    found = parsed.get("claims")
    claims: list[Any] = found if isinstance(found, list) else []
    by_pid = {c.pid: c for c in paper.chunks}
    lines, dropped = [], 0
    for item in claims:
        if not isinstance(item, dict):
            continue
        claim, quote = str(item.get("claim") or "").strip(), str(item.get("quote") or "").strip()
        held = ground(paper, spots, quote, str(item.get("pid") or "")) if quote else ""
        if not held or numbers_in(claim) - numbers_in(by_pid[held].text):
            dropped += 1
            continue
        chunk = by_pid[held]
        value = str(item.get("value") or "").strip()
        # a result is kept only when its value is a number its own paragraph states
        result = (
            bool(value)
            and bool(numbers_in(value))
            and not numbers_in(value) - numbers_in(chunk.text)
        )
        label = run.keep(
            Evidence(
                arxiv_id=paper.record.arxiv_id,
                pid=held,
                section=chunk.section,
                text=chunk.text,
                quote=quote,
                claim=claim,
                metric=str(item.get("metric") or "").strip() if result else "",
                value=value if result else "",
                dataset=str(item.get("dataset") or "").strip() if result else "",
            ),
            paper.record.key,
        )
        lines.append(f"- {claim} [{label}]")
    note = str(parsed.get("note") or "").strip()
    if note:
        lines.append(f"Note: {note}")
    if dropped:
        lines.append(
            f"({dropped} claims dropped: a quote or number in them is not in the paragraph cited)"
        )
    return "\n".join(lines) or "The passages do not address the question."


def describe(schema: Schema) -> str:
    """
    The columns as the table prompt lists them

    Args:
        schema: The table design.

    Returns:
        One line per column.
    """
    lines = []
    for col in schema.columns:
        options = f" Options: {', '.join(col.options)}." if col.options else ""
        lines.append(f"- {col.name} ({col.kind}): {col.description}.{options}")
    return "\n".join(lines)


def fill(ctx: Context, paper: Paper, schema: Schema, question: str, run: Run) -> str:
    """
    One paper's rows of the caller's table, every cell grounded in a paragraph

    Args:
        ctx: The open handles.
        paper: The paper.
        schema: The table design.
        question: What the table is for, which steers retrieval with each column.
        run: Where the rows and their evidence are recorded.

    Returns:
        The rows as text.
    """
    hits: list[int] = []
    for col in schema.columns:
        ask = f"{question} {col.name.replace('_', ' ')} {col.description} {' '.join(col.options)}"
        hits += rank(ask, paper.chunks, paper.vectors, ctx.embed(ask))[:COLUMN_HITS]
    spots = around(list(dict.fromkeys(hits)), len(paper.chunks), 0)
    prompt = TABLE.format(
        title=paper.record.title,
        abstract=paper.record.abstract,
        passages=passages(paper, spots),
        columns=describe(schema),
    )
    parsed = llm.parse_json(ctx.model.ask(prompt, TABLE_TOKENS))
    methods = parsed.get("methods") if isinstance(parsed, dict) else None
    if not isinstance(methods, list):
        return "The reader returned no row."
    by_pid = {c.pid: c for c in paper.chunks}
    lines = []
    for method in methods:
        if not isinstance(method, dict):
            continue
        given = method.get("cells")
        cells: dict[str, Any] = given if isinstance(given, dict) else {}
        row: dict[str, dict[str, str]] = {}
        for col in schema.columns:
            named = cells.get(col.name)
            cell: dict[str, Any] = named if isinstance(named, dict) else {}
            value, quote = (
                str(cell.get("value") or "n/a").strip(),
                str(cell.get("quote") or "").strip(),
            )
            held = ground(paper, spots, quote, str(cell.get("pid") or "")) if quote else ""
            label = ""
            if held:
                chunk = by_pid[held]
                label = run.keep(
                    Evidence(
                        arxiv_id=paper.record.arxiv_id,
                        pid=held,
                        section=chunk.section,
                        text=chunk.text,
                        quote=quote,
                    ),
                    paper.record.key,
                )
            row[col.name] = {"value": value, "quote": quote if held else "", "label": label}
        name = str(method.get("name") or paper.record.title).strip()
        run.add_row(
            schema.model_dump(), {"arxiv_id": paper.record.arxiv_id, "method": name, "cells": row}
        )
        shown = "; ".join(
            f"{k}: {v['value']}" + (f" [{v['label']}]" if v["label"] else "")
            for k, v in row.items()
        )
        lines.append(f"{name}: {shown}")
    return "\n".join(lines) or "The reader returned no row."


def read_paper(
    ctx: Context, run: Run, arxiv_ids: list[str], question: str, schema: Schema | None
) -> str:
    """
    Reads up to a batch of papers at once, answering the question or filling the table for each

    Args:
        ctx: The open handles.
        run: The caller's run.
        arxiv_ids: The papers.
        question: What the caller needs from each.
        schema: Table columns, or None for a plain answer.

    Returns:
        One block per paper.
    """
    ids = list(dict.fromkeys(a.strip() for a in arxiv_ids if a.strip()))
    if not ids:
        return "Pass at least one arxiv id."
    if not question.strip():
        return "Pass a question."

    def one(arxiv_id: str) -> str:
        paper = open_paper(ctx, arxiv_id)
        if paper is None:
            return f"{arxiv_id} is not in the corpus."
        run.read(paper.record)
        body = (
            fill(ctx, paper, schema, question, run) if schema else answer(ctx, paper, question, run)
        )
        source = f"read from {paper.source}, {len(paper.chunks)} paragraphs"
        return f"{paper.record.line(0)}\n{source}\n{body}"

    with ThreadPoolExecutor(max_workers=READERS) as pool:
        return "\n\n".join(pool.map(one, ids))


def parse_schema(raw: dict | str | None) -> Schema | None:
    """
    The caller's schema, accepted as an object or as json text

    Args:
        raw: What the caller passed.

    Returns:
        The schema, or None when nothing usable was passed.
    """
    if not raw:
        return None
    data = json.loads(raw) if isinstance(raw, str) else raw
    schema = Schema.model_validate(data)
    return schema if schema.columns else None
