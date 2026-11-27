"""
Reads papers through the sentences a question needs, so the reader points at evidence, not writes it

A plain question picks sentences, and a schema answers typed fields that each cite one sentence
An answer its sentence does not support is dropped, so every marker leads to the paper's own words
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.core import llm
from src.core.fetch import html_file, pdf_file
from src.mcp.citations import builders
from src.mcp.context import Context, prompt
from src.mcp.papers import Record, read_records
from src.mcp.runs import Evidence, Run
from src.report.check import NUMBER
from src.report.chunks import Chunk, Sentence, around, chunk_html, chunk_pdf, rank, sentences

READERS = 4
HITS = 8
REACH = 1
# each field ranks its own paragraphs, without neighbors, and the field windows are merged
FIELD_HITS = 3
PICKS = 6
MOST_FIELDS = 8
MOST_OPTIONS = 12
POINT_TOKENS = 300
TYPED_TOKENS = 800
KEPT_PAPERS = 300
# newer papers building on a table's rows, listed under each table read
BUILDERS = 10
# the top of that list a deep report must have read
MUST_FORWARD = 4
TYPES = ("bool", "float", "choice", "text", "sentences")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
IDS = re.compile(r'"ids"\s*:\s*\[([^\]]*)\]')
POINT = prompt("read_point.md")
TYPED = prompt("read_typed.md")


class SchemaField(BaseModel):
    """
    One question a schema asks of every paper, and the type its answer takes
    """

    name: str
    type: str
    ask: str
    options: list[str] = Field(default_factory=list)


class Schema(BaseModel):
    """
    The fields the caller asks of every paper
    """

    fields: list[SchemaField]


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


def schema_of(raw: dict[str, Any] | str) -> Schema:
    """
    The caller's schema, checked against the limits the reader keeps to

    Args:
        raw: The schema as an object or as json text.

    Returns:
        The schema.

    Raises:
        ValueError: When a field breaks a limit, naming it.
    """
    schema = Schema.model_validate(json.loads(raw) if isinstance(raw, str) else raw)
    if not 1 <= len(schema.fields) <= MOST_FIELDS:
        raise ValueError(f"pass 1 to {MOST_FIELDS} fields")
    names = [f.name for f in schema.fields]
    for f in schema.fields:
        if not NAME.match(f.name) or names.count(f.name) > 1:
            raise ValueError(f"field names are unique and snake_case, unlike {f.name!r}")
        if f.type not in TYPES:
            raise ValueError(f"{f.name} has type {f.type!r}, not one of {', '.join(TYPES)}")
        if not f.ask.strip():
            raise ValueError(f"{f.name} has no ask")
        if f.type == "choice" and not 2 <= len(f.options) <= MOST_OPTIONS:
            raise ValueError(f"{f.name} is a choice, which takes 2 to {MOST_OPTIONS} options")
    return schema


def stated(value: float, sentence: str) -> bool:
    """
    Whether a sentence writes a number, separators and sign aside

    Args:
        value: The number.
        sentence: The sentence.

    Returns:
        True when the sentence carries it.
    """
    found = set()
    for raw in NUMBER.findall(sentence):
        try:
            found.add(round(float(raw.replace(",", "")), 9))
        except ValueError:
            continue
    return round(abs(value), 9) in found


def supported(field: SchemaField, value: Any, sentence: str) -> str | None:
    """
    An answer as the reply shows it, when it has the field's type and its sentence backs it

    Args:
        field: The field.
        value: What the model answered.
        sentence: The sentence it cited.

    Returns:
        The answer as text, or None when it is dropped.
    """
    if field.type == "bool":
        return str(value).lower() if isinstance(value, bool) else None
    if field.type == "float" and isinstance(value, str):
        # a number written with its uncertainty, like 69.01(2)(21), stays as the sentence writes it
        copied = value.strip()
        return copied if NUMBER.match(copied) and fold(copied) in fold(sentence) else None
    if field.type == "float":
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        return str(value) if number and stated(float(value), sentence) else None
    if field.type == "choice":
        return next((o for o in field.options if o.lower() == str(value).strip().lower()), None)
    phrase = str(value).strip()
    return phrase if phrase and fold(phrase) in fold(sentence) else None


def spot(raw: Any, count: int) -> int | None:
    """
    A sentence number from a reply, accepting the s prefix the plain listing uses

    Args:
        raw: What the model wrote.
        count: How many sentences were shown.

    Returns:
        The position, or None when it names no sentence shown.
    """
    digits = str(raw).strip().removeprefix("s")
    return int(digits) if digits.isdigit() and int(digits) < count else None


def picked(raw: Any, count: int) -> tuple[list[int], int]:
    """
    The distinct sentences a list of numbers names, at most PICKS, and how many were not shown

    Args:
        raw: The list the model wrote.
        count: How many sentences were shown.

    Returns:
        The positions in order, and the count dropped.
    """
    out, dropped = [], 0
    for item in raw if isinstance(raw, list) else []:
        at = spot(item, count)
        if at is None:
            dropped += 1
        elif at not in out:
            out.append(at)
    return out[:PICKS], dropped


def keep(run: Run, paper: Paper, sentence: Sentence, claim: str = "") -> str:
    """
    Records a sentence as evidence under a label

    Args:
        run: The caller's run.
        paper: The paper.
        sentence: The sentence.
        claim: The answer it backs, empty for a plain pick.

    Returns:
        The label.
    """
    chunk = next(c for c in paper.chunks if c.pid == sentence.pid)
    found = Evidence(
        arxiv_id=paper.record.arxiv_id,
        pid=chunk.pid,
        section=chunk.section,
        text=chunk.text,
        quote=sentence.text,
        claim=claim,
    )
    return run.keep(found, paper.record.key)


def point(ctx: Context, paper: Paper, question: str, run: Run) -> str:
    """
    The sentences of one paper that answer a plain question

    Args:
        ctx: The open handles.
        paper: The paper.
        question: What the caller needs from it.
        run: Where the evidence is recorded.

    Returns:
        One sentence a line, each with its marker.
    """
    order = rank(question, paper.chunks, paper.vectors, ctx.embed(question))
    shown = sentences(paper.chunks, around(order[:HITS], len(paper.chunks), REACH))
    listing = "\n".join(f"[s{i}] {s.text}" for i, s in enumerate(shown))
    reply = ctx.model.ask(POINT.format(sentences=listing, question=question), POINT_TOKENS)
    parsed = llm.parse_json(reply)
    if not isinstance(parsed, dict):
        # a note running past the budget cuts the json, but the ids before it are complete
        cut = IDS.search(reply)
        if cut is None:
            return "The reader returned nothing readable."
        parsed = {"ids": re.findall(r"s?\d+", cut.group(1))}
    spots, dropped = picked(parsed.get("ids"), len(shown))
    lines = [f"- {shown[i].text}{run.mark(keep(run, paper, shown[i]))}" for i in spots]
    if not lines:
        note = str(parsed.get("note") or "").strip()
        lines.append(f"Note: {note}" if note else "No sentence shown answers the question.")
    if dropped:
        lines.append(f"({dropped} picks dropped: not a sentence shown)")
    return "\n".join(lines)


def describe(schema: Schema) -> str:
    """
    The fields as the typed prompt lists them

    Args:
        schema: The fields.

    Returns:
        One line per field.
    """
    lines = []
    for f in schema.fields:
        kind = f"choice: {' | '.join(f.options)}" if f.type == "choice" else f.type
        lines.append(f"- {f.name} ({kind}): {f.ask}")
    return "\n".join(lines)


def columns(schema: Schema) -> dict[str, Any]:
    """
    The schema as the webui draws a table's columns

    Args:
        schema: The fields.

    Returns:
        The columns.
    """
    return {
        "columns": [
            {"name": f.name, "kind": f.type, "description": f.ask, "options": f.options}
            for f in schema.fields
        ]
    }


def answer(ctx: Context, paper: Paper, schema: Schema, question: str, run: Run) -> str:
    """
    One paper answered field by field, every answer citing the sentence that backs it

    Args:
        ctx: The open handles.
        paper: The paper.
        schema: The fields.
        question: What the fields are for, which steers each field's paragraphs.
        run: Where the answers and their evidence are recorded.

    Returns:
        One line per field, and the sentences of a sentences field under it.
    """
    hits: list[int] = []
    for f in schema.fields:
        ask = f"{question} {f.ask}"
        hits += rank(ask, paper.chunks, paper.vectors, ctx.embed(ask))[:FIELD_HITS]
    shown = sentences(paper.chunks, around(list(dict.fromkeys(hits)), len(paper.chunks), 0))
    listing = "\n".join(f"[{i}] {s.text}" for i, s in enumerate(shown))
    reply = ctx.model.ask(TYPED.format(sentences=listing, fields=describe(schema)), TYPED_TOKENS)
    parsed = llm.parse_json(reply)
    if not isinstance(parsed, dict):
        return "The reader returned nothing readable."
    # a reply key may echo the listing, as in "limits (sentences)"
    given = {str(k).split(" (")[0]: v for k, v in parsed.items()}
    lines, cells, dropped = [], {}, 0
    for f in schema.fields:
        got = given.get(f.name)
        held: dict[str, Any] = got if isinstance(got, dict) else {}
        cells[f.name] = {"value": "n/a", "quote": "", "label": ""}
        if f.type == "sentences":
            spots, lost = picked(held.get("ids"), len(shown))
            dropped += lost
            labels = [keep(run, paper, shown[i], f.name) for i in spots]
            lines.append(f"{f.name}:" if spots else f"{f.name}: not stated")
            lines += [f"- {shown[i].text}{run.mark(m)}" for i, m in zip(spots, labels, strict=True)]
            if spots:
                first = shown[spots[0]].text
                cells[f.name] = {"value": first, "quote": first, "label": labels[0]}
            continue
        at = spot(held.get("id"), len(shown))
        value = held.get("value")
        shown_answer = None if at is None else supported(f, value, shown[at].text)
        if at is None or shown_answer is None:
            dropped += value is not None
            lines.append(f"{f.name}: not stated")
            continue
        label = keep(run, paper, shown[at], f"{f.name}: {shown_answer}")
        lines.append(f"{f.name}: {shown_answer}{run.mark(label)}")
        cells[f.name] = {"value": shown_answer, "quote": shown[at].text, "label": label}
    run.add_row(
        columns(schema),
        {"arxiv_id": paper.record.arxiv_id, "method": paper.record.title, "cells": cells},
    )
    if dropped:
        lines.append(f"({dropped} answers dropped: not backed by the sentence cited)")
    return "\n".join(lines)


def held_schema(run: Run, raw: dict[str, Any] | str | None) -> dict[str, Any] | str | None:
    """
    The schema a caller passed, or the fields of the table it names by id

    Args:
        run: The caller's run.
        raw: A schema, a table id like T1, or None.

    Returns:
        A schema as fields, or the raw value when it is not a table id.

    Raises:
        ValueError: When it names a table the session lacks.
    """
    # a caller may send the id json-quoted, as "\"T1\""
    tid = raw.strip().strip("\"'").strip() if isinstance(raw, str) else ""
    if not re.fullmatch(r"[Tt]\d+", tid):
        return raw
    table = run.table(tid)
    if table is None:
        raise ValueError(f"this session has no table {tid}")
    return {
        "fields": [
            {"name": c["name"], "type": c["kind"], "ask": c["description"], "options": c["options"]}
            for c in table["schema"]["columns"]
        ]
    }


def read_paper(
    ctx: Context,
    run: Run,
    arxiv_ids: list[str],
    question: str,
    raw_schema: dict[str, Any] | str | None,
) -> str:
    """
    Reads a batch of papers, picking sentences for the question or answering the schema for each

    At deep effort, the papers a report compares are read once with its table schema, so a plain
    read of several papers is refused until the session holds a table.

    Args:
        ctx: The open handles.
        run: The caller's run.
        arxiv_ids: The papers.
        question: What the caller needs from each.
        raw_schema: The fields as the caller passed them, a table id to reuse its fields, or None.

    Returns:
        One block per paper, and the table its rows joined.
    """
    ids = list(dict.fromkeys(a.strip() for a in arxiv_ids if a.strip()))
    if not ids:
        return "Pass at least one arxiv id."
    if not question.strip():
        return "Pass a question."
    if run.tracked and run.effort == "deep" and not raw_schema and len(ids) > 1 and not run.tables:
        return (
            "Not read. At deep effort the papers you compare are read once, with your schema: "
            "design it from the titles and abstracts you have and pass it as schema. "
            "A plain read takes one paper, such as the named paper or an open-area search's hit."
        )
    try:
        resolved = held_schema(run, raw_schema)
        schema = schema_of(resolved) if resolved else None
    except ValueError as error:
        return f"The schema is not usable: {error}."

    def one(arxiv_id: str) -> str:
        paper = open_paper(ctx, arxiv_id)
        if paper is None:
            return f"{arxiv_id} is not in the corpus."
        run.read(paper.record)
        body = (
            answer(ctx, paper, schema, question, run)
            if schema
            else point(ctx, paper, question, run)
        )
        source = f"read from {paper.source}, {len(paper.chunks)} paragraphs"
        return f"{paper.record.line(0)}\n{source}\n{body}"

    with ThreadPoolExecutor(max_workers=READERS) as pool:
        reply = "\n\n".join(pool.map(one, ids))
    tid = run.table_for(columns(schema)) if schema and run.tracked else None
    if tid is None:
        return reply
    table = run.table(tid)
    rows = len(table["rows"]) if table else 0
    held = [r["arxiv_id"] for r in table["rows"]] if table else []
    newer = builders(ctx.dsn, held, set(run.papers), ctx.embed(run.question or question), BUILDERS)
    # fixed at the first table read, so reading them does not put new ones on the list
    if not run.forward:
        run.forward = [line.split(" | ")[0] for line in newer[:MUST_FORWARD]]
    after = (
        "\n\nNewer papers that extend, critique or compare against this table's papers, closest "
        "to the question first:\n" + "\n".join(newer)
        if newer
        else ""
    )
    owed = [a for a in run.forward if a not in run.papers]
    if run.effort == "deep" and owed:
        after += f"\nAt deep effort the report needs these read: {', '.join(owed)}."
    return (
        f"{reply}{after}\n\nThese rows are in Table {tid}, {rows} papers so far. "
        f'Read more papers into it with schema="{tid}". Place it with {{"type": "table", "table": '
        f'"{tid}"}}, and pivot it with {{"type": "coverage", "table": "{tid}", "rows": <a choice '
        'field>, "columns": [<bool fields>] or [<one choice field>]}.'
    )
