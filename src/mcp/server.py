"""
The six tools and both transports, holding the store, the graph and the model for the process

The docstrings here are what the calling model reads, so they say what a tool is for and when
A call passing a session_id that start_session issued is logged into that run, any other is not
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from src.core.config import load_config
from src.mcp import citations, graph, lineages, read, recall, runs, search
from src.mcp.context import config_path, load
from src.mcp.dashboard import mount
from src.mcp.papers import Filter, read_records
from src.report.check import (
    ARXIV,
    EFFORTS,
    Held,
    blocks_of,
    check,
    cites_of,
    contract,
    fold,
    lined,
    name_of,
    place_tables,
    summary,
)
from src.report.check import patch as patch_draft
from src.report.render import render

LOG = logging.getLogger(__name__)

mcp = FastMCP("graphiv")
DEEP = (
    " Deep effort: before save_report, this session needs a table from read_paper with a schema "
    "(a choice field for the strand of work and bool fields for kinds of evidence), a "
    "trace_lineages call whose citing sentences at least three findings cite, and a focused "
    "search checking each open area with one of its nearest hits read; the report places the "
    "table, its coverage, the poset and the open blocks."
)


# an instance set read only writes nothing, so its runs live in memory alone
READONLY = os.environ.get("GRAPHIV_READONLY") == "1"
UNKNOWN = "Unknown session_id {session}. Call start_session for a new one."
UNTRACKED = (
    "\n\nThis call passed no session_id, so it was not recorded and gives no markers a report can "
    "cite. Repeat it with the session_id start_session returned if the report needs it."
)
# start_session's two searches of the question, and how far back the recent one reaches
OPENING = 30
RECENT_DAYS = 730
# the top of each opening search a deep report must have read
MUST_READ = 4


def root() -> Path | None:
    """
    Where runs are written, or None on a read-only instance

    Returns:
        The runs directory or None.
    """
    return None if READONLY else load().runs


async def logged(session_id: str, tool: str, args: dict[str, Any], body: Any) -> str:
    """
    Runs one tool body off the event loop, recording the call when it names an open session

    Args:
        session_id: The id start_session issued, or empty for a call nothing records.
        tool: The tool name.
        args: What the caller passed.
        body: A callable taking the run and returning the reply.

    Returns:
        The reply, or why the session id was refused.
    """
    LOG.info("call %s in %s", tool, session_id or "no session")
    if session_id.strip():
        found = runs.find(root(), session_id.strip())
        if found is None:
            return UNKNOWN.format(session=session_id)
        run = found
    else:
        run = runs.Run(session="", tracked=False)
    began = time.time()
    reply = await to_thread.run_sync(body, run)
    if run.tracked:
        run.log(tool, args, reply, began)
        return reply
    return reply + UNTRACKED


def sources(run: runs.Run) -> dict[str, str]:
    """
    The folded full text behind each quote label, for papers whose text the server still holds

    Args:
        run: The run.

    Returns:
        Text by label.
    """
    opened = load().chunks
    whole: dict[str, str] = {}
    out = {}
    for label, held_quote in run.evidence.items():
        paper = opened.get(held_quote.arxiv_id)
        if paper is None:
            continue
        if held_quote.arxiv_id not in whole:
            parts = [paper.record.title, paper.record.abstract] + [c.text for c in paper.chunks]
            whole[held_quote.arxiv_id] = fold(" ".join(parts))
        out[label] = whole[held_quote.arxiv_id]
    return out


def held(filter: dict[str, Any] | None) -> Filter:
    """
    The caller's filter, empty when none was passed

    Args:
        filter: The filter object.

    Returns:
        The filter.
    """
    return Filter.model_validate(filter or {})


@mcp.tool()
async def start_session(title: str, question: str = "", effort: str = "quick") -> str:
    """
    Opens a study and returns its session_id, which every later call of the study passes

    Call it once before the first search of a study, with the user's question word for word and
    the effort the user's message names.
    At standard and deep effort the reply already holds two focused searches of that question, or
    around the paper it names by arXiv id, one over all years and one over the last two, so start
    from those.
    A call passing the id is logged into the study, and its markers are the ones save_report checks.
    A call without it gets the same answer but leaves no record and carries no markers.

    Args:
        title: What the study is about, in a few words.
        question: The user's question, word for word.
        effort: quick, standard or deep; save_report holds a deep report to the deep steps.
    """
    if not title.strip():
        return "Pass a title."
    level = effort.strip().lower() if effort.strip().lower() in EFFORTS else "quick"
    began = time.time()
    run = await to_thread.run_sync(lambda: runs.start(root(), title, level))
    run.question = question.strip()
    started = f'Session {run.session} started at {level} effort. Pass session_id="{run.session}"'
    reply = f"{started} to every call of it." + (DEEP if level == "deep" else "")
    if question.strip() and level != "quick":
        found = await to_thread.run_sync(lambda: opening(run, question))
        reply += f"\n\n{found}"
        run.log(
            "start_session", {"title": title, "question": question, "effort": level}, reply, began
        )
    return reply


def opening(run: runs.Run, question: str) -> str:
    """
    The question searched over all years and over the last two, around the paper it names if any

    The recent search finds the work an all-time ranking buries, since a new paper has few citers.
    A long question naming a paper embeds near unrelated papers, so the named paper's own vector
    anchors both searches whenever the corpus holds it.

    Args:
        run: The run, which keeps both searches for the report's checks.
        question: The user's question.

    Returns:
        Both result lists under their headings.
    """
    since = (date.today() - timedelta(days=RECENT_DAYS)).isoformat()
    named = next(iter(ARXIV.findall(question)), "")
    anchor = named if named and read_records(load().dsn, [named]) else ""
    text = "" if anchor else question
    before = len(run.searches)
    every = search.search_papers(load(), run, text, anchor, "focused", Filter(), OPENING)
    recent = search.search_papers(
        load(), run, text, anchor, "focused", Filter(date_range=[since, None]), OPENING
    )
    tops = [[p["arxiv_id"] for p in one["papers"][:MUST_READ]] for one in run.searches[before:]]
    run.opening = list(dict.fromkeys(a for top in tops for a in top))
    must = (
        f"\nAt deep effort the report passes only once you have read these, the top {MUST_READ} of "
        f"each list: {', '.join(run.opening)}."
        if run.effort == "deep"
        else ""
    )
    how = f"searched around {anchor}, the paper it names" if anchor else "searched word for word"
    return (
        f"Your question {how}, focused mode, all years:\n{every}\n\n"
        f"The same, published since {since}:\n{recent}{must}"
    )


@mcp.tool()
async def recall_paper(  # noqa: D417
    description: str,
    filter: dict[str, Any] | None = None,
    session_id: str = "",
) -> str:
    """
    Finds one paper from a vague description among about 3 million arXiv papers

    Use it for a paper you know exists but have no id for, by its method, its findings or authors.
    A paper whose arXiv id you already have goes straight to read_paper or list_citations.
    An arXiv id in the description is looked up directly, and its words are used when it is missing.
    It returns the one paper that matches, with its id, title, authors, date, topic and abstract.
    When none matches, it lists the closest candidates, so judge them rather than retry the words.

    Args:
        description: What you remember, like "CaMeL, a prompt injection defense with a privileged
        and a quarantined LLM".
        filter: Optional. {"exclude": [arxiv ids], "authors": [names as written on arXiv, any of
        them must be listed], "date_range": ["YYYY-MM-DD" or null, "YYYY-MM-DD" or null]}.
        session_id: The id start_session returned; without it the call is not recorded.
    """
    args = {"description": description, "filter": filter}
    return await logged(
        session_id,
        "recall_paper",
        args,
        lambda run: recall.recall_paper(load(), description, held(filter)),
    )


@mcp.tool()
async def search_papers(  # noqa: D417
    query: str = "",
    arxiv_id: str = "",
    mode: Literal["focused", "survey", "frontier", "consensus"] = "focused",
    filter: dict[str, Any] | None = None,
    limit: int = 20,
    session_id: str = "",
) -> str:
    """
    Finds the papers answering a research question, or closest to a paper, among 3 million on arXiv

    One search covers one topic, so a question spanning two fields wants one search per field.
    With authors in the filter and no query, it lists those authors' papers, newest first.
    Pass `arxiv_id` instead of `query` to search around that paper.
    Modes:
    - focused: the papers that best match the query, for a specific or recent question; use it by
      default.
    - survey: what a field is built on, the work its reviews cite.
    - frontier: what a broad field is working on now.
    - consensus: what several reviews of a field all cite.

    Args:
        query: A phrase describing what the papers do, without words the whole area shares such as
        "language model".
        arxiv_id: A paper to search around, in place of the query.
        mode: focused, survey, frontier or consensus.
        filter: Optional. {"exclude": [arxiv ids], "authors": [names, any of them], "date_range":
        [start or null, end or null]}.
        limit: How many papers, at most 50.
        session_id: The id start_session returned; without it the call is not recorded.

    Returns:
        One entry per paper, best first: id, date, title, authors, topic, then the opening of its
        abstract.
    """
    args = {"query": query, "arxiv_id": arxiv_id, "mode": mode, "filter": filter, "limit": limit}
    return await logged(
        session_id,
        "search_papers",
        args,
        lambda run: search.search_papers(load(), run, query, arxiv_id, mode, held(filter), limit),
    )


@mcp.tool()
async def read_paper(  # noqa: D417
    arxiv_ids: list[str],
    question: str,
    schema: dict[str, Any] | str | None = None,
    session_id: str = "",
) -> str:
    """
    Reads papers' full text and answers a question about each with the paper's own sentences

    Papers are read one after another, each in a few seconds.
    Each sentence in the reply is the paper's text, under a marker like [Karpukhin7] when the call
    passes a session_id; cite it in your report by that marker, in your own words.
    Ask one specific question per call, such as "what data, what metric, what number against which
    baseline, what fails".

    To compare papers, pass a `schema`, and each paper is answered field by field instead:
    {"fields": [{"name": "uses_retrieval", "type": "bool", "ask": "Does it retrieve passages?"},
    {"name": "retriever", "type": "choice", "ask": "How are candidates found?", "options": ["dense",
    "sparse", "hybrid"]}, {"name": "ndcg10", "type": "float", "ask": "nDCG@10 on BEIR"},
    {"name": "backbone", "type": "text", "ask": "Which model is the backbone?"},
    {"name": "limits", "type": "sentences", "ask": "What do they say fails?"}]}
    A bool is true or false, a float a number as the paper writes it, a choice one of its options,
    a text a short phrase copied from the paper, and sentences up to six of the paper's sentences.
    Every answer cites the sentence it came from, and one the sentence does not support is dropped.
    A field the paper does not state comes back as not stated.
    At most 8 fields, names in snake_case, a choice with 2 to 12 options.
    The rows of every paper go to one table, named in the reply like T1; pass schema="T1" to read
    more papers into it without repeating the fields.

    Args:
        arxiv_ids: The papers to read.
        question: What you need from each.
        schema: Optional fields, or a table id like "T1" to reuse its fields, to answer each paper
        field by field instead of with sentences.
        session_id: The id start_session returned; without it the call is not recorded.
    """
    args = {"arxiv_ids": arxiv_ids, "question": question, "schema": schema}
    return await logged(
        session_id,
        "read_paper",
        args,
        lambda run: read.read_paper(load(), run, arxiv_ids, question, schema),
    )


@mcp.tool()
async def list_citations(  # noqa: D417
    arxiv_id: str,
    direction: Literal["cited_by", "cites"] = "cited_by",
    intents: list[str] | None = None,
    limit: int = 30,
    session_id: str = "",
) -> str:
    """
    Lists the papers citing a paper, or the ones it cites, with each citation's intent and sentence

    Intents are critiques, extends, compares, uses, motivates and background, labeled by a
    classifier, so judge the sentence.
    Use cited_by with critiques, extends and compares to see who took a position on a paper, and
    what came after it.
    Use cites to see what a paper built on.

    Args:
        arxiv_id: The paper.
        direction: cited_by or cites.
        intents: Optional intents to keep.
        limit: How many papers, at most 100.
        session_id: The id start_session returned; without it the call is not recorded.
    """
    args = {"arxiv_id": arxiv_id, "direction": direction, "intents": intents, "limit": limit}
    return await logged(
        session_id,
        "list_citations",
        args,
        lambda run: citations.list_citations(
            load().dsn, run, arxiv_id, direction, intents or [], limit
        ),
    )


@mcp.tool()
async def trace_lineages(  # noqa: D417
    query: str = "",
    past: list[str] | None = None,
    present: list[str] | None = None,
    intents: list[str] | None = None,
    session_id: str = "",
) -> str:
    """
    Maps how a field developed: which papers extend, critique or compare against which

    It searches `query` in survey mode for the field's past and in focused mode over the last two
    years for its present, adds the papers this session read and the papers that argue with several
    of those, and keeps every citation among them that carries an intent.
    The reply gives the longest chain of work, the papers argued with most with what later papers
    say of each, and the newest positions taken; each citing sentence has a marker a report cites.
    The result is a poset with an id like P1, which a report places with a lineage block.

    Args:
        query: A short phrase for the field, like "state space models as an alternative to
        attention for language modeling", never the whole question.
        past: Papers to add to the past, like the one the question names.
        present: Papers to add to the present.
        intents: Which of extends, critiques, compares an edge may carry; all three by default.
        session_id: The id start_session returned; without it the call is not recorded.
    """
    args = {"query": query, "past": past, "present": present, "intents": intents}
    return await logged(
        session_id,
        "trace_lineages",
        args,
        lambda run: lineages.trace_lineages(
            load(), run, query, past or [], present or [], intents or []
        ),
    )


def names_of(run: runs.Run) -> dict[str, str]:
    """
    The short name of every paper the session read or placed in a poset

    Args:
        run: The run.

    Returns:
        Name by arxiv id, for the papers whose title carries one.
    """
    out = {a: name_of(r.title) for a, r in run.papers.items()}
    for poset in run.lineages:
        for p in poset.get("papers", []):
            out.setdefault(p["arxiv_id"], p.get("name", ""))
    return {a: n for a, n in out.items() if n}


def held_by(run: runs.Run, further: list[str]) -> Held:
    """
    What a run holds that its report may cite or place, by the ids a report uses

    Args:
        run: The run.
        further: The ids a report lists under further reading, whose titles are looked up.

    Returns:
        The holdings.
    """
    searches: dict[str, dict[str, Any]] = {}
    for one in run.searches:
        if one.get("query"):
            searches.setdefault(fold(one["query"]), one)
    found = read_records(load().dsn, further) if further else {}
    return Held(
        evidence={m: e.text for m, e in run.evidence.items()},
        sources=sources(run),
        read=set(run.papers),
        served=run.served,
        searches=searches,
        tables={run.table_ids()[k]: t for k, t in run.tables.items()},
        posets={p["id"]: p for p in run.lineages if "id" in p},
        titles={a: f"{r.title} ({r.date.year})" for a, r in found.items()},
        owners={m: e.arxiv_id for m, e in run.evidence.items()},
        names=names_of(run),
        known=fold(" ".join(f"{r.title} {r.abstract}" for r in run.papers.values())),
    )


async def save_report(  # noqa: D417
    session_id: str,
    report: dict[str, Any] | None = None,
    patch: dict[str, Any] | None = None,
) -> str:
    """
    Checks your report against what this session read, and saves it with everything it made

    Send the whole report once as `report`. When it fails, the reply lists each fault under the
    path of its field, and the server keeps your draft: send only those fields as `patch`, a map
    from path to new value, like {"sections.2.blocks.4": {"type": "finding", "text": "...",
    "cites": ["Lee3"]}}. A block path takes the whole block, null deletes it, and one past the
    last index appends.

    The report:
    {"question": the question,
     "tldr": {"text": the answer, like "Hybrids such as Jamba [Lieber4] and Samba [Ren9] ..."},
     "overview": [blocks framing the whole answer: the strands of work and how they relate],
     "sections": [{"heading": ..., "blocks": [blocks]}],
     "further_reading": [arxiv ids found but not read]}
    A block is one of:
    - {"type": "finding", "text": one claim in your own words, its markers inline}, every number
      in the text in a paragraph behind its markers;
    - {"type": "inference", "text": your own reasoning across findings, with no markers};
    - {"type": "open", "text": what is still open, "search": the exact words of a search_papers
      query this session ran to check it, one of whose nearest hits you read};
    - {"type": "table", "table": "T1"}, a table read_paper filled from a schema;
    - {"type": "coverage", "table": "T1", "rows": a choice field, "columns": [bool fields] or [one
      choice field]}, the table's papers counted per cell, each count with its markers;
    - {"type": "lineage", "poset": "P1", "papers": [the ids the section discusses]}, the poset
      trace_lineages returned drawn on those papers with the citing sentences among them.
    The server draws tables, coverage, posets and the searches behind open blocks; tables and
    posets you do not place are added at the end.
    A marker is one a reply gave, like [Karpukhin7], never an arXiv id, and goes in brackets in the
    text right after the claim or the paper name it supports; a paper named in the TL;DR carries
    one of its own markers. An optional "cites" list takes more markers for the end of the text.
    A finding citing a poset's sentence states who extends, critiques or compares against whom.

    Args:
        session_id: The id start_session returned for this study.
        report: The whole report, on the first call or to replace the draft.
        patch: Only the fields a failed check named, by their paths.
    """
    found_run = runs.find(root(), session_id.strip())
    if found_run is None:
        return UNKNOWN.format(session=session_id)
    run = found_run
    began = time.time()
    bad: list[str] = []
    if report is not None:
        draft = report
    elif patch is not None and run.draft is not None:
        draft, bad = patch_draft(run.draft, patch)
    elif patch is not None:
        return "No draft is held for this session yet. Send the whole report first."
    else:
        return "Pass report, or patch after a failed check."
    run.draft = draft

    def body() -> tuple[Any, dict[str, object], str, str, list[str]]:
        further = draft.get("further_reading") if isinstance(draft, dict) else None
        holdings = held_by(
            run, [str(a).strip() for a in further or []] if isinstance(further, list) else []
        )
        drawn = place_tables(draft, holdings) if run.effort == "deep" else draft
        verdict = check(drawn, holdings)
        for path in bad:
            verdict.add(
                path, "path not found", "set a field that exists, or one past the last index"
            )
        traced = any(c["tool"] == "trace_lineages" for c in run.calls)
        owed = run.opening + run.forward + run.argued
        unread = [a for a in dict.fromkeys(owed) if a not in run.papers]
        own = draft.get("sections") if isinstance(draft, dict) else None
        at = len(own) if isinstance(own, list) else None
        contract(drawn, holdings, run.effort, traced, verdict, unread, at)
        kept = lined(verdict, drawn)
        if not verdict.drawable:
            return verdict, kept, "", "", []
        markdown, tldr, unplaced = render(drawn, holdings)
        return verdict, kept, markdown, tldr, unplaced

    verdict, kept, markdown, tldr, unplaced = await to_thread.run_sync(body)
    reply = summary(verdict, draft, unplaced)
    if markdown:
        blocks = [b for _, b in blocks_of(draft) if isinstance(b, dict)] + [draft["tldr"]]
        labels = {c for b in blocks for c in cites_of(b)[0]}
        cited = {run.evidence[m].arxiv_id for m in labels if m in run.evidence}
        run.graph = graph.citation_graph(load(), run.served, set(run.papers), cited)
        out = runs.write(load().runs, run, str(draft["question"]), markdown, kept, tldr)
        reply += f"\nSaved to {out.name}."
    run.log("save_report", {"report": report, "patch": patch}, reply, began)
    if run.folder is not None:
        runs.dump_calls(run.folder, run)
    return reply


# an instance set read only registers nothing that writes a file, which is save_report alone
if not READONLY:
    mcp.tool()(save_report)


@dataclass(frozen=True)
class Serving:
    """
    The transport half of the config, which needs neither the store nor the graph
    """

    host: str
    api_port: int
    frontend_port: int
    allowed_hosts: list[str]
    runs: Path
    pdfs: Path
    fetch: bool


def serving() -> Serving:
    """
    The ports and paths, read without opening the corpus

    Returns:
        The settings.
    """
    config = load_config(str(config_path()))
    block = dict(config["serve"])
    return Serving(
        host=str(block["HOST"]),
        api_port=int(block["API_PORT"]),
        frontend_port=int(block["FRONTEND_PORT"]),
        allowed_hosts=[str(one) for one in block.get("ALLOWED_HOSTS", [])],
        runs=Path(str(block["RUNS_PATH"])),
        pdfs=Path(str(config["dataset"]["PDF_PATH"])),
        fetch=bool(block.get("FETCH", False)),
    )


def secure(server: FastMCP, settings: Serving) -> None:
    """
    Binds one server to its interface, and to the names it answers to

    Args:
        server: The server to configure.
        settings: The transport settings.
    """
    server.settings.host = settings.host
    if settings.allowed_hosts:
        server.settings.transport_security = TransportSecuritySettings(
            allowed_hosts=settings.allowed_hosts,
            allowed_origins=[f"https://{one}" for one in settings.allowed_hosts],
        )


def run(http: bool = False, role: str = "both") -> None:
    """
    Serves the tools over stdio, or the tools and the webui over streamable http

    The frontend role serves a second server with no tools, which opens neither the store nor the
    graph.

    Args:
        http: Serve streamable http instead of stdio.
        role: api for the tools, frontend for the webui, both for one listener carrying each.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = serving()
    if not http:
        _ = load()
        mcp.run()
        return
    if role == "frontend":
        page = FastMCP("graphiv-webui")
        secure(page, settings)
        page.settings.port = settings.frontend_port
        mount(page, settings.runs, settings.pdfs, settings.fetch)
        page.run(transport="streamable-http")
        return
    secure(mcp, settings)
    mcp.settings.port = settings.api_port
    if role != "api":
        mount(mcp, settings.runs, settings.pdfs, settings.fetch)
    _ = load()
    LOG.info(
        "serving %d tools on %s:%d",
        len(mcp._tool_manager.list_tools()),
        mcp.settings.host,
        mcp.settings.port,
    )
    mcp.run(transport="streamable-http")
