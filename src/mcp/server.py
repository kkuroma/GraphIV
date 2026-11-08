"""
The six tools and both transports, holding the store, the graph and the model for the process

The docstrings here are what the calling model reads, so they say what a tool is for and when
Every call is logged into its session's run, which save_report writes to disk for the webui
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from anyio import to_thread
from mcp.server.fastmcp import Context as Session
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from src.core.config import load_config
from src.mcp import citations, graph, lineages, read, recall, runs, search
from src.mcp.context import config_path, load
from src.mcp.dashboard import mount
from src.mcp.papers import Filter, read_records
from src.report.check import (
    MARKER,
    body_lines,
    check,
    fold,
    further_ids,
    relist,
    split_tldr,
    summary,
)

LOG = logging.getLogger(__name__)

mcp = FastMCP("graphiv")
# the answer card holds one sentence
TLDR_WORDS = 40


def session_of(session: Session) -> str:
    """
    The MCP session a call belongs to

    The server holds one session object per client connection on every transport, so its identity
    tells two clients apart where a request header may not be visible.

    Args:
        session: The request context FastMCP passes.

    Returns:
        A key unique to the session.
    """
    return f"session-{id(session.session)}"


async def logged(session: Session, tool: str, args: dict[str, Any], body: Any) -> str:
    """
    Runs one tool body off the event loop and records the call in the session's run

    Args:
        session: The request context.
        tool: The tool name.
        args: What the caller passed.
        body: A callable taking the run and returning the reply.

    Returns:
        The reply.
    """
    LOG.info("call %s from %s", tool, session_of(session))
    run = runs.current(session_of(session))
    began = time.time()
    reply = await to_thread.run_sync(body, run)
    run.log(tool, args, reply, began)
    return reply


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
async def recall_paper(  # noqa: D417
    description: str,
    filter: dict[str, Any] | None = None,
    session: Session = None,  # type: ignore[assignment]
) -> str:
    """
    Finds one paper from a vague description among about 3 million arXiv papers

    Use it for a paper you know exists: a method's name, what the paper did, or who wrote it.
    It returns the one paper that matches, with its arXiv id, title, authors, date, topic and
    abstract, or says that none matches.
    When it says none matches, no paper in the corpus fits the description and filter, so do not
    retry with the same words.

    Args:
        description: What you remember, like "CaMeL, a prompt injection defense with a privileged
        and a quarantined LLM".
        filter: Optional. {"exclude": [arxiv ids], "authors": [names as written on arXiv, any of
        them must be listed], "date_range": ["YYYY-MM-DD" or null, "YYYY-MM-DD" or null]}.
    """
    args = {"description": description, "filter": filter}
    return await logged(
        session,
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
    session: Session = None,  # type: ignore[assignment]
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

    Returns:
        One entry per paper, best first: id, date, title, authors, topic, then the opening of its
        abstract.
    """
    args = {"query": query, "arxiv_id": arxiv_id, "mode": mode, "filter": filter, "limit": limit}
    return await logged(
        session,
        "search_papers",
        args,
        lambda run: search.search_papers(load(), run, query, arxiv_id, mode, held(filter), limit),
    )


@mcp.tool()
async def read_paper(  # noqa: D417
    arxiv_ids: list[str],
    question: str,
    schema: dict[str, Any] | None = None,
    session: Session = None,  # type: ignore[assignment]
) -> str:
    """
    Reads papers' full text and answers a question about each, with verbatim quotes

    Pass several ids at once: four are read in parallel, so a batch of four costs about one.
    Each claim in the reply carries a marker like [Karpukhin7] and the quote behind it.
    Cite a claim in your report with its marker, and the saved report shows the quote with it.
    Ask one specific question per batch, such as "what data, what metric, what number against which
    baseline, what fails".

    To compare methods, pass a `schema`, and each paper comes back as a table row instead of an
    answer:
    {"columns": [{"name": "retrieval_type", "kind": "choice", "description": "how candidates are
    found", "options": ["dense", "sparse", "hybrid"]}, {"name": "result", "kind": "value",
    "description": "headline number and benchmark"}]}
    A choice column lists the options a method picks from, a value column holds a number and what it
    was measured on, and a text column holds a short phrase.

    Args:
        arxiv_ids: The papers to read.
        question: What you need from each.
        schema: Optional table columns, to fill one row per method instead of answering.
    """
    args = {"arxiv_ids": arxiv_ids, "question": question, "schema": schema}
    return await logged(
        session,
        "read_paper",
        args,
        lambda run: read.read_paper(load(), run, arxiv_ids, question, read.parse_schema(schema)),
    )


@mcp.tool()
async def list_citations(  # noqa: D417
    arxiv_id: str,
    direction: Literal["cited_by", "cites"] = "cited_by",
    intents: list[str] | None = None,
    limit: int = 30,
    session: Session = None,  # type: ignore[assignment]
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
    """
    args = {"arxiv_id": arxiv_id, "direction": direction, "intents": intents, "limit": limit}
    return await logged(
        session,
        "list_citations",
        args,
        lambda run: citations.list_citations(
            load().dsn, run, arxiv_id, direction, intents or [], limit
        ),
    )


@mcp.tool()
async def trace_lineages(  # noqa: D417
    arxiv_ids: list[str],
    question: str = "",
    session: Session = None,  # type: ignore[assignment]
) -> str:
    """
    Traces how a field developed through a set of papers: which work led to which, oldest first

    Pass a whole search, 20 to 50 papers on one topic: a handful of papers rarely cite each other,
    so they give no lines.
    The lines are kept for the report, which shows each as a figure.

    Args:
        arxiv_ids: The papers.
        question: What you are working on, which keeps the lines on your topic.
    """
    args = {"arxiv_ids": arxiv_ids, "question": question}
    return await logged(
        session,
        "trace_lineages",
        args,
        lambda run: lineages.trace_lineages(load(), run, arxiv_ids, question),
    )


async def save_report(  # noqa: D417
    question: str,
    markdown: str,
    session: Session = None,  # type: ignore[assignment]
) -> str:
    """
    Checks your report against what this session read, and saves it with everything it did

    The report may open, after its title, on a TL;DR paragraph: "TL;DR: " and the answer in one
    sentence of at most 40 words, citing its strongest evidence.
    Write one whenever a sentence can answer the question; for a list or a history, name two or
    three of the strongest papers and leave the rest to the body.
    It opens the saved report and is checked like every other line, so it must cite.
    The check is mechanical:
    - every marker like [Karpukhin7] must be one a read_paper reply gave, and [Chen3, Karpukhin7]
      cites two,
    - a number on a cited line must appear in the paragraph behind one of its markers,
    - a quote on a line must sit in the paragraph its marker names,
    - every paper cited in the body must have been read; unread papers go under a final "## Further
      reading" heading,
    - a line stating a finding carries a marker, or starts with "Inference:" when it is your own
      reasoning.
    A report that fails is still saved, flagged, and the reply lists each fault, so fix them and
    call again.
    Comparison tables and lineages from this session are added to the saved report by the server; do
    not retype them.

    Args:
        question: The question the report answers.
        markdown: The report, opening on its TL;DR when it has one.
    """
    session_id = session_of(session)
    run = runs.current(session_id)
    began = time.time()
    found = read_records(load().dsn, further_ids(markdown))
    titles = {a: f"{r.title} ({r.date.year})" for a, r in found.items()}
    markdown = relist(markdown, titles, set(run.papers))
    evidence = {m: e.text for m, e in run.evidence.items()}
    queries = {str(c["args"].get("query") or "") for c in run.calls if c["tool"] == "search_papers"}
    tldr, markdown = split_tldr(markdown)
    whole = f"{tldr}\n\n{markdown}"
    verdict = check(whole, evidence, set(run.papers), run.served, sources(run), queries)
    if tldr and not MARKER.search(tldr):
        verdict.faults.insert(0, {"kind": "TL;DR cites nothing", "where": "", "line": tldr})
    words = len(MARKER.sub("", tldr).split())
    if words > TLDR_WORDS:
        verdict.faults.insert(
            0, {"kind": f"TL;DR over {TLDR_WORDS} words", "where": f"at {words}", "line": tldr}
        )
    if not any(MARKER.search(line) for line, further in body_lines(markdown) if not further):
        verdict.faults.insert(
            0, {"kind": "report body cites nothing", "where": "", "line": markdown}
        )
    labels = {m.strip() for group in MARKER.findall(whole) for m in group.split(",")}
    cited = {run.evidence[m].arxiv_id for m in labels if m in run.evidence}
    run.graph = graph.citation_graph(load(), run.served, set(run.papers), cited)
    out = runs.write(load().runs, run, question, markdown, verdict.as_dict(), tldr)
    reply = summary(verdict) + f"\nSaved to {out.name}."
    args = {"question": question, "markdown": markdown, "tldr": tldr}
    run.log("save_report", args, reply, began)
    runs.dump_calls(out, run)
    if verdict.passed:
        runs.close(session_id)
    return reply


# an instance set read only registers nothing that writes a file, which is save_report alone
if os.environ.get("GRAPHIV_READONLY") != "1":
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
