"""
The lines a set of papers developed along, sampled by LeyLine over the citations around them

The graph is the papers plus every paper on a citation path of at most two hops joining two of them
A short list of unrelated papers has no such paths, which is why a whole search is the input
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.parse_graph_export import gather_rows
from src.mcp.context import Context
from src.mcp.papers import clean, read_records
from src.mcp.runs import Run
from src.miner.leyline import LeyLine
from src.miner.store import mean_vector, read_edges, read_papers, read_vectors

HOPS = 2
MOST_PAPERS = 1500
LINES = 4
SHORTEST = 3
STEPS = 6
SENTENCE_CHARS = 200
RANK = {"critiques": 0, "extends": 1, "compares": 2, "motivates": 3, "uses": 4}


def step(ctx: Context, rows: np.ndarray, forward: bool) -> np.ndarray:
    """
    Every paper one citation hop from a set of rows

    Args:
        ctx: The open handles.
        rows: Graph rows.
        forward: Toward citers when True, toward references when False.

    Returns:
        The rows reached, with repeats.
    """
    g = ctx.graph
    ptr, idx = (g.cites_ptr, g.cites_idx) if forward else (g.refs_ptr, g.refs_idx)
    return gather_rows(idx, ptr[rows], ptr[rows + 1] - ptr[rows])


def reach(ctx: Context, seeds: np.ndarray, forward: bool) -> dict[int, int]:
    """
    How many hops each paper sits from the seeds in one direction

    Args:
        ctx: The open handles.
        seeds: Graph rows.
        forward: The direction.

    Returns:
        Hops per row reached, the seeds at zero.
    """
    out = {int(s): 0 for s in seeds}
    frontier = seeds
    for hop in range(1, HOPS + 1):
        if not len(frontier):
            break
        fresh = [int(x) for x in np.unique(step(ctx, frontier, forward)) if int(x) not in out]
        for x in fresh:
            out[x] = hop
        frontier = np.array(fresh, np.int64)
    return out


def between(ctx: Context, ids: list[str]) -> list[str]:
    """
    The papers plus every paper on a short citation path joining two of them

    Args:
        ctx: The open handles.
        ids: The caller's papers.

    Returns:
        The ids of the graph the walks run on.
    """
    rows = np.array([ctx.graph.index[a] for a in ids if a in ctx.graph.index], np.int64)
    up, down = reach(ctx, rows, True), reach(ctx, rows, False)
    keep = {x for x, a in up.items() if x in down and a + down[x] <= HOPS} | {int(r) for r in rows}
    return [ctx.graph.ids[x].decode() for x in keep]


def describe(
    nodes: list[str], edges: list[Any], held: dict[str, Any], given: set[str]
) -> dict[str, Any]:
    """
    One line as its papers oldest first, the citing sentences along it and a figure

    Args:
        nodes: The line's papers, oldest first.
        edges: Every labeled citation in the graph.
        held: The papers by id.
        given: The caller's own papers.

    Returns:
        The line.
    """
    keep = set(nodes)
    best: dict[tuple[str, str], Any] = {}
    for e in edges:
        pair = (e.target, e.source)
        if (
            e.source in keep
            and e.target in keep
            and e.intent in RANK
            and (pair not in best or RANK[e.intent] < RANK[best[pair].intent])
        ):
            best[pair] = e
    marks = {a: f"p{i}" for i, a in enumerate(nodes)}
    boxes = [
        f'{marks[a]}["{a} ({held[a].date.year}) {clean(held[a].title)[:44].replace(chr(34), "")}"]'
        for a in nodes
    ]
    arrows = [f"{marks[a]} -->|{e.intent}| {marks[b]}" for (a, b), e in best.items()]
    said = sorted((e for e in best.values() if e.sentence), key=lambda e: RANK[e.intent])[:STEPS]
    return {
        "papers": [
            {
                "arxiv_id": a,
                "year": held[a].date.year,
                "title": clean(held[a].title),
                "yours": a in given,
            }
            for a in nodes
        ],
        "steps": [
            {
                "citer": e.source,
                "cited": e.target,
                "intent": e.intent,
                "sentence": clean(e.sentence)[:SENTENCE_CHARS],
            }
            for e in said
        ],
        "mermaid": "graph LR\n" + "\n".join(boxes + arrows),
    }


def trace_lineages(ctx: Context, run: Run, arxiv_ids: list[str], question: str) -> str:
    """
    The lines a set of papers developed along

    Args:
        ctx: The open handles.
        run: The caller's run, which keeps the lines for the report.
        arxiv_ids: The papers, best a whole search.
        question: What the caller works on, which holds the walks to the topic.

    Returns:
        Each line as its papers oldest first and the sentences joining them.
    """
    ids = list(dict.fromkeys(a.strip() for a in arxiv_ids if a.strip()))
    if len(ids) < SHORTEST:
        return f"Pass at least {SHORTEST} papers, best a whole search."
    vector = (
        ctx.embed(question)
        if question.strip()
        else mean_vector(list(read_vectors(ctx.dsn, ids).values()))
    )
    graph = between(ctx, ids)
    papers = {p.arxiv_id: p for p in read_papers(ctx.dsn, vector, graph)}
    given = set(ids)
    ranked = sorted(papers, key=lambda a: (a not in given, -(papers[a].cosine or 0.0)))[
        :MOST_PAPERS
    ]
    held = {a: papers[a] for a in ranked}
    edges = read_edges(ctx.dsn, ranked)
    scores = {a: max(p.cosine or 0.0, 0.0) for a, p in held.items()}
    order = {a: float(p.date.toordinal()) for a, p in held.items()}
    walker = LeyLine(edges, ctx.mining.settings, scores, order, focus=scores)
    lines = [x for x in walker.lineages() if len(x.nodes) >= SHORTEST][:LINES]
    found = [describe(x.nodes, edges, held, given) for x in lines]
    citers = {step["citer"] for line in found for step in line["steps"]}
    keys = {a: r.key for a, r in read_records(ctx.dsn, list(citers)).items()}
    for line in found:
        for step in line["steps"]:
            key = keys.get(step["citer"], "Paper")
            step["label"] = run.cite(step["citer"], step["cited"], step["sentence"], key)
    with run.lock:
        run.lineages.append(
            {
                "question": question,
                "inputs": ids,
                "graph": len(held),
                "edges": len(edges),
                "lines": found,
            }
        )
    if not found:
        size = f"{len(held)} papers and {len(edges)} citations"
        return f"No line of {SHORTEST} or more papers over {size}. Pass a whole search instead."
    out = [f"{len(found)} lines over {len(held)} papers and {len(edges)} citations"]
    for spot, line in enumerate(found, 1):
        out.append(f"\nLine {len(run.lineages)}.{spot}")
        out += [
            f"  {p['arxiv_id']} ({p['year']}) {p['title'][:90]}"
            + ("" if p["yours"] else " [not in your list]")
            for p in line["papers"]
        ]
        out += [
            f"  {s['citer']} {s['intent']} {s['cited']}: {s['sentence']} [{s['label']}]"
            for s in line["steps"]
        ]
    return "\n".join(out)
