"""
The citation poset of a field, a survey search as its past and a recent focused one as its present

Every edge is a citation with an intent inside that set, so each is a sentence a report can cite
Papers outside both searches join when they argue with several of the set, closest to the question
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import date, timedelta
from functools import cache
from typing import Any

from src.mcp.context import Context
from src.mcp.papers import Filter, Record, clean, read_records
from src.mcp.runs import Run
from src.mcp.search import rank
from src.miner.store import read_citers, read_cosines, read_edges
from src.report.check import name_of

INTENTS = ("extends", "critiques", "compares")
# how a pair cited with several intents is drawn, the strongest position first
RANK = {"critiques": 0, "extends": 1, "compares": 2}
# config.toml's [lineages] block overrides each of these
DEFAULTS = {
    "SIDE": 50,
    "RECENT_DAYS": 730,
    "LEAST": 3,
    "ADDED": 30,
    "LEAST_READS": 2,
    "MUST_ARGUED": 4,
    "MUST_RECENT": 4,
}
ARGUED = 8
PER_PAPER = 4
NEWEST = 10
SENTENCE_CHARS = 220
TITLE_CHARS = 80


def best_edges(
    edges: list[Any], papers: dict[str, Record], intents: list[str]
) -> dict[tuple[str, str], Any]:
    """
    One citation per citing pair, the strongest intent with a sentence, newer paper citing older

    Args:
        edges: Every labeled citation inside the set.
        papers: The set by id.
        intents: The intents kept.

    Returns:
        The citation per (citer, cited) pair.
    """
    order = {a: (p.date, a) for a, p in papers.items()}
    best: dict[tuple[str, str], Any] = {}
    for e in edges:
        pair = (e.source, e.target)
        if e.intent not in intents or e.source not in order or e.target not in order:
            continue
        # a same day citation both ways would loop, so time and then id decide the direction
        if order[e.source] <= order[e.target]:
            continue
        old = best.get(pair)
        if old is None or (RANK[e.intent], not e.sentence) < (RANK[old.intent], not old.sentence):
            best[pair] = e
    return best


def reduce(pairs: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """
    The pairs no longer path joins, which is what a figure of the poset draws

    Args:
        pairs: The (citer, cited) pairs, acyclic.

    Returns:
        The pairs kept.
    """
    refs: dict[str, set[str]] = {}
    for a, b in pairs:
        refs.setdefault(a, set()).add(b)

    @cache
    def below(a: str) -> frozenset[str]:
        out: set[str] = set()
        for b in refs.get(a, ()):
            out |= {b} | below(b)
        return frozenset(out)

    return {
        (a, b) for a, b in pairs if not any(b in below(c) for c in refs[a] if c != b)
    }  # fmt: skip


def longest(
    pairs: set[tuple[str, str]], papers: dict[str, Record], weight: dict[str, float]
) -> list[str]:
    """
    The chain of citations with the largest summed weight, oldest paper first

    A paper of negative weight ends a chain, so a chain starts where it turns to the question.

    Args:
        pairs: The (citer, cited) pairs.
        papers: The set by id.
        weight: How far each paper sits above the median closeness to the question.

    Returns:
        The chain's ids.
    """
    refs: dict[str, list[str]] = {}
    for a, b in pairs:
        refs.setdefault(a, []).append(b)
    depth: dict[str, tuple[float, str]] = {}
    for a in sorted({x for p in pairs for x in p}, key=lambda a: (papers[a].date, a)):
        below = [(depth[b][0], b) for b in refs.get(a, []) if b in depth]
        top = max(below, default=(0.0, ""))
        top = top if top[0] > 0 else (0.0, "")
        depth[a] = (top[0] + weight.get(a, 0.0), top[1])
    if not depth:
        return []
    end = max(depth, key=lambda a: (depth[a][0], papers[a].date))
    chain = [end]
    while depth[chain[-1]][1]:
        chain.append(depth[chain[-1]][1])
    return chain[::-1]


def named(record: Record) -> str:
    """
    A paper as one line, id, short title and year

    Args:
        record: The paper.

    Returns:
        The line.
    """
    return f"{record.arxiv_id} {clean(record.title)[:TITLE_CHARS]} ({record.date.year})"


def trace_lineages(
    ctx: Context, run: Run, query: str, past: list[str], present: list[str], intents: list[str]
) -> str:
    """
    The poset of citations with an intent among a field's past and present papers

    Args:
        ctx: The open handles.
        run: The caller's run, which keeps the poset for the report and labels its sentences.
        query: A phrase for the field, searched in survey mode as the past, focused as the present.
        past: Papers the caller adds to the past, like one the question names.
        present: Papers the caller adds to the present.
        intents: The intents an edge may carry, empty for extends, critiques and compares.

    Returns:
        The poset's longest chain, the papers argued with most and the newest positions taken.
    """
    knobs = {**DEFAULTS, **ctx.config.get("lineages", {})}
    side, least, reading = int(knobs["SIDE"]), int(knobs["LEAST"]), int(knobs["LEAST_READS"])
    wanted = [i for i in intents if i in RANK] or list(INTENTS)
    since = date.today() - timedelta(days=int(knobs["RECENT_DAYS"]))
    olds = [a.strip() for a in past if a.strip()]
    news = [a.strip() for a in present if a.strip()]
    if query.strip():
        found, _ = rank(ctx, "survey", query, "", Filter(), side)
        olds += [r.arxiv_id for r in found]
        found, _ = rank(
            ctx, "focused", query, "", Filter(date_range=[since.isoformat(), None]), side
        )
        news += [r.arxiv_id for r in found]
    reads = list(run.papers)
    base = list(dict.fromkeys(olds + news + reads))
    if len(base) < 2:
        return "Pass a query, or papers in past and present."
    vector = ctx.embed(query.strip() or run.question or " ".join(base))
    added = read_citers(ctx.dsn, base, wanted, least, vector, int(knobs["ADDED"]))
    papers = read_records(ctx.dsn, base + added)
    best = best_edges(read_edges(ctx.dsn, list(papers)), papers, wanted)
    if not best:
        return (
            f"No citation with an intent of {', '.join(wanted)} joins these {len(papers)} papers."
        )
    kept = {x for pair in best for x in pair}
    near = {a: max(c, 0.0) for a, c in read_cosines(ctx.dsn, vector, list(kept)).items()}
    drawn = reduce(set(best))
    middle = statistics.median(near.values()) if near else 0.0
    chain = longest(drawn, papers, {a: c - middle for a, c in near.items()})
    argued: dict[str, Counter[str]] = {}
    for (_, b), e in best.items():
        argued.setdefault(b, Counter())[e.intent] += 1
    # what the reads critique or extend comes first, as a hub or a baseline is no debate
    by_reads = Counter(
        b for (a, b), e in best.items() if a in run.papers and e.intent in ("critiques", "extends")
    )
    debated = [a for a, n in by_reads.most_common() if n >= reading]
    closest = sorted(argued, key=lambda a: (-sum(argued[a].values()) * near.get(a, 0.0), a))
    top = list(dict.fromkeys(debated + closest))[:ARGUED]
    shown: list[tuple[str, str]] = list(zip(chain[1:], chain[:-1], strict=True))
    weight = {a: sum(c.values()) for a, c in argued.items()}
    for b in top:
        # each intent in turn, its citers the field itself argues with most first
        into = [p for p in best if p[1] == b and best[p].sentence and p not in shown]
        into.sort(key=lambda p: (-weight.get(p[0], 0), -papers[p[0]].date.toordinal()))
        turns = [[p for p in into if best[p].intent == i] for i in RANK]
        while turns and len([p for p in shown if p[1] == b]) < PER_PAPER and any(turns):
            for turn in turns:
                if turn and len([p for p in shown if p[1] == b]) < PER_PAPER:
                    shown.append(turn.pop(0))
    recent = sorted(
        (a for a in kept if papers[a].date >= since and a not in top),
        key=lambda a: near.get(a, 0.0),
        reverse=True,
    )[:NEWEST]
    for a in recent:
        out = sorted((p for p in best if p[0] == a), key=lambda p: RANK[best[p].intent])
        shown += out[:1]
    shown = list(dict.fromkeys(p for p in shown if best[p].sentence))
    labels = {
        pair: run.cite(pair[0], pair[1], best[pair].sentence, papers[pair[0]].key)
        for pair in shown
    }  # fmt: skip
    pid = f"P{len(run.lineages) + 1}"
    with run.lock:
        run.lineages.append(
            {
                "id": pid,
                "query": query,
                "since": since.isoformat(),
                "intents": wanted,
                "papers": [
                    {
                        "arxiv_id": a,
                        "title": clean(papers[a].title),
                        "name": name_of(papers[a].title),
                        "date": papers[a].date.isoformat(),
                        "from": [
                            side
                            for side, ids in (
                                ("past", olds),
                                ("present", news),
                                ("read", reads),
                                ("added", added),
                            )
                            if a in ids
                        ],
                    }
                    for a in sorted(kept, key=lambda a: (papers[a].date, a))
                ],
                "edges": [
                    {
                        "citer": a,
                        "cited": b,
                        "intent": e.intent,
                        "sentence": clean(e.sentence),
                        "implied": (a, b) not in drawn,
                        "label": labels.get((a, b), ""),
                    }
                    for (a, b), e in best.items()
                ],
                "chain": chain,
                "focus": list(dict.fromkeys([a for a in past if a in kept] + top + chain)),
            }
        )
    if run.effort == "deep" and not run.argued:
        disputed = [a for a in debated if a not in run.papers][: int(knobs["MUST_ARGUED"])]
        newest = [a for a in recent if a not in run.papers][: int(knobs["MUST_RECENT"])]
        run.argued = list(dict.fromkeys(disputed + newest))
    owed = [a for a in run.argued if a not in run.papers]
    mix = Counter(e.intent for e in best.values())
    unread = [a for a in kept if a not in run.papers]
    out = [
        f"Poset {pid}: {len(kept)} papers and {len(best)} citations with an intent among them "
        f"({', '.join(f'{n} {i}' for i, n in mix.most_common())}), {len(drawn)} once a citation "
        "a longer path implies is dropped.",
        f"Built from a survey search ({len(olds)} papers with yours), a focused search since "
        f"{since.isoformat()} ({len(news)}), the {len(reads)} papers this session read, and "
        f"{len(added)} papers that argue with {least} or more of those.",
        f'Place it with {{"type": "lineage", "poset": "{pid}", "papers": [the ids a section '
        "discusses]}. Each sentence below is citable by its marker; the other "
        f"{len(unread)} unread papers here are leads to read first.",
    ]

    def step(pair: tuple[str, str]) -> str:
        said = best[pair].sentence[:SENTENCE_CHARS]
        mark = run.mark(labels[pair]) if pair in labels else ""
        return f'{best[pair].intent} {named(papers[pair[1]])}: "{said}"{mark}'

    printed: set[tuple[str, str]] = set()

    def lines(pairs: list[tuple[str, str]]) -> list[str]:
        fresh = [p for p in pairs if p not in printed]
        printed.update(fresh)
        return [f"  {named(papers[a])} {step((a, b))}" for a, b in fresh]

    out.append("\nLongest chain, oldest first:")
    out.append(f"  {named(papers[chain[0]])}")
    out += lines(list(zip(chain[1:], chain[:-1], strict=True)))
    out.append("\nArgued with most by the papers you read, then by the field, and what each says:")
    for b in top:
        counts = ", ".join(f"{n} {i}" for i, n in argued[b].most_common())
        out.append(f"{named(papers[b])}: {counts}")
        out += lines([p for p in shown if p[1] == b])
    out.append("\nNewest papers and the earlier work they take a position on:")
    out += lines([p for p in shown if p[0] in recent])
    if owed:
        out.append(
            f"\nAt deep effort the report passes only once you have read {', '.join(owed)}, the "
            "papers argued with most and the newest ones taking a position; read them with the "
            "question of what each claims and on what evidence, so the report can say who disputes "
            "what and where the field stands now."
        )
    joined = [named(papers[a]) for a in added if a in papers]
    if joined:
        out.append("\nAdded because they argue with this set: " + "; ".join(joined))
    return "\n".join(out)
