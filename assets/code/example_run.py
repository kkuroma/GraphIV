"""
The one report the poster and the pipeline figure draw, read off its run directory and the store

Cached in assets/data/example_run.json, since BoundedField is rerun on its areas to count each step
The rerun is capped at the run's own date, so papers added since cannot change the counts
"""

from __future__ import annotations

import itertools
import json
import re
import time
import tomllib
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from style import DATA

from src.core.embed import EmbeddingEndpoint, embed
from src.core.parse_graph_export import load_graph_export
from src.retrieve.query import Query
from src.retrieve.recipe import build_pipeline, get_seed_limit, read_recipe

CACHE = DATA / "example_run.json"
RUN = "2026-09-28-how-do-temporal-knowledge-graphs-model-a-scienti"
# the successors drawn past the lineages, the strongest by how many answer papers they reach
SHOWN = 6
ARGUED = ("extends", "critiques", "compares")
ACCENTS = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", "~": "\u0303"}


def surname(authors: str) -> str:
    """
    The first author's surname, with arXiv's LaTeX accents turned into the letters they mean

    Args:
        authors: The author list as arXiv lists it.

    Returns:
        The surname.
    """
    first = authors.split(",")[0].split(" and ")[0]
    marked = re.sub(r"\\([`'\"^~])\{?([A-Za-z])\}?", lambda m: m[2] + ACCENTS[m[1]], first)
    return unicodedata.normalize("NFC", marked.replace("{", "").replace("}", "")).split()[-1]


def label(title: str, authors: str, dated: str) -> str:
    """
    A paper's short name, its method's name when the title opens with one, else its first author

    Args:
        title: The title.
        authors: The author list.
        dated: The iso date.

    Returns:
        The name and the year.
    """
    head = title.split(":")[0].strip()
    name = head if " " not in head and len(head) <= 12 else surname(authors)
    return f"{name} {dated[:4]}"


def strongest(intents: set[str]) -> str:
    """
    The intent a citation pair is drawn in, the argued ones first

    Args:
        intents: Every intent the pair was cited with.

    Returns:
        The first argued intent present, else other.
    """
    return next((i for i in ARGUED if i in intents), "other")


def rerun(config: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    """
    BoundedField on each of the run's areas at the run's cap, counting what every step adds

    Args:
        config: The parsed config.
        run: The run's run.json.

    Returns:
        Per area, the pool after each step and the cosine rank of every paper the run kept.
    """
    cap = date.fromisoformat(run["asked"])
    recipe = read_recipe(Path(config["serve"]["RECIPE_PATH"]) / f"{run['mode']}.toml")
    graph = load_graph_export(config["dataset"]["GRAPH_PATH"])
    out: list[dict[str, Any]] = []
    for area in run["areas"]:
        clock = time.perf_counter()
        vector = embed(EmbeddingEndpoint.from_config(config), area["question"])
        pipeline = build_pipeline(recipe, graph, config["database"]["PG_DEST"])
        query = Query(text=area["question"], emb=vector, max_date=cap)
        pool = pipeline.retriever.retrieve(query, get_seed_limit(recipe))
        cosine = {p.arxiv_id: i for i, p in enumerate(pool)}
        steps = [len(pool)]
        for expander, limit in zip(pipeline.expanders, pipeline.limits, strict=True):
            pool = expander.rerank(query, pool, limit)
            steps.append(len(pool))
        ranked = pipeline.score_pool(area["question"], vector, pool, cap)
        print(f"[example_run] {len(ranked):,} ranked in {time.perf_counter() - clock:.1f}s")
        out.append(
            {
                "question": area["question"],
                "steps": steps,
                "kept": [cosine.get(a) for a in area["papers"]],
            }
        )
    return out


def lanes(lineages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The lineages ordered so a paper several of them share sits on adjacent lanes

    Args:
        lineages: The report's lineages.

    Returns:
        Each lineage's name, its papers in order and the intent between each consecutive pair.
    """
    members = [[p["arxiv_id"] for p in line["papers"]] for line in lineages]
    shared = {a for one in members for a in one}

    def spread(order: tuple[int, ...]) -> int:
        """
        How many lanes the shared papers span in one order, summed

        Args:
            order: The lineages' order.

        Returns:
            The total span.
        """
        total = 0
        for a in shared:
            holding = [i for i, k in enumerate(order) if a in members[k]]
            total += holding[-1] - holding[0]
        return total

    best = min(itertools.permutations(range(len(lineages))), key=spread)
    out: list[dict[str, Any]] = []
    for k in best:
        pairs: dict[frozenset[str], set[str]] = {}
        for edge in lineages[k]["edges"]:
            pairs.setdefault(frozenset((edge["source"], edge["target"])), set()).add(edge["intent"])
        steps = zip(members[k], members[k][1:], strict=False)
        out.append(
            {
                "name": lineages[k]["name"],
                "papers": members[k],
                "intents": [strongest(pairs.get(frozenset(step), set())) for step in steps],
            }
        )
    return out


def read(config: dict[str, Any]) -> dict[str, Any]:
    """
    Everything the figures draw about the run, from its files, the store and the rerun

    Args:
        config: The parsed config.

    Returns:
        The run's numbers, its lanes, its successors and the rerun's counts.
    """
    where = Path(config["serve"]["RUNS_PATH"]) / RUN
    run = json.loads((where / "run.json").read_text(encoding="utf-8"))
    report = json.loads((where / "report.json").read_text(encoding="utf-8"))
    papers = {p["arxiv_id"]: p for p in json.loads((where / "papers.json").read_text("utf-8"))}
    ordered = lanes(report["lineages"])
    held = {a for lane in ordered for a in lane["papers"]}
    columns = sorted(held, key=lambda a: papers[a]["date"])
    ahead = sorted(report["successors"], key=lambda s: -s["reach"])[:SHOWN]
    ahead.sort(key=lambda s: s["date"])
    ids = [s["arxiv_id"] for s in ahead]
    with psycopg.connect(config["database"]["PG_DEST"]) as conn:
        cited = conn.execute(
            "SELECT source_id, target_id, intent FROM citations"
            " WHERE source_id = ANY(%s) AND target_id = ANY(%s) AND intent = ANY(%s)",
            (ids, columns, list(ARGUED)),
        ).fetchall()
        authors = dict(
            conn.execute(
                "SELECT arxiv_id, authors FROM papers WHERE arxiv_id = ANY(%s)", (ids,)
            ).fetchall()
        )
    answered: dict[str, dict[int, set[str]]] = {a: {} for a in ids}
    for source, target, intent in cited:
        for k, lane in enumerate(ordered):
            if target in lane["papers"]:
                answered[source].setdefault(k, set()).add(intent)
    # the method table names the pairs nobody tried in one sentence after its cross table
    _, _, untried = report["methods"]["gaps"].partition("Nobody here does:")
    magnets = json.loads((where / "graph.json").read_text(encoding="utf-8"))["nodes"]
    return {
        "question": run["question"],
        "asked": run["asked"],
        "mode": run["mode"],
        "effort": run["effort"],
        "areas": [
            {
                "question": a["question"],
                "kept": len(a["papers"]),
                "top": [papers[k]["title"] for k in a["papers"][:SHOWN] if k in papers],
            }
            for a in run["areas"]
        ],
        "counts": run["counts"],
        "cited": len(run["cited"]),
        "corrected": len(run["corrected"]),
        "cut": len(run["cut"]),
        "answer": report["answer"],
        "lineages": [line["name"] for line in report["lineages"]],
        "nodes": [
            {
                "title": n["title"],
                "magnet": n["magnet"],
                "cited_by": n["cited_by"],
                "role": n["role"],
            }
            for n in sorted(magnets, key=lambda n: -n["magnet"])
        ],
        "disputes": len(report["disputes"]),
        "methods": {
            "rows": len(report["methods"]["rows"]),
            "columns": [c["name"] for c in report["methods"]["columns"]],
            "untried": [pair for pair in untried.strip().rstrip(".").split(", ") if pair],
            "sample": [
                {"name": r["name"], "year": r["year"], **r["values"]}
                for r in report["methods"]["rows"][:SHOWN]
            ],
        },
        "lanes": ordered,
        "columns": [
            {"arxiv_id": a, "label": label(*(papers[a][k] for k in ("title", "authors", "date")))}
            for a in columns
        ],
        "successors": [
            {
                "arxiv_id": s["arxiv_id"],
                "label": label(s["title"], authors[s["arxiv_id"]], s["date"]),
                "title": s["title"],
                "intents": s["intents"],
                "answered": {str(k): strongest(v) for k, v in answered[s["arxiv_id"]].items()},
            }
            for s in ahead
        ],
        "bounded": rerun(config, run),
        "rerun": date.today().isoformat(),
    }


def load(config: Path) -> dict[str, Any]:
    """
    The cached run, read and rerun first when there is none

    Args:
        config: The config naming the runs, the store, the graph and the embedding endpoint.

    Returns:
        The run's numbers as read cached them.
    """
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    out = read(tomllib.loads(config.read_text(encoding="utf-8")))
    _ = CACHE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out
