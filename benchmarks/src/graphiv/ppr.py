"""
Personalized PageRank over each case's BoundedField pool, so the scorer is the one thing that differs

Rank flows from a citing paper to what it cites and teleports back by query cosine
`grid` scores every setting on train cases and `run` ranks with one, for src/methods/ppr.py
"""

import argparse
import itertools
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.core.config import load_config
from src.core.embed import embed
from src.core.parse_graph_export import load_graph_export
from src.retrieve.recipe import build_pipeline, get_seed_limit, read_recipe
from src.retrieve.scorer import GraphScorer

ROOT = Path(__file__).resolve().parents[3]
# the recipe whose pool PageRank ranks, BoundedField's own
POOL = ROOT / "retrieval_recipes" / "survey.toml"
CORE = 200
RETURN = 1000


def pagerank(adjacency: sp.csr_matrix, teleport: np.ndarray, alpha: float) -> np.ndarray:
    # column j of the adjacency is what j cites, so each column splits j's rank among its references
    out = np.asarray(adjacency.sum(axis=0)).ravel()
    walk = adjacency @ sp.diags(np.divide(1.0, out, out=np.zeros_like(out), where=out > 0))
    dangling = out == 0
    rank = teleport.copy()
    for _ in range(200):
        nxt = alpha * (walk @ rank + rank[dangling].sum() * teleport) + (1 - alpha) * teleport
        if np.abs(nxt - rank).sum() < 1e-10:
            return nxt
        rank = nxt
    return rank


class Scorer:
    """
    One pool's adjacency and teleports, built once and shared by every setting
    """

    def __init__(self, graph_scorer: GraphScorer, papers: list) -> None:
        cited_by = graph_scorer.build_adjacency_matrix(graph_scorer.get_paper_rows(papers)).astype(np.float64)
        both = cited_by + cited_by.T
        both.data[:] = 1.0
        self.adjacency = {"back": cited_by.tocsr(), "both": both.tocsr()}
        cosine = np.clip(np.array([p.cosine or 0.0 for p in papers]), 0.0, None)
        core = np.zeros_like(cosine)
        top = np.argsort(-cosine)[:CORE]
        core[top] = cosine[top]
        self.teleport = {"pool": cosine / (cosine.sum() or 1.0), "core": core / (core.sum() or 1.0)}
        self.cosine = cosine

    def rank(self, alpha: float, teleport: str, direction: str) -> np.ndarray:
        score = pagerank(self.adjacency[direction], self.teleport[teleport], alpha)
        return np.lexsort((-self.cosine, -score))[:RETURN]


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("mode", choices=["grid", "run"])
    _ = ap.add_argument("--graph", required=True)
    _ = ap.add_argument("--cases", required=True)
    _ = ap.add_argument("--params", required=True, help="json: alphas, teleports, directions and k for grid, setting for run")
    _ = ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = json.loads(args.params)
    config = load_config(str(ROOT / "config.toml"))
    api, name = str(config["endpoints"]["EMBED_API"]), str(config["endpoints"]["EMBED_NAME"])
    graph = load_graph_export(args.graph)
    recipe = read_recipe(POOL)
    pipeline = build_pipeline(recipe, graph, str(config["database"]["PG_DEST"]))
    limit = get_seed_limit(recipe)
    graph_scorer = GraphScorer(graph)
    with open(args.cases) as f:
        cases = [json.loads(line) for line in f]

    def pool_of(c: dict) -> list:
        v = embed(api, name, c["query"])
        pipeline.block(c["anchors"])
        return pipeline.build_pool(c["query"], v, retrieve_limit=limit, max_date=date.fromisoformat(c["cutoff"]))

    if args.mode == "grid":
        settings = list(itertools.product(params["alphas"], params["teleports"], params["directions"]))
        totals = np.zeros(len(settings))
        for c in cases:
            papers = pool_of(c)
            scorer = Scorer(graph_scorer, papers)
            for j, s in enumerate(settings):
                top = {papers[i].arxiv_id for i in scorer.rank(*s)[: params["k"]]}
                totals[j] += len(top & set(c["key"])) / len(c["key"])
        grid = {"/".join(map(str, s)): float(t / len(cases)) for s, t in zip(settings, totals)}
        with open(args.out, "w") as f:
            json.dump(grid, f)
        return

    setting = tuple(params["setting"])
    with open(args.out, "w") as f:
        for i, c in enumerate(cases):
            started = time.perf_counter()
            papers = pool_of(c)
            order = Scorer(graph_scorer, papers).rank(*setting)
            seconds = time.perf_counter() - started
            f.write(json.dumps({"sample_id": c["sample_id"], "ranked": [papers[j].arxiv_id for j in order], "seconds": seconds}) + "\n")
            if i % 100 == 0:
                print("ppr", i, "of", len(cases), flush=True)


if __name__ == "__main__":
    main()
