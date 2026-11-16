"""
Ablates a tuned recipe one term at a time on its own benchmark's test split, into data/ablation/

The expanded pool is built once per case and every exponent variant rescores it
A stopped shard resumes, skipping the cases its file already holds
"""

import argparse
import copy
import json
from datetime import date
from pathlib import Path

from src.core.config import load_config
from src.core.embed import EmbeddingEndpoint, embed
from src.core.parse_graph_export import load_graph_export
from src.retrieve.recipe import build_pipeline, get_seed_limit, read_recipe

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "benchmarks/data"
RETURN = 1000
KS = (20, 50, 100, 200, 250, 500, 1000)


def variants(recipe: dict) -> dict[str, dict]:
    """
    The full recipe, each one-term removal, and the two reduced recipes

    Args:
        recipe: The tuned recipe as read_recipe returns it.

    Returns:
        Each variant's recipe by its name.
    """
    live = [t for t, w in recipe["weights"].items() if t != "COS" and float(w) != 0.0]

    def with_weights(change: dict[str, float]) -> dict:
        out = copy.deepcopy(recipe)
        out["weights"].update(change)
        return out

    found = {"full": recipe}
    found |= {f"minus_{t.lower()}": with_weights({t: 0.0}) for t in live}
    found["anc_only"] = with_weights({t: 0.0 for t in live if t != "ANC"})
    found["cos_only_expanded"] = with_weights({t: 0.0 for t in live})
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("recipe")
    _ = ap.add_argument("dataset")
    _ = ap.add_argument("shard", type=int)
    _ = ap.add_argument("shards", type=int)
    _ = ap.add_argument("--out", default=str(DATA / "ablation"))
    args = ap.parse_args()
    config = load_config(str(ROOT / "config.toml"))
    endpoint = EmbeddingEndpoint.from_config(config)
    dsn = str(config["database"]["PG_DEST"])
    graph = load_graph_export(str(DATA / "sources/graph_export.npy"))
    recipe = read_recipe(DATA / "recipes" / f"{args.recipe}.toml")
    limit = get_seed_limit(config["retrieve"])
    pipes = {k: build_pipeline(r, config["retrieve"], graph, dsn) for k, r in variants(recipe).items()}
    unexpanded = build_pipeline(recipe, config["retrieve"], graph, dsn)
    unexpanded.expanders, unexpanded.limits = [], []
    out = Path(args.out) / f"{args.recipe}-{args.dataset}_{args.shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["sample_id"] for x in out.open()} if out.exists() else set()
    cases = [json.loads(x) for x in (DATA / f"cases/{args.dataset}/test.jsonl").open()][args.shard :: args.shards]
    with out.open("a") as f:
        for i, c in enumerate(cases):
            if c["sample_id"] in done or not c["key"]:
                continue
            cap = date.fromisoformat(c["cutoff"])
            v = embed(endpoint, c["query"])
            key = set(c["key"])
            for p in [*pipes.values(), unexpanded]:
                p.block(c["anchors"])
            # the live store holds papers newer than the frozen graph, which it cannot score
            pool = [p for p in pipes["full"].build_pool(c["query"], v, retrieve_limit=limit, max_date=cap) if p.arxiv_id in graph.index]
            seed = [p for p in unexpanded.build_pool(c["query"], v, retrieve_limit=limit, max_date=cap) if p.arxiv_id in graph.index]
            ranked = {k: [p.arxiv_id for p in pipe.score_pool(c["query"], v, pool, max_date=cap)[:RETURN]] for k, pipe in pipes.items()}
            ranked["no_expansion"] = [p.arxiv_id for p in unexpanded.score_pool(c["query"], v, seed, max_date=cap)[:RETURN]]
            ranked["cosine_pool_order"] = [p.arxiv_id for p in seed[:RETURN]]
            recall = {k: [len(set(r[:n]) & key) / len(key) for n in KS] for k, r in ranked.items()}
            f.write(json.dumps({"sample_id": c["sample_id"], "pool": len(pool), "recall": recall}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(args.recipe, args.dataset, args.shard, i, "of", len(cases), flush=True)


if __name__ == "__main__":
    main()
