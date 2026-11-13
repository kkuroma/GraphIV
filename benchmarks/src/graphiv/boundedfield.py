"""
Ranks cases with one recipe through GraphIV's own pipeline, for src/methods/boundedfield.py

A case's seconds are the embedding, the pool and the scoring together
With --corpus the ranking keeps only those ids before it is cut to its length
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

from src.core.config import load_config
from src.core.embed import EmbeddingEndpoint, embed
from src.core.parse_graph_export import load_graph_export
from src.retrieve.recipe import build_pipeline, get_seed_limit, read_recipe

ROOT = Path(__file__).resolve().parents[3]
RETURN = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--recipe", required=True)
    _ = ap.add_argument("--graph", required=True)
    _ = ap.add_argument("--cases", required=True)
    _ = ap.add_argument("--out", required=True)
    _ = ap.add_argument("--corpus", help="a file of the ids a ranking may hold, one per line")
    args = ap.parse_args()
    config = load_config(str(ROOT / "config.toml"))
    endpoint = EmbeddingEndpoint.from_config(config)
    recipe = read_recipe(Path(args.recipe))
    pipeline = build_pipeline(recipe, config["retrieve"], load_graph_export(args.graph), str(config["database"]["PG_DEST"]))
    limit = get_seed_limit(config["retrieve"])
    with open(args.cases) as f:
        cases = [json.loads(line) for line in f]
    allowed = set(Path(args.corpus).read_text().split()) if args.corpus else None
    with open(args.out, "w") as f:
        for i, c in enumerate(cases):
            started = time.perf_counter()
            v = embed(endpoint, c["query"])
            cap = date.fromisoformat(c["cutoff"])
            pipeline.block(c["anchors"])
            pool = pipeline.build_pool(c["query"], v, retrieve_limit=limit, max_date=cap)
            papers = pipeline.score_pool(c["query"], v, pool, max_date=cap)
            ranked = [p.arxiv_id for p in papers if allowed is None or p.arxiv_id in allowed][:RETURN]
            seconds = time.perf_counter() - started
            f.write(json.dumps({"sample_id": c["sample_id"], "ranked": ranked, "seconds": seconds}) + "\n")
            if i % 100 == 0:
                print("boundedfield", i, "of", len(cases), flush=True)


if __name__ == "__main__":
    main()
