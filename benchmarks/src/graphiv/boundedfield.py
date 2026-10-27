"""
Ranks cases with one recipe through GraphIV's own pipeline, for src/methods/boundedfield.py

A case's seconds are the embedding, the pool and the scoring together
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

from src.core.config import load_config
from src.core.embed import embed
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
    args = ap.parse_args()
    config = load_config(str(ROOT / "config.toml"))
    api, name = str(config["endpoints"]["EMBED_API"]), str(config["endpoints"]["EMBED_NAME"])
    recipe = read_recipe(Path(args.recipe))
    pipeline = build_pipeline(recipe, load_graph_export(args.graph), str(config["database"]["PG_DEST"]))
    limit = get_seed_limit(recipe)
    with open(args.cases) as f:
        cases = [json.loads(line) for line in f]
    with open(args.out, "w") as f:
        for i, c in enumerate(cases):
            started = time.perf_counter()
            v = embed(api, name, c["query"])
            cap = date.fromisoformat(c["cutoff"])
            pipeline.block(c["anchors"])
            pool = pipeline.build_pool(c["query"], v, retrieve_limit=limit, max_date=cap)
            papers = pipeline.score_pool(c["query"], v, pool, max_date=cap, return_limit=RETURN)
            seconds = time.perf_counter() - started
            f.write(json.dumps({"sample_id": c["sample_id"], "ranked": [p.arxiv_id for p in papers], "seconds": seconds}) + "\n")
            if i % 100 == 0:
                print("boundedfield", i, "of", len(cases), flush=True)


if __name__ == "__main__":
    main()
