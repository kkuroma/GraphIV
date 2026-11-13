"""
Plain nomic cosine from the store's HNSW index, capped at each case's cutoff with its anchors blocked

The search asks for 5,000 and keeps the head, the same call that built the pool GAR reranks
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

from src.core.config import load_config
from src.core.embed import EmbeddingEndpoint, embed
from src.core.schema import Query
from src.retrieve.cosine import CosineRetriever

ROOT = Path(__file__).resolve().parents[3]
LIMIT = 5000
RETURN = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--cases", required=True)
    _ = ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = load_config(str(ROOT / "config.toml"))
    endpoint = EmbeddingEndpoint.from_config(config)
    retriever = CosineRetriever(str(config["database"]["PG_DEST"]))
    with open(args.cases) as f:
        cases = [json.loads(line) for line in f]
    with open(args.out, "w") as f:
        for i, c in enumerate(cases):
            started = time.perf_counter()
            retriever.blocked = c["anchors"]
            vector = embed(endpoint, c["query"])
            papers = retriever.retrieve(Query(text=c["query"], emb=vector, max_date=date.fromisoformat(c["cutoff"])), LIMIT)
            seconds = time.perf_counter() - started
            f.write(json.dumps({"sample_id": c["sample_id"], "ranked": [p.arxiv_id for p in papers][:RETURN], "seconds": seconds}) + "\n")
            if i % 100 == 0:
                print("nomic", i, "of", len(cases), flush=True)


if __name__ == "__main__":
    main()
