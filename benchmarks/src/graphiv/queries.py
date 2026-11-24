"""
Embeds every case's question of the named splits through the router, once per retriever, into cache/<retriever>/queries/

A cache file holds one unit vector per case in the split's file order, beside a json of its ids and a hash of its texts
Run before anything touches the GPU, since the router cannot load a large embedder beside another process's CUDA context
"""

import argparse
import hashlib
import json
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
CACHE = HERE / "data" / "cache"
CASES = HERE / "data" / "cases"
BATCH = 64


def embed_queries(endpoint: str, prompt: str, texts: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        body = json.dumps({"input": [prompt + t for t in texts[i : i + BATCH]]}).encode()
        request = urllib.request.Request(f"{endpoint}/v1/embeddings", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as r:
            out += [d["embedding"] for d in sorted(json.load(r)["data"], key=lambda d: d["index"])]
    v = np.asarray(out, np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def router(endpoint: str, action: str) -> bool:
    base, model = endpoint.split("/embedding/")
    request = urllib.request.Request(f"{base}/embeddings/{action}", json.dumps({"model": model}).encode(), {"Content-Type": "application/json"})
    try:
        return bool(json.load(urllib.request.urlopen(request, timeout=300)).get("success", True))
    except urllib.error.HTTPError:
        return False


def load(endpoint: str) -> None:
    for _ in range(5):
        if router(endpoint, "load"):
            return
        time.sleep(10)
    raise RuntimeError(f"{endpoint} never loaded")


def texts_hash(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]


def cached(retriever: str, dataset: str, split: str, texts: list[str]) -> bool:
    meta = CACHE / retriever / "queries" / f"{dataset}_{split}.json"
    return meta.exists() and json.loads(meta.read_text())["texts"] == texts_hash(texts)


def write(retriever: str, dataset: str, split: str, cases: list[dict], vectors: np.ndarray) -> None:
    out = CACHE / retriever / "queries"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{dataset}_{split}.npy", vectors.astype(np.float32))
    meta = {"sample_ids": [c["sample_id"] for c in cases], "texts": texts_hash([c["query"] for c in cases]), "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / f"{dataset}_{split}.json").write_text(json.dumps(meta))


def read(retriever: str, dataset: str, split: str) -> np.ndarray:
    """
    One split's cached query vectors, checked against the split's cases

    Args:
        retriever: A section of tuning/embedders.toml.
        dataset: The dataset.
        split: The split.

    Returns:
        One vector per case, in file order.

    Raises:
        RuntimeError: When the cache is missing or holds other cases.
    """
    with open(CASES / dataset / f"{split}.jsonl") as f:
        texts = [json.loads(line)["query"] for line in f]
    if not cached(retriever, dataset, split, texts):
        raise RuntimeError(f"no query cache for {retriever} {dataset}_{split}, run queries.py first")
    return np.load(CACHE / retriever / "queries" / f"{dataset}_{split}.npy")


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--retriever", required=True, help="a section of tuning/embedders.toml")
    _ = ap.add_argument("--splits", required=True, help="json list of [dataset, split] pairs")
    args = ap.parse_args()
    spec = tomllib.loads((HERE / "tuning" / "embedders.toml").read_text())[args.retriever]
    todo = []
    for dataset, split in json.loads(args.splits):
        with open(CASES / dataset / f"{split}.jsonl") as f:
            cases = [json.loads(line) for line in f]
        if not cached(args.retriever, dataset, split, [c["query"] for c in cases]):
            todo.append((dataset, split, cases))
    if not todo:
        return
    load(spec["endpoint"])
    for dataset, split, cases in todo:
        started = time.perf_counter()
        write(args.retriever, dataset, split, cases, embed_queries(spec["endpoint"], spec["prompt"], [c["query"] for c in cases]))
        print(f"queries {args.retriever} {dataset}_{split}: {len(cases)} in {time.perf_counter() - started:.0f}s", flush=True)
    if spec.get("unload"):
        router(spec["endpoint"], "unload")


if __name__ == "__main__":
    main()
