"""
Ranks one test set for every job on one retriever, cosine alone, BoundedField or PPR, in one pass over the vectors

Every case's seeds come from one GPU pass, each distinct pool is built once, and every job sharing a pool reads it
A ranking is cut to the set's corpus first and to RETURN after, so a corpus-bound set never loses answers to the cut
"""

import os

# one BLAS thread per worker, set before numpy loads or the forked workers oversubscribe the cores
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

import queries as query_cache
from engine import Store, Stream, bf_order, day_of, ppr_orders
from src.core.config import load_config

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "benchmarks" / "data" / "cases"
RETURN = 1000
BATCH = 1024
STATE: dict[str, Any] = {}


def spec_key(recipe: dict[str, Any]) -> tuple[int, int, int, bool]:
    return (int(recipe["CORE"]), int(recipe["POOL"]), int(recipe["HOPS"]), bool(recipe.get("EXPAND", True)))


def pool_case(j: int) -> dict[tuple, np.ndarray]:
    store: Store = STATE["store"]
    c, (seed, seed_cos), blocked = STATE["cases"][j], STATE["seeds"][j], STATE["blocked"][j]
    cap = day_of(c["cutoff"])
    return {k: store.pool_rows(cap, blocked, seed, seed_cos, {"CORE": k[0], "POOL": k[1], "HOPS": k[2], "EXPAND": k[3]}).astype(np.int32) for k in STATE["specs"]}


def cut(order: np.ndarray) -> np.ndarray:
    allowed = STATE["allowed"]
    if allowed is not None:
        order = order[allowed[order]]
    return order[:RETURN].astype(np.int32)


def rank_case(j: int) -> list[np.ndarray]:
    """
    One case ranked under every job, each pool's terms and graph built once
    """
    store: Store = STATE["store"]
    c, (seed, _), blocked = STATE["cases"][j], STATE["seeds"][j], STATE["blocked"][j]
    cap = day_of(c["cutoff"])
    union, ucos, pools = STATE["union"][j], STATE["cos"][j], STATE["pools"][j]
    postings = store.bm25_postings(c["query"], union) if STATE["any_bf"] else []
    terms: dict[tuple, np.ndarray] = {}
    out = []
    for job in STATE["jobs"]:
        if job["method"] == "none":
            out.append(cut(seed))
            continue
        recipe = job["recipe"]
        key = spec_key(recipe)
        rows = pools[key].astype(np.int64)
        cos = ucos[np.searchsorted(union, rows)]
        if job["method"] == "bf":
            if key not in terms:
                terms[key] = store.terms(rows, cos, cap, c["query"], blocked, key[0], STATE["depth"], STATE["decay"], postings)
            out.append(cut(rows[bf_order(terms[key], recipe["weights"])]))
        else:
            p = recipe["ppr"]
            order = ppr_orders(store, rows, cos, key[0], [(float(p["ALPHA"]), p["TELEPORT"], p["DIRECTION"])])[0]
            out.append(cut(rows[order]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--retriever", required=True)
    _ = ap.add_argument("--dataset", required=True)
    _ = ap.add_argument("--split", required=True)
    _ = ap.add_argument("--jobs", required=True, help="a json file: each job's name, method and effective recipe")
    _ = ap.add_argument("--corpus", default="", help="a file of the ids a ranking is cut to first")
    _ = ap.add_argument("--out", required=True, help="a directory, one <job>.jsonl per job")
    _ = ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    started = time.perf_counter()
    jobs = json.loads(Path(args.jobs).read_text())
    settings = load_config(str(ROOT / "config.toml"))["retrieve"]
    with open(CASES / args.dataset / f"{args.split}.jsonl") as f:
        cases = [json.loads(line) for line in f]
    queries = query_cache.read(args.retriever, args.dataset, args.split)
    store = Store()
    stream = Stream(store, args.retriever)
    blocked = [store.rows_of(c["anchors"]) for c in cases]
    caps = np.array([day_of(c["cutoff"]) for c in cases])
    specs = sorted({spec_key(j["recipe"]) for j in jobs if j["method"] != "none"})
    depth = max([int(settings["COSINE_LIMIT"])] + [s[1] for s in specs])
    allowed = None
    if args.corpus:
        allowed = np.zeros(store.n, bool)
        allowed[store.rows_of(Path(args.corpus).read_text().split())] = True
    seeds: list[tuple[np.ndarray, np.ndarray]] = []
    for s in range(0, len(cases), BATCH):
        seeds += stream.seeds(queries[s : s + BATCH], caps[s : s + BATCH], blocked[s : s + BATCH], depth)
    STATE.update(store=store, cases=cases, seeds=seeds, blocked=blocked, specs=specs, jobs=jobs, allowed=allowed)
    STATE.update(depth=int(settings["DEPTH"]), decay=float(settings["DECAY"]), any_bf=any(j["method"] == "bf" for j in jobs))
    t_seed = time.perf_counter()
    pools: list[dict[tuple, np.ndarray]] = [{} for _ in cases]
    if specs:
        with mp.get_context("fork").Pool(args.workers) as workers:
            pools = workers.map(pool_case, range(len(cases)), chunksize=4)
    union = [np.unique(np.concatenate([np.zeros(0, np.int32)] + list(p.values()))).astype(np.int64) for p in pools]
    cos: list[np.ndarray] = []
    for s in range(0, len(cases), BATCH):
        cos += stream.cosines(queries[s : s + BATCH], union[s : s + BATCH]) if specs else [np.zeros(0)] * len(union[s : s + BATCH])
    del stream
    STATE.update(pools=pools, union=union, cos=cos)
    t_pool = time.perf_counter()
    with mp.get_context("fork").Pool(args.workers) as workers:
        ranked = workers.map(rank_case, range(len(cases)), chunksize=2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, job in enumerate(jobs):
        with open(out / f"{job['name']}.jsonl", "w") as f:
            for c, r in zip(cases, ranked, strict=True):
                f.write(json.dumps({"sample_id": c["sample_id"], "ranked": [bytes(store.ids[x]).decode() for x in r[i]]}) + "\n")
    print(f"rank {args.retriever} {args.dataset}_{args.split}: {len(cases)} cases, {len(jobs)} jobs, {len(specs)} pools, seeds {t_seed - started:.0f}s, pools {t_pool - t_seed:.0f}s, rank {time.perf_counter() - t_pool:.0f}s", flush=True)


if __name__ == "__main__":
    main()
