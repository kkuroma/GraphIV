"""
Caches every train and val case's pool and score terms at each core and pool size of a BoundedField tune, for fit.py

Seeds and cosines stream through the GPU in the main process, then pools and terms run in forked workers
Each worker writes its shard straight to disk, one .npy per array since fit.py memory maps them
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
from engine import Store, Stream, day_of
from src.core.config import load_config

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "benchmarks" / "data" / "cases"
SHARD = 32
BATCH = 1024
# cases held in memory at once, a multiple of SHARD, so a 30k-case split never holds every pool
BLOCK = 4096
STATE: dict[str, Any] = {}


def recipe_of(spec: tuple[int, int]) -> dict[str, Any]:
    return {"CORE": spec[0], "POOL": spec[1], "HOPS": STATE["hops"]}


def pool_case(j: int) -> dict[tuple[int, int], np.ndarray]:
    c, blocked = STATE["cases"][j], STATE["blocked"][j]
    seed, seed_cos = STATE["seeds"][j - STATE["base"]]
    return {s: STATE["store"].pool_rows(day_of(c["cutoff"]), blocked, seed, seed_cos, recipe_of(s)).astype(np.int32) for s in STATE["specs"]}


def write_shard(task: tuple[str, int, int]) -> int:
    """
    Scores one shard of cases at every spec and writes it, so no term array crosses a pipe

    Args:
        task: The split's directory, and the first and one past the last case.

    Returns:
        How many cases it wrote.
    """
    store: Store = STATE["store"]
    out, lo, hi = Path(task[0]), task[1], task[2]
    done: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {s: [] for s in STATE["specs"]}
    n_key = []
    for j in range(lo, hi):
        c, blocked = STATE["cases"][j], STATE["blocked"][j]
        cap = day_of(c["cutoff"])
        key = store.rows_of(c["key"])
        union, ucos = STATE["union"][j - STATE["base"]].astype(np.int64), STATE["cos"][j - STATE["base"]].astype(np.float64)
        postings = store.bm25_postings(c["query"], union)
        for s in STATE["specs"]:
            rows = STATE["pools"][j - STATE["base"]][s].astype(np.int64)
            cos = ucos[np.searchsorted(union, rows)]
            terms = store.terms(rows, cos, cap, c["query"], blocked, s[0], STATE["depth"], STATE["decay"], postings)
            done[s].append((terms.astype(np.float32), np.isin(rows, key)))
        n_key.append(len(set(c["key"])))
    for s, items in done.items():
        target = out / f"core-{s[0]}-pool-{s[1]}" / f"shard-{lo // SHARD:05d}"
        part = out / "staging" / f"{s[0]}-{s[1]}-{lo}"
        part.mkdir(parents=True, exist_ok=True)
        np.save(part / "terms.npy", np.concatenate([t for t, _ in items]))
        np.save(part / "hits.npy", np.concatenate([h for _, h in items]))
        np.save(part / "offsets.npy", np.concatenate([[0], np.cumsum([len(h) for _, h in items])]))
        np.save(part / "n_key.npy", np.array(n_key, np.int32))
        target.parent.mkdir(parents=True, exist_ok=True)
        part.rename(target)
    return hi - lo


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--retriever", required=True)
    _ = ap.add_argument("--dataset", required=True)
    _ = ap.add_argument("--splits", required=True, help="json list of the splits to cache")
    _ = ap.add_argument("--specs", required=True, help="json list of [core, pool] pairs")
    _ = ap.add_argument("--out", required=True, help="a directory, one subdirectory per split")
    _ = ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    settings = load_config(str(ROOT / "config.toml"))["retrieve"]
    specs = [tuple(s) for s in json.loads(args.specs)]
    store = Store()
    stream = Stream(store, args.retriever)
    STATE.update(store=store, specs=specs, hops=int(settings["EXPANSION_LIMIT"]), depth=int(settings["DEPTH"]), decay=float(settings["DECAY"]))
    for split in json.loads(args.splits):
        started = time.perf_counter()
        with open(CASES / args.dataset / f"{split}.jsonl") as f:
            cases = [json.loads(line) for line in f]
        queries = query_cache.read(args.retriever, args.dataset, split)
        blocked = [store.rows_of(c["anchors"]) for c in cases]
        caps = np.array([day_of(c["cutoff"]) for c in cases])
        depth = max(s[1] for s in specs)
        STATE.update(cases=cases, blocked=blocked)
        t_seed = t_pool = 0.0
        for b0 in range(0, len(cases), BLOCK):
            b1 = min(b0 + BLOCK, len(cases))
            t0 = time.perf_counter()
            seeds: list[tuple[np.ndarray, np.ndarray]] = []
            for s in range(b0, b1, BATCH):
                e = min(s + BATCH, b1)
                seeds += stream.seeds(queries[s:e], caps[s:e], blocked[s:e], depth)
            STATE.update(seeds=seeds, base=b0)
            t1 = time.perf_counter()
            with mp.get_context("fork").Pool(args.workers) as workers:
                pools = workers.map(pool_case, range(b0, b1), chunksize=8)
            union = [np.unique(np.concatenate(list(p.values()))) for p in pools]
            cos: list[np.ndarray] = []
            for s in range(0, len(union), BATCH):
                cos += [c.astype(np.float32) for c in stream.cosines(queries[b0 + s : b0 + s + BATCH], union[s : s + BATCH])]
            STATE.update(pools=pools, union=union, cos=cos)
            t2 = time.perf_counter()
            tasks = [(str(Path(args.out) / split), lo, min(lo + SHARD, b1)) for lo in range(b0, b1, SHARD)]
            with mp.get_context("fork").Pool(args.workers) as workers:
                _ = sum(workers.imap_unordered(write_shard, tasks))
            t_seed += t1 - t0
            t_pool += t2 - t1
        n = len(cases)
        total = time.perf_counter() - started
        print(f"terms {args.retriever} {args.dataset}_{split}: {n} cases x {len(specs)} specs, seeds {t_seed:.0f}s, pools {t_pool:.0f}s, all {total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
