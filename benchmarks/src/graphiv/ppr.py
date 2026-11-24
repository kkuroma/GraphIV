"""
Tunes PPR on a seeded sample of one split: every core and pool size, every damping, teleport set and direction

The pick is the first setting reaching the best mean recall@k, in the order the grid lists them
Rank flows over the same pools BoundedField scores, built from the same seeds and expanders
"""

import os

# one BLAS thread per worker, set before numpy loads or the forked workers oversubscribe the cores
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import itertools
import json
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

import queries as query_cache
from engine import Store, Stream, day_of, ppr_orders
from src.core.config import load_config

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "benchmarks" / "data" / "cases"
BATCH = 1024
STATE: dict[str, Any] = {}


def grid_case(j: int) -> list[float]:
    """
    One case's recall@k under every (core, pool) spec and every setting, specs outer
    """
    store: Store = STATE["store"]
    c, (seed, seed_cos), blocked = STATE["cases"][j], STATE["seeds"][j], STATE["blocked"][j]
    cap, k = day_of(c["cutoff"]), STATE["k"]
    key = set(store.rows_of(c["key"]).tolist())
    n_key = len(set(c["key"]))
    out = []
    for core, pool in STATE["specs"]:
        rows = store.pool_rows(cap, blocked, seed, seed_cos, {"CORE": core, "POOL": pool, "HOPS": STATE["hops"]})
        cos = STATE["cos"][j][np.searchsorted(STATE["union"][j], rows)]
        for order in ppr_orders(store, rows, cos, core, STATE["settings"]):
            out.append(len(set(rows[order[:k]].tolist()) & key) / n_key)
    return out


def pool_union(j: int) -> np.ndarray:
    store: Store = STATE["store"]
    c, (seed, seed_cos), blocked = STATE["cases"][j], STATE["seeds"][j], STATE["blocked"][j]
    cap = day_of(c["cutoff"])
    pools = [store.pool_rows(cap, blocked, seed, seed_cos, {"CORE": core, "POOL": pool, "HOPS": STATE["hops"]}) for core, pool in STATE["specs"]]
    return np.unique(np.concatenate(pools))


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--retriever", required=True)
    _ = ap.add_argument("--dataset", required=True)
    _ = ap.add_argument("--spec", required=True, help="json: the [ppr] table of the tuning toml, with cores and pools")
    _ = ap.add_argument("--out", required=True, help="the report json")
    _ = ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    started = time.perf_counter()
    spec = json.loads(args.spec)
    settings = load_config(str(ROOT / "config.toml"))["retrieve"]
    with open(CASES / args.dataset / f"{spec['split']}.jsonl") as f:
        cases = [json.loads(line) for line in f]
    queries = query_cache.read(args.retriever, args.dataset, spec["split"])
    order = list(range(len(cases)))
    random.Random(spec["seed"]).shuffle(order)
    order = order[: spec["cases"]]
    cases, queries = [cases[i] for i in order], queries[order]
    store = Store()
    stream = Stream(store, args.retriever)
    blocked = [store.rows_of(c["anchors"]) for c in cases]
    caps = np.array([day_of(c["cutoff"]) for c in cases])
    specs = [(c, p) for c in spec["cores"] for p in spec["pools"]]
    grid = list(itertools.product(spec["alphas"], spec["teleports"], spec["directions"]))
    seeds = stream.seeds(queries, caps, blocked, max(p for _, p in specs))
    STATE.update(store=store, cases=cases, seeds=seeds, blocked=blocked, specs=specs, settings=grid, k=spec["k"], hops=int(settings["EXPANSION_LIMIT"]))
    with mp.get_context("fork").Pool(args.workers) as workers:
        union = workers.map(pool_union, range(len(cases)), chunksize=2)
    cos = stream.cosines(queries, union)
    STATE.update(union=union, cos=cos)
    with mp.get_context("fork").Pool(args.workers) as workers:
        means = np.array(workers.map(grid_case, range(len(cases)), chunksize=1)).mean(0)
    labels = [f"{c}/{p}/{a}/{t}/{d}" for c, p in specs for a, t, d in grid]
    best = int(np.argmax(means))
    core, pool, alpha, teleport, direction = labels[best].split("/")
    report = {
        "k": spec["k"], "cases": len(cases), "sample_ids": [c["sample_id"] for c in cases],
        "pick": {"core": int(core), "pool": int(pool), "alpha": float(alpha), "teleport": teleport, "direction": direction, "recall": float(means[best])},
        "grid": dict(zip(labels, means.tolist(), strict=True)),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"ppr {args.retriever} {args.dataset}_{spec['split']}: {len(cases)} cases, {len(labels)} settings, pick {labels[best]} R@{spec['k']} {means[best]:.4f}, {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
