"""
Caches every case's pool and score terms at each core size, for tune.py, in GraphIV's own environment

The expanders hop from the core, so each core builds its own pool on top of one shared cosine retrieval
Cases are written in shards of SHARD, each one memory mapped by fit.py and skipped here once written
"""

import argparse
import json
import multiprocessing as mp
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import load_config
from src.core.embed import embed
from src.core.parse_graph_export import load_graph_export
from src.core.schema import Query
from src.retrieve.recipe import build_pipeline, get_seed_limit
from src.retrieve.scorer import GraphScorer, Weights

ROOT = Path(__file__).resolve().parents[3]
TERMS = list(Weights.__dataclass_fields__)
SHARD = 250
WORKERS = 6

# each worker's pipelines, one per core, built once by its initializer
STATE: dict[str, Any] = {}


def start_worker(graph_path: str, cores: list[int]) -> None:
    """
    Builds one worker's pipeline per core from config.toml's [retrieve] block

    Args:
        graph_path: The frozen graph export.
        cores: The core sizes every case is pooled and scored at.
    """
    config = load_config(str(ROOT / "config.toml"))
    graph = load_graph_export(graph_path)
    dsn = str(config["database"]["PG_DEST"])
    pipelines = {c: build_pipeline({"CORE": c, "weights": {}}, config["retrieve"], graph, dsn) for c in cores}
    STATE.update(config=config, pipelines=pipelines, limit=get_seed_limit(config["retrieve"]))


def pool_case(case: dict[str, Any]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """
    One case pooled and scored at every core, its anchors blocked from every step

    Args:
        case: The case, with query, cutoff, anchors and key.

    Returns:
        The terms per core, the answer mask per core over that core's pool and the answer's size.
    """
    endpoints = STATE["config"]["endpoints"]
    pipelines = STATE["pipelines"]
    for pipeline in pipelines.values():
        pipeline.block(case["anchors"])
    cap = date.fromisoformat(case["cutoff"])
    vector = embed(str(endpoints["EMBED_API"]), str(endpoints["EMBED_NAME"]), case["query"])
    query = Query(text=case["query"], emb=vector, max_date=cap)
    first = next(iter(pipelines.values())).retriever.retrieve(query, STATE["limit"])
    answer = set(case["key"])
    terms, hits = {}, {}
    for core, pipeline in pipelines.items():
        pool = list(first)
        for expander, limit in zip(pipeline.expanders, pipeline.limits, strict=True):
            pool = expander.rerank(query, pool, limit)
        scorer = pipeline.scorers[0]
        assert isinstance(scorer, GraphScorer)
        rows = scorer.get_paper_rows(pool)
        each = scorer.get_individual_score(pool, rows, scorer.graph.get_day_number(cap), case["query"])
        terms[core] = np.stack([each[t] for t in TERMS], axis=1).astype(np.float32)
        hits[core] = np.array([p.arxiv_id in answer for p in pool], np.bool_)
    return terms, hits, len(answer)


def write_shard(split: Path, shard: str, cores: list[int], done: list[tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]]) -> None:
    """
    Writes one shard per core, one .npy per array since an .npz is never memory mapped

    Args:
        split: The split's directory, holding core-<c>/shard-<i>/.
        shard: The shard's directory name.
        cores: The core sizes.
        done: The shard's pooled cases.
    """
    n_key = np.array([d[2] for d in done], np.int32)
    for core in cores:
        target = split / f"core-{core}" / shard
        if target.exists():
            continue
        part = split / "staging" / f"{shard}.part"
        part.mkdir(parents=True, exist_ok=True)
        np.save(part / "terms.npy", np.concatenate([d[0][core] for d in done]))
        np.save(part / "hits.npy", np.concatenate([d[1][core] for d in done]))
        np.save(part / "offsets.npy", np.concatenate([[0], np.cumsum([len(d[1][core]) for d in done])]))
        np.save(part / "n_key.npy", n_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        part.rename(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--graph", required=True)
    _ = ap.add_argument("--cases", required=True)
    _ = ap.add_argument("--cores", required=True, help="json list of core sizes")
    _ = ap.add_argument("--out", required=True, help="the split's directory")
    args = ap.parse_args()
    cores = json.loads(args.cores)
    with open(args.cases) as f:
        cases = [json.loads(line) for line in f]
    out = Path(args.out)
    shards = [(f"shard-{i:03d}", cases[s : s + SHARD]) for i, s in enumerate(range(0, len(cases), SHARD))]
    todo = [(name, chunk) for name, chunk in shards if not all((out / f"core-{c}" / name).exists() for c in cores)]
    print(f"terms {out.name}: {len(cases)} cases, {len(shards) - len(todo)} of {len(shards)} shards cached", flush=True)
    if not todo:
        return
    started, seen, total = time.perf_counter(), 0, sum(len(c) for _, c in todo)
    with mp.get_context("spawn").Pool(WORKERS, initializer=start_worker, initargs=(args.graph, cores)) as workers:
        for name, chunk in todo:
            write_shard(out, name, cores, list(workers.imap(pool_case, chunk, chunksize=4)))
            seen += len(chunk)
            print(f"terms {out.name}: {name}, {seen} of {total} cases, {(time.perf_counter() - started) / seen:.2f}s a case", flush=True)


if __name__ == "__main__":
    main()
