"""
Caches every case's pool and score terms at each core size, for tune.py, in GraphIV's own environment

The expanders hop from the core, so each core builds its own pool on top of one shared cosine retrieval
Cases are written in shards of SHARD, each one memory mapped by fit.py and skipped here once written
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import load_config
from src.core.embed import EmbeddingEndpoint, embed
from src.core.parse_graph_export import load_graph_export
from src.core.schema import Query
from src.retrieve.query import PaperQuery
from src.retrieve.recipe import build_pipeline, get_seed_limit
from src.retrieve.scorer import GraphScorer, Weights

ROOT = Path(__file__).resolve().parents[3]
TERMS = list(Weights.__dataclass_fields__)
SHARD = 250
WORKERS = 8

# each worker's pipelines, one per core, built once by its initializer
STATE: dict[str, Any] = {}


def start_worker(graph_path: str, cores: list[int], vectors: dict[str, str] | None = None) -> None:
    """
    Builds one worker's pipeline per core from config.toml's [retrieve] block

    Args:
        graph_path: The frozen graph export.
        cores: The core sizes every case is pooled and scored at.
        vectors: Another embedder's npy, corpus arrow and id array, whose exact pool replaces the store's.
    """
    config = load_config(str(ROOT / "config.toml"))
    graph = load_graph_export(graph_path)
    if vectors is None:
        dsn = str(config["database"]["PG_DEST"])
        pipelines = {c: build_pipeline({"CORE": c, "weights": {}}, config["retrieve"], graph, dsn) for c in cores}
    else:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        import embedder

        embedder.STATE.update(
            corpus=ipc.open_file(pa.memory_map(vectors["arrow"])).read_all(),
            ids=np.load(vectors["ids"], mmap_mode="r"),
            vectors=np.load(vectors["vectors"], mmap_mode="r"),
        )
        r = config["retrieve"]
        hops = int(r["EXPANSION_LIMIT"])
        pipelines = {
            c: PaperQuery(
                embedder.PoolRetriever(),
                [maker(graph, "", top_k=c, hop_limit=hops) for maker in (embedder.Ancestors, embedder.Frontiers)],
                [hops, hops],
                [GraphScorer(graph, Weights(), core=c, depth=int(r["DEPTH"]), decay=float(r["DECAY"]))],
            )
            for c in cores
        }
    STATE.update(config=config, pipelines=pipelines, limit=get_seed_limit(config["retrieve"]))


def pool_case(task: dict[str, Any] | tuple[dict[str, Any], np.ndarray, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """
    One case pooled and scored at every core, its anchors blocked from every step

    Args:
        task: The case, with query, cutoff, anchors and key, or the case with its exact pool rows and query vector.

    Returns:
        The terms per core, the answer mask per core over that core's pool and the answer's size.
    """
    pipelines = STATE["pipelines"]
    case = task[0] if isinstance(task, tuple) else task
    for pipeline in pipelines.values():
        pipeline.block(case["anchors"])
    cap = date.fromisoformat(case["cutoff"])
    retriever = next(iter(pipelines.values())).retriever
    if isinstance(task, tuple):
        vector = task[2].tolist()
        retriever.rows = task[1]
    else:
        vector = embed(EmbeddingEndpoint.from_config(STATE["config"]), case["query"])
    query = Query(text=case["query"], emb=vector, max_date=cap)
    first = retriever.retrieve(query, STATE["limit"])
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


def pool_exactly(args: argparse.Namespace, todo: list[tuple[str, list[dict[str, Any]]]]) -> tuple[dict[str, str], dict[str, list[Any]]]:
    """
    Embeds every uncached case through the router and takes its exact pool from the embedder's npy on the GPU

    Args:
        args: The parsed arguments, carrying the embedder.
        todo: The uncached shards.

    Returns:
        The paths each worker opens, and each shard's cases with their pool rows and query vectors.
    """
    import pyarrow.parquet as pq

    import embedder

    index = Path(args.vectors).parent
    meta = pq.read_table(index / "corpus.parquet", columns=["arxiv_id", "date"])
    ids = meta["arxiv_id"].to_pylist()
    row = {a: i for i, a in enumerate(ids)}
    days = meta["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    cases = [c for _, chunk in todo for c in chunk]
    embedder.load(args.endpoint)
    queries = embedder.embed_queries(args.endpoint, args.prompt, [c["query"] for c in cases])
    print(f"terms: embedded {len(queries)} queries", flush=True)
    if args.unload:
        embedder.router(args.endpoint, "unload")
    caps = np.array([np.datetime64(c["cutoff"], "D").astype(np.int64) for c in cases])
    anchors = [[row[a] for a in c["anchors"] if a in row] for c in cases]
    limit = get_seed_limit(load_config(str(ROOT / "config.toml"))["retrieve"])
    pools = embedder.top_rows_batched(np.load(args.vectors, mmap_mode="r"), queries, caps, anchors, days, limit)
    print(f"terms: pooled {len(pools)} cases", flush=True)
    tasks: dict[str, list[Any]] = {}
    i = 0
    for name, chunk in todo:
        tasks[name] = [(c, pools[i + j], queries[i + j]) for j, c in enumerate(chunk)]
        i += len(chunk)
    paths = {"vectors": args.vectors, "arrow": str(embedder.corpus_arrow(index)), "ids": str(embedder.corpus_ids(index, ids))}
    return paths, tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--graph", required=True)
    _ = ap.add_argument("--cases", required=True)
    _ = ap.add_argument("--cores", required=True, help="json list of core sizes")
    _ = ap.add_argument("--out", required=True, help="the split's directory")
    _ = ap.add_argument("--vectors", help="another embedder's npy in index_cache, pooled exactly instead of from the store")
    _ = ap.add_argument("--endpoint", help="that embedder's router path")
    _ = ap.add_argument("--prompt", default="", help="that embedder's query prompt")
    _ = ap.add_argument("--unload", action="store_true", help="unload that embedder from the router once the queries are embedded")
    _ = ap.add_argument("--workers", type=int, default=WORKERS)
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
    vectors, tasks = None, {name: chunk for name, chunk in todo}
    if args.vectors:
        vectors, tasks = pool_exactly(args, todo)
    started, seen, total = time.perf_counter(), 0, sum(len(c) for _, c in todo)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    with mp.get_context("spawn").Pool(args.workers, initializer=start_worker, initargs=(args.graph, cores, vectors)) as workers:
        for name, chunk in todo:
            write_shard(out, name, cores, list(workers.imap(pool_case, tasks[name], chunksize=4)))
            seen += len(chunk)
            print(f"terms {out.name}: {name}, {seen} of {total} cases, {(time.perf_counter() - started) / seen:.2f}s a case", flush=True)


if __name__ == "__main__":
    main()
