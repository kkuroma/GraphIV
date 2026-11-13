"""
Ranks cases by another embedder's exact cosine over an index_cache npy, alone or as BoundedField's pool, for src/methods/embedder.py

Queries are embedded through the router, and every case's pool comes from one GPU pass over the vectors
A case's seconds are its BoundedField scoring alone, since the embedding and the pool are batched
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch

from src.core.config import load_config
from src.core.parse_graph_export import load_graph_export
from src.core.schema import Paper, Query
from src.retrieve.expand import AncestorExpander, FrontierExpander
from src.retrieve.query import PaperQuery
from src.retrieve.recipe import build_weights, get_seed_limit, read_recipe
from src.retrieve.scorer import GraphScorer

ROOT = Path(__file__).resolve().parents[3]
RETURN = 1000
BATCH = 64
CHUNK = 131072
MASKED = -2.0
STATE: dict[str, Any] = {}


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


def corpus_arrow(index: Path) -> Path:
    path = index / "corpus.arrow"
    if not path.exists():
        table = pq.read_table(index / "corpus.parquet")
        table = table.cast(pa.schema([f.with_type(pa.large_string()) if f.type == pa.string() else f for f in table.schema])).combine_chunks()
        tmp = index / "corpus.arrow.tmp"
        with pa.OSFile(str(tmp), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=table.num_rows)
        os.replace(tmp, path)
    return path


def corpus_ids(index: Path, ids: list[str]) -> Path:
    path = index / "corpus_ids.npy"
    if not path.exists():
        arr = np.array([a.encode() for a in ids])
        assert bool(np.all(arr[:-1] < arr[1:])), "corpus ids are not in byte order"
        np.save(index / "corpus_ids.tmp.npy", arr)
        os.replace(index / "corpus_ids.tmp.npy", path)
    return path


def top_rows(vectors: np.ndarray, queries: np.ndarray, caps: np.ndarray, anchors: list[list[int]], days: np.ndarray, limit: int) -> list[np.ndarray]:
    dev = "cuda"
    q = torch.from_numpy(queries).to(dev)
    cap = torch.from_numpy(caps).to(dev)
    aj = np.array([j for j, rows in enumerate(anchors) for _ in rows], np.int64)
    ar = np.array([r for rows in anchors for r in rows], np.int64)
    best_v = torch.full((len(q), limit), MASKED, device=dev)
    best_i = torch.full((len(q), limit), -1, dtype=torch.long, device=dev)
    for start in range(0, len(vectors), CHUNK):
        end = min(start + CHUNK, len(vectors))
        block = torch.from_numpy(np.ascontiguousarray(vectors[start:end])).to(dev).float()
        s = q @ block.T
        d = torch.from_numpy(days[start:end]).to(dev)
        s[d[None, :] > cap[:, None]] = MASKED
        hit = (ar >= start) & (ar < end)
        if hit.any():
            s[torch.from_numpy(aj[hit]).to(dev), torch.from_numpy(ar[hit] - start).to(dev)] = MASKED
        idx = torch.arange(start, end, device=dev).expand(len(q), -1)
        best_v, pos = torch.cat([best_v, s], 1).topk(limit, dim=1)
        best_i = torch.cat([best_i, idx], 1).gather(1, pos)
    rows, vals = best_i.cpu().numpy(), best_v.cpu().numpy()
    return [r[v > MASKED] for r, v in zip(rows, vals)]


def top_rows_batched(vectors: np.ndarray, queries: np.ndarray, caps: np.ndarray, anchors: list[list[int]], days: np.ndarray, limit: int, batch: int = 2048) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for s in range(0, len(queries), batch):
        out += top_rows(vectors, queries[s : s + batch], caps[s : s + batch], anchors[s : s + batch], days, limit)
        torch.cuda.empty_cache()
    return out


def papers_at(rows: np.ndarray, query: Query) -> list[Paper]:
    if len(rows) == 0:
        return []
    rows = np.sort(rows)
    t = STATE["corpus"].take(pa.array(rows))
    cosine = STATE["vectors"][rows].astype(np.float32) @ np.asarray(query.emb, np.float32)
    cap = query.max_date
    return [
        Paper(arxiv_id=a, title=ti or "", abstract=ab or "", date=d, cosine=float(c))
        for a, ti, ab, d, c in zip(t["arxiv_id"].to_pylist(), t["title"].to_pylist(), t["abstract"].to_pylist(), t["date"].to_pylist(), cosine)
        if cap is None or d <= cap
    ]


class PoolRetriever:
    def __init__(self) -> None:
        self.blocked: list[str] = []
        self.rows = np.zeros(0, np.int64)

    def retrieve(self, query: Query, limit: int) -> list[Paper]:
        return sorted(papers_at(self.rows[:limit], query), key=lambda p: -(p.cosine or 0.0))


class FromVectors:
    def rerank(self, query: Query, papers: list[Paper], limit: int | None) -> list[Paper]:
        wanted = self.get_neighbor_ids(papers)[: limit if limit is not None else self.hop_limit]  # type: ignore[attr-defined]
        if not wanted:
            return papers
        ids = STATE["ids"]
        w = np.array(wanted, dtype=ids.dtype)
        pos = np.minimum(np.searchsorted(ids, w), len(ids) - 1)
        return papers + papers_at(pos[ids[pos] == w].astype(np.int64), query)


class Ancestors(FromVectors, AncestorExpander):
    pass


class Frontiers(FromVectors, FrontierExpander):
    pass


def init(spec: dict[str, Any]) -> None:
    config = load_config(str(ROOT / "config.toml"))
    settings = config["retrieve"]
    graph = load_graph_export(spec["graph"])
    recipe = read_recipe(Path(spec["recipe"]))
    core, hops = int(recipe["CORE"]), int(settings["EXPANSION_LIMIT"])
    corpus = ipc.open_file(pa.memory_map(spec["arrow"])).read_all()
    retriever = PoolRetriever()
    expanders = [maker(graph, "", top_k=core, hop_limit=hops) for maker in (Ancestors, Frontiers)]
    scorer = GraphScorer(graph, build_weights(recipe["weights"]), core=core, depth=int(settings["DEPTH"]), decay=float(settings["DECAY"]))
    STATE.update(
        corpus=corpus,
        ids=np.load(spec["ids"], mmap_mode="r"),
        vectors=np.load(spec["vectors"], mmap_mode="r"),
        retriever=retriever,
        pipeline=PaperQuery(retriever, expanders, [hops, hops], [scorer]),
        allowed=set(Path(spec["corpus"]).read_text().split()) if spec["corpus"] else None,
    )


def score_case(task: tuple[dict[str, Any], np.ndarray, np.ndarray]) -> dict[str, Any]:
    c, rows, qvec = task
    started = time.perf_counter()
    cap = date.fromisoformat(c["cutoff"])
    emb = qvec.tolist()
    STATE["pipeline"].block(c["anchors"])
    STATE["retriever"].rows = rows
    pool = STATE["pipeline"].build_pool(c["query"], emb, retrieve_limit=len(rows), max_date=cap)
    papers = STATE["pipeline"].score_pool(c["query"], emb, pool, max_date=cap)
    allowed = STATE["allowed"]
    ranked = [p.arxiv_id for p in papers if allowed is None or p.arxiv_id in allowed][:RETURN]
    return {"sample_id": c["sample_id"], "ranked": ranked, "seconds": time.perf_counter() - started}


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--vectors", required=True)
    _ = ap.add_argument("--endpoint", required=True)
    _ = ap.add_argument("--prompt", default="")
    _ = ap.add_argument("--unload", action="store_true")
    _ = ap.add_argument("--recipe")
    _ = ap.add_argument("--graph", required=True)
    _ = ap.add_argument("--workers", type=int, default=8)
    _ = ap.add_argument("--cases", required=True)
    _ = ap.add_argument("--out", required=True)
    _ = ap.add_argument("--corpus")
    args = ap.parse_args()
    config = load_config(str(ROOT / "config.toml"))
    with open(args.cases) as f:
        cases = [json.loads(line) for line in f]
    index = Path(args.vectors).parent
    meta = pq.read_table(index / "corpus.parquet", columns=["arxiv_id", "date"])
    ids = meta["arxiv_id"].to_pylist()
    row = {a: i for i, a in enumerate(ids)}
    days = meta["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    load(args.endpoint)
    queries = embed_queries(args.endpoint, args.prompt, [c["query"] for c in cases])
    print("embedded", len(queries), flush=True)
    if args.unload:
        router(args.endpoint, "unload")
    caps = np.array([np.datetime64(c["cutoff"], "D").astype(np.int64) for c in cases])
    anchors = [[row[a] for a in c["anchors"] if a in row] for c in cases]
    vectors = np.load(args.vectors, mmap_mode="r")
    pools = top_rows(vectors, queries, caps, anchors, days, get_seed_limit(config["retrieve"]))
    print("pooled", len(pools), flush=True)
    torch.cuda.empty_cache()
    allowed = set(Path(args.corpus).read_text().split()) if args.corpus else None
    with open(args.out, "w") as f:
        if args.recipe is None:
            for c, rows in zip(cases, pools):
                ranked = [ids[r] for r in rows if allowed is None or ids[r] in allowed][:RETURN]
                f.write(json.dumps({"sample_id": c["sample_id"], "ranked": ranked}) + "\n")
            return
        spec = {"graph": args.graph, "recipe": args.recipe, "arrow": str(corpus_arrow(index)), "ids": str(corpus_ids(index, ids)), "vectors": args.vectors, "corpus": args.corpus}
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = "1"
        with mp.get_context("spawn").Pool(args.workers, initializer=init, initargs=(spec,)) as pool:
            for i, out in enumerate(pool.imap(score_case, zip(cases, pools, queries))):
                f.write(json.dumps(out) + "\n")
                if i % 100 == 0:
                    print("embedder", i, "of", len(cases), flush=True)


if __name__ == "__main__":
    main()
