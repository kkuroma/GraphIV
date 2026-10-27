"""
Graph-based Adaptive Re-ranking through pyterrier_adaptive, scoring 1000 papers of [run] first_stage with OpenScholar

[run] variant is rerank for GAR disabled, sim for 16 nomic neighbours, or cite for the citation graph, after [run] warm
A neighbour past the case's cutoff, or one of its anchors, is never offered, so no variant sees the future
"""

from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyterrier as pt
import torch
from FlagEmbedding import FlagReranker
from pyterrier_adaptive import GAR as Adaptive

from src import common
from src.schema import Case, Method, Ranking

BUDGET = 1000
# b in the GAR paper
BATCH = 16
NEIGHBOURS = 16


class Allowed:
    """
    Which corpus rows a case may see, shared by both graphs and moved per case
    """

    def __init__(self, dates: np.ndarray) -> None:
        self.dates = dates
        self.mask = np.ones(len(dates), dtype=bool)

    def set(self, cutoff: str, blocked: list[int]) -> None:
        self.mask = self.dates <= np.datetime64(cutoff)
        self.mask[blocked] = False


class CiteGraph:
    """
    A paper's references and citers, from the frozen graph
    """

    def __init__(self, ids: list[str], allowed: Allowed) -> None:
        graph = common.load_graph()
        self.refs_ptr, self.refs_idx = graph.refs_ptr, graph.refs_idx
        self.cites_ptr, self.cites_idx = graph.cites_ptr, graph.cites_idx
        row = {a: i for i, a in enumerate(ids)}
        # export row to corpus row, -1 where the corpus lacks it
        self.to_corpus = np.array([row.get(a, -1) for a in graph.ids])
        self.to_export = graph.index
        self.ids, self.allowed = ids, allowed

    def reset(self) -> None:
        pass

    def neighbours(self, docno: str) -> list[str]:
        e = self.to_export.get(docno)
        if e is None:
            return []
        rows = np.concatenate([self.refs_idx[self.refs_ptr[e] : self.refs_ptr[e + 1]], self.cites_idx[self.cites_ptr[e] : self.cites_ptr[e + 1]]])
        rows = self.to_corpus[rows]
        rows = rows[rows >= 0]
        return [self.ids[r] for r in rows[self.allowed.mask[rows]]]


class SimGraph:
    """
    A paper's nearest neighbours by nomic cosine among what the case may see, computed on demand
    """

    def __init__(self, vectors: np.ndarray, ids: list[str], allowed: Allowed) -> None:
        # normalized a chunk at a time, since the whole fp32 matrix does not fit beside the reranker
        unit = np.empty_like(vectors)
        for i in range(0, len(vectors), 500_000):
            v = vectors[i : i + 500_000].astype(np.float32)
            unit[i : i + 500_000] = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.emb = torch.from_numpy(unit).cuda()
        self.row = {a: i for i, a in enumerate(ids)}
        self.ids, self.allowed = ids, allowed
        self.cache: dict[str, list[str]] = {}

    def reset(self) -> None:
        self.cache = {}
        self.mask = torch.from_numpy(self.allowed.mask).cuda()

    def neighbours(self, docno: str) -> list[str]:
        if docno not in self.cache:
            r = self.row[docno]
            score = self.emb @ self.emb[r]
            score[~self.mask] = -2.0
            score[r] = -2.0
            self.cache[docno] = [self.ids[i] for i in torch.topk(score, NEIGHBOURS).indices.tolist()]
        return self.cache[docno]


class Scorer:
    """
    The OpenScholar reranker as a pyterrier scorer, remembering pairs so a case scores a paper once
    """

    def __init__(self, text: dict[str, str]) -> None:
        self.reranker = FlagReranker("OpenScholar/OpenScholar_Reranker", use_fp16=True)
        self.text = text
        self.seen: dict[tuple[str, str], float] = {}

    def __call__(self, df: pd.DataFrame) -> list[float]:
        todo = [(q, d) for q, d in zip(df["query"], df["docno"]) if (q, d) not in self.seen]
        if todo:
            scores = self.reranker.compute_score([[q, self.text[d]] for q, d in todo], batch_size=100)
            self.seen.update(zip(todo, np.atleast_1d(scores).tolist()))
        return [self.seen[(q, d)] for q, d in zip(df["query"], df["docno"])]


class GAR(Method):
    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        first = {r["sample_id"]: r["ranked"] for r in common.read_jsonl(common.RUNS / f"{params['first_stage']}.jsonl")}
        table = pq.read_table(common.corpus() / "corpus.parquet")
        ids = table["arxiv_id"].to_pylist()
        dates = np.array(table["date"].to_pylist(), dtype="datetime64[D]")
        text = {a: t + " " + b for a, t, b in zip(ids, table["title"].to_pylist(), table["abstract"].to_pylist())}
        del table
        row = {a: i for i, a in enumerate(ids)}
        allowed = Allowed(dates)
        variant = params["variant"]
        # warm variants run first on each case and are discarded, leaving their scores behind
        order = [*params.get("warm", []), variant]
        graphs = {"cite": CiteGraph(ids, allowed)}
        if "sim" in order:
            graphs["sim"] = SimGraph(np.load(common.INDEX / "nomic.npy"), ids, allowed)
        scorer = Scorer(text)
        score = pt.apply.doc_score(scorer, batch_size=BATCH)
        pipes = [Adaptive(score, graphs.get(v, graphs["cite"]), num_results=BUDGET, batch_size=BATCH, enabled=v != "rerank") for v in order]
        out = []
        for n, c in enumerate(cases, 1):
            allowed.set(c["cutoff"], [row[a] for a in c["anchors"] if a in row])
            for g in graphs.values():
                g.reset()
            initial = pd.DataFrame({"qid": "q", "query": c["query"], "docno": first[c["sample_id"]][:BUDGET], "score": np.arange(BUDGET, 0, -1, dtype=float)})
            initial["rank"] = np.arange(len(initial))
            # the initial papers in large batches first, which GAR then reads from the cache
            scorer(initial)
            ranked = [p(initial) for p in pipes][-1].sort_values("rank")["docno"].tolist()
            out.append({"sample_id": c["sample_id"], "ranked": ranked})
            scorer.seen.clear()
            if n % 25 == 0:
                common.say(f"gar {variant} {n} of {len(cases)}")
        return out


METHOD = GAR
