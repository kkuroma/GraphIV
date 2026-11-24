"""
BoundedField and PPR over the in-memory graph and a retriever's vectors, the shipped pipeline step for step with no Postgres

Every term is the shipped scorer's formula, imported where it is a function, so a pool scored here matches src/retrieve/scorer.py
A retriever's vectors stream through the GPU in corpus order, once for each batch's seeds and once for its pooled cosines
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import scipy.sparse as sp
import torch

from src.retrieve.keyword import B, K1, get_query_words
from src.retrieve.scorer import compress_long_tail, normalize_to_max

DATA = Path(__file__).resolve().parents[2] / "data"
CACHE = DATA / "cache"
CHUNK = 131072
MASKED = -9.0
# pooled pairs per GPU slice, so the gathered rows stay small beside the router
PAIRS = 65536
# the score terms in the order a term cache stores them
TERMS = ["cos", "rec", "anc", "fro", "anc2", "fro2", "anc_decay", "fro_decay", "cocite", "couple", "bm25", "hub"]
EPOCH = np.datetime64("1970-01-01")


def day_of(when: str) -> int:
    return int((np.datetime64(when) - EPOCH).astype(int))


def gather(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Every edge position out of a set of CSR rows, and which of those rows each came from

    Args:
        ptr: The row pointers.
        rows: The rows to read.

    Returns:
        Edge positions, and the seat in `rows` of each.
    """
    start = ptr[rows].astype(np.int64)
    count = (ptr[rows + 1] - ptr[rows]).astype(np.int64)
    total = int(count.sum())
    if not total:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    seat = np.repeat(np.arange(len(rows)), count)
    offset = np.repeat(start - np.concatenate([[0], np.cumsum(count)[:-1]]), count)
    return offset + np.arange(total), seat


class Store:
    """
    Every graph array a query reads, held in memory and keyed by graph row
    """

    def __init__(self, root: Path = CACHE) -> None:
        def load(name: str, sub: str = "graph") -> np.ndarray:
            return np.load(root / sub / f"{name}.npy")

        self.ids = load("ids")
        self.n = len(self.ids)
        self.index = {a.decode(): i for i, a in enumerate(self.ids)}
        self.day = load("day").astype(np.int64)
        self.embedded = load("embedded")
        self.refs_ptr, self.refs_idx = load("refs_ptr"), load("refs_idx")
        self.cites_ptr, self.cites_idx, self.cites_key = load("cites_ptr"), load("cites_idx"), load("cites_key")
        self.post_ptr, self.post_doc, self.post_tf = load("post_ptr", "keyword"), load("post_doc", "keyword"), load("post_tf", "keyword")
        self.doc_len = load("doc_len", "keyword").astype(np.float64)
        words = ipc.open_file(pa.memory_map(str(root / "keyword" / "vocab.arrow"))).read_all()["word"].to_pylist()
        self.vocab = {w: i for i, w in enumerate(words)}
        # scratch maps from graph row to pool seat, reset after every use so no call allocates one
        self.seat = np.full(self.n, -1, np.int64)
        self.flag = np.zeros(self.n, bool)

    def rows_of(self, ids: list[str]) -> np.ndarray:
        return np.array([self.index[a] for a in ids if a in self.index], np.int64)

    def neighbors(self, rows: np.ndarray, forward: bool, limit: int) -> np.ndarray:
        """
        GraphExport.get_neighbors, the same numpy calls in the same order
        """
        if not len(rows):
            return np.zeros(0, np.int32)
        ptr, idx = (self.cites_ptr, self.cites_idx) if forward else (self.refs_ptr, self.refs_idx)
        pos, _ = gather(ptr, rows)
        reached = idx[pos]
        if not len(reached):
            return np.zeros(0, np.int32)
        uniq, hits = np.unique(reached, return_counts=True)
        if len(uniq) <= limit:
            return uniq[np.argsort(-hits)].astype(np.int32)
        keep = np.argpartition(-hits, limit)[:limit]
        return uniq[keep][np.argsort(-hits[keep])].astype(np.int32)

    def pool_rows(self, cap: int, blocked: np.ndarray, seed: np.ndarray, seed_cos: np.ndarray, recipe: dict[str, Any]) -> np.ndarray:
        """
        The pool a recipe scores: its POOL best seeds, then what its core cites, then what cites its core

        Args:
            cap: The latest day.
            blocked: Rows never added.
            seed: The retriever's rows, best first.
            seed_cos: Their cosines.
            recipe: CORE, POOL and HOPS, with EXPAND false to skip both expanders.

        Returns:
            The pool's rows in the order the shipped pipeline appends them.
        """
        rows, cos = seed[: recipe["POOL"]], seed_cos[: recipe["POOL"]]
        if not recipe.get("EXPAND", True):
            return rows
        targets = rows[np.argsort(-cos, kind="stable")[: recipe["CORE"]]]
        for forward in (False, True):
            found = self.neighbors(targets, forward, recipe["HOPS"])
            self.flag[rows] = True
            self.flag[blocked] = True
            found = found[~self.flag[found]][: recipe["HOPS"]]
            self.flag[rows] = False
            self.flag[blocked] = False
            found = found[self.embedded[found] & (self.day[found] <= cap)]
            rows = np.concatenate([rows, found])
        return rows

    def bm25_postings(self, text: str, rows: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Each query word's postings cut to a set of rows, so several pools of one query read the index once

        Args:
            text: The question.
            rows: Every row any pool of this query may hold.

        Returns:
            Per query word, the rows holding it and its count in each.
        """
        terms = [self.vocab[w] for w in get_query_words(text) if w in self.vocab]
        self.flag[rows] = True
        out = []
        for t in terms:
            a, b = self.post_ptr[t], self.post_ptr[t + 1]
            docs = self.post_doc[a:b]
            live = self.flag[docs]
            out.append((docs[live], self.post_tf[a:b][live].astype(np.float64)))
        self.flag[rows] = False
        return out

    def bm25(self, postings: list[tuple[np.ndarray, np.ndarray]], rows: np.ndarray) -> np.ndarray:
        """
        get_bm25 from cut postings, document frequency counted inside the pool
        """
        scores = np.zeros(len(rows))
        if not postings or not len(rows):
            return scores
        self.seat[rows] = np.arange(len(rows))
        length = self.doc_len[rows]
        norm = K1 * (1 - B + B * length / (float(length.mean()) or 1.0))
        for docs, tfs in postings:
            s = self.seat[docs]
            live = s >= 0
            if not live.any():
                continue
            s, tf = s[live], tfs[live]
            rarity = math.log(1 + len(rows) / len(s))
            scores[s] += rarity * tf * (K1 + 1) / (tf + norm[s])
        self.seat[rows] = -1
        return scores

    def adjacency(self, rows: np.ndarray) -> sp.csr_matrix:
        """
        Entry i, j is 1 when paper j cites paper i, inside the pool
        """
        size = len(rows)
        self.seat[rows] = np.arange(size)
        pos, citing = gather(self.refs_ptr, rows)
        cited = self.seat[self.refs_idx[pos]]
        self.seat[rows] = -1
        live = cited >= 0
        return sp.csr_matrix((np.ones(int(live.sum())), (cited[live], citing[live])), shape=(size, size))

    def hub(self, rows: np.ndarray, cutoff: int, blocked: np.ndarray) -> np.ndarray:
        """
        GraphExport.get_citation_counts_before, a blocked citer not counted
        """
        r = rows.astype(np.int64)
        out = (np.searchsorted(self.cites_key, (r << 16) | cutoff, side="right") - self.cites_ptr[r]).astype(np.float64)
        live = blocked[self.day[blocked] <= cutoff] if len(blocked) else blocked
        if len(live):
            self.seat[rows] = np.arange(len(rows))
            pos, _ = gather(self.refs_ptr, live)
            s = self.seat[self.refs_idx[pos]]
            self.seat[rows] = -1
            np.subtract.at(out, s[s >= 0], 1.0)
        return out

    def terms(self, rows: np.ndarray, cos: np.ndarray, cutoff: int, text: str, blocked: np.ndarray, core: int, depth: int, decay: float, postings: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        GraphScorer.get_individual_score over arrays

        Returns:
            One row per pooled paper, one column per entry of TERMS.
        """
        back = self.adjacency(rows)
        fwd = sp.csr_matrix(back.T)
        weights = np.zeros(len(rows))
        top = np.argsort(-cos)[:core]
        weights[top] = cos[top]
        weights /= weights.sum() or 1.0
        recency = 1.0 / (1.0 + np.clip((cutoff - self.day[rows]) / 365.25, 0.0, None))
        anc, fro = back @ weights, fwd @ weights
        out = {
            "cos": normalize_to_max(cos),
            "rec": normalize_to_max(recency),
            "anc": compress_long_tail(anc),
            "fro": compress_long_tail(fro),
            "anc2": compress_long_tail(back @ anc),
            "fro2": compress_long_tail(fwd @ fro),
            "anc_decay": distance_decay(back, weights, depth, decay),
            "fro_decay": distance_decay(fwd, weights, depth, decay),
            "cocite": compress_long_tail((back @ fro) * recency),
            "couple": compress_long_tail((fwd @ anc) * recency),
            "bm25": normalize_to_max(self.bm25(postings, rows)),
            "hub": compress_long_tail(self.hub(rows, cutoff, blocked)),
        }
        return np.stack([out[t] for t in TERMS], 1)


def distance_decay(step: sp.csr_matrix, weights: np.ndarray, depth: int, decay: float) -> np.ndarray:
    """
    GraphScorer.get_distance_decay
    """
    out = np.zeros(len(weights))
    seen = weights > 0
    out[seen] = 1.0
    current = seen.astype(np.float64)
    for hop in range(1, depth + 1):
        fresh = ((step @ current) > 0) & ~seen
        if not fresh.any():
            break
        out[fresh] = decay**hop
        seen |= fresh
        current = fresh.astype(np.float64)
    return out


def bf_order(terms: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """
    BoundedField's ranking of one pool, ties kept in pool order as the shipped scorer's sort keeps them

    Args:
        terms: The pool's terms, one column per entry of TERMS.
        weights: The recipe's exponents, keys in any case.

    Returns:
        Pool seats, best first.
    """
    e = {k.lower(): float(v) for k, v in weights.items()}
    signs = np.array([-1.0 if t == "hub" else 1.0 for t in TERMS])
    exponents = np.array([e.get(t, 0.0) for t in TERMS]) * signs
    return np.argsort(-(np.log1p(terms) @ exponents), kind="stable")


def pagerank(adjacency: sp.csr_matrix, teleport: np.ndarray, alpha: float) -> np.ndarray:
    """
    Power iteration with teleport, dangling mass returned through the teleport, to a change under 1e-10
    """
    out = np.asarray(adjacency.sum(axis=0)).ravel()
    walk = adjacency @ sp.diags(np.divide(1.0, out, out=np.zeros_like(out), where=out > 0))
    dangling = out == 0
    rank = teleport.copy()
    for _ in range(200):
        nxt = alpha * (walk @ rank + rank[dangling].sum() * teleport) + (1 - alpha) * teleport
        if np.abs(nxt - rank).sum() < 1e-10:
            return nxt
        rank = nxt
    return rank


def ppr_orders(store: Store, rows: np.ndarray, cos: np.ndarray, core: int, settings: list[tuple[float, str, str]]) -> list[np.ndarray]:
    """
    One pool ranked under each PPR setting, ties broken by cosine

    Args:
        store: The graph.
        rows: The pool.
        cos: Its cosines.
        core: How many of the best by cosine the core teleport spreads over.
        settings: Each (alpha, teleport, direction), teleport pool or core and direction back or both.

    Returns:
        Pool seats, best first, one array per setting.
    """
    cited_by = store.adjacency(rows)
    both = (cited_by + cited_by.T).tocsr()
    both.data[:] = 1.0
    adjacency = {"back": cited_by.tocsr(), "both": both}
    cosine = np.clip(cos, 0.0, None)
    core_mass = np.zeros_like(cosine)
    top = np.argsort(-cosine)[:core]
    core_mass[top] = cosine[top]
    teleport = {"pool": cosine / (cosine.sum() or 1.0), "core": core_mass / (core_mass.sum() or 1.0)}
    return [np.lexsort((-cosine, -pagerank(adjacency[d], teleport[t], a))) for a, t, d in settings]


class Stream:
    """
    One retriever's vectors under cache/<name>/corpus.npy, with the map between corpus order and graph rows
    """

    def __init__(self, store: Store, name: str) -> None:
        self.vectors = np.load(CACHE / name / "corpus.npy", mmap_mode="r")
        corpus_ids = np.load(CACHE / "corpus_ids.npy")
        rows = np.array([store.index.get(a.decode(), -1) for a in corpus_ids], np.int64)
        assert (rows >= 0).all(), "a corpus paper has no graph row"
        self.graph_of_corpus = rows
        self.corpus_of_graph = np.full(store.n, -1, np.int64)
        self.corpus_of_graph[rows] = np.arange(len(rows))
        self.day = torch.from_numpy(store.day[rows]).cuda()

    def chunks(self):
        for s in range(0, len(self.vectors), CHUNK):
            e = min(s + CHUNK, len(self.vectors))
            yield s, e, torch.from_numpy(np.ascontiguousarray(self.vectors[s:e])).cuda().float()

    def seeds(self, queries: np.ndarray, caps: np.ndarray, blocked: list[np.ndarray], k: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Each query's exact top `k` by cosine under its cap and without its blocked rows, as graph rows best first
        """
        q = torch.from_numpy(np.ascontiguousarray(queries, np.float32)).cuda()
        cap = torch.from_numpy(np.asarray(caps, np.int64)).cuda()
        bj = np.concatenate([np.full(len(b), j) for j, b in enumerate(blocked)] + [np.zeros(0)]).astype(np.int64)
        br = np.concatenate([self.corpus_of_graph[b] for b in blocked] + [np.zeros(0)]).astype(np.int64)
        keep = br >= 0
        bj, br = bj[keep], br[keep]
        best_v = torch.full((len(q), k), MASKED, device="cuda")
        best_i = torch.full((len(q), k), -1, dtype=torch.long, device="cuda")
        with torch.no_grad():
            for s, e, block in self.chunks():
                score = q @ block.T
                score.masked_fill_(self.day[s:e][None, :] > cap[:, None], MASKED)
                hit = (br >= s) & (br < e)
                if hit.any():
                    score[torch.from_numpy(bj[hit]).cuda(), torch.from_numpy(br[hit] - s).cuda()] = MASKED
                v, i = torch.cat([best_v, score], 1).topk(k, dim=1)
                best_i = torch.cat([best_i, torch.arange(s, e, device="cuda").expand(len(q), -1)], 1).gather(1, i)
                best_v = v
        out = []
        for r, v in zip(best_i.cpu().numpy(), best_v.cpu().numpy(), strict=True):
            live = v > MASKED
            out.append((self.graph_of_corpus[r[live]], v[live].astype(np.float64)))
        return out

    def cosines(self, queries: np.ndarray, pools: list[np.ndarray]) -> list[np.ndarray]:
        """
        The cosine of every pooled paper to its own query, one pass over the vectors for the whole batch
        """
        q = torch.from_numpy(np.ascontiguousarray(queries, np.float32)).cuda()
        case = np.concatenate([np.full(len(p), j) for j, p in enumerate(pools)]).astype(np.int64)
        row = np.concatenate([self.corpus_of_graph[p] for p in pools]).astype(np.int64)
        assert (row >= 0).all(), "a pooled paper has no vector"
        order = np.argsort(row, kind="stable")
        case_s, row_s = case[order], row[order]
        value = np.zeros(len(row))
        with torch.no_grad():
            for s, e, block in self.chunks():
                a, b = np.searchsorted(row_s, [s, e])
                for lo in range(a, b, PAIRS):
                    hi = min(lo + PAIRS, b)
                    c = torch.from_numpy(case_s[lo:hi]).cuda()
                    r = torch.from_numpy(row_s[lo:hi] - s).cuda()
                    value[order[lo:hi]] = (q[c] * block[r]).sum(1).double().cpu().numpy()
        out, at = [], 0
        for p in pools:
            out.append(value[at : at + len(p)])
            at += len(p)
        return out
