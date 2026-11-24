"""
Graph-based Adaptive Re-ranking with the OpenScholar reranker over nomic-{set}-none, every case and variant stepped together

Each case runs pyterrier_adaptive's GAR loop unchanged, but every step's pairs go to the reranker in one large call
Variant rerank turns the graph off, sim offers 16 nomic neighbours and cite the citation graph, all sharing one score cache
"""

from collections import Counter
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import common
from src.schema import Case, Method, Ranking

BUDGET = 1000
# b in the GAR paper
BATCH = 16
NEIGHBOURS = 16
CHUNK = 131072
MASKED = -9.0
# pairs per reranker batch, each call sorted by length so a batch pads little
CALL = 128
MODEL = "OpenScholar/OpenScholar_Reranker"
SCORES = common.CACHE / "openscholar"


class Corpus:
    """
    The corpus as rows, each case's allowed rows, both graphs, and the reranker's text
    """

    def __init__(self, sim: bool) -> None:
        table = common.read_corpus(["arxiv_id", "title", "abstract", "date"])
        self.ids = table["arxiv_id"].to_pylist()
        self.row = {a: i for i, a in enumerate(self.ids)}
        self.days = np.array(table["date"].to_pylist(), dtype="datetime64[D]").astype(np.int64)
        self.text = [t + " " + a for t, a in zip(table["title"].to_pylist(), table["abstract"].to_pylist(), strict=True)]
        graph = common.load_graph()
        self.refs_ptr, self.refs_idx = graph.refs_ptr, graph.refs_idx
        self.cites_ptr, self.cites_idx = graph.cites_ptr, graph.cites_idx
        # export row to corpus row, -1 where the corpus lacks it
        self.to_corpus = np.array([self.row.get(a, -1) for a in graph.ids])
        self.to_export = np.array([graph.index[a] for a in self.ids])
        self.emb = None
        self.outside: torch.Tensor | None = None
        if sim:
            vectors = np.load(common.CACHE / "nomic" / "corpus.npy", mmap_mode="r")
            parts = []
            for s in range(0, len(vectors), CHUNK):
                v = torch.from_numpy(np.ascontiguousarray(vectors[s : s + CHUNK])).cuda().float()
                parts.append(torch.nn.functional.normalize(v, dim=1).half())
            self.emb = torch.cat(parts)
            self.day_t = torch.from_numpy(self.days).cuda()

    def cite_neighbours(self, row: int, allowed: np.ndarray) -> list[int]:
        e = self.to_export[row]
        rows = np.concatenate([self.refs_idx[self.refs_ptr[e] : self.refs_ptr[e + 1]], self.cites_idx[self.cites_ptr[e] : self.cites_ptr[e + 1]]])
        rows = self.to_corpus[rows]
        rows = rows[rows >= 0]
        return rows[allowed[rows]].tolist()

    def sim_neighbours(self, asks: list[tuple[int, int]], cases: list["Case"]) -> dict[tuple[int, int], list[int]]:
        """
        The nearest NEIGHBOURS allowed rows of each (case, row), the row itself excluded, in one pass per 1024 asks
        """
        assert self.emb is not None
        out: dict[tuple[int, int], list[int]] = {}
        for s in range(0, len(asks), 1024):
            part = asks[s : s + 1024]
            q = self.emb[torch.tensor([r for _, r in part], device="cuda")]
            cap = torch.tensor([cases[c]["cap"] for c, _ in part], device="cuda")
            bj = np.array([j for j, (c, r) in enumerate(part) for _ in [r, *cases[c]["blocked"]]], np.int64)
            br = np.array([b for c, r in part for b in [r, *cases[c]["blocked"]]], np.int64)
            best_v = torch.full((len(part), NEIGHBOURS), MASKED, device="cuda")
            best_i = torch.full((len(part), NEIGHBOURS), -1, dtype=torch.long, device="cuda")
            for lo in range(0, len(self.ids), CHUNK):
                hi = min(lo + CHUNK, len(self.ids))
                score = (q @ self.emb[lo:hi].T).float()
                score.masked_fill_(self.day_t[lo:hi][None, :] > cap[:, None], MASKED)
                hit = (br >= lo) & (br < hi)
                if hit.any():
                    score[torch.from_numpy(bj[hit]).cuda(), torch.from_numpy(br[hit] - lo).cuda()] = MASKED
                if self.outside is not None:
                    score.masked_fill_(self.outside[lo:hi][None, :], MASKED)
                v, i = torch.cat([best_v, score], 1).topk(NEIGHBOURS, dim=1)
                best_i = torch.cat([best_i, torch.arange(lo, hi, device="cuda").expand(len(part), -1)], 1).gather(1, i)
                best_v = v
            for (c, r), rows, vals in zip(part, best_i.cpu().numpy(), best_v.cpu().numpy(), strict=True):
                out[(c, r)] = rows[vals > MASKED].tolist()
        return out


class Adaptive:
    """
    pyterrier_adaptive.GAR.transform for one query, split at each scorer call so many queries step together
    """

    def __init__(self, case: int, initial: list[int], enabled: bool, ids: list[str]) -> None:
        self.case, self.ids = case, ids
        self.scores: dict[int, tuple[float, int]] = {}
        self.res_map = [Counter(dict(zip(initial, range(len(initial), 0, -1), strict=True)))]
        if enabled:
            self.res_map.append(Counter())
        self.enabled = enabled
        self.minscore = float("inf")
        self.iteration = 0
        self.batch: list[int] = []

    def next_batch(self) -> list[int]:
        """
        The rows the next iteration scores, empty when the query is done
        """
        while len(self.scores) < BUDGET and any(r for r in self.res_map):
            this = self.res_map[self.iteration % len(self.res_map)]
            if len(this) == 0:
                self.iteration += 1
                continue
            size = min(BATCH, BUDGET - len(self.scores))
            self.batch = [d for d, _ in this.most_common(size)]
            return self.batch
        self.batch = []
        return []

    def absorb(self, score: dict[tuple[int, int], float], neighbours: Any) -> None:
        scored = [(d, score[(self.case, d)]) for d in self.batch]
        self.scores.update({d: (s, self.iteration) for d, s in scored})
        for d, _ in scored:
            for c in self.res_map:
                del c[d]
        if len(self.scores) < BUDGET and self.enabled:
            frontier = self.res_map[1]
            remaining = BUDGET - len(self.scores)
            # ties fall to the larger docno string, as pyterrier sorts (score, docno) pairs
            for s, _, d in sorted(((s, self.ids[d], d) for d, s in scored), reverse=True):
                if len(frontier) < remaining or s >= self.minscore:
                    hit = False
                    for target in neighbours(self.case, d):
                        if target not in self.scores and (target not in frontier or s > frontier[target]):
                            frontier[target] = s
                            hit = True
                    if hit and s < self.minscore:
                        self.minscore = s
        self.iteration += 1

    def ranking(self) -> list[int]:
        # (score, iteration) pairs, so a score tie goes to the later iteration as in pyterrier
        out = [d for d, _ in Counter(self.scores).most_common()]
        if len(self.scores) < BUDGET:
            out += [d for d, _ in self.res_map[0].most_common()][: BUDGET - len(self.scores)]
        return out


class Scorer:
    """
    The OpenScholar reranker's logit, in fp16 with SDPA attention, behind a score cache kept on disk per test set
    """

    def __init__(self, set_name: str, corpus: Corpus, cases: list[Case]) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        self.corpus, self.cases = corpus, cases
        self.path = SCORES / f"{set_name}.npz"
        self.score: dict[tuple[int, int], float] = {}
        if self.path.exists():
            at = {c["sample_id"]: i for i, c in enumerate(cases)}
            with np.load(self.path, allow_pickle=False) as f:
                for sid, row, s in zip(f["sample_ids"][f["case"]], f["row"], f["score"], strict=True):
                    if str(sid) in at:
                        self.score[(at[str(sid)], int(row))] = float(s)

    @torch.inference_mode()
    def fill(self, pairs: list[tuple[int, int]]) -> None:
        todo = sorted({p for p in pairs if p not in self.score})
        for s in range(0, len(todo), CALL * 64):
            part = todo[s : s + CALL * 64]
            enc = self.tokenizer([self.cases[c]["query"] for c, _ in part], [self.corpus.text[r] for _, r in part], truncation="only_second", max_length=512)
            order = sorted(range(len(part)), key=lambda i: -len(enc["input_ids"][i]))
            for b in range(0, len(order), CALL):
                idx = order[b : b + CALL]
                batch = self.tokenizer.pad({k: [enc[k][i] for i in idx] for k in ("input_ids", "attention_mask")}, return_tensors="pt")
                logits = self.model(**{k: v.cuda() for k, v in batch.items()}).logits[:, 0].float().cpu().numpy()
                self.score.update(zip((part[i] for i in idx), (float(v) for v in logits), strict=True))

    def save(self) -> None:
        keys = list(self.score)
        sample_ids = np.array([c["sample_id"] for c in self.cases])
        SCORES.mkdir(parents=True, exist_ok=True)
        part = self.path.with_name(self.path.stem + ".part.npz")
        np.savez(part, sample_ids=sample_ids, case=np.array([c for c, _ in keys], np.int32), row=np.array([r for _, r in keys], np.int32), score=np.array([self.score[k] for k in keys], np.float32))
        part.replace(self.path)


class GAR(Method):
    def retrieve(self, cases: list[Case], jobs: list[dict[str, Any]]) -> dict[str, list[Ranking]]:
        first = jobs[0]
        set_name = f"{first['dataset']}_{first['split']}"
        corpus = Corpus(sim=any(j["method"] == "gar_sim" for j in jobs))
        stage = {r["sample_id"]: r["ranked"] for r in common.read_jsonl(common.TEST_RUNS / f"nomic-{set_name}-none" / "rankings.jsonl")}
        inside = None
        if (common.CASES / first["dataset"] / "corpus.txt").exists():
            inside = np.zeros(len(corpus.ids), bool)
            inside[[corpus.row[a] for a in (common.CASES / first["dataset"] / "corpus.txt").read_text().split() if a in corpus.row]] = True
        state = []
        for c in cases:
            blocked = [corpus.row[a] for a in c["anchors"] if a in corpus.row]
            cap = int((np.datetime64(c["cutoff"]) - np.datetime64("1970-01-01")).astype(int))
            allowed = corpus.days <= cap
            allowed[blocked] = False
            if inside is not None:
                allowed &= inside
            state.append({"query": c["query"], "sample_id": c["sample_id"], "cap": cap, "blocked": blocked, "allowed": allowed})
        if inside is not None and corpus.emb is not None:
            corpus.outside = torch.from_numpy(~inside).cuda()
        scorer = Scorer(set_name, corpus, state)
        cite: dict[tuple[int, int], list[int]] = {}
        sim: dict[tuple[int, int], list[int]] = {}
        out: dict[str, list[Ranking]] = {}
        for job in jobs:
            loops = [Adaptive(i, [corpus.row[a] for a in stage[c["sample_id"]][:BUDGET]], job["method"] != "rerank", corpus.ids) for i, c in enumerate(cases)]
            # the initial papers in large calls first, which every variant then reads from the cache
            scorer.fill([(lp.case, d) for lp in loops for d in lp.res_map[0]])

            def neighbours(case: int, row: int, method: str = job["method"]) -> list[int]:
                if method == "gar_sim":
                    return sim[(case, row)]
                if (case, row) not in cite:
                    cite[(case, row)] = corpus.cite_neighbours(row, state[case]["allowed"])
                return cite[(case, row)]

            active, steps = loops, 0
            while active:
                batches = [(lp, lp.next_batch()) for lp in active]
                active = [lp for lp, b in batches if b]
                scorer.fill([(lp.case, d) for lp in active for d in lp.batch])
                if job["method"] == "gar_sim":
                    asks = sorted({(lp.case, d) for lp in active for d in lp.batch if (lp.case, d) not in sim})
                    sim.update(corpus.sim_neighbours(asks, state))
                for lp in active:
                    lp.absorb(scorer.score, neighbours)
                steps += 1
                if steps % 10 == 0:
                    common.say(f"gar {set_name} {job['method']}: step {steps}, {len(active)} active, {len(scorer.score)} scored pairs")
            scorer.save()
            out[job["name"]] = [{"sample_id": c["sample_id"], "ranked": [corpus.ids[r] for r in lp.ranking()]} for c, lp in zip(cases, loops, strict=True)]
        return out


METHOD = GAR
