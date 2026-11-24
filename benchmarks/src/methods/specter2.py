"""
SPECTER2 over the whole corpus, loaded exactly as the allenai/specter2 model card does

Papers go through the proximity adapter and questions through the adhoc_query adapter, as the card advises for search
The paper vectors are embedded once into cache/specter2/, a shard at a time so a stopped run resumes
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from src import common, datasets
from src.schema import Case, Method, Ranking

# 64 with the GPU to itself, and 16 fits beside whatever model the router holds
BATCH = 64
SHARD = 200_000
TOP = 1000


class Encoder:
    """
    One adapter on the base model, called on a list of texts
    """

    def __init__(self, adapter: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        model.load_adapter(adapter, source="hf", load_as="specter2", set_active=True)
        self.model = model.half().cuda().eval()

    @torch.inference_mode()
    def __call__(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), BATCH):
            inputs = self.tokenizer(texts[i : i + BATCH], padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)
            output = self.model(**{k: v.cuda() for k, v in inputs.items()})
            out.append(output.last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(out).astype(np.float16)


def embed_corpus(make: Callable[[], Callable[[list[str]], np.ndarray]], sep: Callable[[Any], str], root: Path) -> None:
    """
    Embeds every paper as title, separator and abstract into root/corpus.npy, a shard at a time under root/shards/

    Args:
        make: Builds the paper encoder, called only when a shard is missing.
        sep: The separator token, read off the encoder.
        root: The retriever's cache directory.
    """
    if (root / "corpus.npy").exists():
        return
    out = root / "shards"
    table = common.read_corpus(["title", "abstract"])
    titles, abstracts = table["title"].to_pylist(), table["abstract"].to_pylist()
    todo = [s for s in range(0, len(titles), SHARD) if not (out / f"corpus_{s // SHARD:03d}.npy").exists()]
    if not todo:
        return
    out.mkdir(parents=True, exist_ok=True)
    encoder = make()
    for start in todo:
        end = min(start + SHARD, len(titles))
        texts = [titles[i] + sep(encoder) + abstracts[i] for i in range(start, end)]
        # sorting by length keeps padding small, and the order is restored before saving
        order = np.argsort([len(t) for t in texts])
        emb = encoder([texts[i] for i in order])
        restored = np.empty_like(emb)
        restored[order] = emb
        np.save(out / f"corpus_{start // SHARD:03d}.npy", restored)
        common.say(f"embedded {end} of {len(titles)}")
    del encoder
    torch.cuda.empty_cache()
    np.save(root / "corpus.npy", np.concatenate([np.load(p) for p in sorted(out.glob("corpus_*.npy"))]))


def search(cases: list[Case], queries: np.ndarray, root: Path, dataset: str) -> list[Ranking]:
    """
    The top papers by cosine for each case, capped at its cutoff with its anchors removed, inside the set's corpus

    Args:
        cases: The cases.
        queries: One query vector per case.
        root: The retriever's cache directory.
        dataset: The dataset, whose corpus a ranking is cut to before the top is taken.

    Returns:
        One ranking per case.
    """
    meta = common.read_corpus(["arxiv_id", "date"])
    ids = np.array(meta["arxiv_id"].to_pylist())
    dates = np.array(meta["date"].to_pylist(), dtype="datetime64[D]")
    emb = torch.from_numpy(np.load(root / "corpus.npy").astype(np.float32))
    emb = torch.nn.functional.normalize(emb, dim=1)
    index = {a: i for i, a in enumerate(ids)}
    unit = torch.nn.functional.normalize(torch.from_numpy(queries.astype(np.float32)), dim=1)
    corpus = datasets.load(dataset).corpus()
    outside = torch.zeros(len(ids), dtype=torch.bool)
    if corpus is not None:
        outside[:] = True
        outside[[index[a] for a in corpus.read_text().split() if a in index]] = False
    out = []
    for c, q in zip(cases, unit, strict=True):
        score = emb @ q
        score[outside] = -2.0
        score[torch.from_numpy(dates > np.datetime64(c["cutoff"]))] = -2.0
        score[[index[a] for a in c["anchors"] if a in index]] = -2.0
        out.append({"sample_id": c["sample_id"], "ranked": ids[torch.topk(score, TOP).indices.numpy()].tolist()})
    return out


class Specter2(Method):
    def retrieve(self, cases: list[Case], jobs: list[dict[str, Any]]) -> dict[str, list[Ranking]]:
        root = common.CACHE / "specter2"
        embed_corpus(lambda: Encoder("allenai/specter2"), lambda e: e.tokenizer.sep_token, root)
        queries = Encoder("allenai/specter2_adhoc_query")([c["query"] for c in cases])
        torch.cuda.empty_cache()
        return {j["name"]: search(cases, queries, root, j["dataset"]) for j in jobs}


METHOD = Specter2
