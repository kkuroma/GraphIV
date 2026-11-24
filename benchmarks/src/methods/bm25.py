"""
BM25 over titles and abstracts with bm25s, stemmed and with English stopwords as its README does

The corpus is tokenized in chunks, each document kept as an int32 array, since the one-call tokenize runs the box out of memory
"""

from typing import Any

import bm25s
import numpy as np
import Stemmer

from src import common, datasets
from src.schema import Case, Method, Ranking

CHUNK = 100_000
TOP = 1000


class BM25(Method):
    def retrieve(self, cases: list[Case], jobs: list[dict[str, Any]]) -> dict[str, list[Ranking]]:
        table = common.read_corpus(["arxiv_id", "title", "abstract", "date"])
        ids = np.array(table["arxiv_id"].to_pylist())
        dates = np.array(table["date"].to_pylist(), dtype="datetime64[D]")
        texts = [t + " " + a for t, a in zip(table["title"].to_pylist(), table["abstract"].to_pylist(), strict=True)]
        del table
        tokenizer = bm25s.tokenization.Tokenizer(stemmer=Stemmer.Stemmer("english"), stopwords="en")
        tokens = []
        for start in range(0, len(texts), CHUNK):
            chunk = tokenizer.tokenize(texts[start : start + CHUNK], update_vocab=True, return_as="ids", show_progress=False)
            tokens.extend(np.asarray(doc, dtype=np.int32) for doc in chunk)
        del texts
        retriever = bm25s.BM25()
        retriever.index((tokens, tokenizer.get_vocab_dict()), show_progress=False)
        index = {a: i for i, a in enumerate(ids)}
        # a set searched inside its own corpus is cut to it before the top is taken
        corpus = datasets.load(jobs[0]["dataset"]).corpus()
        inside = np.ones(len(ids), np.float32)
        if corpus is not None:
            inside[:] = 0.0
            inside[[index[a] for a in corpus.read_text().split() if a in index]] = 1.0
        out = []
        for c in cases:
            mask = (dates <= np.datetime64(c["cutoff"])).astype(np.float32) * inside
            mask[[index[a] for a in c["anchors"] if a in index]] = 0.0
            query = tokenizer.tokenize([c["query"]], update_vocab=False, return_as="ids", show_progress=False)[0]
            scores = retriever.get_scores_from_ids(query, weight_mask=mask)
            top = np.argpartition(-scores, TOP)[:TOP]
            # a masked paper scores 0 and can still fill the top when few papers match, so it is dropped
            top = top[mask[top] > 0]
            out.append({"sample_id": c["sample_id"], "ranked": ids[top[np.argsort(-scores[top])]].tolist()})
        return {j["name"]: out for j in jobs}


METHOD = BM25
