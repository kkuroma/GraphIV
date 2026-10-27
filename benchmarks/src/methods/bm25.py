"""
BM25 over titles and abstracts with bm25s, stemmed and with English stopwords as its README does

The corpus is tokenized in chunks, each document kept as an int32 array, since the one-call tokenize runs the box out of memory
"""

from typing import Any

import bm25s
import numpy as np
import pyarrow.parquet as pq
import Stemmer

from src import common
from src.schema import Case, Method, Ranking

CHUNK = 100_000
TOP = 1000


class BM25(Method):
    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        table = pq.read_table(common.corpus() / "corpus.parquet")
        ids = np.array(table["arxiv_id"].to_pylist())
        dates = np.array(table["date"].to_pylist(), dtype="datetime64[D]")
        texts = [t + " " + a for t, a in zip(table["title"].to_pylist(), table["abstract"].to_pylist())]
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
        out = []
        for c in cases:
            mask = (dates <= np.datetime64(c["cutoff"])).astype(np.float32)
            mask[[index[a] for a in c["anchors"] if a in index]] = 0.0
            query = tokenizer.tokenize([c["query"]], update_vocab=False, return_as="ids", show_progress=False)[0]
            scores = retriever.get_scores_from_ids(query, weight_mask=mask)
            top = np.argpartition(-scores, TOP)[:TOP]
            out.append({"sample_id": c["sample_id"], "ranked": ids[top[np.argsort(-scores[top])]].tolist()})
        return out


METHOD = BM25
