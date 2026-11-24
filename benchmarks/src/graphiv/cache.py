"""
Builds cache/graph/ and cache/keyword/ from the frozen graph export and the corpus, every array keyed by graph row

Each paper's citers are sorted by day and keyed (row << 16) | day, so a citation count at a cutoff is one searchsorted
BM25 reads an inverted index of every folded word, so a query never tokenizes an abstract
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from src.retrieve.keyword import WORD, fold_word

DATA = Path(__file__).resolve().parents[2] / "data"
CHUNK = 50000
SHIFT = 16
TEXT: dict[str, list[str]] = {}


def tokenize(span: tuple[int, int]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One span of documents as local term ids, counts and lengths

    Args:
        span: The first and one past the last document.

    Returns:
        The span's vocabulary, each posting's document, local term id and count, and each document's length.
    """
    vocab: dict[str, int] = {}
    docs, terms, tfs = [], [], []
    lengths = np.zeros(span[1] - span[0], np.int32)
    for i in range(*span):
        words = [fold_word(w) for w in WORD.findall(TEXT["doc"][i].lower())]
        lengths[i - span[0]] = len(words)
        seen: dict[int, int] = {}
        for w in words:
            t = vocab.setdefault(w, len(vocab))
            seen[t] = seen.get(t, 0) + 1
        docs += [i] * len(seen)
        terms += list(seen)
        tfs += list(seen.values())
    return list(vocab), np.array(docs, np.int32), np.array(terms, np.int32), np.array(tfs, np.int32), lengths


def build_keyword(out: Path, row_of_corpus: np.ndarray, n: int, corpus: pa.Table) -> None:
    title = pc.fill_null(corpus["title"], "").to_pylist()
    abstract = pc.fill_null(corpus["abstract"], "").to_pylist()
    TEXT["doc"] = [f"{t}\n{a}" for t, a in zip(title, abstract, strict=True)]
    spans = [(s, min(s + CHUNK, len(title))) for s in range(0, len(title), CHUNK)]
    vocab: dict[str, int] = {}
    docs, terms, tfs, lengths = [], [], [], np.zeros(len(title), np.int32)
    with mp.get_context("fork").Pool(8) as workers:
        for (s, e), (local, d, t, f, ln) in zip(spans, workers.imap(tokenize, spans), strict=True):
            remap = np.array([vocab.setdefault(w, len(vocab)) for w in local], np.int32)
            docs.append(row_of_corpus[d])
            terms.append(remap[t])
            tfs.append(np.minimum(f, 65535).astype(np.uint16))
            lengths[s:e] = ln
    doc, term, tf = np.concatenate(docs), np.concatenate(terms), np.concatenate(tfs)
    order = np.argsort(term, kind="stable")
    ptr = np.zeros(len(vocab) + 1, np.int64)
    np.cumsum(np.bincount(term, minlength=len(vocab)), out=ptr[1:])
    doc_len = np.zeros(n, np.int32)
    doc_len[row_of_corpus] = lengths
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "post_ptr.npy", ptr)
    np.save(out / "post_doc.npy", doc[order])
    np.save(out / "post_tf.npy", tf[order])
    np.save(out / "doc_len.npy", doc_len)
    words = pa.array(list(vocab), pa.large_string())
    with pa.OSFile(str(out / "vocab.arrow"), "wb") as sink, ipc.new_file(sink, pa.schema([("word", pa.large_string())])) as w:
        w.write_table(pa.table({"word": words}))
    print(f"keyword: {len(doc)} postings, {len(vocab)} words", flush=True)


def build_graph(out: Path, g: dict[str, np.ndarray]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("ids", "day", "stage", "refs_ptr", "refs_idx", "cites_ptr"):
        np.save(out / f"{name}.npy", g[name])
    ptr = g["cites_ptr"].astype(np.int64)
    day = g["day"].astype(np.int64)
    assert day.min() >= 0 and day.max() < (1 << SHIFT)
    row = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
    key = (row << SHIFT) | day[g["cites_idx"]]
    order = np.argsort(key, kind="stable")
    np.save(out / "cites_idx.npy", g["cites_idx"][order])
    np.save(out / "cites_key.npy", key[order])
    print(f"graph: {len(g['ids'])} papers, {len(key)} citations", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", default=str(DATA / "cache"))
    args = ap.parse_args()
    out = Path(args.out)
    g = dict(np.load(DATA / "sources" / "graph_export.npy"))
    n = len(g["ids"])
    build_graph(out / "graph", g)
    corpus = ipc.open_file(pa.memory_map(str(DATA / "sources" / "corpus.arrow"))).read_all()
    corpus_ids = np.array([a.encode() for a in corpus["arxiv_id"].to_pylist()])
    np.save(out / "corpus_ids.npy", corpus_ids)
    order = np.argsort(g["ids"])
    at = np.minimum(np.searchsorted(g["ids"][order], corpus_ids), n - 1)
    inside = g["ids"][order][at] == corpus_ids
    assert inside.all(), "a corpus paper has no graph row"
    row_of_corpus = order[at]
    embedded = np.zeros(n, bool)
    embedded[row_of_corpus] = True
    np.save(out / "graph" / "embedded.npy", embedded)
    build_keyword(out / "keyword", row_of_corpus.astype(np.int32), n, corpus)


if __name__ == "__main__":
    main()
