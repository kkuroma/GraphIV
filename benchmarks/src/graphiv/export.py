"""
The corpus the baselines search, titles and abstracts in corpus.parquet and the stored nomic vectors in nomic.npy

Both are in arxiv_id order over every embedded paper, so a row index means one paper in either file
"""

import argparse
from pathlib import Path

import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.config import load_config

ROOT = Path(__file__).resolve().parents[3]
CORPUS = "SELECT arxiv_id, date, title, coalesce(abstract, '') FROM papers WHERE stage >= 3 ORDER BY arxiv_id"
VECTORS = "SELECT arxiv_id, embed_full::text FROM papers WHERE stage >= 3 ORDER BY arxiv_id"


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    dsn = str(load_config(str(ROOT / "config.toml"))["database"]["PG_DEST"])
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(CORPUS).fetchall()
    ids, dates, titles, abstracts = zip(*rows)
    pq.write_table(pa.table({"arxiv_id": list(ids), "date": list(dates), "title": list(titles), "abstract": list(abstracts)}), out / "corpus.parquet")
    print("corpus", len(ids), flush=True)
    vectors = np.lib.format.open_memmap(out / "nomic.npy", mode="w+", dtype=np.float16, shape=(len(ids), 768))
    with psycopg.connect(dsn) as conn, conn.cursor(name="vectors") as cur:
        cur.itersize = 50_000
        cur.execute(VECTORS)
        for i, (arxiv_id, text) in enumerate(cur):
            assert arxiv_id == ids[i], (i, arxiv_id, ids[i])
            vectors[i] = np.fromstring(text[1:-1], sep=",", dtype=np.float32)
    vectors.flush()
    print("vectors", len(ids), flush=True)


if __name__ == "__main__":
    main()
