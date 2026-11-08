"""
Exports the resolved citation graph as one npy archive of flat arrays

A hop then costs a slice instead of thousands of random reads against a 65GB table
This is an export rather than a queue stage, so it rewrites the whole file every run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import psycopg

from src.core.config import load_config

EPOCH = np.datetime64("1970-01-01")
DAY = np.timedelta64(1, "D")
CHUNK = 2_000_000


def main() -> None:
    """
    Reads papers and citations out of Postgres and writes the graph archive
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--config_path", default="config.toml")
    _ = ap.add_argument("--out", default="")
    args = ap.parse_args()

    # 1) Read the config
    cfg = load_config(str(args.config_path))
    dsn = str(cfg["database"]["PG_DEST"])
    out = Path(str(args.out) or str(cfg["dataset"]["GRAPH_PATH"]))

    with psycopg.connect(dsn) as conn:
        # 2) Read every paper and the columns a ranking reads at query time
        rows = conn.execute(
            "SELECT arxiv_id, date, stage, coalesce(survey_probability, -1)"
            " FROM papers ORDER BY arxiv_id"
        ).fetchall()
        print(f"[stage2] {len(rows):,} papers", flush=True)

        # 3) Map each arxiv id to its row, which every other array is keyed by
        ids = np.array([r[0] for r in rows], dtype="S")
        index = {r[0]: i for i, r in enumerate(rows)}
        day = np.array(
            [-(1 << 30) if r[1] is None else (np.datetime64(r[1]) - EPOCH) // DAY for r in rows],
            np.int32,
        )
        stage = np.array([r[2] for r in rows], np.int8)
        survey = np.array([r[3] for r in rows], np.float32)

        # 4) Stream every citation that resolved to a paper in the corpus
        srcs: list[np.ndarray] = []
        tgts: list[np.ndarray] = []
        seen = 0
        with conn.cursor(name="edges") as cur:
            cur.itersize = CHUNK
            cur.execute("SELECT source_id, target_id FROM citations WHERE target_id IS NOT NULL")
            while True:
                chunk = cur.fetchmany(CHUNK)
                if not chunk:
                    break
                srcs.append(np.array([index.get(s, -1) for s, _ in chunk], np.int32))
                tgts.append(np.array([index.get(t, -1) for _, t in chunk], np.int32))
                seen += len(chunk)
                print(f"[stage2] {seen:,} citations read", flush=True)
        src, tgt = np.concatenate(srcs), np.concatenate(tgts)
        del srcs, tgts

        # 5) Read the topic tree, which the graph carries as each paper's three levels
        level = np.full((len(rows), 3), -1, np.int8)
        for arxiv_id, l0, l1, l2 in conn.execute(
            "SELECT arxiv_id, level_0, level_1, level_2 FROM topic_tree"
            " WHERE stage = 1 AND level_2 IS NOT NULL"
        ).fetchall():
            i = index.get(arxiv_id)
            if i is not None:
                level[i] = (l0, l1, l2)

    # 6) Drop the occurrences whose endpoints are outside the corpus
    keep = (src >= 0) & (tgt >= 0)
    src, tgt = src[keep], tgt[keep]

    # 7) Collapse duplicates, since a paper citing the same target twice is one edge
    pair = np.unique(src.astype(np.int64) * len(rows) + tgt)
    src = (pair // len(rows)).astype(np.int32)
    tgt = (pair % len(rows)).astype(np.int32)
    del pair
    print(f"[stage2] {seen:,} occurrences, {len(src):,} distinct edges", flush=True)

    # 8) Build both directions as csr, so a hop either way is one slice
    arrays: dict[str, np.ndarray] = {
        "ids": ids,
        "day": day,
        "stage": stage,
        "survey": survey,
        "level": level,
    }
    for name, row, col in (("cites", tgt, src), ("refs", src, tgt)):
        order = np.argsort(row, kind="stable")
        ptr = np.zeros(len(rows) + 1, np.int32)
        ptr[1:] = np.cumsum(np.bincount(row, minlength=len(rows)))
        arrays[f"{name}_ptr"] = ptr
        arrays[f"{name}_idx"] = col[order]

    # 9) Write one archive, uncompressed so a reader pays no decode
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        np.savez(fh, **arrays)  # pyright: ignore[reportArgumentType]
    print(f"[stage2] {out} {out.stat().st_size / 1048576:.0f}MB", flush=True)
    for name, a in arrays.items():
        print(f"[stage2]   {name:10s} {str(a.dtype):8s} {a.shape} {a.nbytes / 1048576:7.1f}MB")


if __name__ == "__main__":
    main()
