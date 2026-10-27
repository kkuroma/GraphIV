"""
Export SurveyMatch, FrontierMatch and ConsensusMatch as Parquet, mined up to [benchmarks] CUTOFF

A version is its cutoff, since a new paper never adds references to a review already published
Each case's question is written once here and frozen, so every later run asks the same thing
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from benchmark_files import (
    FOLDERS,
    HEADLINE_K,
    SPLIT_SEED,
    SPLITS,
    assign_splits,
    get_split_group,
    write_split,
)
from case_miner import CaseMiner, Sample
from query_writer import GemmaWriter

from src.core.config import load_config

RATIOS = {
    "SurveyMatch": (0.6, 0.2, 0.2),
    "FrontierMatch": (0.4, 0.2, 0.4),
    "ConsensusMatch": (0.4, 0.1, 0.5),
}
META = """
SELECT arxiv_id, date, survey_probability, title FROM papers WHERE arxiv_id = ANY(%s)
"""
NAMES = """
SELECT level_0, level_1, level_2, name FROM topic_names
"""


def mine(cfg: dict[str, Any], metric: str) -> list[Sample]:
    """
    Every case of one benchmark that exists at the cutoff, uncapped

    Args:
        cfg: The config.
        metric: SurveyMatch, FrontierMatch or ConsensusMatch.

    Returns:
        The samples.
    """
    b = cfg["benchmarks"]
    miner = CaseMiner(
        str(cfg["database"]["PG_DEST"]),
        seed=SPLIT_SEED,
        max_samples=10**9,
        survey_threshold=float(b["SURVEY_THRESHOLD"]),
        min_references=int(b[f"{metric.upper()}_FLOOR"]),
        max_date=date.fromisoformat(str(b["CUTOFF"])),
    )
    floors: dict[str, dict[str, float]] = {
        "SurveyMatch": {},
        "FrontierMatch": {
            "gap_days": float(b["GAP_DAYS"]),
            "cosine": float(b["FRONTIER_COSINE"]),
            "window_days": float(b["FRONTIER_WINDOW_DAYS"]),
            "min_key": float(b["FRONTIER_MIN_KEY"]),
        },
        "ConsensusMatch": {
            "gap_days": float(b["GAP_DAYS"]),
            "cosine": float(b["CONSENSUS_COSINE"]),
            "min_key": float(b["CONSENSUS_MIN_KEY"]),
        },
    }
    return miner.mine(metric, **floors[metric])


def build_rows(
    cfg: dict[str, Any], metric: str, samples: list[Sample], writer: GemmaWriter
) -> list[dict[str, object]]:
    """
    The published rows of one benchmark, each with its question written and its split assigned

    Args:
        cfg: The config.
        metric: SurveyMatch, FrontierMatch or ConsensusMatch.
        samples: The mined samples.
        writer: Writes each case's question, cached by its input.

    Returns:
        One row per sample, keyed by the schema's columns.
    """
    drafter = str(cfg["endpoints"]["LLM_NAME"])
    ids = sorted({a for s in samples for a in s.anchors})
    with psycopg.connect(str(cfg["database"]["PG_DEST"])) as conn:
        rows_meta = conn.execute(META, (ids,)).fetchall()
        meta = {str(r[0]): (r[1], float(r[2] or 0.0), str(r[3])) for r in rows_meta}
        names = {(r[0], r[1], r[2]): str(r[3]) for r in conn.execute(NAMES)}
    splits = assign_splits(samples, RATIOS[metric])
    rows: list[dict[str, object]] = []
    for i, (s, split) in enumerate(zip(samples, splits, strict=True), 1):
        written = writer.write(s.titles, s.abstracts)
        l0, l1, l2 = s.leaf
        rows.append(
            {
                "case_id": s.sample_id,
                "split": split,
                "query_text": written or "; ".join(s.titles),
                "query_drafter": drafter if written else "titles",
                "date_max": date.fromisoformat(s.cutoff),
                "anchors": s.anchors,
                "anchor_titles": [meta[a][2] for a in s.anchors],
                "anchor_dates": [meta[a][0] for a in s.anchors],
                "anchor_survey_probability": [meta[a][1] for a in s.anchors],
                "citations": s.key,
                "n_citations": len(s.key),
                "headline_k": HEADLINE_K[metric],
                "topic_l0": l0,
                "topic_l1": l1,
                "topic_l2": l2,
                "topic_l0_name": names.get((l0, None, None), ""),
                "topic_l1_name": names.get((l0, l1, None), ""),
                "topic_l2_name": names.get((l0, l1, l2), ""),
                "split_group": get_split_group(s.leaf),
                "gap_days": int(s.extra["gap_days"]) if "gap_days" in s.extra else None,
                "cosine": s.extra.get("cosine"),
            }
        )
        if i % 500 == 0:
            print(f"[benchmarks] {metric} {i}/{len(samples)} questions written", flush=True)
    return rows


def main() -> None:
    """
    Mine, write the questions, split and save every benchmark under one versioned directory
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--config_path", default="config.toml")
    _ = ap.add_argument("--dry-run", action="store_true", help="count the cases, write nothing")
    args = ap.parse_args()
    cfg = load_config(str(args.config_path))
    cutoff = str(cfg["benchmarks"]["CUTOFF"])
    out = Path(str(cfg["export"]["EXPORT_PATH"])) / f"graphiv-benchmarks-{cutoff}"
    part = out.with_name(out.name + ".part")
    writer = GemmaWriter(
        str(cfg["endpoints"]["LLM_API"]),
        str(cfg["endpoints"]["LLM_NAME"]),
        Path(str(cfg["export"]["QUERY_CACHE"])),
    )
    counts: dict[str, dict[str, int]] = {}
    for metric in ("SurveyMatch", "FrontierMatch", "ConsensusMatch"):
        samples = mine(cfg, metric)
        print(f"[benchmarks] {metric}: {len(samples)} cases at the cutoff {cutoff}", flush=True)
        if args.dry_run:
            splits = assign_splits(samples, RATIOS[metric])
            print(
                f"[benchmarks] {metric}: " + ", ".join(f"{s} {splits.count(s)}" for s in SPLITS),
                flush=True,
            )
            continue
        rows = build_rows(cfg, metric, samples, writer)
        counts[metric] = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
        for split in SPLITS:
            write_split(
                [r for r in rows if r["split"] == split],
                part / FOLDERS[metric] / f"{split}.parquet",
            )
        print(f"[benchmarks] {metric}: {counts[metric]}", flush=True)
    if args.dry_run:
        return
    manifest = {
        "cutoff": cutoff,
        "split_seed": SPLIT_SEED,
        "ratios": RATIOS,
        "counts": counts,
        "settings": cfg["benchmarks"],
    }
    _ = (part / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    if out.exists():
        shutil.rmtree(out)
    part.rename(out)
    print(f"[benchmarks] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
