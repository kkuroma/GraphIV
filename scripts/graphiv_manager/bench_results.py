"""
What the Benchmarks tab reads: the exported versions, every run per split, and the fit's reports

A run's scores are kept until its file changes, since a SurveyMatch split is thousands of rows
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCHMARKS = ("SurveyMatch", "FrontierMatch", "ConsensusMatch", "PrecisionMatch")
# the cutoff each benchmark reports, PrecisionMatch's being a share rather than a recall
HEADLINE = {
    "SurveyMatch": "1000",
    "FrontierMatch": "1000",
    "ConsensusMatch": "100",
    "PrecisionMatch": "1000",
}
SPLITS = ("test", "val", "train")
# a run written before the split was part of the path sits straight under OUTPUT_PATH
LEGACY = "legacy"

SEEN: dict[Path, tuple[float, dict[str, Any]]] = {}


def summarize(path: Path, metric: str) -> dict[str, Any]:
    """
    One recipe's run as its case count, headline mean and every cutoff's mean

    Args:
        path: Its scores.jsonl.
        metric: The benchmark, which picks the headline.

    Returns:
        The summary, read from the cache while the file is unchanged.
    """
    stamp = path.stat().st_mtime
    held = SEEN.get(path)
    if held and held[0] == stamp:
        return held[1]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    cuts = list(rows[0]["recall"]) if rows else []
    means = {k: sum(r["recall"][k] for r in rows) / len(rows) for k in cuts} if rows else {}
    out = {
        "cases": len(rows),
        "headline": means.get(HEADLINE[metric]),
        "means": means,
        "written": stamp,
    }
    SEEN[path] = (stamp, out)
    return out


def runs(output: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """
    Every scored run under OUTPUT_PATH, by split, benchmark and recipe

    Args:
        output: [recipe_tuning] OUTPUT_PATH.

    Returns:
        Split to benchmark to recipe to its summary.
    """
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for split in (*SPLITS, LEGACY):
        root = output if split == LEGACY else output / split
        for metric in BENCHMARKS:
            for scores in sorted((root / metric).glob("*/scores.jsonl")):
                recipe = scores.parent.name
                found = out.setdefault(split, {}).setdefault(metric, {})
                found[recipe] = summarize(scores, metric)
    return out


def versions(export: Path, cutoff: str) -> list[dict[str, Any]]:
    """
    Every exported benchmark version, newest first, with its counts per split

    Args:
        export: [export] EXPORT_PATH.
        cutoff: [benchmarks] CUTOFF, the version the config names.

    Returns:
        One entry per graphiv-benchmarks-<cutoff> directory.
    """
    out = []
    for root in sorted(export.glob("graphiv-benchmarks-*"), reverse=True):
        if not root.is_dir() or root.name.endswith(".part"):
            continue
        manifest = root / "manifest.json"
        facts = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        size = sum(p.stat().st_size for p in root.rglob("*.parquet"))
        out.append(
            {
                "name": root.name,
                "cutoff": root.name.removeprefix("graphiv-benchmarks-"),
                "current": root.name == f"graphiv-benchmarks-{cutoff}",
                "counts": facts.get("counts", {}),
                "settings": facts.get("settings", {}),
                "bytes": size,
                "written": manifest.stat().st_mtime if manifest.is_file() else root.stat().st_mtime,
            }
        )
    return out


def fits(output: Path) -> list[dict[str, Any]]:
    """
    Every report the fit stage wrote, newest first

    Args:
        output: [recipe_tuning] OUTPUT_PATH.

    Returns:
        One report per version, each recipe's candidates and the one val picked.
    """
    out = []
    for path in sorted((output / "tuning").glob("*.json"), reverse=True):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["written"] = path.stat().st_mtime
        out.append(report)
    return out


def read(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Everything the Benchmarks tab draws, from the config, the export directory and the runs

    Args:
        cfg: The manager's config.

    Returns:
        The tab's data.
    """
    bench = cfg.get("benchmarks", {})
    tuning = cfg.get("recipe_tuning", {})
    output = Path(str(tuning.get("OUTPUT_PATH", "benchmarks/data/recipe_tuning")))
    cutoff = str(bench.get("CUTOFF", ""))
    return {
        "cutoff": cutoff,
        "settings": bench,
        "split": str(tuning.get("SPLIT", "val")),
        "repo": str(cfg.get("export", {}).get("BENCHMARK_REPO", "")),
        "versions": versions(Path(str(cfg.get("export", {}).get("EXPORT_PATH", ""))), cutoff),
        "runs": runs(output),
        "fits": fits(output),
        "headline": HEADLINE,
    }
