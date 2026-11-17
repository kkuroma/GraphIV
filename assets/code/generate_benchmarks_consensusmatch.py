"""
Draws benchmarks-consensusmatch.png, how a ConsensusMatch case is built, on one real case
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_counts import load
from case_diagram import construction
from style import OCHRE, ROOT, Page, apply_theme


def main() -> None:
    """
    Loads the cached case, then draws the figure to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    data = load(Path(str(args.config_path)))
    apply_theme()
    example = data["consensusmatch"]["example"]
    s = data["settings"]
    page = Page(8.15)
    steps = [
        (
            "Three reviews, one topic",
            f"same leaf, cosine {s['CONSENSUS_COSINE']:.2f}+ each pair\n"
            f"within {s['GAP_DAYS']} days",
            "data",
        ),
        (
            "Gemma 4 writes the question",
            "from all three reviews\nnaming no system or author",
            "model",
        ),
        (
            "The answer",
            f"what all three cite, dated before\nthe first, {s['CONSENSUS_MIN_KEY']}+ papers",
            "step",
        ),
    ]
    top = construction(
        page,
        [("ConsensusMatch:", OCHRE), ("find what every review of a topic agrees on", False)],
        steps,
        example,
        set(example["citations"]),
        OCHRE,
        ["first review", "second review", "third review"],
    )
    notes = [
        "; ".join(f"{t} ({a})" for t, a in zip(example["titles"], example["anchors"], strict=True))
        + ". date_max is the first review's date.",
    ]
    page.save(Path(str(args.out)), top, notes)


if __name__ == "__main__":
    main()
