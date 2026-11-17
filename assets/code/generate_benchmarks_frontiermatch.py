"""
Draws benchmarks-frontiermatch.png, how a FrontierMatch case is built, on one real case
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_counts import load
from case_diagram import construction
from style import BLUE, ROOT, Page, apply_theme


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
    example = data["frontiermatch"]["example"]
    s = data["settings"]
    page = Page(7.95)
    steps = [
        (
            "Two reviews, one topic",
            f"same leaf, cosine {s['FRONTIER_COSINE']:.2f}+\nat most {s['GAP_DAYS']} days apart",
            "data",
        ),
        (
            "Gemma 4 writes the question",
            "from the earlier review\nnaming no system or author",
            "model",
        ),
        (
            "The answer",
            f"what the later one cites and\nthe earlier does not, {s['FRONTIER_MIN_KEY']}+ papers",
            "step",
        ),
    ]
    top = construction(
        page,
        [("FrontierMatch:", BLUE), ("given where a topic stood, find what came next", False)],
        steps,
        example,
        set(example["citations"]),
        BLUE,
        ["earlier review", "later review"],
        extra_note=True,
    )
    notes = [
        f"{example['titles'][0]} ({example['anchors'][0]}) and {example['titles'][1]}"
        f" ({example['anchors'][1]}). date_max is the later review's date.",
        f"An answer is published no earlier than {s['FRONTIER_WINDOW_DAYS']} days before the"
        " earlier review, so it is what the topic was doing between the two.",
    ]
    page.save(Path(str(args.out)), top, notes)


if __name__ == "__main__":
    main()
