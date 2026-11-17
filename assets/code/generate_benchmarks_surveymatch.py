"""
Draws benchmarks-surveymatch.png, how a SurveyMatch case is built, on one real case
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_counts import load
from case_diagram import construction
from style import LEAD, ROOT, Page, apply_theme


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
    example = data["surveymatch"]["example"]
    s = data["settings"]
    page = Page(7.1)
    steps = [
        (
            "One review",
            f"review score {s['SURVEY_THRESHOLD']}+\n{s['SURVEYMATCH_FLOOR']}+ resolved references",
            "data",
        ),
        (
            "Gemma 4 writes the question",
            "from its title and abstract\nnaming no system or author",
            "model",
        ),
        ("The answer", "every reference it resolves\nto a paper in GraphIV", "step"),
    ]
    top = construction(
        page,
        [("SurveyMatch:", LEAD), ("given one review, find what it cites", False)],
        steps,
        example,
        set(example["citations"]),
        LEAD,
        ["the review"],
    )
    notes = [
        f"{example['titles'][0]} ({example['anchors'][0]}). A dot is one of its references at that"
        " paper's date, and date_max is the review's own date.",
    ]
    page.save(Path(str(args.out)), top, notes)


if __name__ == "__main__":
    main()
