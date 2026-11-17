"""
Draws benchmarks.png, the benchmark card's title and one panel per benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmark_counts import load
from matplotlib.patches import FancyBboxPatch
from style import (
    BLUE,
    INK,
    LEAD,
    LEFT,
    MUTED,
    OCHRE,
    OUTLINE,
    PANEL,
    PURPLE,
    RIGHT,
    ROOT,
    Page,
    apply_theme,
    tint,
    write_runs,
)

# the cutoffs the GraphIV paper reports each benchmark at
CUTOFFS = {
    "surveymatch": "R@100, R@200, R@500",
    "frontiermatch": "R@20, R@50, R@100",
    "consensusmatch": "R@20, R@50, R@100",
}
# each benchmark as (folder, color, its reviews, what a case asks)
PANELS = [
    ("surveymatch", LEAD, 1, "Given one review,\nfind what it cites"),
    ("frontiermatch", BLUE, 2, "Given where a topic stood,\nfind what came next"),
    ("consensusmatch", OCHRE, 3, "Find what every review\nof a topic agrees on"),
]
SPLIT_SHARES = ("train", "val", "test")
TITLES = {
    "surveymatch": "SurveyMatch",
    "frontiermatch": "FrontierMatch",
    "consensusmatch": "ConsensusMatch",
}


def panel(
    page: Page,
    x: float,
    top: float,
    width: float,
    high: float,
    spec: tuple[str, str, int, str],
    info: dict[str, Any],
) -> None:
    """
    One benchmark as a tall card: its name, its reviews, what it asks, then its numbers

    Args:
        page: The page.
        x: Left edge.
        top: Top edge.
        width: Width.
        high: Height.
        spec: (folder, color, reviews, what it asks).
        info: The benchmark's summary.
    """
    ax = page.ax
    folder, color, reviews, asks = spec
    ax.add_patch(
        FancyBboxPatch(
            (x, top),
            width,
            high,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=PANEL,
            edgecolor=tint(color, 0.6),
            linewidth=1.3,
        )
    )
    inner = x + 0.25
    ax.text(
        inner,
        top + 0.5,
        TITLES[folder],
        fontsize=16,
        fontweight="bold",
        color=color,
        family="monospace",
        va="baseline",
    )
    # one page glyph per review a case is built from
    for index in range(reviews):
        gx = inner + index * 0.32
        ax.add_patch(
            FancyBboxPatch(
                (gx, top + 0.78),
                0.24,
                0.32,
                boxstyle="round,pad=0,rounding_size=0.03",
                facecolor=tint(color, 0.18),
                edgecolor=color,
                linewidth=1.1,
            )
        )
        for line in range(3):
            ax.plot([gx + 0.05, gx + 0.19], [top + 0.88 + line * 0.07] * 2, color=color, lw=0.8)
    label = ax.text(
        inner + reviews * 0.32 + 0.05,
        top + 1.0,
        f"{reviews} review{'s' * (reviews > 1)} a case",
        fontsize=11,
        color=MUTED,
        va="baseline",
    )
    if reviews == 1:
        page.marked.append((label, 1))
    for index, line in enumerate(asks.split("\n")):
        ax.text(
            inner,
            top + 1.55 + index * 0.27,
            line,
            fontsize=13.5,
            fontweight="semibold",
            va="baseline",
        )
    total = sum(info["splits"].values())
    ax.text(inner, top + 2.65, f"{total:,}", fontsize=30, fontweight="bold", va="baseline")
    ax.text(inner, top + 2.92, "cases", fontsize=11.5, color=MUTED, va="baseline")
    # the split as one bar, each part labeled with its count
    bar_top, bar_w = top + 3.2, width - 0.5
    left = inner
    for index, split in enumerate(SPLIT_SHARES):
        n = info["splits"][split]
        part = bar_w * n / total
        ax.add_patch(
            FancyBboxPatch(
                (left, bar_top),
                part - 0.03,
                0.22,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=tint(color, 0.85 - 0.3 * index),
                edgecolor="none",
            )
        )
        name = ax.text(left, bar_top + 0.45, split, fontsize=10, color=MUTED, va="baseline")
        if reviews == 1 and index == 0:
            page.marked.append((name, 2))
        ax.text(left, bar_top + 0.66, f"{n:,}", fontsize=11, fontweight="semibold", va="baseline")
        left += part
    rows = [
        ("answer size", f"{info['answer_median']:g} papers, median"),
        ("scored by", CUTOFFS[folder]),
        ("years", f"{info['years'][0]} to {info['years'][1]}"),
        ("topics", f"{info['topics']} subfields, {info['leaves']:,} leaves"),
    ]
    ax.plot([inner, x + width - 0.25], [top + 4.15, top + 4.15], color=OUTLINE, lw=0.8)
    for index, (name, value) in enumerate(rows):
        y = top + 4.5 + index * 0.32
        ax.text(inner, y, name, fontsize=10.5, color=MUTED, va="baseline")
        ax.text(inner + 1.15, y, value, fontsize=11, color=INK, va="baseline")


def main() -> None:
    """
    Loads the cached summaries, then draws the figure to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    data = load(Path(str(args.config_path)))
    apply_theme()
    page = Page(8.75)
    page.title([[("GraphIV Benchmarks:", True), ("one question, many papers", False)]])
    cutoff = data["settings"]["CUTOFF"]
    subtitle: list[tuple[str, bool | str]] = [
        ("Three ways to score a retriever against what", False),
        ("arXiv's reviews", PURPLE),
        ("cite, version", False),
        (str(cutoff), LEAD),
    ]
    write_runs(page.fig, *page.at(LEFT, 1.6), subtitle, 15, "medium", INK)
    gap = 0.35
    width = (RIGHT - LEFT - 2 * gap) / 3
    for index, spec in enumerate(PANELS):
        panel(page, LEFT + index * (width + gap), 2.0, width, 5.75, spec, data[spec[0]])
    notes = [
        "A review is a paper GraphIV's survey labeler scores 0.95 or more.",
        "A seeded hash of the case's anchors under 69420, so a later version never moves a case.",
    ]
    page.save(Path(str(args.out)), 8.25, notes)


if __name__ == "__main__":
    main()
