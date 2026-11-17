"""
Draws survey-labeler.png: the reviews across arXiv, agreement with the teacher, and other filters
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import release_counts
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
from style import (
    LEAD,
    LEFT,
    MUTED,
    PURPLE,
    RIGHT,
    ROOT,
    Page,
    apply_theme,
)
from survey_scores import GRID, SPEED, THRESHOLD, load

# the filters compared, as (name, the key their scores sit under)
FILTERS = (
    ("survey labeler", "model"),
    ("Semantic Scholar", "Semantic Scholar"),
    ("title filter", "title"),
    ("OpenAlex", "OpenAlex"),
)


def fields(page: Page, x: float, top: float, width: float, rows: list[Any], biggest: int) -> None:
    """
    Fields as rows, each its name, a bar of its reviews, and their share of the field's papers

    Args:
        page: The page.
        x: Left edge.
        top: Top edge.
        width: Width.
        rows: (field, reviews, papers) per field.
        biggest: The most reviews any field has, which every bar is scaled to.
    """
    gutter = max(page.measure(name, 11) for name, _, _ in rows) + 0.2
    span = width - gutter - 1.35
    for index, (name, reviews, papers) in enumerate(rows):
        y = top + index * 0.42
        page.ax.text(x, y + 0.2, name, fontsize=11, va="center")
        length = span * reviews / biggest
        page.ax.add_patch(
            FancyBboxPatch(
                (x + gutter, y + 0.11),
                length,
                0.18,
                boxstyle="round,pad=0,rounding_size=0.03",
                facecolor=PURPLE,
                edgecolor="none",
                zorder=3,
            )
        )
        page.ax.text(
            x + gutter + length + 0.08,
            y + 0.2,
            f"{reviews / 1e3:.1f}k  ·  {reviews / papers:.1%}",
            fontsize=10.5,
            va="center",
        )


def agreement(axis: Axes, data: dict[str, Any]) -> None:
    """
    Precision against recall on the held-out papers weighted to the corpus, with its band

    Args:
        axis: The panel.
        data: The cached scores.
    """
    mid, sd = np.array(data["curve"]), np.array(data["sd"])
    model = data["model"]
    axis.fill_between(GRID, mid - sd, np.minimum(mid + sd, 1.0), color=LEAD, alpha=0.18, lw=0)
    axis.plot(GRID, mid, color=LEAD, linewidth=2.2, zorder=3)
    axis.plot(model["recall"], model["precision"], "o", color=LEAD, markersize=9, zorder=4)
    axis.annotate(
        f"threshold {THRESHOLD}",
        (model["recall"], model["precision"]),
        xytext=(-10, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=11,
        color=LEAD,
        fontweight="semibold",
    )
    axis.set_xlim(0, 1.0)
    axis.set_ylim(0, 1.03)
    axis.set_xticks(np.arange(0, 1.01, 0.2))
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.tick_params(axis="y", labelcolor=MUTED, labelsize=11)
    axis.grid(axis="both")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")


def filters(axis: Axes, data: dict[str, Any]) -> None:
    """
    Recall and precision of the labeler beside the title filter and two scholarly indexes

    Args:
        axis: The panel.
        data: The cached scores.
    """
    rows = np.arange(len(FILTERS))
    scores = [data[key] if key in data else data["sources"][key] for _, key in FILTERS]
    for metric, color, marker, offset in (
        ("recall", LEAD, "o", -0.16),
        ("precision", PURPLE, "s", 0.16),
    ):
        for y, entry in zip(rows, scores, strict=True):
            value = float(entry[metric])
            axis.plot(value, y + offset, marker, color=color, markersize=9, zorder=3)
            right = value < 0.8
            axis.annotate(
                f"{value:.3f}",
                (value, float(y + offset)),
                xytext=(9 if right else -9, 0),
                textcoords="offset points",
                ha="left" if right else "right",
                va="center",
                fontsize=11,
                color=color,
                fontweight="medium",
            )
    axis.set_yticks(rows, [name for name, _ in FILTERS])
    axis.set_ylim(len(FILTERS) - 0.45, -0.55)
    axis.set_xlim(-0.04, 1.06)
    axis.set_xticks(np.arange(0, 1.01, 0.2))
    axis.set_xlabel("Share, weighted to the corpus")


def draw(data: dict[str, Any], release: dict[str, Any], out: Path) -> None:
    """
    The reviews it found across arXiv, its agreement with the teacher, and the other filters

    Args:
        data: The cached scores.
        release: The dataset figures' cached counts.
        out: Where the figure is written.
    """
    page = Page(13.65)
    rows = sorted(release["review_fields"], key=lambda r: -int(r[1]))
    total = int(release["papers"][0][0])
    reviews = sum(int(r[1]) for r in rows)
    biggest = int(rows[0][1])

    page.title(
        [
            [("GraphIV's Survey Labeler:", True)],
            [("DistilBERT for Literature Review Detection", False)],
        ]
    )

    page.panel_title(LEFT, 2.35, "Literature reviews across arXiv", 1)
    page.key(RIGHT, 2.35, [(f"{total / 1e6:.2f}M papers at {SPEED} a second", MUTED)])
    half = len(rows) // 2
    fields(page, LEFT, 2.6, 5.5, rows[:half], biggest)
    fields(page, 6.65, 2.6, RIGHT - 6.65, rows[half:], biggest)

    top = 2.6 + half * 0.42 + 0.9
    page.panel_title(LEFT, top, "Agreement with the teacher", 2)
    page.key(6.0, top, [(f"average precision {data['areas']['weighted']:.3f}", LEAD)])
    agreement(page.place(1.35, top + 0.35, 4.65, 4.0), data)
    page.panel_title(6.65, top, "Comparison with existing filters", 3)
    page.key(RIGHT, top, [("recall", LEAD), ("·", MUTED), ("precision", PURPLE)])
    filters(page.place(8.3, top + 0.35, RIGHT - 8.3, 4.0), data)

    sources = data["sources"]
    notes = [
        f"The {reviews:,} papers in GraphIV's release scoring {THRESHOLD} or more, by the field of"
        " the topic tree they sit in. Each share is of that field's papers.",
        f"Precision against recall on the {data['held']:,} held-out papers, against the teacher's"
        " labels, each weighted by how many real papers its pile stands for. The band is one sd"
        f" over 1,000 resamples, and the dot is the {THRESHOLD} threshold.",
        "On the same weighted papers, against the teacher's labels. Semantic Scholar counts its"
        " Review publication type and OpenAlex its review work type, both fetched on"
        f" {sources['fetched']}, and a paper either leaves untyped counts as not flagged.",
    ]
    page.save(out, top + 5.45, notes)


def main() -> None:
    """
    Scores and counts whatever is not cached, then draws the figure to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    config = Path(str(args.config_path))
    data, release = load(config), release_counts.load(config)
    apply_theme()
    draw(data, release, Path(str(args.out)))


if __name__ == "__main__":
    main()
