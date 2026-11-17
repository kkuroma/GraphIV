"""
Draws citation-labeler.png: the intents across arXiv, agreement with the teacher, and performance
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import release_counts
from labeler_scores import SHOWN, SPEED, load, scores, teacher_truth
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from style import (
    INK,
    LEAD,
    LEFT,
    MUTED,
    PANEL,
    PURPLE,
    RIGHT,
    ROOT,
    Page,
    apply_theme,
    intents,
)


def agreement(axis: Axes, matrix: np.ndarray) -> None:
    """
    The student against the teacher as a confusion matrix, each row summing to one

    Args:
        axis: The panel.
        matrix: Counts, the teacher's intent by row.
    """
    rows = np.arange(len(SHOWN))
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ramp = LinearSegmentedColormap.from_list("lead", [PANEL, LEAD])
    axis.pcolormesh(
        np.arange(len(SHOWN) + 1) - 0.5,
        np.arange(len(SHOWN) + 1) - 0.5,
        shares,
        cmap=ramp,
        vmin=0,
        vmax=1,
        edgecolors=PANEL,
        linewidth=2,
    )
    for i in range(len(SHOWN)):
        for j in range(len(SHOWN)):
            if shares[i, j] >= 0.005:
                axis.text(
                    j,
                    i,
                    f"{shares[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if shares[i, j] > 0.5 else INK,
                )
    axis.set_xticks(rows, list(SHOWN), rotation=35, ha="right", rotation_mode="anchor")
    axis.set_yticks(rows, list(SHOWN))
    for y, n in zip(rows, matrix.sum(axis=1), strict=True):
        axis.annotate(
            f"n = {n}",
            (0, float(y)),
            xycoords=axis.get_yaxis_transform(),
            xytext=(-10, -15),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            color=MUTED,
        )
    axis.set_xlim(-0.5, len(SHOWN) - 0.5)
    axis.set_ylim(len(SHOWN) - 0.5, -0.5)
    axis.grid(False)
    axis.tick_params(axis="both", length=0)
    axis.tick_params(axis="x", labelsize=11.5)
    axis.set_xlabel("Student's intent")


def performance(axis: Axes, student: np.ndarray, teacher: np.ndarray) -> None:
    """
    F1 per intent against human labels, the student beside its teacher, on the matrix's rows

    Args:
        axis: The panel.
        student: The student's counts on the ground truth, the true intent by row.
        teacher: The teacher's counts on the ground truth, in the same order.
    """
    rows = np.arange(len(SHOWN))
    handles = []
    for name, matrix, color, marker, offset in (
        ("teacher", teacher, PURPLE, "s", -0.16),
        ("student", student, LEAD, "o", 0.16),
    ):
        f1 = scores(matrix)[2]
        axis.plot(f1, rows + offset, marker, color=color, markersize=9, zorder=3)
        for y, value in zip(rows, f1, strict=True):
            axis.annotate(
                f"{value:.3f}",
                (float(value), float(y + offset)),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=11,
                color=color,
                fontweight="medium",
            )
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                color=color,
                linestyle="none",
                markersize=9,
                label=f"{name}, macro F1 {f1.mean():.3f}",
            )
        )
    axis.legend(handles=handles, loc="upper left", fontsize=11, borderaxespad=0.5)
    axis.set_xlim(0.88, 1.035)
    axis.set_xticks([0.9, 0.95, 1.0], ["0.90", "0.95", "1.00"])
    axis.set_yticks(rows, [""] * len(SHOWN))
    axis.set_ylim(len(SHOWN) - 0.5, -0.5)
    axis.grid(axis="y", visible=False)
    axis.set_xlabel("F1 against human labels")


def draw(data: dict[str, Any], release: dict[str, Any], out: Path) -> None:
    """
    citation-labeler.png: what it labeled across arXiv, its agreement and its performance

    Args:
        data: The cached scores.
        release: The dataset figures' cached counts.
        out: Where the figure is written.
    """
    page = Page(13.05)
    counts: dict[str, int] = {}
    for _, intent, n in release["cites"]:
        counts[intent] = counts.get(intent, 0) + int(n)
    total = sum(counts.values())
    matrix = np.array(data["confusion"])
    macro = float(scores(matrix)[2].mean())

    page.title(
        [
            [("GraphIV's Citation Labeler:", True)],
            [("DistilBERT for Citation Intent Labeling", False)],
        ]
    )

    page.panel_title(LEFT, 2.35, "Citation intents across arXiv", 1)
    page.key(RIGHT, 2.35, [(f"{total / 1e6:.1f}M citations at {SPEED:,} a second", MUTED)])
    order = sorted(counts, key=lambda k: -counts[k])
    intents(page.ax, LEFT, 2.65, 5.5, counts, order[:3])
    intents(page.ax, 6.65, 2.65, RIGHT - 6.65, counts, order[3:])

    page.panel_title(LEFT, 5.45, "Agreement with the teacher", 2)
    page.key(6.2, 5.45, [(f"macro F1 {macro:.3f}", LEAD)])
    agreement(page.place(1.85, 5.8, 4.35, 4.35), matrix)
    page.panel_title(6.95, 5.45, "Student performance", 3)
    performance(page.place(6.95, 5.8, RIGHT - 6.95, 4.35), np.array(data["truth"]), teacher_truth())

    notes = [
        f"The {total:,} citations in GraphIV's release, each labeled from its sentence and"
        f" section heading, at {SPEED:,} a second end to end on one RTX 3090.",
        f"The {matrix.sum():,} held-out citations of the teacher's labels, at accuracy"
        f" {np.trace(matrix) / matrix.sum():.3f}. Rows are the teacher's intent, and each sums"
        " to 100%.",
        f"The {data['reference']:,} human-labeled citations of the ground truth, which the"
        " teacher's prompt was evolved on and the student never trained on.",
    ]
    page.save(out, 11.75, notes)


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
