"""
Draws citation-labeling-pipeline.png: the teacher's prompt evolved and applied, then the student
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import release_counts
from labeler_scores import GENERATIONS, KEPT, SAMPLE, SPEED, load, scores, teacher_truth
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
from style import (
    LEAD,
    LEFT,
    MUTED,
    OUTLINE,
    PANEL,
    PURPLE,
    RIGHT,
    ROOT,
    Page,
    apply_theme,
)


def curve(
    page: Page, x: float, top: float, width: float, high: float, points: list[float], kept: int
) -> Axes:
    """
    A small line of one score per round, with the first and the kept round labeled

    Args:
        page: The page.
        x: Left edge of the plot.
        top: Top edge of the plot.
        width: Width.
        high: Height.
        points: The score per round.
        kept: Which round was kept.

    Returns:
        The panel.
    """
    axis = page.place(x, top, width, high)
    rounds = np.arange(len(points))
    axis.plot(rounds, points, color=MUTED, linewidth=1.4, zorder=2)
    axis.plot(rounds, points, "o", color=PURPLE, markersize=5, zorder=3)
    axis.plot([kept], [points[kept]], "o", color=LEAD, markersize=8, zorder=4)
    # the first round sits low on a rising line, so its label goes to its right
    for index, text, color, offset, ha, va in (
        (0, f"{points[0]:.3f}", MUTED, (7, 0), "left", "center"),
        (kept, f"{points[kept]:.3f}", LEAD, (0, 7), "center", "bottom"),
    ):
        axis.annotate(
            text,
            (float(index), float(points[index])),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=10.5,
            color=color,
            fontweight="semibold",
        )
    low = min(points)
    axis.set_ylim(low - (1 - low) * 0.15, 1.0 + (1 - low) * 0.25)
    axis.set_xlim(-0.6, len(points) - 0.4)
    axis.set_xticks(rounds[:: 1 if len(points) < 8 else 2])
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="both", labelsize=10)
    axis.tick_params(axis="y", labelcolor=MUTED, pad=4)
    return axis


def draw(data: dict[str, Any], labeled: int, out: Path) -> None:
    """
    citation-labeling-pipeline.png: the teacher's prompt evolved and applied, then the student

    Args:
        data: The cached scores.
        labeled: How many citations the release carries.
        out: Where the figure is written.
    """
    page = Page(12.95)
    ax = page.ax
    sizes = data["sizes"]
    total = sum(sizes.values())
    gens = [a for a, _ in GENERATIONS]
    epochs = [f for _, _, f in data["curve"]]
    best = int(np.argmax(epochs))
    matrix = np.array(data["confusion"])
    truth = np.array(data["truth"])

    page.title([[("GraphIV's Citation Labeling Pipeline", False)]])
    page.kinds(["data", "model"])

    for top, bottom, name in (
        (1.5, 6.5, "Teacher: Gemma 4 26B"),
        (6.85, 10.35, "Student: DistilBERT"),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (LEFT, top),
                RIGHT - LEFT,
                bottom - top,
                boxstyle="round,pad=0,rounding_size=0.12",
                facecolor=PANEL,
                edgecolor=OUTLINE,
                linewidth=1.0,
                zorder=1,
            )
        )
        ax.text(LEFT + 0.3, top + 0.5, name, fontsize=16, fontweight="bold", va="baseline")

    # A, the label, critique and redraft loop over the ground truth
    page.stage(0.8, 2.65, "A", "Automatic prompt engineering", 1)
    row1, row2, high = 3.0, 4.55, 0.95
    truth_box = page.box(
        0.8, row1, 2.05, high, "Ground truth", "985 human-labeled\ncitations", "data", 2
    )
    labeler = page.box(3.3, row1, 1.9, high, "Labeler", "labels all of them,\n25 a call", "model")
    judge = page.box(5.65, row1, 1.85, high, "Judge", "diagnoses\nthe mistakes", "model")
    drafter = page.box(5.65, row2, 1.85, high, "Drafter", "rewrites\nthe prompt", "model")
    prompt = page.box(3.3, row2, 1.9, high, "Prompt", "one per\ngeneration", "data")
    middle1, middle2 = row1 + high / 2, row2 + high / 2
    page.arrow((truth_box[1], middle1), (labeler[0], middle1))
    page.arrow((labeler[1], middle1), (judge[0], middle1))
    page.arrow(((judge[0] + judge[1]) / 2, judge[3]), ((drafter[0] + drafter[1]) / 2, drafter[2]))
    page.arrow((drafter[0], middle2), (prompt[1], middle2))
    page.arrow(
        ((prompt[0] + prompt[1]) / 2, prompt[2]), ((labeler[0] + labeler[1]) / 2, labeler[3])
    )
    chart = ax.text(0.8, row2 + 0.05, "Accuracy per generation", fontsize=11, color=MUTED)
    page.marked.append((chart, 3))
    curve(page, 1.25, row2 + 0.3, 1.6, 1.15, gens, KEPT)
    page.result(
        4.35, 6.0, f"{len(gens)} generations, generation {KEPT} kept", "label, critique, redraft"
    )

    ax.plot([7.85, 7.85], [2.35, 6.2], color=OUTLINE, lw=1.0)

    # B, the kept prompt applied to a sample of real citations
    page.stage(8.15, 2.65, "B", "Teacher labeling", 4)
    kept = page.box(8.15, row1, 1.65, high, "Prompt", f"generation {KEPT}", "data")
    sample = page.box(10.05, row1, 1.65, high, f"{SAMPLE:,}", "citations from\nthe store", "data")
    teacher = page.box(8.9, row2, 2.05, high, "Gemma 4 26B", "25 citations a call", "model")
    page.arrow(((kept[0] + kept[1]) / 2, kept[3]), (teacher[0] + 0.45, teacher[2]))
    page.arrow(((sample[0] + sample[1]) / 2, sample[3]), (teacher[1] - 0.45, teacher[2]))
    page.result(9.925, 6.0, f"{total:,} of {SAMPLE:,} labeled", f"{SAMPLE - total} failed to parse")

    # C, the student trained on those labels, then run over every citation
    page.stage(0.8, 7.95, "C", "Distillation", 5)
    row3 = 8.3
    labels = page.box(
        0.8,
        row3,
        2.35,
        1.0,
        "Teacher labels",
        f"{sizes['train']:,} train, {sizes['val']:,}\nvalidation, {sizes['test']:,} test",
        "data",
    )
    student = page.box(
        3.6, row3, 2.35, 1.0, "DistilBERT", "section and sentence,\n256 tokens", "model"
    )
    ax.text(6.85, row3 - 0.12, "Validation macro F1 per epoch", fontsize=11, color=MUTED)
    curve(page, 6.85, row3 + 0.05, 2.25, 0.95, epochs, best)
    output = page.box(
        9.6, row3, 2.1, 1.0, f"{labeled / 1e6:.1f}M citations", "in GraphIV,\neach labeled", "data"
    )
    middle3 = row3 + 0.5
    page.arrow((labels[1], middle3), (student[0], middle3))
    page.arrow((student[1], middle3), (6.4, middle3))
    page.arrow((9.15, middle3), (output[0], middle3))
    page.result(
        3.375,
        9.85,
        f"macro F1 {float(scores(matrix)[2].mean()):.3f} against the teacher",
        f"{float(scores(truth)[2].mean()):.3f} against the ground truth",
    )
    page.result(7.975, 9.85, f"epoch {best} of {len(epochs)} kept", "by validation macro F1")
    page.result(10.65, 9.85, f"{SPEED:,} citations a second", "on one RTX 3090", 6)

    notes = [
        "The labeler, the judge and the drafter are one Gemma 4 26B at temperature 0, with"
        " reasoning off. The judge reads up to 60 mistakes along with the labeler's reasons.",
        f"{data['reference']:,} citations labeled by hand, 167 per intent and 150 for motivates.",
        f"Scored on the ground truth. Generation {KEPT} is kept, and scored"
        f" {np.trace(teacher_truth()) / teacher_truth().sum():.3f} when rerun.",
        f"{SAMPLE:,} resolved citations drawn at random from the store, with the citation marker"
        " replaced by [CITE]. A citation left unanswered is dropped rather than counted as"
        " background.",
        "Rare intents are oversampled at inverse frequency to the power 0.6, and the epoch with the"
        " best validation macro F1 is kept.",
        "Measured end to end on one RTX 3090 over every resolved citation in the store, writing to"
        " the store included.",
    ]
    page.save(out, 10.95, notes)


def main() -> None:
    """
    Scores and counts whatever is not cached, then draws the figure to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    config = Path(str(args.config_path))
    labeled = sum(int(n) for _, _, n in release_counts.load(config)["cites"])
    apply_theme()
    draw(load(config), labeled, Path(str(args.out)))


if __name__ == "__main__":
    main()
