"""
Draws survey-labeling-pipeline.png: two piles labeled by the teacher, then the student
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import release_counts
from matplotlib.patches import FancyBboxPatch
from style import (
    INK,
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
from survey_scores import AGREEMENT, RATE, SPEED, load

# the held-out average precision three ways, as (label, the key under areas)
AREAS = (
    ("as drawn", "sampled"),
    ("weighted", "weighted"),
    ("at 100 to 1", "stress"),
)


def areas(page: Page, x: float, top: float, width: float, data: dict[str, Any]) -> None:
    """
    The held-out average precision as drawn, weighted to the corpus, and resampled at 100 to 1

    Args:
        page: The page.
        x: Left edge of the labels.
        top: Top edge.
        width: Width, labels included.
        data: The cached scores.
    """
    gutter = max(page.measure(name, 11) for name, _ in AREAS) + 0.15
    span = width - gutter - 0.55
    for index, (name, key) in enumerate(AREAS):
        y = top + index * 0.34
        value = float(data["areas"][key])
        page.ax.text(x, y + 0.13, name, fontsize=11, color=MUTED, va="center")
        page.ax.add_patch(
            FancyBboxPatch(
                (x + gutter, y + 0.04),
                span * value,
                0.18,
                boxstyle="round,pad=0,rounding_size=0.03",
                facecolor=LEAD if key == "weighted" else PURPLE,
                edgecolor="none",
                zorder=3,
            )
        )
        page.ax.text(
            x + gutter + span * value + 0.07,
            y + 0.13,
            f"{value:.3f}",
            fontsize=10.5,
            va="center",
            color=LEAD if key == "weighted" else INK,
        )


def draw(data: dict[str, Any], total: int, reviews: int, out: Path) -> None:
    """
    The piles drawn and labeled by the teacher, then the student trained on them

    Args:
        data: The cached scores.
        total: How many papers the release carries.
        reviews: How many of them score as a review.
        out: Where the figure is written.
    """
    page = Page(12.95)
    ax = page.ax
    word, plain = data["piles"]["word"], data["piles"]["plain"]
    drawn = word[0] + plain[0]
    answered = sum(AGREEMENT.values())
    unanimous = (AGREEMENT[0] + AGREEMENT[3]) / answered

    page.title([[("GraphIV's Survey Labeling Pipeline", False)]])
    page.kinds(["data", "step", "model"])

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

    # A, two equal piles drawn by whether the title carries a review word
    page.stage(0.8, 2.65, "A", "Two piles", 1)
    row1, row2, high = 3.0, 4.55, 0.95
    snapshot = page.box(
        0.8, row1, 1.9, high, "arXiv snapshot", f"{(word[2] + plain[2]) / 1e6:.2f}M papers", "data"
    )
    regex = page.box(3.1, row1, 1.75, high, "Title regex", "11 review\nphrases", "step", 2)
    review = page.box(5.3, row1, 2.2, high, "Review titles", f"{word[0]:,} of\n{word[2]:,}", "data")
    other = page.box(5.3, row2, 2.2, high, "Other titles", f"{plain[0]:,} of\n{plain[2]:,}", "data")
    middle1, middle2 = row1 + high / 2, row2 + high / 2
    page.arrow((snapshot[1], middle1), (regex[0], middle1))
    page.arrow((regex[1], middle1), (review[0], middle1))
    page.arrow(((regex[0] + regex[1]) / 2, regex[3]), (other[0], middle2), bend=0.3)
    page.result(
        4.15, 6.0, f"{drawn:,} papers in two equal piles", "from populations 58 times apart"
    )

    ax.plot([7.85, 7.85], [2.35, 6.2], color=OUTLINE, lw=1.0)

    # B, the teacher's one question, and two other families asking it again
    page.stage(8.15, 2.65, "B", "Teacher labeling", 3)
    teacher = page.box(
        8.15, row1, 1.75, high, "Gemma 4 26B", "is it a literature\nreview?", "model"
    )
    check = page.box(10.1, row1, 1.6, high, "Qwen, GLM", "relabel 400", "model", 4)
    labels = page.box(
        8.85,
        row2,
        2.2,
        high,
        f"{drawn:,} labels",
        f"{word[1] / word[0]:.0%} and {plain[1] / plain[0]:.1%}\nreviews",
        "data",
    )
    page.arrow((review[1], middle1), (teacher[0], middle1))
    page.arrow((other[1], middle2 - 0.2), (teacher[0], middle1 + 0.3))
    page.arrow(((teacher[0] + teacher[1]) / 2, teacher[3]), (labels[0] + 0.45, labels[2]))
    page.arrow(((check[0] + check[1]) / 2, check[3]), (labels[1] - 0.45, labels[2]), dashed=True)
    page.result(
        9.925, 6.0, f"{unanimous:.0%} unanimous", f"on the {answered} papers all three answered"
    )

    # C, the student trained on those labels, then run over every paper
    page.stage(0.8, 7.95, "C", "Distillation", 5)
    row3 = 8.3
    split = page.box(
        0.8,
        row3,
        2.2,
        1.0,
        "Teacher labels",
        f"{data['trained']:,} train,\n{drawn - data['trained']:,} held out",
        "data",
    )
    student = page.box(
        3.45, row3, 2.3, 1.0, "DistilBERT", "title and abstract,\n256 tokens", "model"
    )
    ax.text(6.3, row3 - 0.12, "Held-out average precision", fontsize=11, color=MUTED)
    areas(page, 6.3, row3 + 0.05, 2.75, data)
    output = page.box(
        9.6, row3, 2.1, 1.0, f"{total / 1e6:.2f}M papers", "in GraphIV,\neach scored", "data"
    )
    middle3 = row3 + 0.5
    page.arrow((split[1], middle3), (student[0], middle3))
    page.arrow((student[1], middle3), (6.2, middle3))
    page.arrow((9.15, middle3), (output[0], middle3))
    page.result(3.275, 9.85, "2 epochs", "settings from an Optuna search")
    page.result(
        7.675, 9.85, f"{data['areas']['weighted']:.3f} average precision", "weighted to the corpus"
    )
    page.result(10.65, 9.85, f"{SPEED} papers a second", "on one RTX 3090", 6)

    notes = [
        f"Reviews are {reviews / total:.1%} of GraphIV, so a random sample holds too few to learn"
        " from. Every"
        " score weights a paper back by the size of the pile it was drawn from.",
        "survey, review, overview, tutorial, taxonom, state of the art, systematic, advances in,"
        " progress in, perspectives on and landscape, in any case.",
        "Gemma 4 26B reads the title and the abstract, cut at 2,500 characters, and answers"
        ' "Is this paper a literature review of a research area?" as a JSON boolean.',
        f"Qwen 3.6 35B and GLM 4.7 Flash relabeled 400 papers under the same prompt, and all three"
        f" answered {answered} of them.",
        "Cross-entropy with reviews weighted 2.89, and AdamW at learning rate 4.0e-5 with a linear"
        " warmup over 14.5% of the steps.",
        f"Measured end to end on one RTX 3090 over a {RATE[0]:,}-paper update, writing to the store"
        " included.",
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
    release = release_counts.load(config)
    total = int(release["papers"][0][0])
    reviews = sum(int(n) for _, n, _ in release["review_fields"])
    apply_theme()
    draw(load(config), total, reviews, Path(str(args.out)))


if __name__ == "__main__":
    main()
