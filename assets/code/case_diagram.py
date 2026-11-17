"""
The worked case a benchmark construction figure draws: each review's references on one timeline

One row per review, one dot per reference at its date, and the answer drawn in the benchmark's color
"""

from __future__ import annotations

from datetime import date
from typing import Any

from matplotlib.patches import FancyBboxPatch
from style import INK, LEFT, MUTED, OUTLINE, PANEL, Page, tint

ROW = 0.62


def x_of(day: str, start: date, stop: date, left: float, right: float) -> float:
    """
    Where a date sits on the timeline

    Args:
        day: An iso date.
        start: The timeline's first day.
        stop: Its last day.
        left: Its left edge.
        right: Its right edge.

    Returns:
        The x position in inches.
    """
    span = (stop - start).days or 1
    return left + (right - left) * (date.fromisoformat(day) - start).days / span


def timeline(
    page: Page, top: float, example: dict[str, Any], answer: set[str], color: str, labels: list[str]
) -> float:
    """
    Every reference of the case's reviews on one timeline, a row per review

    Args:
        page: The page.
        top: Top edge.
        example: The worked case with its references.
        answer: The references that are the answer.
        color: The benchmark's color.
        labels: Each review's short name, in anchor order.

    Returns:
        The bottom edge.
    """
    ax = page.ax
    refs = example["references"]
    days = [r["date"] for r in refs.values()] + example["dates"]
    start = date(int(min(days)[:4]), 1, 1)
    stop = date.fromisoformat(max(days))
    stop = date(stop.year + (stop.month > 6), 7 if stop.month <= 6 else 1, 1)
    label_w = 2.1
    left, right = LEFT + label_w, page.right - 0.2
    high = ROW * len(example["anchors"]) + 0.5
    ax.add_patch(
        FancyBboxPatch(
            (LEFT, top),
            page.right - LEFT,
            high,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor=PANEL,
            edgecolor=OUTLINE,
            linewidth=1.0,
        )
    )
    for row, (anchor, label, day) in enumerate(
        zip(example["anchors"], labels, example["dates"], strict=True)
    ):
        y = top + 0.35 + row * ROW
        ax.text(LEFT + 0.2, y, label, fontsize=11, fontweight="semibold", va="center")
        ax.text(LEFT + 0.2, y + 0.2, day, fontsize=9.5, color=MUTED, va="center")
        ax.plot([left, right], [y, y], color=OUTLINE, lw=0.8, zorder=1)
        cited = [(t, r) for t, r in refs.items() if anchor in r["by"]]
        for target, ref in cited:
            x = x_of(ref["date"], start, stop, left, right)
            inside = target in answer
            ax.plot(
                x,
                y,
                "o",
                markersize=5.5 if inside else 4,
                color=color if inside else tint(MUTED, 0.35),
                markeredgewidth=0,
                zorder=3 if inside else 2,
            )
        x = x_of(day, start, stop, left, right)
        ax.plot(x, y, marker="s", markersize=9, color=INK, zorder=4)
    years = range(start.year, stop.year + 1)
    step = max(1, len(years) // 8)
    for year in years[::step]:
        x = x_of(f"{year}-01-01", start, stop, left, right)
        ax.text(x, top + high - 0.08, str(year), fontsize=9.5, color=MUTED, ha="center")
    x = x_of(example["date_max"], start, stop, left, right)
    ax.plot([x, x], [top + 0.12, top + high - 0.28], color=INK, lw=1.2, ls=(0, (3, 2)), zorder=2)
    ax.text(x, top + 0.1, "date_max", fontsize=9.5, color=INK, ha="center", fontweight="semibold")
    return top + high


def question(page: Page, top: float, example: dict[str, Any], answer: int, color: str) -> float:
    """
    The question the case asks and the size of its answer, under the timeline

    Args:
        page: The page.
        top: Top edge.
        example: The worked case.
        answer: How many papers the answer holds.
        color: The benchmark's color.

    Returns:
        The bottom edge.
    """
    ax = page.ax
    ax.text(LEFT, top, "query_text", fontsize=10.5, family="monospace", color=MUTED, va="baseline")
    lines = page.wrapped(example["query"], page.right - LEFT - 1.6, 12.5)
    for index, line in enumerate(lines):
        ax.text(LEFT + 1.6, top + index * 0.24, line, fontsize=12.5, va="baseline")
    y = top + len(lines) * 0.24 + 0.2
    ax.text(LEFT, y, "citations", fontsize=10.5, family="monospace", color=MUTED, va="baseline")
    ax.text(
        LEFT + 1.6,
        y,
        f"{answer} papers, the dots in color",
        fontsize=12.5,
        color=color,
        fontweight="semibold",
        va="baseline",
    )
    return y


def construction(
    page: Page,
    title: list[tuple[str, bool | str]],
    steps: list[tuple[str, str, str]],
    example: dict[str, Any],
    answer: set[str],
    color: str,
    labels: list[str],
    extra_note: bool = False,
) -> float:
    """
    A construction figure: the title, the steps a case is built by, then one real case

    Args:
        page: The page.
        title: The title's runs.
        steps: (name, detail, kind) per step, drawn left to right with arrows between.
        example: The worked case with its references.
        answer: The references that are the answer.
        color: The benchmark's color.
        labels: Each review's short name, in anchor order.
        extra_note: Whether the answer step carries the second footnote.

    Returns:
        Where the footnotes start.
    """
    page.title([title])
    gap = 0.45
    width = (page.right - LEFT - gap * (len(steps) - 1)) / len(steps)
    edges = []
    for index, (name, detail, kind) in enumerate(steps):
        note = 2 if index == len(steps) - 1 and extra_note else 0
        left = LEFT + index * (width + gap)
        edges.append(page.box(left, 1.5, width, 1.15, name, detail, kind, note))
    for a, b in zip(edges, edges[1:], strict=False):
        page.arrow((a[1] + 0.04, 2.075), (b[0] - 0.04, 2.075))
    page.panel_title(LEFT, 3.25, "One real case", note=1)
    bottom = timeline(page, 3.45, example, answer, color, labels)
    return question(page, bottom + 0.45, example, len(answer), color) + 0.55
