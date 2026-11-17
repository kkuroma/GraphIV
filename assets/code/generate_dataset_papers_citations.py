"""
Draws dataset-papers-citations.png: papers and citations per year, the intents, the most discussed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Rectangle
from release_counts import load
from style import (
    INK,
    LEAD,
    LEFT,
    MUTED,
    PANEL,
    PURPLE,
    RIGHT,
    ROOT,
    STANCE_COLORS,
    STANCES,
    Page,
    apply_theme,
    intents,
    short,
    tick,
    tint,
)

FIRST_YEAR = 1991


def per_year(axis: Axes, counts: dict[int, int], color: str, inner: dict[int, int]) -> None:
    """
    A count per year as bars, with a share of each year drawn darker at the bar's foot

    Args:
        axis: The panel.
        counts: Year to count.
        color: The bars' color.
        inner: Year to the share drawn darker, or empty for plain bars in the full color.
    """
    xs = np.array(sorted(y for y in counts if y >= FIRST_YEAR))
    ys = np.array([counts[x] for x in xs], dtype=float)
    axis.bar(xs, ys, width=0.74, color=tint(color, 0.38) if inner else color, zorder=3)
    if inner:
        axis.bar(xs, [inner.get(int(x), 0) for x in xs], width=0.74, color=color, zorder=3)
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    axis.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    axis.set_xticks(range(1995, int(xs[-1]) + 1, 5))
    peak = int(np.argmax(ys))
    axis.annotate(
        short(ys[peak]),
        (float(xs[peak]), float(ys[peak])),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        color=INK,
        fontweight="semibold",
    )
    step = 10 ** int(np.log10(ys.max()))
    step = step * 5 if ys.max() / step > 5 else step / 2 if ys.max() / step < 2 else step
    ticks = np.arange(0, ys.max() * 1.2, step)
    axis.set_yticks(ticks, ["0"] + [tick(t) for t in ticks[1:]])
    axis.set_ylim(0, ys.max() * 1.2)
    axis.tick_params(axis="y", labelcolor=MUTED, labelsize=11, pad=6)
    axis.tick_params(axis="x", labelsize=11)


def discussed(page: Page, x: float, top: float, width: float, rows: list[list[Any]]) -> None:
    """
    The ten papers most discussed, each a title over a bar split by how it is discussed

    Args:
        page: The page.
        x: Left edge.
        top: Top edge.
        width: Width.
        rows: (target, intent, distinct citing papers, title, date) for the ten papers.
    """
    ax = page.ax
    papers: dict[str, dict[str, Any]] = {}
    for target, intent, n, title, date in rows:
        entry = papers.setdefault(target, {"title": " ".join(str(title).split()), "year": date[:4]})
        entry[intent] = int(n)
    order = sorted(papers.values(), key=lambda e: -sum(e.get(s, 0) for s in STANCES))
    biggest = max(sum(e.get(s, 0) for s in STANCES) for e in order)
    span = width - 0.7
    for index, entry in enumerate(order):
        y = top + index * 0.42
        title = page.fit(entry["title"], width - 0.6, 11)
        ax.text(x, y + 0.15, title, fontsize=11, va="baseline")
        ax.text(x + width, y + 0.15, entry["year"], fontsize=10.5, color=MUTED, ha="right")
        left = x
        for stance in STANCES:
            length = span * entry.get(stance, 0) / biggest
            ax.add_patch(
                Rectangle(
                    (left, y + 0.22),
                    length,
                    0.13,
                    facecolor=STANCE_COLORS[stance],
                    edgecolor=PANEL,
                    linewidth=1.2,
                    zorder=3,
                )
            )
            left += length
        total = sum(entry.get(s, 0) for s in STANCES)
        ax.text(left + 0.08, y + 0.285, f"{total:,}", fontsize=10.5, va="center")


def draw(data: dict[str, Any], out: Path) -> None:
    """
    dataset-papers-citations.png: papers and citations per year, the intents, the most discussed

    Args:
        data: The cached counts.
        out: Where the figure is written.
    """
    page = Page(13.3)
    ax = page.ax
    latest = str(data["papers"][0][1])
    counts: dict[str, int] = {}
    cite_years: dict[int, int] = {}
    for year, intent, n in data["cites"]:
        counts[intent] = counts.get(intent, 0) + int(n)
        cite_years[int(year)] = cite_years.get(int(year), 0) + int(n)
    paper_years = {int(y): int(n) for y, n in data["years"]}
    reviews = {int(y): int(n) for y, n in data["reviews"]}
    early = sum(n for y, n in paper_years.items() if y < FIRST_YEAR)

    page.title([[("Papers and Citations", False)]])

    page.panel_title(LEFT, 1.8, "Papers per year", 1)
    papers_axis = page.place(1.15, 2.15, 4.75, 3.0)
    per_year(papers_axis, paper_years, PURPLE, reviews)
    legend = papers_axis.legend(
        handles=[
            Patch(facecolor=tint(PURPLE, 0.38), label="papers"),
            Patch(facecolor=PURPLE, label=f"literature reviews, {sum(reviews.values()):,}"),
        ],
        loc="upper left",
        fontsize=11,
        handlelength=1.0,
        handleheight=0.9,
        borderaxespad=0.6,
    )
    page.marked.append((legend.get_texts()[1], 2))

    page.panel_title(6.65, 1.8, "Citations per year", 3)
    per_year(page.place(7.3, 2.15, 4.7, 3.0), cite_years, LEAD, {})

    page.panel_title(LEFT, 6.35, "Citation intent breakdown", 4)
    intents(ax, LEFT, 6.65, 5.4, counts, sorted(counts, key=lambda k: -counts[k]))

    page.panel_title(6.65, 6.35, "Most discussed papers", 5)
    key: list[tuple[str, bool | str]] = []
    for stance in STANCES:
        key += [("·", MUTED)] if key else []
        key.append((stance, STANCE_COLORS[stance]))
    page.key(RIGHT, 6.35, key)
    discussed(page, 6.65, 6.65, RIGHT - 6.65, data["stance"])

    notes = [
        f"By the date of the first version. {latest[:4]} runs to {latest}, and the {early}"
        f" papers dated before {FIRST_YEAR} are not drawn.",
        "Scored by GraphIV's survey labeler, a DistilBERT student of Gemma 4 26B, at 0.920"
        " average precision on held-out papers. A review scores 0.5 or more.",
        "By the citing paper's year, over citations whose two ends are both in the release.",
        "Labeled by GraphIV's citation labeler, a DistilBERT student of Gemma 4 26B that agrees"
        " with its teacher at 0.815 macro F1.",
        "Discussed means extended, compared with or critiqued. Counted as distinct citing papers,"
        " so a paper that both extends and critiques a work counts once in each.",
    ]
    page.save(out, 11.6, notes)


def main() -> None:
    """
    Counts whatever is not cached, then draws the figure to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    data = load(Path(str(args.config_path)))
    apply_theme()
    draw(data, Path(str(args.out)))


if __name__ == "__main__":
    main()
