"""
Draws dataset.png, the dataset card's title, headline numbers and its three released tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyBboxPatch
from release_counts import load
from style import (
    GREEN,
    INK,
    LEAD,
    LEFT,
    MUTED,
    OUTLINE,
    PANEL,
    PURPLE,
    RIGHT,
    ROOT,
    WIDE,
    Page,
    apply_theme,
    short,
    tint,
    write_runs,
)

REPO = "git.kuroma.dev/kkuroma/graphiv"
HUB = "hf.co/datasets/kkuroma/GraphIV"

TITLE: list[list[tuple[str, bool | str]]] = [
    [("GraphIV:", True), ("the open, comprehensive knowledge graph of arXiv,", False)],
    [("rebuilt biweekly on a reproducible pipeline", False)],
]
SUBTITLE: list[tuple[str, bool | str]] = [
    ("Every", False),
    ("arXiv paper,", PURPLE),
    ("every", False),
    ("citation", LEAD),
    ("between them and its", False),
    ("intent,", LEAD),
    ("and every paper's place among", False),
    ("4,096 topics", GREEN),
]
# each released table's columns as (column, what it holds, its footnote or 0)
COLUMNS: dict[str, list[tuple[str, str, int]]] = {
    "citations": [
        ("source_id", "the citing paper", 0),
        ("target_id", "the cited paper", 0),
        ("section", "its section heading", 0),
        ("resolved_by", "how it was matched", 7),
        ("intent", "why it was cited", 8),
        ("intent_score", "the labeler's confidence", 0),
    ],
    "papers": [
        ("arxiv_id", "e.g. 1706.03762", 0),
        ("title", "as arXiv lists it", 0),
        ("abstract", "as arXiv lists it", 0),
        ("date", "first version's date", 0),
        ("primary_category", "e.g. cs.CL", 0),
        ("authors", "as listed", 0),
        ("surnames", "last names, a list", 0),
        ("latest_version", "e.g. v7", 0),
        ("doi", "when it has one", 0),
        ("journal_ref", "where it was published", 0),
        ("survey_probability", "how likely a review", 9),
        ("embedding", "768-d, title and abstract", 1),
    ],
    "topics": [
        ("arxiv_id", "the paper", 0),
        ("level_0, name_0", "field, 1 of 16", 0),
        ("level_1, name_1", "subfield, 1 of 256", 0),
        ("level_2, name_2", "topic, 1 of 4,096", 0),
    ],
}


def card_size(page: Page, columns: list[tuple[str, str, int]]) -> tuple[float, float]:
    """
    Where a card's descriptions start and how wide it has to be, from its measured columns

    Args:
        page: The page, whose renderer measures.
        columns: (column, what it holds, footnote) per column.

    Returns:
        The gutter and the width, in inches.
    """
    gutter = 0.2 + max(page.measure(c, 9.5, family="monospace") for c, _, _ in columns) + 0.2
    holds = max(page.measure(h, 10, color=MUTED) + (0.12 if n else 0) for _, h, n in columns)
    return gutter, gutter + holds + 0.2


def card(
    page: Page,
    x: float,
    top: float,
    width: float,
    name: str,
    rows: int,
    columns: list[tuple[str, str, int]],
    color: str,
    gutter: float,
) -> dict[str, float]:
    """
    One released table as a card: its name, its row count, then each column and what it holds

    Args:
        page: The page.
        x: Left edge.
        top: Top edge.
        width: Width.
        name: The table's name.
        rows: How many rows it has.
        columns: (column, what it holds, footnote) per column.
        color: The table's color.
        gutter: Where the descriptions start, in inches from the left edge.

    Returns:
        Each column's center line, for the joins.
    """
    ax = page.ax
    first, step = top + 1.13, 0.27
    high = 1.13 + step * (len(columns) - 1) + 0.27
    ax.add_patch(
        FancyBboxPatch(
            (x, top),
            width,
            high,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor=PANEL,
            edgecolor=tint(color, 0.6),
            linewidth=1.3,
            zorder=2,
        )
    )
    ax.text(
        x + 0.2,
        top + 0.45,
        name,
        fontsize=15,
        fontweight="bold",
        color=color,
        va="baseline",
        family="monospace",
    )
    ax.text(
        x + width - 0.2,
        top + 0.45,
        f"{rows:,} rows",
        fontsize=10.5,
        color=MUTED,
        ha="right",
        va="baseline",
    )
    ax.plot([x + 0.2, x + width - 0.2], [top + 0.78, top + 0.78], color=OUTLINE, lw=0.8)
    lines = {}
    for index, (column, holds, note) in enumerate(columns):
        y = first + index * step
        lines[column] = y
        ax.text(x + 0.2, y, column, fontsize=9.5, color=INK, va="center", family="monospace")
        text = ax.text(x + gutter, y, holds, fontsize=10, color=MUTED, va="center")
        if note:
            page.marked.append((text, note))
    return lines


def join(ax: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """
    A join between two tables' key columns, a line with a dot at each end

    Args:
        ax: The page's overlay, in inches from the top left.
        start: One column's end, at a card's edge.
        end: The other's.
    """
    ax.plot([start[0], end[0]], [start[1], end[1]], color=MUTED, lw=1.2, zorder=3)
    for point in (start, end):
        ax.add_patch(Circle(point, 0.045, facecolor=MUTED, edgecolor="none", zorder=4))


def draw(data: dict[str, Any], out: Path) -> None:
    """
    dataset.png: the title, the headline numbers, where it lives, and the three released tables

    Args:
        data: The cached counts.
        out: Where the figure is written.
    """
    page = Page(14.65)
    ax = page.ax
    total, latest = int(data["papers"][0][0]), str(data["papers"][0][1])
    snapshot = int(data["snapshot"][0][0])
    citations = sum(int(n) for _, _, n in data["cites"])
    extracted = int(data["extracted"][0][0])

    page.title(TITLE)
    write_runs(page.fig, *page.at(LEFT, 2.2), SUBTITLE, 16, "medium", INK)

    # the headline numbers, three tiles a row in one rounded panel
    grid_top, row_high, per_row = 2.6, 1.2, 3
    tiles = [
        (f"{total / 1e6:.2f}M", "arXiv papers (and growing)", 1),
        (f"{total / snapshot:.1%}", "of arXiv's snapshot", 2),
        (short(citations), "citations with intent", 3),
        (f"{4096:,}", "named topics", 4),
        ("2 weeks", "between rebuilds", 5),
        ("RTX 3090", "rebuilds all of it", 6),
    ]
    rows = len(tiles) // per_row
    ax.add_patch(
        FancyBboxPatch(
            (LEFT, grid_top),
            RIGHT - LEFT,
            row_high * rows,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=PANEL,
            edgecolor=OUTLINE,
            linewidth=1.0,
        )
    )
    width = (RIGHT - LEFT) / per_row
    for index, (value, name, note) in enumerate(tiles):
        row, col = divmod(index, per_row)
        top, middle = grid_top + row * row_high, LEFT + width * (col + 0.5)
        ax.text(middle, top + 0.7, value, fontsize=30, fontweight="bold", ha="center")
        label = ax.text(middle, top + 1.0, name, fontsize=12.5, color=MUTED, ha="center")
        page.marked.append((label, note))
        if col:
            x = LEFT + width * col
            ax.plot([x, x], [top + 0.22, top + row_high - 0.22], color=OUTLINE, lw=1.0)
    for row in range(1, rows):
        y = grid_top + row * row_high
        ax.plot([LEFT + 0.25, RIGHT - 0.25], [y, y], color=OUTLINE, lw=1.0)

    # where it lives, under which license, and how fresh it is
    x, line = page.at(LEFT, 0)[0], page.at(0, 5.5)[1]
    for index, (label, value) in enumerate(
        [
            ("Data", f"{HUB}, CC BY-NC 4.0"),
            ("Code", f"{REPO}, AGPLv3"),
            ("Papers up to", latest),
        ]
    ):
        if index:
            x += 0.42 / WIDE
        x = write_runs(page.fig, x, line, [(label, False)], 12, "medium", MUTED)
        x = write_runs(page.fig, x + 0.1 / WIDE, line, [(value, False)], 12, "semibold", INK)
    if x > RIGHT / WIDE:
        print(f"[dataset] the access line runs {x * WIDE - RIGHT:.2f} in past the margin")

    # the three tables, joined on the arXiv id
    ax.plot([LEFT, RIGHT], [5.95, 5.95], color=OUTLINE, lw=1.0)
    runs: list[tuple[str, bool | str]] = [
        ("What's released:", INK),
        ("three tables joined by arxiv_id", MUTED),
    ]
    write_runs(page.fig, *page.at(LEFT, 6.45), runs, 16, "bold")
    tables = [("citations", citations, LEAD), ("papers", total, PURPLE), ("topics", total, GREEN)]
    sizes = [card_size(page, COLUMNS[name]) for name, _, _ in tables]
    gap = (RIGHT - LEFT - sum(w for _, w in sizes)) / 2
    x, edges, keys = LEFT, [], []
    for (name, count, color), (gutter, width) in zip(tables, sizes, strict=True):
        keys.append(card(page, x, 6.8, width, name, count, COLUMNS[name], color, gutter))
        edges.append((x, x + width))
        x += width + gap
    for column in ("source_id", "target_id"):
        join(ax, (edges[0][1], keys[0][column]), (edges[1][0], keys[1]["arxiv_id"]))
    join(ax, (edges[2][0], keys[2]["arxiv_id"]), (edges[1][1], keys[1]["arxiv_id"]))
    for (low, high), text in (
        ((edges[0][1], edges[1][0]), "n : 1"),
        ((edges[1][1], edges[2][0]), "1 : 1"),
    ):
        ax.text(
            (low + high) / 2,
            keys[1]["arxiv_id"] - 0.14,
            text,
            fontsize=9.5,
            color=MUTED,
            ha="center",
        )

    notes = [
        f"Every paper in arXiv's metadata snapshot up to {latest} that the pipeline fetched and"
        f" embedded, {total:,} in all, embedded with nomic-embed-text-v2-moe.",
        f"{total:,} of the {snapshot:,} papers in that snapshot. The other"
        f" {snapshot - total:,} are still in the pipeline's queue.",
        f"Citations whose cited paper is also in the release, {citations:,} of the"
        f" {extracted:,} extracted from the full text. The citing sentence stays out, since it is"
        " the citing paper's own text.",
        "A three-level tree of 16 fields, 256 subfields and 4,096 topics, clustered from the"
        " embeddings and named by Gemma 4, with every released paper placed in it.",
        f"Rebuilt from arXiv's weekly metadata snapshot every two weeks, by the open pipeline at"
        f" {REPO}.",
        "Every stage, from fetching and embedding to clustering and labeling, runs on one"
        " machine with one RTX 3090.",
        "Matched from the reference entry by the first that works: an arXiv link, an arXiv id in"
        " its text, a DOI, a journal reference, then its title, exact or fuzzy.",
        "Labeled by GraphIV's citation labeler, a DistilBERT student of Gemma 4 26B that agrees"
        " with its teacher at 0.815 macro F1.",
        "Scored by GraphIV's survey labeler, a DistilBERT student of Gemma 4 26B, at 0.920"
        " average precision on held-out papers. A review scores 0.5 or more.",
    ]
    page.save(out, 11.8, notes)


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
