"""
Draws dataset-topics.png, the topic tree from its 16 fields down to one subfield's 16 topics
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from matplotlib.patches import FancyBboxPatch, Polygon
from release_counts import load
from style import (
    GREEN,
    INK,
    LEAD,
    LEFT,
    MUTED,
    PURPLE,
    RIGHT,
    ROOT,
    Page,
    apply_theme,
    compact,
    tint,
)

# the field and subfield the tree opens, and the color of each level down that path
OPENED = ("Artificial Intelligence", "Reasoning and Alignment")
LEVEL_COLORS = (GREEN, PURPLE, LEAD)
TREE_HIGH = 7.2


def tree_column(
    page: Page,
    x: float,
    top: float,
    width: float,
    rows: list[tuple[str, int]],
    level: int,
    opens: float | None,
) -> None:
    """
    One level of the tree as a column of tiles, each shaded to its paper count

    The opened tile takes the next level's color and fans out to the column that lists its children

    Args:
        page: The page.
        x: Left edge.
        top: Top edge.
        width: Width.
        rows: (name, papers) per tile, the largest first.
        level: The tree level, 0 for fields, which picks the color and the opened tile.
        opens: The next column's left edge, or None for the last level.
    """
    ax = page.ax
    opened = OPENED[level] if opens is not None else None
    slot = TREE_HIGH / len(rows)
    biggest = max(n for _, n in rows)
    pad = 0.035
    wrap = opens is None
    for index, (name, n) in enumerate(rows):
        picked = name == opened
        color = LEVEL_COLORS[level + 1] if picked else LEVEL_COLORS[level]
        y0, high = top + index * slot + pad, slot - 2 * pad
        for share, fill in (
            (1.0, 0.17 if picked else 0.07),
            (n / biggest, 0.5 if picked else 0.22),
        ):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y0),
                    width * share,
                    high,
                    boxstyle="round,pad=0,rounding_size=0.06",
                    facecolor=tint(color, fill),
                    edgecolor=color if picked and share == 1.0 else "none",
                    linewidth=1.3,
                    zorder=2,
                )
            )
        center = top + (index + 0.5) * slot
        lines = page.wrapped(name, width - 0.6, 10) if wrap else [name]
        if len(lines) > 2:
            print(f"[dataset] {name} wraps onto {len(lines)} lines")
        for row, line in enumerate(lines):
            ax.text(
                x + 0.12,
                center + (row - (len(lines) - 1) / 2) * 0.158,
                line,
                va="center",
                fontsize=10 if wrap else 10.5,
                fontweight="semibold" if picked else "regular",
                zorder=4,
            )
        ax.text(
            x + width - 0.1,
            center,
            compact(n),
            va="center",
            ha="right",
            fontsize=10.5,
            color=INK if picked else MUTED,
            fontweight="semibold" if picked else "regular",
            zorder=4,
        )
        if picked and opens is not None:
            corners = [(x + width, y0), (opens, top + pad), (opens, top + TREE_HIGH - pad)]
            ax.add_patch(
                Polygon(
                    [*corners, (x + width, y0 + high)],
                    facecolor=tint(color, 0.14),
                    edgecolor="none",
                    zorder=1,
                )
            )


def draw(data: dict[str, Any], out: Path) -> None:
    """
    dataset-topics.png: the tree from its 16 fields down to one subfield's 16 topics

    Args:
        data: The cached counts.
        out: Where the figure is written.
    """
    page = Page(10.45)
    page.title([[("Topic Tree", False)]])
    page.marked.append((page.fig.texts[-1], 1))
    names = {(int(a), None if b is None else int(b)): str(n) for a, b, n in data["upper"]}
    fields: Counter[int] = Counter()
    subfields: Counter[tuple[int, int]] = Counter()
    for a, b, _, _, n in data["leaves"]:
        fields[int(a)] += int(n)
        subfields[(int(a), int(b))] += int(n)
    field = next(a for (a, b), n in names.items() if b is None and n == OPENED[0])
    sub = next(b for (a, b), n in names.items() if a == field and n == OPENED[1])
    levels = [
        [(names[(a, None)], n) for a, n in fields.items()],
        [(names[(a, b)], n) for (a, b), n in subfields.items() if a == field],
        [(str(t), int(n)) for a, b, _, t, n in data["leaves"] if (a, b) == (field, sub)],
    ]
    tree_top, gap = 2.0, 0.32
    # a tile holds its name, a space and its count, and the topics take what is left
    fixed = [
        max(
            page.measure(name, 10.5, fontweight="semibold") + page.measure(compact(n), 10.5)
            for name, n in rows
        )
        + 0.45
        for rows in levels[:2]
    ]
    widths = (*fixed, RIGHT - LEFT - sum(fixed) - 2 * gap)
    lefts = (LEFT, LEFT + widths[0] + gap, LEFT + widths[0] + widths[1] + 2 * gap)
    heads = ("16 fields", "16 subfields in each field", "16 topics in each subfield")
    for index, rows in enumerate(levels):
        rows.sort(key=lambda r: -r[1])
        last = index == len(levels) - 1
        tree_column(
            page,
            lefts[index],
            tree_top,
            widths[index],
            rows,
            index,
            None if last else lefts[index + 1],
        )
        page.ax.text(
            lefts[index],
            tree_top - 0.22,
            heads[index],
            fontsize=12,
            fontweight="semibold",
            color=LEVEL_COLORS[index],
            va="baseline",
        )
    notes = [
        "Clustered from the paper embeddings and named by Gemma 4, with every released paper"
        " placed in one topic. Each count is papers.",
    ]
    page.save(out, tree_top + TREE_HIGH + 0.6, notes)


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
