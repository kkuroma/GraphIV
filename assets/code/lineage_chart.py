"""
A run's lineages as subway lanes in date order, with the successors past a dashed line

One paper several lineages share is one station, and a successor sits on every lane it argues with
"""

from __future__ import annotations

import itertools
from typing import Any

from style import GRAY, INK, MUTED, PURPLE, STANCE_COLORS, Page, tint


def lanes(
    page: Page,
    first: float,
    last: float,
    top: float,
    run: dict[str, Any],
    gap: float = 0.45,
    scale: float = 1.0,
) -> list[float]:
    """
    Draws the lanes, their stations, the successors and every paper's short name above them

    Args:
        page: The page.
        first: Where the oldest paper sits.
        last: Where the last successor sits.
        top: The first lane's height, with the names rotated above it.
        run: The example run, whose lanes, columns and successors are drawn.
        gap: The space between two lanes.
        scale: How much larger than the poster's the type and the marks are.

    Returns:
        Each lane's height.
    """
    lines, columns, ahead = run["lanes"], run["columns"], run["successors"]
    step = (last - first) / (len(columns) + len(ahead))
    where = {c["arxiv_id"]: first + i * step for i, c in enumerate(columns)}
    for i, s in enumerate(ahead):
        where[s["arxiv_id"]] = first + (len(columns) + 1 + i) * step
    ys = [top + gap * k for k in range(len(lines))]
    divider = first + (len(columns) + 0.5) * step
    page.ax.plot(
        [divider] * 2,
        [ys[0] - 0.2, ys[-1] + 0.15],
        color=MUTED,
        linewidth=1,
        linestyle=(0, (3, 2.5)),
        zorder=4,
    )
    for k, lane in enumerate(lines):
        page.ax.text(
            first - 0.08,
            ys[k] - 0.13 * scale,
            lane["name"],
            fontsize=8.5 * scale,
            color=PURPLE,
            fontweight="semibold",
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )
        for (a, b), intent in zip(itertools.pairwise(lane["papers"]), lane["intents"], strict=True):
            color = STANCE_COLORS.get(intent, tint(GRAY, 0.6))
            width = 2.6 * scale
            page.ax.plot([where[a], where[b]], [ys[k]] * 2, color=color, linewidth=width, zorder=4)
        stations = [where[a] for a in lane["papers"]]
        page.ax.scatter(
            stations,
            [ys[k]] * len(stations),
            s=26 * scale**2,
            facecolor="white",
            edgecolors=INK,
            linewidths=1.1 * scale,
            zorder=5,
        )
        for s in ahead:
            intent = s["answered"].get(str(k))
            if intent:
                page.ax.scatter(
                    [where[s["arxiv_id"]]],
                    [ys[k]],
                    s=34 * scale**2,
                    color=STANCE_COLORS[intent],
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=5,
                )
    # a paper on several lanes is one station, so its lanes are joined
    for c in columns:
        held = [k for k, lane in enumerate(lines) if c["arxiv_id"] in lane["papers"]]
        if len(held) > 1:
            page.ax.plot(
                [where[c["arxiv_id"]]] * 2,
                [ys[held[0]], ys[held[-1]]],
                color=INK,
                linewidth=1.0,
                zorder=4,
            )
    for item, color in [(c, INK) for c in columns] + [(s, PURPLE) for s in ahead]:
        page.ax.text(
            where[item["arxiv_id"]],
            top - 0.32 * scale,
            item["label"],
            fontsize=8 * scale,
            rotation=50,
            rotation_mode="anchor",
            color=color,
            zorder=5,
        )
    return ys


def key(run: dict[str, Any]) -> list[tuple[str, bool | str]]:
    """
    The colors a lane is drawn in, then how many of the run's successors are shown

    Args:
        run: The example run.

    Returns:
        The key's runs, for write_runs or Page.key.
    """
    out: list[tuple[str, bool | str]] = []
    for intent in ("extends", "critiques", "compares"):
        out += [(intent, STANCE_COLORS[intent]), ("·", MUTED)]
    shown = (
        f"past the line, {len(run['successors'])} of the {run['counts']['successors']} successors"
    )
    return [*out, ("other", GRAY), ("·", MUTED), (shown, PURPLE)]
