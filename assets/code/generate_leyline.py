"""
Draws leyline.png: how LeyLine walks, the Unseen Newcomers its successors reach, and its lineages
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import example_run
import leyline_scores
import lineage_chart
import numpy as np
import overview
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
from style import (
    GRAY,
    GREEN,
    LEFT,
    MUTED,
    PURPLE,
    RIGHT,
    ROOT,
    STANCE_COLORS,
    Page,
    apply_theme,
    tint,
    write_runs,
)

from src.core.config import load_config

# the strongest magnets the magnet panel names, beside the single most cited paper
NAMED = 3
# every term, its formula and what it computes
TERMS = [
    (
        "magnet",
        r"$m(p) = \frac{1}{Z}\,\mathrm{stance}(p) \cdot"
        r" \frac{\mathrm{stance}(p)}{\mathrm{total}(p)} \cdot"
        r" \left(\frac{s(p)}{\max\, s}\right)^{\phi}$",
        "weighted sum of the argument into $p$, times its share and the closeness of $p$ to the"
        " question",
    ),
    (
        "reach",
        r"$\rho(v) = \Sigma_u\, w(u)\, \lambda^{h(v, u)}$",
        "sum of the magnet weights after $v$, decayed by $\\lambda$ per hop",
    ),
    (
        "step",
        r"$\mathrm{argmax}_{u\ \mathrm{cites}\ v}\ U_u^{1/\rho(u)}$",
        "a citer of $v$ sampled in proportion to its reach, with $U_u$ uniform on $(0,1)$",
    ),
    (
        "knockdown",
        r"$w(v) \leftarrow \kappa\, w(v)$",
        "every magnet a walk collects, scaled by $\\kappa$ after the walk",
    ),
    (
        "rank",
        r"$\mathrm{runs}(\pi) \cdot \Sigma_{v \in \pi}\, m(v)$",
        "runs of a path times the magnet weight it collects",
    ),
    (
        "successor",
        r"$\mathrm{pull}(c) = \mathrm{reach}(c) \cdot \cos(q, c)$",
        "list papers $c$ extends, critiques or compares against, times its cosine to the question",
    ),
]
FACTORS = [
    r"$\mathrm{stance}$ is 0 for a paper cited only as background.",
    r"$\mathrm{stance}/\mathrm{total}$ is small for a hub most papers cite as background.",
    r"$(s(p)/\max\,s)^{\phi}$ is small for a paper far from the question.",
]
# a toy typed graph: each paper's place and magnet weight, then each citation and its intent
PAPERS = {
    "a": (0.5, 1.55, 0.05),
    "root": (1.15, 1.0, 0.9),
    "c": (1.6, 1.95, 0.1),
    "b": (1.95, 0.5, 0.45),
    "d": (2.65, 1.25, 1.0),
    "e": (3.05, 0.4, 0.1),
    "f": (3.35, 1.95, 0.55),
    "g": (3.95, 1.05, 0.35),
}
CITATIONS = [
    ("root", "a", "background"),
    ("b", "root", "extends"),
    ("c", "root", "uses"),
    ("d", "b", "extends"),
    ("d", "root", "critiques"),
    ("e", "b", "compares"),
    ("f", "d", "extends"),
    ("f", "c", "background"),
    ("g", "f", "compares"),
    ("g", "d", "critiques"),
]
WALK = ("root", "b", "d", "f", "g")
# the successor sits past the graph, citing two of its papers
SUCCESSOR = (4.85, 2.05, (("g", "extends"), ("f", "critiques")))


def radius(weight: float) -> float:
    """
    A toy paper's radius, its area growing with its magnet weight

    Args:
        weight: The magnet weight, on [0, 1].

    Returns:
        The radius in inches.
    """
    return 0.08 + 0.16 * math.sqrt(weight)


def toy(page: Page, left: float, top: float) -> None:
    """
    A toy typed graph, its magnets sized by weight, with one walk from the root drawn under it

    Args:
        page: The page.
        left: Left edge.
        top: Top edge.
    """
    page.ax.add_patch(
        FancyBboxPatch(
            (left + 0.05, top),
            4.35,
            2.45,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor="white",
            edgecolor=PURPLE,
            linewidth=1.3,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    )
    page.ax.text(
        left + 0.22, top + 0.32, "typed graph", fontsize=10.5, color=PURPLE, fontweight="semibold"
    )
    at = {name: (left + x, top + y) for name, (x, y, _) in PAPERS.items()}
    size = {name: radius(m) for name, (_, _, m) in PAPERS.items()}
    page.ax.plot(
        [at[n][0] for n in WALK],
        [at[n][1] for n in WALK],
        color=tint(PURPLE, 0.3),
        linewidth=11,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )
    for citing, cited, intent in CITATIONS:
        color = STANCE_COLORS.get(intent, tint(GRAY, 0.75))
        overview.cite(page, at[citing], at[cited], color, (size[citing], size[cited]))
    for name, (_, _, m) in PAPERS.items():
        overview.node(page, at[name], tint(PURPLE, 0.12 + 0.88 * m), PURPLE, size[name])
    overview.label(page, at["root"], "root", below=False)
    x, y, cites = SUCCESSOR
    where = (left + x, top + y)
    overview.node(page, where, "white", PURPLE, 0.12, dashed=True)
    overview.label(page, where, "successor")
    for cited, intent in cites:
        overview.cite(page, where, at[cited], STANCE_COLORS[intent], (0.12, size[cited]), 0.0, True)
    overview.timeline(page, left + 0.1, left + 4.75, top + 2.7)


def newcomers(
    page: Page, x: float, top: float, width: float, topics: list[dict[str, Any]]
) -> float:
    """
    Each topic's Unseen Newcomers as a bar in a ruled table, with the share the successors reached

    Args:
        page: The page.
        x: Left edge.
        top: Top edge.
        width: Width.
        topics: Each topic's counts.

    Returns:
        The table's bottom edge.
    """
    header, step = 0.36, 0.42
    first = x + max(page.measure(t["topic"], 10.5) for t in topics) + 0.2
    second = first + 1.2
    bars, middle = second + 0.12, (first + second) / 2
    span = x + width - bars - 0.95
    most = max(t["newcomers"] for t in topics)
    for text, at, align in (
        ("Topic", x, "left"),
        ("BoundedField", middle, "center"),
        ("Successors", bars, "left"),
    ):
        page.ax.text(
            at, top + header / 2, text, fontsize=10.5, fontweight="semibold", ha=align, va="center"
        )
    for i, t in enumerate(topics):
        y = top + header + step * (i + 0.5)
        page.ax.text(x, y, t["topic"], fontsize=10.5, va="center")
        page.ax.text(
            middle,
            y,
            str(t["graph"]),
            fontsize=11,
            va="center",
            ha="center",
            color=GREEN,
            fontweight="semibold",
        )
        layers = (
            (t["newcomers"], tint(GRAY, 0.35)),
            (t["successor_set"], tint(PURPLE, 0.45)),
            (t["successor_top"], PURPLE),
        )
        for count, color in layers:
            page.ax.add_patch(
                FancyBboxPatch(
                    (bars, y - 0.1),
                    max(span * count / most, 0.03),
                    0.2,
                    boxstyle="round,pad=0,rounding_size=0.03",
                    facecolor=color,
                    edgecolor="none",
                    zorder=3,
                )
            )
        label = f"{t['successor_set']} of {t['newcomers']}"
        page.ax.text(
            bars + span * t["newcomers"] / most + 0.08, y, label, fontsize=10.5, va="center"
        )
    bottom = top + header + step * len(topics)
    overview.grid(
        page,
        (x - 0.1, x + width),
        (top, top + header, bottom),
        [top + header + step * (i + 1) for i in range(len(topics) - 1)],
        [first - 0.1, second],
    )
    return bottom


def magnets(axis: Axes, page: Page, nodes: list[dict[str, Any]]) -> None:
    """
    Every paper the run mined, its magnet weight against the citations it holds across GraphIV

    Args:
        axis: The panel.
        page: The page, for measuring names.
        nodes: The run's papers, with their weight and citations.
    """
    held = [n for n in nodes if n["cited_by"] > 0]
    xs = np.array([n["cited_by"] for n in held], dtype=float)
    ys = np.array([n["magnet"] for n in held])
    axis.scatter(xs, ys, s=22, color=tint(PURPLE, 0.55), edgecolors="none", zorder=3)
    strongest = sorted(held, key=lambda n: -n["magnet"])[:NAMED]
    cited = max(held, key=lambda n: n["cited_by"])
    for n in strongest + ([cited] if cited not in strongest else []):
        name = page.fit(n["title"].split(":")[0], 1.6, 9.5)
        axis.scatter([n["cited_by"]], [n["magnet"]], s=30, color=PURPLE, zorder=4)
        axis.annotate(
            name,
            (n["cited_by"], n["magnet"]),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9.5,
        )
    axis.set_xscale("log")
    axis.set_ylim(-0.04, 1.08)
    axis.set_xlabel("citations across GraphIV")
    axis.set_ylabel("magnet weight")
    axis.tick_params(axis="y", labelcolor=MUTED, labelsize=10.5)
    axis.grid(axis="both")


def draw(config: Path, out: Path) -> None:
    """
    leyline.png: the overview, the Unseen Newcomers, the magnets and the lineages

    Args:
        config: The config the example run's cache falls back to.
        out: Where the figure is written.
    """
    cold, run = leyline_scores.load(ROOT), example_run.load(config)
    topics, setup = cold["topics"], cold["setup"]
    miner = load_config(config)["miner"]
    page = Page(20.9)
    page.title(
        [
            [("GraphIV's", True), ("LeyLine:", PURPLE)],
            [("Magnetic Monte Carlo Trajectory Miner", False)],
        ]
    )

    page.panel_title(LEFT, 2.35, "Overview", 1)
    key: list[tuple[str, bool | str]] = [
        ("extends", STANCE_COLORS["extends"]),
        ("·", MUTED),
        ("critiques", STANCE_COLORS["critiques"]),
        ("·", MUTED),
        ("compares", STANCE_COLORS["compares"]),
        ("·", MUTED),
        ("other", GRAY),
        ("·", MUTED),
        ("one walk", tint(PURPLE, 0.55)),
    ]
    page.key(RIGHT, 2.35, key)
    boxes = [
        ("Typed graph", f"{miner['WALK_PREFIX']} papers and their references"),
        ("Magnets", "papers weighted by argument"),
        ("Monte Carlo walks", f"{miner['WALKS']} walks for each of {miner['SEEDS']} seeds"),
        ("Lineages", "paths ranked by runs and weight"),
    ]
    below = overview.steps(page, 2.6, boxes, PURPLE)
    toy(page, LEFT, below + 0.3)
    right = 5.9
    last = overview.terms(page, right, below + 0.3, (1.05, 2.9, RIGHT - right - 3.95), TERMS)
    end = overview.meanings(
        page,
        max(last, below + 3.25) + 0.5,
        "Each factor of the magnet discounts one kind of paper:",
        FACTORS,
        PURPLE,
    )

    top = end + 0.75
    page.panel_title(LEFT, top, "Unseen Newcomers found by each ranking", 2)
    ranked: list[tuple[str, bool | str]] = [
        ("BoundedField's top 20", GREEN),
        ("·", MUTED),
        ("successor top 20", PURPLE),
        ("·", MUTED),
        ("successor set", tint(PURPLE, 0.45)),
        ("·", MUTED),
        ("all Unseen Newcomers", GRAY),
    ]
    write_runs(page.fig, *page.at(LEFT, top + 0.37), ranked, 10.5, "semibold")
    newcomers(page, LEFT, top + 0.6, 6.7, topics)
    page.panel_title(7.6, top, "Magnet weight against citations", 3)
    magnets(page.place(8.4, top + 0.3, RIGHT - 8.4, 2.6), page, run["nodes"])

    top += 4.1
    page.panel_title(LEFT, top, "Lineages and successors for one question", 4)
    page.key(RIGHT, top, lineage_chart.key(run))
    ys = lineage_chart.lanes(page, LEFT + 0.1, RIGHT - 0.7, top + 1.4, run, gap=0.52, scale=1.2)

    total = sum(t["newcomers"] for t in topics)
    best = min(t["best_rank"] for t in topics)
    named = run["question"].split(",")[0]
    weights = miner["weights"]
    notes = [
        "An arrow points from a paper to a paper it cites, colored by its intent, and a paper's"
        " area grows with its magnet weight. $\\mathrm{stance}(p)$ sums $W_i\\,s(c)$ over the"
        " citations into $p$ and $\\mathrm{total}(p)$ sums $s(c)$, where $s$ is the cosine to the"
        f" question floored at 0 and $W$ is {weights['extends']:g} for extends,"
        f" {weights['critiques']:g}"
        f" for critiques, {weights['compares']:g} for compares, {weights['uses']:g} for uses,"
        f" {weights['motivates']:g} for motivates and {weights['background']:g} for background."
        f" Every walk starts at the root, $\\mathrm{{argmax}}_v\\,\\rho(v)\\,w(v)$, with"
        f" $\\phi={miner['FOCUS']:g}$, $\\lambda={miner['DECAY']:g}$ and"
        f" $\\kappa={miner['KNOCKDOWN']:g}$.",
        f"We asked each of the {len(topics)} Topic Queries in docs/03_tools.md on"
        f" {setup['question date']}, with the citation graph running to"
        f" {setup['citation graph runs to'].split(',')[0]}. An Unseen Newcomer is a paper from the"
        " 365 days before the question with no citations on that date and 20 or more in the 18"
        f" months after. BoundedField's top 20 held none of the {total:,}, the"
        f" highest any reached being rank {best}. The successor set is every paper of that window"
        " that extends, critiques or compares against the answer, and needs no citations of its"
        " own.",
        f"The {len(run['nodes'])} papers mined for “{named}” on {run['asked']}, with each weight"
        " scaled to the strongest magnet.",
        f"The run's {len(run['lanes'])} lineages, each segment in the strongest intent between its"
        f" two papers, and the {len(run['successors'])} successors that reach the most answer"
        " papers, each on every lane it extends, critiques or compares against.",
    ]
    page.save(out, ys[-1] + 0.85, notes)


def main() -> None:
    """
    Reads the findings and the example run, then draws the figure to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    apply_theme()
    draw(Path(str(args.config_path)), Path(str(args.out)))


if __name__ == "__main__":
    main()
