"""
Draws poster.png, the landing page's figure: arXiv's fields, the pipeline, and its two hallmarks

BoundedField and LeyLine each zoom out of the pipeline step they run in, on example_run.py's run
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import benchmark_scores
import example_run
import field_map
import lineage_chart
import numpy as np
import release_counts
from matplotlib import patheffects
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import FancyBboxPatch, Polygon
from style import (
    GRAY,
    GREEN,
    LEAD,
    LEFT,
    MUTED,
    OUTLINE,
    PANEL,
    PURPLE,
    ROOT,
    STANCE_COLORS,
    Page,
    apply_theme,
    tint,
    write_runs,
)

# wider than a card, so the two hallmarks can be read beside the pipeline they come from
WIDE = 16.0
BOTTOM = 12.1
BODY = 2.75
LINES = [
    [
        ("BoundedField", GREEN),
        ("- an LLM-free graph-driven retrieval at twice the recall of embedding search", MUTED),
    ],
    [
        ("LeyLine", PURPLE),
        (
            "- magnetic Monte Carlo lineage and trajectory mining, and a cited review in under"
            " 3 minutes on a single 3090",
            MUTED,
        ),
    ],
]
# what each term of the score walks, in the words of src/retrieve/README.md
TERMS = {
    "cos": "cosine to the question",
    "rec": "recency",
    "anc": "cited by the core",
    "fro": "cites the core",
    "anc_decay": "decay back from the core",
    "fro_decay": "decay forward from the core",
    "cocite": "cited by the core's citers",
    "couple": "cites what the core cites",
    "anc2": "two steps back",
    "fro2": "two steps forward",
    "hub": "citations so far, divided out",
}
# the data card's own claim about how often the release is rebuilt
REBUILT = "2 weeks"


def mix(a: str, b: str, share: float = 0.5) -> str:
    """
    A blend of two colors

    Args:
        a: The first color.
        b: The second color.
        share: How much of the second, 0 for the first alone.

    Returns:
        The blend as hex.
    """
    first, second = to_rgb(a), to_rgb(b)
    red, green, blue = ((1 - share) * x + share * y for x, y in zip(first, second, strict=True))
    return to_hex((red, green, blue))


def card(
    page: Page, box: tuple[float, float, float, float], edge: str, fill: str, line: float
) -> None:
    """
    A rounded box drawn behind whatever text is placed in it

    Args:
        page: The page.
        box: Left, top, width and height.
        edge: The outline's color.
        fill: The fill.
        line: The outline's width.
    """
    x, top, w, h = box
    page.ax.add_patch(
        FancyBboxPatch(
            (x, top),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge,
            linewidth=line,
            zorder=3,
        )
    )


def zoom(page: Page, box: tuple[float, ...], panel: tuple[float, ...], color: str) -> None:
    """
    A translucent wedge from a pipeline step's right edge to the panel that zooms into it

    Args:
        page: The page.
        box: The step's left, top, width and height.
        panel: The panel's left, top, width and height.
        color: The step's color.
    """
    right, top, bottom = box[0] + box[2], box[1] + 0.08, box[1] + box[3] - 0.08
    left, near, far = panel[0], panel[1] + 0.08, panel[1] + panel[3] - 0.08
    corners = [(right, top), (left, near), (left, far), (right, bottom)]
    page.ax.add_patch(Polygon(corners, closed=True, facecolor=tint(color, 0.12), edgecolor="none"))
    for start, end in ((corners[0], corners[1]), (corners[3], corners[2])):
        page.ax.plot(*zip(start, end, strict=True), color=tint(color, 0.5), linewidth=1.0)


def heading(page: Page, x: float, top: float, name: str, color: str, what: str) -> None:
    """
    A hallmark's name in its color, then what it is in smaller muted type on the same baseline

    Args:
        page: The page.
        x: Left edge.
        top: Baseline.
        name: The hallmark.
        color: Its color.
        what: What it is.
    """
    right = write_runs(page.fig, *page.at(x, top), [(name, color)], 14, "semibold")
    write_runs(
        page.fig, right + 0.08 / page.wide, page.at(x, top)[1], [(what, MUTED)], 12, "semibold"
    )


def colors(layout: dict[str, Any]) -> dict[int, str]:
    """
    One color per field, a theme role or a blend of two per discipline, alternating two shades in it

    Args:
        layout: The field map.

    Returns:
        Field to color.
    """
    hues = [LEAD, GREEN, PURPLE, mix(LEAD, PURPLE), mix(PURPLE, GREEN)]
    field, xy = layout["field"], layout["xy"]
    middle = {f: xy[field == f].mean(axis=0) for f in range(len(layout["names"]))}
    out: dict[int, str] = {}
    for d in sorted(set(layout["discipline"].tolist())):
        members = [f for f in middle if layout["discipline"][f] == d]
        spread = np.stack([middle[f] for f in members])
        # neighbors alternate shades, ordered along the discipline's long axis
        axis = (
            np.linalg.svd(spread - spread.mean(axis=0))[2][0] if len(members) > 1 else np.zeros(2)
        )
        for i, f in enumerate(sorted(members, key=lambda f: float(middle[f] @ axis))):
            out[f] = hues[d] if i % 2 == 0 else mix(hues[d], "#FFFFFF", 0.4)
    return out


def fields(page: Page, x: float, top: float, side: float, layout: dict[str, Any]) -> None:
    """
    The sampled papers as dots in their discipline's colors, each field named at its densest spot

    Args:
        page: The page.
        x: Left edge.
        top: Top edge.
        side: The square's side.
        layout: The field map.
    """
    ax = page.place(x, top, side, side)
    ax.set_facecolor(PANEL)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    xy, field, names = layout["xy"], layout["field"], [str(n) for n in layout["names"]]
    hue = colors(layout)
    lo, hi = np.percentile(xy, 0.3, axis=0), np.percentile(xy, 99.7, axis=0)
    pad = (hi - lo) * 0.05
    extent = [lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1]]
    shuffle = np.random.default_rng(0).permutation(len(xy))
    dots = np.array([hue[f] for f in field.tolist()])[shuffle]
    ax.scatter(*xy[shuffle].T, s=0.16, c=dots, linewidths=0, alpha=0.5, rasterized=True)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    spots: dict[int, np.ndarray] = {}
    for f in range(len(names)):
        own = xy[field == f]
        grid, xe, ye = np.histogram2d(*own.T, bins=20, range=[extent[:2], extent[2:]])
        i, j = np.unravel_index(grid.argmax(), grid.shape)
        spots[f] = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
    per_inch = np.array([extent[1] - extent[0], extent[3] - extent[2]]) / side
    size = 7.5
    room = {
        f: np.array([page.measure(n, size, fontweight="semibold") + 0.06, 0.17]) * per_inch
        for f, n in enumerate(names)
    }
    floor, ceiling = np.array(extent[::2]), np.array(extent[1::2])
    # names are pushed apart vertically until none overlap, and kept inside the square
    for _ in range(500):
        moved = False
        for a, b in itertools.combinations(spots, 2):
            gap = spots[b] - spots[a]
            overlap = (room[a] + room[b]) / 2 - np.abs(gap)
            if (overlap > 0).all():
                step = np.array([0.0, overlap[1] / 2 + 1e-6]) * (1 if gap[1] >= 0 else -1)
                spots[a] -= step
                spots[b] += step
                moved = True
        for f in spots:
            spots[f] = np.clip(spots[f], floor + room[f] / 2, ceiling - room[f] / 2)
        if not moved:
            break
    for f, name in enumerate(names):
        ax.text(
            float(spots[f][0]),
            float(spots[f][1]),
            name,
            fontsize=size,
            ha="center",
            va="center",
            fontweight="semibold",
            path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")],
            zorder=5,
        )


def counts(page: Page, x: float, top: float, width: float, release: dict[str, Any]) -> None:
    """
    The release's headline counts, then a bar per intent

    Args:
        page: The page.
        x: Left edge.
        top: The first count's baseline.
        width: Width.
        release: The dataset figures' cached counts.
    """
    leaves = release["leaves"]
    by_intent: dict[str, int] = {}
    for _, intent, n in release["cites"]:
        by_intent[intent] = by_intent.get(intent, 0) + int(n)
    facts = [
        (
            f"{int(release['papers'][0][0]) / 1e6:.2f}M",
            "papers, each with its metadata and an embedding",
        ),
        (f"{sum(by_intent.values()) / 1e6:.1f}M", "citations between them, each labeled by intent"),
        (
            f"{len(leaves):,}",
            f"topics, under {len({r[0] for r in leaves})} fields"
            f" and {len({(r[0], r[1]) for r in leaves})} subfields",
        ),
        (REBUILT, "between rebuilds, from the arXiv snapshot up"),
    ]
    y = top
    for big, small in facts:
        page.ax.text(x, y, big, fontsize=19, fontweight="semibold", color=LEAD, va="baseline")
        page.ax.text(x, y + 0.27, small, fontsize=10.5, color=MUTED, va="baseline")
        y += 0.72
    page.panel_title(x, y + 0.05, "Citations by intent")
    biggest = max(by_intent.values())
    for i, intent in enumerate(sorted(by_intent, key=lambda k: -by_intent[k])):
        yy = y + 0.42 + 0.3 * i
        page.ax.text(x, yy, intent, fontsize=10, fontweight="semibold", va="baseline")
        span = (width - 1.85) * by_intent[intent] / biggest
        page.ax.add_patch(
            FancyBboxPatch(
                (x + 0.95, yy - 0.12),
                span,
                0.14,
                boxstyle="round,pad=0,rounding_size=0.03",
                facecolor=STANCE_COLORS.get(intent, tint(GRAY, 0.6)),
                edgecolor="none",
                zorder=3,
            )
        )
        label = f"{by_intent[intent] / 1e6:.1f}M"
        page.ax.text(x + 1.02 + span, yy, label, fontsize=9.5, va="baseline", color=MUTED)


def pipeline(
    page: Page, x: float, width: float, run: dict[str, Any], core: int
) -> dict[str, tuple[float, float, float, float]]:
    """
    The question, then each tool the agent calls with what it returned on this run

    Args:
        page: The page.
        x: Left edge.
        width: Width.
        run: The example run.
        core: How many papers the scorer's walks start from.

    Returns:
        Each tool's box, for the zooms to start from.
    """
    page.panel_title(x, BODY, "Automated Research Pipeline")
    lines = page.wrapped(f"“{run['question']}”", width - 0.4, 10.5)
    asked = 0.2 + 0.2 * len(lines)
    card(page, (x, BODY + 0.25, width, asked), OUTLINE, PANEL, 1.2)
    for i, line in enumerate(lines):
        page.ax.text(x + 0.2, BODY + 0.54 + 0.2 * i, line, fontsize=10.5, zorder=4)
    c, pools = run["counts"], [area["steps"] for area in run["bounded"]]
    steps = [
        (
            "refine",
            "",
            "",
            [
                "a model turns the question into one",
                f"this corpus can answer, tagged {run['mode']}",
            ],
        ),
        ("identify_areas", "", "", [f"{len(run['areas'])} areas, each searched on its own"]),
        (
            "retrieve",
            "BoundedField",
            GREEN,
            [
                f"{pools[0][0]:,} by cosine plus one citation hop,",
                f"{' and '.join(f'{p[-1]:,}' for p in pools)} papers ranked by the graph,",
                f"then a model grades {core} and keeps {run['areas'][0]['kept']}",
            ],
        ),
        (
            "mine",
            "LeyLine",
            PURPLE,
            [
                f"{c['union']} papers and {c['edges']:,} citations,",
                f"{c['magnets']} magnets and {c['lineages']} walked paths,",
                f"{len(run['lineages'])} lineages and {c['successors']} successors",
            ],
        ),
        (
            "read_paper",
            "",
            "",
            [
                f"{run['methods']['rows']} full texts into a method table,",
                f"{len(run['methods']['untried'])} method pairs nobody has tried",
            ],
        ),
        (
            "write_report",
            "",
            "",
            [
                f"{run['cited']} arXiv ids cited, each checked",
                "against the corpus before it is written",
            ],
        ),
    ]
    heights = [0.5 + 0.2 * len(detail) for *_, detail in steps]
    above = BODY + 0.25 + asked
    gap = (BOTTOM - above - sum(heights)) / len(steps)
    boxes: dict[str, tuple[float, float, float, float]] = {}
    for (name, alias, hue, detail), high in zip(steps, heights, strict=True):
        y = above + gap
        page.arrow((x + width / 2, above), (x + width / 2, y))
        box = (x, y, width, high)
        card(page, box, hue or OUTLINE, tint(hue, 0.1) if hue else "white", 1.8 if hue else 1.2)
        runs: list[tuple[str, bool | str]] = [(name, False)]
        if alias:
            runs += [("·", MUTED), (alias, hue)]
        write_runs(page.fig, *page.at(x + 0.2, y + 0.3), runs, 12, "semibold")
        for i, line in enumerate(detail):
            page.ax.text(x + 0.2, y + 0.52 + 0.2 * i, line, fontsize=10, color=MUTED, zorder=4)
        boxes[name] = box
        above = y + high
    return boxes


def bounded_field(
    page: Page,
    panel: tuple[float, float, float, float],
    run: dict[str, Any],
    scorer: dict[str, Any],
    scores: dict[str, dict[str, dict[str, float]]],
) -> None:
    """
    BoundedField zoomed in: its score, this mode's exponents, its recall, and this question

    Args:
        page: The page.
        panel: Left, top, width and height.
        run: The example run.
        scorer: The run's recipe, its core and its exponents.
        scores: Mean recall per benchmark, recipe and cut.
    """
    card(page, panel, GREEN, "white", 1.8)
    x, top, width, high = panel
    x0 = x + 0.25
    heading(page, x0, top + 0.42, "BoundedField", GREEN, "LLM-free graph-driven retrieval")
    formula = (
        r"$\mathrm{score}(p) = \prod_{t} \, (1 + t_p)^{w_t}"
        r" \;/\; (1 + \mathrm{hub}_p)^{w_{\mathrm{hub}}}$"
    )
    page.ax.text(x0, top + 0.92, formula, fontsize=13, zorder=4)
    note = (
        "each term on [0, 1], the graph terms walked from the core,"
        f" the top {scorer['core']} by cosine"
    )
    page.ax.text(x0, top + 1.32, note, fontsize=9.5, color=MUTED, zorder=4)
    weights = scorer["weights"]
    terms = [t for t in weights if t != "hub"] + ["hub"]
    left, span, most = x0 + 2.65, 0.8, max(weights.values())
    page.ax.text(left, top + 1.62, f"{run['mode']} exponents", fontsize=9.5, color=MUTED, zorder=4)
    for i, term in enumerate(terms):
        y = top + 1.9 + 0.255 * i
        page.ax.text(x0, y, term.replace("_", " "), fontsize=10, fontweight="semibold", zorder=4)
        page.ax.text(x0 + 0.85, y, TERMS[term], fontsize=9.5, color=MUTED, zorder=4)
        length = span * weights[term] / most
        page.ax.add_patch(
            FancyBboxPatch(
                (left, y - 0.12),
                length,
                0.13,
                boxstyle="round,pad=0,rounding_size=0.03",
                facecolor=GREEN if term != "hub" else tint(GREEN, 0.45),
                edgecolor="none",
                zorder=4,
            )
        )
        page.ax.text(left + length + 0.06, y, f"{weights[term]:g}", fontsize=9.5, zorder=4)
    right = x + 4.3
    room = width - 4.55
    page.ax.text(
        right, top + 1.62, "recall, cosine to BoundedField", fontsize=9.5, color=MUTED, zorder=4
    )
    for i, (bench, (recipe, cut)) in enumerate(benchmark_scores.HEADLINE.items()):
        y = top + 1.9 + 0.6 * i
        low = scores[bench][benchmark_scores.BASELINE][cut]
        best = scores[bench][recipe][cut]
        page.ax.text(right, y, f"{bench} @{cut}", fontsize=10, fontweight="semibold", zorder=4)
        y += 0.22
        page.ax.plot([right, right + room], [y, y], color=OUTLINE, linewidth=1, zorder=3)
        page.ax.plot(
            [right + room * low, right + room * best], [y, y], color=GREEN, linewidth=2.4, zorder=4
        )
        page.ax.scatter([right + room * low], [y], s=30, color=GRAY, zorder=5)
        page.ax.scatter([right + room * best], [y], s=38, color=GREEN, zorder=5)
        for value, color, weight in ((low, MUTED, "regular"), (best, GREEN, "semibold")):
            page.ax.text(
                right + room * value,
                y + 0.2,
                f"{value:.3f}",
                fontsize=9,
                color=color,
                ha="center",
                fontweight=weight,
                zorder=5,
            )
    page.ax.text(
        right, top + 3.68, "each with its own tuned exponents", fontsize=8.5, color=MUTED, zorder=4
    )
    ranks = [r for area in run["bounded"] for r in area["kept"]]
    outside = sum(1 for r in ranks if r is None or r >= 1000)
    never = sum(1 for r in ranks if r is None)
    pool = run["bounded"][0]["steps"][0]
    first: list[tuple[str, bool | str]] = [
        ("On this question,", MUTED),
        (f"{outside} of the {len(ranks)}", GREEN),
        ("papers kept sat outside cosine's top 1,000,", MUTED),
    ]
    second: list[tuple[str, bool | str]] = [
        ("and", MUTED),
        (f"{never}", GREEN),
        (f"were never among its {pool:,} at all", MUTED),
    ]
    write_runs(page.fig, *page.at(x0, top + high - 0.5), first, 10.5, "medium")
    write_runs(page.fig, *page.at(x0, top + high - 0.25), second, 10.5, "medium")


def leyline(
    page: Page, panel: tuple[float, float, float, float], run: dict[str, Any], miner: dict[str, Any]
) -> None:
    """
    LeyLine zoomed in: what a magnet is, how the walks run, and this run's lineages and successors

    Args:
        page: The page.
        panel: Left, top, width and height.
        run: The example run.
        miner: The recipe's [miner] block.
    """
    card(page, panel, PURPLE, "white", 1.8)
    x, top, width, high = panel
    x0 = x + 0.25
    heading(
        page,
        x0,
        top + 0.42,
        "LeyLine",
        PURPLE,
        "magnetic Monte Carlo lineage and trajectory mining",
    )
    about = (
        "A magnet's weight is the weighted sum of the argument into a paper, times its share and"
        " the paper's closeness to the question. LeyLine runs"
        f" {miner['WALKS']} walks for each of {miner['SEEDS']} seeds, sampling every step in"
        " proportion to the magnet weight ahead, and ranks the paths by their runs times the"
        " magnet weight they collect."
    )
    for i, line in enumerate(page.wrapped(about, width - 0.5, 9.5)):
        page.ax.text(x0, top + 0.75 + 0.19 * i, line, fontsize=9.5, color=MUTED, zorder=4)
    lineage_chart.lanes(page, x0 + 0.1, x + width - 0.6, top + 2.5, run)
    write_runs(page.fig, *page.at(x0, top + high - 0.22), lineage_chart.key(run), 9.5, "semibold")


def draw(config: Path, out: Path) -> None:
    """
    The whole poster, left to right: the fields, the pipeline, then the two zooms

    Args:
        config: The config every cache falls back to.
        out: Where the figure is written.
    """
    layout, run, release = (
        field_map.load(config),
        example_run.load(config),
        release_counts.load(config),
    )
    recipe = benchmark_scores.exponents(config, run["mode"])
    scores = benchmark_scores.recall(config)
    page = Page(BOTTOM + 0.45, WIDE)
    title: list[tuple[str, bool | str]] = [
        ("GraphIV:", True),
        ("a rich, open source, and updating citation graph of arXiv", False),
    ]
    write_runs(page.fig, *page.at(LEFT, 0.95), title, 24, "bold")
    page.ax.plot(
        [LEFT, page.right], [1.25, 1.25], color=OUTLINE, linewidth=1.4, solid_capstyle="butt"
    )
    for i, runs in enumerate(LINES):
        write_runs(page.fig, *page.at(LEFT, 1.72 + 0.37 * i), list(runs), 14, "semibold")

    side = 3.55
    page.panel_title(LEFT, BODY, "Identified research fields in arXiv")
    fields(page, LEFT, BODY + 0.25, side, layout)
    counts(page, LEFT, BODY + 0.25 + side + 0.55, side, release)

    middle, width = LEFT + side + 0.4, 3.9
    boxes = pipeline(page, middle, width, run, recipe["core"])

    right = middle + width + 0.65
    upper = (right, BODY - 0.25, page.right - right, 4.4)
    below = upper[1] + upper[3] + 0.3
    lower = (right, below, page.right - right, BOTTOM - below)
    zoom(page, boxes["retrieve"], upper, GREEN)
    zoom(page, boxes["mine"], lower, PURPLE)
    bounded_field(page, upper, run, recipe, scores)
    leyline(page, lower, run, recipe["miner"])
    page.save(out, page.high - 0.3, [])


def main() -> None:
    """
    Reads or computes every cache, then draws the poster to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    apply_theme()
    draw(Path(str(args.config_path)), Path(str(args.out)))


if __name__ == "__main__":
    main()
