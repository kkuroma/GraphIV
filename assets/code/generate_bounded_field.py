"""
Draws bounded-field.png: how BoundedField scores, its recall against embedding search, its recipes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_scores
import overview
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
from style import (
    GRAY,
    GREEN,
    LEFT,
    MUTED,
    RIGHT,
    ROOT,
    Page,
    apply_theme,
    tint,
)

MODES = ("survey", "frontier", "consensus")
CUTS = ("100", "200", "500", "1000")
TERMS = ("cos", "rec", "anc", "fro", "anc_decay", "cocite", "hub")
# how far above the first row of the ablation and recipe grids their header starts
HEAD = 1.25
# what each tuned recipe is meant to put first
PURPOSE = {
    "survey": "prioritizes what a literature review on the topic would cite",
    "frontier": "prioritizes recent, frontier papers on a topic",
    "consensus": "prioritizes the papers every review of a topic agrees on",
}
# the pool's steps, each with what it adds
STEPS = [
    ("Initial retrieval", "the 5,000 closest papers by cosine"),
    ("Ancestor hop", "adds the papers the core cites"),
    ("Frontier hop", "adds the papers citing the core"),
    ("Graph-aware scoring", "multiplies every term into one score"),
]
SCORE = (
    r"$\mathrm{score}(p) = \dfrac{\prod_{\tau \in T}\, (1 + x_\tau(p))^{w_\tau}}"
    r"{(1 + x_{\mathrm{hub}}(p))^{w_{\mathrm{hub}}}}$"
)
# every term, its formula and what it computes
WALKS = [
    ("cos", r"$\cos(q, p)$", "cosine similarity of $p$ to the question"),
    ("rec", r"$1 / (1 + \mathrm{age})$", "recency of $p$, from its age in years"),
    ("anc", r"$A c$", "core papers citing $p$, weighted by their cosine"),
    ("fro", r"$A^{\mathrm{T}} c$", "core papers $p$ cites, weighted by their cosine"),
    ("anc2, fro2", r"$A^2 c, \ (A^{\mathrm{T}})^2 c$", "the same, two hops from the core"),
    ("cocite", r"$(A A^{\mathrm{T}} c) \odot r$", "co-citation with the core, times recency"),
    (
        "couple",
        r"$(A^{\mathrm{T}} A c) \odot r$",
        "bibliographic coupling with the core, times recency",
    ),
    ("anc_decay, fro_decay", r"$\delta^{d(p)}$", "decay by the hops between $p$ and the core"),
    ("hub", r"citations at $t$", "citations $p$ holds across GraphIV at $t$"),
]
PROPERTIES = [
    "A term at 0 gives a factor of 1, so it never zeroes a score.",
    "An exponent of 0 also gives a factor of 1, which removes the term.",
    r"A term changes a score by at most a factor of $2^{w}$.",
]


def toy(page: Page, left: float, top: float) -> None:
    """
    The pool as a toy graph around one core, each paper marked with the walk that reaches it

    Args:
        page: The page.
        left: Left edge.
        top: Top edge.
    """
    page.ax.add_patch(
        FancyBboxPatch(
            (left + 0.05, top),
            4.75,
            2.45,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor="white",
            edgecolor=GREEN,
            linewidth=1.3,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    )
    page.ax.text(left + 0.22, top + 0.32, "pool", fontsize=10.5, color=GREEN, fontweight="semibold")
    row = top + 1.2
    at = {
        "anc2": (left + 0.55, row),
        "anc": (left + 1.5, row),
        "core": (left + 2.55, row),
        "fro": (left + 3.6, row),
        "fro2": (left + 4.4, row),
        "couple": (left + 2.05, top + 0.4),
        "cocite": (left + 2.1, top + 1.95),
    }
    big = 0.3
    for name, xy in at.items():
        if name == "core":
            overview.node(page, xy, GREEN, GREEN, big)
            page.ax.text(
                *xy,
                "core",
                fontsize=10.5,
                color="white",
                fontweight="semibold",
                ha="center",
                va="center",
                zorder=6,
            )
            continue
        overview.node(page, xy, tint(GREEN, 0.25), GREEN)
        overview.label(page, xy, name, below=name != "couple")
    page.ax.text(
        at["core"][0], row + 0.53, "200 closest", fontsize=9.5, color=MUTED, ha="center", zorder=6
    )
    overview.cite(page, at["core"], at["anc"], radii=(big, overview.NODE))
    overview.cite(page, at["anc"], at["anc2"])
    overview.cite(page, at["fro"], at["core"], radii=(overview.NODE, big))
    overview.cite(page, at["fro2"], at["fro"])
    overview.cite(page, at["fro"], at["cocite"], bend=-0.25)
    overview.cite(page, at["couple"], at["anc"], bend=0.2)
    overview.timeline(page, left + 0.1, left + 4.75, top + 2.7)


def curves(
    axis: Axes, bench: str, scores: dict[str, dict[str, float]], tuned: str, cut: str, top: float
) -> None:
    """
    One benchmark's score at every K, BoundedField's own recipe against embedding search

    Args:
        axis: The panel.
        bench: The benchmark.
        scores: Recipe to K to the mean score.
        tuned: The recipe tuned on this benchmark.
        cut: The K the benchmark's headline is read at, which is annotated.
        top: BoundedField's ceiling.
    """
    xs = list(range(len(CUTS)))
    ours = [scores[tuned][k] for k in CUTS]
    cosine = [scores[benchmark_scores.BASELINE][k] for k in CUTS]
    axis.fill_between(xs, cosine, ours, color=tint(GREEN, 0.14), linewidth=0, zorder=1)
    if top:
        axis.axhline(top, color=MUTED, linewidth=1.1, linestyle=(0, (1.5, 2)), zorder=2)
        axis.text(-0.15, top + 0.025, "ceiling", fontsize=9.5, color=MUTED)
    axis.plot(xs, cosine, color=GRAY, linewidth=1.8, marker="o", markersize=4, zorder=3)
    axis.plot(xs, ours, color=GREEN, linewidth=2.6, marker="o", markersize=4.5, zorder=4)
    i = CUTS.index(cut)
    high, low = scores[tuned][cut], scores[benchmark_scores.BASELINE][cut]
    axis.annotate(
        f"{high:.3f}",
        (i, high),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=10.5,
        color=GREEN,
        fontweight="semibold",
    )
    axis.text(
        i + (0.12 if i == 0 else -0.12),
        (high + low) / 2,
        f"{high / low:.1f}x",
        ha="left" if i == 0 else "right",
        va="center",
        fontsize=12,
        color=GREEN,
        fontweight="bold",
    )
    axis.set_title(bench, fontsize=12.5)
    axis.set_xticks(xs, list(CUTS))
    axis.set_xlim(-0.25, len(CUTS) - 0.75)
    axis.set_ylim(0, 1.08)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.tick_params(axis="y", labelcolor=MUTED, labelsize=10.5)
    axis.grid(axis="both")


def ablation(
    page: Page, x: float, top: float, lost: dict[str, dict[str, float]], tuned: dict[str, float]
) -> None:
    """
    The recall each benchmark lost when one term was dropped from its own tuned recipe

    Args:
        page: The page.
        x: Left edge.
        top: The first row's middle.
        lost: Benchmark to term to the recall lost.
        tuned: Benchmark to its tuned recall at the headline K.
    """
    gutter, step, span = 1.05, 1.6, 0.95
    rows = 0.3
    most = max(v for terms in lost.values() for v in terms.values())
    columns = [x + gutter - 0.12 + step * j for j in range(len(benchmark_scores.HEADLINE))]
    bottom = top + rows * (len(TERMS) - 0.5)
    overview.grid(
        page,
        (x - 0.1, columns[-1] + step),
        (top - HEAD, top - rows / 2, bottom),
        [top + rows * (i + 0.5) for i in range(len(TERMS) - 1)],
        columns,
    )
    page.ax.text(x, top - 0.4, "Term", fontsize=11, fontweight="semibold")
    for j, (bench, (_, cut)) in enumerate(benchmark_scores.HEADLINE.items()):
        column = x + gutter + step * j
        page.ax.text(column, top - 0.62, bench, fontsize=11, fontweight="semibold")
        page.ax.text(
            column, top - 0.4, f"of {tuned[bench]:.3f} at {int(cut):,}", fontsize=9.5, color=MUTED
        )
    for i, term in enumerate(TERMS):
        y = top + rows * i
        page.ax.text(x, y, term.replace("_", " "), fontsize=11, va="center", fontweight="semibold")
        for j, bench in enumerate(benchmark_scores.HEADLINE):
            column = x + gutter + step * j
            if term not in lost[bench]:
                page.ax.text(column, y, "off", fontsize=10, va="center", color=MUTED)
                continue
            length = max(span * lost[bench][term] / most, 0.02)
            page.ax.add_patch(
                FancyBboxPatch(
                    (column, y - 0.075),
                    length,
                    0.15,
                    boxstyle="round,pad=0,rounding_size=0.03",
                    facecolor=GREEN,
                    edgecolor="none",
                    zorder=3,
                )
            )
            page.ax.text(
                column + length + 0.07, y, f"{lost[bench][term]:.3f}", fontsize=10, va="center"
            )


def recipes(page: Page, x: float, top: float, config: Path) -> None:
    """
    Each tuned recipe with what it is meant for and an exponent per term, a disc growing with it

    Args:
        page: The page.
        x: Left edge.
        top: The first row's middle.
        config: The config naming the recipes.
    """
    tuned = {mode: benchmark_scores.exponents(config, mode)["weights"] for mode in MODES}
    most = max(w for weights in tuned.values() for w in weights.values())
    gutter, step = 1.05, 1.47
    columns = [x + gutter - 0.12 + step * j for j in range(len(MODES))]
    overview.grid(
        page,
        (x - 0.1, RIGHT),
        (top - HEAD, top - 0.15, top + 0.3 * (len(TERMS) - 0.5)),
        [top + 0.3 * (i + 0.5) for i in range(len(TERMS) - 1)],
        columns,
    )
    page.ax.text(x, top - 0.4, "Term", fontsize=11, fontweight="semibold")
    for j, mode in enumerate(MODES):
        column = x + gutter + step * j
        lines = page.wrapped(PURPOSE[mode], step - 0.15, 9.5)
        head = top - 0.4 - 0.19 * len(lines)
        page.ax.text(column, head - 0.03, mode, fontsize=11.5, color=GREEN, fontweight="semibold")
        for k, line in enumerate(lines):
            page.ax.text(column, head + 0.19 * (k + 1), line, fontsize=9.5, color=MUTED)
    for i, term in enumerate(TERMS):
        y = top + 0.3 * i
        page.ax.text(x, y, term.replace("_", " "), fontsize=11, va="center", fontweight="semibold")
        for j, mode in enumerate(MODES):
            weight = tuned[mode].get(term, 0.0)
            column = x + gutter + step * j
            if weight == 0:
                page.ax.text(column + 0.1, y, "0", fontsize=10, va="center", color=MUTED)
                continue
            color = GREEN if term != "hub" else tint(GREEN, 0.45)
            page.ax.scatter([column + 0.1], [y], s=230 * weight / most, color=color, zorder=3)
            page.ax.text(column + 0.33, y, f"{weight:g}", fontsize=10.5, va="center")


def draw(config: Path, out: Path) -> None:
    """
    bounded-field.png: the overview, the recall panels, the ablation and the tuned recipes

    Args:
        config: The config naming the benchmarks' output and the recipes.
        out: Where the figure is written.
    """
    scores = benchmark_scores.recall(config)
    counted = benchmark_scores.cases(config)
    _, agree = benchmark_scores.agreement(ROOT)
    lost = benchmark_scores.dropped(ROOT)
    page = Page(20.7)
    page.title(
        [
            [("GraphIV's", True), ("BoundedField:", GREEN)],
            [("Citation-Aware, LLM-free Retriever and Reranker", False)],
        ]
    )

    page.panel_title(LEFT, 2.35, "Overview", 1)
    below = overview.steps(page, 2.6, STEPS, GREEN)
    toy(page, LEFT, below + 0.3)
    right = 5.6
    page.ax.text((right + RIGHT) / 2, below + 0.7, SCORE, fontsize=17, ha="center", va="center")
    last = overview.terms(page, right, below + 1.3, (1.6, 1.35, RIGHT - right - 2.95), WALKS)
    end = overview.meanings(
        page,
        max(last, below + 3.25) + 0.5,
        r"Every term is normalized to $[0, 1]$, so each factor lies on $[1, 2^{w}]$:",
        PROPERTIES,
        GREEN,
    )

    top = end + 0.75
    page.panel_title(LEFT, top, "Recall at top K against embedding search", 2)
    key: list[tuple[str, bool | str]] = [
        ("BoundedField", GREEN),
        ("·", MUTED),
        ("embedding search", GRAY),
        ("·", MUTED),
        ("ceiling", MUTED),
    ]
    page.key(RIGHT, top, key)
    benches = list(benchmark_scores.HEADLINE)
    gap = 0.45
    each = (RIGHT - LEFT - 0.35 - gap * (len(benches) - 1)) / len(benches)
    for i, bench in enumerate(benches):
        axis = page.place(LEFT + 0.35 + i * (each + gap), top + 0.65, each, 2.1)
        tuned, cut = benchmark_scores.HEADLINE[bench]
        curves(
            axis, bench, scores[bench], tuned, cut, benchmark_scores.ceiling(config, bench, tuned)
        )
        if i == 0:
            axis.set_ylabel("recall")
        else:
            axis.tick_params(axis="y", labelleft=False)
    page.ax.text(
        LEFT + 0.35 + (RIGHT - LEFT - 0.35) / 2,
        top + 3.25,
        "top K",
        ha="center",
        fontsize=11.5,
        color=MUTED,
    )

    top += 3.85
    page.panel_title(LEFT, top, "Ablation of each term", 3)
    page.panel_title(6.55, top, "Tuned Recipes for BoundedField", 4)
    rows = top + 1.85
    tuned = {b: scores[b][r][c] for b, (r, c) in benchmark_scores.HEADLINE.items()}
    page.ax.text(
        6.55,
        top + 0.4,
        "the exponent of each term, per recipe",
        fontsize=10.5,
        color=MUTED,
    )
    page.ax.text(
        LEFT,
        top + 0.4,
        "recall lost when one term's exponent is set to 0",
        fontsize=10.5,
        color=MUTED,
    )
    ablation(page, LEFT, rows, lost, tuned)
    recipes(page, 6.55, rows, config)
    after = rows + 0.3 * (len(TERMS) - 0.5) + 0.35
    survey = scores["SurveyMatch"]
    hop = survey["survey"]["1000"] - survey["no_expansion"]["1000"]
    page.ax.text(
        LEFT,
        after,
        f"Dropping both citation hops instead loses SurveyMatch {hop:.3f}.",
        fontsize=10,
        color=MUTED,
    )
    page.ax.text(
        6.55,
        after,
        "ancestry is survey with fro at 0, for what led to a paper",
        fontsize=10,
        color=MUTED,
    )
    page.ax.text(
        6.55,
        after + 0.22,
        "evolution is frontier with anc and anc decay at 0, for what came of a paper",
        fontsize=10,
        color=MUTED,
    )

    survey_cases, frontier, consensus = (counted[b] for b in benches)
    notes = [
        "An arrow points from a paper to a paper it cites. $A$ is the citation graph on the pool,"
        " with $A_{ij}=1$ when $j$ cites $i$, $c$ is the core's cosine normalized to sum to 1, $r$"
        " is $1/(1+\\mathrm{age})$, $\\delta$ is 0.5 per hop and $t$ is the date the corpus is"
        " capped at.",
        f"Mean over {survey_cases} SurveyMatch, {frontier} FrontierMatch and {consensus}"
        f" ConsensusMatch cases built from arXiv's literature"
        " reviews, each capped at its review's date. Embedding search ranks the 5,000 closest"
        " papers by cosine alone, and the ceiling is the share of the ground truth the pool holds."
        f" Two reviews of the same topic find each other's references at recall {agree:.3f}.",
        "We set one exponent to 0 at its recipe's tuned point, keep the rest, and report the recall"
        " lost at the benchmark's headline K. A term already at 0 in that recipe is off.",
        "We tuned each recipe by coordinate descent on its own benchmark. A disc's area grows with"
        " the exponent, and hub divides the score rather than multiplying it. ancestry and"
        " evolution are not tuned.",
    ]
    page.save(out, after + 0.75, notes)


def main() -> None:
    """
    Reads the benchmarks' scores, then draws the figure to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    apply_theme()
    draw(Path(str(args.config_path)), Path(str(args.out)))


if __name__ == "__main__":
    main()
