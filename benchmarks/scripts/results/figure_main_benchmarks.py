"""
Figure main_benchmarks: the three proposed benchmarks side by side, each a schematic of one case and its counts

Counts come from data/cases and the day windows from the checkout config that mined them
The dots of each schematic are illustrative, drawn from a seeded generator
"""

import random
import statistics
import tomllib
from pathlib import Path

from common import COLORS, CONFIG, PATHS, cases, output, plt, style
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle

STEM = Path(__file__).stem
SPEC = CONFIG[STEM]
MINED = tomllib.loads(PATHS["ROOT_CONFIG"].read_text())["benchmarks"]
WIDTH, HEIGHT, GAP = 6.75, 1.86, 0.08


def tint(color: str, share: float) -> str:
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - v) * share):02X}" for v in rgb)


def counts(dataset: str) -> tuple[dict[str, int], float]:
    """
    Cases per split and the median answer size of one benchmark

    Args:
        dataset: The dataset directory under data/cases.

    Returns:
        The case count of each split and the median key size over all splits.
    """
    split = {s: len(cases(f"{dataset}_{s}")) for s in ("train", "val", "test")}
    sizes = [len(c["key"]) for s in split for c in cases(f"{dataset}_{s}").values()]
    return split, statistics.median(sizes)


def blurb(dataset: str) -> str:
    gap = MINED["GAP_DAYS"]
    return {
        "surveymatch": "One review. The answer is\nwhat it cites.",
        "frontiermatch": f"Two reviews at most {gap} days apart. The\nanswer is what the later one adds.",
        "consensusmatch": f"Three reviews within {gap} days. The\nanswer is what all three cite.",
    }[dataset]


def cutoffs(dataset: str) -> str:
    """The paper cutoffs of a benchmark's test split, as table_main_benchmarks scores it"""
    ks = CONFIG["table_main_benchmarks"]["proposed"]["columns"][f"{dataset}_test"]
    return "R@" + ", ".join(k.split("@")[1] for k in ks)


def lane(ax: Axes, x0: float, x1: float, y: float, refs: list[float], hits: set[float], color: str, label: str, end: float) -> None:
    """One review on a timeline: its references as dots, the answer ones in color, the review as a square"""
    ax.plot([x0, x1], [y, y], color=COLORS["RULE"], lw=0.6, zorder=1)
    for r in refs:
        hit = r in hits
        ax.add_patch(plt.Circle((x0 + r * (x1 - x0), y), 0.022 if hit else 0.017, fc=color if hit else COLORS["FAINT"], ec="none", zorder=2))
    s = 0.05
    ax.add_patch(Rectangle((x0 + end * (x1 - x0) - s / 2, y - s / 2), s, s, fc=COLORS["INK"], ec="none", zorder=3))
    ax.text(x0 - 0.05, y, label, fontsize=5.6, color=COLORS["MUTED"], ha="right", va="center")


def panel(ax: Axes, x: float, w: float, dataset: str, rng: random.Random) -> None:
    """
    Draws one benchmark's box: its title, its rule, a schematic case and its counts

    Args:
        ax: The full-figure axes.
        x: The box's left edge in inches.
        w: The box's width in inches.
        dataset: The dataset directory under data/cases.
        rng: The generator the schematic's positions come from.
    """
    name, color = SPEC["datasets"][dataset], SPEC["hues"][dataset]
    split, median = counts(dataset)
    top, h = HEIGHT - 0.01, HEIGHT - 0.02
    ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=tint(color, 0.06), ec=tint(color, 0.55), lw=0.6))
    ax.text(x + 0.09, top - 0.07, name, fontsize=7.2, color=color, weight="semibold", va="top")
    ax.text(x + 0.09, top - 0.25, blurb(dataset), fontsize=5.9, color=COLORS["INK"], va="top", linespacing=1.25)

    x0, x1 = x + 0.42, x + w - 0.12
    ends = {"surveymatch": [0.95], "frontiermatch": [0.84, 0.95], "consensusmatch": [0.85, 0.9, 0.95]}[dataset]
    labels = {1: ["review"], 2: ["earlier", "later"], 3: ["first", "second", "third"]}[len(ends)]
    pool = sorted(rng.uniform(0.05, 0.8) for _ in range(22))
    if dataset == "surveymatch":
        lanes = [(pool, set(pool))]
    elif dataset == "frontiermatch":
        early = [p for i, p in enumerate(pool) if i % 3]
        late = [p for i, p in enumerate(pool) if i % 2] + [rng.uniform(0.5, 0.92) for _ in range(9)]
        added = {p for p in late if p not in early and p > 0.4}
        lanes = [(early, set()), (late, added)]
        arrow = {"arrowstyle": "<->", "color": color, "lw": 0.6, "shrinkA": 0, "shrinkB": 0}
        ax.annotate("", (x0 + 0.4 * (x1 - x0), top - 1.0), (x0 + 0.84 * (x1 - x0), top - 1.0), arrowprops=arrow)
        window = f"{MINED['FRONTIER_WINDOW_DAYS']} days before the earlier"
        ax.text(x0 + 0.62 * (x1 - x0), top - 1.03, window, fontsize=5.2, color=color, ha="center", va="top")
    else:
        shared = pool[::4]
        lanes = []
        for _ in range(3):
            own = [rng.uniform(0.05, 0.85) for _ in range(9)]
            lanes.append((sorted(shared + own), set(shared)))
        for s in shared:
            ax.add_patch(
                Rectangle(
                    (x0 + s * (x1 - x0) - 0.018, top - 0.68 - 2 * 0.15 - 0.04), 0.036, 2 * 0.15 + 0.08, fc=tint(color, 0.2), ec="none", zorder=1.5
                )
            )
    for i, ((refs, hits), end, label) in enumerate(zip(lanes, ends, labels, strict=True)):
        lane(ax, x0, x1, top - 0.68 - i * 0.15, refs, hits, color, label, end)
    cap = x0 + ends[-1 if dataset != "consensusmatch" else 0] * (x1 - x0)
    ax.plot([cap, cap], [top - 0.55, top - 0.68 - (len(lanes) - 1) * 0.15 - 0.08], color=color, lw=0.6, ls=(0, (2, 1.5)))
    ax.text(cap, top - 0.53, "date cap", fontsize=5.2, color=color, ha="center", va="bottom")

    y = top - 1.2
    ax.plot([x + 0.09, x + w - 0.09], [y + 0.06, y + 0.06], color=COLORS["RULE"], lw=0.5)
    rows = [
        ("cases", f"{sum(split.values()):,}"),
        ("train / val / test", f"{split['train']:,} / {split['val']:,} / {split['test']:,}"),
        ("median answer", f"{median:.0f} papers"),
        ("scored at", cutoffs(dataset)),
    ]
    for k, (a, b) in enumerate(rows):
        ax.text(x + 0.09, y - k * 0.15, a, fontsize=5.8, color=COLORS["MUTED"], va="top")
        ax.text(x + 0.95, y - k * 0.15, b, fontsize=5.8, color=COLORS["INK"], va="top")


def main() -> None:
    style()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    w = (WIDTH - 2 * GAP) / 3
    rng = random.Random(SPEC["seed"])
    for i, dataset in enumerate(SPEC["datasets"]):
        panel(ax, i * (w + GAP), w, dataset, rng)
    fig.savefig(output(STEM).with_suffix(".pdf"))
    fig.savefig(output(STEM).with_suffix(".png"), dpi=200)


if __name__ == "__main__":
    main()
