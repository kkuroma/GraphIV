"""
Figure 2: the three proposed benchmarks side by side, each a schematic of one case and its counts

Usage: direnv exec . uv run python paper/scripts/benchmarks.py
Counts come from the 2026-10-22 release's cases, the schematic positions are illustrative
"""

import json
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CASES = ROOT / "benchmarks" / "data" / "cases"
from matplotlib import font_manager

BIOLINUM = Path("/nix/store/5xhrl01vhivcmwvr3sdkz0ygdbsyd84l-texlive-2025-r78234-final-env-texmfdist/fonts/opentype/public/libertine")
for f in BIOLINUM.glob("LinBiolinum_R*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Linux Biolinum O"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "stix"

WIDTH, HEIGHT = 6.75, 1.86
GAP = 0.08
INK, MUTED, FAINT, RULE = "#2A1E18", "#60504A", "#ABA09B", "#D8C8C0"
HUES = {"SurveyMatch": "#902020", "FrontierMatch": "#1F74A8", "ConsensusMatch": "#806000"}
CUTS = {"SurveyMatch": "R@100, 200, 500", "FrontierMatch": "R@20, 50, 100", "ConsensusMatch": "R@20, 50, 100"}
TEXT = {
    "SurveyMatch": "One review. The answer is\nwhat it cites.",
    "FrontierMatch": "Two reviews at most 190 days apart. The\nanswer is what the later one adds.",
    "ConsensusMatch": "Three reviews within 190 days. The\nanswer is what all three cite.",
}


def tint(color: str, share: float) -> str:
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - v) * share):02X}" for v in rgb)


def counts() -> dict:
    """Cases per split and the median answer size of each benchmark, from the cases on disk"""
    out = {}
    for name in HUES:
        sizes, split = [], {}
        for s in ("train", "val", "test"):
            rows = [json.loads(l) for l in open(CASES / name.lower() / f"{s}.jsonl")]
            split[s] = len(rows)
            sizes += [len(r["key"]) for r in rows]
        out[name] = (split, statistics.median(sizes))
    return out


def lane(ax, x0, x1, y, refs, hits, color, label, end):
    """One review on a timeline: its references as dots, the answer ones in color, the review as a square"""
    ax.plot([x0, x1], [y, y], color=RULE, lw=0.6, zorder=1)
    for r in refs:
        hit = r in hits
        ax.add_patch(plt.Circle((x0 + r * (x1 - x0), y), 0.022 if hit else 0.017, fc=color if hit else FAINT, ec="none", zorder=2))
    s = 0.05
    ax.add_patch(Rectangle((x0 + end * (x1 - x0) - s / 2, y - s / 2), s, s, fc=INK, ec="none", zorder=3))
    ax.text(x0 - 0.05, y, label, fontsize=5.6, color=MUTED, ha="right", va="center")


def panel(ax, x, w, name, split, median, rng):
    color = HUES[name]
    top = HEIGHT - 0.01
    h = HEIGHT - 0.02
    ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=tint(color, 0.06), ec=tint(color, 0.55), lw=0.6))
    ax.text(x + 0.09, top - 0.07, name, fontsize=7.2, color=color, weight="semibold", va="top")
    ax.text(x + 0.09, top - 0.25, TEXT[name], fontsize=5.9, color=INK, va="top", linespacing=1.25)

    x0, x1 = x + 0.42, x + w - 0.12
    ends = {"SurveyMatch": [0.95], "FrontierMatch": [0.84, 0.95], "ConsensusMatch": [0.85, 0.9, 0.95]}[name]
    labels = {1: ["review"], 2: ["earlier", "later"], 3: ["first", "second", "third"]}[len(ends)]
    pool = sorted(rng.uniform(0.05, 0.8) for _ in range(22))
    lanes = []
    if name == "SurveyMatch":
        lanes = [(pool, set(pool))]
    elif name == "FrontierMatch":
        early = [p for i, p in enumerate(pool) if i % 3]
        late = [p for i, p in enumerate(pool) if i % 2] + [rng.uniform(0.5, 0.92) for _ in range(9)]
        added = {p for p in late if p not in early and p > 0.4}
        lanes = [(early, set()), (late, added)]
        ax.annotate("", (x0 + 0.4 * (x1 - x0), top - 1.0), (x0 + 0.84 * (x1 - x0), top - 1.0), arrowprops=dict(arrowstyle="<->", color=color, lw=0.6, shrinkA=0, shrinkB=0))
        ax.text(x0 + 0.62 * (x1 - x0), top - 1.03, "730 days before the earlier", fontsize=5.2, color=color, ha="center", va="top")
    else:
        shared = pool[::4]
        for k in range(3):
            own = [rng.uniform(0.05, 0.85) for _ in range(9)]
            lanes.append((sorted(shared + own), set(shared)))
        for s in shared:
            ax.add_patch(Rectangle((x0 + s * (x1 - x0) - 0.018, top - 0.68 - 2 * 0.15 - 0.04), 0.036, 2 * 0.15 + 0.08, fc=tint(color, 0.2), ec="none", zorder=1.5))
    for i, ((refs, hits), end, label) in enumerate(zip(lanes, ends, labels)):
        lane(ax, x0, x1, top - 0.68 - i * 0.15, refs, hits, color, label, end)
    cap = x0 + ends[-1 if name != "ConsensusMatch" else 0] * (x1 - x0)
    ax.plot([cap, cap], [top - 0.55, top - 0.68 - (len(lanes) - 1) * 0.15 - 0.08], color=color, lw=0.6, ls=(0, (2, 1.5)))
    ax.text(cap, top - 0.53, "date cap", fontsize=5.2, color=color, ha="center", va="bottom")

    y = top - 1.2
    ax.plot([x + 0.09, x + w - 0.09], [y + 0.06, y + 0.06], color=RULE, lw=0.5)
    total = sum(split.values())
    rows = [
        ("cases", f"{total:,}"),
        ("train / val / test", f"{split['train']:,} / {split['val']:,} / {split['test']:,}"),
        ("median answer", f"{median:.0f} papers"),
        ("scored at", CUTS[name]),
    ]
    for k, (a, b) in enumerate(rows):
        ax.text(x + 0.09, y - k * 0.15, a, fontsize=5.8, color=MUTED, va="top")
        ax.text(x + 0.95, y - k * 0.15, b, fontsize=5.8, color=INK, va="top")


def main():
    data = counts()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    w = (WIDTH - 2 * GAP) / 3
    rng = random.Random(7)
    for i, name in enumerate(HUES):
        split, median = data[name]
        panel(ax, i * (w + GAP), w, name, split, median, rng)
    fig.savefig(HERE.parent / "figures/benchmarks.pdf")


if __name__ == "__main__":
    main()
