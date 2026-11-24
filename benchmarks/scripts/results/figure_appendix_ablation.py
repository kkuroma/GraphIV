"""
Figure appendix_ablation: the change in recall from zeroing or refitting without one term, over every embedder

Each recipe is scored on its own benchmark at the middle cutoff, and a dagger marks a score significantly below the full recipe
"""

import json
from pathlib import Path

import numpy as np
from common import (
    ALPHA,
    COLORS,
    EMBEDDERS,
    OWN,
    RECIPES,
    SETS,
    TERMS,
    holm_tests,
    mean,
    middle,
    output,
    plt,
    style,
)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

STEM = Path(__file__).stem
KINDS = {"zero": "Zero the term", "without": "Refit without the term"}
ABLATED = [t for t in TERMS if t != "cos"]


def run(e: str, r: str, variant: str) -> str:
    return f"{e}-{OWN[r]}-{r}_bf_{variant}_seed0"


def main() -> None:
    pairs = {
        (k, r, e, t): (f"{e}-{OWN[r]}-{r}_bf_seed0", run(e, r, f"{k}_{t}"), middle(OWN[r]))
        for k in KINDS
        for r in RECIPES
        for e in EMBEDDERS
        for t in ABLATED
    }
    adjusted = holm_tests(pairs)
    delta = {key: mean(b, m) - mean(a, m) for key, (a, b, m) in pairs.items()}
    low, high = min(delta.values()), max(delta.values())
    style()
    cmap = LinearSegmentedColormap.from_list("loss", [COLORS["RED"], "#F6EEE8", COLORS["BLUE"]])
    fig, axes = plt.subplots(len(KINDS), len(RECIPES), figsize=(6.75, 4.6), constrained_layout=True)
    for i, k in enumerate(KINDS):
        for j, r in enumerate(RECIPES):
            ax = axes[i][j]
            grid = np.array([[delta[k, r, e, t] for e in EMBEDDERS] for t in ABLATED])
            im = ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(0, low, high), aspect="auto")
            for a, t in enumerate(ABLATED):
                for b, e in enumerate(EMBEDDERS):
                    if adjusted[k, r, e, t] <= ALPHA:
                        ax.text(b, a, "†", ha="center", va="center", fontsize=5, color=COLORS["INK"])
            ax.set_xticks(range(len(EMBEDDERS)), list(EMBEDDERS.values()), rotation=90, fontsize=4.8)
            ax.set_yticks(range(len(ABLATED)), ABLATED if j == 0 else [], fontsize=5.2)
            ax.grid(False)
            ax.tick_params(length=0)
            if i == 0:
                ax.set_title(f"{r}, {SETS[OWN[r]]['label']}", fontsize=6.5, color=COLORS["INK"])
            if j == 0:
                ax.set_ylabel(KINDS[k], fontsize=6.5, color=COLORS["MUTED"])
            for side in ax.spines.values():
                side.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.5, aspect=30, location="right")
    cbar.set_label("change in recall", fontsize=6, color=COLORS["MUTED"])
    cbar.ax.tick_params(labelsize=5)
    fig.savefig(output(STEM).with_suffix(".pdf"))
    fig.savefig(output(STEM).with_suffix(".png"), dpi=200)
    data = {" ".join(key): {"delta": delta[key], "holm": adjusted[key]} for key in pairs}
    output(STEM).with_suffix(".json").write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
