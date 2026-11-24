"""
Figure appendix_tuning: val recall over BoundedField's core and pool grid, one panel per recipe

Every point is the picked candidate of one grid cell in a seed-0 tune's report.json, and a ring marks the pick
"""

from pathlib import Path

from common import COLORS, CONFIG, EMBEDDERS, RECIPES, output, plt, report, style

STEM = Path(__file__).stem
SPEC = CONFIG[STEM]


def main() -> None:
    style()
    fig, axes = plt.subplots(1, len(RECIPES), figsize=(6.75, 2.1), constrained_layout=True)
    for ax, r in zip(axes, RECIPES, strict=True):
        for e, name in EMBEDDERS.items():
            rep = report(f"{e}-{r}_bf_seed0")
            cells = rep["grid"]
            keys = sorted(cells, key=lambda k: (cells[k]["core"], cells[k]["pool"]))
            ys = [cells[k]["candidates"][cells[k]["picked"]]["judged"] for k in keys]
            color = COLORS[SPEC["families"][e.split("_")[0]]]
            ax.plot(range(len(keys)), ys, SPEC["dashes"][e], color=color, lw=0.8, label=name)
            pick = next(i for i, k in enumerate(keys) if (cells[k]["core"], cells[k]["pool"]) == (rep["pick"]["core"], rep["pick"]["pool"]))
            ax.plot([pick], [ys[pick]], "o", ms=3.2, mfc="none", mec=color, mew=0.7)
        ax.set_xticks(range(len(keys)), [f"{cells[k]['core']}/{cells[k]['pool']}" for k in keys], rotation=90, fontsize=5)
        ax.set_title(f"{r}, R@{rep['k']}", fontsize=7, color=COLORS["INK"])
        ax.grid(axis="y", color=COLORS["GRID"], lw=0.4)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_xlabel("core / pool", fontsize=6, color=COLORS["MUTED"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("val recall", fontsize=6.5, color=COLORS["MUTED"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=8, fontsize=5.5, frameon=False, handlelength=1.8, columnspacing=1.0)
    fig.savefig(output(STEM).with_suffix(".pdf"))
    fig.savefig(output(STEM).with_suffix(".png"), dpi=200)


if __name__ == "__main__":
    main()
