"""
Table appendix_transfer: cosine, Nomic's recipe applied unchanged and the recipe refit, per embedder and set

A dagger marks an unchanged recipe significantly below the embedder's own refit on that set
"""

from pathlib import Path

from common import (
    ALPHA,
    CONFIG,
    EMBEDDERS,
    SETS,
    f3,
    float_table,
    holm_tests,
    mean,
    middle,
    save,
    subtable,
)

STEM = Path(__file__).stem
SOURCE = CONFIG[STEM]["source"]
OTHERS = [e for e in EMBEDDERS if e != SOURCE]


def runs(e: str, s: str) -> dict[str, str]:
    r = SETS[s]["recipe"]
    return {"cosine": f"{e}-{s}-none", "unchanged": f"{e}-{s}-{r}_bf_from_{SOURCE}_seed0", "refit": f"{e}-{s}-{r}_bf_seed0"}


def main() -> None:
    pairs = {(e, s): (runs(e, s)["refit"], runs(e, s)["unchanged"], middle(s)) for e in OTHERS for s in SETS}
    adjusted = holm_tests(pairs)
    names = {"cosine": "cosine", "unchanged": f"{EMBEDDERS[SOURCE].split()[0]} recipe", "refit": "refit"}
    rows, data = [], {}
    for i, e in enumerate(OTHERS):
        if i:
            rows.append("\\addlinespace")
        for kind, label in names.items():
            cells = []
            for s in SETS:
                v = mean(runs(e, s)[kind], middle(s))
                data[f"{e} {kind} {s}"] = {"mean": v, "holm": adjusted[e, s] if kind == "unchanged" else None}
                mark = "$^\\dagger$" if kind == "unchanged" and adjusted[e, s] <= ALPHA else ""
                cells.append(f3(v) + mark)
            rows.append(f"{EMBEDDERS[e]}, {label} & " + " & ".join(cells) + " \\\\")
    groups = [(v["label"], [f"R@{v['cutoff']}"]) for v in SETS.values()]
    caption = (
        f"Recall at the middle cutoff of every set, by embedder: cosine search alone, BoundedField with {EMBEDDERS[SOURCE].split()[0]}'s recipes applied unchanged, and BoundedField with the recipes refit on the embedder."
        " $^\\dagger$ marks the unchanged recipe significantly below the refit, by a one-sided Wilcoxon signed-rank test on per-case recall,"
        f" Holm-corrected over the {len(pairs)} comparisons, $p < {ALPHA}$."
    )
    save(STEM, float_table(STEM, [subtable("tab:transfer-all", "", groups, rows, "Embedder, method")], caption, "tab:transfer", "scriptsize"), data)


if __name__ == "__main__":
    main()
