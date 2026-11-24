"""
Table appendix_tuning: the core and pool BoundedField picked for each recipe and embedder, with its val recall

Every value is read from the seed-0 BoundedField tune's report.json
"""

from pathlib import Path

from common import EMBEDDERS, RECIPES, f3, float_table, report, save, subtable

STEM = Path(__file__).stem


def main() -> None:
    reps = {(e, r): report(f"{e}-{r}_bf_seed0") for e in EMBEDDERS for r in RECIPES}
    groups = [(f"\\term{{{r}}}, R@{reps[next(iter(EMBEDDERS)), r]['k']}", ["Core", "Pool", "Val"]) for r in RECIPES]
    rows = []
    for e in EMBEDDERS:
        cells = []
        for r in RECIPES:
            p = reps[e, r]["pick"]
            cells += [str(p["core"]), str(p["pool"]), f3(p["judged"])]
        rows.append(f"{EMBEDDERS[e]} & " + " & ".join(cells) + " \\\\")
    caption = (
        "The core and pool BoundedField picked for each recipe and embedder, with the recall it reached on the val split of the recipe's benchmark."
    )
    data = {f"{e}-{r}": v["pick"] for (e, r), v in reps.items()}
    save(STEM, float_table(STEM, [subtable("tab:tuning", "", groups, rows, "Embedder")], caption, "tab:tuning-picks"), data)


if __name__ == "__main__":
    main()
