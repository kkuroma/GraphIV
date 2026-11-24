"""
Table appendix_seeds: the mean and range of val and test recall over five tune seeds

Val is the pick's recall in each seed's report.json and test is each seed's mean at the set's middle cutoff
"""

from pathlib import Path

from common import (
    CONFIG,
    EMBEDDERS,
    RECIPES,
    SETS,
    float_table,
    mean,
    middle,
    report,
    save,
    subtable,
)

STEM = Path(__file__).stem
SEEDS = CONFIG[STEM]["seeds"]


def cell(values: list[float]) -> str:
    return f"{sum(values) / len(values):.3f} ({max(values) - min(values):.3f})"


def main() -> None:
    data: dict = {}
    val_rows, test_rows = [], []
    for e in EMBEDDERS:
        val = {r: [report(f"{e}-{r}_bf_seed{n}")["pick"]["judged"] for n in SEEDS] for r in RECIPES}
        test = {s: [mean(f"{e}-{s}-{v['recipe']}_bf_seed{n}", middle(s)) for n in SEEDS] for s, v in SETS.items()}
        data[e] = {"val": val, "test": test}
        val_rows.append(f"{EMBEDDERS[e]} & " + " & ".join(cell(val[r]) for r in RECIPES) + " \\\\")
        test_rows.append(f"{EMBEDDERS[e]} & " + " & ".join(cell(test[s]) for s in SETS) + " \\\\")
    ks = {r: report(f"nomic-{r}_bf_seed0")["k"] for r in RECIPES}
    subs = [
        subtable(
            "tab:seeds-val", "Val split of each recipe's benchmark.", [(f"\\term{{{r}}}", [f"R@{ks[r]}"]) for r in RECIPES], val_rows, "Embedder"
        ),
        subtable("tab:seeds-test", "Test split of every set.", [(v["label"], [f"R@{v['cutoff']}"]) for v in SETS.values()], test_rows, "Embedder"),
    ]
    caption = f"Recall of BoundedField over tune seeds {SEEDS[0]} to {SEEDS[-1]}, the mean and then the range max minus min in parentheses."
    save(STEM, float_table(STEM, subs, caption, "tab:seeds", "scriptsize"), data)


if __name__ == "__main__":
    main()
