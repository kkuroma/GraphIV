"""
Table appendix_ablation: each recipe on its own benchmark with terms removed, on Nomic

A dagger marks a score significantly below the full recipe, one-sided Wilcoxon signed-rank, Holm over every cell
"""

from pathlib import Path

from common import (
    ALPHA,
    CONFIG,
    OWN,
    RECIPES,
    SETS,
    TERMS,
    f3,
    float_table,
    holm_tests,
    mean,
    middle,
    save,
    subtable,
)
from table_main_recipes import term

STEM = Path(__file__).stem
SPEC = CONFIG[STEM]
E = SPEC["embedder"]
ABLATED = [t for t in TERMS if t != "cos"]


def variants() -> list[tuple[str, str, str]]:
    """Each variant's key, its row label and its run suffix, after the full recipe"""
    out = [(f"zero_{t}", f"Zero {term(t)}", f"bf_zero_{t}_seed0") for t in ABLATED]
    out += [(f"without_{t}", f"Refit without {term(t)}", f"bf_without_{t}_seed0") for t in ABLATED]
    out += [(k, label, f"bf_{k}_seed0") for k, label in SPEC["variants"].items()]
    return out + [("none", "Cosine, no scorer", "none")]


def run(r: str, suffix: str) -> str:
    s = OWN[r]
    return f"{E}-{s}-none" if suffix == "none" else f"{E}-{s}-{r}_{suffix}"


def main() -> None:
    rows_def = variants()
    pairs = {(k, r): (run(r, "bf_seed0"), run(r, suffix), middle(OWN[r])) for k, _, suffix in rows_def for r in RECIPES}
    adjusted = holm_tests(pairs)
    data, rows = {}, []
    rows.append("Full recipe & " + " & ".join(f3(mean(run(r, "bf_seed0"), middle(OWN[r]))) for r in RECIPES) + " \\\\")
    for k, label, suffix in rows_def:
        if k in ("zero_" + ABLATED[0], "without_" + ABLATED[0], next(iter(SPEC["variants"]))):
            rows.append("\\midrule")
        cells = []
        for r in RECIPES:
            v = mean(run(r, suffix), middle(OWN[r]))
            data[f"{k} {r}"] = {"mean": v, "holm": adjusted[k, r]}
            cells.append(f3(v) + ("$^\\dagger$" if adjusted[k, r] <= ALPHA else ""))
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    groups = [(f"\\term{{{r}}}", [f"R@{SETS[OWN[r]]['cutoff']}"]) for r in RECIPES]
    caption = (
        "Each recipe on its own benchmark's test split with the Nomic embedding, with terms removed: \\term{focused} on AutoScholarQuery, \\term{survey} on SurveyMatch, \\term{frontier} on FrontierMatch, and \\term{consensus} on ConsensusMatch."
        " Zero sets one exponent to zero as tuned, and refit tunes the recipe again without the term."
        " The last rows keep only the named terms, apply the tuned exponents to the cosine pool without the citation hop, and order the cosine pool by cosine."
        " $^\\dagger$ marks a score significantly below the full recipe, by a one-sided Wilcoxon signed-rank test on per-case recall,"
        f" Holm-corrected over the {len(pairs)} comparisons, $p < {ALPHA}$."
    )
    save(STEM, float_table(STEM, [subtable("tab:ablation-all", "", groups, rows, "Variant")], caption, "tab:ablation", "scriptsize", star=False), data)


if __name__ == "__main__":
    main()
