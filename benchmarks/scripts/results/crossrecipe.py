"""
The cross-recipe tables of the appendix: every recipe as PPR and BoundedField on one subtable's sets, on Nomic

A dagger marks a score significantly below the same set's own-mode BoundedField, which is the reference of its column
"""

from common import (
    ALPHA,
    CONFIG,
    RECIPES,
    SETS,
    arxiv_only,
    float_table,
    fmt,
    holm_tests,
    mean,
    near,
    save,
    subtable,
)
from table_main_benchmarks import columns, heads

MAIN = CONFIG["table_main_benchmarks"]
E = "nomic"


def methods(sets: list[str]) -> list[tuple[str, str, dict[str, str]]]:
    """The baselines of table_main_benchmarks that run on Nomic, then each recipe as PPR and BoundedField"""
    out = [(f"{r}-{m}", label, {s: f"{r}-{s}-{m}" for s in sets}) for r, m, label in MAIN["baselines"]]
    for r in RECIPES:
        for m, title in (("ppr", "PPR"), ("bf", "BoundedField")):
            out.append((f"{r}-{m}", f"{title}, \\term{{{r}}}", {s: f"{E}-{s}-{r}_{m}_seed0" for s in sets}))
    return out


def build(stem: str) -> None:
    spec = CONFIG[stem]
    part = spec["part"]
    cols = columns(part)
    sets = list(MAIN[part]["columns"])
    rows = methods(sets)
    ref = {s: f"{E}-{s}-{SETS[s]['recipe']}_bf_seed0" for s in sets}
    pairs = {(key, s, m): (ref[s], runs[s], m) for key, _, runs in rows for s, m in cols if runs[s] != ref[s]}
    adjusted = holm_tests(pairs)
    sc = part == "scholarcatalyst"
    data, means = {}, {}
    for key, _, runs in rows:
        for s, m in cols:
            cell = {"run": runs[s], "mean": mean(runs[s], m), "holm": adjusted.get((key, s, m))}
            if sc:
                cell["arxiv"] = arxiv_only(runs[s], s, m)
            data[f"{key} {s} {m}"] = cell
            means[key, s, m] = cell["mean"]
    best = {(s, m): max(means[k, s, m] for k, _, _ in rows) for s, m in cols}
    lines = []
    for key, label, _ in rows:
        if key == "focused-ppr":
            lines.append("\\midrule")
        texts = []
        for s, m in cols:
            c = data[f"{key} {s} {m}"]
            sup = "\\dagger" if (c["holm"] if c["holm"] is not None else 1.0) <= ALPHA else ""
            t = fmt(c["mean"], near(c["mean"], best[s, m]), False, sup)
            texts.append(t + (f" ({c['arxiv']:.3f})" if sc else ""))
        lines.append(f"{label} & " + " & ".join(texts) + " \\\\")
    caption = (
        "Every recipe as PPR and as BoundedField on the Nomic embedding, beside the baselines."
        " Bold marks the best of each column."
        " $^\\dagger$ marks a score significantly below BoundedField run with the recipe of that benchmark's own mode,"
        " by a one-sided Wilcoxon signed-rank test on per-case scores,"
        f" Holm-corrected over the {len(pairs)} comparisons, $p < {ALPHA}$."
    )
    if sc:
        caption += " Each cell scores every answer, then in parentheses only the answers on arXiv, as our corpus holds no other papers, and the marks test the score over every answer."
    sub = subtable("", "", heads(part), lines)
    save(stem, float_table(stem, [sub], caption, spec["label"]), data)
