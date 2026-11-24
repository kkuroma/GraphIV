"""
Table main_benchmarks: every method we ran on every test set at the paper cutoffs, beside the reported rows

A dagger marks a score significantly below BoundedField on the first embedder, a double dagger on the second
Reported rows are parsed from each paper's html and have no per-case scores, so they carry no marks
"""

import json
from pathlib import Path

from common import (
    ALPHA,
    ARXIV,
    BENCH,
    CONFIG,
    EMBEDDERS,
    SETS,
    arxiv_only,
    cases,
    fmt,
    holm,
    html_table,
    near,
    number,
    one_sided,
    output,
    per_case,
    subtable,
)

STEM = Path(__file__).stem
SPEC = CONFIG[STEM]
PARTS = ("proposed", "existing", "scholarcatalyst")
SC = set(SPEC["scholarcatalyst"]["columns"])


def methods() -> list[tuple[str, str, dict[str, str]]]:
    """
    Every row we ran, in table order

    Returns:
        A key, a label and the run name on each set, for the baselines, then PPR, then BoundedField.
    """
    every = [s for part in PARTS for s in SPEC[part]["columns"]]
    out = [(f"{r}-{m}", label, {s: f"{r}-{s}-{m}" for s in every}) for r, m, label in SPEC["baselines"]]
    for method, title in (("ppr", "PPR"), ("bf", "BoundedField")):
        for e in SPEC["embedders"]:
            runs = {s: f"{e}-{s}-{SETS[s]['recipe']}_{method}_seed0" for s in every}
            out.append((f"{e}-{method}", f"{title} ({EMBEDDERS[e]})", runs))
    return out


def columns(part: str) -> list[tuple[str, str]]:
    return [(s, metric) for s, metrics in SPEC[part]["columns"].items() for metric in metrics]


def collect() -> dict:
    """
    Means, arXiv-only means on ScholarCatalyst, and both Holm families

    Returns:
        A dict keyed by row then by "set metric" holding the mean, the arXiv-only mean, and each reference's raw and Holm p.
    """
    rows = methods()
    out: dict = {key: {} for key, _, _ in rows}
    cols = [c for part in PARTS for c in columns(part)]
    for key, _, runs in rows:
        for s, metric in cols:
            scores = per_case(runs[s], metric)
            cell = {"run": runs[s], "mean": sum(scores.values()) / len(scores)}
            if s in SC:
                cell["arxiv"] = arxiv_only(runs[s], s, metric)
            out[key][f"{s} {metric}"] = cell
    for e in SPEC["embedders"]:
        ref = next(r for r in rows if r[0] == f"{e}-bf")
        family = {
            (key, s, metric): one_sided(per_case(ref[2][s], metric), per_case(runs[s], metric))
            for key, _, runs in rows
            if key != ref[0]
            for s, metric in cols
        }
        for (key, s, metric), p in holm(family).items():
            out[key][f"{s} {metric}"][f"p_{e}"] = family[(key, s, metric)]
            out[key][f"{s} {metric}"][f"holm_{e}"] = p
    return out


def reported(part: str) -> list[tuple[str, list[float | None]]]:
    """
    The reported rows of one subtable, parsed from the paper's html

    Args:
        part: proposed, existing or scholarcatalyst.

    Returns:
        Each reported row's label and its values, None where it reports nothing.
    """
    spec = SPEC[part]
    if "reported_html" not in spec:
        return []
    tables = {t: html_table(BENCH / spec["reported_html"], t) for t, *_ in spec["reported_columns"].values()}
    out = []
    for label in spec["reported_rows"]:
        values: list[float | None] = []
        for s, metrics in spec["columns"].items():
            if s not in spec["reported_columns"]:
                values += [None] * len(metrics)
                continue
            table, *cells = spec["reported_columns"][s]
            values += [number(tables[table][label][i]) for i in cells]
        out.append((label, values))
    return out


def body(part: str, data: dict, heading: str) -> list[str]:
    """
    The rows of one subtable: ours with their marks, a rule before BoundedField, then the reported rows

    Args:
        part: proposed, existing or scholarcatalyst.
        data: The output of collect.
        heading: The italic line above the reported rows.

    Returns:
        The tabular body lines.
    """
    cols = columns(part)
    ours = [(key, label, runs, [data[key][f"{s} {m}"] for s, m in cols]) for key, label, runs in methods()]
    quoted = reported(part)
    best = [max(r[3][i]["mean"] for r in ours) for i in range(len(cols))]
    top = [max([best[i]] + [v[i] for _, v in quoted if v[i] is not None]) for i in range(len(cols))]
    lines = []
    for key, label, runs, cells in ours:
        if key == f"{SPEC['embedders'][0]}-bf":
            lines.append("\\midrule")
        lines.append("% " + ", ".join(dict.fromkeys(runs[s] for s, _ in cols)))
        texts = []
        for i, c in enumerate(cells):
            sup = "".join(mark for e, mark in zip(SPEC["embedders"], SPEC["marks"], strict=True) if c.get(f"holm_{e}", 1.0) <= ALPHA)
            text = fmt(c["mean"], near(c["mean"], best[i]), near(c["mean"], top[i]), sup)
            texts.append(text + (f" ({c['arxiv']:.3f})" if "arxiv" in c else ""))
        lines.append(f"{label} & " + " & ".join(texts) + " \\\\")
    if quoted:
        lines += ["\\midrule", f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        lines.append(
            f"% {SPEC[part]['reported_html']}, " + ", ".join(f"{s} from {v[0]} cells {v[1:]}" for s, v in SPEC[part]["reported_columns"].items())
        )
        for label, values in quoted:
            lines.append(label + " & " + " & ".join(fmt(x, under=near(x, top[i])) for i, x in enumerate(values)) + " \\\\")
    return lines


def heads(part: str, titles: dict[str, str] | None = None) -> list[tuple[str, list[str]]]:
    name = {"recall": "R", "ndcg": "nDCG"}
    out = []
    for s, metrics in SPEC[part]["columns"].items():
        title = (titles or {}).get(s, SETS[s]["label"])
        out.append((title, [f"{name[m.split('@')[0]]}@{m.split('@')[1]}" for m in metrics]))
    return out


def answers(set_name: str) -> tuple[int, int, int]:
    """
    Answer counts of one ScholarCatalyst split

    Args:
        set_name: The split as runs name it.

    Returns:
        The answers on arXiv, all answers, and the cases with no answer on arXiv.
    """
    keys = [c["key"] for c in cases(set_name).values()]
    on = [sum(bool(ARXIV.fullmatch(a)) for a in k) for k in keys]
    return sum(on), sum(map(len, keys)), sum(n == 0 for n in on)


def write(data: dict) -> None:
    first, second = (EMBEDDERS[e] for e in SPEC["embedders"])
    tests = len(methods()) - 1
    n = tests * sum(len(columns(p)) for p in PARTS)
    core, sub = answers("scholarcatalyst_core"), answers("scholarcatalyst_subfield")
    caption = (
        "Retrieval against existing methods."
        " Every method we ran uses the Nomic embedding unless named,"
        " and PPR and BoundedField run the recipe of each benchmark's mode, tuned on the embedder named:"
        " \\term{survey} on SurveyMatch and SurGE, \\term{frontier} and \\term{consensus} on their own,"
        " and \\term{focused} on PaSa and ScholarCatalyst."
        " Bold marks the best of our runs in each column, and an underline the best of every row, reported ones included."
        f" $^\\dagger$ and $^\\ddagger$ mark a score significantly below BoundedField on {first} and on {second},"
        f" by a one-sided Wilcoxon signed-rank test on per-case scores, Holm-corrected over the {n:,} comparisons against each, $p < {ALPHA}$."
        " Reported rows have no per-case scores and are not tested."
    )
    sc_caption = (
        "ScholarCatalyst, scored inside the benchmark's own corpus."
        " Each of our cells scores every answer, then in parentheses only the answers on arXiv, as our corpus holds no other papers:"
        f" {core[0]:,} of the {core[1]:,} core answers and {sub[0]:,} of the {sub[1]:,} subfield answers are on arXiv,"
        f" and the arXiv-only score leaves out the {core[2]} core and {sub[2]} subfield queries with none."
        " The marks test the score over every answer."
    )
    sc_titles = {s: f"{s.rsplit('_', 1)[1].capitalize()}, {len(cases(s))} queries" for s in SPEC["scholarcatalyst"]["columns"]}
    lines = [
        f"% generated by benchmarks/scripts/results/{STEM}.py from benchmarks/data/runs/test",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        *subtable("tab:proposed", "The proposed benchmarks, test splits.", heads("proposed"), body("proposed", data, "")),
        "\\vspace{6pt}",
        *subtable(
            "tab:existing",
            "PaSa's AutoScholarQuery test split and RealScholarQuery, and SurGE. PaSa's reported runs search Google rather than our corpus.",
            heads("existing"),
            body("existing", data, "Reported by \\citet{he2025pasa}"),
        ),
        f"\\caption{{{caption}}}",
        "\\label{tab:retrieval}",
        "\\end{table*}",
        "",
        "\\begin{table*}[t]",
        "\\ContinuedFloat",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        *subtable(
            "tab:scholarcatalyst",
            sc_caption,
            heads("scholarcatalyst", sc_titles),
            body("scholarcatalyst", data, "Reported by \\citet{kim2026scholarcatalyst}"),
        ),
        "\\caption[]{Retrieval against existing methods, continued.}",
        "\\end{table*}",
        "",
    ]
    (output(STEM).with_suffix(".tex")).write_text("\n".join(lines))


def main() -> None:
    data = collect()
    write(data)
    (output(STEM).with_suffix(".json")).write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
