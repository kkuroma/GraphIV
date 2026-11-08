"""
Writes every table the paper reports from runs/ and fit_cache/, to [paths] TABLES

A run is read from its summary, and a cell stays pending until the run covers every case
"""

import json

import numpy as np

from src import common, datasets

ROWS = ["survey", "frontier", "consensus", "ancestry", "evolution", "focused", "ppr", "gar_cite", "gar_sim", "rerank", "nomic_cosine", "bm25", "scincl", "specter2"]
# each eval's experiment suffix and the name the paper gives it
EVALS = {
    "surveymatch": "SurveyMatch",
    "frontiermatch": "FrontierMatch",
    "consensusmatch": "ConsensusMatch",
    "pasa_test": "AutoScholarQuery",
    "pasa_real": "RealScholarQuery",
    "surge": "SurGE",
}
REPORTED = {
    "pasa_test": [("PaSa-7b, reported", {"20": 0.530, "50": 0.633, "100": 0.695}), ("Google with GPT-4o, reported", {"20": 0.192, "50": 0.245, "100": 0.268})],
    "pasa_real": [("PaSa-7b, reported", {"20": 0.580, "50": 0.656, "100": 0.693})],
}
# their Table 3, nDCG@20 then recall at 5, 20 and 100, per split
CATALYST = {
    "core": [
        ("BM25, reported", 0.16, 0.12, 0.23, 0.39),
        ("SPECTER2, reported", 0.13, 0.10, 0.20, 0.38),
        ("OpenScholar, reported", 0.16, 0.13, 0.23, 0.40),
        ("Qwen3-Emb-4B, reported", 0.25, 0.18, 0.39, 0.59),
        ("Qwen3-Emb-8B, reported", 0.26, 0.21, 0.37, 0.58),
        ("Gemini-Emb-2, reported", 0.27, 0.21, 0.38, 0.62),
        ("GPT-4.1 tool-calling agent, reported", 0.25, 0.19, 0.37, 0.60),
    ],
    "subfield": [
        ("BM25, reported", 0.21, 0.12, 0.26, 0.45),
        ("SPECTER2, reported", 0.18, 0.11, 0.23, 0.42),
        ("OpenScholar, reported", 0.19, 0.12, 0.24, 0.46),
        ("Qwen3-Emb-4B, reported", 0.35, 0.21, 0.44, 0.69),
        ("Qwen3-Emb-8B, reported", 0.41, 0.25, 0.51, 0.73),
        ("Gemini-Emb-2, reported", 0.38, 0.24, 0.46, 0.70),
        ("GPT-4.1 tool-calling agent, reported", 0.37, 0.24, 0.43, 0.70),
    ],
}
SECONDS = ["survey", "ppr", "nomic_cosine"]
TERMS = ["rec", "anc", "fro", "anc2", "cocite", "couple", "hub"]


def summary(row: str, suffix: str) -> dict | None:
    path = common.RUNS / f"{row}-{suffix}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cell(v: float | None, best: float) -> str:
    if v is None:
        return "pending"
    return f"**{v:.3f}**" if round(v, 3) == round(best, 3) else f"{v:.3f}"


def table(title: str, rows: list[tuple[str, dict[str, float]]]) -> list[str]:
    ks = sorted({k for _, r in rows for k in r}, key=int)
    best = {k: max(r[k] for _, r in rows if k in r) for k in ks}
    lines = [title, "", "| method | " + " | ".join(f"R@{k}" for k in ks) + " |", "| --- " * (len(ks) + 1) + "|"]
    lines += [f"| {label} | " + " | ".join(cell(r.get(k), best[k]) for k in ks) + " |" for label, r in rows]
    return lines + [""]


def bench(suffix: str) -> list[str]:
    rows = list(REPORTED.get(suffix, []))
    cases = None
    for row in ROWS:
        s = summary(row, suffix)
        rows.append((row, s["recall"] if s else {}))
        cases = cases or (s and s["cases"])
    return table(f"## {EVALS[suffix]}, {cases:,} cases", rows)


def outside() -> list[str]:
    # the SurGE cases whose survey anchors no train or val case of the release
    held = {c["sample_id"] for c in datasets.load("surge").get("test") if not c["in_train"]}
    rows = []
    for row in ROWS:
        path = common.RUNS / f"{row}-surge.jsonl"
        lines = [r for r in common.read_jsonl(path) if r["sample_id"] in held] if path.exists() else []
        rows.append((row, {k: float(np.mean([r["recall"][k] for r in lines])) for k in lines[0]["recall"]} if lines else {}))
    return table(f"## SurGE, the {len(held):,} cases outside our train and val", rows)


def fits(method: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((common.FITS / method).glob("*.json"))]


def ppr_picks() -> list[str]:
    lines = ["## PPR settings, picked on train", "", "| train set | split | cases | cut-off | alpha | teleport | edges | train recall |", "| --- " * 8 + "|"]
    order = list(EVALS) + ["pasa"]
    for f in sorted(fits("ppr"), key=lambda f: order.index(f["fit"]["train"])):
        r = f["result"]
        a, t, e = r["pick"]
        lines.append(f"| {f['fit']['train']} | {f['fit']['split']} | {r['cases']} | R@{r['k']} | {a} | {t} | {e} | {r['grid']['/'.join(map(str, r['pick']))]:.4f} |")
    return lines + [
        "",
        "| Footnote | |",
        "| --- | --- |",
        "| pool | the same expanded pool BoundedField scores, cosine 5,000 plus the ancestor and frontier hops |",
        "| grid | alpha 0.1, 0.3, 0.5, 0.7, 0.85, 0.95; teleport by cosine over the pool or its top 200; citing to cited or both ways |",
        "| no train split | SurGE takes SurveyMatch's pick and RealScholarQuery takes AutoScholarQuery's |",
        "",
    ]


def seconds() -> list[str]:
    lines = ["## Seconds per query, embedding included", "", "| run | " + " | ".join(EVALS.values()) + " |", "| --- " * (len(EVALS) + 1) + "|"]
    for row in SECONDS:
        values = [summary(row, suffix) for suffix in EVALS]
        lines.append(f"| {row} | " + " | ".join(f"{s['seconds']:.2f}" if s and s["seconds"] is not None else "pending" for s in values) + " |")
    return lines + ["", "| Footnote | |", "| --- | --- |", "| measured | as each run went, so a set run first on a cold store reads slower |", ""]


def focused() -> list[str]:
    lines = ["## Focused recipe, AutoScholarQuery val", "", "| point | val recall | " + " | ".join(TERMS) + " |", "| --- " * (len(TERMS) + 2) + "|"]
    footer = []
    for f in fits("boundedfield"):
        r, p = f["result"], f["fit"]
        for label, c in r["candidates"].items():
            lines.append(f"| {label} | {c['val']:.4f} | " + " | ".join(f"{c['exponents'][t]:g}" for t in TERMS) + " |")
        footer.append(f"| fit | {r['train_cases']:,} train cases, {p['steps']} steps of {p['batch']}, {r['val_cases']} val cases at R@{p['k']}; picked: {r['picked']} |")
    return lines + ["", "| Footnote | |", "| --- | --- |", *footer, ""]


def catalyst() -> list[str]:
    lines = []
    for split, reported in CATALYST.items():
        rows = [(label, v) for label, *v in reported]
        for path in sorted(common.RUNS.glob(f"*-scholarcatalyst_{split}.json")):
            s = json.loads(path.read_text())
            rows.append((path.stem.rsplit("-", 1)[0], [s["ndcg"].get("20"), *(s["recall"].get(k) for k in ("5", "20", "100"))]))
        best = [max(v[i] for _, v in rows if v[i] is not None) for i in range(4)]
        lines += [f"## ScholarCatalyst {split} queries, inside their corpus", "", "| method | nDCG@20 | R@5 | R@20 | R@100 |", "| --- " * 5 + "|"]
        lines += [f"| {label} | " + " | ".join(cell(x, b) for x, b in zip(v, best, strict=True)) + " |" for label, v in rows]
        lines.append("")
    return lines + [
        "| Footnote | |",
        "| --- | --- |",
        "| reported | the paper's two-decimal figures, their 191K corpus cut at the source paper's month |",
        "| ours | ranked over the whole store with its citation graph, cut to their corpus ids, capped at the source paper's day |",
        "",
    ]


def main() -> None:
    setting = [
        "# Main Tables",
        "",
        "| Item | Value |",
        "| --- | --- |",
        "| Split | `graphiv-benchmarks-2026-10-22` test: SurveyMatch, FrontierMatch, ConsensusMatch |",
        "| PaSa | AutoScholarQuery test and RealScholarQuery, each capped at its source paper's date |",
        "| SurGE | the survey task, each survey's title capped at its date and blocked from it |",
        "| ScholarCatalyst | each author-verified question capped at its source paper's date, scored inside the benchmark's corpus |",
        "| Recipes | survey, frontier, consensus tuned on their train split; ancestry, evolution zeroed; focused tuned on AutoScholarQuery train |",
        "| pending | the run has not been made yet |",
        "| bold | the highest recall at that N |",
        "",
    ]
    lines = setting + [line for suffix in EVALS for line in bench(suffix)] + outside() + catalyst() + ppr_picks() + seconds() + focused()
    common.TABLES.parent.mkdir(parents=True, exist_ok=True)
    common.TABLES.write_text("\n".join(lines))
    print(common.TABLES)


if __name__ == "__main__":
    main()
