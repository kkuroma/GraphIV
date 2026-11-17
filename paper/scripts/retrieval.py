import json
import math
import re
from functools import cache
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("pdf")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "benchmarks/data/runs"
PAPER = Path(__file__).resolve().parents[1]
FONTS = Path("/nix/store/5xhrl01vhivcmwvr3sdkz0ygdbsyd84l-texlive-2025-r78234-final-env-texmfdist/fonts/opentype/public/libertine")
RED = "#902020"
BLUE = "#1F74A8"
BASE = "#A09692"
INK = "#2A1E18"
MUTED = "#60504A"
GRID = "#EDE3DD"

BENCH = [
    ("surveymatch", "SurveyMatch", "200", "survey"),
    ("frontiermatch", "FrontierMatch", "50", "frontier"),
    ("consensusmatch", "ConsensusMatch", "50", "consensus"),
    ("pasa_test", "AutoScholarQuery", "50", "focused"),
    ("pasa_real", "RealScholarQuery", "50", "focused"),
    ("surge", "SurGE", "200", "survey"),
    ("scholarcatalyst_core", "ScholarCatalyst core", "20", "focused"),
    ("scholarcatalyst_subfield", "ScholarCatalyst subfield", "20", "focused"),
]
MODE = {b: r for b, _, _, r in BENCH}
CASES = {
    "surveymatch": "surveymatch/test.jsonl",
    "frontiermatch": "frontiermatch/test.jsonl",
    "consensusmatch": "consensusmatch/test.jsonl",
    "pasa_test": "pasa/test.jsonl",
    "pasa_real": "pasa/real.jsonl",
    "surge": "surge/test.jsonl",
}
METHODS = [
    ("bm25", "BM25"),
    ("specter2", "SPECTER2"),
    ("scincl", "SciNCL"),
    ("nomic_cosine", "Nomic cosine"),
    ("rerank", "OpenScholar rerank"),
    ("gar_sim", "GAR (similarity)"),
    ("gar_cite", "GAR (citations)"),
    ("ppr", "PPR"),
]
RECIPES = [("focused", "\\term{focused}"), ("survey", "\\term{survey}"), ("frontier", "\\term{frontier}"), ("consensus", "\\term{consensus}")]
EMBEDDERS = [
    ("nomic", "Nomic 0.5B"),
    ("qwen3_0_6b", "Qwen3 0.6B"),
    ("harrier_0_6b", "Harrier 0.6B"),
    ("nemotron_1b", "Nemotron 1B"),
    ("qwen3_4b", "Qwen3 4B"),
    ("qwen3_8b", "Qwen3 8B"),
    ("nemotron_8b", "Nemotron 8B"),
    ("harrier_27b", "Harrier 27B"),
]
BEST = "nemotron_8b"
PASA_PAPER = "data/arxiv/html/2501/2501.10120.html"
SC_PAPER = "data/arxiv/html/2610/2610.02202.html"
REPORTED_FIG = {
    "scholarcatalyst_core": {"qwen3_4b": 0.39, "qwen3_8b": 0.37},
    "scholarcatalyst_subfield": {"qwen3_4b": 0.44, "qwen3_8b": 0.51},
}
PASA_REPORTED = [
    ("Google", (0.1568, 0.1891, 0.2015), (0.1834, 0.2342, 0.2535)),
    ("Google Scholar", (0.0609, 0.0970, 0.1130), (0.1514, 0.2155, 0.2809)),
    ("Google with GPT-4o", (0.1921, 0.2450, 0.2683), (0.2020, 0.2573, 0.2946)),
    ("PaSa-7b", (0.5301, 0.6334, 0.6947), (0.5798, 0.6563, 0.6929)),
    ("PaSa-7b-ensemble", (0.5326, 0.6386, 0.7099), (0.5986, 0.6877, 0.7281)),
]
SC_REPORTED = [
    ("BM25", (0.16, 0.12, 0.23, 0.39, 0.21, 0.12, 0.26, 0.45)),
    ("LateOn 0.1B", (0.16, 0.12, 0.24, 0.37, 0.27, 0.16, 0.33, 0.55)),
    ("SPECTER2", (0.13, 0.10, 0.20, 0.38, 0.18, 0.11, 0.23, 0.42)),
    ("OpenScholar", (0.16, 0.13, 0.23, 0.40, 0.19, 0.12, 0.24, 0.46)),
    ("Qwen3-Emb-4B", (0.25, 0.18, 0.39, 0.59, 0.35, 0.21, 0.44, 0.69)),
    ("Qwen3-Emb-8B", (0.26, 0.21, 0.37, 0.58, 0.41, 0.25, 0.51, 0.73)),
    ("Gemini-Emb-2", (0.27, 0.21, 0.38, 0.62, 0.38, 0.24, 0.46, 0.70)),
    ("ReasonEmbed", (0.23, 0.18, 0.33, 0.53, 0.33, 0.19, 0.40, 0.65)),
    ("Inf-Retriever-v1-Pro", (0.26, 0.20, 0.37, 0.53, 0.31, 0.19, 0.38, 0.63)),
    ("GPT-4.1 + tool-calling agent", (0.25, 0.19, 0.37, 0.60, 0.37, 0.24, 0.43, 0.70)),
    ("o3 + deep research agent", (0.24, 0.21, 0.33, 0.59, 0.33, 0.22, 0.38, 0.69)),
]
S, F = ("100", "200", "500"), ("20", "50", "100")
ARXIV = re.compile(r"\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7}")
SC = ("scholarcatalyst_core", "scholarcatalyst_subfield")


@cache
def keys(bench: str) -> dict[str, set[str]]:
    path = ROOT / "benchmarks/data/cases" / CASES[bench]
    return {c["sample_id"]: set(c["key"]) for c in map(json.loads, open(path))}


@cache
def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in open(RUNS / f"{name}.jsonl")]


def run(name: str, k: str, metric: str = "recall") -> float | None:
    if not (RUNS / f"{name}.jsonl").exists():
        return None
    rs = rows(name)
    bench = name.rsplit("-", 1)[1]
    if metric == "ndcg":
        return sum(r["ndcg"][k] for r in rs) / len(rs)
    if bench not in CASES:
        return sum(r["recall"][k] for r in rs) / len(rs)
    key = keys(bench)
    n = int(k)
    return sum(len(set(r["ranked"][:n]) & key[r["sample_id"]]) / len(key[r["sample_id"]]) for r in rs) / len(rs)


def arxiv_only(name: str, k: str, metric: str) -> float:
    bench = name.rsplit("-", 1)[1]
    cases = ROOT / "benchmarks/data/cases/scholarcatalyst" / f"{bench.split('_')[1]}.jsonl"
    key = {c["sample_id"]: {p for p in c["key"] if ARXIV.fullmatch(p)} for c in map(json.loads, open(cases))}
    scored = [(r["ranked"], key[r["sample_id"]]) for r in rows(name) if key[r["sample_id"]]]
    n = int(k)
    if metric == "ndcg":
        return sum(ndcg(ranked, answers, n) for ranked, answers in scored) / len(scored)
    return sum(len(set(ranked[:n]) & answers) / len(answers) for ranked, answers in scored) / len(scored)


def ndcg(ranked: list[str], key: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in key)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(key), k)))
    return dcg / ideal if ideal else 0.0


def embedder_run(kind: str, embedder: str, bench: str) -> str:
    if kind == "bfrefit" and embedder == "nomic":
        kind = "bf"
    return f"{kind}_{embedder}-{bench}"


def values(names: list[str], cols: list[tuple[str, tuple]]) -> list[float | None]:
    out = []
    for name, (bench, ks) in zip(names, cols):
        for k in ks:
            out.append(run(name, "20", "ndcg") if k == "ndcg" else run(name, k))
    return out


def arxiv_values(names: list[str], cols: list[tuple[str, tuple]]) -> list[float] | None:
    if not all(b in SC for b, _ in cols):
        return None
    return [arxiv_only(name, "20" if k == "ndcg" else k, "ndcg" if k == "ndcg" else "recall") for name, (_, ks) in zip(names, cols) for k in ks]


def source(names: list[str]) -> str:
    return "% " + ", ".join(f"benchmarks/data/runs/{n}.jsonl" for n in names)


def fmt(v: float | None, bold: bool = False, under: bool = False) -> str:
    if v is None:
        return "--"
    t = f"{v:.2f}" if round(v, 2) == v else f"{v:.3f}"
    if bold:
        t = f"\\textbf{{{t}}}"
    if under:
        t = f"\\underline{{{t}}}"
    return t


def block(ours: list[tuple], reported: list[tuple], heading: str, n: int) -> list[str]:
    near = lambda x, t: x is not None and t is not None and abs(x - t) < 5e-4
    best = [max((r[1][i] for r in ours if r[1][i] is not None), default=None) for i in range(n)]
    top = [max((r[1][i] for r in ours + reported if r[1][i] is not None), default=None) for i in range(n)] if reported else [None] * n
    out, split = [], False
    for label, v, src, extra in ours:
        if not split and ("BoundedField" in label or "term" in label):
            split = True
            out.append("\\midrule")
            if "term" in label:
                out.append(f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{BoundedField}}}} \\\\")
        out.append(src)
        cells = [fmt(x, near(x, best[i]), near(x, top[i])) + (f" ({extra[i]:.3f})" if extra else "") for i, x in enumerate(v)]
        out.append(label + " & " + " & ".join(cells) + " \\\\")
    if reported:
        out += ["\\midrule", f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        for label, v, src in reported:
            out.append(src)
            out.append(label + " & " + " & ".join(fmt(x, under=near(x, top[i])) for i, x in enumerate(v)) + " \\\\")
    return out


def tabular(groups: list[tuple[str, tuple]], body: list[str]) -> list[str]:
    n = sum(len(ks) for _, ks in groups)
    head = " & " + " & ".join(f"\\multicolumn{{{len(ks)}}}{{c}}{{{name}}}" for name, ks in groups) + " \\\\"
    rule, start = "", 2
    for _, ks in groups:
        rule += f"\\cmidrule(lr){{{start}-{start + len(ks) - 1}}}"
        start += len(ks)
    cuts = "Method & " + " & ".join(k for _, ks in groups for k in ks) + " \\\\"
    return [f"\\begin{{tabular}}{{l{'r' * n}}}", "\\toprule", head, rule, cuts, "\\midrule", *body, "\\bottomrule", "\\end{tabular}"]


def subtable(label: str, caption: str, groups: list[tuple[str, tuple]], body: list[str]) -> list[str]:
    return ["\\begin{subtable}{\\textwidth}", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", *tabular(groups, body), "\\end{subtable}"]


def rk(ks: tuple) -> tuple:
    return tuple(f"R@{k}" for k in ks)


def ours(cols: list[tuple[str, tuple]], with_best: bool = True) -> list[tuple]:
    out = []
    for m, label in METHODS:
        names = [f"{m}-{b}" for b, _ in cols]
        if all((RUNS / f"{n}.jsonl").exists() for n in names):
            out.append((label, values(names, cols), source(names), arxiv_values(names, cols)))
    names = [f"{MODE[b]}-{b}" for b, _ in cols]
    out.append(("BoundedField (Nomic)", values(names, cols), source(names), arxiv_values(names, cols)))
    if with_best:
        names = [embedder_run("bfrefit", BEST, b) for b, _ in cols]
        out.append((f"BoundedField ({dict(EMBEDDERS)[BEST]})", values(names, cols), source(names), arxiv_values(names, cols)))
    return out


def tables() -> None:
    proposed = [("surveymatch", S), ("frontiermatch", F), ("consensusmatch", F)]
    existing = [("pasa_test", F), ("pasa_real", F), ("surge", S)]
    sc = [(b, ("ndcg", "5", "20", "100")) for b in SC]
    pasa = [(label, list(a) + list(r) + [None] * 3, f"% {PASA_PAPER}, Table 5 (AutoScholarQuery test) and Table 6 (RealScholarQuery), Recall@20, @50, @100") for label, a, r in PASA_REPORTED]
    screp = [(label, list(v), f"% {SC_PAPER}, Table 3, pre-cutoff rows, nDCG@20, R@5, R@20, R@100 for core then subfield") for label, v in SC_REPORTED]
    sc_heads = [("Core, 207 queries", ("nDCG@20", "R@5", "R@20", "R@100")), ("Subfield, 687 queries", ("nDCG@20", "R@5", "R@20", "R@100"))]
    best_label = dict(EMBEDDERS)[BEST]
    note = "% recall is recomputed from each run's rankings against benchmarks/data/cases/<benchmark>/<split>.jsonl, and ScholarCatalyst's nDCG and recall are the means of each run's per-case values"
    CAPTION = f"\\caption{{Retrieval against existing methods. Every method we ran uses the Nomic embedding unless named, and BoundedField runs the recipe of each benchmark's mode: \\term{{survey}} on SurveyMatch and SurGE, \\term{{frontier}} and \\term{{consensus}} on their own, and \\term{{focused}} on PaSa and ScholarCatalyst. The {best_label} row runs the same recipes refit on that embedder. Bold marks the best of our runs in each column, and an underline the best of every row, reported ones included.}}"
    lines = [
        "% generated by paper/scripts/retrieval.py",
        note,
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        *subtable("tab:proposed", "The proposed benchmarks, test splits.", [("SurveyMatch", rk(S)), ("FrontierMatch", rk(F)), ("ConsensusMatch", rk(F))], block(ours(proposed), [], "", 9)),
        "\\vspace{6pt}",
        *subtable("tab:existing", "PaSa's AutoScholarQuery test split and RealScholarQuery, and SurGE. PaSa's reported runs search Google rather than our corpus.", [("AutoScholarQuery", rk(F)), ("RealScholarQuery", rk(F)), ("SurGE", rk(S))], block(ours(existing), pasa, "Reported by \\citet{he2025pasa}", 9)),
        CAPTION,
        "\\label{tab:retrieval}",
        "\\end{table*}",
        "",
        "\\begin{table*}[t]",
        "\\ContinuedFloat",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        *subtable("tab:scholarcatalyst", "ScholarCatalyst, scored inside the benchmark's own corpus. Each of our cells scores every answer, then in parentheses only the answers on arXiv, as our corpus holds no other papers: 604 of the 661 core answers and 3,904 of the 4,184 subfield answers are on arXiv, and the arXiv-only score leaves out the 6 core and 5 subfield queries with none.", sc_heads, block(ours(sc), screp, "Reported by \\citet{kim2026scholarcatalyst}", 8)),
        "\\caption[]{Retrieval against existing methods, continued.}",
        "\\end{table*}",
        "",
    ]
    (PAPER / "tables/results_retrieval.tex").write_text("\n".join(lines))

    def recipes(cols, names, label, caption, heads=None):
        body = [(r[0], r[1], r[2], None) for r in ours(cols, with_best=False) if not r[0].startswith("BoundedField")]
        for m, title in RECIPES:
            ns = [f"{m}-{b}" for b, _ in cols]
            body.append((title, values(ns, cols), source(ns), None))
        groups = heads or list(zip(names, [rk(ks) for _, ks in cols]))
        out = ["% generated by paper/scripts/retrieval.py", note, "\\begin{table}[H]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
               *tabular(groups, block(body, [], "", sum(len(ks) for _, ks in cols))), "\\end{table}", ""]
        return "\n".join(out)

    (PAPER / "tables/recipes_proposed.tex").write_text(recipes(proposed, ["SurveyMatch", "FrontierMatch", "ConsensusMatch"], "tab:recipes-proposed", "Every BoundedField recipe on the test splits of the proposed benchmarks, every method on the Nomic embedding."))
    (PAPER / "tables/recipes_sc.tex").write_text(recipes(sc, None, "tab:recipes-sc", "Every BoundedField recipe on ScholarCatalyst, scored inside the benchmark's own corpus over every answer, every method on the Nomic embedding.", sc_heads))
    (PAPER / "tables/recipes_existing.tex").write_text(recipes(existing, ["AutoScholarQuery", "RealScholarQuery", "SurGE"], "tab:recipes-existing", "Every BoundedField recipe on PaSa's AutoScholarQuery test split and RealScholarQuery, and SurGE, every method on the Nomic embedding."))


def transfer() -> None:
    cols = " & ".join(f"{name.replace('ScholarCatalyst ', 'SC ').replace('ScholarQuery', 'SQ')} R@{k}" for _, name, k, _ in BENCH)
    lines = ["% generated by paper/scripts/retrieval.py", "% benchmarks/data/runs/{cosine,bf,bfrefit}_<embedder>-<benchmark>.jsonl, and nomic's own recipe is bf_nomic", "\\begin{table}[H]",
             "\\caption{R@$k$ at the middle cutoff of every benchmark by embedder: cosine search alone, BoundedField with the recipes fit on Nomic and applied unchanged, and BoundedField with the recipes refit on each embedder. AutoSQ and RealSQ are PaSa's AutoScholarQuery and RealScholarQuery, and SC is ScholarCatalyst. Italics mark a BoundedField cell below that embedder's cosine search.}",
             "\\label{tab:transfer}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{l{'r' * len(BENCH)}}}", "\\toprule", f"Embedder & {cols} \\\\", "\\midrule"]
    for kind, title in (("cosine", "Cosine search"), ("bf", "BoundedField, Nomic's recipes"), ("bfrefit", "BoundedField, refit recipes")):
        lines.append(f"\\multicolumn{{{len(BENCH) + 1}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for e, label in EMBEDDERS:
            cells = []
            for b, _, k, _ in BENCH:
                v = run(f"{'bf' if kind == 'bfrefit' and e == 'nomic' else kind}_{e}-{b}", k)
                c = run(f"cosine_{e}-{b}", k)
                cells.append(f"\\textit{{{v:.3f}}}" if kind != "cosine" and v < c else f"{v:.3f}")
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if kind != "bfrefit" else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    (PAPER / "tables/transfer.tex").write_text("\n".join(lines))


def figure() -> None:
    for f in FONTS.glob("LinBiolinum_R*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Linux Biolinum O",
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.5,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.size": 0,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.3))
    ys = list(range(len(EMBEDDERS)))[::-1]
    for ax, (b, name, c, _) in zip(axes.flat, BENCH):
        cos = [run(embedder_run("cosine", e, b), c) for e, _ in EMBEDDERS]
        bf = [run(embedder_run("bfrefit", e, b), c) for e, _ in EMBEDDERS]
        for y, a, z in zip(ys, cos, bf):
            ax.plot([a, z], [y, y], color=RED, alpha=0.3, linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter(cos, ys, s=16, facecolors="white", edgecolors=BASE, linewidths=1.2, zorder=3, label="cosine")
        ax.scatter(bf, ys, s=16, color=RED, edgecolors="white", linewidths=0.8, zorder=4, label="BoundedField")
        for e, v in REPORTED_FIG.get(b, {}).items():
            y = ys[[k for k, _ in EMBEDDERS].index(e)]
            ax.scatter([v], [y], marker="D", s=16, color=BLUE, edgecolors="white", linewidths=0.8, zorder=5, label="cosine, reported by ScholarCatalyst")
        lo = min(cos + bf + list(REPORTED_FIG.get(b, {}).values()))
        hi = max(cos + bf + list(REPORTED_FIG.get(b, {}).values()))
        pad = (hi - lo) * 0.12
        top = max(range(len(bf)), key=lambda i: bf[i])
        ax.set_yticks(ys, [label for _, label in EMBEDDERS])
        ax.tick_params(axis="y", labelleft=True, labelsize=6.5, pad=1)
        ax.get_yticklabels()[top].set_fontweight("bold")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"{name}, R@{c}", fontsize=8, color=INK, pad=4)
        ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=6.5, length=2)
        ax.locator_params(axis="x", nbins=4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[1, 3].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    order = ("cosine", "BoundedField", "cosine, reported by ScholarCatalyst")
    fig.legend([seen[k] for k in order], order, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.4, h_pad=1.0)
    fig.savefig(PAPER / "figures/embedders.pdf")


tables()
transfer()
figure()
