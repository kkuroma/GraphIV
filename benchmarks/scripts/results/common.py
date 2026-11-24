"""
What every results script shares: the config, per-case scores, the significance tests and reported tables

A script imports this from its own directory, which python puts first on the path when running it
"""

import html
import json
import math
import re
import tomllib
from functools import cache
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from scipy.stats import wilcoxon

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

BENCH = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((Path(__file__).parent / "config.toml").read_text())
PATHS = {k: BENCH / v for k, v in CONFIG["paths"].items()}
COLORS = CONFIG["colors"]
SETS = CONFIG["sets"]
EMBEDDERS = CONFIG["embedders"]
ALPHA = CONFIG["significance"]["ALPHA"]
TUNES = PATHS["TUNES"]
RECIPES = CONFIG["table_main_recipes"]["recipes"]
TERMS = CONFIG["table_main_recipes"]["terms"]
# Each recipe's own benchmark, the first set that runs it
OWN = {r: next(s for s, v in SETS.items() if v["recipe"] == r) for r in RECIPES}
ARXIV = re.compile(r"\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7}")


def output(script: str) -> Path:
    """
    Where one script's files go, results/<name>/<script> with the extension left to the caller

    Args:
        script: The script's file stem, like figure_main_embedders, whose name drops the kind.

    Returns:
        The path without its extension, its directory created.
    """
    path = PATHS["RESULTS"] / script.split("_", 1)[1]
    path.mkdir(parents=True, exist_ok=True)
    return path / script


@cache
def rankings(run: str) -> tuple[dict, ...]:
    """
    Reads one test run's rankings.jsonl

    Args:
        run: The run's directory name under data/runs/test.

    Returns:
        One dict per case.
    """
    with open(PATHS["RUNS"] / run / "rankings.jsonl") as f:
        return tuple(map(json.loads, f))


def per_case(run: str, metric: str) -> dict[str, float]:
    """
    One run's per-case score as the run stored it

    Args:
        run: The run's directory name under data/runs/test.
        metric: recall@k or ndcg@k with k a stored cutoff.

    Returns:
        The score keyed by sample id.
    """
    kind, k = metric.split("@")
    return {r["sample_id"]: r[kind][k] for r in rankings(run)}


@cache
def cases(set_name: str) -> dict[str, dict]:
    """
    The cases of one test set, read from data/cases/<dataset>/<split>.jsonl

    Args:
        set_name: The set as runs name it, like scholarcatalyst_core.

    Returns:
        Each case keyed by sample id.
    """
    dataset, split = set_name.rsplit("_", 1)
    with open(PATHS["CASES"] / dataset / f"{split}.jsonl") as f:
        return {c["sample_id"]: c for c in map(json.loads, f)}


def ndcg(ranked: list[str], key: set[str], k: int) -> float:
    gain = sum(1 / math.log2(i + 2) for i, a in enumerate(ranked[:k]) if a in key)
    return gain / sum(1 / math.log2(i + 2) for i in range(min(len(key), k)))


def arxiv_only(run: str, set_name: str, metric: str) -> float:
    """
    A run's mean score against only the answers on arXiv, leaving out cases with none

    Args:
        run: The run's directory name under data/runs/test.
        set_name: The set the run ranked.
        metric: recall@k or ndcg@k.

    Returns:
        The mean over the cases that keep at least one answer.
    """
    kind, k = metric.split("@")
    n = int(k)
    scores = []
    for r in rankings(run):
        key = {a for a in cases(set_name)[r["sample_id"]]["key"] if ARXIV.fullmatch(a)}
        if not key:
            continue
        ranked = r["ranked"]
        scores.append(ndcg(ranked, key, n) if kind == "ndcg" else len(set(ranked[:n]) & key) / len(key))
    return sum(scores) / len(scores)


def one_sided(better: dict[str, float], worse: dict[str, float]) -> float:
    """
    Wilcoxon signed-rank p of the first run beating the second over their shared cases

    Args:
        better: Per-case scores of the run hypothesized higher.
        worse: Per-case scores of the other run.

    Returns:
        The one-sided p, 1.0 when every paired difference is zero.
    """
    ids = sorted(better.keys() & worse.keys())
    diffs = [better[i] - worse[i] for i in ids]
    if not any(diffs):
        return 1.0
    return float(wilcoxon(diffs, alternative="greater").pvalue)


def holm(pvalues: dict) -> dict:
    """
    Holm step-down adjustment over one family

    Args:
        pvalues: Raw p keyed by comparison.

    Returns:
        Adjusted p keyed by the same comparisons.
    """
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    out, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def html_table(path: Path, table_id: str) -> dict[str, list[str]]:
    """
    Parses one table of an arXiv html page into its rows

    Args:
        path: The html file.
        table_id: The id of the table's figure element, like S4.T3.

    Returns:
        Each row's cells keyed by its first cell.
    """
    page = path.read_text()
    start = page.index(f'id="{table_id}"')
    table = page[start : page.index("</table>", start)]
    rows = {}
    for tr in re.findall(r"<tr.*?</tr>", table, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if cells:
            rows[cells[0]] = cells
    return rows


def number(cell: str) -> float:
    """A reported cell as a float, dropping footnote marks like a trailing dagger"""
    return float(re.sub(r"[^\d.]", "", cell))


def report(tune: str) -> dict:
    """One tune's report.json, read from data/runs/tune/<tune>"""
    return json.loads((TUNES / tune / "report.json").read_text())


def weights(tune: str) -> dict[str, float]:
    """One tune's exponents keyed by lowercase term"""
    return {k.lower(): v for k, v in tomllib.loads((TUNES / tune / "recipe.toml").read_text())["weights"].items()}


def middle(set_name: str) -> str:
    """The metric a set is scored at in the middle of the paper's cutoffs"""
    return f"recall@{SETS[set_name]['cutoff']}"


def mean(run: str, metric: str) -> float:
    scores = per_case(run, metric)
    return sum(scores.values()) / len(scores)


def holm_tests(pairs: dict) -> dict:
    """
    Holm-adjusted one-sided Wilcoxon p over one family

    Args:
        pairs: Each comparison keyed freely, mapping to (better run, worse run, metric).

    Returns:
        The adjusted p keyed like pairs.
    """
    return holm({k: one_sided(per_case(a, m), per_case(b, m)) for k, (a, b, m) in pairs.items()})


def f3(v: float, bold: bool = False) -> str:
    t = f"{v:.3f}"
    return f"\\textbf{{{t}}}" if bold else t


def subtable(label: str, caption: str, groups: list[tuple[str, list[str]]], lines: list[str], first: str = "Method") -> list[str]:
    """
    One booktabs subtable with grouped column heads

    Args:
        label: The latex label.
        caption: The subcaption.
        groups: Each group's title and its column heads.
        lines: The body rows.
        first: The head of the first column.

    Returns:
        The latex lines.
    """
    n = sum(len(ks) for _, ks in groups)
    head = " & " + " & ".join(f"\\multicolumn{{{len(ks)}}}{{c}}{{{name}}}" for name, ks in groups) + " \\\\"
    rule, start = "", 2
    for _, ks in groups:
        rule += f"\\cmidrule(lr){{{start}-{start + len(ks) - 1}}}"
        start += len(ks)
    cuts = f"{first} & " + " & ".join(k for _, ks in groups for k in ks) + " \\\\"
    wrap = bool(caption)
    return [
        *(["\\begin{subtable}{\\textwidth}", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"] if wrap else []),
        f"\\begin{{tabular}}{{l{'r' * n}}}",
        "\\toprule",
        head,
        rule,
        cuts,
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        *(["\\end{subtable}"] if wrap else []),
    ]


def blocks_table(first: str, heads: list[str], blocks: list[tuple[str, list[str]]]) -> list[str]:
    """
    One tabular whose rows fall in titled blocks sharing a column head

    Args:
        first: The head of the first column.
        heads: The other column heads.
        blocks: Each block's italic title and its body rows.

    Returns:
        The latex lines.
    """
    n = len(heads) + 1
    out = [f"\\begin{{tabular}}{{l{'r' * len(heads)}}}", "\\toprule", f"{first} & " + " & ".join(heads) + " \\\\"]
    for title, rows in blocks:
        out += ["\\midrule", f"\\multicolumn{{{n}}}{{l}}{{\\textit{{{title}}}}} \\\\", *rows]
    return out + ["\\bottomrule", "\\end{tabular}"]


def float_table(stem: str, subs: list[list[str]], caption: str, label: str, size: str = "footnotesize", star: bool = True) -> str:
    """A table of stacked tabulars with one caption, headed by the line saying which script wrote it, one column wide unless star"""
    env = "table*" if star else "table"
    body = []
    for i, s in enumerate(subs):
        body += (["\\vspace{6pt}"] if i else []) + s
    lines = [
        f"% generated by benchmarks/scripts/results/{stem}.py from benchmarks/data/runs",
        f"\\begin{{{env}}}[!htbp]",
        "\\centering",
        f"\\{size}",
        "\\setlength{\\tabcolsep}{4pt}",
        *body,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\end{{{env}}}",
        "",
    ]
    return "\n".join(lines)


def save(stem: str, tex: str, data: dict) -> None:
    out = output(stem)
    out.with_suffix(".tex").write_text(tex)
    out.with_suffix(".json").write_text(json.dumps(data, indent=1))


def fmt(v: float | None, bold: bool = False, under: bool = False, sup: str = "") -> str:
    if v is None:
        return "--"
    t = f"{v:.2f}" if round(v, 2) == v else f"{v:.3f}"
    if bold:
        t = f"\\textbf{{{t}}}"
    if under:
        t = f"\\underline{{{t}}}"
    return t + (f"$^{{{sup}}}$" if sup else "")


def near(x: float | None, t: float | None) -> bool:
    return x is not None and t is not None and abs(x - t) < 5e-4


def style() -> None:
    for f in Path(CONFIG["paths"]["FONTS"]).glob("LinBiolinum_R*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update(
        {
            "font.family": "Linux Biolinum O",
            "font.size": 8,
            "axes.edgecolor": COLORS["MUTED"],
            "axes.linewidth": 0.5,
            "xtick.color": COLORS["MUTED"],
            "ytick.color": COLORS["INK"],
            "xtick.major.width": 0.5,
            "ytick.major.size": 0,
            "pdf.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )
