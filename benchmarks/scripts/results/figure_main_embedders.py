"""
Figure main_embedders: cosine, PPR and BoundedField recall by embedder on every test set

A star after an embedder marks BoundedField significantly above cosine, a cross above PPR
The ScholarCatalyst diamonds are parsed from that paper's html, every other number from data/runs/test
"""

import json
from pathlib import Path

from common import (
    ALPHA,
    BENCH,
    COLORS,
    CONFIG,
    EMBEDDERS,
    SETS,
    holm,
    html_table,
    number,
    one_sided,
    output,
    per_case,
    plt,
    style,
)

STEM = Path(__file__).stem
SPEC = CONFIG[STEM]


def reported() -> dict[str, dict[str, float]]:
    """
    The cosine recall ScholarCatalyst reported for the Qwen3 embedders

    Returns:
        Recall keyed by set and then by embedder.
    """
    rows = html_table(BENCH / SPEC["reported_html"], SPEC["reported_table"])
    return {s: {e: number(rows[label][col]) for e, label in SPEC["reported_rows"].items()} for s, col in SPEC["reported_columns"].items()}


def collect() -> dict:
    """
    Means and significance for every set and embedder

    Returns:
        A dict keyed by set and then embedder holding each method's mean recall and the raw and Holm p.
    """
    out: dict = {s: {} for s in SETS}
    raw: dict[str, dict[tuple, float]] = {"cosine": {}, "ppr": {}}
    for s, spec in SETS.items():
        metric = f"recall@{spec['cutoff']}"
        for e in EMBEDDERS:
            runs = {
                "cosine": f"{e}-{s}-none",
                "ppr": f"{e}-{s}-{spec['recipe']}_ppr_seed0",
                "bf": f"{e}-{s}-{spec['recipe']}_bf_seed0",
            }
            scores = {m: per_case(r, metric) for m, r in runs.items()}
            out[s][e] = {"runs": runs, **{m: sum(v.values()) / len(v) for m, v in scores.items()}}
            for m in raw:
                raw[m][(s, e)] = one_sided(scores["bf"], scores[m])
    for m, family in raw.items():
        for (s, e), p in holm(family).items():
            out[s][e][f"p_{m}"] = family[(s, e)]
            out[s][e][f"holm_{m}"] = p
    return out


def marks(row: dict) -> str:
    """
    The superscript after an embedder's name, a star for BoundedField above cosine and a cross above PPR

    Args:
        row: One embedder's entry from collect.

    Returns:
        A mathtext superscript, or an empty string when neither test passes.
    """
    sup = ("\\star" if row["holm_cosine"] <= ALPHA else "") + ("\\times" if row["holm_ppr"] <= ALPHA else "")
    return f"$^{{{sup}}}$" if sup else ""


def figure(data: dict, quoted: dict[str, dict[str, float]]) -> None:
    """
    Draws one panel per set with a row per embedder and writes the pdf and a png preview

    Args:
        data: The output of collect.
        quoted: The output of reported.
    """
    style()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.4))
    ys = list(range(len(EMBEDDERS)))[::-1]
    for ax, (s, spec) in zip(axes.flat, SETS.items(), strict=True):
        rows = [data[s][e] for e in EMBEDDERS]
        cos, ppr, bf = ([r[m] for r in rows] for m in ("cosine", "ppr", "bf"))
        extra = quoted.get(s, {})
        for y, r in zip(ys, rows, strict=True):
            lo, hi = min(r["cosine"], r["ppr"], r["bf"]), max(r["cosine"], r["ppr"], r["bf"])
            ax.plot([lo, hi], [y, y], color=COLORS["RULE"], linewidth=2, zorder=1)
        ax.scatter(cos, ys, s=16, facecolors="white", edgecolors=COLORS["BASE"], linewidths=1.2, zorder=3, label="cosine")
        ax.scatter(ppr, ys, s=16, color=COLORS["AMBER"], edgecolors="white", linewidths=0.8, zorder=3, label="PPR")
        ax.scatter(bf, ys, s=16, color=COLORS["RED"], edgecolors="white", linewidths=0.8, zorder=4, label="BoundedField")
        for e, v in extra.items():
            y = ys[list(EMBEDDERS).index(e)]
            ax.scatter(
                [v],
                [y],
                marker="D",
                s=14,
                color=COLORS["BLUE"],
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                label="cosine, reported by ScholarCatalyst",
            )
        values = cos + ppr + bf + list(extra.values())
        pad = (max(values) - min(values)) * 0.1
        ax.set_xlim(min(values) - pad, max(values) + pad)
        ax.set_yticks(ys, [label + marks(r) for label, r in zip(EMBEDDERS.values(), rows, strict=True)])
        ax.tick_params(axis="y", labelsize=6.5, pad=1)
        ax.get_yticklabels()[max(range(len(bf)), key=bf.__getitem__)].set_fontweight("bold")
        ax.set_title(f"{spec['label']}, R@{spec['cutoff']}", fontsize=8, color=COLORS["INK"], pad=4)
        ax.grid(axis="x", color=COLORS["GRID"], linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=6.5, length=2)
        ax.locator_params(axis="x", nbins=4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    seen: dict = {}
    for ax in axes.flat:
        for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(label, h)
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="lower center",
        ncol=len(seen),
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
        handletextpad=0.2,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.2, h_pad=1.0)
    fig.savefig(output(STEM).with_suffix(".pdf"))
    fig.savefig(output(STEM).with_suffix(".png"), dpi=200)


def alt(data: dict, quoted: dict[str, dict[str, float]]) -> str:
    """
    The figure's alt text as a latex Description, every value read from data

    Args:
        data: The output of collect.
        quoted: The output of reported.

    Returns:
        The \\Description command.
    """
    clear = [SETS[s]["label"] for s in SETS if min(r["bf"] for r in data[s].values()) > max(r["cosine"] for r in data[s].values())]
    parts = [
        f"Eight dot-line panels in two rows, one per benchmark, each listing eight embedders from top to bottom by size: {', '.join(EMBEDDERS.values())}."
        " Each row joins a hollow gray dot, recall with cosine search, an amber dot, recall with PPR, and a red dot, recall with BoundedField,"
        " and the embedder with the highest BoundedField recall in each panel is in bold."
        " A star beside an embedder marks BoundedField significantly above its cosine search and a cross significantly above its PPR."
        f" On {', '.join(clear)}, every BoundedField dot sits right of every cosine dot."
        " On the two ScholarCatalyst panels, blue diamonds mark the cosine recall ScholarCatalyst reported for Qwen3 4B and 8B."
        " Values, cosine to PPR to BoundedField, are:"
    ]
    for s, v in SETS.items():
        best = max(data[s], key=lambda e: data[s][e]["bf"])
        vals = "; ".join(f"{EMBEDDERS[e]} {c['cosine']:.3f} to {c['ppr']:.3f} to {c['bf']:.3f}" for e, c in data[s].items())
        note = "".join(f"; reported cosine {x:.2f} for {EMBEDDERS[e]}" for e, x in quoted.get(s, {}).items())
        parts.append(f"{v['label']} at R@{v['cutoff']}: {vals}; best with BoundedField, {EMBEDDERS[best]}{note}.")
    return "\\Description{" + " ".join(parts) + "}\n"


def main() -> None:
    data = collect()
    quoted = reported()
    figure(data, quoted)
    output(STEM).with_suffix(".alt.tex").write_text(alt(data, quoted))
    (output(STEM).with_suffix(".json")).write_text(json.dumps({"sets": data, "reported": quoted}, indent=1))


if __name__ == "__main__":
    main()
