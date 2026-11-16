import json
import sys
from pathlib import Path

RUN = Path("/Vault/graphiv/runs/2026-11-09-045354-how-speculative-decoding-for-llm-inference-developed-since/run.json")
HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
SHORT = {
    "2302.01318": "Spec. Sampling",
    "2211.17192": "Spec. Decoding",
    "2401.15077": "EAGLE",
    "2401.10774": "Medusa",
    "2406.16858": "EAGLE-2",
}
GROUPS = {
    "textmatch": ("cos", "bm25"),
    "cites": ("anc", "anc2", "fro", "fro2", "anc_decay", "fro_decay", "cocite", "couple"),
    "recent": ("rec",),
}
LABEL_X = 3.45
BAR_X = 5.55
BAR_W = 2.1
ROW_TOP = 3.22
ROW_GAP = 0.34


def bars(run: dict) -> str:
    survey = next(s for s in run["searches"] if s["mode"] == "survey")
    rows = survey["papers"][:5]
    totals = [sum(max(v, 0) for v in p["terms"].values()) for p in rows]
    scale = BAR_W / max(totals)
    year = {n["arxiv_id"]: n["date"][2:4] for n in run["graph"]["nodes"]} | {a: p["date"][2:4] for a, p in run["papers"].items()}
    out = []
    for i, p in enumerate(rows):
        y = ROW_TOP - i * ROW_GAP
        name = SHORT[p["arxiv_id"]]
        out.append(f"\\node[row] at ({LABEL_X},{y:.2f}) {{{i + 1}\\ \\ {name} '{year[p['arxiv_id']]}}};")
        x = BAR_X
        for group, keys in GROUPS.items():
            w = sum(p["terms"].get(k, 0) for k in keys) * scale
            out.append(f"\\fill[{group}] ({x:.3f},{y - 0.11:.2f}) rectangle ({x + w - 0.02:.3f},{y + 0.11:.2f});")
            x += w
    return "\n".join(out)


run = json.loads(RUN.read_text())
template = (HERE / (sys.argv[1] if len(sys.argv) > 1 else "teaser.in.tex")).read_text()
(FIGURES / "teaser.tex").write_text(template.replace("%BARS%", bars(run)))
