"""
Draws research-pipeline.png: one question through the six MCP tools, each with what it returned
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import example_run
from matplotlib.patches import FancyBboxPatch
from style import (
    GREEN,
    INK,
    LEFT,
    MUTED,
    OUTLINE,
    PANEL,
    PURPLE,
    RIGHT,
    ROOT,
    Page,
    apply_theme,
    tint,
    write_runs,
)

TOOL = 2.55
SIZE = 10.5
LINE = 0.205
# how many of each list a row writes out
SHOWN = 3
# what each tool answers, in the words of src/mcp's table
ANSWERS = {
    "refine": "what focused question this corpus can answer",
    "identify_areas": "which areas have to be searched apart",
    "retrieve": "what one area is, graded by role",
    "mine": "what it argues with and how it got here",
    "read_paper": "a schema the caller invented, filled from full text",
    "write_report": "the document, grounded and written to disk",
}


def plain(markdown: str) -> str:
    """
    The report's TL;DR with each arXiv link reduced to its id

    Args:
        markdown: The TL;DR as the report writes it.

    Returns:
        The text.
    """
    return re.sub(r"\[arXiv:([^\]]+)\]\([^)]*\)", r"[\1]", markdown)


def rows(run: dict[str, Any]) -> list[tuple[str, str, str, list[tuple[str, str]]]]:
    """
    What each tool returned, as lines of text each marked plain, muted or emphasized

    Args:
        run: The example run.

    Returns:
        Per tool, its name, its hallmark, its color and its lines.
    """
    c, pools = run["counts"], [area["steps"] for area in run["bounded"]]
    areas = [(f"{i}. {a['question']}", "text") for i, a in enumerate(run["areas"], 1)]
    retrieved = [
        (
            f"area {i}: {p[0]:,} by cosine, {p[1]:,} with what the core cites, {p[2]:,} with what"
            f" cites it, ranked by the graph, then a model kept {a['kept']}",
            "text",
        )
        for i, (p, a) in enumerate(zip(pools, run["areas"], strict=True), 1)
    ]
    top = list(dict.fromkeys(t for a in run["areas"] for t in a["top"]))[:SHOWN]
    magnets = [f"{n['title'].split(':')[0]} ({n['magnet']:.2f})" for n in run["nodes"][:SHOWN]]
    ahead = [f"{s['label']}, {' and '.join(s['intents'])}" for s in run["successors"][:SHOWN]]
    sample = [
        f"{r['name']} {r['year']}: {r['modeling_paradigm']}, {r['top_performance']}"
        for r in run["methods"]["sample"][:SHOWN]
    ]
    return [
        (
            "refine",
            "",
            "",
            [(f"tagged {run['mode']}, with the corpus capped at {run['asked']}", "text")],
        ),
        ("identify_areas", "", "", areas),
        (
            "retrieve",
            "BoundedField",
            GREEN,
            [*retrieved, ("kept first: " + "; ".join(top), "muted")],
        ),
        (
            "mine",
            "LeyLine",
            PURPLE,
            [
                (
                    f"{c['union']} papers and {c['edges']:,} citations, {c['magnets']} magnets,"
                    f" {c['lineages']} walked paths and {c['successors']} successors",
                    "text",
                ),
                ("strongest magnets: " + "; ".join(magnets), "muted"),
                ("lineages: " + "; ".join(run["lineages"]), "muted"),
                ("successors: " + "; ".join(ahead), "muted"),
            ],
        ),
        (
            "read_paper",
            "",
            "",
            [
                (
                    f"{run['methods']['rows']} full texts into a table of"
                    f" {len(run['methods']['columns'])} columns: "
                    + ", ".join(col.replace("_", " ") for col in run["methods"]["columns"]),
                    "text",
                ),
                *[(line, "muted") for line in sample],
                (
                    f"{len(run['methods']['untried'])} pairs nobody tried: "
                    + "; ".join(run["methods"]["untried"][:SHOWN])
                    + "...",
                    "muted",
                ),
            ],
        ),
        (
            "write_report",
            "",
            "",
            [
                (f"“{plain(run['answer'])}”", "quote"),
                (
                    f"{run['cited']} arXiv ids cited, {run['corrected']} corrected and"
                    f" {run['cut']} removed by the check against the corpus",
                    "muted",
                ),
            ],
        ),
    ]


def draw(config: Path, out: Path) -> None:
    """
    research-pipeline.png: the question, then one row per tool

    Args:
        config: The config the example run's cache falls back to.
        out: Where the figure is written.
    """
    run = example_run.load(config)
    page = Page(13.3)
    page.title(
        [
            [("GraphIV's", True), ("Automated Research Pipeline:", False)],
            [("One Question Through Six MCP Tools", False)],
        ]
    )
    page.panel_title(LEFT, 2.1, f"The {run['mode']} run of {run['asked']}", 1)
    asked = page.wrapped(f"“{run['question']}”", RIGHT - LEFT - 0.4, 12)
    high = 0.25 + 0.24 * len(asked)
    page.ax.add_patch(
        FancyBboxPatch(
            (LEFT, 2.35),
            RIGHT - LEFT,
            high,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=PANEL,
            edgecolor=OUTLINE,
            linewidth=1.2,
            zorder=3,
        )
    )
    for i, line in enumerate(asked):
        page.ax.text(LEFT + 0.2, 2.35 + 0.35 + 0.24 * i, line, fontsize=12, zorder=4)
    text_x = LEFT + TOOL + 0.35
    room = RIGHT - text_x
    y = 2.35 + high + 0.35
    above = 2.35 + high
    for name, alias, hue, lines in rows(run):
        wrapped: list[tuple[str, str]] = []
        for text, kind in lines:
            wrapped += [(piece, kind) for piece in page.wrapped(text, room, SIZE)]
        answer = page.wrapped(ANSWERS[name], TOOL - 0.4, 10)
        tall = max(0.5 + 0.19 * len(answer), 0.3 + LINE * len(wrapped))
        page.arrow((LEFT + TOOL / 2, above), (LEFT + TOOL / 2, y))
        page.ax.add_patch(
            FancyBboxPatch(
                (LEFT, y),
                TOOL,
                tall,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=tint(hue, 0.1) if hue else "white",
                edgecolor=hue or OUTLINE,
                linewidth=1.8 if hue else 1.2,
                zorder=3,
            )
        )
        runs: list[tuple[str, bool | str]] = [(name, False)]
        if alias:
            runs += [("·", MUTED), (alias, hue)]
        write_runs(page.fig, *page.at(LEFT + 0.2, y + 0.33), runs, 12, "semibold")
        for i, piece in enumerate(answer):
            page.ax.text(LEFT + 0.2, y + 0.56 + 0.19 * i, piece, fontsize=10, color=MUTED, zorder=4)
        for i, (piece, kind) in enumerate(wrapped):
            color = MUTED if kind == "muted" else INK
            style = "italic" if kind == "quote" else "normal"
            page.ax.text(text_x, y + 0.3 + LINE * i, piece, fontsize=SIZE, color=color, style=style)
        above = y + tall
        y = above + 0.3
    notes = [
        f"The {run['mode']} run of {run['asked']} at the {run['effort']} effort, read back from its"
        " own run directory. The retrieve row counts BoundedField rerun on each area at that date.",
    ]
    page.save(out, y + 0.15, notes)


def main() -> None:
    """
    Reads the example run, then draws the figure to the path it is given
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--out", required=True, help="where the figure is written")
    _ = ap.add_argument("--config_path", default=str(ROOT / "config.toml"))
    args = ap.parse_args()
    apply_theme()
    draw(Path(str(args.config_path)), Path(str(args.out)))


if __name__ == "__main__":
    main()
