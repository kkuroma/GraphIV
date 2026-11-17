"""
The cold-start measurement behind LeyLine's successors, read off the findings file that kept it

Its code was deleted with the scratchpad task on 2026-09-26, so the findings file is the source
The file sits in assets/data as git last held it, at de451c2~1:experiments/findings/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

FINDINGS = "assets/data/frontier-system-REPORT.md"
# the topic each short name of the findings file stands for, as docs/03_tools.md lists them
TOPICS = {
    "tabular": "Tabular foundation models",
    "mechinterp": "Mechanistic interpretability",
    "litsearch": "Retrieval for scientific literature",
    "diffusionlm": "Diffusion language models",
    "testtime": "Test-time compute and reasoning",
}


def table(text: str, header: str) -> list[list[str]]:
    """
    One markdown table out of a document, found by its header row's first cell

    Args:
        text: The document.
        header: The first cell of the table's header row.

    Returns:
        The body rows, each a list of cells.
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {header} |"))
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def load(root: Path) -> dict[str, Any]:
    """
    Every topic's Unseen Newcomers and how many each ranking reached, plus the dates they hold at

    Args:
        root: The repository.

    Returns:
        The topics' rows and the experiment's configuration.
    """
    text = (root / FINDINGS).read_text(encoding="utf-8")
    topics = [
        {
            "topic": TOPICS[row[0]],
            "window": int(row[1].replace(",", "")),
            "newcomers": int(row[2]),
            "graph": int(row[3]),
            "successor_top": int(row[4]),
            "successor_set": int(row[5]),
            "best_rank": int(row[6]),
        }
        for row in table(text, "persona")
    ]
    setup = {row[0]: row[1] for row in table(text, "item")}
    return {"topics": topics, "setup": setup}
