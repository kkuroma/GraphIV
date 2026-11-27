"""
The report object rendered as markdown, with a table, its coverage, a poset and a checked search
"""

from src.report.check import Held
from src.report.render import render

ROW = {
    "arxiv_id": "2503.18813",
    "method": "CaMeL",
    "cells": {
        "split": {"value": "yes", "quote": "q", "label": "Debenedetti3"},
        "trained": {"value": "true", "quote": "q", "label": "Debenedetti4"},
    },
}
TABLE = {
    "schema": {
        "columns": [
            {"name": "split", "kind": "choice", "description": "x", "options": ["yes", "no"]},
            {"name": "trained", "kind": "bool", "description": "x", "options": []},
        ]
    },
    "rows": [ROW],
}
POSET = {
    "id": "P1",
    "papers": [
        {"arxiv_id": "2302.12173", "title": "Indirect injection", "name": "", "date": "2023-02-23"},
        {
            "arxiv_id": "2503.18813",
            "title": "CaMeL: Defeating",
            "name": "CaMeL",
            "date": "2025-03-24",
        },
    ],
    "edges": [
        {
            "citer": "2503.18813",
            "cited": "2302.12173",
            "intent": "extends",
            "sentence": "s",
            "label": "Debenedetti5",
            "implied": False,
        }
    ],
    "focus": ["2302.12173", "2503.18813"],
}
HELD = Held(
    evidence={},
    sources={},
    read={"2503.18813"},
    served={"2503.18813", "2404.13208"},
    searches={
        "graph rerankers": {
            "query": "graph rerankers",
            "mode": "focused",
            "papers": [{"arxiv_id": "2404.13208", "title": "Instruction Hierarchy"}],
        }
    },
    tables={"T1": TABLE},
    posets={"P1": POSET, "P2": POSET},
    titles={"2404.13208": "The Instruction Hierarchy (2024)"},
    owners={"Debenedetti3": "2503.18813"},
    names={"2503.18813": "CaMeL"},
)
REPORT = {
    "question": "Does separating flows stop injection?",
    "tldr": {"text": "Yes, on AgentDojo", "cites": ["Debenedetti3"]},
    "overview": [{"type": "finding", "text": "CaMeL splits the flows.", "cites": ["Debenedetti3"]}],
    "sections": [
        {
            "heading": "Defenses",
            "blocks": [
                {"type": "table", "table": "T1"},
                {"type": "coverage", "table": "T1", "rows": "split", "columns": ["trained"]},
                {"type": "lineage", "poset": "P1"},
                {"type": "open", "text": "No graph reranker for code", "search": "Graph rerankers"},
            ],
        }
    ],
    "further_reading": ["2404.13208", "2503.18813"],
}


def test_every_part_is_drawn_and_the_unplaced_poset_goes_last():
    markdown, tldr, unplaced = render(REPORT, HELD)
    assert tldr == "Yes, on AgentDojo [Debenedetti3]."
    assert "CaMeL [Debenedetti3] splits the flows." in markdown
    assert "| CaMeL (2503.18813) | yes [Debenedetti3] | true [Debenedetti4] |" in markdown
    assert "| yes | 1 | 1 [Debenedetti4] |" in markdown
    assert "| no | 0 | 0 |" in markdown
    assert "**Poset P1**" in markdown and "p1 -->|extends| p0" in markdown
    assert "- CaMeL extends Indirect injection: s [Debenedetti5]" in markdown
    assert 'Checked by a focused search for "graph rerankers", which returned 1 papers' in markdown
    assert unplaced == ["Poset P2"]
    assert markdown.index("## Also from this study") < markdown.index("**Poset P2**")
    assert "- 2404.13208 The Instruction Hierarchy (2024)" in markdown
    assert "2503.18813" not in markdown.split("## Further reading")[1]


def test_inline_markers_stay_and_listed_ones_go_beside_names():
    block = {
        "type": "finding",
        "text": "Flows split [Debenedetti5] , and CaMeL wins .",
        "cites": ["Debenedetti3"],
    }
    report = {**REPORT, "overview": [block]}
    markdown, _, _ = render(report, HELD)
    assert "Flows split [Debenedetti5], and CaMeL [Debenedetti3] wins." in markdown
