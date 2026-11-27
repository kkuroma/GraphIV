"""
Session runs: an issued id with its effort, a run read back after a restart, and tables by shape
"""

from pathlib import Path

from src.mcp import runs
from src.mcp.runs import Evidence


def columns(options: list[str]) -> dict:
    return {
        "columns": [
            {"name": "kind", "kind": "choice", "description": "a", "options": options},
            {"name": "trained", "kind": "bool", "description": "b", "options": []},
        ]
    }


def test_a_run_survives_a_restart_with_its_effort_and_draft(tmp_path: Path):
    run = runs.start(tmp_path, "dense retrieval", "deep")
    label = run.keep(
        Evidence(arxiv_id="2101.00001", pid="p3", section="", text="t", quote="q"), "Lee"
    )
    run.draft = {"question": "q"}
    run.log("read_paper", {"q": 1}, "reply 2101.00001", 0.0)
    runs.RUNS.clear()
    back = runs.find(tmp_path, run.session)
    assert back is not None
    assert back.evidence[label].pid == "p3"
    assert back.served == {"2101.00001"}
    assert (back.effort, back.draft) == ("deep", {"question": "q"})


def test_a_revised_schema_adds_rows_to_the_same_table():
    run = runs.Run(session="s-1")
    first = run.add_row(columns(["dense", "sparse"]), {"arxiv_id": "1", "method": "a", "cells": {}})
    second = run.add_row(
        columns(["sparse", "graph"]), {"arxiv_id": "2", "method": "b", "cells": {}}
    )
    again = run.add_row(columns(["dense"]), {"arxiv_id": "1", "method": "a2", "cells": {}})
    assert first == second == again == "T1"
    table = run.table("t1")
    assert table is not None
    assert table["schema"]["columns"][0]["options"] == ["dense", "sparse", "graph"]
    assert [r["method"] for r in table["rows"]] == ["b", "a2"]


def test_an_unrecorded_run_shows_no_marker():
    assert runs.Run(session="", tracked=False).mark("Lee1") == ""
    assert runs.Run(session="s-1").mark("Lee1") == " [Lee1]"
