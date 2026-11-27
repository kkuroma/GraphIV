"""
The report object check, patches and the deep contract, against a small hand-made session
"""

from typing import Any

from src.report.check import (
    Held,
    check,
    contract,
    lined,
    name_of,
    numbers_in,
    patch,
    place_tables,
)

EVIDENCE = {
    "E1": "CaMeL extracts the control and data flows from the user query.",
    "E2": "The number of successful attacks drops from 300 to 0 with CaMeL.",
}
SEARCH = {"query": "graph rerankers", "mode": "focused", "papers": [{"arxiv_id": "2404.13208"}]}
POSET = {
    "id": "P1",
    "papers": [{"arxiv_id": "2302.12173"}, {"arxiv_id": "2503.18813"}],
    "edges": [
        {"citer": "2503.18813", "cited": "2302.12173", "intent": "critiques", "label": "E3"},
    ],
    "focus": ["2302.12173", "2503.18813"],
}
TABLE = {
    "schema": {
        "columns": [
            {"name": "split", "kind": "choice", "description": "x", "options": ["yes", "no"]},
            {"name": "trained", "kind": "bool", "description": "x", "options": []},
        ]
    },
    "rows": [],
}


def held(read: set[str] | None = None, sources: dict[str, str] | None = None) -> Held:
    return Held(
        evidence=EVIDENCE,
        sources=sources or {},
        read={"2503.18813"} if read is None else read,
        served={"2503.18813", "2404.13208"},
        searches={"graph rerankers": SEARCH},
        tables={"T1": TABLE},
        posets={"P1": POSET},
    )


def report(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": "Does separating flows stop injection?",
        "tldr": {"text": "Attacks fall from 300 to 0.", "cites": ["E2"]},
        "overview": [{"type": "finding", "text": "CaMeL separates the flows.", "cites": ["E1"]}],
        "sections": [
            {"heading": "Defenses", "blocks": list(blocks) or [{"type": "inference", "text": "x"}]}
        ],
    }


def faults(r: Any, h: Held | None = None) -> list[tuple[str, str]]:
    return [(f["path"], f["kind"]) for f in check(r, h or held()).faults]


def test_a_cited_report_passes():
    assert check(report(), held()).passed


def test_each_fault_names_the_path_of_its_field():
    bad = report(
        {"type": "finding", "text": "Attacks fall from 300 to 70.", "cites": ["E2"]},
        {"type": "finding", "text": "CaMeL separates flows.", "cites": ["E9"]},
        {"type": "finding", "text": "CaMeL separates flows."},
    )
    assert faults(bad) == [
        ("sections.0.blocks.0.text", "number not in its paragraph"),
        ("sections.0.blocks.1.cites", "unknown marker"),
        ("sections.0.blocks.2.cites", "cites nothing"),
    ]


def test_markers_left_in_the_text_count_as_cites():
    assert check(
        report({"type": "finding", "text": "Attacks fall from 300 to 0 [E2]."}), held()
    ).passed


def test_a_name_passes_in_a_cited_paper_or_a_read_paper_title():
    sources = {"E1": "camel extracts the control and data flows", "E2": "attacks drop"}
    line = {"type": "finding", "text": "CaMeL explains flows with LIME.", "cites": ["E1"]}
    assert faults(report(line), held(sources=sources)) == [
        ("sections.0.blocks.0.text", "name not in the papers cited")
    ]
    known = held(sources=sources)
    known.known = "lime: local explanations of any classifier"
    assert check(report(line), known).passed


def test_an_open_block_needs_its_search_run_and_a_nearest_hit_read():
    block = {"type": "open", "text": "No graph reranker for code.", "search": "Graph rerankers"}
    assert faults(report(block)) == [("sections.0.blocks.0.search", "nearest hits never read")]
    assert check(report(block), held(read={"2503.18813", "2404.13208"})).passed
    numbered = {**block, "text": "Rerankers reach 97% on code."}
    assert faults(report(numbered), held(read={"2404.13208"})) == [
        ("sections.0.blocks.0.cites", "cites nothing")
    ]
    unrun = {"type": "open", "text": "x", "search": "dense rerankers"}
    assert faults(report(unrun)) == [("sections.0.blocks.0.search", "search never run")]


def test_placed_ids_must_exist_and_a_coverage_pivots_choice_by_bools():
    assert faults(report({"type": "table", "table": "T2"})) == [
        ("sections.0.blocks.0.table", "no such table")
    ]
    assert faults(report({"type": "lineage", "poset": "P1"})) == []
    assert faults(report({"type": "lineage", "poset": "P2"})) == [
        ("sections.0.blocks.0.poset", "no such poset")
    ]
    stray = {"type": "lineage", "poset": "P1", "papers": ["2503.18813", "1706.03762"]}
    assert faults(report(stray)) == [("sections.0.blocks.0.papers", "papers not in the poset")]
    good = {"type": "coverage", "table": "T1", "rows": "split", "columns": ["trained"]}
    assert faults(report(good)) == []
    flipped = {
        "type": "coverage",
        "table": "T1",
        "rows": "trained",
        "columns": ["split", "trained"],
    }
    assert [k for _, k in faults(report(flipped))] == [
        "not a choice field",
        "not bool fields or one choice field",
    ]


def test_a_deep_report_missing_parts_gets_one_fault_at_a_new_section():
    r = report()
    verdict = check(r, held())
    contract(r, held(), "deep", False, verdict, [])
    assert [(f["path"], f["kind"]) for f in verdict.faults] == [
        ("sections.1", "deep effort needs more blocks")
    ]
    quick = check(r, held())
    contract(r, held(), "quick", False, quick, ["2404.13208"])
    assert quick.passed
    unread = check(r, held())
    contract(r, held(), "deep", True, unread, ["2404.13208"])
    assert ("reads", "deep effort needs these papers read") in [
        (f["path"], f["kind"]) for f in unread.faults
    ]


def test_a_patch_sets_paths_appends_deletes_and_replaces_whole_lists():
    r = report({"type": "inference", "text": "a"}, {"type": "inference", "text": "b"})
    out, bad = patch(
        r,
        {
            "sections/0/blocks/1": None,
            "sections.0.blocks.2": {"type": "inference", "text": "c"},
            "sections.0.heading": "Flows",
            "nowhere.3": 1,
        },
    )
    assert [b["text"] for b in out["sections"][0]["blocks"]] == ["a", "c"]
    assert out["sections"][0]["heading"] == "Flows"
    assert bad == ["nowhere.3"]
    assert r["sections"][0]["heading"] == "Defenses"
    grown, bad = patch(
        r, {"sections.1.heading": "Gaps", "sections.1.blocks": [{"type": "inference", "text": "d"}]}
    )
    assert not bad and grown["sections"][1] == {
        "heading": "Gaps",
        "blocks": [{"type": "inference", "text": "d"}],
    }


def test_a_whole_overview_replaces_even_when_shorter():
    r = report()
    r["overview"] = [{"type": "inference", "text": t} for t in ("a", "b", "c")]
    out, bad = patch(r, {"overview": [{"type": "inference", "text": "b2"}]})
    assert not bad and [b["text"] for b in out["overview"]] == ["b2"]


def test_years_and_separators_in_numbers():
    assert numbers_in("1,804 videos in 2023 at 59.17%") == {"1804", "59.17"}
    assert numbers_in("Qwen3-VL on the J1-J2 model in 1D, 4 runs, 20x20 lattice") == set()


def test_a_rounded_number_is_backed_by_its_paragraph():
    h = held()
    h.evidence = {**EVIDENCE, "E3": "SRCC reaches 0.891 and PLCC 0.914 on the benchmark."}
    ok = {"type": "finding", "text": "SRCC reaches 0.89.", "cites": ["E3"]}
    off = {"type": "finding", "text": "SRCC reaches 0.95.", "cites": ["E3"]}
    assert check(report(ok), h).passed
    assert not check(report(off), h).passed


def test_a_deep_report_without_a_table_gets_one_with_its_coverage():
    drawn = place_tables(report(), held())
    assert drawn["sections"][-1] == {
        "heading": "Comparison",
        "blocks": [
            {"type": "table", "table": "T1"},
            {"type": "coverage", "table": "T1", "rows": "split", "columns": ["trained"]},
        ],
    }
    placed = report({"type": "table", "table": "T1"})
    assert place_tables(placed, held()) == placed
    verdict = check(drawn, held())
    contract(drawn, held(), "deep", True, verdict, [], at=1)
    assert [f["path"] for f in verdict.faults] == ["sections.1"]


def test_a_saved_fault_carries_the_text_it_sits_in():
    bad = report({"type": "finding", "text": "Attacks fall from 300 to 70.", "cites": ["E2"]})
    kept = lined(check(bad, held()), bad)
    fault = kept["faults"][0]
    assert (fault["line"], fault["where"]) == ("Attacks fall from 300 to 70", "70")


def test_a_title_names_a_paper_only_when_its_head_reads_as_a_name():
    assert name_of("Mamba: Linear-Time Sequence Modeling with Selective State Spaces") == "Mamba"
    assert name_of("Hungry Hungry Hippos: Towards Language Modeling") == "Hungry Hungry Hippos"
    assert name_of("Can Mamba Learn How to Learn? A Comparative Study") == ""
    assert name_of("Is the whole thing broken yet: a study") == ""


def test_a_tldr_naming_a_paper_cites_that_paper():
    h = held()
    h.owners = {"E1": "2503.18813", "E2": "2503.18813", "E3": "2302.12173"}
    h.names = {"2503.18813": "CaMeL", "2302.12173": "Greshake"}
    r = report()
    r["tldr"] = {"text": "CaMeL stops every attack.", "cites": []}
    assert ("tldr.cites", "cites nothing") in faults(r, h)
    r["tldr"] = {"text": "CaMeL stops attacks that Greshake showed [E1].", "cites": []}
    assert faults(r, h) == [("tldr.cites", "names a paper without its marker")]
    r["tldr"] = {"text": "CaMeL [E1] stops attacks that Greshake [E3] showed.", "cites": []}
    h.evidence = {**EVIDENCE, "E3": "Greshake shows indirect injection."}
    assert faults(r, h) == []


def test_a_deep_report_cites_the_poset_sentences():
    h = held()
    h.evidence = {**EVIDENCE, "E3": "CaMeL critiques prompt filters."}
    r = report({"type": "table", "table": "T1"})
    r["sections"][0]["blocks"] += [
        {"type": "coverage", "table": "T1", "rows": "split", "columns": ["trained"]},
        {"type": "open", "text": "x", "search": "graph rerankers"},
    ]
    verdict = check(r, held(read={"2503.18813", "2404.13208"}))
    contract(r, h, "deep", True, verdict, [])
    assert [f["path"] for f in verdict.faults] == ["sections.1"]
    assert "trace_lineages' sentences" in verdict.faults[0]["detail"]


def test_a_deep_report_cites_every_paper_of_its_table():
    h = held()
    rows = [
        {"arxiv_id": "2503.18813", "cells": {"split": {"value": "yes"}}},
        {"arxiv_id": "2502.03593", "cells": {"split": {"value": "n/a"}}},
    ]
    h.tables = {"T1": {**TABLE, "rows": rows}}
    r = report({"type": "table", "table": "T1"})
    verdict = check(r, h)
    contract(r, h, "deep", False, verdict, [])
    assert "2503.18813" in verdict.faults[0]["detail"]
    assert "2502.03593" not in verdict.faults[0]["detail"]
    h.owners = {"E1": "2503.18813"}
    verdict = check(r, h)
    contract(r, h, "deep", False, verdict, [])
    assert "2503.18813" not in verdict.faults[0]["detail"]


def test_a_whole_sections_list_appends_when_all_new_and_replaces_otherwise():
    r = report({"type": "inference", "text": "a"})
    gap = {"heading": "Gaps", "blocks": [{"type": "inference", "text": "d"}]}
    kept, bad = patch(r, {"sections": [r["sections"][0], gap]})
    assert not bad and [x["heading"] for x in kept["sections"]] == ["Defenses", "Gaps"]
    added, bad = patch(r, {"sections": [gap]})
    assert not bad and [x["heading"] for x in added["sections"]] == ["Defenses", "Gaps"]
    dropped, bad = patch(kept, {"sections": [gap]})
    assert not bad and [x["heading"] for x in dropped["sections"]] == ["Gaps"]
