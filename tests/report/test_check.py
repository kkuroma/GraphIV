"""
The mechanical report check, on hand-written reports against a two-paragraph evidence store
"""

from src.report.check import check, numbers_in, split_tldr

EVIDENCE = {
    "E1": "CaMeL extracts the control and data flows from the user query.",
    "E2": "The number of successful attacks drops from 300 to 0 with CaMeL.",
}
READ = {"2503.18813"}


def kinds(markdown: str, served: set[str] | None = None) -> list[str]:
    return [f["kind"] for f in check(markdown, EVIDENCE, READ, served or set()).faults]


def test_a_cited_line_with_its_number_passes():
    verdict = check("Attacks fall from 300 to 0 [E2].", EVIDENCE, READ, set())
    assert verdict.passed


def test_an_unknown_marker_fails():
    assert kinds("CaMeL separates the flows [E9].") == ["unknown marker"]


def test_a_quote_outside_its_paragraph_fails():
    line = 'CaMeL "drops every attack to zero" [E1].'
    assert kinds(line) == ["quote not in its paragraph"]


def test_a_number_outside_its_paragraph_fails():
    assert kinds("Attacks fall from 300 to 70 [E2].") == ["number not in its paragraph"]


def test_an_unread_paper_in_the_body_fails_and_is_fine_under_further_reading():
    body = "CaMeL extracts flows [E1], unlike 2404.13208 which trains the model."
    assert "paper cited but never read" in kinds(body)
    further = "Flows [E1].\n\n## Further reading\n\n- 2404.13208 The Instruction Hierarchy"
    assert kinds(further, {"2404.13208"}) == []


def test_a_long_uncited_statement_fails_unless_labeled_inference():
    claim = "Every defense in this area trades a measurable amount of utility for its security."
    assert kinds(f"Flows [E1].\n\n{claim}") == ["statement with no marker"]
    assert kinds(f"Flows [E1].\n\nInference: {claim}") == []


def test_years_and_separators_in_numbers():
    assert numbers_in("1,804 videos in 2023 at 59.17%") == {"1804", "59.17"}
    assert numbers_in("Qwen3-VL on the J1-J2 model in 1D, 4 runs, 20x20 lattice") == set()


def test_one_bracket_can_cite_several_labels():
    verdict = check("CaMeL cuts attacks from 300 to 0 by design [E1, E2].", EVIDENCE, READ, set())
    assert verdict.passed and verdict.markers == 2


def test_an_arxiv_id_on_a_cited_line_is_not_a_number():
    assert check("2503.18813 cuts attacks from 300 to 0 [E2].", EVIDENCE, READ, set()).passed


def test_every_sentence_of_a_cited_line_needs_its_own_marker():
    claim = "every defense in this whole area trades a measurable share of utility for security."
    assert kinds(f"Attacks fall from 300 to 0 [E2]. Then {claim}") == ["statement with no marker"]
    assert kinds(f"Attacks fall from 300 to 0 [E2]. Inference: {claim}") == []


def test_a_name_absent_from_the_cited_paper_fails():
    sources = {"E1": "camel extracts the control and data flows from the user query.", "E2": ""}
    line = "CaMeL explains its predictions with LIME over the extracted flows [E1]."
    faults = check(line, EVIDENCE, READ, set(), sources).faults
    assert [f["where"] for f in faults] == ["LIME"]
    assert check("CaMeL extracts the flows [E1].", EVIDENCE, READ, set(), sources).passed


def test_a_claimed_search_must_be_one_the_run_made():
    line = 'Inference: open, since a focused search for "graph rerankers" found none.'
    faults = check(
        f"Flows [E1].\n\n{line}", EVIDENCE, READ, set(), None, {"dense rerankers"}
    ).faults
    assert [f["kind"] for f in faults] == ["search claimed but never run"]
    assert check(f"Flows [E1].\n\n{line}", EVIDENCE, READ, set(), None, {"Graph rerankers"}).passed


def test_a_rounded_number_is_backed_by_its_paragraph():
    evidence = {"E3": "SRCC reaches 0.891 and PLCC 0.914 on the benchmark."}
    assert check("SRCC reaches 0.89 on it [E3].", evidence, READ, set()).passed
    assert not check("SRCC reaches 0.95 on it [E3].", evidence, READ, set()).passed


def test_a_tldr_opening_the_report_is_split_off():
    report = "# Q\n\n**TL;DR:** Dense retrieval wins [Karpukhin7].\n\n## Findings\n\nBody."
    tldr, body = split_tldr(report)
    assert tldr == "Dense retrieval wins [Karpukhin7]."
    assert "TL;DR" not in body and body.startswith("# Q")


def test_a_report_without_a_tldr_is_left_whole():
    report = "# Q\n\n## Findings\n\nTL;DR: not an opener here."
    assert split_tldr(report) == ("", report)
