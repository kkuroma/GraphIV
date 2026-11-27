"""
Paragraph chunking of a small LaTeXML-shaped page, the fused ranking, and the sentence split
"""

from pathlib import Path

from src.report.chunks import Chunk, around, bm25, chunk_html, number, rank, sentences, split

PAGE = """<html><body>
<section class="ltx_section"><h2>1 Introduction</h2>
<div class="ltx_para"><p class="ltx_p">We study dense retrieval for scientific question
answering at scale with a bi-encoder trained on citation pairs <math alttext="x_i"></math>
across many fields.</p></div>
<div class="ltx_para"><p class="ltx_p">Short.</p></div>
<div class="ltx_para"><p class="ltx_p">Our method improves recall at 100 by twelve points on the
benchmark compared against the sparse baseline that ranks by term overlap.</p></div>
</section>
<figure class="ltx_table"><figcaption>Table 1: Recall at 100.</figcaption>
<table class="ltx_tabular"><tr><td>BM25</td><td>0.41</td></tr>
<tr><td>Ours</td><td>0.53</td></tr></table>
</figure>
<section class="ltx_bibliography"><p class="ltx_p">A reference that must not be read.</p></section>
</body></html>"""


def test_html_keeps_math_tables_and_drops_the_bibliography(tmp_path: Path):
    path = tmp_path / "paper.html"
    path.write_text(PAGE)
    chunks = chunk_html(path)
    texts = [c.text for c in chunks]
    assert any("$x_i$" in t for t in texts)
    assert any("BM25 | 0.41" in t for t in texts)
    assert not any("must not be read" in t for t in texts)
    assert chunks[0].section == "1 Introduction"
    assert [c.pid for c in chunks] == [f"p{i}" for i in range(1, len(chunks) + 1)]


def test_a_short_paragraph_joins_its_neighbor():
    chunks = number([("A", "x" * 10), ("A", "long enough " * 20), ("B", "other")])
    assert len(chunks) == 2
    assert chunks[0].text.startswith("x" * 10)


def test_bm25_prefers_the_chunk_with_the_terms():
    chunks = number([("", "dense retrieval " * 20), ("", "protein folding " * 20)])
    scores = bm25("dense retrieval", chunks)
    assert scores[0] > scores[1] == 0


def test_rank_fuses_and_around_adds_neighbors():
    chunks = number([("", f"topic {w} " * 20) for w in ("alpha", "beta", "gamma", "delta")])
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    order = rank("beta", chunks, vectors, [0.0, 1.0])
    assert order[0] == 1
    assert around([1], 4, 1) == [0, 1, 2]


def test_split_keeps_short_forms_and_joins_a_colon():
    text = (
        "It follows Ref. [12] and Fig. 3 shows it. The values are: Recall is 0.53. A. Smith agrees."
    )
    assert split(text) == [
        "It follows Ref. [12] and Fig. 3 shows it.",
        "The values are: Recall is 0.53.",
        "A. Smith agrees.",
    ]


def test_sentences_keep_a_table_whole_and_carry_the_paragraph():
    table = "Table 1: Recall.\nBM25 | 0.41\nOurs | 0.53\nDense | 0.50\nGraph | 0.49\nAll | 0.55"
    chunks = [Chunk("p1", "", "One claim here. Another claim there."), Chunk("p2", "", table)]
    shown = sentences(chunks, [0, 1])
    assert [(s.pid, s.text) for s in shown[:2]] == [
        ("p1", "One claim here."),
        ("p1", "Another claim there."),
    ]
    assert shown[2].text == table
