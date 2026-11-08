"""
Checks the score against numbers worked out by hand on the eight paper fixture
"""

from __future__ import annotations

from datetime import date

import numpy as np
import scipy.sparse as sp

from src.core.parse_graph_export import GraphExport
from src.core.schema import Paper, Query
from src.retrieve.scorer import (
    TERMS,
    GraphScorer,
    Weights,
    compress_long_tail,
    get_combined_score,
    normalize_to_max,
)

CUTOFF = date(2030, 1, 1)


def _only(*live: str) -> Weights:
    """
    Exponents with every term at zero but the ones named

    Args:
        live: Which terms keep an exponent of one.

    Returns:
        The exponents.
    """
    return Weights(**{name: 1.0 if name in live else 0.0 for name in TERMS + ("hub",)})


def _query() -> Query:
    """
    A query capped past every fixture date

    Returns:
        The query.
    """
    return Query(text="t", emb=[0.0], max_date=CUTOFF)


def test_normalize_to_max_clips_and_scales() -> None:
    """
    Negatives go to zero and the largest entry becomes one
    """
    assert normalize_to_max(np.array([-1.0, 1.0, 2.0])).tolist() == [0.0, 0.5, 1.0]
    assert normalize_to_max(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]


def test_compress_long_tail_keeps_zero_at_zero() -> None:
    """
    A zero stays zero, the maximum stays one, and a hundredth of the max lifts above a hundredth
    """
    got = compress_long_tail(np.array([0.0, 0.01, 1.0]))
    assert got[0] == 0.0
    assert got[2] == 1.0
    assert got[1] > 0.1


def test_combined_score_is_neutral_where_a_term_is_zero() -> None:
    """
    An exponent of zero makes its factor exactly one, so the score is the remaining terms
    """
    individual = {name: np.array([0.5]) for name in TERMS + ("hub",)}
    assert get_combined_score(individual, _only("cos")).tolist() == [1.5]
    assert get_combined_score(individual, _only()).tolist() == [1.0]


def test_combined_score_divides_by_the_hub(graph: GraphExport) -> None:
    """
    The hub denominator is the one term that lowers a score
    """
    individual = {name: np.array([0.0]) for name in TERMS}
    individual["hub"] = np.array([1.0])
    assert get_combined_score(individual, _only("hub")).tolist() == [0.5]


def test_adjacency_is_one_where_the_column_cites_the_row(
    graph: GraphExport, pool: list[Paper]
) -> None:
    """
    p3 cites p0, so the entry at row p0 and column p3 is the only one in that row
    """
    scorer = GraphScorer(graph)
    rows = scorer.get_paper_rows(pool)
    dense = scorer.build_adjacency_matrix(rows).toarray()
    assert dense[0, 3] == 1.0
    assert dense[3, 0] == 0.0
    assert dense[4].sum() == 2.0
    assert dense.sum() == len(("p0p4", "p0p5", "p1p4", "p2p5", "p2p6", "p3p0", "p3p1", "p7p3"))


def test_core_weights_take_the_top_by_cosine(graph: GraphExport, pool: list[Paper]) -> None:
    """
    A core of two keeps p0 and p1 and normalizes their cosine to sum to one
    """
    weights = GraphScorer(graph, core=2).get_core_weights(pool)
    assert weights[2:].tolist() == [0.0] * 6
    assert weights[0] == 0.9 / 1.7
    assert abs(weights.sum() - 1.0) < 1e-12


def test_distance_decay_falls_off_with_each_hop(graph: GraphExport, pool: list[Paper]) -> None:
    """
    The core is 1.0, what it cites is 0.5, and what nothing reaches stays 0.0
    """
    scorer = GraphScorer(graph, core=2, decay=0.5)
    rows = scorer.get_paper_rows(pool)
    adjacency = scorer.build_adjacency_matrix(rows)
    got = scorer.get_distance_decay(adjacency, scorer.get_core_weights(pool))
    assert got.tolist() == [1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]


def test_distance_decay_forward_reaches_the_second_hop(
    graph: GraphExport, pool: list[Paper]
) -> None:
    """
    p3 cites the core so it is one step out, and p7 cites p3 so it is two

    This is the pair that no one-step channel can separate, since both are zero on `fro`.
    """
    scorer = GraphScorer(graph, core=2, decay=0.5)
    rows = scorer.get_paper_rows(pool)
    forward = sp.csr_matrix(scorer.build_adjacency_matrix(rows).T)
    got = scorer.get_distance_decay(forward, scorer.get_core_weights(pool))
    assert got.tolist() == [1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25]


def test_second_hop_channels_score_only_the_far_paper(
    graph: GraphExport, pool: list[Paper]
) -> None:
    """
    Only p7 is two forward hops from the core, and nothing is two backward hops

    `fro` is zero at p7 and `fro2` is one there, which is what the term was added for.
    """
    scorer = GraphScorer(graph, core=2)
    rows = scorer.get_paper_rows(pool)
    individual = scorer.get_individual_score(pool, rows, graph.get_day_number(CUTOFF))
    assert individual["fro"][7] == 0.0
    assert individual["fro2"].tolist() == [0.0] * 7 + [1.0]
    assert individual["anc2"].tolist() == [0.0] * 8


def test_individual_scores_are_all_bounded(graph: GraphExport, pool: list[Paper]) -> None:
    """
    Every term comes back on [0, 1], which is what makes each factor sit on [1, 2 ** exponent]
    """
    scorer = GraphScorer(graph, core=2)
    rows = scorer.get_paper_rows(pool)
    individual = scorer.get_individual_score(pool, rows, graph.get_day_number(CUTOFF))
    assert set(individual) == set(TERMS) | {"hub"}
    for name, values in individual.items():
        assert values.min() >= 0.0, name
        assert values.max() <= 1.0, name


def test_ancestors_rank_what_the_core_cites(graph: GraphExport, pool: list[Paper]) -> None:
    """
    p4 is cited by both core papers and p5 by one, so p4 leads and p2 scores nothing
    """
    scorer = GraphScorer(graph, core=2)
    rows = scorer.get_paper_rows(pool)
    anc = scorer.get_individual_score(pool, rows, graph.get_day_number(CUTOFF))["anc"]
    assert anc[4] == 1.0
    assert 0.0 < anc[5] < anc[4]
    assert anc[2] == 0.0


def test_frontier_ranks_what_cites_the_core(graph: GraphExport, pool: list[Paper]) -> None:
    """
    Only p3 cites a core paper, so it is the one nonzero entry
    """
    scorer = GraphScorer(graph, core=2)
    rows = scorer.get_paper_rows(pool)
    fro = scorer.get_individual_score(pool, rows, graph.get_day_number(CUTOFF))["fro"]
    assert fro[3] == 1.0
    assert np.count_nonzero(fro) == 1


def test_rerank_puts_the_ancestor_first(graph: GraphExport, pool: list[Paper]) -> None:
    """
    Ranking on ancestors alone returns p4 first, since both core papers cite it
    """
    weights = _only("anc")
    got = GraphScorer(graph, weights, core=2).rerank(_query(), pool, 3)
    assert [p.arxiv_id for p in got][0] == "p4"
    assert len(got) == 3


def test_rerank_puts_the_citer_first_on_the_frontier(graph: GraphExport, pool: list[Paper]) -> None:
    """
    Ranking on the frontier alone returns p3, which is the only paper citing the core
    """
    weights = _only("fro")
    got = GraphScorer(graph, weights, core=2).rerank(_query(), pool, 1)
    assert [p.arxiv_id for p in got] == ["p3"]


def test_rerank_fills_the_signals_and_the_score(graph: GraphExport, pool: list[Paper]) -> None:
    """
    Every returned paper carries one signal per term and a score that reproduces them
    """
    got = GraphScorer(graph, core=2).rerank(_query(), pool, None)
    assert len(got) == 8
    assert [p.score for p in got] == sorted((p.score or 0.0 for p in got), reverse=True)
    for paper in got:
        assert set(paper.signals) == set(TERMS) | {"hub"}


def test_rerank_caps_citations_at_the_query_date(graph: GraphExport, pool: list[Paper]) -> None:
    """
    p3 is dated 2022, so a 2021 cap hides its citations of p0 and p1 from the hub term
    """
    scorer = GraphScorer(graph, core=2)
    rows = scorer.get_paper_rows(pool)
    late = scorer.get_individual_score(pool, rows, graph.get_day_number(date(2030, 1, 1)))
    early = scorer.get_individual_score(pool, rows, graph.get_day_number(date(2021, 6, 1)))
    assert late["hub"][0] > 0.0
    assert early["hub"][0] == 0.0


def test_rerank_returns_nothing_for_an_empty_pool(graph: GraphExport) -> None:
    """
    An empty pool is not an error
    """
    assert GraphScorer(graph).rerank(_query(), [], 10) == []


def test_a_blocked_citer_never_counts_toward_the_hub(graph: GraphExport, pool: list[Paper]) -> None:
    """
    p3 is the only citer of p0, so blocking it zeroes p0's hub as a case's own review must
    """
    scorer = GraphScorer(graph, core=2)
    rows = scorer.get_paper_rows(pool)
    cutoff = graph.get_day_number(CUTOFF)
    assert scorer.get_individual_score(pool, rows, cutoff)["hub"][0] > 0.0
    scorer.blocked = ["p3"]
    assert scorer.get_individual_score(pool, rows, cutoff)["hub"][0] == 0.0


def test_rerank_drops_a_blocked_paper_from_the_pool(graph: GraphExport, pool: list[Paper]) -> None:
    """
    A blocked paper handed in by a caller is neither returned nor walked through
    """
    scorer = GraphScorer(graph, _only("anc"), core=2)
    scorer.blocked = ["p4"]
    assert "p4" not in [p.arxiv_id for p in scorer.rerank(_query(), pool, None)]
