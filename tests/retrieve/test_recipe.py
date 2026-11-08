"""
Checks a recipe and the [retrieve] block build the pipeline both describe
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.parse_graph_export import GraphExport
from src.retrieve.expand import AncestorExpander, FrontierExpander
from src.retrieve.recipe import build_pipeline, get_seed_limit, read_recipe
from src.retrieve.scorer import GraphScorer

SETTINGS = {"COSINE_LIMIT": 5000, "EXPANSION_LIMIT": 300, "DEPTH": 3, "DECAY": 0.25}
RECIPE = {"NAME": "t", "CORE": 7, "weights": {"COS": 1.0, "ANC": 2.0}}


def test_expanders_hop_from_the_core_the_scorer_walks_from(graph: GraphExport) -> None:
    """
    One CORE sets both, and the expansion limit caps each expander
    """
    pipeline = build_pipeline(RECIPE, SETTINGS, graph, "unused")
    ancestor, frontier = pipeline.expanders
    assert isinstance(ancestor, AncestorExpander) and isinstance(frontier, FrontierExpander)
    assert ancestor.top_k == frontier.top_k == 7
    assert pipeline.limits == [300, 300]
    scorer = pipeline.scorers[0]
    assert isinstance(scorer, GraphScorer)
    assert (scorer.core, scorer.depth, scorer.decay) == (7, 3, 0.25)
    assert scorer.weights.anc == 2.0


def test_the_seed_limit_comes_from_the_retrieve_block() -> None:
    assert get_seed_limit(SETTINGS) == 5000


def test_a_recipe_without_a_core_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "r.toml"
    path.write_text("[weights]\nCOS = 1\n")
    with pytest.raises(ValueError, match="CORE"):
        read_recipe(path)
