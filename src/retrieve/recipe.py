"""
Builds a pipeline out of a recipe file and the [retrieve] block of config.toml

A recipe holds only what is tuned per mode, its core and its exponents
Everything the modes share, the pool sizes and the decay, lives in config.toml
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from src.core.parse_graph_export import GraphExport
from src.core.schema import Reranker, Retriever
from src.retrieve.cosine import CosineRetriever
from src.retrieve.expand import AncestorExpander, FrontierExpander
from src.retrieve.query import PaperQuery
from src.retrieve.scorer import GraphScorer, Weights

LOG = logging.getLogger(__name__)


def read_recipe(path: str | Path) -> dict[str, Any]:
    """
    Reads one recipe file

    Args:
        path: The toml holding the recipe's core and exponents.

    Returns:
        The parsed recipe.

    Raises:
        ValueError: When the file carries no core or no exponents.
    """
    parsed = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if "CORE" not in parsed or "weights" not in parsed:
        raise ValueError(f"{path} needs a CORE and a [weights] table")
    return parsed


def build_weights(block: dict[str, Any]) -> Weights:
    """
    One set of exponents out of a recipe's weights table

    Args:
        block: The `weights` table.

    Returns:
        The exponents, with anything the file leaves out taking its default.
    """
    return Weights(**{k.lower(): float(v) for k, v in block.items()})


def build_pipeline(
    recipe: dict[str, Any],
    settings: dict[str, Any],
    graph: GraphExport,
    pg_dest: str,
    blocked: list[str] | None = None,
) -> PaperQuery:
    """
    Assembles the retriever, both expanders and the scorer

    The expanders hop from the same core the scorer walks from.

    Args:
        recipe: A parsed recipe.
        settings: The [retrieve] block of config.toml.
        graph: The citation graph export.
        pg_dest: libpq connection string.
        blocked: Papers the query must never return.

    Returns:
        The pipeline, ready to be called.
    """
    core = int(recipe["CORE"])
    hops = int(settings["EXPANSION_LIMIT"])
    retriever: Retriever = CosineRetriever(pg_dest, blocked=blocked)
    expanders: list[Reranker] = [
        maker(graph, pg_dest, top_k=core, hop_limit=hops)
        for maker in (AncestorExpander, FrontierExpander)
    ]
    scorer = GraphScorer(
        graph,
        build_weights(recipe["weights"]),
        core=core,
        depth=int(settings["DEPTH"]),
        decay=float(settings["DECAY"]),
    )
    LOG.info("recipe %s: core %d", recipe.get("NAME", "unnamed"), core)
    return PaperQuery(retriever, expanders, [hops, hops], [scorer])


def get_seed_limit(settings: dict[str, Any]) -> int:
    """
    How many papers the cosine retriever returns

    Args:
        settings: The [retrieve] block of config.toml.

    Returns:
        The retriever's limit.
    """
    return int(settings["COSINE_LIMIT"])
