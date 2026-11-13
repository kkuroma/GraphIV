"""
What every tool needs, opened once per process: store, graph, a pipeline per mode, the model

The config path comes from GRAPHIV_CONFIG when set, which is how a deployment hands it a file
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.config import load_config
from src.core.embed import EmbeddingEndpoint, embed, embed_documents
from src.core.llm import Gemma
from src.core.parse_graph_export import GraphExport, load_graph_export
from src.miner.recipe import Recipe, build_recipe
from src.retrieve.query import PaperQuery
from src.retrieve.recipe import build_pipeline, read_recipe

REPO = Path(__file__).resolve().parents[2]
MODES = ("focused", "survey", "frontier", "consensus")


def prompt(name: str) -> str:
    """
    One prompt from its markdown file, which is where every prompt the tools send lives

    Args:
        name: The file name under src/mcp/prompts.

    Returns:
        The prompt text, without the file's closing newline.
    """
    return (Path(__file__).parent / "prompts" / name).read_text(encoding="utf-8").removesuffix("\n")


def config_path() -> Path:
    """
    Which config this process reads

    Returns:
        GRAPHIV_CONFIG when set, the checkout's config.toml otherwise.
    """
    return Path(os.environ.get("GRAPHIV_CONFIG") or REPO / "config.toml")


@dataclass
class Context:
    """
    The handles that outlive every call
    """

    config: dict[str, Any]
    dsn: str
    graph: GraphExport
    model: Gemma
    pipes: dict[str, PaperQuery]
    weights: dict[str, dict[str, float]]
    seed_limit: int
    mining: Recipe
    html: Path
    pdfs: Path
    runs: Path
    pdftotext: str
    fetch: bool
    chunks: dict[str, Any] = field(default_factory=dict)

    def embed(self, text: str) -> list[float]:
        """
        One query vector

        Args:
            text: The question.

        Returns:
            The unit vector.
        """
        return embed(EmbeddingEndpoint.from_config(self.config), text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Document vectors with the corpus's own prefix

        Args:
            texts: The documents.

        Returns:
            One unit vector each.
        """
        return embed_documents(EmbeddingEndpoint.from_config(self.config), texts)


@lru_cache(maxsize=1)
def load() -> Context:
    """
    Opens everything once

    Returns:
        The context every tool reads.
    """
    config = load_config(str(config_path()))
    serve, retrieve = config["serve"], config["retrieve"]
    dsn = str(config["database"]["PG_DEST"])
    graph = load_graph_export(str(Path(str(config["dataset"]["GRAPH_PATH"]))))
    recipes = Path(str(serve["RECIPE_PATH"]))
    recipe = {mode: read_recipe(recipes / f"{mode}.toml") for mode in MODES}
    pipes = {mode: build_pipeline(recipe[mode], retrieve, graph, dsn) for mode in MODES}
    weights = {m: {k.lower(): float(v) for k, v in r["weights"].items()} for m, r in recipe.items()}
    ends = config["endpoints"]
    return Context(
        config=config,
        dsn=dsn,
        graph=graph,
        model=Gemma(
            str(ends["LLM_ENDPOINT"]),
            str(ends["LLM_NAME"]),
            Path(str(serve["CACHE_PATH"])),
            str(ends["LLM_API_KEY"]),
        ),
        pipes=pipes,
        weights=weights,
        seed_limit=int(retrieve["COSINE_LIMIT"]),
        mining=build_recipe(config["miner"]),
        html=Path(str(config["dataset"]["HTML_PATH"])),
        pdfs=Path(str(config["dataset"]["PDF_PATH"])),
        runs=Path(str(serve["RUNS_PATH"])),
        pdftotext=str(serve.get("PDFTOTEXT", "pdftotext")),
        fetch=bool(serve.get("FETCH", False)),
    )
