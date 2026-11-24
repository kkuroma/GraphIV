"""
Every method family by the module that ranks it, each module exposing its class as METHOD

A module is imported only when a test names it, since the GPU baselines load torch on import
"""

import importlib
from typing import Any

from src.schema import Method

# retrievers with their own search code, run only as the method none
BASELINES = ("bm25", "specter2", "scincl")
GAR = ("gar_cite", "gar_sim", "rerank")


def module_of(spec: dict[str, Any]) -> str:
    """
    The module ranking a test toml

    Args:
        spec: The test toml.

    Returns:
        A module name under src/methods/.
    """
    if spec["retriever"] in BASELINES:
        return spec["retriever"]
    return "gar" if spec["method"] in GAR else "graph"


def load(name: str) -> Method:
    """
    The method family a module holds

    Args:
        name: What module_of returned.

    Returns:
        An instance of it.
    """
    return importlib.import_module(f"src.methods.{name}").METHOD()
