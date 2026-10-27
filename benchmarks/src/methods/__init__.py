"""
Every method by the name an experiment gives it, each module exposing its class as METHOD

A module is imported only when an experiment names it, since the GPU baselines load torch on import
"""

import importlib

from src.schema import Method

NAMES = ("boundedfield", "nomic", "ppr", "bm25", "specter2", "scincl", "gar")


def load(name: str) -> Method:
    """
    The method an experiment names

    Args:
        name: One of NAMES.

    Returns:
        An instance of it.
    """
    return importlib.import_module(f"src.methods.{name}").METHOD()
