"""
Every dataset by the name an experiment gives it, each module exposing its class as DATASET
"""

import importlib

from src.schema import Dataset

NAMES = ("surveymatch", "frontiermatch", "consensusmatch", "pasa", "surge")


def load(name: str) -> Dataset:
    """
    The dataset an experiment names

    Args:
        name: One of NAMES.

    Returns:
        An instance of it.
    """
    return importlib.import_module(f"src.datasets.{name}").DATASET()
