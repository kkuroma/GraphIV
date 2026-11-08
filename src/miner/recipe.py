"""
The `[miner]` block of config.toml, which is where every constant this package uses lives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.miner.leyline import Settings

ARGUES = ("extends", "critiques", "compares")
WEIGHTS = {
    "extends": 1.0,
    "critiques": 1.0,
    "compares": 0.8,
    "uses": 0.4,
    "motivates": 0.15,
    "background": 0.0,
}


@dataclass(frozen=True)
class Recipe:
    """
    What the miner does above the ranking, from the hop out to how far back a successor may sit
    """

    settings: Settings = field(default_factory=Settings)
    hop: int = 1
    walk_prefix: int = 30
    connectors: int = 40
    successor_days: int = 240
    successor_floor: float = 0.60
    successor_intents: tuple[str, ...] = ARGUES


def build_recipe(block: dict[str, Any]) -> Recipe:
    """
    The miner's constants, with every value the block leaves out taking its default

    Args:
        block: The [miner] table of config.toml.

    Returns:
        The recipe.
    """
    weights = {str(k): float(v) for k, v in dict(block.get("weights", WEIGHTS)).items()}
    argues = tuple(str(x) for x in block.get("ARGUES", ARGUES))
    return Recipe(
        settings=Settings(
            weights=weights,
            argues=argues,
            focus=float(block.get("FOCUS", 4.0)),
            decay=float(block.get("DECAY", 0.5)),
            knockdown=float(block.get("KNOCKDOWN", 0.4)),
            seeds=int(block.get("SEEDS", 10)),
            walks=int(block.get("WALKS", 50)),
            branch=int(block.get("BRANCH", 1)),
        ),
        hop=int(block.get("HOP", 1)),
        walk_prefix=int(block.get("WALK_PREFIX", 30)),
        connectors=int(block.get("CONNECTORS", 40)),
        successor_days=int(block.get("SUCCESSOR_DAYS", 240)),
        successor_floor=float(block.get("SUCCESSOR_FLOOR", 0.60)),
        successor_intents=tuple(str(x) for x in block.get("SUCCESSOR_INTENTS", argues)),
    )
