"""
Plain nomic cosine from the store's HNSW index, through src/graphiv/nomic.py
"""

from typing import Any

from src.methods import bridge
from src.schema import Case, Method, Ranking


class Nomic(Method):
    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        return bridge.rank("nomic.py", cases)


METHOD = Nomic
