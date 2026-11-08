"""
BoundedField, a recipe tune.py wrote under data/recipes/, run through GraphIV's own pipeline by src/graphiv/boundedfield.py

[run] recipe names the recipe, which carries its own core and exponents, so nothing is fit here
"""

from typing import Any

from src import common
from src.methods import bridge
from src.schema import Case, Method, Ranking


class BoundedField(Method):
    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        recipe = common.TUNED_RECIPES / f"{params['recipe']}.toml"
        extra = ["--corpus", params["corpus"]] if "corpus" in params else []
        return bridge.rank("boundedfield.py", cases, "--recipe", str(recipe), "--graph", str(common.GRAPH), *extra)


METHOD = BoundedField
