"""
Personalized PageRank over the tuned survey recipe's pool, through src/graphiv/ppr.py

The fit scores every setting of the grid on a seeded sample of train cases and keeps the best at [fit] k
"""

import itertools
import json
import random
from typing import Any

from src import common
from src.methods import bridge
from src.schema import Case, Dataset, Method, Ranking

# the pool PageRank ranks, BoundedField's own
POOL = common.TUNED_RECIPES / "survey.toml"


class PPR(Method):
    # 1 since the pool blocks a case's anchors past the retriever
    FIT_VERSION = 1

    def fit(self, data: Dataset, params: dict[str, Any]) -> dict[str, Any]:
        cases = data.get(params["split"])
        random.Random(params["seed"]).shuffle(cases)
        cases = cases[: params["cases"]]
        grid = bridge.compute("ppr.py", cases, "grid", "--graph", str(common.GRAPH), "--recipe", str(POOL), "--params", json.dumps(params))
        settings = list(itertools.product(params["alphas"], params["teleports"], params["directions"]))
        # the first setting reaching the best mean, in grid order
        pick = max(settings, key=lambda s: grid["/".join(map(str, s))])
        return {"k": params["k"], "cases": len(cases), "pick": list(pick), "grid": grid}

    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        return bridge.rank("ppr.py", cases, "run", "--graph", str(common.GRAPH), "--recipe", str(POOL), "--params", json.dumps({"setting": fitted["pick"]}))


METHOD = PPR
