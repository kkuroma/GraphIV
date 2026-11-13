"""
Another embedder's exact cosine over index_cache/<vectors>.npy, alone or under a BoundedField recipe, through src/graphiv/embedder.py

[run] vectors names the npy, endpoint the router's embedding path and prompt the query prompt
"""

from typing import Any

from src import common
from src.methods import bridge
from src.schema import Case, Method, Ranking


class Embedder(Method):
    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        args = ["--vectors", str(common.INDEX / f"{params['vectors']}.npy"), "--endpoint", params["endpoint"], "--prompt", params.get("prompt", ""), "--graph", str(common.GRAPH)]
        if "recipe" in params:
            args += ["--recipe", str(common.TUNED_RECIPES / f"{params['recipe']}.toml")]
        if params.get("unload"):
            args += ["--unload"]
        if "workers" in params:
            args += ["--workers", str(params["workers"])]
        if "corpus" in params:
            args += ["--corpus", params["corpus"]]
        return bridge.rank("embedder.py", cases, *args)


METHOD = Embedder
