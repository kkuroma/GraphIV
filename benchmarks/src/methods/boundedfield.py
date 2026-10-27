"""
BoundedField, the shipped recipes run through GraphIV's own pipeline by src/graphiv/boundedfield.py

[run] recipe names a toml of the checkout's retrieval_recipes/, and a fit replaces its weights, which is how focused is made
The fit is src/graphiv/focused.py, on the dataset's train split and picked on [fit] val
"""

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from src import common
from src.methods import bridge
from src.schema import Case, Dataset, Method, Ranking


class BoundedField(Method):
    def fit(self, data: Dataset, params: dict[str, Any]) -> dict[str, Any]:
        rest = {k: v for k, v in params.items() if k != "val"} | {"rows": data.source_size("train")}
        with tempfile.TemporaryDirectory() as tmp:
            val = Path(tmp) / "val.jsonl"
            common.write_jsonl(val, data.get(params["val"]))
            return bridge.compute(
                "focused.py",
                data.get("train"),
                "--val",
                str(val),
                "--params",
                json.dumps(rest),
                "--graph",
                str(common.GRAPH),
                "--cache",
                str(common.FITS / "boundedfield" / f"terms_{data.NAME}"),
                path="scripts/recipe_tuning",
            )

    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        text = (common.ROOT / "retrieval_recipes" / f"{params['recipe']}.toml").read_text()
        for term, value in fitted.get("exponents", {}).items():
            text = re.sub(rf"^{term.upper()} = .*$", f"{term.upper()} = {value:g}", text, flags=re.M)
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "recipe.toml"
            recipe.write_text(text)
            return bridge.rank("boundedfield.py", cases, "--recipe", str(recipe), "--graph", str(common.GRAPH))


METHOD = BoundedField
