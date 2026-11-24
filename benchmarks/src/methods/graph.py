"""
Cosine alone, BoundedField and PPR on one retriever's pool, every job of a test set ranked by one call to src/graphiv/rank.py

A tuned job reads its recipe from runs/tune/, with the test toml's [override] merged over it
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from src import common
from src.schema import Case, Method, Ranking


def effective_recipe(job: dict[str, Any]) -> dict[str, Any] | None:
    """
    The recipe a tuned job runs, its tune run's recipe.toml with the job's [override] over it

    Args:
        job: The test toml.

    Returns:
        The recipe, None for cosine alone.

    Raises:
        FileNotFoundError: When the tune run has not written its recipe.
    """
    if job["method"] not in common.TUNED_METHODS:
        return None
    path = common.TUNE_RUNS / common.tune_source(job) / "recipe.toml"
    if not path.exists():
        raise FileNotFoundError(f"{job['name']} needs {path}, run scripts/tune.py first")
    change = {k: v for k, v in job.get("override", {}).items() if k != "tune"}
    return common.overlay(common.read_toml(path), change)


class Graph(Method):
    def retrieve(self, cases: list[Case], jobs: list[dict[str, Any]]) -> dict[str, list[Ranking]]:
        first = jobs[0]
        dataset, split = first["dataset"], first["split"]
        common.run_graphiv("queries.py", "--retriever", first["retriever"], "--splits", json.dumps([[dataset, split]]))
        with tempfile.TemporaryDirectory() as tmp:
            spec = [{"name": j["name"], "method": j["method"], "recipe": effective_recipe(j)} for j in jobs]
            (Path(tmp) / "jobs.json").write_text(json.dumps(spec))
            corpus = common.CASES / dataset / "corpus.txt"
            args = ["--retriever", first["retriever"], "--dataset", dataset, "--split", split, "--jobs", str(Path(tmp) / "jobs.json"), "--out", tmp]
            common.run_graphiv("rank.py", *args, *(["--corpus", str(corpus)] if corpus.exists() else []))
            return {j["name"]: common.read_jsonl(Path(tmp) / f"{j['name']}.jsonl") for j in jobs}


METHOD = Graph
