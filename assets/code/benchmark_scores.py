"""
What the retrieval benchmarks last scored, read off the runs scripts/recipe_tuning wrote

Nothing is cached, since every number comes from experiments/, which the repository keeps
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

# each benchmark's own tuned recipe and the cut its headline is read at
HEADLINE = {
    "SurveyMatch": ("survey", "1000"),
    "FrontierMatch": ("frontier", "1000"),
    "ConsensusMatch": ("consensus", "100"),
}
BASELINE = "cosine_only"
# where the two-review agreement was measured, kept after its scratchpad task was deleted
AGREEMENT = "experiments/findings/survey-agreement-REPORT.md"
# where the tuning set each exponent to 0 from each benchmark's tuned point
TUNING = "experiments/findings/frontier-terms-REPORT.md"


def recall(config: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Mean recall at every cut, for every recipe each benchmark ran

    Args:
        config: The config whose [recipe_tuning] names the benchmarks' output.

    Returns:
        Benchmark to recipe to cut to the mean over its cases.
    """
    block = tomllib.loads(config.read_text(encoding="utf-8"))["recipe_tuning"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for bench in HEADLINE:
        out[bench] = {}
        for name in block["recipes"][bench]:
            stem = name.removesuffix(".toml")
            path = Path(block["OUTPUT_PATH"]) / bench / stem / "scores.jsonl"
            cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            cuts = cases[0]["recall"]
            out[bench][stem] = {k: sum(c["recall"][k] for c in cases) / len(cases) for k in cuts}
    return out


def ceiling(config: Path, bench: str, recipe: str) -> float:
    """
    The share of the answer key one recipe's pool holds at all, averaged over the cases

    Args:
        config: The config whose [recipe_tuning] names the benchmarks' output.
        bench: The benchmark.
        recipe: The recipe.

    Returns:
        The mean ceiling.
    """
    block = tomllib.loads(config.read_text(encoding="utf-8"))["recipe_tuning"]
    path = Path(block["OUTPUT_PATH"]) / bench / recipe / "scores.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return sum(row["ceiling"] for row in rows) / len(rows)


def dropped(root: Path) -> dict[str, dict[str, float]]:
    """
    The recall each benchmark lost when one exponent was set to 0 from its own tuned point

    Args:
        root: The repository.

    Returns:
        Benchmark to term to the recall lost, for each term the tuned point did not already zero.
    """
    text = (root / TUNING).read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| move |"))
    benches = [cell.strip() for cell in lines[start].strip("|").split("|")][1:]
    out: dict[str, dict[str, float]] = {bench: {} for bench in benches}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        found = re.fullmatch(r"`(\w+)` to 0", cells[0])
        if found is None:
            continue
        for bench, cell in zip(benches, cells[1:], strict=True):
            if cell:
                out[bench][found[1]] = -float(cell.split()[0])
    return out


def exponents(config: Path, mode: str) -> dict[str, Any]:
    """
    One recipe's scorer, its core and the exponent of every term it keeps

    Args:
        config: The config whose [serve] names the recipes.
        mode: The recipe's name.

    Returns:
        The core size and each nonzero exponent, keyed by the term in lower case.
    """
    where = Path(tomllib.loads(config.read_text(encoding="utf-8"))["serve"]["RECIPE_PATH"])
    recipe = tomllib.loads((where / f"{mode}.toml").read_text(encoding="utf-8"))
    scorer = next(c for c in recipe["component"] if c["KIND"] == "GraphScorer")
    weights = {k.lower(): float(v) for k, v in scorer["weights"].items() if float(v) > 0}
    return {"core": int(scorer["CORE"]), "weights": weights, "miner": recipe.get("miner", {})}


def cases(config: Path) -> dict[str, int]:
    """
    How many cases each benchmark scored

    Args:
        config: The config whose [recipe_tuning] names the benchmarks' output.

    Returns:
        Benchmark to its number of cases.
    """
    block = tomllib.loads(config.read_text(encoding="utf-8"))["recipe_tuning"]
    out: dict[str, int] = {}
    for bench in HEADLINE:
        path = Path(block["OUTPUT_PATH"]) / bench / BASELINE / "scores.jsonl"
        out[bench] = len(path.read_text(encoding="utf-8").splitlines())
    return out


def agreement(root: Path) -> tuple[int, float]:
    """
    How well two reviews of one topic written close together agree, as the findings file states it

    Args:
        root: The repository.

    Returns:
        The number of review pairs and their mean recall of each other.
    """
    text = (root / AGREEMENT).read_text(encoding="utf-8")
    found = re.search(r"All (\d+) pairs at cosine [\d.]+ or above: mean recall ([\d.]+)", text)
    if found is None:
        raise ValueError(f"{AGREEMENT} no longer states the pairs' mean recall")
    return int(found[1]), float(found[2])
