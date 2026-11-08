"""
Runs benchmark experiments, each a toml under experiments/ naming a method, a dataset and its [fit] and [run]

A fit is kept in fit_cache/ under a hash of what made it, so experiments sharing one fit it once
A run whose results are already in runs/ is skipped unless --rerun is passed
"""

import argparse
import hashlib
import json
import math
import time
import tomllib
from pathlib import Path
from typing import Any

from src import common, datasets, methods
from src.schema import Case, Method, Ranking

EXPERIMENTS = common.HERE / "experiments"


def find(spec: str) -> Path:
    """
    An experiment toml from its path or its name

    Args:
        spec: A path to a toml, or the stem of one under experiments/.

    Returns:
        The toml's path.

    Raises:
        FileNotFoundError: When neither exists.
    """
    path = Path(spec)
    if path.suffix == ".toml" and path.exists():
        return path
    path = EXPERIMENTS / f"{spec}.toml"
    if not path.exists():
        raise FileNotFoundError(f"no experiment {spec}")
    return path


def fit_key(method: str, method_class: Method, table: dict[str, Any]) -> str:
    """
    The name a fit is kept under, a hash of the method, its FIT_VERSION and the [fit] table

    Args:
        method: The method's name.
        method_class: The method, for its FIT_VERSION.
        table: The [fit] table.

    Returns:
        Twelve hex characters.
    """
    blob = json.dumps([method, method_class.FIT_VERSION, table], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def fitted(spec: dict[str, Any], method: Method) -> tuple[str | None, dict[str, Any]]:
    """
    What the experiment's fit learned, read from fit_cache/ or fitted and written there

    Args:
        spec: The experiment.
        method: Its method.

    Returns:
        The fit's key and what it learned, (None, {}) when the experiment has no [fit].
    """
    table = spec.get("fit")
    if not table:
        return None, {}
    key = fit_key(spec["method"], method, table)
    path = common.FITS / spec["method"] / f"{key}.json"
    if path.exists():
        common.say(f"fit {key} cached")
        return key, json.loads(path.read_text())["result"]
    common.say(f"fit {key} on {table['train']}")
    started = time.time()
    params = {k: v for k, v in table.items() if k != "train"}
    result = method.fit(datasets.load(table["train"]), params)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"method": spec["method"], "fit_version": method.FIT_VERSION, "fit": table, "seconds": time.time() - started, "result": result}
    path.write_text(json.dumps(record, indent=1))
    return key, result


def ndcg(ranked: list[str], key: set[str], k: int) -> float:
    """
    Binary nDCG at k, the ideal list holding as many hits as the key allows within k

    Args:
        ranked: The ranking.
        key: The answers.
        k: The cutoff.

    Returns:
        The score on [0, 1].
    """
    gain = sum(1 / math.log2(i + 2) for i, a in enumerate(ranked[:k]) if a in key)
    return gain / sum(1 / math.log2(i + 2) for i in range(min(len(key), k)))


def write(name: str, spec: dict[str, Any], cases: list[Case], rankings: list[Ranking], key: str | None, allowed: set[str] | None) -> dict[str, Any]:
    """
    Writes runs/<name>.jsonl, each case's ranking and recall at every k, and runs/<name>.json, the summary

    Args:
        name: The experiment's name.
        spec: The experiment.
        cases: Its cases.
        rankings: One per case, in any order.
        key: The fit's key, None without a fit.
        allowed: The dataset's corpus, which each ranking is cut to first, None for the whole store.

    Returns:
        The summary.

    Raises:
        ValueError: When a case has no ranking.
    """
    ks, ndcg_ks = spec["run"]["ks"], spec["run"].get("ndcg", [])
    by_id = {r["sample_id"]: r for r in rankings}
    missing = [c["sample_id"] for c in cases if c["sample_id"] not in by_id]
    if missing:
        raise ValueError(f"{name}: {len(missing)} cases have no ranking, the first {missing[0]}")
    rows = []
    for c in cases:
        r = by_id[c["sample_id"]]
        key_set = set(c["key"])
        ranked = [a for a in r["ranked"] if a in allowed] if allowed is not None else r["ranked"]
        row = {"sample_id": c["sample_id"], "ranked": ranked, "recall": {str(k): len(set(ranked[:k]) & key_set) / len(key_set) for k in ks}}
        if ndcg_ks:
            row["ndcg"] = {str(k): ndcg(ranked, key_set, k) for k in ndcg_ks}
        if "seconds" in r:
            row["seconds"] = r["seconds"]
        rows.append(row)
    common.write_jsonl(common.RUNS / f"{name}.jsonl", rows)
    seconds = [r["seconds"] for r in rows if "seconds" in r]
    summary = {
        "name": name,
        "method": spec["method"],
        "dataset": spec["dataset"],
        "split": spec.get("split", "test"),
        "cases": len(rows),
        "recall": {str(k): sum(r["recall"][str(k)] for r in rows) / len(rows) for k in ks},
        "ndcg": {str(k): sum(r["ndcg"][str(k)] for r in rows) / len(rows) for k in ndcg_ks},
        "seconds": sum(seconds) / len(seconds) if len(seconds) == len(rows) else None,
        "fit": key,
        "config": spec,
        # the recipe file as it ran, since a later tune rewrites it in place
        "recipe": (common.TUNED_RECIPES / f"{spec['run']['recipe']}.toml").read_text() if "recipe" in spec["run"] else None,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (common.RUNS / f"{name}.json").write_text(json.dumps(summary, indent=1))
    return summary


def run(path: Path, rerun: bool) -> None:
    """
    One experiment, fitting first when its fit is not cached

    Args:
        path: Its toml.
        rerun: Run it even when its results exist.
    """
    name = path.stem
    spec = tomllib.loads(path.read_text())
    summary = common.RUNS / f"{name}.json"
    if summary.exists() and not rerun:
        if json.loads(summary.read_text())["config"] != spec:
            print(f"{name}: results exist but the toml changed since, pass --rerun to redo it", flush=True)
        else:
            print(f"{name}: results exist, skipped", flush=True)
        return
    first = spec["run"].get("first_stage")
    if first and not (common.RUNS / f"{first}.json").exists():
        run(find(first), rerun=False)
    common.LOGS.mkdir(parents=True, exist_ok=True)
    common.LOG = common.LOGS / f"{name}.log"
    common.say(f"== {name} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    method = methods.load(spec["method"])
    key, result = fitted(spec, method)
    data = datasets.load(spec["dataset"])
    cases = data.get(spec.get("split", "test"))
    params = {k: v for k, v in spec["run"].items() if k not in ("ks", "ndcg")}
    corpus = data.corpus()
    if corpus is not None:
        params["corpus"] = str(corpus)
    started = time.time()
    rankings = method.retrieve(cases, params, result)
    allowed = set(corpus.read_text().split()) if corpus is not None else None
    done = write(name, spec, cases, rankings, key, allowed)
    scores = [f"R@{k} {v:.3f}" for k, v in done["recall"].items()] + [f"nDCG@{k} {v:.3f}" for k, v in done["ndcg"].items()]
    common.say(f"== {name} {len(cases)} cases in {time.time() - started:.0f}s, {', '.join(scores)}")
    common.LOG = None


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--experiment", help="a toml path, or a name under experiments/")
    _ = ap.add_argument("--run-all", action="store_true", help="every toml under experiments/")
    _ = ap.add_argument("--rerun", action="store_true", help="redo runs whose results already exist")
    args = ap.parse_args()
    if bool(args.experiment) == args.run_all:
        ap.error("pass --experiment or --run-all, not both")
    common.guard()
    paths = sorted(EXPERIMENTS.glob("*.toml")) if args.run_all else [find(args.experiment)]
    # a run another reads as its first stage goes first, so a rerun never hands on stale results
    paths.sort(key=lambda p: "first_stage" in tomllib.loads(p.read_text())["run"])
    for path in paths:
        run(path, args.rerun)


if __name__ == "__main__":
    main()
