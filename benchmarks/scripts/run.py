"""
Runs test tomls under experiments/test/, each writing runs/test/<name>/ with its rankings, summary, test.toml and log

Tests sharing a retriever and a test set are ranked by one call to their method family, so a pool is built once for all
A tuned test also keeps the recipe.toml it ran, with its [override] merged in, since a later tune rewrites the source
"""

import argparse
import itertools
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from src import common, datasets, methods
from src.methods.graph import effective_recipe
from src.schema import Case, Ranking


def read(path: Path) -> dict[str, Any]:
    """
    One test toml, checked against the name its fields give

    Args:
        path: The toml.

    Returns:
        The spec, its name under "name".

    Raises:
        ValueError: When the file is not named as its fields say, or carries an override with no variant.
    """
    spec = common.read_toml(path)
    spec["name"] = common.test_name(spec)
    if spec["name"] != path.stem:
        raise ValueError(f"{path.name} should be named {spec['name']}.toml")
    if spec.get("override") and not spec.get("variant"):
        raise ValueError(f"{path.name} overrides its recipe without naming a variant")
    return spec


def done(spec: dict[str, Any]) -> bool:
    out = common.TEST_RUNS / spec["name"]
    return (out / "summary.json").exists() and common.read_toml(out / "test.toml") == {k: v for k, v in spec.items() if k != "name"}


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


def write(spec: dict[str, Any], src: Path, cases: list[Case], rankings: list[Ranking], allowed: set[str] | None, seconds: float, log: list[str]) -> dict[str, Any]:
    """
    Writes one run's directory: rankings.jsonl with each case's scores, summary.json, test.toml, recipe.toml and log

    Args:
        spec: The test toml.
        src: Its path.
        cases: Its cases.
        rankings: One per case, in any order.
        allowed: The set's corpus, which each ranking is cut to first, None for the whole corpus.
        seconds: The group's wall time over its tests.
        log: The group's log lines.

    Returns:
        The summary.

    Raises:
        ValueError: When a case has no ranking.
    """
    by_id = {r["sample_id"]: r for r in rankings}
    missing = [c["sample_id"] for c in cases if c["sample_id"] not in by_id]
    if missing:
        raise ValueError(f"{spec['name']}: {len(missing)} cases have no ranking, the first {missing[0]}")
    rows = []
    for c in cases:
        key = set(c["key"])
        ranked = by_id[c["sample_id"]]["ranked"]
        ranked = ([a for a in ranked if a in allowed] if allowed is not None else ranked)[: common.RETURN]
        rows.append({
            "sample_id": c["sample_id"], "ranked": ranked,
            "recall": {str(k): len(set(ranked[:k]) & key) / len(key) for k in common.KS},
            "ndcg": {str(k): ndcg(ranked, key, k) for k in common.NDCG_KS},
        })
    out = common.TEST_RUNS / spec["name"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    common.write_jsonl(out / "rankings.jsonl", rows)
    shutil.copyfile(src, out / "test.toml")
    recipe = effective_recipe(spec) if spec["method"] in common.TUNED_METHODS else None
    if recipe is not None:
        (out / "recipe.toml").write_text(f"# {common.tune_source(spec)}/recipe.toml with this test's [override] merged in\n" + common.dump_toml(recipe))
    summary = {
        "name": spec["name"], "retriever": spec["retriever"], "dataset": spec["dataset"], "split": spec["split"], "method": spec["method"],
        "tune": common.tune_source(spec) if recipe is not None else None, "cases": len(rows),
        "recall": {str(k): sum(r["recall"][str(k)] for r in rows) / len(rows) for k in common.KS},
        "ndcg": {str(k): sum(r["ndcg"][str(k)] for r in rows) / len(rows) for k in common.NDCG_KS},
        "group_seconds": seconds, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "log").write_text("".join(log))
    return summary


def run_group(rows: list[tuple[dict[str, Any], Path]]) -> None:
    """
    Every test of one retriever, test set and method family, ranked by one call
    """
    first = rows[0][0]
    data = datasets.load(first["dataset"])
    cases = data.get(first["split"])
    corpus = data.corpus()
    allowed = set(corpus.read_text().split()) if corpus is not None else None
    log_path = common.SCRATCH / "logs" / f"{first['retriever']}-{first['dataset']}_{first['split']}-{methods.module_of(first)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    common.LOG = log_path
    common.say(f"== {log_path.stem}: {len(rows)} tests, {time.strftime('%Y-%m-%d %H:%M:%S')}")
    started = time.time()
    jobs = [s for s, _ in rows]
    ranked = methods.load(methods.module_of(first)).retrieve(cases, jobs)
    seconds = time.time() - started
    common.LOG = None
    log = log_path.read_text().splitlines(keepends=True)
    for spec, src in rows:
        summary = write(spec, src, cases, ranked[spec["name"]], allowed, seconds, log)
        recall = summary["recall"]
        print(f"{spec['name']}: R@20 {recall['20']:.3f} R@100 {recall['100']:.3f} R@1000 {recall['1000']:.3f}", flush=True)
    print(f"== {log_path.stem}: {len(rows)} tests in {seconds:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("tests", nargs="+", help="toml paths or glob patterns of names under experiments/test/")
    _ = ap.add_argument("--rerun", action="store_true", help="redo tests whose results exist")
    args = ap.parse_args()
    common.guard()
    paths: list[Path] = []
    for t in args.tests:
        paths += [Path(t)] if t.endswith(".toml") else sorted((common.EXPERIMENTS / "test").glob(f"{t}.toml"))
    specs = [(read(p), p) for p in dict.fromkeys(paths)]
    todo = [(s, p) for s, p in specs if args.rerun or not done(s)]
    print(f"{len(specs)} tests, {len(specs) - len(todo)} done", flush=True)
    # GAR reads nomic's cosine run as its first stage, so every graph group goes first
    key = lambda sp: (methods.module_of(sp[0]) == "gar", sp[0]["retriever"], sp[0]["dataset"], sp[0]["split"], methods.module_of(sp[0]))  # noqa: E731
    todo.sort(key=key)
    for _, group in itertools.groupby(todo, key=key):
        run_group(list(group))


if __name__ == "__main__":
    main()
