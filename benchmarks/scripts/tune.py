"""
Runs tune tomls under experiments/tune/, each writing runs/tune/<name>/ with its recipe.toml, report.json, tune.toml and log

A BoundedField tune fits every core and pool size of tuning/<recipe>.toml, one pool size's term cache in RAM at a time
A tune with at_pick fits only at the core and pool its seed-0 base picked, sharing that one term cache with every other
"""

import argparse
import itertools
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src import common

WEIGHTS = ["cos", "rec", "anc", "fro", "anc2", "fro2", "anc_decay", "fro_decay", "cocite", "couple", "bm25", "hub"]
# fits running at once, each on THREADS of the 16 cores
PARALLEL = 4
THREADS = 4


def read(path: Path) -> dict[str, Any]:
    """
    One tune toml, checked against the name its fields give

    Args:
        path: The toml.

    Returns:
        The spec, its name under "name".

    Raises:
        ValueError: When the file is not named as its fields say, or carries an override with no variant.
    """
    spec = common.read_toml(path)
    spec["name"] = common.tune_name(spec)
    if spec["name"] != path.stem:
        raise ValueError(f"{path.name} should be named {spec['name']}.toml")
    if spec.get("override") and not spec.get("variant"):
        raise ValueError(f"{path.name} overrides its spec without naming a variant")
    return spec


def base_name(spec: dict[str, Any]) -> str:
    return common.tune_name({"retriever": spec["retriever"], "recipe": spec["recipe"], "method": spec["method"], "seed": 0})


def method_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    The [bf] or [ppr] table a tune fits with, its grid, seed and override applied

    Args:
        spec: The tune toml.

    Returns:
        The table, holding dataset, cores and pools beside its own keys.
    """
    shared = common.read_toml(common.TUNING / f"{spec['recipe']}.toml")
    table = {k: v for k, v in shared[spec["method"]].items()}
    table |= {"dataset": shared["dataset"], "cores": shared["cores"], "pools": shared["pools"]}
    if spec["method"] == "bf":
        table["fit"] = dict(table["fit"], seed=spec["seed"])
    else:
        table["seed"] = spec["seed"]
    table = common.overlay(table, spec.get("override", {}))
    if spec.get("at_pick"):
        pick = common.read_toml(common.TUNE_RUNS / base_name(spec) / "recipe.toml")
        table |= {"cores": [pick["CORE"]], "pools": [pick["POOL"]]}
    return table


def done(spec: dict[str, Any]) -> bool:
    out = common.TUNE_RUNS / spec["name"]
    return (out / "recipe.toml").exists() and common.read_toml(out / "tune.toml") == {k: v for k, v in spec.items() if k != "name"}


def write_recipe(spec: dict[str, Any], core: int, pool: int, body: dict[str, Any]) -> None:
    settings = common.CONFIG["retrieve"]
    recipe = {"NAME": spec["name"], "CORE": core, "POOL": pool, "HOPS": int(settings["EXPANSION_LIMIT"])} | body
    head = f"# The {spec['recipe']} recipe for {spec['method']} on {spec['retriever']}, written by benchmarks/scripts/tune.py\n"
    out = common.TUNE_RUNS / spec["name"]
    (out / "recipe.toml").write_text(head + common.dump_toml(recipe))


def start(spec: dict[str, Any], src: Path) -> Path:
    out = common.TUNE_RUNS / spec["name"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(src, out / "tune.toml")
    return out


def fit_one(spec: dict[str, Any], table: dict[str, Any], terms: Path, core: int, pool: int) -> dict[str, Any]:
    out = common.TUNE_RUNS / spec["name"] / "fits"
    out.mkdir(exist_ok=True)
    target = out / f"core-{core}-pool-{pool}.json"
    args = ["--terms", str(terms), "--spec", json.dumps(table), "--core", str(core), "--pool", str(pool), "--out", str(target)]
    common.wait(common.graphiv("fit.py", *args, threads=THREADS), log=common.TUNE_RUNS / spec["name"] / "log")
    return json.loads(target.read_text())


def build_terms(retriever: str, table: dict[str, Any], specs: list[tuple[int, int]], log: Path) -> Path:
    terms = common.SCRATCH / "terms" / f"{retriever}-{table['dataset']}"
    shutil.rmtree(terms, ignore_errors=True)
    args = ["--retriever", retriever, "--dataset", table["dataset"], "--splits", json.dumps([table["train"], table["val"]]), "--specs", json.dumps(specs), "--out", str(terms)]
    common.wait(common.graphiv("terms.py", *args), log=log)
    return terms


def finish_bf(spec: dict[str, Any], table: dict[str, Any], fits: dict[tuple[int, int], dict[str, Any]], seconds: float) -> None:
    """
    Picks the core, pool and point best on val, the first in grid order on a tie, and writes the recipe and report
    """
    grid = [(c, p) for c in table["cores"] for p in table["pools"]]
    best = max(grid, key=lambda g: fits[g]["candidates"][fits[g]["picked"]]["judged"])
    pick = fits[best]["candidates"][fits[best]["picked"]]
    report = {"k": table["k"], "spec": table, "seconds": seconds, "pick": {"core": best[0], "pool": best[1], "point": fits[best]["picked"], "judged": pick["judged"]}}
    report["grid"] = {f"core-{c}-pool-{p}": fits[(c, p)] for c, p in grid}
    (common.TUNE_RUNS / spec["name"] / "report.json").write_text(json.dumps(report, indent=1))
    write_recipe(spec, best[0], best[1], {"weights": {t.upper(): pick["exponents"][t] for t in WEIGHTS}})
    common.say(f"{spec['name']}: core {best[0]}, pool {best[1]}, {fits[best]['picked']}, judged R@{table['k']} {pick['judged']:.4f}")


def tune_bf(group: list[tuple[dict[str, Any], Path]]) -> None:
    """
    One retriever and recipe's BoundedField tunes, the grid ones pool by pool and then the at_pick ones together
    """
    retriever = group[0][0]["retriever"]
    grid_tunes = [(s, p) for s, p in group if not s.get("at_pick")]
    pick_tunes = [(s, p) for s, p in group if s.get("at_pick")]
    if grid_tunes:
        tables = {s["name"]: method_spec(s) for s, _ in grid_tunes}
        logs = {s["name"]: start(s, p) / "log" for s, p in grid_tunes}
        started = time.time()
        first = tables[grid_tunes[0][0]["name"]]
        fits: dict[str, dict[tuple[int, int], dict[str, Any]]] = {s["name"]: {} for s, _ in grid_tunes}
        for pool in first["pools"]:
            terms = build_terms(retriever, first, [(c, pool) for c in first["cores"]], logs[grid_tunes[0][0]["name"]])
            jobs = [(s, c) for s, _ in grid_tunes for c in first["cores"]]
            with ThreadPoolExecutor(PARALLEL) as ex:
                results = list(ex.map(lambda j, terms=terms, pool=pool: fit_one(j[0], tables[j[0]["name"]], terms, j[1], pool), jobs))
            for (s, c), r in zip(jobs, results, strict=True):
                fits[s["name"]][(c, pool)] = r
            shutil.rmtree(terms)
        for s, _ in grid_tunes:
            finish_bf(s, tables[s["name"]], fits[s["name"]], time.time() - started)
    if pick_tunes:
        tables = {s["name"]: method_spec(s) for s, _ in pick_tunes}
        logs = {s["name"]: start(s, p) / "log" for s, p in pick_tunes}
        first = tables[pick_tunes[0][0]["name"]]
        at = (first["cores"][0], first["pools"][0])
        started = time.time()
        terms = build_terms(retriever, first, [at], logs[pick_tunes[0][0]["name"]])
        with ThreadPoolExecutor(PARALLEL) as ex:
            results = list(ex.map(lambda sp: fit_one(sp[0], tables[sp[0]["name"]], terms, *at), pick_tunes))
        shutil.rmtree(terms)
        for (s, _), r in zip(pick_tunes, results, strict=True):
            finish_bf(s, tables[s["name"]], {at: r}, time.time() - started)


def tune_ppr(spec: dict[str, Any], src: Path) -> None:
    out = start(spec, src)
    table = method_spec(spec)
    started = time.time()
    report_path = out / "report.json"
    common.wait(common.graphiv("ppr.py", "--retriever", spec["retriever"], "--dataset", table["dataset"], "--spec", json.dumps(table), "--out", str(report_path)), log=out / "log")
    report = json.loads(report_path.read_text())
    report |= {"spec": table, "seconds": time.time() - started}
    report_path.write_text(json.dumps(report, indent=1))
    p = report["pick"]
    write_recipe(spec, p["core"], p["pool"], {"ppr": {"ALPHA": p["alpha"], "TELEPORT": p["teleport"], "DIRECTION": p["direction"]}})
    common.say(f"{spec['name']}: core {p['core']}, pool {p['pool']}, {p['alpha']}/{p['teleport']}/{p['direction']}, R@{table['k']} {p['recall']:.4f}")


def splits_needed(spec: dict[str, Any]) -> list[list[str]]:
    table = method_spec(spec) if not spec.get("at_pick") else common.read_toml(common.TUNING / f"{spec['recipe']}.toml")[spec["method"]]
    dataset = common.read_toml(common.TUNING / f"{spec['recipe']}.toml")["dataset"]
    return [[dataset, table["train"]], [dataset, table["val"]]] if spec["method"] == "bf" else [[dataset, table["split"]]]


def promote() -> None:
    """
    Copies the four nomic BoundedField recipes into the checkout's retrieval_recipes/

    Raises:
        ValueError: When a recipe's pool differs from the COSINE_LIMIT the server reads.
    """
    target = common.ROOT / "retrieval_recipes"
    for recipe in common.RECIPES:
        source = common.TUNE_RUNS / f"nomic-{recipe}_bf_seed0" / "recipe.toml"
        tuned = common.read_toml(source)
        if tuned["POOL"] != int(common.CONFIG["retrieve"]["COSINE_LIMIT"]):
            raise ValueError(f"{source} picked pool {tuned['POOL']}, which the server's COSINE_LIMIT does not follow")
        shutil.copyfile(source, target / f"{recipe}.toml")
        print(f"copied {source} to {target / f'{recipe}.toml'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("tunes", nargs="*", help="toml paths or glob patterns of names under experiments/tune/")
    _ = ap.add_argument("--rerun", action="store_true", help="redo tunes whose recipe exists")
    _ = ap.add_argument("--promote", action="store_true", help="copy the nomic BoundedField recipes into retrieval_recipes/")
    args = ap.parse_args()
    if args.promote:
        promote()
        return
    common.guard()
    paths: list[Path] = []
    for t in args.tunes:
        paths += [Path(t)] if t.endswith(".toml") else sorted((common.EXPERIMENTS / "tune").glob(f"{t}.toml"))
    specs = [(read(p), p) for p in dict.fromkeys(paths)]
    todo = [(s, p) for s, p in specs if args.rerun or not done(s)]
    print(f"{len(specs)} tunes, {len(specs) - len(todo)} done", flush=True)
    # base tunes before the at_pick ones that read their pick
    todo.sort(key=lambda sp: (sp[0]["retriever"], sp[0]["recipe"], sp[0]["method"], bool(sp[0].get("at_pick"))))
    for retriever, rows in itertools.groupby(todo, key=lambda sp: sp[0]["retriever"]):
        rows = list(rows)
        needed = sorted({tuple(x) for s, _ in rows for x in splits_needed(s)})
        common.run_graphiv("queries.py", "--retriever", retriever, "--splits", json.dumps(needed))
        for (_, recipe, method), group in itertools.groupby(rows, key=lambda sp: (sp[0]["retriever"], sp[0]["recipe"], sp[0]["method"])):
            group = list(group)
            common.say(f"== {retriever} {recipe} {method}: {len(group)} tunes, {time.strftime('%Y-%m-%d %H:%M:%S')}")
            if method == "bf":
                tune_bf(group)
            else:
                for s, p in group:
                    tune_ppr(s, p)


if __name__ == "__main__":
    main()
