"""
Tunes BoundedField's recipes on the benchmarks' own train and val splits, then writes each one whole to data/recipes/

Each recipe is one toml under tuning/, and the pipeline it tunes over is the [retrieve] block of the checkout's config.toml
--promote copies the tuned recipes into the checkout's retrieval_recipes/, the one write this directory makes outside itself
"""

import argparse
import json
import random
import shutil
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

from src import common, datasets

TUNING = common.HERE / "tuning"
RECIPES = ["survey", "frontier", "consensus", "focused"]
WEIGHTS = ["cos", "rec", "anc", "fro", "anc2", "fro2", "anc_decay", "fro_decay", "cocite", "couple", "bm25", "hub"]


def read_spec(name: str) -> dict[str, Any]:
    """
    One recipe's tuning toml

    Args:
        name: The recipe.

    Returns:
        The spec.
    """
    return tomllib.loads((TUNING / f"{name}.toml").read_text())


def split_cases(spec: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """
    One split's cases, a seeded sample of train when the spec caps it

    Args:
        spec: The tuning spec.
        split: train or val.

    Returns:
        The cases.
    """
    cases = datasets.load(spec["dataset"]).get(spec[split])
    cap = spec["fit"]["train_cases"] if split == "train" else 0
    if cap and cap < len(cases):
        cases = random.Random(spec["fit"]["seed"]).sample(cases, cap)
    return cases


def cache_terms(name: str, spec: dict[str, Any]) -> Path:
    """
    Pools and scores every train and val case at every core, through src/graphiv/terms.py

    Args:
        name: The recipe.
        spec: Its spec.

    Returns:
        The recipe's term directory.

    Raises:
        RuntimeError: When a split's cached cases are not the ones the spec now asks for.
    """
    root = common.TUNED / name / "terms"
    for split in ("train", "val"):
        cases = split_cases(spec, split)
        out = root / split
        manifest = out / "cases.json"
        ids = [c["sample_id"] for c in cases]
        if manifest.exists() and json.loads(manifest.read_text()) != ids:
            raise RuntimeError(f"{out} holds other cases than the spec asks for; delete it to rebuild")
        out.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(ids))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            common.write_jsonl(path, cases)
            common.graphiv("terms.py", "--graph", str(common.GRAPH), "--cases", str(path), "--cores", json.dumps(spec["cores"]), "--out", str(out))
    return root


def write_recipe(name: str, pick: dict[str, Any]) -> Path:
    """
    The picked core and exponents, written under data/recipes/ in the form src reads

    Args:
        name: The recipe.
        pick: The fit report's pick.

    Returns:
        The written recipe.
    """
    weights = "".join(f"{t.upper()} = {pick['exponents'][t]:g}\n" for t in WEIGHTS)
    text = (
        f"# The {name} recipe written by benchmarks/tune.py\n"
        f'NAME = "{name}"\n'
        f"CORE = {pick['core']} # papers the expanders and the walks start from\n\n"
        f"# Exponent per score term\n[weights]\n{weights}"
    )
    out = common.TUNED_RECIPES / f"{name}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def tune(name: str) -> None:
    """
    Caches the terms, fits every core, and writes the picked recipe

    Args:
        name: The recipe.
    """
    spec = read_spec(name)
    common.LOGS.mkdir(parents=True, exist_ok=True)
    common.LOG = common.LOGS / f"tune-{name}.log"
    common.say(f"== tune {name} {time.strftime('%Y-%m-%d %H:%M:%S')}, cores {spec['cores']}, R@{spec['k']}")
    root = cache_terms(name, spec)
    report = common.TUNED / name / "report.json"
    common.graphiv("fit.py", "--terms", str(root), "--spec", json.dumps(spec), "--out", str(report))
    pick = json.loads(report.read_text())["pick"]
    path = write_recipe(name, pick)
    common.say(f"== tune {name} wrote {path}, core {pick['core']}, {pick['point']}, R@{spec['k']} {pick['judged']:.4f}")
    common.LOG = None


def promote() -> None:
    """
    Copies every tuned recipe into the checkout's retrieval_recipes/

    Raises:
        FileNotFoundError: When any of the four is not tuned yet.
    """
    missing = [n for n in RECIPES if not (common.TUNED_RECIPES / f"{n}.toml").exists()]
    if missing:
        raise FileNotFoundError(f"not tuned yet: {', '.join(missing)}")
    target = common.ROOT / "retrieval_recipes"
    target.mkdir(exist_ok=True)
    for n in RECIPES:
        shutil.copyfile(common.TUNED_RECIPES / f"{n}.toml", target / f"{n}.toml")
        print(f"copied {n}.toml to {target}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--recipe", choices=RECIPES, action="append", help="a recipe to tune, repeatable, in the order given")
    _ = ap.add_argument("--promote", action="store_true", help="copy data/recipes/ into the checkout's retrieval_recipes/")
    args = ap.parse_args()
    if bool(args.recipe) == args.promote:
        ap.error("pass --recipe or --promote, not both")
    if args.promote:
        promote()
        return
    common.guard()
    for name in args.recipe:
        tune(name)


if __name__ == "__main__":
    main()
