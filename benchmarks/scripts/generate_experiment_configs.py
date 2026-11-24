"""
Writes every tune toml under experiments/tune/ and every test toml under experiments/test/ from experiments/grid.toml

Each file is named by its fields, so a file whose spec changed is rewritten and stale ones are listed, never deleted
"""

from pathlib import Path
from typing import Any

from src import common

TERMS = ["rec", "anc", "fro", "anc2", "fro2", "anc_decay", "fro_decay", "cocite", "couple", "bm25", "hub"]


def tunes(grid: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in grid["retrievers"]:
        for recipe in common.RECIPES:
            out += [{"retriever": r, "recipe": recipe, "method": m, "seed": 0} for m in common.TUNED_METHODS]
            out += [{"retriever": r, "recipe": recipe, "method": "bf", "seed": s, "at_pick": True} for s in grid["seeds"]]
            free = common.read_toml(common.TUNING / f"{recipe}.toml")["bf"]["terms"]
            for t in free:
                kept = [x for x in free if x != t]
                out.append({"retriever": r, "recipe": recipe, "method": "bf", "seed": 0, "variant": f"without_{t}", "at_pick": True, "override": {"terms": kept}})
    return out


def zeroed(keep: list[str]) -> dict[str, float]:
    return {t.upper(): 0.0 for t in TERMS if t not in keep}


def tests(grid: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for set_name, (dataset, split) in common.SETS.items():
        own = common.OWN_RECIPE[set_name]
        head = {"dataset": dataset, "split": split}
        out += [{"retriever": b, **head, "method": "none"} for b in grid["baselines"]]
        out += [{"retriever": "nomic", **head, "method": g} for g in grid["gar"]]
        for r in grid["retrievers"]:
            out.append({"retriever": r, **head, "method": "none"})
            for recipe in common.RECIPES:
                out += [{"retriever": r, **head, "method": m, "recipe": recipe, "seed": 0} for m in common.TUNED_METHODS]
            bf = {"retriever": r, **head, "method": "bf", "recipe": own}
            out += [bf | {"seed": s} for s in grid["seeds"]]
            if r != grid["transfer_from"]:
                source = common.tune_name({"retriever": grid["transfer_from"], "recipe": own, "method": "bf", "seed": 0})
                out.append(bf | {"seed": 0, "variant": f"from_{grid['transfer_from']}", "override": {"tune": source}})
            free = common.read_toml(common.TUNING / f"{own}.toml")["bf"]["terms"]
            for t in free:
                out.append(bf | {"seed": 0, "variant": f"zero_{t}", "override": {"weights": {t.upper(): 0.0}}})
                out.append(bf | {"seed": 0, "tune_variant": f"without_{t}"})
            out.append(bf | {"seed": 0, "variant": "only_anc", "override": {"weights": zeroed(["anc"])}})
            out.append(bf | {"seed": 0, "variant": "only_cos", "override": {"weights": zeroed([])}})
            out.append(bf | {"seed": 0, "variant": "nohop", "override": {"EXPAND": False}})
    return out


def write(folder: Path, specs: list[dict[str, Any]], name) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    wanted = set()
    changed = 0
    for spec in specs:
        path = folder / f"{name(spec)}.toml"
        wanted.add(path.name)
        text = common.dump_toml(spec)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
            changed += 1
    stale = sorted(p.name for p in folder.glob("*.toml") if p.name not in wanted)
    print(f"{folder.name}: {len(specs)} tomls, {changed} written" + (f", stale: {stale}" if stale else ""), flush=True)


def main() -> None:
    grid = common.read_toml(common.EXPERIMENTS / "grid.toml")
    write(common.EXPERIMENTS / "tune", tunes(grid), common.tune_name)
    write(common.EXPERIMENTS / "test", tests(grid), common.test_name)


if __name__ == "__main__":
    main()
