"""
Fits BoundedField's exponents at recall@k on PaSa train by minibatch gradient, picked on val, for src/methods/boundedfield.py

Train is pooled in shards of raw source rows and read memory mapped, so memory is one batch whatever the shard count
Runs with scripts/recipe_tuning on PYTHONPATH, whose term cache and smooth recall this reuses
"""

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import torch
from metrics import Sample
from stage0_cache_terms import STATE, TERMS, Pooled, pool_case, start_worker
from stage1_fit_recipes import FIXED, from_exponents, recall_at, to_exponents

from src.core.config import load_config

ROOT = Path(__file__).resolve().parents[3]
SHARD = 1000
WORKERS = 6


def pool_row(c: dict) -> Pooled:
    sample = Sample(sample_id=c["sample_id"], anchors=c["anchors"], titles=[], leaf=[], cutoff=c["cutoff"], key=c["key"], query=c["query"])
    return pool_case(sample)


def write(out: Path, done: list[Pooled]) -> None:
    # one .npy per array, since an .npz is never memory mapped
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "terms.npy", np.concatenate([d[0] for d in done]))
    np.save(out / "hits.npy", np.concatenate([d[1] for d in done]))
    np.save(out / "offsets.npy", np.concatenate([[0], np.cumsum([len(d[1]) for d in done])]))
    np.save(out / "n_key.npy", np.array([d[2] for d in done], np.int32))


def read(path: Path) -> dict[str, np.ndarray]:
    return {n: np.load(path / f"{n}.npy", mmap_mode="r") for n in ("terms", "hits", "offsets", "n_key")}


def cache(config: dict, jobs: list[tuple[Path, list[dict]]]) -> None:
    jobs = [(p, cases) for p, cases in jobs if cases and not p.exists()]
    if not jobs:
        return
    with mp.get_context("spawn").Pool(WORKERS, initializer=start_worker, initargs=(config,)) as workers:
        for out, cases in jobs:
            write(out, list(workers.imap(pool_row, cases, chunksize=4)))
            print(f"cached {out.name}, {len(cases)} cases", flush=True)


class Shards:
    """
    Every train shard memory mapped, drawing one case at a time as tensors
    """

    def __init__(self, paths: list[Path]) -> None:
        self.data = [read(p) for p in paths]
        self.index = [(s, i) for s, d in enumerate(self.data) for i in range(len(d["n_key"]))]

    def get(self, j: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        s, i = self.index[j]
        d = self.data[s]
        lo, hi = int(d["offsets"][i]), int(d["offsets"][i + 1])
        x = np.log1p(np.asarray(d["terms"][lo:hi], np.float32))
        x[:, TERMS.index("hub")] *= -1
        return torch.from_numpy(x), torch.from_numpy(np.asarray(d["hits"][lo:hi])), int(d["n_key"][i])


def fit(shards: Shards, start: torch.Tensor, params: dict) -> torch.Tensor:
    # the same smooth recall as stage 1, sharpened over steps rather than epochs
    rng = np.random.default_rng(params["seed"])
    steps, batch, k = params["steps"], params["batch"], params["k"]
    free = from_exponents(start).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([free], lr=0.05)
    for step in range(steps):
        sharpness = 0.3 * (0.05 / 0.3) ** (step / (steps - 1))
        optimizer.zero_grad()
        exponents = to_exponents(free)
        loss = torch.zeros(())
        for j in rng.choice(len(shards.index), batch, replace=False):
            x, y, n_key = shards.get(int(j))
            score = x @ exponents
            edge = torch.topk(score, min(k, len(score))).values[-1]
            hit = torch.sigmoid((score[y] - edge) / (sharpness * (score.std() + 1e-6)))
            loss = loss - hit.sum() / n_key
        (loss / batch).backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step} loss {float(loss.detach()) / batch:.4f}", flush=True)
    return to_exponents(free).detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--train", required=True, help="jsonl of train cases, each carrying its raw source row")
    _ = ap.add_argument("--val", required=True)
    _ = ap.add_argument("--params", required=True, help="json: rows, shards, steps, batch, k, seed and the start exponents")
    _ = ap.add_argument("--graph", required=True)
    _ = ap.add_argument("--cache", required=True, help="where each shard's pooled terms are kept")
    _ = ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = json.loads(args.params)
    config = load_config(str(ROOT / "config.toml"))
    config["dataset"]["GRAPH_PATH"] = args.graph
    with open(args.train) as f:
        by_row = {c["row"]: c for c in map(json.loads, f)}
    with open(args.val) as f:
        val = [json.loads(line) for line in f]
    # raw rows are shuffled before any is dropped, so a shard is the same rows whatever is kept
    order = list(range(params["rows"]))
    random.Random(params["seed"]).shuffle(order)
    cache_dir = Path(args.cache)
    jobs = [(cache_dir / "dev", val)]
    for s in range(params["shards"]):
        jobs.append((cache_dir / f"train-{s:02d}", [by_row[r] for r in order[s * SHARD : (s + 1) * SHARD] if r in by_row]))
    cache(config, jobs)
    torch.manual_seed(params["seed"])
    shards = Shards([cache_dir / f"train-{s:02d}" for s in range(params["shards"])])
    held = Shards([cache_dir / "dev"])
    dev = [held.get(j) for j in range(len(held.index))]
    print(f"{len(shards.index)} train cases, {len(dev)} val", flush=True)
    start = torch.tensor([float(params["start"].get(t, 0.0)) for t in TERMS])
    fitted = fit(shards, start, params)
    rows = {}
    for label, e in (("start", start), ("fit", fitted)):
        e = e / e[FIXED]
        rows[label] = {"val": recall_at(dev, e, params["k"]), "exponents": {t: round(float(v), 3) for t, v in zip(TERMS, e)}}
        print(label, f"val R@{params['k']} {rows[label]['val']:.4f}", rows[label]["exponents"], flush=True)
    best = max(rows, key=lambda r: rows[r]["val"])
    report = {"train_cases": len(shards.index), "val_cases": len(dev), "picked": best, "candidates": rows, "exponents": rows[best]["exponents"]}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
