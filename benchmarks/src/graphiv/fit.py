"""
Fits one recipe's exponents at each cached core size and picks the core and the point on val, for tune.py

The log of the score is linear in the exponents, so a smooth recall@k is fit by minibatch gradient from a neutral start
CMA-ES then searches the exact recall from the fitted point, and val picks among the start, the fit and the search
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

from src.retrieve.scorer import Weights

TERMS = list(Weights.__dataclass_fields__)
COS, HUB = TERMS.index("cos"), TERMS.index("hub")
LR = 0.05
# optuna 4.9 marks x0 and sigma0 for removal in 6.0, and they still set the start and the step
warnings.filterwarnings("ignore", message="`(x0|sigma0)` has been deprecated")

Case = tuple[torch.Tensor, torch.Tensor, int]


class Shards:
    """
    Every shard of one split at one core, memory mapped and drawn one case at a time
    """

    def __init__(self, split: Path) -> None:
        self.data = [{n: np.load(p / f"{n}.npy", mmap_mode="r") for n in ("terms", "hits", "offsets", "n_key")} for p in sorted(split.glob("shard-*"))]
        self.index = [(s, i) for s, d in enumerate(self.data) for i in range(len(d["n_key"]))]

    def get(self, j: int) -> Case:
        s, i = self.index[j]
        d = self.data[s]
        lo, hi = int(d["offsets"][i]), int(d["offsets"][i + 1])
        # every term enters as log(1 + term), and the hub's sign flips since it divides the score
        x = np.log1p(np.asarray(d["terms"][lo:hi], np.float32))
        x[:, HUB] *= -1
        return torch.from_numpy(x), torch.from_numpy(np.array(d["hits"][lo:hi])), int(d["n_key"][i])

    def take(self, picks: list[int]) -> list[Case]:
        return [self.get(j) for j in picks]


def exponents_of(free: torch.Tensor, slots: list[int]) -> torch.Tensor:
    """
    The full exponent vector, cos pinned at 1, the free terms kept non-negative and the rest at 0

    Args:
        free: One unconstrained value per free term.
        slots: Where each free term sits in TERMS.

    Returns:
        One exponent per term.
    """
    out = torch.zeros(len(TERMS))
    out[COS] = 1.0
    return out.index_put((torch.tensor(slots),), torch.nn.functional.softplus(free))


def recall_at(cases: list[Case], exponents: torch.Tensor, k: int) -> float:
    """
    The exact mean recall@k, ties broken toward the answer

    Args:
        cases: The cases.
        exponents: One exponent per term.
        k: The cutoff.

    Returns:
        The mean over cases.
    """
    total = 0.0
    with torch.no_grad():
        for x, y, n_key in cases:
            score = x @ exponents
            top = score >= torch.topk(score, min(k, len(score))).values[-1]
            total += float((top & y).sum()) / n_key
    return total / len(cases)


def fit(train: Shards, slots: list[int], start: float, k: int, params: dict[str, Any]) -> torch.Tensor:
    """
    Minibatch gradient on a smooth recall@k, the sigmoid sharpening over the steps toward the exact count

    Args:
        train: The train shards.
        slots: Where each free term sits in TERMS.
        start: Every free exponent's starting value.
        k: The cutoff.
        params: steps, batch and seed.

    Returns:
        The fitted exponents.
    """
    rng = np.random.default_rng(params["seed"])
    steps, batch = params["steps"], min(params["batch"], len(train.index))
    free = torch.full((len(slots),), float(np.log(np.expm1(start))), requires_grad=True)
    optimizer = torch.optim.Adam([free], lr=LR)
    for step in range(steps):
        sharpness = 0.3 * (0.05 / 0.3) ** (step / max(steps - 1, 1))
        optimizer.zero_grad()
        exponents = exponents_of(free, slots)
        loss = torch.zeros(())
        for j in rng.choice(len(train.index), batch, replace=False):
            x, y, n_key = train.get(int(j))
            score = x @ exponents
            # the cut and the scale move with the exponents, so inflating every score gains nothing
            edge = torch.topk(score, min(k, len(score))).values[-1]
            hit = torch.sigmoid((score[y] - edge) / (sharpness * (score.std() + 1e-6)))
            loss = loss - hit.sum() / n_key
        (loss / batch).backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"fit step {step} smooth R@{k} {-float(loss.detach()) / batch:.4f}", flush=True)
    return exponents_of(free.detach(), slots)


def search(cases: list[Case], origin: torch.Tensor, slots: list[int], k: int, params: dict[str, Any]) -> torch.Tensor:
    """
    CMA-ES on the exact recall@k, started at the fitted point

    Args:
        cases: The train sample it scores each trial on.
        origin: The fitted exponents.
        slots: Where each free term sits in TERMS.
        k: The cutoff.
        params: trials and seed.

    Returns:
        The best exponents it found.
    """
    names = [TERMS[s] for s in slots]

    def full(values: dict[str, float]) -> torch.Tensor:
        out = torch.zeros(len(TERMS))
        out[COS] = 1.0
        for s, n in zip(slots, names, strict=True):
            out[s] = max(values[n], 0.0)
        return out

    x0 = {n: float(origin[s]) for s, n in zip(slots, names, strict=True)}
    # CMA-ES refuses a start outside its bounds, so each ceiling covers the fitted point
    ceiling = {n: max(10.0, x0[n]) for n in names}

    def objective(trial: optuna.Trial) -> float:
        return recall_at(cases, full({n: trial.suggest_float(n, -1.0, ceiling[n]) for n in names}), k)

    sampler = optuna.samplers.CmaEsSampler(x0=x0, sigma0=0.5, seed=params["seed"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # the fitted point is scored first, so the search can only match or beat it on train
    study.enqueue_trial(x0)
    study.optimize(objective, n_trials=params["trials"])
    return full(study.best_params)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--terms", required=True, help="the recipe's term directory, holding train/ and val/")
    _ = ap.add_argument("--spec", required=True, help="json: the tuning toml, with cores resolved")
    _ = ap.add_argument("--out", required=True, help="the report json")
    args = ap.parse_args()
    spec = json.loads(args.spec)
    params, k = spec["fit"], spec["k"]
    slots = [TERMS.index(t) for t in spec["terms"]]
    torch.manual_seed(params["seed"])
    root = Path(args.terms)
    report: dict[str, Any] = {"k": k, "terms": spec["terms"], "cores": {}}
    for core in spec["cores"]:
        train, val = Shards(root / "train" / f"core-{core}"), Shards(root / "val" / f"core-{core}")
        rng = np.random.default_rng(params["seed"])
        n = len(train.index)
        sample = sorted(rng.choice(n, min(params["search_cases"] or n, n), replace=False).tolist())
        searched_on = train.take(sample)
        judge = val.take(list(range(len(val.index))))
        start = exponents_of(torch.full((len(slots),), float(np.log(np.expm1(spec["start"])))), slots)
        fitted = fit(train, slots, spec["start"], k, params)
        found = search(searched_on, fitted, slots, k, params)
        rows = {}
        for label, e in (("start", start), ("fit", fitted), ("search", found)):
            rows[label] = {
                "train_sample": recall_at(searched_on, e, k),
                "judged": recall_at(judge, e, k),
                "exponents": {t: round(float(v), 3) for t, v in zip(TERMS, e, strict=True)},
            }
            print(f"core {core} {label:6} train R@{k} {rows[label]['train_sample']:.4f} judged R@{k} {rows[label]['judged']:.4f}", flush=True)
        best = max(rows, key=lambda r: rows[r]["judged"])
        report["cores"][str(core)] = {"train_cases": n, "judged_cases": len(judge), "picked": best, "candidates": rows}
    core = max(report["cores"], key=lambda c: report["cores"][c]["candidates"][report["cores"][c]["picked"]]["judged"])
    pick = report["cores"][core]
    report["pick"] = {"core": int(core), "point": pick["picked"], "judged": pick["candidates"][pick["picked"]]["judged"], "exponents": pick["candidates"][pick["picked"]]["exponents"]}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"picked core {core}, {pick['picked']}, judged R@{k} {report['pick']['judged']:.4f}", flush=True)


if __name__ == "__main__":
    main()
