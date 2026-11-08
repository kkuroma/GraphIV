# benchmarks

GraphIV's retrieval against the baselines it is compared with, on our three benchmarks, PaSa, SurGE and ScholarCatalyst.
An experiment is one toml naming a method, a dataset and its `[fit]` and `[run]`, and one experiment is one row of one table.
The baselines run in this directory's own environment, since `adapters` pins `transformers` 4 and the root project runs 5, while BoundedField, PPR and nomic run on GraphIV's own code through `src/graphiv/`.

```
benchmarks/
  flake.nix, pyproject.toml   python 3.13 and the baselines' own uv lock
  config.toml                 what the checkout's config.toml leaves out: where the data lives, the checkout itself, and where the tables go
  tune.py                     tunes BoundedField's four recipes into data/recipes/, and --promote copies them to ../retrieval_recipes/
  run.py                      runs experiments, fitting only when the fit is not cached
  make_table.py               every table the paper reports, from runs/ and fit_cache/
  tuning/                     one toml per tuned recipe, over the pipeline in the checkout's config.toml
  experiments/                {row}-{eval}.toml, one per method and eval, plus focused on ScholarCatalyst's two
  src/
    schema.py                 Dataset, a split's cases, and Method, fit and retrieve
    common.py                 paths, jsonl, the frozen graph, and the call into graphiv/
    graphiv/                  scripts run in the checkout's environment, importing its src and nothing here, terms.py and fit.py among them
    datasets/                 surveymatch, frontiermatch, consensusmatch, pasa, surge, scholarcatalyst
    methods/                  boundedfield, nomic, ppr, bm25, specter2, scincl, gar
  data -> /Vault/graphiv/benchmarks
```

## Running It

From this directory, under the checkout's direnv, which carries the native libraries the wheels load.

```sh
direnv exec .. .venv/bin/python run.py --experiment survey-surge
direnv exec .. .venv/bin/python run.py --run-all
direnv exec .. .venv/bin/python run.py --experiment gar_cite-surge --rerun
direnv exec .. .venv/bin/python make_table.py
```

`--experiment` takes a name under `experiments/` or a path to a toml anywhere.
A run whose results exist is skipped, and named as stale when its toml has changed since, until `--rerun` is passed.
A fit is never redone by `--rerun`; deleting its file under `fit_cache/` is what refits it.

## Tuning

BoundedField's recipes are tuned here and nowhere else, on the same case readers the experiments score with.

```sh
direnv exec .. .venv/bin/python tune.py --recipe survey --recipe frontier --recipe focused --recipe consensus
direnv exec .. .venv/bin/python tune.py --promote
```

Each recipe pools every train and val case at every core of its grid and caches the terms under `data/tuning/<recipe>/terms/`.
The expanders hop from the core, so each core builds its own pool on one shared cosine retrieval.
`fit.py` then fits the free exponents at each core by minibatch gradient on a smooth recall@k from a neutral start, searches the exact recall with CMA-ES from there, and picks the core and the point on val.
A case's anchors are blocked from the retriever, both expanders and the hub count, so no case sees its own review.

| key | what it does |
| --- | --- |
| `dataset`, `train`, `val` | the dataset under `src/datasets/` and the splits fit and picked on |
| `k` | the recall the fit maximizes and val picks on |
| `cores` | the core grid, one term cache per value from one pool per case |
| `terms` | the exponents the fit may move, every other at 0 and `cos` pinned at 1 |
| `start` | every free exponent's starting value |
| `[fit]` | the gradient steps and batch, the CMA-ES trials and the train sample they score, the seed, and a cap on train cases |

## An Experiment

```toml
method = "ppr"
dataset = "surge"
split = "test"

# SurGE has no train split, so it takes SurveyMatch's pick
[fit]
train = "surveymatch"
split = "train"
cases = 200
seed = 0
k = 1000
alphas = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]
teleports = ["pool", "core"]
directions = ["back", "both"]

[run]
ks = [100, 1000]
```

| key | what it does |
| --- | --- |
| `method`, `dataset`, `split` | the module under `src/methods/` and `src/datasets/`, and the split scored |
| `[fit] train` | the dataset the fit reads, which may differ from `dataset`, so a cross-fit is one line |
| the rest of `[fit]` | the method's fit settings, hashed with the method and its `FIT_VERSION` into the fit's file name |
| `[run] ks` | the cut-offs recall is written at |
| `[run] ndcg` | the cut-offs nDCG is written at, none when absent |
| the rest of `[run]` | the method's run settings |

## The Data

| path | holds |
| --- | --- |
| `sources/` | the 2026-10-22 release, PaSa's two query sets, SurGE with its title map, ScholarCatalyst from HF at `a5a7346`, and `graph_export.npy`, the graph frozen on 2026-10-21 |
| `cases/{dataset}/{split}.jsonl` | each split as cases, built from `sources/` on first use |
| `index_cache/` | `corpus.parquet` and `nomic.npy` exported from the store, and the SPECTER2 and SciNCL paper vectors in shards |
| `fit_cache/{method}/{key}.json` | what a baseline's fit learned, with the `[fit]` table that made it |
| `tuning/{recipe}/` | `terms/{split}/core-{c}/shard-{i}/`, the cached terms, and `report.json`, every core's candidates and the pick |
| `recipes/{recipe}.toml` | the tuned recipe as `tune.py` wrote it, which BoundedField and PPR run |
| `leaked-2026-11-05/` | the recipes, runs, fits and tuning made before anchors were blocked past the retriever, kept to compare against |
| `tables.md` | what `make_table.py` writes |
| `runs/{name}.jsonl` | one line per case: its ranking, its recall at each k, and its seconds when measured |
| `runs/{name}.json` | the summary: mean recall at each k, mean seconds, the fit's key, the toml it ran with, and the recipe file as it ran |
| `logs/{name}.log` | each run's output |

| Footnote | |
| --- | --- |
| store | BoundedField, PPR and nomic still search the live store's HNSW index, so after an ingest they reproduce only against the 2026-10-21 dump |
| gar_cite | `warm = ["sim"]` runs gar_sim first on each case and reuses its reranker scores, as the stored runs were made; without it the order shifts within the same 1,000 papers |
| GAR cost | each variant scores its own 1,000 papers per case, where the stored runs shared one scoring pass across the three |
| corpus | a dataset with a corpus of its own, ScholarCatalyst, has every ranking cut to it before scoring, and BoundedField cuts before its 1,000 |
