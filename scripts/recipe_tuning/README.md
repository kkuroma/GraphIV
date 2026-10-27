# recipe_tuning

Fits BoundedField's tuned recipes on the published benchmarks and scores every recipe on them, which [`docs/benchmarks.md`](../../docs/benchmarks.md) defines.
The cases and their questions come from `graphiv-benchmarks-<cutoff>/` under `[export] EXPORT_PATH`, which `scripts/export/stage2_export_benchmarks.py` writes, so nothing here mines a case or calls a model.

```
driver.py               reading a split, pooling each case once and scoring every recipe on that pool
metrics.py              a published case, its score, and SurveyMatch, FrontierMatch and ConsensusMatch
precision.py            PrecisionMatch, which judges the SurveyMatch cases by the topic tree instead of a key
report.py               the run directory each recipe writes, and the figure comparing them
stage0_cache_terms.py   every train and val case's pool and terms, into [recipe_tuning] CACHE_PATH
stage1_fit_recipes.py   the survey, frontier and consensus exponents, fit on train and picked on val
stage2_evaluate.py      every recipe config.toml lists, on one split of all four benchmarks
```

## Running It

```sh
uv run python scripts/recipe_tuning/stage0_cache_terms.py
uv run python scripts/recipe_tuning/stage1_fit_recipes.py --dry-run
uv run python scripts/recipe_tuning/stage1_fit_recipes.py
uv run python scripts/recipe_tuning/stage2_evaluate.py --split test
```

Stage 0 is the slow one, a pool per case against the store, and a split already cached under the same cutoff is skipped.
Stage 1 fits a smooth recall@k by gradient, since the log of the score is linear in the exponents, then runs CMA-ES on the exact recall from that point.
Val picks among the shipped point, the fit and the search, except for consensus, whose 13 val cases are pooled with train.
It rewrites only each recipe's weights block and its first comment line, and writes the report to `benchmarks/data/recipe_tuning/tuning/<version>.json`.
`ancestry` and `evolution` follow as the survey and frontier points with one direction of the walk at zero.

`[recipe_tuning] SPLIT` is stage 2's default split: train to tune on, val to choose, test to report.
`MAX_SAMPLES` caps how many cases of it a run reads, every case at 0.

## What Comes Out

```
benchmarks/data/recipe_tuning/<split>/<Metric>/
    run.log                     every line, console and file
    performance.png             one horizontal box per recipe
    {recipe}/config.toml        the settings and the recipe, copied verbatim
    {recipe}/scores.jsonl       one row per case
```

Each row carries the query, the cutoff, the anchors, the whole answer key and the whole ranked answer, so any statistic can be recomputed later without the store.
Recipes that share a retrieval prefix share one pool, so six of them cost one retrieval.
