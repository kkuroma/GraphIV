# benchmarks

GraphIV's retrieval against the baselines it is compared with, on our three benchmarks, PaSa, SurGE and ScholarCatalyst.
A tune is one toml under `experiments/tune/` and a test is one toml under `experiments/test/`, and each writes one directory under `data/runs/` named by its fields.
The baselines run in this directory's own environment, since `adapters` pins `transformers` 4 and the root project runs 5, while BoundedField, PPR and cosine run on GraphIV's own code through `src/graphiv/`.

```
benchmarks/
  flake.nix, pyproject.toml   python 3.13 and the baselines' own uv lock
  config.toml                 what the checkout's config.toml leaves out: where the data lives and the checkout itself
  scripts/generate_experiment_configs.py   writes every toml under experiments/ from experiments/grid.toml
  scripts/tune.py             runs tune tomls into data/runs/tune/, and --promote copies the nomic recipes to ../retrieval_recipes/
  scripts/run.py              runs test tomls into data/runs/test/
  scripts/results/            one script per paper figure and table, and config.toml with where each is copied
  scripts/copy.sh             copies results/ into ../paper
  results/                    what those scripts wrote, one directory per figure or table name
  tuning/                     one toml per recipe, its grid and its [bf] and [ppr] tables, and embedders.toml
  experiments/                grid.toml, tune/{name}.toml and test/{name}.toml
  src/
    schema.py                 Dataset, a split's cases, and Method, one family of methods ranking a group of tests
    common.py                 paths, run names, jsonl and toml, the frozen graph, and the call into graphiv/
    graphiv/                  the in-memory engine and every script run in the checkout's environment
    datasets/                 surveymatch, frontiermatch, consensusmatch, pasa, surge, scholarcatalyst
    methods/                  graph (cosine, BoundedField, PPR), gar, bm25, specter2, scincl
  data -> /Vault/graphiv/benchmarks
```

## Running It

From this directory, under the checkout's direnv, which carries the native libraries the wheels load.

```sh
direnv exec .. .venv/bin/python -m scripts.generate_experiment_configs
direnv exec .. .venv/bin/python -m scripts.tune 'qwen3_8b-*'
direnv exec .. .venv/bin/python -m scripts.run 'qwen3_8b-surveymatch_test-*'
direnv exec .. .venv/bin/python -m scripts.run experiments/test/nomic-pasa_test-gar_cite.toml --rerun
```

Both take toml paths or glob patterns of names, and skip a run whose directory holds the same toml until `--rerun` is passed.
`scripts/run.py` ranks every test of one retriever and test set in one call, so a pool is built once for all of them.
A query is embedded once per retriever and split into `data/cache/<retriever>/queries/`, by `src/graphiv/queries.py` before anything touches the GPU.

## Names

| run | name |
| --- | --- |
| tune | `{retriever}-{recipe}_{bf,ppr}[_{variant}]_seed{n}` |
| test, tuned | `{retriever}-{dataset}_{split}-{recipe}_{bf,ppr}[_{tune variant}][_{variant}]_seed{n}` |
| test, untuned | `{retriever}-{dataset}_{split}-{none,gar_cite,gar_sim,rerank}` |

`none` is the retriever's own ranking, and `bm25`, `specter2` and `scincl` are retrievers that run only as `none`.
GAR reads `nomic-{dataset}_{split}-none` as its first stage.
A file whose name is not the one its fields give is refused, and so is an `[override]` with no variant in the name.

## A Tune

```toml
retriever = "nomic"
recipe = "survey"
method = "bf"
seed = 0
variant = "without_anc"
at_pick = true

[override]
terms = ["rec", "fro", "anc2", "fro2", "anc_decay", "fro_decay", "cocite", "couple", "bm25", "hub"]
```

| Key | What it does |
| --- | --- |
| `retriever` | A section of `tuning/embedders.toml`, whose vectors are `data/cache/<retriever>/corpus.npy` |
| `recipe` | `tuning/<recipe>.toml`, its dataset, its grid of cores and pools, and its `[bf]` and `[ppr]` tables |
| `seed` | The fit's seed for BoundedField, the sample's seed for PPR |
| `at_pick` | Fit only at the core and pool the seed-0 tune picked, instead of the whole grid |
| `[override]` | Merged over the method's table, so a variant changes only what it names |

A BoundedField tune caches every train and val case's terms at each core of one pool size in `/dev/shm`, fits each core in its own process, and moves to the next pool size.
`src/graphiv/fit.py` fits the free exponents by minibatch gradient on a smooth recall@k from a neutral start, searches the exact recall with CMA-ES from there, and keeps the point best on val.
The core and pool best on val are picked, the first in grid order on a tie.
A PPR tune scores every core, pool, damping, teleport set and direction on a seeded sample of one split and keeps the first best.
A case's anchors are blocked from the retriever, both expanders and the hub count, so no case sees its own review.

| File in `runs/tune/<name>/` | Holds |
| --- | --- |
| `recipe.toml` | CORE, POOL, HOPS and the exponents or the PPR setting, which tests read |
| `report.json` | The method's table as it ran, the pick, and every grid point's candidates and val score |
| `fits/` | One BoundedField fit's report per core and pool |
| `tune.toml`, `log` | The toml as it ran and the output |

## A Test

```toml
retriever = "qwen3_8b"
dataset = "surveymatch"
split = "test"
method = "bf"
recipe = "survey"
seed = 0
variant = "zero_anc"

[override.weights]
ANC = 0.0
```

| Key | What it does |
| --- | --- |
| `method` | `none`, `bf`, `ppr`, `gar_cite`, `gar_sim` or `rerank` |
| `recipe`, `seed`, `tune_variant` | Which tune run's `recipe.toml` a tuned test reads |
| `[override]` | Merged over that recipe, `tune` naming another retriever's tune run in place of its own |

| File in `runs/test/<name>/` | Holds |
| --- | --- |
| `rankings.jsonl` | One line per case: its 1,000-paper ranking, recall at 5 to 1,000 and nDCG at 10, 20 and 100 |
| `summary.json` | The means, the tune run read, and the group's wall time |
| `recipe.toml` | The recipe as it ran, override merged in |
| `test.toml`, `log` | The toml as it ran and its group's output |

## The Data

| Path | Holds |
| --- | --- |
| `sources/` | The 2026-10-22 release, PaSa's two query sets, SurGE with its title map, ScholarCatalyst from HF at `a5a7346`, `graph_export.npy`, the graph frozen on 2026-10-21, and `corpus.arrow`, every searchable paper's title, abstract and date |
| `cases/{dataset}/{split}.jsonl` | Each split as cases, built from `sources/` on first use |
| `cache/` | Everything rebuildable from `sources/` and model weights, below |
| `runs/tune/`, `runs/test/` | One directory per run |
| `leaked-2026-11-05/` | Runs made before anchors were blocked past the retriever |
| `superseded-2026-11-23/` | PPR runs pooled from Postgres HNSW |
| `prelayout-2026-11-23/` | Every run, recipe, fit, log and ablation of the flat layout before this one |

| Path in `cache/` | Holds | Rebuilt by |
| --- | --- | --- |
| `corpus_ids.npy` | The corpus order every vector file shares | `src/graphiv/cache.py` |
| `graph/` | CSR both ways, citers sorted by day with a `(row << 16) \| day` key for the hub count, and which rows are in the corpus | `src/graphiv/cache.py` |
| `keyword/` | BoundedField's BM25 postings, document lengths and vocabulary | `src/graphiv/cache.py` |
| `{retriever}/corpus.npy` | Every corpus paper's vector, fp16, in corpus order | the embedder over `corpus.arrow`, hours for a large one |
| `{retriever}/queries/` | Each split's query vectors and a hash of its texts | `src/graphiv/queries.py` |
| `openscholar/{set}.npz` | Every reranker score GAR has computed, by case and paper | `src/methods/gar.py` |

| Footnote | |
| --- | --- |
| Pools | Every retriever's top `POOL` is exact cosine over its own `corpus.npy`, never the store's HNSW index |
| Corpus | ScholarCatalyst has every ranking cut to its corpus before the 1,000-paper cut |
| GAR | The three variants step every case together and share one score cache, and `src/methods/gar.py` matches `pyterrier_adaptive` ranking for ranking on the same scores |
| Promote | `scripts.tune --promote` refuses a recipe whose POOL differs from `COSINE_LIMIT`, since the server does not read POOL yet |
