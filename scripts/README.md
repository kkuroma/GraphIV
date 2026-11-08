# scripts

Everything you run.
Each script does one job, and nothing imports it.

## Building the Store

The store is built by four task directories.
Every stage reads its queue off a `stage` column, does one thing and advances the rows that succeeded, so any stage can be stopped and rerun.

```
papers_preprocess/    the papers table, from the Kaggle snapshot to embeddings and review scores
topic_tree_creation/  the topic tree and its labels, from the embeddings
citation_preprocess/  the citations table, parsed from the html and labeled by intent
citation_labeler/     trains the intent labeler the citation stage loads
```

Run `papers_preprocess` first.
The topic tree and the citations only read papers at stage 3 and write their own tables, so those two can run in either order, or at the same time.

`citation_labeler/` sits off the main path.
It writes no table, only the checkpoint that `citation_preprocess/stage1_label_citation.py` loads.

Each directory has its own README with its stages, and `docs/preprocess.md` walks through the whole build in order.

## Flags

Plain argparse.
Every script reads `config.toml` from the repo root, or another file with `--config_path`.

```sh
uv run python scripts/papers_preprocess/stage1_download_pdf.py --limit 1000
uv run python scripts/citation_preprocess/stage0_create_citation.py --workers 16
```

## Benchmarks

`export/stage2_export_benchmarks.py` mines and writes the published benchmarks.
Tuning and scoring the recipes on them lives in `benchmarks/`, which writes `retrieval_recipes/`.

## Export

`export/` writes the full `pg_dump` that `docs/restore.md` restores and the Parquet release we publish, both under `[export] EXPORT_PATH`.

```sh
uv run python scripts/export/stage0_dump_store.py
uv run python scripts/export/stage1_export_parquet.py
```

## The Manager

`graphiv_manager/` is the page that keeps the store current.
It fetches the weekly snapshot and runs every stage above in order, one job at a time.

```sh
uv run python scripts/graphiv_manager/main.py
```

## The Server

```sh
uv run python scripts/serve.py                         # stdio
uv run python scripts/serve.py --http --role api       # the six tools
uv run python scripts/serve.py --http --role frontend  # the dashboard, with no tools at all
```

The agent runs the loop and writes the report, so nothing here answers a question on its own.
