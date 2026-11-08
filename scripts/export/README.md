# export

Turns the store into files that can leave it, under `[export] EXPORT_PATH`, and publishes them.
Every stage only reads the store, and we run them from the manager's Export tab.

```
stage0_dump_store.py      graphiv-<date>.dump, the whole store as one pg_dump archive, sentences included
stage1_export_parquet.py  graphiv-<date>/, the Parquet release with both classifiers beside it
stage2_export_benchmarks.py  graphiv-benchmarks-<cutoff>/, the three benchmarks split into train, val and test
stage3_publish_hf.py      uploads the newest release, the benchmarks and the cards to the four Hugging Face repos
case_miner.py             mines the benchmark cases out of the store, the same ones for the same seed and cutoff
query_writer.py           writes each case's question once, cached by its input
benchmark_files.py        the published schema and the split assigned per case
cards/<name>/             the README, figures and license of each Hugging Face repo
```

```sh
uv run python scripts/export/stage0_dump_store.py
uv run python scripts/export/stage1_export_parquet.py
uv run python scripts/export/stage2_export_benchmarks.py --dry-run
uv run python scripts/export/stage3_publish_hf.py --dry-run
```

The dump is the backup that `docs/restore.md` restores, and the Parquet release is what we publish:

```
graphiv-<date>/
  papers/       metadata, abstract, review probability, and the embedding as 768 float16
  topics/       each paper's three tree levels and their labels
  citations/    resolved citations between exported papers, with intent and score, and no sentence
  classifiers/  citation_intent and survey, each loadable with from_pretrained
  manifest.json the row counts and each classifier's sha256
```

A citation sentence is the citing paper's own text, and most arXiv licenses do not allow redistributing it, so the release carries none.
The dump and the Parquet export write under a `.part` name and rename it only once everything has landed, so a stopped run leaves nothing that looks finished.
The dump stops once its disk falls under 10 GB free, since that disk may also hold the cluster's WAL.

## The Benchmarks

`[benchmarks] CUTOFF` is the version, since a newer paper never adds a reference to a review already published.
Every case that exists at the cutoff is mined, with no sample cap, and each case's question is written once and frozen in the file.
The split is a hash of each case's sorted anchors under the seed 69420 hardcoded in `benchmark_files.py`, so appending cases never moves an existing one.

```
graphiv-benchmarks-<cutoff>/
  surveymatch/     train.parquet, val.parquet, test.parquet, 60/20/20
  frontiermatch/   the same, 40/20/40
  consensusmatch/  the same, 40/10/50
  manifest.json    the settings, the seed and the counts per split
  README.md        the card, with this version's cutoff and counts filled in
```

`cards/benchmarks/README.md` is the card and lists every column, and [`docs/benchmarks.md`](../../docs/benchmarks.md) defines each benchmark.

## Publishing

The repos are `DATA_REPO`, `CITATION_REPO`, `SURVEY_REPO` and `BENCHMARK_REPO` in `[export]`, and the token is the file `[manager] HF_TOKEN_PATH` points at, or `HF_TOKEN` when that is empty.
A fine-grained token with write access to the four repos is enough.

We compare every file with the hub before uploading anything, by sha256 for a file the hub keeps in LFS and by git blob id otherwise, so an unchanged release uploads nothing.
Each table directory goes up as its own commit, since uploading the whole dataset at once hung twice near the end.
Each release repo is then tagged `GraphIV-<newest paper date>` and the benchmark repo `GraphIV-benchmarks-<cutoff>`, and a tag that already exists stays where it is.
`--dry-run` prints the same plan without uploading or tagging.

The numbers in `cards/data/README.md` are written by hand, so update them before publishing a new release.
