# papers_preprocess

Builds the `papers` table from the Kaggle metadata snapshot.
Each stage reads its queue off the `stage` column, does one thing and advances the rows that succeeded, so any stage can be stopped and rerun without losing work.

```
schema.sql                   the papers and citations tables, applied once before anything else
indexes.sql                  every index, applied after the bulk loads
stage0_fetch_source_json.py  refreshes the snapshot and inserts every id the table is missing
stage1_download_pdf.py       downloads the PDF of every stage 0 paper
stage2_download_html.py      downloads the LaTeXML HTML of every stage 1 paper
stage3_embed_papers.py       embeds every stage 2 paper
stage4_label_survey.py       scores every stage 3 paper for how much it looks like a review
```

## The Stage Column

```
0  in the table, nothing fetched
1  PDF on disk
2  HTML on disk
3  embedded, embed_full written
4  scored, survey_probability written
```

A paper keeps one PDF, under the version `latest_version` holds.
The bulk mirror only carries v1 of most revised papers, so stage 1 falls back to v1 and saves it under the latest name.
No stage rewrites `latest_version`, or stage 0 would see a new version on every run and fetch the paper again forever.

`is_citation_parsable` is not a stage.
It stays null until `citation_preprocess` has parsed the paper once, then records whether the html had a bibliography, so the fifth of the corpus without one is only ever parsed once.

The cascade is strict: a paper that fails a stage never reaches the next one, so it keeps its metadata and gets no vector.

## Running It

```sh
psql -h "$PWD/.pg" -p 5433 -d arxivkg -f scripts/papers_preprocess/schema.sql

uv run python scripts/papers_preprocess/stage0_fetch_source_json.py
uv run python scripts/papers_preprocess/stage1_download_pdf.py
uv run python scripts/papers_preprocess/stage2_download_html.py
uv run python scripts/papers_preprocess/stage3_embed_papers.py
uv run python scripts/papers_preprocess/stage4_label_survey.py

psql -h "$PWD/.pg" -p 5433 -d arxivkg -f scripts/papers_preprocess/indexes.sql
```

`stage0` takes `--no-fetch` to use the snapshot already on disk, and `--dry-run` to only report what it would insert.
Stage 3 needs the embedding server on `[endpoints] EMBEDDING_ENDPOINT`, which is `llama-router`'s path for the embedding model on port 11434.
