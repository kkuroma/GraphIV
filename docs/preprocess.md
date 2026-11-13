# Building the store

Reproduce the PostGreSQL DB (`papers`, `citations`, `topic_tree` and `topic_names`) creation. `docs/restore.md` is the shortcut: restore the published dump and skip this page.

## What you supply

| input | config key | note |
| --- | --- | --- |
| metadata dump URL | `[source] SOURCE_URL` | the Kaggle arXiv snapshot, 5.1 GB of JSONL |
| PDF mirror | `[source] PDF_BASE_URL` | Google's arxiv-dataset bucket |
| HTML source | `[source] HTML_BASE_URL`, `HTML_FALLBACK_URL` | arXiv's render, ar5iv before 2024-01 |
| embedding endpoint | `[endpoints] EMBEDDING_ENDPOINT` | llama-server only, since it needs `/tokenize` |
| LLM endpoint | `[endpoints] LLM_ENDPOINT` | any OpenAI-compatible host |
| CUDA GPU | not configured | the student trains and infers on it |
| disk space | `[dataset] PDF_PATH`, `HTML_PATH` | 5.6 TB of PDFs, 87 GB of database |

Everything else defaults in `config.toml`, and every script takes `--config_path`.

## Reproducing the database

Run the following in this exact order:

```sh
nix develop
uv sync
nix run .#pg-start          # applies both schema.sql files, idempotent

P=scripts/papers_preprocess
T=scripts/topic_tree_creation
uv run python $P/stage0_fetch_source_json.py
uv run python $P/stage1_download_pdf.py
uv run python $P/stage2_download_html.py
uv run python $P/stage3_embed_papers.py
uv run python $P/stage4_label_survey.py

uv run python $T/stage0_umap.py
uv run python $T/stage1_create_topic_tree.py
psql -h ./.pg -p 5433 -d arxivkg -f $T/indexes.sql
uv run python $T/stage2_topic_labeling.py

uv run python scripts/citation_preprocess/stage0_create_citation.py --workers 16
psql -h ./.pg -p 5433 -d arxivkg -f $P/indexes.sql
uv run python scripts/citation_preprocess/stage1_label_citation.py
```

Indexes go in after the parse and before the labeling, never before a bulk load.
Every stage is idempotent, so a kill costs only the batch in flight.

`scripts/citation_labeler/` is not in that list.
It writes no table, and stage1 only reads the `best.pt` it leaves behind.

```sh
L=scripts/citation_labeler
uv run python $L/stage0_ape_prompt.py
uv run python $L/stage1_teacher_label.py
uv run python $L/stage2_train_student.py
```

Each of the three directories carries a README with its stage list and queue semantics.

## Time taken

One RTX 3090, one NVMe, 3,141,674 papers.

| step | scope | time |
| --- | --- | --- |
| stage0, insert from the JSON | 3,141,674 rows | 49 s |
| stage1, PDF | network bound | days, resumable |
| stage2, HTML | network bound, rate limited | days, resumable |
| stage3, embed and reduce | 499,537 embedded, 3,103,758 reduced | 2 h 32 m |
| papers stage4, review verdict | one forward pass a paper | 2 h 15 m |
| tree stage1, cluster | three kmeans levels | 8 m |
| tree stage2, name the tree | 4,368 nodes | 1 h 45 m |
| citations stage0, parse | 3,103,758 papers into 180,690,765 rows | 8 h 15 m |
| indexes | six, including HNSW | 9 m 34 s |
| citations stage1, label | 68,911,128 rows at 3,412 rows/s | 5 h 37 m |
| release dump | 87 GB into 18 GB | 4 m 29 s |

## Stage column

`stage` is the only queue, and the papers cascade is strict.
A paper that fails one stage never reaches the next, so it keeps its metadata and gets no vector.

```
papers      0 in the table  1 pdf  2 html  3 embedded  4 clustered
citations   0 extracted     1 labeled
```

30,762 papers stall with no PDF, 7,154 with no HTML, 3,103,758 reach stage 4.
Only a citation that resolved to a paper is labeled, so 72,835,096 of the 180,690,765 rows carry
a target and the other 107.9M sit at stage 0 forever.

The tree is three levels of MiniBatchKMeans at k=16, so 16 grandparents over 256 parents over
4,096 leaves, with a median leaf of 585 papers.

## Delta re-ingest

New papers arrive about twice a month, and the whole list above runs again.
Clustering is global, so tree stage1 refits from scratch and **tree stage2 always relabels the whole tree**.
`fit_umap` refuses to run while the stage 2 queue is non-empty, since a half-finished embed pass
would otherwise fit the model on a biased sample and fail with no error.

## Verification

Never `SELECT count(*)` on `citations` casually, since it is a 53 GB scan.

```sql
SELECT stage, count(*) FROM papers GROUP BY stage ORDER BY stage;
SELECT stage, count(*) FROM citations WHERE target_id IS NOT NULL GROUP BY stage;

SELECT arxiv_id, doi, journal_ref, primary_category, latest_version, surnames
FROM papers WHERE arxiv_id = '0704.0001';
```

The last one returns `10.1103/PhysRevD.76.013009`, `Phys.Rev.D76:013009,2007`, `hep-ph`, `v2` and
`{Balázs,Berger,Nadolsky,Yuan}`, where `Balázs` is the encoding canary.
Then read ten titles out of one leaf, which is the only check that says the tree means anything.

## Reasoning for certain hyperparameter

| decision | evidence |
| --- | --- |
| kmeans at every level, not HDBSCAN at the leaf | HDBSCAN called 47% of papers noise |
| k=16, so 16/256/4,096 | 4000 sounds "about right" |
| DistilBERT, not ModernBERT | ModernBERT is slower', the benefit from long context and FA2 is not used |
| SDPA, not flash-attn 2 | FA2's speed measured 0.76x at max_length and 1.10x dynamic |
| no `paragraph` column | costs over double the database's size |
| length-sorted label batches | 2x on the same GPU, since a global max pads away half the compute |
| `reasoning_effort: "none"` on every labeling call | a reasoning model returns empty content, which parses as a failure |
