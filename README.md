# GraphIV: a rich, open source, and updating citation graph of arXiv

[![demo](https://img.shields.io/badge/demo-graphiv.kuroma.dev-C04868)](https://graphiv.kuroma.dev)
[![dataset](https://img.shields.io/badge/dataset-kkuroma%2FGraphIV-6248B8?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/kkuroma/GraphIV)
[![citation intent](https://img.shields.io/badge/model-citation%20intent-2A8A60?logo=huggingface&logoColor=white)](https://huggingface.co/kkuroma/GraphIV-citation-intent)
[![survey](https://img.shields.io/badge/model-survey-3060B0?logo=huggingface&logoColor=white)](https://huggingface.co/kkuroma/GraphIV-survey)
[![license](https://img.shields.io/badge/license-AGPLv3-565662)](LICENSE)

GraphIV ("graph" + arXiv) is an open source, reproducible citation graph of all of arXiv, rebuilt every two weeks on a single PC. Each resolved citation between two arXiv papers is labeled with its intent, each paper carries an embedding and a place in a topic tree of 4,096 topics, and the whole graph is served as an MCP server that turns a question into a grounded literature review.

There are tens of thousands of papers released to arXiv each month, and even a single topic easily exceeds several hundred. LLM deep research tools take upwards of 30 minutes per question and often miss niche and recent publications, while the closest tool, [consensus](https://consensus.app), still misses the newest papers and runs on proprietary algorithms. I built GraphIV to catch up with modern AI research using infrastructure anyone can host.

On top of the graph sit two algorithms of my own. BoundedField retrieves and ranks papers by where they sit in the citation graph, with no language model in the ranking, and LeyLine mines the lineages that led a field to where it is and the new papers that have already answered them.

![GraphIV's research fields, its automated research pipeline, and BoundedField and LeyLine zoomed in](assets/figures/poster.png)

## Table of contents
- [Using GraphIV](#using-graphiv)
  - [Live Demo](#live-demo)
  - [Reproduction](#reproduction)
  - [Self-hosting](#self-hosting)
- [Components](#components)
  - [Data](#graphivs-data)
  - [BoundedField](#boundedfield---citation-aware-llm-free-retriever--reranker)
  - [LeyLine](#leyline---magnetic-monte-carlo-trajectory-miner)
  - [Pipeline](#automated-research-pipeline)
- [Repository layout](#repository-layout)
- [Future directions](#future-directions)
- [License](#license) and [citation](#citation)

## Using GraphIV

### Live Demo

[graphiv.kuroma.dev](https://graphiv.kuroma.dev) serves the dashboard over the reports my own server has written. Each report opens on six tabs: the report itself, its method table, its papers, the lineages as a citation graph, where its papers sit in the topic tree, and the run's details. The demo serves the dashboard alone, so asking a new question needs a server of your own, below.

### Reproduction

Everything in GraphIV is rebuilt from public sources: [arXiv's metadata snapshot on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv), arXiv's own HTML and PDF files, and open weight models. The build is ten stages filling four tables, run in this order, and [`docs/preprocess.md`](docs/preprocess.md) is the runbook with what each stage reads and writes. To skip the build, [`docs/restore.md`](docs/restore.md) restores the published dump and serves from it.

```sh
uv run python scripts/papers_preprocess/stage0_fetch_source_json.py --dry-run   # what a new snapshot adds
uv run python scripts/papers_preprocess/stage0_fetch_source_json.py --no-fetch
uv run python scripts/papers_preprocess/stage1_download_pdf.py
uv run python scripts/papers_preprocess/stage2_download_html.py
uv run python scripts/papers_preprocess/stage3_embed_papers.py
uv run python scripts/papers_preprocess/stage4_label_survey.py
uv run python scripts/topic_tree_creation/stage0_umap.py
uv run python scripts/topic_tree_creation/stage1_create_topic_tree.py
uv run python scripts/topic_tree_creation/stage2_topic_labeling.py
uv run python scripts/citation_preprocess/stage0_create_citation.py --workers 16
uv run python scripts/citation_preprocess/stage1_label_citation.py
```

The two labelers the build loads are trained by [`scripts/citation_labeler/`](scripts/citation_labeler/README.md) and described on their cards, [GraphIV-citation-intent](https://huggingface.co/kkuroma/GraphIV-citation-intent) and [GraphIV-survey](https://huggingface.co/kkuroma/GraphIV-survey). The four retrieval benchmarks, defined in [`docs/benchmarks.md`](docs/benchmarks.md) and published at [GraphIV_benchmarks](https://huggingface.co/datasets/kkuroma/GraphIV_benchmarks), are tuned and rerun by [`benchmarks/`](benchmarks/README.md), which writes every case and every ranking under its own data directory, and every figure on this page and on the cards is drawn by `direnv exec . uv run python assets/run.py`.

### Self-hosting

GraphIV ships as an MCP server, which any agent that speaks MCP can drive, plus a dashboard over what it wrote. Everything below was measured on the one PC GraphIV runs on: a Ryzen 7 7700X, 64 GB of RAM and an RTX 3090 with 24 GB.

| Tier | What you get | What it needs |
|---|---|---|
| data | the three tables | a Hugging Face login and 6.1 GB of download, see [the dataset card](https://huggingface.co/datasets/kkuroma/GraphIV) |
| server | the six MCP tools and the dashboard | Postgres 17 with pgvector holding the 90 GB store, about 1 GB of RAM for the citation graph and a ranking, an embedding endpoint serving nomic-embed-text-v2-moe through llama-server (0.9 GB of VRAM), and any OpenAI-compatible chat endpoint, here Wordslop-Qwen3.6-27B in IQ2_M with a 262k context (about 19 GB of VRAM) |
| updates | the rebuild every two weeks | the server tier, plus the manager, the GPU for both labelers (2,012 citations and 214 papers a second) and the embeddings, and the chat endpoint for naming 4,368 topics |
| build | everything from scratch | the update tier, plus 496 GB of HTML and 5.6 TB of PDF for the full text |

The store holds the citation sentences, which are each paper's own text, so it is not published as a dump, and the server tier needs a build of its own for now.

```sh
nix develop                                               # Python 3.13, uv, postgres17 with pgvector
uv sync
uv run python scripts/serve.py --http --role api          # the tools, at http://127.0.0.1:6767/mcp
uv run python scripts/serve.py --http --role frontend     # the dashboard, at http://127.0.0.1:6969
claude mcp add --transport http graphiv http://127.0.0.1:6767/mcp
```

On NixOS, `services.graphiv` runs the cluster, the server, the dashboard and the manager as units, and [`nix/README.md`](nix/README.md) is its page. Every path, port and endpoint is a key of [`config.toml`](config.toml).

## Components

### GraphIV's data

| Table | Rows | One row is |
|---|---|---|
| `papers` | 3,165,827 | an arXiv paper with its metadata, its embedding and how likely it is a literature review |
| `citations` | 74,764,672 | a citation between two of those papers, and its intent |
| `topics` | 3,165,827 | a paper's field, subfield and topic, in a tree of 16, 256 and 4,096 |

The release is gated on [kkuroma/GraphIV](https://huggingface.co/datasets/kkuroma/GraphIV), with one tag per rebuild, the present one being `GraphIV-2026-10-01`. Its card covers the columns, how citations are resolved and how the topic tree is built and named.

### BoundedField - Citation Aware LLM-free Retriever & Reranker

BoundedField takes the 5,000 papers closest to a question, adds what the top 200 cite and what cites them, and ranks the pool by a product of cosine, recency and citation-graph terms whose exponents are tuned per mode, because embedding search finds the papers that sound like a question rather than the ones its field cites. Read more at [`docs/retrieval.md`](docs/retrieval.md).

![How BoundedField scores a paper, its recall at top K against embedding search, what each term adds, and its tuned recipes](assets/figures/bounded-field.png)

### LeyLine - Magnetic Monte Carlo Trajectory Miner

LeyLine walks the papers a field takes positions on, drawing every step by how much the field argues with the next paper, keeps the paths that recur as lineages, and collects the newest papers extending, critiquing or comparing against the answer, because a ranking built on citations cannot see a paper too new to be cited. Read more at [`docs/leyline.md`](docs/leyline.md), and the Topic Queries behind its evaluation are in [`docs/supplement.md`](docs/supplement.md).

![How LeyLine walks toward magnets, the Unseen Newcomers only its successors reach, magnet weight against citations, and one question's lineages](assets/figures/leyline.png)

### Automated research pipeline

![One question through the six MCP tools, each with what it returned](assets/figures/research-pipeline.png)

The agent owns the loop and writes the report, and the server answers one call at a time while holding the corpus, the tuned recipes and the rankings. It logs every call into the session's run and checks the report mechanically when the agent saves it. Over the nine public questions, a standard run took 385 seconds on average, 232 of them in `read_paper` and 137 in the agent's own turns.

| Tool | What it answers |
|---|---|
| `recall_paper` | the one paper a description points at, or that none matches |
| `search_papers` | BoundedField from a question or a paper in one of four modes, or an author's papers |
| `read_paper` | answers from the paragraphs a question needs, every claim under a quote label like [Karpukhin7], or one table row per method from a schema |
| `list_citations` | a paper's citers or references, each with its intent and labeled sentence |
| `trace_lineages` | LeyLine's lines through a set of papers, oldest first, with labeled citing sentences |
| `save_report` | the mechanical check on the agent's report, and the saved run |

Four modes pick the recipe, `focused`, `survey`, `frontier` and `consensus`, each tuned on its own benchmark. [`src/mcp/README.md`](src/mcp/README.md) covers the tools, the skills and connecting opencode.

## Repository Layout

| Path | What it holds |
|---|---|
| [`src/`](src/README.md) | the library, in layers: [`core`](src/core/README.md), [`retrieve`](src/retrieve/README.md) (BoundedField), [`miner`](src/miner/README.md) (LeyLine), [`report`](src/report/README.md), [`mcp`](src/mcp/README.md) and [`webui`](src/webui/README.md) |
| [`scripts/`](scripts/README.md) | the build stages, the labelers, the benchmarks, the export, the manager and `serve.py` |
| [`retrieval_recipes/`](retrieval_recipes) | one toml per mode, its core and its exponents, as `benchmarks/tune.py` wrote them |
| [`benchmarks/`](benchmarks/README.md) | the recipe tuning, and GraphIV's retrieval against the baselines on our three benchmarks, PaSa, SurGE and ScholarCatalyst |
| [`assets/`](assets/README.md) | every figure here and on the cards, and the code that draws them |
| [`docs/`](docs/README.md) | the methods, the benchmarks and the runbooks |
| [`nix/`](nix/README.md) | the package and the NixOS module |
| [`tests/`](tests/README.md) | the tests of `src` |
| [`themes/`](themes/README.md) | the palette and the typeface every figure uses |
| [`config.toml`](config.toml) | every path, endpoint and stage default |

## Future Directions

- My vision is to eventually unify the web of science into a singular graph database, where querying becomes a simple graph operation, and arXiv is the first part.
- Anyone should be able to host the server, which needs a loader from the Hugging Face release into Postgres that fetches the citation sentences from arXiv's HTML.
- A paper too new to be cited is still the open problem: the successors reached 136 of 557 such papers that later took off, and BoundedField's top 20 none.
- The reports pass a mechanical check, yet over the nine public questions an outside judge finds 9% to 21% of their cited sentences unsupported by the quotes behind them, against 3% for Consensus on the six both answered.

### Acknowledgments

Thank you to arXiv for use of its open access interoperability. GraphIV is built on [arXiv's metadata snapshot on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv), on [arXiv](https://arxiv.org)'s own HTML renders and [its PDF bucket](https://www.kaggle.com/datasets/Cornell-University/arxiv) on Google Cloud, and on [ar5iv](https://ar5iv.labs.arxiv.org) for the papers from before arXiv rendered HTML itself.

| Model | Used for |
|---|---|
| [nomic-ai/nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe), served from [its GGUF](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF) | every embedding |
| Wordslop-Qwen3.6-27B | the research agent, and every model call of the MCP tools and the stages |
| [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | the topic names, both labelers' training labels and the benchmark questions of the current release |
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | checking the survey labels against two other model families |
| [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) | both labelers |

## License

| Part | License |
|---|---|
| the code in this repository | [AGPLv3](LICENSE) |
| the dataset, [kkuroma/GraphIV](https://huggingface.co/datasets/kkuroma/GraphIV) | CC BY-NC 4.0, with the arXiv metadata columns CC0 as arXiv releases them |
| the two labelers | Apache 2.0 |

The code is AGPLv3 from the commit after `c484e9a`, and every commit up to and including `c484e9a` was released under GPLv3.

## Citation

```bibtex
@misc{graphiv2026,
  title={GraphIV: An Open, Reproducible Citation Knowledge Graph of arXiv},
  author={kkuroma},
  year={2026},
  howpublished={\url{https://huggingface.co/datasets/kkuroma/GraphIV}}
}
```
