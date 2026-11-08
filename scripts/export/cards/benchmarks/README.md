---
license: cc-by-nc-4.0
pretty_name: GraphIV Benchmarks
language:
- en
size_categories:
- 1K<n<10K
task_categories:
- text-retrieval
tags:
- arxiv
- literature-review
- scientific-retrieval
- information-retrieval
configs:
- config_name: surveymatch
  data_files:
  - split: train
    path: surveymatch/train.parquet
  - split: validation
    path: surveymatch/val.parquet
  - split: test
    path: surveymatch/test.parquet
  default: true
- config_name: frontiermatch
  data_files:
  - split: train
    path: frontiermatch/train.parquet
  - split: validation
    path: frontiermatch/val.parquet
  - split: test
    path: frontiermatch/test.parquet
- config_name: consensusmatch
  data_files:
  - split: train
    path: consensusmatch/train.parquet
  - split: validation
    path: consensusmatch/val.parquet
  - split: test
    path: consensusmatch/test.parquet
---

# GraphIV Benchmarks

![The three benchmarks, their splits and their answers](benchmarks.png)

## Motivation

Existing literature search benchmark, like LitSearch, often matches a query to a list of one or a few papers. As many research questions can be answered by multiple research papers that often complement each other, often in the form of a literature review cohort, I find existing benchmarks lacking for automated research agents. As such, I built three benchmarks that score a retriever against what arXiv's own reviews cite, often as a list of dozens, if not hundreds, of papers. Every case is mined from [GraphIV](https://huggingface.co/datasets/kkuroma/GraphIV), and every id in it is an arXiv id in that corpus.

A version is specified by its date cutoff, which is the newest date of publication for the reviews inside of this dataset. As newer papers aren't cited by already published reviews, a later cutoff only adds new cases. This is `GraphIV-benchmarks-{{cutoff}}`.

## How Each Benchmark Is Built

Every review is a paper GraphIV's survey labeler scores 0.95 or more, and Gemma-4-26B writes one question per case without mentioning any system, dataset or author. The answer is always what the reviews cite, resolved to papers in GraphIV, which is asking what other papers in arXiv the reviewer cites.

### SurveyMatch

SurveyMatch is the simplest of the three benchmarks. It simply asks: given a research question, what do reviewers of that question cite for their literature reviews.

![How a SurveyMatch case is built](benchmarks-surveymatch.png)

### FrontierMatch

FrontierMatch was designed to evaluate how well a retriever finds "rising" papers in a given field. For each research question, two reviews of the same topic at most 190 days apart are retrieved, and papers cited by the later (and not former) of the two are labeled as "rising frontier" in the dataset.

![How a FrontierMatch case is built](benchmarks-frontiermatch.png)

### ConsensusMatch

ConsensusMatch asks what every review of a field agrees on. For each research question, three reviews of the same topic released within 190 days of each other are retrieved, and only the papers all three of them cite, published before the first of the three, are labeled as the answer. These answers are small, 9 papers at the median, so it is scored at 100 rather than 1,000.

![How a ConsensusMatch case is built](benchmarks-consensusmatch.png)

### Splits

The split is a hash of each case's sorted anchor ids under the seed 69420, so a case keeps its split whatever else is mined, and anyone can recompute it from `anchors`.

{{counts}}

## Downloading and Scoring

```python
from datasets import load_dataset

test = load_dataset("kkuroma/GraphIV_benchmarks", "frontiermatch", split="test")
```

To score a case, retrieve from the GraphIV corpus with every paper dated after `date_max` and every paper in `anchors` removed, then report the share of `citations` found in the top `headline_k`.

## Columns

| column | type | holds |
| --- | --- | --- |
| `case_id` | string | the anchors joined by `_` |
| `split` | string | train, val or test |
| `query_text` | string | the question the case asks |
| `query_drafter` | string | the model that wrote it |
| `date_max` | date | the latest date a retrieved paper may carry: the review's date, the later review's for frontiermatch, the first review's for consensusmatch |
| `anchors` | list of string | the case's own reviews, which a retriever must never return |
| `anchor_titles` | list of string | |
| `anchor_dates` | list of date | |
| `anchor_survey_probability` | list of float | the survey labeler's score for each anchor |
| `citations` | list of string | the answer |
| `n_citations` | int | |
| `headline_k` | int | the cutoff recall is reported at |
| `topic_l0`, `topic_l1`, `topic_l2` | int | the case's field, subfield and topic in the topic tree |
| `topic_l0_name`, `topic_l1_name`, `topic_l2_name` | string | |
| `split_group` | string | the case's subfield, as `topic_l0.topic_l1` |
| `gap_days`, `cosine` | int, float | frontiermatch only, the gap and the similarity between the two reviews |

## Limitations

Every answer is what a later document cited, so all three reward work that was already being noticed, and none of them can score a paper too new to be cited. Two reviews on one topic written within three months agree at recall 0.265, so past a point a higher score means matching one author's selection more than the field's.

The full definitions are in [docs/benchmarks.md](https://git.kuroma.dev/kkuroma/graphiv/src/branch/main/docs/benchmarks.md), and the code that mines, splits and exports these is at [git.kuroma.dev/kkuroma/graphiv](https://git.kuroma.dev/kkuroma/graphiv), licensed as AGPLv3.
