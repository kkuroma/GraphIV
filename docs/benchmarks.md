# Benchmarks

GraphIV's retrieval is scored on three benchmarks of my own: SurveyMatch, FrontierMatch and ConsensusMatch. Each asks one question and expects many papers back, and each answer is what arXiv's own literature reviews cite.

Existing literature search benchmarks such as LitSearch match a query to one or a few known papers. A research question is usually answered by a cohort of papers that complement each other, which is what a literature review collects, so I score a retriever against the reviews themselves. The benchmarks are published at [kkuroma/GraphIV_benchmarks](https://huggingface.co/datasets/kkuroma/GraphIV_benchmarks) and every id in them is an arXiv id in [GraphIV](https://huggingface.co/datasets/kkuroma/GraphIV).

![The three benchmarks, their splits and their answers](../scripts/export/cards/benchmarks/benchmarks.png)

The cases are mined and exported by [`scripts/export/stage2_export_benchmarks.py`](../scripts/export/README.md), and the recipes are tuned and scored on them by [`benchmarks/`](../benchmarks/README.md).

## Reviews and Their References

> **Definition (review).** A review is a paper GraphIV's survey labeler scores $0.95$ or more on `survey_probability`, placed in a leaf of the topic tree. At that threshold the labeler's precision is $0.937$.

> **Definition (references).** The references $R(s)$ of a review $s$ are the papers it cites that resolve to a paper in GraphIV, dated before $s$.

A review needs at least $100$ references to be a SurveyMatch case and $25$ to be part of a FrontierMatch or ConsensusMatch case.

Every case carries a question. Gemma 4 26B writes it once from the reviews' titles and abstracts, without mentioning any system, dataset or author, and the question is frozen in the published file so every later run asks the same thing.

## SurveyMatch

Given a research question, what does a review of it cite?

> **Definition (SurveyMatch).** A case is one review $s$. The answer is $R(s)$ and the cap is $s$'s date.

![How a SurveyMatch case is built](../scripts/export/cards/benchmarks/benchmarks-surveymatch.png)

## FrontierMatch

Given where a topic stood, what came next? FrontierMatch is how I evaluate whether a retriever finds the papers a field is rising on.

> **Definition (FrontierMatch).** A case is two reviews $s_1, s_2$ in one leaf, $s_2$ published at most $190$ days after $s_1$, at cosine $0.80$ or more. The answer is $\{ p \in R(s_2) \setminus R(s_1) : p \text{ dated at most } 730 \text{ days before } s_1 \}$, kept when it holds $10$ or more papers, and the cap is $s_2$'s date.

The question is written from $s_1$ alone, so a case asks what the later review added to where the earlier one left the topic.

![How a FrontierMatch case is built](../scripts/export/cards/benchmarks/benchmarks-frontiermatch.png)

## ConsensusMatch

What does every review of a topic agree on?

> **Definition (ConsensusMatch).** A case is three reviews $s_1, s_2, s_3$ in one leaf, all within $190$ days, every pair at cosine $0.75$ or more. The answer is $R(s_1) \cap R(s_2) \cap R(s_3)$ dated on or before the first of them, kept when it holds $5$ or more papers, and the cap is the first review's date.

Its answers are small, $9$ papers at the median, so it is scored at $100$ rather than $1{,}000$.

![How a ConsensusMatch case is built](../scripts/export/cards/benchmarks/benchmarks-consensusmatch.png)

## Versions and Splits

A version is specified by its cutoff, the newest date a mined review may carry, set as `[benchmarks] CUTOFF` in `config.toml`. As newer papers aren't cited by already published reviews, a later cutoff only adds new cases. Every case that exists at the cutoff is mined, with no sample cap.

The split is assigned per subfield of the topic tree, never per case, with the seed $69420$. Subfields are taken in hash order and each goes to the split furthest under its share. Two reviews of one topic share most of their references, so a split by case would let a retriever tune on the answers it is tested on.

| benchmark | cases | train | val | test | answer, median | scored at |
| --- | --- | --- | --- | --- | --- | --- |
| SurveyMatch | 6,109 | 3,665 | 1,215 | 1,229 | 150 | 1,000 |
| FrontierMatch | 518 | 210 | 101 | 207 | 32 | 1,000 |
| ConsensusMatch | 144 | 58 | 13 | 73 | 9 | 100 |

These are the counts of the version `2026-10-22`.

## Scoring

A retriever searches GraphIV with every paper dated after `date_max` and every paper in `anchors` removed, and its score is recall at `headline_k`: the share of `citations` in its top `headline_k`. We run them with `scripts/recipe_tuning/` on the split `[recipe_tuning] SPLIT` sets, so train is for tuning, val for choosing, and test for reporting.

PrecisionMatch reuses the SurveyMatch cases with no answer key, and calls a retrieved paper on topic when it shares the case's leaf or another review in that leaf cites it. It is not published, since its judge is the topic tree itself.

## Limitations

- Every answer is what a later document cited, so all three reward work that was already being noticed. A paper too new to be cited can't be an answer, and [LeyLine](leyline.md) is scored on that case separately.
- Two reviews on one topic written within three months agree at recall $0.265$ over $93$ pairs, so past a point a higher score means matching one author's selection more than the field's.
- A ConsensusMatch question is written from all three reviews while its cap is the first one's date, so its wording can borrow from reviews published up to $190$ days later. The answer can't, since it is capped.
