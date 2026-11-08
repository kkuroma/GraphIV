# BoundedField

BoundedField is GraphIV's retriever and reranker. 

As we tend towards LLM driven searching and reranking, the cost of retrieval explodes with the depth of the research. BoundedField is designed to be LLM-free and citation graph-aware, leveraging the citation graph to improve the accuracy of retrieval over plain cosine similarity.

By designing a score system based on citation graph proximity, I was able to double the recall score from pure cosine similarity, tested on past literature reviews from arXiv itself. This work also invents several reproducible [benchmarks](benchmarks.md) for arXiv citation retrieval itself, using the newly introduced `survey_probability` score.

P.S. I named it after Fate/Type-Moon's bounded field, a ward drawn around a territory. The retrieval similarly draws the bounds of a research field around a question, bounding each term in $[0,1]$, and runs [LeyLine](leyline.md) inside the field it draws.

![How BoundedField scores a paper, its recall at top K against embedding search, what each term adds, and its tuned recipes](../assets/figures/bounded-field.png)

The code is [`src/retrieve/`](../src/retrieve/README.md).

## Initial retrieval

We first define the "pool" and "core" from a retrieval process

> **Definition (pool, core).** Let $q$ be the question's embedding, $t$ the date the corpus is capped at, and $\cos(q, p)$ the cosine between $q$ and paper $p$'s embedding. The **pool** $P_0$ is the $5{,}000$ papers dated on or before $t$ with the highest $\cos(q, p)$. For any pool $P$, its **core** $K(P) \subseteq P$ is the $200$ papers of $P$ with the highest cosine.

Using the citation graph, we're able to grow the pool by one citation hop bidirectionally in time. An expander only increases the size of the pool without deleting what is already in the pool:

$$
P_1 = P_0 \cup \{\, p \le t : \exists k \in K(P_0),\ k \text{ cites } p \,\}, \qquad
P = P_1 \cup \{\, p \le t : \exists k \in K(P_1),\ p \text{ cites } k \,\}
$$

where $p \le t$ means $p$ is dated on or before the cap, and each hop keeps at most $20{,}000$ papers.

For the specific case where two areas of the temporal knowledge graph run of 2026-09-28, the pool grew from 5,000 to 6,844 and then 8,481, and from 5,000 to 6,510 and then 7,962. This sets the initial set in which we begin the pruning process.

## Graph-aware Scoring

Every graph term is a walk over the citations among the pool, starting from the same core.

> **Definition (adjacency).** Let $A \in \{0, 1\}^{|P| \times |P|}$ be the citation graph induced on $P$, where $A_{ij} = 1$ iff paper $j$ cites paper $i$. We can walk the graph using matrix multiplication: multiplying by $A$ walks one hop back in time, and multiplying by $A^\top$ walks one hop forward.

> **Definition (core weights, recency).** Let $c \in \mathbb{R}^{|P|}$ put the core's cosine on the core and nothing elsewhere, normalized to sum to one:
> $$c_p = \frac{\cos(q, p)}{\sum_{k \in K(P)} \cos(q, k)} \text{ for } p \in K(P), \qquad c_p = 0 \text{ otherwise}$$
> and let $r_p = 1 / (1 + a_p)$, where $a_p$ is $p$'s age in years at the cap.

We define the following scores as follows

$$
\begin{aligned}
\text{anc} &= A c & \text{fro} &= A^\top c \\
\text{anc2} &= A^2 c & \text{fro2} &= (A^\top)^2 c \\
\text{cocite} &= (A A^\top c) \odot r & \text{couple} &= (A^\top A c) \odot r
\end{aligned}
$$

Intuitively, each term translates to the following

| Term | What it walks |
|---|---|
| `anc` | what the core cites |
| `fro` | what cites the core |
| `anc2`, `fro2` | the same one step further back, or further forward |
| `cocite` | what the core's citers also cite, weighted by recency |
| `couple` | what cites the same papers the core cites, weighted by recency |

`cocite` is co-citation (Small, 1973) and `couple` is bibliographic coupling (Kessler, 1963), both seeded by the core instead of computed over all pairs.

`anc` and `fro` are exactly zero for any paper the core does not touch directly. Two decay terms let a paper further out score at all:

> **Definition (decay).** Let $d^{\leftarrow}(p)$ be the fewest hops from the core to $p$ walking $A$, and $d^{\rightarrow}(p)$ the same walking $A^\top$. Then
> $$\text{anc\_decay}_p = \delta^{\,d^{\leftarrow}(p)}, \qquad \text{fro\_decay}_p = \delta^{\,d^{\rightarrow}(p)}$$
> with $\delta = 0.5$, a value of $1$ on the core, and $0$ past $4$ hops or where the walk never arrives.

The last two terms need no walk: $\cos$ is the cosine itself, and **hub** is how many citations $p$ held across all of GraphIV at the cap, not only within the pool.

## Normalizing the Scores

The scores above span orders of magnitude. Without normalization, a single top paper dominates the score disproportionately.

> **Definition (normalizing, compressing).** For $v \in \mathbb{R}^n$,
> $$N(v) = \frac{\max(v, 0)}{\max_i \max(v_i, 0)}, \qquad C(v) = N\big(\log(1 + 100\, N(v))\big)$$

The walks and hub pass through $C$, cosine and recency through $N$, and the two decay terms are already on $[0, 1]$, since compressing a decay curve would only rescale it onto its own maximum. Every term $x_\tau(p)$ therefore lies on $[0, 1]$, and $C(v)_i = 0$ iff $v_i \le 0$, so a zero stays a zero.

## The Final Score

$$
\mathrm{score}(p) = \frac{\prod_{\tau \in T} \big(1 + x_\tau(p)\big)^{w_\tau}}{\big(1 + x_{\text{hub}}(p)\big)^{w_{\text{hub}}}}
$$

where $T$ is every term but hub, and each exponent $w_\tau \ge 0$ comes from the recipe. Hub divides rather than multiplies, since a paper the whole of arXiv cites tells you nothing about this question.

Since $x_\tau(p) \in [0, 1]$, each factor lies on $[1, 2^{w_\tau}]$, which gives the three properties the whole design rests on:

1. A paper scoring nothing on a term is left alone by it, rather than zeroed. A brand new paper with no citations keeps whatever its cosine and recency earn.
2. $w_\tau = 0$ makes its factor exactly $1$, so dropping a term is an edit to a recipe and needs no new pool.
3. No term can move a paper by more than a factor of $2^{w_\tau}$, so the exponent *is* the term's influence.

Taking logs,

$$
\log \mathrm{score}(p) = \sum_{\tau \in T} w_\tau \log\big(1 + x_\tau(p)\big) - w_{\text{hub}} \log\big(1 + x_{\text{hub}}(p)\big)
$$

so BoundedField is a log-linear ranker over bounded features, and tuning it is choosing eleven nonnegative weights.

## Recipes

A recipe sets the core and every exponent for one mode, and the pool sizes it shares with every other mode live in the `[retrieve]` block of [`config.toml`](../config.toml). For example, this is [`frontier.toml`](../retrieval_recipes/frontier.toml), the recipe designed to prioritize recent, frontier papers on a topic:

```toml
NAME = "frontier"
CORE = 200

[weights]
COS = 1
REC = 1.231
ANC = 0.246
FRO = 0.33
ANC2 = 0.004
FRO2 = 0.153
ANC_DECAY = 0.093
FRO_DECAY = 0.007
COCITE = 1.764
COUPLE = 0.057
BM25 = 0.207
HUB = 2.089
```

```toml
[retrieve]
COSINE_LIMIT = 5000
EXPANSION_LIMIT = 20000
DEPTH = 4
DECAY = 0.5
```

There are four modes/recipes defined in this project, each written by [`benchmarks/tune.py`](../benchmarks/README.md) with `cos` pinned at 1, as follows:

| Term | `focused` | `survey` | `frontier` | `consensus` |
|---|---|---|---|---|
| tuned on | PaSa | SurveyMatch | FrontierMatch | ConsensusMatch |
| `cos` | 1 | 1 | 1 | 1 |
| `rec` | 0.547 | 0.951 | 1.231 | 1.473 |
| `anc` | 0.518 | 1.384 | 0.246 | 2.643 |
| `fro` | 0.011 | 0.182 | 0.33 | 0.022 |
| `anc2` | 0.124 | 0.258 | 0.004 | 0.088 |
| `fro2` | 0.009 | 0.023 | 0.153 | 0.015 |
| `anc_decay` | 0.009 | 0.029 | 0.093 | 0.03 |
| `fro_decay` | 0.005 | 0.017 | 0.007 | 0.018 |
| `cocite` | 0.362 | 1.89 | 1.764 | 0.29 |
| `couple` | 0.046 | 0.017 | 0.057 | 0.011 |
| `bm25` | 0.933 | 0.529 | 0.207 | 0.61 |
| `hub` | 0.005 | 0.223 | 2.089 | 0.027 |

Passing a paper instead of a question replaces the question's vector with that paper's stored one, so the paper sits at a cosine of 1 and the recipe is unchanged.

## Tuning

I tuned the first three modes using their corresponding [benchmarks](benchmarks.md) with lattice search. With $11$ terms at $0$ or $1$, this is $2^{11} = 2{,}048$ settings to be tuned. Then coordinate descent optimizes one exponent at a time over $\{0, 0.25, 0.5, 1, 2, 3, 4\}$ and kept a decision only when the benchmark's headline recall rose, at 1,000 for SurveyMatch and FrontierMatch and at 100 for ConsensusMatch, from three starting points: the lattice's best, the previously shipped point, and every term at $1$.

| Benchmark | Initialization | Initial Score | Optimized Score |
|---|---|---|---|
| SurveyMatch | the lattice's best | 0.8000 | 0.8108 |
| | the shipped point | 0.7704 | **0.8123** |
| | every term at 1 | 0.6416 | **0.8123** |
| FrontierMatch | the lattice's best | 0.7551 | **0.7698** |
| | the shipped point | 0.7365 | 0.7697 |
| | every term at 1 | 0.6659 | 0.7687 |
| ConsensusMatch | the lattice's best | 0.7963 | 0.8000 |
| | the shipped point | 0.6734 | **0.8095** |
| | every term at 1 | 0.5076 | 0.8021 |

We'll explain what each benchmark entails in the next section.

## Evaluation

I designed four benchmarks for this study, defined in [benchmarks.md](benchmarks.md), all of which are LLM-free apart from writing each case's query. The benchmarks use human-written literature reviews (identified with our own survey labeler) as ground truth, measuring our recall with respect to these citations. We can rerun them with [`benchmarks/`](../benchmarks/README.md).

| Benchmark | Number of Papers | Input | Ground truth |
|---|---|---|---|
| SurveyMatch | 300 | One literature review | Its reference list within arXiv |
| FrontierMatch | 300 | Two reviews in the same topic leaf, at most 190 days apart and at cosine 0.80 or more | Papers the later cites but not the earlier |
| ConsensusMatch | 151 | Three reviews in one topic within 190 days | Papers all three cite |
| PrecisionMatch | 300 | The SurveyMatch cases | Papers that either share the case's leaf of the topic tree or are cited by another review in that leaf |

Below is the score (recall, except precision for PrecisionMatch) comparing BoundedField against pure cosine retrieval, scored at each retrieval count.

| Benchmark | Recipe | @100 | @200 | @500 | @1000 |
|---|---|---|---|---|---|
| SurveyMatch | Cosine only | 0.144 | 0.198 | 0.291 | 0.370 |
| | `survey` | **0.408** | **0.559** | **0.743** | **0.812** |
| FrontierMatch | Cosine only | 0.119 | 0.180 | 0.287 | 0.376 |
| | `frontier` | **0.322** | **0.496** | **0.685** | **0.770** |
| ConsensusMatch | Cosine only | 0.269 | 0.372 | 0.512 | 0.609 |
| | `consensus` | **0.809** | **0.887** | **0.956** | **0.980** |
| PrecisionMatch | Cosine only | 0.340 | 0.298 | 0.237 | 0.188 |
| | `survey` | **0.549** | **0.479** | **0.371** | **0.283** |

And here is a breakdown of SurveyMatch recall at 1,000:

| Recipe | Recall |
|---|---|
| cosine only | 0.370 |
| the tuned score without the citation hop | 0.475 |
| every exponent at 1 | 0.642 |
| `anc` alone | 0.800 |
| the tuned score without `fro` | 0.811 |
| the tuned score | **0.812** |

For reference, each score has a ceiling, which is the maximum score we could get, should we be restricted to the expanded pool. We can see that BoundedField comes extremely close to the ceiling while taking less than a second to perform a query on consumer hardware.

| Benchmark | Mean Ceiling | Median Pool | Mean Time Per Query (s) |
|---|---|---|---|
| SurveyMatch | 0.831 | 7,314 | 0.78 |
| FrontierMatch | 0.816 | 8,466 | 0.95 |
| ConsensusMatch | 0.993 | 8,549 | 0.73 |

For scale, two reviews of one topic at cosine 0.80 or more find each other's references at recall 0.265, over 93 pairs. Two experts on the same topic overlap that little, which is what a review's own references are being recovered against.

## After BoundedField

`search_papers` returns the top of the ranking that passes the caller's filter, with each paper's score split by term, and no model enters it. Reading, citations and lineages are separate tools the agent calls on the papers it picks, covered in [`src/mcp/README.md`](../src/mcp/README.md).

## Using it

```python
from src.core.parse_graph_export import load_graph_export
from src.retrieve.recipe import build_pipeline, get_seed_limit, read_recipe

recipe = read_recipe("retrieval_recipes/frontier.toml")
pipeline = build_pipeline(recipe, config["retrieve"], load_graph_export(GRAPH_PATH), PG_DEST)
pool = pipeline.build_pool(question, vector, get_seed_limit(config["retrieve"]), cap)
ranked = pipeline.score_pool(question, vector, pool, cap)
```

`vector` is the question embedded with nomic-embed-text-v2-moe under the `search_query: ` prefix, and every ranked paper carries its terms in `paper.signals`. The citation graph is a 407 MB CSR export that loads once per process, and a ranking takes about half a second warm.

## Discussion and Limitations

- A paper too new to be cited scores exactly zero on `anc`, `anc2`, `cocite` and `hub` since they require at least some citation. BoundedField's top 20 held none of the 557 papers that had no citations then and 20 or more within 18 months. LeyLine's successors are the answer to that case.
- A paper that is neither among the 5,000 nearest nor one hop from a core can never be ranked, which is the ceiling as discussed above. Expanding to two hops greatly increases the pool size and ceiling, but taints the pool with more noise.
- The terms used are classic bibliometrics, co-citation and coupling included. What BoundedField adds is putting them together, a bounded product, the per-mode exponents and the measurement behind them, and using them in a citation graph.

## References

- Small, H. (1973). Co-citation in the scientific literature: a new measure of the relationship between two documents. *Journal of the American Society for Information Science*, 24(4).
- Kessler, M. M. (1963). Bibliographic coupling between scientific papers. *American Documentation*, 14(1).
