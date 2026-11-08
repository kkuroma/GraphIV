# LeyLine

LeyLine is GraphIV's trajectory miner. It takes the papers a search kept and returns what the field argues with, the lineages that led to it, and the newest papers that have already answered it. Its walks are Monte Carlo: they are drawn toward the papers the field takes positions on, which I call *magnets*, and the paths they keep are the *lineages*.

A ranking, however good, is still a static list, and I think a research needs a story: where does an idea come from, what are the arguments, and where does that lead us today. A ranking build on citation alone is temporally static, while I'm able to use my labeled citation intentes, like extend or critique, to sample the stores out of a list of retrieved papers.

The name is LeyLine refercing Fate/Type-Moon's comcept of a magical power grid that runs from node to node. Here, the power is the argument each citation made (classified into intents). It runs inside what the field [BoundedField](retrieval.md) draws.

![How LeyLine walks toward magnets, the Unseen Newcomers only its successors reach, magnet weight against citations, and one question's lineages](../assets/figures/leyline.png)

The code is [`src/miner/`](../src/miner/README.md). [`leyline.py`](../src/miner/leyline.py) is the algorithm on a typed graph and knows nothing about papers, so [`tests/miner/test_leyline.py`](../tests/miner/test_leyline.py) runs it on a hand-built graph with no store and no GPU, and [`miner.py`](../src/miner/miner.py) is what feeds it citations.

## The graph it reads

We first define what LeyLine takes as input.

> **Definition (typed graph).** A typed graph is $G = (V, E)$ with $E \subseteq V \times V \times I$, where $(c, p, i) \in E$ means $c$ cites $p$ with intent $i$, and
> $$I = \{\text{background}, \text{uses}, \text{compares}, \text{critiques}, \text{extends}, \text{motivates}\}$$
> Each node carries a closeness $s(v) = \max(\cos(q, v), 0)$ to the question $q$ and a date $o(v)$.

The miner builds $G$ from the list a search returns. Papers a reader kept rarely cite each other, and the papers a field argues with are the ones they all cite, so the first 30 of the list are expanded by everything they cite, and $E$ is every labeled citation among that union. On the temporal knowledge graph run of 2026-09-28, the first 30 of 68 kept papers gave a union of 370 papers and 7,415 citations.

## Magnets

A magnet is a paper the field takes a position on, rather than a paper it merely names.

> **Definition (intent weights).** $W : I \to [0, 1]$ says how much citing with intent $i$ is taking a position:
>
> | Intent | $W_i$ |
> |---|---|
> | extends | 1.0 |
> | critiques | 1.0 |
> | compares | 0.8 |
> | uses | 0.4 |
> | motivates | 0.15 |
> | background | 0.0 |

> **Definition (magnet).** For each $p$ that anything cites,
> $$\text{stance}(p) = \sum_{(c, p, i) \in E} W_i\, s(c), \qquad \text{total}(p) = \sum_{(c, p, i) \in E} s(c)$$
> $$m(p) = \frac{1}{Z}\ \text{stance}(p) \cdot \frac{\text{stance}(p)}{\text{total}(p)} \cdot \left(\frac{s(p)}{\max_{v \in V} s(v)}\right)^{\phi}$$
> where $Z$ scales the strongest magnet to $1$ and $\phi = 4$. Each $(c, p, i)$ counts once however many times $c$ cites $p$ that way.

Each factor removes one way a paper can look important without being argued with:

| Factor | What it removes |
|---|---|
| $\text{stance}(p)$ | a paper nobody takes a position on, however often it is named |
| $\text{stance}(p) / \text{total}(p)$ | a paper cited as background by everything, which is the generic hub |
| $(s(p) / \max s)^{\phi}$ | a paper argued about for a reason other than this question |

The closeness enters as a power and not as a factor, since at a power of one a backbone paper the whole pool cites outranks the field's own landmarks.

On the temporal knowledge graph run, the most cited papers in the graph are not its strongest magnets:

| Paper | Citations across GraphIV | Magnet weight |
|---|---|---|
| Learning Sequence Encoders for Temporal Knowledge Graph Completion | 161 | **1.000** |
| Recurrent Event Network | 162 | 0.733 |
| Modeling Relational Data with Graph Convolutional Networks | 2,093 | 0.569 |
| Semi-Supervised Classification with Graph Convolutional Networks | 11,675 | 0.219 |
| Inductive Representation Learning on Large Graphs | 7,260 | 0.214 |
| Graph Attention Networks | 8,006 | 0.174 |

## Reach

A walk needs to know which way the magnets lie, which is what reach measures.

> **Definition (reach).** Let $w : V \to [0, 1]$ be the current magnet weights, starting at $m$. For $v, u \in V$, let $h(v, u)$ be the fewest hops from $u$ back to $v$ along citations, so $v$ is an ancestor of $u$. Then
> $$\rho(v) = \sum_{u : w(u) > 0} w(u)\ \lambda^{h(v, u)}$$
> is the magnet mass reachable forward in time from $v$, discounted by $\lambda = 0.5$ per hop, with $h(v, v) = 0$ and unreachable pairs contributing nothing.

Without the discount, a magnet six hops away counts as much as an adjacent one, and every walk would start at the oldest paper in the graph, since it reaches all of them.

## The walk

A walk starts where the most argued-with mass is reachable and moves forward in time, each step drawn toward the mass still ahead.

> **Algorithm (one walk).** Given $w$ and $\rho$:
> 1. Pick the root $r = \arg\max_{v} \rho(v)\, w(v)$ over the $v$ with $\rho(v) > 0$ and $w(v) > 0$.
> 2. At the current node $v$, collect it if $w(v) > 0$.
> 3. Among the papers citing $v$ with $\rho > 0$, draw the next one with probability proportional to its $\rho$, by giving each candidate $u$ the key $U_u^{1/\rho(u)}$ with $U_u \sim \mathrm{Uniform}(0, 1)$ and taking the largest (Efraimidis and Spirakis, 2006).
> 4. Repeat from step 2 until no candidate is left, and return the path and what it collected.

Rooting on reach alone would start every path at the oldest node again, which is why the root multiplies reach by the node's own weight.

> **Algorithm (lineages).** For each seed $1, \dots, S$:
> 1. Reset $w \gets m$ and seed the sampler with the seed's index.
> 2. Run $T$ walks. After each, recompute $\rho$, count the path, sorted by date, as one more run of that path, and knock down every magnet it collected, $w(v) \gets \kappa\, w(v)$.
>
> Then drop every path another path contains, and rank the rest by
> $$\text{rank}(\pi) = \text{runs}(\pi) \cdot \sum_{v \in \text{collected}(\pi)} m(v)$$

Here $S = 10$, $T = 50$ and $\kappa = 0.4$. The knockdown is what sends the next walk to a part of the graph the last one left alone, and the seeds make the whole thing reproducible: the same graph always returns the same lineages.

On the temporal knowledge graph run, at most 500 walks found 444 distinct paths, of which the report keeps 5.

## Successors

The lineages look backward. The successors look at what has just appeared.

> **Definition (successor).** Let $L$ be the kept list, $t$ the cap and $\Delta = 240$ days. A successor is a paper $c \notin L$ dated in $[t - \Delta, t]$ with at least one citation $(c, p, i)$, $p \in L$, where $i \in \{\text{extends}, \text{critiques}, \text{compares}\}$ and the labeler's confidence in $i$ is at least $0.60$. Its reach is how many papers of $L$ it cites that way, and successors are ranked by
> $$\text{pull}(c) = \text{reach}(c) \cdot \cos(q, c)$$

A successor needs no citations of its own, only its own position on the answer, which is exactly what a citation-based ranking cannot see. The walk runs on the head of the list and the successors on all of it, since the two pull opposite ways: lineages stop reproducing once the graph is wide, while successors keep scaling with every paper added.

## Disputes and connectors

Two smaller reads come off the same graph.

1. A dispute is one sentence per citing pair that took a position, at most two per cited paper, ordered by the cited paper's magnet weight, so the report can quote the field arguing in its own words.
2. A connector is a paper outside the list sitting on a path between two papers in it, with one or two papers between them, which is as far as a bridge stays legible.

## Evaluation

There is no benchmark for a lineage, since no document lists a field's lineages to score against. What can be measured is the cold start, by asking at a date in the past and scoring against what happened after. Five Topic Queries ([`supplement.md`](supplement.md#topic-queries)) were asked on 2025-03-01, with the citation graph running to 2026-09-10. An Unseen Newcomer is a paper of the 365 days before the question with no citations at the question date and 20 or more in the 18 months after.

| Topic | Unseen Newcomers | BoundedField's Top 20 | Successor Top 20 | Successor Set |
|---|---|---|---|---|
| Tabular foundation models | 30 | 0 | 2 | 3 |
| Mechanistic interpretability | 49 | 0 | 5 | 14 |
| Retrieval for scientific literature | 43 | 0 | 4 | 11 |
| Diffusion language models | 195 | 0 | 7 | 45 |
| Test-time compute and reasoning | 240 | 0 | 2 | 63 |
| **All** | **557** | **0** | **20** | **136** |

The best rank any Unseen Newcomer reached in BoundedField's ranking was 41, and the median citation count the successor top 20 held at the question date was 0 for every topic.

## Using it

On a graph of your own, LeyLine takes the edges, a closeness and a date per node, and the settings:

```python
from src.core.schema import Edge
from src.miner.leyline import LeyLine, Settings

edges = [
    Edge(source="c", target="a", intent="extends", sentence="..."),
    Edge(source="d", target="a", intent="critiques", sentence="..."),
    Edge(source="d", target="c", intent="compares", sentence="..."),
]
settings = Settings(
    weights={"extends": 1.0, "critiques": 1.0, "compares": 0.8, "background": 0.0},
    argues=("extends", "critiques", "compares"),
)
closeness = {"a": 0.9, "c": 0.8, "d": 0.7}
order = {"a": 1.0, "c": 2.0, "d": 3.0}
line = LeyLine(edges, settings, closeness, order, focus=closeness)
magnets, lineages = line.magnets(), line.lineages()
```

On GraphIV, the miner does the expansion, the successors and the disputes around it:

```python
from src.miner.miner import Miner
from src.miner.recipe import read_recipe

found = Miner(PG_DEST, graph, read_recipe("retrieval_recipes/frontier.toml")).mine(papers, vector, asked)
```

Every constant above is in the `[miner]` block of the mode's recipe, the same file BoundedField reads, so one mode always picks a matched retriever and miner.

## Limitations

- Lineages have no ground truth, so the walk's design was settled by reading the reports it produced rather than by a number.
- The walk's constants were set by reading the lineages rather than by a score: $\lambda$ came down from $1$ because at $1$ every walk started at the oldest paper, $\phi = 4$ reuses the area gate's exponent, and every mode shares one `[miner]` block.
- At the standard effort the walks rarely repeat a path. In 8 of the 10 runs on the demo every kept lineage was walked once, so their rank falls to the magnet mass they collected.
- Every intent comes from GraphIV's DistilBERT labeler, at macro F1 0.815 against its teacher, so a mislabeled citation moves a magnet.
- The cold-start measurement is five questions on one date, and the successor set it reports is a reach, not a ranking that put the Unseen Newcomers on top.

## Related work

- Main path analysis (Hummon and Doreian, 1989) traces a field's development along the citation path with the highest traversal count. LeyLine instead weights papers by how much the field argues with them and samples many paths toward that mass, keeping the distinct paths rather than the single heaviest one.
- Randomized shortest paths (Saerens et al., 2009) and the betweenness measures built on them ([arXiv:1509.03147](https://arxiv.org/abs/1509.03147), [arXiv:1210.4276](https://arxiv.org/abs/1210.4276)) sample paths between two chosen endpoints. A LeyLine walk has no target, and its steps depend only on the mass reachable ahead.
- Intermediacy ([arXiv:1812.08259](https://arxiv.org/abs/1812.08259)) measures how much a publication lies on paths between two others, which is the question LeyLine's connectors answer by enumeration.

## References

- Efraimidis, P. S., and Spirakis, P. G. (2006). Weighted random sampling with a reservoir. *Information Processing Letters*, 97(5).
- Hummon, N. P., and Doreian, P. (1989). Connectivity in a citation network: the development of DNA theory. *Social Networks*, 11(1).
- Saerens, M., Achbany, Y., Fouss, F., and Yen, L. (2009). Randomized shortest-path problems: two related models. *Neural Computation*, 21(8).
- Kivimäki, I., Lebichot, B., Saramäki, J., and Saerens, M. Two betweenness centrality measures based on randomized shortest paths. [arXiv:1509.03147](https://arxiv.org/abs/1509.03147).
- Lebichot, B., Kivimäki, I., Françoisse, K., and Saerens, M. Semi-supervised classification through the bag-of-paths group betweenness. [arXiv:1210.4276](https://arxiv.org/abs/1210.4276).
- Šubelj, L., Waltman, L., Traag, V., and van Eck, N. J. Intermediacy of publications. [arXiv:1812.08259](https://arxiv.org/abs/1812.08259).
