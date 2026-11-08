# retrieve

BoundedField, the retriever and reranker.
A question goes in and a ranked pool of papers comes out, with no language model anywhere on the way.
The math and the benchmarks are in [`docs/retrieval.md`](../../docs/retrieval.md).

```
cosine.py   CosineRetriever, which picks the first pool by cosine
expand.py   AncestorExpander and FrontierExpander, which add one citation hop in each direction
scorer.py   GraphScorer and its Weights, the only step that ranks
query.py    PaperQuery, which runs the steps in order
recipe.py   builds a pipeline out of a recipe file and config.toml's [retrieve] block
people.py   one person's papers under a shared name, from a known paper, and their topics
```

```python
from src.retrieve.recipe import build_pipeline, get_seed_limit, read_recipe

recipe = read_recipe("retrieval_recipes/frontier.toml")
pipeline = build_pipeline(recipe, config["retrieve"], graph, PG_DEST)
pool = pipeline.build_pool(question, vector, get_seed_limit(config["retrieve"]), cap)
ranked = pipeline.score_pool(question, vector, pool, cap)
```

## Why the Steps Split Where They Do

An expander only adds papers and never ranks them, so the two expanders commute and neither can drop what the other found.
Turning off an expander changes which papers are reachable, while turning off a term changes how they are ordered, so the two questions never mix.

The scorer cuts once, with every term read together.
In a chain, whichever term ran first would decide what the next one was allowed to see.

`GraphScorer` opens no connection.
The steps before it fill `title`, `abstract`, `date` and `cosine`, which sit on the same heap page as the id and cost nothing extra, so scoring needs only the graph export and numpy.

## The Terms

Every graph term is a walk from the same core, the top of the pool by cosine, over the citations among the pool.

```
anc        what the core cites
fro        what cites the core
cocite     what the core's citers also cite
couple     what cites the same papers as the core
anc2       anc, one step further back
fro2       fro, one step further forward
anc_decay  decay to the power of the hops back from the core
fro_decay  decay to the power of the hops forward from the core
```

Each term is normalized to `[0, 1]`, so its factor sits on `[1, 2 ** exponent]` and a term a paper scores nothing on leaves it alone.
The walks go through `compress_long_tail` and the decay terms do not, since compressing a decay curve would only rescale it.

An exponent of 0 drops its term exactly, so an ablation is `replace(weights, couple=0.0)` with no new pool.
The exponents differ per mode, which is why they live in the recipe files and not in the code.
