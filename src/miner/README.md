# miner

LeyLine, the miner.
It takes the papers a search kept, never the question, and reads them as one citation graph to find what the field argues about and how it got there.
The math is in [`docs/leyline.md`](../../docs/leyline.md).

```
leyline.py   LeyLine itself, on a typed graph, knowing nothing about papers
miner.py     Miner, which turns papers into that graph and calls LeyLine once
store.py     every query this package sends to Postgres
recipe.py    reads config.toml's [miner] block, where every constant lives
```

## Why LeyLine Takes No Papers

Its whole input is an edge list typed by intent, a score and an order per node, and a weight per intent.
So `tests/miner/test_leyline.py` runs it on a hand-built ten-node graph with no store and no GPU, and only `miner.py` knows what an arXiv id is.

## What a Call Does

```python
found = Miner(dsn, graph, build_recipe(config["miner"])).mine(papers, vector, asked)
```

The list is first expanded by one citation hop, because papers a reader kept rarely cite each other, while the papers a field argues with are the ones they all cite.
The walks run on the head of the list and the successors on all of it.
Lineages stop repeating once the graph gets wide, while successors keep growing with every paper added.

| Field | What it holds |
| --- | --- |
| `magnets` | the papers the list takes positions on, with their counts per intent |
| `lineages` | dated paths through the graph, and how many runs repeated each one |
| `disputes` | one sentence per citing pair that took a position |
| `successors` | recent papers that extend, critique or compare against the list, which need no citations of their own |
| `connectors` | papers outside the list that sit on a path between two papers inside it |
