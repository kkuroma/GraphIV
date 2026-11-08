# tests

Tests for the logic in `src`, not for the entrypoints in `scripts`. Every test builds a situation small enough to check by hand, so the suite needs no database, no model and no network and finishes in a few seconds.

```
run.py                runs the whole suite, or one subdirectory
core/                 how config paths resolve, and where a paper's full text is looked for
retrieve/conftest.py  an eight paper citation graph written in the format stage2 exports
retrieve/             the graph reader, the score, the expanders, the BM25 term, the recipe, the pipeline and author lookup
miner/                LeyLine on a hand-built graph, and the successor order
report/               the report check, the TL;DR split and the paragraph chunking
export/               what the publish stage uploads
graphiv_manager/      the manager's runner against fake stages, its disk gate, its schedule, its publish button and its fetch against a local server
```

```sh
uv run python tests/run.py
uv run python tests/run.py retrieve -v
```

## The fixture graph

`retrieve/conftest.py` builds eight papers and eight edges, chosen so that every quantity the retrieval code computes has one obviously correct answer.

```
p0 -> p4, p5      p1 -> p4      p2 -> p5, p6      p3 -> p0, p1      p7 -> p3
```

p0 and p1 carry the two highest cosines, so they are the core. p4 and p5 are what the core cites, which makes them the ancestors, and p4 leads because both core papers reach it. p3 is what cites the core, which makes it the frontier. p7 sits two hops out and scores zero on every one-step channel, which is what the distance channel exists to fix. p3 is dated after p0 and p1, so a date cap of 2021 removes it and the citation counts of p0 and p1 fall to zero.

`-m "not slow and not gpu"` is the default, so a heavy test added later is skipped unless `--slow` is passed.
