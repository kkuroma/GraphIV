# miner

The queries `trace_lineages` builds its poset from, and the paper reads the citation tools share.
Nothing here reads a question, so a caller passes the vector a ranking needs.

```
store.py     every statement this package sends to Postgres
```

| Function | Returns |
| --- | --- |
| `read_edges` | every labeled citation with both ends inside a set of papers |
| `read_citers` | papers outside a set citing at least some of it with an intent, closest to a question first |
| `read_cosines` | each paper's cosine to a vector |
| `read_papers` | the fields a `Paper` needs for a list of ids |
| `read_seed` | one paper and its stored vector, which a search around that paper sends |

The poset itself is described in [`docs/03_tools.md`](../../docs/03_tools.md#the-poset).
