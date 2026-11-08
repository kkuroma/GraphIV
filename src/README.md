# src

The library everything else imports.
Nothing here runs on its own, and nothing here imports `scripts/`.

```
core/      the data types and clients every layer shares
retrieve/  BoundedField, which turns a question into a ranked pool with no language model
miner/     LeyLine, which reads a list of papers for what the field argues about
report/    a paper cut into paragraphs, and the mechanical check on an agent's report
mcp/       the tools an outside agent calls, one file per tool
webui/     the research library over saved runs, in html and css
```

The layers stack as `mcp`, `report`, `miner`, `retrieve`, `core`, top to bottom.
A layer may import the ones below it and never the ones above, and the import-linter contract in `pyproject.toml` checks that.

A recipe file under `retrieval_recipes/` holds what each mode tunes, its core and its exponents.
The pool sizes and the miner's constants are shared by every mode and live in `config.toml` under `[retrieve]` and `[miner]`.

## Where the Models Are

| Layer | Calls a model |
| --- | --- |
| `core`, `retrieve`, `miner`, `report` | never |
| `mcp` | to answer a question from a paper's paragraphs in `read_paper`, and to rule on each candidate in `recall_paper` |

Everything else is arithmetic over the citation graph or string matching, so a ranking can be reproduced from the query vector alone and a report's check needs no model.

The code from before the rewrite is in git at `d0ff854` and earlier.
The preprocessing has no code here: its helper modules sit beside the scripts that use them.
