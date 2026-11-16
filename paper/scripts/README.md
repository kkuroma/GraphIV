# scripts

The generators behind the paper's figures and retrieval tables, each writing straight into `paper/figures` or `paper/tables`.

```
scripts/
  teaser.py        Figure 1, fills teaser.in.tex with one run's search scores into figures/teaser.tex
  teaser.in.tex    the TikZ template of Figure 1
  dataset.py       Figure 2, figures/dataset.pdf, from the counts pinned in dataset.json
  dataset.json     the store's counts on 2026-11-11, as the paper states them
  benchmarks.py    Figure 3, figures/benchmarks.pdf, from the frozen cases in benchmarks/data/cases
  retrieval.py     Figure 4, the retrieval tables and the appendix recipe tables, from benchmarks/data/runs
  runs.py          the appendix tables of the 63 test runs, from their public directories under data/runs
  runs.json        each test run's directory and the client's wall time in seconds
  ablation.py      the appendix ablation table, from benchmarks/data/ablation
```

Run any of them from the repository root, then build the paper.

```sh
direnv exec . uv run python paper/scripts/retrieval.py
cd paper && direnv exec .. latexmk -g -pdf -interaction=nonstopmode main.tex
```

Every row of `tables/runs.tex` names its run's url on graphiv.kuroma.dev.
`dataset.py --refresh` queries the live store instead, which no longer matches the counts the paper states.
