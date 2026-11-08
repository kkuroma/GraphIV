# graphiv_manager

The page that keeps the store current.
It fetches the arXiv snapshot Kaggle publishes each week and runs it through every stage, one job at a time.
A step that exits nonzero halts its job, so the next stage never reads a queue that did not fill.
`citation_labeler/` is left out, since training a labeler is a one-time job we run by hand.
Exporting the benchmarks at `[benchmarks] CUTOFF` is a stage of the Export table, and tuning and scoring the recipes happen in `benchmarks/`, never here.
Publishing to Hugging Face runs only from its own Publish button, never as part of a table or of every stage, and its panel shows which account the token belongs to and which repos it can write.

```
main.py      the entrypoint, which stops a running job with SIGINT before it exits
schedule.py  the loop that picks the next job, and every number the page reads
jobs.py      a job as ordered steps, and how a stop escalates from SIGINT to SIGKILL
stages.py    every stage as a recipe: its script and what it reads, writes and fetches
stats.py     one query per dashboard chart, cached as json in the state directory
discover.py  finds every runnable script and its flags from the files themselves
probes.py    the health checks for the store, the endpoints, the GPU, the disks and the queues
snapshot.py  the snapshot in use, the resumable fetch of a new one, and the commit
routes.py    the json routes and the page
webui/       the page, in the same four palettes as the dashboard
```

```sh
uv run python scripts/graphiv_manager/main.py              # [manager] PORT and STATE_PATH
uv run python scripts/graphiv_manager/main.py --port 3001  # a scratch copy beside the unit
uv run pytest tests/graphiv_manager -q
```

The state directory is `[manager] STATE_PATH`, created on first start.
It holds `jobs/<id>/{job.json,log}`, `stats/<query>.json`, `rates.json` from finished stages, and one snapshot check per file.
Two managers on one state directory would both run jobs, so a scratch copy needs a directory of its own.
The stages connect as `PGUSER` when it is set, which is how the `graphiv-manager` unit writes tables its own user can only read.
