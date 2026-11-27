# mcp

The tool server an outside agent researches with, and the routes the webui reads saved runs from.
The agent owns the loop and writes the report, and the server holds the corpus, answers one call at a time and checks what it is given.

```
server.py      start_session and the six tools, their docstrings as the agent reads them, both transports
context.py     the store, the graph, a pipeline per mode and the model, opened once
papers.py      the paper record every tool returns, and the filter every lookup takes
search.py      search_papers: BoundedField from a question or a paper
recall.py      recall_paper: one paper from a description, or none
read.py        read_paper: the sentences a question needs, or a schema's typed fields into a table
citations.py   list_citations, and the newer papers arguing with a table's papers
graph.py       the citation graph among a run's papers, which save_report stores for the webui's timeline
lineages.py    trace_lineages: the poset of citations with an intent among a field's past and present
runs.py        the run: its effort, every call, every quote under its label, its tables and draft
dashboard.py   the read-only routes the webui fetches runs from
skills/        the agent's skills, one directory each, in the SKILL.md format
prompts/       every prompt the tools send to the model, one markdown file each
```

## Running it

```sh
uv run python scripts/serve.py --http --role api         # the tools
uv run python scripts/serve.py --http --role frontend    # the webui over saved runs
```

A run opens on `start_session`, which issues its id, fixes its effort and searches the question word for word, over all years and the last two.
Every call passing that id is logged into the run, whose state is written to `state.json` on every call, so a server restart loses nothing.
`save_report` takes a report object and checks it, and each fault names the path of its field, so a fix resends only that field as a patch.
It writes `report.md`, `report.json`, `run.json` and `calls.jsonl` into the run's directory under `RUNS_PATH`, and a client may add `session.json`, its own transcript, which the webui interleaves with the calls.

| deep effort holds a report to | so |
| --- | --- |
| a table from `read_paper` with a schema, and its coverage | the server places them when the report does not |
| a `trace_lineages` call, and three findings citing its sentences | the report says who extends, critiques or compares against whom |
| an open area checked by a search, one of whose three nearest hits was read | an absence is looked for before it is claimed |
| the top four hits of each opening search, and of the first list of newer papers arguing with the table, read | the writer reads past its own wording |

At deep effort a plain read of several papers is refused until a table exists, so the papers compared are read once.

## Connecting opencode

The configuration the measured runs used, with `<checkout>` the repo and the model on the router's OpenAI-compatible endpoint:

```json
{
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:11434/v1"},
      "models": {"Wordslop-Qwen3.6-27B": {"tool_call": true, "limit": {"context": 262144, "output": 32768}}}
    }
  },
  "model": "local/Wordslop-Qwen3.6-27B",
  "mcp": {"servers": {"graphiv": {"type": "remote", "url": "http://127.0.0.1:6767/mcp", "codemode": false}}},
  "tools": {"bash": false, "edit": false, "write": false, "patch": false, "read": false, "glob": false, "grep": false, "list": false, "webfetch": false, "websearch": false, "codesearch": false, "lsp": false, "task": false, "todowrite": false, "todoread": false, "question": false, "skill": true},
  "tool_output": {"max_bytes": 600000, "max_lines": 20000},
  "skills": ["<checkout>/src/mcp/skills"],
  "instructions": ["<checkout>/src/mcp/skills/graphiv-research/SKILL.md"],
  "compaction": {"auto": true, "buffer": 16000}
}
```

`codemode` off makes every tool a plain call, since code mode let an agent build a call it never awaited.
The built-in tools are off so every answer comes from the corpus, and the skill tool stays on so the agent can load the other three skills.
The question tool is off since a run has no one to answer it.
The research skill goes in as instructions because an agent rarely loads an optional skill on its own.
The output limits are raised so a long `read_paper` reply reaches the agent whole.

## Connecting Claude Code

```sh
claude mcp add --scope user --transport http graphiv https://graphiv.zaphkiel/mcp
```

The api answers only the hosts in `ALLOWED_HOSTS`, so a loopback url gets a 421.
The four skill directories go under `~/.claude/skills`, and `/graphiv-research <question>` starts a report.
A `PostToolUse` hook on `mcp__graphiv__save_report` writes the transcript into the saved run as `session.json`, and zaphkiel's is in `home/dev/claude.nix` of the nixos configs.
