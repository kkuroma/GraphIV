# mcp

The tool server an outside agent researches with, and the routes the webui reads saved runs from.
The agent owns the loop and writes the report, and the server holds the corpus, answers one call at a time and checks what it is given.

```
server.py      the six tools, their docstrings as the agent reads them, and both transports
context.py     the store, the graph, a pipeline per mode and the model, opened once
papers.py      the paper record every tool returns, and the filter every lookup takes
search.py      search_papers: BoundedField from a question or a paper
recall.py      recall_paper: one paper from a description, or none
read.py        read_paper: paragraph retrieval into the reader, quotes kept only when found
citations.py   list_citations: a paper's citers or references with intent and sentence
graph.py       the citation graph among a run's papers, which save_report stores for the webui's timeline
lineages.py    trace_lineages: LeyLine lines over a set of papers
runs.py        the run log: every call, every quote under its label, saved by save_report
dashboard.py   the read-only routes the webui fetches runs from
skills/        the agent's skills, one directory each, in the SKILL.md format
prompts/       every prompt the tools send to the model, one markdown file each
```

## Running it

```sh
uv run python scripts/serve.py --http --role api         # the tools
uv run python scripts/serve.py --http --role frontend    # the webui over saved runs
```

A run is every call of one MCP session since its last passing `save_report`.
`save_report` writes `report.md`, `run.json` and `calls.jsonl` into one directory under `RUNS_PATH`, and a client may add `session.json`, its own transcript, which the webui interleaves with the calls.

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
  "tools": {"bash": false, "edit": false, "write": false, "patch": false, "read": false, "glob": false, "grep": false, "list": false, "webfetch": false, "websearch": false, "codesearch": false, "lsp": false, "task": false, "todowrite": false, "todoread": false, "skill": true},
  "tool_output": {"max_bytes": 600000, "max_lines": 20000},
  "skills": ["<checkout>/src/mcp/skills"],
  "instructions": ["<checkout>/src/mcp/skills/graphiv-research/SKILL.md"],
  "compaction": {"auto": true, "buffer": 16000}
}
```

`codemode` off makes every tool a plain call, since code mode let an agent build a call it never awaited.
The built-in tools are off so every answer comes from the corpus, and the skill tool stays on so the agent can load the other three skills.
The research skill goes in as instructions because an agent rarely loads an optional skill on its own.
The output limits are raised so a long `read_paper` reply reaches the agent whole.

## Connecting Claude Code

```sh
claude mcp add --scope user --transport http graphiv https://graphiv.zaphkiel/mcp
```

The api answers only the hosts in `ALLOWED_HOSTS`, so a loopback url gets a 421.
The four skill directories go under `~/.claude/skills`, and `/graphiv-research <question>` starts a report.
A `PostToolUse` hook on `mcp__graphiv__save_report` writes the transcript into the saved run as `session.json`, and zaphkiel's is in `home/dev/claude.nix` of the nixos configs.
