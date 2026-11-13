# nix

The daemon as a package and as a NixOS module, which is how GraphIV is deployed.
The dev shell and the postgres helpers stay in `flake.nix`, since they are for working on the code rather than for running it.

```
package.nix   the tree the daemons run from, and the wrappers that start them over a uv venv
module.nix    services.graphiv: the cluster, the three units, and config.toml in the store
```

## The package

Nix supplies the interpreter, uv and the native libraries the wheels load, and no Python package.
Every Python dependency comes from `uv.lock`, the same file the dev shell's `.venv` is synced from, so what runs deployed is what the code was run against.

`src/`, `scripts/`, `retrieval_recipes/`, `pyproject.toml`, `uv.lock` and `.python-version` are copied into `share/graphiv`, and `src` goes on `PYTHONPATH` rather than being installed.

| wrapper | runs |
| --- | --- |
| `bin/graphiv-sync` | `uv sync --frozen --no-install-project` into `GRAPHIV_VENV`, on the package's own `python313` |
| `bin/graphiv-serve` | `scripts/serve.py` on that venv |
| `bin/graphiv-manager` | `scripts/graphiv_manager/main.py` on that venv |

`GRAPHIV_VENV` defaults to `$XDG_STATE_HOME/graphiv/venv`, and the module sets it to `venvDir`.
The wrappers put `/run/opengl-driver/lib`, libstdc++, zlib and libGL on `LD_LIBRARY_PATH`, which is where a manylinux wheel expects them on any other distribution.
uv never downloads an interpreter and rebuilds the venv whenever it was made with another one, so a nixpkgs bump that moves `python313` moves the venv with it.
The dev shell sets `UV_PYTHON` to that same interpreter, so `.venv` and the deployed venv sit on one Python.

```sh
nix build .#graphiv
GRAPHIV_VENV=/tmp/graphiv-venv ./result/bin/graphiv-sync
```

## The module

`settings` is `config.toml`, one typed option per key, and every key has a default, so a host names only what differs.
A key the module does not declare fails evaluation, and `nix flake check` holds the options and the checkout's `config.toml` to the same set of keys.

```nix
{
  services.graphiv = {
    enable = true;
    stateDir = "/srv/graphiv";
    settings.endpoints.LLM_NAME = "Wordslop-Qwen3.6-27B";
    settings.serve.ALLOWED_HOSTS = [ "graphiv.example.com" ];
  };
}
```

| unit | port | holds |
| --- | --- | --- |
| `graphiv-pg` | socket only | the cluster, one postmaster of its own |
| `graphiv-venv` | | a oneshot running `graphiv-sync` into `venvDir`, which every unit below requires |
| `graphiv-api` | `serve.API_PORT`, 6767 | the six tools, the store, the graph export |
| `graphiv-frontend` | `serve.FRONTEND_PORT`, 6969 | the dashboard, from a second process registering no tools |
| `graphiv-manager` | `manager.PORT`, 6868 | the snapshot fetch and every preprocess stage, on the GPU |
| `graphiv-graph` | | a path unit restarting the api when the graph export is rewritten |
| `graphiv-config` | | a path unit restarting the api and the frontend when the config override is rewritten |

The frontend opens neither the store nor the graph export, so it cannot reach the corpus.
The manager writes the store and the corpus, so it belongs on a private vhost only.

`stateDir` and `dataDir` are the layout, and every path in `settings` defaults under one of them:

```
<stateDir>/venv          the venv uv fills from uv.lock, torch and its CUDA wheels included
<stateDir>/data          the snapshot, the models, the graph export and the cluster
<stateDir>/data/export   the manager's store dumps and Parquet releases
<stateDir>/data/arxiv    the corpus, as trees of YYMM/<id>.html and YYMM/<id>v<n>.pdf
<stateDir>/cache         model replies keyed by their prompt, safe to delete at any time
<stateDir>/runs          one directory per report written
<stateDir>/manager       the manager's jobs, logs and cached charts
```

Point `settings.dataset.HTML_PATH` and `PDF_PATH` at their resolved paths when the full text is on another mount.
A symlink under `dataDir` leaves every unit ordering on the wrong mount, which turns every full-text read into a silent miss.

`postgres.enable` runs the cluster under `postgres.dataDir`, created by `initdb` when it holds no `PG_VERSION`, and `postgres.settings` reaches the server as `-c` flags.
Turn it off to point `settings.database.PG_DEST` at a cluster run elsewhere.
The api and the frontend only ever read, so their role needs `CONNECT` and `SELECT`.
The stages insert, truncate and copy, so `manager.databaseUser` names the role that owns the four tables when it is not the unit's own user.
The manager's path carries `postgres.package` for the dump stage, and with `postgres.enable` off, `manager.path` has to name a `pg_dump` no older than the cluster.
`manager.huggingface.tokenFile` is the file holding the Hugging Face token the publish stage uploads with, such as a sops secret, and systemd hands it to the manager's unit alone as a credential.
Name it as a string, since a path literal copies the token into the store.
`settings.export.DATA_REPO`, `CITATION_REPO` and `SURVEY_REPO` say where it publishes, and an empty one is skipped.

No option names a tuned exponent.
The exponents live in the recipe files that ship with the package, because an option is a promise to support whatever value is put there, and a detuned retriever still reports numbers.

## The config override

The generated `config.toml` is the default, linked as `<stateDir>/config.toml`, and `<stateDir>/config-override.toml` holds only the keys the manager's graphiv-mcp tab changed.
Every unit reads the default with the override merged over it, and reverting a key in the tab deletes it from the override.
The override is 0600, since it holds `settings.endpoints.LLM_API_KEY` and `EMBEDDING_API_KEY`, which have to stay empty in nix because a value set there lands in the world-readable store.
The tab renders `graphiv-config-schema.json`, every section and option with its type and description, which the module writes and hands the manager as `GRAPHIV_CONFIG_SCHEMA`.
A path, a port, `HOST`, `ALLOWED_HOSTS` and `PG_DEST` are marked built in the tab, since the units, caddy and the sandbox are built around their nix value until the next rebuild.

## What a host has to supply

| | |
| --- | --- |
| network on the first sync | PyPI, for the wheels `uv.lock` names, cached under `CACHE_PATH/uv` |
| the store | the dump restored into the cluster, or built by the manager from a snapshot |
| the graph export | the manager's last stage writes it, and the dump does not carry it |
| two endpoints | a llama-server for embedding, any OpenAI compatible host for the model |
| a GPU, for the manager | the two classifiers load onto it, and `manager.path` carries `nvidia-smi` |
| the corpus, optionally | the two trees above, which the manager's stages fill |

With no corpus, a paper is fetched from arXiv on the first read, one request every three seconds, into those same trees, so the second read is local.
Set `settings.serve.FETCH = false` when they cannot be written, and the daemon then answers out of whatever is already there.
