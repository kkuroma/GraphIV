"""
Paths, run names, jsonl files, the frozen citation graph, and the call into GraphIV's own environment

The checkout's config.toml is read first and this directory's config.toml only fills what it leaves out
A run's name is a function of its toml's fields, so every artifact under runs/ is found by name alone
"""

import json
import os
import subprocess
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def merge(root: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    """
    Two parsed configs as one, the root's value winning wherever both set a key

    Args:
        root: The checkout's config.
        local: This directory's config.

    Returns:
        The merged config.
    """
    out = dict(local)
    for key, value in root.items():
        held = out.get(key)
        out[key] = merge(value, held) if isinstance(value, dict) and isinstance(held, dict) else value
    return out


def overlay(base: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    """
    A table with another merged over it, nested tables merged key by key

    Args:
        base: The table.
        change: What replaces its values.

    Returns:
        The merged copy.
    """
    out = dict(base)
    for key, value in change.items():
        held = out.get(key)
        out[key] = overlay(held, value) if isinstance(value, dict) and isinstance(held, dict) else value
    return out


HERE = Path(__file__).resolve().parent.parent
LOCAL = tomllib.loads((HERE / "config.toml").read_text())
ROOT = (HERE / LOCAL["paths"]["ROOT"]).resolve()
CONFIG = merge(tomllib.loads((ROOT / "config.toml").read_text()), LOCAL)
PATHS = CONFIG["paths"]
DATA = HERE / PATHS["DATA"]
SOURCES = DATA / "sources"
CASES = DATA / "cases"
CACHE = DATA / "cache"
RUNS = DATA / "runs"
TUNE_RUNS = RUNS / "tune"
TEST_RUNS = RUNS / "test"
EXPERIMENTS = HERE / "experiments"
TUNING = HERE / "tuning"
# the graph every method reads, copied from the store's export so a later ingest cannot move it
GRAPH = SOURCES / "graph_export.npy"
# title, abstract and date of every paper a retriever may return, in corpus order
CORPUS = SOURCES / "corpus.arrow"
CORPUS_IDS = CACHE / "corpus_ids.npy"
# term caches live in RAM, since writing and deleting gigabytes on disk stalls
SCRATCH = Path("/dev/shm/graphiv-benchmarks")
# the process stops rather than let the shared box swap
FLOOR_GB = 10

# test set name to its dataset and split
SETS = {
    "surveymatch_test": ("surveymatch", "test"),
    "frontiermatch_test": ("frontiermatch", "test"),
    "consensusmatch_test": ("consensusmatch", "test"),
    "pasa_test": ("pasa", "test"),
    "pasa_real": ("pasa", "real"),
    "surge_test": ("surge", "test"),
    "scholarcatalyst_core": ("scholarcatalyst", "core"),
    "scholarcatalyst_subfield": ("scholarcatalyst", "subfield"),
}
# the recipe each test set runs under in the main results
OWN_RECIPE = {
    "surveymatch_test": "survey",
    "surge_test": "survey",
    "frontiermatch_test": "frontier",
    "consensusmatch_test": "consensus",
    "pasa_test": "focused",
    "pasa_real": "focused",
    "scholarcatalyst_core": "focused",
    "scholarcatalyst_subfield": "focused",
}
RECIPES = ["survey", "frontier", "consensus", "focused"]
# methods ranked by src/graphiv/rank.py from a retriever's pool, the rest have a src/methods/ module
GRAPH_METHODS = ("none", "bf", "ppr")
TUNED_METHODS = ("bf", "ppr")
KS = [5, 10, 20, 50, 100, 200, 500, 1000]
NDCG_KS = [10, 20, 100]
RETURN = 1000

# where graphiv() copies its child's output, set per run
LOG: Path | None = None
CHILDREN: list[subprocess.Popen] = []


def tune_name(spec: dict[str, Any]) -> str:
    """
    A tune run's name, {retriever}-{recipe}_{method}[_{variant}]_seed{n}

    Args:
        spec: The tune toml.

    Returns:
        The name.
    """
    variant = f"_{spec['variant']}" if spec.get("variant") else ""
    return f"{spec['retriever']}-{spec['recipe']}_{spec['method']}{variant}_seed{spec['seed']}"


def test_name(spec: dict[str, Any]) -> str:
    """
    A test run's name, {retriever}-{set}-{method}, a tuned method with its recipe, variants and seed

    Args:
        spec: The test toml.

    Returns:
        The name.
    """
    head = f"{spec['retriever']}-{spec['dataset']}_{spec['split']}-"
    if spec["method"] not in TUNED_METHODS:
        return head + spec["method"]
    parts = [spec["recipe"], spec["method"], spec.get("tune_variant", ""), spec.get("variant", ""), f"seed{spec['seed']}"]
    return head + "_".join(p for p in parts if p)


def tune_source(spec: dict[str, Any]) -> str:
    """
    The tune run a tuned test reads its recipe from, its own unless [override] names another

    Args:
        spec: The test toml.

    Returns:
        The tune run's name.
    """
    own = {"retriever": spec["retriever"], "recipe": spec["recipe"], "method": spec["method"], "variant": spec.get("tune_variant"), "seed": spec["seed"]}
    return spec.get("override", {}).get("tune", tune_name(own))


def read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Every line of a jsonl file

    Args:
        path: The file.

    Returns:
        One dict per line.
    """
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Writes a jsonl file whole, through a temporary name so a stopped write leaves nothing

    Args:
        path: The file.
        rows: One dict per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(part, path)


def say(text: str) -> None:
    """
    Prints a line and appends it to the current run's log

    Args:
        text: The line.
    """
    print(text, flush=True)
    if LOG is not None:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(text + "\n")


def guard() -> None:
    """
    Kills the running child and exits when available memory falls under the floor
    """

    def watch() -> None:
        while True:
            with open("/proc/meminfo") as f:
                avail = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable"))
            if avail < FLOOR_GB * 1048576:
                say(f"memory guard: {avail / 1048576:.1f} GB available, exiting")
                for child in CHILDREN:
                    os.killpg(child.pid, 9)
                os._exit(3)
            time.sleep(1)

    threading.Thread(target=watch, daemon=True).start()


def graphiv(script: str, *args: str, threads: int = 0, quiet: bool = False) -> subprocess.Popen:
    """
    Starts one script of src/graphiv/ in GraphIV's own environment, where the store and src are

    Args:
        script: The file name under src/graphiv/.
        args: Its arguments.
        threads: The BLAS and torch threads it may use, all when 0.
        quiet: Send its output to the run log only.

    Returns:
        The started child, which wait() finishes.
    """
    # the benchmarks venv would otherwise leak into the checkout's uv
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    if threads:
        env |= {k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    command = ["direnv", "exec", str(ROOT), "env", f"PYTHONPATH={ROOT}:{HERE / 'src' / 'graphiv'}", "uv", "run", "python", str(HERE / "src" / "graphiv" / script), *args]
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    child.quiet = quiet  # type: ignore[attr-defined]
    CHILDREN.append(child)
    return child


def wait(child: subprocess.Popen, log: Path | None = None) -> None:
    """
    Copies a started child's output to the log until it exits

    Args:
        child: What graphiv() started.
        log: Where its lines go, the current run's log when None.

    Raises:
        RuntimeError: When it exits nonzero.
    """
    assert child.stdout is not None
    for line in child.stdout:
        if "direnv:" in line[:12]:
            continue
        if log is not None:
            with open(log, "a") as f:
                f.write(line)
        elif child.quiet:  # type: ignore[attr-defined]
            if LOG is not None:
                with open(LOG, "a") as f:
                    f.write(line)
        else:
            say(line.rstrip())
    code = child.wait()
    CHILDREN.remove(child)
    if code:
        raise RuntimeError(f"{child.args[8:10]} exited {code}")


def run_graphiv(script: str, *args: str) -> None:
    """
    Runs one script of src/graphiv/ to the end, its output in the run log

    Args:
        script: The file name under src/graphiv/.
        args: Its arguments.
    """
    wait(graphiv(script, *args))


@dataclass
class Graph:
    """
    The frozen citation graph as plain arrays, `cites` indexed by the cited paper and `refs` by the citing one
    """

    ids: list[str]
    index: dict[str, int]
    day: np.ndarray
    stage: np.ndarray
    cites_ptr: np.ndarray
    cites_idx: np.ndarray
    refs_ptr: np.ndarray
    refs_idx: np.ndarray

    def day_number(self, when: str) -> int:
        """
        A date as the integer the export stores, days since 1970

        Args:
            when: An iso date.

        Returns:
            The day number.
        """
        return int((np.datetime64(when) - np.datetime64("1970-01-01")) // np.timedelta64(1, "D"))

    def citing(self, row: int) -> np.ndarray:
        """
        Every paper citing paper `row`

        Args:
            row: A paper row.

        Returns:
            The citing rows.
        """
        return self.cites_idx[self.cites_ptr[row] : self.cites_ptr[row + 1]]


def load_graph() -> Graph:
    """
    Reads the frozen graph

    Returns:
        The graph with its id to row map built.
    """
    with np.load(GRAPH) as g:
        ids = [a.decode() for a in g["ids"]]
        arrays = {n: g[n] for n in ("day", "stage", "cites_ptr", "cites_idx", "refs_ptr", "refs_idx")}
    return Graph(ids=ids, index={a: i for i, a in enumerate(ids)}, **arrays)


def read_corpus(columns: list[str]) -> Any:
    """
    Columns of the corpus every retriever searches, memory mapped

    Args:
        columns: Any of arxiv_id, title, abstract, date.

    Returns:
        The pyarrow table.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    return ipc.open_file(pa.memory_map(str(CORPUS))).read_all().select(columns)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return repr(value) if isinstance(value, float) else str(value)


def dump_toml(table: dict[str, Any], prefix: str = "") -> str:
    """
    A table as toml text, plain keys first and each nested table under its dotted header

    Args:
        table: Scalars, lists of scalars and nested tables.
        prefix: The dotted path of this table, empty at the top.

    Returns:
        The text.
    """
    plain = [f"{k} = {toml_value(v)}\n" for k, v in table.items() if not isinstance(v, dict)]
    nested = [f"\n[{prefix}{k}]\n" + dump_toml(v, f"{prefix}{k}.") for k, v in table.items() if isinstance(v, dict)]
    return "".join(plain) + "".join(nested)
