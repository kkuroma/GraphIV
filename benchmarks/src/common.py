"""
Paths, jsonl files, the frozen citation graph, and the call into GraphIV's own environment

Every path comes from config.toml, so moving the data is one line there
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

HERE = Path(__file__).resolve().parent.parent
PATHS = tomllib.loads((HERE / "config.toml").read_text())["paths"]
DATA = HERE / PATHS["DATA"]
ROOT = (HERE / PATHS["ROOT"]).resolve()
TABLES = HERE / PATHS["TABLES"]
SOURCES = DATA / "sources"
CASES = DATA / "cases"
INDEX = DATA / "index_cache"
FITS = DATA / "fit_cache"
RUNS = DATA / "runs"
LOGS = DATA / "logs"
# the graph every method reads, copied from the store's export so a later ingest cannot move it
GRAPH = SOURCES / "graph_export.npy"
# the process stops rather than let the shared box swap
FLOOR_GB = 10

# where graphiv() copies its child's output, set by run.py per experiment
LOG: Path | None = None
CHILDREN: list[subprocess.Popen] = []


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
    Prints a line and appends it to the current experiment's log

    Args:
        text: The line.
    """
    print(text, flush=True)
    if LOG is not None:
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


def graphiv(script: str, *args: str, path: str = "") -> None:
    """
    Runs one script of src/graphiv/ in GraphIV's own environment, where the store and src are

    Args:
        script: The file name under src/graphiv/.
        args: Its arguments.
        path: Another directory of the checkout to put on PYTHONPATH, relative to it.

    Raises:
        RuntimeError: When the script exits nonzero.
    """
    pythonpath = f"{ROOT}:{ROOT / path}" if path else str(ROOT)
    # the benchmarks venv would otherwise leak into the checkout's uv
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    command = ["direnv", "exec", str(ROOT), "env", f"PYTHONPATH={pythonpath}", "uv", "run", "python", str(HERE / "src" / "graphiv" / script), *args]
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    CHILDREN.append(child)
    assert child.stdout is not None
    for line in child.stdout:
        if "direnv:" not in line[:12]:
            say(line.rstrip())
    code = child.wait()
    CHILDREN.remove(child)
    if code:
        raise RuntimeError(f"src/graphiv/{script} exited {code}")


def corpus() -> Path:
    """
    The corpus every method searches, exported from the store on first use

    Returns:
        index_cache/, holding corpus.parquet and nomic.npy in one row order.
    """
    if not (INDEX / "corpus.parquet").exists() or not (INDEX / "nomic.npy").exists():
        INDEX.mkdir(parents=True, exist_ok=True)
        graphiv("export.py", "--out", str(INDEX))
    return INDEX


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
