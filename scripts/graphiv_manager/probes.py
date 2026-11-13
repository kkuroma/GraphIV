"""
What the page shows before anything runs: the store, both endpoints, the GPU, disk and the queues

The model endpoints are asked for their model list and never for a completion
Counting a queue scans millions of rows, so the counts are cached and refreshed in the background
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, LiteralString

import httpx
import psycopg
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from src.core.llm import headers

TIMEOUT = 4.0
# a download step stops waiting below this, since a full disk fails half way through a batch
DISK_FLOOR = 20 * 1024**3
# a task script, which leaves out serve.py and the daemon the nix package runs
TASK = re.compile(r"(?:^|/)scripts/([^/]+)/([^/]+\.py)$")

QUEUES: dict[str, LiteralString] = {
    "papers": "SELECT stage, count(*) FROM papers GROUP BY stage",
    "tree": "SELECT stage, count(*) FROM topic_tree GROUP BY stage",
    "unparsed": "SELECT count(*) FROM papers WHERE stage >= 3 AND is_citation_parsable IS NULL",
    "unlabeled": "SELECT count(*) FROM citations WHERE stage = 0 AND target_id IS NOT NULL",
    "names": "SELECT count(*) FROM topic_names",
    "newest": "SELECT max(date) FROM papers",
}


def writable(who: dict[str, Any], repo: str) -> bool:
    """
    Whether the token whoami describes may push to one repo

    A write token reaches every repo its user can, and a fine-grained one only what its scopes list

    Args:
        who: What whoami returned.
        repo: The repo, as owner/name.

    Returns:
        Whether a commit to it would be accepted.
    """
    access = who.get("auth", {}).get("accessToken", {})
    if access.get("role") == "write":
        return True
    owner = repo.split("/", 1)[0]
    for scope in access.get("fineGrained", {}).get("scoped", []):
        entity = scope.get("entity", {})
        whole = entity.get("type") in ("user", "org") and entity.get("name") == owner
        if (whole or entity.get("name") == repo) and "repo.write" in scope.get("permissions", []):
            return True
    return False


def hub(token_path: str, repos: list[str]) -> dict[str, Any]:
    """
    Who the publish token belongs to, and which repos it may push to, from whoami alone

    Args:
        token_path: [manager] HF_TOKEN_PATH, empty when there is none.
        repos: The repos [export] gives an id to.

    Returns:
        Whether there is a token, its user and role, each repo with whether it may push, and a note.
    """
    out: dict[str, Any] = {"token": False, "user": "", "role": "", "taken": time.time()}
    out["repos"] = [{"id": r, "write": False} for r in repos]
    if not token_path or not Path(token_path).is_file():
        return out | {"note": "no token, so set [manager] HF_TOKEN_PATH"}
    try:
        who = HfApi(token=Path(token_path).read_text(encoding="utf-8").strip()).whoami()
    except (HfHubHTTPError, httpx.HTTPError, OSError) as error:
        return out | {"token": True, "note": f"whoami failed: {str(error)[:200]}"}
    access = who.get("auth", {}).get("accessToken", {})
    return out | {
        "token": True,
        "user": str(who.get("name", "")),
        "role": str(access.get("role", "")),
        "repos": [{"id": r, "write": writable(who, r)} for r in repos],
        "note": "",
    }


def store(dsn: str) -> tuple[bool, str]:
    """
    Whether a new connection is accepted, which is what a postmaster refusing them fails

    Args:
        dsn: libpq connection string.

    Returns:
        Whether it answered, and what it said.
    """
    try:
        with psycopg.connect(dsn, connect_timeout=int(TIMEOUT)) as conn:
            _ = conn.execute("SELECT 1").fetchone()
    except psycopg.Error as err:
        return False, str(err).strip().splitlines()[0][:200]
    return True, "accepting connections"


def models(url: str, want: str = "", key: str = "") -> tuple[bool, str]:
    """
    Whether an OpenAI compatible endpoint lists the model this config names

    Args:
        url: The base url, with or without /v1.
        want: The model name, empty to accept any.
        key: The API key, empty for an endpoint that takes none.

    Returns:
        Whether it answered with the model, and what it said.
    """
    base = url.rstrip("/")
    listing = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    request = urllib.request.Request(listing, headers=headers(key))
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as reply:
            data = json.loads(reply.read())
    except (urllib.error.URLError, OSError, ValueError) as err:
        return False, f"no answer from {listing}: {err}"[:200]
    names = [str(m.get("id", "")) for m in data.get("data", [])]
    if want and want not in names:
        return False, f"{want} is not among the {len(names)} models it serves"
    paused = router(base.removesuffix("/v1"))
    if paused:
        return False, paused
    return True, want or f"{len(names)} models"


def router(base: str) -> str:
    """
    Whether a llama-router in front of the models has been switched off, which still lists them

    Args:
        base: The endpoint without /v1.

    Returns:
        What is wrong, empty when it is active or is not a llama-router.
    """
    try:
        with urllib.request.urlopen(f"{base}/router", timeout=TIMEOUT) as reply:
            data = json.loads(reply.read())
    except (urllib.error.URLError, OSError, ValueError):
        return ""
    status = data.get("status") if isinstance(data, dict) else None
    if status is None or status in ("active", "idle"):
        return ""
    return f"the router is {status}, so every completion is refused"


def gpu() -> dict[str, Any]:
    """
    The card's memory and load, as nvidia-smi reports them

    Returns:
        ok, used and total in MiB, and utilization in percent.
    """
    try:
        said = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as err:
        return {"ok": False, "note": str(err)[:200]}
    name, used, total, load = [x.strip() for x in said.splitlines()[0].split(",")]
    return {"ok": True, "name": name, "used": int(used), "total": int(total), "load": int(load)}


def disks(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """
    Free space under each path the stages write

    Args:
        paths: A label per path.

    Returns:
        Free and total bytes per label, and whether it clears the floor.
    """
    out: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        try:
            use = shutil.disk_usage(path.resolve())
        except OSError as err:
            out[label] = {"ok": False, "note": str(err)[:200]}
            continue
        out[label] = {"ok": use.free > DISK_FLOOR, "free": use.free, "total": use.total}
    return out


def running(mine: set[int]) -> list[dict[str, Any]]:
    """
    Any task script running outside this manager, such as a stage started by hand

    Args:
        mine: Process ids this manager started, which are not foreign.

    Returns:
        One pid and script per foreign process.
    """
    out: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in mine or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        words = [a.decode(errors="replace") for a in argv if a]
        if "--help" in words:
            continue
        for word in words:
            found = TASK.search(word)
            if found and found.group(1) != "graphiv_manager":
                out.append({"pid": int(entry.name), "script": f"{found.group(1)}/{found.group(2)}"})
                break
    return out


def queues(dsn: str) -> dict[str, Any]:
    """
    How much sits at every stage, which is what says how far a delta has drained

    Args:
        dsn: libpq connection string.

    Returns:
        The counts, and when they were taken.
    """
    clock = time.time()
    out: dict[str, Any] = {}
    with psycopg.connect(dsn, autocommit=True, connect_timeout=int(TIMEOUT)) as conn:
        _ = conn.execute("SET application_name = 'graphiv-manager:queues'")
        papers = dict(conn.execute(QUEUES["papers"]).fetchall())
        tree = dict(conn.execute(QUEUES["tree"]).fetchall())
        out["papers"] = {str(k): int(v) for k, v in papers.items()}
        out["tree"] = {str(k): int(v) for k, v in tree.items()}
        for key in ("unparsed", "unlabeled", "names", "newest"):
            row = conn.execute(QUEUES[key]).fetchone()
            value: Any = row[0] if row else None
            out[key] = value.isoformat() if hasattr(value, "isoformat") else value
    out["embedded"] = sum(v for k, v in out["papers"].items() if int(k) >= 3)
    out["placed"] = sum(out["tree"].values())
    out["taken"] = clock
    out["seconds"] = round(time.time() - clock, 1)
    return out
