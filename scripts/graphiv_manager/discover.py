"""
Every runnable file under scripts, read off the files themselves rather than a list kept here

A script's flags come from its own --help, so a new flag appears on the page with no edit here
What a script needs is read from its source: torch means the GPU, LLM_ENDPOINT the model, and so on
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import re
import signal
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# one-time training work, which is run by hand and never from a page
SKIP = {"citation_labeler", "graphiv_manager", "__pycache__"}
HIDDEN = {"-h", "--help", "--config_path"}
FLAG = re.compile(r"^\s{2}(--?[\w-]+)(?:,\s*(--[\w-]+))?(?: ([A-Z_]+))?(?:\s{2,}(.*))?$")
HELP_SECONDS = 180.0
# a --help imports torch in half of these, so a few at a time
HELP_PARALLEL = 4


@dataclass
class Flag:
    """
    One option a script takes, and whether it takes a value
    """

    flag: str
    value: str = ""
    help: str = ""


@dataclass
class Script:
    """
    One runnable file, what its docstring says it does, and what it needs to run
    """

    path: str
    task: str
    name: str
    title: str
    flags: list[Flag] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    mtime: float = 0.0


def runnable(path: Path) -> bool:
    """
    Whether a file is an entrypoint rather than a helper its siblings import

    Args:
        path: A python file.

    Returns:
        True when it carries a main guard.
    """
    return '__name__ == "__main__"' in path.read_text(encoding="utf-8", errors="replace")


def needs(source: str) -> list[str]:
    """
    What a script reaches outside itself, read from its source text

    Args:
        source: The script.

    Returns:
        A subset of store, gpu, llm and embed.
    """
    out = ["store"] if "PG_DEST" in source else []
    if re.search(r"^\s*(import|from) (torch|transformers)\b", source, re.MULTILINE):
        out.append("gpu")
    if "LLM_ENDPOINT" in source or "LABELER_APE_MODEL" in source:
        out.append("llm")
    if "EmbeddingEndpoint" in source:
        out.append("embed")
    return out


def parse_help(text: str) -> list[Flag]:
    """
    The options argparse printed, leaving out the ones the manager sets itself

    Args:
        text: What --help printed.

    Returns:
        One flag per option.
    """
    out: list[Flag] = []
    for line in text.splitlines():
        found = FLAG.match(line)
        if not found:
            if out and line.startswith(" " * 10) and line.strip():
                out[-1].help = f"{out[-1].help} {line.strip()}".strip()
            continue
        short, long, value, said = found.groups()
        name = long or short
        if name in HIDDEN or short in HIDDEN:
            continue
        out.append(Flag(flag=name, value=value or "", help=(said or "").strip()))
    return out


async def read_help(python: str, path: Path) -> list[Flag]:
    """
    Asks one script for its flags, which imports it and so costs what its imports cost

    Args:
        python: The interpreter.
        path: The script.

    Returns:
        Its flags, empty when it did not answer in time.
    """
    if "argparse" not in path.read_text(encoding="utf-8", errors="replace"):
        return []
    proc = await asyncio.create_subprocess_exec(
        python,
        str(path),
        "--help",
        cwd=path.parents[2],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        said, _ = await asyncio.wait_for(proc.communicate(), HELP_SECONDS)
    except TimeoutError:
        # a script that ignores --help would otherwise keep running, with whatever it spawned
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        return []
    return parse_help(said.decode("utf-8", errors="replace"))


async def discover(root: Path, python: str, cache: Path) -> list[Script]:
    """
    Every entrypoint under the scripts directory, with its flags read once per edit

    Args:
        root: The scripts directory.
        python: The interpreter.
        cache: A json file holding flags keyed by path and mtime.

    Returns:
        The scripts, grouped by task directory.
    """
    try:
        held: dict[str, Any] = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        held = {}
    found: list[Script] = []
    for path in sorted(root.glob("*/*.py")):
        if path.parent.name in SKIP or not runnable(path):
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        doc = ast.get_docstring(tree) or ""
        # a sibling module the script imports reaches the same things, as driver.py does
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                sibling = path.parent / f"{node.module}.py"
                if sibling.is_file():
                    source += sibling.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(root))
        found.append(
            Script(
                path=rel,
                task=path.parent.name,
                name=path.stem,
                title=doc.strip().splitlines()[0] if doc.strip() else path.stem,
                needs=needs(source),
                mtime=path.stat().st_mtime,
            )
        )
    stale = [s for s in found if held.get(s.path, {}).get("mtime") != s.mtime]
    gate = asyncio.Semaphore(HELP_PARALLEL)

    async def one(script: Script) -> list[Flag]:
        async with gate:
            return await read_help(python, root / script.path)

    answers = await asyncio.gather(*(one(s) for s in stale))
    for script, flags in zip(stale, answers, strict=True):
        held[script.path] = {"mtime": script.mtime, "flags": [asdict(f) for f in flags]}
    for script in found:
        script.flags = [Flag(**f) for f in held.get(script.path, {}).get("flags", [])]
    cache.parent.mkdir(parents=True, exist_ok=True)
    _ = cache.write_text(json.dumps(held, indent=2), encoding="utf-8")
    return found
