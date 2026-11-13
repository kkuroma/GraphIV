"""
A job is an ordered list of steps run one process group at a time, kept on disk as json

A step that exits nonzero halts the job, since every stage reads a queue the one before it fills
Each step owns its process group, so a cancel reaches the workers and the aria2c it spawned
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import OVERRIDE_ENV

# a stage finishing its current batch needs longer than a network timeout, and less than a COPY
GRACE_INT = 60.0
GRACE_TERM = 30.0
TAIL_BYTES = 16384
PROGRESS = re.compile(r"\] +([\d,]+)/([\d,]+)")
PROJECTED = re.compile(r"projected ([\d,]+)")
INSERTED = re.compile(r"inserted ([\d,]+) rows, reset ([\d,]+) papers")
FOUND = re.compile(r"\] ([\d,]+) new rows")

ACTIVE = ("queued", "scheduled", "waiting", "running", "stopping")
FINISHED = ("done", "nothing new", "failed", "blocked", "cancelled", "interrupted")


@dataclass
class Step:
    """
    One script and its arguments, or one of the two steps the manager runs itself: fetch and commit
    """

    label: str
    script: str = ""
    args: list[str] = field(default_factory=list)
    kind: str = "script"
    # what a resumed step runs instead, so a resumed relabel does not reset the rows it already did
    resume_args: list[str] | None = None
    gate: bool = False
    # a config other than the manager's, which is how a dry run reads the staged snapshot
    config: str = ""
    # the stage this step runs, so the page finds a stage's last log
    stage: str = ""
    status: str = "pending"
    code: int | None = None
    started: float = 0.0
    ended: float = 0.0
    note: str = ""


@dataclass
class Job:
    """
    What was asked for, when it may start, and how far it got
    """

    id: str
    title: str
    steps: list[Step]
    created: float = field(default_factory=time.time)
    start_at: float = 0.0
    after: str = ""
    status: str = "queued"
    reason: str = ""
    current: int = 0
    pid: int = 0
    resumed: bool = False
    before: dict[str, Any] = field(default_factory=dict)
    after_counts: dict[str, Any] = field(default_factory=dict)
    ended: float = 0.0

    def to_json(self) -> dict[str, Any]:
        """
        The job as the page reads it

        Returns:
            A json-ready dict.
        """
        return asdict(self)

    @staticmethod
    def from_json(raw: dict[str, Any]) -> Job:
        """
        A job read back off disk

        Args:
            raw: What to_json wrote.

        Returns:
            The job.
        """
        steps = [Step(**s) for s in raw.pop("steps", [])]
        return Job(steps=steps, **raw)


class Store:
    """
    One directory per job holding job.json and the log every step appends to
    """

    def __init__(self, root: Path) -> None:
        """
        Opens the store, marking any job a dead manager left running as interrupted

        Args:
            root: The jobs directory.
        """
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        for found in sorted(self.root.glob("*/job.json")):
            try:
                job = Job.from_json(json.loads(found.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
            if job.status in ("running", "stopping") and not alive(job.pid):
                job.status = "interrupted"
                job.reason = "the manager stopped while this was running"
                job.steps[job.current].status = "interrupted"
                self.save(job)
            self.jobs[job.id] = job

    def new(self, title: str, steps: list[Step], start_at: float = 0.0, after: str = "") -> Job:
        """
        Adds a job to the queue

        Args:
            title: What the page calls it.
            steps: What it runs, in order.
            start_at: The earliest epoch second it may start, zero for now.
            after: A job id that has to finish first.

        Returns:
            The job.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        job = Job(id=f"{stamp}-{secrets.token_hex(2)}", title=title, steps=steps)
        job.start_at = start_at
        job.after = after
        job.status = "scheduled" if start_at > time.time() else "queued"
        self.jobs[job.id] = job
        self.save(job)
        return job

    def log(self, job: Job) -> Path:
        """
        The file every step of one job writes to

        Args:
            job: The job.

        Returns:
            The path.
        """
        return self.root / job.id / "log"

    def save(self, job: Job) -> None:
        """
        Writes one job, replacing the file whole so a crash never leaves half of it

        Args:
            job: The job.
        """
        where = self.root / job.id
        where.mkdir(parents=True, exist_ok=True)
        part = where / "job.json.part"
        _ = part.write_text(json.dumps(job.to_json(), indent=2), encoding="utf-8")
        os.replace(part, where / "job.json")

    def tail(self, job: Job, start: int = 0) -> tuple[str, int]:
        """
        What the log gained since an offset, capped so a first read is not the whole file

        Args:
            job: The job.
            start: The byte offset the reader already has.

        Returns:
            The text and the new offset.
        """
        path = self.log(job)
        if not path.exists():
            return "", 0
        size = path.stat().st_size
        start = max(start, size - TAIL_BYTES) if start == 0 else min(start, size)
        with path.open("rb") as fh:
            _ = fh.seek(start)
            return fh.read().decode("utf-8", errors="replace"), size


def alive(pid: int) -> bool:
    """
    Whether a process still exists

    Args:
        pid: The process id, zero meaning none.

    Returns:
        True when it does.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def progress(text: str) -> tuple[int, int] | None:
    """
    The last done/total pair a stage printed

    Args:
        text: The log tail.

    Returns:
        Done and total, zero total when only a count is printed, None when there is no counter.
    """
    found = PROGRESS.findall(text)
    if not found:
        alone = PROJECTED.findall(text)
        return (int(alone[-1].replace(",", "")), 0) if alone else None
    done, total = found[-1]
    return int(done.replace(",", "")), int(total.replace(",", ""))


def nothing_new(text: str) -> bool:
    """
    Whether the snapshot stage found nothing to insert, which ends a delta right there

    Args:
        text: The log of that one step.

    Returns:
        True when it inserted and reset nothing, or its dry run found no new rows.
    """
    done = INSERTED.findall(text)
    if done:
        rows, reset = done[-1]
        return int(rows.replace(",", "")) + int(reset.replace(",", "")) == 0
    found = FOUND.findall(text)
    return bool(found) and int(found[-1].replace(",", "")) == 0


def verdict(text: str) -> tuple[str, str]:
    """
    Tells a crash from a stage refusing to run, since a refusal names its own cause

    Args:
        text: The log of the step that exited nonzero.

    Returns:
        failed or blocked, and the line that says why.
    """
    lines = [one.strip() for one in text.splitlines() if one.strip()]
    last = lines[-1] if lines else "exited with no output"
    crashed = any(one.startswith("Traceback") for one in lines[-60:])
    return ("failed" if crashed else "blocked"), last[:300]


class Runner:
    """
    Runs one job's steps in order, and stops one by escalating from SIGINT to SIGKILL
    """

    def __init__(
        self,
        store: Store,
        root: Path,
        python: str,
        config: Path,
        commit: Callable[[Path], None] | None = None,
        fetch: Callable[[Path, threading.Event], object] | None = None,
        finished: Callable[[Step, str], None] | None = None,
        grace: tuple[float, float] = (GRACE_INT, GRACE_TERM),
    ) -> None:
        """
        Holds what every step is launched with

        Args:
            store: Where jobs live.
            root: The directory scripts are named relative to, which is also the working directory.
            python: The interpreter every script runs under.
            config: The config.toml every script is handed.
            commit: The snapshot commit, called with the log path.
            fetch: The snapshot download, called with the log path and the stop flag.
            finished: Called with each step and what it printed once it exits zero.
            grace: Seconds to wait after SIGINT, then after SIGTERM.
        """
        self.store = store
        self.root = root
        self.python = python
        self.config = config
        self.commit = commit
        self.fetch = fetch
        self.halt = threading.Event()
        self.finished = finished
        self.grace = grace
        self.proc: asyncio.subprocess.Process | None = None
        self.job: Job | None = None
        self.stopping = False
        self.halted = "cancelled"

    async def run(self, job: Job) -> None:
        """
        Runs a job from its current step to the end, or to the first step that does not succeed

        Args:
            job: The job, updated and saved as it goes.
        """
        self.job, self.stopping, self.halted = job, False, "cancelled"
        self.halt.clear()
        job.status, job.reason = "running", ""
        self.store.save(job)
        log = self.store.log(job)
        log.parent.mkdir(parents=True, exist_ok=True)
        while job.current < len(job.steps):
            step = job.steps[job.current]
            step.status, step.started, step.code, step.note = "running", time.time(), None, ""
            self.store.save(job)
            start = log.stat().st_size if log.exists() else 0
            with log.open("ab") as fh:
                _ = fh.write(f"\n=== {step.label} ===\n".encode())
            code = await self.step(job, step, log)
            with log.open("rb") as fh:
                _ = fh.seek(start)
                said = fh.read().decode("utf-8", errors="replace")
            step.code, step.ended, job.pid = code, time.time(), 0
            if self.stopping:
                step.status, job.status = self.halted, self.halted
                job.reason = (
                    "stopped from the page"
                    if self.halted == "cancelled"
                    else "the manager was stopped while this ran"
                )
                break
            if code != 0:
                job.status, step.note = verdict(said)
                step.status, job.reason = job.status, step.note
                break
            step.status = "done"
            if self.finished is not None:
                self.finished(step, said)
            if step.gate and nothing_new(said):
                job.status = "nothing new"
                job.reason = "the snapshot holds no paper the store is missing"
                for rest in job.steps[job.current + 1 :]:
                    rest.status = "skipped"
                job.current = len(job.steps)
                break
            job.current += 1
            self.store.save(job)
        else:
            job.status = "done"
        job.ended = time.time()
        self.store.save(job)
        self.job, self.proc = None, None

    async def step(self, job: Job, step: Step, log: Path) -> int:
        """
        Runs one step with its output appended to the job's log

        Args:
            job: The job it belongs to.
            step: The step.
            log: The log file.

        Returns:
            The exit code, -1 for a step that raised.
        """
        if step.kind in ("commit", "fetch"):
            try:
                if step.kind == "commit" and self.commit is not None:
                    await asyncio.to_thread(self.commit, log)
                elif step.kind == "fetch" and self.fetch is not None:
                    _ = await asyncio.to_thread(self.fetch, log, self.halt)
                else:
                    raise ValueError(f"this runner cannot {step.kind}")
            except (OSError, ValueError, InterruptedError) as err:
                with log.open("ab") as fh:
                    _ = fh.write(f"{err}\n".encode())
                return 1
            return 0
        args = step.resume_args if job.resumed and step.resume_args is not None else step.args
        env = dict(os.environ, PGAPPNAME=f"graphiv-manager:{job.id}", PYTHONUNBUFFERED="1")
        if step.config:
            # a step's own config is already merged, so no override may land on it again
            env[OVERRIDE_ENV] = os.devnull
        with log.open("ab") as fh:
            self.proc = await asyncio.create_subprocess_exec(
                self.python,
                str(self.root / step.script),
                *args,
                "--config_path",
                step.config or str(self.config),
                cwd=self.root.parent,
                stdout=fh,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
            job.pid = self.proc.pid
            self.store.save(job)
            code = await self.proc.wait()
        job.resumed = False
        return code

    async def stop(self, halted: str = "cancelled") -> None:
        """
        Stops the running step: SIGINT so it can roll back, then SIGTERM, then SIGKILL

        Args:
            halted: What the job is recorded as, cancelled from the page or interrupted by a stop.
        """
        self.halted = halted
        self.halt.set()
        proc = self.proc
        if proc is None or proc.returncode is not None:
            self.stopping = self.job is not None
            return
        self.stopping = True
        if self.job is not None:
            self.job.status = "stopping"
            self.store.save(self.job)
        for sent, wait in ((signal.SIGINT, self.grace[0]), (signal.SIGTERM, self.grace[1])):
            group(proc.pid, sent)
            try:
                _ = await asyncio.wait_for(proc.wait(), wait)
                return
            except TimeoutError:
                continue
        group(proc.pid, signal.SIGKILL)
        _ = await proc.wait()


def group(pid: int, sent: signal.Signals) -> None:
    """
    Signals a whole process group, which is every worker a stage spawned

    Args:
        pid: The group leader, whose pid is the group id.
        sent: The signal.
    """
    try:
        os.killpg(pid, sent)
    except ProcessLookupError:
        pass
