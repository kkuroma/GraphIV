"""
The one loop that decides what runs next, and every number the page reads

Jobs start strictly in the order they were queued, since a later stage depends on an earlier one
A job that cannot start yet waits and says why, rather than failing, and is checked again shortly
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import discover
import probes
import snapshot
import stages
import stats
from jobs import ACTIVE, Job, Runner, Step, Store, progress
from src.core.config import dumps, load_config

LOG = logging.getLogger(__name__)

TICK = 3.0
QUEUE_EVERY = 300.0
HEALTH_EVERY = 15.0
# Kaggle publishes weekly, so asking which version it serves twice an hour is plenty
REMOTE_EVERY = 1800.0
# the [export] keys holding the repos the publish stage uploads to
HUB_REPOS = ("DATA_REPO", "CITATION_REPO", "SURVEY_REPO", "BENCHMARK_REPO")
# a rate from a step shorter than this is mostly startup, not throughput
RATE_FLOOR = 60.0
PHASE = re.compile(r"^\[(\w+)\]", re.MULTILINE)
CHECKED = {
    "new": re.compile(r"\[stage0\] ([\d,]+) new rows"),
    "reset": re.compile(r"\[stage0\] ([\d,]+) papers moved to a new version"),
    "doomed": re.compile(r"\[stage0\] ([\d,]+) superseded files"),
    "edges": re.compile(r"\[stage0\] ([\d,]+) citations to drop"),
}


class Manager:
    """
    Everything the page reads and every action it can take
    """

    def __init__(self, scripts: Path, config: Path, state: Path, python: str) -> None:
        """
        Reads the config and opens the job store, creating the state directory on first start

        Args:
            scripts: The scripts directory.
            config: The config.toml every script is handed.
            state: The directory this manager owns.
            python: The interpreter scripts run under.
        """
        self.scripts = scripts
        self.config = config
        self.state = state
        self.python = python
        state.mkdir(parents=True, exist_ok=True)
        self.remote: dict[str, Any] = {}
        self.hub: dict[str, Any] = {}
        self.store = Store(state / "jobs")
        self.runner = Runner(
            self.store,
            scripts,
            python,
            config,
            commit=lambda log: snapshot.commit(self.source, log),
            fetch=self.fetched,
            finished=self.measured,
        )
        self.found: list[discover.Script] = []
        self.counts: dict[str, Any] = {}
        self.counting = False
        self.summing = False
        self.health: dict[str, Any] = {}
        self.checked = 0.0
        self.tasks: list[asyncio.Task[None]] = []

    # -- the config, read again on every use so a saved override applies at the next job or probe

    @property
    def cfg(self) -> dict[str, Any]:
        """
        The config with the override merged over it
        """
        return load_config(self.config)

    @property
    def dsn(self) -> str:
        """
        The store's connection string
        """
        return str(self.cfg["database"]["PG_DEST"])

    @property
    def source(self) -> Path:
        """
        The snapshot in use
        """
        return Path(str(self.cfg["source"]["SOURCE_PATH"]))

    @property
    def url(self) -> str:
        """
        Where the snapshot is fetched from
        """
        return str(self.cfg["source"].get("SOURCE_URL", ""))

    # -- files this manager owns

    def read_json(self, name: str) -> dict[str, Any]:
        """
        One of the small json files in the state directory

        Args:
            name: The file name.

        Returns:
            Its contents, empty when missing or unreadable.
        """
        try:
            return json.loads((self.state / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write_json(self, name: str, value: dict[str, Any]) -> None:
        """
        Replaces one of the small json files whole

        Args:
            name: The file name.
            value: What to write.
        """
        part = self.state / f"{name}.part"
        _ = part.write_text(json.dumps(value, indent=2), encoding="utf-8")
        _ = part.replace(self.state / name)

    def staged_config(self) -> Path:
        """
        The merged config pointing the snapshot stage at the staged file, for its dry run

        The runner hands a step with its own config no override, so the staged path stands

        Returns:
            The copy's path.
        """
        merged = self.cfg
        merged["source"]["SOURCE_PATH"] = str(snapshot.staged(self.source))
        path = self.state / "staged.toml"
        _ = path.write_text(dumps(merged), encoding="utf-8")
        return path

    def measured(self, step: Step, said: str) -> None:
        """
        Keeps the rows a second a finished step managed, which the forecast then uses

        Args:
            step: The step that exited zero.
            said: What it printed.
        """
        found = progress(said)
        seconds = time.time() - step.started
        if not step.stage or found is None or found[0] <= 0 or seconds < RATE_FLOOR:
            return
        rates = self.read_json("rates.json")
        rates[step.stage] = {"rate": found[0] / seconds, "rows": found[0], "seconds": seconds}
        self.write_json("rates.json", rates)

    def fetched(self, log: Path, halt: threading.Event) -> dict[str, Any]:
        """
        Downloads the snapshot to the staged path, then keeps which version it was

        Args:
            log: The job's log.
            halt: The runner's stop flag.

        Returns:
            The staged file's description.
        """
        said = snapshot.fetch(self.source, self.url, log, halt)
        self.write_json("source.json", {**self.remote, **said, "fetched": time.time()})
        return said

    def rates(self) -> dict[str, float]:
        """
        Rows a second per stage, measured here where it ran and from the recipe otherwise

        Returns:
            The rate per stage key.
        """
        held = {k: float(v["rate"]) for k, v in self.read_json("rates.json").items()}
        return {**stages.RATES, **held}

    def check(self, staged: bool = True) -> dict[str, Any]:
        """
        The last check of the staged file or of the one in use, kept while that file is unchanged

        Args:
            staged: Which of the two files.

        Returns:
            The counts it reported and the file it read, empty when stale.
        """
        said = self.read_json(f"check-{'staged' if staged else 'current'}.json")
        path = Path(str(said.get("path", "")))
        if not said or not path.exists() or path.stat().st_mtime != said.get("mtime"):
            return {}
        return said

    # -- what the page reads

    def view(self) -> dict[str, Any]:
        """
        Everything the page draws except the dashboard numbers, in one reply

        Returns:
            The health, the queues, the snapshot, the stages, the jobs and the scripts.
        """
        jobs = sorted(self.store.jobs.values(), key=lambda j: j.created, reverse=True)
        held = stats.read(self.state / "stats")
        shot = snapshot.info(self.source)
        checked = self.check()
        staged = "staged" in shot
        found = int(checked.get("new", 0)) + int(checked.get("reset", 0)) if staged else 0
        recipe = stages.recipe()
        for row in recipe:
            row["waiting"] = stages.waiting(stages.BY_KEY[row["key"]], self.counts, held, found)
            row["last"] = self.last(row["key"])
        return {
            "health": self.health,
            "queues": self.counts,
            "counting": self.counting,
            "summing": self.summing,
            "snapshot": shot,
            "check": checked,
            "remote": self.remote,
            "hub": self.hub,
            "source": self.read_json("source.json"),
            "checked": self.check(staged=False),
            "forecast": stages.forecast(self.counts, held, found, self.rates()) if staged else [],
            "stages": recipe,
            "tables": stages.TABLES,
            "rates": self.rates(),
            "jobs": [self.shown(j) for j in jobs[:80]],
            "scripts": [asdict(s) for s in self.found],
            "stats_taken": {k: v.get("taken") for k, v in held.items()},
            "now": time.time(),
        }

    def last(self, key: str) -> dict[str, Any] | None:
        """
        The most recent job step that ran one stage, which is what the stage's console follows

        Args:
            key: The stage.

        Returns:
            The job id, step index and status, None when it never ran here.
        """
        for job in sorted(self.store.jobs.values(), key=lambda j: j.created, reverse=True):
            for spot, step in enumerate(job.steps):
                if step.stage == key and step.status != "pending":
                    return {"job": job.id, "step": spot, "status": step.status, "label": step.label}
        return None

    def shown(self, job: Job) -> dict[str, Any]:
        """
        One job as the page reads it, with the progress of the step running now

        Args:
            job: The job.

        Returns:
            The job with its progress.
        """
        out = job.to_json()
        if job.status in ("running", "stopping"):
            said, _ = self.store.tail(job)
            section = said.split("=== ")[-1]
            found = progress(section)
            out["progress"] = list(found) if found else None
            tags = PHASE.findall(section)
            out["phase"] = tags[-1] if tags else ""
        return out

    async def refresh_health(self) -> None:
        """
        Asks the store, both endpoints, the GPU and the disks how they are
        """
        ep = self.cfg.get("endpoints", {})
        data = self.cfg.get("dataset", {})
        paths = {
            "State": self.state,
            "Snapshot": self.source.resolve().parent,
            "PDF store": Path(str(data.get("PDF_PATH", self.state))),
            "HTML store": Path(str(data.get("HTML_PATH", self.state))),
            "Export": Path(str(self.cfg.get("export", {}).get("EXPORT_PATH", self.state))),
        }
        store, llm, embed, card, disk = await asyncio.gather(
            asyncio.to_thread(probes.store, self.dsn),
            asyncio.to_thread(
                probes.models,
                str(ep.get("LLM_ENDPOINT", "")),
                str(ep.get("LLM_NAME", "")),
                str(ep.get("LLM_API_KEY", "")),
            ),
            asyncio.to_thread(
                probes.models,
                str(ep.get("EMBEDDING_ENDPOINT", "")),
                "",
                str(ep.get("EMBEDDING_API_KEY", "")),
            ),
            asyncio.to_thread(probes.gpu),
            asyncio.to_thread(probes.disks, paths),
        )
        mine = {self.runner.proc.pid} if self.runner.proc else set()
        self.health = {
            "store": {"ok": store[0], "note": store[1]},
            "llm": {"ok": llm[0], "note": llm[1]},
            "embed": {"ok": embed[0], "note": embed[1]},
            "gpu": card,
            "disk": disk,
            "foreign": await asyncio.to_thread(probes.running, mine),
            "taken": time.time(),
        }
        self.checked = time.time()

    async def refresh_remote(self) -> None:
        """
        Asks Kaggle which version it serves, keeping the last answer when it does not reply
        """
        said = await asyncio.to_thread(snapshot.remote, self.url)
        if said:
            self.remote = said

    async def refresh_hub(self) -> None:
        """
        Asks Hugging Face who the publish token belongs to and which repos it may push to
        """
        export = self.cfg.get("export", {})
        repos = [str(export[k]) for k in HUB_REPOS if export.get(k)]
        path = str(self.cfg.get("manager", {}).get("HF_TOKEN_PATH", ""))
        self.hub = await asyncio.to_thread(probes.hub, path, repos)

    async def refresh_counts(self) -> None:
        """
        Recounts every queue, which takes seconds and so never blocks the page
        """
        if self.counting:
            return
        self.counting = True
        try:
            self.counts = await asyncio.to_thread(probes.queues, self.dsn)
        except Exception as err:  # noqa: BLE001
            LOG.warning("counting the queues failed: %s", err)
            self.counts = {**self.counts, "error": str(err)[:200]}
        finally:
            self.counting = False

    async def refresh_stats(self, only: str = "") -> None:
        """
        Reruns the dashboard queries in the background, one table or all of them

        Args:
            only: A table name, empty for every table.
        """
        if self.summing:
            return
        self.summing = True
        try:
            await asyncio.to_thread(stats.refresh, self.dsn, self.state / "stats", only)
        except Exception as err:  # noqa: BLE001
            LOG.warning("the dashboard queries failed: %s", err)
        finally:
            self.summing = False

    async def rediscover(self) -> None:
        """
        Reads the scripts directory again, and the flags of any script edited since
        """
        self.found = await discover.discover(self.scripts, self.python, self.state / "help.json")

    # -- what the page does

    def add(self, title: str, steps: list[Step], start_at: float = 0.0, after: str = "") -> Job:
        """
        Queues a job

        Args:
            title: What the page calls it.
            steps: What it runs.
            start_at: The earliest epoch second it may start.
            after: A job that has to finish first.

        Returns:
            The job.
        """
        job = self.store.new(title, steps, start_at, after)
        LOG.info("queued %s, %s", job.id, title)
        return job

    def run_stages(
        self, keys: list[str], title: str, extra: list[str], start_at: float, after: str
    ) -> Job:
        """
        Queues some stages as one job, in recipe order

        Args:
            keys: The stages, run in recipe order.
            title: What the page calls the job.
            extra: Flags for a single stage.
            start_at: The earliest epoch second it may start.
            after: A job that has to finish first.

        Returns:
            The job.
        """
        staged = snapshot.staged(self.source).exists()
        steps = stages.steps(keys, staged, extra)
        return self.add(title, steps, start_at, after)

    def run_fetch(self) -> Job:
        """
        Queues the download of the snapshot Kaggle serves, followed by its check

        Returns:
            The job.
        """
        fetch = stages.step(stages.BY_KEY["papers.fetch"])
        check = Step(
            label="Compare the snapshot with the store",
            script=stages.BY_KEY["papers.0"].script,
            args=["--dry-run"],
            config=str(self.staged_config()),
            stage="papers.0",
        )
        version = self.remote.get("version")
        title = f"Fetch the snapshot, version {version}" if version else "Fetch the snapshot"
        return self.add(title, [fetch, check])

    def run_check(self) -> Job:
        """
        Queues the snapshot stage's own dry run, against the staged file when there is one

        Returns:
            The job.
        """
        staged = snapshot.staged(self.source).exists()
        step = Step(
            label="Compare the snapshot with the store",
            script=stages.BY_KEY["papers.0"].script,
            args=["--dry-run"],
            config=str(self.staged_config()) if staged else "",
            stage="papers.0",
        )
        title = "Check the staged snapshot" if staged else "Check the snapshot in use"
        return self.add(title, [step])

    def plan(self, keys: list[str]) -> str:
        """
        What running some stages would do, without running anything

        Args:
            keys: The stages.

        Returns:
            The console text.
        """
        checked = self.check()
        staged = snapshot.staged(self.source).exists()
        found = int(checked.get("new", 0)) + int(checked.get("reset", 0)) if staged else 0
        held = stats.read(self.state / "stats")
        return stages.plan(keys, staged, self.counts, held, found, self.remote)

    def record_check(self, job: Job) -> None:
        """
        Keeps what a finished snapshot check reported, tied to the file it read

        Args:
            job: A job whose one step ran the snapshot stage with --dry-run.
        """
        said, _ = self.store.tail(job)
        out: dict[str, Any] = {}
        for key, pattern in CHECKED.items():
            found = pattern.findall(said)
            if found:
                out[key] = int(found[-1].replace(",", ""))
        if "new" not in out:
            return
        dry = next((s for s in job.steps if "--dry-run" in s.args), None)
        staged = bool(dry and dry.config)
        path = snapshot.staged(self.source) if staged else self.source.resolve()
        if not path.exists():
            return
        out.update(
            path=str(path), mtime=path.stat().st_mtime, staged=staged, taken=time.time(), job=job.id
        )
        self.write_json(f"check-{'staged' if staged else 'current'}.json", out)

    async def cancel(self, job_id: str) -> str:
        """
        Stops a running job or takes a waiting one off the queue

        Args:
            job_id: The job.

        Returns:
            What happened.
        """
        job = self.store.jobs.get(job_id)
        if job is None:
            return "no such job"
        if job.status in ("running", "stopping") and self.runner.job is job:
            asyncio.get_running_loop().create_task(self.runner.stop())
            return "stopping"
        if job.status in ACTIVE:
            job.status, job.reason, job.ended = "cancelled", "taken off the queue", time.time()
            self.store.save(job)
            return "cancelled"
        return f"already {job.status}"

    def resume(self, job_id: str) -> str:
        """
        Queues a stopped job again from the step it stopped on

        Args:
            job_id: The job.

        Returns:
            What happened.
        """
        job = self.store.jobs.get(job_id)
        if job is None:
            return "no such job"
        if job.status not in ("failed", "blocked", "cancelled", "interrupted"):
            return f"a {job.status} job cannot be resumed"
        job.status, job.reason, job.resumed, job.start_at, job.after = "queued", "", True, 0.0, ""
        for step in job.steps[job.current :]:
            step.status, step.note = "pending", ""
        self.store.save(job)
        return "queued"

    # -- the loop

    def blocker(self, job: Job) -> str:
        """
        Why a job cannot start right now, if anything

        Args:
            job: The job at the front of the queue.

        Returns:
            The reason, empty when it may start.
        """
        health = self.health
        if health.get("foreign"):
            one = health["foreign"][0]
            return f"{one['script']} is already running as pid {one['pid']}"
        named = {s.path: s for s in self.found}
        disk = health.get("disk", {})
        wanted: set[str] = set()
        touched: set[str] = set()
        for step in job.steps[job.current :]:
            script = named.get(step.script)
            wanted |= set(script.needs) if script else {"store"}
            stage = stages.BY_KEY.get(step.stage)
            if stage is not None:
                touched |= set(stage.disks)
            elif step.kind == "script":
                touched |= set(disk)
        names = {
            "store": "the database",
            "llm": "the language model",
            "embed": "the embedding model",
        }
        for key, label in names.items():
            if key in wanted and not health.get(key, {}).get("ok"):
                return f"waiting for {label}: {health.get(key, {}).get('note', 'not checked yet')}"
        if "gpu" in wanted and not health.get("gpu", {}).get("ok"):
            return "waiting for the GPU, which nvidia-smi cannot see"
        mine = {k: v for k, v in disk.items() if k in touched}
        for key, said in mine.items():
            if "note" in said:
                return f"cannot reach the {key}: {said['note']}"
        low = [k for k, v in mine.items() if not v.get("ok")]
        if low:
            return f"waiting for disk space on {', '.join(low)}"
        return ""

    def next_job(self) -> Job | None:
        """
        The job at the front of the queue, marking what it waits on

        Returns:
            The job to start, or None.
        """
        now = time.time()
        waiting = [
            j for j in self.store.jobs.values() if j.status in ("queued", "scheduled", "waiting")
        ]
        for job in sorted(waiting, key=lambda j: j.created):
            if job.start_at > now:
                continue
            if job.after:
                first = self.store.jobs.get(job.after)
                if first is not None and first.status in ACTIVE:
                    self.mark(job, "waiting", f"waiting for {first.title} to finish")
                    continue
                if first is not None and first.status not in ("done", "nothing new"):
                    job.status, job.ended = "cancelled", now
                    job.reason = f"{first.title} ended as {first.status}"
                    self.store.save(job)
                    continue
            why = self.blocker(job)
            if why:
                self.mark(job, "waiting", why)
                return None
            return job
        return None

    def mark(self, job: Job, status: str, reason: str) -> None:
        """
        Records what a job waits on, writing only when it changed

        Args:
            job: The job.
            status: Its status.
            reason: Why.
        """
        if (job.status, job.reason) != (status, reason):
            job.status, job.reason = status, reason
            self.store.save(job)

    async def loop(self) -> None:
        """
        Starts the next job whenever nothing is running, forever
        """
        counted = 0.0
        asked = 0.0
        while True:
            if time.time() - asked > REMOTE_EVERY:
                asked = time.time()
                asyncio.get_running_loop().create_task(self.refresh_remote())
                asyncio.get_running_loop().create_task(self.refresh_hub())
            if time.time() - self.checked > HEALTH_EVERY:
                await self.refresh_health()
            if time.time() - counted > QUEUE_EVERY:
                counted = time.time()
                asyncio.get_running_loop().create_task(self.refresh_counts())
            if self.runner.job is None:
                job = self.next_job()
                if job is not None:
                    job.before = dict(self.counts)
                    await self.runner.run(job)
                    if any("--dry-run" in s.args for s in job.steps):
                        self.record_check(job)
                    await self.refresh_counts()
                    job.after_counts = dict(self.counts)
                    self.store.save(job)
                    counted = time.time()
                    if job.status == "done" and not any("--dry-run" in s.args for s in job.steps):
                        asyncio.get_running_loop().create_task(self.refresh_stats())
            await asyncio.sleep(TICK)

    async def start(self) -> None:
        """
        Reads the scripts, checks health, counts the queues, fills a missing dashboard and loops
        """
        await self.refresh_health()
        loop = asyncio.get_running_loop()
        self.tasks = [
            loop.create_task(self.rediscover()),
            loop.create_task(self.loop()),
        ]
        if not any((self.state / "stats").glob("*.json")):
            self.tasks.append(loop.create_task(self.refresh_stats()))

    async def stop(self) -> None:
        """
        Stops the running job so it can roll back, then the loop, which is what a unit stop does
        """
        if self.runner.job is not None:
            await self.runner.stop("interrupted")
            # the runner records the step once its process is reaped, which is a loop turn later
            for _ in range(100):
                if self.runner.job is None:
                    break
                await asyncio.sleep(0.05)
        for task in self.tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
