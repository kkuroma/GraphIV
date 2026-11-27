"""
The runner against fake stages: halting, refusing, the snapshot gate, cancelling and restarting
"""

import asyncio
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from discover import parse_help
from jobs import Runner, Step, Store, nothing_new, progress, verdict


def stage(root: Path, name: str, body: str) -> str:
    """
    Writes one fake stage that accepts --config_path like the real ones
    """
    path = root / "task" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "import argparse, sys, time, signal\nap = argparse.ArgumentParser()\n"
    head += "ap.add_argument('--config_path')\nap.parse_known_args()\n"
    path.write_text(head + textwrap.dedent(body))
    return f"task/{name}.py"


@pytest.fixture
def world(tmp_path: Path) -> tuple[Store, Runner, Path]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    store = Store(tmp_path / "state" / "jobs")
    runner = Runner(store, scripts, sys.executable, tmp_path / "config.toml", grace=(1.0, 1.0))
    return store, runner, scripts


def test_a_failing_step_halts_the_job(world: tuple[Store, Runner, Path]) -> None:
    store, runner, scripts = world
    ok = stage(scripts, "ok", "print('[stage1] 5/10 tried')\n")
    bad = stage(scripts, "bad", "raise RuntimeError('boom')\n")
    after = stage(scripts, "after", "print('never')\n")
    job = store.new("t", [Step("a", ok), Step("b", bad), Step("c", after)])
    asyncio.run(runner.run(job))
    assert job.status == "failed"
    assert [s.status for s in job.steps] == ["done", "failed", "pending"]
    assert "boom" in job.reason
    assert "never" not in store.log(job).read_text()


def test_a_refusal_is_blocked_not_failed(world: tuple[Store, Runner, Path]) -> None:
    store, runner, scripts = world
    refuse = stage(scripts, "refuse", "raise SystemExit('9 papers still at stage 2')\n")
    job = store.new("t", [Step("a", refuse)])
    asyncio.run(runner.run(job))
    assert job.status == "blocked"
    assert job.reason == "9 papers still at stage 2"


def test_an_empty_snapshot_ends_the_delta(world: tuple[Store, Runner, Path]) -> None:
    store, runner, scripts = world
    first = stage(scripts, "first", "print('[stage0] inserted 0 rows, reset 0 papers')\n")
    rest = stage(scripts, "rest", "print('ran')\n")
    job = store.new("t", [Step("a", first, gate=True), Step("b", rest)])
    asyncio.run(runner.run(job))
    assert job.status == "nothing new"
    assert job.steps[1].status == "skipped"


def test_cancel_escalates_past_an_ignored_sigint(world: tuple[Store, Runner, Path]) -> None:
    store, runner, scripts = world
    stubborn = stage(
        scripts,
        "stubborn",
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\nprint('up', flush=True)\ntime.sleep(60)\n",
    )
    job = store.new("t", [Step("a", stubborn)])

    async def go() -> float:
        running = asyncio.create_task(runner.run(job))
        while runner.proc is None:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)
        clock = time.time()
        await runner.stop()
        await running
        return time.time() - clock

    took = asyncio.run(go())
    assert job.status == "cancelled"
    assert 0.9 < took < 5


def test_a_step_runs_on_the_first_threads_cpus(
    world: tuple[Store, Runner, Path], tmp_path: Path
) -> None:
    store, runner, scripts = world
    _ = (tmp_path / "config.toml").write_text("[manager]\nTHREADS = 1\n")
    count = stage(scripts, "count", "import os\nprint('cpus', len(os.sched_getaffinity(0)))\n")
    job = store.new("t", [Step("a", count)])
    asyncio.run(runner.run(job))
    assert "cpus 1" in store.log(job).read_text()
    assert len(os.sched_getaffinity(0)) == len(runner.cpus)


def test_a_job_left_running_reads_back_interrupted(tmp_path: Path) -> None:
    store = Store(tmp_path / "jobs")
    job = store.new("t", [Step("a", "task/x.py")])
    job.status, job.pid = "running", 999999999
    store.save(job)
    again = Store(tmp_path / "jobs")
    assert again.jobs[job.id].status == "interrupted"


def test_a_resumed_relabel_drops_the_reset(world: tuple[Store, Runner, Path]) -> None:
    store, runner, scripts = world
    echo = stage(scripts, "echo", "print('args', sys.argv[1:])\n")
    job = store.new("t", [Step("a", echo, args=["--relabel"], resume_args=[])])
    job.resumed = True
    asyncio.run(runner.run(job))
    said = store.log(job).read_text()
    assert "--relabel" not in said


def test_the_parsers() -> None:
    assert progress("[label] 1,000/73,808,475  (1,762/s)") == (1000, 73808475)
    assert progress("nothing") is None
    assert nothing_new("[stage0] 0 new rows\n[stage0] dry run, nothing written")
    assert not nothing_new("[stage0] 12 new rows")
    assert verdict("Traceback (most recent call last):\nValueError: x")[0] == "failed"
    flags = parse_help(
        "options:\n  -h, --help            show\n  --config_path CONFIG_PATH\n"
        "  --workers WORKERS\n  --relabel             redo the whole table\n"
    )
    assert [(f.flag, f.value) for f in flags] == [("--workers", "WORKERS"), ("--relabel", "")]
