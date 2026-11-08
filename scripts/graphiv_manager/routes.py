"""
The http surface: one read of the state, the dashboard numbers, and the actions

Every action queues a job rather than running anything in the request, so a closed tab stops nothing
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import snapshot
import stages
import stats
from jobs import Step
from schedule import Manager

HERE = Path(__file__).resolve().parent / "webui"


def when(raw: object) -> float:
    """
    A start time from the page, which sends a local datetime or nothing

    Args:
        raw: The value, such as 2026-09-28T02:00.

    Returns:
        Epoch seconds, zero for now.
    """
    if not raw:
        return 0.0
    return datetime.fromisoformat(str(raw)).timestamp()


def chosen(body: dict[str, Any]) -> tuple[list[str], str]:
    """
    Which stages a request names: one stage, one table, or everything a delta runs

    Args:
        body: The request, carrying stage, table or all.

    Returns:
        The stage keys and a title for the job.
    """
    if body.get("all"):
        return stages.everything(), "Run every stage"
    table = str(body.get("table", ""))
    if table in stages.TABLES:
        return stages.of_table(table), f"Run {stages.TABLES[table].lower()}"
    key = str(body.get("stage", ""))
    if key in stages.BY_KEY:
        return [key], stages.BY_KEY[key].label
    return [], ""


def routes(manager: Manager) -> list[Route | Mount]:
    """
    Every route, bound to one manager

    Args:
        manager: The manager.

    Returns:
        The routes.
    """

    async def state(request: Request) -> Response:
        _ = request
        return JSONResponse(manager.view())

    async def numbers(request: Request) -> Response:
        _ = request
        return JSONResponse(stats.read(manager.state / "stats"))

    async def recount(request: Request) -> Response:
        only = str(request.query_params.get("table", ""))
        asyncio.get_running_loop().create_task(manager.refresh_stats(only))
        return JSONResponse({"started": not manager.summing})

    async def refresh(request: Request) -> Response:
        _ = request
        asyncio.get_running_loop().create_task(manager.refresh_counts())
        await manager.refresh_health()
        return JSONResponse({"ok": True})

    async def rescan(request: Request) -> Response:
        _ = request
        await manager.rediscover()
        return JSONResponse({"scripts": len(manager.found)})

    def fetching() -> bool:
        active = ("queued", "scheduled", "waiting", "running", "stopping")
        return any(
            j.status in active and any(s.kind == "fetch" for s in j.steps)
            for j in manager.store.jobs.values()
        )

    async def fetch(request: Request) -> Response:
        _ = request
        if fetching():
            return JSONResponse({"error": "a fetch is already queued or running"}, status_code=400)
        return JSONResponse(manager.shown(manager.run_fetch()))

    async def commit(request: Request) -> Response:
        _ = request
        if not snapshot.staged(manager.source).exists():
            return JSONResponse({"error": "no snapshot is staged"}, status_code=400)
        job = manager.add("Put the staged snapshot in place", [Step(label="Commit", kind="commit")])
        return JSONResponse(manager.shown(job))

    async def discard(request: Request) -> Response:
        _ = request
        return JSONResponse({"removed": snapshot.discard(manager.source)})

    async def check(request: Request) -> Response:
        _ = request
        return JSONResponse(manager.shown(manager.run_check()))

    async def run(request: Request) -> Response:
        body: dict[str, Any] = await request.json()
        keys, title = chosen(body)
        if not keys:
            return JSONResponse({"error": "name a stage, a table or all"}, status_code=400)
        try:
            start_at = when(body.get("start_at"))
        except ValueError:
            return JSONResponse({"error": "that start time does not parse"}, status_code=400)
        after = str(body.get("after", ""))
        if after and after not in manager.store.jobs:
            return JSONResponse({"error": "the job to wait for does not exist"}, status_code=400)
        if keys == ["papers.fetch"]:
            if fetching():
                return JSONResponse(
                    {"error": "a fetch is already queued or running"}, status_code=400
                )
            return JSONResponse(manager.shown(manager.run_fetch()))
        extra = [str(a) for a in body.get("args", [])]
        job = manager.run_stages(keys, title, extra, start_at, after)
        return JSONResponse(manager.shown(job))

    async def plan(request: Request) -> Response:
        body: dict[str, Any] = await request.json()
        keys, _ = chosen(body)
        if not keys:
            return JSONResponse({"error": "name a stage, a table or all"}, status_code=400)
        return JSONResponse({"text": manager.plan(keys)})

    async def script(request: Request) -> Response:
        body: dict[str, Any] = await request.json()
        path = str(body.get("script", ""))
        known = {s.path: s for s in manager.found}
        if path not in known:
            return JSONResponse({"error": f"{path} is not a script here"}, status_code=400)
        args = [str(a) for a in body.get("args", [])]
        steps = [Step(label=known[path].title, script=path, args=args)]
        return JSONResponse(manager.shown(manager.add(known[path].name, steps)))

    async def cancel(request: Request) -> Response:
        said = await manager.cancel(str(request.path_params["job"]))
        return JSONResponse({"said": said})

    async def resume(request: Request) -> Response:
        return JSONResponse({"said": manager.resume(str(request.path_params["job"]))})

    async def log(request: Request) -> Response:
        job = manager.store.jobs.get(str(request.path_params["job"]))
        if job is None:
            return JSONResponse({"error": "no such job"}, status_code=404)
        start = int(request.query_params.get("from", "0") or 0)
        text, size = manager.store.tail(job, start)
        return JSONResponse({"text": text, "offset": size})

    async def page(request: Request) -> Response:
        _ = request
        return FileResponse(HERE / "index.html")

    return [
        Route("/", page),
        Route("/api/state", state),
        Route("/api/stats", numbers),
        Route("/api/stats/refresh", recount, methods=["POST"]),
        Route("/api/refresh", refresh, methods=["POST"]),
        Route("/api/rescan", rescan, methods=["POST"]),
        Route("/api/fetch", fetch, methods=["POST"]),
        Route("/api/snapshot/commit", commit, methods=["POST"]),
        Route("/api/snapshot/discard", discard, methods=["POST"]),
        Route("/api/check", check, methods=["POST"]),
        Route("/api/run", run, methods=["POST"]),
        Route("/api/plan", plan, methods=["POST"]),
        Route("/api/script", script, methods=["POST"]),
        Route("/api/jobs/{job}/cancel", cancel, methods=["POST"]),
        Route("/api/jobs/{job}/resume", resume, methods=["POST"]),
        Route("/api/jobs/{job}/log", log),
        Mount("/", StaticFiles(directory=HERE)),
    ]
