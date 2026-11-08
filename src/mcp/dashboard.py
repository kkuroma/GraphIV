"""
The read-only routes the dashboard runs on, which are the run directory and the static page

A run's facts come out of its own run.json, so listing the library reads no report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from src.core.fetch import pdf_file

LOG = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parents[1] / "webui"
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
FILES = ("report.md", "run.json", "calls.jsonl")


def read_runs(root: Path) -> list[dict[str, Any]]:
    """
    Every run on disk, newest first

    Args:
        root: The directory runs are written to.

    Returns:
        One row per run, carrying nothing the library does not show.
    """
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for found in sorted(root.iterdir(), reverse=True):
        meta = found / "run.json"
        if not meta.is_file():
            continue
        try:
            parsed: dict[str, Any] = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        check = parsed.get("check") or {}
        out.append(
            {
                "slug": found.name,
                "question": str(parsed.get("question", found.name)),
                "tldr": str(parsed.get("tldr", "")),
                "saved": str(parsed.get("saved", "")),
                "seconds": parsed.get("seconds", 0),
                "passed": bool(check.get("passed")),
                "unverified": len({str(f.get("line", "")) for f in check.get("faults") or []}),
                "papers": len(parsed.get("papers") or {}),
                "tables": len(parsed.get("tables") or []),
                "lineages": sum(len(x.get("lines", [])) for x in parsed.get("lineages") or []),
            }
        )
    return out


def known(root: Path, slug: str) -> bool:
    """
    Whether one name is a run on disk, checked rather than joined so no path escapes the root

    Args:
        root: The directory runs are written to.
        slug: The name to check.

    Returns:
        True when a directory of that name is there.
    """
    return root.is_dir() and slug in {d.name for d in root.iterdir() if d.is_dir()}


def read_one(root: Path, slug: str) -> dict[str, Any] | None:
    """
    One run: its facts, the report, every call, and the client's transcript when one is attached

    Args:
        root: The directory runs are written to.
        slug: The run's directory name.

    Returns:
        The run, or None when nothing carries that name.
    """
    if not known(root, slug):
        return None
    where = root / slug
    try:
        parsed: dict[str, Any] = json.loads((where / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    parsed["slug"] = slug
    report = where / "report.md"
    parsed["report"] = report.read_text(encoding="utf-8") if report.is_file() else ""
    calls = where / "calls.jsonl"
    parsed["calls"] = (
        [json.loads(x) for x in calls.read_text(encoding="utf-8").splitlines() if x.strip()]
        if calls.is_file()
        else []
    )
    session = where / "session.json"
    parsed["session"] = (
        json.loads(session.read_text(encoding="utf-8")) if session.is_file() else None
    )
    return parsed


def mount(mcp: FastMCP, root: Path, pdfs: Path, fetch: bool = False) -> None:
    """
    Registers the dashboard on the http transport, which serves nothing that writes

    Args:
        mcp: The server.
        root: The directory runs are written to.
        pdfs: The pdf store, which the reading panel shows a paper out of.
        fetch: Whether a paper the store never had may be fetched from arXiv.
    """

    @mcp.custom_route("/api/runs", methods=["GET"])
    async def listing(request: Request) -> Response:
        """
        Every run, for the library

        Args:
            request: The http request, which carries nothing this reads.

        Returns:
            The rows as json.
        """
        _ = request
        return JSONResponse(read_runs(root))

    @mcp.custom_route("/api/runs/{slug}", methods=["GET"])
    async def one(request: Request) -> Response:
        """
        One run and its document

        Args:
            request: The http request, whose path carries the run's name.

        Returns:
            The run as json, or a 404.
        """
        found = read_one(root, str(request.path_params.get("slug", "")))
        if found is None:
            return JSONResponse({"error": "no such run"}, status_code=404)
        return JSONResponse(found)

    @mcp.custom_route("/", methods=["GET"])
    async def page(request: Request) -> Response:
        """
        The dashboard itself

        Args:
            request: The http request, which carries nothing this reads.

        Returns:
            The page.
        """
        _ = request
        return FileResponse(HERE / "index.html", media_type=TYPES[".html"])

    @mcp.custom_route("/style.css", methods=["GET"])
    async def styles(request: Request) -> Response:
        """
        The stylesheet

        Args:
            request: The http request, which carries nothing this reads.

        Returns:
            The file.
        """
        _ = request
        return FileResponse(HERE / "style.css", media_type=TYPES[".css"])

    @mcp.custom_route("/app.js", methods=["GET"])
    async def script(request: Request) -> Response:
        """
        The dashboard script

        Args:
            request: The http request, which carries nothing this reads.

        Returns:
            The file.
        """
        _ = request
        return FileResponse(HERE / "app.js", media_type=TYPES[".js"])

    @mcp.custom_route("/vendor/marked.min.js", methods=["GET"])
    async def markdown(request: Request) -> Response:
        """
        The markdown renderer the page draws a report with

        Args:
            request: The http request, which carries nothing this reads.

        Returns:
            The file.
        """
        _ = request
        return FileResponse(HERE / "vendor/marked.min.js", media_type=TYPES[".js"])

    @mcp.custom_route("/icon.svg", methods=["GET"])
    async def icon(request: Request) -> Response:
        """
        The wordmark, a G in a circle in the palette's own accent

        Args:
            request: The http request, which carries nothing this reads.

        Returns:
            The file.
        """
        _ = request
        return FileResponse(HERE / "icon.svg", media_type=TYPES[".svg"])

    @mcp.custom_route("/files/{slug}/{name}", methods=["GET"])
    async def download(request: Request) -> Response:
        """
        One of a run's own files, for the download menu

        Args:
            request: The http request, whose path carries the run and the file.

        Returns:
            The file, or a 404.
        """
        slug = str(request.path_params.get("slug", ""))
        name = str(request.path_params.get("name", ""))
        found = root / slug / name
        if not known(root, slug) or name not in FILES or not found.is_file():
            return JSONResponse({"error": "no such file"}, status_code=404)
        return FileResponse(found)

    @mcp.custom_route("/pdf/{arxiv_id:path}", methods=["GET", "HEAD"])
    async def paper(request: Request) -> Response:
        """
        One paper's pdf out of the local store, which the reading panel shows inline

        A paper the store never had is fetched when this instance is allowed to, and is
        otherwise a 404 that the page turns back into an arXiv link.

        Args:
            request: The http request, whose path carries the arXiv id.

        Returns:
            The pdf, or a 404.
        """
        found = pdf_file(pdfs, str(request.path_params.get("arxiv_id", "")), fetch)
        if found is None:
            return JSONResponse({"error": "not in the local store"}, status_code=404)
        return FileResponse(found, media_type="application/pdf")

    LOG.info("dashboard mounted over %s", root)
