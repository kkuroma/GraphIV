"""
The arXiv metadata snapshot: the one in use, a new one fetched and staged beside it, and the commit

The configured path may be a symlink, so every file lives beside the one it resolves to
Records run in id order with the old slash ids after the numeric ones, so the newest sit between
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, BinaryIO

TAIL_BYTES = 2 * 1024**2
STAGED = "arxiv-new.json"
ARCHIVE = "arxiv-new.zip"
CHUNK = 1 << 22
# the page polls every few seconds, so progress written more often only grows the log
SAY_EVERY = 2.0
# Kaggle answers 404 to the default Python-urllib agent, on the download and the metadata alike
AGENT = {"User-Agent": "graphiv-manager"}
FETCH_TIMEOUT = 60.0
PROBE_TIMEOUT = 8.0
# the page asks every few seconds and a 5 GB file does not change under it
SEEN: dict[tuple[str, int, float], str] = {}


def line_at(fh: BinaryIO, offset: int) -> str:
    """
    The first whole record starting after a byte offset

    Args:
        fh: The snapshot, opened for binary reads.
        offset: Where to look from.

    Returns:
        That line, empty at the end of the file.
    """
    _ = fh.seek(offset)
    if offset:
        _ = fh.readline()
    return fh.readline().decode("utf-8", errors="replace")


def numeric(line: str) -> bool:
    """
    Whether a record carries a new-style numeric id, which is every paper since April 2007

    Args:
        line: One record.

    Returns:
        True for 0704.0001 and later, False for hep-th/9901001 and the end of the file.
    """
    try:
        return str(json.loads(line).get("id", ""))[:1].isdigit()
    except ValueError:
        return False


def newest(path: Path) -> str:
    """
    The latest first-submission date in a snapshot, read just before the old ids begin

    Args:
        path: The snapshot.

    Returns:
        The date as YYYY-MM-DD, empty when nothing parsed.
    """
    size = path.stat().st_size
    with path.open("rb") as fh:
        low, high = 0, size
        while high - low > TAIL_BYTES:
            mid = (low + high) // 2
            if numeric(line_at(fh, mid)):
                low = mid
            else:
                high = mid
        _ = fh.seek(max(0, high - TAIL_BYTES))
        lines = (
            fh.read(TAIL_BYTES + (high - low)).decode("utf-8", errors="replace").splitlines()[1:]
        )
    best: datetime | None = None
    for line in lines:
        try:
            versions = json.loads(line).get("versions") or []
            when = parsedate_to_datetime(versions[0]["created"])
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            continue
        if best is None or when > best:
            best = when
    return best.date().isoformat() if best else ""


def staged(configured: Path) -> Path:
    """
    Where an uploaded snapshot waits, beside the one in use

    Args:
        configured: SOURCE_PATH from the config.

    Returns:
        The staged path, which may not exist.
    """
    return configured.resolve().with_name(STAGED)


def describe(path: Path) -> dict[str, Any]:
    """
    One snapshot file as the page shows it

    Args:
        path: The file.

    Returns:
        Its size, modification time and newest paper.
    """
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime)
    if key not in SEEN:
        SEEN[key] = newest(path)
    return {"path": str(path), "size": stat.st_size, "modified": stat.st_mtime, "newest": SEEN[key]}


def info(configured: Path) -> dict[str, Any]:
    """
    The snapshot in use, the staged one, and a download that stopped before it was staged

    Args:
        configured: SOURCE_PATH from the config.

    Returns:
        Each file's description keyed current and staged, and partial with its bytes so far.
    """
    out: dict[str, Any] = {"path": str(configured)}
    for key, path in (("current", configured.resolve()), ("staged", staged(configured))):
        if path.exists():
            out[key] = describe(path)
    archive = staged(configured).with_name(ARCHIVE)
    part = archive.with_name(archive.name + ".part")
    tag = part.with_name(part.name + ".etag")
    if archive.exists():
        size = archive.stat().st_size
        out["partial"] = {"size": size, "total": size, "unzip": True}
    elif part.exists():
        said = tag.read_text(encoding="utf-8").split() if tag.exists() else []
        total = int(said[1]) if len(said) > 1 else 0
        out["partial"] = {"size": part.stat().st_size, "total": total, "unzip": False}
    return out


def fetch(configured: Path, url: str, log: Path, halt: threading.Event) -> dict[str, Any]:
    """
    Downloads the snapshot archive, resuming a partial one, and unzips its json to the staged path

    Kaggle answers the download url with a redirect to a signed storage url that serves the zip.
    A partial archive is resumed only while the storage etag still matches the one it started on.

    Args:
        configured: SOURCE_PATH from the config.
        url: SOURCE_URL from the config.
        log: Where progress goes, as done/total lines the page reads.
        halt: Set when the job is stopped, checked between chunks.

    Returns:
        The staged file's description plus the archive's etag and size.

    Raises:
        InterruptedError: When the job was stopped, leaving the partial archive to resume.
        ValueError: When the archive holds no json, or the json holds no arXiv record.
    """
    target = staged(configured)
    archive = target.with_name(ARCHIVE)
    part = archive.with_name(archive.name + ".part")
    tag = part.with_name(part.name + ".etag")
    if archive.exists():
        # an earlier attempt finished the download and was stopped while unzipping
        return unpack(configured, archive, log, halt)

    def say(text: str) -> None:
        with log.open("ab") as fh:
            _ = fh.write(f"{text}\n".encode())

    have = part.stat().st_size if part.exists() else 0
    said = tag.read_text(encoding="utf-8").split() if tag.exists() else []
    held = said[0] if said else ""
    request = urllib.request.Request(
        url, headers=AGENT | ({"Range": f"bytes={have}-"} if have else {})
    )
    clock = time.time()
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as reply:
        etag = str(reply.headers.get("ETag", "")).strip('"')
        resumed = reply.status == 206 and bool(held) and held == etag
        if have and not resumed:
            say("[fetch] the archive changed since the last attempt, starting over")
            have = 0
            if reply.status == 206:
                return fetch_fresh(configured, url, log, halt, part, tag)
        total = have + int(reply.headers.get("Content-Length", 0))
        # the total rides beside the etag so the page can show how far a stopped download got
        _ = tag.write_text(f"{etag} {total}", encoding="utf-8")
        say(
            f"[fetch] resuming at {have:,} of {total:,} bytes"
            if have
            else f"[fetch] {total:,} bytes"
        )
        shown = 0.0
        with part.open("ab" if have else "wb") as fh:
            while chunk := reply.read(CHUNK):
                if halt.is_set():
                    raise InterruptedError(f"stopped at {have:,} of {total:,} bytes, which resume")
                _ = fh.write(chunk)
                have += len(chunk)
                if time.time() - shown > SAY_EVERY:
                    say(f"[fetch] {have:,}/{total:,} bytes")
                    shown = time.time()
    say(f"[fetch] {have:,}/{total:,} bytes in {time.time() - clock:.0f}s")
    os.replace(part, archive)
    tag.unlink(missing_ok=True)
    return unpack(configured, archive, log, halt) | {"etag": etag, "archive": total}


def fetch_fresh(
    configured: Path, url: str, log: Path, halt: threading.Event, part: Path, tag: Path
) -> dict[str, Any]:
    """
    Throws away a partial archive whose source changed, then fetches from the start

    Args:
        configured: SOURCE_PATH from the config.
        url: SOURCE_URL from the config.
        log: Where progress goes.
        halt: Set when the job is stopped.
        part: The partial archive.
        tag: The etag it was started against.

    Returns:
        What fetch returns.
    """
    part.unlink(missing_ok=True)
    tag.unlink(missing_ok=True)
    return fetch(configured, url, log, halt)


def unpack(configured: Path, archive: Path, log: Path, halt: threading.Event) -> dict[str, Any]:
    """
    Unzips the archive's json to the staged path and deletes the archive

    Args:
        configured: SOURCE_PATH from the config.
        archive: The downloaded zip.
        log: Where progress goes.
        halt: Set when the job is stopped.

    Returns:
        The staged file's description.

    Raises:
        InterruptedError: When the job was stopped, keeping the archive for the next attempt.
        ValueError: When the archive holds no json, or the json holds no arXiv record.
    """
    target = staged(configured)
    out = target.with_name(target.name + ".part")
    try:
        opened = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as err:
        archive.unlink()
        raise ValueError(f"the download is not a whole zip, so it was deleted: {err}") from err
    with opened as z:
        member = next((m for m in z.infolist() if m.filename.endswith(".json")), None)
        if member is None:
            raise ValueError("the archive holds no json")
        total, done, shown = member.file_size, 0, 0.0
        with z.open(member) as src, out.open("wb") as dst:
            while chunk := src.read(CHUNK):
                if halt.is_set():
                    raise InterruptedError("stopped while unzipping, the archive is kept")
                _ = dst.write(chunk)
                done += len(chunk)
                if time.time() - shown > SAY_EVERY:
                    with log.open("ab") as fh:
                        _ = fh.write(f"[unzip] {done:,}/{total:,} bytes\n".encode())
                    shown = time.time()
    found = newest(out)
    if not found:
        out.unlink()
        raise ValueError("the unzipped file holds no arXiv record, so nothing was staged")
    os.replace(out, target)
    archive.unlink()
    said = describe(target)
    with log.open("ab") as fh:
        _ = fh.write(
            f"[unzip] {done:,}/{total:,} bytes, staged {target}, newest paper {found}\n".encode()
        )
    return said


def remote(url: str) -> dict[str, Any]:
    """
    Which version of the dataset Kaggle serves, read from its metadata api and never downloaded

    Args:
        url: SOURCE_URL, whose view twin names the version.

    Returns:
        The version, when it was published and its size unzipped, empty when unreachable.
    """
    view = url.replace("/datasets/download/", "/datasets/view/")
    try:
        with urllib.request.urlopen(
            urllib.request.Request(view, headers=AGENT), timeout=PROBE_TIMEOUT
        ) as reply:
            data = json.loads(reply.read())
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    versions = data.get("versions") or [{}]
    return {
        "version": data.get("currentVersionNumberNullable"),
        "published": data.get("lastUpdated"),
        "size": data.get("totalBytesNullable"),
        "note": versions[0].get("versionNotes", ""),
        "taken": time.time(),
    }


def commit(configured: Path, log: Path) -> None:
    """
    Replaces the snapshot in use with the staged one

    Args:
        configured: SOURCE_PATH from the config.
        log: Where to say what happened.

    Raises:
        ValueError: When nothing is staged.
    """
    source = staged(configured)
    if not source.exists():
        raise ValueError("no snapshot is staged")
    real = configured.resolve()
    found = newest(source)
    os.replace(source, real)
    with log.open("ab") as fh:
        _ = fh.write(f"[snapshot] {real} replaced, newest paper {found}\n".encode())


def discard(configured: Path) -> bool:
    """
    Deletes the staged snapshot and whatever a fetch left beside it

    Args:
        configured: SOURCE_PATH from the config.

    Returns:
        Whether there was anything to delete.
    """
    target = staged(configured)
    names = [STAGED, STAGED + ".part", ARCHIVE, ARCHIVE + ".part", ARCHIVE + ".part.etag"]
    found = [target.with_name(n) for n in names if target.with_name(n).exists()]
    for path in found:
        path.unlink()
    return bool(found)
