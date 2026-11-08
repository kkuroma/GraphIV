"""
The server's own record of a run: every call with its arguments, reply and timing, and the reads

A run belongs to one MCP session and ends at save_report, so one session can hold several runs
This log, not the client, is what the webui draws the timeline from
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.mcp.papers import Record
from src.report.check import ARXIV

SLUG_WORDS = 8
# a session quiet this long has finished its run, which is how a run that never saved ends
IDLE_SECONDS = 180


class Evidence(BaseModel):
    """
    One quote a reply gave, from one paragraph, which is what a report's marker points at
    """

    arxiv_id: str
    pid: str
    section: str
    text: str
    quote: str
    claim: str = ""
    metric: str = ""
    value: str = ""
    dataset: str = ""


@dataclass
class Run:
    """
    Everything one run did, kept in memory until it is saved
    """

    session: str
    started: float = field(default_factory=time.time)
    last: float = field(default_factory=time.time)
    calls: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    papers: dict[str, Record] = field(default_factory=dict)
    served: set[str] = field(default_factory=set)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    lineages: list[dict[str, Any]] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=dict)
    folder: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, tool: str, args: dict[str, Any], reply: str, began: float) -> None:
        """
        Records one call

        Args:
            tool: The tool name.
            args: What the caller passed.
            reply: What it got back.
            began: When the call started, epoch seconds.
        """
        with self.lock:
            self.last = time.time()
            self.calls.append(
                {
                    "tool": tool,
                    "args": args,
                    "reply": reply,
                    "start": round(began - self.started, 2),
                    "seconds": round(time.time() - began, 2),
                }
            )
            self.served |= set(ARXIV.findall(reply))

    def keep(self, evidence: Evidence, key: str) -> str:
        """
        Records a quote under a short label that a report cites it by

        The label carries the first author's surname, so a writer citing it for another paper's
        claim can see the mismatch, and a short label is not corrupted in copying the way an id is.

        Args:
            evidence: The paragraph and its quote.
            key: The first author's surname.

        Returns:
            The label, like Karpukhin7, the same one when the same quote comes back.
        """
        with self.lock:
            same = (evidence.arxiv_id, evidence.pid, evidence.quote)
            for label, held in self.evidence.items():
                if (held.arxiv_id, held.pid, held.quote) == same:
                    return label
            label = f"{key}{len(self.evidence) + 1}"
            self.evidence[label] = evidence
            return label

    def cite(self, citer: str, cited: str, sentence: str, key: str) -> str:
        """
        Records a citing sentence as evidence of what the citing paper says about the cited one

        Args:
            citer: The citing paper.
            cited: The cited paper.
            sentence: The sentence, as the store holds it.
            key: The citing paper's first author surname.

        Returns:
            The sentence's label.
        """
        found = Evidence(
            arxiv_id=citer,
            pid=f"c-{cited}",
            section=f"citing {cited}",
            text=sentence,
            quote=sentence,
        )
        return self.keep(found, key)

    def read(self, record: Record) -> None:
        """
        Records a paper the reader went through

        Args:
            record: The paper.
        """
        with self.lock:
            self.papers[record.arxiv_id] = record

    def add_row(self, schema: dict[str, Any], row: dict[str, Any]) -> None:
        """
        Adds one method row to the table its schema defines

        Args:
            schema: The columns.
            row: The method and its cells.
        """
        key = json.dumps(schema, sort_keys=True)
        with self.lock:
            table = self.tables.setdefault(key, {"schema": schema, "rows": []})
            table["rows"] = [
                r
                for r in table["rows"]
                if (r["arxiv_id"], r["method"]) != (row["arxiv_id"], row["method"])
            ] + [row]


RUNS: dict[str, Run] = {}
GUARD = threading.Lock()


def current(session: str) -> Run:
    """
    The open run of one session, started on its first call or after the session went quiet

    Args:
        session: The MCP session id.

    Returns:
        The run.
    """
    with GUARD:
        held = RUNS.get(session)
        if held is None or time.time() - held.last > IDLE_SECONDS:
            RUNS[session] = Run(session=session)
        return RUNS[session]


def close(session: str) -> None:
    """
    Ends a session's run, so its next call starts a new one

    Args:
        session: The MCP session id.
    """
    with GUARD:
        RUNS.pop(session, None)


def slug(question: str, when: datetime) -> str:
    """
    A run directory name from its date and question

    Args:
        question: What the report answers.
        tldr: The one-sentence answer, which opens report.md.
        when: When it was saved.

    Returns:
        The name.
    """
    words = re.findall(r"[a-z0-9]+", question.lower())[:SLUG_WORDS]
    return f"{when:%Y-%m-%d-%H%M%S}-{'-'.join(words) or 'report'}"


def write(
    root: Path, run: Run, question: str, markdown: str, verdict: dict[str, Any], tldr: str = ""
) -> Path:
    """
    Saves a run as one directory the webui reads, the same directory on every save of the run

    Args:
        root: The runs directory.
        run: The run.
        question: What the report answers.
        markdown: The caller's report.
        verdict: The check's result.
        tldr: The one-sentence answer, which opens report.md.

    Returns:
        The directory.
    """
    when = datetime.now()
    if run.folder is None:
        run.folder = root / slug(question, when)
    out = run.folder
    out.mkdir(parents=True, exist_ok=True)
    opening = f"> **TL;DR** {tldr}\n\n" if tldr else ""
    (out / "report.md").write_text(opening + markdown, encoding="utf-8")
    payload = {
        "question": question,
        "tldr": tldr,
        "saved": when.isoformat(timespec="seconds"),
        "seconds": round(time.time() - run.started, 1),
        "check": verdict,
        "papers": {a: r.model_dump(mode="json") for a, r in run.papers.items()},
        "evidence": {m: e.model_dump() for m, e in run.evidence.items()},
        "tables": list(run.tables.values()),
        "lineages": run.lineages,
        "searches": run.searches,
        "graph": run.graph,
    }
    (out / "run.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    dump_calls(out, run)
    return out


def dump_calls(out: Path, run: Run) -> None:
    """
    Writes every call of a run, one json line each

    Args:
        out: The run directory.
        run: The run.
    """
    with open(out / "calls.jsonl", "w", encoding="utf-8") as f:
        for call in run.calls:
            f.write(json.dumps(call) + "\n")
