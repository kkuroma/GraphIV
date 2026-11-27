"""
The server's own record of a run: every call with its arguments, reply and timing, and the reads

A run opens on start_session, which issues its id and fixes its effort, and stays open after saving
Its state is written to state.json on every call, so a restart of the server loses nothing
"""

from __future__ import annotations

import json
import re
import secrets
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
STATE = "state.json"


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
    title: str = ""
    question: str = ""
    effort: str = "quick"
    tracked: bool = True
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
    draft: dict[str, Any] | None = None
    # papers a deep report must have read: the question's top hits and the table's newest citers
    opening: list[str] = field(default_factory=list)
    forward: list[str] = field(default_factory=list)
    # the papers a deep run's first poset argues with most that it had not read, frozen there
    argued: list[str] = field(default_factory=list)
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
            self.persist()

    def mark(self, label: str) -> str:
        """
        A label as a reply shows it, which is nothing on a call outside a session

        Args:
            label: The label.

        Returns:
            The label in brackets after a space, or empty.
        """
        return f" [{label}]" if self.tracked and label else ""

    def state(self) -> dict[str, Any]:
        """
        Everything a restarted server needs to carry the run on

        Returns:
            The run as json-ready data.
        """
        return {
            "session": self.session,
            "title": self.title,
            "question": self.question,
            "effort": self.effort,
            "started": self.started,
            "calls": self.calls,
            "evidence": {m: e.model_dump() for m, e in self.evidence.items()},
            "papers": {a: r.model_dump(mode="json") for a, r in self.papers.items()},
            "served": sorted(self.served),
            "tables": self.tables,
            "lineages": self.lineages,
            "searches": self.searches,
            "draft": self.draft,
            "opening": self.opening,
            "forward": self.forward,
            "argued": self.argued,
        }

    def persist(self) -> None:
        """
        Writes the run's state into its directory, when it has one
        """
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / STATE).write_text(json.dumps(self.state()), encoding="utf-8")

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

    def add_row(self, schema: dict[str, Any], row: dict[str, Any]) -> str:
        """
        Adds one paper's row to the table of its fields' names and types

        A schema revised between batches, with a choice gaining an option or an ask reworded,
        adds to the same table, whose choices carry every option either schema offered.

        Args:
            schema: The columns.
            row: The paper and its cells.

        Returns:
            The table's id, like T1.
        """
        key = shape(schema)
        with self.lock:
            table = self.tables.setdefault(
                key, {"schema": json.loads(json.dumps(schema)), "rows": []}
            )
            for held, given in zip(table["schema"]["columns"], schema["columns"], strict=True):
                held["options"] = list(dict.fromkeys(held["options"] + given["options"]))
            table["rows"] = [r for r in table["rows"] if r["arxiv_id"] != row["arxiv_id"]] + [row]
            return self.table_ids()[key]

    def table_for(self, schema: dict[str, Any]) -> str | None:
        """
        The id of the table a schema's rows go to, once it has any

        Args:
            schema: The columns.

        Returns:
            The id, or None.
        """
        return self.table_ids().get(shape(schema))

    def table_ids(self) -> dict[str, str]:
        """
        Each table's id, in the order the tables were made

        Returns:
            Ids like T1 by table key.
        """
        return {key: f"T{i + 1}" for i, key in enumerate(self.tables)}

    def table(self, tid: str) -> dict[str, Any] | None:
        """
        The table under an id

        Args:
            tid: Like T1.

        Returns:
            The table, or None.
        """
        found = [k for k, t in self.table_ids().items() if t == tid.strip().upper()]
        return self.tables[found[0]] if found else None


def shape(schema: dict[str, Any]) -> str:
    """
    A table's key, its fields' names and types, so a reworded ask still lands in the same table

    Args:
        schema: The columns.

    Returns:
        The key.
    """
    return json.dumps([(c["name"], c["kind"]) for c in schema["columns"]])


RUNS: dict[str, Run] = {}
GUARD = threading.Lock()


def start(root: Path | None, title: str, effort: str = "quick") -> Run:
    """
    Opens a run under a fresh id, in its own directory when the server may write

    Args:
        root: The runs directory, None on a read-only server.
        title: What the caller says the study is.
        effort: quick, standard or deep, which save_report holds a report to.

    Returns:
        The run.
    """
    with GUARD:
        session = f"s-{secrets.token_hex(4)}"
        while session in RUNS:
            session = f"s-{secrets.token_hex(4)}"
        run = Run(session=session, title=title.strip(), effort=effort)
        if root is not None:
            run.folder = root / slug(title, datetime.now())
        run.persist()
        RUNS[session] = run
        return run


def restore(folder: Path) -> Run:
    """
    A run as its state.json left it

    Args:
        folder: The run's directory.

    Returns:
        The run.
    """
    data = json.loads((folder / STATE).read_text(encoding="utf-8"))
    return Run(
        session=data["session"],
        title=data.get("title", ""),
        question=data.get("question", ""),
        effort=data.get("effort", "quick"),
        started=float(data["started"]),
        last=time.time(),
        calls=data.get("calls", []),
        evidence={m: Evidence.model_validate(e) for m, e in data.get("evidence", {}).items()},
        papers={a: Record.model_validate(r) for a, r in data.get("papers", {}).items()},
        served=set(data.get("served", [])),
        tables=data.get("tables", {}),
        lineages=data.get("lineages", []),
        searches=data.get("searches", []),
        draft=data.get("draft"),
        opening=data.get("opening", []),
        forward=data.get("forward", []),
        argued=data.get("argued", []),
        folder=folder,
    )


def find(root: Path | None, session: str) -> Run | None:
    """
    The run under an id, read back from disk when the server restarted since it opened

    Args:
        root: The runs directory, None on a read-only server.
        session: The id start_session issued.

    Returns:
        The run, or None when no run carries that id.
    """
    with GUARD:
        held = RUNS.get(session)
        if held is None and root is not None and root.is_dir():
            for state in root.glob(f"*/{STATE}"):
                if json.loads(state.read_text(encoding="utf-8")).get("session") == session:
                    held = restore(state.parent)
                    RUNS[session] = held
                    break
        return held


def slug(question: str, when: datetime) -> str:
    """
    A run directory name from its date and question

    Args:
        question: What the run studies.
        when: When it opened.

    Returns:
        The name.
    """
    words = re.findall(r"[a-z0-9]+", question.lower())[:SLUG_WORDS]
    return f"{when:%Y-%m-%d-%H%M%S}-{'-'.join(words) or 'report'}"


def write(
    root: Path,
    run: Run,
    question: str,
    markdown: str,
    verdict: dict[str, Any],
    tldr: str = "",
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
    if run.draft is not None:
        (out / "report.json").write_text(json.dumps(run.draft, indent=1), encoding="utf-8")
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
