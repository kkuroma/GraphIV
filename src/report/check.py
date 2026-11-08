"""
The mechanical check on a caller's report, string matching only, against what the run actually read

A marker names a quote a read gave, its numbers must sit in that paragraph, its paper was read
A line stating something carries a marker or is labeled as the writer's inference
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# new-style ids and the old archive/number form
ID = r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})"
# a marker cites quotes by their run labels, one or several in one bracket
LABEL = r"[A-Z][A-Za-z]*\d+"
MARKER = re.compile(rf"\[({LABEL}(?:\s*,\s*{LABEL})*)\]")
ARXIV = re.compile(rf"\b{ID}\b")
QUOTE = re.compile(r"[\"“]([^\"“”\n]{8,})[\"”]")
FENCE = re.compile(r"^(```|~~~)")
FURTHER = re.compile(r"^#{1,6}\s*further reading", re.I)
HEADING = re.compile(r"^#{1,6}\s")
INFERENCE = re.compile(r"^\W*(inference|my inference|note)\b", re.I)
# a report may open on a paragraph starting TL;DR, bold or not, after its title
TLDR = re.compile(r"^\W*tl;?\s?dr\W*", re.I)
# a number as written, with thousands separators allowed, years left out by numbers_in
NUMBER = re.compile(r"(?<![A-Za-z\d.])\d+(?:[.,]\d+)*(?![A-Za-z\d])")
# a name is a token with an inner capital or a digit, which ordinary words and sentence starts lack
NAME = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*|\d[A-Za-z0-9]*)(?:-[A-Za-z0-9]+)*\b")
# a suffix that turns a name into an adjective, as in IPM-based
SUFFIX = r"-(?:based|like|style|free|only|aware)$"
# acronyms that name a field or a unit rather than one paper or method
COMMON = {
    "AI", "API", "CL", "CNN", "CPU", "GNN", "GPU", "GeV", "LLM", "LLMs", "MLP", "ML", "MeV", "NLP",
    "QA", "RL", "RNN", "SOTA", "TeV", "VLM", "VLMs", "LMM", "LMMs", "MLLM", "MLLMs", "RAG",
}  # fmt: skip
# a search the writer says it ran, as in: a focused search for "dense retrieval for code"
CLAIMED = re.compile(r"search(?:ed)?\s+for\s+[\"“]([^\"”]+)[\"”]", re.I)
# a sentence ends at a stop followed by a capital, a list item or a table cell starts its own
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# a sentence this short is a transition or a label rather than a finding
STATEMENT_WORDS = 12
SHOWN = 20


@dataclass
class Verdict:
    """
    What the check found, by kind of fault
    """

    markers: int = 0
    faults: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """
        Whether the report has no fault

        Returns:
            True when it passed.
        """
        return self.markers > 0 and not self.faults

    def as_dict(self) -> dict[str, object]:
        """
        The verdict as the caller and the run file read it

        Returns:
            The fields.
        """
        return {"passed": self.passed, "markers": self.markers, "faults": self.faults}


def fold(text: str) -> str:
    """
    Text compared for quoting, with typography and case folded away

    Args:
        text: Any text.

    Returns:
        The folded text.
    """
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "\xa0": " "})
    return " ".join(unicodedata.normalize("NFKC", text).translate(table).split()).lower()


def names_in(text: str) -> list[str]:
    """
    The distinctive names a text carries, such as a method or a dataset

    Args:
        text: Any text.

    Returns:
        The names, in order.
    """
    found = [re.sub(SUFFIX, "", n) for n in NAME.findall(MARKER.sub(" ", ARXIV.sub(" ", text)))]
    return [n for n in dict.fromkeys(found) if len(n) >= 3 and n not in COMMON]


def numbers_in(text: str) -> set[str]:
    """
    The numbers a text states, without years, single digits or digits inside a name like Qwen3

    Args:
        text: Any text.

    Returns:
        The numbers as strings.
    """
    out = set()
    for raw in NUMBER.findall(MARKER.sub(" ", ARXIV.sub(" ", text))):
        number = raw.replace(",", "")
        if re.fullmatch(r"(19|20)\d\d|\d", number):
            continue
        out.add(number)
    return out


def backed(number: str, held: set[str]) -> bool:
    """
    Whether a stated number is one of the paragraph's numbers, or that number rounded

    Args:
        number: The number as the report states it.
        held: The numbers the cited paragraphs state.

    Returns:
        True when some paragraph number rounds to it at its own precision.
    """
    if number in held:
        return True
    places = len(number.split(".")[1]) if "." in number else 0
    try:
        value = float(number)
        return any(round(float(m), places) == value for m in held if m.replace(".", "").isdigit())
    except ValueError:
        return False


def body_lines(markdown: str) -> list[tuple[str, bool]]:
    """
    The report's lines outside code fences, each with whether it sits under Further reading

    Args:
        markdown: The report.

    Returns:
        Line and section flag pairs.
    """
    out, fenced, further = [], False, False
    for raw in markdown.splitlines():
        line = raw.strip()
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced or not line:
            continue
        if HEADING.match(line):
            further = bool(FURTHER.match(line))
        out.append((line, further))
    return out


def is_statement(line: str) -> bool:
    """
    Whether a sentence states something a reader would take as a finding

    Args:
        line: One sentence or table row.

    Returns:
        True for prose and table rows long enough to claim something.
    """
    if HEADING.match(line) or INFERENCE.match(line) or line.endswith(":"):
        return False
    if set(line) <= set("|-: "):
        return False
    words = re.findall(r"[A-Za-z0-9]+", line)
    return len(words) >= STATEMENT_WORDS


def check(
    markdown: str,
    evidence: dict[str, str],
    read: set[str],
    served: set[str],
    sources: dict[str, str] | None = None,
    queries: set[str] | None = None,
) -> Verdict:
    """
    Every fault a report carries against what its run read

    A name on a cited sentence must appear in the full text of a paper it cites, which catches a
    marker pointing at a paper that says something else. A sentence citing a paper whose text the
    server no longer holds is not checked for names.

    Args:
        markdown: The report.
        evidence: Paragraph text by label, for every quote a read gave.
        read: Papers the reader went through.
        served: Papers any tool returned.
        sources: The folded full text behind each label, for the name rule.
        queries: Every query the run searched with, which a claimed search must be one of.

    Returns:
        The verdict.
    """
    verdict = Verdict()
    folded = {m: fold(t) for m, t in evidence.items()}
    for line, further in body_lines(markdown):
        marks = [m.strip() for group in MARKER.findall(line) for m in group.split(",")]
        verdict.markers += len(marks)
        for mark in marks:
            if mark not in folded:
                verdict.faults.append({"kind": "unknown marker", "where": mark, "line": line})
        if marks:
            held = set().union(*(numbers_in(folded.get(m, "")) for m in marks))
            stated = numbers_in(MARKER.sub("", line))
            missing = sorted(n for n in stated if not backed(n, held))
            if missing and all(m in folded for m in marks):
                verdict.faults.append(
                    {
                        "kind": "number not in its paragraph",
                        "where": ", ".join(missing),
                        "line": line,
                    }
                )
        for quote in QUOTE.findall(line):
            if marks and not any(fold(quote) in folded.get(m, "") for m in marks):
                verdict.faults.append(
                    {"kind": "quote not in its paragraph", "where": quote, "line": line}
                )
        named = set(ARXIV.findall(MARKER.sub("", line)))
        for arxiv_id in sorted(named):
            if further and arxiv_id not in served:
                verdict.faults.append(
                    {"kind": "paper never returned by a tool", "where": arxiv_id, "line": line}
                )
            if not further and arxiv_id not in read:
                verdict.faults.append(
                    {"kind": "paper cited but never read", "where": arxiv_id, "line": line}
                )
        ran = {fold(q) for q in queries or set()}
        for claimed in CLAIMED.findall(line):
            if queries is not None and fold(claimed) not in ran:
                verdict.faults.append(
                    {"kind": "search claimed but never run", "where": claimed, "line": line}
                )
        if further or HEADING.match(line) or INFERENCE.match(line):
            continue
        for sentence in SENTENCE.split(line):
            cited = [m.strip() for g in MARKER.findall(sentence) for m in g.split(",")]
            texts = [sources.get(m, "") for m in cited] if sources else []
            if cited and texts and all(texts):
                absent = [
                    n for n in names_in(sentence)
                    if not any(re.search(rf"\b{re.escape(n.lower())}\b", t) for t in texts)
                ]  # fmt: skip
                if absent:
                    verdict.faults.append(
                        {
                            "kind": "name not in the papers cited",
                            "where": ", ".join(absent),
                            "line": sentence,
                        }
                    )
            if not MARKER.search(sentence) and is_statement(sentence):
                verdict.faults.append(
                    {"kind": "statement with no marker", "where": "", "line": sentence}
                )
    return verdict


def split_tldr(markdown: str) -> tuple[str, str]:
    """
    The report's TL;DR, when its first paragraph after the title is one, and the rest

    Args:
        markdown: The report as the writer sent it.

    Returns:
        The TL;DR on one line without its opener, or empty, and the report without it.
    """
    blocks = markdown.strip().split("\n\n")
    for i, block in enumerate(blocks):
        if re.fullmatch(r"#\s[^\n]*", block.strip()):
            continue
        if not TLDR.match(block.strip()):
            break
        tldr = " ".join(TLDR.sub("", block.strip()).split())
        return tldr, "\n\n".join(blocks[:i] + blocks[i + 1 :]) + "\n"
    return "", markdown


def further_ids(markdown: str) -> list[str]:
    """
    The papers listed under Further reading, in order

    Args:
        markdown: The report.

    Returns:
        The ids.
    """
    return list(
        dict.fromkeys(
            a for line, further in body_lines(markdown) if further for a in ARXIV.findall(line)
        )
    )


def relist(markdown: str, titles: dict[str, str], read: set[str]) -> str:
    """
    The report with its Further reading section rewritten from the store, one paper a line

    The writer supplies ids and the store supplies titles, so a listed title can never be invented.

    Args:
        markdown: The report.
        titles: The line each id is listed as, from the store.
        read: Papers the run read, which belong in the body rather than this list.

    Returns:
        The report.
    """
    lines = markdown.splitlines()
    start = next((i for i, x in enumerate(lines) if FURTHER.match(x.strip())), None)
    if start is None:
        return markdown
    stop = next(
        (i for i in range(start + 1, len(lines)) if HEADING.match(lines[i].strip())), len(lines)
    )
    ids = [
        a
        for a in dict.fromkeys(a for x in lines[start + 1 : stop] for a in ARXIV.findall(x))
        if a not in read
    ]
    listed = [f"- {a} {titles[a]}" if a in titles else f"- {a}" for a in ids]
    if not listed:
        return "\n".join(lines[:start] + lines[stop:]).rstrip() + "\n"
    return "\n".join(lines[: start + 1] + [""] + listed + [""] + lines[stop:]).rstrip() + "\n"


def summary(verdict: Verdict) -> str:
    """
    The verdict as the caller reads it, every fault on one line

    Args:
        verdict: The check's result.

    Returns:
        The text.
    """
    if verdict.passed:
        return f"PASSED: {verdict.markers} markers, every one resolved."
    if not verdict.markers:
        head = "NOT PASSED: the report cites no paragraph. Cite reads by their markers."
    else:
        head = f"NOT PASSED: {len(verdict.faults)} faults over {verdict.markers} markers."
    lines = [head]
    kinds = {f["kind"] for f in verdict.faults}
    if "statement with no marker" in kinds:
        lines.append(
            "A statement with no marker needs the marker of the read that supports it. "
            "Label a line Inference only when no read supports it, and never the answer."
        )
    if "name not in the papers cited" in kinds:
        lines.append(
            "A name on a cited sentence must appear in a paper it cites. "
            "Cite the read that names it, or drop the name."
        )
    if "search claimed but never run" in kinds:
        lines.append(
            "Run the search you describe, with those exact words, or describe the one you ran."
        )
    if "paper cited but never read" in kinds:
        lines.append("Read a paper before citing it, or move it under ## Further reading.")
    for fault in verdict.faults[:SHOWN]:
        where = f" {fault['where']}" if fault["where"] else ""
        lines.append(f"- {fault['kind']}{where}: {fault['line'][:140]}")
    if len(verdict.faults) > SHOWN:
        lines.append(f"- and {len(verdict.faults) - SHOWN} more")
    return "\n".join(lines)
