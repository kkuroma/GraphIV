"""
The mechanical check on a caller's report object, string matching only, against what the run held

Every fault names the path of the field it sits in, so a fix resends that field alone as a patch
A marker names a paragraph a reply gave, and every number its text states must sit in that paragraph
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# new-style ids and the old archive/number form
ID = r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})"
# a marker cites quotes by their run labels, one or several in one bracket
LABEL = r"[A-Z][A-Za-z]*\d+"
MARKER = re.compile(rf"\[({LABEL}(?:\s*,\s*{LABEL})*)\]")
ARXIV = re.compile(rf"\b{ID}\b")
QUOTE = re.compile(r"[\"“]([^\"“”\n]{8,})[\"”]")
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
KINDS = ("finding", "inference", "open", "table", "coverage", "lineage")
# a dict key a patch may create, any other must exist already
FIELDS = {
    "question", "tldr", "overview", "sections", "further_reading", "heading", "blocks", "type",
    "text", "cites", "search", "table", "rows", "columns", "poset", "papers",
}  # fmt: skip
# faults that leave nothing to draw, so the report is checked but not rendered
SHAPE = {
    "not an object", "unknown type", "no such table", "no such poset", "missing", "empty",
    "empty text", "not a choice field", "not bool fields or one choice field",
}  # fmt: skip
# the fields of a block a fault can name, whose block is the text the fault sits in
LEAVES = (
    ".text", ".cites", ".search", ".table", ".poset", ".papers", ".rows", ".columns", ".type",
)  # fmt: skip
# an open block's search must have had one of its nearest hits read
NEAREST = 3
EFFORTS = ("quick", "standard", "deep")
# a deep report cites at least this many citing sentences from its posets
RELATIONS = 3
# a title prefix before its colon counts as the paper's name up to this many words
NAME_WORDS = 3


@dataclass
class Held:
    """
    What a session holds that a report may cite or place
    """

    evidence: dict[str, str]
    sources: dict[str, str]
    read: set[str]
    served: set[str]
    searches: dict[str, dict[str, Any]]
    tables: dict[str, dict[str, Any]]
    posets: dict[str, dict[str, Any]]
    titles: dict[str, str] = field(default_factory=dict)
    # the paper each marker quotes, and each paper's short name, which a TL;DR cites beside it
    owners: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    # the folded titles and abstracts of the papers read, where a read paper's own name is found
    known: str = ""


@dataclass
class Verdict:
    """
    Every fault by the path of its field, and how many markers the report carries
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

    @property
    def drawable(self) -> bool:
        """
        Whether every id the report names exists, so it can be rendered

        Returns:
            True when no fault is one of shape.
        """
        return not any(f["kind"] in SHAPE for f in self.faults)

    def add(self, path: str, kind: str, detail: str = "") -> None:
        """
        Records one fault

        Args:
            path: The field it sits in, like sections.2.blocks.4.text.
            kind: What is wrong.
            detail: What to do about it, or the value at fault.
        """
        self.faults.append({"path": path, "kind": kind, "detail": detail})

    def as_dict(self) -> dict[str, object]:
        """
        The verdict as the run file holds it

        Returns:
            The fields.
        """
        return {"passed": self.passed, "markers": self.markers, "faults": self.faults}


def name_of(title: str) -> str:
    """
    A paper's short name, the part of its title before a colon when that reads as a name

    Args:
        title: The title.

    Returns:
        The name, like Mamba or Hungry Hungry Hippos, or empty when the title has none.
    """
    head, colon, _ = " ".join(str(title).split()).partition(":")
    words = head.split()
    if not colon or not 0 < len(words) <= NAME_WORDS or head in COMMON:
        return ""
    return head if any(c.isupper() for c in head) and len(head) >= 3 else ""


def relations(held: Held) -> set[str]:
    """
    The markers of the citing sentences a session's posets labeled

    Args:
        held: What the session holds.

    Returns:
        The markers.
    """
    return {e["label"] for p in held.posets.values() for e in p["edges"] if e.get("label")}


def name_check(path: str, block: dict[str, Any], held: Held, verdict: Verdict) -> None:
    """
    A TL;DR naming a paper cites one of that paper's own markers, which goes beside the name

    Only a paper the session holds a marker of counts, since nothing else could be cited.

    Args:
        path: The block's path.
        block: The TL;DR.
        held: What the session holds.
        verdict: Where faults go.
    """
    cites, text = cites_of(block)
    mine = {held.owners.get(c) for c in cites}
    have = set(held.owners.values())
    missing = [
        name
        for arxiv_id, name in held.names.items()
        if name and arxiv_id in have and arxiv_id not in mine
        and re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text)
    ]  # fmt: skip
    if missing:
        verdict.add(
            f"{path}.cites",
            "names a paper without its marker",
            f"{', '.join(sorted(set(missing)))}; cite a marker from each paper you name",
        )


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


def cites_of(block: dict[str, Any]) -> tuple[list[str], str]:
    """
    A block's markers, from its cites list and any bracket left in its text, and the bare text

    Args:
        block: A finding, an open block or the TL;DR.

    Returns:
        The markers in order without repeats, and the text without brackets.
    """
    text = str(block.get("text") or "")
    raw = block.get("cites") or []
    listed = [str(c).strip().strip("[]") for c in (raw if isinstance(raw, list) else [raw])]
    inline = [m.strip() for g in MARKER.findall(text) for m in g.split(",")]
    bare = re.sub(r"\s+([.,;:!?)])", r"\1", " ".join(MARKER.sub("", text).split()))
    return list(dict.fromkeys(c for c in listed + inline if c)), bare


def claim(path: str, block: dict[str, Any], held: Held, verdict: Verdict, need: bool) -> None:
    """
    The rules a cited text follows: known markers, its numbers, quotes and names behind them

    A name passes when a cited paper's full text holds it, or the title or abstract of a paper
    the session read, since a follow-up cites the work it extends by its authors, not its name.

    Args:
        path: The block's path.
        block: The block.
        held: What the session holds.
        verdict: Where faults go.
        need: Whether the text must carry a marker.
    """
    cites, text = cites_of(block)
    if not text:
        verdict.add(f"{path}.text", "empty text")
        return
    verdict.markers += len(cites)
    if need and not cites:
        verdict.add(
            f"{path}.cites", "cites nothing", "cite the read behind it, or make it an inference"
        )
        return
    unknown = [c for c in cites if c not in held.evidence]
    for c in unknown:
        hint = f"{c} is an arXiv id; cite a marker a reply gave for it" if ARXIV.fullmatch(c) else c
        verdict.add(f"{path}.cites", "unknown marker", hint)
    known = [c for c in cites if c in held.evidence]
    if not known:
        return
    folded = [fold(held.evidence[c]) for c in known]
    numbers = set().union(*(numbers_in(t) for t in folded))
    missing = sorted(n for n in numbers_in(text) if not backed(n, numbers))
    if missing and not unknown:
        verdict.add(f"{path}.text", "number not in its paragraph", ", ".join(missing))
    for quote in QUOTE.findall(text):
        if not any(fold(quote) in t for t in folded):
            verdict.add(f"{path}.text", "quote not in its paragraph", quote)
    texts = [held.sources.get(c, "") for c in known]
    if all(texts):
        absent = [
            n for n in names_in(text)
            if not any(re.search(rf"\b{re.escape(n.lower())}\b", t) for t in texts + [held.known])
        ]  # fmt: skip
        if absent:
            verdict.add(
                f"{path}.text",
                "name not in the papers cited",
                f"{', '.join(absent)}; add a marker from a read that names it, or drop the name",
            )
    for arxiv_id in sorted(set(ARXIV.findall(text))):
        if arxiv_id not in held.read:
            verdict.add(f"{path}.text", "paper named but never read", arxiv_id)


def block_check(path: str, block: Any, held: Held, verdict: Verdict) -> None:
    """
    One block against the rules of its type

    Args:
        path: The block's path.
        block: The block as the caller sent it.
        held: What the session holds.
        verdict: Where faults go.
    """
    if not isinstance(block, dict):
        verdict.add(path, "not an object", f"a block is one of {', '.join(KINDS)}")
        return
    kind = block.get("type")
    if kind == "finding":
        claim(path, block, held, verdict, need=True)
    elif kind == "inference":
        if cites_of(block)[0]:
            verdict.add(f"{path}.cites", "an inference carries no marker", "make it a finding")
        if not str(block.get("text") or "").strip():
            verdict.add(f"{path}.text", "empty text")
    elif kind == "open":
        open_check(path, block, held, verdict)
    elif kind in ("table", "coverage"):
        table = str(block.get("table") or "")
        if table not in held.tables:
            verdict.add(
                f"{path}.table",
                "no such table",
                f"this session has {', '.join(held.tables) or 'none'}",
            )
        elif kind == "coverage":
            coverage_check(path, block, held.tables[table], verdict)
    elif kind == "lineage":
        pid = str(block.get("poset") or "")
        if pid not in held.posets:
            verdict.add(
                f"{path}.poset",
                "no such poset",
                f"this session has {', '.join(held.posets) or 'none'}",
            )
            return
        inside = {p["arxiv_id"] for p in held.posets[pid]["papers"]}
        wanted = block.get("papers") or []
        strays = [str(a) for a in (wanted if isinstance(wanted, list) else [wanted])]
        strays = [a for a in strays if a not in inside]
        if strays:
            verdict.add(f"{path}.papers", "papers not in the poset", ", ".join(strays))
    else:
        verdict.add(f"{path}.type", "unknown type", f"one of {', '.join(KINDS)}")


def open_check(path: str, block: dict[str, Any], held: Held, verdict: Verdict) -> None:
    """
    An open block names a search the session ran, one of whose nearest hits it read

    Args:
        path: The block's path.
        block: The block.
        held: What the session holds.
        verdict: Where faults go.
    """
    words = fold(str(block.get("search") or ""))
    if not words or words not in held.searches:
        verdict.add(
            f"{path}.search", "search never run", "run search_papers with these exact words first"
        )
    else:
        near = [p["arxiv_id"] for p in held.searches[words].get("papers", [])[:NEAREST]]
        if near and not set(near) & held.read:
            verdict.add(
                f"{path}.search",
                "nearest hits never read",
                f"read one of {', '.join(near)}; if it covers this area, write a finding instead",
            )
    # a number in an open area is a claim, so it needs the marker of the paragraph stating it
    if block.get("cites") or numbers_in(str(block.get("text") or "")):
        claim(path, block, held, verdict, need=True)
    elif not str(block.get("text") or "").strip():
        verdict.add(f"{path}.text", "empty text")


def coverage_check(
    path: str, block: dict[str, Any], table: dict[str, Any], verdict: Verdict
) -> None:
    """
    A coverage block pivots one choice field against bool fields or one more choice field

    Args:
        path: The block's path.
        block: The block.
        table: The table it pivots.
        verdict: Where faults go.
    """
    kinds = {c["name"]: c["kind"] for c in table["schema"]["columns"]}
    listed = ", ".join(f"{k} ({t})" for k, t in kinds.items())
    if kinds.get(str(block.get("rows") or "")) != "choice":
        verdict.add(f"{path}.rows", "not a choice field", f"fields: {listed}")
    cols = block.get("columns") or []
    types = [kinds.get(str(c)) for c in (cols if isinstance(cols, list) else [cols])]
    one_choice = len(types) == 1 and types[0] == "choice"
    if not types or not (all(t == "bool" for t in types) or one_choice):
        verdict.add(f"{path}.columns", "not bool fields or one choice field", f"fields: {listed}")


def blocks_of(report: dict[str, Any]) -> list[tuple[str, Any]]:
    """
    Every block of a report with its path, the overview first

    Args:
        report: The report.

    Returns:
        Path and block pairs.
    """
    out = [(f"overview.{i}", b) for i, b in enumerate(report.get("overview") or [])]
    for i, section in enumerate(report.get("sections") or []):
        if isinstance(section, dict):
            out += [
                (f"sections.{i}.blocks.{j}", b) for j, b in enumerate(section.get("blocks") or [])
            ]
    return out


def check(report: Any, held: Held) -> Verdict:
    """
    Every fault a report carries, each under the path of its field

    Args:
        report: The report object as the caller sent it.
        held: What the session holds.

    Returns:
        The verdict.
    """
    verdict = Verdict()
    if not isinstance(report, dict):
        verdict.add("", "not an object")
        return verdict
    if not str(report.get("question") or "").strip():
        verdict.add("question", "empty")
    tldr = report.get("tldr")
    if not isinstance(tldr, dict):
        verdict.add("tldr", "missing", 'pass {"text": ..., "cites": [...]}')
    else:
        claim("tldr", tldr, held, verdict, need=True)
        name_check("tldr", tldr, held, verdict)
    if not isinstance(report.get("overview"), list) or not report["overview"]:
        verdict.add("overview", "missing", "finding or inference blocks framing the answer")
    sections = report.get("sections")
    if not isinstance(sections, list) or not sections:
        verdict.add("sections", "missing")
    for i, section in enumerate(sections if isinstance(sections, list) else []):
        if not isinstance(section, dict):
            verdict.add(f"sections.{i}", "not an object", 'pass {"heading": ..., "blocks": [...]}')
            continue
        if not str(section.get("heading") or "").strip():
            verdict.add(f"sections.{i}.heading", "empty")
        if not isinstance(section.get("blocks"), list) or not section["blocks"]:
            verdict.add(f"sections.{i}.blocks", "empty")
    for path, block in blocks_of(report):
        block_check(path, block, held, verdict)
    further = report.get("further_reading") or []
    for i, arxiv_id in enumerate(further if isinstance(further, list) else [further]):
        if str(arxiv_id).strip() not in held.served:
            verdict.add(f"further_reading.{i}", "paper never returned by a tool", str(arxiv_id))
    return verdict


def place_tables(report: Any, held: Held) -> Any:
    """
    A deep report with a table section added when it places none, the table and its coverage

    The coverage pivots the table's first choice field against its bool fields, the layout the
    skill asks for, so the writer spends no save on placing what the server can draw.

    Args:
        report: The report object.
        held: What the session holds.

    Returns:
        The report, with the section appended when it placed no table.
    """
    if not isinstance(report, dict) or not isinstance(report.get("sections"), list):
        return report
    placed = {b.get("type") for _, b in blocks_of(report) if isinstance(b, dict)}
    if "table" in placed or not held.tables:
        return report
    blocks: list[dict[str, Any]] = []
    for tid, table in held.tables.items():
        blocks.append({"type": "table", "table": tid})
        kinds = [(c["name"], c["kind"]) for c in table["schema"]["columns"]]
        choices = [n for n, k in kinds if k == "choice"]
        bools = [n for n, k in kinds if k == "bool"]
        if choices and bools and "coverage" not in placed:
            blocks.append({"type": "coverage", "table": tid, "rows": choices[0], "columns": bools})
    return {
        **report,
        "sections": report["sections"] + [{"heading": "Comparison", "blocks": blocks}],
    }


def contract(
    report: Any,
    held: Held,
    effort: str,
    traced: bool,
    verdict: Verdict,
    unread: list[str],
    at: int | None = None,
) -> None:
    """
    What a deep report must hold, every missing part in one fault at the path of a new section

    Args:
        report: The report object.
        held: What the session holds.
        effort: The effort start_session was given.
        traced: Whether the session called trace_lineages.
        verdict: Where the fault goes.
        unread: Papers a deep report must have read that the session has not.
        at: Where a new section goes in the writer's own draft, when the server added sections.
    """
    if effort != "deep" or not isinstance(report, dict):
        return
    kinds = [b.get("type") for _, b in blocks_of(report) if isinstance(b, dict)]
    need = []
    if "table" not in kinds:
        need.append(
            '{"type": "table", "table": "T1"}'
            if held.tables
            else "a table: first read the papers you compare with read_paper and a schema"
        )
    if "coverage" not in kinds:
        need.append('{"type": "coverage", "table": "T1", "rows": <choice>, "columns": [<bools>]}')
    if not traced:
        need.append("a poset: first call trace_lineages with a phrase for the field")
    used = {c for _, b in blocks_of(report) if isinstance(b, dict) for c in cites_of(b)[0]}
    if traced and len(used & relations(held)) < RELATIONS:
        need.append(
            f"at least {RELATIONS} findings citing trace_lineages' sentences, each saying which "
            "paper extends, critiques or compares against which"
        )
    said = {
        held.owners.get(c)
        for _, b in blocks_of(report)
        if isinstance(b, dict) and b.get("type") in ("finding", "open")
        for c in cites_of(b)[0]
    }
    # a row whose every choice field the paper leaves unanswered is off the question
    rows = [
        r["arxiv_id"]
        for t in held.tables.values()
        for r in t.get("rows", [])
        if any(
            (r.get("cells", {}).get(c["name"]) or {}).get("value") not in (None, "n/a")
            for c in t["schema"]["columns"]
            if c["kind"] == "choice"
        )
    ]
    unsaid = [a for a in dict.fromkeys(rows) if a not in said]
    if unsaid:
        need.append(
            f"findings citing each paper of your table, none cites yet: {', '.join(unsaid)}"
        )
    if "open" not in kinds:
        need.append(
            '{"type": "open", "text": ..., "search": <exact words of a focused search you ran>}, '
            "one per checked open area"
        )
    if unread:
        verdict.add(
            "reads",
            "deep effort needs these papers read",
            f"{', '.join(unread)}: the top of your question's own searches, the newest papers "
            "building on your table and the papers your poset argues with most; read them, with "
            'schema="T1" where they belong in the table, then save again',
        )
    if need:
        sections = report.get("sections")
        spot = at if at is not None else len(sections) if isinstance(sections, list) else 0
        verdict.add(
            f"sections.{spot}",
            "deep effort needs more blocks",
            f'add one new section with patch={{"sections.{spot}": {{"heading": "<a heading>", '
            '"blocks": [<the blocks>]}}, or add blocks to an existing section at '
            "sections.<i>.blocks.<next index>; it needs: " + "; ".join(need),
        )


def parts(path: str) -> list[str]:
    """
    A path split into its keys and indices, dots or slashes alike

    Args:
        path: Like sections.2.blocks.4 or sections/2/blocks/4.

    Returns:
        The parts.
    """
    return [x for x in re.split(r"[./]", path.strip("./ ")) if x]


def apply(root: dict[str, Any], path: str, value: Any) -> bool:
    """
    Sets one path in place, null deleting it and an index one past the end appending

    Args:
        root: The report.
        path: Where.
        value: The new value, or None to delete.

    Returns:
        Whether the path existed or could be created.
    """
    keys = parts(path)
    if not keys:
        return False
    node: Any = root
    for key in keys[:-1]:
        # a path through the index one past the end opens a new section or block there
        if isinstance(node, list) and key.isdigit() and int(key) == len(node) and value is not None:
            node.append({})
        if isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return False
    last = keys[-1]
    if isinstance(node, list) and last.isdigit():
        k = int(last)
        if value is None and k < len(node):
            node.pop(k)
        elif k < len(node):
            node[k] = value
        elif k == len(node) and value is not None:
            node.append(value)
        else:
            return False
        return True
    if isinstance(node, dict):
        if value is None:
            return node.pop(last, None) is not None
        if last not in node and last not in FIELDS:
            return False
        node[last] = value
        return True
    return False


def key_of(item: Any) -> str:
    """
    What identifies a section or a block when a whole list is resent, its heading or its text

    Args:
        item: A section or a block.

    Returns:
        The folded heading or text, or the item as text.
    """
    if isinstance(item, dict):
        return fold(str(item.get("heading") or item.get("text") or item))
    return fold(str(item))


def patch(draft: dict[str, Any], changes: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    The draft with each path set, shallow paths first, and deletions last from the back

    A whole sections list holding only new headings is appended, as the writer adds a section.
    Any other whole list replaces the old one, since a refusal leaves the writer resending it.

    Args:
        draft: The report the server holds.
        changes: New values by path, null deleting.

    Returns:
        The patched copy and the paths that could not be applied.
    """
    bad: list[str] = []
    out = copy.deepcopy(draft)
    changes = dict(changes)
    for whole in ("sections", "overview"):
        if whole not in changes:
            continue
        given = changes.pop(whole)
        found = out.get(whole)
        old: list[Any] = found if isinstance(found, list) else []
        if not isinstance(given, list):
            bad.append(f"{whole}, which takes a list")
            continue
        keys = [key_of(x) for x in old]
        new = [key_of(x) for x in given]
        if whole == "sections" and given and not set(keys) & set(new):
            out[whole] = old + given
        else:
            out[whole] = given
    sets = sorted(
        ((p, x) for p, x in changes.items() if x is not None), key=lambda c: len(parts(c[0]))
    )
    for path, value in sets:
        if not apply(out, path, value):
            bad.append(path)
    # deletions from the back, so an earlier index still means what the writer saw
    deletes = [p for p, x in changes.items() if x is None]
    for path in sorted(
        deletes, key=lambda p: [int(x) if x.isdigit() else -1 for x in parts(p)], reverse=True
    ):
        if not apply(out, path, None):
            bad.append(path)
    return out, bad


def at(report: Any, path: str) -> Any:
    """
    The value at a path, or None

    Args:
        report: The report.
        path: The path.

    Returns:
        The value.
    """
    node: Any = report
    for key in parts(path):
        if isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        elif isinstance(node, dict):
            node = node.get(key)
        else:
            return None
    return node


def owner(path: str) -> str:
    """
    The path of the block or field a fault's path points into

    Args:
        path: A fault's path, like sections.2.blocks.4.text.

    Returns:
        The path without its last key when that key is a block's field.
    """
    return path.rsplit(".", 1)[0] if path.endswith(LEAVES) else path


def lined(verdict: Verdict, report: Any) -> dict[str, object]:
    """
    The verdict as the run file keeps it, each fault with the text it sits in for the webui

    Args:
        verdict: The check's result.
        report: The report checked.

    Returns:
        The verdict, its faults carrying where and line beside their path.
    """
    out = verdict.as_dict()
    faults = []
    for fault in verdict.faults:
        shown = at(report, owner(fault["path"]))
        text = (
            cites_of(shown)[1]
            if isinstance(shown, dict)
            else shown if isinstance(shown, str) else ""
        )
        faults.append({**fault, "where": fault["detail"], "line": text.rstrip(".!?")})
    out["faults"] = faults
    return out


def summary(verdict: Verdict, report: Any, unplaced: list[str]) -> str:
    """
    The verdict as the writer reads it, each fault under the path a patch sets

    Args:
        verdict: The check's result.
        report: The report checked.
        unplaced: Tables and lines the server adds since the report did not place them.

    Returns:
        The text.
    """
    if verdict.passed:
        return (
            f"PASSED: {verdict.markers} markers, every one resolved. The run is finished: "
            "reply with one line saying where the report was saved, and call no more tools."
        )
    lines = [
        f"NOT PASSED: {len(verdict.faults)} faults. The server keeps this draft. "
        'Fix only these fields with save_report(session_id, patch={"<path>": <new value>}); '
        "a block path takes the whole block, null deletes it."
    ]
    for fault in verdict.faults:
        shown = at(report, owner(fault["path"]))
        detail = f" {fault['detail']}" if fault["detail"] else ""
        lines.append(f"- {fault['path']}: {fault['kind']}{detail}")
        if shown is not None and not isinstance(shown, list):
            lines.append(f"    now: {json.dumps(shown, ensure_ascii=False)[:220]}")
    if unplaced:
        lines.append(
            f"Not placed, so added by the server under Also from this study: {', '.join(unplaced)}."
        )
    return "\n".join(lines)
