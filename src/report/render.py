"""
The report object as markdown, with tables, coverage, lines and searches drawn from the session

A table or poset the writer did not place goes under Also from this study, so nothing made is lost
A marker goes beside the name of the paper it quotes when the text names it, else at the end
"""

from __future__ import annotations

import re
from typing import Any

from src.report.check import MARKER, Held, blocks_of, cites_of, fold

CELL_CHARS = 110
TITLE_CHARS = 60
STEP_CHARS = 260
NOT_STATED = "n/a"
# a poset drawn on chosen papers adds up to this many papers citing or cited by two of them
JOINERS = 6


def marks(labels: list[str]) -> str:
    """
    Markers as a bracket after a space, or nothing

    Args:
        labels: The markers.

    Returns:
        The bracket.
    """
    return f" [{', '.join(labels)}]" if labels else ""


def cited(text: str, labels: list[str]) -> str:
    """
    A sentence with its markers placed before its closing stop

    Args:
        text: The sentence.
        labels: Its markers.

    Returns:
        The sentence ending on a stop.
    """
    text = text.strip()
    stop = text[-1] if text[-1:] in (".", "!", "?") else "."
    body = text[:-1] if text[-1:] in (".", "!", "?") else text
    return f"{body}{marks(labels)}{stop}"


def beside(text: str, labels: list[str], held: Held) -> str:
    """
    A sentence with each marker after the name of the paper it quotes, the rest before its stop

    Args:
        text: The sentence.
        labels: Its markers.
        held: What the session holds, which names the paper behind each marker.

    Returns:
        The sentence ending on a stop.
    """
    at: dict[int, list[str]] = {}
    rest = []
    for label in labels:
        name = held.names.get(held.owners.get(label, ""), "")
        hit = re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) if name else None
        if hit:
            at.setdefault(hit.end(), []).append(label)
        else:
            rest.append(label)
    for end in sorted(at, reverse=True):
        text = text[:end] + marks(at[end]) + text[end:]
    return cited(text, rest)


def in_place(block: dict[str, Any], held: Held) -> str:
    """
    A block's text with the markers the writer put inline left where they are, the others placed

    Args:
        block: A finding, an open block or the TL;DR.
        held: What the session holds.

    Returns:
        The sentence ending on a stop.
    """
    cites, _ = cites_of(block)
    raw = " ".join(str(block.get("text") or "").split())
    inline = {m.strip() for g in MARKER.findall(raw) for m in g.split(",")}
    raw = re.sub(r"\s+([.,;:!?)])", r"\1", raw)
    return beside(raw, [c for c in cites if c not in inline], held)


def short(text: str, most: int) -> str:
    """
    Text cut to a length, marked when cut

    Args:
        text: Any text.
        most: The longest it may be.

    Returns:
        The text.
    """
    text = " ".join(str(text).split())
    return text if len(text) <= most else text[: most - 3].rstrip() + "..."


def table_md(tid: str, table: dict[str, Any]) -> str:
    """
    One table, a paper a row, every cell with the marker of the sentence behind it

    Args:
        tid: Its id, like T1.
        table: Its schema and rows.

    Returns:
        The markdown.
    """
    cols = table["schema"]["columns"]
    out = [
        f"**Table {tid}**",
        "",
        "| paper | " + " | ".join(c["name"] for c in cols) + " |",
        "| --- " * (len(cols) + 1) + "|",
    ]
    for row in table["rows"]:
        cells = [f"{short(row['method'], TITLE_CHARS)} ({row['arxiv_id']})"]
        for c in cols:
            cell = row["cells"].get(c["name"], {})
            value = str(cell.get("value", NOT_STATED))
            shown = "not stated" if value == NOT_STATED else short(value, CELL_CHARS)
            cells.append(
                (shown + marks([cell["label"]] if cell.get("label") else [])).replace("|", "/")
            )
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def coverage_md(tid: str, table: dict[str, Any], block: dict[str, Any]) -> str:
    """
    A table's papers counted by one choice field against bool fields or another choice field

    Args:
        tid: The table's id.
        table: Its schema and rows.
        block: The coverage block naming the rows and columns.

    Returns:
        The markdown, each count with the markers of the answers behind it.
    """
    cols = {c["name"]: c for c in table["schema"]["columns"]}
    by = cols[block["rows"]]
    picked = [
        str(c)
        for c in (block["columns"] if isinstance(block["columns"], list) else [block["columns"]])
    ]
    choice = len(picked) == 1 and cols[picked[0]]["kind"] == "choice"
    heads = list(cols[picked[0]]["options"]) if choice else picked

    def cell_of(row: dict[str, Any], head: str) -> dict[str, Any]:
        return row["cells"].get(picked[0] if choice else head, {})

    def hit(row: dict[str, Any], head: str) -> bool:
        return cell_of(row, head).get("value") == (head if choice else "true")

    out = [
        f"**Coverage from Table {tid}**, papers by {by['name']} against {', '.join(picked)}; "
        "a count is the papers whose answer is true, each with the marker of its sentence, "
        "and a paper that does not state it is not counted",
        "",
        f"| {by['name']} | papers | " + " | ".join(heads) + " |",
        "| --- " * (len(heads) + 2) + "|",
    ]
    options = list(by["options"])

    def value(row: dict[str, Any]) -> str:
        return str(row["cells"].get(by["name"], {}).get("value", NOT_STATED))

    for option in options + ["not stated"]:
        members = [
            r for r in table["rows"]
            if (value(r) not in options if option == "not stated" else value(r) == option)
        ]  # fmt: skip
        if option == "not stated" and not members:
            continue
        counts = []
        for h in heads:
            got = [r for r in members if hit(r, h)]
            labels = [cell_of(r, h)["label"] for r in got if cell_of(r, h).get("label")]
            counts.append(f"{len(got)}{marks(labels)}" if got else "0")
        out.append(f"| {option} | {len(members)} | " + " | ".join(counts) + " |")
    return "\n".join(out)


def poset_md(pid: str, poset: dict[str, Any], wanted: list[str]) -> str:
    """
    A poset drawn on some of its papers, the citations among them and their citing sentences

    Args:
        pid: Its id, like P1.
        poset: The poset as trace_lineages kept it.
        wanted: The papers to draw, empty for its longest chain and the papers argued with most.

    Returns:
        The markdown, a mermaid graph and one line per labeled citation.
    """
    papers = {p["arxiv_id"]: p for p in poset["papers"]}
    keep = [a for a in wanted if a in papers] or [a for a in poset["focus"] if a in papers]
    # papers the chosen ones cite or are cited by, two of them at least, so the figure connects
    touch: dict[str, int] = {}
    for e in poset["edges"]:
        for a, b in ((e["citer"], e["cited"]), (e["cited"], e["citer"])):
            if b in keep and a not in keep:
                touch[a] = touch.get(a, 0) + 1
    keep += sorted((a for a, n in touch.items() if n >= 2), key=lambda a: -touch[a])[:JOINERS]
    spot = {a: f"p{i}" for i, a in enumerate(keep)}
    edges = [e for e in poset["edges"] if e["citer"] in spot and e["cited"] in spot]
    implied = {(e["citer"], e["cited"]) for e in edges if e.get("implied")}
    boxes = []
    for a in keep:
        p = papers[a]
        title = (p.get("name") or short(p["title"], 40)).replace('"', "")
        boxes.append(f'{spot[a]}["{title} ({p["date"][:4]})"]')
    arrows = [
        f"{spot[e['citer']]} -->|{e['intent']}| {spot[e['cited']]}"
        for e in edges
        if (e["citer"], e["cited"]) not in implied
    ]
    out = [
        f"**Poset {pid}**: {len(keep)} of its {len(papers)} papers, older ones on the left.",
        "",
        "```mermaid",
        "graph RL",
        *boxes,
        *arrows,
        "```",
        "",
    ]
    for e in edges:
        if e.get("label"):
            who = papers[e["citer"]].get("name") or short(papers[e["citer"]]["title"], 50)
            whom = papers[e["cited"]].get("name") or short(papers[e["cited"]]["title"], 50)
            said = short(e["sentence"], STEP_CHARS)
            out.append(f"- {who} {e['intent']} {whom}: {said}{marks([e['label']])}")
    return "\n".join(out)


def open_md(block: dict[str, Any], held: Held) -> str:
    """
    An open area with the search that checked it and the nearest papers it returned

    Args:
        block: The open block.
        held: What the session holds.

    Returns:
        The line.
    """
    search = held.searches.get(fold(str(block.get("search") or "")), {})
    papers = search.get("papers", [])
    near = " and ".join(f"{short(p['title'], 70)} ({p['arxiv_id']})" for p in papers[:2])
    checked = (
        f"Checked by a {search.get('mode', 'focused')} search for \"{search.get('query', '')}\", "
        f"which returned {len(papers)} papers" + (f", the nearest being {near}." if near else ".")
    )
    return f"Open: {in_place(block, held)} {checked}"


def block_md(block: dict[str, Any], held: Held) -> tuple[str, bool]:
    """
    One block as markdown, and whether it stands apart from the prose around it

    Args:
        block: The block.
        held: What the session holds.

    Returns:
        The markdown and whether it is a figure.
    """
    kind = block["type"]
    if kind == "finding":
        return in_place(block, held), False
    if kind == "inference":
        return f"Inference: {' '.join(str(block['text']).split())}", False
    if kind == "open":
        return open_md(block, held), False
    if kind == "table":
        return table_md(block["table"], held.tables[block["table"]]), True
    if kind == "coverage":
        return coverage_md(block["table"], held.tables[block["table"]], block), True
    papers = block.get("papers") or []
    wanted = [str(a) for a in (papers if isinstance(papers, list) else [papers])]
    return poset_md(block["poset"], held.posets[block["poset"]], wanted), True


def blocks_md(blocks: list[dict[str, Any]], held: Held) -> list[str]:
    """
    Blocks in order, prose one line each and figures set apart by blank lines

    Args:
        blocks: The blocks.
        held: What the session holds.

    Returns:
        The lines.
    """
    out: list[str] = []
    for block in blocks:
        text, apart = block_md(block, held)
        out += ["", text, ""] if apart else [text]
    return out


def render(report: dict[str, Any], held: Held) -> tuple[str, str, list[str]]:
    """
    The report as markdown without its TL;DR, the TL;DR, and what it left unplaced

    The report must be drawable, so every table and poset it names exists.

    Args:
        report: The report object.
        held: What the session holds.

    Returns:
        The markdown, the TL;DR line, and the unplaced tables and posets by name.
    """
    placed = [b for _, b in blocks_of(report) if isinstance(b, dict)]
    tables = {b["table"] for b in placed if b.get("type") in ("table", "coverage")}
    posets = {b["poset"] for b in placed if b.get("type") == "lineage"}
    out = [f"# {str(report['question']).strip()}", ""]
    out += blocks_md(report["overview"], held) + [""]
    for section in report["sections"]:
        out += [f"## {str(section.get('heading') or '').strip()}", ""]
        out += blocks_md(section.get("blocks") or [], held) + [""]
    left_tables = [t for t in held.tables if t not in tables]
    left_posets = [x for x in held.posets if x not in posets]
    if left_tables or left_posets:
        out += ["## Also from this study", ""]
        out += [x for t in left_tables for x in ["", table_md(t, held.tables[t]), ""]]
        out += [x for pid in left_posets for x in ["", poset_md(pid, held.posets[pid], []), ""]]
    further = [str(a).strip() for a in report.get("further_reading") or []]
    unread = [a for a in dict.fromkeys(further) if a not in held.read]
    if unread:
        out += ["## Further reading", ""] + [
            f"- {a} {held.titles.get(a, '')}".rstrip() for a in unread
        ]
    markdown = "\n".join(out)
    while "\n\n\n" in markdown:
        markdown = markdown.replace("\n\n\n", "\n\n")
    unplaced = [f"Table {t}" for t in left_tables] + [f"Poset {x}" for x in left_posets]
    return markdown.strip() + "\n", in_place(report["tldr"], held), unplaced
