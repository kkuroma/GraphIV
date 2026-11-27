"""
Parse the rendered HTML of every finished paper into citation rows

A paper is asked once and its verdict kept in is_citation_parsable, so a null is the queue
Workers are forked after the index is built, which lets them share it instead of pickling 3M rows
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import psycopg
from citation_regex import SENTINEL, Index, extract_citations

from src.core.config import load_config

MODERN = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")
LEGACY = re.compile(r"^[a-z-]+(?:\.[A-Za-z-]+)?/(\d{2})(\d{2})\d{3}$")
MARK = "[CITE]"
ANY_MARK = re.compile(rf"{SENTINEL}(\d+){SENTINEL}")

# a flattened LaTeXML footnote renders its marker three times in a row
FOOTNOTE = re.compile(r"(?<!\S)(\S{1,3})(?: \1){2}(?!\S)")
OPEN_GAP = re.compile(r"([(\[])\s+")
CLOSE_GAP = re.compile(r"\s+([)\]])")
SPACE_PUNCT = re.compile(r"\s+([,.;:!?])")
DANGLING = re.compile(r"(?<![\w)\]])\s*-(?=[A-Za-z])")

"""
Abbreviations, initials and decimals are masked before splitting, since their full stops are not
sentence ends and `et al.` alone accounts for a large share of citing sentences.
"""
ABBREV = (
    "et al|e\\.g|i\\.e|cf|vs|resp|approx|Fig|Figs|Eq|Eqs|Sec|Secs|Ref|Refs|Tab|Tabs"
    "|Thm|Def|Prop|Lem|Cor|Alg|App|Ch|No|Vol|pp|St|Dr|Prof|Mr|Ms|Mrs|Inc|Ltd|vol|no"
)
ABBREV_DOT = re.compile(rf"\b({ABBREV})\.", re.IGNORECASE)
INITIAL = re.compile(r"\b([A-Z])\.(?=\s*[A-Z])")
DECIMAL = re.compile(r"(\d)\.(\d)")
GUARD = "\ue002"
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"\'(\[]?[A-Z0-9])")


def clean_paragraph(text: str) -> str:
    """
    Normalize a rendered paragraph into readable prose

    Collapses whitespace, drops the tripled footnote markers LaTeXML leaves behind, and closes up
    the spaces it puts inside brackets and before punctuation.

    Args:
        text: the paragraph as rendered by the HTML parser.

    Returns:
        The tidied paragraph, still carrying SENTINEL.
    """
    out = " ".join(text.split())
    out = FOOTNOTE.sub("", out)
    out = OPEN_GAP.sub(r"\1", out)
    out = CLOSE_GAP.sub(r"\1", out)
    out = DANGLING.sub(" ", out)
    out = SPACE_PUNCT.sub(r"\1", out)
    return " ".join(out.split()).strip()


def extract_sentence(paragraph: str, target: str) -> str:
    """
    Return the sentence of `paragraph` that carries this citation's marker

    Args:
        paragraph: a cleaned paragraph containing the markers.
        target: the marker belonging to the citation being extracted.

    Returns:
        The sentence holding `target`, or the whole paragraph when it is absent.
    """
    if target not in paragraph:
        return paragraph

    masked = ABBREV_DOT.sub(lambda m: m.group(1) + GUARD, paragraph)
    masked = INITIAL.sub(lambda m: m.group(1) + GUARD, masked)
    masked = DECIMAL.sub(lambda m: m.group(1) + GUARD + m.group(2), masked)

    pos = 0
    at = masked.find(target)
    for piece in SPLIT.split(masked):
        end = pos + len(piece)
        if pos <= at < end:
            return piece.replace(GUARD, ".").strip()
        pos = end + 1
    return paragraph


# forked workers inherit these rather than receiving them down a pipe
INDEX: Index | None = None
HTML_ROOT: Path | None = None


@dataclass(frozen=True)
class Citation:
    """
    One citation occurrence as the citations table holds it
    """

    target_id: str | None
    section: str | None
    sentence: str | None
    resolved_by: str


def yymm(arxiv_id: str) -> str | None:
    """
    The YYMM directory a paper belongs in, or None when the id is unparsable

    Args:
        arxiv_id: a modern `YYMM.NNNNN` or legacy `archive/YYMMNNN` id.

    Returns:
        The four-digit YYMM, or None.
    """
    m = MODERN.match(arxiv_id) or LEGACY.match(arxiv_id)
    return f"{m.group(1)}{m.group(2)}" if m else None


def html_path(root: Path, arxiv_id: str) -> Path | None:
    """
    Canonical on-disk path for a paper's HTML

    Args:
        root: the HTML store root.
        arxiv_id: the paper id.

    Returns:
        The path, or None when the id is unparsable.
    """
    ym = yymm(arxiv_id)
    if ym is None:
        return None
    return root / ym / f"{arxiv_id.replace('/', '')}.html"


def citations_for(arxiv_id: str) -> tuple[str, list[Citation]]:
    """
    Extract every citation occurrence in one paper

    Args:
        arxiv_id: the paper to parse.

    Returns:
        The paper id and its citations, empty when the file is missing or unreadable.
    """
    assert INDEX is not None and HTML_ROOT is not None
    path = html_path(HTML_ROOT, arxiv_id)
    if path is None or not path.is_file():
        return arxiv_id, []
    try:
        bibitems, occurrences, marker_text = extract_citations(path, INDEX)
    except (OSError, ValueError, RuntimeError):
        return arxiv_id, []

    out: list[Citation] = []
    for occ in occurrences:
        bib = bibitems.get(occ.bib_local_id)
        mine = f"{SENTINEL}{occ.ordinal}{SENTINEL}"
        para = clean_paragraph(occ.paragraph)
        sentence = extract_sentence(para, mine)
        out.append(
            Citation(
                target_id=bib.arxiv_id if bib else None,
                section=occ.section,
                sentence=render(sentence, mine, marker_text),
                resolved_by=bib.method if bib else "none",
            )
        )
    return arxiv_id, out


def render(text: str, mine: str, marker_text: dict[int, str]) -> str:
    """
    Show this citation as MARK and put every other citation back to how it was written

    Args:
        text: a paragraph or sentence carrying the markers.
        mine: the marker belonging to this citation.
        marker_text: the original rendered text of every citation, by ordinal.

    Returns:
        The text as a reader would see it, with one MARK in it.
    """
    out = text.replace(mine, MARK)
    return ANY_MARK.sub(lambda m: marker_text.get(int(m.group(1)), ""), out)


class CitationBuilder:
    """
    The HTML store and the papers still missing from the citations table
    """

    def __init__(self, html_path: str) -> None:
        """
        Args:
            html_path: root of the local HTML store.
        """
        self.html_path = Path(html_path)

    def pending(self, pg_dest: str, min_stage: int) -> list[str]:
        """
        Every finished paper that has never been asked for its citations

        `is_citation_parsable` is the queue rather than the presence of rows, since about a fifth
        of the corpus renders html that carries no bibliography and would be parsed on every run.

        Args:
            pg_dest: libpq connection string.
            min_stage: the stage a paper must have reached.

        Returns:
            The paper ids to parse.
        """
        with psycopg.connect(pg_dest) as conn:
            rows: list[tuple[str]] = conn.execute(
                "SELECT arxiv_id FROM papers" " WHERE stage >= %s AND is_citation_parsable IS NULL",
                (min_stage,),
            ).fetchall()
        return [r[0] for r in rows]

    def populate_citation_table(
        self,
        pg_dest: str,
        min_stage: int = 3,
        workers: int = 10,
        batch: int = 2000,
        limit: int = 0,
    ) -> int:
        """
        Parse every pending paper, write its citations and record whether it had any

        Args:
            pg_dest: libpq connection string.
            min_stage: the stage a paper must have reached.
            workers: parallel parser processes.
            batch: papers per commit.
            limit: stop after this many papers, or 0 for all of them.

        Returns:
            How many citation rows were written.
        """
        global INDEX, HTML_ROOT
        todo = self.pending(pg_dest, min_stage)
        if limit:
            todo = todo[:limit]
        print(f"[stage0] {len(todo):,} papers to parse", flush=True)
        if not todo:
            return 0

        INDEX = Index.load(pg_dest)
        HTML_ROOT = self.html_path
        written, parsable = 0, 0
        ctx = multiprocessing.get_context("fork")
        with psycopg.connect(pg_dest, autocommit=True) as conn:
            _ = conn.execute("CREATE TEMP TABLE q (arxiv_id text, v boolean)")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                for i in range(0, len(todo), batch):
                    chunk = todo[i : i + batch]
                    verdicts: list[tuple[str, bool]] = []
                    with conn.cursor().copy(
                        "COPY citations (source_id, target_id, section, resolved_by,"
                        " sentence, stage) FROM STDIN"
                    ) as cp:
                        for arxiv_id, found in pool.map(citations_for, chunk):
                            verdicts.append((arxiv_id, bool(found)))
                            for c in found:
                                cp.write_row(
                                    (
                                        arxiv_id,
                                        c.target_id,
                                        c.section,
                                        c.resolved_by,
                                        c.sentence,
                                        0,
                                    )
                                )
                                written += 1
                    with conn.cursor().copy("COPY q (arxiv_id, v) FROM STDIN") as cp:
                        for arxiv_id, verdict in verdicts:
                            cp.write_row((arxiv_id, verdict))
                    _ = conn.execute(
                        "UPDATE papers p SET is_citation_parsable = q.v"
                        " FROM q WHERE p.arxiv_id = q.arxiv_id"
                    )
                    _ = conn.execute("TRUNCATE q")
                    parsable += sum(1 for _a, v in verdicts if v)
                    print(
                        f"[stage0] {i + len(chunk):,}/{len(todo):,} papers, {written:,} citations,"
                        f" {parsable:,} with a bibliography",
                        flush=True,
                    )
        return written


def main() -> None:
    """
    Read the config, then parse every finished paper that has never been asked
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--config_path", default="config.toml")
    _ = ap.add_argument("--min-stage", type=int, default=3)
    _ = ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    _ = ap.add_argument("--limit", type=int, default=0, help="stop after N papers")
    args = ap.parse_args()

    cfg = load_config(str(args.config_path))
    builder = CitationBuilder(html_path=str(cfg["dataset"]["HTML_PATH"]))
    _ = builder.populate_citation_table(
        str(cfg["database"]["PG_DEST"]),
        min_stage=int(args.min_stage),  # pyright: ignore[reportAny]
        workers=int(args.workers),  # pyright: ignore[reportAny]
        limit=int(args.limit),  # pyright: ignore[reportAny]
    )


if __name__ == "__main__":
    main()
