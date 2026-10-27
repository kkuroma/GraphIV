"""
PaSa's AutoScholarQuery as train, val and test, and RealScholarQuery as real, each capped at its source paper's date

PaSa records only the source's date, so the source is the paper of that date citing the most answers, and it is blocked
A row is kept only when every answer is an embedded paper of the graph, and keeps its raw position as `row`
"""

import json
from datetime import date

from src import common
from src.schema import Case, Dataset

SOURCE = common.SOURCES / "pasa"
# split to folder, file and sample id prefix
FILES = {
    "train": ("AutoScholarQuery", "train", "asqt"),
    "val": ("AutoScholarQuery", "dev", "asqv"),
    "test": ("AutoScholarQuery", "test", "asq"),
    "real": ("RealScholarQuery", "test", "rsq"),
}
# days either side of the published date a citing paper may sit and still count as the source
WINDOW = 3


def find_sources(graph: common.Graph, key: list[str], when: date) -> list[str]:
    """
    The papers dated near `when` that cite the most answers, when they cite at least half of them

    Args:
        graph: The frozen graph.
        key: The answers.
        when: The source's published date.

    Returns:
        The source's arXiv ids, several on a tie and none when nothing qualifies.
    """
    day = graph.day_number(when.isoformat())
    rows = [graph.index[a] for a in key if a in graph.index]
    hits: dict[int, int] = {}
    for r in rows:
        for c in graph.citing(r):
            if abs(int(graph.day[c]) - day) <= WINDOW:
                hits[int(c)] = hits.get(int(c), 0) + 1
    if not hits:
        return []
    best = max(hits.values())
    return [graph.ids[c] for c, n in hits.items() if n == best and n * 2 >= len(rows)]


class Pasa(Dataset):
    NAME = "pasa"
    SPLITS = tuple(FILES)

    def path(self, split: str) -> str:
        folder, file, _ = FILES[split]
        return str(SOURCE / folder / f"{file}.jsonl")

    def source_size(self, split: str) -> int:
        with open(self.path(split)) as f:
            return sum(1 for _ in f)

    def build(self, split: str) -> list[Case]:
        graph = common.load_graph()
        tag = FILES[split][2]
        cases = []
        with open(self.path(split)) as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                key, t = r["answer_arxiv_id"], r["source_meta"].get("published_time")
                if not key or not t or not all(a in graph.index and graph.stage[graph.index[a]] >= 3 for a in key):
                    continue
                when = date(int(t[:4]), int(t[4:6]), int(t[6:8]))
                anchors = find_sources(graph, key, when)
                cases.append({"sample_id": f"{tag}{i:04d}", "query": r["question"], "cutoff": when.isoformat(), "anchors": anchors, "key": key, "row": i})
        return cases


DATASET = Pasa
