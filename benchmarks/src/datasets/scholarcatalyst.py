"""
ScholarCatalyst's core and subfield queries, each capped at its source paper's date and scored inside the benchmark's own corpus

The key keeps every positive the authors gave, so one from S2 or OpenAlex counts against recall as it does in their scoring
The cutoff is the source paper's day where theirs is its month, as BoundedField takes a date
"""

import json
from pathlib import Path

from src import common
from src.schema import Case, Dataset

SOURCE = common.SOURCES / "scholarcatalyst"
# split to the relevance file and its query type
FILES = {"core": "core_query", "subfield": "subfield_query"}


def strip(doc_id: str) -> str:
    """
    A corpus id as the store spells it, the arXiv prefix dropped and any other id left whole

    Args:
        doc_id: The benchmark's id.

    Returns:
        The id.
    """
    return doc_id.removeprefix("arxiv_")


class ScholarCatalyst(Dataset):
    NAME = "scholarcatalyst"
    SPLITS = tuple(FILES)

    def corpus(self) -> Path:
        path = common.CASES / self.NAME / "corpus.txt"
        if not path.exists():
            with open(SOURCE / "corpus.jsonl") as f:
                ids = [strip(json.loads(line)["id"]) for line in f]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(ids) + "\n")
        return path

    def build(self, split: str) -> list[Case]:
        queries = {q["id"]: q for q in common.read_jsonl(SOURCE / "queries.jsonl")}
        cases = []
        for r in common.read_jsonl(SOURCE / "rels" / f"{FILES[split]}.jsonl"):
            q = queries[r["query_id"]]
            key = [strip(p["id"] if isinstance(p, dict) else p) for p in r["positive_docs"]]
            cases.append({"sample_id": q["id"], "query": q["question"], "cutoff": q["paper_published"][:10], "anchors": [strip(q["paper_id"])], "key": key})
        return cases


DATASET = ScholarCatalyst
