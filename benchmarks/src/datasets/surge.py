"""
SurGE's survey task, each survey's references retrieved from its title and capped at its date

SurGE's corpus carries no arXiv id, so papers are matched to the store by title in sources/surge/arxiv_map.json
A survey that anchors any train or val case of GraphIV's release is flagged `in_train`, since the recipes were tuned there
"""

import json

from src import common, datasets
from src.schema import Case, Dataset

SOURCE = common.SOURCES / "surge"
RELEASED = ("surveymatch", "frontiermatch", "consensusmatch")


class Surge(Dataset):
    NAME = "surge"
    SPLITS = ("test",)

    def build(self, split: str) -> list[Case]:
        if not (SOURCE / "arxiv_map.json").exists():
            common.graphiv("surge_map.py", "--source", str(SOURCE))
        with open(SOURCE / "surveys.json") as f:
            surveys = json.load(f)
        with open(SOURCE / "arxiv_map.json") as f:
            mapping = json.load(f)
        docs, places = mapping["doc"], mapping["survey"]
        tuned = {a for name in RELEASED for s in ("train", "val") for c in datasets.load(name).get(s) for a in c["anchors"]}
        cases = []
        for i, s in enumerate(surveys):
            if not places[i]:
                continue
            anchor, cutoff = places[i]
            cited = (docs.get(str(d)) for d in s["all_cites"])
            key = sorted({d[0] for d in cited if d and d[1] <= cutoff} - {anchor})
            if key:
                cases.append({"sample_id": f"surge{i:03d}", "query": s["survey_title"].strip(), "cutoff": cutoff, "anchors": [anchor], "key": key, "in_train": anchor in tuned})
        return cases


DATASET = Surge
