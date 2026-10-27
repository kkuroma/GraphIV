"""
The three benchmarks GraphIV publishes, read from the Parquet release copied into sources/

Each folder of the release holds train, val and test, one case a row
"""

import pyarrow.parquet as pq

from src import common
from src.schema import Case, Dataset

RELEASE = common.SOURCES / "graphiv-benchmarks-2026-10-22"


class ReleaseDataset(Dataset):
    """
    One folder of the release, NAME being the folder
    """

    SPLITS = ("train", "val", "test")

    def build(self, split: str) -> list[Case]:
        rows = pq.read_table(RELEASE / self.NAME / f"{split}.parquet").to_pylist()
        return [
            {
                "sample_id": str(r["case_id"]),
                "query": str(r["query_text"]),
                "cutoff": r["date_max"].isoformat(),
                "anchors": list(r["anchors"]),
                "key": list(r["citations"]),
            }
            for r in rows
        ]
