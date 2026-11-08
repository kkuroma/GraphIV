"""
The published benchmark files: their schema, and the split assigned per case from its anchors

A split is a hash of the anchors, so appending cases to a version never moves an existing one
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from case_miner import Sample

SPLITS = ("train", "val", "test")
# the one seed every published version is split with
SPLIT_SEED = 69420
FOLDERS = {
    "SurveyMatch": "surveymatch",
    "FrontierMatch": "frontiermatch",
    "ConsensusMatch": "consensusmatch",
}
HEADLINE_K = {"SurveyMatch": 1000, "FrontierMatch": 1000, "ConsensusMatch": 100}

SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("split", pa.string()),
        ("query_text", pa.string()),
        ("query_drafter", pa.string()),
        ("date_max", pa.date32()),
        ("anchors", pa.list_(pa.string())),
        ("anchor_titles", pa.list_(pa.string())),
        ("anchor_dates", pa.list_(pa.date32())),
        ("anchor_survey_probability", pa.list_(pa.float32())),
        ("citations", pa.list_(pa.string())),
        ("n_citations", pa.int32()),
        ("headline_k", pa.int32()),
        ("topic_l0", pa.int32()),
        ("topic_l1", pa.int32()),
        ("topic_l2", pa.int32()),
        ("topic_l0_name", pa.string()),
        ("topic_l1_name", pa.string()),
        ("topic_l2_name", pa.string()),
        ("split_group", pa.string()),
        ("gap_days", pa.int32()),
        ("cosine", pa.float32()),
    ]
)


def get_split_group(leaf: list[int]) -> str:
    """
    A case's parent topic, published beside its split

    Args:
        leaf: The three tree levels.

    Returns:
        The parent as `level_0.level_1`.
    """
    return f"{leaf[0]}.{leaf[1]}"


def assign_splits(samples: list[Sample], ratios: tuple[float, float, float]) -> list[str]:
    """
    One split per sample, from the seeded hash of its sorted anchors

    The hash maps each case to a point on [0, 1), and the shares cut that interval in split order.

    Args:
        samples: The mined samples.
        ratios: The train, val and test shares.

    Returns:
        The split of each sample, in the same order.
    """
    edges = [sum(ratios[: i + 1]) for i in range(len(SPLITS))]
    out: list[str] = []
    for s in samples:
        key = ",".join(sorted(s.anchors))
        digest = hashlib.sha1(f"{SPLIT_SEED}:{key}".encode()).digest()[:8]
        point = int.from_bytes(digest, "big") / 2**64
        out.append(next((sp for sp, e in zip(SPLITS, edges, strict=True) if point < e), SPLITS[-1]))
    return out


def write_split(rows: list[dict[str, object]], path: Path) -> None:
    """
    One split of one benchmark as a Parquet file

    Args:
        rows: The cases, one dict per row keyed by the schema's columns.
        path: Where the file lands.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
