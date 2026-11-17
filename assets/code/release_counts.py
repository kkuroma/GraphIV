"""
Counts the release from the store for the dataset figures, under the export's own filters

The counts are cached in assets/data/dataset.json, so a redraw needs no store
A query missing from the cache is counted again from the store config.toml names
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, LiteralString

import psycopg
from style import DATA

from src.core.config import load_config

CACHE = DATA / "dataset.json"

# the filters scripts/export/stage1_export_parquet.py writes the release with
BOTH_ENDS: LiteralString = """
    FROM citations c
    JOIN papers s ON s.arxiv_id = c.source_id AND s.stage >= 3
    JOIN papers t ON t.arxiv_id = c.target_id AND t.stage >= 3
    WHERE c.target_id IS NOT NULL"""
QUERIES: dict[str, LiteralString] = {
    "papers": "SELECT count(*), max(date)::text FROM papers WHERE stage >= 3",
    "snapshot": "SELECT count(*) FROM papers",
    "years": """
        SELECT extract(year FROM date)::int, count(*) FROM papers WHERE stage >= 3
        GROUP BY 1 ORDER BY 1""",
    # 0.5 is where the survey labeler's probability is read as a review
    "reviews": """
        SELECT extract(year FROM date)::int, count(*) FROM papers
        WHERE stage >= 3 AND survey_probability >= 0.5
        GROUP BY 1 ORDER BY 1""",
    "cites": "SELECT extract(year FROM s.date)::int, c.intent, count(*)"
    + BOTH_ENDS
    + " GROUP BY 1, 2",
    "extracted": "SELECT count(*) FROM citations",
    "stance": "WITH counted AS (SELECT c.target_id, c.intent, count(DISTINCT c.source_id) AS n"
    + BOTH_ENDS
    + """ AND c.intent IN ('extends', 'compares', 'critiques') GROUP BY 1, 2),
        top AS (SELECT target_id FROM counted GROUP BY 1 ORDER BY sum(n) DESC LIMIT 10)
        SELECT counted.target_id, counted.intent, counted.n, p.title, p.date::text
        FROM counted JOIN top USING (target_id) JOIN papers p ON p.arxiv_id = counted.target_id""",
    "leaves": """
        SELECT t.level_0, t.level_1, t.level_2, n.name, count(*) FROM topic_tree t
        JOIN papers p ON p.arxiv_id = t.arxiv_id AND p.stage >= 3
        JOIN topic_names n
          ON n.level_0 = t.level_0 AND n.level_1 = t.level_1 AND n.level_2 = t.level_2
        GROUP BY 1, 2, 3, 4""",
    "upper": "SELECT level_0, level_1, name FROM topic_names WHERE level_2 IS NULL",
    "review_fields": """
        SELECT n.name, count(*) FILTER (WHERE p.survey_probability >= 0.5), count(*)
        FROM topic_tree t
        JOIN papers p ON p.arxiv_id = t.arxiv_id AND p.stage >= 3
        JOIN topic_names n ON n.level_0 = t.level_0 AND n.level_1 IS NULL
        GROUP BY 1""",
}


def load(config: Path) -> dict[str, Any]:
    """
    Counts whatever the cache does not hold yet, straight from the store

    Args:
        config: The config whose PG_DEST names the store.

    Returns:
        Each query's rows, keyed as in QUERIES.
    """
    cached = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    out = {name: cached[name] for name in QUERIES if name in cached}
    missing = [name for name in QUERIES if name not in out]
    if not missing:
        return out
    dsn = str(load_config(config)["database"]["PG_DEST"])
    with psycopg.connect(dsn) as conn:
        # counting distinct citing papers per target sorts in memory rather than on disk
        _ = conn.execute("SET work_mem = '1GB'")
        for name in missing:
            print(f"[dataset] {name}", flush=True)
            out[name] = [list(row) for row in conn.execute(QUERIES[name]).fetchall()]
    _ = CACHE.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out
