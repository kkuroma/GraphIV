"""
Matches SurGE's surveys and the papers they cite to arXiv ids in the store by normalized title

SurGE's corpus carries no arXiv id, so the title is the one field both sides share, and a shared title goes to the lowest id
Writes arxiv_map.json beside the source, each SurGE doc id and each survey's position to its arXiv id and date
"""

import argparse
import json
import re
from pathlib import Path

import psycopg

from src.core.config import load_config

ROOT = Path(__file__).resolve().parents[3]


def norm(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--source", required=True)
    source = Path(ap.parse_args().source)
    dsn = str(load_config(str(ROOT / "config.toml"))["database"]["PG_DEST"])
    with open(source / "surveys.json") as f:
        surveys = json.load(f)
    need = {int(c) for s in surveys for c in s["all_cites"]}
    with open(source / "corpus.json") as f:
        corpus = {int(p["doc_id"]): p for p in json.load(f) if int(p["doc_id"]) in need}
    want = {norm(p["Title"]) for p in corpus.values()} | {norm(s["survey_title"]) for s in surveys}
    hit: dict[str, tuple[str, str]] = {}
    with psycopg.connect(dsn) as conn:
        for arxiv_id, title, day in conn.execute("SELECT arxiv_id, title, date FROM papers ORDER BY arxiv_id"):
            key = norm(title)
            if key in want:
                hit.setdefault(key, (arxiv_id, str(day)))
    docs = {i: hit.get(norm(p["Title"])) for i, p in corpus.items()}
    places = [hit.get(norm(s["survey_title"])) for s in surveys]
    with open(source / "arxiv_map.json", "w") as f:
        json.dump({"doc": {i: v for i, v in docs.items() if v}, "survey": places}, f)
    print("cited", len(need), "mapped", sum(1 for v in docs.values() if v), "surveys mapped", sum(1 for v in places if v), "of", len(surveys), flush=True)


if __name__ == "__main__":
    main()
