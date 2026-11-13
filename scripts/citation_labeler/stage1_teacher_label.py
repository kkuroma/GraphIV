"""
Label a sample of real citations with the winning prompt to make the student's training set

The sample is drawn with TABLESAMPLE, since ordering 181M rows at random is a full table scan
Only resolved citations are drawn, as an unresolved one has no target to reason about
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psycopg
from labels import parse_labels

from src.core.config import load_config
from src.core.llm import headers

SAMPLE = """
SELECT source_id, target_id, COALESCE(section, ''), sentence
FROM citations TABLESAMPLE SYSTEM (%s)
WHERE target_id IS NOT NULL AND sentence IS NOT NULL AND length(sentence) > 20
LIMIT %s
"""


class Teacher:
    """
    The winning prompt applied to real citations
    """

    def __init__(
        self, model: str, prompt_path: str, api: str, parallel: int, key: str = ""
    ) -> None:
        """
        Args:
            model: model name the endpoint answers to.
            prompt_path: the winning prompt written by the APE stage.
            api: OpenAI-compatible base URL.
            parallel: concurrent requests.
            key: the API key, empty for an endpoint that takes none.
        """
        self.model = model
        self.root = Path(prompt_path).parent
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.api = api.rstrip("/")
        self.key = key
        self.parallel = parallel

    def _chat(self, user: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        }
        req = urllib.request.Request(
            f"{self.api}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers(self.key),
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                body: dict[str, Any] = json.load(r)  # pyright: ignore[reportAny, reportExplicitAny]
            return str(body["choices"][0]["message"]["content"] or "")  # pyright: ignore[reportAny]
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError):
            return ""

    def label_one(self, section: str, sentence: str) -> tuple[str, float] | None:
        """
        Label a single citation

        The score is always 1.0 because the teacher is a prompt and reports no calibrated
        confidence; the student is what produces a real score.

        Args:
            section: the section heading the citation sits under.
            sentence: the sentence carrying the citation.

        Returns:
            The intent and its score, or None if the model answered nothing usable.
        """
        return self._batch([(section, sentence)])[0]

    def _batch(self, items: list[tuple[str, str]]) -> list[tuple[str, float] | None]:
        """
        Label one batch and align the answers back onto the inputs

        Args:
            items: (section, sentence) pairs.

        Returns:
            One (intent, score) per input, None where the model answered nothing usable.
        """
        listing = "\n".join(
            f"{i}. section: {s or '(unknown)'}\n   sentence: {t}" for i, (s, t) in enumerate(items)
        )
        user = (
            f"{listing}\n\nAnswer with one line per item, formatted "
            "`index: intent - the phrase that decided it`."
        )
        text = self._chat(user, max_tokens=len(items) * 48 + 256)
        got = parse_labels(text, len(items))
        if any(got) or len(items) == 1:
            return [(g[0], 1.0) if g else None for g in got]

        """
        A call that answers nothing is a whole batch lost, and it happens under load rather than
        on particular content, so the same items are retried in halves before being given up on.
        """
        half = len(items) // 2
        return self._batch(items[:half]) + self._batch(items[half:])

    def label(self, pg_dest: str, sample: int, batch: int) -> Path:
        """
        Draw a sample, label it, and write the training set

        Args:
            pg_dest: libpq connection string.
            sample: how many citations to label.
            batch: citations per endpoint call.

        Returns:
            The path of the written CSV.
        """
        percent = min(100.0, max(0.01, sample / 181_500_000 * 100 * 4))
        with psycopg.connect(pg_dest) as conn:
            rows: list[tuple[str, str, str, str]] = conn.execute(
                SAMPLE, (percent, sample)
            ).fetchall()
        print(f"[teacher] drew {len(rows):,} citations", flush=True)

        chunks = [rows[i : i + batch] for i in range(0, len(rows), batch)]
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            results = list(pool.map(lambda c: self._batch([(r[2], r[3]) for r in c]), chunks))
        labeled = [g for chunk in results for g in chunk]

        """
        An unanswered citation is dropped rather than written as background, since that label is
        the majority class and a silent default there is indistinguishable from a real one.
        """
        out = self.root / "teacher_labeled.csv"
        written = 0
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            _ = w.writerow(("source_id", "target_id", "section", "sentence", "intent", "score"))
            for (src, tgt, section, sentence), got in zip(rows, labeled, strict=True):
                if got is None:
                    continue
                _ = w.writerow((src, tgt, section, sentence, got[0], got[1]))
                written += 1

        counts = Counter(got[0] for got in labeled if got)
        print(f"[teacher] {len(labeled) - written:,} citations went unanswered", flush=True)
        print(f"[teacher] wrote {written:,} rows -> {out}", flush=True)
        print(f"[teacher] classes: {dict(counts.most_common())}", flush=True)
        return out


def main() -> None:
    """
    Read the config, then label a fresh sample with the winning prompt
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--config_path", default="config.toml")
    args = ap.parse_args()

    cfg = load_config(str(args.config_path))
    cl = cfg["citation_labeler"]
    root = str(cl["LABELER_PATH"])
    teacher = Teacher(
        model=str(cl["LABELER_APE_MODEL"]),
        prompt_path=str(Path(root) / "best.txt"),
        api=str(cfg["endpoints"]["LLM_ENDPOINT"]),
        parallel=int(cl["LABELER_APE_PARALLEL"]),
        key=str(cfg["endpoints"]["LLM_API_KEY"]),
    )
    _ = teacher.label(
        str(cfg["database"]["PG_DEST"]),
        sample=int(cl["LABELER_TEACHER_SAMPLE"]),
        batch=25,
    )


if __name__ == "__main__":
    main()
