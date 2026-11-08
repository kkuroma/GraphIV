"""
Mines benchmark samples out of the store, deterministically from a seed

A candidate is kept when the hash of its id under the seed falls below the sample rate
Deciding per candidate rather than in order means raising the cap grows the set
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import psycopg

Row = tuple[Any, ...]

LOG = logging.getLogger(__name__)

STAGE = """
SET work_mem = '4GB'
"""
POOL = """
CREATE TEMP TABLE sv AS
SELECT p.arxiv_id, p.title, p.abstract, p.date, p.embed_full,
       t.level_0, t.level_1, t.level_2
FROM papers p JOIN topic_tree t USING (arxiv_id)
WHERE p.survey_probability >= %s AND p.stage = 4 AND t.stage = 1 AND t.level_2 IS NOT NULL
  AND p.date <= %s
"""
REFS = """
CREATE TEMP TABLE rr AS
SELECT DISTINCT c.source_id, c.target_id, t.date AS target_date
FROM citations c JOIN sv ON sv.arxiv_id = c.source_id
JOIN papers t ON t.arxiv_id = c.target_id
WHERE t.stage >= 3 AND t.date < sv.date
"""
USABLE = """
CREATE TEMP TABLE usable AS
SELECT source_id FROM rr GROUP BY 1 HAVING count(*) >= %s
"""


@dataclass(frozen=True)
class Sample:
    """
    One benchmark question, self contained so a runner reads nothing else
    """

    metric: str
    sample_id: str
    anchors: list[str]
    titles: list[str]
    abstracts: list[str]
    leaf: list[int]
    cutoff: str
    key: list[str]
    query: str = ""
    extra: dict[str, float] = field(default_factory=dict)


def keep(seed: int, sample_id: str, rate: float) -> bool:
    """
    Whether a candidate survives the seeded draw

    Hashing the id rather than drawing in order means the same candidate decides the same way
    however many other candidates there are.

    Args:
        seed: The run seed.
        sample_id: The candidate's identity.
        rate: The share of candidates to keep.

    Returns:
        True when the candidate is in.
    """
    if rate >= 1.0:
        return True
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode()).digest()[:8]
    return int.from_bytes(digest, "big") / float(1 << 64) < rate


def shuffle_key(seed: int, sample_id: str) -> str:
    """
    The order candidates are considered in, which is stable under the seed

    Args:
        seed: The run seed.
        sample_id: The candidate's identity.

    Returns:
        A sortable digest.
    """
    return hashlib.sha1(f"{seed}:order:{sample_id}".encode()).hexdigest()


class CaseMiner:
    """
    Reads the store once and hands out samples for any of the three benchmarks
    """

    def __init__(
        self,
        pg_dest: str,
        *,
        seed: int = 0,
        max_samples: int = 300,
        sample_rate: float = 1.0,
        survey_threshold: float = 0.95,
        min_references: int = 25,
        max_date: date | None = None,
    ) -> None:
        """
        Holds the settings that decide which samples exist at all

        Args:
            pg_dest: libpq connection string.
            seed: Fixes which candidates are kept and in what order.
            max_samples: The most samples to return per benchmark.
            sample_rate: The share of candidates considered, applied before the cap.
            survey_threshold: The classifier floor a paper must clear to count as a review.
            min_references: How many resolved references a review needs to be usable.
            max_date: The last day a review may be published, today when unset.
        """
        self.pg_dest = pg_dest
        self.seed = seed
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.survey_threshold = survey_threshold
        self.min_references = min_references
        self.max_date = max_date or date.today()

    def _stage(self, conn: psycopg.Connection) -> None:
        """
        Builds the review pool, its resolved references and the usable subset

        Args:
            conn: An open connection.
        """
        _ = conn.execute(STAGE)
        _ = conn.execute(POOL, (self.survey_threshold, self.max_date))
        _ = conn.execute(REFS)
        _ = conn.execute("CREATE INDEX ON rr (source_id)")
        _ = conn.execute("CREATE INDEX ON sv (arxiv_id)")
        _ = conn.execute(USABLE, (self.min_references,))
        _ = conn.execute("CREATE INDEX ON usable (source_id)")
        pool = conn.execute("SELECT count(*) FROM sv").fetchone()
        usable = conn.execute("SELECT count(*) FROM usable").fetchone()
        LOG.info(
            "staged %s reviews at threshold %.2f, %s with %d or more references",
            f"{pool[0]:,}" if pool else "?",
            self.survey_threshold,
            f"{usable[0]:,}" if usable else "?",
            self.min_references,
        )

    def mine(self, metric: str, **kwargs: float) -> list[Sample]:
        """
        Every sample of one benchmark, in seeded order and capped

        Args:
            metric: SurveyMatch, FrontierMatch or ConsensusMatch.
            kwargs: The per-metric floors, which the caller reads out of config.

        Returns:
            At most `max_samples` samples.
        """
        with psycopg.connect(self.pg_dest) as conn:
            self._stage(conn)
            if metric == "SurveyMatch":
                out = self._mine_single(conn, metric)
            elif metric == "FrontierMatch":
                out = self._mine_pairs(conn, **kwargs)
            elif metric == "ConsensusMatch":
                out = self._mine_triples(conn, **kwargs)
            else:
                raise ValueError(f"unknown metric {metric}")
        LOG.info("%s: %d samples, seed %d", metric, len(out), self.seed)
        return out

    def _mine_single(self, conn: psycopg.Connection, metric: str) -> list[Sample]:
        """
        One sample per review, scored against its own bibliography

        Args:
            conn: An open connection with the temp tables staged.
            metric: Which benchmark the samples belong to.

        Returns:
            The samples.
        """
        rows = conn.execute(
            "SELECT s.arxiv_id, s.title, s.abstract, s.date, s.level_0, s.level_1, s.level_2"
            " FROM sv s JOIN usable u ON u.source_id = s.arxiv_id"
        ).fetchall()
        chosen = self._draw([(str(r[0]), r) for r in rows])
        keys = self._keys(conn, [str(r[0]) for _i, r in chosen])
        return [
            Sample(
                metric=metric,
                sample_id=str(r[0]),
                anchors=[str(r[0])],
                titles=[str(r[1])],
                abstracts=[str(r[2] or "")],
                leaf=[int(r[4]), int(r[5]), int(r[6])],
                cutoff=r[3].isoformat(),
                key=sorted(keys[str(r[0])]),
            )
            for _i, r in chosen
        ]

    def _mine_pairs(
        self,
        conn: psycopg.Connection,
        gap_days: float = 190,
        cosine: float = 0.80,
        window_days: float = 730,
        min_key: float = 10,
    ) -> list[Sample]:
        """
        One sample per pair of reviews written at the same moment on the same leaf

        The key is what the later review added, so the cap is the later date and every key paper
        is retrievable.

        Args:
            conn: An open connection with the temp tables staged.
            gap_days: The widest gap between the two.
            cosine: How similar the two must be.
            window_days: How far before the earlier review a key paper may be published.
            min_key: The smallest key that makes a usable sample.

        Returns:
            The samples, no review appearing twice.
        """
        rows = conn.execute(
            "SELECT a.arxiv_id, b.arxiv_id, a.title, a.abstract, a.date, b.date,"
            "       a.level_0, a.level_1, a.level_2, 1 - (a.embed_full <=> b.embed_full)"
            " FROM sv a JOIN usable ua ON ua.source_id = a.arxiv_id"
            " JOIN sv b ON b.level_0 = a.level_0 AND b.level_1 = a.level_1"
            "          AND b.level_2 = a.level_2 AND b.arxiv_id <> a.arxiv_id"
            " JOIN usable ub ON ub.source_id = b.arxiv_id"
            " WHERE b.date > a.date AND b.date <= a.date + %s"
            "   AND 1 - (a.embed_full <=> b.embed_full) >= %s",
            (int(gap_days), cosine),
        ).fetchall()

        out: list[Sample] = []
        seen: set[str] = set()
        for _digest, r in self._draw([(f"{r[0]}_{r[1]}", r) for r in rows], cap=False):
            early, late = str(r[0]), str(r[1])
            if early in seen or late in seen:
                continue
            key = conn.execute(
                "SELECT rb.target_id FROM rr rb"
                " WHERE rb.source_id = %s AND rb.target_date >= %s AND rb.target_date <= %s"
                "   AND NOT EXISTS (SELECT 1 FROM rr ra WHERE ra.source_id = %s"
                "                   AND ra.target_id = rb.target_id)",
                (late, r[4] - timedelta(days=int(window_days)), r[5], early),
            ).fetchall()
            if len(key) < min_key:
                continue
            seen.update((early, late))
            out.append(
                Sample(
                    metric="FrontierMatch",
                    sample_id=f"{early}_{late}",
                    anchors=[early, late],
                    titles=[str(r[2])],
                    abstracts=[str(r[3] or "")],
                    leaf=[int(r[6]), int(r[7]), int(r[8])],
                    cutoff=r[5].isoformat(),
                    key=sorted(str(k[0]) for k in key),
                    extra={"cosine": float(r[9]), "gap_days": float((r[5] - r[4]).days)},
                )
            )
            if len(out) >= self.max_samples:
                break
        return out

    def _mine_triples(
        self,
        conn: psycopg.Connection,
        gap_days: float = 190,
        cosine: float = 0.75,
        min_key: float = 5,
    ) -> list[Sample]:
        """
        One sample per three reviews of the same leaf written inside one span

        The key is what all three cite, which is the part of a topic its reviewers agree on.

        Args:
            conn: An open connection with the temp tables staged.
            gap_days: The widest span the three may cover.
            cosine: How similar every pair must be.
            min_key: The smallest key that makes a usable sample.

        Returns:
            The samples, no review appearing twice.
        """
        _ = conn.execute(
            "CREATE TEMP TABLE e AS"
            " SELECT a.arxiv_id AS a, b.arxiv_id AS b, a.date AS da, b.date AS db,"
            "        a.level_0, a.level_1, a.level_2"
            " FROM sv a JOIN usable ua ON ua.source_id = a.arxiv_id"
            " JOIN sv b ON b.level_0 = a.level_0 AND b.level_1 = a.level_1"
            "          AND b.level_2 = a.level_2 AND a.arxiv_id < b.arxiv_id"
            " JOIN usable ub ON ub.source_id = b.arxiv_id"
            " WHERE abs(b.date - a.date) <= %s AND 1 - (a.embed_full <=> b.embed_full) >= %s",
            (int(gap_days), cosine),
        )
        _ = conn.execute("CREATE INDEX ON e (a)")
        _ = conn.execute("CREATE INDEX ON e (b)")
        rows = conn.execute(
            "SELECT e1.a, e1.b, e2.b, least(e1.da, e1.db, e2.db),"
            "       e1.level_0, e1.level_1, e1.level_2"
            " FROM e e1 JOIN e e2 ON e2.a = e1.a AND e2.b > e1.b"
            "           JOIN e e3 ON e3.a = e1.b AND e3.b = e2.b"
            " WHERE greatest(e1.da, e1.db, e2.db) - least(e1.da, e1.db, e2.db) <= %s",
            (int(gap_days),),
        ).fetchall()

        out: list[Sample] = []
        seen: set[str] = set()
        for _digest, r in self._draw([("_".join(map(str, r[:3])), r) for r in rows], cap=False):
            trio = [str(r[0]), str(r[1]), str(r[2])]
            if seen.intersection(trio):
                continue
            key = conn.execute(
                "SELECT r1.target_id FROM rr r1"
                " JOIN rr r2 ON r2.source_id = %s AND r2.target_id = r1.target_id"
                " JOIN rr r3 ON r3.source_id = %s AND r3.target_id = r1.target_id"
                " WHERE r1.source_id = %s AND r1.target_date <= %s",
                (trio[1], trio[2], trio[0], r[3]),
            ).fetchall()
            if len(key) < min_key:
                continue
            meta = {
                str(m[0]): (str(m[1]), str(m[2] or ""))
                for m in conn.execute(
                    "SELECT arxiv_id, title, abstract FROM sv WHERE arxiv_id = ANY(%s)", (trio,)
                ).fetchall()
            }
            seen.update(trio)
            out.append(
                Sample(
                    metric="ConsensusMatch",
                    sample_id="_".join(trio),
                    anchors=trio,
                    titles=[meta[a][0] for a in trio],
                    abstracts=[meta[a][1] for a in trio],
                    leaf=[int(r[4]), int(r[5]), int(r[6])],
                    cutoff=r[3].isoformat(),
                    key=sorted(str(k[0]) for k in key),
                )
            )
            if len(out) >= self.max_samples:
                break
        return out

    def _draw(self, candidates: list[tuple[str, Row]], cap: bool = True) -> list[tuple[str, Row]]:
        """
        The seeded subset of the candidates, in seeded order

        Args:
            candidates: Pairs of identity and row.
            cap: Whether to apply the sample cap here, which a metric that filters later turns off.

        Returns:
            The kept candidates, ordered by their shuffle digest.
        """
        kept = [
            (shuffle_key(self.seed, name), row)
            for name, row in candidates
            if keep(self.seed, name, self.sample_rate)
        ]
        kept.sort(key=lambda pair: pair[0])
        LOG.info(
            "drew %d of %d candidates at rate %.3f", len(kept), len(candidates), self.sample_rate
        )
        return kept[: self.max_samples] if cap else kept

    def _keys(self, conn: psycopg.Connection, ids: list[str]) -> dict[str, set[str]]:
        """
        The resolved bibliography of each review

        Args:
            conn: An open connection with the temp tables staged.
            ids: The reviews to read.

        Returns:
            One reference set per review.
        """
        out: dict[str, set[str]] = {a: set() for a in ids}
        for source, target in conn.execute(
            "SELECT source_id, target_id FROM rr WHERE source_id = ANY(%s)", (ids,)
        ).fetchall():
            out[str(source)].add(str(target))
        return out
