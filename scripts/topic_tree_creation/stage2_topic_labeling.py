"""
Name every node of the topic tree from the leaves upward and store one row per node

A parent is asked for three words fewer than its shortest child, so abstraction rises with the level
Siblings above the leaves are named in one call each, since a node named alone collides with them
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psycopg
from pydantic import BaseModel

from src.core.config import load_config
from src.core.llm import headers

MIN_CAP = 3
LEAF_CAP = 15
NUMBERED = re.compile(r"^\s*(\d+)\s*[.):]\s*(.+)$")


class Paper(BaseModel):
    """
    The two fields the namer reads off a paper
    """

    title: str
    abstract: str


def word_cap(children: list[str]) -> int:
    """
    The word budget for a node, three under its shortest child and never below the floor

    Args:
        children: the names already assigned to the node's children.

    Returns:
        The cap in words.
    """
    lengths = [len(c.split()) for c in children if c]
    return MIN_CAP if not lengths else max(MIN_CAP, min(lengths) - 3)


class TopicTreeLabeler:
    """
    A namer for one topic tree, holding the endpoint and the width of its request pool
    """

    def __init__(
        self, base_url: str, model: str, batch_size: int, n_papers: int, key: str = ""
    ) -> None:
        """
        Args:
            base_url: an OpenAI-compatible root, normally llama-router.
            model: the model name the endpoint answers to.
            batch_size: nodes named at once, which is also the pool width.
            n_papers: papers shown per leaf.
            key: the API key, empty for an endpoint that takes none.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = key
        self.batch_size = batch_size
        self.n_papers = n_papers
        self.fallbacks = 0

    def _complete(self, prompt: str, max_tokens: int) -> str:
        """
        Send one query, retrying a network or parse failure twice

        Reasoning is off because the model otherwise spends the whole budget on chain of thought
        and returns an empty completion.

        Args:
            prompt: the whole user turn.
            max_tokens: the completion budget.

        Returns:
            The reply, or an empty string once the attempts are spent.
        """
        body: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "reasoning_effort": "none",
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers(self.key),
        )
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    payload: dict[str, Any] = json.load(
                        r
                    )  # pyright: ignore[reportExplicitAny, reportAny]
                raw = str(payload["choices"][0]["message"]["content"])  # pyright: ignore[reportAny]
            except (urllib.error.URLError, OSError, KeyError, ValueError):  # noqa: PERF203
                continue
            if raw.strip():
                return raw
        return ""

    @staticmethod
    def _clean(raw: str) -> str:
        """
        Strip the model's decoration, keeping a name over its word cap whole

        Cutting it at the cap left names ending mid-phrase, like Algebraic Geometry And.

        Args:
            raw: one line of the reply.

        Returns:
            The bare name.
        """
        return " ".join(raw.split()).strip("\"'*`.,")

    def label_topics(self, papers: list[Paper]) -> str:
        """
        Fold a leaf's sample papers into one query and return the leaf's name

        Args:
            papers: the papers nearest the leaf centroid.

        Returns:
            The name, or an empty string if every attempt failed.
        """
        lines: list[str] = []
        for i, paper in enumerate(papers, 1):
            lines.append(f"{i}. {paper.title}")
            lines.append(f"   {' '.join(paper.abstract.split())[:700]}")
        prompt = (
            "Name the specific research topic these papers share.\n"
            f"Answer with a title-case noun phrase of at most {LEAF_CAP} words and nothing else.\n"
            "Be precise enough that a different cluster in the same field would need"
            " a different name.\n"
            "No preamble, no quotes, no trailing period.\n"
            "Do not use the words Papers, Research, Studies, Approaches or Methods.\n\n"
        ) + "\n".join(lines)
        return self._clean(self._complete(prompt, 60))

    def label_meta_topic(self, topics: list[str], cap: int) -> str:
        """
        Name the umbrella covering one node's children

        Args:
            topics: the children's names.
            cap: the word budget.

        Returns:
            The name, or an empty string if every attempt failed.
        """
        listed = "\n".join(f"- {t}" for t in topics if t)
        prompt = (
            "These are the subtopics of one parent node in a topic tree.\n"
            "Name the parent: the narrowest field that covers all of them.\n"
            f"Answer with a title-case noun phrase of at most {cap} words and nothing else.\n"
            "No preamble, no quotes, no trailing period.\n\n" + listed
        )
        return self._clean(self._complete(prompt, 40))

    def label_sibling_group(self, groups: list[list[str]], cap: int) -> list[str]:
        """
        Name a whole set of siblings in one call so no two of them land on the same field

        Falls back to naming each sibling on its own when the reply cannot be parsed, which
        restores the old behavior rather than losing the level.

        Args:
            groups: one child-name list per sibling, in output order.
            cap: the word budget for every name.

        Returns:
            One name per sibling, in input order.
        """
        blocks: list[str] = []
        for i, children in enumerate(groups, 1):
            listed = "\n".join(f"   - {c}" for c in children if c)
            blocks.append(f"CLUSTER {i}\n{listed}")
        prompt = (
            f"Below are {len(groups)} sibling clusters in a topic tree,"
            " each shown as its own subtopics.\n"
            "Name every cluster: the narrowest field that covers its subtopics.\n"
            f"Each name is a title-case noun phrase of at most {cap} words.\n"
            "The names must all differ from one another and must not be interchangeable.\n"
            f"Answer with exactly {len(groups)} lines, each `N. Name`, and nothing else.\n\n"
            + "\n\n".join(blocks)
        )
        reply = self._complete(prompt, 40 * len(groups))
        got: dict[int, str] = {}
        for line in reply.splitlines():
            hit = NUMBERED.match(line)
            if hit:
                got[int(hit.group(1))] = self._clean(hit.group(2))
        if len(got) == len(groups):
            return [got[i] for i in range(1, len(groups) + 1)]
        return [self.label_meta_topic(children, cap) for children in groups]

    def _named(self, jobs: list[str], run: Callable[[int], str], tag: str) -> list[str]:
        """
        Run one level's queries through the pool, substituting a placeholder for a failure

        Args:
            jobs: the placeholder for each node, used when the call comes back empty.
            run: the per-node callable producing a name.
            tag: the level's name in the progress line.

        Returns:
            One name per node, in input order.
        """
        with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            got = list(pool.map(run, range(len(jobs))))
        out: list[str] = []
        for name, spare in zip(got, jobs, strict=True):
            if name:
                out.append(name)
            else:
                out.append(spare)
                self.fallbacks += 1
        print(f"[stage2] named {len(out):,} {tag}", flush=True)
        return out

    def _sample(self, pg_dest: str) -> tuple[list[tuple[int, int, int]], list[list[Paper]]]:
        """
        The leaf list and the papers nearest each leaf's centroid

        Args:
            pg_dest: libpq connection string.

        Returns:
            The leaves in tree order and their sampled papers.

        Raises:
            SystemExit: if any paper is still waiting to be clustered.
        """
        with psycopg.connect(pg_dest) as conn:
            row = conn.execute("SELECT count(*) FROM topic_tree WHERE stage = 0").fetchone()
            waiting = int(row[0]) if row else 0
            if waiting:
                raise SystemExit(f"{waiting:,} papers still at stage 0, cluster them before naming")

            leaves: list[tuple[int, int, int]] = conn.execute(
                "SELECT level_0, level_1, level_2 FROM topic_tree WHERE stage = 1"
                " GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
            ).fetchall()
            print(f"[stage2] {len(leaves):,} leaves to name", flush=True)

            """
            The sample is pulled up front rather than inside a worker, so the pool holds one
            connection rather than one per thread.
            """
            samples: list[list[Paper]] = []
            for l0, l1, l2 in leaves:
                rows: list[tuple[str, str]] = conn.execute(
                    "WITH c AS ("
                    "  SELECT avg(embed_umap) AS mid FROM topic_tree"
                    "  WHERE level_0 = %s AND level_1 = %s AND level_2 = %s"
                    ")"
                    " SELECT p.title, p.abstract FROM topic_tree t"
                    " JOIN papers p ON p.arxiv_id = t.arxiv_id"
                    " CROSS JOIN c"
                    " WHERE t.level_0 = %s AND t.level_1 = %s AND t.level_2 = %s"
                    " AND p.title IS NOT NULL AND p.abstract IS NOT NULL"
                    " ORDER BY t.embed_umap <-> c.mid LIMIT %s",
                    (l0, l1, l2, l0, l1, l2, self.n_papers),
                ).fetchall()
                samples.append([Paper(title=t, abstract=a) for t, a in rows])
        print(f"[stage2] sampled {sum(len(s) for s in samples):,} papers", flush=True)
        return leaves, samples

    def label_tree(self, pg_dest: str) -> int:
        """
        Name every leaf from its papers, then every parent and grandparent from its children

        Args:
            pg_dest: libpq connection string.

        Returns:
            How many nodes were written.
        """
        leaves, samples = self._sample(pg_dest)
        leaf_names = self._named(
            [f"Cluster {a}/{b}/{c}" for a, b, c in leaves],
            lambda i: self.label_topics(samples[i]),
            "leaves",
        )

        mids = sorted({(a, b) for a, b, _c in leaves})
        mid_of = {key: i for i, key in enumerate(mids)}
        mid_children: list[list[str]] = [[] for _ in mids]
        for (a, b, _c), name in zip(leaves, leaf_names, strict=True):
            mid_children[mid_of[(a, b)]].append(name)

        tops = sorted({a for a, _b in mids})
        top_of = {key: i for i, key in enumerate(tops)}

        """
        Siblings are named together, so the model has to tell them apart rather than reaching
        for the same umbrella twice.
        """
        mid_names: list[str] = [""] * len(mids)
        under: list[list[int]] = [[] for _ in tops]
        for i, (a, _b) in enumerate(mids):
            under[top_of[a]].append(i)

        def name_group(g: int) -> list[str]:
            picked = [mid_children[i] for i in under[g]]
            return self.label_sibling_group(picked, word_cap([n for c in picked for n in c]))

        with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            grouped = list(pool.map(name_group, range(len(tops))))
        for g, names in enumerate(grouped):
            for i, name in zip(under[g], names, strict=True):
                mid_names[i] = name
        print(f"[stage2] named {len(mids):,} parents", flush=True)

        top_children: list[list[str]] = [[] for _ in tops]
        for (a, _b), name in zip(mids, mid_names, strict=True):
            top_children[top_of[a]].append(name)
        top_names = self.label_sibling_group(
            top_children, word_cap([n for group in top_children for n in group])
        )
        print(f"[stage2] named {len(top_names):,} grandparents", flush=True)

        return self._write(pg_dest, tops, top_names, mids, mid_names, leaves, leaf_names)

    def _write(
        self,
        pg_dest: str,
        tops: list[int],
        top_names: list[str],
        mids: list[tuple[int, int]],
        mid_names: list[str],
        leaves: list[tuple[int, int, int]],
        leaf_names: list[str],
    ) -> int:
        """
        Rebuild the table, since a refit changes what a given leaf id means

        Args:
            pg_dest: libpq connection string.
            tops: the level 0 labels.
            top_names: their names.
            mids: the level 0 and 1 pairs.
            mid_names: their names.
            leaves: the full triples.
            leaf_names: their names.

        Returns:
            How many nodes were written.
        """
        with psycopg.connect(pg_dest) as conn:
            _ = conn.execute("TRUNCATE topic_names")
            with conn.cursor().copy(
                "COPY topic_names (level_0, level_1, level_2, name) FROM STDIN"
            ) as cp:
                for top, name in zip(tops, top_names, strict=True):
                    cp.write_row((top, None, None, name))
                for (a, b), name in zip(mids, mid_names, strict=True):
                    cp.write_row((a, b, None, name))
                for (a, b, c), name in zip(leaves, leaf_names, strict=True):
                    cp.write_row((a, b, c, name))
            conn.commit()

        written = len(tops) + len(mids) + len(leaves)
        print(f"[stage2] wrote {written:,} nodes, {self.fallbacks:,} placeholders", flush=True)
        return written


def main() -> None:
    """
    Read the config and name the whole tree
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--config_path", default="config.toml")
    args = ap.parse_args()

    cfg = load_config(str(args.config_path))
    ep, tp = cfg["endpoints"], cfg["topics"]
    labeler = TopicTreeLabeler(
        base_url=str(ep["LLM_ENDPOINT"]),
        model=str(ep["LLM_NAME"]),
        batch_size=int(tp["TOPIC_BATCH"]),
        n_papers=int(tp["N_TOPIC_LABEL"]),
        key=str(ep["LLM_API_KEY"]),
    )
    _ = labeler.label_tree(str(cfg["database"]["PG_DEST"]))


if __name__ == "__main__":
    main()
