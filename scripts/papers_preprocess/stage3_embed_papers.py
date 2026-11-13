"""
Embed the title and abstract of every stage 2 paper into embed_full

The document is `Title: ...\\nAbstract: ...` because the existing corpus vectors were built that way
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import psycopg

from src.core.config import load_config
from src.core.embed import EmbeddingEndpoint, embed_documents


class Embedder:
    """
    The embedding endpoint, plus the pass that fills the vector column
    """

    def __init__(self, endpoint: EmbeddingEndpoint, parallel: int = 1) -> None:
        """
        Args:
            endpoint: the embedding server, its document prefix and its window.
            parallel: batches in flight against the endpoint at once.
        """
        self.endpoint = endpoint
        self.parallel = parallel

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of documents through one endpoint call, cut to the window the way a read is

        Args:
            texts: composed documents, without the prompt prefix.

        Returns:
            One L2-normalized vector per input, in input order.
        """
        return embed_documents(self.endpoint, texts)

    def embed_to_db(self, pg_dest: str, batch: int) -> int:
        """
        Embed every stage 2 paper and advance it to stage 3

        Args:
            pg_dest: libpq connection string.
            batch: documents per endpoint call.

        Returns:
            How many papers were embedded.
        """
        with psycopg.connect(pg_dest) as conn:
            rows: list[tuple[str, str, str]] = conn.execute(
                "SELECT arxiv_id, title, abstract FROM papers WHERE stage = 2"
            ).fetchall()
        print(f"[stage3] {len(rows):,} papers to embed", flush=True)
        if not rows:
            return 0

        done = 0
        with psycopg.connect(pg_dest, autocommit=True) as conn:
            _ = conn.execute("CREATE TEMP TABLE t_emb (arxiv_id text PRIMARY KEY, v text)")
            chunks = [rows[i : i + batch] for i in range(0, len(rows), batch)]
            """
            Batches are independent, so `parallel` of them are in flight at once.
            The COPY stays on the single connection, so writes are serialized behind the fetches.
            """
            with ThreadPoolExecutor(max_workers=self.parallel) as pool:
                for group in range(0, len(chunks), self.parallel):
                    wave = chunks[group : group + self.parallel]
                    docs = [
                        [f"Title: {t or ''}\nAbstract: {a or ''}" for _, t, a in c] for c in wave
                    ]
                    for chunk, vecs in zip(wave, pool.map(self.embed_many, docs), strict=True):
                        with conn.cursor().copy("COPY t_emb (arxiv_id, v) FROM STDIN") as cp:
                            for (arxiv_id, _t, _a), v in zip(chunk, vecs, strict=True):
                                cp.write_row(
                                    (arxiv_id, "[" + ",".join(f"{x:.5g}" for x in v) + "]")
                                )
                        done += len(chunk)
                    """
                    Each wave lands before the next starts, so a kill costs one wave rather
                    than the whole pass.
                    """
                    _ = conn.execute(
                        "UPDATE papers p SET embed_full = t.v::halfvec, stage = 3"
                        " FROM t_emb t WHERE p.arxiv_id = t.arxiv_id"
                    )
                    _ = conn.execute("TRUNCATE t_emb")
                    print(f"[stage3]   {done:,}/{len(rows):,}", flush=True)
        return done


def main() -> None:
    """
    Read the config, then embed every paper waiting at stage 2
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--config_path", default="config.toml")
    args = ap.parse_args()

    cfg = load_config(str(args.config_path))
    ep = cfg["endpoints"]
    dsn = str(cfg["database"]["PG_DEST"])
    emb = Embedder(EmbeddingEndpoint.from_config(cfg), parallel=int(ep["EMBEDDING_PARALLEL"]))
    _ = emb.embed_to_db(dsn, batch=int(ep["EMBEDDING_BATCH"]))


if __name__ == "__main__":
    main()
