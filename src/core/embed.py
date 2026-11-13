"""
Query and document vectors, from the endpoint the corpus was encoded with

Documents carry a prompt prefix, so a query missing its own lands elsewhere in the space
A document is cut to the window with the server's own tokenizer, so the host must be llama-server
"""

from __future__ import annotations

import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from src.core.llm import headers

TIMEOUT = 180
# the window loses the two special tokens the server adds
SPECIAL_TOKENS = 2
# no text this short can overflow the window, so it skips the tokenizer
SAFE_CHARS = 400
MAX_CHARS = 4000
FITTERS = 8


@dataclass(frozen=True)
class EmbeddingEndpoint:
    """
    The embedding server and everything a request to it needs, from `[endpoints]`
    """

    url: str
    name: str
    key: str = ""
    query_prefix: str = ""
    document_prefix: str = ""
    max_seq: int = 512

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EmbeddingEndpoint:
        """
        Reads the endpoint from a parsed config

        Args:
            config: The parsed config.toml.

        Returns:
            The endpoint.
        """
        ep = config["endpoints"]
        return cls(
            url=str(ep["EMBEDDING_ENDPOINT"]).rstrip("/"),
            name=str(ep["EMBEDDING_NAME"]),
            key=str(ep["EMBEDDING_API_KEY"]),
            query_prefix=str(ep["EMBEDDING_QUERY_PREFIX"]),
            document_prefix=str(ep["EMBEDDING_DOCUMENT_PREFIX"]),
            max_seq=int(ep["EMBEDDING_MAX_SEQ"]),
        )


def embed(endpoint: EmbeddingEndpoint, text: str) -> list[float]:
    """
    One query vector, unit normalized

    Args:
        endpoint: The embedding server.
        text: The question.

    Returns:
        The vector.
    """
    body = {"model": endpoint.name, "input": [f"{endpoint.query_prefix}{text}"]}
    raw = post(endpoint, "/v1/embeddings", body)["data"][0]["embedding"]
    return unit([float(x) for x in raw])


def unit(raw: list[float]) -> list[float]:
    """
    One vector scaled to length one

    Args:
        raw: The vector.

    Returns:
        The unit normalized vector.
    """
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def post(endpoint: EmbeddingEndpoint, route: str, payload: dict[str, Any]) -> Any:
    """
    One json request to the embedding server

    Args:
        endpoint: The embedding server.
        route: The path under its url.
        payload: The request body.

    Returns:
        The decoded reply.
    """
    request = urllib.request.Request(
        f"{endpoint.url}{route}", json.dumps(payload).encode(), headers(endpoint.key)
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def fit(endpoint: EmbeddingEndpoint, text: str) -> str:
    """
    Cuts one document to the model's window using the server's own tokenizer

    Args:
        endpoint: The embedding server.
        text: The prefixed document.

    Returns:
        The document, shortened only when it overflows.
    """
    if len(text) <= SAFE_CHARS:
        return text
    window = endpoint.max_seq - SPECIAL_TOKENS
    tokens = post(endpoint, "/tokenize", {"content": text, "add_special": False})["tokens"]
    if len(tokens) <= window:
        return text
    keep = window
    while True:
        cut = str(post(endpoint, "/detokenize", {"tokens": tokens[:keep]})["content"])
        # detokenized text can tokenize longer than the tokens it came from
        again = post(endpoint, "/tokenize", {"content": cut, "add_special": False})["tokens"]
        if len(again) <= window:
            return cut
        keep -= 1


def embed_documents(endpoint: EmbeddingEndpoint, texts: list[str]) -> list[list[float]]:
    """
    Many document vectors in one call, each prefixed and cut to the window first

    Args:
        endpoint: The embedding server.
        texts: The documents, without their prefix.

    Returns:
        One unit vector per document, in order.
    """
    if not texts:
        return []
    with ThreadPoolExecutor(max_workers=FITTERS) as pool:
        held = list(
            pool.map(lambda t: fit(endpoint, endpoint.document_prefix + t[:MAX_CHARS]), texts)
        )
    rows = post(endpoint, "/v1/embeddings", {"model": endpoint.name, "input": held})["data"]
    return [
        unit([float(x) for x in r["embedding"]]) for r in sorted(rows, key=lambda r: r["index"])
    ]
