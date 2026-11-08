"""
The query vector, from the endpoint the documents were encoded with

Documents carry a prompt prefix, so a query missing its own lands elsewhere in the space
"""

from __future__ import annotations

import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

QUERY_PREFIX = "search_query: "
TIMEOUT = 180
# the model's window in tokens, less the two special tokens
WINDOW = 510
# no text this short can overflow the window, so it skips the tokenizer
SAFE_CHARS = 400
MAX_CHARS = 4000
FITTERS = 8


def embed(api: str, model: str, text: str) -> list[float]:
    """
    One query vector, unit normalized

    Args:
        api: The embedding server's base url.
        model: Which model to ask.
        text: The question.

    Returns:
        The vector.
    """
    request = urllib.request.Request(
        f"{api.rstrip('/')}/v1/embeddings",
        data=json.dumps({"model": model, "input": [f"{QUERY_PREFIX}{text}"]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        raw = [float(x) for x in json.load(response)["data"][0]["embedding"]]
    return unit(raw)


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


def post(url: str, payload: dict[str, Any]) -> Any:
    """
    One json request to the embedding server

    Args:
        url: The full endpoint url.
        payload: The request body.

    Returns:
        The decoded reply.
    """
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def fit(api: str, text: str) -> str:
    """
    Cuts one document to the model's window using the server's own tokenizer

    Args:
        api: The embedding server's base url.
        text: The prefixed document.

    Returns:
        The document, shortened only when it overflows.
    """
    if len(text) <= SAFE_CHARS:
        return text
    base = api.rstrip("/")
    tokens = post(f"{base}/tokenize", {"content": text, "add_special": False})["tokens"]
    if len(tokens) <= WINDOW:
        return text
    return str(post(f"{base}/detokenize", {"tokens": tokens[:WINDOW]})["content"])


def embed_documents(api: str, model: str, texts: list[str], prefix: str) -> list[list[float]]:
    """
    Many document vectors in one call, each cut to the window first

    Args:
        api: The embedding server's base url.
        model: Which model to ask.
        texts: The documents, without their prefix.
        prefix: The document prompt the corpus was encoded with.

    Returns:
        One unit vector per document, in order.
    """
    if not texts:
        return []
    with ThreadPoolExecutor(max_workers=FITTERS) as pool:
        held = list(pool.map(lambda t: fit(api, prefix + t[:MAX_CHARS]), texts))
    rows = post(f"{api.rstrip('/')}/v1/embeddings", {"model": model, "input": held})["data"]
    return [
        unit([float(x) for x in r["embedding"]]) for r in sorted(rows, key=lambda r: r["index"])
    ]
