"""
Writes the one question a set of reviews would have been written to answer

Naming a system in the question would hand back part of the answer key, so the prompt forbids it
Every completion is cached by the hash of its input, which freezes a question once it exists
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

from src.core.llm import headers

LOG = logging.getLogger(__name__)

ABSTRACT_CHARS = 1400

PROMPT = """You are given {n} literature review{s} on one research topic.

Write the single search question a researcher would have typed to find the papers {these} \
review{s2} cite.

Rules:
- One question, 15 to 40 words, no preamble and no quotes.
- Name the research problem and its scope, the way a person asks for reading material.
- Do not name any specific system, model, dataset, benchmark or author.
- Do not use the words survey, review, overview or literature.

{body}

Question:"""


class GemmaWriter:
    """
    Asks the local router for one question, and remembers what it said
    """

    def __init__(self, api: str, model: str, cache: str | Path, key: str = "") -> None:
        """
        Holds the endpoint and where a written question is kept

        Args:
            api: The OpenAI-compatible base url.
            model: Which model to ask.
            cache: Directory the questions are written into.
            key: The API key, empty for an endpoint that takes none.
        """
        self.api = api.rstrip("/")
        self.model = model
        self.key = key
        self.cache = Path(cache)

    def _complete(self, prompt: str, max_tokens: int = 160) -> str:
        """
        One completion, retried, with reasoning off so the budget goes to the answer

        Args:
            prompt: The user message.
            max_tokens: The completion budget.

        Returns:
            The reply, empty when every attempt failed.
        """
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "reasoning_effort": "none",
        }
        request = urllib.request.Request(
            f"{self.api}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers(self.key),
        )
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = json.load(response)
                reply = str(payload["choices"][0]["message"]["content"] or "")
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                KeyError,
                ValueError,
            ):  # noqa: PERF203
                continue
            if reply.strip():
                return reply
        LOG.warning("the router returned nothing after three attempts")
        return ""

    def write(self, titles: list[str], abstracts: list[str]) -> str:
        """
        The question these reviews answer, read from the cache when it is already written

        Args:
            titles: The review titles.
            abstracts: Their abstracts, in the same order.

        Returns:
            The question, empty when the router gave nothing and the caller should fall back.
        """
        self.cache.mkdir(parents=True, exist_ok=True)
        raw = json.dumps([titles, abstracts])
        path = self.cache / f"{hashlib.sha1(raw.encode()).hexdigest()[:12]}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

        count = len(titles)
        body = "\n\n".join(
            f"Title: {t}\nAbstract: {(a or '')[:ABSTRACT_CHARS]}"
            for t, a in zip(titles, abstracts, strict=True)
        )
        prompt = PROMPT.format(
            n=count,
            s="" if count == 1 else "s",
            s2="s" if count == 1 else "",
            these="this" if count == 1 else "these",
            body=body,
        )
        written = clean_reply(self._complete(prompt))
        if written:
            _ = path.write_text(written, encoding="utf-8")
        return written


def clean_reply(reply: str) -> str:
    """
    The first usable line of a completion

    Args:
        reply: The raw reply.

    Returns:
        One question, stripped of quoting and numbering, empty when nothing survives.
    """
    for line in reply.splitlines():
        got = line.strip().strip('"').strip()
        got = re.sub(r"^(question|query)\s*[:.-]\s*", "", got, flags=re.IGNORECASE)
        got = re.sub(r"^[-*\d.)\s]+", "", got)
        if len(got) >= 25:
            return got
    return ""
