"""
Scores a pool by the question's own words with BM25, the one signal that reads exact text

Rarity is measured inside the pool, so a word the whole pool shares carries almost nothing
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from numpy.typing import NDArray

K1 = 1.5
B = 0.75

WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""
    a an and are as at be been being but by can could did do does doing done each for from had has
    have how if in into is it its may might more most much must not of on one or our other out over
    per rather should so some such than that the their them then there these they this those
    through to under up us use used uses using very via was we were what when where whether which
    while who why will with within without would you your question
    """.split())


def fold_word(word: str) -> str:
    """
    Strips a plural, so a query's "requests" meets a title's "request"

    Args:
        word: A lowercase word.

    Returns:
        The word, without a trailing s when it looks like a plural.
    """
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def get_query_words(text: str) -> list[str]:
    """
    The words of a question worth matching, each once, in the order it gives them

    Args:
        text: The question.

    Returns:
        Folded words, stopwords and single characters dropped.
    """
    words = WORD.findall(text.lower())
    kept = [fold_word(w) for w in words if w not in STOPWORDS]
    return list(dict.fromkeys(w for w in kept if len(w) > 1))


def get_bm25(text: str, documents: list[str]) -> NDArray[np.float64]:
    """
    One BM25 score per document, with document frequency counted over these documents alone

    Hyphens and punctuation split words, so "pull-request" and "pull request" read the same.

    Args:
        text: The question.
        documents: Each paper's title and abstract.

    Returns:
        One raw score per document, zero where it shares no word with the question.
    """
    scores = np.zeros(len(documents))
    wanted = set(get_query_words(text))
    if not wanted or not documents:
        return scores
    counts: list[Counter[str]] = []
    lengths = np.zeros(len(documents))
    for seat, document in enumerate(documents):
        words = [fold_word(w) for w in WORD.findall(document.lower())]
        lengths[seat] = len(words)
        counts.append(Counter(w for w in words if w in wanted))
    frequency = Counter(word for count in counts for word in count)
    rarity = {word: math.log(1 + len(documents) / df) for word, df in frequency.items()}
    mean = float(lengths.mean()) or 1.0
    for seat, count in enumerate(counts):
        norm = K1 * (1 - B + B * lengths[seat] / mean)
        scores[seat] = sum(rarity[w] * c * (K1 + 1) / (c + norm) for w, c in count.items())
    return scores
