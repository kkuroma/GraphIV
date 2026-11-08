"""
The BM25 term on a handful of documents, with no store and no graph
"""

import numpy as np

from src.retrieve.keyword import fold_word, get_bm25, get_query_words


def test_query_words_drop_stopwords_and_fold_plurals() -> None:
    assert get_query_words("What are the pull requests?") == ["pull", "request"]


def test_fold_word_keeps_double_s() -> None:
    assert fold_word("loss") == "loss"
    assert fold_word("papers") == "paper"


def test_hyphens_and_plurals_match() -> None:
    scores = get_bm25("pull request", ["Pull-Requests are reviewed", "nothing related"])
    assert scores[0] > 0
    assert scores[1] == 0


def test_rare_word_outweighs_a_word_the_pool_shares() -> None:
    documents = ["llm alpha", "llm beta", "llm gamma", "llm malicious"]
    scores = get_bm25("llm malicious", documents)
    assert int(np.argmax(scores)) == 3


def test_empty_question_and_empty_pool() -> None:
    assert get_bm25("", ["anything"]).tolist() == [0.0]
    assert get_bm25("anything", []).tolist() == []
