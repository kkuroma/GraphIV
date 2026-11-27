"""
The reader's checks on a schema and on each typed answer, with no model and no store
"""

import pytest

from src.mcp.read import SchemaField, held_schema, picked, schema_of, supported
from src.mcp.runs import Run


def test_schema_limits_name_the_broken_field():
    good = {"fields": [{"name": "uses_rag", "type": "bool", "ask": "Does it retrieve?"}]}
    assert schema_of(good).fields[0].name == "uses_rag"
    with pytest.raises(ValueError, match="Bad Name"):
        schema_of({"fields": [{"name": "Bad Name", "type": "bool", "ask": "x"}]})
    with pytest.raises(ValueError, match="2 to 12 options"):
        schema_of({"fields": [{"name": "kind", "type": "choice", "ask": "x", "options": ["a"]}]})


def test_an_answer_stands_only_when_its_sentence_backs_it():
    sentence = (
        "Our model reaches 71.3 nDCG@10 on BEIR with a BERT-base backbone, at 160,000 queries."
    )
    number = SchemaField(name="n", type="float", ask="x")
    assert supported(number, 71.3, sentence) == "71.3"
    assert supported(number, 160000, sentence) == "160000"
    assert supported(number, 72.0, sentence) is None
    assert supported(number, "71.3", sentence) == "71.3"
    assert supported(number, "71.3 points", sentence) is None
    phrase = SchemaField(name="b", type="text", ask="x")
    assert supported(phrase, "BERT-base", sentence) == "BERT-base"
    assert supported(phrase, "RoBERTa", sentence) is None
    choice = SchemaField(name="c", type="choice", ask="x", options=["dense", "sparse"])
    assert supported(choice, "Dense", sentence) == "dense"
    assert supported(SchemaField(name="u", type="bool", ask="x"), "yes", sentence) is None


def test_picks_drop_unshown_and_repeated_numbers():
    assert picked(["s2", "s2", "s9", 1], 5) == ([2, 1], 1)


def test_a_table_id_reuses_its_fields_and_an_unknown_one_is_refused():
    run = Run(session="s-1")
    run.add_row(
        {
            "columns": [
                {"name": "kind", "kind": "choice", "description": "How?", "options": ["a", "b"]}
            ]
        },
        {"arxiv_id": "1", "method": "m", "cells": {}},
    )
    reused = held_schema(run, "T1")
    assert isinstance(reused, dict)
    assert schema_of(reused).fields[0].ask == "How?"
    assert held_schema(run, '"T1"') == reused
    with pytest.raises(ValueError, match="no table T2"):
        held_schema(run, "T2")
