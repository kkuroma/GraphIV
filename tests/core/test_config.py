"""
How config paths resolve, over a config file written into a temp directory
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from src.core.config import dumps, load_config

TEXT = """
[database]
PG_DEST = "host=.pg port=5433 dbname=arxivkg"

[serve]
RUNS_PATH = "data/runs"
CACHE_PATH = "/abs/cache"
HOST = "127.0.0.1"

[manager]
HF_TOKEN_PATH = ""

[benchmarks.recipes]
SurveyMatch = ["survey.toml"]
"""


def _load(tmp_path: Path, text: str = TEXT) -> dict:
    path = tmp_path / "config.toml"
    _ = path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_a_relative_path_resolves_against_the_file(tmp_path: Path, monkeypatch):
    monkeypatch.chdir("/")
    assert _load(tmp_path)["serve"]["RUNS_PATH"] == str(tmp_path.resolve() / "data/runs")


def test_an_absolute_path_is_kept(tmp_path: Path):
    assert _load(tmp_path)["serve"]["CACHE_PATH"] == "/abs/cache"


def test_an_empty_path_stays_off(tmp_path: Path):
    assert _load(tmp_path)["manager"]["HF_TOKEN_PATH"] == ""


def test_a_value_that_is_not_a_path_is_kept(tmp_path: Path):
    config = _load(tmp_path)
    assert config["serve"]["HOST"] == "127.0.0.1"
    assert config["benchmarks"]["recipes"]["SurveyMatch"] == ["survey.toml"]


def test_a_relative_socket_directory_is_anchored(tmp_path: Path):
    dsn = _load(tmp_path)["database"]["PG_DEST"]
    assert dsn == f"host={tmp_path.resolve() / '.pg'} port=5433 dbname=arxivkg"


def test_an_absolute_socket_and_a_hostname_are_kept(tmp_path: Path):
    text = '[a]\nPG_DEST = "host=/run/graphiv port=5433"\n[b]\nPG_DEST = "host=db.local"\n'
    config = _load(tmp_path, text)
    assert config["a"]["PG_DEST"] == "host=/run/graphiv port=5433"
    assert config["b"]["PG_DEST"] == "host=db.local"


def test_the_override_wins_and_merges_nested_tables(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GRAPHIV_CONFIG_OVERRIDE", raising=False)
    text = '[serve]\nHOST = "127.0.0.1"\nFETCH = true\n[miner.weights]\nuses = 0.4\nextends = 1.0\n'
    _ = (tmp_path / "config-override.toml").write_text(
        "[serve]\nFETCH = false\n[miner.weights]\nuses = 0.5\n", encoding="utf-8"
    )
    config = _load(tmp_path, text)
    assert config["serve"] == {"HOST": "127.0.0.1", "FETCH": False}
    assert config["miner"]["weights"] == {"uses": 0.5, "extends": 1.0}


def test_a_key_the_config_lacks_is_skipped(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GRAPHIV_CONFIG_OVERRIDE", raising=False)
    _ = (tmp_path / "config-override.toml").write_text(
        '[endpoints]\nEMBED_API = "old"\n[gone]\nX = 1\n', encoding="utf-8"
    )
    config = _load(tmp_path, '[endpoints]\nEMBEDDING_ENDPOINT = "new"\n')
    assert config == {"endpoints": {"EMBEDDING_ENDPOINT": "new"}}


def test_each_layer_anchors_to_its_own_directory(tmp_path: Path, monkeypatch):
    elsewhere = tmp_path / "state"
    elsewhere.mkdir()
    _ = (elsewhere / "o.toml").write_text('[serve]\nRUNS_PATH = "runs"\n', encoding="utf-8")
    monkeypatch.setenv("GRAPHIV_CONFIG_OVERRIDE", str(elsewhere / "o.toml"))
    config = _load(tmp_path)
    assert config["serve"]["RUNS_PATH"] == str(elsewhere.resolve() / "runs")
    assert config["serve"]["CACHE_PATH"] == "/abs/cache"


def test_a_missing_override_and_devnull_are_no_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GRAPHIV_CONFIG_OVERRIDE", str(tmp_path / "absent.toml"))
    assert _load(tmp_path)["serve"]["HOST"] == "127.0.0.1"
    _ = (tmp_path / "config-override.toml").write_text('[serve]\nHOST = "x"\n', encoding="utf-8")
    monkeypatch.delenv("GRAPHIV_CONFIG_OVERRIDE")
    path = tmp_path / "config.toml"
    assert load_config(path, override=os.devnull)["serve"]["HOST"] == "127.0.0.1"


def test_dumps_reads_back_as_the_same_tables(tmp_path: Path):
    tables = {
        "a": {"S": 'quote " and \\ and\nline', "I": 3, "F": 0.15, "B": False, "L": [0.8, 0.1]},
        "miner": {"ARGUES": ["extends"], "weights": {"uses": 0.4}},
    }
    assert tomllib.loads(dumps(tables)) == tables
