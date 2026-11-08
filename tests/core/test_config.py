"""
How config paths resolve, over a config file written into a temp directory
"""

from __future__ import annotations

from pathlib import Path

from src.core.config import load_config

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
