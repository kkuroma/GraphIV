"""
The config tab: the schema off config.toml, saving and reverting the override, the staged config
"""

import os
import stat
import tomllib
from pathlib import Path

import pytest

import settings
from schedule import Manager
from src.core.config import load_config

TEXT = """# The tool server
[serve]
HOST = "127.0.0.1" # interface to bind
FETCH = true # whether a missing paper is fetched
ALLOWED_HOSTS = ["a", "b"] # proxied hosts

# Weight of each intent
[miner.weights]
uses = 0.4 # uses

# Model endpoints
[endpoints]
LLM_API_KEY = "" # chat key
EMBEDDING_BATCH = 256 # papers per batch

# Where the snapshot is
[source]
SOURCE_PATH = "a.json" # the snapshot
"""


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("GRAPHIV_CONFIG_OVERRIDE", raising=False)
    monkeypatch.delenv(settings.SCHEMA_ENV, raising=False)
    path = tmp_path / "config.toml"
    _ = path.write_text(TEXT, encoding="utf-8")
    return path


def override(config: Path) -> dict:
    return tomllib.loads((config.parent / "config-override.toml").read_text(encoding="utf-8"))


def test_the_schema_reads_sections_kinds_and_comments(config: Path) -> None:
    sections, source = settings.schema(config)
    assert source == "config.toml"
    assert [s["name"] for s in sections] == ["serve", "miner.weights", "endpoints", "source"]
    serve = sections[0]
    assert serve["description"] == "The tool server"
    assert [(k["key"], k["kind"]) for k in serve["keys"]] == [
        ("HOST", "str"),
        ("FETCH", "bool"),
        ("ALLOWED_HOSTS", "list"),
    ]
    assert serve["keys"][2]["item"] == "str"
    assert serve["keys"][0]["description"] == "interface to bind"


def test_a_save_writes_only_what_differs_and_is_private(config: Path) -> None:
    changes = {"serve": {"FETCH": False, "HOST": "127.0.0.1"}, "miner.weights": {"uses": 1}}
    assert settings.save(config, changes, check=False) == []
    assert override(config) == {"serve": {"FETCH": False}, "miner": {"weights": {"uses": 1.0}}}
    mode = stat.S_IMODE(os.stat(config.parent / "config-override.toml").st_mode)
    assert mode == 0o600
    assert load_config(config)["miner"]["weights"]["uses"] == 1.0


def test_a_wrong_kind_or_an_unknown_key_writes_nothing(config: Path) -> None:
    said = settings.save(config, {"serve": {"FETCH": "no"}, "serve.x": {"Y": 1}}, check=False)
    assert len(said) == 2
    assert not (config.parent / "config-override.toml").exists()


def test_saving_the_default_again_reverts_it(config: Path) -> None:
    _ = settings.save(config, {"endpoints": {"EMBEDDING_BATCH": 64}}, check=False)
    _ = settings.save(config, {"endpoints": {"EMBEDDING_BATCH": 256}}, check=False)
    assert override(config) == {}


def test_revert_one_key_and_everything(config: Path) -> None:
    _ = settings.save(config, {"serve": {"FETCH": False, "HOST": "0.0.0.0"}}, check=False)
    settings.revert(config, "serve", "FETCH")
    assert override(config) == {"serve": {"HOST": "0.0.0.0"}}
    settings.revert(config)
    assert override(config) == {}


def test_the_view_masks_a_key_and_lists_stale_ones(config: Path) -> None:
    _ = settings.save(config, {"endpoints": {"LLM_API_KEY": "sk-secret"}}, check=False)
    layer = config.parent / "config-override.toml"
    _ = layer.write_text(layer.read_text() + "\n[gone]\nOLD = 1\n", encoding="utf-8")
    seen = settings.view(config)
    keys = {k["key"]: k for s in seen["sections"] for k in s["keys"]}
    assert keys["LLM_API_KEY"]["value"] == "set"
    assert keys["LLM_API_KEY"]["overridden"]
    assert "sk-secret" not in str(seen)
    assert seen["stale"] == ["gone.OLD"]
    assert settings.secret(config, "endpoints", "LLM_API_KEY") == "sk-secret"
    with pytest.raises(ValueError):
        _ = settings.secret(config, "serve", "HOST")


def test_built_keys_are_marked(config: Path) -> None:
    keys = {k["key"]: k["built"] for s in settings.view(config)["sections"] for k in s["keys"]}
    assert keys["HOST"] and keys["ALLOWED_HOSTS"] and keys["SOURCE_PATH"]
    assert not keys["FETCH"] and not keys["EMBEDDING_BATCH"]


def test_the_staged_config_beats_an_overridden_snapshot(config: Path) -> None:
    _ = settings.save(config, {"source": {"SOURCE_PATH": "/elsewhere/b.json"}}, check=False)
    manager = Manager(config.parent / "scripts", config, config.parent / "state", "python")
    staged = manager.staged_config()
    read = load_config(staged, override=os.devnull)
    assert read["source"]["SOURCE_PATH"] == "/elsewhere/arxiv-new.json"
    assert read["serve"]["FETCH"] is True
