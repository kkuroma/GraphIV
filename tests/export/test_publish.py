"""
What the publish stage uploads: only what differs from the hub, and never the hub's own root files
"""

import hashlib
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from stage3_publish_hf import ROOT, Digest, Repo, changes, digest, files, group, release_tag

TABLES = ("papers", "citations")


def test_a_table_file_commits_with_its_table() -> None:
    assert group("papers/papers-00000.parquet", TABLES) == "papers"
    assert group("README.md", TABLES) == ROOT
    assert group("other/file.txt", TABLES) == ROOT


def test_an_lfs_file_compares_by_sha256() -> None:
    hub = {"papers/a.parquet": Digest("same", "old blob")}
    assert changes({"papers/a.parquet": Digest("same", "new blob")}, hub, TABLES) == {}
    moved = changes({"papers/a.parquet": Digest("other", "new blob")}, hub, TABLES)
    assert moved == {"papers": (["papers/a.parquet"], [])}


def test_a_plain_file_compares_by_blob() -> None:
    hub = {"README.md": Digest(None, "abc")}
    assert changes({"README.md": Digest("x", "abc")}, hub, TABLES) == {}
    assert changes({"README.md": Digest("x", "abd")}, hub, TABLES) == {ROOT: (["README.md"], [])}


def test_only_a_table_is_pruned() -> None:
    hub = {
        ".gitattributes": Digest(None, "g"),
        "papers/papers-00009.parquet": Digest("s", "b"),
    }
    assert changes({}, hub, TABLES) == {"papers": ([], ["papers/papers-00009.parquet"])}


def test_digest_matches_git_and_sha256(tmp_path: Path) -> None:
    file = tmp_path / "hello.txt"
    file.write_bytes(b"hello\n")
    found = digest(file)
    assert found.blob == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert found.sha256 == hashlib.sha256(b"hello\n").hexdigest()


def test_files_lay_out_a_repo(tmp_path: Path) -> None:
    cards, release = tmp_path / "cards", tmp_path / "graphiv-2026-10-16"
    (cards / "card").mkdir(parents=True)
    (cards / "card" / "README.md").write_text("card")
    (release / "classifiers" / "model").mkdir(parents=True)
    (release / "classifiers" / "model" / "config.json").write_text("{}")
    (release / "papers").mkdir()
    (release / "papers" / "papers-00000.parquet").write_bytes(b"")
    data = Repo("DATA_REPO", "dataset", "card", tables=("papers",))
    model = Repo("CITATION_REPO", "model", "card", model="model")
    assert set(files(data, release, cards)) == {"README.md", "papers/papers-00000.parquet"}
    assert set(files(model, release, cards)) == {"README.md", "config.json"}


def test_the_benchmark_card_comes_from_its_version(tmp_path: Path) -> None:
    cards, bench = tmp_path / "cards", tmp_path / "graphiv-benchmarks-2026-10-22"
    (cards / "card").mkdir(parents=True)
    (cards / "card" / "README.md").write_text("template")
    (cards / "card" / "figure.png").write_bytes(b"")
    bench.mkdir()
    (bench / "README.md").write_text("rendered")
    repo = Repo("BENCHMARK_REPO", "dataset", "card", benchmarks=True)
    found = files(repo, bench, cards)
    assert found["README.md"] == bench / "README.md"
    assert "figure.png" in found


def test_the_tag_is_the_newest_paper(tmp_path: Path) -> None:
    (tmp_path / "papers").mkdir()
    dates = pa.array([date(2026, 9, 30), date(2026, 10, 1)], type=pa.date32())
    pq.write_table(pa.table({"date": dates}), tmp_path / "papers" / "papers-00000.parquet")
    assert release_tag(tmp_path) == "GraphIV-2026-10-01"
