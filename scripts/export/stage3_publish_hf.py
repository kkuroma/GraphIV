"""
Publish the newest Parquet release, the benchmarks and the cards to the four Hugging Face repos

Only a file whose hash differs from the hub's is uploaded, with one commit per table directory
A release repo is tagged GraphIV-<newest paper date>, the benchmark repo GraphIV-benchmarks-<cutoff>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.hf_api import RepoFile

from src.core.config import load_config

CARDS = Path(__file__).resolve().parent / "cards"
RELEASE = re.compile(r"^graphiv-\d{4}-\d{2}-\d{2}$")
# the commit a repo's root files go in, apart from its table directories
ROOT = "card"


@dataclass(frozen=True)
class Repo:
    """
    One Hugging Face repo and where its files come from
    """

    key: str
    kind: str
    card: str
    tables: tuple[str, ...] = ()
    model: str = ""
    benchmarks: bool = False


REPOS = (
    Repo("DATA_REPO", "dataset", "data", tables=("papers", "citations", "topics")),
    Repo("CITATION_REPO", "model", "citation_labeler", model="citation_intent"),
    Repo("SURVEY_REPO", "model", "survey_labeler", model="survey"),
    Repo(
        "BENCHMARK_REPO",
        "dataset",
        "benchmarks",
        tables=("surveymatch", "frontiermatch", "consensusmatch"),
        benchmarks=True,
    ),
)


class Digest(NamedTuple):
    """
    What a file hashes to: its sha256, which the hub keeps for LFS files, and its git blob id
    """

    sha256: str | None
    blob: str


def newest_release(root: Path) -> Path:
    """
    The newest finished Parquet release under the export directory

    Args:
        root: EXPORT_PATH.

    Returns:
        Its directory.
    """
    found = sorted(p for p in root.iterdir() if p.is_dir() and RELEASE.match(p.name))
    if not found:
        sys.exit(f"[publish] no graphiv-<date>/ under {root}, so run the Parquet export first")
    return found[-1]


def files(repo: Repo, release: Path, cards: Path = CARDS) -> dict[str, Path]:
    """
    Every file one repo carries, by its path in the repo

    Args:
        repo: The repo.
        release: The Parquet release.
        cards: The directory holding one card directory per repo.

    Returns:
        Path in the repo to the local file.
    """
    out = {p.name: p for p in sorted((cards / repo.card).iterdir()) if p.is_file()}
    if repo.model:
        out |= {p.name: p for p in sorted((release / "classifiers" / repo.model).iterdir())}
    if repo.benchmarks:
        out["README.md"] = release / "README.md"
    for table in repo.tables:
        out |= {f"{table}/{p.name}": p for p in sorted((release / table).glob("*.parquet"))}
    return out


def digest(path: Path) -> Digest:
    """
    A local file's sha256 and git blob id, read in one pass

    Args:
        path: The file.

    Returns:
        Both hashes.
    """
    sha = hashlib.sha256()
    blob = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
            blob.update(chunk)
    return Digest(sha.hexdigest(), blob.hexdigest())


def remote(api: HfApi, repo_id: str, kind: str) -> dict[str, Digest]:
    """
    Every file on the hub, with the hash the hub keeps for it

    Args:
        api: The authenticated client.
        repo_id: The repo.
        kind: dataset or model.

    Returns:
        Path in the repo to its digest, the sha256 only for an LFS file.
    """
    out: dict[str, Digest] = {}
    for entry in api.list_repo_tree(repo_id, repo_type=kind, recursive=True, expand=True):
        if isinstance(entry, RepoFile):
            out[entry.path] = Digest(entry.lfs.sha256 if entry.lfs else None, entry.blob_id)
    return out


def group(path: str, tables: tuple[str, ...]) -> str:
    """
    The commit a file goes in: its table directory, or the card for anything at the root

    Args:
        path: The path in the repo.
        tables: The repo's table directories.

    Returns:
        The commit's name.
    """
    top = path.split("/", 1)[0]
    return top if "/" in path and top in tables else ROOT


def changes(
    local: dict[str, Digest], hub: dict[str, Digest], tables: tuple[str, ...]
) -> dict[str, tuple[list[str], list[str]]]:
    """
    What each commit has to upload and delete for the hub to match the local files

    A file the hub stores in LFS is compared by sha256 and any other file by its git blob id.
    Only a table directory is pruned, so the hub's own root files such as .gitattributes stay.

    Args:
        local: The local files' digests.
        hub: The hub's digests.
        tables: The repo's table directories.

    Returns:
        Commit name to the paths to upload and the paths to delete, for commits with any.
    """
    out: dict[str, tuple[list[str], list[str]]] = {}
    for path, mine in sorted(local.items()):
        there = hub.get(path)
        same = there is not None and (
            there.sha256 == mine.sha256 if there.sha256 else there.blob == mine.blob
        )
        if not same:
            out.setdefault(group(path, tables), ([], []))[0].append(path)
    for path in sorted(set(hub) - set(local)):
        if group(path, tables) != ROOT:
            out.setdefault(group(path, tables), ([], []))[1].append(path)
    return out


def release_tag(release: Path) -> str:
    """
    The tag a release is published under, named by its newest paper

    Args:
        release: The Parquet release.

    Returns:
        GraphIV-<date>.
    """
    dates = pq.read_table(sorted((release / "papers").glob("*.parquet")), columns=["date"])
    newest = pc.max(dates["date"]).as_py()  # pyright: ignore[reportAttributeAccessIssue]
    return f"GraphIV-{newest.isoformat()}"


def token(cfg: dict[str, Any]) -> str:
    """
    The token publishing uses: the file [manager] HF_TOKEN_PATH points at, or HF_TOKEN

    Args:
        cfg: The config.

    Returns:
        The token, after printing why and exiting when there is none.
    """
    path = str(cfg.get("manager", {}).get("HF_TOKEN_PATH", ""))
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8").strip()
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    sys.exit("[publish] no Hugging Face token, so set [manager] HF_TOKEN_PATH or HF_TOKEN")


def publish(api: HfApi, repo: Repo, repo_id: str, release: Path, tag: str, dry: bool) -> None:
    """
    Brings one repo up to the release, commit by commit, then tags it

    Args:
        api: The authenticated client.
        repo: The repo.
        repo_id: Its id on the hub.
        release: The Parquet release.
        tag: The tag to give it.
        dry: Whether to only say what would change.
    """
    local = files(repo, release)
    digests: dict[str, Digest] = {}
    for i, (path, file) in enumerate(local.items(), 1):
        digests[path] = digest(file)
        print(f"[publish] {i}/{len(local)} files hashed for {repo_id}", flush=True)
    todo = changes(digests, remote(api, repo_id, repo.kind), repo.tables)
    tagged = tag in {t.name for t in api.list_repo_refs(repo_id, repo_type=repo.kind).tags}
    if not todo:
        print(f"[publish] {repo_id}: every file matches the hub", flush=True)
    for name, (uploads, deletes) in todo.items():
        size = sum(local[p].stat().st_size for p in uploads)
        print(
            f"[publish] {repo_id} {name}: {len(uploads)} to upload ({size / 1e6:,.1f} MB),"
            f" {len(deletes)} to delete",
            flush=True,
        )
        for path in uploads:
            print(f"  upload {path}", flush=True)
        for path in deletes:
            print(f"  delete {path}", flush=True)
        if dry:
            continue
        operations: list[CommitOperationAdd | CommitOperationDelete] = [
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(local[p])) for p in uploads
        ]
        operations += [CommitOperationDelete(path_in_repo=p) for p in deletes]
        api.create_commit(repo_id, operations, commit_message=f"{tag}: {name}", repo_type=repo.kind)
        print(f"[publish] {repo_id} {name}: committed", flush=True)
    if tagged:
        print(f"[publish] {repo_id}: tag {tag} exists and stays where it is", flush=True)
    elif dry:
        print(f"[publish] {repo_id}: would tag {tag}", flush=True)
    else:
        api.create_tag(repo_id, tag=tag, repo_type=repo.kind)
        print(f"[publish] {repo_id}: tagged {tag}", flush=True)


def main() -> None:
    """
    Read the config and the token, then publish every repo [export] gives an id to
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--config_path", default="config.toml")
    _ = ap.add_argument("--dry-run", action="store_true", help="list changes, upload nothing")
    _ = ap.add_argument("--release", default="", help="a graphiv-<date>/ other than the newest")
    args = ap.parse_args()

    cfg = load_config(str(args.config_path))
    export = cfg["export"]
    release = Path(args.release) if args.release else newest_release(Path(export["EXPORT_PATH"]))
    cutoff = str(cfg["benchmarks"]["CUTOFF"])
    bench = Path(export["EXPORT_PATH"]) / f"graphiv-benchmarks-{cutoff}"
    api = HfApi(token=token(cfg))
    tag = release_tag(release)
    print(f"[publish] {release.name} as {tag}, {'dry run' if args.dry_run else 'uploading'}")
    for repo in REPOS:
        repo_id = str(export.get(repo.key, ""))
        if not repo_id:
            print(f"[publish] [export] {repo.key} is empty, so that repo is skipped", flush=True)
            continue
        if repo.benchmarks and not (bench / "README.md").is_file():
            print(
                f"[publish] no {bench.name}/README.md, so rerun the benchmark export first",
                flush=True,
            )
            continue
        if repo.benchmarks:
            publish(api, repo, repo_id, bench, f"GraphIV-benchmarks-{cutoff}", bool(args.dry_run))
        else:
            publish(api, repo, repo_id, release, tag, bool(args.dry_run))


if __name__ == "__main__":
    main()
