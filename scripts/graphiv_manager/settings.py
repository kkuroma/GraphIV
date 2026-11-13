"""
The override over config.toml: what every key is, what it is now, and saving or reverting one

The schema comes from the NixOS module when GRAPHIV_CONFIG_SCHEMA names it, else from config.toml
The override is written 0600, since it holds the API keys
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import tomllib
from pathlib import Path
from typing import Any

import psycopg

import probes
from src.core.config import dumps, load_config, override_path
from src.core.embed import EmbeddingEndpoint, embed_documents, post

SCHEMA_ENV = "GRAPHIV_CONFIG_SCHEMA"
SECRET = re.compile(r"_API_KEY$")
# keys the units, caddy and the sandbox are also built around, so an override waits for a rebuild
BUILT = {
    ("serve", "HOST"),
    ("serve", "API_PORT"),
    ("serve", "FRONTEND_PORT"),
    ("serve", "ALLOWED_HOSTS"),
    ("manager", "PORT"),
}
PORTS = {("serve", "API_PORT"), ("serve", "FRONTEND_PORT"), ("manager", "PORT")}
HEADER = re.compile(r"^\[([\w.]+)\]\s*$")
ENTRY = re.compile(r"^(\w+)\s*=\s*(.*?)\s*(?:#\s*(.*))?$")
# a re-embedded paper is the stored vector up to halfvec rounding
SAME_SPACE = 0.99
# papers read for the shortest abstract, which no window can cut
PROBE_PAPERS = 50


def built(section: str, key: str) -> bool:
    """
    Whether nix also uses a key when it builds the units

    Args:
        section: The dotted table.
        key: The key.

    Returns:
        True for a path, a connection string, a port, the bind address and the proxied hosts.
    """
    return (section, key) in BUILT or key.endswith(("_PATH", "_DEST"))


def kind_of(value: Any) -> str:
    """
    The schema kind of a value read from config.toml

    Args:
        value: The value.

    Returns:
        bool, int, float, str, or list.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "list" if isinstance(value, list) else "str"


def from_toml(text: str) -> list[dict[str, Any]]:
    """
    A schema read off config.toml itself, for a manager running without the module's

    The comment above a table describes the section and the comment after a key describes the key

    Args:
        text: The config file.

    Returns:
        One section per table, in file order.
    """
    sections: list[dict[str, Any]] = []
    above = ""
    for line in text.splitlines():
        if line.startswith("#"):
            above = line.lstrip("# ").strip()
            continue
        header = HEADER.match(line)
        if header:
            sections.append({"name": header.group(1), "description": above, "keys": []})
            above = ""
            continue
        entry = ENTRY.match(line)
        if entry and sections:
            value = tomllib.loads(f"v = {entry.group(2)}")["v"]
            item = value[0] if isinstance(value, list) and value else ""
            key = {
                "key": entry.group(1),
                "description": entry.group(3) or "",
                "kind": kind_of(value),
            }
            if key["kind"] == "list":
                key["item"] = kind_of(item)
            sections[-1]["keys"].append(key)
        if not line.strip():
            above = ""
    return sections


def schema(config: Path) -> tuple[list[dict[str, Any]], str]:
    """
    Every section and key the tab shows, grouped as the module groups them

    Args:
        config: Where config.toml lives.

    Returns:
        The sections, and which source they came from.
    """
    named = os.environ.get(SCHEMA_ENV, "")
    if not named:
        return from_toml(config.read_text(encoding="utf-8")), "config.toml"
    loaded = json.loads(Path(named).read_text(encoding="utf-8"))
    rank = {name: i for i, name in enumerate(loaded.get("order", []))}
    last = len(rank)
    sections = sorted(loaded["sections"], key=lambda s: rank.get(s["name"], last))
    for section in sections:
        section["keys"].sort(key=lambda k: rank.get(f"{section['name']}.{k['key']}", last))
    return sections, "nix"


def at(tables: dict[str, Any], section: str) -> dict[str, Any]:
    """
    One table of a parsed config by its dotted name

    Args:
        tables: The parsed config.
        section: The dotted table, such as miner.weights.

    Returns:
        The table, empty when it is missing.
    """
    for part in section.split("."):
        tables = tables.get(part, {}) if isinstance(tables, dict) else {}
    return tables if isinstance(tables, dict) else {}


def read_override(config: Path) -> dict[str, Any]:
    """
    The override as it is on disk

    Args:
        config: Where config.toml lives.

    Returns:
        Its tables, empty when there is no file.
    """
    path = override_path(config)
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_override(config: Path, layer: dict[str, Any]) -> None:
    """
    Replaces the override whole, readable by its owner alone, dropping tables left empty

    Args:
        config: Where config.toml lives.
        layer: Its tables.
    """
    path = override_path(config)
    part = path.with_name(path.name + ".part")
    kept = prune(layer)
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        _ = fh.write(dumps(kept) if kept else "")
    os.chmod(part, 0o600)
    _ = part.replace(path)


def prune(layer: dict[str, Any]) -> dict[str, Any]:
    """
    The override without its empty tables

    Args:
        layer: Its tables.

    Returns:
        The same keys, with every table that holds none dropped.
    """
    out: dict[str, Any] = {}
    for key, value in layer.items():
        if isinstance(value, dict):
            kept = prune(value)
            if kept:
                out[key] = kept
        else:
            out[key] = value
    return out


def view(config: Path) -> dict[str, Any]:
    """
    Every key with its default, its current value and whether the override sets it

    Args:
        config: Where config.toml lives.

    Returns:
        The sections, the keys the override holds that the config no longer has, and the paths.
    """
    sections, source = schema(config)
    default = load_config(config, override=os.devnull)
    merged = load_config(config)
    layer = read_override(config)
    out = []
    for section in sections:
        name = str(section["name"])
        keys = []
        for key in section["keys"]:
            k = str(key["key"])
            secret = bool(SECRET.search(k))
            now = at(merged, name).get(k)
            keys.append(
                key
                | {
                    "default": "" if secret else at(default, name).get(k),
                    "value": ("set" if now else "") if secret else now,
                    "overridden": k in at(layer, name),
                    "secret": secret,
                    "built": built(name, k),
                }
            )
        out.append(section | {"keys": keys})
    known = {(str(s["name"]), str(k["key"])) for s in sections for k in s["keys"]}
    stale = [f"{s}.{k}" for s, k in flatten(layer) if (s, k) not in known]
    return {
        "sections": out,
        "stale": stale,
        "source": source,
        "config": str(config),
        "override": str(override_path(config)),
    }


def flatten(layer: dict[str, Any], where: str = "") -> list[tuple[str, str]]:
    """
    Every key of the override as its section and name

    Args:
        layer: The override's tables.
        where: The dotted table so far.

    Returns:
        One pair per key.
    """
    out: list[tuple[str, str]] = []
    for key, value in layer.items():
        if isinstance(value, dict):
            out += flatten(value, f"{where}.{key}" if where else key)
        elif where:
            out.append((where, key))
    return out


def secret(config: Path, section: str, key: str) -> str:
    """
    One API key in the clear, for the reveal button

    Args:
        config: Where config.toml lives.
        section: The dotted table.
        key: The key, which has to be an API key.

    Returns:
        The key's value.

    Raises:
        ValueError: When the key is not an API key.
    """
    if not SECRET.search(key):
        raise ValueError(f"{section}.{key} is not an API key")
    return str(at(load_config(config), section).get(key, ""))


def coerce(spec: dict[str, Any], raw: Any) -> Any:
    """
    One value from the page as its schema kind

    Args:
        spec: The key's schema entry.
        raw: What the page sent.

    Returns:
        The value.

    Raises:
        ValueError: When it is not of that kind.
    """
    kind = spec["kind"]
    if kind == "bool" and isinstance(raw, bool):
        return raw
    if kind == "int" and isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if kind == "float" and isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw)
    if kind == "str" and isinstance(raw, str):
        return raw
    if kind == "list" and isinstance(raw, list):
        item = {"kind": spec.get("item", "str")}
        return [coerce(item, x) for x in raw]
    raise ValueError(f"{spec['key']} takes a {kind}, not {json.dumps(raw)}")


def save(config: Path, changes: dict[str, dict[str, Any]], check: bool = True) -> list[str]:
    """
    Writes changed keys into the override, dropping any that equal the default

    Args:
        config: Where config.toml lives.
        changes: The new values, by dotted table and key.
        check: Whether the probes have to pass before anything is written.

    Returns:
        What is wrong, empty when the override was written.
    """
    sections, _ = schema(config)
    specs = {(str(s["name"]), str(k["key"])): k for s in sections for k in s["keys"]}
    default = load_config(config, override=os.devnull)
    wanted: dict[tuple[str, str], Any] = {}
    problems: list[str] = []
    for section, keys in changes.items():
        for key, raw in keys.items():
            spec = specs.get((section, key))
            if spec is None:
                problems.append(f"{section}.{key} is not a setting")
                continue
            try:
                wanted[(section, key)] = coerce(spec, raw)
            except ValueError as err:
                problems.append(str(err))
    if problems:
        return problems
    merged = load_config(config)
    after = json.loads(json.dumps(merged))
    for (section, key), value in wanted.items():
        at(after, section)[key] = value
    if check:
        problems = probe(merged, after, wanted)
        if problems:
            return problems
    layer = read_override(config)
    for (section, key), value in wanted.items():
        table = layer
        for part in section.split("."):
            table = table.setdefault(part, {})
        if value == at(default, section).get(key):
            table.pop(key, None)
        else:
            table[key] = value
    write_override(config, layer)
    return []


def revert(config: Path, section: str = "", key: str = "") -> None:
    """
    Deletes one key from the override, or the whole override

    Args:
        config: Where config.toml lives.
        section: The dotted table, empty to revert everything.
        key: The key, empty to revert the whole table.
    """
    layer = read_override(config)
    if not section:
        layer = {}
    elif key:
        _ = at(layer, section).pop(key, None)
    else:
        parent, _, last = section.rpartition(".")
        _ = (at(layer, parent) if parent else layer).pop(last, None)
    write_override(config, layer)


def probe(
    before: dict[str, Any], after: dict[str, Any], wanted: dict[tuple[str, str], Any]
) -> list[str]:
    """
    Checks what a save would change before it is written

    Args:
        before: The config as it is.
        after: The config as the save would leave it.
        wanted: The changed keys.

    Returns:
        What failed.
    """
    changed = {k for k, v in wanted.items() if at(before, k[0]).get(k[1]) != v}
    out: list[str] = []
    ep = after["endpoints"]
    if any(s == "endpoints" and k.startswith("LLM_") for s, k in changed):
        ok, said = probes.models(
            str(ep["LLM_ENDPOINT"]), str(ep["LLM_NAME"]), str(ep["LLM_API_KEY"])
        )
        if not ok:
            out.append(f"LLM: {said}")
    if any(s == "endpoints" and k.startswith("EMBEDDING_") for s, k in changed):
        said = same_space(EmbeddingEndpoint.from_config(after), str(after["database"]["PG_DEST"]))
        if said:
            out.append(f"embedding: {said}")
    for section, key in sorted(changed):
        value = at(after, section)[key]
        if (
            key.endswith("_PATH")
            and value
            and not (Path(value).exists() or Path(value).parent.is_dir())
        ):
            out.append(f"{section}.{key}: neither {value} nor its directory exists")
        if (section, key) in PORTS and not free(str(after["serve"]["HOST"]), int(value)):
            out.append(f"{section}.{key}: something already listens on {value}")
    return out


def same_space(endpoint: EmbeddingEndpoint, dsn: str) -> str:
    """
    Whether an embedding endpoint puts a stored paper where the store has it

    Args:
        endpoint: The endpoint the save would set.
        dsn: The store.

    Returns:
        What is wrong, empty when it tokenizes and the shortest stored paper re-embeds in place.
    """
    try:
        _ = post(endpoint, "/tokenize", {"content": "probe", "add_special": False})["tokens"]
    except (OSError, ValueError, KeyError) as err:
        return f"no /tokenize at {endpoint.url}: {err}"[:200]
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            rows = conn.execute(
                "SELECT title, abstract, embed_full::text FROM papers"
                " WHERE stage >= 3 AND embed_full IS NOT NULL LIMIT %s",
                (PROBE_PAPERS,),
            ).fetchall()
    except psycopg.Error as err:
        return f"the store did not answer: {err}"[:200]
    if not rows:
        return "the store has no embedded paper to compare against"
    title, abstract, stored = min(rows, key=lambda r: len(r[1] or ""))
    try:
        fresh = embed_documents(endpoint, [f"Title: {title or ''}\nAbstract: {abstract or ''}"])[0]
    except (OSError, ValueError, KeyError) as err:
        return f"no embedding from {endpoint.url}: {err}"[:200]
    kept = [float(x) for x in str(stored).strip("[]").split(",")]
    if len(kept) != len(fresh):
        return f"{len(fresh)} dimensions against the store's {len(kept)}"
    cosine = sum(a * b for a, b in zip(kept, fresh, strict=True)) / (
        math.sqrt(sum(a * a for a in kept)) or 1.0
    )
    if cosine < SAME_SPACE:
        return f"a stored paper re-embeds at cosine {cosine:.3f}, under {SAME_SPACE}"
    return ""


def free(host: str, port: int) -> bool:
    """
    Whether nothing listens on a port

    Args:
        host: The bind address.
        port: The port.

    Returns:
        True when it can be bound.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True
