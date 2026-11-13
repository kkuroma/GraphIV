"""
Reads config.toml, with the manager's override merged over it and every relative path resolved

The override holds only the keys the manager changed, so deleting one reverts it to config.toml
A relative path means the same thing from any working directory, anchored to the file it sits in
"""

import json
import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

SOCKET_HOST = re.compile(r"(?<![^\s])host=(\.\S*)")
OVERRIDE_ENV = "GRAPHIV_CONFIG_OVERRIDE"
OVERRIDE_NAME = "config-override.toml"


def override_path(path: str | Path) -> Path:
    """
    Where the override for one config file lives

    Args:
        path: Where config.toml lives.

    Returns:
        GRAPHIV_CONFIG_OVERRIDE when set, config-override.toml beside the config otherwise.
    """
    named = os.environ.get(OVERRIDE_ENV, "")
    return Path(named) if named else Path(path).resolve().parent / OVERRIDE_NAME


def load_config(path: str | Path, override: str | Path | None = None) -> dict[str, Any]:
    """
    Parses a config file, merges its override over it and anchors each relative path

    An empty value stays empty, since it means the setting is off
    A `*_DEST` DSN whose `host=` starts with a dot names a socket directory, anchored the same way

    Args:
        path: Where config.toml lives.
        override: The override file, or None for the one override_path names.

    Returns:
        The parsed blocks, with every path value absolute.
    """
    path = Path(path).resolve()
    config: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    _anchor(config, path.parent)
    layer_path = override_path(path) if override is None else Path(override)
    if layer_path.is_file():
        layer: dict[str, Any] = tomllib.loads(layer_path.read_text(encoding="utf-8"))
        _anchor(layer, layer_path.resolve().parent)
        merge(config, layer)
    return config


def merge(base: dict[str, Any], layer: dict[str, Any], where: str = "") -> None:
    """
    Writes every key of the override into the config in place, table by table

    A key the config lacks is skipped, so a renamed key in an old override never reaches the code

    Args:
        base: The parsed config.
        layer: The parsed override.
        where: The dotted table the two sit at, for the warning.
    """
    for key, value in layer.items():
        if key not in base:
            LOG.warning("%s%s is in the override but not the config, so it is skipped", where, key)
        elif isinstance(base[key], dict) and isinstance(value, dict):
            merge(base[key], value, f"{where}{key}.")
        else:
            base[key] = value


def dumps(config: dict[str, Any]) -> str:
    """
    Writes a parsed config back as toml, the inverse of tomllib for the value types config uses

    Args:
        config: Tables of strings, numbers, booleans and lists of them, nested at any depth.

    Returns:
        The toml text, one table per section.
    """
    lines: list[str] = []
    _dump_table(config, [], lines)
    return "\n".join(lines).lstrip("\n") + "\n"


def _dump_table(table: dict[str, Any], where: list[str], lines: list[str]) -> None:
    """
    Appends one table's plain keys under its header, then each nested table after it

    Args:
        table: The table.
        where: The dotted path to it, empty at the root.
        lines: The text so far, written into in place.
    """
    plain = {k: v for k, v in table.items() if not isinstance(v, dict)}
    if where and (plain or not table):
        lines += ["", f"[{'.'.join(where)}]"]
    lines += [f"{k} = {_dump_value(v)}" for k, v in plain.items()]
    for key, value in table.items():
        if isinstance(value, dict):
            _dump_table(value, [*where, key], lines)


def _dump_value(value: Any) -> str:
    """
    One value in toml

    Args:
        value: A string, a number, a boolean or a list of them.

    Returns:
        Its toml text.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_dump_value(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def _anchor(block: dict[str, Any], root: Path) -> None:
    """
    Rewrites the relative path values of one block and of every block nested in it, in place

    Args:
        block: One table of the parsed file.
        root: The directory the file sits in.
    """
    for key, value in block.items():
        if isinstance(value, dict):
            _anchor(value, root)
        elif not isinstance(value, str) or not value:
            continue
        elif key.endswith("_PATH"):
            block[key] = str(root / value)
        elif key.endswith("_DEST"):
            block[key] = SOCKET_HOST.sub(lambda m: f"host={root / m.group(1)}", value)
