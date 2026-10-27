"""
Hands cases to a script of src/graphiv/ and reads its rankings back, for the methods that run on GraphIV's own code
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from src import common
from src.schema import Case, Ranking


def rank(script: str, cases: list[Case], *args: str) -> list[Ranking]:
    """
    Runs a src/graphiv/ script on the cases, which writes one ranking per line

    Args:
        script: The file under src/graphiv/.
        cases: The cases.
        args: The script's other arguments.

    Returns:
        The rankings.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path, out = Path(tmp) / "cases.jsonl", Path(tmp) / "out.jsonl"
        common.write_jsonl(path, cases)
        common.graphiv(script, *args, "--cases", str(path), "--out", str(out))
        return common.read_jsonl(out)


def compute(script: str, cases: list[Case], *args: str, path: str = "") -> Any:
    """
    Runs a src/graphiv/ script on the cases, which writes one json document

    Args:
        script: The file under src/graphiv/.
        cases: The cases.
        args: The script's other arguments.
        path: Another directory of the checkout for PYTHONPATH.

    Returns:
        The document.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cases_path, out = Path(tmp) / "cases.jsonl", Path(tmp) / "out.json"
        common.write_jsonl(cases_path, cases)
        common.graphiv(script, *args, "--cases", str(cases_path), "--out", str(out), path=path)
        return json.loads(out.read_text())
