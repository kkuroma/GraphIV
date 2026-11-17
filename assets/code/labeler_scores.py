"""
Scores the citation labeler for its two figures, on the CPU, and caches what it finds

The student is scored against its test split and the ground truth, cached in citation_labeler.json
APE scores are the stage0 run's printed log of 2026-08-30, which no file on disk keeps
"""

from __future__ import annotations

import csv
import json
import random
import re
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from style import DATA
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

CACHE = DATA / "citation_labeler.json"

# the order stage2_train_student.py trained with, since a checkpoint stores indices and not names
INTENTS = ("uses", "extends", "compares", "motivates", "background", "critiques")
# the order the figures read, most common first
SHOWN = ("background", "uses", "compares", "critiques", "extends", "motivates")
# accuracy on the ground truth and unparsed replies per generation, from the stage0 log
GENERATIONS = [(0.817, 0), (0.929, 0), (0.968, 0), (0.988, 0), (0.993, 2), (0.955, 31)]
KEPT = 4
# the kept prompt rerun on the ground truth, in INTENTS order, from logs/teacher_matrix.png
TEACHER_TRUTH = [
    [167, 0, 0, 0, 0, 0],
    [1, 166, 0, 0, 0, 0],
    [0, 0, 162, 0, 0, 5],
    [0, 0, 0, 148, 2, 0],
    [0, 0, 0, 0, 167, 0],
    [0, 0, 2, 0, 2, 163],
]
# citations a second stage1 labeled the whole store at, as the paper states it
SPEED = 1_762
SAMPLE = 10_000
EPOCH = re.compile(r"epoch (\d+) loss [\d.]+ val_acc ([\d.]+) val_macro_f1 ([\d.]+)")


def split(path: Path, seed: int, splits: list[float]) -> dict[str, list[dict[str, str]]]:
    """
    Cuts the teacher's labels exactly as stage2_train_student.py does

    Args:
        path: teacher_labeled.csv.
        seed: The shuffle seed.
        splits: Train, validation and test fractions.

    Returns:
        The three splits.
    """
    with path.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("intent") in INTENTS]
    random.Random(seed).shuffle(rows)
    a = int(len(rows) * splits[0])
    b = a + int(len(rows) * splits[1])
    return {"train": rows[:a], "val": rows[a:b], "test": rows[b:]}


def predict(cl: dict[str, Any], rows: list[dict[str, str]]) -> list[int]:
    """
    Labels rows with best.pt on the CPU, encoded as the stage encodes them

    Args:
        cl: The [citation_labeler] block of the config.
        rows: The citations to label.

    Returns:
        One intent index per row.
    """
    root = Path(str(cl["LABELER_PATH"]))
    config = AutoConfig.from_pretrained(str(cl["LABELER_MODEL"]), num_labels=len(INTENTS))
    model = AutoModelForSequenceClassification.from_config(config)
    _ = model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    tok = AutoTokenizer.from_pretrained(root / "tokenizer")
    out: list[int] = []
    for start in range(0, len(rows), 64):
        batch = rows[start : start + 64]
        enc = tok(
            [r["section"] or "(unknown)" for r in batch],
            [r["sentence"] for r in batch],
            max_length=int(cl["LABELER_MAX_SEQ"]),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        out += logits.argmax(-1).tolist()
    return out


def confusion(truth: list[str], guesses: list[int]) -> list[list[int]]:
    """
    Counts in SHOWN order, the true intent by row and the student's by column

    Args:
        truth: The true intent of each row.
        guesses: The student's intent index for each row.

    Returns:
        The matrix.
    """
    matrix = np.zeros((len(SHOWN), len(SHOWN)), dtype=int)
    for name, guess in zip(truth, guesses, strict=True):
        matrix[SHOWN.index(name), SHOWN.index(INTENTS[guess])] += 1
    return matrix.tolist()


def load(config: Path) -> dict[str, Any]:
    """
    Scores the student on whatever the cache does not hold yet

    Args:
        config: The config whose [citation_labeler] block names the run.

    Returns:
        Both confusion matrices, the validation curve, the teacher's counts and the split sizes.
    """
    cl = tomllib.loads(config.read_text(encoding="utf-8"))["citation_labeler"]
    root = Path(str(cl["LABELER_PATH"]))
    data = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    if "confusion" not in data:
        parts = split(
            root / "teacher_labeled.csv", int(cl["LABELER_SEED"]), list(cl["LABELER_SPLITS"])
        )
        print(f"[intent] scoring {len(parts['test']):,} test citations on the CPU", flush=True)
        guesses = predict(cl, parts["test"])
        data["confusion"] = confusion([r["intent"] for r in parts["test"]], guesses)
        data["sizes"] = {k: len(v) for k, v in parts.items()}
        data["teacher"] = {}
        for part in parts.values():
            for row in part:
                data["teacher"][row["intent"]] = data["teacher"].get(row["intent"], 0) + 1
        log = (root / "logs" / "student.log").read_text(encoding="utf-8")
        data["curve"] = [[int(e), float(a), float(f)] for e, a, f in EPOCH.findall(log)]
    if "truth" not in data:
        with (root / "ground_truth.csv").open(encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["sentence"].strip()]
        print(f"[intent] scoring {len(rows):,} ground truth citations on the CPU", flush=True)
        data["truth"] = confusion([r["label"] for r in rows], predict(cl, rows))
        data["reference"] = len(rows)
    _ = CACHE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def scores(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precision, recall and F1 per intent from a confusion matrix with the truth on its rows

    Args:
        matrix: Counts, the true label by row and the predicted one by column.

    Returns:
        Precision, recall and F1, one value per intent.
    """
    hit = np.diag(matrix).astype(float)
    precision = hit / np.maximum(matrix.sum(axis=0), 1)
    recall = hit / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return precision, recall, f1


def teacher_truth() -> np.ndarray:
    """
    The teacher against the ground truth, reordered from INTENTS to SHOWN

    Returns:
        The matrix, the true intent by row.
    """
    order = [INTENTS.index(name) for name in SHOWN]
    return np.array(TEACHER_TRUTH)[np.ix_(order, order)]
