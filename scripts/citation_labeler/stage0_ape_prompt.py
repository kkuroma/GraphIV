"""
Evolve the teacher prompt against the hand-labeled set and keep the best generation

Each round labels the ground truth, shows a judge only the mistakes, and redrafts from that critique
Every generation is written out, so a winner is a file on disk rather than a line in a log
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from labels import INTENTS, parse_labels

from src.core.config import load_config
from src.core.llm import headers

BATCH = 25

JUDGE = """You are improving a citation-intent labeling policy.
Below is the policy, then cases it got wrong, each with the text, the correct label, the label the
policy chose, and the reason the labeler gave for choosing it.
Read the reasons, not just the labels, since they show what the policy made the labeler believe.
Say what the policy is systematically getting wrong. Be specific about which classes it confuses
and what rule would separate them. Do not rewrite the policy, just diagnose it."""

DRAFTER = """You are rewriting a citation-intent labeling policy.
Below is the current policy and a diagnosis of what it gets wrong.
Rewrite the policy to fix those failures while keeping everything that already works.
Keep the same output contract and the same six labels. Return only the new policy text."""


class PromptRefiner:
    """
    One LLM endpoint and the label, critique, redraft loop over it
    """

    def __init__(
        self, model: str, generations: int, api: str, parallel: int, key: str = ""
    ) -> None:
        """
        Args:
            model: model name the endpoint answers to.
            generations: how many rounds to run.
            api: OpenAI-compatible base URL.
            parallel: concurrent requests.
            key: the API key, empty for an endpoint that takes none.
        """
        self.model = model
        self.generations = generations
        self.api = api.rstrip("/")
        self.key = key
        self.parallel = parallel

    def _chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """
        One completion, returning empty text on any transport failure

        Args:
            system: the system message.
            user: the user message.
            max_tokens: completion budget.

        Returns:
            The assistant message, or an empty string.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        }
        req = urllib.request.Request(
            f"{self.api}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers(self.key),
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                body: dict[str, Any] = json.load(r)  # pyright: ignore[reportAny, reportExplicitAny]
            return str(body["choices"][0]["message"]["content"] or "")  # pyright: ignore[reportAny]
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError):
            return ""

    def _one_batch(self, prompt: str, rows: list[dict[str, str]]) -> list[tuple[str, str] | None]:
        """
        Label one batch and align the answers back onto the inputs

        The deciding phrase is asked for because naming it lifts accuracy, and because the judge
        diagnoses the policy far better from the labeler's reasoning than from its verdict alone.

        Args:
            prompt: the policy under test.
            rows: reference rows carrying `section` and `sentence`.

        Returns:
            One (intent, reason) per row, None where the answer was unusable.
        """
        items = "\n".join(
            f"{i}. section: {r['section'] or '(unknown)'}\n   sentence: {r['sentence']}"
            for i, r in enumerate(rows)
        )
        user = (
            f"{items}\n\nAnswer with one line per item, formatted "
            "`index: intent - the phrase that decided it`."
        )
        text = self._chat(prompt, user, max_tokens=len(rows) * 48 + 256)
        got = parse_labels(text, len(rows))
        if any(got) or len(rows) == 1:
            return got

        """
        The router holds one model per gpu and evicts on lru, so a swap empties whatever calls
        were in flight. Retrying the same items in halves recovers them after the reload.
        """
        half = len(rows) // 2
        return self._one_batch(prompt, rows[:half]) + self._one_batch(prompt, rows[half:])

    def label(self, prompt: str, rows: list[dict[str, str]]) -> list[tuple[str, str] | None]:
        """
        Run the policy over every row

        Args:
            prompt: the policy under test.
            rows: the reference rows.

        Returns:
            One (intent, reason) per row in input order, None where nothing parsed.
        """
        chunks = [rows[i : i + BATCH] for i in range(0, len(rows), BATCH)]
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            results = list(pool.map(lambda c: self._one_batch(prompt, c), chunks))
        return [got for chunk in results for got in chunk]

    def critique(self, prompt: str, wrong: list[tuple[dict[str, str], str, str]]) -> str:
        """
        Ask the judge what the policy is systematically getting wrong

        Args:
            prompt: the policy under test.
            wrong: misclassified rows, each with the label chosen and the reason given.

        Returns:
            The diagnosis text.
        """
        shown = wrong[:60]
        cases = "\n\n".join(
            f"section: {r['section'] or '(unknown)'}\nsentence: {r['sentence']}\n"
            f"correct: {r['label']}\npolicy said: {pred}\nits reason: {why}"
            for r, pred, why in shown
        )
        return self._chat(JUDGE, f"POLICY:\n{prompt}\n\nMISTAKES:\n{cases}")

    def redraft(self, prompt: str, notes: str) -> str:
        """
        Rewrite the policy from the diagnosis

        Args:
            prompt: the policy under test.
            notes: the judge's diagnosis.

        Returns:
            The next policy, or the current one when the model returns nothing.
        """
        out = self._chat(DRAFTER, f"POLICY:\n{prompt}\n\nDIAGNOSIS:\n{notes}", max_tokens=4096)
        return out.strip() or prompt

    def refine(self, labeler_path: str) -> Path:
        """
        Run the loop and write every generation, plus the winner as best.txt

        Args:
            labeler_path: the citation_labeler data directory.

        Returns:
            The path of the winning prompt.
        """
        root = Path(labeler_path)
        rows = read_ground_truth(root / "ground_truth.csv")
        print(f"[ape] {len(rows):,} ground truth rows", flush=True)

        prompt = (root / "prompts" / "prompt_00.txt").read_text(encoding="utf-8")
        scores: list[tuple[int, float]] = []
        for gen in range(self.generations):
            path = root / "prompts" / f"prompt_{gen:02d}.txt"
            _ = path.write_text(prompt, encoding="utf-8")
            preds = self.label(prompt, rows)
            hits = sum(
                1 for got, r in zip(preds, rows, strict=True) if got and got[0] == r["label"]
            )
            missed = sum(1 for got in preds if got is None)
            acc = hits / len(rows)
            scores.append((gen, acc))
            print(f"[ape] gen {gen}: accuracy {acc:.3f}, {missed:,} unparsed", flush=True)

            if gen == self.generations - 1:
                break
            """
            An unparsed answer counts against the policy, since a prompt whose output cannot be
            read is not a usable prompt.
            """
            wrong = [
                (r, got[0] if got else "unparsed", got[1] if got else "")
                for r, got in zip(rows, preds, strict=True)
                if not got or got[0] != r["label"]
            ]
            prompt = self.redraft(prompt, self.critique(prompt, wrong))

        best_gen = max(scores, key=lambda s: s[1])[0]
        winner = (root / "prompts" / f"prompt_{best_gen:02d}.txt").read_text(encoding="utf-8")
        out = root / "best.txt"
        _ = out.write_text(winner, encoding="utf-8")
        print(f"[ape] best is generation {best_gen} -> {out}", flush=True)
        return out


def read_ground_truth(path: Path) -> list[dict[str, str]]:
    """
    Load the reference set, dropping rows with no usable sentence

    Args:
        path: the ground_truth.csv the agreement build writes.

    Returns:
        One dict per row.
    """
    with path.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("sentence", "").strip()]


def report(labeler_path: str, refiner: PromptRefiner) -> None:
    """
    Score best.txt against the ground truth and write the confusion matrix

    Args:
        labeler_path: the citation_labeler data directory.
        refiner: the endpoint used to run the winning prompt.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from plot import apply_theme  # noqa: PLC0415

    apply_theme()

    root = Path(labeler_path)
    rows = read_ground_truth(root / "ground_truth.csv")
    labeled = refiner.label((root / "best.txt").read_text(encoding="utf-8"), rows)

    idx = {name: i for i, name in enumerate(INTENTS)}
    cm = np.zeros((len(INTENTS), len(INTENTS)), dtype=int)
    for r, got in zip(rows, labeled, strict=True):
        if got:
            cm[idx[r["label"]], idx[got[0]]] += 1
    acc = float(np.trace(cm)) / max(1, cm.sum())
    missed = sum(1 for got in labeled if got is None)

    print(
        f"[ape] best.txt accuracy {acc:.3f} over {cm.sum():,} parsed rows, {missed:,} unparsed",
        flush=True,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    _ = ax.imshow(cm, cmap="viridis")
    _ = ax.set_title("teacher against the reference set")
    _ = ax.set_xticks(range(len(INTENTS)), INTENTS, rotation=45, ha="right")
    _ = ax.set_yticks(range(len(INTENTS)), INTENTS)
    _ = ax.set_xlabel("predicted")
    _ = ax.set_ylabel("ground truth")
    for i in range(len(INTENTS)):
        for j in range(len(INTENTS)):
            """
            Viridis runs light at the top of its range, so a fixed white number vanishes there.
            """
            shade = "#000000" if cm[i, j] > cm.max() * 0.6 else "#ffffff"
            _ = ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=shade, fontsize=8)
    fig.tight_layout()
    out = root / "logs" / "teacher_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[ape] wrote {out}", flush=True)


def main() -> None:
    """
    Read the config, evolve the prompt, then score the winner
    """
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--config_path", default="config.toml")
    args = ap.parse_args()

    cfg = load_config(str(args.config_path))
    cl = cfg["citation_labeler"]
    refiner = PromptRefiner(
        model=str(cl["LABELER_APE_MODEL"]),
        generations=int(cl["LABELER_APE_GENERATIONS"]),
        api=str(cfg["endpoints"]["LLM_ENDPOINT"]),
        parallel=int(cl["LABELER_APE_PARALLEL"]),
        key=str(cfg["endpoints"]["LLM_API_KEY"]),
    )
    _ = refiner.refine(str(cl["LABELER_PATH"]))
    report(str(cl["LABELER_PATH"]), refiner)


if __name__ == "__main__":
    main()
