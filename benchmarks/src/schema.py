"""
The two interfaces every benchmark file implements, a dataset's splits as cases and a method family's rankings

A case is a dict with sample_id, query, cutoff, anchors and key, and may carry more a dataset needs
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src import common

Case = dict[str, Any]
# sample_id and ranked, plus seconds when the method measured them
Ranking = dict[str, Any]


class Dataset(ABC):
    """
    A benchmark's splits, built from its sources once and read from cases/ after
    """

    NAME: str
    SPLITS: tuple[str, ...]

    def get(self, split: str) -> list[Case]:
        """
        The cases of one split

        Args:
            split: One of SPLITS.

        Returns:
            The cases, in the order the source lists them.

        Raises:
            KeyError: When the dataset has no such split.
        """
        if split not in self.SPLITS:
            raise KeyError(f"{self.NAME} has no split {split}, only {self.SPLITS}")
        path = common.CASES / self.NAME / f"{split}.jsonl"
        if not path.exists():
            common.write_jsonl(path, self.build(split))
        return common.read_jsonl(path)

    def corpus(self) -> Path | None:
        """
        The ids a ranking is cut to before it is scored, for a benchmark that searches its own corpus

        Returns:
            A file of one id per line, or None to score the whole store.
        """
        return None

    @abstractmethod
    def build(self, split: str) -> list[Case]:
        """
        Reads one split out of the sources

        Args:
            split: One of SPLITS.

        Returns:
            The cases.
        """


class Method(ABC):
    """
    One family of retrieval methods, ranking every job of one retriever on one test set together
    """

    @abstractmethod
    def retrieve(self, cases: list[Case], jobs: list[dict[str, Any]]) -> dict[str, list[Ranking]]:
        """
        Ranks the corpus for every case under every job

        Args:
            cases: The test set's cases, each to be capped at its cutoff with its anchors removed.
            jobs: Test tomls sharing one retriever and test set, each with its run name under "name".

        Returns:
            One ranking per case for each job, by the job's name.
        """
