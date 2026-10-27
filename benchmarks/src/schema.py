"""
The two interfaces every benchmark file implements, a dataset's splits as cases and a method's fit and retrieve

A case is a dict with sample_id, query, cutoff, anchors and key, and may carry more a dataset needs
"""

from abc import ABC, abstractmethod
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

    def source_size(self, split: str) -> int:
        """
        How many rows the source holds before any is dropped, which a fit that shuffles raw rows needs

        Args:
            split: One of SPLITS.

        Returns:
            The row count, the case count unless a dataset says otherwise.
        """
        return len(self.get(split))

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
    One retrieval method, fit on a train split when it has anything to fit
    """

    # bumped by hand when what fit learns changes, so every older fit stops matching
    FIT_VERSION = 0

    def fit(self, data: Dataset, params: dict[str, Any]) -> dict[str, Any]:
        """
        Learns whatever the method tunes, called only when the experiment has a [fit] table

        Args:
            data: The dataset [fit] names, whose splits the method picks from.
            params: The rest of [fit].

        Returns:
            What was learned, as json, which retrieve receives as `fitted`.
        """
        raise NotImplementedError(f"{type(self).__name__} has nothing to fit")

    @abstractmethod
    def retrieve(self, cases: list[Case], params: dict[str, Any], fitted: dict[str, Any]) -> list[Ranking]:
        """
        Ranks the corpus for every case

        Args:
            cases: The cases, each to be capped at its cutoff with its anchors removed.
            params: [run] without ks.
            fitted: What fit returned, empty when the experiment has no [fit].

        Returns:
            One ranking per case.
        """
