"""
SurveyMatch, the test split of GraphIV's release and its train and val for fitting
"""

from src.datasets.release import ReleaseDataset


class SurveyMatch(ReleaseDataset):
    NAME = "surveymatch"


DATASET = SurveyMatch
