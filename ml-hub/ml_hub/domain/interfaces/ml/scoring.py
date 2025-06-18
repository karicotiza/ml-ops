"""Scoring ML model interface."""

from typing import Protocol

from sklearn.ensemble import RandomForestClassifier


class ScoringModel(Protocol):
    """Scoring ML model interface."""

    @property
    def model(self) -> RandomForestClassifier:
        """Get model.

        Returns:
            RandomForestClassifier: sklearn's random forest classifier.

        """
        ...
