"""Scoring ML model interface."""

from typing import Protocol

from ml_hub.domain.value_objects.scoring_features import ScoringFeatures


class Scoring(Protocol):
    """Scoring ML model interface."""

    @property
    def name(self) -> str:
        """Get scoring model name.

        Returns:
            str: name.

        """
        ...

    @property
    def version(self) -> str | None:
        """Get scoring model version.

        Returns:
            str: version.

        """
        ...

    def predict(self, features: ScoringFeatures) -> float:
        """Predict scoring.

        Args:
            features (ScoringFeatures): scoring features.

        Returns:
            float: prediction.

        """
        ...
