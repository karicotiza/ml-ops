"""Scoring models repository interface."""

from typing import Protocol

from sklearn.base import BaseEstimator

from ml_hub.domain.interfaces.ml.scoring import Scoring
from ml_hub.domain.value_objects.metadata import Metadata


class ScoringModels(Protocol):
    """Scoring models repository interface."""

    def get(
        self,
        name: str,
        version: str,
    ) -> Scoring:
        """Get scoring model by name and version.

        Args:
            name (str): name of the model.
            version (str): version of the model.

        Returns:
            Scoring: scoring model.

        """
        ...

    def add(
        self,
        model: BaseEstimator,
        metadata: Metadata,
    ) -> None:
        """Add scoring model.

        Args:
            model (BaseEstimator): scoring model instance.
            metadata (Metadata): metadata.

        """
        ...
