"""Scoring models repository interface."""

from typing import Protocol

from sklearn.base import BaseEstimator

from ml_hub.domain.interfaces.ml.scoring import Scoring
from ml_hub.domain.value_objects.model_info import ModelInfo


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
        model_info: ModelInfo,
    ) -> None:
        """Add scoring model.

        Args:
            model (BaseEstimator): scoring model instance.
            model_info (ModelInfo): scoring model info.

        """
        ...
