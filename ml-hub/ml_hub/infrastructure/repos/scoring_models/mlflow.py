"""MLFlow scoring models repository."""

from typing import Any

from mlflow import (
    log_metrics,
    log_params,
    set_experiment,
    set_tracking_uri,
    start_run,
)
from mlflow.sklearn import load_model, log_model
from sklearn.base import BaseEstimator

from ml_hub.domain.value_objects.model_info import ModelInfo
from ml_hub.infrastructure.ml.scoring.mlflow import MLFlowScoring


class MLFLowScoringModels:
    """MLFLow scoring models."""

    def __init__(
        self,
        uri: str,
        experiment: str = "scoring-models",
    ) -> None:
        """Create new instance.

        Args:
            uri (str): MLFlow uri.
            experiment (str, optional): experiment with scoring models.
                Defaults to "scoring-models".

        """
        self._uri: str = uri
        set_tracking_uri(self._uri)

        self._experiment_id: str = set_experiment(experiment).experiment_id

    def get(
        self,
        name: str,
        version: str,
    ) -> MLFlowScoring:
        """Get scoring model by name and version.

        Args:
            name (str): name of the model.
            version (str): version of the model.

        Returns:
            Scoring: scoring model.

        """
        set_tracking_uri(self._uri)

        model_uri: str = f"models:/{name}/{version}"
        model_instance: Any = load_model(model_uri) or ""

        return MLFlowScoring(
            name=name,
            version=version,
            model=model_instance,
        )

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
        set_tracking_uri(self._uri)

        with start_run(experiment_id=self._experiment_id):
            log_params(model_info.training_parameters)
            log_metrics(model_info.metrics)
            log_model(
                sk_model=model,
                name=model_info.name,
                registered_model_name=model_info.name,
            )
