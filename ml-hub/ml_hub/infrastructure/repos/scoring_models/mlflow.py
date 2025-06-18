"""MLFlow scoring models repositories code."""

from typing import Any

from mlflow import (
    log_metrics,
    log_params,
    set_experiment,
    set_tracking_uri,
    start_run,
)
from mlflow.sklearn import load_model, log_model
from sklearn.ensemble import RandomForestClassifier

from ml_hub.infrastructure.ml.scoring.random_forest import RandomForest


class ScoringModels:
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
    ) -> RandomForest:
        """Get scoring model by name and version.

        Args:
            name (str): name of the model.
            version (str): version of the model.

        Returns:
            RandomForest: random forest scoring model.

        """
        set_tracking_uri(self._uri)

        model_uri: str = f"models:/{name}/{version}"
        model_instance: Any = load_model(model_uri) or ""

        return RandomForest(
            model_name=name,
            model_version=version,
            model_instance=model_instance,
        )

    def add(
        self,
        model: RandomForestClassifier,
        model_name: str,
        model_metrics: dict[str, float],
        model_parameters: dict[Any, Any],
    ) -> None:
        """Add scoring model.

        Args:
            model (RandomForestClassifier): scoring model.
            model_name (str): scoring model name.
            model_metrics (dict[str, float]): scoring model metrics.
            model_parameters (dict[Any, Any]): scoring model parameters.

        """
        set_tracking_uri(self._uri)

        with start_run(experiment_id=self._experiment_id):
            log_params(model_parameters)
            log_metrics(model_metrics)
            log_model(
                sk_model=model,
                name=model_name,
                registered_model_name=model_name,
            )
