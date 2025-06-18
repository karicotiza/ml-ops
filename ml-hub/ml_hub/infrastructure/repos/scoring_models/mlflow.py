"""MLFlow scoring models repositories code."""

from typing import TYPE_CHECKING

from mlflow import (
    log_params,
    search_registered_models,
    set_experiment,
    set_tracking_uri,
    start_run,
)
from mlflow.sklearn import log_model

from ml_hub.domain.interfaces.ml.scoring import ScoringModel

if TYPE_CHECKING:
    from mlflow.entities import Experiment


class ScoringModels:
    """MLFLow scoring models."""

    def __init__(
        self,
        tracking_uri: str,
    ) -> None:
        """Create new instance.

        Args:
            tracking_uri (str): MLFlow tracking uri.

        """
        set_tracking_uri(tracking_uri)

    def all(self) -> list[ScoringModel]:
        """Get all scoring models.

        Returns:
            list(str): list of scoring models.

        """
        return [model.name for model in search_registered_models()]

    def add(
        self,
        model_name: str,
        experiment_name: str,
        scoring_model: ScoringModel,
    ) -> None:
        experiment: Experiment = set_experiment(experiment_name)

        with start_run(experiment_id=experiment.experiment_id):
            log_params(params=scoring_model.model.get_params())

            log_model(
                sk_model=scoring_model.model,
                name=model_name,
                registered_model_name=model_name,
            )
