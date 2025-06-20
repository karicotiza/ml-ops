"""MLFlow scoring models repository."""

from typing import Any

from mlflow import (
    log_input,
    log_metrics,
    log_params,
    set_experiment,
    set_tag,
    set_tags,
    set_tracking_uri,
    start_run,
)
from mlflow.data.numpy_dataset import from_numpy as make_dataset_from_numpy
from mlflow.models.signature import infer_signature
from mlflow.sklearn import load_model, log_model
from mlflow.tracking.client import MlflowClient
from numpy import array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_hub.domain.value_objects.metadata import Metadata
from ml_hub.infrastructure.ml.scoring.mlflow import MLFlowScoring

type AnySklearnModel = RandomForestRegressor | RandomForestClassifier


class MLFLowScoringModels:
    """MLFLow scoring models."""

    _description_tag_name: str = "mlflow.note.content"

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
        model: AnySklearnModel,
        metadata: Metadata,
    ) -> None:
        """Add scoring model.

        Args:
            model (BaseEstimator): scoring model instance.
            metadata (Metadata): metadata.

        """
        set_tracking_uri(self._uri)

        self._add_run_info(
            model=model,
            metadata=metadata,
        )

        self._add_model_info(
            metadata=metadata,
        )

    def _add_run_info(
        self,
        model: AnySklearnModel,
        metadata: Metadata,
    ) -> None:
        with start_run(
            experiment_id=self._experiment_id,
            run_name=metadata.run_name,
        ):
            set_tag(self._description_tag_name, metadata.run_description)
            log_metrics(metadata.run_metrics)
            log_params(metadata.run_parameters)
            set_tags(metadata.run_tags)
            log_model(
                sk_model=model,
                name=metadata.model_name,
                registered_model_name=metadata.model_name,
                signature=infer_signature(
                    model_input=array(metadata.dataset_features),
                    model_output=array(
                        object=model.predict(metadata.dataset_features),
                    ),
                ),
                input_example=metadata.dataset_features[:5],
            )
            log_input(
                dataset=make_dataset_from_numpy(
                    features=array(metadata.dataset_features),
                    targets=array(metadata.dataset_targets),
                    name=metadata.dataset_name,
                ),
                context=metadata.dataset_split,
                tags=metadata.dataset_tags,
            )

    def _add_model_info(
        self,
        metadata: Metadata,
    ) -> None:
        client: MlflowClient = MlflowClient()

        latest_version: str = client.get_latest_versions(
            name=metadata.model_name,
        )[0].version

        client.update_model_version(
            name=metadata.model_name,
            version=latest_version,
            description=metadata.version_description,
        )

        client.update_registered_model(
            name=metadata.model_name,
            description=metadata.model_description,
        )

        for tag_name, tag_value in metadata.version_tags.items():
            client.set_model_version_tag(
                name=metadata.model_name,
                version=latest_version,
                key=tag_name,
                value=tag_value,
            )

        for tag_name, tag_value in metadata.model_tags.items():
            client.set_registered_model_tag(
                name=metadata.model_name,
                key=tag_name,
                value=tag_value,
            )
