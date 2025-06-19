"""MLFLow models repository integration tests."""

from random import random
from typing import TYPE_CHECKING

import pytest

from ml_hub.domain.value_objects.model_info import ModelInfo
from ml_hub.domain.value_objects.scoring_features import ScoringFeatures
from ml_hub.infrastructure.repos.scoring_models.mlflow import (
    MLFLowScoringModels,
)
from tests.mock.infrastructure.ml.scoring import ScoringModelMock

if TYPE_CHECKING:
    from ml_hub.infrastructure.ml.scoring.mlflow import MLFlowScoring


@pytest.fixture
def mlflow_scoring_models() -> MLFLowScoringModels:
    """Create MLFLowScoringModels fixture.

    Returns:
        MLFLowScoringModels: MLFLowScoringModels instance.

    """
    return MLFLowScoringModels(
        uri="http://localhost:5000",
        experiment="tests-integration",
    )


@pytest.mark.parametrize(
    argnames=(
        "expected_name",
        "expected_version",
    ),
    argvalues=[
        ("scoring-model-001", 5),
    ],
)
def test_mlflow_scoring_models_get(
    expected_name: str,
    expected_version: int,
    mlflow_scoring_models: MLFLowScoringModels,
) -> None:
    """Test MLFLowScoringModels's get method.

    Args:
        expected_name (str): expected name.
        expected_version (int): expected version.
        mlflow_scoring_models (MLFLowScoringModels): MLFLowScoringModels
            fixture.

    """
    for index in range(expected_version):
        mock: ScoringModelMock = ScoringModelMock(index)

        mlflow_scoring_models.add(
            model=mock.model,
            model_info=ModelInfo(
                name=expected_name,
                description="Integration tests model.",
                metrics={
                    "accuracy": random(),  # noqa: S311
                    "auc": random(),  # noqa: S311
                },
                training_parameters=mock.training_parameters,
                tags=[
                    "tests",
                    "integration",
                ],
            ),
        )

    downloaded_model: MLFlowScoring = mlflow_scoring_models.get(
        name=expected_name,
        version="1",
    )

    prediction: float = downloaded_model.predict(
        features=ScoringFeatures(
            credit_utilization_ratio=0,
            payment_history=0,
            length_of_credit_history=0,
            number_of_open_credit_accounts=0,
        ),
    )

    assert isinstance(prediction, float)


@pytest.mark.parametrize(
    argnames=(
        "expected_name",
        "expected_version",
    ),
    argvalues=[
        (
            "scoring-model-002",
            "1",
        ),
    ],
)
def test_mlflow_scoring_models_add(
    expected_name: str,
    expected_version: str,
    mlflow_scoring_models: MLFLowScoringModels,
) -> None:
    """Test MLFLowScoringModels's add method.

    Args:
        expected_name (str): expected name.
        expected_version (str): expected version.
        mlflow_scoring_models (MLFLowScoringModels): MLFLowScoringModels
            fixture.

    """
    mock: ScoringModelMock = ScoringModelMock(0)

    mlflow_scoring_models.add(
        model=mock.model,
        model_info=ModelInfo(
            name=expected_name,
            description="Integration tests model.",
            metrics={
                "metric-a": 1,
                "metric-b": 1,
                "metric-c": 1,
            },
            training_parameters=mock.training_parameters,
            tags=[
                "tests",
                "integration",
            ],
        ),
    )

    downloaded_model: MLFlowScoring = mlflow_scoring_models.get(
        name=expected_name,
        version=expected_version,
    )

    assert downloaded_model.name == expected_name
    assert downloaded_model.version == expected_version
