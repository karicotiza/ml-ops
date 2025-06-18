"""MLFLow models repository integration tests."""

from random import random
from typing import TYPE_CHECKING

import pytest
from sklearn.ensemble import RandomForestClassifier

from ml_hub.infrastructure.repos.scoring_models.mlflow import ScoringModels
from tests.mock.infrastructure.ml.scoring import ScoringModelMock

if TYPE_CHECKING:
    from ml_hub.infrastructure.ml.scoring.random_forest import RandomForest


@pytest.fixture
def scoring_models() -> ScoringModels:
    """Create ScoringModels fixture.

    Returns:
        ScoringModels: ScoringModels instance.

    """
    return ScoringModels(
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
def test_scoring_models_get(
    expected_name: str,
    expected_version: int,
    scoring_models: ScoringModels,
) -> None:
    """Test ScoringModels's get method.

    Args:
        expected_name (str): expected name.
        expected_version (int): expected version.
        scoring_models (ScoringModels): ScoringModels fixture.

    """
    for index in range(expected_version):
        model: ScoringModelMock = ScoringModelMock(index)

        scoring_models.add(
            model=model.get_model(),
            model_name=expected_name,
            model_metrics={
                "accuracy": random(),  # noqa: S311
                "auc": random(),  # noqa: S311
            },
            model_parameters=model.get_parameters(),
        )

    downloaded_model: RandomForest = scoring_models.get(
        name=expected_name,
        version="1",
    )

    prediction: bool = downloaded_model.predict(
        credit_utilization_ratio=0,
        payment_history=0,
        length_of_credit_history=0,
        number_of_open_credit_accounts=0,
    )

    assert isinstance(prediction, bool)


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
def test_scoring_models_add(
    expected_name: str,
    expected_version: str,
    scoring_models: ScoringModels,
) -> None:
    """Test ScoringModels's add method.

    Args:
        expected_name (str): expected name.
        expected_version (str): expected version.
        scoring_models (ScoringModels): ScoringModels fixture.

    """
    model: ScoringModelMock = ScoringModelMock(0)

    scoring_models.add(
        model=model.get_model(),
        model_name=expected_name,
        model_metrics={
            "metric-a": 1,
            "metric-b": 1,
            "metric-c": 1,
        },
        model_parameters=model.get_parameters(),
    )

    downloaded_model: RandomForest = scoring_models.get(
        name=expected_name,
        version=expected_version,
    )

    assert downloaded_model.get_name() == expected_name
    assert downloaded_model.get_version() == expected_version
    assert isinstance(downloaded_model.get_parameters(), dict)
    assert isinstance(downloaded_model.get_model(), RandomForestClassifier)
