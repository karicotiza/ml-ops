"""MLFLow models repository integration tests."""

from typing import TYPE_CHECKING

import pytest

from ml_hub.infrastructure.repos.scoring_models.mlflow import ScoringModels
from tests.mock.infrastructure.ml.scoring import ScoringModelMock

if TYPE_CHECKING:
    from ml_hub.domain.interfaces.ml.scoring import ScoringModel


@pytest.fixture
def scoring_models() -> ScoringModels:
    """Create ScoringModels fixture.

    Returns:
        ScoringModels: ScoringModels instance.

    """
    return ScoringModels(
        tracking_uri="http://localhost:5000",
    )


@pytest.mark.parametrize(
    argnames="expected_amount_of_models",
    argvalues=[
        3,
    ],
)
def test_scoring_models_all(
    expected_amount_of_models: int,
    scoring_models: ScoringModels,
) -> None:
    """Test ScoringModels's all method.

    Args:
        expected_amount_of_models (int): expected amount of models.
        scoring_models (ScoringModels): ScoringModels fixture.

    """
    for index in range(expected_amount_of_models):
        scoring_mock: ScoringModelMock = ScoringModelMock(random_state=index)
        scoring_models.add(
            model_name=f"scoring-mock-{index}",
            experiment_name="integration_tests",
            scoring_model=scoring_mock,
        )

    models: list[ScoringModel] = scoring_models.all()

    print("!!!", models)

    assert isinstance(models, list)
    assert len(models) == expected_amount_of_models
