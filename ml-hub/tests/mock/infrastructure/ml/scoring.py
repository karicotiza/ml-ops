"""Scoring ML model mock."""

from typing import TYPE_CHECKING, Any

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ScoringModelMock:
    """Scoring ML model mock."""

    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 3,
        random_state: int = 42,
        n_samples: int = 100,
        n_features: int = 5,
    ) -> None:
        """Create new instance.

        Args:
            n_estimators (int, optional): number of estimators. Defaults to 5.
            max_depth (int, optional): max depth. Defaults to 3.
            random_state (int, optional): random state. Defaults to 42.
            n_samples (int, optional): number of samples. Defaults to 100.
            n_features (int, optional): number of features. Defaults to 5.

        """
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        self._train(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
        )

    def _train(
        self,
        n_samples: int,
        n_features: int,
        random_state: int,
    ) -> None:
        training_data: tuple[NDArray[Any], NDArray[Any]] = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
        )

        self.model.fit(
            X=training_data[0],
            y=training_data[1],
        )
