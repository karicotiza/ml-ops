"""Scoring ML model mock."""

from dataclasses import fields

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_hub.domain.value_objects.scoring_features import ScoringFeatures


class ScoringModelMock:
    """Scoring ML model mock."""

    _n_estimators: int = 5
    _max_depth: int = 3
    _n_samples: int = 100
    _n_features: int = len(fields(ScoringFeatures))

    def __init__(self, random_state: int) -> None:
        """Create new instance.

        Args:
            random_state (int, optional): random state.

        """
        self._model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=random_state,
        )

        self._model.fit(
            *make_classification(
                n_samples=self._n_samples,
                n_features=self._n_features,
                random_state=random_state,
            ),
        )

    @property
    def model(self) -> RandomForestClassifier:
        """Get model.

        Returns:
            RandomForestClassifier: sklearn's random forest classifier.

        """
        return self._model

    @property
    def training_parameters(self) -> dict[str, str]:
        """Get scoring model parameters.

        Returns:
            dict: parameters as dict.

        """
        return {
            str(key): str(param_value)
            for key, param_value in self._model.get_params().items()
        }
