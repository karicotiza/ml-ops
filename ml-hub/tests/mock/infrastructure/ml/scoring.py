"""Scoring ML model mock."""

from typing import Any

from numpy import array
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


class ScoringModelMock:
    """Scoring ML model mock."""

    _n_estimators: int = 5
    _max_depth: int = 3
    _n_samples: int = 100
    _n_features: int = 4

    def __init__(self, random_state: int) -> None:
        """Create new instance.

        Args:
            random_state (int, optional): random state.

        """
        self._model_instance: RandomForestClassifier = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=random_state,
        )

        self._model_instance.fit(
            *make_classification(
                n_samples=self._n_samples,
                n_features=self._n_features,
                random_state=random_state,
            ),
        )

    def get_model(self) -> RandomForestClassifier:
        """Get model.

        Returns:
            RandomForestClassifier: sklearn's random forest classifier.

        """
        return self._model_instance

    def get_parameters(self) -> dict[Any, Any]:
        """Get scoring model parameters.

        Returns:
            dict: parameters as dict.

        """
        return self._model_instance.get_params()

    def predict(
        self,
        credit_utilization_ratio: float,
        payment_history: float,
        length_of_credit_history: float,
        number_of_open_credit_accounts: float,
    ) -> bool:
        """Predict scoring.

        Args:
            credit_utilization_ratio (float): credit_utilization_ratio
            payment_history (float): payment_history
            length_of_credit_history (float): length of credit history
            number_of_open_credit_accounts (float): number of open credit
                accounts.

        Returns:
            bool: prediction.

        """
        prediction: int = self._model_instance.predict(
            X=array(
                object=[
                    [
                        credit_utilization_ratio,
                        payment_history,
                        length_of_credit_history,
                        number_of_open_credit_accounts,
                    ],
                ],
            ),
        )[0]

        return bool(prediction)
