"""Scoring ML model mock."""

from dataclasses import fields
from typing import TYPE_CHECKING, Any

from numpy import array, float64, ndarray
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_hub.domain.value_objects.scoring_features import ScoringFeatures

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sklearn.utils import Bunch


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

        self._training_data: (
            tuple[
                Any | list[Any] | NDArray[float64],
                Any | list[Any] | ndarray[tuple[int], Any],
            ]
            | Bunch
        ) = make_classification(
            n_samples=self._n_samples,
            n_features=self._n_features,
            random_state=random_state,
        )

        self._model.fit(*self._training_data)

    @property
    def model(self) -> RandomForestClassifier:
        """Get model.

        Returns:
            RandomForestClassifier: sklearn's random forest classifier.

        """
        return self._model

    @property
    def dataset_features(self) -> list[list[str | float]]:
        """Get scoring model data.

        Returns:
            dict: data as dict.

        """
        if isinstance(self._training_data, tuple) and isinstance(
            self._training_data[0],
            ndarray,
        ):
            return self._training_data[0].tolist()

        msg: str = "Unexpected type"
        raise TypeError(msg)

    @property
    def dataset_targets(self) -> list[str | float]:
        """Get scoring model data.

        Returns:
            dict: data as dict.

        """
        if isinstance(self._training_data, tuple) and isinstance(
            self._training_data[1],
            ndarray,
        ):
            return self._training_data[1].tolist()

        msg: str = "Unexpected type"
        raise TypeError(msg)

    @property
    def run_parameters(self) -> dict[str, str]:
        """Get scoring model parameters.

        Returns:
            dict: parameters as dict.

        """
        return {
            str(key): str(param_value)
            for key, param_value in self._model.get_params().items()
        }

    def predict(
        self,
        credit_utilization_ratio: float,
        payment_history: float,
        length_of_credit_history: float,
        number_of_open_credit_accounts: float,
    ) -> float:
        """Predict scoring.

        Args:
            credit_utilization_ratio (float): credit_utilization_ratio
            payment_history (float): payment_history
            length_of_credit_history (float): length of credit history
            number_of_open_credit_accounts (float): number of open credit
                accounts.

        Returns:
            float: prediction.

        """
        prediction: float = float(
            self._model.predict(
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
            )[0],
        )

        return prediction
