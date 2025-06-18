"""Random forest scoring ML model code."""

from typing import Any

from numpy import array
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    """Scoring ML model interface."""

    def __init__(
        self,
        model_name: str,
        model_version: str,
        model_instance: RandomForestClassifier,
    ) -> None:
        """Create new instance.

        Args:
            model_name (str): model name.
            model_version (str): model version.
            model_instance (RandomForestClassifier): model instance.

        """
        self._model_name: str = model_name
        self._model_version: str = model_version
        self._model_instance: RandomForestClassifier = model_instance

    def get_name(self) -> str:
        """Get scoring model name.

        Returns:
            str: name.

        """
        return self._model_name

    def get_version(self) -> str | None:
        """Get scoring model version.

        Returns:
            str: version.

        """
        return self._model_version

    def get_model(self) -> RandomForestClassifier:
        """Get model.

        Returns:
            RandomForestClassifier: sklearn's random forest classifier.

        """
        return self._model_instance

    def get_parameters(self) -> dict[Any, Any]:
        """Get scoring model parameters.

        Returns:
            dict[Any, Any]: parameters as dict.

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
