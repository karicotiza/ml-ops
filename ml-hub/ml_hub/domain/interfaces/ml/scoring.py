"""Scoring ML model interface."""

from typing import Any, Protocol

from sklearn.ensemble import RandomForestClassifier


class ScoringModel(Protocol):
    """Scoring ML model interface."""

    def get_name(self) -> str:
        """Get scoring model name.

        Returns:
            str: name.

        """
        ...

    def get_version(self) -> str | None:
        """Get scoring model version.

        Returns:
            str: version.

        """
        ...

    def get_model(self) -> RandomForestClassifier:
        """Get model.

        Returns:
            RandomForestClassifier: sklearn's random forest classifier.

        """
        ...

    def get_parameters(self) -> dict[Any, Any]:
        """Get scoring model parameters.

        Returns:
            dict[Any, Any]: parameters as dict.

        """
        ...

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
        ...
