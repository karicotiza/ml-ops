"""Random forest scoring ML model."""

from numpy import array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_hub.domain.value_objects.scoring_features import ScoringFeatures

type AnySklearnModel = RandomForestRegressor | RandomForestClassifier


class MLFlowScoring:
    """MLFlow scoring ML model."""

    def __init__(
        self,
        name: str,
        version: str,
        model: AnySklearnModel,
    ) -> None:
        """Create new instance.

        Args:
            name (str): model name.
            version (str): model version.
            model (Sklearn): model instance.

        """
        self._name: str = name
        self._version: str = version
        self._model: AnySklearnModel = model

    @property
    def name(self) -> str:
        """Get scoring model name.

        Returns:
            str: name.

        """
        return self._name

    @property
    def version(self) -> str | None:
        """Get scoring model version.

        Returns:
            str: version.

        """
        return self._version

    def predict(self, features: ScoringFeatures) -> float:
        """Predict scoring.

        Args:
            features (ScoringFeatures): scoring features.

        Returns:
            float: prediction.

        """
        return float(
            self._model.predict(
                X=array(
                    object=[
                        [
                            features.credit_utilization_ratio,
                            features.payment_history,
                            features.length_of_credit_history,
                            features.number_of_open_credit_accounts,
                        ],
                    ],
                ),
            )[0],
        )
