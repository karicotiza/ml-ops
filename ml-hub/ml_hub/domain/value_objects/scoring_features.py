"""Scoring features value object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoringFeatures:
    """Scoring features value object."""

    credit_utilization_ratio: float
    payment_history: float
    length_of_credit_history: float
    number_of_open_credit_accounts: float
