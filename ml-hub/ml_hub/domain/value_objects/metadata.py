"""Metadata value object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Metadata:
    """Metadata value object."""

    model_name: str
    model_description: str
    model_tags: dict[str, str]

    version_description: str
    version_tags: dict[str, str]

    run_name: str
    run_description: str
    run_tags: dict[str, str]
    run_metrics: dict[str, float]
    run_parameters: dict[str, str]

    dataset_name: str
    dataset_tags: dict[str, str]
    dataset_features: list[list[str | float]]
    dataset_targets: list[str | float]
    dataset_split: str
