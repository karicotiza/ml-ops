"""Model info value object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Model info value object."""

    name: str
    description: str
    metrics: dict[str, float]
    training_parameters: dict[str, str]
    tags: list[str]
