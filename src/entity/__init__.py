"""
Entity / configuration data structures.

These lightweight dataclasses centralise the configuration values loaded from
the saved artifacts (artifacts/reports/config.json) and the fixed hyper-parameters
defined in the notebook. Using typed dataclasses makes the configuration
explicit, self-documenting and easy to consume across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Configuration describing the trained model and its pre-processing."""

    image_size: int
    batch_size: int
    num_classes: int
    class_names: List[str]
    mean: List[float]
    std: List[float]

    # Populated from the classification report (best model = AlexNet Raw).
    best_model: str = "AlexNet"
    best_f1_score: float = 0.7377
    best_test_accuracy: float = 0.7393

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelConfig":
        """Create a ModelConfig from a config dictionary (e.g. config.json)."""
        return cls(
            image_size=int(data["IMAGE_SIZE"]),
            batch_size=int(data["BATCH_SIZE"]),
            num_classes=int(data["NUM_CLASSES"]),
            class_names=list(data["CLASS_NAMES"]),
            mean=list(data["MEAN"]),
            std=list(data["STD"]),
        )


@dataclass(frozen=True)
class PredictionResult:
    """Result of a single inference."""

    predicted_index: int
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    top_k: List[tuple] = field(default_factory=list)

    # OOD / calibration metadata (added to improve inference UX without
    # modifying the trained model).
    is_ood: bool = False
    confidence_threshold: float = 0.75
    entropy: float = 0.0
    inference_time_ms: float = 0.0

    @property
    def top_classes(self) -> List[str]:
        """Return the ordered list of class names (highest confidence first)."""
        return [name for name, _ in self.top_k]

    @property
    def top_confidences(self) -> List[float]:
        """Return the ordered list of confidences (highest first)."""
        return [conf for _, conf in self.top_k]

    @property
    def ood_message(self) -> str:
        """Return a user-friendly message when the image is out-of-distribution."""
        if self.is_ood:
            return (
                "This image does not appear to belong to any of the supported "
                "animal classes. The model confidence is below the "
                f"{self.confidence_threshold * 100:.0f}% threshold."
            )
        return ""


@dataclass(frozen=True)
class InferenceInput:
    """Represents a single inference input (image path or PIL image)."""

    image_path: Optional[str] = None
    image = None  # type: ignore  # PIL or numpy image


__all__ = ["ModelConfig", "PredictionResult", "InferenceInput"]
