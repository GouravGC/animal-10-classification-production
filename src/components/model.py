"""
Model architectures.

This module contains the exact CNN architectures defined in the authoritative
notebook (notebooks/animals-10-classification-updated.ipynb, cell 19 for
AlexNet). The best model is the custom AlexNet, trained on the RAW dataset.

IMPORTANT: The architecture MUST NOT be modified. It is identical to the
notebook and is required to load the pre-trained weights and produce identical
predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.logger import get_logger

logger = get_logger(__name__)


class AlexNet(nn.Module):
    """
    Custom AlexNet model (exact copy from the notebook).

    The architecture is defined as:
        features: Sequential of Conv/BatchNorm/ReLU/MaxPool blocks.
        classifier: Sequential of Flatten/Linear/ReLU/Dropout/Linear.

    This is the model that achieved the best F1 score (AlexNet_Raw) and was
    saved as `best_model.pth` and `best_model.onnx`.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------------------
# The other architectures defined in the notebook (MiniCNN, LeNet) are kept
# here for completeness and reproducibility of the training pipeline. They are
# NOT used for inference (the best model is AlexNet).
# ----------------------------------------------------------------------------
class MiniCNN(nn.Module):
    """MiniCNN architecture (exact copy from the notebook, cell 17)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    """LeNet architecture (exact copy from the notebook, cell 18)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def initialize_weights(model: nn.Module) -> None:
    """
    Reproduce the notebook's weight initialization (cell 21).

    Args:
        model (nn.Module): Model whose weights will be initialized.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


__all__ = ["AlexNet", "MiniCNN", "LeNet", "initialize_weights"]
