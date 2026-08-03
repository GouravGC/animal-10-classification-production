"""
Image transforms / pre-processing.

This module reproduces the EXACT inference pre-processing pipeline used in the
authoritative notebook (the `raw_test_transform` from cells 6 and 8):

    Resize((IMAGE_SIZE, IMAGE_SIZE))
    -> ToTensor()
    -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

IMPORTANT: These transforms must NOT be changed. They are the exact pipeline
used to train and evaluate the best model, so they are required to produce
identical predictions.
"""

from __future__ import annotations

from typing import List

from torchvision import transforms

from src.constants import IMAGE_SIZE, MEAN, STD


def get_inference_transform(
    image_size: int = IMAGE_SIZE,
    mean: List[float] = MEAN,
    std: List[float] = STD,
) -> transforms.Compose:
    """
    Return the inference (test) transform, identical to the notebook's
    `raw_test_transform`.

    Args:
        image_size (int): Target size (both width and height). Defaults to 224.
        mean (List[float]): Normalization mean. Defaults to ImageNet mean.
        std (List[float]): Normalization std. Defaults to ImageNet std.

    Returns:
        transforms.Compose: The composed transform pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


__all__ = ["get_inference_transform"]
