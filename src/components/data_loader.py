"""
Data loading utilities.

Reproduces the notebook's `CustomSubset` wrapper (cell 10) and provides helpers
to build the raw/augmented datasets and dataloaders for the training and
evaluation pipelines (reproducibility only).
"""

from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.constants import BATCH_SIZE, NUM_WORKERS, SEED
from src.logger import get_logger

logger = get_logger(__name__)


class CustomSubset(Dataset):
    """
    Dataset wrapper that applies a transform to a subset (exact copy from the
    notebook, cell 10).
    """

    def __init__(self, subset: Dataset, transform: Optional[object] = None) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_image_folder(root: str, transform: Optional[object] = None) -> ImageFolder:
    """
    Load a torchvision ImageFolder dataset (as in the notebook, cell 9).

    Args:
        root (str): Root directory of the dataset.
        transform (Optional[object]): Transform to apply (None in the notebook).

    Returns:
        ImageFolder: The loaded dataset.
    """
    dataset = ImageFolder(root=root, transform=transform)
    logger.info("Loaded ImageFolder from %s with %d classes", root, len(dataset.classes))
    return dataset


def build_dataloaders(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for a dataset (mirrors notebook DataLoader setup).

    Args:
        dataset (Dataset): The dataset to wrap.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


__all__ = ["CustomSubset", "load_image_folder", "build_dataloaders"]
