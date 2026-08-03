"""
Utility helpers used across the project.

These helpers reproduce the exact reproducibility and helper behaviour from the
authoritative notebook (e.g. seed setting, parameter counting, JSON loading).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch

from src.constants import SEED
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = SEED) -> None:
    """
    Reproduce the notebook's seed-setting logic to ensure deterministic
    behaviour. Exact copy of the notebook's `set_seed` function.

    Args:
        seed (int): Random seed. Defaults to 42 (from the notebook).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model. Exact copy of the
    notebook's `count_parameters` function.

    Args:
        model (torch.nn.Module): Model to count parameters for.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_json(path: str) -> Any:
    """
    Load and return the contents of a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON content.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        raise CustomException(f"Failed to load JSON from {path}: {exc}")


def load_class_names(path: str) -> List[str]:
    """
    Load the class names from a JSON list file.

    Args:
        path (str): Path to the class_names.json file.

    Returns:
        List[str]: Ordered list of class names.
    """
    data = load_json(path)
    if not isinstance(data, list):
        raise CustomException(f"Expected a list of class names in {path}")
    return [str(name) for name in data]


def load_config(path: str) -> Dict[str, Any]:
    """
    Load the model configuration from a config JSON file.

    Args:
        path (str): Path to the config.json file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    data = load_json(path)
    if not isinstance(data, dict):
        raise CustomException(f"Expected a dict config in {path}")
    return data


def ensure_directory(path: str) -> None:
    """
    Create a directory (and parents) if it does not exist.

    Args:
        path (str): Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


__all__ = [
    "set_seed",
    "count_parameters",
    "load_json",
    "load_class_names",
    "load_config",
    "ensure_directory",
]
