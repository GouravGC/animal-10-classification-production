"""
Custom logging configuration for the project.

Provides a consistent, reusable logger across all modules. The logger writes
both to the console and to a rotating log file inside the `logs/` directory.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from src.constants import PROJECT_ROOT

# ----------------------------------------------------------------------------
# Logger configuration
# ----------------------------------------------------------------------------
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "animals_classification.log")

LOG_FORMAT = (
    "[%(asctime)s] %(levelname)s | %(name)s | %(lineno)d | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _ensure_log_dir() -> None:
    """Create the log directory if it does not exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str = "animals-classification") -> logging.Logger:
    """
    Return a configured logger instance.

    Args:
        name (str): Logger name. Defaults to the project logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    _ensure_log_dir()

    logger = logging.getLogger(name)

    # Avoid duplicate handlers when the logger is re-created.
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        # Console handler.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

        # Rotating file handler.
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        # Prevent propagation to the root logger.
        logger.propagate = False

    return logger


__all__ = ["get_logger"]
