"""
Custom exception handling for the project.

Defines a base project exception and a helper to format detailed error
messages (including file, line number and the original exception) for robust
logging and debugging.
"""

from __future__ import annotations

import sys
from typing import Any, Optional


class CustomException(Exception):
    """
    Base exception for the project.

    Wraps the underlying error message with extra context (file and line
    number) to aid debugging in production.
    """

    def __init__(self, error_message: Optional[Any] = None) -> None:
        self.error_message = error_message
        # Capture the current exception information (if any).
        self._file_name: Optional[str] = None
        self._line_number: Optional[int] = None

        if sys.exc_info()[0] is not None:
            _, _, exc_tb = sys.exc_info()
            if exc_tb is not None:
                self._file_name = exc_tb.tb_frame.f_code.co_filename
                self._line_number = exc_tb.tb_lineno

        super().__init__(self._build_message())

    def _build_message(self) -> str:
        """Build a human-readable error message with context."""
        base = f"Error: {self.error_message}"
        if self._file_name and self._line_number:
            base += f" occurred in file: {self._file_name}, line: {self._line_number}"
        return base

    def __str__(self) -> str:
        return self._build_message()


def error_message_detail(error: Exception, error_detail: Any = None) -> str:
    """Helper to build a detailed error message string from an exception."""
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown"
    line_number = exc_tb.tb_lineno if exc_tb else -1
    return (
        f"Error occurred in python script: [{file_name}] "
        f"line number: [{line_number}] error message: [{error}]"
    )


__all__ = ["CustomException", "error_message_detail"]
