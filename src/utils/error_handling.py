"""
Error Handling

This module provides error handling utilities for the Trading Algorithm System.
"""

import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

# Type variables for function signatures
F = TypeVar("F", bound=Callable[..., Any])
R = TypeVar("R")


class AppError(Exception):
    """Base exception class for application errors."""

    def __init__(
        self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the error.

        Args:
            message: The error message.
            status_code: The HTTP status code.
            details: Additional error details.
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AppError):
    """Exception raised for validation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.

        Args:
            message: The error message.
            details: Additional error details.
        """
        super().__init__(message, 400, details)


class NotFoundError(AppError):
    """Exception raised when a resource is not found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.

        Args:
            message: The error message.
            details: Additional error details.
        """
        super().__init__(message, 404, details)


class AuthenticationError(AppError):
    """Exception raised for authentication errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.

        Args:
            message: The error message.
            details: Additional error details.
        """
        super().__init__(message, 401, details)


class AuthorizationError(AppError):
    """Exception raised for authorization errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.

        Args:
            message: The error message.
            details: Additional error details.
        """
        super().__init__(message, 403, details)


def handle_errors(
    error_types: Union[Type[Exception], tuple] = Exception,
    default_message: str = "An error occurred",
    log_level: int = logging.ERROR,
    reraise: bool = False,
    default_return: Optional[Any] = None,
) -> Callable[[F], F]:
    """
    Decorator to handle errors in functions.

    Args:
        error_types: The exception type(s) to catch.
        default_message: The default error message.
        log_level: The log level for errors.
        reraise: Whether to reraise the error after handling.
        default_return: The default return value if an error occurs and reraise is False.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except error_types as e:
                # Get the logger for the function's module
                logger = logging.getLogger(func.__module__)

                # Log the error
                error_message = str(e) or default_message
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {error_message}",
                    exc_info=True,
                )

                # Reraise the error if requested
                if reraise:
                    raise

                # Return the default value
                return default_return

        return cast(F, wrapper)

    return decorator