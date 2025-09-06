"""
Performance Monitoring

This module provides performance monitoring utilities for the Trading Algorithm System.
"""

import functools
import logging
import time
from typing import Any, Callable, TypeVar, cast

# Type variables for function signatures
F = TypeVar("F", bound=Callable[..., Any])


def performance_timer(func: F) -> F:
    """
    Decorator to measure and log the execution time of a function.

    Args:
        func: The function to measure.

    Returns:
        The decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get the logger for the function's module
        logger = logging.getLogger(func.__module__)

        # Record the start time
        start_time = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # Calculate the execution time
        execution_time = time.time() - start_time

        # Log the execution time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")

        return result

    return cast(F, wrapper)