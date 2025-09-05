"""
Error handling module.

This module provides standardized error handling mechanisms for the application.
"""

import sys
import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

# Import logging service
from src.utils.logging_service import get_logger

# Create logger
logger = get_logger("ErrorHandler")

# Type variables for function annotations
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')


class ApplicationError(Exception):
    """Base exception class for application-specific errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            error_code: Optional error code
            details: Optional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ConfigurationError(ApplicationError):
    """Exception raised for configuration errors."""
    pass


class DataError(ApplicationError):
    """Exception raised for data-related errors."""
    pass


class APIError(ApplicationError):
    """Exception raised for API-related errors."""
    pass


class ModelError(ApplicationError):
    """Exception raised for model-related errors."""
    pass


def handle_exception(e: Exception, log_exception: bool = True) -> Dict[str, Any]:
    """
    Handle an exception and return a standardized error response.
    
    Args:
        e: The exception to handle
        log_exception: Whether to log the exception
        
    Returns:
        A standardized error response dictionary
    """
    # Get exception details
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    # Create error response
    error_response = {
        "success": False,
        "error": str(e),
        "error_type": e.__class__.__name__
    }
    
    # Add error code and details if available
    if isinstance(e, ApplicationError):
        if e.error_code:
            error_response["error_code"] = e.error_code
        if e.details:
            error_response["details"] = e.details
    
    # Log exception if requested
    if log_exception:
        logger.error(f"Exception occurred: {str(e)}")
        if exc_traceback:
            traceback_str = ''.join(traceback.format_tb(exc_traceback))
            logger.error(f"Traceback:\n{traceback_str}")
    
    return error_response


def exception_handler(
    fallback_return: Optional[Any] = None,
    expected_exceptions: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = None,
    log_exception: bool = True,
    raise_unexpected: bool = False
) -> Callable[[F], F]:
    """
    Decorator for handling exceptions in functions.
    
    Args:
        fallback_return: Value to return if an exception occurs
        expected_exceptions: Expected exception types to handle
        log_exception: Whether to log the exception
        raise_unexpected: Whether to re-raise unexpected exceptions
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if this is an expected exception
                is_expected = expected_exceptions is None or isinstance(e, expected_exceptions)
                
                # Handle the exception
                if is_expected or not raise_unexpected:
                    # Log the exception if requested
                    if log_exception:
                        logger.exception(f"Exception in {func.__name__}: {str(e)}")
                    
                    # Return fallback value
                    return fallback_return
                else:
                    # Re-raise unexpected exceptions
                    raise
        
        return cast(F, wrapper)
    
    return decorator


def retry(
    max_attempts: int = 3,
    retry_exceptions: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = None,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    log_retries: bool = True
) -> Callable[[F], F]:
    """
    Decorator for retrying functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_exceptions: Exception types to retry on
        delay_seconds: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by after each retry
        log_retries: Whether to log retry attempts
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            
            attempts = 0
            current_delay = delay_seconds
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    
                    # Check if we should retry this exception
                    if retry_exceptions is not None and not isinstance(e, retry_exceptions):
                        raise
                    
                    # Check if we've reached max attempts
                    if attempts >= max_attempts:
                        if log_retries:
                            logger.error(f"Max retry attempts ({max_attempts}) reached for {func.__name__}")
                        raise
                    
                    # Log retry attempt
                    if log_retries:
                        logger.warning(
                            f"Retry attempt {attempts}/{max_attempts} for {func.__name__} "
                            f"after error: {str(e)}. Retrying in {current_delay:.2f}s"
                        )
                    
                    # Wait before retrying
                    time.sleep(current_delay)
                    
                    # Increase delay for next attempt
                    current_delay *= backoff_factor
        
        return cast(F, wrapper)
    
    return decorator


if __name__ == "__main__":
    # Example usage
    
    # Using the exception handler decorator
    @exception_handler(fallback_return={"status": "error"})
    def risky_function(x: int, y: int) -> Dict[str, Any]:
        result = x / y
        return {"status": "success", "result": result}
    
    # Using the retry decorator
    @retry(max_attempts=3, retry_exceptions=(ConnectionError, TimeoutError))
    def api_call(endpoint: str) -> Dict[str, Any]:
        # Simulate API call that might fail
        import random
        if random.random() < 0.7:
            raise ConnectionError("Connection failed")
        return {"status": "success", "data": "API response"}
    
    # Test exception handler
    print("Testing exception handler:")
    print(risky_function(10, 2))  # Should succeed
    print(risky_function(10, 0))  # Should return fallback value
    
    # Test retry decorator
    print("\nTesting retry decorator:")
    try:
        result = api_call("https://example.com/api")
        print(result)
    except Exception as e:
        print(f"API call failed after retries: {e}")
