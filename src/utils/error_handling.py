"""
Error handling module.

This module provides standardized error handling mechanisms for the application,
including custom exception classes, decorators for exception handling and retries,
and utilities for consistent error reporting.
"""

import sys
import time
import traceback
import asyncio
import contextlib
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast, Awaitable

# Import logging service
from src.utils.logging_service import get_logger, log_function_call

# Create logger
logger = get_logger("ErrorHandler")

# Type variables for function annotations
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')
T = TypeVar('T')
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])


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


class ValidationError(ApplicationError):
    """Exception raised for validation errors."""
    pass


class AuthenticationError(ApplicationError):
    """Exception raised for authentication errors."""
    pass


class AuthorizationError(ApplicationError):
    """Exception raised for authorization errors."""
    pass


class NetworkError(ApplicationError):
    """Exception raised for network-related errors."""
    pass


class ResourceError(ApplicationError):
    """Exception raised for resource-related errors (not found, already exists, etc.)."""
    pass


class TimeoutError(ApplicationError):
    """Exception raised for timeout errors."""
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
    jitter: float = 0.1,
    log_retries: bool = True
) -> Callable[[F], F]:
    """
    Decorator for retrying functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_exceptions: Exception types to retry on
        delay_seconds: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by after each retry
        jitter: Random factor to add to delay to prevent thundering herd
        log_retries: Whether to log retry attempts
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import random
            
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
                    
                    # Add jitter to delay
                    jitter_amount = random.uniform(-jitter * current_delay, jitter * current_delay)
                    actual_delay = max(0.001, current_delay + jitter_amount)
                    
                    # Log retry attempt
                    if log_retries:
                        logger.warning(
                            f"Retry attempt {attempts}/{max_attempts} for {func.__name__} "
                            f"after error: {str(e)}. Retrying in {actual_delay:.2f}s"
                        )
                    
                    # Wait before retrying
                    time.sleep(actual_delay)
                    
                    # Increase delay for next attempt
                    current_delay *= backoff_factor
        
        return cast(F, wrapper)
    
    return decorator


async def async_retry(
    max_attempts: int = 3,
    retry_exceptions: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = None,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    log_retries: bool = True
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator for retrying async functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_exceptions: Exception types to retry on
        delay_seconds: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by after each retry
        jitter: Random factor to add to delay to prevent thundering herd
        log_retries: Whether to log retry attempts
        
    Returns:
        Decorated async function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import random
            
            attempts = 0
            current_delay = delay_seconds
            
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
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
                    
                    # Add jitter to delay
                    jitter_amount = random.uniform(-jitter * current_delay, jitter * current_delay)
                    actual_delay = max(0.001, current_delay + jitter_amount)
                    
                    # Log retry attempt
                    if log_retries:
                        logger.warning(
                            f"Retry attempt {attempts}/{max_attempts} for {func.__name__} "
                            f"after error: {str(e)}. Retrying in {actual_delay:.2f}s"
                        )
                    
                    # Wait before retrying
                    await asyncio.sleep(actual_delay)
                    
                    # Increase delay for next attempt
                    current_delay *= backoff_factor
        
        return cast(AsyncF, wrapper)
    
    return decorator


@contextlib.contextmanager
def error_context(
    error_message: str,
    fallback_value: Optional[T] = None,
    expected_exceptions: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = Exception,
    log_exception: bool = True,
    raise_unexpected: bool = True,
    error_code: Optional[str] = None
) -> Any:
    """
    Context manager for handling exceptions.
    
    Args:
        error_message: Message to log if an exception occurs
        fallback_value: Value to return if an exception occurs
        expected_exceptions: Expected exception types to handle
        log_exception: Whether to log the exception
        raise_unexpected: Whether to re-raise unexpected exceptions
        error_code: Optional error code to include in ApplicationError
        
    Yields:
        None
    """
    try:
        yield
    except Exception as e:
        # Check if this is an expected exception
        is_expected = isinstance(e, expected_exceptions) if expected_exceptions else True
        
        if log_exception:
            if isinstance(e, ApplicationError):
                logger.exception(f"{error_message}: {e.message}")
            else:
                logger.exception(f"{error_message}: {str(e)}")
        
        if is_expected and not raise_unexpected:
            return fallback_value
        
        # Wrap in ApplicationError if it's not already one
        if not isinstance(e, ApplicationError):
            raise ApplicationError(
                message=f"{error_message}: {str(e)}",
                error_code=error_code,
                details={"original_error": str(e), "error_type": e.__class__.__name__}
            ) from e
        
        # Re-raise the exception
        raise


@contextlib.asynccontextmanager
async def async_error_context(
    error_message: str,
    fallback_value: Optional[T] = None,
    expected_exceptions: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = Exception,
    log_exception: bool = True,
    raise_unexpected: bool = True,
    error_code: Optional[str] = None
) -> Any:
    """
    Async context manager for handling exceptions.
    
    Args:
        error_message: Message to log if an exception occurs
        fallback_value: Value to return if an exception occurs
        expected_exceptions: Expected exception types to handle
        log_exception: Whether to log the exception
        raise_unexpected: Whether to re-raise unexpected exceptions
        error_code: Optional error code to include in ApplicationError
        
    Yields:
        None
    """
    try:
        yield
    except Exception as e:
        # Check if this is an expected exception
        is_expected = isinstance(e, expected_exceptions) if expected_exceptions else True
        
        if log_exception:
            if isinstance(e, ApplicationError):
                logger.exception(f"{error_message}: {e.message}")
            else:
                logger.exception(f"{error_message}: {str(e)}")
        
        if is_expected and not raise_unexpected:
            return fallback_value
        
        # Wrap in ApplicationError if it's not already one
        if not isinstance(e, ApplicationError):
            raise ApplicationError(
                message=f"{error_message}: {str(e)}",
                error_code=error_code,
                details={"original_error": str(e), "error_type": e.__class__.__name__}
            ) from e
        
        # Re-raise the exception
        raise


def safe_execute(
    func: Callable[..., R],
    *args: Any,
    error_message: str = "Error executing function",
    fallback_value: Optional[R] = None,
    expected_exceptions: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = Exception,
    log_exception: bool = True,
    **kwargs: Any
) -> Optional[R]:
    """
    Safely execute a function and handle exceptions.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        error_message: Message to log if an exception occurs
        fallback_value: Value to return if an exception occurs
        expected_exceptions: Expected exception types to handle
        log_exception: Whether to log the exception
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, expected_exceptions):
            raise
        
        if log_exception:
            logger.exception(f"{error_message}: {str(e)}")
        
        return fallback_value


async def safe_execute_async(
    func: Callable[..., Awaitable[R]],
    *args: Any,
    error_message: str = "Error executing async function",
    fallback_value: Optional[R] = None,
    expected_exceptions: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = Exception,
    log_exception: bool = True,
    **kwargs: Any
) -> Optional[R]:
    """
    Safely execute an async function and handle exceptions.
    
    Args:
        func: Async function to execute
        *args: Arguments to pass to the function
        error_message: Message to log if an exception occurs
        fallback_value: Value to return if an exception occurs
        expected_exceptions: Expected exception types to handle
        log_exception: Whether to log the exception
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function or fallback value
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, expected_exceptions):
            raise
        
        if log_exception:
            logger.exception(f"{error_message}: {str(e)}")
        
        return fallback_value


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
    
    # Using the error context manager
    def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
        with error_context("Error processing data", fallback_value={"status": "error"}):
            if "key" not in data:
                raise KeyError("Missing required key")
            return {"status": "success", "result": data["key"]}
    
    # Using safe_execute
    def divide(x: int, y: int) -> float:
        return x / y
    
    result = safe_execute(
        divide, 10, 0,
        error_message="Division error",
        fallback_value=0.0,
        expected_exceptions=ZeroDivisionError
    )
    
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
    
    # Test error context manager
    print("\nTesting error context manager:")
    print(process_data({"key": "value"}))  # Should succeed
    print(process_data({}))  # Should return fallback value
    
    # Test safe_execute
    print("\nTesting safe_execute:")
    print(f"Result of safe divide: {result}")  # Should be 0.0
