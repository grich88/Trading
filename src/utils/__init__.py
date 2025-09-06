"""
Utilities package.

This package provides utility functions and classes for the application.
"""

from src.utils.logging_service import (
    LoggingService,
    get_logger,
    default_logger,
    log_function_call,
    JsonFormatter,
    configure_root_logger
)

from src.utils.error_handling import (
    ApplicationError,
    ConfigurationError,
    DataError,
    APIError,
    ModelError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NetworkError,
    ResourceError,
    TimeoutError,
    handle_exception,
    exception_handler,
    retry,
    async_retry,
    error_context,
    async_error_context,
    safe_execute,
    safe_execute_async
)

from src.utils.performance import (
    performance_monitor,
    MemoryMonitor,
    get_memory_monitor,
    adaptive_batch_processing
)

__all__ = [
    # Logging
    'LoggingService',
    'get_logger',
    'default_logger',
    'log_function_call',
    'JsonFormatter',
    'configure_root_logger',
    
    # Error handling
    'ApplicationError',
    'ConfigurationError',
    'DataError',
    'APIError',
    'ModelError',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'NetworkError',
    'ResourceError',
    'TimeoutError',
    'handle_exception',
    'exception_handler',
    'retry',
    'async_retry',
    'error_context',
    'async_error_context',
    'safe_execute',
    'safe_execute_async',
    
    # Performance
    'performance_monitor',
    'MemoryMonitor',
    'get_memory_monitor',
    'adaptive_batch_processing'
]
