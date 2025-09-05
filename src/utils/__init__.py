"""
Utilities package.

This package provides utility functions and classes for the application.
"""

from src.utils.logging_service import (
    LoggingService,
    get_logger,
    default_logger
)

from src.utils.error_handling import (
    ApplicationError,
    ConfigurationError,
    DataError,
    APIError,
    ModelError,
    handle_exception,
    exception_handler,
    retry
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
    
    # Error handling
    'ApplicationError',
    'ConfigurationError',
    'DataError',
    'APIError',
    'ModelError',
    'handle_exception',
    'exception_handler',
    'retry',
    
    # Performance
    'performance_monitor',
    'MemoryMonitor',
    'get_memory_monitor',
    'adaptive_batch_processing'
]
