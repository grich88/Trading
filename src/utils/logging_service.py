"""
Logging service module.

This module provides a centralized logging service with configurable
outputs, log levels, and formatting. It supports console logging, file logging,
and JSON formatting for structured logging.
"""

import os
import sys
import json
import logging
import traceback
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any, Union, List, Callable
from datetime import datetime
import functools

# Import configuration
try:
    from src.config import Config
    LOG_LEVEL = Config.get_str("LOG_LEVEL", "INFO")
    APP_MODE = Config.get_str("APP_MODE", "development")
except ImportError:
    # Default values if config not available
    LOG_LEVEL = "INFO"
    APP_MODE = "development"


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    This formatter outputs log records as JSON objects for easier parsing
    and integration with log aggregation systems.
    """
    
    def __init__(self, include_timestamp: bool = True):
        """
        Initialize the JSON formatter.
        
        Args:
            include_timestamp: Whether to include a timestamp in the log
        """
        super().__init__()
        self.include_timestamp = include_timestamp
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON string representation of the log record
        """
        log_data = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        
        if self.include_timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(record.created).isoformat()
        
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        if hasattr(record, "extra") and record.extra:
            log_data["extra"] = record.extra
        
        return json.dumps(log_data)


class LoggingService:
    """
    Centralized logging service with configurable outputs and formatting.
    
    This class provides a consistent logging interface across the application
    with support for console output, file output, JSON formatting, and configurable log levels.
    """
    
    # Log level mapping
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    # Singleton instance
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(LoggingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, 
                 name: str = "TradingSystem", 
                 log_level: str = LOG_LEVEL,
                 log_to_console: bool = True,
                 log_to_file: bool = True,
                 log_file_path: Optional[str] = None,
                 max_file_size_mb: int = 10,
                 backup_count: int = 5,
                 use_json_format: bool = False,
                 use_timed_rotation: bool = False,
                 rotation_interval: str = 'D',
                 include_context: bool = True):
        """
        Initialize the logging service.
        
        Args:
            name: Logger name
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_file_path: Path to log file (default: logs/{name}_{date}.log)
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup log files to keep
            use_json_format: Whether to use JSON formatting for logs
            use_timed_rotation: Whether to use time-based log rotation instead of size-based
            rotation_interval: Interval for timed rotation ('S', 'M', 'H', 'D', 'W0'-'W6', 'midnight')
            include_context: Whether to include context information in logs (e.g., app mode)
        """
        # Skip initialization if already initialized
        if self._initialized:
            return
        
        self.name = name
        self.log_level_name = log_level.upper()
        self.log_level = self.LOG_LEVELS.get(self.log_level_name, logging.INFO)
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.use_json_format = use_json_format
        self.use_timed_rotation = use_timed_rotation
        self.rotation_interval = rotation_interval
        self.include_context = include_context
        
        # Set up log file path
        if log_file_path is None:
            os.makedirs("logs", exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            self.log_file_path = f"logs/{name.lower()}_{date_str}.log"
        else:
            self.log_file_path = log_file_path
            # Ensure directory exists
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self.backup_count = backup_count
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters
        if use_json_format:
            formatter = JsonFormatter(include_timestamp=True)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Add console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler
        if log_to_file:
            if use_timed_rotation:
                file_handler = TimedRotatingFileHandler(
                    self.log_file_path,
                    when=rotation_interval,
                    backupCount=self.backup_count
                )
            else:
                file_handler = RotatingFileHandler(
                    self.log_file_path,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count
                )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self._initialized = True
        
        # Log initialization
        init_message = f"Logging service initialized: {name} ({self.log_level_name})"
        
        # Add context information if requested
        if include_context:
            context = {
                "app_mode": APP_MODE,
                "json_format": use_json_format,
                "timed_rotation": use_timed_rotation
            }
            self.logger.info(init_message, extra={"extra": context})
        else:
            self.logger.info(init_message)
            
        if log_to_file:
            self.logger.info(f"Logging to file: {self.log_file_path}")
    
    def get_logger(self) -> logging.Logger:
        """
        Get the logger instance.
        
        Returns:
            The logger instance
        """
        return self.logger
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a debug message.
        
        Args:
            message: The message to log
            extra: Extra data to include in the log
        """
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an info message.
        
        Args:
            message: The message to log
            extra: Extra data to include in the log
        """
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning message.
        
        Args:
            message: The message to log
            extra: Extra data to include in the log
        """
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error message.
        
        Args:
            message: The message to log
            extra: Extra data to include in the log
        """
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a critical message.
        
        Args:
            message: The message to log
            extra: Extra data to include in the log
        """
        self.logger.critical(message, extra=extra)
    
    def exception(self, message: str, exc_info: bool = True, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an exception message.
        
        Args:
            message: The message to log
            exc_info: Whether to include exception info
            extra: Extra data to include in the log
        """
        self.logger.exception(message, exc_info=exc_info, extra=extra)


def log_function_call(logger=None, level: str = "DEBUG"):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Logger to use (default: create a new logger)
        level: Log level to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create logger
            log = logger or LoggingService(name=func.__module__)
            
            # Get log method based on level
            log_method = getattr(log, level.lower(), log.debug)
            
            # Format arguments for logging
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            
            # Log function call
            log_method(f"Calling {func.__name__}({signature})")
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Log the return value
                log_method(f"{func.__name__} returned {repr(result)}")
                
                return result
            except Exception as e:
                # Log the exception
                log.exception(f"Exception in {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


def configure_root_logger(log_level: str = LOG_LEVEL, 
                         log_to_file: bool = True,
                         log_file_path: Optional[str] = "logs/app.log",
                         use_json_format: bool = False):
    """
    Configure the root logger for the application.
    
    This is useful for third-party libraries that use the root logger.
    
    Args:
        log_level: Log level to use
        log_to_file: Whether to log to a file
        log_file_path: Path to log file
        use_json_format: Whether to use JSON formatting
    """
    # Create a LoggingService instance
    logger = LoggingService(
        name="root",
        log_level=log_level,
        log_to_console=True,
        log_to_file=log_to_file,
        log_file_path=log_file_path,
        use_json_format=use_json_format
    )
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Set the log level
    root_logger.setLevel(logger.log_level)
    
    # Clear existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Add the handlers from our logger
    for handler in logger.logger.handlers:
        root_logger.addHandler(handler)
    
    return root_logger


# Create default logger instance
default_logger = LoggingService()


def get_logger(name: str = "TradingSystem") -> LoggingService:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: The logger name
        
    Returns:
        A LoggingService instance
    """
    return LoggingService(name=name)


if __name__ == "__main__":
    # Example usage
    logger = get_logger("ExampleLogger")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Example with JSON formatting
    json_logger = LoggingService(
        name="JsonLogger",
        use_json_format=True
    )
    json_logger.info("This is a JSON formatted log")
    
    # Example with timed rotation
    timed_logger = LoggingService(
        name="TimedLogger",
        use_timed_rotation=True,
        rotation_interval="H"  # Rotate every hour
    )
    timed_logger.info("This log will be rotated hourly")
    
    # Example with decorator
    @log_function_call()
    def example_function(a, b, c=None):
        return a + b + (c or 0)
    
    result = example_function(1, 2, c=3)
    
    try:
        # Generate an exception
        result = 1 / 0
    except Exception as e:
        logger.exception(f"An exception occurred: {e}")
