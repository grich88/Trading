"""
Logging service module.

This module provides a centralized logging service with configurable
outputs, log levels, and formatting.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Union
from datetime import datetime

# Import configuration
try:
    from src.config import LOG_LEVEL, APP_MODE
except ImportError:
    # Default values if config not available
    LOG_LEVEL = "INFO"
    APP_MODE = "development"


class LoggingService:
    """
    Centralized logging service with configurable outputs and formatting.
    
    This class provides a consistent logging interface across the application
    with support for console output, file output, and configurable log levels.
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
                 backup_count: int = 5):
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
        """
        # Skip initialization if already initialized
        if self._initialized:
            return
        
        self.name = name
        self.log_level_name = log_level.upper()
        self.log_level = self.LOG_LEVELS.get(self.log_level_name, logging.INFO)
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        
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
        
        # Create formatter
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
            file_handler = RotatingFileHandler(
                self.log_file_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self._initialized = True
        
        # Log initialization
        self.logger.info(f"Logging service initialized: {name} ({self.log_level_name})")
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
    
    try:
        # Generate an exception
        result = 1 / 0
    except Exception as e:
        logger.exception(f"An exception occurred: {e}")
