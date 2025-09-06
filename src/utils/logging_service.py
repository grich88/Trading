"""
Logging Service

This module provides centralized logging functionality for the Trading Algorithm System.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Union

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log level
DEFAULT_LEVEL = logging.INFO

# Default log file
DEFAULT_LOG_FILE = "app.log"

# Maximum log file size (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Maximum number of backup log files
BACKUP_COUNT = 5


def setup_logger(
    name: str,
    level: Optional[Union[int, str]] = None,
    log_format: str = DEFAULT_FORMAT,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: The name of the logger.
        level: The log level. Defaults to the value from the LOG_LEVEL environment variable,
               or INFO if not set.
        log_format: The log format string.
        log_file: The log file path. If None, logs will only be sent to the console.
        console: Whether to log to the console.

    Returns:
        The configured logger.
    """
    # Get the logger
    logger = logging.getLogger(name)

    # Set the log level
    if level is None:
        level = os.environ.get("LOG_LEVEL", DEFAULT_LEVEL)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Create the directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger