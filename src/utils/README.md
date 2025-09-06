# Utils Module

This module contains utility functions and helpers for the Trading Algorithm System.

## Purpose

The utils module provides common utility functions that can be used by any other module in the system, including:

- Logging
- Error handling
- Performance monitoring
- Date and time utilities
- Math and statistics utilities
- File and I/O utilities

## Structure

- `__init__.py`: Exports the public API of the module
- `logging_service.py`: Centralized logging functionality
- `error_handling.py`: Error handling utilities
- `performance.py`: Performance monitoring utilities
- `date_utils.py`: Date and time utilities
- `math_utils.py`: Math and statistics utilities
- `file_utils.py`: File and I/O utilities

## Usage

Utilities should be simple, focused functions or classes that solve a specific problem:

```python
from src.utils import setup_logger, performance_timer

# Set up logging
logger = setup_logger(__name__)

# Use performance timer
@performance_timer
def my_function():
    # Function implementation
    pass
```

## Dependencies

The utils module should have minimal dependencies on external libraries and no dependencies on other modules in the system to avoid circular dependencies.
