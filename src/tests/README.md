# Tests Module

This module contains test files for the Trading Algorithm System.

## Purpose

The tests module provides a centralized location for tests that span multiple modules or require special setup, including:

- Integration tests
- End-to-end tests
- Performance tests
- Fixtures and test utilities

## Structure

- `__init__.py`: Exports test utilities
- `conftest.py`: Shared pytest fixtures
- `integration/`: Integration tests
- `e2e/`: End-to-end tests
- `performance/`: Performance tests
- `fixtures/`: Test fixtures and data

## Testing Approach

Most tests should be co-located with the code they test. For example:

- `src/models/base_model.py` → `src/models/base_model_test.py`
- `src/services/data_service.py` → `src/services/data_service_test.py`

This module is for tests that don't fit that pattern.

## Running Tests

Tests can be run using pytest:

```bash
# Run all tests
pytest

# Run tests in a specific module
pytest src/models

# Run a specific test file
pytest src/models/base_model_test.py

# Run a specific test function
pytest src/models/base_model_test.py::test_function_name
```

## Test Naming Conventions

- Test files: `*_test.py`
- Test classes: `Test*`
- Test functions: `test_*`

## Dependencies

Tests may depend on any module in the system, as well as testing libraries like pytest, pytest-mock, etc.
