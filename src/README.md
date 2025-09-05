# Source Code

This directory contains the source code for the Trading Algorithm System.

## Directory Structure

- `api/`: API endpoints and interfaces
- `config/`: Configuration management
- `core/`: Core application logic
- `models/`: Trading models and algorithms
- `services/`: Shared services
- `tests/`: Test files (co-located with source)
- `utils/`: Utility functions and helpers

## Module Organization

Each module follows a consistent structure:

1. `__init__.py`: Exports the public API of the module
2. Module-specific files and submodules
3. Tests co-located with the code they test

## Import Structure

To avoid circular dependencies, imports should follow this hierarchy:

1. `utils` can be imported by any module
2. `config` can be imported by any module except `utils`
3. `services` can import `utils` and `config`
4. `models` can import `utils`, `config`, and `services`
5. `core` can import `utils`, `config`, `services`, and `models`
6. `api` can import any module

## Naming Conventions

- Module names: lowercase, underscore_separated
- Class names: CamelCase
- Function names: lowercase, underscore_separated
- Constants: UPPERCASE, UNDERSCORE_SEPARATED
