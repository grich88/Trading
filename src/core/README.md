# Core Module

This module contains the core application logic for the Trading Algorithm System.

## Purpose

The core module orchestrates the interaction between different components of the system, including:

- Application lifecycle management
- Workflow coordination
- System-wide event handling
- Integration of models and services

## Structure

- `__init__.py`: Exports the public API of the module
- `app.py`: Main application class
- `workflow.py`: Workflow coordination
- `events.py`: Event handling

## Usage

The core module should be the primary entry point for the application:

```python
from src.core import App

# Initialize the application
app = App()

# Run the application
app.run()
```

## Dependencies

The core module may depend on:

- `utils` module
- `config` module
- `services` module
- `models` module

But it should not depend on the `api` module to avoid circular dependencies.
