# Configuration Module

This module handles configuration management for the Trading Algorithm System.

## Purpose

The configuration module provides a centralized way to manage application settings, including:

- Environment-specific configuration
- Feature flags
- API credentials
- Performance tuning parameters

## Structure

- `__init__.py`: Exports the public API of the module
- `config.py`: Main configuration class
- `defaults.py`: Default configuration values
- `validators.py`: Configuration validation functions

## Usage

Configuration should be accessed through the `Config` class:

```python
from src.config import Config

# Access configuration values
api_key = Config.get("API_KEY")
debug_mode = Config.get("DEBUG_MODE", default=False)

# Set configuration values
Config.set("BATCH_SIZE", 100)
```

## Environment Variables

Configuration values can be overridden using environment variables. See `.env.example` for a list of supported variables.
