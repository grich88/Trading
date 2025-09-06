"""
Configuration Validators

This module provides validation functions for configuration values.
"""

import os
from typing import Any, Dict, List, Optional, Set, Union

from src.utils.error_handling import ValidationError
from src.utils.logging_service import setup_logger

# Set up logger
logger = setup_logger(__name__)


def validate_required_config(
    config: Dict[str, Any], required_keys: List[str]
) -> None:
    """
    Validate that all required configuration keys are present and not empty.

    Args:
        config: The configuration dictionary to validate.
        required_keys: The list of required configuration keys.

    Raises:
        ValidationError: If any required configuration key is missing or empty.
    """
    missing_keys = []
    empty_keys = []

    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
        elif not config[key] and not isinstance(config[key], (bool, int, float)):
            empty_keys.append(key)

    if missing_keys:
        raise ValidationError(
            f"Missing required configuration keys: {', '.join(missing_keys)}"
        )

    if empty_keys:
        raise ValidationError(
            f"Empty required configuration keys: {', '.join(empty_keys)}"
        )


def validate_directory(directory: str, create: bool = False) -> None:
    """
    Validate that a directory exists and is writable.

    Args:
        directory: The directory path to validate.
        create: Whether to create the directory if it doesn't exist.

    Raises:
        ValidationError: If the directory doesn't exist or isn't writable.
    """
    if not os.path.exists(directory):
        if create:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                raise ValidationError(
                    f"Failed to create directory {directory}: {str(e)}"
                )
        else:
            raise ValidationError(f"Directory does not exist: {directory}")

    if not os.path.isdir(directory):
        raise ValidationError(f"Not a directory: {directory}")

    if not os.access(directory, os.W_OK):
        raise ValidationError(f"Directory is not writable: {directory}")


def validate_api_credentials(
    api_key: str, api_secret: Optional[str] = None
) -> None:
    """
    Validate API credentials.

    Args:
        api_key: The API key to validate.
        api_secret: The API secret to validate (optional).

    Raises:
        ValidationError: If the API credentials are invalid.
    """
    if not api_key:
        raise ValidationError("API key is required")

    if api_secret is not None and not api_secret:
        raise ValidationError("API secret is required")


def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    name: str = "Value",
) -> None:
    """
    Validate that a numeric value is within a specified range.

    Args:
        value: The value to validate.
        min_value: The minimum allowed value (inclusive).
        max_value: The maximum allowed value (inclusive).
        name: The name of the value for error messages.

    Raises:
        ValidationError: If the value is not within the specified range.
    """
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be at least {min_value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be at most {max_value}")


def validate_enum(value: Any, allowed_values: Set[Any], name: str = "Value") -> None:
    """
    Validate that a value is one of the allowed values.

    Args:
        value: The value to validate.
        allowed_values: The set of allowed values.
        name: The name of the value for error messages.

    Raises:
        ValidationError: If the value is not one of the allowed values.
    """
    if value not in allowed_values:
        allowed_str = ", ".join(str(v) for v in allowed_values)
        raise ValidationError(f"{name} must be one of: {allowed_str}")


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> None:
    """
    Validate a configuration dictionary against a schema.

    Args:
        config: The configuration dictionary to validate.
        schema: The schema to validate against.

    Raises:
        ValidationError: If the configuration doesn't match the schema.
    """
    for key, value in config.items():
        if key not in schema:
            logger.warning(f"Unknown configuration key: {key}")
            continue

        key_schema = schema[key]
        key_type = key_schema.get("type")

        # Check type
        if key_type and not isinstance(value, key_type):
            raise ValidationError(
                f"Invalid type for {key}: expected {key_type.__name__}, got {type(value).__name__}"
            )

        # Check enum
        if "enum" in key_schema and value not in key_schema["enum"]:
            allowed_str = ", ".join(str(v) for v in key_schema["enum"])
            raise ValidationError(f"{key} must be one of: {allowed_str}")

        # Check range
        if "min" in key_schema and value < key_schema["min"]:
            raise ValidationError(f"{key} must be at least {key_schema['min']}")

        if "max" in key_schema and value > key_schema["max"]:
            raise ValidationError(f"{key} must be at most {key_schema['max']}")

        # Check pattern
        if "pattern" in key_schema and not key_schema["pattern"].match(str(value)):
            raise ValidationError(f"{key} does not match the required pattern")

        # Check custom validation
        if "validate" in key_schema:
            try:
                key_schema["validate"](value)
            except ValidationError as e:
                raise ValidationError(f"Invalid value for {key}: {str(e)}")
            except Exception as e:
                raise ValidationError(f"Validation error for {key}: {str(e)}")

    # Check required keys
    required_keys = [k for k, v in schema.items() if v.get("required", False)]
    validate_required_config(config, required_keys)
