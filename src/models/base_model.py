"""
Base Model

This module provides a base class for all models in the Trading Algorithm System.
"""

import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import Config
from src.utils.error_handling import AppError
from src.utils.logging_service import setup_logger
from src.utils.performance import performance_timer


class ModelError(AppError):
    """Exception raised for model errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.

        Args:
            message: The error message.
            details: Additional error details.
        """
        super().__init__(message, 500, details)


class BaseModel(ABC):
    """
    Base class for all models.

    This class provides common functionality for all models, including:
    - Model persistence
    - Performance evaluation
    - Hyperparameter management
    - Model versioning
    """

    def __init__(self, name: str, version: str = "0.1.0"):
        """
        Initialize the model.

        Args:
            name: The name of the model.
            version: The version of the model.
        """
        self.name = name
        self.version = version
        self.logger = setup_logger(f"model.{name}")
        self.hyperparameters: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "name": name,
            "version": version,
            "created_at": datetime.now().isoformat(),
        }
        self.model_dir = os.path.join(
            Config.get("MODEL_WEIGHTS_DIR", "data/weights"), name
        )
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def set_hyperparameters(self, **kwargs: Any) -> None:
        """
        Set hyperparameters for the model.

        Args:
            **kwargs: Hyperparameter values.
        """
        self.hyperparameters.update(kwargs)
        self.logger.info(f"Set hyperparameters: {kwargs}")

    def get_hyperparameter(self, name: str, default: Any = None) -> Any:
        """
        Get a hyperparameter value.

        Args:
            name: The name of the hyperparameter.
            default: The default value to return if the hyperparameter is not set.

        Returns:
            The hyperparameter value.
        """
        return self.hyperparameters.get(name, default)

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the model to disk.

        Args:
            filename: The filename to save the model to. If None, a default filename
                     will be generated based on the model name and version.

        Returns:
            The path to the saved model.

        Raises:
            ModelError: If the model fails to save.
        """
        if filename is None:
            filename = f"{self.name}_v{self.version.replace('.', '_')}.pkl"
        
        filepath = os.path.join(self.model_dir, filename)
        
        try:
            # Save model state
            model_state = self._get_model_state()
            
            # Add metadata
            model_state["metadata"] = {
                **self.metadata,
                "saved_at": datetime.now().isoformat(),
                "hyperparameters": self.hyperparameters,
            }
            
            # Save to disk
            with open(filepath, "wb") as f:
                pickle.dump(model_state, f)
            
            self.logger.info(f"Model saved to {filepath}")
            
            # Save metadata separately as JSON for easy inspection
            metadata_path = os.path.splitext(filepath)[0] + ".json"
            with open(metadata_path, "w") as f:
                json.dump(model_state["metadata"], f, indent=2)
            
            return filepath
        except Exception as e:
            error_msg = f"Failed to save model to {filepath}: {str(e)}"
            self.logger.error(error_msg)
            raise ModelError(error_msg)

    def load(self, filepath: str) -> None:
        """
        Load the model from disk.

        Args:
            filepath: The path to the saved model.

        Raises:
            ModelError: If the model fails to load.
        """
        try:
            with open(filepath, "rb") as f:
                model_state = pickle.load(f)
            
            # Load metadata
            if "metadata" in model_state:
                self.metadata = model_state["metadata"]
                self.name = self.metadata.get("name", self.name)
                self.version = self.metadata.get("version", self.version)
                self.hyperparameters = self.metadata.get("hyperparameters", {})
            
            # Load model state
            self._set_model_state(model_state)
            
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            error_msg = f"Failed to load model from {filepath}: {str(e)}"
            self.logger.error(error_msg)
            raise ModelError(error_msg)

    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """
        Get the model state for saving.

        Returns:
            The model state as a dictionary.
        """
        pass

    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """
        Set the model state after loading.

        Args:
            state: The model state as a dictionary.
        """
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            data: The training data.
            **kwargs: Additional training parameters.

        Returns:
            A dictionary containing training metrics.
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Generate predictions.

        Args:
            data: The input data.
            **kwargs: Additional prediction parameters.

        Returns:
            A DataFrame containing the predictions.
        """
        pass

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluate the model.

        Args:
            data: The evaluation data.
            **kwargs: Additional evaluation parameters.

        Returns:
            A dictionary containing evaluation metrics.
        """
        pass

    def get_version_info(self) -> Dict[str, Any]:
        """
        Get version information for the model.

        Returns:
            A dictionary containing version information.
        """
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.metadata.get("created_at"),
            "saved_at": self.metadata.get("saved_at"),
        }

    @performance_timer
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before training or prediction.

        Args:
            data: The input data.

        Returns:
            The preprocessed data.
        """
        return data.copy()

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data.

        Args:
            data: The input data.

        Raises:
            ModelError: If the data is invalid.
        """
        if data is None or data.empty:
            raise ModelError("Input data is empty")
        
        # Check for required columns (to be implemented by subclasses)
        required_columns = self._get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ModelError(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
    
    def _get_required_columns(self) -> List[str]:
        """
        Get the list of required columns for input data.

        Returns:
            A list of required column names.
        """
        return []