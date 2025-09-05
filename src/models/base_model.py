"""
Base model module.

This module provides a base class for all models with common functionality.
"""

from typing import Any, Dict, List, Optional, Union
import os
import json

# Import utilities
from src.utils import get_logger, exception_handler, performance_monitor

# Import configuration
from src.config import MODEL_WEIGHTS_DIR


class BaseModel:
    """
    Base class for all models.
    
    This class provides common functionality for all models, including:
    - Logging
    - Weight management
    - Serialization
    """
    
    def __init__(self, name: str, asset_type: str = "BTC"):
        """
        Initialize the base model.
        
        Args:
            name: Model name
            asset_type: Asset type (BTC, SOL, BONK)
        """
        self.name = name
        self.asset_type = asset_type
        self.logger = get_logger(f"{name}Model")
        
        # Create weights directory if it doesn't exist
        os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
        
        self.logger.info(f"{self.name} model initialized for {self.asset_type}")
    
    def get_weights_path(self) -> str:
        """
        Get the path to the weights file.
        
        Returns:
            Path to the weights file
        """
        return os.path.join(MODEL_WEIGHTS_DIR, f"{self.name}_{self.asset_type}_weights.json")
    
    def save_weights(self, weights: Dict[str, Any]) -> None:
        """
        Save model weights to a file.
        
        Args:
            weights: Model weights
        """
        try:
            weights_path = self.get_weights_path()
            
            # Add metadata
            weights_with_metadata = {
                "name": self.name,
                "asset_type": self.asset_type,
                "weights": weights
            }
            
            # Save to file
            with open(weights_path, 'w') as f:
                json.dump(weights_with_metadata, f, indent=2)
            
            self.logger.info(f"Saved weights to {weights_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving weights: {str(e)}")
    
    def load_weights(self) -> Optional[Dict[str, Any]]:
        """
        Load model weights from a file.
        
        Returns:
            Model weights, or None if not found
        """
        try:
            weights_path = self.get_weights_path()
            
            # Check if file exists
            if not os.path.exists(weights_path):
                self.logger.warning(f"Weights file not found: {weights_path}")
                return None
            
            # Load from file
            with open(weights_path, 'r') as f:
                weights_with_metadata = json.load(f)
            
            # Extract weights
            weights = weights_with_metadata.get("weights", {})
            
            self.logger.info(f"Loaded weights from {weights_path}")
            return weights
        
        except Exception as e:
            self.logger.error(f"Error loading weights: {str(e)}")
            return None
    
    @performance_monitor()
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Make a prediction.
        
        This method should be overridden by subclasses.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Prediction result
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.name,
            "asset_type": self.asset_type,
            "weights_path": self.get_weights_path(),
            "has_weights": os.path.exists(self.get_weights_path())
        }
