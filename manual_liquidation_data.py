"""
Manual Liquidation Data Input for BTC and SOL

This script adds liquidation data from Coinglass heatmaps to our model input.
"""

import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_latest_model_input():
    """
    Load the latest model input file
    
    Returns:
    --------
    tuple
        (filepath, data)
    """
    # Find the most recent model_input_*.json file
    data_dir = 'data'
    if not os.path.exists(data_dir):
        logger.error("Data directory not found")
        return None, None
    
    files = [f for f in os.listdir(data_dir) if f.startswith('model_input_') and f.endswith('.json')]
    if not files:
        logger.error("No model input files found")
        return None, None
    
    # Sort by timestamp (newest first)
    files.sort(reverse=True)
    latest_file = os.path.join(data_dir, files[0])
    
    # Load data
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return latest_file, data
    except Exception as e:
        logger.error(f"Error loading data from {latest_file}: {e}")
        return None, None

def save_data(data, filepath):
    """
    Save data to a JSON file
    
    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str
        Filepath to save to
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def main():
    """Main function"""
    # Load latest model input
    input_filepath, model_input = load_latest_model_input()
    if not model_input:
        return
    
    logger.info(f"Loaded model input from {input_filepath}")
    
    # BTC liquidation data from Coinglass heatmap
    btc_liquidation_data = {
        'clusters': [
            # Support levels (below current price) - yellow/green areas
            (115000, 0.9),  # Strong support level with high intensity
            (110000, 0.7),  # Moderate support level
            
            # Resistance levels (above current price) - yellow/green areas
            (120000, 0.6),  # Moderate resistance
            (125000, 0.4),  # Light resistance
        ],
        'cleared_zone': True  # Price has cleared major liquidation zones
    }
    
    # SOL liquidation data from Coinglass heatmap
    sol_liquidation_data = {
        'clusters': [
            # Support levels (below current price) - yellow/green areas
            (190, 0.8),  # Strong support level with high intensity
            (180, 0.9),  # Very strong support level
            (177, 0.7),  # Moderate support level
            
            # Resistance levels (above current price) - yellow/green areas
            (210, 0.5),  # Moderate resistance
            (220, 0.4),  # Light resistance
        ],
        'cleared_zone': True  # Price has cleared major liquidation zones
    }
    
    # Update model input with liquidation data
    if 'BTC' in model_input:
        model_input['BTC']['liquidation_data'] = btc_liquidation_data
        logger.info("Added liquidation data for BTC")
    
    if 'SOL' in model_input:
        model_input['SOL']['liquidation_data'] = sol_liquidation_data
        logger.info("Added liquidation data for SOL")
    
    # Save updated model input
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = os.path.join('data', f"model_input_with_liquidation_{timestamp}.json")
    save_data(model_input, output_filepath)
    
    logger.info("Liquidation data added successfully")
    logger.info(f"Updated model input saved to {output_filepath}")
    
    return output_filepath, model_input

if __name__ == "__main__":
    main()
