"""
Liquidation Data Input Tool

This script provides a simple command-line interface to manually input liquidation data
from Coinglass for use with the Enhanced RSI + Volume Predictive Scoring Model.

Usage:
    python liquidation_data_input.py [--input-file data/model_data_YYYYMMDD_HHMMSS.json] [--output-file data/model_input_YYYYMMDD_HHMMSS.json]
"""

import argparse
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

def load_data(filepath):
    """
    Load data from a JSON file
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
    
    Returns:
    --------
    dict
        Loaded data
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def save_data(data, filepath):
    """
    Save data to a JSON file
    
    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str
        Path to save the data to
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Successfully saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")

def input_float(prompt, default=None):
    """
    Get float input from user
    
    Parameters:
    -----------
    prompt : str
        Prompt to display
    default : float, optional
        Default value if user enters nothing
    
    Returns:
    --------
    float
        User input as float
    """
    while True:
        if default is not None:
            user_input = input(f"{prompt} [{default}]: ")
            if user_input == "":
                return default
        else:
            user_input = input(f"{prompt}: ")
        
        try:
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a number.")

def input_yes_no(prompt, default=None):
    """
    Get yes/no input from user
    
    Parameters:
    -----------
    prompt : str
        Prompt to display
    default : bool, optional
        Default value if user enters nothing
    
    Returns:
    --------
    bool
        True if yes, False if no
    """
    default_str = None
    if default is not None:
        default_str = "y" if default else "n"
    
    while True:
        if default_str is not None:
            user_input = input(f"{prompt} (y/n) [{default_str}]: ").lower()
            if user_input == "":
                return default
        else:
            user_input = input(f"{prompt} (y/n): ").lower()
        
        if user_input in ["y", "yes"]:
            return True
        elif user_input in ["n", "no"]:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def input_liquidation_data(asset_name, current_price):
    """
    Get liquidation data input from user
    
    Parameters:
    -----------
    asset_name : str
        Name of the asset
    current_price : float
        Current price of the asset
    
    Returns:
    --------
    dict
        Liquidation data
    """
    print(f"\n{'-' * 50}")
    print(f"Liquidation Data Input for {asset_name}")
    print(f"Current price: {current_price}")
    print(f"{'-' * 50}")
    
    # Initialize liquidation data
    liquidation_data = {
        'clusters': [],
        'cleared_zone': False
    }
    
    # Input cleared zone
    liquidation_data['cleared_zone'] = input_yes_no(
        "Has the price cleared major liquidation zones? (Is price above most yellow clusters on Coinglass?)",
        default=False
    )
    
    # Input support clusters
    print("\nEnter support clusters (liquidation levels below current price):")
    print("Leave price empty to finish")
    
    while True:
        price = input_float(f"Support level price (below {current_price})", default=None)
        if price is None:
            break
        
        if price >= current_price:
            print(f"Support level must be below current price ({current_price})")
            continue
        
        intensity = input_float("Intensity (0.0-1.0, where 1.0 is brightest yellow)", default=0.5)
        intensity = max(0.0, min(1.0, intensity))  # Clamp to [0.0, 1.0]
        
        liquidation_data['clusters'].append((price, intensity))
    
    # Input resistance clusters
    print("\nEnter resistance clusters (liquidation levels above current price):")
    print("Leave price empty to finish")
    
    while True:
        price = input_float(f"Resistance level price (above {current_price})", default=None)
        if price is None:
            break
        
        if price <= current_price:
            print(f"Resistance level must be above current price ({current_price})")
            continue
        
        intensity = input_float("Intensity (0.0-1.0, where 1.0 is brightest yellow)", default=0.5)
        intensity = max(0.0, min(1.0, intensity))  # Clamp to [0.0, 1.0]
        
        liquidation_data['clusters'].append((price, intensity))
    
    # Sort clusters by price
    liquidation_data['clusters'].sort(key=lambda x: x[0])
    
    return liquidation_data

def process_data(data):
    """
    Process data to add liquidation data
    
    Parameters:
    -----------
    data : dict
        Data to process
    
    Returns:
    --------
    dict
        Processed data
    """
    model_input = {}
    
    for asset_name, asset_data in data['assets'].items():
        # Get current price
        current_price = asset_data['price']
        
        # Get liquidation data from user
        liquidation_data = input_liquidation_data(asset_name, current_price)
        
        # Create model input for asset
        model_input[asset_name] = {
            'price': asset_data['price_series'],
            'rsi_raw': asset_data['rsi_raw_series'],
            'rsi_sma': asset_data['rsi_sma_series'],
            'volume': asset_data['volume_series'],
            'webtrend_status': asset_data['webtrend_status'],
            'liquidation_data': liquidation_data
        }
    
    return model_input

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Input liquidation data for the Enhanced RSI + Volume Predictive Scoring Model')
    parser.add_argument('--input-file', type=str, help='Input file path (JSON file from model_data_collector.py)')
    parser.add_argument('--output-file', type=str, help='Output file path')
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set default input file if not provided
    if args.input_file is None:
        # Find the most recent model_data_*.json file
        data_dir = 'data'
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.startswith('model_data_') and f.endswith('.json')]
            if files:
                files.sort(reverse=True)  # Sort in descending order
                args.input_file = os.path.join(data_dir, files[0])
                logger.info(f"Using most recent data file: {args.input_file}")
    
    # Set default output file if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"data/model_input_{timestamp}.json"
    
    # Check if input file exists
    if args.input_file is None or not os.path.exists(args.input_file):
        logger.error("Input file not found. Please run model_data_collector.py first or specify an input file.")
        return
    
    # Load data
    data = load_data(args.input_file)
    if data is None:
        return
    
    # Process data
    model_input = process_data(data)
    
    # Save processed data
    save_data(model_input, args.output_file)
    
    logger.info("Liquidation data input completed successfully")
    logger.info(f"Model input saved to {args.output_file}")
    
    return model_input

if __name__ == "__main__":
    main()
