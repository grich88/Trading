"""
Run Enhanced RSI + Volume Predictive Scoring Model with Collected Data

This script:
1. Collects data using model_data_collector.py
2. Allows for manual input of liquidation data using liquidation_data_input.py
3. Runs the Enhanced RSI + Volume Predictive Scoring Model
4. Visualizes and saves the results

Usage:
    python run_model_with_data.py [--skip-collection] [--skip-liquidation-input] [--input-file path/to/input.json]
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime

# Import model and data collection modules
try:
    from updated_rsi_volume_model import EnhancedRsiVolumePredictor, analyze_market_data, get_market_assessment
    from model_data_collector import ModelDataCollector
except ImportError:
    print("Error: Required modules not found. Please ensure updated_rsi_volume_model.py and model_data_collector.py are in the current directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_run.log"),
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

def collect_data():
    """
    Collect data using model_data_collector.py
    
    Returns:
    --------
    tuple
        (raw_data, model_input_data)
    """
    logger.info("Collecting data...")
    
    # Initialize data collector
    collector = ModelDataCollector()
    
    # Define symbols to collect data for
    symbols = ['BTC/USDT', 'SOL/USDT']
    
    # Try to add BONK
    try:
        collector.exchange.fetch_ticker('BONK/USDT')
        symbols.append('BONK/USDT')
    except:
        logger.warning("BONK/USDT not available, trying BONK/USDC")
        try:
            collector.exchange.fetch_ticker('BONK/USDC')
            symbols.append('BONK/USDC')
        except:
            logger.warning("BONK not available, skipping")
    
    # Collect data
    raw_data = collector.collect_data(symbols)
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collector.save_data(raw_data, f"model_data_{timestamp}.json")
    
    # Prepare model input (without liquidation data)
    model_input = collector.prepare_model_input(raw_data)
    
    return raw_data, model_input

def input_liquidation_data(model_input):
    """
    Input liquidation data manually
    
    Parameters:
    -----------
    model_input : dict
        Model input data
    
    Returns:
    --------
    dict
        Updated model input data with liquidation data
    """
    logger.info("Please input liquidation data from Coinglass...")
    
    for asset_name, asset_data in model_input.items():
        print(f"\n{'-' * 50}")
        print(f"Liquidation Data Input for {asset_name}")
        print(f"Current price: {asset_data['price'][-1]}")
        print(f"{'-' * 50}")
        
        # Initialize liquidation data
        liquidation_data = {
            'clusters': [],
            'cleared_zone': False
        }
        
        # Input cleared zone
        cleared_zone = input("Has the price cleared major liquidation zones? (y/n): ").lower()
        liquidation_data['cleared_zone'] = cleared_zone in ['y', 'yes']
        
        # Input support clusters
        print("\nEnter support clusters (liquidation levels below current price):")
        print("Enter 'done' when finished")
        
        current_price = asset_data['price'][-1]
        
        while True:
            price_str = input(f"Support level price (below {current_price}): ")
            if price_str.lower() == 'done' or price_str == '':
                break
            
            try:
                price = float(price_str)
                if price >= current_price:
                    print(f"Support level must be below current price ({current_price})")
                    continue
                
                intensity_str = input("Intensity (0.0-1.0, where 1.0 is brightest yellow): ")
                intensity = float(intensity_str) if intensity_str else 0.5
                intensity = max(0.0, min(1.0, intensity))  # Clamp to [0.0, 1.0]
                
                liquidation_data['clusters'].append((price, intensity))
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Input resistance clusters
        print("\nEnter resistance clusters (liquidation levels above current price):")
        print("Enter 'done' when finished")
        
        while True:
            price_str = input(f"Resistance level price (above {current_price}): ")
            if price_str.lower() == 'done' or price_str == '':
                break
            
            try:
                price = float(price_str)
                if price <= current_price:
                    print(f"Resistance level must be above current price ({current_price})")
                    continue
                
                intensity_str = input("Intensity (0.0-1.0, where 1.0 is brightest yellow): ")
                intensity = float(intensity_str) if intensity_str else 0.5
                intensity = max(0.0, min(1.0, intensity))  # Clamp to [0.0, 1.0]
                
                liquidation_data['clusters'].append((price, intensity))
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Sort clusters by price
        liquidation_data['clusters'].sort(key=lambda x: x[0])
        
        # Update model input
        asset_data['liquidation_data'] = liquidation_data
    
    return model_input

def run_model(model_input):
    """
    Run the Enhanced RSI + Volume Predictive Scoring Model
    
    Parameters:
    -----------
    model_input : dict
        Model input data
    
    Returns:
    --------
    dict
        Model results
    """
    logger.info("Running Enhanced RSI + Volume Predictive Scoring Model...")
    
    # Convert model input to format expected by analyze_market_data
    assets_data = {}
    
    for asset_name, asset_data in model_input.items():
        assets_data[asset_name] = {
            'price': asset_data['price'],
            'rsi_raw': asset_data['rsi_raw'],
            'rsi_sma': asset_data['rsi_sma'],
            'volume': asset_data['volume'],
            'liquidation_data': asset_data['liquidation_data'],
            'webtrend_status': asset_data['webtrend_status']
        }
    
    # Run model
    results = analyze_market_data(assets_data)
    
    # Get market assessment
    assessment = get_market_assessment(results)
    
    # Create complete results
    complete_results = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'assessment': assessment
    }
    
    return complete_results

def display_results(results):
    """
    Display model results
    
    Parameters:
    -----------
    results : dict
        Model results
    """
    print("\n" + "=" * 80)
    print("ENHANCED RSI + VOLUME PREDICTIVE SCORING MODEL RESULTS")
    print("=" * 80)
    
    # Display individual asset results
    for asset, analysis in results['results'].items():
        print(f"\n{'-' * 50}")
        print(f"{asset} ANALYSIS")
        print(f"{'-' * 50}")
        print(f"Current Price: {analysis['price']}")
        print(f"Current RSI: {analysis['rsi']:.2f}")
        print(f"Current RSI SMA: {analysis['rsi_sma']:.2f}")
        
        print(f"\nModel Component Scores:")
        for component, score in analysis['components'].items():
            print(f"{component.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\nFinal Momentum Score: {analysis['final_score']:.3f}")
        print(f"Signal: {analysis['signal']}")
        
        print(f"\nTarget Prices:")
        print(f"TP1: {analysis['targets']['TP1']}")
        print(f"TP2: {analysis['targets']['TP2']}")
        print(f"SL: {analysis['targets']['SL']}")
    
    # Display market assessment
    assessment = results['assessment']
    
    print("\n" + "=" * 50)
    print("MARKET ASSESSMENT")
    print("=" * 50)
    print(f"Market Condition: {assessment['market_condition']}")
    print(f"Average Score: {assessment['average_score']}")
    print(f"\nBest Opportunity: {assessment['best_opportunity']['asset']} ({assessment['best_opportunity']['signal']})")
    print(f"Weakest Asset: {assessment['weakest_asset']['asset']} ({assessment['weakest_asset']['signal']})")
    
    print(f"\nRotation Strategy:")
    for strategy in assessment['rotation_strategy']:
        print(f"- {strategy}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Enhanced RSI + Volume Predictive Scoring Model with collected data')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection step')
    parser.add_argument('--skip-liquidation-input', action='store_true', help='Skip manual liquidation data input')
    parser.add_argument('--input-file', type=str, help='Input file path (JSON file with model input data)')
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    try:
        model_input = None
        
        # Step 1: Collect data (unless skipped)
        if not args.skip_collection and args.input_file is None:
            _, model_input = collect_data()
        
        # Step 2: Load data from file if provided
        if args.input_file:
            model_input = load_data(args.input_file)
            if model_input is None:
                logger.error("Failed to load input file. Exiting.")
                return
        
        # Step 3: Input liquidation data (unless skipped)
        if not args.skip_liquidation_input:
            model_input = input_liquidation_data(model_input)
        
        # Step 4: Save model input
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_data(model_input, f"data/model_input_{timestamp}.json")
        
        # Step 5: Run model
        results = run_model(model_input)
        
        # Step 6: Save results
        save_data(results, f"data/model_results_{timestamp}.json")
        
        # Step 7: Display results
        display_results(results)
        
        logger.info("Model run completed successfully")
        
    except Exception as e:
        logger.error(f"Error running model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
