"""
Unified Trading Application - Entry Point

This script serves as the unified entry point for the trading application,
combining data collection, model execution, visualization, and backtesting.
"""

import os
import argparse
import logging
from datetime import datetime

from model_data_collector import ModelDataCollector
from liquidation_data_input import process_data
from run_model_with_data import run_model
from backtester import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified Trading Application')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection step')
    parser.add_argument('--skip-liquidation-input', action='store_true', help='Skip manual liquidation data input')
    parser.add_argument('--input-file', type=str, help='Input file path (JSON file with model input data)')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    return parser.parse_args()

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    try:
        model_input = None
        
        # Step 1: Collect data (unless skipped)
        if not args.skip_collection and args.input_file is None:
            logger.info("Collecting data...")
            collector = ModelDataCollector()
            symbols = ['BTC/USDT', 'SOL/USDT']  # Add more symbols as needed
            data = collector.collect_data(symbols)
            model_input = collector.prepare_model_input(data)
        
        # Step 2: Load data from file if provided
        if args.input_file:
            logger.info(f"Loading data from {args.input_file}...")
            with open(args.input_file, 'r') as f:
                import json
                model_input = json.load(f)
            if model_input is None:
                logger.error("Failed to load input file. Exiting.")
                return
        
        # Step 3: Get liquidation data (unless skipped)
        if not args.skip_liquidation_input and model_input is not None:
            logger.info("Getting liquidation data...")
            # Import necessary modules
            from coinglass_api import get_liquidation_heatmap, extract_liquidation_clusters
            from image_extractors import extract_liq_clusters_from_image, auto_detect_heatmap_scale
            import os
            
            # Process each asset separately
            for asset_name, asset_data in model_input.items():
                current_price = asset_data['price'][-1]  # Get the latest price
                price_series = asset_data['price']
                logger.info(f"Getting liquidation data for {asset_name} (Current price: {current_price})")
                
                # Try to get liquidation data from Coinglass API
                liquidation_data = None
                try:
                    heatmap = get_liquidation_heatmap(asset_name)
                    if heatmap:
                        liquidation_data = extract_liquidation_clusters(heatmap, price_series)
                        logger.info(f"Successfully retrieved liquidation data from Coinglass API")
                except Exception as e:
                    logger.warning(f"Error getting liquidation data from Coinglass API: {e}")
                
                # If API fails, try to extract from heatmap images in the data/coinglass_cache directory
                if liquidation_data is None:
                    logger.info(f"Trying to extract liquidation data from heatmap images...")
                    heatmap_dir = "data/coinglass_cache"
                    if os.path.exists(heatmap_dir):
                        for filename in os.listdir(heatmap_dir):
                            if asset_name.lower() in filename.lower() and filename.endswith(".png"):
                                try:
                                    image_path = os.path.join(heatmap_dir, filename)
                                    with open(image_path, "rb") as f:
                                        image_bytes = f.read()
                                    
                                    # Try to auto-detect price scale
                                    price_range = auto_detect_heatmap_scale(image_bytes)
                                    if price_range:
                                        price_top, price_bottom = price_range
                                    else:
                                        # Use current price ±20% as fallback
                                        price_top = current_price * 1.2
                                        price_bottom = current_price * 0.8
                                    
                                    # Extract clusters from image
                                    clusters = extract_liq_clusters_from_image(
                                        image_bytes, 
                                        price_top=price_top, 
                                        price_bottom=price_bottom
                                    )
                                    
                                    if clusters:
                                        # Determine if price has cleared major liquidation zones
                                        clusters_above = [c for c in clusters if c[0] > current_price]
                                        clusters_below = [c for c in clusters if c[0] < current_price]
                                        intensity_above = sum(c[1] for c in clusters_above)
                                        intensity_below = sum(c[1] for c in clusters_below)
                                        cleared_zone = intensity_below > intensity_above
                                        
                                        liquidation_data = {
                                            'clusters': clusters,
                                            'cleared_zone': cleared_zone
                                        }
                                        logger.info(f"Successfully extracted liquidation data from image: {filename}")
                                        break
                                except Exception as e:
                                    logger.warning(f"Error extracting liquidation data from image: {e}")
                
                # If still no data, use empty liquidation data
                if liquidation_data is None:
                    logger.warning(f"No liquidation data available for {asset_name}. Using empty data.")
                    liquidation_data = {
                        'clusters': [],
                        'cleared_zone': False
                    }
                
                # Update the model input with the liquidation data
                model_input[asset_name]['liquidation_data'] = liquidation_data
            
        # Check if we have model input data
        if model_input is None and args.skip_collection and not args.input_file:
            logger.error("No model input data available. Please provide an input file or enable data collection.")
            return
        
        # Step 4: Save model input
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_dir("data")
        model_input_path = f"data/model_input_{timestamp}.json"
        with open(model_input_path, 'w') as f:
            import json
            json.dump(model_input, f, indent=2)
        logger.info(f"Saved model input to {model_input_path}")
        
        # Step 5: Run model
        logger.info("Running model...")
        results = run_model(model_input)
        
        # Step 6: Save results
        results_path = f"data/model_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")
        
        # Step 7: Run backtesting (if requested)
        if args.backtest:
            logger.info("Running backtesting...")
            # Convert results to DataFrame for backtesting
            import pandas as pd
            results_df = pd.DataFrame(results)
            backtester = Backtester(results_df)
            backtest_results = backtester.run_backtest()
            
            # Save backtest results
            backtest_path = f"data/backtest_results_{timestamp}.csv"
            backtest_results.to_csv(backtest_path)
            logger.info(f"Saved backtest results to {backtest_path}")
        
        # Step 8: Visualize results (if requested)
        if args.visualize:
            logger.info("Visualizing results...")
            try:
                # Create a simple visualization of the results
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract data
                assets = list(results['results'].keys())
                scores = [results['results'][asset]['final_score'] for asset in assets]
                signals = [results['results'][asset]['signal'] for asset in assets]
                
                # Set colors based on signal
                colors = []
                for signal in signals:
                    if signal == 'STRONG BUY':
                        colors.append('darkgreen')
                    elif signal == 'BUY':
                        colors.append('green')
                    elif signal == 'NEUTRAL':
                        colors.append('gray')
                    elif signal == 'SELL':
                        colors.append('red')
                    elif signal == 'STRONG SELL':
                        colors.append('darkred')
                    else:
                        colors.append('blue')
                
                # Create bar chart
                bars = ax.bar(assets, scores, color=colors)
                
                # Add labels
                ax.set_title('Model Scores by Asset', fontsize=16)
                ax.set_xlabel('Asset', fontsize=12)
                ax.set_ylabel('Score', fontsize=12)
                
                # Add signal labels
                for i, (bar, signal) in enumerate(zip(bars, signals)):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                           signal, ha='center', va='bottom', fontweight='bold')
                
                # Add assessment
                assessment = results['assessment']
                ax.text(0.5, -0.2, f"Market Condition: {assessment['market_condition']}\n"
                       f"Best Opportunity: {assessment['best_opportunity']['asset']} ({assessment['best_opportunity']['signal']})\n"
                       f"Weakest Asset: {assessment['weakest_asset']['asset']} ({assessment['weakest_asset']['signal']})",
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                # Save figure
                plt.tight_layout()
                visualization_path = f"data/visualization_{timestamp}.png"
                plt.savefig(visualization_path)
                plt.close()
                
                logger.info(f"Visualization saved to {visualization_path}")
            except Exception as e:
                logger.error(f"Error visualizing results: {e}")
        
        logger.info("Unified application completed successfully")
        
    except Exception as e:
        logger.error(f"Error in unified application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
