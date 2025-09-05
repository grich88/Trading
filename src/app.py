"""
Main application entry point.

This module provides the main entry point for the application.
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Any

# Import core application
from src.core import TradingApp

# Import configuration
from src.config import print_config

# Import utilities
from src.utils import get_logger

# Initialize logger
logger = get_logger("Main")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Trading Algorithm System")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["analysis", "backtest", "live"],
        default="analysis",
        help="Application mode"
    )
    
    # Asset selection
    parser.add_argument(
        "--assets",
        type=str,
        default="BTC,SOL,BONK",
        help="Comma-separated list of assets to analyze"
    )
    
    # Timeframe
    parser.add_argument(
        "--timeframe",
        type=str,
        default="4h",
        help="Timeframe for data (e.g., 1h, 4h, 1d)"
    )
    
    # Date range
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date for data (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data (YYYY-MM-DD)"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Window size for analysis"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.425,
        help="Minimum absolute score for signals"
    )
    
    # Output options
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files"
    )
    
    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print configuration and exit"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration if requested
    if args.print_config:
        print_config()
        return 0
    
    # Set debug mode
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Parse assets
    assets = [asset.strip() for asset in args.assets.split(",")]
    
    # Create application
    app = TradingApp()
    
    # Start application
    if not app.start():
        logger.error("Failed to start application")
        return 1
    
    try:
        # Run in selected mode
        if args.mode == "analysis":
            # Run analysis
            results = app.run_full_analysis(
                assets=assets,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                window_size=args.window_size,
                min_abs_score=args.min_score
            )
            
            # Print results
            print("\nAnalysis Results:")
            for asset, signal in results["signals"].items():
                print(f"{asset}: {signal['signal']} (Score: {signal['final_score']:.3f})")
                print(f"  Targets: TP1={signal['targets']['TP1']}, TP2={signal['targets']['TP2']}, SL={signal['targets']['SL']}")
            
            # Print cross-asset insights
            print("\nCross-Asset Insights:")
            for asset, bias in results["cross_asset_biases"].items():
                if abs(bias.get("bias", 0)) > 0.01:
                    print(f"  BTC → {asset}: {bias['bias']:.3f} (confidence: {bias['confidence']:.2f})")
            
            # Save results if requested
            if args.save_results:
                import json
                from datetime import datetime
                
                # Create output directory
                os.makedirs(args.output_dir, exist_ok=True)
                
                # Create output file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(args.output_dir, f"analysis_{timestamp}.json")
                
                # Save results
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)
                
                print(f"\nResults saved to {output_file}")
        
        elif args.mode == "backtest":
            print("Backtest mode not implemented yet")
            return 1
        
        elif args.mode == "live":
            print("Live mode not implemented yet")
            return 1
    
    finally:
        # Stop application
        app.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
