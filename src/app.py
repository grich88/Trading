#!/usr/bin/env python3
"""
Trading Algorithm System

Main application entry point.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import streamlit.web.bootstrap
from dotenv import load_dotenv

from src.config import Config
from src.core import App
from src.utils.logging_service import setup_logger

# Set up logger
logger = setup_logger("main")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Trading Algorithm System")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["app", "web", "backtest", "signal"],
        default="app",
        help="Application mode",
    )
    
    # Web mode options
    parser.add_argument(
        "--port",
        type=int,
        default=Config.get("WEB_APP_PORT", 8501),
        help="Web app port (for web mode)",
    )
    parser.add_argument(
        "--host",
        default=Config.get("WEB_APP_HOST", "localhost"),
        help="Web app host (for web mode)",
    )
    
    # Backtest mode options
    parser.add_argument(
        "--asset",
        default="BTC",
        help="Asset to backtest (for backtest mode)",
    )
    parser.add_argument(
        "--start-date",
        default=Config.get("HISTORICAL_DATA_START_DATE", "2023-01-01"),
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=Config.get("HISTORICAL_DATA_END_DATE", None),
        help="End date for backtest (YYYY-MM-DD)",
    )
    
    # Signal mode options
    parser.add_argument(
        "--assets",
        default="BTC,SOL,BONK",
        help="Comma-separated list of assets to generate signals for (for signal mode)",
    )
    
    # General options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=Config.get("LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file",
    )
    
    return parser.parse_args()


def run_app_mode() -> None:
    """
    Run the application in app mode.
    """
    logger.info("Starting application in app mode")
    
    # Create and run the application
    app = App()
    
    # Register services (to be implemented)
    # app.register_service(DataService("data"))
    # app.register_service(AnalysisService("analysis"))
    
    # Run the application
    app.run()


def run_web_mode(port: int, host: str) -> None:
    """
    Run the application in web mode.

    Args:
        port: The port to run the web app on.
        host: The host to run the web app on.
    """
    logger.info(f"Starting web app on {host}:{port}")
    
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = host
    
    # Get the path to the web app script
    web_app_path = os.path.join(os.path.dirname(__file__), "web_app.py")
    
    # Check if the web app script exists
    if not os.path.exists(web_app_path):
        logger.error(f"Web app script not found: {web_app_path}")
        sys.exit(1)
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", web_app_path]
    streamlit.web.bootstrap.run()


def run_backtest_mode(asset: str, start_date: str, end_date: Optional[str]) -> None:
    """
    Run the application in backtest mode.

    Args:
        asset: The asset to backtest.
        start_date: The start date for the backtest.
        end_date: The end date for the backtest.
    """
    logger.info(f"Running backtest for {asset} from {start_date} to {end_date or 'now'}")
    
    # Import backtester here to avoid circular imports
    # from src.models import Backtester
    
    # Run backtest
    # backtester = Backtester(asset)
    # results = backtester.run(start_date, end_date)
    
    # Print results
    # print(f"Backtest results for {asset}:")
    # print(f"Total return: {results['total_return']:.2%}")
    # print(f"Sharpe ratio: {results['sharpe']:.2f}")
    # print(f"Win rate: {results['win_rate']:.2%}")
    # print(f"Total trades: {results['total_trades']}")
    
    logger.info("Backtest completed")


def run_signal_mode(assets: List[str]) -> None:
    """
    Run the application in signal mode.

    Args:
        assets: The list of assets to generate signals for.
    """
    logger.info(f"Generating signals for {', '.join(assets)}")
    
    # Import signal generator here to avoid circular imports
    # from src.models import SignalGenerator
    
    # Generate signals
    # signal_generator = SignalGenerator()
    # signals = signal_generator.generate(assets)
    
    # Print signals
    # for asset, signal in signals.items():
    #     print(f"{asset}: {signal['signal']} (Score: {signal['score']:.2f})")
    #     print(f"  Entry: ${signal['entry']:.2f}")
    #     print(f"  TP1: ${signal['tp1']:.2f} ({signal['tp1_pct']:.2%})")
    #     print(f"  SL: ${signal['sl']:.2f} ({signal['sl_pct']:.2%})")
    
    logger.info("Signal generation completed")


def main() -> None:
    """
    Main entry point for the application.
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(args.log_level)
    
    # Set configuration from file if provided
    if args.config:
        # Load configuration from file (to be implemented)
        pass
    
    # Run in the specified mode
    if args.mode == "app":
        run_app_mode()
    elif args.mode == "web":
        run_web_mode(args.port, args.host)
    elif args.mode == "backtest":
        run_backtest_mode(args.asset, args.start_date, args.end_date)
    elif args.mode == "signal":
        run_signal_mode(args.assets.split(","))
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()