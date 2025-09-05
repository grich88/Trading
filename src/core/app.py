"""
Core application module.

This module provides the main application class that orchestrates
the interaction between all services and components.
"""

import os
import sys
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Import services
from src.services import BaseService, DataService, AnalysisService

# Import utilities
from src.utils import get_logger, exception_handler, performance_monitor

# Import configuration
from src.config import (
    APP_MODE,
    DATA_DIR,
    DEFAULT_TIMEFRAME,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_ABS_SCORE
)


class TradingApp:
    """
    Main application class.
    
    This class orchestrates the interaction between all services and components,
    providing a unified interface for the application.
    """
    
    def __init__(self, 
                 app_mode: str = APP_MODE,
                 data_dir: str = DATA_DIR,
                 default_timeframe: str = DEFAULT_TIMEFRAME,
                 default_window_size: int = DEFAULT_WINDOW_SIZE,
                 default_min_abs_score: float = DEFAULT_MIN_ABS_SCORE):
        """
        Initialize the application.
        
        Args:
            app_mode: Application mode (development, testing, production)
            data_dir: Directory for storing data
            default_timeframe: Default timeframe for data
            default_window_size: Default window size for analysis
            default_min_abs_score: Default minimum absolute score for signals
        """
        self.app_mode = app_mode
        self.data_dir = data_dir
        self.default_timeframe = default_timeframe
        self.default_window_size = default_window_size
        self.default_min_abs_score = default_min_abs_score
        
        # Initialize logger
        self.logger = get_logger("TradingApp")
        
        # Initialize services
        self.services = {}
        self._initialize_services()
        
        # Application state
        self.running = False
        self.start_time = None
        
        self.logger.info(f"Trading application initialized (mode: {app_mode})")
    
    def _initialize_services(self) -> None:
        """Initialize all services."""
        # Data service
        self.services["data"] = DataService(
            data_dir=self.data_dir,
            default_timeframe=self.default_timeframe
        )
        
        # Analysis service
        self.services["analysis"] = AnalysisService(
            default_window_size=self.default_window_size,
            default_min_abs_score=self.default_min_abs_score
        )
        
        self.logger.info(f"Initialized {len(self.services)} services")
    
    def start(self) -> bool:
        """
        Start the application and all services.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            self.logger.warning("Application is already running")
            return True
        
        try:
            self.logger.info("Starting application")
            
            # Start all services
            for name, service in self.services.items():
                if not service.start():
                    self.logger.error(f"Failed to start service: {name}")
                    return False
            
            # Set application state
            self.running = True
            self.start_time = datetime.now()
            
            self.logger.info("Application started successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error starting application: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the application and all services.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            self.logger.warning("Application is not running")
            return True
        
        try:
            self.logger.info("Stopping application")
            
            # Stop all services in reverse order
            for name, service in reversed(list(self.services.items())):
                if not service.stop():
                    self.logger.error(f"Failed to stop service: {name}")
            
            # Set application state
            self.running = False
            
            self.logger.info("Application stopped successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping application: {str(e)}")
            return False
    
    def get_service(self, name: str) -> Optional[BaseService]:
        """
        Get a service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance, or None if not found
        """
        return self.services.get(name)
    
    def get_data_service(self) -> DataService:
        """
        Get the data service.
        
        Returns:
            Data service instance
        """
        return self.services["data"]
    
    def get_analysis_service(self) -> AnalysisService:
        """
        Get the analysis service.
        
        Returns:
            Analysis service instance
        """
        return self.services["analysis"]
    
    @performance_monitor()
    def run_full_analysis(self, 
                        assets: List[str],
                        timeframe: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        window_size: Optional[int] = None,
                        min_abs_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Run a full analysis for the specified assets.
        
        Args:
            assets: List of assets to analyze
            timeframe: Timeframe for data
            start_date: Start date for data
            end_date: End date for data
            window_size: Window size for analysis
            min_abs_score: Minimum absolute score for signals
            
        Returns:
            Analysis results
        """
        if not self.running:
            self.logger.error("Application is not running")
            return {"error": "Application is not running"}
        
        try:
            # Set default values
            timeframe = timeframe or self.default_timeframe
            window_size = window_size or self.default_window_size
            min_abs_score = min_abs_score or self.default_min_abs_score
            
            # Get services
            data_service = self.get_data_service()
            analysis_service = self.get_analysis_service()
            
            # Fetch data for all assets
            self.logger.info(f"Fetching data for {len(assets)} assets")
            data = data_service.get_data_for_assets(
                assets=assets,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if not data:
                self.logger.error("Failed to fetch data for any assets")
                return {"error": "Failed to fetch data"}
            
            # Calculate WebTrend for all assets
            self.logger.info("Calculating WebTrend indicators")
            webtrend_data = {}
            for asset, df in data.items():
                try:
                    webtrend = analysis_service.calculate_webtrend(df)
                    webtrend_data[asset] = webtrend
                except Exception as e:
                    self.logger.error(f"Error calculating WebTrend for {asset}: {str(e)}")
            
            # Analyze cross-asset relationships
            self.logger.info("Analyzing cross-asset relationships")
            cross_asset_biases = analysis_service.analyze_cross_asset(data)
            
            # Generate signals
            self.logger.info("Generating signals")
            signals = analysis_service.generate_signals(
                data=data,
                webtrend_data=webtrend_data,
                cross_asset_biases=cross_asset_biases,
                min_abs_score=min_abs_score
            )
            
            # Compile results
            results = {
                "signals": signals,
                "webtrend_data": webtrend_data,
                "cross_asset_biases": cross_asset_biases,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("Analysis completed successfully")
            return results
        
        except Exception as e:
            self.logger.error(f"Error running analysis: {str(e)}")
            return {"error": str(e)}
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get health status for the application and all services.
        
        Returns:
            Dictionary with health information
        """
        health = {
            "app": {
                "mode": self.app_mode,
                "running": self.running,
                "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            },
            "services": {}
        }
        
        # Get health for all services
        for name, service in self.services.items():
            health["services"][name] = service.get_health()
        
        return health


def main():
    """Main entry point for the application."""
    # Create application
    app = TradingApp()
    
    # Start application
    if not app.start():
        sys.exit(1)
    
    try:
        # Run analysis for example assets
        results = app.run_full_analysis(
            assets=["BTC", "SOL", "BONK"],
            timeframe="4h",
            start_date="2023-01-01"
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
    
    finally:
        # Stop application
        app.stop()


if __name__ == "__main__":
    main()
