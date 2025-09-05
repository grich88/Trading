#!/usr/bin/env python3
"""
Advanced debugging tool for trading system and liquidation analysis.
This script provides comprehensive debugging capabilities with detailed logging.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import argparse
from typing import List, Dict, Tuple, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AdvancedDebugger")

# Try to import necessary modules
try:
    from image_extractors import (
        extract_liq_clusters_from_image, 
        auto_detect_heatmap_scale, 
        detect_token_from_image
    )
    from cross_asset_analyzer import CrossAssetAnalyzer
    logger.info("Successfully imported extraction modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class AdvancedDebugger:
    """Advanced debugging tool for trading system and liquidation analysis"""
    
    def __init__(self, verbose=True):
        """Initialize the debugger with timestamp and output directory"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"advanced_debug_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.verbose = verbose
        
        # Asset-specific configuration
        self.price_ranges = {
            "BTC": {"min": 60000.0, "max": 200000.0, "typical": 140000.0},
            "ETH": {"min": 1000.0, "max": 10000.0, "typical": 3500.0},
            "SOL": {"min": 20.0, "max": 500.0, "typical": 180.0},
            "XRP": {"min": 0.1, "max": 2.0, "typical": 0.6},
            "DOGE": {"min": 0.01, "max": 1.0, "typical": 0.15},
            "BONK": {"min": 0.00001, "max": 0.01, "typical": 0.0001}
        }
        
        # Storage for analysis results
        self.all_clusters = []
        self.clusters_by_asset = {}
        self.combined_clusters = {}
        self.cross_asset_results = {}
        
        logger.info(f"Initialized AdvancedDebugger with output dir: {self.output_dir}")
    
    def debug_function(self, func, *args, **kwargs):
        """Debug a function call with detailed logging"""
        func_name = func.__name__
        logger.info(f"Debugging function: {func_name}")
        logger.info(f"Args: {args}")
        logger.info(f"Kwargs: {kwargs}")
        
        try:
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Function {func_name} completed in {duration:.2f} seconds")
            logger.info(f"Result type: {type(result)}")
            
            if isinstance(result, (list, tuple)) and len(result) < 10:
                logger.info(f"Result: {result}")
            elif isinstance(result, dict) and len(result) < 10:
                logger.info(f"Result: {result}")
            else:
                logger.info(f"Result summary: {type(result)} of length {len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            return result
        except Exception as e:
            logger.error(f"Error in function {func_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def debug_image_extraction(self, image_path: str) -> Dict:
        """Debug the image extraction process"""
        logger.info(f"Debugging image extraction for: {image_path}")
        
        try:
            # Read image file
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            
            # Step 1: Auto-detect scale
            logger.info("Step 1: Auto-detecting scale")
            scale_result = self.debug_function(auto_detect_heatmap_scale, img_bytes)
            
            if scale_result:
                if len(scale_result) == 3:
                    top_price, bottom_price, detected_token = scale_result
                    logger.info(f"Detected scale: {top_price} - {bottom_price} for {detected_token}")
                else:
                    top_price, bottom_price = scale_result
                    logger.info(f"Detected scale: {top_price} - {bottom_price}")
                    
                    # Try to detect token separately
                    detected_token = self.debug_function(detect_token_from_image, img_bytes)
                    logger.info(f"Separately detected token: {detected_token}")
            else:
                logger.warning("Failed to auto-detect scale")
                # Use default values
                detected_token = "BTC"  # Default token
                top_price = self.price_ranges[detected_token]["typical"] * 1.2
                bottom_price = self.price_ranges[detected_token]["typical"] * 0.8
                logger.info(f"Using default scale: {top_price} - {bottom_price} for {detected_token}")
            
            # Step 2: Extract clusters
            logger.info("Step 2: Extracting liquidation clusters")
            clusters = self.debug_function(
                extract_liq_clusters_from_image,
                img_bytes,
                price_top=top_price,
                price_bottom=bottom_price
            )
            
            # Save the image for reference
            img = Image.open(io.BytesIO(img_bytes))
            img_save_path = os.path.join(self.output_dir, f"debug_image_{os.path.basename(image_path)}")
            img.save(img_save_path)
            logger.info(f"Saved debug image to: {img_save_path}")
            
            # Save clusters to file
            clusters_save_path = os.path.join(self.output_dir, f"clusters_{os.path.basename(image_path)}.json")
            with open(clusters_save_path, 'w') as f:
                # Convert clusters to serializable format
                serializable_clusters = []
                for cluster in clusters:
                    if len(cluster) > 2:
                        price, strength, metadata = cluster
                        serializable_clusters.append({
                            "price": float(price),
                            "strength": float(strength),
                            "metadata": metadata
                        })
                    else:
                        price, strength = cluster
                        serializable_clusters.append({
                            "price": float(price),
                            "strength": float(strength)
                        })
                json.dump(serializable_clusters, f, indent=2)
            logger.info(f"Saved clusters to: {clusters_save_path}")
            
            # Plot clusters
            self.plot_clusters(clusters, detected_token, os.path.join(self.output_dir, f"plot_{os.path.basename(image_path)}.png"))
            
            return {
                "image_path": image_path,
                "top_price": top_price,
                "bottom_price": bottom_price,
                "detected_token": detected_token,
                "clusters": clusters,
                "clusters_count": len(clusters)
            }
        except Exception as e:
            logger.error(f"Error debugging image extraction: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "image_path": image_path,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def plot_clusters(self, clusters, token, save_path):
        """Plot clusters and save to file"""
        try:
            # Extract prices and strengths
            prices = [c[0] for c in clusters]
            strengths = [c[1] for c in clusters]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot clusters as bars
            ax.bar(prices, strengths, width=max(prices)/100 if prices else 1, alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('Price')
            ax.set_ylabel('Strength')
            ax.set_title(f'Liquidation Clusters for {token}')
            
            # Set y-axis limits
            ax.set_ylim(0, 1.1)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved cluster plot to: {save_path}")
        except Exception as e:
            logger.error(f"Error plotting clusters: {str(e)}")
    
    def debug_directory(self, directory_path: str) -> Dict:
        """Debug all images in a directory"""
        logger.info(f"Debugging all images in directory: {directory_path}")
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Debug each image
        results = []
        for image_path in image_files:
            result = self.debug_image_extraction(image_path)
            results.append(result)
        
        # Summarize results
        summary = {
            "total_images": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "tokens_detected": set(r["detected_token"] for r in results if "detected_token" in r),
            "total_clusters": sum(r.get("clusters_count", 0) for r in results)
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "debug_summary.json")
        with open(summary_path, 'w') as f:
            # Convert set to list for JSON serialization
            summary["tokens_detected"] = list(summary["tokens_detected"])
            json.dump(summary, f, indent=2)
        
        logger.info(f"Debug summary: {summary}")
        return summary
    
    def debug_cross_asset_analysis(self, clusters_by_asset, current_prices=None):
        """Debug cross-asset analysis"""
        logger.info("Debugging cross-asset analysis")
        
        try:
            # Initialize cross-asset analyzer
            analyzer = CrossAssetAnalyzer(clusters_by_asset, current_prices)
            
            # Calculate cluster distances
            logger.info("Calculating cluster distances")
            distances = self.debug_function(analyzer.calculate_cluster_distance)
            
            # Identify lead/lag patterns
            logger.info("Identifying lead/lag patterns")
            patterns = self.debug_function(analyzer.identify_lead_lag_patterns)
            
            # Plot cross-asset clusters
            logger.info("Plotting cross-asset clusters")
            plot_path = os.path.join(self.output_dir, "cross_asset_plot.png")
            fig = analyzer.plot_cross_asset_clusters()
            if fig:
                fig.savefig(plot_path, dpi=300)
                plt.close(fig)
                logger.info(f"Saved cross-asset plot to: {plot_path}")
            
            return {
                "distances": distances,
                "patterns": patterns,
                "plot_path": plot_path if fig else None
            }
        except Exception as e:
            logger.error(f"Error debugging cross-asset analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

def main():
    """Main function to run the advanced debugger"""
    parser = argparse.ArgumentParser(description="Advanced debugging tool for trading system")
    parser.add_argument("--image", help="Path to a single image to debug")
    parser.add_argument("--dir", help="Path to a directory of images to debug")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    debugger = AdvancedDebugger(verbose=args.verbose)
    
    if args.image:
        result = debugger.debug_image_extraction(args.image)
        print(f"Debug result for {args.image}:")
        print(json.dumps(result, indent=2))
    
    elif args.dir:
        summary = debugger.debug_directory(args.dir)
        print(f"Debug summary for directory {args.dir}:")
        print(json.dumps(summary, indent=2))
    
    else:
        print("Please provide either --image or --dir argument")
        parser.print_help()

if __name__ == "__main__":
    main()
