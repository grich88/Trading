#!/usr/bin/env python3
"""
Comprehensive debug script for analyzing liquidation clusters from heatmap images.
This script follows the full process of analysis from image loading to cluster extraction
and cross-asset correlation analysis.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LiquidationAnalysis")

# Import our extraction functions
try:
    from image_extractors import extract_liq_clusters_from_image, auto_detect_heatmap_scale, detect_token_from_image
    from cross_asset_analyzer import CrossAssetAnalyzer
    logger.info("Successfully imported extraction modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class LiquidationAnalysisDebugger:
    """Comprehensive debugger for liquidation cluster analysis"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"debug_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
        logger.info(f"Initialized LiquidationAnalysisDebugger with output dir: {self.output_dir}")
    
    def find_image_files(self, directory: str) -> List[str]:
        """Find all image files in the given directory"""
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_files)} image files in {directory}")
        return image_files
    
    def process_single_image(self, image_path: str, auto_detect: bool = True) -> Tuple[List, str]:
        """Process a single image and extract liquidation clusters"""
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Read image file
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            
            # Auto-detect price scale and token
            detected_range = None
            detected_token = None
            
            if auto_detect:
                try:
                    price_range = auto_detect_heatmap_scale(img_bytes)
                    if price_range:
                        detected_top, detected_bottom, detected_token = price_range
                        logger.info(f"Auto-detected price range: {detected_top} - {detected_bottom} for {detected_token}")
                        detected_range = (detected_top, detected_bottom)
                except Exception as e:
                    logger.error(f"Error during price scale detection: {str(e)}")
            
            # If auto-detection failed, use token-specific defaults
            if not detected_range and detected_token:
                token_range = self.price_ranges.get(detected_token, {"min": 100.0, "max": 1000.0, "typical": 500.0})
                detected_top = token_range["typical"] * 1.2
                detected_bottom = token_range["typical"] * 0.8
                logger.info(f"Using default range for {detected_token}: {detected_top} - {detected_bottom}")
                detected_range = (detected_top, detected_bottom)
            
            # If we still don't have a range or token, try to detect token from image
            if not detected_token:
                detected_token = detect_token_from_image(img_bytes)
                logger.info(f"Detected token from image content: {detected_token}")
            
            # If we have a token but no range, use defaults
            if detected_token and not detected_range:
                token_range = self.price_ranges.get(detected_token, {"min": 100.0, "max": 1000.0, "typical": 500.0})
                detected_top = token_range["typical"] * 1.2
                detected_bottom = token_range["typical"] * 0.8
                logger.info(f"Using default range for {detected_token}: {detected_top} - {detected_bottom}")
                detected_range = (detected_top, detected_bottom)
            
            # If we still don't have a range or token, use BTC as default
            if not detected_range:
                detected_token = "BTC"
                detected_top = self.price_ranges["BTC"]["typical"] * 1.2
                detected_bottom = self.price_ranges["BTC"]["typical"] * 0.8
                logger.info(f"Using default BTC range: {detected_top} - {detected_bottom}")
                detected_range = (detected_top, detected_bottom)
            
            # Extract clusters with the detected or default range
            price_top, price_bottom = detected_range
            clusters = extract_liq_clusters_from_image(
                img_bytes, 
                price_top=price_top, 
                price_bottom=price_bottom,
                auto_adjust_range=True
            )
            
            # Apply asset-specific filtering
            if detected_token and clusters:
                asset_range = self.price_ranges.get(detected_token, {"min": 0, "max": float('inf')})
                min_valid_price = asset_range["min"]
                max_valid_price = asset_range["max"]
                
                original_clusters = clusters.copy()
                valid_clusters = [c for c in clusters if min_valid_price <= c[0] <= max_valid_price]
                
                if len(valid_clusters) < len(clusters):
                    filtered_out = [c for c in clusters if c not in valid_clusters]
                    logger.warning(f"Filtered out {len(clusters) - len(valid_clusters)} clusters that were outside reasonable price range ({min_valid_price:.2f} - {max_valid_price:.2f})")
                    logger.info(f"Filtered clusters: {', '.join([f'{c[0]:.2f}' for c in filtered_out])}")
                    clusters = valid_clusters
                
                # Check if we have any valid clusters left
                if not clusters and original_clusters:
                    # Sort by strength and take top 2
                    strongest_clusters = sorted(original_clusters, key=lambda x: x[1], reverse=True)[:2]
                    logger.warning(f"Restoring {len(strongest_clusters)} strongest clusters as fallback")
                    clusters = strongest_clusters
            
            # Add image metadata
            enriched_clusters = []
            for cluster in clusters:
                # Handle the new format that includes metadata
                if len(cluster) > 2:
                    # Already has metadata
                    cluster[2]["image_path"] = image_path
                    cluster[2]["token"] = detected_token
                    enriched_clusters.append(cluster)
                else:
                    # Old format, convert to new format with metadata
                    price, strength = cluster
                    metadata = {
                        "image_path": image_path,
                        "token": detected_token
                    }
                    enriched_clusters.append((price, strength, metadata))
            
            logger.info(f"Extracted {len(enriched_clusters)} clusters from {image_path}")
            return enriched_clusters, detected_token
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return [], None
    
    def process_all_images(self, image_dir: str) -> Dict[str, List]:
        """Process all images in the directory and organize clusters by asset"""
        image_files = self.find_image_files(image_dir)
        
        # Reset storage
        self.all_clusters = []
        self.clusters_by_asset = {}
        
        for i, image_path in enumerate(image_files):
            logger.info(f"Processing image {i+1}/{len(image_files)}: {image_path}")
            
            clusters, detected_token = self.process_single_image(image_path)
            
            if clusters:
                # Add to all clusters
                self.all_clusters.extend(clusters)
                
                # Add to asset-specific clusters
                if detected_token:
                    if detected_token not in self.clusters_by_asset:
                        self.clusters_by_asset[detected_token] = []
                    self.clusters_by_asset[detected_token].extend(clusters)
        
        # Log summary
        logger.info(f"Processed {len(image_files)} images")
        logger.info(f"Extracted {len(self.all_clusters)} total clusters")
        for asset, clusters in self.clusters_by_asset.items():
            logger.info(f"Asset {asset}: {len(clusters)} clusters")
        
        return self.clusters_by_asset
    
    def combine_clusters_for_asset(self, asset: str, similarity_threshold: float = 0.01) -> List:
        """Combine clusters for a specific asset based on price similarity"""
        if asset not in self.clusters_by_asset:
            logger.warning(f"No clusters found for asset {asset}")
            return []
        
        asset_clusters = self.clusters_by_asset[asset]
        if not asset_clusters:
            logger.warning(f"Empty cluster list for asset {asset}")
            return []
        
        logger.info(f"Combining {len(asset_clusters)} clusters for {asset}")
        
        # Group by similar price levels
        combined_clusters = {}
        
        for cluster in asset_clusters:
            # Handle both tuple formats (price, strength) or (price, strength, metadata)
            price = cluster[0]
            strength = cluster[1]
            metadata = cluster[2] if len(cluster) > 2 else {}
            
            found_match = False
            for existing_price in list(combined_clusters.keys()):
                # If within threshold of an existing price point, combine them
                if existing_price > 0 and abs(price - existing_price) / existing_price < similarity_threshold:
                    # Keep the highest strength
                    if strength > combined_clusters[existing_price][0]:
                        combined_clusters[existing_price] = (strength, metadata)
                    found_match = True
                    break
            
            if not found_match:
                combined_clusters[price] = (strength, metadata)
        
        # Convert back to list of tuples with metadata
        final_clusters = []
        for price, strength_data in combined_clusters.items():
            strength = strength_data[0]
            metadata = strength_data[1]
            final_clusters.append((price, strength, metadata))
        
        logger.info(f"Combined into {len(final_clusters)} final clusters for {asset}")
        
        # Store in the combined clusters dictionary
        self.combined_clusters[asset] = final_clusters
        
        return final_clusters
    
    def visualize_clusters(self, asset: str, save_path: Optional[str] = None) -> None:
        """Visualize clusters for a specific asset"""
        if asset not in self.combined_clusters:
            logger.warning(f"No combined clusters found for asset {asset}")
            return
        
        clusters = self.combined_clusters[asset]
        if not clusters:
            logger.warning(f"Empty combined cluster list for asset {asset}")
            return
        
        # Create DataFrame for visualization
        df_data = []
        for i, cluster in enumerate(clusters):
            price = cluster[0]
            strength = cluster[1]
            row = {"Index": i, "Price": price, "Strength": strength}
            df_data.append(row)
        
        df_clusters = pd.DataFrame(df_data)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate appropriate bar width based on price range
        if len(df_clusters["Price"]) > 1:
            price_range = max(df_clusters["Price"]) - min(df_clusters["Price"])
            bar_width = price_range * 0.01  # 1% of range
        else:
            # Default width if only one price point
            bar_width = max(df_clusters["Price"]) * 0.01 if not df_clusters["Price"].empty else 1.0
        
        ax.bar(df_clusters["Price"], df_clusters["Strength"], width=bar_width)
        ax.set_xlabel("Price")
        ax.set_ylabel("Strength")
        ax.set_title(f"Combined Liquidation Clusters for {asset}")
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        else:
            save_path = os.path.join(self.output_dir, f"{asset}_clusters.png")
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        plt.close()
    
    def perform_cross_asset_analysis(self) -> Dict:
        """Perform cross-asset analysis using the CrossAssetAnalyzer"""
        if not self.combined_clusters:
            logger.warning("No combined clusters available for cross-asset analysis")
            return {}
        
        logger.info("Performing cross-asset analysis")
        
        try:
            # Create analyzer with combined clusters
            analyzer = CrossAssetAnalyzer(self.combined_clusters)
            
            # Set current prices (using midpoint of clusters as a proxy)
            current_prices = {}
            for asset, clusters in self.combined_clusters.items():
                if clusters:
                    prices = [c[0] for c in clusters]
                    current_prices[asset] = sum(prices) / len(prices)
            
            analyzer.set_current_prices(current_prices)
            
            # Calculate distances
            distances = analyzer.calculate_cluster_distance()
            logger.info(f"Calculated cluster distances: {json.dumps(distances, indent=2)}")
            
            # Identify lead/lag patterns
            relationships = analyzer.identify_lead_lag_patterns()
            logger.info(f"Identified relationships: {json.dumps(relationships, indent=2)}")
            
            # Generate report
            report = analyzer.generate_cross_asset_report()
            logger.info(f"Cross-asset report:\n{report}")
            
            # Save report
            report_path = os.path.join(self.output_dir, "cross_asset_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            
            # Generate visualization
            plot_path = os.path.join(self.output_dir, "cross_asset_plot.png")
            analyzer.plot_cross_asset_clusters(save_path=plot_path)
            
            # Store results
            self.cross_asset_results = {
                "distances": distances,
                "relationships": relationships,
                "report": report,
                "plot_path": plot_path
            }
            
            return self.cross_asset_results
            
        except Exception as e:
            logger.error(f"Error in cross-asset analysis: {str(e)}")
            return {}
    
    def run_full_analysis(self, image_dir: str) -> Dict:
        """Run the full analysis pipeline"""
        start_time = time.time()
        logger.info(f"Starting full analysis of images in {image_dir}")
        
        # Process all images
        self.process_all_images(image_dir)
        
        # Combine clusters for each asset
        for asset in self.clusters_by_asset.keys():
            self.combine_clusters_for_asset(asset)
        
        # Visualize clusters for each asset
        for asset in self.combined_clusters.keys():
            self.visualize_clusters(asset)
        
        # Perform cross-asset analysis
        if len(self.combined_clusters) >= 2:
            self.perform_cross_asset_analysis()
        
        # Save all results
        self.save_results()
        
        end_time = time.time()
        logger.info(f"Full analysis completed in {end_time - start_time:.2f} seconds")
        
        return {
            "clusters_by_asset": self.clusters_by_asset,
            "combined_clusters": self.combined_clusters,
            "cross_asset_results": self.cross_asset_results,
            "output_dir": self.output_dir
        }
    
    def save_results(self) -> None:
        """Save all analysis results to files"""
        # Save clusters by asset
        clusters_file = os.path.join(self.output_dir, "clusters_by_asset.json")
        with open(clusters_file, "w") as f:
            # Convert tuples to lists for JSON serialization
            serializable_clusters = {}
            for asset, clusters in self.clusters_by_asset.items():
                serializable_clusters[asset] = [list(c) for c in clusters]
            json.dump(serializable_clusters, f, indent=2)
        
        # Save combined clusters
        combined_file = os.path.join(self.output_dir, "combined_clusters.json")
        with open(combined_file, "w") as f:
            # Convert tuples to lists for JSON serialization
            serializable_combined = {}
            for asset, clusters in self.combined_clusters.items():
                serializable_combined[asset] = [list(c) for c in clusters]
            json.dump(serializable_combined, f, indent=2)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "analysis_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"Analysis completed at: {datetime.now()}\n\n")
            f.write(f"Total clusters extracted: {len(self.all_clusters)}\n")
            f.write(f"Assets detected: {', '.join(self.clusters_by_asset.keys())}\n\n")
            
            for asset, clusters in self.clusters_by_asset.items():
                f.write(f"{asset}: {len(clusters)} raw clusters, ")
                if asset in self.combined_clusters:
                    f.write(f"{len(self.combined_clusters[asset])} combined clusters\n")
                else:
                    f.write("0 combined clusters\n")
            
            if self.cross_asset_results:
                f.write("\nCross-asset analysis completed.\n")
                if "relationships" in self.cross_asset_results:
                    f.write("Relationships detected:\n")
                    for asset1, relations in self.cross_asset_results["relationships"].items():
                        for asset2, relation in relations.items():
                            f.write(f"  {asset1} → {asset2}: {relation}\n")
        
        logger.info(f"Saved all results to {self.output_dir}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Debug liquidation cluster analysis")
    parser.add_argument("--image_dir", type=str, default="./data/heatmaps", 
                      help="Directory containing heatmap images")
    args = parser.parse_args()
    
    # Run analysis
    debugger = LiquidationAnalysisDebugger()
    results = debugger.run_full_analysis(args.image_dir)
    
    print(f"\nAnalysis complete! Results saved to: {results['output_dir']}")
    print(f"Assets detected: {', '.join(results['clusters_by_asset'].keys())}")
    for asset, clusters in results['clusters_by_asset'].items():
        print(f"{asset}: {len(clusters)} clusters")
