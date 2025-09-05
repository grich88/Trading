import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

class CrossAssetAnalyzer:
    """
    Analyzes relationships between clusters across different assets
    to identify lead/lag patterns and correlation effects.
    """
    
    def __init__(self, clusters_by_asset):
        self.clusters_by_asset = clusters_by_asset
        self.assets = list(clusters_by_asset.keys())
        self.current_prices = {}
        self.relationships = {}
        
    def set_current_prices(self, price_dict):
        self.current_prices = price_dict
        
    def calculate_cluster_distance(self, normalize=True):
        results = {}
        
        for asset, clusters in self.clusters_by_asset.items():
            if not clusters or asset not in self.current_prices:
                continue
            
            current_price = self.current_prices[asset]
            max_price = max(c[0] for c in clusters)
            min_price = min(c[0] for c in clusters)
            price_range = max_price - min_price
            
            # Calculate distance of each cluster from current price
            cluster_distances = []
            for price, strength in clusters:
                abs_distance = abs(price - current_price)
                pct_distance = abs_distance / current_price
                norm_distance = abs_distance / price_range if normalize and price_range > 0 else abs_distance
                direction = "above" if price > current_price else "below"
                
                cluster_distances.append({
                    "price": price,
                    "strength": strength,
                    "abs_distance": abs_distance,
                    "pct_distance": pct_distance,
                    "norm_distance": norm_distance,
                    "direction": direction
                })
                
                results[asset] = {
                "current_price": current_price,
                "price_range": price_range,
                "cluster_distances": cluster_distances
                }
        
        return results
    
    def identify_lead_lag_patterns(self):
        if len(self.assets) < 2:
            return {"error": "Need at least 2 assets to identify lead/lag relationships"}
            
        if not self.current_prices:
            return {"error": "Current prices must be set before analyzing lead/lag patterns"}
        
        # Safety check: ensure all assets have current prices
        for asset in self.assets:
            if asset not in self.current_prices:
                return {"error": f"Missing current price for {asset}"}
                
        try:    
            distance_data = self.calculate_cluster_distance()
            relationships = {}
            
            # Analyze each pair of assets
            for i, asset1 in enumerate(self.assets):
                if asset1 not in distance_data:
                    continue
                
                for asset2 in self.assets[i+1:]:  # Note the slice here
                    if asset2 not in distance_data:
                    continue
                
                    # Get the closest strong liquidation cluster for each asset
                    asset1_data = distance_data[asset1]
                    asset2_data = distance_data[asset2]
                    
                    # Get strongest clusters above and below current price
                    asset1_above = [c for c in asset1_data["cluster_distances"] if c["direction"] == "above"]
                    asset1_below = [c for c in asset1_data["cluster_distances"] if c["direction"] == "below"]
                    asset2_above = [c for c in asset2_data["cluster_distances"] if c["direction"] == "above"]
                    asset2_below = [c for c in asset2_data["cluster_distances"] if c["direction"] == "below"]
                    
                    # Sort by strength and get the strongest
                    if asset1_above:
                        asset1_above = sorted(asset1_above, key=lambda x: x["strength"], reverse=True)[0]
                    if asset1_below:
                        asset1_below = sorted(asset1_below, key=lambda x: x["strength"], reverse=True)[0]
                    if asset2_above:
                        asset2_above = sorted(asset2_above, key=lambda x: x["strength"], reverse=True)[0]
                    if asset2_below:
                        asset2_below = sorted(asset2_below, key=lambda x: x["strength"], reverse=True)[0]
                    
                    # Calculate relative distances to resistance/support
                    resistance_ratio = None
                    support_ratio = None
                    
                    if asset1_above and asset2_above:
                        resistance_ratio = asset1_above["norm_distance"] / asset2_above["norm_distance"] if asset2_above["norm_distance"] > 0 else 1.0
                        
                    if asset1_below and asset2_below:
                        support_ratio = asset1_below["norm_distance"] / asset2_below["norm_distance"] if asset2_below["norm_distance"] > 0 else 1.0
                    
                    # Determine lead/lag relationship
                    relationship = {
                        "assets": [asset1, asset2],
                        "resistance_ratio": resistance_ratio,
                        "support_ratio": support_ratio,
                        "resistance_clusters": {
                            asset1: asset1_above if asset1_above else None,
                            asset2: asset2_above if asset2_above else None
                        },
                        "support_clusters": {
                            asset1: asset1_below if asset1_below else None,
                            asset2: asset2_below if asset2_below else None
                        }
                    }
                    
                    # Determine which asset is leading based on cluster proximity
                    if resistance_ratio is not None:
                        if resistance_ratio < 0.8:
                            relationship["resistance_leader"] = asset1
                        elif resistance_ratio > 1.2:
                            relationship["resistance_leader"] = asset2
                    
                    if support_ratio is not None:
                        if support_ratio < 0.8:
                            relationship["support_leader"] = asset1
                        elif support_ratio > 1.2:
                            relationship["support_leader"] = asset2
                    
                    # Store the relationship
                    pair_key = f"{asset1}_{asset2}"
                    relationships[pair_key] = relationship
            
            self.relationships = relationships
            return relationships
        except Exception as e:
            return {"error": f"Error analyzing lead/lag patterns: {str(e)}"}
    
    def generate_cross_asset_report(self):
        if not self.relationships:
            self.identify_lead_lag_patterns()
            
        if not self.relationships:
            return "No cross-asset relationships identified. Ensure multiple assets have been analyzed."
            
        report = []
        report.append("# Cross-Asset Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add current prices section
        report.append("## Current Prices")
        for asset, price in self.current_prices.items():
            if asset in self.assets:
                report.append(f"- {asset}: {price}")
        report.append("")
        
        # Add lead/lag relationships
        report.append("## Lead/Lag Relationships")
        
        for pair_key, relationship in self.relationships.items():
            if "error" in relationship:
                continue
                
            asset1, asset2 = relationship["assets"]
            report.append(f"### {asset1} vs {asset2}")
            
            # Resistance analysis
            if "resistance_leader" in relationship:
                leader = relationship["resistance_leader"]
                follower = asset2 if leader == asset1 else asset1
                report.append(f"- **Resistance**: {leader} leads {follower}")
                
                # Add details about the clusters
                if relationship["resistance_clusters"][leader]:
                    leader_price = relationship["resistance_clusters"][leader]["price"]
                    leader_strength = relationship["resistance_clusters"][leader]["strength"]
                    report.append(f"  - {leader} resistance at {leader_price} (strength: {leader_strength:.2f})")
                
                if relationship["resistance_clusters"][follower]:
                    follower_price = relationship["resistance_clusters"][follower]["price"]
                    follower_strength = relationship["resistance_clusters"][follower]["strength"]
                    report.append(f"  - {follower} resistance at {follower_price} (strength: {follower_strength:.2f})")
            
            # Support analysis
            if "support_leader" in relationship:
                leader = relationship["support_leader"]
                follower = asset2 if leader == asset1 else asset1
                report.append(f"- **Support**: {leader} leads {follower}")
                
                # Add details about the clusters
                if relationship["support_clusters"][leader]:
                    leader_price = relationship["support_clusters"][leader]["price"]
                    leader_strength = relationship["support_clusters"][leader]["strength"]
                    report.append(f"  - {leader} support at {leader_price} (strength: {leader_strength:.2f})")
                
                if relationship["support_clusters"][follower]:
                    follower_price = relationship["support_clusters"][follower]["price"]
                    follower_strength = relationship["support_clusters"][follower]["strength"]
                    report.append(f"  - {follower} support at {follower_price} (strength: {follower_strength:.2f})")
            
            report.append("")
        
        return "\n".join(report)
    
    def plot_cross_asset_clusters(self, save_path=None):
        """
        Plot liquidation clusters for each asset with current price markers
        """
        if not self.assets or not self.current_prices:
            return None
            
        # Create a figure with subplots for each asset
        n_assets = len(self.assets)
        if n_assets == 0:
            return None
            
        fig, axes = plt.subplots(n_assets, 1, figsize=(10, 4 * n_assets))
        if n_assets == 1:
            axes = [axes]  # Make it iterable
            
        for i, asset in enumerate(self.assets):
            ax = axes[i]
            
            if asset not in self.clusters_by_asset or not self.clusters_by_asset[asset]:
                ax.text(0.5, 0.5, f"No clusters for {asset}", 
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title(f"{asset} - No Data")
                continue
                
            clusters = self.clusters_by_asset[asset]
            prices = [c[0] for c in clusters]
            strengths = [c[1] for c in clusters]
            
            # Safety check for very small or negative prices
            if min(prices) <= 0:
                ax.text(0.5, 0.5, f"Invalid price data for {asset}", 
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title(f"{asset} - Invalid Data")
                continue
            
            # Adjust bar width based on price range
            price_range = max(prices) - min(prices)
            if price_range <= 0:
                bar_width = 0.01
            else:
                # For very small prices (like BONK), use a smaller relative width
                if max(prices) < 0.01:
                    bar_width = price_range * 0.01
                else:
                    bar_width = price_range * 0.02
            
            # Plot the clusters as bars
            ax.bar(prices, strengths, width=bar_width, alpha=0.7, color='blue')
            
            # Add current price marker if available
            if asset in self.current_prices:
                current_price = self.current_prices[asset]
                ax.axvline(x=current_price, color='red', linestyle='--', label=f"Current Price: {current_price}")
            
            ax.set_title(f"{asset} Liquidation Clusters")
            ax.set_xlabel("Price")
            ax.set_ylabel("Strength")
            ax.legend()
            
            # Set x-axis limits with some padding
            if price_range > 0:
                padding = price_range * 0.1
                ax.set_xlim(min(prices) - padding, max(prices) + padding)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def save_analysis(self, filepath):
        """
        Save the analysis results to a JSON file
        """
        if not self.relationships:
            self.identify_lead_lag_patterns()
            
        data = {
            "timestamp": datetime.now().isoformat(),
            "assets": self.assets,
            "current_prices": self.current_prices,
            "relationships": self.relationships
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath