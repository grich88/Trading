#!/usr/bin/env python3
"""
Script to create GitHub issues for the Trading Algorithm System project.
"""

import subprocess
import json
import time

# Define the issues to create
issues = [
    {
        "title": "Initialize Project Repository",
        "body": """**Problem Statement:**
We need to establish a well-structured repository following the Master Guide principles to ensure clean development and maintainability.

**Definition of Done:**
- Repository created with proper folder structure
- Basic README.md with project overview
- .gitignore configured for Python projects
- MASTER_GUIDE.md added with project principles
- .env.example with all required environment variables""",
        "category": "Loose Ideas"
    },
    {
        "title": "Set Up Configuration Management",
        "body": """**Problem Statement:**
We need a centralized configuration system that loads settings from environment variables with sensible defaults.

**Definition of Done:**
- Configuration module created with proper error handling
- Support for different environment types (dev, test, prod)
- Validation for required configuration values
- Documentation for all configuration options
- Unit tests for configuration loading""",
        "category": "Backlog"
    },
    {
        "title": "Implement Logging and Error Handling",
        "body": """**Problem Statement:**
We need comprehensive logging and error handling to ensure system reliability and debuggability.

**Definition of Done:**
- Centralized logging service with configurable outputs
- Error handling decorators for functions
- Performance monitoring capabilities
- Log rotation and management
- Unit tests for logging functionality""",
        "category": "Backlog"
    },
    {
        "title": "Create Base Service Architecture",
        "body": """**Problem Statement:**
We need a standardized service architecture for all components to ensure consistency and maintainability.

**Definition of Done:**
- BaseService class with lifecycle management (start/stop)
- Health monitoring capabilities
- Memory management for long-running services
- Service registration and discovery
- Unit tests for base service functionality""",
        "category": "ToDo"
    },
    {
        "title": "Implement Historical Data Collection Service",
        "body": """**Problem Statement:**
We need to collect and store historical price data for BTC, SOL, and BONK for analysis.

**Definition of Done:**
- Service to fetch historical OHLCV data from exchanges
- Support for different timeframes (1h, 4h, 1d)
- Data storage and caching mechanism
- Automatic data updates
- Unit tests for data collection""",
        "category": "ToDo"
    },
    {
        "title": "Implement Technical Indicators Calculation",
        "body": """**Problem Statement:**
We need to calculate common technical indicators for market analysis.

**Definition of Done:**
- RSI calculation with configurable periods
- Moving averages (SMA, EMA)
- Volume indicators
- Bollinger Bands
- ATR and volatility metrics
- Unit tests for indicator calculations""",
        "category": "ToDo"
    },
    {
        "title": "Implement Market Structure Analysis Model",
        "body": """**Problem Statement:**
We need to analyze market structure and price action for BTC, SOL, and BONK.

**Definition of Done:**
- 4H chart analysis with RSI, MAs, and key levels
- Support and resistance detection
- Trend analysis
- Candlestick pattern recognition
- Unit tests for market structure analysis""",
        "category": "In Progress"
    },
    {
        "title": "Implement Open Interest vs Price Divergence Analysis",
        "body": """**Problem Statement:**
We need to analyze divergence between Open Interest and Price to identify hidden distribution/accumulation.

**Definition of Done:**
- OI vs Price comparison algorithm
- Divergence detection logic
- Signal generation for bullish/bearish divergence
- Visualization of OI vs Price
- Unit tests for OI vs Price divergence analysis""",
        "category": "In Progress"
    },
    {
        "title": "Implement Spot vs Perp CVD Analysis",
        "body": """**Problem Statement:**
We need to analyze Spot vs Perp CVD to detect whether spot or perp markets are leading price action.

**Definition of Done:**
- Spot vs Perp CVD comparison algorithm
- Lead/lag detection logic
- Signal generation for spot-led vs perp-led moves
- Visualization of Spot vs Perp CVD
- Unit tests for Spot vs Perp CVD analysis""",
        "category": "Backlog"
    },
    {
        "title": "Implement Delta Volume Analysis",
        "body": """**Problem Statement:**
We need to analyze Delta (Aggressor) Volume to identify trap zones and stop-hunt setups.

**Definition of Done:**
- Delta volume analysis algorithm
- Trap zone detection logic
- Signal generation for buy/sell imbalances
- Visualization of delta volume
- Unit tests for delta volume analysis""",
        "category": "Backlog"
    },
    {
        "title": "Implement Liquidation Analysis",
        "body": """**Problem Statement:**
We need to analyze liquidation heatmaps to identify magnet zones for price.

**Definition of Done:**
- Liquidation cluster analysis algorithm
- Magnet zone detection logic
- Signal generation for liquidation-driven moves
- Visualization of liquidation heatmap
- Unit tests for liquidation analysis""",
        "category": "Backlog"
    },
    {
        "title": "Implement Funding Rate Analysis",
        "body": """**Problem Statement:**
We need to analyze funding rates to detect crowd bias and potential squeeze triggers.

**Definition of Done:**
- Funding rate analysis algorithm
- Crowd bias detection logic
- Signal generation for potential squeezes
- Visualization of funding rates
- Unit tests for funding rate analysis""",
        "category": "Backlog"
    },
    {
        "title": "Implement Gamma Exposure Analysis",
        "body": """**Problem Statement:**
We need to analyze gamma exposure to identify volatility clusters and resistance bands.

**Definition of Done:**
- Gamma exposure analysis algorithm
- Volatility cluster detection logic
- Resistance/support identification from GEX
- Visualization of gamma exposure
- Unit tests for gamma exposure analysis""",
        "category": "Loose Ideas"
    },
    {
        "title": "Implement Macro Events Analysis",
        "body": """**Problem Statement:**
We need to analyze macro events to predict volatility spikes from economic events.

**Definition of Done:**
- Macro events analysis algorithm
- Volatility prediction logic
- Signal generation for upcoming events
- Visualization of macro events calendar
- Unit tests for macro events analysis""",
        "category": "Loose Ideas"
    },
    {
        "title": "Implement Cross-Asset Correlation Analysis",
        "body": """**Problem Statement:**
We need to analyze correlations between BTC, SOL, and BONK to identify leading indicators.

**Definition of Done:**
- Cross-asset correlation algorithm
- Lead/lag detection logic
- Signal generation for correlated moves
- Visualization of cross-asset correlations
- Unit tests for cross-asset correlation analysis""",
        "category": "In Review"
    },
    {
        "title": "Implement On-Chain Analysis",
        "body": """**Problem Statement:**
We need to analyze on-chain data to monitor large wallet movements for mid-term signals.

**Definition of Done:**
- On-chain data analysis algorithm
- Whale movement detection logic
- Signal generation for significant on-chain activity
- Visualization of on-chain flows
- Unit tests for on-chain analysis""",
        "category": "Loose Ideas"
    },
    {
        "title": "Implement Signal Integration Service",
        "body": """**Problem Statement:**
We need to integrate signals from all analysis models to generate comprehensive trading signals.

**Definition of Done:**
- Signal integration algorithm
- Weighting and scoring mechanism
- Confidence level calculation
- Signal filtering based on thresholds
- Unit tests for signal integration""",
        "category": "ToDo"
    },
    {
        "title": "Implement Trading Signal Generator",
        "body": """**Problem Statement:**
We need to generate actionable trading signals with entry, exit, and risk management parameters.

**Definition of Done:**
- Trading signal generation algorithm
- Entry/exit point calculation
- Stop loss and take profit calculation
- Risk/reward ratio calculation
- Unit tests for trading signal generation""",
        "category": "ToDo"
    },
    {
        "title": "Implement Signal Visualization",
        "body": """**Problem Statement:**
We need to visualize trading signals and analysis results for easy interpretation.

**Definition of Done:**
- Chart generation with signal overlays
- Dashboard layout with key metrics
- Interactive visualization components
- Export functionality for charts
- Unit tests for visualization components""",
        "category": "Done"
    },
    {
        "title": "Implement Notification Service",
        "body": """**Problem Statement:**
We need to notify users of trading signals and important events.

**Definition of Done:**
- Email notification system
- Webhook integration for external platforms
- Customizable notification settings
- Rate limiting for notifications
- Unit tests for notification service""",
        "category": "Done"
    }
]

def create_issue(title, body, repo="grich88/Trading"):
    """Create a GitHub issue using the gh CLI."""
    try:
        cmd = [
            "gh", "issue", "create",
            "--title", title,
            "--body", body,
            "--repo", repo
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        issue_url = result.stdout.strip()
        print(f"Created issue: {title} - {issue_url}")
        return issue_url
    except subprocess.CalledProcessError as e:
        print(f"Error creating issue: {e.stderr}")
        return None

def main():
    """Create all issues."""
    issue_map = {}
    
    for issue in issues:
        title = issue["title"]
        body = issue["body"]
        category = issue["category"]
        
        # Create the issue
        issue_url = create_issue(title, body)
        if issue_url:
            # Extract issue number from URL
            issue_number = issue_url.split("/")[-1]
            issue_map[issue_number] = category
            
            # Wait a bit to avoid rate limiting
            time.sleep(1)
    
    # Save the issue map to a file for later use
    with open("issue_map.json", "w") as f:
        json.dump(issue_map, f, indent=2)
    
    print(f"Created {len(issue_map)} issues and saved mapping to issue_map.json")

if __name__ == "__main__":
    main()
