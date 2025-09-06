# Implement Liquidation Map Analysis (Ticket #39)

## Description
This PR implements a comprehensive Liquidation Map Analysis model for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced analysis of liquidation data, including liquidation clustering, cascade detection, impact analysis, and combined signal generation.

## Changes
- Created `src/models/liquidation_map_analyzer.py` with:
  - `LiquidationMapAnalyzer` class for comprehensive liquidation analysis
  - Liquidation data normalization
  - Liquidation cluster detection at specific price levels
  - Liquidation cascade detection
  - Liquidation impact analysis on price and volume
  - Combined signal generation
  - Standalone helper functions
- Created `src/models/liquidation_map_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- Advanced liquidation data processing:
  - Multiple normalization methods (z-score, percent-of-volume, price-relative)
  - Support for various data formats
- Liquidation cluster detection:
  - Identification of liquidation clusters at specific price levels
  - Support/resistance level identification from liquidation data
  - Imbalance detection between liquidations above and below current price
- Liquidation cascade detection:
  - Momentum and acceleration analysis
  - Correlation with price movements
  - Early warning signals for potential liquidation cascades
- Liquidation impact analysis:
  - Analysis of price reactions to high-impact liquidation events
  - Pattern detection (mean reversion vs. trend continuation)
  - Lagged correlation between liquidations and price changes
- Combined signal generation:
  - Multi-factor signal scoring
  - Weighted component integration
  - Confidence calculation

## Testing
- Added unit tests for all major functionality
- Tests for liquidation normalization
- Tests for cluster detection
- Tests for cascade detection
- Tests for impact analysis
- Tests for signal generation
- Tests for error handling

## Compliance with Master Guide
This implementation adheres to the following principles from the Master Guide:
- Modular design
- Separation of concerns
- Comprehensive test coverage
- Performance monitoring
- Error handling

## Related Issues
Closes #39
