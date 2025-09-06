# Implement Delta (Aggressor) Volume Imbalance Analysis (Ticket #38)

## Description
This PR implements a comprehensive Delta (Aggressor) Volume Imbalance Analysis model for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced analysis of Delta Volume patterns, including imbalance detection, trend analysis, divergence detection, VWAP analysis, and combined signal generation.

## Changes
- Created `src/models/delta_volume_analyzer.py` with:
  - `DeltaVolumeAnalyzer` class for comprehensive Delta Volume analysis
  - Delta volume calculation and cumulative delta tracking
  - Volume imbalance detection
  - Delta trend analysis
  - Price-delta divergence detection
  - VWAP delta analysis
  - Combined signal generation
  - Standalone helper functions
- Created `src/models/delta_volume_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- Advanced Delta Volume calculation and analysis:
  - Raw delta volume calculation from market order volumes
  - Cumulative delta tracking
  - Multiple analysis methods
- Volume imbalance detection:
  - Buy/sell market order imbalance calculation
  - Imbalance ratio and strength metrics
  - Z-score calculation for statistical significance
- Delta trend analysis:
  - Short-term, medium-term, and long-term trends
  - Trend consistency detection
  - Trend acceleration detection
- Price-delta divergence detection:
  - Bullish divergence (price down, delta up)
  - Bearish divergence (price up, delta down)
  - Extreme divergence detection
  - Correlation analysis
- VWAP delta analysis:
  - Buy VWAP vs Sell VWAP calculation
  - VWAP delta direction and strength
- Combined signal generation:
  - Multi-factor signal scoring
  - Weighted component integration
  - Confidence calculation

## Testing
- Added unit tests for all major functionality
- Tests for delta volume calculation
- Tests for imbalance detection
- Tests for trend analysis
- Tests for divergence detection
- Tests for VWAP analysis
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
Closes #38
