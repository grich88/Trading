# Implement Spot vs Perp CVD Analysis (Ticket #37)

## Description
This PR implements a comprehensive Spot vs Perpetual CVD (Cumulative Volume Delta) Analysis model for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced analysis of Spot vs Perp CVD patterns, including divergence detection, trend analysis, and combined signal generation.

## Changes
- Created `src/models/cvd_analyzer.py` with:
  - `CVDAnalyzer` class for comprehensive Spot vs Perp CVD analysis
  - CVD calculation and normalization
  - Spot vs Perp CVD divergence detection
  - CVD trend analysis
  - CVD momentum analysis
  - Combined signal generation
  - Standalone helper functions
- Created `src/models/cvd_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- Advanced CVD calculation and normalization:
  - Raw CVD calculation from buy/sell volume
  - Multiple normalization methods (z-score, min-max, percent-change)
- Spot vs Perp CVD divergence detection:
  - Bullish divergence (spot up, perp down)
  - Bearish divergence (spot down, perp up)
  - Extreme divergence detection
  - Correlation analysis
- CVD trend analysis:
  - Short-term, medium-term, and long-term trends
  - Trend consistency detection
  - Trend acceleration detection
- CVD momentum analysis:
  - Short-term and medium-term momentum
  - Relative momentum between spot and perp
  - Spot/Perp ratio analysis
- Combined signal generation:
  - Multi-factor signal scoring
  - Weighted component integration
  - Confidence calculation
- Comprehensive test coverage

## Testing
- Added unit tests for all major functionality
- Tests for CVD calculation and normalization
- Tests for divergence detection
- Tests for trend analysis
- Tests for momentum analysis
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
Closes #37
