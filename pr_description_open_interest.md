# Implement Open Interest vs Price Divergence Analysis (Ticket #36)

## Description
This PR implements a comprehensive Open Interest vs Price Divergence Analysis model for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced analysis of Open Interest (OI) and price movements, including divergence detection, trend analysis, and combined signal generation.

## Changes
- Created `src/models/open_interest_analyzer.py` with:
  - `OpenInterestAnalyzer` class for comprehensive OI and price analysis
  - OI metrics calculation
  - OI vs price divergence detection
  - OI trend analysis
  - Significant change detection
  - Combined signal generation
  - Standalone helper functions
- Created `src/models/open_interest_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- Advanced OI metrics calculation:
  - OI change and percentage change
  - OI rate of change (ROC)
  - OI to price ratio
- OI vs price divergence detection:
  - Bullish divergence (price down, OI up)
  - Bearish divergence (price up, OI down)
  - Extreme divergence detection
  - Local extrema divergence detection
- OI trend analysis:
  - Short-term, medium-term, and long-term trends
  - Trend consistency detection
  - Trend acceleration detection
- Significant change analysis:
  - OI increase with price increase (bullish continuation)
  - OI increase with price decrease (potential reversal)
  - OI decrease with price increase (weak rally)
  - OI decrease with price decrease (bearish continuation)
- Combined signal generation:
  - Multi-factor signal scoring
  - Weighted component integration
  - Confidence calculation
- Comprehensive test coverage

## Testing
- Added unit tests for all major functionality
- Tests for OI metrics calculation
- Tests for divergence detection
- Tests for trend analysis
- Tests for significant change detection
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
Closes #36
