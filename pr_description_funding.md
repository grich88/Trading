# Implement Funding Rate & Bias Tracking (Ticket #40)

## Description
This PR implements a comprehensive Funding Rate & Bias Tracking analysis model for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced analysis of funding rates, market bias detection, pattern recognition, and combined signal generation.

## Changes
- Created `src/models/funding_rate_analyzer.py` with:
  - `FundingRateAnalyzer` class for comprehensive funding rate analysis
  - Funding rate anomaly detection
  - Market bias tracking based on funding rates
  - Historical funding rate pattern recognition
  - Funding rate impact analysis
  - Combined signal generation
  - Standalone helper functions
- Created `src/models/funding_rate_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- Advanced funding rate analysis:
  - Statistical anomaly detection
  - Z-score calculation
  - Historical anomaly tracking
- Market bias tracking:
  - Bias direction determination (bullish/bearish/neutral)
  - Bias strength calculation
  - Net bias metrics
- Funding rate pattern recognition:
  - Correlation with price
  - Sign change detection
  - Peak and trough identification
  - Convergence/divergence pattern detection
  - Common pattern identification (sustained funding, reversals, spikes)
- Funding rate impact analysis:
  - Price reaction to different funding conditions
  - Mean-reverting vs. trend-following pattern detection
  - Predictive correlation calculation
- Combined signal generation:
  - Multi-factor signal scoring
  - Weighted component integration
  - Confidence calculation

## Testing
- Added unit tests for all major functionality
- Tests for funding rate anomaly detection
- Tests for market bias tracking
- Tests for pattern recognition
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
Closes #40
