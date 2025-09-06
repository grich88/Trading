# Implement RSI and Volume Analysis (Ticket #35)

## Description
This PR implements a comprehensive RSI and Volume Analysis model for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced analysis of RSI (Relative Strength Index) and volume patterns, including divergence detection, volume confirmation, and combined signal generation.

## Changes
- Created `src/models/rsi_volume_analyzer.py` with:
  - `RSIVolumeAnalyzer` class for comprehensive RSI and volume analysis
  - RSI calculation and analysis
  - Volume pattern detection
  - RSI-volume divergence detection
  - Combined signal generation
  - Standalone helper functions
- Created `src/models/rsi_volume_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- Advanced RSI calculation and analysis:
  - Overbought/oversold detection
  - Zone analysis
  - Momentum calculation
- Volume pattern analysis:
  - Volume moving average calculation
  - Volume anomaly detection
  - Volume confirmation of price movements
  - Climax volume detection
- RSI divergence detection:
  - Bearish divergence (price makes higher high, RSI makes lower high)
  - Bullish divergence (price makes lower low, RSI makes higher low)
  - Divergence strength calculation
- Combined signal generation:
  - Multi-factor signal scoring
  - Weighted component integration
  - Confidence calculation
- Comprehensive test coverage

## Testing
- Added unit tests for all major functionality
- Tests for RSI calculation
- Tests for volume analysis
- Tests for divergence detection
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
Closes #35
