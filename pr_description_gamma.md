# Implement Gamma Exposure / Option Flow Analysis (Ticket #41)

## Description
This PR implements a comprehensive Gamma Exposure / Option Flow Analysis model for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced analysis of gamma exposure, dealer positioning, option flow patterns, and combined signal generation.

## Changes
- Created `src/models/gamma_exposure_analyzer.py` with:
  - `GammaExposureAnalyzer` class for comprehensive gamma and option flow analysis
  - Gamma exposure calculation from options data
  - Dealer positioning analysis
  - Option flow analysis
  - Gamma dynamics analysis over time
  - Gamma flip detection
  - Combined signal generation
  - Standalone helper functions
- Created `src/models/gamma_exposure_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- Advanced gamma exposure calculation:
  - Dealer gamma calculation (negative for calls, positive for puts)
  - Strike-based gamma aggregation
  - Cumulative gamma exposure
  - Gamma flip point detection with interpolation
  - Spot gamma calculation
- Option flow analysis:
  - Call/put volume and premium ratios
  - Buy/sell flow detection
  - Large trade identification
  - Unusual activity detection
  - Flow sentiment determination
- Gamma dynamics analysis:
  - Gamma momentum tracking
  - Flip point stability measurement
  - Gamma effectiveness as support/resistance
  - Gamma regime identification
  - Gamma squeeze detection
- Combined signal generation:
  - Multi-factor signal scoring
  - Weighted component integration
  - Confidence calculation

## Testing
- Added unit tests for all major functionality
- Tests for gamma exposure calculation
- Tests for option flow analysis
- Tests for gamma dynamics analysis
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
Closes #41
