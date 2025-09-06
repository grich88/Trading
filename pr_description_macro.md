# Implement CPI / Macro Events Tracking (Ticket #42)

## Description
This PR implements a comprehensive CPI / Macro Events Tracking model for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced analysis of macroeconomic events and their impact on crypto markets, including CPI data, FOMC events, economic calendar analysis, and combined signal generation.

## Changes
- Created `src/models/macro_events_analyzer.py` with:
  - `MacroEventsAnalyzer` class for comprehensive macro event analysis
  - CPI impact analysis with surprise correlation
  - FOMC event impact analysis with volatility tracking
  - Economic calendar analysis with event clustering
  - Macro sentiment calculation
  - Combined signal generation with risk level assessment
  - Standalone helper functions
- Created `src/models/macro_events_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- CPI Impact Analysis:
  - CPI surprise calculation and correlation with price movements
  - Analysis of positive vs negative surprise impacts
  - Current inflationary regime detection
  - Historical impact pattern analysis
- FOMC Event Analysis:
  - Pre/post event volatility analysis
  - Decision-based impact tracking
  - Statement sentiment analysis
  - Upcoming event monitoring
- Economic Calendar Analysis:
  - High-impact event identification
  - Event clustering measurement
  - Weekly event density calculation
  - Risk categorization
- Macro Sentiment:
  - Multi-factor sentiment scoring
  - CPI, FOMC, and calendar contributions
  - Confidence calculation
- Combined Signal Generation:
  - Weighted component integration
  - Signal dampening during high uncertainty
  - Risk level assessment

## Testing
- Added unit tests for all major functionality
- Tests for CPI impact analysis
- Tests for FOMC impact analysis
- Tests for economic calendar analysis
- Tests for macro sentiment calculation
- Tests for signal generation
- Tests for error handling and edge cases

## Compliance with Master Guide
This implementation adheres to the following principles from the Master Guide:
- Modular design
- Separation of concerns
- Comprehensive test coverage
- Performance monitoring
- Error handling

## Related Issues
Closes #42
