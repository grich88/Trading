# Implement Correlations Analysis (BTC vs SOL vs BONK) (Ticket #43)

## Description
This PR implements a comprehensive Correlations Analysis model for BTC vs SOL vs BONK, following the principles outlined in the Project Master Guide v1.0.0. The implementation provides advanced correlation analysis between crypto assets, including rolling correlations, correlation breakdowns, divergence detection, lead-lag relationships, beta calculations, and combined signal generation.

## Changes
- Created `src/models/correlations_analyzer.py` with:
  - `CorrelationsAnalyzer` class for comprehensive multi-asset correlation analysis
  - Rolling correlation calculation with multiple windows
  - Correlation breakdown detection
  - Price divergence analysis
  - Lead-lag relationship analysis
  - Beta calculations and interpretations
  - Market regime determination
  - Combined signal generation
  - Standalone helper functions
- Created `src/models/correlations_analyzer_test.py` with comprehensive tests
- Updated `src/models/__init__.py` to export the new model and functions

## Features
- Rolling Correlation Analysis:
  - Short and long-term correlation windows
  - Correlation breakdown detection
  - Decorrelation identification
  - Regime change detection
- Divergence Analysis:
  - Multi-period price divergence calculation
  - Significant divergence identification
  - Leader/laggard determination
- Lead-Lag Analysis:
  - Cross-correlation at multiple lags
  - Optimal lag detection
  - Leader identification
- Beta Relationships:
  - Beta calculation between asset pairs
  - R-squared for relationship strength
  - Beta interpretation (high_beta, amplified, correlated, etc.)
- Market Regime Determination:
  - Decorrelated regime
  - High beta risk-on regime
  - Low beta risk-off regime
  - Normal correlation regime
- Combined Signal Generation:
  - Multi-factor signal scoring
  - Risk regime adjustment
  - Confidence calculation

## Testing
- Added unit tests for all major functionality
- Tests for rolling correlation calculation
- Tests for correlation breakdown detection
- Tests for divergence analysis
- Tests for lead-lag relationships
- Tests for beta calculations
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
Closes #43
