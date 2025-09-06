# Set Up Signal Integration (Ticket #34)

## Description
This PR implements a comprehensive signal integration service for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0. The service builds on the existing market analysis service and provides advanced features for integrating signals from multiple sources, applying custom weighting strategies, and generating actionable trading signals.

## Changes
- Created `src/services/signal_integration.py` with:
  - `SignalIntegrationService` class that extends `LongRunningService`
  - Multi-source signal integration
  - Custom weighting strategies
  - Signal filtering and validation
  - Backtesting capabilities
  - Signal history tracking
  - Automated signal generation
- Created `src/services/signal_integration_test.py` with comprehensive tests
- Updated `src/services/__init__.py` to export the new service

## Features
- Integration of signals from multiple sources:
  - Market analysis signals
  - Technical signals
  - Fundamental signals
  - Sentiment signals
  - On-chain signals
- Multiple signal processing strategies:
  - Default processor
  - Weighted average processor
  - Majority vote processor
  - Threshold filter processor
- Signal history tracking and persistence
- Customizable signal weights and thresholds
- Backtesting capabilities
- Custom source and processor registration
- Comprehensive test coverage

## Testing
- Added unit tests for all major functionality
- Mocked dependencies for reliable testing
- Tests cover both success and error scenarios
- Tests for all signal processors

## Compliance with Master Guide
This implementation adheres to the following principles from the Master Guide:
- Service-oriented architecture
- Centralized signal integration
- Multi-source signal processing
- Comprehensive test coverage
- Integration with existing services

## Related Issues
Closes #34
