# Set Up Performance Monitoring (Ticket #30)

## Description
This PR implements a comprehensive performance monitoring system for the trading algorithm project, following the principles outlined in the Project Master Guide v1.0.0.

## Changes
- Enhanced `src/utils/performance.py` with:
  - Async performance monitor decorator for async functions
  - Context managers for timing, memory usage, and profiling
  - System resource profiler for CPU and memory monitoring
  - Helper functions for profiling both sync and async functions
  - Enhanced adaptive batch processing with progress callbacks and error handling
  - Async version of adaptive batch processing
- Created `src/utils/performance_test.py` with comprehensive tests for all features
- Updated `src/utils/__init__.py` to export all performance monitoring features

## Features
- Function timing and memory usage tracking
- CPU and memory profiling
- Adaptive batch processing for memory management
- System resource monitoring
- Support for both synchronous and asynchronous code
- Context managers for easy performance monitoring
- Comprehensive test coverage

## Testing
- Added unit tests for performance monitor decorators
- Added unit tests for memory monitor
- Added unit tests for adaptive batch processing
- Added unit tests for system profiler
- All tests are passing

## Compliance with Master Guide
This implementation adheres to the following principles from the Master Guide:
- Centralized performance monitoring
- Memory management for large datasets
- Profiling tools for performance optimization
- Comprehensive test coverage
- Support for both synchronous and asynchronous code

## Related Issues
Closes #30
