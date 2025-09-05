"""
Test runner module.

This module provides a test runner for the application.
"""

import os
import sys
import unittest
import argparse


def discover_tests(start_dir='.', pattern='*_test.py'):
    """
    Discover and load tests from the specified directory.
    
    Args:
        start_dir: Directory to start discovery
        pattern: Pattern for test files
        
    Returns:
        Test suite
    """
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern=pattern)
    return suite


def run_tests(start_dir='.', pattern='*_test.py', verbosity=1):
    """
    Run tests from the specified directory.
    
    Args:
        start_dir: Directory to start discovery
        pattern: Pattern for test files
        verbosity: Verbosity level
        
    Returns:
        Test result
    """
    suite = discover_tests(start_dir, pattern)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Runner")
    
    parser.add_argument(
        "--dir",
        type=str,
        default="src",
        help="Directory to start test discovery"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_test.py",
        help="Pattern for test files"
    )
    
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="Verbosity level"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the test runner."""
    # Parse command line arguments
    args = parse_args()
    
    # Run tests
    result = run_tests(args.dir, args.pattern, args.verbosity)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
