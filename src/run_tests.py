#!/usr/bin/env python3
"""
Test Runner

This script runs all tests for the Trading Algorithm System.
"""

import argparse
import os
import sys
from typing import List, Optional

import pytest


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run tests for the Trading Algorithm System")
    
    parser.add_argument(
        "--path",
        help="Path to test file or directory",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report",
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML coverage report",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel",
    )
    
    return parser.parse_args()


def run_tests(
    path: Optional[str] = None,
    verbose: bool = False,
    coverage: bool = False,
    html: bool = False,
    xml: bool = False,
    parallel: bool = False,
) -> int:
    """
    Run tests.

    Args:
        path: Path to test file or directory.
        verbose: Whether to use verbose output.
        coverage: Whether to generate a coverage report.
        html: Whether to generate an HTML coverage report.
        xml: Whether to generate an XML coverage report.
        parallel: Whether to run tests in parallel.

    Returns:
        The exit code (0 for success, non-zero for failure).
    """
    # Set up pytest arguments
    args: List[str] = []
    
    # Add path if specified
    if path:
        args.append(path)
    else:
        args.append("src")
    
    # Add verbosity
    if verbose:
        args.append("-v")
    
    # Add coverage
    if coverage or html or xml:
        args.append("--cov=src")
        
        if html:
            args.append("--cov-report=html")
        
        if xml:
            args.append("--cov-report=xml")
        
        if coverage:
            args.append("--cov-report=term-missing")
    
    # Add parallel
    if parallel:
        args.append("-xvs")
    
    # Run pytest
    return pytest.main(args)


def main() -> None:
    """
    Main entry point for the test runner.
    """
    args = parse_args()
    
    # Run tests
    exit_code = run_tests(
        path=args.path,
        verbose=args.verbose,
        coverage=args.coverage,
        html=args.html,
        xml=args.xml,
        parallel=args.parallel,
    )
    
    # Exit with the test result
    sys.exit(exit_code)


if __name__ == "__main__":
    main()