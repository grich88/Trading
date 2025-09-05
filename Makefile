.PHONY: help setup clean lint format test coverage docs

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup      - Set up development environment"
	@echo "  make clean      - Clean up build artifacts"
	@echo "  make lint       - Run linters (flake8, mypy)"
	@echo "  make format     - Format code (black, isort)"
	@echo "  make test       - Run tests"
	@echo "  make coverage   - Run tests with coverage report"
	@echo "  make docs       - Build documentation"

# Set up development environment
setup:
	@echo "Setting up development environment..."
	pip install -e ".[dev]"
	pre-commit install

# Clean up build artifacts
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Run linters
lint:
	@echo "Running linters..."
	flake8 src
	mypy src

# Format code
format:
	@echo "Formatting code..."
	isort src
	black src

# Run tests
test:
	@echo "Running tests..."
	pytest

# Run tests with coverage report
coverage:
	@echo "Running tests with coverage report..."
	pytest --cov=src --cov-report=term-missing --cov-report=html

# Build documentation
docs:
	@echo "Building documentation..."
	mkdocs build
