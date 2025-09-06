# Development Guide

This document provides guidelines and instructions for setting up and working with the Trading Algorithm System development environment.

## Python Version

This project requires **Python 3.9+**. We recommend using Python 3.10 for the best balance of features and stability.

## Setting Up Your Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/grich88/Trading.git
cd Trading
```

### 2. Create a Virtual Environment

#### Windows

```powershell
# Using venv
python -m venv venv
.\venv\Scripts\activate

# Using conda
conda create -n trading-algo python=3.10
conda activate trading-algo
```

#### macOS/Linux

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# Using conda
conda create -n trading-algo python=3.10
conda activate trading-algo
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies only
pip install -r requirements-dev.txt
```

## Environment Configuration

The project uses environment variables for configuration. Create a `.env` file in the project root by copying the example:

```bash
cp .env.example .env
```

Then edit the `.env` file to set the appropriate values for your environment.

## Development Tools

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting with a line length of 88 characters:

```bash
# Format a single file
black src/file.py

# Format the entire project
black .
```

### Import Sorting

We use [isort](https://pycqa.github.io/isort/) to sort imports:

```bash
# Sort imports in a single file
isort src/file.py

# Sort imports in the entire project
isort .
```

### Linting

We use [Flake8](https://flake8.pycqa.org/) for linting:

```bash
# Lint a single file
flake8 src/file.py

# Lint the entire project
flake8
```

### Type Checking

We use [mypy](https://mypy.readthedocs.io/) for static type checking:

```bash
# Type check a single file
mypy src/file.py

# Type check the entire project
mypy .
```

### Pre-commit Hooks

We recommend setting up pre-commit hooks to automatically run formatting, linting, and type checking before each commit:

```bash
pip install pre-commit
pre-commit install
```

## Running Tests

We use [pytest](https://docs.pytest.org/) for testing:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src

# Run a specific test file
pytest src/tests/test_file.py
```

## Git Workflow

We follow the Git Flow branching model:

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, commit them, and push to the remote repository:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   git push -u origin feature/your-feature-name
   ```

3. Create a pull request to merge your feature branch into `develop`.

4. After your PR is reviewed and approved, it will be merged into `develop`.

## Continuous Integration

We use GitHub Actions for continuous integration. The CI pipeline runs automatically on pull requests to the `develop` and `main` branches.

## Documentation

We use Markdown for documentation. All documentation files should be placed in the `docs` directory.

## Getting Help

If you have any questions or need help with the development environment, please reach out to the team.
