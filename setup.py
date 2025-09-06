from setuptools import setup, find_packages

setup(
    name="trading-algorithm-system",
    version="0.1.0",
    description="A comprehensive trading algorithm system for cryptocurrency market analysis",
    author="Trading Team",
    author_email="team@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "python-dotenv>=1.0.0",
        "psutil>=5.9.5",
        "pydantic>=2.0.3",
        
        # API and data fetching
        "ccxt>=4.0.0",
        "requests>=2.31.0",
        "aiohttp>=3.8.5",
        
        # Web interface
        "streamlit>=1.24.0",
        "plotly>=5.15.0",
        "matplotlib>=3.7.2",
        
        # Image processing
        "Pillow>=9.5.0",
        "pytesseract>=0.3.10",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        
        # Data processing and analysis
        "scipy>=1.10.1",
        "scikit-learn>=1.2.2",
        "statsmodels>=0.14.0",
        
        # Performance optimization
        "numba>=0.57.1",
        "joblib>=1.3.1",
        
        # Utilities
        "tqdm>=4.65.0",
        "pytz>=2023.3",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pylint>=2.17.4",
            "bandit>=1.7.5",
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.10.0",
            "pytest-xdist>=3.3.1",
            "coverage>=7.2.7",
            "mkdocs>=1.4.3",
            "mkdocs-material>=9.1.15",
            "pre-commit>=3.3.2",
            "ipython>=8.14.0",
            "ipdb>=0.13.13",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
