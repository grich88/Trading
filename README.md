# Trading Algorithm System

A comprehensive trading algorithm system for cryptocurrency market analysis, backtesting, and signal generation.

## Overview

This system provides an integrated solution for analyzing cryptocurrency markets, generating trading signals, and backtesting strategies. It combines technical indicators, liquidation data, and cross-asset relationships to produce high-quality trading signals.

## Features

- **Enhanced RSI + Volume Model**: Core predictive model combining RSI trends, volume analysis, and price action
- **Liquidation Data Integration**: Incorporates liquidation clusters from Coinglass or image extraction
- **Cross-Asset Analysis**: Analyzes relationships between different assets to improve signal quality
- **Adaptive Timeframe Selection**: Dynamically selects optimal timeframes based on market conditions
- **Signal Calibration**: Calibrates signals based on historical performance
- **Comprehensive Backtesting**: Robust backtesting framework with detailed performance metrics
- **Memory Management**: Optimized for long-running processes with adaptive batch processing
- **Web Interface**: Interactive dashboard for visualizing signals and backtesting results

## Project Structure

```
├── src/                    # Source code
│   ├── api/                # API endpoints and interfaces
│   ├── config/             # Configuration management
│   ├── core/               # Core application logic
│   ├── models/             # Trading models and algorithms
│   ├── services/           # Shared services
│   ├── tests/              # Test files (co-located with source)
│   └── utils/              # Utility functions and helpers
├── docs/                   # Documentation
│   ├── guides/             # User and developer guides
│   ├── implementation_docs/ # Implementation details
│   ├── diagrams/           # System architecture diagrams
│   ├── api/                # API documentation
│   └── scripts/            # Utility scripts
├── data/                   # Data storage (not in repository)
│   ├── historical/         # Historical price data
│   ├── weights/            # Model weights
│   └── backtest_results/   # Backtest results
├── .env.example            # Example environment variables
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-algorithm-system.git
   cd trading-algorithm-system
   ```

2. Create and activate a virtual environment:
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Copy the example file and fill in your values
   cp .env.example .env
   ```

## Usage

### Running the Web Interface

```bash
python src/app.py
```

This will start the Streamlit web interface, accessible at http://localhost:8501.

### Running Backtests

```bash
python src/run_backtest.py --asset BTC --start_date 2023-01-01 --end_date 2023-12-31
```

### Generating Signals

```bash
python src/generate_signals.py --assets BTC SOL BONK
```

## Configuration

The system is configured through environment variables. See `.env.example` for all available options.

Key configuration parameters:

- `APP_MODE`: Set to `development`, `testing`, or `production`
- `LOG_LEVEL`: Set logging verbosity
- `COINGLASS_API_KEY`: API key for Coinglass integration
- `DEFAULT_TIMEFRAME`: Default timeframe for data collection (e.g., `4h`)
- `DEFAULT_WINDOW_SIZE`: Default window size for analysis

## Development

### Running Tests

```bash
pytest
```

### Code Style

This project follows PEP 8 guidelines. To check your code:

```bash
flake8 src
```

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) for exchange API integration
- [Streamlit](https://streamlit.io/) for the web interface
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [NumPy](https://numpy.org/) for numerical computations