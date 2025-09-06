# Crypto Trading Algorithm System

A comprehensive cryptocurrency trading algorithm system that integrates 10 critical signal categories for powerful market predictions and automated trading decisions.

## 🚀 Features

### Signal Categories
1. **RSI and Volume Analysis** - Advanced RSI calculations with volume pattern detection
2. **Open Interest vs Price Divergence** - Identifies divergences between OI and price movements
3. **Spot vs Perp CVD Analysis** - Analyzes cumulative volume delta across spot and perpetual markets
4. **Delta Volume Imbalance** - Detects aggressor volume imbalances
5. **Liquidation Map Analysis** - Identifies liquidation clusters and cascade risks
6. **Funding Rate & Bias Tracking** - Monitors funding rate anomalies and market bias
7. **Gamma Exposure / Option Flow** - Tracks dealer positioning and gamma dynamics
8. **CPI / Macro Events Tracking** - Analyzes economic events impact on crypto markets
9. **Correlations Analysis** - Monitors correlations between BTC, SOL, and BONK
10. **On-Chain Flows Analysis** - Tracks exchange flows and whale movements

### Core Features
- **Unified Dashboard** - Beautiful Streamlit web interface with real-time monitoring
- **Signal Integration** - Weighted signal scoring across all categories
- **Alert System** - Multi-level alerts with convergence/divergence detection
- **Trading Recommendations** - Risk-adjusted recommendations with position sizing
- **Performance Tracking** - System and recommendation performance metrics
- **Multi-Exchange Support** - Ready for integration with major exchanges via CCXT

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/grich88/Trading.git
   cd Trading
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your exchange API keys and other configuration.

## 🚀 Quick Start

### Running the Dashboard

```bash
streamlit run src/ui/dashboard_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest src/models/rsi_volume_analyzer_test.py
```

### Development Tools

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 📁 Project Structure

```
Trading/
├── src/
│   ├── api/              # API endpoints (ready for REST API implementation)
│   ├── config/           # Configuration management
│   ├── core/             # Core application logic
│   ├── models/           # Signal analyzer models
│   ├── services/         # Business logic services
│   ├── tests/            # Test configuration
│   ├── ui/               # Streamlit dashboard
│   └── utils/            # Utilities (logging, error handling, performance)
├── docs/                 # Documentation
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## 💻 Usage Examples

### Basic Signal Analysis

```python
from src.models import RSIVolumeAnalyzer
from src.services import DataService

# Initialize services
data_service = DataService()
analyzer = RSIVolumeAnalyzer()

# Get market data
data = data_service.get_latest_data('BTC/USDT', timeframe='1h', limit=100)

# Analyze signals
result = analyzer.analyze(data)
print(f"Signal: {result['combined_signal']}")
```

### Using the Unified Dashboard Service

```python
from src.services import UnifiedDashboardService

# Initialize dashboard
dashboard = UnifiedDashboardService()

# Get comprehensive analysis
analysis = dashboard.get_dashboard_data(
    symbols=['BTC/USDT', 'ETH/USDT'],
    timeframe='1h'
)

# Access components
signals = analysis['active_signals']
alerts = analysis['alerts']
recommendations = analysis['recommendations']
```

### Async Monitoring

```python
import asyncio
from src.services import UnifiedDashboardService

async def monitor_markets():
    dashboard = UnifiedDashboardService()
    
    async def handle_update(data):
        print(f"New alerts: {len(data['alerts'])}")
        for alert in data['alerts']:
            print(f"- {alert.level}: {alert.message}")
    
    await dashboard.start_monitoring(
        symbols=['BTC/USDT'],
        callback=handle_update
    )

# Run monitoring
asyncio.run(monitor_markets())
```

## 🎨 Dashboard Features

### Main Dashboard Views
- **Market Overview** - Real-time prices, trends, and volatility
- **Active Signals** - All signals organized by category
- **Alerts Panel** - Priority-sorted alerts with action requirements
- **Trading Recommendations** - Clear buy/sell/hold recommendations
- **Signal Correlations** - Visual correlation matrix
- **Performance Metrics** - System accuracy and win rates

### Interactive Controls
- Symbol selection (BTC, ETH, SOL, etc.)
- Timeframe adjustment (1m to 1d)
- Auto-refresh toggle
- Export functionality (JSON, CSV, HTML)

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Exchange API Configuration
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_SECRET=your_secret_here
EXCHANGE_NAME=binance

# Database Configuration
DATABASE_URL=sqlite:///trading.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

# Dashboard Configuration
DASHBOARD_REFRESH_INTERVAL=60
DASHBOARD_DEFAULT_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
```

### Signal Weights

Adjust signal category weights in `src/services/unified_dashboard.py`:

```python
category_weights = {
    SignalCategory.TECHNICAL: 0.20,
    SignalCategory.VOLUME: 0.20,
    SignalCategory.DERIVATIVES: 0.25,
    SignalCategory.ONCHAIN: 0.20,
    SignalCategory.MACRO: 0.10,
    SignalCategory.SENTIMENT: 0.05
}
```

## 🧪 Testing

The project includes comprehensive test coverage:

- **Unit Tests** - Each analyzer has 300-500 lines of tests
- **Integration Tests** - Service-level testing
- **Mocking** - External dependencies are mocked
- **Edge Cases** - Comprehensive edge case handling

Run specific test suites:

```bash
# Test all analyzers
pytest src/models/

# Test services
pytest src/services/

# Test utilities
pytest src/utils/
```

## 📊 Performance Considerations

- **Batch Processing** - Adaptive batch processing for large datasets
- **Memory Management** - Built-in memory monitoring and limits
- **Async Operations** - Async support for concurrent operations
- **Caching** - Results caching for expensive calculations
- **Rate Limiting** - Built-in rate limiting for API calls

## 🚧 Deployment

### Docker (Coming Soon)

```dockerfile
# Dockerfile will be added in next update
FROM python:3.9-slim
...
```

### Production Checklist

- [ ] Set secure API keys in environment
- [ ] Configure proper database (PostgreSQL recommended)
- [ ] Set up logging aggregation
- [ ] Configure monitoring/alerting
- [ ] Set up SSL/TLS for dashboard
- [ ] Configure backup strategy
- [ ] Set up CI/CD pipeline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8
- Add type hints
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages

## 📚 Documentation

- [Development Guide](docs/development_guide.md)
- [API Reference](docs/api_reference.md) (Coming Soon)
- [Signal Documentation](docs/signals.md) (Coming Soon)
- [Deployment Guide](docs/deployment.md) (Coming Soon)

## 🐛 Troubleshooting

### Common Issues

1. **ImportError**: Make sure you're in the project root and have activated the virtual environment
2. **Missing dependencies**: Run `pip install -r requirements.txt` again
3. **API errors**: Check your `.env` file has valid API credentials
4. **Dashboard not loading**: Ensure port 8501 is not in use

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Roadmap

- [ ] Add more exchanges support
- [ ] Implement backtesting framework
- [ ] Add machine learning models
- [ ] Create mobile app
- [ ] Add more cryptocurrencies
- [ ] Implement automated trading execution
- [ ] Add portfolio management features

## 📝 License

This project is proprietary software. All rights reserved.

## 👥 Team

- **Lead Developer** - Implementation of core system and signal analyzers
- **Contributors** - See GitHub contributors page

## 🙏 Acknowledgments

- CCXT for exchange connectivity
- Streamlit for the dashboard framework
- The crypto trading community for insights and strategies

---

For questions or support, please open an issue on GitHub.