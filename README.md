# 🚀 Advanced Financial Market Analyzer

A sophisticated real-time financial news analysis and trading signal generation system that monitors multiple news sources, performs advanced sentiment analysis, and generates actionable trading signals for major financial assets.

## ✨ Features

### 📰 Multi-Source News Aggregation
- **NewsAPI Integration**: Comprehensive financial news coverage
- **Finnhub News**: Professional-grade financial data
- **Truth Social Integration**: Specialized analysis of influential social media posts
- **Real-time Updates**: Continuous monitoring with configurable intervals

### 🧠 Advanced Sentiment Analysis
- **Multi-Model Approach**: Combines VADER, TextBlob, and transformer models
- **Financial Context**: Specialized FinBERT model for financial sentiment
- **Composite Scoring**: Weighted sentiment analysis for accurate market predictions
- **Confidence Metrics**: Reliability scoring for each analysis

### 📊 Intelligent Signal Generation
- **Multi-Asset Support**: Gold (XAU/USD), Bitcoin (BTC/USD), S&P 500 (SPX), USD Index (DXY)
- **Contextual Analysis**: Asset-specific logic for different market conditions
- **Signal Strength Classification**: WEAK, MODERATE, STRONG, CRITICAL
- **Market Impact Assessment**: Keyword-weighted impact scoring

### 🎯 Specialized Features
- **Trump Post Analysis**: Dedicated analyzer for high-impact social media posts
- **Audio Alerts**: Sound notifications for critical market events
- **Terminal Dashboard**: Real-time colored terminal interface
- **Streamlit Web Dashboard**: Interactive web-based monitoring
- **Historical Tracking**: Signal and article history management

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- API keys for news sources
- Optional: GPU for advanced NLP models

### Required Dependencies
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Required API Keys
NEWS_API_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alphavantage_key_here
FINNHUB_KEY=your_finnhub_key_here

# Optional: Truth Social Credentials
TRUTHSOCIAL_USERNAME=your_username
TRUTHSOCIAL_PASSWORD=your_password
```

### API Key Setup

1. **NewsAPI**: Get free API key at [newsapi.org](https://newsapi.org)
2. **Alpha Vantage**: Register at [alphavantage.co](https://www.alphavantage.co)
3. **Finnhub**: Sign up at [finnhub.io](https://finnhub.io)
4. **Truth Social**: Optional - requires valid account credentials

### Configuration Parameters

The script includes several configurable parameters in the `Config` class:

```python
UPDATE_INTERVAL = 60          # News update frequency (seconds)
HISTORY_LENGTH = 100          # Articles to keep in memory
SIGNAL_THRESHOLD = 0.3        # Minimum confidence for signal generation
MAJOR_EVENT_THRESHOLD = 0.7   # Threshold for audio alerts
```

## 🚀 Usage

### Terminal Mode (Default)
```bash
python news_scan.py 
```

Features:
- Real-time colored output
- Signal strength indicators
- Market state summary every 15 minutes
- Audio alerts for critical events
- Keyboard interrupt support (Ctrl+C)

### Web Dashboard Mode
```bash
streamlit run news_scan.py -- --mode dashboard
```

Features:
- Interactive Streamlit web interface
- Real-time signal visualization
- Configurable parameters
- Historical signal charts
- Multi-column asset display

### Command Line Options
```bash
python news_scan.py --help

Options:
  --mode {terminal,dashboard}  Run mode: terminal or dashboard
  --config CONFIG             Configuration file path
```

## 📈 Understanding Signals

### Signal Types
- **🟢 BULLISH**: Positive outlook, potential price increase
- **🔴 BEARISH**: Negative outlook, potential price decrease  
- **🟡 NEUTRAL**: Mixed or unclear signals

### Signal Strength
- **WEAK**: Low confidence, minor market impact
- **MODERATE**: Medium confidence, noticeable impact
- **STRONG**: High confidence, significant impact
- **CRITICAL**: Very high confidence, major market event

### Monitored Assets

| Asset | Symbol | Key Factors |
|-------|--------|-------------|
| **Gold** | XAU/USD | Inflation, safe-haven demand, USD strength, interest rates |
| **Bitcoin** | BTC/USD | Regulation, adoption, risk sentiment, technology developments |
| **S&P 500** | SPX | Economic growth, Fed policy, corporate earnings |
| **US Dollar** | DXY | Fed policy, economic data, geopolitical events |

## 🎯 Specialized Analysis Features

### Trump Post Analysis
The system includes specialized logic for analyzing posts from Donald Trump's Truth Social account:

- **Trade & Tariffs**: Detects trade war implications
- **Federal Reserve**: Analyzes Fed policy commentary
- **Economic Performance**: Processes economic outlook statements
- **Cryptocurrency**: Evaluates crypto-related posts
- **Geopolitical Events**: Assesses global stability comments

### Market Impact Keywords
The system uses weighted keyword analysis for market impact assessment:

**High Impact (0.8-0.9)**:
- inflation, CPI, PPI, unemployment, NFP, GDP, recession
- Federal Reserve, interest rates, rate hike/cut
- war, conflict, sanctions, trade war

**Medium Impact (0.6-0.8)**:
- growth, jobs, monetary policy, hawkish/dovish
- tariff, volatility, correction, regulation

## 📊 Output Examples

### Terminal Output
```
[14:30:15] BULLISH STRONG signal for GOLD
  Confidence: 0.85 | Market Impact: 0.78
  Sentiment Score: 0.42
  • Rising inflation increases gold's appeal as hedge
  • Geopolitical uncertainty drives safe-haven demand
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Signal Summary Table
```
╔══════════════════════════════════════════════════════════════╗
║           MARKET STATE SUMMARY (Last 15 Minutes)             ║
╠════════════╤══════════════════╤══════════════════╤════════════╣
║  ASSET     │    DIRECTION     │     STRENGTH     │ CONFIDENCE ║
╠════════════╪══════════════════╪══════════════════╪════════════╣
║    GOLD    │     BULLISH      │      STRONG      │    0.85    ║
║  BITCOIN   │     BEARISH      │     MODERATE     │    0.67    ║
║   SP500    │     NEUTRAL      │       WEAK       │    0.32    ║
║    USD     │     BULLISH      │     MODERATE     │    0.58    ║
╚════════════╧══════════════════╧══════════════════╧════════════╝
```

## 🔧 Troubleshooting

### Common Issues

**Missing API Keys**
```
RuntimeError: Missing required environment variable: NEWS_API_KEY
```
Solution: Ensure all required API keys are set in your `.env` file.

**Model Loading Failures**
```
⚠️ All transformer models failed to load. Using basic sentiment only.
```
Solution: Install transformers with `pip install transformers torch` or use basic analysis.

**Audio Alerts Not Working**
```
⚠️ pygame not available. Audio alerts disabled.
```
Solution: Install pygame with `pip install pygame`.

**Truth Social Connection Issues**
```
⚠️ Failed to initialize Truth Social client
```
Solution: Verify credentials in `.env` file or disable Truth Social integration.

### Performance Optimization

- **Memory Usage**: Adjust `HISTORY_LENGTH` for memory-constrained environments
- **Update Frequency**: Increase `UPDATE_INTERVAL` to reduce API calls
- **Model Selection**: Disable transformers for faster startup on limited hardware
- **Signal Filtering**: Adjust `SIGNAL_THRESHOLD` to reduce noise

## 📝 Logging

The application creates detailed logs in `market_analyzer.log` including:
- API request/response status
- Signal generation details
- Error messages and warnings
- Performance metrics

## ⚠️ Disclaimers

1. **Not Financial Advice**: This tool is for educational and research purposes only
2. **Market Risk**: All trading involves risk of financial loss
3. **API Limitations**: Respect rate limits and terms of service for all APIs
4. **Data Accuracy**: News sentiment may not always reflect market reality
5. **Testing Required**: Thoroughly backtest any strategies before live trading

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional news sources
- Enhanced sentiment models  
- New asset classes
- Backtesting capabilities
- Performance optimizations

## 📄 License

This project is provided as-is for educational purposes. Please ensure compliance with all API terms of service and applicable regulations.

## 🔗 Resources

- [NewsAPI Documentation](https://newsapi.org/docs)
- [Finnhub API Reference](https://finnhub.io/docs/api)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io)

---

**Made with ❤️ for the trading community**
