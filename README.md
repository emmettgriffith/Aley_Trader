# Aley Trader 📊

A powerful, professional trading application with advanced charting capabilities, technical analysis, and AI-powered insights.

![Aley Trader](https://img.shields.io/badge/Version-2.2-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## ✨ Features

### 🔥 Core Trading Features
- **Real-time Market Data** - Live stock prices, volume, and historical data via Yahoo Finance
- **Advanced Charting** - Multiple timeframes (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 1d)
- **Technical Analysis** - RSI, Bollinger Bands, Volume Profile, MACD, VWAP, SMA/EMA overlays
- **Pre/Post Market Data** - Extended hours trading information
- **Interactive Charts** - Zoom, pan, and scroll through price history
- **Live Last-Candle Update** - Lightweight 5s updates that move only the newest candlestick without redrawing the chart
- **Resizable Split View** - Drag the sash to resize chart vs. side panel; expand/collapse individual panels
- **Multi-Window** - Launch a new application window from the UI (“New Window”)

### 🌊 Overflow Chart System
- **RSI Overflow Charts** - Specialized charts for overbought/oversold conditions
- **Volume Spike Analysis** - Identify unusual volume patterns
- **Price Breakout Detection** - Spot significant price movements
- **Seamless Switching** - Quickly switch chart types from a dropdown
- **Threshold Analysis** - Statistical insights on market extremes

### 🤖 AI-Powered Analysis
- **OpenAI Integration** - In-depth financial analysis using GPT models
- **Technical Signal Detection** - Automated buy/sell/hold recommendations
- **Risk Assessment** - Balance sheet and cash flow analysis
- **Market Sentiment** - AI-driven market insights

### 🔐 User Management
- **Secure Authentication** - Password-protected user accounts with PBKDF2 hashing
- **Remember Me** - Encrypted credential storage for convenience
- **Multi-User Support** - Individual user data and settings
- **Session Management** - Automatic login and secure logout

### 🎨 Professional UI
- **Deep Sea Theme** - Beautiful dark blue color scheme
- **Responsive Design** - Scales to any screen size
- **Intuitive Navigation** - Tabs for Home, News, Heatmap, Screener, and Charts
 - **Indicator Persistence** - Remembers your selections per symbol (and a global default) across app restarts
- **Real-time Updates** - Auto-refresh capabilities with customizable intervals
 - **Side Panel Modules** - Watchlist, DOM (simulated ladder), Notes, and AI quick view with expand/collapse

### 📈 Order Book (DOM)
- **Simulated Ladder** - Bid/ask ladders around the latest price
- **1s Refresh** - Updates every second based on the most recent trade price
- Designed for low overhead while providing an order-book style view

## 🆕 What’s New

September 2025
- Live 5-second last-candle updater on candlestick charts (moves only the latest bar)
- Resizable split pane between chart and side panel; panel expand/collapse controls
- DOM panel now updates every second using the latest price as baseline (lightweight simulation)
- “New Window” button to spawn additional app instances
- Trading toolbar docked on the left (placeholders for drawing tools)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Internet connection for market data
- OpenAI API key (optional, for AI analysis)
- TAAPI.io API key (optional, for enhanced technical analysis)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/emmettgriffith/Aley_Trader.git
   cd Aley_Trader
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   Use the path that exists in your checkout:
   ```bash
   # Option A (project root script)
   python aley_trader.py

   # Option B (module inside folder)
   python Aley_trader/aley_trader.py
   ```

### First Time Setup

1. **Create an account** - Click "Create Account" on the login screen
2. **Choose a symbol** - Search for any stock ticker (e.g., AAPL, TSLA, MSFT)
3. **Explore chart types** - Use the dropdown to switch between candlestick and overflow charts
4. **Customize timeframes** - Click timeframe buttons to change chart intervals

## 📖 User Guide

### Navigation

- **Home** - Startup page in a tab with quick actions (Screener, News, Heatmap, quick “Open” box)
- **Chart Tabs** - Individual stock charts with full technical analysis; indicator choices persist per symbol
- **News Flow** - Live headlines with symbol filter and auto‑refresh; double‑click to open
- **Heatmap** - Color‑coded performance grid for selected universes
- **Screener** - Scan universes; open or add tickers to watchlist via context menu

### Chart Controls

| Control | Function |
|---------|----------|
| **Timeframe Buttons** | Switch between 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 1d |
| **Chart Type Dropdown** | Choose between Candlestick, RSI Overflow, Volume Overflow, Price Breakout |
| **Auto Refresh Toggle** | Enable/disable automatic chart updates |
| **Manual Refresh** | Force immediate chart reload |
| **Left Toolbar** | Access drawing-tool placeholders |
| **Panel Expand (⇅/⤢)** | Expand/collapse Watchlist, DOM, Notes, AI panels |
| **Resize Sash** | Drag between chart and side panel to resize |
| **Mouse Scroll** | Zoom in/out on chart |
| **Mouse Drag** | Pan around chart area |

### Overflow Charts

Overflow charts provide specialized analysis for extreme market conditions:

- **RSI Overbought (>70)** - Identify potential sell signals
- **RSI Oversold (<30)** - Spot potential buy opportunities  
- **Volume Spike (80th percentile)** - Detect unusual trading activity
- **Price Breakout (90th percentile)** - Find significant price movements

### API Integration

#### OpenAI (Optional)
Add your OpenAI API key to `.env` for AI-powered analysis:
```
OPENAI_API_KEY=your_api_key_here
```

#### TAAPI.io (Optional)
Add your TAAPI key for enhanced technical signals:
```
TAAPI_KEY=your_taapi_key_here
```

## 🛠️ Technical Details

### Architecture

- **Frontend** - Tkinter with custom styling and responsive design
- **Data Source** - Yahoo Finance API via yfinance library
- **Charting** - Matplotlib with custom candlestick rendering
- **Security** - PBKDF2 password hashing with device-specific encryption
- **Storage** - JSON-based user data with file encryption

### Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `yfinance` | Market data | ≥0.2.0 |
| `matplotlib` | Charting | ≥3.5.0 |
| `pandas` | Data processing | ≥1.3.0 |
| `numpy` | Numerical computing | ≥1.21.0 |
| `requests` | API requests | ≥2.25.0 |
| `openai` | AI analysis | ≥0.27.0 |
| `python-dotenv` | Environment variables | ≥0.19.0 |
| `seaborn` | Statistical visualization | ≥0.12.0 |
| `plotly` | Interactive charts | ≥5.15.0 |
| `cryptography` | Secure credential storage | ≥3.0.0 |

### File Structure

```
Aley_Trader/
├── aley_trader.py              # Main application
├── overflow_chart.py           # Overflow chart module
├── overflow_demo.py            # Standalone overflow chart demo
├── overflow_chart_integration.ipynb  # Development notebook
├── requirements.txt            # Python dependencies
├── .env.template              # Environment variables template
├── README.md                  # This file
└── user_data_[username]/      # User-specific data (auto-created)
    ├── watchlist.json         # Personal watchlist
    ├── custom_tabs.json       # Quick tab configuration
    └── indicator_prefs.json   # Saved indicator selections (per symbol + default)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Optional: Default stock symbol to load
DEFAULT_SYMBOL=MSFT

# Optional: OpenAI API key for AI analysis
OPENAI_API_KEY=your_openai_api_key

# Optional: TAAPI.io API key for technical analysis
TAAPI_KEY=your_taapi_key
```

### User Data

Each user gets their own data directory with:
- **watchlist.json** - Personal stock watchlist
- **custom_tabs.json** - Quick access tab configuration
- **remember_me.dat** - Encrypted login credentials (if enabled)
- **indicator_prefs.json** - Indicator selections per symbol and a global “_default” profile

### Screener “All US” Setup (optional)
Place a symbols file in the project (root, `data/`, or `tools/`) to enable a broad US scan:
- `symbols.txt` or `tickers.txt` (one ticker per line)
- `nasdaq_screener.csv` (CSV with a `Symbol` column)
- `symbols.csv` / `tickers.csv` (CSV with `Symbol` or `Ticker` column)

If none is present, the Screener uses a fast Mega‑Caps set.

## 🎯 Usage Examples

### Basic Stock Analysis
1. Launch Aley Trader
2. Search for "AAPL" in the search bar
3. Press Enter or click "Go"
4. Explore different timeframes and chart types

### Overflow Analysis
1. Open any stock chart
2. Click the chart type dropdown
3. Select "Overflow - RSI Overbought"
4. Analyze periods where RSI exceeded 70

### Custom Watchlist
1. Go to the Home page
2. Configure Quick Tabs with your favorite symbols
3. Click any Quick Tab for instant access

## 🚨 Troubleshooting

### Common Issues

**"No data available"**
- Check internet connection
- Try a different stock symbol
- Verify the symbol exists and is actively traded

**"API key not configured"**
- Add your API keys to the `.env` file
- Restart the application after adding keys

**Charts not loading**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check for any error messages in the console

**Login issues**
- Verify username/password are correct
- Try creating a new account if persistent issues

### Performance Tips

- Use longer timeframes (1h, 1d) for better performance
- Disable auto-refresh when not needed
- Close unused chart tabs
- For older systems, stick to candlestick charts over overflow charts

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📧 Support

For support, questions, or feature requests:
- Open an issue on GitHub
- Contact: emmettg@griffithind.com

## 🎉 Acknowledgments

- **Yahoo Finance** - Market data provider
- **OpenAI** - AI analysis capabilities
- **TAAPI.io** - Technical analysis indicators
- **Python Community** - Amazing libraries and tools

---

**⚡ Happy Trading! ⚡**

*Built with ❤️ by Emmett Griffith*
