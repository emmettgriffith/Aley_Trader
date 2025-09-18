#!/usr/bin/env python3
import logging
import math
import subprocess  # Added for spawning new application windows
import yfinance as yf
import pandas as pd
# Import pandas_ta with error handling
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"pandas_ta not available ({e}). Using simplified indicators.")
    PANDAS_TA_AVAILABLE = False
    ta = None
import requests
import re
import time
import random
import openai
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import messagebox, ttk
import tkinter.simpledialog as simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import numpy as np  # <-- Add this import
try:
    import scipy  # noqa: F401 - presence check for yfinance repair
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
from dotenv import load_dotenv
import json
import os
import threading
import webbrowser
import html
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import sys
import hashlib
import hmac
import base64
import platform
import overflow_chart  # Add overflow chart module
import footprint_chart  # Footprint chart module
from typing import Iterable

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# AI patterns module for advanced pattern recognition
try:
    import ai_patterns_lite as ai_patterns
    AI_PATTERNS_AVAILABLE = True
    print("AI Patterns Lite module loaded successfully")
except ImportError as e:
    try:
        import ai_patterns
        AI_PATTERNS_AVAILABLE = True
        print("AI Patterns module loaded successfully")
    except ImportError as e2:
        print(f"Warning: AI Patterns not available ({e2}). Advanced pattern recognition will be limited.")
        AI_PATTERNS_AVAILABLE = False
        ai_patterns = None

# Deep Sea Blue Theme Color Palette
DEEP_SEA_THEME = {
    'primary_bg': '#0A1628',        # Deep ocean blue - main background
    'secondary_bg': '#1B2838',      # Darker sea blue - secondary panels
    'accent_bg': '#2C3E50',         # Steel blue - accent elements
    'surface_bg': '#34495E',        # Lighter blue-gray - surfaces
    'card_bg': '#2E4057',           # Card backgrounds
    'text_primary': '#ECF0F1',      # Light gray - primary text
    'text_secondary': '#BDC3C7',    # Medium gray - secondary text
    'text_accent': '#85C1E9',       # Light blue - accent text
    'success': '#27AE60',           # Ocean green - success/positive
    'danger': '#E74C3C',            # Coral red - danger/negative
    'warning': '#F39C12',           # Deep gold - warning
    'info': '#3498DB',              # Ocean blue - info
    'border': '#4A5568',            # Border color
    'hover': '#5D6D7E',             # Hover state
    'active': '#76D7C4',            # Active state - turquoise
    'shadow': '#0F1A2B'             # Shadow color
}

# Global reference to the active main notebook (chart/news/heatmap tabs)
GLOBAL_NOTEBOOK = None

# Simple RSI calculation function for when pandas_ta is not available
def calculate_simple_rsi(prices, period=14):
    """Calculate simple RSI without pandas_ta"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception:
        # Return a series of NaN if calculation fails
        return pd.Series([float('nan')] * len(prices), index=prices.index)


FOOTPRINT_INTERVAL_MAP = {
    "15s": "1m",
    "30s": "1m",
    "1m": "1m",
    "5m": "5m",
}

FOOTPRINT_TICK_CACHE: dict[tuple[str, str], tuple[pd.DataFrame, float, float]] = {}
FOOTPRINT_CACHE_TTL = 15.0  # seconds


def estimate_tick_size(symbol: str, price_samples: Iterable[float]) -> float:
    """Heuristic for tick size based on symbol and price samples."""

    upper = symbol.upper()
    if upper.endswith("=F") or upper.startswith(("ES", "NQ", "YM", "RTY")):
        return 0.25

    prices = [float(p) for p in price_samples if pd.notna(p)]
    if len(prices) >= 2:
        diffs = sorted({round(abs(a - b), 6) for a, b in zip(prices[:-1], prices[1:]) if abs(a - b) > 0})
        if diffs:
            return max(min(diffs), 0.0001)

    try:
        latest_price = float(prices[-1]) if prices else float(yf.Ticker(symbol).fast_info.get("lastPrice") or 0)
    except Exception:
        latest_price = 0.0

    if latest_price >= 1000:
        return 0.1
    if latest_price >= 100:
        return 0.01
    if latest_price >= 10:
        return 0.01
    if latest_price >= 1:
        return 0.001
    return 0.0001


def fetch_symbol_ticks(symbol: str, interval: str = "1m", limit: int | None = None) -> tuple[pd.DataFrame, float]:
    """Fetch synthetic tick data for footprint rendering using yfinance.

    Results are cached briefly so repeated redraws do not refetch from the
    network. The data is a light approximation derived from minute candles to
    keep initial load within a few seconds.
    """

    key = (symbol.upper(), interval)
    now = time.time()
    cache_entry = FOOTPRINT_TICK_CACHE.get(key)
    if cache_entry and now - cache_entry[2] < FOOTPRINT_CACHE_TTL:
        cached_df, cached_tick, _ = cache_entry
        return cached_df.copy(deep=True), cached_tick

    yf_interval = FOOTPRINT_INTERVAL_MAP.get(interval, "1m")
    period = "5d" if yf_interval in ("5m", "15m") else "1d"
    if limit is None:
        limit = 90 if interval in ("15s", "30s", "1m") else 150

    try:
        hist = yf.Ticker(symbol).history(period=period, interval=yf_interval, auto_adjust=False, prepost=True)
    except Exception as exc:
        logging.warning("Footprint history fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=["ts", "price", "size", "side"]), 0.01

    if hist.empty:
        return pd.DataFrame(columns=["ts", "price", "size", "side"]), 0.01

    hist = hist.tail(limit)
    rows = []
    price_samples = []
    for ts, row in hist.iterrows():
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        volume = float(row.get("Volume", 0.0) or 0.0)
        slice_volume = max(volume / 6.0, 1.0)
        price_points = [
            (row.get("Open"), "ASK"),
            (row.get("Close"), "ASK" if row.get("Close", 0) >= row.get("Open", 0) else "BID"),
            (row.get("Low"), "BID"),
            (row.get("High"), "ASK"),
        ]
        for price, side in price_points:
            if pd.isna(price):
                continue
            price_samples.append(price)
            rows.append({
                "ts": ts,
                "price": float(price),
                "size": slice_volume,
                "side": side,
            })

    ticks = pd.DataFrame(rows, columns=["ts", "price", "size", "side"])
    if not ticks.empty:
        ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True)

    tick_size = estimate_tick_size(symbol, price_samples)
    FOOTPRINT_TICK_CACHE[key] = (ticks.copy(deep=True), tick_size, now)
    return ticks, tick_size

# Load environment variables from .env file in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Also try loading from current directory as fallback
if not os.getenv('OPENAI_API_KEY'):
    load_dotenv('.env')

# Debug: Log if environment variables are loaded
if os.getenv('OPENAI_API_KEY'):
    logging.info("OpenAI API key loaded successfully")
else:
    logging.warning("OpenAI API key not found in environment")

# Default ticker symbol from environment or fallback to MSFT
symbol = os.getenv("DEFAULT_SYMBOL", "MSFT")
ticker = yf.Ticker(symbol)

# Initialize global variables for later use
prev_close = None
latest_close = None
percent_gain = None

def initialize_market_data():
    """Initialize market data - call this when needed, not during import"""
    global prev_close, latest_close, percent_gain
    
    try:
        # Get historical data for the last 2 days
        hist = ticker.history(period="2d")

        if len(hist) < 2:
            print("Not enough data to calculate percent gain.")
        else:
            # Get closing prices for the last two days
            prev_close = hist['Close'].iloc[0]
            latest_close = hist['Close'].iloc[1]

            # Calculate percent gain
            percent_gain = ((latest_close - prev_close) / prev_close) * 100

            logging.info(f"{symbol} previous close: {prev_close}")
            logging.info(f"{symbol} latest close: {latest_close}")
            logging.info(f"Percent gain: {percent_gain:.2f}%")
    except Exception as e:
        logging.warning(f"Error initializing market data: {e}")
        # Set defaults
        prev_close = 100.0
        latest_close = 102.0
        percent_gain = 2.0

# --- HTTP helper with timeout/retries ---
def http_get_json(url, params=None, timeout=10, retries=2, backoff=1.5):
    """GET JSON with basic retries and timeout."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            else:
                break
    raise last_err if last_err else RuntimeError("Unknown request error")

def http_get_text(url, params=None, timeout=10, retries=2, backoff=1.5, headers=None):
    """GET text with basic retries, timeout, and a browser-like User-Agent."""
    last_err = None
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            resp.raise_for_status()
            resp.encoding = resp.encoding or "utf-8"
            return resp.text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            else:
                break
    raise last_err if last_err else RuntimeError("Unknown request error")

# --- Technical Analysis Signal Section ---
TAAPI_KEY = os.getenv("TAAPI_KEY", "YOUR_TAAPI_KEY")  # Load from environment variable

def get_ta_signal(symbol):
    if not TAAPI_KEY or TAAPI_KEY == "YOUR_TAAPI_KEY":
        logging.warning("TAAPI_KEY missing; skipping TA signal fetch")
        return "Signal unavailable"
    url = "https://api.taapi.io/summary"
    params = {
        "secret": TAAPI_KEY,
        "exchange": "NASDAQ",
        "symbol": symbol,
        "interval": "1d",
    }
    try:
        data = http_get_json(url, params=params, timeout=10, retries=2)
        if isinstance(data, dict) and "recommendation" in data:
            recommendation = str(data.get("recommendation", "")).lower()
            if "buy" in recommendation:
                return "Buy"
            if "sell" in recommendation:
                return "Sell"
            if "hold" in recommendation:
                return "Hold"
            return f"Signal: {data.get('recommendation')}"
        return "No signal available"
    except Exception as e:
        logging.warning(f"TAAPI signal error: {e}")
        return "Signal unavailable"

def get_rsi(symbol):
    if not TAAPI_KEY or TAAPI_KEY == "YOUR_TAAPI_KEY":
        return None
    url = "https://api.taapi.io/rsi"
    params = {
        "secret": TAAPI_KEY,
        "exchange": "NASDAQ",
        "symbol": symbol,
        "interval": "1d",
    }
    try:
        data = http_get_json(url, params=params, timeout=10, retries=2)
        if isinstance(data, dict) and "value" in data:
            return float(data["value"])  # may raise ValueError; that's fine
    except Exception as e:
        logging.warning(f"TAAPI RSI error: {e}")
    return None

# Initialize global variables for technical analysis
ta_signal = None
rsi = None

def initialize_technical_analysis():
    """Initialize technical analysis data - call this when needed"""
    global ta_signal, rsi
    
    try:
        # Display technical analysis signal
        ta_signal = get_ta_signal(symbol)
        logging.info(f"Technical Analysis Signal for {symbol}: {ta_signal}")

        # Display RSI and overbought status
        rsi = get_rsi(symbol)
        if rsi is not None:
            logging.info(f"RSI for {symbol}: {rsi:.2f}")
            if rsi > 70:
                logging.info("The stock is overbought (RSI > 70)!")
            elif rsi < 30:
                logging.info("The stock is oversold (RSI < 30).")
        else:
            logging.info("Could not retrieve RSI value.")
    except Exception as e:
        logging.warning(f"Error initializing technical analysis: {e}")
        ta_signal = "Signal unavailable"
        rsi = 50.0  # Default RSI

# --- Helper function for watchlist percent gain ---
def get_percent_gain(symbol):
    """Get 2-day percent gain for a symbol"""
    try:
        ticker_obj = yf.Ticker(symbol)
        hist = ticker_obj.history(period="2d")
        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[0]
            latest_close = hist['Close'].iloc[1]
            return ((latest_close - prev_close) / prev_close) * 100
        return None
    except Exception:
        return None

# --- OpenAI Financial Analysis Section ---
openai.api_key = os.getenv("OPENAI_API_KEY")  # Load from environment variable for security

def get_in_depth_analysis(symbol):
    # Check if OpenAI API key is available
    if not openai.api_key:
        return f"OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file to enable AI analysis.\n\nYou can get an API key from: https://platform.openai.com/account/api-keys\n\nFor now, showing basic stock information for {symbol}..."
    
    # Fetch company info and financials using yfinance
    ticker_obj = yf.Ticker(symbol)  # Create new ticker object for this symbol
    info = ticker_obj.info
    financials = ticker_obj.financials.replace({np.nan: np.nan}) if hasattr(ticker_obj, 'financials') else pd.DataFrame()
    balance_sheet = ticker_obj.balance_sheet.replace({np.nan: np.nan}) if hasattr(ticker_obj, 'balance_sheet') else pd.DataFrame()
    cashflow = ticker_obj.cashflow.replace({np.nan: np.nan}) if hasattr(ticker_obj, 'cashflow') else pd.DataFrame()

    def summarize_info(info_dict):
        keys = [
            'longName','sector','industry','marketCap',
            'trailingPE','forwardPE','profitMargins','revenueGrowth','earningsGrowth',
            'beta','dividendYield','payoutRatio','debtToEquity'
        ]
        return {k: info_dict.get(k) for k in keys if k in info_dict}

    def summarize_df(df, rows):
        if df is None or df.empty:
            return {}
        out = {}
        for r in rows:
            try:
                s = df.loc[r].dropna()
                # Take last up to 4 points
                out[r] = {str(k): float(v) if pd.notna(v) else None for k, v in list(s.to_dict().items())[-4:]}
            except Exception:
                continue
        return out

    fin_sum = summarize_df(financials, ['Total Revenue','Gross Profit','Operating Income','Net Income'])
    bs_sum = summarize_df(balance_sheet, ['Total Assets','Total Liab','Cash','Total Stockholder Equity'])
    cf_sum = summarize_df(cashflow, ['Total Cash From Operating Activities','Capital Expenditures','Free Cash Flow'])

    # Prepare a summary for OpenAI
    prompt = (
        f"Provide a concise, insight-driven financial analysis of {info.get('longName', symbol)} ({symbol}). "
        f"Discuss profitability, growth, leverage, cash generation, and risks. Use the data below.\n"
        f"Company Info (key fields): {summarize_info(info)}\n"
        f"Financials (last periods): {fin_sum}\n"
        f"Balance Sheet (last periods): {bs_sum}\n"
        f"Cash Flow (last periods): {cf_sum}\n"
        f"2-day price change: {percent_gain:.2f}% | TA signal: {ta_signal} | RSI: {rsi if rsi is not None else 'N/A'}"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error fetching analysis from OpenAI: {e}"

# Initialize global variable for analysis
analysis = None

def initialize_openai_analysis():
    """Initialize OpenAI analysis - call this when needed"""
    global analysis
    
    try:
        # Get and display in-depth analysis
        analysis = get_in_depth_analysis(symbol)
        logging.info("--- In-Depth Financial Analysis (OpenAI) ---")
        logging.info(analysis)
    except Exception as e:
        logging.warning(f"Error initializing OpenAI analysis: {e}")
        analysis = "Analysis unavailable - please check API configuration"

# --- Chart display window with stock chart and technical indicators ---
def show_chart_with_points(symbol, ticker, prev_close, latest_close, percent_gain, ta_signal, rsi, analysis, notebook, tab_title):
    # Dynamic chart sizing - maximize available space
    import tkinter as tk
    
    # Get screen dimensions
    root_window = notebook.winfo_toplevel()
    screen_width = root_window.winfo_screenwidth()
    screen_height = root_window.winfo_screenheight()
    
    # Calculate optimal chart size based on available space
    # Account for side panel (220px), margins, and other UI elements
    available_width = screen_width - 320  # Side panel + margins
    available_height = screen_height - 200  # Top bars + bottom elements
    
    # Convert to inches for matplotlib (assuming 100 DPI)
    width = max(12, min(available_width / 100, 20))  # Min 12", max 20"
    height = max(8, min(available_height / 100, 14))  # Min 8", max 14"
    
    print(f"Chart size: {width:.1f}\" x {height:.1f}\"")  # Debug info

    # --- Deep Sea Blue Chart Colors ---
    plt.style.use('dark_background')
    grid_color = DEEP_SEA_THEME['border']          # Deep sea grid
    candle_up = DEEP_SEA_THEME['success']          # Ocean green for bullish candles
    candle_down = DEEP_SEA_THEME['danger']         # Coral red for bearish candles
    bb_color = DEEP_SEA_THEME['warning']           # Deep gold for bollinger bands
    rsi_color = DEEP_SEA_THEME['info']             # Ocean blue for RSI
    overbought_color = DEEP_SEA_THEME['danger']    # Coral for overbought
    oversold_color = DEEP_SEA_THEME['active']      # Turquoise for oversold
    volume_profile_color = DEEP_SEA_THEME['accent_bg']  # Steel blue for volume profile

    # Container for chart
    frame = tk.Frame(notebook)
    frame.pack(fill=tk.BOTH, expand=1)

    # --- Chart container with toolbar on left ---
    chart_container = tk.Frame(frame, bg=DEEP_SEA_THEME['primary_bg'])
    chart_container.pack(fill=tk.BOTH, expand=1)

    # Drawing Toolbar (left side of chart)
    drawing_tools = [
        ("Trendline", "📈"),
        ("Horizontal", "━"),
        ("Vertical", "┃"),
        ("Fibonacci", "𝑭"),
        ("Text", "📝"),
        ("Rectangle", "▭"),
        ("Ellipse", "◯"),
        ("Triangle", "△"),
        ("Freehand", "✏️"),
        ("Eraser", "🧹"),
        ("Clear", "❌"),
        ("Crosshair", "+"),
        ("Zoom", "🔍")
    ]
    toolbar_frame = tk.Frame(chart_container, bg=DEEP_SEA_THEME['secondary_bg'], width=56)
    toolbar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2), pady=2)
    toolbar_frame.pack_propagate(False)
    for tool_name, icon in drawing_tools:
        btn = tk.Button(toolbar_frame, text=icon, width=3, height=1, font=("Segoe UI", 14),
                       bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'])
        btn.pack(side=tk.TOP, pady=3, padx=2)

    # Ensure chart canvas is packed to the right of the toolbar
    # Find or create the chart canvas after toolbar_frame
    # Example for candlestick chart:
    # canvas = FigureCanvasTkAgg(fig, master=chart_container)
    # canvas.draw()
    # canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    current_timeframe = tk.StringVar(value="1d")
    auto_refresh_enabled = tk.BooleanVar(value=True)
    refresh_interval = 30000  # 30 seconds for auto-refresh
    refresh_timer = None
    # Lightweight last-tick updater (5s) that only moves the most recent candle
    last_tick_timer = None

    # Timeframe mapping for yfinance - Extended periods for better pre/post market data
    timeframe_mapping = {
        "1m": {"period": "1d", "interval": "1m"},
        "5m": {"period": "5d", "interval": "5m"},
        "10m": {"period": "5d", "interval": "5m"},  # Use 5m and aggregate
        "15m": {"period": "7d", "interval": "15m"},
        "30m": {"period": "30d", "interval": "30m"},
        "1h": {"period": "30d", "interval": "1h"},
        "2h": {"period": "60d", "interval": "1h"},  # Use 1h and aggregate
        "3h": {"period": "60d", "interval": "1h"},  # Use 1h and aggregate
        "4h": {"period": "60d", "interval": "1h"},  # Use 1h and aggregate
        "6h": {"period": "60d", "interval": "1h"},  # Use 1h and aggregate
        "1d": {"period": "1y", "interval": "1d"}
    }

    def fetch_chart_data():
        tf = current_timeframe.get()
        tf_config = timeframe_mapping[tf]
        
        try:
            data_note = ""
            # Optimized data fetching - simplified and faster
            include_prepost = tf in ["1m", "5m", "10m", "15m", "30m", "1h"]
            
            print(f"Fetching {tf} data with config: {tf_config}")
            
            # Try multiple data fetching strategies for robustness
            chart_hist = None
            
            def fetch_history_safe(params):
                """Call ticker.history with compatibility fallbacks."""
                try:
                    return ticker.history(**params)
                except TypeError as type_err:
                    if "unexpected keyword argument 'repair'" in str(type_err).lower() and "repair" in params:
                        print("  'repair' unsupported by yfinance version; retrying without it")
                        retry_params = dict(params)
                        retry_params.pop("repair", None)
                        return ticker.history(**retry_params)
                    raise
                except ModuleNotFoundError as mod_err:
                    if "scipy" in str(mod_err).lower() and "repair" in params:
                        print("  SciPy missing; retrying history fetch without repair")
                        retry_params = dict(params)
                        retry_params.pop("repair", None)
                        return ticker.history(**retry_params)
                    raise

            def build_strategy(period, interval, prepost, allow_repair=True):
                strat = {"period": period, "interval": interval, "prepost": prepost}
                if allow_repair and SCIPY_AVAILABLE:
                    strat["repair"] = True
                return strat

            strategies = [
                # Strategy 1: Original request
                build_strategy(tf_config["period"], tf_config["interval"], include_prepost),
                # Strategy 2: Shorter period, same interval
                build_strategy("5d", tf_config["interval"], include_prepost),
                # Strategy 3: Even shorter period
                build_strategy("2d", tf_config["interval"], include_prepost),
                # Strategy 4: Daily fallback
                build_strategy("30d", "1d", False),
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    print(f"  Trying strategy {i+1}: {strategy}")
                    temp_hist = fetch_history_safe(strategy)
                    if not temp_hist.empty and len(temp_hist) > 0:
                        chart_hist = temp_hist
                        print(f"  Success! Got {len(chart_hist)} data points")
                        break
                    else:
                        print(f"  Strategy {i+1} returned empty data")
                except Exception as strategy_error:
                    print(f"  Strategy {i+1} failed: {strategy_error}")
                    continue
            
            if chart_hist is None or len(chart_hist) < 2:
                print(f"Primary fetch returned insufficient data ({'none' if chart_hist is None else len(chart_hist)} bars). Trying extended daily fallback...")
                try:
                    daily_fallback = fetch_history_safe(build_strategy("2y", "1d", False, allow_repair=False))
                except Exception as daily_err:
                    print(f"  Daily fallback failed: {daily_err}")
                    daily_fallback = pd.DataFrame()
                if daily_fallback is not None and not daily_fallback.empty:
                    chart_hist = daily_fallback
                    data_note = "daily fallback"
                    print(f"  Daily fallback succeeded with {len(chart_hist)} bars")
            
            if chart_hist is None or chart_hist.empty:
                print(f"All strategies failed for {tf}, creating minimal test data...")
                # Create minimal test data to prevent crash
                from datetime import datetime, timedelta
                dates = pd.date_range(start=datetime.now() - timedelta(days=5), end=datetime.now(), freq='D')
                base_price = None
                try:
                    if latest_close is not None:
                        base_price = float(latest_close)
                    elif prev_close is not None:
                        base_price = float(prev_close)
                except Exception:
                    base_price = None
                if base_price is None:
                    base_price = 100.0

                trend = np.linspace(-1.5, 1.5, len(dates))
                close_vals = base_price + trend
                open_vals = np.concatenate([[close_vals[0]], close_vals[:-1]])
                high_vals = np.maximum(open_vals, close_vals) + 0.8
                low_vals = np.minimum(open_vals, close_vals) - 0.8
                volume_vals = np.linspace(800_000, 1_400_000, len(dates))

                chart_hist = pd.DataFrame({
                    'Open': open_vals,
                    'High': high_vals,
                    'Low': low_vals,
                    'Close': close_vals,
                    'Volume': volume_vals
                }, index=dates)
                print(f"Created {len(chart_hist)} test data points")
                data_note = "synthetic"

            print(f"Final data check: {len(chart_hist)} data points for {tf}")
            
            # Handle custom aggregation more efficiently
            if tf == "10m" and tf_config["interval"] == "5m":
                print("Aggregating 5m data to 10m...")
                chart_hist = chart_hist.groupby(chart_hist.index.floor('10T')).agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
            elif tf in ["2h", "3h", "4h", "6h"] and tf_config["interval"] == "1h":
                hours = int(tf[0])
                freq = f"{hours}H"
                print(f"Aggregating 1h data to {tf}...")
                chart_hist = chart_hist.groupby(chart_hist.index.floor(freq)).agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
            
            if chart_hist.empty:
                raise ValueError("No data available after aggregation")
            
            print(f"Final data points after aggregation: {len(chart_hist)}")
            
            # More conservative data limiting for better charts
            max_points = {
                "1m": 300, "5m": 400, "10m": 300, "15m": 250, 
                "30m": 200, "1h": 168, "2h": 120, "3h": 100, 
                "4h": 80, "6h": 60, "1d": 100
            }
            
            original_length = len(chart_hist)
            if tf in max_points and len(chart_hist) > max_points[tf]:
                chart_hist = chart_hist.tail(max_points[tf])
                print(f"Limited data from {original_length} to {len(chart_hist)} points for performance")
            else:
                print(f"No data limiting needed: {len(chart_hist)} <= {max_points.get(tf, 'unlimited')}")
            
            # Simplified technical indicators - only if needed
            try:
                # Only calculate RSI for performance if pandas_ta is available
                if PANDAS_TA_AVAILABLE and ta is not None:
                    chart_hist["RSI"] = ta.rsi(chart_hist["Close"], length=14)
                else:
                    # Simple RSI calculation fallback
                    chart_hist["RSI"] = calculate_simple_rsi(chart_hist["Close"])
                # Skip Bollinger Bands for better performance
            except Exception as rsi_error:
                print(f"Technical indicators error: {rsi_error}")
                pass  # Skip all technical indicators if any error
            
            chart_hist = chart_hist.reset_index()
            chart_hist['Date'] = mdates.date2num(chart_hist['Date'])
            
            # Simplified volume profile - fewer bins for speed
            raw_bins = len(chart_hist) // 3
            num_bins = min(20, raw_bins if raw_bins >= 2 else 2)
            price_bins = np.linspace(chart_hist['Low'].min(), chart_hist['High'].max(), num_bins)
            volume_profile = np.zeros(len(price_bins) - 1)
            
            # Vectorized volume profile calculation
            for i in range(len(price_bins) - 1):
                mask = (chart_hist['Low'] < price_bins[i+1]) & (chart_hist['High'] > price_bins[i])
                volume_profile[i] = chart_hist.loc[mask, 'Volume'].sum()
                
            high_vol_idx = np.argmax(volume_profile) if len(volume_profile) > 0 else 0
            low_vol_idx = np.argmin(volume_profile) if len(volume_profile) > 0 else 0
            high_vol_price = (price_bins[high_vol_idx] + price_bins[high_vol_idx+1]) / 2 if len(price_bins) > 1 else 0
            low_vol_price = (price_bins[low_vol_idx] + price_bins[low_vol_idx+1]) / 2 if len(price_bins) > 1 else 0
            
            return chart_hist, price_bins, volume_profile, high_vol_price, low_vol_price, data_note
        except Exception as e:
            print(f"Error fetching data for {tf}: {e}")
            # Fallback to daily data with error handling
            try:
                print("Attempting fallback to daily data...")
                chart_hist = ticker.history(period="60d", interval="1d", prepost=False, repair=True)
                
                if chart_hist.empty:
                    raise ValueError("No fallback data available")
                    
                print(f"Fallback data retrieved: {len(chart_hist)} daily points")
                
                try:
                    if PANDAS_TA_AVAILABLE and ta is not None:
                        chart_hist["RSI"] = ta.rsi(chart_hist["Close"], length=14)
                    else:
                        # Simple RSI calculation fallback
                        chart_hist["RSI"] = calculate_simple_rsi(chart_hist["Close"])
                except Exception as ta_err:
                    print(f"Skipping technical indicators in fallback: {ta_err}")
                    
                chart_hist = chart_hist.reset_index()
                chart_hist['Date'] = mdates.date2num(chart_hist['Date'])
                
                # Simple volume profile for fallback
                if len(chart_hist) > 0:
                    price_bins = np.linspace(chart_hist['Low'].min(), chart_hist['High'].max(), 20)
                    volume_profile = np.zeros(len(price_bins) - 1)
                    for i in range(len(price_bins) - 1):
                        mask = (chart_hist['Low'] < price_bins[i+1]) & (chart_hist['High'] > price_bins[i])
                        volume_profile[i] = chart_hist.loc[mask, 'Volume'].sum()
                    
                    high_vol_idx = np.argmax(volume_profile) if len(volume_profile) > 0 else 0
                    low_vol_idx = np.argmin(volume_profile) if len(volume_profile) > 0 else 0
                    high_vol_price = (price_bins[high_vol_idx] + price_bins[high_vol_idx+1]) / 2 if len(price_bins) > 1 else 0
                    low_vol_price = (price_bins[low_vol_idx] + price_bins[low_vol_idx+1]) / 2 if len(price_bins) > 1 else 0
                else:
                    price_bins = np.array([0, 1])
                    volume_profile = np.array([0])
                    high_vol_price = 0
                    low_vol_price = 0
                
                return chart_hist, price_bins, volume_profile, high_vol_price, low_vol_price, "daily fallback"
            except Exception as fallback_error:
                print(f"Fallback failed: {fallback_error}")
                # Return empty data structure to prevent crash
                empty_df = pd.DataFrame({
                    'Date': [mdates.date2num(datetime.now())],
                    'Open': [100], 'High': [100], 'Low': [100], 'Close': [100], 'Volume': [0]
                })
                return empty_df, np.array([99, 101]), np.array([0]), 100, 100, "synthetic"

    def draw_chart(chart_type="Candlestick"):
        nonlocal refresh_timer, last_tick_timer
        footprint_active = chart_type == "Order Flow Footprint"
        if not footprint_active and hasattr(frame, "footprint_settings"):
            fp_state = frame.footprint_settings
            if fp_state.get("frame") and fp_state["frame"].winfo_ismapped():
                fp_state["frame"].pack_forget()
            if fp_state.get("fetch_job"):
                try:
                    frame.after_cancel(fp_state["fetch_job"])
                except Exception:
                    pass
                fp_state["fetch_job"] = None
            worker = fp_state.get("worker")
            if worker:
                try:
                    worker.stop()
                except Exception:
                    pass
                fp_state["worker"] = None
        # --- Drawing Toolbar Placeholder (left side) ---
        drawing_tools = [
            ("Trendline", "📈"),
            ("Horizontal", "━"),
            ("Vertical", "┃"),
            ("Fibonacci", "𝑭"),
            ("Text", "📝"),
            ("Rectangle", "▭"),
            ("Ellipse", "◯"),
            ("Triangle", "△"),
            ("Freehand", "✏️"),
            ("Eraser", "🧹"),
            ("Clear", "❌"),
            ("Crosshair", "+"),
            ("Zoom", "🔍")
        ]
        toolbar_frame = tk.Frame(frame, bg=DEEP_SEA_THEME['secondary_bg'], width=56)
        toolbar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2), pady=2)
        toolbar_frame.pack_propagate(False)
        for tool_name, icon in drawing_tools:
            btn = tk.Button(toolbar_frame, text=icon, width=3, height=1, font=("Segoe UI", 14),
                           bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'])
            btn.pack(side=tk.TOP, pady=3, padx=2)
        # --- TradingView-style Drawing Toolbar ---
        class DrawingToolbar(tk.Frame):
            TOOL_LIST = [
                ("trendline", "Trendline", "📈"),
                ("hline", "Horizontal Line", "━"),
                ("vline", "Vertical Line", "┃"),
                ("fibonacci", "Fibonacci", "𝑭"),
                ("text", "Text Label", "📝"),
                ("rectangle", "Rectangle", "▭"),
                ("ellipse", "Ellipse", "◯"),
                ("triangle", "Triangle", "△"),
                ("pencil", "Pencil", "✏️"),
                ("eraser", "Eraser", "🧹"),
                ("clear", "Clear All", "❌"),
                ("crosshair", "Crosshair", "+"),
                ("zoom", "Zoom", "🔍")
            ]

            def __init__(self, parent, on_tool_select):
                super().__init__(parent, bg=DEEP_SEA_THEME['secondary_bg'], width=52)
                self.on_tool_select = on_tool_select
                self.selected_tool = None
                self.tool_buttons = {}
                self.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2), pady=2)
                for tool_id, tool_name, icon in self.TOOL_LIST:
                    btn = tk.Button(self, text=icon, width=3, height=1, font=("Segoe UI", 14),
                                    bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'],
                                    command=lambda t=tool_id: self.select_tool(t))
                    btn.pack(side=tk.TOP, pady=3, padx=2)
                    btn.tooltip = tool_name  # For future tooltip support
                    self.tool_buttons[tool_id] = btn

            def select_tool(self, tool_id):
                self.selected_tool = tool_id
                self.on_tool_select(tool_id)
                for t, btn in self.tool_buttons.items():
                    btn.config(bg=DEEP_SEA_THEME['info'] if t == tool_id else DEEP_SEA_THEME['surface_bg'])

        # --- Drawing Manager (placeholder for chart interaction logic) ---
        class DrawingManager:
            def __init__(self, chart_canvas):
                self.chart_canvas = chart_canvas
                self.active_tool = None
                self.drawings = []  # Store drawing objects

            def set_tool(self, tool_id):
                self.active_tool = tool_id
                # Placeholder: integrate with chart_canvas for drawing

            # Placeholder methods for each tool
            def draw_trendline(self): pass
            def draw_hline(self): pass
            def draw_vline(self): pass
            def draw_fibonacci(self): pass
            def draw_text(self): pass
            def draw_rectangle(self): pass
            def draw_ellipse(self): pass
            def draw_triangle(self): pass
            def draw_pencil(self): pass
            def erase(self): pass
            def clear_all(self): pass
            def crosshair(self): pass
            def zoom(self): pass

        # --- Integrate Toolbar and DrawingManager ---
        # Find chart canvas (Matplotlib FigureCanvasTkAgg)
        chart_canvas = None
        for widget in frame.winfo_children():
            if hasattr(widget, 'get_tk_widget'):
                chart_canvas = widget
                break
        drawing_manager = DrawingManager(chart_canvas)
        drawing_toolbar = DrawingToolbar(frame, drawing_manager.set_tool)
        # --- Drawing Toolbar (TradingView-style) ---
        class DrawingToolbar(tk.Frame):
            def __init__(self, parent, on_tool_select):
                super().__init__(parent, bg=DEEP_SEA_THEME['secondary_bg'], width=48)
                self.on_tool_select = on_tool_select
                self.selected_tool = None
                self.tools = [
                    ("trendline", "📈"),
                    ("ray", "➖"),
                    ("hline", "━"),
                    ("vline", "┃"),
                    ("fibonacci", "𝑭"),
                    ("text", "📝"),
                    ("rectangle", "▭"),
                    ("ellipse", "◯"),
                    ("triangle", "△"),
                    ("pencil", "✏️"),
                    ("eraser", "🧹"),
                    ("clear", "❌"),
                    ("zoom", "🔍"),
                    ("crosshair", "+")
                ]
                self.tool_buttons = {}
                self.collapsed = False
                self.collapse_btn = tk.Button(self, text="⏴", command=self.toggle_collapse, width=2, bg=DEEP_SEA_THEME['surface_bg'])
                self.collapse_btn.pack(side=tk.TOP, pady=2)
                self.tools_frame = tk.Frame(self, bg=DEEP_SEA_THEME['secondary_bg'])
                self.tools_frame.pack(side=tk.TOP, fill=tk.Y, expand=1)
                for tool, icon in self.tools:
                    btn = tk.Button(self.tools_frame, text=icon, width=2, height=1, font=("Segoe UI", 14),
                                    bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'],
                                    command=lambda t=tool: self.select_tool(t))
                    btn.pack(side=tk.TOP, pady=2, padx=2)
                    self.tool_buttons[tool] = btn
                self.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2), pady=2)

            def select_tool(self, tool):
                self.selected_tool = tool
                self.on_tool_select(tool)
                for t, btn in self.tool_buttons.items():
                    btn.config(bg=DEEP_SEA_THEME['info'] if t == tool else DEEP_SEA_THEME['surface_bg'])

            def toggle_collapse(self):
                self.collapsed = not self.collapsed
                if self.collapsed:
                    self.tools_frame.pack_forget()
                    self.collapse_btn.config(text="⏵")
                else:
                    self.tools_frame.pack(side=tk.TOP, fill=tk.Y, expand=1)
                    self.collapse_btn.config(text="⏴")

        # Placeholder drawing logic handler
        class DrawingManager:
            def __init__(self, canvas):
                self.canvas = canvas
                self.active_tool = None
                # Placeholder for storing drawings
                self.drawings = []

            def set_tool(self, tool):
                self.active_tool = tool
                # TODO: Integrate tool activation logic

            # Placeholder methods for each tool
            def draw_trendline(self): pass
            def draw_ray(self): pass
            def draw_hline(self): pass
            def draw_vline(self): pass
            def draw_fibonacci(self): pass
            def draw_text(self): pass
            def draw_rectangle(self): pass
            def draw_ellipse(self): pass
            def draw_triangle(self): pass
            def draw_pencil(self): pass
            def erase(self): pass
            def clear_all(self): pass
            def zoom(self): pass
            def crosshair(self): pass
            # TODO: Add undo/redo, edit/delete, sync, etc.

    # (moved) Integrate Drawing Toolbar and Manager above
        
        # Cancel any existing refresh timer
        if refresh_timer:
            frame.after_cancel(refresh_timer)
            refresh_timer = None
        # Cancel any previous last-tick updater
        if last_tick_timer:
            try:
                frame.after_cancel(last_tick_timer)
            except Exception:
                pass
            last_tick_timer = None
        
        # Clear all existing widgets in chart frame
        for widget in frame.winfo_children():
            widget.destroy()
            
        # Add control bar at the top of chart frame
        control_frame = tk.Frame(frame, bg=DEEP_SEA_THEME['secondary_bg'])
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # --- Indicators state (persist across redraws) ---
        saved_state = get_indicator_state(symbol)
        if not hasattr(frame, "indicator_vars"):
            frame.indicator_vars = {
                "RSI": tk.BooleanVar(value=bool(saved_state.get("RSI", True))),
                "SMA 20": tk.BooleanVar(value=bool(saved_state.get("SMA 20", False))),
                "EMA 50": tk.BooleanVar(value=bool(saved_state.get("EMA 50", False))),
                "Bollinger (20,2)": tk.BooleanVar(value=bool(saved_state.get("Bollinger (20,2)", False))),
                "VWAP": tk.BooleanVar(value=bool(saved_state.get("VWAP", False))),
                "MACD": tk.BooleanVar(value=bool(saved_state.get("MACD", False))),
            }
        indicator_vars = frame.indicator_vars
        
        # Chart type selector (new overflow chart integration)
        chart_selector_frame = tk.Frame(control_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        chart_selector_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # Chart type dropdown
        current_chart_type = tk.StringVar(value="Candlestick")
        chart_types = [
            "Candlestick",
            "Volume Profile - Order Book",
            "Volume Profile Plus",
            "Order Flow Footprint",
            "Overflow - Momentum",
            "Overflow - RSI Overbought", 
            "Overflow - RSI Oversold",
            "Overflow - Volume Spike",
            "Overflow - Price Breakout"
        ]
        
        chart_dropdown = ttk.Combobox(
            chart_selector_frame,
            textvariable=current_chart_type,
            values=chart_types,
            state="readonly",
            font=("Segoe UI", 9),
            width=18
        )
        chart_dropdown.pack(side=tk.LEFT, padx=5, pady=3)
        
        def on_chart_type_change(event=None):
            selected = current_chart_type.get()
            print(f"Chart type changed to: {selected}")
            # Trigger chart redraw with new type
            frame.after(10, lambda: draw_chart(chart_type=selected))
        
        chart_dropdown.bind('<<ComboboxSelected>>', on_chart_type_change)
        
        # Timeframe buttons
        timeframe_frame = tk.Frame(control_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        timeframe_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        timeframes = ["1m", "5m", "10m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "1d"]
        
        def change_timeframe(tf):
            current_timeframe.set(tf)
            # Immediate redraw for responsiveness
            frame.after(10, lambda: draw_chart("Candlestick"))
        
        for tf in timeframes:
            is_selected = (tf == current_timeframe.get())
            btn_bg = DEEP_SEA_THEME['info'] if is_selected else DEEP_SEA_THEME['surface_bg']
            btn_fg = DEEP_SEA_THEME['text_primary'] if is_selected else DEEP_SEA_THEME['text_secondary']
            
            tf_btn = tk.Button(
                timeframe_frame, 
                text=tf, 
                command=lambda t=tf: change_timeframe(t),
                font=("Segoe UI", 9, "bold"), 
                bg=btn_bg, 
                fg=btn_fg,
                activebackground=DEEP_SEA_THEME['active'] if is_selected else DEEP_SEA_THEME['hover'], 
                activeforeground=DEEP_SEA_THEME['text_primary'], 
                bd=2, 
                highlightthickness=0,
                relief="raised",
                padx=6,
                pady=3
            )
            tf_btn.pack(side=tk.LEFT, padx=1)

        # Footprint chart controls (interval & volume profile toggle)
        # Recreate the UI each redraw but preserve state objects/workers.
        fp_state = getattr(frame, "footprint_settings", None) or {}
        fp_interval_var = fp_state.get("interval_var") or tk.StringVar(value="1m")
        fp_profile_var = fp_state.get("show_profile_var") or tk.BooleanVar(value=True)

        fp_ctrl_frame = tk.Frame(control_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        fp_interval_box = ttk.Combobox(
            fp_ctrl_frame,
            textvariable=fp_interval_var,
            values=["15s", "30s", "1m", "5m"],
            state="readonly",
            width=6
        )
        fp_interval_box.pack(side=tk.LEFT, padx=(0, 6), pady=2)

        fp_profile_toggle = tk.Checkbutton(
            fp_ctrl_frame,
            text="Volume Profile",
            variable=fp_profile_var,
            onvalue=True,
            offvalue=False,
            bg=DEEP_SEA_THEME['secondary_bg'],
            fg=DEEP_SEA_THEME['text_primary'],
            selectcolor=DEEP_SEA_THEME['surface_bg'],
            activebackground=DEEP_SEA_THEME['hover'],
            activeforeground=DEEP_SEA_THEME['text_primary'],
            font=("Segoe UI", 9)
        )
        fp_profile_toggle.pack(side=tk.LEFT, padx=(0, 6))

        frame.footprint_settings = {
            "frame": fp_ctrl_frame,
            "interval_var": fp_interval_var,
            "show_profile_var": fp_profile_var,
            "interval_box": fp_interval_box,
            "profile_toggle": fp_profile_toggle,
            "worker": fp_state.get("worker"),
            "fetch_job": fp_state.get("fetch_job"),
        }

        def on_fp_interval_change(event=None):
            if current_chart_type.get() == "Order Flow Footprint":
                frame.after(10, lambda: draw_chart("Order Flow Footprint"))

        def on_fp_profile_toggle():
            if current_chart_type.get() == "Order Flow Footprint":
                frame.after(10, lambda: draw_chart("Order Flow Footprint"))

        fp_interval_box.bind('<<ComboboxSelected>>', on_fp_interval_change)
        fp_profile_toggle.configure(command=on_fp_profile_toggle)


        # Indicator dropdown button (beside Add Tab)
        def open_indicator_menu(event=None):
            try:
                menu.tk_popup(indicator_btn.winfo_rootx(), indicator_btn.winfo_rooty() + indicator_btn.winfo_height())
            finally:
                menu.grab_release()

        indicator_btn = tk.Button(
            control_frame,
            text="Indicators ▾",
            command=open_indicator_menu,
            font=("Segoe UI", 9, "bold"),
            bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['hover'], activeforeground=DEEP_SEA_THEME['text_primary'],
            bd=2, relief="raised", width=12, height=1
        )
        indicator_btn.pack(side=tk.RIGHT, padx=(4, 2), pady=2)

        menu = tk.Menu(indicator_btn, tearoff=0, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_primary'])
        def toggle_and_redraw(name):
            # Save current indicator state for this symbol and as global default
            try:
                state_now = {k: var.get() for k, var in indicator_vars.items()}
                set_indicator_state(symbol, state_now)
                set_default_indicator_state(state_now)
            except Exception:
                pass
            frame.after(10, lambda: draw_chart(chart_type=current_chart_type.get()))

        for name in ["RSI", "SMA 20", "EMA 50", "Bollinger (20,2)", "VWAP", "MACD"]:
            menu.add_checkbutton(
                label=name,
                onvalue=True,
                offvalue=False,
                variable=indicator_vars[name],
                command=lambda n=name: toggle_and_redraw(n)
            )

        # Indicator bar shows active indicator tags
        indicator_bar = tk.Frame(control_frame, bg=DEEP_SEA_THEME['surface_bg'], width=220, height=32, highlightbackground=DEEP_SEA_THEME['border'], highlightthickness=1)
        indicator_bar.pack(side=tk.RIGHT, padx=(8, 4), pady=2)
        indicator_bar.pack_propagate(False)
        for k, var in indicator_vars.items():
            if var.get():
                tag = tk.Label(
                    indicator_bar,
                    text=k,
                    bg=DEEP_SEA_THEME['accent_bg'], fg=DEEP_SEA_THEME['text_primary'],
                    font=("Segoe UI", 8, "bold"), padx=6, pady=2
                )
                tag.pack(side=tk.LEFT, padx=2)

        # Add Tab button (next to indicator bar)
        def add_tab():
            new_symbol = simpledialog.askstring("Add Tab", "Enter the stock ticker symbol (e.g., MSFT):", parent=frame)
            if not new_symbol:
                return
            new_ticker = yf.Ticker(new_symbol)
            new_hist = new_ticker.history(period="2d")
            if len(new_hist) < 2:
                messagebox.showerror("Error", f"Not enough data for {new_symbol}.")
                return
            new_prev_close = new_hist['Close'].iloc[0]
            new_latest_close = new_hist['Close'].iloc[1]
            new_percent_gain = ((new_latest_close - new_prev_close) / new_prev_close) * 100
            new_ta_signal = get_ta_signal(new_symbol)
            new_rsi = get_rsi(new_symbol)
            new_analysis = get_in_depth_analysis(new_symbol)
            show_chart_with_points(new_symbol, new_ticker, new_prev_close, new_latest_close, new_percent_gain, new_ta_signal, new_rsi, new_analysis, notebook, new_symbol)

        add_tab_btn = tk.Button(control_frame, text="+ Add Tab", command=add_tab, font=("Segoe UI", 9, "bold"),
                               bg="#4CAF50", fg="#E8F5E8", activebackground="#66BB6A", activeforeground="#FFFFFF",
                               bd=2, relief="raised", width=10, height=1)
        add_tab_btn.pack(side=tk.RIGHT, padx=(4, 2), pady=2)

        # Close Tab button (next to Add Tab)
        def close_tab():
            current = notebook.index(notebook.select())
            if current >= 0:
                notebook.forget(current)

        close_tab_btn = tk.Button(control_frame, text="X Close Tab", command=close_tab, font=("Segoe UI", 9, "bold"),
                                 bg="#D32F2F", fg="#FFEBEE", activebackground="#F44336", activeforeground="#FFFFFF",
                                 bd=2, relief="raised", width=10, height=1)
        close_tab_btn.pack(side=tk.RIGHT, padx=(2, 2), pady=2)

        # Control buttons frame
        controls_frame = tk.Frame(control_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Auto-refresh toggle
        auto_refresh_btn = tk.Button(
            controls_frame, 
            text=f"AUTO {'ON' if auto_refresh_enabled.get() else 'OFF'}", 
            command=lambda: toggle_auto_refresh(),
            font=("Segoe UI", 9, "bold"), 
            bg=DEEP_SEA_THEME['success'] if auto_refresh_enabled.get() else DEEP_SEA_THEME['danger'],
            fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'] if auto_refresh_enabled.get() else DEEP_SEA_THEME['danger'], 
            activeforeground=DEEP_SEA_THEME['text_primary'], 
            bd=2, 
            highlightthickness=0,
            relief="raised",
            padx=8,
            pady=3
        )
        auto_refresh_btn.pack(side=tk.RIGHT, padx=2)
        
        # Manual refresh button
        refresh_btn = tk.Button(
            controls_frame, 
            text="REFRESH", 
            command=lambda: frame.after(10, lambda: draw_chart("Candlestick")),
            font=("Segoe UI", 9, "bold"), 
            bg=DEEP_SEA_THEME['info'],
            fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'], 
            activeforeground=DEEP_SEA_THEME['text_primary'], 
            bd=2, 
            highlightthickness=0,
            relief="raised",
            padx=8,
            pady=3
        )
        refresh_btn.pack(side=tk.RIGHT, padx=2)
        
        # Status and info frame
        info_frame = tk.Frame(control_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=15)
        
        # Status label
        status_label = tk.Label(
            info_frame,
            text=f"Loading {current_timeframe.get().upper()} data...",
            font=("Segoe UI", 9),
            bg=DEEP_SEA_THEME['secondary_bg'],
            fg=DEEP_SEA_THEME['text_secondary']
        )
        status_label.pack(side=tk.LEFT)
        
        def toggle_auto_refresh():
            auto_refresh_enabled.set(not auto_refresh_enabled.get())
            draw_chart("Candlestick")  # Redraw to update button state
        
        try:
            chart_hist, price_bins, volume_profile, high_vol_price, low_vol_price, data_note = fetch_chart_data()
            tf = current_timeframe.get()
            has_prepost = tf in ["1m", "5m", "10m", "15m", "20m", "30m", "1h"]
            status_text = f"OK {tf.upper()} - {len(chart_hist)} bars"
            if data_note:
                status_text += f" ({data_note})"
            if has_prepost:
                status_text += " (Extended Hours)"
        except Exception as e:
            error_label = tk.Label(frame, text=f"Error loading chart: {e}", bg=DEEP_SEA_THEME['primary_bg'], fg=DEEP_SEA_THEME['danger'], font=("Segoe UI", 12))
            error_label.pack(expand=1)
            return
            
        # Handle different chart types
        if chart_type == "Order Flow Footprint":
            fp_state = frame.footprint_settings
            fp_ctrl = fp_state["frame"]
            if not fp_ctrl.winfo_ismapped():
                fp_ctrl.pack(side=tk.LEFT, padx=(12, 8), pady=2)

            interval_choice = fp_state["interval_var"].get()
            show_profile_var = fp_state["show_profile_var"]

            chart_container = tk.Frame(frame, bg=DEEP_SEA_THEME['primary_bg'])
            chart_container.pack(fill=tk.BOTH, expand=1)

            def update_canvas(fig):
                if not chart_container.winfo_exists():
                    plt.close(fig)
                    return
                for child in chart_container.winfo_children():
                    child.destroy()
                canvas = FigureCanvasTkAgg(fig, master=chart_container)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
                fp_state['canvas'] = canvas
                plt.close(fig)

            ticks_df, inferred_tick = fetch_symbol_ticks(symbol, interval_choice)
            tick_size = inferred_tick if inferred_tick > 0 else 0.01

            if ticks_df.empty:
                msg = tk.Label(
                    chart_container,
                    text="No footprint data available",
                    bg=DEEP_SEA_THEME['primary_bg'],
                    fg=DEEP_SEA_THEME['danger'],
                    font=("Segoe UI", 12)
                )
                msg.pack(expand=1)
                status_label.config(text=f"Footprint unavailable for {interval_choice}")
            else:
                initial_fig = footprint_chart.render_footprint(
                    ticks_df,
                    tick_size=tick_size,
                    interval=interval_choice,
                    show_volume_profile=show_profile_var.get(),
                )
                update_canvas(initial_fig)
                note_suffix = f" ({data_note})" if data_note else ""
                status_label.config(text=f"Footprint {interval_choice.upper()} ready{note_suffix}")

            worker = fp_state.get("worker")
            if worker is None:
                worker = footprint_chart.FootprintRenderWorker(
                    tick_size=tick_size,
                    interval=interval_choice,
                    show_profile_fn=lambda: fp_state["show_profile_var"].get(),
                    update_callback=lambda fig: frame.after(0, lambda f=fig: update_canvas(f)),
                    batch_ms=400,
                    max_ticks=12000,
                    max_bars=60,
                )
                fp_state["worker"] = worker
                worker.start()
            else:
                worker.set_tick_size(tick_size)
                worker.set_interval(interval_choice)
                worker.set_max_bars(60)

            if not ticks_df.empty:
                worker.submit(ticks_df)

            if fp_state.get("fetch_job"):
                try:
                    frame.after_cancel(fp_state["fetch_job"])
                except Exception:
                    pass
                fp_state["fetch_job"] = None

            def schedule_tick_refresh():
                if not chart_container.winfo_exists():
                    return
                worker_ref = fp_state.get("worker")
                if worker_ref is None:
                    return
                fresh_ticks, _ = fetch_symbol_ticks(symbol, fp_state["interval_var"].get())
                if not fresh_ticks.empty:
                    worker_ref.submit(fresh_ticks)
                fp_state["fetch_job"] = frame.after(5000, schedule_tick_refresh)

            fp_state["fetch_job"] = frame.after(500, schedule_tick_refresh)
            return

        if chart_type == "Volume Profile Plus":
            try:
                plt.close('all')
                fig, axes, stats = overflow_chart.plot_volume_profile_plus(
                    chart_hist,
                    price_col="Close",
                    volume_col="Volume",
                    open_col="Open",
                    high_col="High",
                    low_col="Low",
                    bins=40,
                    title=f"{symbol} - Volume Profile Plus ({current_timeframe.get().upper()})",
                    figsize=(width, height)
                )

                fig.patch.set_facecolor(DEEP_SEA_THEME['primary_bg'])
                for ax in axes:
                    ax.set_facecolor(DEEP_SEA_THEME['primary_bg'])
                    ax.tick_params(colors=DEEP_SEA_THEME['text_primary'])
                    ax.xaxis.label.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.yaxis.label.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.title.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.grid(True, color=DEEP_SEA_THEME['border'], linestyle='--', linewidth=0.3, alpha=0.4)

                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

                total_buy = stats.get('total_buy_volume', 0.0)
                total_sell = stats.get('total_sell_volume', 0.0)
                dominance = stats.get('buy_sell_ratio', 0.0)
                status_label.config(
                    text=(
                        f"{status_text} | Buy {total_buy:,.0f} vs Sell {total_sell:,.0f} "
                        f"({dominance:+.1f}% net)"
                    )
                )

            except Exception as e:
                print(f"Error creating volume profile plus chart: {e}")
                error_label = tk.Label(
                    frame,
                    text=f"Error rendering Volume Profile Plus: {e}",
                    bg=DEEP_SEA_THEME['primary_bg'],
                    fg=DEEP_SEA_THEME['danger'],
                    font=("Segoe UI", 12)
                )
                error_label.pack(expand=1)
            return

        if chart_type == "Volume Profile - Order Book":
            # Render volume profile chart
            try:
                # Create volume profile chart using the overflow_chart module
                plt.close('all')
                fig, axes, stats = overflow_chart.plot_volume_profile_chart(
                    chart_hist,
                    price_col="Close",
                    volume_col="Volume",
                    bins=40,  # Good balance between detail and performance
                    title=f"{symbol} - Volume Profile & Order Book Analysis ({current_timeframe.get().upper()})",
                    figsize=(width, height),
                    show_poc=True,
                    show_vah_val=True
                )
                
                # Apply Deep Sea theme
                fig.patch.set_facecolor(DEEP_SEA_THEME['primary_bg'])
                for ax in axes:
                    ax.set_facecolor(DEEP_SEA_THEME['primary_bg'])
                    ax.tick_params(colors=DEEP_SEA_THEME['text_primary'])
                    ax.xaxis.label.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.yaxis.label.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.title.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.grid(True, color=DEEP_SEA_THEME['border'], linestyle='-', linewidth=0.3)
                
                # Update legends
                for ax in axes:
                    legend = ax.get_legend()
                    if legend:
                        legend.get_frame().set_facecolor(DEEP_SEA_THEME['secondary_bg'])
                        for text in legend.get_texts():
                            text.set_color(DEEP_SEA_THEME['text_primary'])
                
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
                
                print(f"Volume Profile chart rendered for {symbol}")
                print(f"  POC: ${stats['poc_price']:.2f}")
                print(f"  Value Area: ${stats['val_price']:.2f} - ${stats['vah_price']:.2f}")
                
            except Exception as e:
                print(f"Error creating volume profile chart: {e}")
                # Fallback to regular candlestick chart
                draw_chart("Candlestick")
                return
                
        elif chart_type.startswith("Overflow"):
            # Render overflow chart
            try:
                # Prepare data for overflow chart
                overflow_df = pd.DataFrame({
                    'time': chart_hist['Date'],
                    'value': chart_hist['Close']  # Default to closing prices
                })
                
                # Determine threshold based on chart type
                ylabel = "Value"
                if "RSI Overbought" in chart_type:
                    # Calculate RSI and use overbought threshold
                    if 'RSI' in chart_hist.columns:
                        overflow_df['value'] = chart_hist['RSI']
                    threshold = 70
                elif "RSI Oversold" in chart_type:
                    # Calculate RSI and use oversold threshold  
                    if 'RSI' in chart_hist.columns:
                        overflow_df['value'] = chart_hist['RSI']
                    threshold = 30
                elif "Volume Spike" in chart_type:
                    overflow_df['value'] = chart_hist['Volume']
                    threshold = chart_hist['Volume'].quantile(0.8)
                elif "Price Breakout" in chart_type:
                    threshold = chart_hist['Close'].quantile(0.9)
                elif "Momentum" in chart_type:
                    momentum = chart_hist['Close'].pct_change().fillna(0)
                    momentum = momentum.rolling(3, min_periods=1).mean() * 100
                    overflow_df['value'] = momentum
                    threshold = momentum.quantile(0.85)
                    ylabel = "Momentum (%)"
                else:
                    threshold = chart_hist['Close'].mean()
                
                # Remove any NaN values
                overflow_df = overflow_df.dropna()
                
                if len(overflow_df) > 0:
                    # Create overflow chart using the overflow_chart module
                    plt.close('all')
                    fig, ax, stats = overflow_chart.plot_overflow_matplotlib(
                        overflow_df,
                        x="time",
                        y="value",
                        threshold=threshold,
                        title=f"{symbol} - {chart_type} ({current_timeframe.get().upper()})",
                        ylabel=ylabel,
                        xlabel="Date",
                        figsize=(width, height),
                        use_seaborn_theme=False
                    )
                    
                    # Apply Deep Sea theme
                    fig.patch.set_facecolor(DEEP_SEA_THEME['primary_bg'])
                    ax.set_facecolor(DEEP_SEA_THEME['primary_bg'])
                    ax.tick_params(colors=DEEP_SEA_THEME['text_primary'])
                    ax.xaxis.label.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.yaxis.label.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.title.set_color(DEEP_SEA_THEME['text_primary'])
                    ax.grid(True, color=DEEP_SEA_THEME['border'], linestyle='-', linewidth=0.3)
                    
                    # Create chart container
                    chart_container = tk.Frame(frame, bg=DEEP_SEA_THEME['primary_bg'])
                    chart_container.pack(fill=tk.BOTH, expand=1, padx=2, pady=2)
                    
                    # Create canvas
                    canvas = FigureCanvasTkAgg(fig, master=chart_container)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
                    
                    print(f"Overflow chart rendered: {chart_type}")
                    print(f"  Data points: {len(overflow_df)}")
                    print(f"  Threshold: {threshold:.2f}")
                    print(f"  Points above threshold: {stats['count_over']} ({stats['pct_over']:.1f}%)")
                    
                    # Set up auto-refresh for overflow charts
                    if auto_refresh_enabled.get():
                        tf = current_timeframe.get()
                        if tf in ["1m"]:
                            interval = 30000
                        elif tf in ["5m", "10m"]:
                            interval = 60000
                        else:
                            interval = 300000
                        refresh_timer = frame.after(interval, lambda: draw_chart(chart_type))
                    
                    suffix = f" ({data_note})" if data_note else ""
                    status_label.config(text=f"OK {chart_type} loaded successfully{suffix}")
                    return
                else:
                    print("Warning: No data available for overflow chart, falling back to candlestick")
                    
            except Exception as e:
                print(f"Error creating overflow chart: {e}")
                print("Falling back to candlestick chart")
        
        # Create matplotlib figure with enhanced interactivity (for candlestick charts)
        plt.close('all')  # Close any existing figures
        fig = plt.figure(figsize=(width, height), facecolor=DEEP_SEA_THEME['primary_bg'])

        # Determine which subplots are needed
        rsi_enabled = bool(indicator_vars.get("RSI") and indicator_vars["RSI"].get())
        macd_enabled = bool(indicator_vars.get("MACD") and indicator_vars["MACD"].get())

        # Dynamic grid specification based on selected indicators
        if rsi_enabled and macd_enabled:
            height_ratios = [12, 0.6, 2.7, 2.7]
        elif rsi_enabled or macd_enabled:
            height_ratios = [12, 0.6, 3.6, 0.3]
        else:
            height_ratios = [12, 0.6, 0.3, 0.3]

        gs = fig.add_gridspec(
            4, 2,
            width_ratios=[8, 1],
            height_ratios=height_ratios,
            wspace=0.02,
            hspace=0.05
        )

        # Main price chart
        ax1 = fig.add_subplot(gs[0, 0])
        # Bottom indicator slots (we'll assign RSI/MACD as needed)
        ax_bottom1 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax_bottom2 = fig.add_subplot(gs[3, 0], sharex=ax1)
        # Volume profile at right
        ax_vp = fig.add_subplot(gs[0:3, 1], sharey=ax1)
        
        # Enable interactive navigation
        ax1.set_navigate(True)
        ax_bottom1.set_navigate(True)
        
        # Chart container with scrollable canvas
        chart_container = tk.Frame(frame, bg=DEEP_SEA_THEME['primary_bg'])
        chart_container.pack(fill=tk.BOTH, expand=1, padx=2, pady=2)

        width_candle = 0.6

        # Draw candlesticks with optimized rendering
        prepost_count = 0
        regular_count = 0
        
        try:
            # Pre-allocate arrays for better performance
            dates = chart_hist['Date'].values
            opens = chart_hist['Open'].values
            closes = chart_hist['Close'].values
            highs = chart_hist['High'].values
            lows = chart_hist['Low'].values
            # Handles for live-updating last candle
            last_body_patch = None
            last_wick_line = None
            last_open_val = None
            last_low_val = None
            last_high_val = None
            last_date_val = None
            
            print(f"Drawing candlesticks: {len(dates)} candles")
            print(f"Price range: {lows.min():.2f} - {highs.max():.2f}")
            print(f"Date range: {dates.min():.1f} - {dates.max():.1f} (matplotlib dates)")
            
            if len(dates) == 0:
                print("ERROR: No data to draw!")
                return
            
            # Ensure we have valid price data
            if np.isnan(lows.min()) or np.isnan(highs.max()):
                print("ERROR: NaN values in price data!")
                return
            
            # Simplified pre/post market detection
            if current_timeframe.get() in ["1m", "5m", "10m", "15m", "30m", "1h"]:
                # Vectorized time-based detection for better performance
                times = pd.to_datetime(chart_hist['Date'], unit='D', origin='1970-01-01')
                hours = times.dt.hour
                minutes = times.dt.minute
                # Market hours: 9:30 AM - 4:00 PM ET
                is_regular_hours = ((hours > 9) | ((hours == 9) & (minutes >= 30))) & (hours < 16)
                # Batch draw candlesticks
                for i, (date_val, open_val, close_val, high_val, low_val, is_regular) in enumerate(
                    zip(dates, opens, closes, highs, lows, is_regular_hours)
                ):
                    if is_regular:
                        color = candle_up if close_val >= open_val else candle_down
                        edgecolor = '#37474F'
                        alpha = 1.0
                        linewidth = 0.5
                        regular_count += 1
                    else:
                        color = '#8BC34A' if close_val >= open_val else '#FF5722'
                        edgecolor = '#FFC107'
                        alpha = 0.9
                        linewidth = 1.0
                        prepost_count += 1
                    # Body
                    body_patch = Rectangle(
                        (date_val - width_candle/2, min(open_val, close_val)),
                        width_candle,
                        abs(close_val - open_val),
                        color=color,
                        ec=edgecolor,
                        linewidth=linewidth,
                        alpha=alpha
                    )
                    ax1.add_patch(body_patch)
                    # Wick
                    wick_line, = ax1.plot([date_val, date_val], [low_val, high_val], color=color, linewidth=1.0, alpha=alpha)
                    # Track last candle handles
                    if i == len(dates) - 1:
                        last_body_patch = body_patch
                        last_wick_line = wick_line
                        last_open_val = float(open_val)
                        last_low_val = float(low_val)
                        last_high_val = float(high_val)
                        last_date_val = float(date_val)
            else:
                # Regular candlesticks for daily timeframe
                for i, (date_val, open_val, close_val, high_val, low_val) in enumerate(
                    zip(dates, opens, closes, highs, lows)
                ):
                    color = candle_up if close_val >= open_val else candle_down
                    regular_count += 1
                    body_patch = Rectangle(
                        (date_val - width_candle/2, min(open_val, close_val)),
                        width_candle,
                        abs(close_val - open_val),
                        color=color,
                        ec='#37474F',
                        linewidth=0.5
                    )
                    ax1.add_patch(body_patch)
                    wick_line, = ax1.plot([date_val, date_val], [low_val, high_val], color=color, linewidth=1.0)
                    if i == len(dates) - 1:
                        last_body_patch = body_patch
                        last_wick_line = wick_line
                        last_open_val = float(open_val)
                        last_low_val = float(low_val)
                        last_high_val = float(high_val)
                        last_date_val = float(date_val)
        
        except Exception as candlestick_error:
            print(f"Error drawing candlesticks: {candlestick_error}")
            # Draw a simple line chart as fallback
            if len(chart_hist) > 0:
                ax1.plot(chart_hist['Date'], chart_hist['Close'], color=candle_up, linewidth=2, label='Close Price')
                regular_count = len(chart_hist)
        
        # Debug output
        print(f"Drew {prepost_count} pre/post market candles and {regular_count} regular market candles")

        # Set chart face colors and styling
        ax1.set_facecolor(DEEP_SEA_THEME['primary_bg'])
        ax_bottom1.set_facecolor(DEEP_SEA_THEME['primary_bg'])
        ax_bottom2.set_facecolor(DEEP_SEA_THEME['primary_bg'])
        ax_vp.set_facecolor(DEEP_SEA_THEME['primary_bg'])

        # Set proper axis limits to ensure candlesticks are visible
        if len(chart_hist) > 0:
            x_min = chart_hist['Date'].min() - 0.5
            x_max = chart_hist['Date'].max() + 0.5
            y_min = chart_hist['Low'].min() * 0.995  # Small padding
            y_max = chart_hist['High'].max() * 1.005
            
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            
            print(f"Chart limits: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.2f}, {y_max:.2f})")

        # Add simplified technical indicators
        if 'RSI' in chart_hist.columns and not chart_hist['RSI'].empty:
            ax1.plot([], [], color=bb_color, linestyle='-', linewidth=1, label='RSI Available')

        ax1.xaxis_date()
        # Simplified date formatting
        tf = current_timeframe.get()
        if tf in ['1m', '5m', '10m', '15m', '30m']:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif tf in ['1h', '2h', '3h', '4h', '6h']:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        else:  # 1d
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        fig.autofmt_xdate()
        ax1.set_title(f"{symbol} - {current_timeframe.get().upper()}", color='#fff', fontsize=14)
        ax1.set_ylabel("Price", color='#fff')
        ax1.tick_params(axis='x', colors='#aaa')
        ax1.tick_params(axis='y', colors='#aaa')
        ax1.grid(True, color=grid_color, linestyle='-', linewidth=0.3)

        # Simplified annotations
        if len(chart_hist) >= 2:
            latest_price = chart_hist['Close'].iloc[-1]
            prev_price = chart_hist['Close'].iloc[-2]
            percent_change = ((latest_price - prev_price) / prev_price) * 100
            
            ax1.scatter(chart_hist['Date'].iloc[-1], latest_price, color='#ff1744', s=30, zorder=5)
            ax1.text(0.02, 0.98, f"Latest: ${latest_price:.2f}\nChange: {percent_change:+.2f}%",
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle="round", fc="#111", alpha=0.8), color='#fff')

        # Simplified volume profile lines
        if len(volume_profile) > 0:
            ax1.axhline(high_vol_price, color='#ffd600', linestyle='--', linewidth=1, alpha=0.7)
            ax1.axhline(low_vol_price, color='#00e5ff', linestyle='--', linewidth=1, alpha=0.7)
        
        # --- Technical indicator overlays on price chart ---
        try:
            if indicator_vars.get("SMA 20") and indicator_vars["SMA 20"].get():
                chart_hist["SMA20"] = pd.Series(chart_hist['Close']).rolling(20, min_periods=1).mean()
                ax1.plot(chart_hist['Date'], chart_hist['SMA20'], color="#00BCD4", linewidth=1.4, label="SMA 20")
            if indicator_vars.get("EMA 50") and indicator_vars["EMA 50"].get():
                chart_hist["EMA50"] = pd.Series(chart_hist['Close']).ewm(span=50, adjust=False).mean()
                ax1.plot(chart_hist['Date'], chart_hist['EMA50'], color="#FFC107", linewidth=1.4, label="EMA 50")
            if indicator_vars.get("Bollinger (20,2)") and indicator_vars["Bollinger (20,2)"].get():
                mid = pd.Series(chart_hist['Close']).rolling(20, min_periods=20).mean()
                std = pd.Series(chart_hist['Close']).rolling(20, min_periods=20).std()
                upper = mid + 2 * std
                lower = mid - 2 * std
                ax1.plot(chart_hist['Date'], upper, color=bb_color, linewidth=1.0, linestyle='--', label='BB Upper')
                ax1.plot(chart_hist['Date'], lower, color=bb_color, linewidth=1.0, linestyle='--', label='BB Lower')
                ax1.fill_between(chart_hist['Date'], lower, upper, color=bb_color, alpha=0.08)
            if indicator_vars.get("VWAP") and indicator_vars["VWAP"].get():
                tp = (chart_hist['High'] + chart_hist['Low'] + chart_hist['Close']) / 3.0
                vol = chart_hist['Volume'].replace(0, np.nan)
                vwap = (tp * vol).cumsum() / vol.cumsum()
                ax1.plot(chart_hist['Date'], vwap, color="#9C27B0", linewidth=1.2, label='VWAP')
        except Exception as ind_err:
            print(f"Indicator overlay error: {ind_err}")

        # --- Subplot assignment for RSI/MACD ---
        rsi_enabled = bool(indicator_vars.get("RSI") and indicator_vars["RSI"].get())
        macd_enabled = bool(indicator_vars.get("MACD") and indicator_vars["MACD"].get())
        ax_rsi = None
        ax_macd = None
        if rsi_enabled and macd_enabled:
            ax_rsi = ax_bottom1
            ax_macd = ax_bottom2
        elif rsi_enabled and not macd_enabled:
            ax_rsi = ax_bottom1
            ax_bottom2.set_visible(False)
        elif macd_enabled and not rsi_enabled:
            ax_macd = ax_bottom1
            ax_bottom2.set_visible(False)
        else:
            # Neither enabled
            ax_bottom1.set_visible(False)
            ax_bottom2.set_visible(False)

        # RSI subplot (only if selected and data exists)
        if ax_rsi is not None and ('RSI' in chart_hist.columns and not chart_hist['RSI'].isna().all()):
            ax_rsi.plot(chart_hist['Date'], chart_hist['RSI'], color=rsi_color, linewidth=1.5)
            ax_rsi.axhline(70, color=overbought_color, linestyle='--', linewidth=1)
            ax_rsi.axhline(30, color=oversold_color, linestyle='--', linewidth=1)
            ax_rsi.set_ylabel("RSI", color='#fff')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_title("RSI", color='#fff', fontsize=12)
            ax_rsi.tick_params(axis='x', colors='#aaa')
            ax_rsi.tick_params(axis='y', colors='#aaa')
            ax_rsi.grid(True, color=grid_color, linestyle='-', linewidth=0.3)

        # MACD subplot (only if selected)
        if ax_macd is not None:
            try:
                close_series = pd.Series(chart_hist['Close'])
                ema12 = close_series.ewm(span=12, adjust=False).mean()
                ema26 = close_series.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                hist = macd_line - signal_line
                # Plot histogram bars
                colors = np.where(hist >= 0, '#26A69A', '#EF5350')
                ax_macd.bar(chart_hist['Date'], hist, color=colors, alpha=0.6, width=0.6)
                # Plot MACD and Signal lines
                ax_macd.plot(chart_hist['Date'], macd_line, color='#42A5F5', linewidth=1.3, label='MACD')
                ax_macd.plot(chart_hist['Date'], signal_line, color='#FF7043', linewidth=1.1, label='Signal')
                ax_macd.set_ylabel("MACD", color='#fff')
                ax_macd.set_title("MACD", color='#fff', fontsize=12)
                ax_macd.tick_params(axis='x', colors='#aaa')
                ax_macd.tick_params(axis='y', colors='#aaa')
                ax_macd.grid(True, color=grid_color, linestyle='-', linewidth=0.3)
            except Exception as macd_err:
                print(f"MACD error: {macd_err}")

        # Simplified volume profile
        if len(volume_profile) > 0:
            ax_vp.barh(
                (price_bins[:-1] + price_bins[1:]) / 2,
                volume_profile,
                height=(price_bins[1] - price_bins[0]) * 0.9,
                color=volume_profile_color,
                alpha=0.6
            )
            ax_vp.set_xlabel("Vol", color='#fff', fontsize=10)
            ax_vp.invert_xaxis()
            ax_vp.tick_params(axis='x', colors='#aaa', labelsize=8)
            ax_vp.tick_params(axis='y', colors='#aaa', labelsize=8)
            ax_vp.yaxis.tick_right()
        else:
            ax_vp.set_visible(False)

        # Apply tight layout for better spacing
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=chart_container)
        canvas.draw()
        
        # Simplified toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, chart_container)
        toolbar.update()
        toolbar.config(bg=DEEP_SEA_THEME['secondary_bg'])
        
        # Pack canvas and toolbar
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        
        # Simplified mouse interaction
        def on_scroll(event):
            if event.inaxes == ax1 and event.xdata and event.ydata:
                scale = 0.9 if event.button == 'up' else 1.1
                xlim = ax1.get_xlim()
                ylim = ax1.get_ylim()
                
                new_width = (xlim[1] - xlim[0]) * scale
                new_height = (ylim[1] - ylim[0]) * scale
                
                relx = (event.xdata - xlim[0]) / (xlim[1] - xlim[0])
                rely = (event.ydata - ylim[0]) / (ylim[1] - ylim[0])
                
                ax1.set_xlim([event.xdata - new_width * relx, event.xdata + new_width * (1 - relx)])
                ax1.set_ylim([event.ydata - new_height * rely, event.ydata + new_height * (1 - rely)])
                canvas.draw_idle()
        
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        # --- 5s lightweight updater: move only the newest candle to latest price ---
        def fetch_last_trade_price():
            try:
                # Use 1m interval for freshest trade price
                recent = ticker.history(period="1d", interval="1m", prepost=True, repair=True)
                if recent is not None and not recent.empty:
                    return float(recent['Close'].iloc[-1])
            except Exception as _e:
                pass
            return None

        def update_last_candle():
            nonlocal last_tick_timer, last_low_val, last_high_val
            if not frame.winfo_exists() or last_body_patch is None or last_wick_line is None or last_open_val is None:
                return
            new_px = fetch_last_trade_price()
            if new_px is None:
                # Try again later
                last_tick_timer = frame.after(5000, update_last_candle)
                return
            # Update body geometry
            top = max(new_px, last_open_val)
            bottom = min(new_px, last_open_val)
            last_body_patch.set_y(bottom)
            last_body_patch.set_height(top - bottom)
            # Update color based on up/down
            is_up = new_px >= last_open_val
            last_body_patch.set_facecolor(candle_up if is_up else candle_down)
            # Update wick extents to include new price
            last_low_val = min(last_low_val, new_px)
            last_high_val = max(last_high_val, new_px)
            last_wick_line.set_data([last_date_val, last_date_val], [last_low_val, last_high_val])
            # Adjust y-limits if price moved outside
            y0, y1 = ax1.get_ylim()
            expanded = False
            if last_high_val > y1:
                y1 = last_high_val * 1.005
                expanded = True
            if last_low_val < y0:
                y0 = last_low_val * 0.995
                expanded = True
            if expanded:
                ax1.set_ylim(y0, y1)
            # Redraw only this figure
            canvas.draw_idle()
            # Reschedule
            last_tick_timer = frame.after(5000, update_last_candle)

        # Start the 5-second updater only for candlestick chart
        last_tick_timer = frame.after(5000, update_last_candle)
        
        # Optimized auto-refresh with longer intervals
        if auto_refresh_enabled.get():
            tf = current_timeframe.get()
            # Longer refresh intervals for better performance
            if tf in ["1m"]:
                interval = 30000  # 30 seconds for 1m
            elif tf in ["5m", "10m"]:
                interval = 60000  # 1 minute for 5m/10m
            elif tf in ["15m", "30m"]:
                interval = 120000  # 2 minutes for 15m/30m
            else:
                interval = 300000  # 5 minutes for longer timeframes
            
            refresh_timer = frame.after(interval, draw_chart)
            status_label.config(text=f"{status_text} (Refresh: {interval//1000}s)")
        else:
            status_label.config(text=status_text)

    # Show chart by default
    draw_chart("Candlestick")

    # Add the tab to the notebook
    notebook.add(frame, text=tab_title)

WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    try:
        path = get_user_data_path("watchlist.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"Loaded watchlist with {len(data)} items: {data}")
                    return data
        print("No existing watchlist found, creating new one")
    except Exception as e:
        print(f"Error loading watchlist: {e}")
    return []

def save_watchlist(watchlist):
    try:
        path = get_user_data_path("watchlist.json")
        with open(path, "w") as f:
            json.dump(watchlist, f, indent=2)
        print(f"Saved watchlist with {len(watchlist)} items to {path}")
        return True
    except Exception as e:
        print(f"Error saving watchlist: {e}")
        return False

# --- Indicator preferences persistence ---
INDICATOR_PREFS_FILENAME = "indicator_prefs.json"

def _load_indicator_prefs():
    try:
        path = get_user_data_path(INDICATOR_PREFS_FILENAME)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}

def _save_indicator_prefs(prefs: dict):
    try:
        path = get_user_data_path(INDICATOR_PREFS_FILENAME)
        with open(path, "w") as f:
            json.dump(prefs, f, indent=2)
        return True
    except Exception:
        return False

def get_indicator_state(symbol: str) -> dict:
    """Return saved indicator booleans for a symbol, or defaults if none."""
    prefs = _load_indicator_prefs()
    sym = (symbol or "").upper()
    state = prefs.get(sym) or prefs.get("_default")
    if not isinstance(state, dict):
        state = {}
    # Defaults
    base = {
        "RSI": True,
        "SMA 20": False,
        "EMA 50": False,
        "Bollinger (20,2)": False,
        "VWAP": False,
        "MACD": False,
    }
    base.update({k: bool(v) for k, v in state.items() if k in base})
    return base

def set_indicator_state(symbol: str, state: dict):
    """Save indicator booleans for a symbol."""
    prefs = _load_indicator_prefs()
    sym = (symbol or "").upper()
    prefs[sym] = {k: bool(v) for k, v in state.items()}
    _save_indicator_prefs(prefs)

def set_default_indicator_state(state: dict):
    """Save indicator booleans as the global default for all symbols."""
    prefs = _load_indicator_prefs()
    prefs["_default"] = {k: bool(v) for k, v in state.items()}
    _save_indicator_prefs(prefs)

# --- Persistent Quick Tabs (Home) --- per-user
DEFAULT_CUSTOM_TABS = ["", "", "", "", ""]

def load_custom_tabs():
    try:
        path = get_user_data_path("custom_tabs.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) == 5:
                    return [str(x).upper() for x in data]
    except Exception:
        pass
    return DEFAULT_CUSTOM_TABS.copy()

def save_custom_tabs(tabs):
    try:
        path = get_user_data_path("custom_tabs.json")
        with open(path, "w") as f:
            json.dump([str(x).upper() for x in tabs], f, indent=2)
    except Exception:
        pass

# --- Simple local account system ---
USERS_FILE = "users.json"
REMEMBER_ME_FILE = "remember_me.dat"
current_user = None  # Global variable to track logged-in user

def get_user_data_path(filename):
    """Get the path to a user-specific data file"""
    global current_user
    if not current_user:
        raise ValueError("No user logged in")
    
    # Create user data directory if it doesn't exist
    user_dir = f"user_data_{current_user}"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    return os.path.join(user_dir, filename)

def _load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}

def _save_users(data):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _hash_password(password: str, salt_b64: str | None = None, iterations: int = 200000):
    try:
        if salt_b64 is None:
            salt = os.urandom(16)
            salt_b64 = base64.b64encode(salt).decode("utf-8")
        else:
            salt = base64.b64decode(salt_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        hash_b64 = base64.b64encode(dk).decode("utf-8")
        return salt_b64, hash_b64, iterations
    except Exception:
        # Fallback (not recommended) - very unlikely to hit
        return "", "", iterations

def _verify_password(password: str, salt_b64: str, hash_b64: str, iterations: int = 200000) -> bool:
    try:
        salt = base64.b64decode(salt_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(base64.b64encode(dk).decode("utf-8"), hash_b64)
    except Exception:
        return False

def _get_device_key():
    """Generate a device-specific key for encrypting saved credentials"""
    try:
        import platform
        import socket
        
        # Create a device fingerprint from hostname, platform, and user
        device_info = f"{platform.node()}-{platform.system()}-{platform.machine()}-{os.getlogin()}"
        device_hash = hashlib.sha256(device_info.encode()).digest()
        return device_hash[:32]  # Use first 32 bytes as AES key
    except Exception:
        # Fallback to a static key (less secure but functional)
        return hashlib.sha256(b"aley_trader_device_key").digest()[:32]

def _encrypt_credentials(username, password):
    """Encrypt username and password for remember me functionality"""
    try:
        from cryptography.fernet import Fernet
        
        # Generate key from device fingerprint
        device_key = _get_device_key()
        fernet_key = base64.urlsafe_b64encode(device_key)
        fernet = Fernet(fernet_key)
        
        # Combine username and password with a separator
        credentials = f"{username}|{password}"
        encrypted = fernet.encrypt(credentials.encode())
        return base64.b64encode(encrypted).decode()
    except ImportError:
        logging.warning("cryptography not installed; remember-me disabled for security")
        return None
    except Exception as e:
        logging.warning(f"Encryption error: {e}")
        return None

def _decrypt_credentials(encrypted_data):
    """Decrypt saved credentials"""
    try:
        from cryptography.fernet import Fernet
        
        # Generate key from device fingerprint
        device_key = _get_device_key()
        fernet_key = base64.urlsafe_b64encode(device_key)
        fernet = Fernet(fernet_key)
        
        # Decrypt the data
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted = fernet.decrypt(encrypted_bytes).decode()
        
        # Split username and password
        parts = decrypted.split("|", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    except ImportError:
        # Fallback from base64 encoding
        try:
            decrypted = base64.b64decode(encrypted_data).decode()
            parts = decrypted.split("|", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
        except Exception:
            pass
    except Exception:
        pass
    return None, None

def save_remember_me_credentials(username, password):
    """Save credentials for remember me functionality"""
    try:
        encrypted = _encrypt_credentials(username, password)
        if encrypted:
            with open(REMEMBER_ME_FILE, "w") as f:
                f.write(encrypted)
            return True
        else:
            logging.warning("Failed to save remember-me credentials (encryption unavailable)")
    except Exception:
        logging.warning("Failed to save remember-me credentials")
    return False

def load_remember_me_credentials():
    """Load saved credentials if remember me was enabled"""
    try:
        if os.path.exists(REMEMBER_ME_FILE):
            with open(REMEMBER_ME_FILE, "r") as f:
                encrypted_data = f.read().strip()
            return _decrypt_credentials(encrypted_data)
    except Exception:
        pass
    return None, None

def clear_remember_me_credentials():
    """Clear saved credentials"""
    try:
        if os.path.exists(REMEMBER_ME_FILE):
            os.remove(REMEMBER_ME_FILE)
    except Exception:
        pass

def check_auto_login():
    """Check if auto-login should be performed"""
    username, password = load_remember_me_credentials()
    if username and password:
        # Verify credentials are still valid
        users = _load_users()
        rec = users.get("users", {}).get(username)
        if rec and _verify_password(password, rec.get("salt", ""), rec.get("hash", ""), rec.get("iterations", 200000)):
            return username
    return None

def require_login():
    # Modal login window; returns username on success, None otherwise
    login = tk.Tk()
    login.title("Aley Trader - Sign In")
    login.configure(bg=DEEP_SEA_THEME['primary_bg'])
    login.geometry("380x300")  # Increased height for remember me checkbox
    login.resizable(False, False)

    pad = 14
    frm = tk.Frame(login, bg=DEEP_SEA_THEME['primary_bg'])
    frm.pack(fill=tk.BOTH, expand=1, padx=pad, pady=pad)

    tk.Label(frm, text="Sign in to continue", bg=DEEP_SEA_THEME['primary_bg'], fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

    row1 = tk.Frame(frm, bg=DEEP_SEA_THEME['primary_bg'])
    row1.pack(fill=tk.X, pady=4)
    tk.Label(row1, text="Username", width=10, anchor="w", bg=DEEP_SEA_THEME['primary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT)
    ent_user = tk.Entry(row1, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_primary'], insertbackground=DEEP_SEA_THEME['text_accent'])
    ent_user.pack(side=tk.LEFT, fill=tk.X, expand=1)

    row2 = tk.Frame(frm, bg=DEEP_SEA_THEME['primary_bg'])
    row2.pack(fill=tk.X, pady=4)
    tk.Label(row2, text="Password", width=10, anchor="w", bg=DEEP_SEA_THEME['primary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT)
    ent_pass = tk.Entry(row2, show="*", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_primary'], insertbackground=DEEP_SEA_THEME['text_accent'])
    ent_pass.pack(side=tk.LEFT, fill=tk.X, expand=1)

    # Remember Me checkbox
    remember_frame = tk.Frame(frm, bg=DEEP_SEA_THEME['primary_bg'])
    remember_frame.pack(fill=tk.X, pady=(8, 4))
    remember_var = tk.BooleanVar(value=False)
    remember_checkbox = tk.Checkbutton(
        remember_frame,
        text="Remember me on this device",
        variable=remember_var,
        bg=DEEP_SEA_THEME['primary_bg'],
        fg=DEEP_SEA_THEME['text_secondary'],
        selectcolor=DEEP_SEA_THEME['secondary_bg'],
        activebackground=DEEP_SEA_THEME['primary_bg'],
        activeforeground=DEEP_SEA_THEME['text_primary'],
        font=("Segoe UI", 10),
        bd=0,
        highlightthickness=0
    )
    remember_checkbox.pack(anchor="w", padx=(95, 0))  # Align with entry fields

    msg_var = tk.StringVar(value="")
    tk.Label(frm, textvariable=msg_var, bg=DEEP_SEA_THEME['primary_bg'], fg=DEEP_SEA_THEME['danger']).pack(anchor="w", pady=(4, 0))

    btns = tk.Frame(frm, bg=DEEP_SEA_THEME['primary_bg'])
    btns.pack(fill=tk.X, pady=(12, 0))

    result = {"user": None}

    def do_sign_in(event=None):
        uname = ent_user.get().strip()
        pwd = ent_pass.get()
        if not uname or not pwd:
            msg_var.set("Enter username and password")
            return
        users = _load_users()
        rec = users.get("users", {}).get(uname)
        if rec and _verify_password(pwd, rec.get("salt", ""), rec.get("hash", ""), rec.get("iterations", 200000)):
            result["user"] = uname
            
            # Handle remember me functionality
            if remember_var.get():
                if save_remember_me_credentials(uname, pwd):
                    print(f"Credentials saved for {uname}")
                else:
                    print("Warning: Failed to save credentials")
            else:
                # Clear any existing saved credentials if user unchecks remember me
                clear_remember_me_credentials()
            
            login.destroy()
        else:
            msg_var.set("Invalid credentials")

    def do_create_account():
        uname = ent_user.get().strip()
        pwd = ent_pass.get()
        if not re.match(r"^[A-Za-z0-9_\-\.]{3,32}$", uname or ""):
            msg_var.set("Username: 3-32 chars (letters, numbers, _-. )")
            return
        if len(pwd) < 6:
            msg_var.set("Password must be 6+ characters")
            return
        users = _load_users()
        users.setdefault("users", {})
        if uname in users["users"]:
            msg_var.set("Username already exists")
            return
        salt_b64, hash_b64, iters = _hash_password(pwd)
        users["users"][uname] = {"salt": salt_b64, "hash": hash_b64, "iterations": iters}
        _save_users(users)
        msg_var.set("Account created. Sign in now.")
        ent_pass.delete(0, tk.END)

    # Load saved credentials if they exist
    saved_username, saved_password = load_remember_me_credentials()
    if saved_username and saved_password:
        ent_user.insert(0, saved_username)
        ent_pass.insert(0, saved_password)
        remember_var.set(True)
        msg_var.set("Credentials loaded from previous session")

    sign_in = tk.Button(btns, text="Sign In", command=do_sign_in, font=("Segoe UI", 10, "bold"),
                        bg=DEEP_SEA_THEME['success'], fg=DEEP_SEA_THEME['text_primary'], activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['primary_bg'], bd=2)
    sign_in.pack(side=tk.RIGHT, padx=4)

    create_btn = tk.Button(btns, text="Create Account", command=do_create_account, font=("Segoe UI", 10, "bold"),
                           bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'], activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['primary_bg'], bd=2)
    create_btn.pack(side=tk.RIGHT, padx=4)

    ent_user.focus_set()
    login.bind('<Return>', do_sign_in)
    try:
        login.mainloop()
    except Exception:
        pass

    return result.get("user")

# --- Main window with tabs ---
def main_tabbed_chart():
    from tkinter import ttk

    root = tk.Tk()
    root.title("Aley Trader")
    root.configure(bg=DEEP_SEA_THEME['primary_bg'])
    root.geometry("1200x800")  # Set initial window size

    # --- Main content frame (for home or charting) ---
    main_content = tk.Frame(root, bg=DEEP_SEA_THEME['primary_bg'])
    main_content.pack(fill=tk.BOTH, expand=1)

    def clear_main_content():
        for widget in main_content.winfo_children():
            widget.destroy()

    def spawn_new_instance():
        """Launch a completely new Aley Trader window as a separate process.
        Uses the same Python executable and this script's file path.
        Falls back to a Toplevel clone if subprocess fails.
        """
        try:
            script_path = os.path.abspath(__file__)
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            print(f"[WARN] Failed to spawn new process: {e}. Falling back to Toplevel window.")
            try:
                # Minimal fallback: open an extra root-level like window via Toplevel
                fallback = tk.Toplevel(root)
                fallback.title("Aley Trader — Extra Window (Fallback)")
                tk.Label(fallback, text="Fallback window (multi-instance)", bg=DEEP_SEA_THEME['primary_bg'], fg=DEEP_SEA_THEME['text_primary']).pack(padx=20, pady=20)
            except Exception as inner:
                print(f"[ERROR] Fallback Toplevel failed: {inner}")

    def make_card(parent, title, subtitle="", width=340, height=150):
        card_outer = tk.Frame(parent, bg=DEEP_SEA_THEME['primary_bg'])
        shadow = tk.Frame(card_outer, bg=DEEP_SEA_THEME['shadow'])
        card = tk.Frame(card_outer, bg=DEEP_SEA_THEME['card_bg'], bd=1, relief="solid", highlightthickness=0)
        def _on_resize(e):
            w = max(160, e.width)
            h = max(100, e.height)
            shadow.place(x=6, y=8, width=w, height=h)
            card.place(x=0, y=0, width=w-6, height=h-8)
        card_outer.bind("<Configure>", _on_resize)
        tk.Label(card, text=title, font=("Segoe UI", 13, "bold"), fg=DEEP_SEA_THEME['text_primary'], bg=DEEP_SEA_THEME['card_bg']).pack(anchor="w", padx=16, pady=(12, 2))
        if subtitle:
            tk.Label(card, text=subtitle, font=("Segoe UI", 10), fg=DEEP_SEA_THEME['text_secondary'], bg=DEEP_SEA_THEME['card_bg']).pack(anchor="w", padx=16)
        return card_outer, card

    def create_news_tab(notebook, symbol_hint: str | None = None):
        """Create a 'News' tab inside the main notebook."""
        tab = tk.Frame(notebook, bg=DEEP_SEA_THEME['primary_bg'])
        notebook.add(tab, text="News")
        notebook.select(tab)

        # Top controls
        controls = tk.Frame(tab, bg=DEEP_SEA_THEME['secondary_bg'])
        controls.pack(side=tk.TOP, fill=tk.X)

        tk.Label(controls, text="Filter symbol:", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT, padx=(10, 4), pady=6)
        sym_var = tk.StringVar(value=(symbol_hint or "").upper())
        sym_entry = tk.Entry(controls, textvariable=sym_var, bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'], insertbackground=DEEP_SEA_THEME['text_accent'])
        sym_entry.pack(side=tk.LEFT, padx=(0, 8), pady=6)

        status_var = tk.StringVar(value="Idle")
        tk.Label(controls, textvariable=status_var, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.RIGHT, padx=10)

        auto_var = tk.BooleanVar(value=True)
        def toggle_auto():
            auto_var.set(not auto_var.get())
            auto_btn.config(text=f"Auto: {'ON' if auto_var.get() else 'OFF'}",
                            bg=DEEP_SEA_THEME['success'] if auto_var.get() else DEEP_SEA_THEME['danger'])
        auto_btn = tk.Button(controls, text="Auto: ON", command=toggle_auto,
                             font=("Segoe UI", 9, "bold"),
                             bg=DEEP_SEA_THEME['success'], fg=DEEP_SEA_THEME['text_primary'],
                             activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary'], bd=2)
        auto_btn.pack(side=tk.RIGHT, padx=(4, 8), pady=4)

        # Will wire after defining refresh
        refresh_btn = tk.Button(controls, text="Refresh Now", font=("Segoe UI", 9, "bold"), bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'], bd=2,
                                activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary'])
        refresh_btn.pack(side=tk.RIGHT, padx=6, pady=4)

        # News list area
        content = tk.Frame(tab, bg=DEEP_SEA_THEME['primary_bg'])
        content.pack(fill=tk.BOTH, expand=1)

        # Treeview styled to dark theme
        style = ttk.Style(tab)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('LiveNews.Treeview',
                        background=DEEP_SEA_THEME['primary_bg'],
                        fieldbackground=DEEP_SEA_THEME['primary_bg'],
                        foreground=DEEP_SEA_THEME['text_primary'],
                        rowheight=32,
                        bordercolor=DEEP_SEA_THEME['border'],
                        font=("Segoe UI", 12))
        style.map('LiveNews.Treeview',
                   background=[('selected', DEEP_SEA_THEME['surface_bg'])],
                   foreground=[('selected', DEEP_SEA_THEME['text_primary'])])

        columns = ("time", "symbol", "headline", "provider")
        tree = ttk.Treeview(content, columns=columns, show='headings', style='LiveNews.Treeview')
        tree.heading('time', text='Time')
        tree.heading('symbol', text='Symbol')
        tree.heading('headline', text='Headline')
        tree.heading('provider', text='Provider')
        tree.column('time', width=120, anchor='w')
        tree.column('symbol', width=100, anchor='w')
        tree.column('headline', width=720, anchor='w')
        tree.column('provider', width=160, anchor='w')

        vsb = ttk.Scrollbar(content, orient='vertical', command=tree.yview)
        tree.configure(yscroll=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tag styles
        # Default row style: black background with white text
        tree.tag_configure('newsrow', background=DEEP_SEA_THEME['primary_bg'], foreground=DEEP_SEA_THEME['text_primary'])
        # Optional tag for newly arrived items (slightly lighter gray)
        tree.tag_configure('new', background=DEEP_SEA_THEME['surface_bg'])

        # Data + refresh loop
        last_seen = set()
        refresh_job = {"id": None}

        FEEDS = [
            # Broad business feeds
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.reuters.com/reuters/USBusinessNews",
            # Yahoo Finance top stories
            "https://finance.yahoo.com/news/rssindex",
        ]

        def parse_items(xml_text: str, url: str):
            items = []
            try:
                root = ET.fromstring(xml_text)
            except Exception:
                return items
            # RSS items
            for item in root.findall('.//item'):
                try:
                    title_el = item.find('title')
                    link_el = item.find('link')
                    date_el = item.find('pubDate')
                    source_el = item.find('source')
                    title = (title_el.text or '').strip() if title_el is not None else ''
                    link = (link_el.text or '').strip() if link_el is not None else ''
                    src = (source_el.text or '').strip() if source_el is not None else urlparse(url).netloc
                    dt = None
                    if date_el is not None and date_el.text:
                        try:
                            dt = parsedate_to_datetime(date_el.text)
                        except Exception:
                            dt = None
                    items.append({'title': title, 'link': link, 'source': src, 'dt': dt})
                except Exception:
                    continue
            # Atom entries
            atom_ns = '{http://www.w3.org/2005/Atom}'
            for entry in root.findall(f'.//{atom_ns}entry'):
                try:
                    title_el = entry.find(f'{atom_ns}title')
                    link_el = entry.find(f'{atom_ns}link')
                    updated_el = entry.find(f'{atom_ns}updated') or entry.find(f'{atom_ns}published')
                    source_el = entry.find(f'{atom_ns}source') or entry.find(f'{atom_ns}author')
                    title = (title_el.text or '').strip() if title_el is not None else ''
                    link = ''
                    if link_el is not None:
                        href = link_el.attrib.get('href')
                        link = (href or '').strip()
                    src = urlparse(url).netloc
                    if source_el is not None:
                        name_el = source_el.find(f'{atom_ns}title') or source_el.find(f'{atom_ns}name')
                        if name_el is not None and name_el.text:
                            src = name_el.text.strip()
                    dt = None
                    if updated_el is not None and updated_el.text:
                        try:
                            dt = parsedate_to_datetime(updated_el.text)
                        except Exception:
                            dt = None
                    items.append({'title': title, 'link': link, 'source': src, 'dt': dt})
                except Exception:
                    continue
            return items

        def fetch_news(symbol_filter: str | None):
            headlines = []
            sym = (symbol_filter or '').upper().strip()
            urls = list(FEEDS)
            if sym:
                # Add a symbol-specific Yahoo feed when possible
                urls.append(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US")
            for url in urls:
                try:
                    xml_text = http_get_text(url, timeout=8, retries=2)
                    if xml_text:
                        items = parse_items(xml_text, url)
                        headlines.extend(items)
                except Exception:
                    continue
            # Filter by symbol in title if provided
            if sym:
                sym_re = re.compile(rf"\b{re.escape(sym)}\b", re.I)
                headlines = [h for h in headlines if sym_re.search(h['title'])]
            # Deduplicate by link or title
            seen_links = set()
            unique = []
            for h in headlines:
                key = h['link'] or h['title']
                if key and key not in seen_links:
                    seen_links.add(key)
                    unique.append(h)
            # Sort newest first
            unique.sort(key=lambda x: x.get('dt') or datetime.now(timezone.utc), reverse=True)
            return unique[:200]

        def open_link(event=None):
            sel = tree.selection()
            if not sel:
                return
            # We stored link in item values at index 4 via hidden column; pull from item data
            item_id = sel[0]
            data = tree.item(item_id, 'values')
            # Data values: [time, symbol, headline, provider, link]
            if len(data) >= 5 and data[4]:
                webbrowser.open_new_tab(data[4])

        # Reconfigure columns to hold link hidden
        tree['columns'] = ("time", "symbol", "headline", "provider", "_link")
        tree.column("_link", width=0, stretch=False)
        tree.heading("_link", text="")
        tree.bind('<Double-1>', open_link)

        def render_news(items, highlight_new=True):
            # Preserve current selection and scroll position (best-effort)
            tree.delete(*tree.get_children())
            for h in items:
                dt = h.get('dt')
                timestr = dt.astimezone().strftime('%H:%M:%S') if isinstance(dt, datetime) else ''
                title = h.get('title', '')
                src = h.get('source', '')
                symcol = (sym_var.get() or '').upper()
                link = h.get('link', '')
                is_new = link not in last_seen
                tags = ['newsrow']
                if highlight_new and is_new:
                    tags.append('new')
                tree.insert('', 'end', values=(timestr, symcol, title, src, link), tags=tuple(tags))
                if link:
                    last_seen.add(link)

        def refresh_news(force=False):
            status_var.set("Refreshing…")
            tab.update_idletasks()
            try:
                items = fetch_news(sym_var.get())
                if not items:
                    # Provide a small sample if nothing fetched (e.g., offline/restricted)
                    now = datetime.now()
                    items = [{
                        'title': 'Sample: Markets open mixed as investors weigh data',
                        'link': '',
                        'source': 'Local',
                        'dt': now
                    }, {
                        'title': 'Sample: Tech stocks lead early trading',
                        'link': '',
                        'source': 'Local',
                        'dt': now
                    }]
                    render_news(items, highlight_new=False)
                    status_var.set("No live headlines fetched — showing samples.")
                else:
                    render_news(items, highlight_new=not force)
                    status_var.set(f"Updated at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                status_var.set(f"Error: {e}")
            # Schedule next refresh
            if auto_var.get() and tab.winfo_exists():
                # 30s refresh cadence
                refresh_job['id'] = tab.after(30000, refresh_news)
        # Wire refresh button now
        refresh_btn.config(command=lambda: refresh_news(force=True))

        # Cancel timer on tab destroy
        def on_destroy(_event=None):
            if refresh_job['id'] is not None:
                try:
                    tab.after_cancel(refresh_job['id'])
                except Exception:
                    pass
        tab.bind("<Destroy>", on_destroy)

        # Initial load
        refresh_news(force=True)

    def open_news_tab(symbol_hint: str | None = None):
        """Open or focus the News tab in the main notebook."""
        global GLOBAL_NOTEBOOK
        if GLOBAL_NOTEBOOK is None or not GLOBAL_NOTEBOOK.winfo_exists():
            # Create a trading interface to host the notebook (use default symbol)
            try:
                open_stock_layout_with_symbol(symbol)
            except Exception:
                # Fallback: open_stock_layout may not be available from Home; ensure trading interface
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[0]
                    latest_close = hist['Close'].iloc[1]
                    pg = ((latest_close - prev_close) / prev_close) * 100
                else:
                    prev_close = latest_close = 0
                    pg = 0
                create_trading_interface(symbol, ticker, prev_close, latest_close, pg, get_ta_signal(symbol), get_rsi(symbol) or 0, get_in_depth_analysis(symbol))
        # Check if News tab exists
        for i in range(len(GLOBAL_NOTEBOOK.tabs())):
            tab_id = GLOBAL_NOTEBOOK.tabs()[i]
            if GLOBAL_NOTEBOOK.tab(tab_id, 'text') == 'News':
                GLOBAL_NOTEBOOK.select(tab_id)
                return
        # Create new News tab
        create_news_tab(GLOBAL_NOTEBOOK, symbol_hint)

    def create_heatmap_tab(notebook):
        """Create a 'Heatmap' tab inside the main notebook."""
        tab = tk.Frame(notebook, bg=DEEP_SEA_THEME['primary_bg'])
        notebook.add(tab, text="Heatmap")
        notebook.select(tab)

        # Controls
        controls = tk.Frame(tab, bg=DEEP_SEA_THEME['secondary_bg'])
        controls.pack(side=tk.TOP, fill=tk.X)
        tk.Label(controls, text="Universe:", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT, padx=(10, 4), pady=6)
        uni_var = tk.StringVar(value="Mega-Caps")
        uni_dropdown = ttk.Combobox(controls, textvariable=uni_var, values=["Mega-Caps", "Tech+Banks"], state="readonly", width=14)
        uni_dropdown.pack(side=tk.LEFT, padx=(0, 8), pady=6)

        status_var = tk.StringVar(value="")
        tk.Label(controls, textvariable=status_var, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.RIGHT, padx=10)

        refresh_btn = tk.Button(controls, text="Refresh", font=("Segoe UI", 9, "bold"), bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'], bd=2,
                                activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary'])
        refresh_btn.pack(side=tk.RIGHT, padx=6, pady=4)

        # Canvas area
        content = tk.Frame(tab, bg=DEEP_SEA_THEME['primary_bg'])
        content.pack(fill=tk.BOTH, expand=1)

        def get_universe(name):
            if name == "Mega-Caps":
                return [
                    "AAPL","MSFT","NVDA","AMZN","META",
                    "TSLA","GOOGL","BRK-B","LLY","JPM",
                    "V","WMT","XOM","MA","UNH",
                    "PG","HD","JNJ","AVGO","COST",
                ]
            else:
                return [
                    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","AMD","INTC","CSCO",
                    "JPM","BAC","WFC","C","GS","MS","V","MA","PYPL","AXP"
                ]

        def pct_color(p):
            # Map -5..+5 to red..green
            if p is None:
                return "#555555"
            v = max(-5.0, min(5.0, p))
            t = (v + 5.0) / 10.0
            # interpolate between red (231,76,60) and green (39,174,96)
            r1,g1,b1 = (231,76,60)
            r2,g2,b2 = (39,174,96)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            return f"#{r:02x}{g:02x}{b:02x}"

        def fetch_gains(tickers):
            data = {}
            for tkr in tickers:
                try:
                    pg = get_percent_gain(tkr)
                    if pg is None:
                        # Fallback small random
                        pg = round(random.uniform(-2, 2), 2)
                except Exception:
                    pg = round(random.uniform(-2, 2), 2)
                data[tkr] = pg
            return data

        def render():
            for w in content.winfo_children():
                w.destroy()
            tickers = get_universe(uni_var.get())
            gains = fetch_gains(tickers)

            # Compute grid size
            cols = 5
            rows = (len(tickers) + cols - 1) // cols

            # Create matplotlib fig
            plt.close('all')
            fig = plt.figure(figsize=(12, 6), facecolor=DEEP_SEA_THEME['primary_bg'])
            ax = fig.add_subplot(111)
            ax.set_facecolor(DEEP_SEA_THEME['primary_bg'])
            ax.set_axis_off()

            cell_w = 1.0 / cols
            cell_h = 1.0 / rows
            for idx, tkr in enumerate(tickers):
                r = idx // cols
                c = idx % cols
                x0 = c * cell_w
                y0 = 1.0 - (r + 1) * cell_h
                p = gains.get(tkr)
                color = pct_color(p)
                rect = Rectangle((x0, y0), cell_w*0.98, cell_h*0.94, facecolor=color, edgecolor=DEEP_SEA_THEME['border'])
                ax.add_patch(rect)
                label = f"{tkr}\n{p:+.2f}%" if p is not None else f"{tkr}\nN/A"
                ax.text(x0 + cell_w/2, y0 + cell_h/2, label, ha='center', va='center', color=DEEP_SEA_THEME['text_primary'], fontsize=10, weight='bold')

            canvas = FigureCanvasTkAgg(fig, master=content)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
            status_var.set(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        refresh_btn.config(command=render)
        uni_dropdown.bind('<<ComboboxSelected>>', lambda e: render())
        render()

    def open_heatmap_tab():
        """Open or focus the Heatmap tab in the main notebook."""
        global GLOBAL_NOTEBOOK
        if GLOBAL_NOTEBOOK is None or not GLOBAL_NOTEBOOK.winfo_exists():
            try:
                open_stock_layout_with_symbol(symbol)
            except Exception:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[0]
                    latest_close = hist['Close'].iloc[1]
                    pg = ((latest_close - prev_close) / prev_close) * 100
                else:
                    prev_close = latest_close = 0
                    pg = 0
                create_trading_interface(symbol, ticker, prev_close, latest_close, pg, get_ta_signal(symbol), get_rsi(symbol) or 0, get_in_depth_analysis(symbol))
        # Focus if exists
        for tab_id in GLOBAL_NOTEBOOK.tabs():
            if GLOBAL_NOTEBOOK.tab(tab_id, 'text') == 'Heatmap':
                GLOBAL_NOTEBOOK.select(tab_id)
                return
        create_heatmap_tab(GLOBAL_NOTEBOOK)

    # --- Screener (TradingView-like) ---
    def _load_symbols_from_local():
        """Try to load a large US symbol universe from local files.
        Supports txt (one symbol per line) and csv (with a 'Symbol' column).
        """
        search_names = [
            'symbols.txt', 'tickers.txt', 'tickers_us.txt',
            'symbols.csv', 'tickers.csv', 'nasdaq_screener.csv',
        ]
        roots = [script_dir, os.getcwd(), os.path.join(script_dir, 'data'), os.path.join(script_dir, 'tools')]
        found_path = None
        for rootp in roots:
            if not rootp or not os.path.isdir(rootp):
                continue
            for name in search_names:
                path = os.path.join(rootp, name)
                if os.path.isfile(path):
                    found_path = path
                    break
            if found_path:
                break
        symbols = []
        if not found_path:
            return symbols
        try:
            if found_path.lower().endswith('.csv'):
                import csv
                with open(found_path, 'r', newline='', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    # Try several possible column names
                    cols = [c for c in reader.fieldnames or []]
                    sym_col = None
                    for c in cols:
                        if c.lower() in ('symbol', 'ticker', 'symbol/ticker', 'ticker symbol'):
                            sym_col = c
                            break
                    if sym_col:
                        for row in reader:
                            val = (row.get(sym_col) or '').strip().upper()
                            if val:
                                symbols.append(val)
                    else:
                        # fallback: first column
                        f.seek(0)
                        for i, line in enumerate(f):
                            if i == 0:
                                continue
                            parts = line.split(',')
                            if parts:
                                s = parts[0].strip().upper()
                                if s and s.isascii():
                                    symbols.append(s)
            else:
                with open(found_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        s = line.strip().upper()
                        if s and s.isascii():
                            symbols.append(s)
        except Exception:
            symbols = []
        # Deduplicate while preserving order
        seen = set()
        out = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def create_screener_tab(notebook):
        tab = tk.Frame(notebook, bg=DEEP_SEA_THEME['primary_bg'])
        notebook.add(tab, text="Screener")
        notebook.select(tab)

        # Controls
        controls = tk.Frame(tab, bg=DEEP_SEA_THEME['secondary_bg'])
        controls.pack(side=tk.TOP, fill=tk.X)

        tk.Label(controls, text="Universe:", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT, padx=(10, 4), pady=6)
        uni_var = tk.StringVar(value="Mega-Caps")
        uni_dropdown = ttk.Combobox(controls, textvariable=uni_var, values=["Mega-Caps", "Tech+Banks", "All (Local)"] , state="readonly", width=14)
        uni_dropdown.pack(side=tk.LEFT, padx=(0, 8), pady=6)

        tk.Label(controls, text="Min Price:", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT, padx=(6, 4))
        min_price_var = tk.StringVar(value="0")
        tk.Entry(controls, textvariable=min_price_var, width=6, bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary']).pack(side=tk.LEFT)

        tk.Label(controls, text="Min %Chg:", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT, padx=(6, 4))
        min_chg_var = tk.StringVar(value="0")
        tk.Entry(controls, textvariable=min_chg_var, width=6, bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary']).pack(side=tk.LEFT)

        tk.Label(controls, text="Min Vol:", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT, padx=(6, 4))
        min_vol_var = tk.StringVar(value="0")
        tk.Entry(controls, textvariable=min_vol_var, width=8, bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary']).pack(side=tk.LEFT)

        rsi_ob_var = tk.BooleanVar(value=False)
        rsi_os_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="RSI > 70", variable=rsi_ob_var, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary'], selectcolor=DEEP_SEA_THEME['secondary_bg']).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(controls, text="RSI < 30", variable=rsi_os_var, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary'], selectcolor=DEEP_SEA_THEME['secondary_bg']).pack(side=tk.LEFT)

        status_var = tk.StringVar(value="")
        tk.Label(controls, textvariable=status_var, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.RIGHT, padx=10)

        scan_btn = tk.Button(controls, text="Scan", font=("Segoe UI", 9, "bold"), bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'], bd=2,
                             activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary'])
        scan_btn.pack(side=tk.RIGHT, padx=6, pady=4)

        # Results table
        content = tk.Frame(tab, bg=DEEP_SEA_THEME['primary_bg'])
        content.pack(fill=tk.BOTH, expand=1)

        cols = ("ticker","price","chg%","volume","rsi")
        tv = ttk.Treeview(content, columns=cols, show='headings')
        for c, w in [("ticker",100),("price",100),("chg%",100),("volume",140),("rsi",80)]:
            tv.heading(c, text=c.upper())
            tv.column(c, width=w, anchor='w')
        vsb = ttk.Scrollbar(content, orient='vertical', command=tv.yview)
        tv.configure(yscroll=vsb.set)
        tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def get_universe(name):
            if name == "Mega-Caps":
                return [
                    "AAPL","MSFT","NVDA","AMZN","META",
                    "TSLA","GOOGL","BRK-B","LLY","JPM",
                    "V","WMT","XOM","MA","UNH",
                    "PG","HD","JNJ","AVGO","COST",
                ]
            elif name == "Tech+Banks":
                return [
                    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","AMD","INTC","CSCO",
                    "JPM","BAC","WFC","C","GS","MS","V","MA","PYPL","AXP"
                ]
            else:
                syms = _load_symbols_from_local()
                if not syms:
                    messagebox.showinfo("Screener", "No local symbol list found. Place symbols.txt or nasdaq_screener.csv in the project folder to enable 'All (Local)'. Falling back to Mega-Caps.")
                    return get_universe("Mega-Caps")
                return syms

        def scan_universe():
            tickers = get_universe(uni_var.get())
            try:
                min_price = float(min_price_var.get() or 0)
            except Exception:
                min_price = 0
            try:
                min_chg = float(min_chg_var.get() or 0)
            except Exception:
                min_chg = 0
            try:
                min_vol = float(min_vol_var.get() or 0)
            except Exception:
                min_vol = 0

            tv.delete(*tv.get_children())
            status_var.set("Scanning…")
            scan_btn.config(state=tk.DISABLED)

            results = []

            def worker():
                total = len(tickers)
                for i, tkr in enumerate(tickers, start=1):
                    price = None
                    chg = None
                    vol = None
                    rsi_val = None
                    try:
                        t = yf.Ticker(tkr)
                        hist = t.history(period="60d", interval="1d", prepost=False, repair=True)
                        if hist is not None and len(hist) >= 2:
                            price = float(hist['Close'].iloc[-1])
                            prev = float(hist['Close'].iloc[-2])
                            chg = ((price - prev) / prev) * 100.0
                            vol = int(hist['Volume'].iloc[-1])
                            # RSI from close
                            ser = pd.Series(hist['Close'])
                            rsi_series = calculate_simple_rsi(ser)
                            rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None
                    except Exception:
                        pass

                    # Apply filters
                    if price is None: continue
                    if price < min_price: continue
                    if chg is not None and chg < min_chg: continue
                    if vol is not None and vol < min_vol: continue
                    if rsi_ob_var.get() and (rsi_val is None or rsi_val <= 70): continue
                    if rsi_os_var.get() and (rsi_val is None or rsi_val >= 30): continue

                    results.append((tkr, price, chg, vol, rsi_val))
                    if i % 50 == 0:
                        # periodic progress update
                        def _progress():
                            status_var.set(f"Scanning… {i}/{total}")
                        tab.after(1, _progress)

                # Update UI on main thread
                def finish():
                    for tkr, price, chg, vol, rsi_val in results:
                        tv.insert('', 'end', values=(tkr, f"{price:.2f}", f"{(chg or 0):+.2f}", f"{vol or 0:,}", f"{(rsi_val or 0):.1f}"))
                    status_var.set(f"{len(results)} results")
                    scan_btn.config(state=tk.NORMAL)

                tab.after(10, finish)

            threading.Thread(target=worker, daemon=True).start()

        scan_btn.config(command=scan_universe)
        uni_dropdown.bind('<<ComboboxSelected>>', lambda e: scan_universe())
        scan_universe()

    def open_screener_tab():
        global GLOBAL_NOTEBOOK
        if GLOBAL_NOTEBOOK is None or not GLOBAL_NOTEBOOK.winfo_exists():
            try:
                open_stock_layout_with_symbol(symbol)
            except Exception:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[0]
                    latest_close = hist['Close'].iloc[1]
                    pg = ((latest_close - prev_close) / prev_close) * 100
                else:
                    prev_close = latest_close = 0
                    pg = 0
                create_trading_interface(symbol, ticker, prev_close, latest_close, pg, get_ta_signal(symbol), get_rsi(symbol) or 0, get_in_depth_analysis(symbol))
        # Focus if exists
        for tab_id in GLOBAL_NOTEBOOK.tabs():
            if GLOBAL_NOTEBOOK.tab(tab_id, 'text') == 'Screener':
                GLOBAL_NOTEBOOK.select(tab_id)
                return
        create_screener_tab(GLOBAL_NOTEBOOK)

    def create_home_tab(notebook):
        """Create a lightweight Home dashboard inside the notebook."""
        tab = tk.Frame(notebook, bg=DEEP_SEA_THEME['primary_bg'])
        # Determine a unique tab title
        if not hasattr(notebook, 'home_counter'):
            notebook.home_counter = 1
        title = 'Home' if notebook.home_counter == 1 else f"Home {notebook.home_counter}"
        notebook.home_counter += 1
        notebook.add(tab, text=title)
        notebook.select(tab)

        header = tk.Frame(tab, bg=DEEP_SEA_THEME['primary_bg'])
        header.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(12, 6))
        tk.Label(header, text=title, font=("Segoe UI", 18, "bold"), fg=DEEP_SEA_THEME['text_primary'], bg=DEEP_SEA_THEME['primary_bg']).pack(side=tk.LEFT)

        # Utilities -----------------------------------------------------
        def open_symbol_tab(target_symbol: str):
            target = (target_symbol or "").strip().upper()
            if not target:
                return
            try:
                ticker_obj = yf.Ticker(target)
                hist = ticker_obj.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[0]
                    latest_close = hist['Close'].iloc[1]
                    percent_gain = ((latest_close - prev_close) / prev_close) * 100
                else:
                    prev_close = latest_close = 0.0
                    percent_gain = 0.0
                ta_signal = get_ta_signal(target)
                rsi_val = get_rsi(target) or 0
                analysis = get_in_depth_analysis(target)
                create_trading_interface(
                    target,
                    ticker_obj,
                    prev_close,
                    latest_close,
                    percent_gain,
                    ta_signal,
                    rsi_val,
                    analysis,
                )
            except Exception as exc:
                logging.error("Failed to open symbol %s: %s", target, exc)
                messagebox.showerror("Error", f"Unable to open {target}. Please check the symbol and try again.")

        # Quick open symbol
        quick = tk.Frame(header, bg=DEEP_SEA_THEME['primary_bg'])
        quick.pack(side=tk.RIGHT)
        tk.Label(quick, text="Open: ", bg=DEEP_SEA_THEME['primary_bg'], fg=DEEP_SEA_THEME['text_secondary']).pack(side=tk.LEFT)
        open_var = tk.StringVar()
        ent = tk.Entry(quick, textvariable=open_var, width=10, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_primary'], insertbackground=DEEP_SEA_THEME['text_accent'])
        ent.pack(side=tk.LEFT)
        tk.Button(quick, text="Go", command=lambda: open_symbol_tab(open_var.get().strip().upper()),
                 font=("Segoe UI", 10, "bold"), bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'], bd=2,
                 activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary']).pack(side=tk.LEFT, padx=6)

        # Cards grid
        grid = tk.Frame(tab, bg=DEEP_SEA_THEME['primary_bg'])
        grid.pack(fill=tk.BOTH, expand=1, padx=16, pady=8)
        for i in range(3):
            grid.grid_columnconfigure(i, weight=1, uniform="cols")
        for r in range(2):
            grid.grid_rowconfigure(r, weight=1)

        # Screeners card
        scr_outer, scr = make_card(grid, "Screeners", "Find anything with a simple scan", width=340, height=180)
        scr_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        tk.Button(scr, text="Open Screener", command=lambda: open_screener_tab(),
                  font=("Segoe UI", 11, "bold"), bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'], bd=2,
                  activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary']).pack(anchor='w', padx=14, pady=10)

        # News card
        news_outer, news_card = make_card(grid, "News Flow", "US stock headlines", width=340, height=180)
        news_outer.grid(row=0, column=1, sticky="nsew", padx=12, pady=(0, 12))
        tk.Button(news_card, text="Open News", command=lambda: open_news_tab(),
                  font=("Segoe UI", 11, "bold"), bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'], bd=2,
                  activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary']).pack(anchor='w', padx=14, pady=10)

        # Heatmaps card
        heat_outer, heat_card = make_card(grid, "Heatmaps", "Market performance", width=340, height=180)
        heat_outer.grid(row=0, column=2, sticky="nsew", padx=(12, 0), pady=(0, 12))
        tk.Button(heat_card, text="Open Heatmap", command=lambda: open_heatmap_tab(),
                  font=("Segoe UI", 11, "bold"), bg=DEEP_SEA_THEME['success'], fg=DEEP_SEA_THEME['text_primary'], bd=2,
                  activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary']).pack(anchor='w', padx=14, pady=10)


    def create_home_page():
        clear_main_content()
        home_frame = tk.Frame(main_content, bg=DEEP_SEA_THEME['primary_bg'])
        home_frame.pack(fill=tk.BOTH, expand=1)

        # Header row
        header = tk.Frame(home_frame, bg=DEEP_SEA_THEME['primary_bg'])
        header.pack(side=tk.TOP, fill=tk.X, padx=24, pady=(18, 8))

        title_lbl = tk.Label(header, text="Supercharts", font=("Segoe UI", 24, "bold"), fg=DEEP_SEA_THEME['text_primary'], bg=DEEP_SEA_THEME['primary_bg'])
        title_lbl.pack(side=tk.LEFT)

    # User info and logout section
        user_section = tk.Frame(header, bg=DEEP_SEA_THEME['primary_bg'])
        user_section.pack(side=tk.RIGHT, padx=(10, 0))

        # New Window button (multi-instance launcher) - pack first so it sits at far right
        new_window_btn = tk.Button(
            user_section,
            text="New Window",
            command=spawn_new_instance,
            font=("Segoe UI", 10, "bold"),
            bg=DEEP_SEA_THEME['info'],
            fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'],
            activeforeground=DEEP_SEA_THEME['primary_bg'],
            bd=2,
            relief="raised",
            padx=10,
            pady=2
        )
        new_window_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # Current user label
        global current_user
        user_label = tk.Label(user_section, text=f"Logged in as: {current_user}", 
                             font=("Segoe UI", 10), fg=DEEP_SEA_THEME['text_secondary'], 
                             bg=DEEP_SEA_THEME['primary_bg'])
        user_label.pack(side=tk.RIGHT, padx=(0, 10))

        # Logout button
        def logout():
            if messagebox.askyesno("Logout", "Do you want to logout and clear saved credentials?"):
                clear_remember_me_credentials()
                print("Saved credentials cleared")
                root.quit()
                sys.exit(0)

        logout_btn = tk.Button(
            user_section,
            text="Logout",
            command=logout,
            font=("Segoe UI", 10, "bold"),
            bg=DEEP_SEA_THEME['danger'],
            fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'],
            activeforeground=DEEP_SEA_THEME['primary_bg'],
            bd=2,
            relief="raised",
            padx=10,
            pady=2
        )
        logout_btn.pack(side=tk.RIGHT)

        # Search bar (wired to open chart)
        search_wrap = tk.Frame(header, bg=DEEP_SEA_THEME['primary_bg'])
        search_wrap.pack(side=tk.RIGHT, fill=tk.X, padx=(0, 20))
        search_frame = tk.Frame(search_wrap, bg=DEEP_SEA_THEME['secondary_bg'], bd=1, relief="solid")
        search_frame.pack()
        tk.Label(search_frame, text="[S]", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_accent'], font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(10, 6))

        placeholder_text = "Search symbol (e.g., MSFT)"
        search_entry = tk.Entry(search_frame, bd=0, bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_secondary'], insertbackground=DEEP_SEA_THEME['text_accent'], width=32)
        search_entry.insert(0, placeholder_text)

        def handle_search_submit(event=None):
            sym = search_entry.get().strip().upper()
            if not sym or sym == placeholder_text.upper() or sym == placeholder_text:
                return
            # keep only valid ticker characters
            sym = re.sub(r"[^A-Z0-9\.\-]", "", sym)
            if not sym:
                return
            open_stock_layout_with_symbol(sym)

        def on_focus_in(_):
            if search_entry.get() == placeholder_text:
                search_entry.delete(0, tk.END)
                search_entry.config(fg=DEEP_SEA_THEME['text_primary'])

        def on_focus_out(_):
            if not search_entry.get():
                search_entry.insert(0, placeholder_text)
                search_entry.config(fg=DEEP_SEA_THEME['text_secondary'])

        search_entry.bind("<Return>", handle_search_submit)
        search_entry.bind("<FocusIn>", on_focus_in)
        search_entry.bind("<FocusOut>", on_focus_out)
        search_entry.pack(side=tk.LEFT, ipady=7, padx=(0, 6))

        go_btn = tk.Button(
            search_frame,
            text="Go",
            command=handle_search_submit,
            font=("Segoe UI", 10, "bold"),
            bg=DEEP_SEA_THEME['info'],
            fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'],
            activeforeground=DEEP_SEA_THEME['primary_bg'],
            bd=1,
            relief="raised",
            padx=10,
        )
        go_btn.pack(side=tk.LEFT, padx=(0, 10), ipady=3)

        # --- Quick Tabs Row (5 custom tabs) ---
        quick_row = tk.Frame(home_frame, bg=DEEP_SEA_THEME['primary_bg'])
        quick_row.pack(fill=tk.X, padx=24, pady=(0, 8))

        tk.Label(quick_row, text="Quick tabs:", fg=DEEP_SEA_THEME['text_secondary'], bg=DEEP_SEA_THEME['primary_bg'], font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=(0, 8))

        quick_btns = []

        def refresh_quick_tabs():
            tabs = load_custom_tabs()
            for i, btn in enumerate(quick_btns):
                sym = tabs[i].strip().upper()
                if sym:
                    btn.config(text=sym, bg=DEEP_SEA_THEME['success'], fg=DEEP_SEA_THEME['primary_bg'], activebackground=DEEP_SEA_THEME['active'])
                else:
                    btn.config(text="+ Set", bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'], activebackground=DEEP_SEA_THEME['hover'])

        def configure_quick_tab(i):
            tabs = load_custom_tabs()
            current = tabs[i].strip().upper()
            prompt = f"Enter ticker for Quick Tab {i+1}" + (f" (currently: {current})" if current else "")
            new = simpledialog.askstring("Set Quick Tab", prompt + ":", parent=root)
            if new is None:
                return
            new = new.strip().upper()
            new = re.sub(r"[^A-Z0-9\.\-]", "", new)
            tabs[i] = new
            save_custom_tabs(tabs)
            refresh_quick_tabs()

        def handle_quick_tab_click(i):
            tabs = load_custom_tabs()
            sym = tabs[i].strip().upper()
            if sym:
                open_stock_layout_with_symbol(sym)
            else:
                configure_quick_tab(i)

        # Create 5 buttons
        init_tabs = load_custom_tabs()
        for i in range(5):
            initial_sym = init_tabs[i].strip().upper() if i < len(init_tabs) and init_tabs[i] else ""
            btn = tk.Button(
                quick_row,
                text=(initial_sym if initial_sym else "+ Set"),
                command=lambda idx=i: handle_quick_tab_click(idx),
                font=("Segoe UI", 10, "bold"),
                bg=(DEEP_SEA_THEME['success'] if initial_sym else DEEP_SEA_THEME['surface_bg']),
                fg=(DEEP_SEA_THEME['primary_bg'] if initial_sym else DEEP_SEA_THEME['text_primary']),
                activebackground=(DEEP_SEA_THEME['active'] if initial_sym else DEEP_SEA_THEME['hover']),
                activeforeground=DEEP_SEA_THEME['primary_bg'],
                bd=2,
                relief="raised",
                padx=10,
                pady=4,
                width=8,
            )
            btn.pack(side=tk.LEFT, padx=6)
            btn.bind("<Button-3>", lambda e, idx=i: configure_quick_tab(idx))  # Right-click to set/change
            quick_btns.append(btn)

        # Ensure styling is up to date
        refresh_quick_tabs()

        # Layouts row (cards)
        layouts_row = tk.Frame(home_frame, bg=DEEP_SEA_THEME['primary_bg'])
        layouts_row.pack(fill=tk.X, padx=24, pady=(4, 8))

        # Create new layout card (button-like, not wired)
        create_card, create_inner = make_card(layouts_row, "+  Create new layout", "", width=340, height=120)
        create_card.pack(side=tk.LEFT, padx=(0, 16), fill=tk.BOTH, expand=1)

        # Placeholder small layout cards
        small1, _ = make_card(layouts_row, "* Unnamed", "SMCI, 1h - 11 Aug '25", width=340, height=120)
        small1.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=1)
        small2, _ = make_card(layouts_row, "* Unnamed", "TEM, 15m - 4 Aug '25", width=340, height=120)
        small2.pack(side=tk.LEFT, padx=8, fill=tk.BOTH, expand=1)

        # Grid container for main content cards
        grid = tk.Frame(home_frame, bg=DEEP_SEA_THEME['primary_bg'])
        grid.pack(fill=tk.BOTH, expand=1, padx=24, pady=(8, 0))

        # Configure grid columns and rows
        for i in range(3):
            grid.grid_columnconfigure(i, weight=1, uniform="cols")
        for r in range(3):
            grid.grid_rowconfigure(r, weight=1)

        # Row 1: Screeners (left, span 2 cols) and Calendars (right)
        scr_outer, scr = make_card(grid, "Screeners", "Find anything with a simple scan", width=720, height=220)
        scr_outer.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 12), pady=(0, 12))
        # Screener quick actions
        actions = tk.Frame(scr, bg=DEEP_SEA_THEME['card_bg'])
        actions.pack(fill=tk.X, padx=14, pady=8)
        tk.Button(actions, text="Open Screener", command=lambda: open_screener_tab(),
                  font=("Segoe UI", 11, "bold"), bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'], bd=2,
                  activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary']).pack(side=tk.LEFT, padx=(0, 8))
        # Popular tickers pills
        pills = tk.Frame(scr, bg=DEEP_SEA_THEME['card_bg'])
        pills.pack(fill=tk.X, padx=14, pady=(0, 10))
        for txt, col in [("AAPL", DEEP_SEA_THEME['accent_bg']), ("TSLA", DEEP_SEA_THEME['accent_bg']), ("GOOGL", DEEP_SEA_THEME['accent_bg']), ("WMT", DEEP_SEA_THEME['accent_bg'])]:
            tk.Label(pills, text=f"  {txt}  ", bg=col, fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 10, "bold"), bd=0).pack(side=tk.LEFT, padx=6, pady=6)

        cal_outer, _ = make_card(grid, "Calendars", "Explore the world's financial events", width=340, height=220)
        cal_outer.grid(row=0, column=2, sticky="nsew", padx=(12, 0), pady=(0, 12))

        # Row 2: News Flow (left), Heatmaps (center), Options (right)
        news_outer, news_card = make_card(grid, "News Flow", "US stock headlines", width=340, height=220)
        news_outer.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=12)
        # Add a button to open the live news window
        news_btn = tk.Button(
            news_card,
            text="Open Live News",
            command=lambda: open_news_tab(),
            font=("Segoe UI", 11, "bold"),
            bg=DEEP_SEA_THEME['info'], fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary'],
            bd=2, relief="raised"
        )
        news_btn.pack(padx=16, pady=16, anchor="w")
        # Also allow clicking the card to open
        news_card.bind("<Button-1>", lambda e: open_news_tab())

        heat_outer, heat_card = make_card(grid, "Heatmaps", "See the full picture for global markets", width=340, height=220)
        heat_outer.grid(row=1, column=1, sticky="nsew", padx=12, pady=12)
        heat_btn = tk.Button(
            heat_card,
            text="Open Heatmap",
            command=lambda: open_heatmap_tab(),
            font=("Segoe UI", 11, "bold"),
            bg=DEEP_SEA_THEME['success'], fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['text_primary'],
            bd=2, relief="raised"
        )
        heat_btn.pack(padx=16, pady=16, anchor="w")
        # Mini preview heatmap
        preview = tk.Canvas(heat_card, width=300, height=130, bg=DEEP_SEA_THEME['card_bg'], highlightthickness=0)
        preview.pack(padx=14, pady=(0, 12), anchor="w")
        labels = ["AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","JPM","WMT","XOM"]
        cols = 5
        cell_w = 56
        cell_h = 52
        def color_for(p):
            v = max(-5.0, min(5.0, p))
            t = (v + 5.0) / 10.0
            r1,g1,b1 = (231,76,60)
            r2,g2,b2 = (39,174,96)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            return f"#{r:02x}{g:02x}{b:02x}"
        for i, tkr in enumerate(labels):
            r = i // cols
            c = i % cols
            x0 = c * (cell_w + 2)
            y0 = r * (cell_h + 2)
            val = round(random.uniform(-2.5, 2.5), 2)
            preview.create_rectangle(x0, y0, x0+cell_w, y0+cell_h, fill=color_for(val), outline=DEEP_SEA_THEME['border'])
            preview.create_text(x0+cell_w/2, y0+18, text=tkr, fill=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 9, "bold"))
            preview.create_text(x0+cell_w/2, y0+36, text=f"{val:+.2f}%", fill=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 9))

        opts_outer, _ = make_card(grid, "Options", "Build your best strategy", width=340, height=220)
        opts_outer.grid(row=1, column=2, sticky="nsew", padx=(12, 0), pady=12)

        # Bottom spacer
        tk.Frame(home_frame, bg=DEEP_SEA_THEME['primary_bg'], height=8).pack(fill=tk.X)

    def open_stock_layout():
        symbol = simpledialog.askstring(
            "Enter Stock Symbol",
            "Enter the stock ticker symbol (e.g., MSFT, AAPL, TSLA):",
            parent=root
        )
        if symbol:
            open_stock_layout_with_symbol(symbol.upper())
    
    def open_stock_layout_with_symbol(symbol):
        # Clear only main content and create trading interface
        clear_main_content()
        # Get stock data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        
        if len(hist) < 2:
            messagebox.showerror("Error", f"Not enough data for {symbol}.")
            create_home_page()  # Return to home page
            return
        
        prev_close = hist['Close'].iloc[0]
        latest_close = hist['Close'].iloc[1]
        percent_gain = ((latest_close - prev_close) / prev_close) * 100
        ta_signal = get_ta_signal(symbol)
        rsi_val = get_rsi(symbol)
        if rsi_val is None:
            rsi_val = 0
        analysis = get_in_depth_analysis(symbol)
        
        # Create trading interface with the stock data
        create_trading_interface(symbol, ticker, prev_close, latest_close, percent_gain, ta_signal, rsi_val, analysis)
        
    def create_trading_interface(symbol, ticker, prev_close, latest_close, percent_gain, ta_signal, rsi_val, analysis):
        # Home button bar (without AI panel)
        top_control_bar = tk.Frame(main_content, bg=DEEP_SEA_THEME['secondary_bg'], height=40)
        top_control_bar.pack(side=tk.TOP, fill=tk.X)
        top_control_bar.pack_propagate(False)
        
        # Home button on the left
        home_btn = tk.Button(
            top_control_bar,
            text="HOME",
            command=create_home_page,
            font=("Segoe UI", 11, "bold"),
            bg=DEEP_SEA_THEME['surface_bg'],
            fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['hover'],
            activeforeground=DEEP_SEA_THEME['text_primary'],
            bd=2,
            highlightthickness=0,
            relief="raised",
            padx=15
        )
        home_btn.pack(side=tk.LEFT, padx=10, pady=5)

        # New Window button on the right (multi-instance)
        new_window_btn2 = tk.Button(
            top_control_bar,
            text="New Window",
            command=spawn_new_instance,
            font=("Segoe UI", 10, "bold"),
            bg=DEEP_SEA_THEME['info'],
            fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'],
            activeforeground=DEEP_SEA_THEME['primary_bg'],
            bd=2,
            relief="raised",
            padx=10,
            pady=4
        )
        new_window_btn2.pack(side=tk.RIGHT, padx=10, pady=5)

        # --- Nature-inspired ttk theme ---
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background=DEEP_SEA_THEME['secondary_bg'], borderwidth=0)
        style.configure("TNotebook.Tab",
            background=DEEP_SEA_THEME['surface_bg'],
            foreground=DEEP_SEA_THEME['text_primary'],
            font=("Segoe UI", 13, "bold"),
            padding=[24, 12],
            borderwidth=1
        )
        style.map("TNotebook.Tab",
            background=[("selected", DEEP_SEA_THEME['info']), ("active", DEEP_SEA_THEME['active'])],
            foreground=[("selected", DEEP_SEA_THEME['text_primary']), ("active", DEEP_SEA_THEME['primary_bg'])]
        )
        style.configure("TFrame", background=DEEP_SEA_THEME['secondary_bg'])
        style.configure("TLabel", background=DEEP_SEA_THEME['secondary_bg'], foreground=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 12))
        style.configure("TButton", background=DEEP_SEA_THEME['surface_bg'], foreground=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 11, "bold"), borderwidth=1)
        style.map("TButton",
            background=[("active", DEEP_SEA_THEME['info'])],
            foreground=[("active", DEEP_SEA_THEME['text_primary'])]
        )

        # --- Paned window to allow user to resize chart area vs side panel ---
        paned = tk.PanedWindow(main_content, orient=tk.HORIZONTAL, sashwidth=6, bd=0, relief='flat', bg=DEEP_SEA_THEME['primary_bg'])
        paned.pack(fill=tk.BOTH, expand=1)

        left_area = tk.Frame(paned, bg=DEEP_SEA_THEME['primary_bg'])
        side_panel_width = 260  # slightly wider default
        side_area = tk.Frame(paned, bg=DEEP_SEA_THEME['secondary_bg'], width=side_panel_width)

        paned.add(left_area)
        paned.add(side_area)

        notebook = ttk.Notebook(left_area, style="TNotebook")
        notebook.pack(fill=tk.BOTH, expand=1, padx=0, pady=0)

        # Simple header bar with a plus button to go to startup Home page
        tabs_header = tk.Frame(left_area, bg=DEEP_SEA_THEME['secondary_bg'])
        def go_to_startup_home():
            # Redirect to the main startup Home page (outside of tabs)
            create_home_page()
        plus_btn = tk.Button(
            tabs_header, text="+", command=go_to_startup_home, width=3,
            font=("Segoe UI", 12, "bold"),
            bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['hover'], activeforeground=DEEP_SEA_THEME['text_primary'],
            bd=1, relief="raised"
        )
        plus_btn.pack(side=tk.RIGHT, padx=6, pady=2)
        # Pack header before the notebook so it appears above the tabs row
        tabs_header.pack(fill=tk.X, padx=0, pady=(0, 2), before=notebook)

        # Expose notebook globally for adding News/Heatmap tabs from elsewhere
        global GLOBAL_NOTEBOOK
        GLOBAL_NOTEBOOK = notebook

        show_chart_with_points(symbol, ticker, prev_close, latest_close, percent_gain, ta_signal, rsi_val, analysis, notebook, symbol)

        side_panel = tk.Frame(side_area, bg=DEEP_SEA_THEME['secondary_bg'], bd=0, highlightthickness=0)
        side_panel.pack(fill=tk.BOTH, expand=1)
        side_panel.pack_propagate(False)

        # --- Top icon bar for switching panels ---
        icon_bar = tk.Frame(side_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        icon_bar.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        watchlist_btn = tk.Button(icon_bar, text="*", command=lambda: show_panel("watchlist"),
                                  font=("Segoe UI", 18), bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['success'],
                                  activebackground=DEEP_SEA_THEME['success'], activeforeground=DEEP_SEA_THEME['text_primary'], bd=1, highlightthickness=0, relief="raised")
        watchlist_btn.pack(side=tk.LEFT, padx=(8, 4), pady=0)

        dom_btn = tk.Button(icon_bar, text="=", command=lambda: show_panel("dom"),
                            font=("Segoe UI", 18), bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['text_accent'],
                            activebackground=DEEP_SEA_THEME['info'], activeforeground=DEEP_SEA_THEME['text_primary'], bd=1, highlightthickness=0, relief="raised")
        dom_btn.pack(side=tk.LEFT, padx=4, pady=0)

        notes_btn = tk.Button(icon_bar, text="N", command=lambda: show_panel("notes"),
                              font=("Segoe UI", 18), bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['warning'],
                              activebackground=DEEP_SEA_THEME['warning'], activeforeground=DEEP_SEA_THEME['primary_bg'], bd=1, highlightthickness=0, relief="raised")
        notes_btn.pack(side=tk.LEFT, padx=4, pady=0)

        ai_btn = tk.Button(icon_bar, text="AI", command=lambda: show_panel("ai_quick"),
                          font=("Segoe UI", 14), bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['info'],
                          activebackground=DEEP_SEA_THEME['info'], activeforeground=DEEP_SEA_THEME['text_primary'], bd=1, highlightthickness=0, relief="raised")
        ai_btn.pack(side=tk.LEFT, padx=4, pady=0)

        dynamic_panel = tk.Frame(side_panel, bg=DEEP_SEA_THEME['secondary_bg'], bd=0, highlightthickness=0)
        dynamic_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        panel_states = {}
        expand_buttons = {}
        default_sash_pos = [None]

        def update_expand_icons():
            for key, btn in expand_buttons.items():
                btn.config(text="⤢" if panel_states.get(key, False) else "⇅")

        def toggle_expand(key):
            targets = {
                'watchlist': watchlist_frame,
                'dom': dom_frame,
                'notes': notes_frame,
                'ai_quick': ai_quick_frame,
            }
            frame = targets[key]
            expanded = panel_states.get(key, False)
            if not expanded:
                for k, f in targets.items():
                    if k != key and f.winfo_ismapped():
                        f.forget()
                frame.pack(fill=tk.BOTH, expand=1)
                panel_states[key] = True
                try:
                    if default_sash_pos[0] is None:
                        default_sash_pos[0] = paned.sash_coord(0)[0]
                    total = paned.winfo_width()
                    if total <= 0:
                        paned.update_idletasks()
                        total = paned.winfo_width()
                    new_left = max(420, int(total * 0.45))
                    paned.sash_place(0, new_left, 0)
                except Exception:
                    pass
            else:
                panel_states[key] = False
                show_panel(key)
                try:
                    if default_sash_pos[0] is not None:
                        paned.sash_place(0, default_sash_pos[0], 0)
                except Exception:
                    pass
            update_expand_icons()

        # --- Watchlist UI ---
        watchlist_frame = tk.Frame(dynamic_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        watchlist_header = tk.Frame(watchlist_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        watchlist_header.pack(fill=tk.X, pady=(10,5))
        watchlist_label = tk.Label(watchlist_header, text="* Watchlist", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['success'], font=("Segoe UI", 13, "bold"))
        watchlist_label.pack(side=tk.LEFT, padx=(6,0))
        wl_expand = tk.Button(watchlist_header, text="⇅", command=lambda: toggle_expand('watchlist'),
                      font=("Segoe UI", 10, "bold"), bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'], bd=1, relief="raised", padx=6)
        wl_expand.pack(side=tk.RIGHT, padx=6)
        expand_buttons['watchlist'] = wl_expand

        watchlist_listbox = tk.Listbox(
            watchlist_frame, bg=DEEP_SEA_THEME['accent_bg'], fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 12, "bold"),
            selectbackground=DEEP_SEA_THEME['info'], selectforeground=DEEP_SEA_THEME['text_primary'], activestyle="none", width=18, bd=1, highlightthickness=0, relief="solid"
        )
        watchlist_listbox.pack(fill=tk.BOTH, expand=1, padx=8, pady=5)

        def add_to_watchlist():
            try:
                symbol_input = simpledialog.askstring("Add to Watchlist", "Enter the stock ticker symbol (e.g., MSFT):", parent=root)
                if not symbol_input:
                    return
                symbol_input = symbol_input.strip().upper()
                symbol_input = re.sub(r'[^A-Z0-9\.\-]', '', symbol_input)
                if not symbol_input:
                    messagebox.showerror("Error", "Please enter a valid ticker symbol.")
                    return
                watchlist = load_watchlist()
                if symbol_input not in watchlist:
                    watchlist.append(symbol_input)
                    save_watchlist(watchlist)
                    update_watchlist_listbox(force_update=True)
                else:
                    messagebox.showinfo("Info", f"{symbol_input} is already in your watchlist.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add to watchlist: {e}")

        add_watch_btn = tk.Button(
            watchlist_frame, text="+ Add", command=add_to_watchlist,
            font=("Segoe UI", 10, "bold"), bg=DEEP_SEA_THEME['success'], fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['primary_bg'], bd=2, highlightthickness=0, relief="raised"
        )
        add_watch_btn.pack(pady=(0, 10), fill=tk.X, padx=8)

        watchlist_cache = {}
        last_watchlist_update = 0

        def update_watchlist_listbox(force_update=False):
            nonlocal last_watchlist_update
            current_time = time.time()
            if not force_update and current_time - last_watchlist_update < 30:
                return
            try:
                watchlist_listbox.delete(0, tk.END)
                watchlist = load_watchlist()
                for symbol_item in watchlist:
                    if not force_update and symbol_item in watchlist_cache and current_time - watchlist_cache[symbol_item]['time'] < 60:
                        percent = watchlist_cache[symbol_item]['percent']
                    else:
                        gp = globals().get('get_percent_gain')
                        percent = gp(symbol_item) if callable(gp) else None
                        watchlist_cache[symbol_item] = {'percent': percent, 'time': current_time}
                    if percent is not None:
                        percent_str = f"{percent:+.1f}%"
                        color = DEEP_SEA_THEME['success'] if percent >= 0 else DEEP_SEA_THEME['danger']
                    else:
                        percent_str = "N/A"
                        color = DEEP_SEA_THEME['text_secondary']
                    watchlist_listbox.insert(tk.END, f"{symbol_item} {percent_str}")
                    idx = watchlist_listbox.size() - 1
                    watchlist_listbox.itemconfig(idx, foreground=color)
                last_watchlist_update = current_time
            except Exception:
                pass

        def on_watchlist_double_click(event):
            selection = watchlist_listbox.curselection()
            if not selection:
                return
            idx = selection[0]
            selected_symbol = load_watchlist()[idx]
            open_stock_layout_with_symbol(selected_symbol)

        watchlist_listbox.bind("<Double-Button-1>", on_watchlist_double_click)
        update_watchlist_listbox()

        # --- DOM UI ---
        dom_frame = tk.Frame(dynamic_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        dom_header = tk.Frame(dom_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        dom_header.pack(fill=tk.X, pady=(10,5))
        dom_label = tk.Label(dom_header, text="[DOM]", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['info'], font=("Segoe UI", 13, "bold"))
        dom_label.pack(side=tk.LEFT, padx=(6,0))
        dom_expand = tk.Button(dom_header, text="⇅", command=lambda: toggle_expand('dom'),
                       font=("Segoe UI", 10, "bold"), bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'], bd=1, relief="raised", padx=6)
        dom_expand.pack(side=tk.RIGHT, padx=6)
        expand_buttons['dom'] = dom_expand

        dom_table = tk.Frame(dom_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        dom_table.pack(fill=tk.BOTH, expand=1, padx=4, pady=4)
        dom_columns = ["Bid Size", "Bid Price", "Ask Price", "Ask Size"]
        header_labels = []
        for i, col in enumerate(dom_columns):
            lbl = tk.Label(dom_table, text=col, bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 11, "bold"), width=9, borderwidth=1, relief="solid")
            lbl.grid(row=0, column=i, sticky="nsew", padx=1, pady=1)
            header_labels.append(lbl)
        # Pre-create cell labels for performance
        order_rows = 10
        cell_labels = []  # list of lists [ [bid_size,bid,ask,ask_size], ...]
        for r in range(1, order_rows+1):
            row_cells = []
            for c in range(4):
                base_bg = DEEP_SEA_THEME['accent_bg']
                lbl = tk.Label(dom_table, text="", bg=base_bg, fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 11, "bold"), width=9, borderwidth=1, relief="solid")
                lbl.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
                row_cells.append(lbl)
            cell_labels.append(row_cells)
        for i in range(len(dom_columns)):
            dom_table.grid_columnconfigure(i, weight=1)

        # Flag to stop updates when panel destroyed
        dom_active = True

        def build_simulated_book(base_price: float):
            # Create symmetrical ladders around base price
            book = []
            spread = max(0.01, base_price * 0.0005)  # 5 bps minimal
            for i in range(order_rows):
                # Price levels further away get larger size sometimes
                bid_price = base_price - (i * spread)
                ask_price = base_price + (i * spread)
                bid_size = random.randint(10, 500) * (1 + int(i/5))
                ask_size = random.randint(10, 500) * (1 + int(i/5))
                book.append((bid_size, bid_price, ask_price, ask_size))
            return book

        def fetch_last_price():
            try:
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            except Exception:
                pass
            # Fallback to previous latest_close if available else random baseline
            return latest_close if latest_close else 100.0

        def update_dom():
            if not dom_frame.winfo_exists():
                return
            # Only refresh when DOM panel is visible to reduce load
            if dom_frame.winfo_ismapped():
                base = fetch_last_price()
                book = build_simulated_book(base)
                for i, (bid_sz, bid_p, ask_p, ask_sz) in enumerate(book):
                    if i >= len(cell_labels):
                        break
                    cells = cell_labels[i]
                    # Update text
                    cells[0].config(text=str(bid_sz), fg=DEEP_SEA_THEME['success'])
                    cells[1].config(text=f"{bid_p:.2f}", fg=DEEP_SEA_THEME['success'])
                    cells[2].config(text=f"{ask_p:.2f}", fg=DEEP_SEA_THEME['danger'])
                    cells[3].config(text=str(ask_sz), fg=DEEP_SEA_THEME['danger'])
            # Schedule next update
            dom_frame.after(1000, update_dom)

        update_dom()

        # --- Notes UI ---
        notes_frame = tk.Frame(dynamic_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        notes_header = tk.Frame(notes_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        notes_header.pack(fill=tk.X, pady=(10,5))
        notes_label = tk.Label(notes_header, text="Notes", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['warning'], font=("Segoe UI", 13, "bold"))
        notes_label.pack(side=tk.LEFT, padx=(6,0))
        notes_expand = tk.Button(notes_header, text="⇅", command=lambda: toggle_expand('notes'),
                     font=("Segoe UI", 10, "bold"), bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'], bd=1, relief="raised", padx=6)
        notes_expand.pack(side=tk.RIGHT, padx=6)
        expand_buttons['notes'] = notes_expand
        notes_text = tk.Text(notes_frame, bg=DEEP_SEA_THEME['accent_bg'], fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 12, "bold"), height=10, width=18, 
                              insertbackground=DEEP_SEA_THEME['active'], bd=1, highlightthickness=0, selectbackground=DEEP_SEA_THEME['info'], 
                              selectforeground=DEEP_SEA_THEME['text_primary'], wrap=tk.WORD, relief="solid")
        notes_text.pack(fill=tk.BOTH, expand=1, padx=8, pady=5)

        # --- AI Quick Panel ---
        ai_quick_frame = tk.Frame(dynamic_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        ai_header = tk.Frame(ai_quick_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        ai_header.pack(fill=tk.X, pady=(10,5))
        ai_quick_label = tk.Label(ai_header, text="AI Analysis", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['info'], font=("Segoe UI", 13, "bold"))
        ai_quick_label.pack(side=tk.LEFT, padx=(6,0))
        ai_expand = tk.Button(ai_header, text="⇅", command=lambda: toggle_expand('ai_quick'),
                      font=("Segoe UI", 10, "bold"), bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'], bd=1, relief="raised", padx=6)
        ai_expand.pack(side=tk.RIGHT, padx=6)
        expand_buttons['ai_quick'] = ai_expand
        ai_summary_text = tk.Text(
            ai_quick_frame, 
            bg="#263238",
            fg="#ECEFF1",
            font=("Segoe UI", 10), 
            height=12, 
            width=18,
            insertbackground="#1976D2",
            bd=1, 
            highlightthickness=0, 
            selectbackground="#1976D2",
            selectforeground="#FFFFFF", 
            wrap=tk.WORD,
            relief="solid"
        )
        ai_summary_text.pack(fill=tk.BOTH, expand=1, padx=8, pady=5)
        try:
            short_analysis = analysis[:500] + "..." if len(analysis) > 500 else analysis
            ai_summary_text.insert(tk.END, short_analysis)
            ai_summary_text.config(state=tk.DISABLED)
        except Exception:
            ai_summary_text.insert(tk.END, "AI analysis will appear here...")
            ai_summary_text.config(state=tk.DISABLED)

        def open_full_ai():
            show_panel("watchlist")

        full_ai_btn = tk.Button(
            ai_quick_frame,
            text="View Full Analysis",
            command=open_full_ai,
            font=("Segoe UI", 9, "bold"),
            bg="#1976D2",
            fg="#E3F2FD",
            activebackground="#2196F3",
            activeforeground="#FFFFFF",
            bd=2,
            highlightthickness=0,
            relief="raised"
        )
        full_ai_btn.pack(pady=(5, 10), fill=tk.X, padx=8)

        def get_notes_filename(note_symbol):
            return get_user_data_path(f"notes_{note_symbol.upper()}.txt")

        def load_notes(note_symbol):
            try:
                with open(get_notes_filename(note_symbol), "r") as f:
                    return f.read()
            except Exception:
                return ""

        def save_notes(note_symbol, content):
            with open(get_notes_filename(note_symbol), "w") as f:
                f.write(content)

        def save_current_notes():
            try:
                current_tab = notebook.select()
                tab_text = notebook.tab(current_tab, "text")
                content = notes_text.get("1.0", tk.END)
                save_notes(tab_text, content)
            except Exception:
                pass

        def on_tab_changed(event):
            save_current_notes()
            if notes_frame.winfo_ismapped():
                try:
                    current_tab = notebook.select()
                    tab_text = notebook.tab(current_tab, "text")
                    notes_text.delete("1.0", tk.END)
                    notes_text.insert(tk.END, load_notes(tab_text))
                except Exception:
                    notes_text.delete("1.0", tk.END)

        notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

        def show_panel(panel):
            for widget in dynamic_panel.winfo_children():
                widget.pack_forget()
            if panel == "watchlist":
                watchlist_frame.pack(fill=tk.BOTH, expand=1)
            elif panel == "dom":
                dom_frame.pack(fill=tk.BOTH, expand=1)
            elif panel == "notes":
                notes_frame.pack(fill=tk.BOTH, expand=1)
            elif panel == "ai_quick":
                ai_quick_frame.pack(fill=tk.BOTH, expand=1)

        show_panel("watchlist")

        def add_tab():
            new_symbol = simpledialog.askstring("Add Tab", "Enter the stock ticker symbol (e.g., MSFT):", parent=root)
            if not new_symbol:
                return
            new_ticker = yf.Ticker(new_symbol)
            new_hist = new_ticker.history(period="2d")
            if len(new_hist) < 2:
                messagebox.showerror("Error", f"Not enough data for {new_symbol}.")
                return
            new_prev_close = new_hist['Close'].iloc[0]
            new_latest_close = new_hist['Close'].iloc[1]
            new_percent_gain = ((new_latest_close - new_prev_close) / new_prev_close) * 100
            new_ta_signal = get_ta_signal(new_symbol)
            new_rsi = get_rsi(new_symbol)
            new_analysis = get_in_depth_analysis(new_symbol)
            show_chart_with_points(new_symbol, new_ticker, new_prev_close, new_latest_close, new_percent_gain, new_ta_signal, new_rsi, new_analysis, notebook, new_symbol)

    # Start with home page
    create_home_page()
    root.mainloop()

# Start the application with tabs and watchlist
if __name__ == "__main__":
    # Check for auto-login first
    auto_user = check_auto_login()
    if auto_user:
        print(f"Auto-login successful for {auto_user}")
        user = auto_user
    else:
        user = require_login()
        if not user:
            sys.exit(0)
    
    # Set the global current user for per-user data isolation
    current_user = user
    
    # Initialize data if needed (optional - data will be fetched when charts are opened)
    try:
        initialize_market_data()
        initialize_technical_analysis()
        initialize_openai_analysis()
    except Exception as e:
        print(f"Warning: Could not initialize all data: {e}")
    
    # Start the main application
    main_tabbed_chart()
