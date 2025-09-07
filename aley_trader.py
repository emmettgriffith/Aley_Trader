#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
# Import pandas_ta with error handling
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pandas_ta not available ({e}). Technical indicators will be simplified.")
    PANDAS_TA_AVAILABLE = False
    ta = None
import requests
import re
import time
import openai
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import messagebox, ttk
import tkinter.simpledialog as simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import requests
import re
import time
import yfinance as yf
import pandas as pd
import numpy as np  # <-- Add this import
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

# Load environment variables from .env file in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Also try loading from current directory as fallback
if not os.getenv('OPENAI_API_KEY'):
    load_dotenv('.env')

# Debug: Print if environment variables are loaded
if os.getenv('OPENAI_API_KEY'):
    print("OpenAI API key loaded successfully")
else:
    print("Warning: OpenAI API key not found in environment")

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

            print(f"{symbol} previous close: {prev_close}")
            print(f"{symbol} latest close: {latest_close}")
            print(f"Percent gain: {percent_gain:.2f}%")
    except Exception as e:
        print(f"Error initializing market data: {e}")
        # Set defaults
        prev_close = 100.0
        latest_close = 102.0
        percent_gain = 2.0

# --- Technical Analysis Signal Section ---
TAAPI_KEY = os.getenv("TAAPI_KEY", "YOUR_TAAPI_KEY")  # Load from environment variable

def get_ta_signal(symbol):
    url = f"https://api.taapi.io/summary"
    params = {
        "secret": TAAPI_KEY,
        "exchange": "NASDAQ",
        "symbol": symbol,
        "interval": "1d"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "recommendation" in data:
            recommendation = data["recommendation"].lower()
            if "buy" in recommendation:
                return "Buy"
            elif "sell" in recommendation:
                return "Sell"
            elif "hold" in recommendation:
                return "Hold"
            else:
                return f"Signal: {data['recommendation']}"
        else:
            return f"No signal available: {data}"
    except Exception as e:
        return f"Error fetching TA signal: {e}"

def get_rsi(symbol):
    url = f"https://api.taapi.io/rsi"
    params = {
        "secret": TAAPI_KEY,
        "exchange": "NASDAQ",
        "symbol": symbol,
        "interval": "1d"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "value" in data:
            return float(data["value"])
        else:
            return None
    except Exception as e:
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
        print(f"Technical Analysis Signal for {symbol}: {ta_signal}")

        # Display RSI and overbought status
        rsi = get_rsi(symbol)
        if rsi is not None:
            print(f"RSI for {symbol}: {rsi:.2f}")
            if rsi > 70:
                print("Warning: The stock is overbought (RSI > 70)!")
            elif rsi < 30:
                print("The stock is oversold (RSI < 30).")
        else:
            print("Could not retrieve RSI value.")
    except Exception as e:
        print(f"Error initializing technical analysis: {e}")
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
    financials = ticker_obj.financials.replace({np.nan: np.nan})  # Fix NaN to nan
    balance_sheet = ticker_obj.balance_sheet.replace({np.nan: np.nan})  # Fix NaN to nan
    cashflow = ticker_obj.cashflow.replace({np.nan: np.nan})  # Fix NaN to nan

    # Prepare a summary for OpenAI
    prompt = (
        f"Provide an in-depth financial analysis of {info.get('longName', symbol)} ({symbol}). "
        f"Include recent financial performance, balance sheet health, cash flow, and long-term growth prospects. "
        f"Here is some data:\n"
        f"Company Info: {info}\n"
        f"Financials: {financials.to_dict() if not financials.empty else 'N/A'}\n"
        f"Balance Sheet: {balance_sheet.to_dict() if not balance_sheet.empty else 'N/A'}\n"
        f"Cash flow: {cashflow.to_dict() if not cashflow.empty else 'N/A'}\n"
        f"Percent gain over last 2 days: {percent_gain:.2f}%\n"
        f"Technical Analysis Signal: {ta_signal}\n"
        f"RSI: {rsi if rsi is not None else 'N/A'}"
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
        print("\n--- In-Depth Financial Analysis (OpenAI) ---")
        print(analysis)
    except Exception as e:
        print(f"Error initializing OpenAI analysis: {e}")
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
            # Optimized data fetching - simplified and faster
            include_prepost = tf in ["1m", "5m", "10m", "15m", "30m", "1h"]
            
            print(f"Fetching {tf} data with config: {tf_config}")
            
            # Try multiple data fetching strategies for robustness
            chart_hist = None
            strategies = [
                # Strategy 1: Original request
                {"period": tf_config["period"], "interval": tf_config["interval"], "prepost": include_prepost, "repair": True},
                # Strategy 2: Shorter period, same interval
                {"period": "5d", "interval": tf_config["interval"], "prepost": include_prepost, "repair": True},
                # Strategy 3: Even shorter period
                {"period": "2d", "interval": tf_config["interval"], "prepost": include_prepost, "repair": True},
                # Strategy 4: Daily fallback
                {"period": "30d", "interval": "1d", "prepost": False, "repair": True}
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    print(f"  Trying strategy {i+1}: {strategy}")
                    temp_hist = ticker.history(**strategy)
                    if not temp_hist.empty and len(temp_hist) > 0:
                        chart_hist = temp_hist
                        print(f"  Success! Got {len(chart_hist)} data points")
                        break
                    else:
                        print(f"  Strategy {i+1} returned empty data")
                except Exception as strategy_error:
                    print(f"  Strategy {i+1} failed: {strategy_error}")
                    continue
            
            if chart_hist is None or chart_hist.empty:
                print(f"All strategies failed for {tf}, creating minimal test data...")
                # Create minimal test data to prevent crash
                from datetime import datetime, timedelta
                dates = pd.date_range(start=datetime.now() - timedelta(days=5), end=datetime.now(), freq='D')
                chart_hist = pd.DataFrame({
                    'Open': [100] * len(dates),
                    'High': [105] * len(dates), 
                    'Low': [95] * len(dates),
                    'Close': [102] * len(dates),
                    'Volume': [1000000] * len(dates)
                }, index=dates)
                print(f"Created {len(chart_hist)} test data points")
            
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
            num_bins = min(20, len(chart_hist) // 3)  # Reduced bins
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
            
            return chart_hist, price_bins, volume_profile, high_vol_price, low_vol_price
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
                
                return chart_hist, price_bins, volume_profile, high_vol_price, low_vol_price
            except Exception as fallback_error:
                print(f"Fallback failed: {fallback_error}")
                # Return empty data structure to prevent crash
                empty_df = pd.DataFrame({
                    'Date': [mdates.date2num(datetime.now())],
                    'Open': [100], 'High': [100], 'Low': [100], 'Close': [100], 'Volume': [0]
                })
                return empty_df, np.array([99, 101]), np.array([0]), 100, 100

    def draw_chart(chart_type="Candlestick"):
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

        # --- Integrate Drawing Toolbar and Manager ---
        drawing_manager = DrawingManager(None)  # Pass chart canvas when available
        drawing_toolbar = DrawingToolbar(frame, drawing_manager.set_tool)
        nonlocal refresh_timer
        
        # Cancel any existing refresh timer
        if refresh_timer:
            frame.after_cancel(refresh_timer)
            refresh_timer = None
        
        # Clear all existing widgets in chart frame
        for widget in frame.winfo_children():
            widget.destroy()
            
        # Add control bar at the top of chart frame
        control_frame = tk.Frame(frame, bg=DEEP_SEA_THEME['secondary_bg'])
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Chart type selector (new overflow chart integration)
        chart_selector_frame = tk.Frame(control_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        chart_selector_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # Chart type dropdown
        current_chart_type = tk.StringVar(value="Candlestick")
        chart_types = [
            "Candlestick",
            "Volume Profile - Order Book",
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
        

        # Indicator bar (top right, empty for now, space for future indicators)
        indicator_bar = tk.Frame(control_frame, bg=DEEP_SEA_THEME['surface_bg'], width=180, height=32, highlightbackground=DEEP_SEA_THEME['border'], highlightthickness=1)
        indicator_bar.pack(side=tk.RIGHT, padx=(30, 4), pady=2)
        indicator_bar.pack_propagate(False)

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
            chart_hist, price_bins, volume_profile, high_vol_price, low_vol_price = fetch_chart_data()
            tf = current_timeframe.get()
            has_prepost = tf in ["1m", "5m", "10m", "15m", "20m", "30m", "1h"]
            status_text = f"OK {tf.upper()} - {len(chart_hist)} bars"
            if has_prepost:
                status_text += " (Extended Hours)"
        except Exception as e:
            error_label = tk.Label(frame, text=f"Error loading chart: {e}", bg=DEEP_SEA_THEME['primary_bg'], fg=DEEP_SEA_THEME['danger'], font=("Segoe UI", 12))
            error_label.pack(expand=1)
            return
            
        # Handle different chart types
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
                        ylabel="Value",
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
                    
                    status_label.config(text=f"OK {chart_type} loaded successfully")
                    return
                else:
                    print("Warning: No data available for overflow chart, falling back to candlestick")
                    
            except Exception as e:
                print(f"Error creating overflow chart: {e}")
                print("Falling back to candlestick chart")
        
        # Create matplotlib figure with enhanced interactivity (for candlestick charts)
        plt.close('all')  # Close any existing figures
        fig = plt.figure(figsize=(width, height), facecolor=DEEP_SEA_THEME['primary_bg'])
        
        # Enhanced grid specification for better layout
        gs = fig.add_gridspec(
            4, 2,
            width_ratios=[8, 1],  # More space for main chart
            height_ratios=[12, 1, 3, 1],  # Better proportions
            wspace=0.02,
            hspace=0.05
        )
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0, 0])
        # RSI subplot  
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
        # Volume profile
        ax_vp = fig.add_subplot(gs[0:3, 1], sharey=ax1)
        
        # Enable interactive navigation
        ax1.set_navigate(True)
        ax2.set_navigate(True)
        
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
                        edgecolor = '#37474F'  # Blue-gray edge
                        alpha = 1.0
                        linewidth = 0.5
                        regular_count += 1
                    else:
                        color = '#8BC34A' if close_val >= open_val else '#FF5722'  # Light green/deep orange for pre/post
                        edgecolor = '#FFC107'  # Amber edge for pre/post market
                        alpha = 0.9
                        linewidth = 1.0
                        prepost_count += 1
                    
                    # Simplified rectangle drawing
                    ax1.add_patch(Rectangle(
                        (date_val - width_candle/2, min(open_val, close_val)),
                        width_candle,
                        abs(close_val - open_val),
                        color=color,
                        ec=edgecolor,
                        linewidth=linewidth,
                        alpha=alpha
                    ))
                    
                    # Simplified wick drawing
                    ax1.plot([date_val, date_val], [low_val, high_val], 
                            color=color, linewidth=1.0, alpha=alpha)
            else:
                # Regular candlesticks for daily timeframe
                for i, (date_val, open_val, close_val, high_val, low_val) in enumerate(
                    zip(dates, opens, closes, highs, lows)
                ):
                    color = candle_up if close_val >= open_val else candle_down
                    regular_count += 1
                    
                    ax1.add_patch(Rectangle(
                        (date_val - width_candle/2, min(open_val, close_val)),
                        width_candle,
                        abs(close_val - open_val),
                        color=color,
                        ec='#37474F',  # Blue-gray edge
                        linewidth=0.5
                    ))
                    
                    ax1.plot([date_val, date_val], [low_val, high_val], 
                            color=color, linewidth=1.0)
        
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
        ax2.set_facecolor(DEEP_SEA_THEME['primary_bg'])
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
        
        # Simplified RSI subplot - only if data exists
        if 'RSI' in chart_hist.columns and not chart_hist['RSI'].isna().all():
            ax2.plot(chart_hist['Date'], chart_hist['RSI'], color=rsi_color, linewidth=1.5)
            ax2.axhline(70, color=overbought_color, linestyle='--', linewidth=1)
            ax2.axhline(30, color=oversold_color, linestyle='--', linewidth=1)
            ax2.set_ylabel("RSI", color='#fff')
            ax2.set_ylim(0, 100)
            ax2.set_title("RSI", color='#fff', fontsize=12)
            ax2.tick_params(axis='x', colors='#aaa')
            ax2.tick_params(axis='y', colors='#aaa')
            ax2.grid(True, color=grid_color, linestyle='-', linewidth=0.3)
        else:
            # Hide RSI subplot if no data
            ax2.set_visible(False)

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
        # Fallback to base64 encoding (less secure but functional)
        credentials = f"{username}|{password}"
        return base64.b64encode(credentials.encode()).decode()
    except Exception:
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
    except Exception:
        pass
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
        # Placeholder pill rows inside screeners
        pills = tk.Frame(scr, bg=DEEP_SEA_THEME['card_bg'])
        pills.pack(fill=tk.X, padx=14, pady=8)
        for txt, col in [("AAPL", DEEP_SEA_THEME['accent_bg']), ("TSLA", DEEP_SEA_THEME['accent_bg']), ("GOOGL", DEEP_SEA_THEME['accent_bg']), ("WMT", DEEP_SEA_THEME['accent_bg'])]:
            tk.Label(pills, text=f"  {txt}  ", bg=col, fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 10, "bold"), bd=0).pack(side=tk.LEFT, padx=6, pady=6)

        cal_outer, _ = make_card(grid, "Calendars", "Explore the world's financial events", width=340, height=220)
        cal_outer.grid(row=0, column=2, sticky="nsew", padx=(12, 0), pady=(0, 12))

        # Row 2: News Flow (left), Heatmaps (center), Options (right)
        news_outer, news_card = make_card(grid, "News Flow", "US stock headlines", width=340, height=220)
        news_outer.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=12)

        heat_outer, _ = make_card(grid, "Heatmaps", "See the full picture for global markets", width=340, height=220)
        heat_outer.grid(row=1, column=1, sticky="nsew", padx=12, pady=12)

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

        notebook = ttk.Notebook(main_content, style="TNotebook")
        notebook.pack(fill=tk.BOTH, expand=1, side=tk.LEFT, padx=0, pady=0)

        # Add initial tab for the selected symbol
        show_chart_with_points(symbol, ticker, prev_close, latest_close, percent_gain, ta_signal, rsi_val, analysis, notebook, symbol)

        # --- Thin TradingView-style side panel ---
        side_panel_width = 220
        side_panel = tk.Frame(main_content, bg=DEEP_SEA_THEME['secondary_bg'], bd=0, highlightthickness=0, width=side_panel_width)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y)
        side_panel.pack_propagate(False)

        # --- Top icon bar for switching panels ---
        icon_bar = tk.Frame(side_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        icon_bar.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        # Use unicode icons for a TradingView feel
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

        # --- Dynamic panel for watchlist, DOM, notes ---
        dynamic_panel = tk.Frame(side_panel, bg=DEEP_SEA_THEME['secondary_bg'], bd=0, highlightthickness=0)
        dynamic_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        for widget in dynamic_panel.winfo_children():
            widget.pack_forget()

        # --- Watchlist UI ---
        watchlist_frame = tk.Frame(dynamic_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        watchlist_label = tk.Label(watchlist_frame, text="* Watchlist", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['success'], font=("Segoe UI", 13, "bold"))
        watchlist_label.pack(pady=(10, 5))

        watchlist_listbox = tk.Listbox(
            watchlist_frame, bg=DEEP_SEA_THEME['accent_bg'], fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 12, "bold"),
            selectbackground=DEEP_SEA_THEME['info'], selectforeground=DEEP_SEA_THEME['text_primary'], activestyle="none", width=18, bd=1, highlightthickness=0, relief="solid"
        )
        watchlist_listbox.pack(fill=tk.BOTH, expand=1, padx=8, pady=5)

        # Button to add to watchlist
        def add_to_watchlist():
            try:
                symbol_input = simpledialog.askstring("Add to Watchlist", "Enter the stock ticker symbol (e.g., MSFT):", parent=root)
                if not symbol_input:
                    return
                symbol_input = symbol_input.strip().upper()
                # Remove any non-alphanumeric characters except dots and dashes
                symbol_input = re.sub(r'[^A-Z0-9\.\-]', '', symbol_input)
                
                if not symbol_input:
                    messagebox.showerror("Error", "Please enter a valid ticker symbol.")
                    return
                    
                watchlist = load_watchlist()
                if symbol_input not in watchlist:
                    watchlist.append(symbol_input)
                    save_watchlist(watchlist)
                    print(f"Added {symbol_input} to watchlist")
                    # Force update the watchlist display immediately
                    update_watchlist_listbox(force_update=True)
                else:
                    messagebox.showinfo("Info", f"{symbol_input} is already in your watchlist.")
            except Exception as e:
                print(f"Error adding to watchlist: {e}")
                messagebox.showerror("Error", f"Failed to add to watchlist: {e}")

        add_watch_btn = tk.Button(
            watchlist_frame, text="+ Add", command=add_to_watchlist,
            font=("Segoe UI", 10, "bold"), bg=DEEP_SEA_THEME['success'], fg=DEEP_SEA_THEME['text_primary'],
            activebackground=DEEP_SEA_THEME['active'], activeforeground=DEEP_SEA_THEME['primary_bg'], bd=2, highlightthickness=0, relief="raised"
        )
        add_watch_btn.pack(pady=(0, 10), fill=tk.X, padx=8)

        # Optimized helper to update watchlist with cached data
        watchlist_cache = {}
        last_watchlist_update = 0
        
        def update_watchlist_listbox(force_update=False):
              # ensure time is available for periodic refresh
            nonlocal last_watchlist_update
            current_time = time.time()
            
            # Only update every 30 seconds to reduce lag, unless forced
            if not force_update and current_time - last_watchlist_update < 30:
                return
                
            try:
                watchlist_listbox.delete(0, tk.END)
                watchlist = load_watchlist()
                print(f"Updating watchlist display with {len(watchlist)} items")
                
                for symbol_item in watchlist:
                    # Use cached data if available and recent, or if forced update skip cache
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
                print("Watchlist display updated successfully")
            except Exception as e:
                print(f"Error updating watchlist display: {e}")

        # Double-click to open tab from watchlist
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
        dom_label = tk.Label(dom_frame, text="[DOM]", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['info'], font=("Segoe UI", 13, "bold"))
        dom_label.pack(pady=(10, 5))

        dom_table = tk.Frame(dom_frame, bg=DEEP_SEA_THEME['secondary_bg'])
        dom_table.pack(fill=tk.BOTH, expand=1, padx=4, pady=4)

        dom_columns = ["Bid Size", "Bid Price", "Ask Price", "Ask Size"]
        for i, col in enumerate(dom_columns):
            lbl = tk.Label(dom_table, text=col, bg=DEEP_SEA_THEME['surface_bg'], fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 11, "bold"), width=8, borderwidth=1, relief="solid")
            lbl.grid(row=0, column=i, sticky="nsew", padx=1, pady=1)

        dom_data = [
            [120,  99.50, 99.55, 110],
            [100,  99.45, 99.60, 90],
            [80,   99.40, 99.65, 70],
            [60,   99.35, 99.70, 50],
            [40,   99.30, 99.75, 30],
            [20,   99.25, 99.80, 10],
        ]
        for r, row in enumerate(dom_data, start=1):
            for c, val in enumerate(row):
                fg = DEEP_SEA_THEME['success'] if c == 1 else DEEP_SEA_THEME['danger'] if c == 2 else DEEP_SEA_THEME['text_primary']
                bg = DEEP_SEA_THEME['accent_bg']
                tk.Label(dom_table, text=val, bg=bg, fg=fg, font=("Segoe UI", 11, "bold"), width=8, borderwidth=1, relief="solid").grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
        for i in range(len(dom_columns)):
            dom_table.grid_columnconfigure(i, weight=1)

        # --- Notes UI ---
        notes_frame = tk.Frame(dynamic_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        notes_label = tk.Label(notes_frame, text="Notes", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['warning'], font=("Segoe UI", 13, "bold"))
        notes_label.pack(pady=(10, 5))

        notes_text = tk.Text(notes_frame, bg=DEEP_SEA_THEME['accent_bg'], fg=DEEP_SEA_THEME['text_primary'], font=("Segoe UI", 12, "bold"), height=10, width=18, 
                            insertbackground=DEEP_SEA_THEME['active'], bd=1, highlightthickness=0, selectbackground=DEEP_SEA_THEME['info'], 
                            selectforeground=DEEP_SEA_THEME['text_primary'], wrap=tk.WORD, relief="solid")
        notes_text.pack(fill=tk.BOTH, expand=1, padx=8, pady=5)

        # --- AI Quick Panel ---
        ai_quick_frame = tk.Frame(dynamic_panel, bg=DEEP_SEA_THEME['secondary_bg'])
        ai_quick_label = tk.Label(ai_quick_frame, text="AI Analysis", bg=DEEP_SEA_THEME['secondary_bg'], fg=DEEP_SEA_THEME['info'], font=("Segoe UI", 13, "bold"))
        ai_quick_label.pack(pady=(10, 5))

        # AI summary display
        ai_summary_text = tk.Text(
            ai_quick_frame, 
            bg="#263238",  # Dark blue-gray
            fg="#ECEFF1",  # Light gray text
            font=("Segoe UI", 10), 
            height=12, 
            width=18,
            insertbackground="#1976D2",  # Ocean blue cursor
            bd=1, 
            highlightthickness=0, 
            selectbackground="#1976D2",  # Ocean blue selection
            selectforeground="#FFFFFF", 
            wrap=tk.WORD,
            relief="solid"
        )
        ai_summary_text.pack(fill=tk.BOTH, expand=1, padx=8, pady=5)

        # Add truncated analysis to AI quick panel
        try:
            # Get first 500 characters of analysis
            short_analysis = analysis[:500] + "..." if len(analysis) > 500 else analysis
            ai_summary_text.insert(tk.END, short_analysis)
            ai_summary_text.config(state=tk.DISABLED)
        except:
            ai_summary_text.insert(tk.END, "AI analysis will appear here...")
            ai_summary_text.config(state=tk.DISABLED)

        # Full analysis button
        def open_full_ai():
            # Switch to main chart tab first, then toggle to AI
            show_panel("watchlist")  # Reset side panel
            # Note: Full AI analysis accessible via main chart toggle button
            
        full_ai_btn = tk.Button(
            ai_quick_frame,
            text="View Full Analysis",
            command=open_full_ai,
            font=("Segoe UI", 9, "bold"),
            bg="#1976D2",  # Ocean blue
            fg="#E3F2FD",  # Light blue
            activebackground="#2196F3",  # Lighter ocean blue
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
            dynamic_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            if panel == "watchlist":
                watchlist_frame.pack(fill=tk.BOTH, expand=1)
            elif panel == "dom":
                dom_frame.pack(fill=tk.BOTH, expand=1)
            elif panel == "notes":
                notes_frame.pack(fill=tk.BOTH, expand=1)
            elif panel == "ai_quick":
                ai_quick_frame.pack(fill=tk.BOTH, expand=1)

        # Show watchlist by default
        show_panel("watchlist")

        # --- Tabs logic ---
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