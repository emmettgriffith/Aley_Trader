#!/usr/bin/env python3
"""
Debug script to test chart functionality
"""

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def test_simple_chart():
    """Test basic chart functionality"""
    print("Testing simple chart creation...")
    
    # Get some test data
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d", interval="1d")
    
    if hist.empty:
        print("❌ No data retrieved")
        return False
    
    print(f"✅ Data retrieved: {len(hist)} points")
    print(f"   Columns: {list(hist.columns)}")
    print(f"   Date range: {hist.index[0]} to {hist.index[-1]}")
    
    # Create a simple matplotlib chart
    plt.figure(figsize=(10, 6))
    plt.plot(hist.index, hist['Close'], label='Close Price')
    plt.title('AAPL - Test Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save the chart instead of showing it
    plt.savefig('test_chart.png', bbox_inches='tight', dpi=100)
    plt.close()
    
    print("✅ Chart saved as test_chart.png")
    return True

def test_tkinter_chart():
    """Test tkinter integration"""
    print("Testing tkinter chart integration...")
    
    try:
        root = tk.Tk()
        root.title("Chart Test")
        root.geometry("800x600")
        
        # Get test data
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="5d", interval="1d")
        
        if hist.empty:
            print("❌ No data for tkinter test")
            root.destroy()
            return False
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(hist.index, hist['Close'], color='blue', linewidth=2)
        ax.set_title('AAPL - Tkinter Test')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        
        # Create tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        
        print("✅ Tkinter chart created successfully")
        
        # Run for 2 seconds then close
        root.after(2000, root.destroy)
        root.mainloop()
        
        return True
        
    except Exception as e:
        print(f"❌ Tkinter test failed: {e}")
        return False

if __name__ == "__main__":
    print("Chart Debugging Test")
    print("=" * 30)
    
    success1 = test_simple_chart()
    success2 = test_tkinter_chart()
    
    if success1 and success2:
        print("\n✅ All tests passed! Chart functionality should work.")
    else:
        print("\n❌ Some tests failed. There may be an issue with chart functionality.")
