"""ai_patterns_lite.py
Lightweight AI-assisted pattern recognition utilities for time-series.

This is a simplified version that provides core functionality without heavy dependencies.
Features:
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)  
  - Rule-based signals (MA crossovers, RSI zones, breakouts, volatility regimes)
  - Basic anomaly detection using statistical methods
  - Clean, typed APIs designed to integrate with pandas DataFrames

Dependencies:
  numpy, pandas (scipy optional)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import zscore  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# -----------------------------
# Indicators
# -----------------------------
def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(period, min_periods=period).mean()
    avg_loss = down.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = sma(close, window)
    sd = close.rolling(window, min_periods=window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return upper, ma, lower

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(period, min_periods=period).mean()

# -----------------------------
# Rule-based detections
# -----------------------------
def detect_ma_crossover(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    return cross.fillna(False)

def detect_rsi_zones(rsi_series: pd.Series, overbought: float = 70, oversold: float = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "rsi_overbought": (rsi_series >= overbought),
        "rsi_oversold": (rsi_series <= oversold),
    }).fillna(False)

def detect_breakout(close: pd.Series, lookback: int = 20, min_atr_mult: float = 0.5,
                    high: Optional[pd.Series] = None, low: Optional[pd.Series] = None) -> pd.Series:
    rolling_max = close.rolling(lookback, min_periods=lookback).max()
    base_breakout = close > rolling_max.shift(1)
    if high is not None and low is not None:
        # add a volatility gate to reduce noise
        local_atr = atr(high, low, close, period=14)
        gate = local_atr > (min_atr_mult * local_atr.rolling(lookback, min_periods=lookback).mean())
        return (base_breakout & gate).fillna(False)
    return base_breakout.fillna(False)

def detect_volatility_regime(close: pd.Series, window: int = 20) -> pd.Series:
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(window, min_periods=window).std()
    if _HAS_SCIPY:
        volz = pd.Series(zscore(vol.dropna()), index=vol.dropna().index)
        volz = volz.reindex(vol.index)
    else:
        # manual z-score fallback
        m = vol.mean()
        s = vol.std()
        volz = (vol - m) / s if s and not math.isclose(s, 0.0) else vol * 0.0
    return volz

# -----------------------------
# Simple anomaly detection (statistical)
# -----------------------------
def detect_anomalies_statistical(df: pd.DataFrame, feature_cols: List[str], 
                                 threshold: float = 3.0) -> pd.Series:
    """Simple statistical anomaly detection using z-scores"""
    # Calculate z-scores for each feature
    z_scores = pd.DataFrame(index=df.index)
    
    for col in feature_cols:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                z_scores[col] = np.abs((df[col] - mean_val) / std_val)
    
    # Mark as anomaly if any feature exceeds threshold
    if len(z_scores.columns) > 0:
        max_z = z_scores.max(axis=1)
        anomaly = max_z > threshold
    else:
        anomaly = pd.Series(False, index=df.index)
    
    return anomaly.fillna(False)

# -----------------------------
# Feature engineering
# -----------------------------
def compute_features(df: pd.DataFrame,
                     price_col: str = "close",
                     high_col: str = "high",
                     low_col: str = "low",
                     volume_col: Optional[str] = "volume") -> pd.DataFrame:
    close = df[price_col]
    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = close.pct_change(fill_method=None)
    feats["ret_5"] = close.pct_change(5, fill_method=None)
    feats["ret_20"] = close.pct_change(20, fill_method=None)
    feats["sma_20"] = sma(close, 20)
    feats["sma_50"] = sma(close, 50)
    feats["ema_20"] = ema(close, 20)
    feats["rsi_14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    feats["macd"] = macd_line
    feats["macd_sig"] = signal_line
    feats["macd_hist"] = hist
    u, ma, l = bollinger(close)
    feats["bb_upper"] = u
    feats["bb_mid"] = ma
    feats["bb_lower"] = l
    feats["vol_z20"] = detect_volatility_regime(close, 20)
    if all(c in df.columns for c in [high_col, low_col]):
        feats["atr_14"] = atr(df[high_col], df[low_col], close, 14)
    if volume_col and volume_col in df.columns:
        feats["vol_1"] = df[volume_col]
        feats["vol_z"] = (df[volume_col] - df[volume_col].rolling(20).mean()) / (df[volume_col].rolling(20).std())
    return feats

# -----------------------------
# Orchestrator
# -----------------------------
@dataclass
class PatternConfig:
    fast_ma: int = 20
    slow_ma: int = 50
    breakout_lookback: int = 20
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    anomaly_threshold: float = 3.0

def analyze_patterns(df: pd.DataFrame,
                     price_col: str = "close",
                     high_col: str = "high",
                     low_col: str = "low",
                     volume_col: Optional[str] = "volume",
                     config: Optional[PatternConfig] = None) -> Dict[str, pd.Series]:
    if config is None:
        config = PatternConfig()

    close = df[price_col]
    signals: Dict[str, pd.Series] = {}

    # Core signals
    signals["ma_bull_cross"] = detect_ma_crossover(close, config.fast_ma, config.slow_ma)
    rsi_series = rsi(close, 14)
    rsi_df = detect_rsi_zones(rsi_series, config.rsi_overbought, config.rsi_oversold)
    signals.update(rsi_df.to_dict(orient="series"))
    signals["breakout"] = detect_breakout(close, config.breakout_lookback,
                                          min_atr_mult=0.5,
                                          high=df.get(high_col),
                                          low=df.get(low_col))

    # Statistical anomaly detection on engineered features
    feats = compute_features(df, price_col, high_col, low_col, volume_col)
    feature_cols = [c for c in feats.columns if feats[c].dtype != "O"]
    signals["anomaly"] = detect_anomalies_statistical(feats, feature_cols, config.anomaly_threshold)

    return signals

# -----------------------------
# Demo generator
# -----------------------------
def demo_ohlcv(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # random walk close
    close = np.cumsum(rng.normal(0, 1, size=n)) * 0.8 + 100
    high = close + rng.normal(0.5, 0.4, size=n).clip(min=0)
    low = close - rng.normal(0.5, 0.4, size=n).clip(min=0)
    open_ = close + rng.normal(0, 0.2, size=n)
    volume = rng.integers(1000, 5000, size=n)
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"time": t, "open": open_, "high": high, "low": low, "close": close, "volume": volume}).set_index("time")


if __name__ == "__main__":
    df = demo_ohlcv()
    signals = analyze_patterns(df)
    # Show the last 10 rows of boolean signals
    preview = pd.DataFrame({k: v for k, v in signals.items()}).tail(10)
    print(preview)
