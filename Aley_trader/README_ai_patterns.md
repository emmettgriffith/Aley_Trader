# AI-Assisted Pattern Recognition (Python)

This module adds intelligent pattern detection for time-series (e.g., OHLCV market data or sensor data). It mixes robust **rule-based signals** with **ML-powered anomaly detection** and includes a **supervised ML template** for your own labels.

## Install
```bash
pip install numpy pandas scikit-learn scipy
```

## What it provides
- Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- Signals: MA crossovers, RSI overbought/oversold, breakouts, volatility regime z-score
- Anomaly detection: IsolationForest on engineered features
- Supervised training template: RandomForest pipeline with scaler + metrics
- Demo data: `demo_ohlcv()`

## Usage
```python
import pandas as pd
from ai_patterns import analyze_patterns, compute_features, train_classifier_labeled, demo_ohlcv

# Get data (or load your own OHLCV DataFrame with index=time)
df = demo_ohlcv()

# 1) Rule-based + anomaly signals
signals = analyze_patterns(df)
signals_df = pd.DataFrame(signals)
print(signals_df.tail())

# 2) Features for ML
feats = compute_features(df)

# 3) Optional: Supervised learning if you have labels
# Suppose you add a column df['label'] with your target pattern (0/1)
# feature_cols = [c for c in feats.columns if feats[c].dtype != 'O']
# model, report = train_classifier_labeled(pd.concat([feats, df['label']], axis=1), feature_cols, 'label')
# print(report)
```

## Data expectations
- Index: time-like (datetime). If not, it still works but time-based logic is simpler with proper indexing.
- Columns required for full functionality:
  - `close` (must)
  - `high`, `low` (for ATR, better breakouts)
  - `volume` (optional for volume features)

## Extend it
- Add new rule-based detectors in the **Rule-based detections** section.
- Swap/extend the ML model in `train_classifier_labeled` (e.g., XGBoost, SVM).
- Use your own labeled patterns to train a classifier and then apply `model.predict()` on new features.
