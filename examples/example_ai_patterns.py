#!/usr/bin/env python3
"""
Example: AI-Assisted Pattern Recognition for Trading Data

This script demonstrates how to use the ai_patterns module for:
1. Generating demo OHLCV data
2. Running pattern analysis (rule-based + ML anomaly detection)
3. Computing features for supervised learning
4. Training a classifier (when labels are available)
"""

import pandas as pd
import sys
import os

# Add parent directory to path to import ai_patterns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_patterns import (
    analyze_patterns, 
    compute_features, 
    train_classifier_labeled, 
    demo_ohlcv,
    PatternConfig
)

def main():
    print("🔍 AI-Assisted Pattern Recognition Demo")
    print("=" * 50)
    
    # 1. Generate demo OHLCV data
    print("\n📊 Generating demo OHLCV data...")
    df = demo_ohlcv(n=200, seed=42)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # 2. Run pattern analysis
    print("\n🎯 Running pattern analysis...")
    
    # Custom configuration
    config = PatternConfig(
        fast_ma=10,
        slow_ma=30,
        breakout_lookback=15,
        rsi_overbought=75.0,
        rsi_oversold=25.0,
        anomaly_contamination=0.05
    )
    
    signals = analyze_patterns(df, config=config)
    signals_df = pd.DataFrame(signals)
    
    print("\nPattern signals detected:")
    for signal_name, signal_series in signals.items():
        count = signal_series.sum()
        percentage = (count / len(signal_series)) * 100
        print(f"  {signal_name:20}: {count:4d} occurrences ({percentage:5.1f}%)")
    
    print("\nLast 10 signal rows:")
    print(signals_df.tail(10))
    
    # 3. Compute features for ML
    print("\n🧠 Computing ML features...")
    feats = compute_features(df, price_col="close", high_col="high", low_col="low", volume_col="volume")
    print(f"Features shape: {feats.shape}")
    print(f"Feature columns: {list(feats.columns)}")
    
    # Display feature statistics
    print("\nFeature statistics (last 5 rows):")
    print(feats.tail().round(4))
    
    # 4. Demo supervised learning (create synthetic labels)
    print("\n🤖 Supervised Learning Demo...")
    
    # Create synthetic labels based on future returns
    future_returns = df['close'].pct_change(5).shift(-5)  # 5-day forward return
    df['label'] = (future_returns > 0.02).astype(int)  # 1 if >2% gain in next 5 days
    
    # Combine features with labels
    ml_data = pd.concat([feats, df[['label']]], axis=1).dropna()
    
    if len(ml_data) > 50:  # Need sufficient data for training
        feature_cols = [c for c in feats.columns if feats[c].dtype != 'O']
        
        try:
            model, report = train_classifier_labeled(
                ml_data, 
                feature_cols, 
                'label', 
                test_size=0.3,
                random_state=42
            )
            
            print(f"✓ Model trained successfully!")
            print(f"Features used: {len(feature_cols)}")
            print(f"Training samples: {len(ml_data)}")
            print("\nClassification Report:")
            print(report)
            
            # Show feature importance (Random Forest)
            if hasattr(model.named_steps['rf'], 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.named_steps['rf'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(importance_df.head(10).round(4))
                
        except Exception as e:
            print(f"⚠ Model training failed: {e}")
    else:
        print("⚠ Insufficient data for supervised learning demo")
    
    # 5. Trading signals summary
    print("\n📈 Trading Signals Summary")
    print("=" * 30)
    
    # Recent signals
    recent_signals = signals_df.tail(20)
    active_signals = []
    
    for col in recent_signals.columns:
        if recent_signals[col].iloc[-1]:  # If signal is active in latest period
            active_signals.append(col)
    
    if active_signals:
        print(f"🚨 Active signals in latest period: {', '.join(active_signals)}")
    else:
        print("✅ No active signals in latest period")
    
    # Signal frequency over last 50 periods
    recent_50 = signals_df.tail(50)
    print(f"\nSignal frequency (last 50 periods):")
    for col in recent_50.columns:
        freq = recent_50[col].sum()
        print(f"  {col:20}: {freq:2d}/50 ({freq*2:3.0f}%)")
    
    print(f"\n✅ Pattern analysis complete!")
    print(f"💡 Tip: Integrate these signals into your trading strategy")
    print(f"🔗 See README_ai_patterns.md for more details")

if __name__ == "__main__":
    main()
