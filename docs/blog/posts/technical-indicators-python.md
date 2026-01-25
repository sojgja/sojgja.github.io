---
date: 2025-01-25
authors:
  - soigia
categories: [Technical Analysis, Python]
title: Technical Indicators với Python - RSI, MACD, Bollinger Bands
description: >
  Implement các technical indicators phổ biến với Python: RSI, MACD, Bollinger Bands, và cách sử dụng chúng trong trading strategies.
---

# Technical Indicators với Python

Technical indicators là công cụ quan trọng trong trading. Trong bài viết này, chúng ta sẽ implement các indicators phổ biến với Python.

<!-- more -->

## Các Indicators phổ biến

1. **RSI** - Relative Strength Index
2. **MACD** - Moving Average Convergence Divergence
3. **Bollinger Bands**
4. **Moving Averages**
5. **Stochastic Oscillator**

## Bước 1: RSI Implementation

```python
# indicators/rsi.py
import pandas as pd
import numpy as np

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
```

## Bước 2: MACD Implementation

```python
# indicators/macd.py
def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram
```

## Bước 3: Bollinger Bands

```python
# indicators/bollinger.py
def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band
```

## Kết luận

Technical indicators giúp:
- Identify trends
- Find entry/exit points
- Manage risk
- Improve trading decisions

Use indicators wisely in your strategies!
