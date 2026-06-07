---
id: strategy
title: Trading Strategy Engine
sidebar_label: Trading Strategy
sidebar_position: 2
description: Build and backtest trading strategies — signal generators, risk management, position sizing, and strategy evaluation metrics.
keywords: [trading, strategy, backtest, signal, risk-management, position-sizing, sharpe-ratio]
---

# Trading Strategy Engine

Framework for designing, backtesting, and deploying automated trading strategies.

## Signal Generator

```python
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df['sma_fast'] = df['close'].rolling(20).mean()
    df['sma_slow'] = df['close'].rolling(50).mean()
    df['rsi'] = compute_rsi(df['close'], 14)

    df['signal'] = np.where(
        (df['sma_fast'] > df['sma_slow']) & (df['rsi'] < 70), 1,   # BUY
        np.where(
            (df['sma_fast'] < df['sma_slow']) & (df['rsi'] > 30), -1,  # SELL
            0
        )
    )
    return df
```

## Position Sizing (Kelly Criterion)

```python
def kelly_position(win_rate: float, win_loss_ratio: float, capital: float) -> float:
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    return capital * max(kelly_fraction * 0.25, 0)  # quarter-Kelly
```

## Risk Management

```python
class RiskManager:
    def __init__(self, max_drawdown: float = 0.15, max_position: float = 0.05):
        self.max_drawdown = max_drawdown
        self.max_position = max_position
        self.peak = 0

    def can_trade(self, equity: float, position_size: float) -> bool:
        self.peak = max(self.peak, equity)
        drawdown = (self.peak - equity) / self.peak
        if drawdown > self.max_drawdown:
            return False
        return position_size / equity <= self.max_position
```

## Backtest Metrics

```python
def evaluate(returns: pd.Series) -> dict:
    return {
        'sharpe': returns.mean() / returns.std() * np.sqrt(252),
        'max_drawdown': (returns.cummax() - returns).max(),
        'win_rate': (returns > 0).mean(),
        'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
    }
```
