---
date: 2025-01-25
authors:
  - soigia
categories: [Algorithmic Trading, Backend Development]
title: Backtesting Trading Strategies với Python - Từ Zero đến Hero
description: >
  Hướng dẫn xây dựng hệ thống backtesting hoàn chỉnh để đánh giá hiệu quả của trading strategies trước khi deploy thực tế.
---

# Backtesting Trading Strategies với Python

Backtesting là quá trình test một trading strategy trên dữ liệu lịch sử để đánh giá hiệu quả trước khi trade thật. Trong bài viết này, chúng ta sẽ xây dựng một backtesting engine hoàn chỉnh từ đầu.

<!-- more -->

## Tại sao cần Backtesting?

1. **Validate Strategy**: Kiểm tra xem strategy có hoạt động không
2. **Risk Assessment**: Đánh giá rủi ro và drawdown
3. **Optimization**: Tìm parameters tốt nhất
4. **Confidence**: Tăng tự tin trước khi trade thật

## Kiến trúc Backtesting Engine

```
┌──────────────┐
│ Historical   │
│ Data         │
└──────┬───────┘
       │
┌──────▼──────────────┐
│  Data Loader        │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Strategy Engine    │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Portfolio Manager  │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Performance        │
│  Analyzer           │
└─────────────────────┘
```

## Bước 1: Data Loader

```python
# backtesting/data_loader.py
import pandas as pd
import yfinance as yf
from typing import Optional
import logging

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_from_yahoo(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Load data từ Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return None
            
            # Rename columns
            df.columns = [col.lower() for col in df.columns]
            df.reset_index(inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
    
    def load_from_csv(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load data từ CSV file"""
        try:
            df = pd.read_csv(filepath)
            
            # Ensure required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Need: {required_cols}")
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return None
    
    def load_from_binance(self, symbol: str, interval: str = '1h', limit: int = 1000):
        """Load data từ Binance API"""
        from binance.client import Client
        
        try:
            client = Client()
            klines = client.get_historical_klines(
                symbol, interval, limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamp
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            self.logger.error(f"Error loading from Binance: {e}")
            return None
```

## Bước 2: Strategy Base Class

```python
# backtesting/strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class Signal:
    action: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0-1
    price: float
    timestamp: pd.Timestamp

class BaseStrategy(ABC):
    """Base class cho tất cả strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals từ data"""
        pass
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        return data
```

## Bước 3: Implement Moving Average Strategy

```python
# backtesting/strategies/ma_strategy.py
from backtesting.strategy import BaseStrategy
import pandas as pd
import numpy as np

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, short_window=10, long_window=30):
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        df = data.copy()
        
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals"""
        df = self.calculate_indicators(data.copy())
        df['signal'] = 0
        df['position'] = 0
        
        # Golden Cross: Buy signal
        df.loc[
            (df['sma_short'] > df['sma_long']) & 
            (df['sma_short'].shift(1) <= df['sma_long'].shift(1)) &
            (df['rsi'] < 70),
            'signal'
        ] = 1
        
        # Death Cross: Sell signal
        df.loc[
            (df['sma_short'] < df['sma_long']) & 
            (df['sma_short'].shift(1) >= df['sma_long'].shift(1)) &
            (df['rsi'] > 30),
            'signal'
        ] = -1
        
        # Forward fill position
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return df
```

## Bước 4: Portfolio Manager

```python
# backtesting/portfolio.py
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'LONG' or 'SHORT'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

class Portfolio:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: quantity}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.current_trade: Optional[Trade] = None
    
    def execute_trade(self, signal: int, price: float, timestamp: pd.Timestamp, 
                     symbol: str = 'ASSET', position_size: float = 0.1):
        """Execute a trade based on signal"""
        
        # Close existing position if opposite signal
        if self.current_trade:
            if (signal == 1 and self.current_trade.side == 'SHORT') or \
               (signal == -1 and self.current_trade.side == 'LONG'):
                self.close_position(price, timestamp)
        
        # Open new position
        if signal == 1:  # Buy signal
            if not self.current_trade or self.current_trade.side != 'LONG':
                self.open_long(price, timestamp, symbol, position_size)
        elif signal == -1:  # Sell signal
            if not self.current_trade or self.current_trade.side != 'SHORT':
                self.open_short(price, timestamp, symbol, position_size)
    
    def open_long(self, price: float, timestamp: pd.Timestamp, 
                 symbol: str, position_size: float):
        """Open long position"""
        investment = self.cash * position_size
        quantity = investment / price
        
        if investment <= self.cash:
            self.cash -= investment
            self.positions[symbol] = quantity
            
            self.current_trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price,
                exit_price=None,
                quantity=quantity,
                side='LONG'
            )
    
    def open_short(self, price: float, timestamp: pd.Timestamp,
                  symbol: str, position_size: float):
        """Open short position (simplified - assumes we can short)"""
        investment = self.cash * position_size
        quantity = investment / price
        
        # For short, we "borrow" and sell
        self.cash += investment
        self.positions[symbol] = -quantity
        
        self.current_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            side='SHORT'
        )
    
    def close_position(self, price: float, timestamp: pd.Timestamp):
        """Close current position"""
        if not self.current_trade:
            return
        
        self.current_trade.exit_time = timestamp
        self.current_trade.exit_price = price
        
        # Calculate PnL
        if self.current_trade.side == 'LONG':
            pnl = (price - self.current_trade.entry_price) * self.current_trade.quantity
            self.cash += price * self.current_trade.quantity
        else:  # SHORT
            pnl = (self.current_trade.entry_price - price) * self.current_trade.quantity
            self.cash -= price * self.current_trade.quantity
        
        self.current_trade.pnl = pnl
        self.current_trade.pnl_pct = (
            (price - self.current_trade.entry_price) / self.current_trade.entry_price * 100
            if self.current_trade.side == 'LONG'
            else (self.current_trade.entry_price - price) / self.current_trade.entry_price * 100
        )
        
        self.trades.append(self.current_trade)
        self.current_trade = None
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            qty * current_price for qty in self.positions.values()
        )
        return self.cash + position_value
    
    def update_equity_curve(self, current_price: float):
        """Update equity curve"""
        total_value = self.get_portfolio_value(current_price)
        self.equity_curve.append(total_value)
```

## Bước 5: Backtesting Engine

```python
# backtesting/engine.py
import pandas as pd
from backtesting.portfolio import Portfolio
from backtesting.strategy import BaseStrategy
from typing import Dict
import logging

class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, initial_capital: float = 10000):
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.logger = logging.getLogger(__name__)
    
    def run(self, data: pd.DataFrame, position_size: float = 0.1) -> Dict:
        """Run backtest"""
        self.logger.info(f"Starting backtest with {self.strategy.name}")
        
        # Generate signals
        df = self.strategy.generate_signals(data.copy())
        
        # Run simulation
        for idx, row in df.iterrows():
            signal = row['signal']
            price = row['close']
            
            # Update equity curve
            self.portfolio.update_equity_curve(price)
            
            # Execute trade if signal
            if signal != 0:
                self.portfolio.execute_trade(
                    signal, price, idx, 
                    position_size=position_size
                )
        
        # Close any open positions at the end
        if self.portfolio.current_trade:
            final_price = df.iloc[-1]['close']
            self.portfolio.close_position(final_price, df.index[-1])
        
        # Calculate performance metrics
        results = self.calculate_performance(df)
        
        return results
    
    def calculate_performance(self, data: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        equity_series = pd.Series(self.portfolio.equity_curve, index=data.index)
        
        # Total return
        total_return = (
            (equity_series.iloc[-1] - self.portfolio.initial_capital) / 
            self.portfolio.initial_capital * 100
        )
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        winning_trades = [t for t in self.portfolio.trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / len(self.portfolio.trades) * 100 if self.portfolio.trades else 0
        
        # Average win/loss
        wins = [t.pnl for t in winning_trades if t.pnl]
        losses = [t.pnl for t in self.portfolio.trades if t.pnl and t.pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        return {
            'strategy': self.strategy.name,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.portfolio.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_value': equity_series.iloc[-1],
            'equity_curve': equity_series,
            'trades': self.portfolio.trades
        }

# Usage
if __name__ == '__main__':
    from backtesting.data_loader import DataLoader
    from backtesting.strategies.ma_strategy import MovingAverageStrategy
    
    # Load data
    loader = DataLoader()
    data = loader.load_from_yahoo('BTC-USD', period='1y')
    
    if data is not None:
        # Create strategy
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        
        # Run backtest
        engine = BacktestEngine(strategy, initial_capital=10000)
        results = engine.run(data)
        
        print(f"Strategy: {results['strategy']}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
```

## Visualization

```python
# backtesting/visualizer.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_backtest_results(results: Dict, data: pd.DataFrame):
    """Visualize backtest results"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Price and signals
    axes[0].plot(data.index, data['close'], label='Price', alpha=0.7)
    
    buy_signals = data[data['signal'] == 1]
    sell_signals = data[data['signal'] == -1]
    
    axes[0].scatter(buy_signals.index, buy_signals['close'], 
                   color='green', marker='^', s=100, label='Buy')
    axes[0].scatter(sell_signals.index, sell_signals['close'],
                   color='red', marker='v', s=100, label='Sell')
    axes[0].set_title('Price and Trading Signals')
    axes[0].legend()
    axes[0].grid(True)
    
    # Equity curve
    equity = results['equity_curve']
    axes[1].plot(equity.index, equity.values, label='Equity Curve')
    axes[1].axhline(y=results['initial_capital'], color='r', 
                   linestyle='--', label='Initial Capital')
    axes[1].set_title('Equity Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    # Drawdown
    cumulative = (1 + equity.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    axes[2].fill_between(drawdown.index, drawdown.values, 0, 
                        color='red', alpha=0.3)
    axes[2].set_title('Drawdown')
    axes[2].set_ylabel('Drawdown %')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Best Practices

1. **Survivorship Bias**: Chỉ test trên assets còn tồn tại
2. **Look-ahead Bias**: Không dùng thông tin tương lai
3. **Transaction Costs**: Tính phí giao dịch
4. **Slippage**: Giá thực tế khác giá lý thuyết
5. **Overfitting**: Tránh optimize quá mức

## Kết luận

Với backtesting engine này, bạn có thể:
- Test nhiều strategies khác nhau
- Optimize parameters
- Đánh giá risk và performance
- So sánh strategies

Trong bài tiếp theo, chúng ta sẽ xây dựng optimization framework để tìm parameters tốt nhất.
