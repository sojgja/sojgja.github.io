---
date: 2025-01-25
authors:
  - soigia
categories: [Algorithmic Trading, Risk Management]
title: Risk Management và Automation - Bảo vệ vốn của bạn
description: >
  Xây dựng hệ thống quản lý rủi ro tự động với position sizing, stop-loss, take-profit và portfolio management.
---

# Risk Management và Automation

Risk management là yếu tố quan trọng nhất trong trading. Dù strategy tốt đến đâu, nếu không quản lý risk tốt, bạn sẽ thua lỗ. Trong bài viết này, chúng ta sẽ xây dựng một hệ thống risk management tự động hoàn chỉnh.

<!-- more -->

## Tại sao Risk Management quan trọng?

1. **Bảo vệ vốn**: Tránh mất toàn bộ vốn trong một trade
2. **Consistency**: Đảm bảo trading nhất quán
3. **Emotion Control**: Loại bỏ cảm xúc khỏi trading
4. **Long-term Success**: Duy trì lợi nhuận dài hạn

## Các thành phần Risk Management

1. **Position Sizing**: Tính toán kích thước position
2. **Stop Loss**: Giới hạn thua lỗ
3. **Take Profit**: Chốt lời
4. **Portfolio Risk**: Quản lý risk toàn portfolio
5. **Correlation**: Tránh over-exposure

## Bước 1: Position Sizing Calculator

```python
# risk_management/position_sizing.py
from typing import Dict, Optional
from enum import Enum
import math

class RiskMethod(Enum):
    FIXED_PERCENT = "fixed_percent"  # Fixed % of capital
    KELLY = "kelly"  # Kelly Criterion
    VOLATILITY = "volatility"  # Based on volatility
    EQUAL_WEIGHT = "equal_weight"  # Equal weight all positions

class PositionSizer:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
    
    def calculate_fixed_percent(
        self, 
        capital: float, 
        risk_percent: float = 0.02,
        entry_price: float = 0,
        stop_loss_price: float = 0
    ) -> float:
        """
        Calculate position size using fixed risk percentage
        
        Args:
            capital: Current capital
            risk_percent: Risk per trade (e.g., 0.02 = 2%)
            entry_price: Entry price
            stop_loss_price: Stop loss price
        
        Returns:
            Position size in base currency
        """
        if entry_price == 0 or stop_loss_price == 0:
            # Simple fixed percentage
            return capital * risk_percent
        
        # Risk amount
        risk_amount = capital * risk_percent
        
        # Price difference
        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        # Position size
        position_size = risk_amount / price_diff
        
        return position_size
    
    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion
        
        f* = (p * b - q) / b
        where:
            f* = fraction of capital to bet
            p = win probability
            q = loss probability (1 - p)
            b = win/loss ratio
        """
        if avg_loss == 0:
            return 0
        
        p = win_rate / 100
        q = 1 - p
        b = avg_win / abs(avg_loss)
        
        kelly = (p * b - q) / b
        
        # Use fractional Kelly (half) for safety
        return max(0, kelly / 2)
    
    def calculate_volatility_based(
        self,
        capital: float,
        entry_price: float,
        volatility: float,  # Daily volatility as decimal
        risk_multiplier: float = 2.0
    ) -> float:
        """
        Calculate position size based on volatility
        
        Position size inversely proportional to volatility
        """
        # Risk per trade
        risk_percent = 0.02  # 2% base risk
        
        # Adjust for volatility
        adjusted_risk = risk_percent / (volatility * risk_multiplier)
        adjusted_risk = min(adjusted_risk, 0.05)  # Cap at 5%
        
        risk_amount = capital * adjusted_risk
        position_size = risk_amount / entry_price
        
        return position_size
    
    def calculate_equal_weight(
        self,
        capital: float,
        num_positions: int,
        max_positions: int = 10
    ) -> float:
        """
        Calculate position size for equal weight portfolio
        """
        if num_positions >= max_positions:
            return 0
        
        position_value = capital / max_positions
        return position_value
```

## Bước 2: Stop Loss Manager

```python
# risk_management/stop_loss.py
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class StopLoss:
    entry_price: float
    stop_loss_price: float
    take_profit_price: Optional[float]
    trailing_stop: bool = False
    trailing_percent: float = 0.02
    current_stop: Optional[float] = None

class StopLossManager:
    def __init__(self):
        self.active_stops: Dict[str, StopLoss] = {}
        self.logger = logging.getLogger(__name__)
    
    def set_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_percent: float = 0.02,
        take_profit_percent: Optional[float] = None,
        trailing: bool = False,
        trailing_percent: float = 0.02
    ):
        """Set stop loss for a position"""
        stop_loss_price = entry_price * (1 - stop_loss_percent)
        take_profit_price = None
        
        if take_profit_percent:
            take_profit_price = entry_price * (1 + take_profit_percent)
        
        stop = StopLoss(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            trailing_stop=trailing,
            trailing_percent=trailing_percent,
            current_stop=stop_loss_price
        )
        
        self.active_stops[symbol] = stop
        self.logger.info(f"Stop loss set for {symbol}: {stop_loss_price}")
    
    def update_trailing_stop(self, symbol: str, current_price: float):
        """Update trailing stop loss"""
        if symbol not in self.active_stops:
            return
        
        stop = self.active_stops[symbol]
        
        if not stop.trailing_stop:
            return
        
        # Calculate new stop
        if stop.current_stop is None:
            stop.current_stop = stop.stop_loss_price
        
        # If price moves up, move stop up
        if current_price > stop.entry_price:
            new_stop = current_price * (1 - stop.trailing_percent)
            
            # Only move stop up, never down
            if new_stop > stop.current_stop:
                stop.current_stop = new_stop
                self.logger.info(f"Trailing stop updated for {symbol}: {stop.current_stop}")
    
    def check_stop_loss(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if stop loss or take profit is hit
        
        Returns:
            'STOP_LOSS', 'TAKE_PROFIT', or None
        """
        if symbol not in self.active_stops:
            return None
        
        stop = self.active_stops[symbol]
        
        # Update trailing stop
        if stop.trailing_stop:
            self.update_trailing_stop(symbol, current_price)
        
        # Check stop loss
        stop_price = stop.current_stop if stop.trailing_stop else stop.stop_loss_price
        
        if current_price <= stop_price:
            self.logger.warning(f"Stop loss hit for {symbol} at {current_price}")
            del self.active_stops[symbol]
            return 'STOP_LOSS'
        
        # Check take profit
        if stop.take_profit_price and current_price >= stop.take_profit_price:
            self.logger.info(f"Take profit hit for {symbol} at {current_price}")
            del self.active_stops[symbol]
            return 'TAKE_PROFIT'
        
        return None
    
    def remove_stop_loss(self, symbol: str):
        """Remove stop loss"""
        if symbol in self.active_stops:
            del self.active_stops[symbol]
```

## Bước 3: Portfolio Risk Manager

```python
# risk_management/portfolio_risk.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import logging

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: str  # 'LONG' or 'SHORT'
    
    @property
    def value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl(self) -> float:
        if self.side == 'LONG':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def pnl_pct(self) -> float:
        if self.side == 'LONG':
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

class PortfolioRiskManager:
    def __init__(self, max_portfolio_risk: float = 0.10, max_correlation: float = 0.7):
        """
        Args:
            max_portfolio_risk: Maximum total portfolio risk (e.g., 0.10 = 10%)
            max_correlation: Maximum correlation between positions
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.positions: Dict[str, Position] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_position(self, position: Position):
        """Add position to portfolio"""
        self.positions[position.symbol] = position
    
    def remove_position(self, symbol: str):
        """Remove position from portfolio"""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def get_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        return sum(pos.value for pos in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """Calculate total PnL"""
        return sum(pos.pnl for pos in self.positions.values())
    
    def get_portfolio_risk(self, capital: float) -> float:
        """Calculate current portfolio risk as % of capital"""
        total_exposure = self.get_total_exposure()
        return total_exposure / capital if capital > 0 else 0
    
    def can_add_position(self, position_value: float, capital: float) -> bool:
        """Check if can add new position"""
        current_exposure = self.get_total_exposure()
        new_exposure = current_exposure + position_value
        
        portfolio_risk = new_exposure / capital if capital > 0 else 0
        
        return portfolio_risk <= self.max_portfolio_risk
    
    def get_position_weights(self) -> Dict[str, float]:
        """Get position weights in portfolio"""
        total_value = self.get_total_exposure()
        
        if total_value == 0:
            return {}
        
        return {
            symbol: pos.value / total_value
            for symbol, pos in self.positions.items()
        }
    
    def check_concentration_risk(self, max_weight: float = 0.30) -> List[str]:
        """Check for concentration risk (single position too large)"""
        weights = self.get_position_weights()
        concentrated = [
            symbol for symbol, weight in weights.items()
            if weight > max_weight
        ]
        
        if concentrated:
            self.logger.warning(f"Concentration risk: {concentrated}")
        
        return concentrated
    
    def calculate_var(self, confidence: float = 0.95, lookback: int = 252) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Simplified VaR calculation
        """
        if not self.positions:
            return 0
        
        # Get historical returns (simplified)
        # In practice, you'd use actual historical data
        returns = [pos.pnl_pct / 100 for pos in self.positions.values()]
        
        if not returns:
            return 0
        
        # Calculate VaR
        var = np.percentile(returns, (1 - confidence) * 100)
        
        return abs(var)
    
    def get_risk_summary(self, capital: float) -> Dict:
        """Get comprehensive risk summary"""
        return {
            'total_positions': len(self.positions),
            'total_exposure': self.get_total_exposure(),
            'portfolio_risk_pct': self.get_portfolio_risk(capital) * 100,
            'total_pnl': self.get_total_pnl(),
            'total_pnl_pct': (self.get_total_pnl() / capital) * 100 if capital > 0 else 0,
            'position_weights': self.get_position_weights(),
            'concentration_risk': self.check_concentration_risk(),
            'var_95': self.calculate_var(0.95)
        }
```

## Bước 4: Complete Risk Management System

```python
# risk_management/risk_manager.py
from risk_management.position_sizing import PositionSizer, RiskMethod
from risk_management.stop_loss import StopLossManager
from risk_management.portfolio_risk import PortfolioRiskManager, Position
import logging

class RiskManager:
    """Complete risk management system"""
    
    def __init__(self, initial_capital: float = 10000):
        self.capital = initial_capital
        self.position_sizer = PositionSizer(initial_capital)
        self.stop_loss_manager = StopLossManager()
        self.portfolio_manager = PortfolioRiskManager()
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_method: RiskMethod = RiskMethod.FIXED_PERCENT,
        risk_percent: float = 0.02
    ) -> float:
        """Calculate position size"""
        if risk_method == RiskMethod.FIXED_PERCENT:
            return self.position_sizer.calculate_fixed_percent(
                self.capital,
                risk_percent,
                entry_price,
                stop_loss_price
            )
        else:
            # Default to fixed percent
            return self.position_sizer.calculate_fixed_percent(
                self.capital,
                risk_percent,
                entry_price,
                stop_loss_price
            )
    
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str = 'LONG',
        stop_loss_percent: float = 0.02,
        take_profit_percent: Optional[float] = None,
        trailing_stop: bool = False
    ) -> bool:
        """Open a new position with risk management"""
        position_value = quantity * entry_price
        
        # Check if can add position
        if not self.portfolio_manager.can_add_position(position_value, self.capital):
            self.logger.warning(f"Cannot add position {symbol}: portfolio risk limit")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            side=side
        )
        
        # Add to portfolio
        self.portfolio_manager.add_position(position)
        
        # Set stop loss
        self.stop_loss_manager.set_stop_loss(
            symbol,
            entry_price,
            stop_loss_percent,
            take_profit_percent,
            trailing_stop
        )
        
        self.logger.info(f"Position opened: {symbol} {quantity} @ {entry_price}")
        return True
    
    def update_position(self, symbol: str, current_price: float):
        """Update position price and check stop loss"""
        if symbol not in self.portfolio_manager.positions:
            return None
        
        # Update position price
        position = self.portfolio_manager.positions[symbol]
        position.current_price = current_price
        
        # Check stop loss
        stop_result = self.stop_loss_manager.check_stop_loss(symbol, current_price)
        
        if stop_result:
            # Close position
            self.close_position(symbol, current_price, stop_result)
            return stop_result
        
        return None
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'MANUAL'):
        """Close a position"""
        if symbol not in self.portfolio_manager.positions:
            return
        
        position = self.portfolio_manager.positions[symbol]
        position.current_price = exit_price
        
        pnl = position.pnl
        pnl_pct = position.pnl_pct
        
        # Update capital
        self.capital += pnl
        
        # Remove from portfolio
        self.portfolio_manager.remove_position(symbol)
        self.stop_loss_manager.remove_stop_loss(symbol)
        
        self.logger.info(
            f"Position closed: {symbol} @ {exit_price} "
            f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%) Reason: {reason}"
        )
    
    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report"""
        risk_summary = self.portfolio_manager.get_risk_summary(self.capital)
        
        return {
            'capital': self.capital,
            'active_positions': len(self.portfolio_manager.positions),
            **risk_summary
        }

# Usage example
if __name__ == '__main__':
    risk_manager = RiskManager(initial_capital=10000)
    
    # Open position
    entry_price = 50000
    stop_loss = entry_price * 0.98  # 2% stop loss
    
    quantity = risk_manager.calculate_position_size(
        entry_price,
        stop_loss,
        risk_percent=0.02
    )
    
    risk_manager.open_position(
        'BTCUSDT',
        entry_price,
        quantity,
        stop_loss_percent=0.02,
        take_profit_percent=0.04,
        trailing_stop=True
    )
    
    # Update position
    risk_manager.update_position('BTCUSDT', 51000)
    
    # Get risk report
    report = risk_manager.get_risk_report()
    print(report)
```

## Best Practices

1. **Never risk more than 2% per trade**
2. **Use stop loss always**
3. **Diversify positions**
4. **Monitor correlation**
5. **Review risk regularly**
6. **Keep position sizes consistent**

## Kết luận

Với hệ thống risk management này, bạn có thể:
- Tự động tính position size
- Quản lý stop loss và take profit
- Monitor portfolio risk
- Bảo vệ vốn hiệu quả

Nhớ: **Preservation of capital is more important than making profits.**
