---
sidebar_position: 4
---

# SDK Design Patterns

This document outlines the SDK architecture and design patterns used across our trading system integrations.

## Package Structure

```
sojgja-sdk/
├── core/           # Base classes and interfaces
├── exchange/       # Exchange-specific adapters
├── data/           # Market data client
├── order/          # Order management
├── risk/           # Risk calculation utilities
└── utils/          # Shared helpers
```

## Design Patterns

### 1. Adapter Pattern (Exchange Integration)

```python
from abc import ABC, abstractmethod

class ExchangeAdapter(ABC):
    @abstractmethod
    async def get_balance(self, asset: str) -> float: ...
    
    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult: ...

class BinanceAdapter(ExchangeAdapter):
    def __init__(self, api_key: str, secret: str):
        self.client = BinanceClient(api_key, secret)
    
    async def get_balance(self, asset: str) -> float:
        account = await self.client.get_account()
        return account['balances'][asset]
```

### 2. Strategy Pattern (Signal Generation)

```python
class SignalStrategy(ABC):
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> Signal: ...

class MACrossoverStrategy(SignalStrategy):
    def generate(self, data: pd.DataFrame) -> Signal:
        if data['ma_fast'].iloc[-1] > data['ma_slow'].iloc[-1]:
            return Signal.BUY
        return Signal.NEUTRAL
```

### 3. Observer Pattern (Event System)

```python
class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
    
    def subscribe(self, event: str, callback: Callable):
        self._subscribers.setdefault(event, []).append(callback)
    
    async def emit(self, event: str, data: dict):
        for callback in self._subscribers.get(event, []):
            await callback(data)
```

## SDK Installation

```bash
pip install sojgja-sdk

# With exchange support
pip install sojgja-sdk[binance,bybit,okx]
```

## Usage Example

```python
from sojgja_sdk import TradingClient
from sojgja_sdk.exchange import BinanceAdapter

client = TradingClient(
    exchange=BinanceAdapter(api_key="...", secret="..."),
    risk_manager=DefaultRiskManager(max_drawdown=0.05),
)

await client.start()
await client.place_order(Order(symbol="BTCUSDT", side="BUY", qty=0.01))
```
