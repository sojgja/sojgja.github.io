---
id: design-patterns
title: SDK Design Patterns
sidebar_label: SDK Patterns
sidebar_position: 1
description: Essential design patterns for SDK development — Adapter, Strategy, Observer, Factory, Singleton patterns with Python trading SDK examples.
keywords: [design-pattern, sdk, adapter, strategy, observer, factory, singleton, python]
---

# SDK Design Patterns

Reusable patterns for building clean, extensible trading SDKs.

## Adapter Pattern

```python
from abc import ABC, abstractmethod

class ExchangeAdapter(ABC):
    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict: ...
    @abstractmethod
    async def place_order(self, order: dict) -> str: ...

class BinanceAdapter(ExchangeAdapter):
    def __init__(self, api_key: str, secret: str):
        self.client = Client(api_key, secret)

    async def get_ticker(self, symbol: str) -> dict:
        return await self.client.get_symbol_ticker(symbol=symbol)

class BybitAdapter(ExchangeAdapter):
    async def get_ticker(self, symbol: str) -> dict:
        resp = await self.session.get(f"/v5/market/tickers?symbol={symbol}")
        return resp['result']['list'][0]
```

## Strategy Pattern

```python
class TradingStrategy(ABC):
    @abstractmethod
    def should_enter(self, data: pd.DataFrame) -> bool: ...
    @abstractmethod
    def should_exit(self, data: pd.DataFrame, position: dict) -> bool: ...

class MACrossover(TradingStrategy):
    def should_enter(self, data: pd.DataFrame) -> bool:
        return data['ma_fast'].iloc[-1] > data['ma_slow'].iloc[-1]

class MeanReversion(TradingStrategy):
    def should_enter(self, data: pd.DataFrame) -> bool:
        z_score = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        return z_score.iloc[-1] < -2.0
```

## Observer Pattern

```python
class PriceFeed:
    def __init__(self):
        self._observers: list[callable] = []

    def subscribe(self, callback: callable):
        self._observers.append(callback)

    async def _on_price(self, price: dict):
        for cb in self._observers:
            await cb(price)

# Usage
feed = PriceFeed()
feed.subscribe(strategy.on_price)
feed.subscribe(risk_manager.on_price)
feed.subscribe(logger.log_price)
```

## Factory Pattern

```python
class ExchangeFactory:
    _adapters = {
        'binance': BinanceAdapter,
        'bybit': BybitAdapter,
        'okx': OKXAdapter,
    }

    @classmethod
    def create(cls, exchange: str, **credentials) -> ExchangeAdapter:
        adapter = cls._adapters.get(exchange)
        if not adapter:
            raise ValueError(f"Unsupported exchange: {exchange}")
        return adapter(**credentials)
```
