---
id: orderbook
title: Order Book Engine
sidebar_label: Order Book
sidebar_position: 1
description: Low-latency order book matching engine — bid/ask management, price-time priority, order lifecycle, and exchange integration patterns.
keywords: [orderbook, trading, matching-engine, bid-ask, exchange, websocket, low-latency]
---

# Order Book Engine

A **price-time priority** matching engine for cryptocurrency and forex trading.

## Data Structure

```python
from sortedcontainers import SortedDict
from dataclasses import dataclass

@dataclass
class Order:
    order_id: str
    symbol: str
    side: str       # BUY or SELL
    price: float
    quantity: float
    timestamp: float

class OrderBook:
    def __init__(self):
        self.bids = SortedDict(lambda k: -k)  # descending
        self.asks = SortedDict()               # ascending
        self.orders: dict[str, Order] = {}
```

## Add Order

```python
def add_order(self, order: Order):
    book = self.bids if order.side == 'BUY' else self.asks
    if order.price not in book:
        book[order.price] = []
    book[order.price].append(order)
    self.orders[order.order_id] = order
```

## Match Engine

```python
def match(self, incoming: Order) -> list[tuple]:
    fills = []
    book = self.asks if incoming.side == 'BUY' else self.bids

    for price in list(book.keys()):
        if incoming.side == 'BUY' and price > incoming.price:
            break
        if incoming.side == 'SELL' and price < incoming.price:
            break

        for resting in book[price][:]:
            matched_qty = min(resting.quantity, incoming.quantity)
            fills.append((resting.order_id, incoming.order_id, price, matched_qty))
            resting.quantity -= matched_qty
            incoming.quantity -= matched_qty
            if resting.quantity == 0:
                book[price].remove(resting)
                del self.orders[resting.order_id]
            if incoming.quantity == 0:
                return fills
        if not book[price]:
            del book[price]
    return fills
```

## WebSocket Feed

```python
import websockets
import json

async def stream_orderbook(symbol: str):
    async with websockets.connect(f"wss://stream.exchange.com/ws/{symbol}@depth") as ws:
        async for msg in ws:
            data = json.loads(msg)
            # Update local orderbook from delta
            apply_delta(book, data['bids'], data['asks'])
```
