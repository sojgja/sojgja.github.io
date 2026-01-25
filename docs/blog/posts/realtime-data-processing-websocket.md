---
date: 2025-01-25
authors:
  - soigia
categories: [Algorithmic Trading, Backend Development]
title: Real-time Data Processing với WebSocket - Xây dựng Trading Data Pipeline
description: >
  Hướng dẫn xây dựng hệ thống xử lý dữ liệu real-time với WebSocket cho trading, bao gồm connection management, data streaming và error handling.
---

# Real-time Data Processing với WebSocket

Trong trading, việc nhận dữ liệu real-time là cực kỳ quan trọng. WebSocket cung cấp kết nối hai chiều, low-latency để stream market data. Trong bài viết này, chúng ta sẽ xây dựng một hệ thống xử lý dữ liệu real-time hoàn chỉnh.

<!-- more -->

## Tại sao WebSocket cho Trading?

1. **Low Latency**: Kết nối persistent, không cần HTTP overhead
2. **Real-time**: Push data ngay khi có thay đổi
3. **Efficient**: Ít bandwidth hơn polling
4. **Bidirectional**: Có thể gửi commands và nhận data

## Kiến trúc hệ thống

```
┌─────────────┐
│  Exchange   │
│  WebSocket   │
└──────┬───────┘
       │
┌──────▼──────────────┐
│  WebSocket Client   │
│  (Connection Mgr)  │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Data Processor     │
│  (Normalize/Filter) │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Event Bus          │
│  (Redis/RabbitMQ)   │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Consumers          │
│  (Strategy/DB)      │
└─────────────────────┘
```

## Bước 1: WebSocket Client Base Class

```python
# websocket/base_client.py
import asyncio
import websockets
import json
import logging
from typing import Callable, Optional, Dict, List
from abc import ABC, abstractmethod

class BaseWebSocketClient(ABC):
    """Base class cho WebSocket clients"""
    
    def __init__(self, url: str, reconnect_interval: int = 5):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.ws = None
        self.running = False
        self.subscriptions = []
        self.callbacks: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def subscribe(self, channels: List[str]):
        """Subscribe to channels"""
        pass
    
    @abstractmethod
    def handle_message(self, message: dict):
        """Handle incoming message"""
        pass
    
    def on(self, event: str, callback: Callable):
        """Register event callback"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def emit(self, event: str, data: dict):
        """Emit event to callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
    
    async def connect(self):
        """Connect to WebSocket"""
        while not self.running:
            try:
                self.logger.info(f"Connecting to {self.url}")
                async with websockets.connect(self.url) as ws:
                    self.ws = ws
                    self.running = True
                    self.emit('connect', {})
                    
                    # Subscribe to channels
                    if self.subscriptions:
                        await self.subscribe(self.subscriptions)
                    
                    # Listen for messages
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            self.handle_message(data)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid JSON: {message}")
                        except Exception as e:
                            self.logger.error(f"Error handling message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("Connection closed, reconnecting...")
                self.running = False
                await asyncio.sleep(self.reconnect_interval)
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
                self.running = False
                await asyncio.sleep(self.reconnect_interval)
    
    async def send(self, message: dict):
        """Send message to WebSocket"""
        if self.ws and self.running:
            await self.ws.send(json.dumps(message))
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        if self.ws:
            await self.ws.close()
        self.emit('disconnect', {})
```

## Bước 2: Binance WebSocket Client

```python
# websocket/binance_client.py
from websocket.base_client import BaseWebSocketClient
from typing import List
import json

class BinanceWebSocketClient(BaseWebSocketClient):
    """Binance WebSocket client"""
    
    def __init__(self, stream_type: str = 'spot'):
        base_url = 'wss://stream.binance.com:9443/ws/' if stream_type == 'spot' else 'wss://fstream.binance.com/ws/'
        super().__init__(base_url)
        self.stream_type = stream_type
    
    async def subscribe(self, channels: List[str]):
        """Subscribe to Binance streams"""
        # Binance uses stream names like btcusdt@ticker
        streams = [f"{symbol.lower()}@ticker" for symbol in channels]
        stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        # Reconnect with new URL
        self.url = stream_url
        self.subscriptions = channels
    
    def handle_message(self, message: dict):
        """Handle Binance message"""
        if 'stream' in message and 'data' in message:
            stream = message['stream']
            data = message['data']
            
            if '@ticker' in stream:
                symbol = stream.split('@')[0].upper()
                self.emit('ticker', {
                    'symbol': symbol,
                    'price': float(data['c']),
                    'volume': float(data['v']),
                    'change': float(data['P']),
                    'timestamp': data['E']
                })
            
            elif '@trade' in stream:
                symbol = stream.split('@')[0].upper()
                self.emit('trade', {
                    'symbol': symbol,
                    'price': float(data['p']),
                    'quantity': float(data['q']),
                    'timestamp': data['T']
                })
            
            elif '@kline' in stream:
                symbol = stream.split('@')[0].upper()
                kline = data['k']
                self.emit('kline', {
                    'symbol': symbol,
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'timestamp': kline['t'],
                    'is_closed': kline['x']
                })
```

## Bước 3: Bybit WebSocket Client

```python
# websocket/bybit_client.py
from websocket.base_client import BaseWebSocketClient
from typing import List
import json

class BybitWebSocketClient(BaseWebSocketClient):
    """Bybit WebSocket client"""
    
    def __init__(self):
        super().__init__('wss://stream.bybit.com/v5/public/spot')
    
    async def subscribe(self, channels: List[str]):
        """Subscribe to Bybit topics"""
        topics = [f"tickers.{symbol}" for symbol in channels]
        
        subscribe_msg = {
            "op": "subscribe",
            "args": topics
        }
        
        await self.send(subscribe_msg)
        self.subscriptions = channels
    
    def handle_message(self, message: dict):
        """Handle Bybit message"""
        if message.get('topic', '').startswith('tickers.'):
            symbol = message['topic'].split('.')[1]
            data = message['data']
            
            self.emit('ticker', {
                'symbol': symbol,
                'price': float(data['lastPrice']),
                'volume': float(data['volume24h']),
                'change': float(data['price24hPcnt']) * 100,
                'timestamp': message['ts']
            })
```

## Bước 4: Data Processor

```python
# websocket/data_processor.py
from typing import Dict, Optional
from datetime import datetime
import logging

class DataProcessor:
    """Process and normalize data from different exchanges"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_cache: Dict[str, float] = {}
        self.volume_cache: Dict[str, float] = {}
    
    def normalize_ticker(self, exchange: str, data: dict) -> Optional[Dict]:
        """Normalize ticker data from different exchanges"""
        try:
            if exchange == 'binance':
                return {
                    'exchange': 'binance',
                    'symbol': data['symbol'],
                    'price': data['price'],
                    'volume_24h': data['volume'],
                    'change_24h': data['change'],
                    'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000),
                    'source': 'websocket'
                }
            elif exchange == 'bybit':
                return {
                    'exchange': 'bybit',
                    'symbol': data['symbol'],
                    'price': data['price'],
                    'volume_24h': data['volume'],
                    'change_24h': data['change'],
                    'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000),
                    'source': 'websocket'
                }
        except Exception as e:
            self.logger.error(f"Error normalizing ticker: {e}")
            return None
    
    def detect_price_change(self, symbol: str, new_price: float, threshold: float = 0.01) -> bool:
        """Detect significant price change"""
        if symbol in self.price_cache:
            old_price = self.price_cache[symbol]
            change_pct = abs((new_price - old_price) / old_price)
            
            if change_pct >= threshold:
                self.price_cache[symbol] = new_price
                return True
        
        self.price_cache[symbol] = new_price
        return False
    
    def filter_volume(self, symbol: str, volume: float, min_volume: float = 1000) -> bool:
        """Filter by minimum volume"""
        return volume >= min_volume
```

## Bước 5: Event Bus với Redis

```python
# websocket/event_bus.py
import redis
import json
from typing import Callable, List
import asyncio
import logging

class RedisEventBus:
    """Event bus using Redis pub/sub"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
    
    def publish(self, channel: str, data: dict):
        """Publish event to channel"""
        try:
            self.redis_client.publish(channel, json.dumps(data))
        except Exception as e:
            self.logger.error(f"Error publishing to {channel}: {e}")
    
    def subscribe(self, channel: str, callback: Callable):
        """Subscribe to channel"""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
            self.pubsub.subscribe(channel)
        
        self.subscribers[channel].append(callback)
    
    async def listen(self):
        """Listen for messages"""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel']
                data = json.loads(message['data'])
                
                if channel in self.subscribers:
                    for callback in self.subscribers[channel]:
                        try:
                            await callback(data) if asyncio.iscoroutinefunction(callback) else callback(data)
                        except Exception as e:
                            self.logger.error(f"Callback error: {e}")
```

## Bước 6: Complete System

```python
# websocket/trading_data_pipeline.py
import asyncio
from websocket.binance_client import BinanceWebSocketClient
from websocket.bybit_client import BybitWebSocketClient
from websocket.data_processor import DataProcessor
from websocket.event_bus import RedisEventBus
import logging

logging.basicConfig(level=logging.INFO)

class TradingDataPipeline:
    """Complete trading data pipeline"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.event_bus = RedisEventBus()
        
        # Initialize exchange clients
        self.binance = BinanceWebSocketClient()
        self.bybit = BybitWebSocketClient()
        
        # Setup callbacks
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup WebSocket callbacks"""
        # Binance callbacks
        self.binance.on('ticker', self.handle_binance_ticker)
        self.binance.on('trade', self.handle_binance_trade)
        self.binance.on('kline', self.handle_binance_kline)
        
        # Bybit callbacks
        self.bybit.on('ticker', self.handle_bybit_ticker)
    
    def handle_binance_ticker(self, data: dict):
        """Handle Binance ticker"""
        normalized = self.processor.normalize_ticker('binance', data)
        
        if normalized:
            # Check for significant price change
            if self.processor.detect_price_change(
                normalized['symbol'], 
                normalized['price'],
                threshold=0.005  # 0.5%
            ):
                # Publish to event bus
                self.event_bus.publish('price_alerts', normalized)
            
            # Always publish to ticker stream
            self.event_bus.publish('tickers', normalized)
    
    def handle_bybit_ticker(self, data: dict):
        """Handle Bybit ticker"""
        normalized = self.processor.normalize_ticker('bybit', data)
        
        if normalized:
            self.event_bus.publish('tickers', normalized)
    
    def handle_binance_trade(self, data: dict):
        """Handle Binance trade"""
        self.event_bus.publish('trades', {
            'exchange': 'binance',
            'symbol': data['symbol'],
            'price': data['price'],
            'quantity': data['quantity'],
            'timestamp': data['timestamp']
        })
    
    def handle_binance_kline(self, data: dict):
        """Handle Binance kline"""
        if data['is_closed']:
            self.event_bus.publish('klines', {
                'exchange': 'binance',
                'symbol': data['symbol'],
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume'],
                'timestamp': data['timestamp']
            })
    
    async def start(self, symbols: List[str]):
        """Start the pipeline"""
        # Subscribe to symbols
        self.binance.subscriptions = symbols
        self.bybit.subscriptions = symbols
        
        # Start WebSocket connections
        tasks = [
            asyncio.create_task(self.binance.connect()),
            asyncio.create_task(self.bybit.connect()),
            asyncio.create_task(self.event_bus.listen())
        ]
        
        await asyncio.gather(*tasks)

# Usage
async def main():
    pipeline = TradingDataPipeline()
    
    # Subscribe to symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Setup consumers
    def on_price_alert(data):
        print(f"Price alert: {data}")
    
    def on_ticker(data):
        print(f"Ticker: {data['symbol']} = ${data['price']}")
    
    pipeline.event_bus.subscribe('price_alerts', on_price_alert)
    pipeline.event_bus.subscribe('tickers', on_ticker)
    
    await pipeline.start(symbols)

if __name__ == '__main__':
    asyncio.run(main())
```

## Best Practices

1. **Connection Management**: Auto-reconnect với exponential backoff
2. **Error Handling**: Handle tất cả exceptions gracefully
3. **Rate Limiting**: Respect exchange rate limits
4. **Data Validation**: Validate data trước khi process
5. **Monitoring**: Log và monitor connection health
6. **Backpressure**: Handle khi consumer chậm hơn producer

## Kết luận

Với hệ thống này, bạn có thể:
- Stream real-time data từ multiple exchanges
- Normalize data từ different sources
- Publish events cho consumers
- Scale horizontally với Redis

Trong bài tiếp theo, chúng ta sẽ xây dựng risk management system.
