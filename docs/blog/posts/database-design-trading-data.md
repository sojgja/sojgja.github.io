---
date: 2025-01-25
authors:
  - soigia
categories: [Trading Infrastructure, Database Design]
title: Database Design cho Trading Data - Time-Series và Market Data
description: >
  Thiết kế database tối ưu cho trading data: time-series databases, schema design, indexing strategies và query optimization.
---

# Database Design cho Trading Data

![Database Design for Trading](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&h=600&fit=crop)

Trong thế giới của algorithmic trading và quantitative finance, dữ liệu là tài sản quý giá nhất, và việc lưu trữ, quản lý, và truy xuất dữ liệu một cách hiệu quả có thể quyết định thành bại của toàn bộ hệ thống trading. Trading data có những đặc thù riêng biệt mà không phải database nào cũng có thể xử lý tốt: khối lượng dữ liệu cực lớn (hàng triệu records mỗi ngày cho một trading pair), tính chất time-series với các query patterns đặc biệt, yêu cầu về tốc độ truy xuất real-time, và nhu cầu về khả năng scale horizontal. Một database được thiết kế tốt không chỉ giúp bạn lưu trữ dữ liệu một cách hiệu quả, mà còn cho phép bạn query và phân tích dữ liệu với tốc độ cao, hỗ trợ các operations phức tạp như aggregations, window functions, và time-based joins.

Trong bài viết chi tiết này, chúng ta sẽ cùng nhau thiết kế một database system tối ưu cho trading data, từ việc lựa chọn loại database phù hợp (time-series databases như InfluxDB, TimescaleDB, hay traditional relational databases như PostgreSQL với optimizations), thiết kế schema để tối ưu hóa storage và query performance, implement indexing strategies cho time-series data, đến việc xây dựng data partitioning và archiving strategies. Chúng ta sẽ học cách xử lý các challenges đặc thù như compression cho historical data, partitioning theo thời gian để tăng tốc độ query, và implement caching layers để giảm latency cho real-time queries. Chúng ta cũng sẽ thảo luận về các trade-offs giữa different database technologies và khi nào nên sử dụng cái nào.

Bài viết này sẽ hướng dẫn bạn từng bước một, từ việc phân tích requirements và query patterns, thiết kế schema với các best practices cho time-series data, implement indexing và partitioning strategies, đến việc optimize queries và monitor performance. Chúng ta cũng sẽ học cách implement data retention policies, backup và recovery strategies, và scaling strategies cho khi dữ liệu tăng trưởng. Cuối cùng, bạn sẽ có trong tay một database system mạnh mẽ, có thể xử lý hàng triệu records mỗi ngày và query chúng với tốc độ cực nhanh.

<!-- more -->

![Trading Data Characteristics](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&h=400&fit=crop)

## Đặc điểm Trading Data

1. **High Volume**: Hàng triệu records mỗi ngày
2. **Time-Series**: Data theo thời gian
3. **Append-Only**: Chủ yếu là insert, ít update/delete
4. **Query Patterns**: Range queries, aggregations
5. **Real-time**: Cần access data real-time

## Database Options

### 1. Time-Series Databases
- **InfluxDB**: Popular, easy to use
- **TimescaleDB**: PostgreSQL extension
- **QuestDB**: High-performance, SQL-like

### 2. Traditional Databases
- **PostgreSQL**: With time-series extensions
- **MySQL**: For relational data

### 3. NoSQL
- **MongoDB**: Flexible schema
- **Cassandra**: Distributed, high write throughput

## Bước 1: Schema Design cho Market Data

```sql
-- PostgreSQL schema for market data

-- Tickers table (current prices)
CREATE TABLE tickers (
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    volume_24h DECIMAL(20, 8),
    change_24h DECIMAL(10, 4),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, exchange)
);

CREATE INDEX idx_tickers_updated ON tickers(updated_at);

-- OHLCV data (candlestick)
CREATE TABLE ohlcv (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL, -- '1m', '5m', '1h', '1d'
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    UNIQUE(symbol, exchange, interval, timestamp)
);

-- Partition by time (monthly)
CREATE TABLE ohlcv_2025_01 PARTITION OF ohlcv
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE INDEX idx_ohlcv_symbol_time ON ohlcv(symbol, exchange, interval, timestamp DESC);
CREATE INDEX idx_ohlcv_timestamp ON ohlcv(timestamp DESC);

-- Trades table (individual trades)
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    side VARCHAR(5) NOT NULL, -- 'BUY' or 'SELL'
    timestamp TIMESTAMP NOT NULL,
    trade_id VARCHAR(100) -- Exchange trade ID
);

-- Partition trades by time
CREATE TABLE trades_2025_01 PARTITION OF trades
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE INDEX idx_trades_symbol_time ON trades(symbol, exchange, timestamp DESC);
CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);

-- Order book snapshots
CREATE TABLE orderbook_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    bids JSONB NOT NULL, -- [{price, quantity}, ...]
    asks JSONB NOT NULL,
    UNIQUE(symbol, exchange, timestamp)
);

CREATE INDEX idx_orderbook_symbol_time ON orderbook_snapshots(symbol, exchange, timestamp DESC);
```

## Bước 2: TimescaleDB Setup

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert ohlcv to hypertable
SELECT create_hypertable('ohlcv', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day');

-- Convert trades to hypertable
SELECT create_hypertable('trades', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour');

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW ohlcv_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    exchange,
    interval,
    FIRST(open, timestamp) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, timestamp) AS close,
    SUM(volume) AS volume
FROM ohlcv
GROUP BY bucket, symbol, exchange, interval;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('ohlcv_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

## Bước 3: Python Database Layer

```python
# database/market_data_db.py
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Optional
from datetime import datetime
import logging

class MarketDataDB:
    """Database layer for market data"""
    
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        self.logger = logging.getLogger(__name__)
    
    def insert_ticker(self, symbol: str, exchange: str, price: float, 
                     volume: Optional[float] = None, change: Optional[float] = None):
        """Insert or update ticker"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO tickers (symbol, exchange, price, volume_24h, change_24h, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (symbol, exchange)
                DO UPDATE SET
                    price = EXCLUDED.price,
                    volume_24h = EXCLUDED.volume_24h,
                    change_24h = EXCLUDED.change_24h,
                    updated_at = NOW()
            """, (symbol, exchange, price, volume, change))
            self.conn.commit()
    
    def insert_ohlcv_batch(self, data: List[Dict]):
        """Batch insert OHLCV data"""
        if not data:
            return
        
        values = [
            (
                d['symbol'], d['exchange'], d['interval'],
                d['timestamp'], d['open'], d['high'],
                d['low'], d['close'], d['volume']
            )
            for d in data
        ]
        
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO ohlcv (symbol, exchange, interval, timestamp, 
                                  open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, exchange, interval, timestamp) DO NOTHING
                """,
                values
            )
            self.conn.commit()
    
    def get_ohlcv(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get OHLCV data for time range"""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s AND exchange = %s AND interval = %s
                AND timestamp >= %s AND timestamp <= %s
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.conn.cursor() as cur:
            cur.execute(query, (symbol, exchange, interval, start_time, end_time))
            
            return [
                {
                    'timestamp': row[0],
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': float(row[5])
                }
                for row in cur.fetchall()
            ]
    
    def get_latest_price(self, symbol: str, exchange: str) -> Optional[float]:
        """Get latest price"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT price FROM tickers
                WHERE symbol = %s AND exchange = %s
            """, (symbol, exchange))
            
            result = cur.fetchone()
            return float(result[0]) if result else None
    
    def get_price_history(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Get price history with aggregations"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    time_bucket('1 hour', timestamp) AS hour,
                    FIRST(open, timestamp) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, timestamp) AS close,
                    SUM(volume) AS volume
                FROM ohlcv
                WHERE symbol = %s AND exchange = %s
                    AND timestamp >= %s AND timestamp <= %s
                GROUP BY hour
                ORDER BY hour DESC
            """, (symbol, exchange, start_time, end_time))
            
            return [
                {
                    'hour': row[0],
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': float(row[5])
                }
                for row in cur.fetchall()
            ]
```

## Bước 4: InfluxDB Integration

```python
# database/influxdb_client.py
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import List, Dict
import logging

class InfluxDBMarketData:
    """InfluxDB client for market data"""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org
        self.logger = logging.getLogger(__name__)
    
    def write_ticker(self, symbol: str, exchange: str, price: float, 
                    volume: Optional[float] = None):
        """Write ticker data"""
        point = Point("ticker") \
            .tag("symbol", symbol) \
            .tag("exchange", exchange) \
            .field("price", price)
        
        if volume:
            point.field("volume", volume)
        
        self.write_api.write(bucket=self.bucket, record=point)
    
    def write_ohlcv(self, symbol: str, exchange: str, interval: str,
                   timestamp: datetime, open: float, high: float,
                   low: float, close: float, volume: float):
        """Write OHLCV data"""
        point = Point("ohlcv") \
            .tag("symbol", symbol) \
            .tag("exchange", exchange) \
            .tag("interval", interval) \
            .field("open", open) \
            .field("high", high) \
            .field("low", low) \
            .field("close", close) \
            .field("volume", volume) \
            .time(timestamp)
        
        self.write_api.write(bucket=self.bucket, record=point)
    
    def query_price_history(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h"
    ) -> List[Dict]:
        """Query price history"""
        query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "ohlcv")
            |> filter(fn: (r) => r["symbol"] == "{symbol}")
            |> filter(fn: (r) => r["exchange"] == "{exchange}")
            |> filter(fn: (r) => r["interval"] == "{interval}")
            |> aggregateWindow(every: {interval}, fn: mean, createEmpty: false)
        '''
        
        result = self.query_api.query(org=self.org, query=query)
        
        data = []
        for table in result:
            for record in table.records:
                data.append({
                    'timestamp': record.get_time(),
                    'value': record.get_value(),
                    'field': record.get_field()
                })
        
        return data
```

## Bước 5: Caching Layer

```python
# database/cache.py
import redis
import json
from typing import Optional, Dict
from datetime import timedelta

class MarketDataCache:
    """Redis cache for market data"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
    
    def cache_ticker(self, symbol: str, exchange: str, price: float, ttl: int = 60):
        """Cache ticker price"""
        key = f"ticker:{exchange}:{symbol}"
        self.redis.setex(key, ttl, json.dumps({'price': price}))
    
    def get_ticker(self, symbol: str, exchange: str) -> Optional[float]:
        """Get cached ticker price"""
        key = f"ticker:{exchange}:{symbol}"
        data = self.redis.get(key)
        
        if data:
            return json.loads(data)['price']
        return None
    
    def cache_ohlcv(self, symbol: str, exchange: str, interval: str,
                   data: List[Dict], ttl: int = 3600):
        """Cache OHLCV data"""
        key = f"ohlcv:{exchange}:{symbol}:{interval}"
        self.redis.setex(key, ttl, json.dumps(data))
    
    def get_ohlcv(self, symbol: str, exchange: str, interval: str) -> Optional[List[Dict]]:
        """Get cached OHLCV data"""
        key = f"ohlcv:{exchange}:{symbol}:{interval}"
        data = self.redis.get(key)
        
        if data:
            return json.loads(data)
        return None
```

## Best Practices

1. **Partitioning**: Partition by time for better performance
2. **Indexing**: Index on (symbol, timestamp) for range queries
3. **Batch Inserts**: Use batch inserts for better throughput
4. **Caching**: Cache frequently accessed data
5. **Compression**: Compress old data
6. **Retention**: Archive old data to cold storage

## Kết luận

Database design cho trading data cần:
- Time-series optimization
- Efficient indexing
- Partitioning strategy
- Caching layer
- Query optimization

Chọn database phù hợp với use case của bạn!
