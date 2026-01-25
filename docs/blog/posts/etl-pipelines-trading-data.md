---
date: 2025-01-25
authors:
  - soigia
categories: [Data Engineering, Trading]
title: ETL Pipelines cho Trading Data - Apache Airflow và Data Processing
description: >
  Xây dựng ETL pipelines để xử lý trading data: data extraction, transformation, loading với Apache Airflow và Python.
---

# ETL Pipelines cho Trading Data

ETL (Extract, Transform, Load) pipelines là critical cho việc xử lý và lưu trữ trading data. Trong bài viết này, chúng ta sẽ xây dựng ETL pipelines với Apache Airflow.

<!-- more -->

## ETL Pipeline Architecture

```
┌─────────────┐
│   Extract   │  ← Exchange APIs, Files, Databases
└──────┬──────┘
       │
┌──────▼──────┐
│ Transform   │  ← Data cleaning, validation, enrichment
└──────┬──────┘
       │
┌──────▼──────┐
│    Load     │  → Databases, Data Warehouses
└─────────────┘
```

## Bước 1: Airflow DAG Setup

```python
# etl/airflow_dags/trading_data_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from etl.extractors import BinanceExtractor
from etl.transformers import MarketDataTransformer
from etl.loaders import DatabaseLoader

default_args = {
    'owner': 'trading_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'trading_data_etl',
    default_args=default_args,
    description='ETL pipeline for trading data',
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2025, 1, 1),
    catchup=False
)

def extract_binance_data(**context):
    """Extract data from Binance"""
    extractor = BinanceExtractor()
    data = extractor.extract_ohlcv(
        symbol='BTCUSDT',
        interval='1h',
        limit=100
    )
    return data

def transform_data(**context):
    """Transform extracted data"""
    ti = context['ti']
    raw_data = ti.xcom_pull(task_ids='extract_binance')
    
    transformer = MarketDataTransformer()
    transformed = transformer.transform(raw_data)
    
    return transformed

def load_to_database(**context):
    """Load transformed data to database"""
    ti = context['ti']
    transformed_data = ti.xcom_pull(task_ids='transform_data')
    
    loader = DatabaseLoader()
    loader.load(transformed_data)

# Define tasks
extract_task = PythonOperator(
    task_id='extract_binance',
    python_callable=extract_binance_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_to_database',
    python_callable=load_to_database,
    dag=dag
)

# Set dependencies
extract_task >> transform_task >> load_task
```

## Bước 2: Data Extractors

```python
# etl/extractors.py
from binance.client import Client
import pandas as pd
from typing import List, Dict
from datetime import datetime
import logging

class BinanceExtractor:
    """Extract data from Binance"""
    
    def __init__(self):
        self.client = Client()
        self.logger = logging.getLogger(__name__)
    
    def extract_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Extract OHLCV data"""
        try:
            if start_time and end_time:
                klines = self.client.get_historical_klines(
                    symbol, interval, start_time.strftime('%d %b %Y %H:%M:%S'),
                    end_time.strftime('%d %b %Y %H:%M:%S')
                )
            else:
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            data = []
            for kline in klines:
                data.append({
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            self.logger.info(f"Extracted {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error extracting data: {e}")
            return []
    
    def extract_trades(
        self,
        symbol: str,
        limit: int = 1000
    ) -> List[Dict]:
        """Extract recent trades"""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            
            data = []
            for trade in trades:
                data.append({
                    'symbol': symbol,
                    'trade_id': trade['id'],
                    'price': float(trade['price']),
                    'quantity': float(trade['qty']),
                    'timestamp': datetime.fromtimestamp(trade['time'] / 1000),
                    'is_buyer_maker': trade['isBuyerMaker']
                })
            
            return data
        except Exception as e:
            self.logger.error(f"Error extracting trades: {e}")
            return []

class FileExtractor:
    """Extract data from files"""
    
    def extract_csv(self, filepath: str) -> pd.DataFrame:
        """Extract from CSV file"""
        return pd.read_csv(filepath)
    
    def extract_json(self, filepath: str) -> List[Dict]:
        """Extract from JSON file"""
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
```

## Bước 3: Data Transformers

```python
# etl/transformers.py
import pandas as pd
from typing import List, Dict
import logging

class MarketDataTransformer:
    """Transform market data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def transform(self, data: List[Dict]) -> pd.DataFrame:
        """Transform raw data"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Data cleaning
        df = self.clean_data(df)
        
        # Data validation
        df = self.validate_data(df)
        
        # Data enrichment
        df = self.enrich_data(df)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['symbol', 'timestamp'])
        
        # Remove invalid prices
        df = df[df['close'] > 0]
        df = df[df['volume'] >= 0]
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data"""
        # Check for missing values
        required_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=required_cols)
        
        # Validate price relationships
        # High should be >= Low
        df = df[df['high'] >= df['low']]
        # High should be >= Open and Close
        df = df[(df['high'] >= df['open']) & (df['high'] >= df['close'])]
        # Low should be <= Open and Close
        df = df[(df['low'] <= df['open']) & (df['low'] <= df['close'])]
        
        return df
    
    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich data with calculated fields"""
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate volatility (rolling 24h)
        df['volatility'] = df['returns'].rolling(window=24).std()
        
        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

class TradeDataTransformer:
    """Transform trade data"""
    
    def transform(self, data: List[Dict]) -> pd.DataFrame:
        """Transform trade data"""
        df = pd.DataFrame(data)
        
        # Add exchange column if missing
        if 'exchange' not in df.columns:
            df['exchange'] = 'binance'
        
        # Calculate trade value
        df['value'] = df['price'] * df['quantity']
        
        # Classify trade size
        df['trade_size'] = pd.cut(
            df['value'],
            bins=[0, 1000, 10000, 100000, float('inf')],
            labels=['small', 'medium', 'large', 'whale']
        )
        
        return df
```

## Bước 4: Data Loaders

```python
# etl/loaders.py
import pandas as pd
from database.market_data_db import MarketDataDB
from typing import List, Dict
import logging

class DatabaseLoader:
    """Load data to database"""
    
    def __init__(self, connection_string: str):
        self.db = MarketDataDB(connection_string)
        self.logger = logging.getLogger(__name__)
    
    def load(self, df: pd.DataFrame):
        """Load DataFrame to database"""
        if df.empty:
            self.logger.warning("Empty DataFrame, nothing to load")
            return
        
        # Convert DataFrame to list of dicts
        data = df.to_dict('records')
        
        # Group by symbol and exchange
        grouped = {}
        for record in data:
            key = (record['symbol'], record.get('exchange', 'binance'))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(record)
        
        # Load each group
        for (symbol, exchange), records in grouped.items():
            try:
                self.db.insert_ohlcv_batch(records)
                self.logger.info(f"Loaded {len(records)} records for {symbol} on {exchange}")
            except Exception as e:
                self.logger.error(f"Error loading {symbol}: {e}")

class FileLoader:
    """Load data to files"""
    
    def load_csv(self, df: pd.DataFrame, filepath: str):
        """Load to CSV file"""
        df.to_csv(filepath, index=False)
    
    def load_parquet(self, df: pd.DataFrame, filepath: str):
        """Load to Parquet file (compressed)"""
        df.to_parquet(filepath, compression='snappy')
```

## Bước 5: Complete ETL Pipeline

```python
# etl/pipeline.py
from etl.extractors import BinanceExtractor
from etl.transformers import MarketDataTransformer
from etl.loaders import DatabaseLoader
from datetime import datetime, timedelta
import logging

class TradingDataETL:
    """Complete ETL pipeline"""
    
    def __init__(self, db_connection_string: str):
        self.extractor = BinanceExtractor()
        self.transformer = MarketDataTransformer()
        self.loader = DatabaseLoader(db_connection_string)
        self.logger = logging.getLogger(__name__)
    
    def run(self, symbols: List[str], intervals: List[str]):
        """Run ETL for multiple symbols and intervals"""
        for symbol in symbols:
            for interval in intervals:
                try:
                    # Extract
                    self.logger.info(f"Extracting {symbol} {interval}")
                    raw_data = self.extractor.extract_ohlcv(
                        symbol=symbol,
                        interval=interval,
                        limit=1000
                    )
                    
                    if not raw_data:
                        continue
                    
                    # Transform
                    self.logger.info(f"Transforming {symbol} {interval}")
                    transformed = self.transformer.transform(raw_data)
                    
                    # Load
                    self.logger.info(f"Loading {symbol} {interval}")
                    self.loader.load(transformed)
                    
                    self.logger.info(f"Completed {symbol} {interval}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol} {interval}: {e}")
                    continue

# Usage
if __name__ == '__main__':
    etl = TradingDataETL('postgresql://user:pass@localhost/db')
    etl.run(
        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        intervals=['1h', '4h', '1d']
    )
```

## Best Practices

1. **Idempotency**: Make pipelines idempotent
2. **Error Handling**: Handle errors gracefully
3. **Logging**: Log all operations
4. **Monitoring**: Monitor pipeline health
5. **Incremental Loading**: Load only new data
6. **Data Quality**: Validate data at each stage

## Kết luận

ETL pipelines enable:
- Automated data processing
- Data quality assurance
- Scalable data processing
- Reliable data pipelines

Build robust ETL pipelines for your trading data!
