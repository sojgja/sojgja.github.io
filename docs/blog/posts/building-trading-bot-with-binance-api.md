---
date: 2025-01-25
authors:
  - soigia
categories: [Algorithmic Trading, Backend Development]
title: Xây dựng Trading Bot với Binance API - Hướng dẫn từ A đến Z
description: >
  Hướng dẫn chi tiết cách xây dựng một trading bot hoàn chỉnh sử dụng Binance API, từ setup đến deployment.
---

# Xây dựng Trading Bot với Binance API

Trong bài viết này, chúng ta sẽ cùng xây dựng một trading bot hoàn chỉnh sử dụng Binance API. Bot này sẽ có khả năng theo dõi thị trường, phân tích tín hiệu và thực hiện giao dịch tự động.

<!-- more -->

## Tổng quan

Trading bot là một công cụ mạnh mẽ giúp tự động hóa quá trình giao dịch, loại bỏ cảm xúc và thực hiện các chiến lược một cách nhất quán. Với Binance API, chúng ta có thể truy cập vào một trong những sàn giao dịch lớn nhất thế giới.

## Yêu cầu

- Python 3.8+
- Tài khoản Binance (testnet hoặc real)
- API Key và Secret Key từ Binance
- Kiến thức cơ bản về Python và trading

## Bước 1: Setup môi trường

Đầu tiên, chúng ta cần cài đặt các thư viện cần thiết:

```bash
pip install python-binance pandas python-dotenv
```

Tạo file `.env` để lưu trữ API keys:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=True
```

## Bước 2: Tạo cấu trúc project

```
trading-bot/
├── config/
│   └── settings.py
├── strategies/
│   └── simple_strategy.py
├── utils/
│   ├── logger.py
│   └── risk_manager.py
├── bot.py
└── requirements.txt
```

## Bước 3: Implement Binance Client

```python
# bot.py
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import logging

load_dotenv()

class BinanceTradingBot:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.testnet = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'
        
        # Khởi tạo client
        if self.testnet:
            self.client = Client(
                self.api_key,
                self.secret_key,
                testnet=True
            )
        else:
            self.client = Client(self.api_key, self.secret_key)
        
        # Setup logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def get_account_balance(self):
        """Lấy số dư tài khoản"""
        try:
            account = self.client.get_account()
            balances = {b['asset']: float(b['free']) 
                       for b in account['balances'] 
                       if float(b['free']) > 0}
            return balances
        except BinanceAPIException as e:
            self.logger.error(f"Error getting balance: {e}")
            return {}
    
    def get_ticker_price(self, symbol):
        """Lấy giá hiện tại của symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol, side, quantity):
        """Đặt lệnh market"""
        try:
            if side.upper() == 'BUY':
                order = self.client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
            else:
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
            self.logger.info(f"Order placed: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def get_klines(self, symbol, interval, limit=100):
        """Lấy dữ liệu kline (candlestick)"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return klines
        except BinanceAPIException as e:
            self.logger.error(f"Error getting klines: {e}")
            return []

if __name__ == '__main__':
    bot = BinanceTradingBot()
    balance = bot.get_account_balance()
    print(f"Account balance: {balance}")
    
    price = bot.get_ticker_price('BTCUSDT')
    print(f"BTC Price: ${price}")
```

## Bước 4: Implement Strategy

```python
# strategies/simple_strategy.py
import pandas as pd
from typing import Dict, Optional

class SimpleMovingAverageStrategy:
    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate_indicators(self, klines):
        """Tính toán các chỉ báo kỹ thuật"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        
        # Calculate moving averages
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df) -> Optional[str]:
        """Tạo tín hiệu mua/bán"""
        if len(df) < self.long_window:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Golden Cross: SMA ngắn cắt lên SMA dài
        if (prev['sma_short'] <= prev['sma_long'] and 
            latest['sma_short'] > latest['sma_long'] and
            latest['rsi'] < 70):
            return 'BUY'
        
        # Death Cross: SMA ngắn cắt xuống SMA dài
        if (prev['sma_short'] >= prev['sma_long'] and 
            latest['sma_short'] < latest['sma_long'] and
            latest['rsi'] > 30):
            return 'SELL'
        
        return None
```

## Bước 5: Risk Management

```python
# utils/risk_manager.py
class RiskManager:
    def __init__(self, max_position_size=0.1, stop_loss_pct=0.02, take_profit_pct=0.04):
        self.max_position_size = max_position_size  # 10% của balance
        self.stop_loss_pct = stop_loss_pct  # 2%
        self.take_profit_pct = take_profit_pct  # 4%
    
    def calculate_position_size(self, balance, price):
        """Tính toán kích thước position"""
        max_investment = balance * self.max_position_size
        quantity = max_investment / price
        return round(quantity, 6)
    
    def should_place_order(self, current_price, entry_price, side):
        """Kiểm tra xem có nên đặt lệnh không"""
        if side == 'SELL':
            # Check stop loss
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                return True, 'STOP_LOSS'
            # Check take profit
            if current_price >= entry_price * (1 + self.take_profit_pct):
                return True, 'TAKE_PROFIT'
        
        return False, None
```

## Bước 6: Kết hợp tất cả lại

```python
# bot.py (updated)
import time
from strategies.simple_strategy import SimpleMovingAverageStrategy
from utils.risk_manager import RiskManager

class TradingBot(BinanceTradingBot):
    def __init__(self, symbol='BTCUSDT'):
        super().__init__()
        self.symbol = symbol
        self.strategy = SimpleMovingAverageStrategy()
        self.risk_manager = RiskManager()
        self.position = None  # {'side': 'BUY', 'price': 50000, 'quantity': 0.001}
    
    def run(self):
        """Chạy bot chính"""
        self.logger.info("Starting trading bot...")
        
        while True:
            try:
                # Lấy dữ liệu kline
                klines = self.get_klines(self.symbol, '1h', 100)
                
                if not klines:
                    time.sleep(60)
                    continue
                
                # Tính toán indicators
                df = self.strategy.calculate_indicators(klines)
                
                # Tạo signal
                signal = self.strategy.generate_signal(df)
                
                if signal:
                    self.logger.info(f"Signal generated: {signal}")
                    self.handle_signal(signal, df)
                
                # Kiểm tra position hiện tại
                if self.position:
                    self.check_position()
                
                # Đợi 1 phút trước khi check lại
                time.sleep(60)
                
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)
    
    def handle_signal(self, signal, df):
        """Xử lý tín hiệu trading"""
        balance = self.get_account_balance()
        current_price = float(df.iloc[-1]['close'])
        
        if signal == 'BUY' and not self.position:
            # Tính toán quantity
            usdt_balance = balance.get('USDT', 0)
            if usdt_balance > 10:  # Minimum $10
                quantity = self.risk_manager.calculate_position_size(
                    usdt_balance, current_price
                )
                
                order = self.place_market_order(
                    self.symbol, 'BUY', quantity
                )
                
                if order:
                    self.position = {
                        'side': 'BUY',
                        'price': current_price,
                        'quantity': quantity,
                        'order_id': order['orderId']
                    }
                    self.logger.info(f"Position opened: {self.position}")
        
        elif signal == 'SELL' and self.position:
            if self.position['side'] == 'BUY':
                order = self.place_market_order(
                    self.symbol, 'SELL', self.position['quantity']
                )
                
                if order:
                    profit = (current_price - self.position['price']) * self.position['quantity']
                    profit_pct = ((current_price - self.position['price']) / self.position['price']) * 100
                    self.logger.info(f"Position closed. Profit: ${profit:.2f} ({profit_pct:.2f}%)")
                    self.position = None
    
    def check_position(self):
        """Kiểm tra và quản lý position hiện tại"""
        current_price = self.get_ticker_price(self.symbol)
        
        if not current_price or not self.position:
            return
        
        should_close, reason = self.risk_manager.should_place_order(
            current_price,
            self.position['price'],
            'SELL'
        )
        
        if should_close:
            self.logger.info(f"Closing position due to: {reason}")
            order = self.place_market_order(
                self.symbol, 'SELL', self.position['quantity']
            )
            if order:
                self.position = None

if __name__ == '__main__':
    bot = TradingBot(symbol='BTCUSDT')
    bot.run()
```

## Best Practices

1. **Luôn test trên testnet trước**: Binance cung cấp testnet miễn phí
2. **Implement logging**: Ghi lại tất cả các hoạt động để debug
3. **Error handling**: Xử lý lỗi một cách graceful
4. **Rate limiting**: Binance có giới hạn số request, cần implement rate limiting
5. **Backtesting**: Test strategy trên dữ liệu lịch sử trước khi trade thật

## Kết luận

Trong bài viết này, chúng ta đã xây dựng một trading bot cơ bản với Binance API. Bot này có thể:
- Kết nối với Binance API
- Phân tích thị trường với moving averages
- Tự động đặt lệnh mua/bán
- Quản lý risk với stop loss và take profit

Trong các bài viết tiếp theo, chúng ta sẽ nâng cấp bot với các tính năng nâng cao hơn như backtesting, multiple strategies, và real-time monitoring.

---

**Lưu ý**: Trading có rủi ro. Luôn test kỹ trên testnet và chỉ đầu tư số tiền bạn có thể chấp nhận mất.
