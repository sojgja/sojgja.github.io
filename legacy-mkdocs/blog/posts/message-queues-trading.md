---
date: 2025-01-25
authors:
  - soigia
categories: [Trading Infrastructure, System Architecture]
title: Message Queues cho Trading Systems - RabbitMQ, Kafka và Event-Driven Architecture
description: >
  Sử dụng message queues để xây dựng scalable, reliable trading systems với RabbitMQ, Kafka và event-driven patterns.
---

# Message Queues cho Trading Systems

![Message Queue Architecture](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&h=600&fit=crop)

Trong kiến trúc của modern trading systems, message queues đóng vai trò như những đường ống dẫn quan trọng, kết nối các components khác nhau và đảm bảo dữ liệu được truyền tải một cách reliable, scalable, và efficient. Không giống như synchronous request-response patterns, message queues cho phép các services giao tiếp với nhau một cách asynchronous, decoupling producers và consumers, cho phép hệ thống scale từng component độc lập, và quan trọng nhất là đảm bảo không mất dữ liệu ngay cả khi một component tạm thời không available. Trong trading systems, nơi mà market data có thể đến với tốc độ hàng nghìn messages mỗi giây, và việc mất một message có thể dẫn đến quyết định trading sai lầm, message queues không chỉ là một nice-to-have mà là một requirement thiết yếu.

Trong bài viết chi tiết này, chúng ta sẽ cùng nhau khám phá cách sử dụng message queues trong trading systems, từ việc lựa chọn message queue phù hợp (RabbitMQ cho simple use cases, Apache Kafka cho high-throughput scenarios, Redis Streams cho low-latency requirements), thiết kế message schemas và routing strategies, implement producers và consumers, đến việc xử lý các edge cases như message ordering, duplicate detection, và dead letter queues. Chúng ta sẽ học cách implement các patterns phổ biến như pub/sub để broadcast market data đến nhiều consumers, request-reply patterns cho synchronous operations, và event sourcing để maintain state. Chúng ta cũng sẽ thảo luận về các trade-offs giữa different message queue technologies và khi nào nên sử dụng cái nào.

Bài viết này sẽ hướng dẫn bạn từng bước một, từ việc setup message queue infrastructure, thiết kế message schemas và topics, implement producers và consumers với error handling và retry logic, đến việc monitor và optimize performance. Chúng ta cũng sẽ học cách handle backpressure khi consumers không thể keep up với producers, implement message prioritization, và scale systems horizontally. Cuối cùng, bạn sẽ có trong tay kiến thức và tools cần thiết để xây dựng trading systems có khả năng scale và reliable cao.

<!-- more -->

## Tại sao cần Message Queues?

1. **Decoupling**: Tách các components độc lập
2. **Scalability**: Scale từng component riêng biệt
3. **Reliability**: Guaranteed delivery, retry mechanisms
4. **Buffering**: Handle traffic spikes
5. **Asynchronous Processing**: Non-blocking operations

## Message Queue Options

### 1. RabbitMQ
- **Pros**: Easy setup, good for simple use cases
- **Cons**: Performance limitations at very high throughput

### 2. Apache Kafka
- **Pros**: High throughput, distributed, durable
- **Cons**: More complex, overkill for simple cases

### 3. Redis Streams
- **Pros**: Simple, fast, good for real-time
- **Cons**: Less features than dedicated MQ

## Bước 1: RabbitMQ Setup

```python
# messaging/rabbitmq_setup.py
import pika
import json
import logging
from typing import Callable, Optional

class RabbitMQClient:
    """RabbitMQ client wrapper"""
    
    def __init__(self, host: str = 'localhost', port: int = 5672,
                 username: str = 'guest', password: str = 'guest'):
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=pika.PlainCredentials(username, password)
        )
        self.connection = None
        self.channel = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Establish connection"""
        self.connection = pika.BlockingConnection(self.connection_params)
        self.channel = self.connection.channel()
    
    def declare_exchange(self, exchange: str, exchange_type: str = 'topic'):
        """Declare exchange"""
        self.channel.exchange_declare(
            exchange=exchange,
            exchange_type=exchange_type,
            durable=True
        )
    
    def declare_queue(self, queue: str, durable: bool = True):
        """Declare queue"""
        self.channel.queue_declare(queue=queue, durable=durable)
    
    def bind_queue(self, queue: str, exchange: str, routing_key: str):
        """Bind queue to exchange"""
        self.channel.queue_bind(
            queue=queue,
            exchange=exchange,
            routing_key=routing_key
        )
    
    def publish(self, exchange: str, routing_key: str, message: dict):
        """Publish message"""
        self.channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
            )
        )
    
    def consume(self, queue: str, callback: Callable, auto_ack: bool = False):
        """Consume messages"""
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=callback,
            auto_ack=auto_ack
        )
        self.channel.start_consuming()
    
    def close(self):
        """Close connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
```

## Bước 2: Trading Event Publisher

```python
# messaging/trading_events.py
from messaging.rabbitmq_setup import RabbitMQClient
from typing import Dict
import logging

class TradingEventPublisher:
    """Publish trading events to message queue"""
    
    def __init__(self, rabbitmq: RabbitMQClient):
        self.mq = rabbitmq
        self.exchange = 'trading_events'
        self.logger = logging.getLogger(__name__)
        
        # Setup exchange
        self.mq.declare_exchange(self.exchange, 'topic')
    
    def publish_ticker_update(self, symbol: str, exchange: str, price: float):
        """Publish ticker update event"""
        event = {
            'type': 'ticker_update',
            'symbol': symbol,
            'exchange': exchange,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
        
        routing_key = f"ticker.{exchange}.{symbol}"
        self.mq.publish(self.exchange, routing_key, event)
        self.logger.debug(f"Published ticker update: {routing_key}")
    
    def publish_trade_executed(self, order_id: str, symbol: str, 
                             price: float, quantity: float, side: str):
        """Publish trade executed event"""
        event = {
            'type': 'trade_executed',
            'order_id': order_id,
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'side': side,
            'timestamp': datetime.now().isoformat()
        }
        
        routing_key = f"trade.{symbol}"
        self.mq.publish(self.exchange, routing_key, event)
        self.logger.info(f"Published trade executed: {order_id}")
    
    def publish_signal(self, symbol: str, signal: str, strength: float):
        """Publish trading signal"""
        event = {
            'type': 'signal',
            'symbol': symbol,
            'signal': signal,  # 'BUY', 'SELL', 'HOLD'
            'strength': strength,
            'timestamp': datetime.now().isoformat()
        }
        
        routing_key = f"signal.{symbol}"
        self.mq.publish(self.exchange, routing_key, event)
        self.logger.info(f"Published signal: {symbol} - {signal}")
```

## Bước 3: Event Consumers

```python
# messaging/event_consumers.py
from messaging.rabbitmq_setup import RabbitMQClient
import json
import logging
from typing import Dict

class TradingEventConsumer:
    """Consume trading events"""
    
    def __init__(self, rabbitmq: RabbitMQClient):
        self.mq = rabbitmq
        self.exchange = 'trading_events'
        self.logger = logging.getLogger(__name__)
    
    def setup_ticker_consumer(self, queue: str, routing_key: str):
        """Setup consumer for ticker updates"""
        self.mq.declare_queue(queue)
        self.mq.bind_queue(queue, self.exchange, routing_key)
        
        def callback(ch, method, properties, body):
            try:
                event = json.loads(body)
                self.handle_ticker_update(event)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                self.logger.error(f"Error processing ticker: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        self.mq.channel.basic_consume(
            queue=queue,
            on_message_callback=callback
        )
    
    def setup_trade_consumer(self, queue: str, routing_key: str):
        """Setup consumer for trade events"""
        self.mq.declare_queue(queue)
        self.mq.bind_queue(queue, self.exchange, routing_key)
        
        def callback(ch, method, properties, body):
            try:
                event = json.loads(body)
                self.handle_trade_executed(event)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                self.logger.error(f"Error processing trade: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        self.mq.channel.basic_consume(
            queue=queue,
            on_message_callback=callback
        )
    
    def handle_ticker_update(self, event: Dict):
        """Handle ticker update"""
        self.logger.info(f"Ticker update: {event['symbol']} = ${event['price']}")
        # Update database, cache, etc.
    
    def handle_trade_executed(self, event: Dict):
        """Handle trade executed"""
        self.logger.info(f"Trade executed: {event['order_id']}")
        # Update portfolio, send notifications, etc.
```

## Bước 4: Kafka Producer

```python
# messaging/kafka_producer.py
from kafka import KafkaProducer
import json
import logging
from typing import Dict

class KafkaTradingProducer:
    """Kafka producer for trading events"""
    
    def __init__(self, bootstrap_servers: List[str]):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        self.logger = logging.getLogger(__name__)
    
    def publish_ticker(self, topic: str, symbol: str, exchange: str, price: float):
        """Publish ticker to Kafka"""
        message = {
            'symbol': symbol,
            'exchange': exchange,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
        
        future = self.producer.send(topic, value=message, key=symbol.encode())
        
        # Wait for result
        try:
            record_metadata = future.get(timeout=10)
            self.logger.debug(
                f"Published to {record_metadata.topic} "
                f"partition {record_metadata.partition} "
                f"offset {record_metadata.offset}"
            )
        except Exception as e:
            self.logger.error(f"Error publishing: {e}")
    
    def flush(self):
        """Flush pending messages"""
        self.producer.flush()
    
    def close(self):
        """Close producer"""
        self.producer.close()
```

## Bước 5: Kafka Consumer

```python
# messaging/kafka_consumer.py
from kafka import KafkaConsumer
import json
import logging
from typing import Callable, List

class KafkaTradingConsumer:
    """Kafka consumer for trading events"""
    
    def __init__(self, bootstrap_servers: List[str], group_id: str):
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=False
        )
        self.logger = logging.getLogger(__name__)
    
    def subscribe(self, topics: List[str]):
        """Subscribe to topics"""
        self.consumer.subscribe(topics)
    
    def consume(self, callback: Callable):
        """Consume messages"""
        try:
            for message in self.consumer:
                try:
                    callback(message.value)
                    self.consumer.commit()
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    # Don't commit on error
        except KeyboardInterrupt:
            self.logger.info("Consumer stopped")
        finally:
            self.consumer.close()
```

## Bước 6: Event-Driven Trading System

```python
# messaging/event_driven_system.py
from messaging.trading_events import TradingEventPublisher
from messaging.event_consumers import TradingEventConsumer
from messaging.rabbitmq_setup import RabbitMQClient
import threading

class EventDrivenTradingSystem:
    """Event-driven trading system"""
    
    def __init__(self):
        # Setup message queue
        self.mq = RabbitMQClient()
        self.mq.connect()
        
        # Publishers
        self.event_publisher = TradingEventPublisher(self.mq)
        
        # Consumers
        self.event_consumer = TradingEventConsumer(self.mq)
        
        # Setup consumers
        self.setup_consumers()
    
    def setup_consumers(self):
        """Setup all consumers"""
        # Ticker consumer
        self.event_consumer.setup_ticker_consumer(
            'ticker_updates',
            'ticker.*.*'
        )
        
        # Trade consumer
        self.event_consumer.setup_trade_consumer(
            'trade_executions',
            'trade.*'
        )
    
    def start_consumers(self):
        """Start consuming in background thread"""
        def run():
            self.mq.channel.start_consuming()
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    
    def publish_ticker(self, symbol: str, exchange: str, price: float):
        """Publish ticker update"""
        self.event_publisher.publish_ticker_update(symbol, exchange, price)
    
    def publish_trade(self, order_id: str, symbol: str, price: float,
                     quantity: float, side: str):
        """Publish trade"""
        self.event_publisher.publish_trade_executed(
            order_id, symbol, price, quantity, side
        )

# Usage
if __name__ == '__main__':
    system = EventDrivenTradingSystem()
    system.start_consumers()
    
    # Publish events
    system.publish_ticker('BTCUSDT', 'binance', 50000)
    system.publish_trade('order123', 'BTCUSDT', 50000, 0.001, 'BUY')
```

## Best Practices

1. **Message Durability**: Enable persistence for critical messages
2. **Acknowledgment**: Use manual ack for reliability
3. **Error Handling**: Implement retry and dead letter queues
4. **Idempotency**: Make consumers idempotent
5. **Partitioning**: Partition by symbol for better parallelism
6. **Monitoring**: Monitor queue depth, consumer lag
7. **Backpressure**: Handle when consumers are slow

## Kết luận

Message queues enable:
- Scalable architecture
- Reliable message delivery
- Decoupled components
- Better fault tolerance

Chọn message queue phù hợp với requirements của bạn!
