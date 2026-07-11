---
id: event-driven
title: Event-Driven Architecture (EDA)
sidebar_label: ⚡ Event-Driven Architecture
sidebar_position: 38
---

# Event-Driven Architecture (EDA)

> *"Events are the atoms of the reactive universe. They are immutable, they happened in the past, and they carry meaning."* — **Gregor Hohpe**, *Enterprise Integration Patterns*

**Event-Driven Architecture (EDA)** là một kiểu kiến trúc phần mềm nơi các thành phần giao tiếp với nhau thông qua **sự kiện (events)** — những thông điệp bất biến ghi lại một điều gì đó đã xảy ra trong hệ thống. Không giống như kiến trúc request-response truyền thống nơi client gọi server và chờ phản hồi, EDA cho phép producer phát event mà không cần biết ai sẽ xử lý nó, và consumer nhận event mà không cần biết ai đã phát ra nó. Sự tách rời (decoupling) này là nền tảng cho các hệ thống real-time, scalable, và resilient.

---

## Bài toán

### Vấn đề: Xử lý đồng bộ trong thế giới real-time

Hãy tưởng tượng bạn đang xây dựng **một hệ thống giao dịch chứng khoán** cho một công ty chứng khoán lớn tại Việt Nam (giống như SSI hay VNDirect). Hệ thống phải xử lý hàng triệu lệnh giao dịch mỗi ngày, cập nhật giá cổ phiếu real-time từ HOSE và HNX, gửi cảnh báo cho hàng trăm nghìn user, và tạo báo cáo compliance cho UBCKNN.

Trong kiến trúc request-response truyền thống, flow xử lý một lệnh mua cổ phiếu sẽ như sau:

```
1. User gửi lệnh BUY 100 VIC @ 85,000
2. Hệ thống kiểm tra số dư (call User Service)
3. Hệ thống kiểm tra tồn kho chứng khoán (call Portfolio Service)
4. Hệ thống kiểm tra hạn mức (call Risk Service)
5. Hệ thống khớp lệnh (call Matching Engine)
6. Hệ thống ghi nhận giao dịch (call Database)
7. Hệ thống gửi email xác nhận (call Email Service)
8. Hệ thống gửi push notification (call Notification Service)
9. Hệ thống cập nhật dashboard (call WebSocket Service)
10. Hệ thống ghi audit log (call Audit Service)
```

Vấn đề với cách tiếp cận này:

**1. Tight coupling**: Module xử lý lệnh phải biết tất cả các service khác. Mỗi lần thêm một service mới (ví dụ: Anti-Money Laundering check), bạn phải sửa code ở module chính.

**2. Performance bottleneck**: Flow đồng bộ: nếu Email Service chậm (gọi API SendGrid mất 2 giây), user phải chờ 2 giây để nhận response mặc dù lệnh đã được khớp thành công.

**3. Reliability cascade**: Nếu Notification Service bị lỗi, toàn bộ flow đặt lệnh thất bại — user không thể đặt lệnh mặc dù mọi thứ đều ổn.

**4. Không scale được**: Mỗi service có nhu cầu scale khác nhau. Email Service có thể scale theo số lượng email, WebSocket scale theo số connection. Với flow đồng bộ, tất cả đều bị giới hạn bởi throughput của module chính.

**5. Khó thêm tính năng mới**: Muốn thêm AI-based fraud detection? Bạn phải sửa toàn bộ flow. Muốn thêm real-time data feed cho đối tác? Lại phải sửa.

### Giải pháp từ Event-Driven Architecture

EDA giải quyết triệt để các vấn đề trên bằng cách:

- **Order Service** chỉ phát event `OrderPlaced` khi user đặt lệnh thành công
- Các service khác (Email, Notification, Analytics, Audit, Risk) **lắng nghe** event này và xử lý độc lập
- Order Service **không cần biết** service nào đang lắng nghe
- Nếu một service bị lỗi, các service khác vẫn hoạt động bình thường
- Có thể thêm service mới chỉ bằng cách viết consumer mới — không cần sửa code cũ

Hãy tưởng tượng flow với EDA:

```
User đặt lệnh → Order Service kiểm tra và khớp lệnh (200ms)
              → Phát event OrderPlaced (5ms)
              → Trả response "Success" ngay lập tức

Các consumer xử lý bất đồng bộ:
  └─ Email Service: gửi email xác nhận (có thể mất 2 giây, user không cần chờ)
  └─ Notification Service: push notification (100ms)
  └─ Audit Service: ghi audit log (10ms)
  └─ Analytics Service: cập nhật báo cáo (500ms)
  └─ Risk Service: cập nhật hạn mức (50ms)
  └─ AML Service: kiểm tra rửa tiền (có thể mất 10 giây, nhưng không block)
```

---

## Nguyên lý thiết kế

### 1. Event là nguồn sự thật duy nhất

Event là một **sự thật đã xảy ra trong quá khứ** — nó không thể thay đổi, chỉ có thể được ghi nhận. Event dùng thì quá khứ để đặt tên: `OrderPlaced`, `PaymentReceived`, `InventoryUpdated`.

Mỗi event phải chứa:
- **Event ID**: Unique identifier
- **Timestamp**: Thời gian xảy ra
- **Aggregate ID**: ID của entity liên quan
- **Event Data**: Dữ liệu cụ thể (có thể là delta hoặc full snapshot)
- **Metadata**: Trace ID, user ID, version

### 2. Producer-Consumer Separation (Tách rời)

Producer **không biết** consumer nào đang lắng nghe. Consumer **không biết** producer nào phát ra event. Sự tách rời này cho phép:

- **Thêm consumer mới** mà không cần sửa producer
- **Loại bỏ consumer** mà không ảnh hưởng đến producer
- **Consumer tự quyết định** tốc độ xử lý của mình (back-pressure)

### 3. Eventually Consistency

Trong EDA, không có ACID transactions xuyên service. Thay vào đó, hệ thống đạt được **eventual consistency** — sau một khoảng thời gian, tất cả service sẽ đồng bộ. Chiến lược:

- **Saga Pattern**: Chuỗi local transactions, nếu một step fail thì thực hiện compensating transactions
- **Outbox Pattern**: Ghi event vào database trước, sau đó publish lên message broker
- **Idempotent Consumers**: Consumer phải handle trùng lặp event an toàn

### 4. Event Ordering and Partitioning

Thứ tự event có thể quan trọng (ví dụ: `AccountCreated` trước `DepositMade`). Kafka giải quyết bằng:

- **Key-based Partitioning**: Tất cả event của cùng một aggregate (ví dụ: account ID) vào cùng partition
- **Single Partition per Aggregate**: Đảm bảo thứ tự trong cùng partition
- **No Global Ordering**: Không cần thứ tự toàn cục — chỉ cần thứ tự trong aggregate

### 5. At-Least-Once Delivery

Message brokers cung cấp **at-least-once** delivery guarantee. Consumer phải là **idempotent**:

```python
async def handle_order_placed(event):
    # Check if already processed using event ID
    if await is_duplicate(event.event_id):
        return  # Skip duplicate
    # Process event
    await process(event)
```

### 6. Event Schema Evolution

Event schema thay đổi theo thời gian. Chiến lược quản lý:

- **Event versioning**: Mỗi event có version field (`OrderPlacedV1`, `OrderPlacedV2`)
- **Avro / Protobuf / JSON Schema**: Schema registry cho compatibility check
- **Backward compatible**: Consumer cũ vẫn đọc được event mới (chỉ thêm field, không xóa/rename)
- **Upcasting**: Transform event từ cũ sang mới khi đọc

---

## Cấu trúc chi tiết

### EDA Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EVENT-DRIVEN ARCHITECTURE                         │
│                                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐                      │
│  │    EVENT PRODUCERS   │    │    EVENT CONSUMERS   │                      │
│  │                      │    │                      │                      │
│  │  ┌─────────────────┐ │    │  ┌─────────────────┐ │                      │
│  │  │  Order Service   │─┼────┼─▶│ Email Service    │ │                      │
│  │  │  (order.placed)  │ │    │  │ (send email)     │ │                      │
│  │  └─────────────────┘ │    │  └─────────────────┘ │                      │
│  │  ┌─────────────────┐ │    │  ┌─────────────────┐ │                      │
│  │  │  Payment Service │─┼────┼─▶│ Audit Service    │ │                      │
│  │  │ (payment.recv'd) │ │    │  │ (audit log)      │ │                      │
│  │  └─────────────────┘ │    │  └─────────────────┘ │                      │
│  │  ┌─────────────────┐ │    │  ┌─────────────────┐ │                      │
│  │  │ Inventory Service│─┼────┼─▶│ Analytics Ser.  │ │                      │
│  │  │ (inv.updated)    │ │    │  │ (report)         │ │                      │
│  │  └─────────────────┘ │    │  └─────────────────┘ │                      │
│  │  ┌─────────────────┐ │    │  ┌─────────────────┐ │                      │
│  │  │ User Service     │─┼────┼─▶│ Notification Se.│ │                      │
│  │  │ (user.registered)│ │    │  │ (push/email/SMS) │ │                      │
│  │  └─────────────────┘ │    │  └─────────────────┘ │                      │
│  └─────────────────────┘    └─────────────────────┘                      │
│                     │                    ▲                                │
│                     │  Publish Events    │  Subscribe                     │
│                     ▼                    │                                │
│          ┌─────────────────────────────────────────┐                     │
│          │          MESSAGE BROKER / EVENT BUS      │                     │
│          │                                         │                     │
│          │  ┌──────────┐  ┌──────────┐  ┌────────┐ │                     │
│          │  │  Topic:   │  │  Topic:  │  │ Topic: │ │                     │
│          │  │order.evts│  │pay.evts  │  │user.evt│ │                     │
│          │  │Part 0..N │  │Part 0..N │  │Pa0..N  │ │                     │
│          │  └──────────┘  └──────────┘  └────────┘ │                     │
│          │                                         │                     │
│          │  Kafka / RabbitMQ / AWS SQS / NATS      │                     │
│          └─────────────────────────────────────────┘                     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │                    EVENT STORE (Optional)                       │      │
│  │  Lưu trữ tất cả event theo thời gian — dùng cho Event Sourcing │      │
│  └────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Detailed Structure

#### Event Producer

```
producer/
├── src/
│   ├── domain/
│   │   ├── events.py          # Domain event definitions
│   │   └── models.py          # Domain entities
│   ├── application/
│   │   ├── service.py         # Business logic
│   │   └── interfaces.py      # Ports (event publisher interface)
│   └── infrastructure/
│       ├── messaging/
│       │   ├── publisher.py   # Kafka/RabbitMQ publisher
│       │   └── serializers.py # Event serialization
│       ├── api/
│       │   ├── routes.py      # HTTP endpoints
│       │   └── middlewares.py # Auth, logging
│       └── persistence/
│           ├── models.py      # ORM models
│           └── repositories.py
```

#### Event Consumer

```
consumer/
├── src/
│   ├── domain/
│   │   ├── events.py          # Event handlers (import events from shared lib)
│   │   └── models.py
│   ├── application/
│   │   ├── handlers.py        # Event handler implementations
│   │   └── interfaces.py
│   └── infrastructure/
│       ├── messaging/
│       │   ├── consumer.py    # Kafka/RabbitMQ consumer loop
│       │   └── serializers.py # Deserialization
│       └── persistence/
│           ├── models.py
│           └── repositories.py
```

---

## Sơ đồ kiến trúc

```
   EDA — HỆ THỐNG GIAO DỊCH CHỨNG KHOÁN THỜI GIAN THỰC
   ======================================================

   ┌──────────────────┐      ┌──────────────────┐
   │  MOBILE APP      │      │  WEB DASHBOARD   │
   │  (Flutter/Kotlin)│      │  (React)         │
   └────────┬─────────┘      └────────┬─────────┘
            │                         │
            ▼                         ▼
   ┌──────────────────────────────────────────────────┐
   │              API GATEWAY (Kong / NGINX)           │
   └────┬────────────────────┬──────────────────┬─────┘
        │                    │                  │
        ▼                    ▼                  ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │ ORDER    │    │ MARKET DATA  │    │ USER         │
   │ SERVICE  │    │ SERVICE      │    │ SERVICE      │
   ├──────────┤    ├──────────────┤    ├──────────────┤
   │ - Place  │    │ - Price Feed │    │ - Auth       │
   │   Order  │    │ - HOSE/HNX   │    │ - Portfolio  │
   │ - Cancel │    │ - Real-time  │    │ - Watchlist  │
   │ - Modify │    │   Quotes     │    │ - Profile    │
   └─────┬────┘    └──────┬───────┘    └──────┬───────┘
         │                │                    │
         └────────────────┼────────────────────┘
                          │ PUBLISH EVENTS
                          ▼
   ╔══════════════════════════════════════════════════════╗
   ║               KAFKA EVENT STORE                       ║
   ╠══════════════════════════════════════════════════════╣
   ║  Topics:                                              ║
   ║  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ║
   ║  │ order.trades │  │ market.price │  │ user.act.  │  ║
   ║  │ Part: 0-9    │  │ Part: 0-3    │  │ Part: 0-3  │  ║
   ║  │ Retention: ∞ │  │ Retention:7d │  │ Retention∞ │  ║
   ║  └──────────────┘  └──────────────┘  └────────────┘  ║
   ║  More: order.alert, market.snapshot, risk.breach      ║
   ╚══════════════════════════════════════════════════════╝
         │                    │                    │
         │ SUBSCRIBE         │                    │
         ▼                    ▼                    ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │ NOTIF.   │    │ ALERT        │    │ COMPLIANCE   │
   │ SERVICE  │    │ ENGINE       │    │ AUDIT SERVICE│
   ├──────────┤    ├──────────────┤    ├──────────────┤
   │ - Email  │    │ - Price      │    │ - Trade Log  │
   │ - SMS    │    │   Threshold  │    │ - Report     │
   │ - Push   │    │ - Volume     │    │ - Analytics  │
   │          │    │   Alert      │    │ - AML Check  │
   └──────────┘    └──────────────┘    └──────────────┘
        │                  │                    │
        ▼                  ▼                    ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │ SENDGRID │    │ TWILIO       │    │ ELASTICSEARCH│
   │ (Email)  │    │ (SMS)        │    │ (Logging)    │
   └──────────┘    └──────────────┘    └──────────────┘

   EVENT FLOW:
   ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌────────┐
   │ Order   │────▶│ Kafka    │────▶│ Consumer │────▶│ Process│
   │ Service │     │ Producer │     │ Group    │     │ Logic  │
   │ (Source)│     │ (Topic)  │     │ (Handler)│     │ (Sink) │
   └─────────┘     └──────────┘     └──────────┘     └────────┘
```

---

## Ví dụ code hoàn chỉnh

### Cấu trúc project

```
trading_system/
├── common/
│   ├── __init__.py
│   ├── events.py              # Shared event definitions
│   ├── serializers.py         # Avro/JSON serialization
│   └── config.py              # Shared configuration
├── order-service/
│   ├── main.py                # FastAPI app + Kafka producer
│   ├── domain/
│   │   ├── models.py          # Order, Trade, Portfolio
│   │   └── exceptions.py
│   ├── application/
│   │   └── service.py         # Order matching logic
│   └── infrastructure/
│       ├── persistence.py     # Order repository
│       └── producer.py        # Kafka event producer
├── alert-engine/
│   ├── main.py                # Kafka consumer
│   ├── domain/
│   │   └── alerts.py          # Alert rules
│   └── infrastructure/
│       └── consumer.py        # Kafka consumer
├── audit-service/
│   ├── main.py
│   ├── domain/
│   │   └── audit_log.py
│   └── infrastructure/
│       ├── consumer.py
│       └── elastic.py         # Elasticsearch client
├── notification-service/
│   ├── main.py
│   ├── domain/
│   │   └── notifications.py
│   └── infrastructure/
│       ├── consumer.py
│       └── email.py           # SendGrid client
└── docker-compose.yml
```

### File: `common/events.py`

```python
"""Shared domain events for the trading system.
All services use these event definitions for inter-service communication."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional
from uuid import uuid4


class EventType(Enum):
    ORDER_PLACED = "order.placed"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_MODIFIED = "order.modified"
    ORDER_FILLED = "order.filled"
    ORDER_REJECTED = "order.rejected"
    PRICE_UPDATED = "market.price.updated"
    PORTFOLIO_CHANGED = "portfolio.changed"
    RISK_BREACH = "risk.breach"
    ACCOUNT_SUSPENDED = "account.suspended"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class DomainEvent:
    """Base event — all events inherit from this."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: EventType = field(init=False)
    aggregate_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    trace_id: str = ""
    user_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "data": self._data(),
        }

    def _data(self) -> Dict[str, Any]:
        return {}


@dataclass
class OrderPlaced(DomainEvent):
    event_type: EventType = EventType.ORDER_PLACED
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    price: float = 0.0
    quantity: int = 0
    total_value: float = 0.0

    def _data(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "price": self.price,
            "quantity": self.quantity,
            "total_value": self.total_value,
        }


@dataclass
class OrderFilled(DomainEvent):
    event_type: EventType = EventType.ORDER_FILLED
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    filled_price: float = 0.0
    filled_quantity: int = 0
    matched_order_id: str = ""

    def _data(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "matched_order_id": self.matched_order_id,
        }


@dataclass
class PriceUpdated(DomainEvent):
    event_type: EventType = EventType.PRICE_UPDATED
    symbol: str = ""
    exchange: str = ""
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    volume: int = 0

    def _data(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "bid": self.bid,
            "ask": self.ask,
            "last_price": self.last_price,
            "volume": self.volume,
        }


@dataclass
class RiskBreach(DomainEvent):
    event_type: EventType = EventType.RISK_BREACH
    account_id: str = ""
    breach_type: str = ""
    current_value: float = 0.0
    threshold: float = 0.0

    def _data(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "breach_type": self.breach_type,
            "current_value": self.current_value,
            "threshold": self.threshold,
        }
```

### File: `common/serializers.py`

```python
"""Event serialization and deserialization."""

import json
from datetime import datetime
from typing import Any, Dict, Type

from common.events import (
    DomainEvent,
    EventType,
    OrderFilled,
    OrderPlaced,
    PriceUpdated,
    RiskBreach,
)


class EventSerializer:
    """JSON serializer for domain events."""

    EVENT_TYPE_MAP: Dict[EventType, Type[DomainEvent]] = {
        EventType.ORDER_PLACED: OrderPlaced,
        EventType.ORDER_FILLED: OrderFilled,
        EventType.PRICE_UPDATED: PriceUpdated,
        EventType.RISK_BREACH: RiskBreach,
    }

    @staticmethod
    def serialize(event: DomainEvent) -> bytes:
        """Serialize event to JSON bytes."""
        payload = event.to_dict()
        return json.dumps(payload, default=str).encode("utf-8")

    @staticmethod
    def deserialize(data: bytes) -> DomainEvent:
        """Deserialize JSON bytes back to domain event."""
        payload = json.loads(data.decode("utf-8"))
        event_type = EventType(payload["event_type"])
        event_cls = EventSerializer.EVENT_TYPE_MAP.get(event_type)

        if event_cls is None:
            # Fall back to base event
            event = DomainEvent()
            event.event_type = event_type
            event.event_id = payload["event_id"]
            event.aggregate_id = payload["aggregate_id"]
            event.timestamp = datetime.fromisoformat(payload["timestamp"])
            event.version = payload["version"]
            event.trace_id = payload.get("trace_id", "")
            event.user_id = payload.get("user_id", "")
            return event

        data = payload.get("data", {})
        return event_cls(
            event_id=payload["event_id"],
            aggregate_id=payload["aggregate_id"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            version=payload["version"],
            trace_id=payload.get("trace_id", ""),
            user_id=payload.get("user_id", ""),
            **data,
        )
```

### File: `common/config.py`

```python
"""Shared configuration for all trading services."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    schema_registry_url: Optional[str] = None
    client_id: str = "trading-system"
    acks: str = "all"
    compression_type: str = "snappy"
    linger_ms: int = 5
    batch_size: int = 16384
    enable_idempotence: bool = True


@dataclass
class TradingServiceConfig:
    kafka: KafkaConfig = KafkaConfig()
    service_name: str = "unknown"
    consumer_group_id: str = "unknown-group"
    topics: tuple = ()
```

### File: `order-service/domain/models.py`

```python
"""Order domain models for the trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional
from uuid import uuid4


class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """A trading order."""
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    symbol: str = ""
    side: str = ""  # BUY / SELL
    order_type: str = ""  # MARKET / LIMIT / STOP
    price: float = 0.0
    quantity: int = 0
    filled_quantity: int = 0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    reject_reason: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    def fill(self, fill_quantity: int) -> None:
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot fill order in status {self.status}")
        self.filled_quantity += fill_quantity
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        self.updated_at = datetime.now()

    def cancel(self) -> None:
        if self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            raise ValueError(f"Cannot cancel order in status {self.status}")
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now()

    def reject(self, reason: str) -> None:
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason
        self.updated_at = datetime.now()


@dataclass
class Trade:
    """A matched trade between a buy and a sell order."""
    id: str = field(default_factory=lambda: str(uuid4()))
    buy_order_id: str = ""
    sell_order_id: str = ""
    symbol: str = ""
    price: float = 0.0
    quantity: int = 0
    total_value: float = 0.0
    executed_at: datetime = field(default_factory=datetime.now)
```

### File: `order-service/application/service.py`

```python
"""Order matching engine — core business logic."""

import logging
from typing import Dict, List, Optional

from order_service.domain.models import Order, OrderStatus, Trade
from common.events import OrderFilled, OrderPlaced, OrderSide

logger = logging.getLogger(__name__)


class MatchingEngine:
    """Simple price-time priority matching engine."""

    def __init__(self) -> None:
        # symbol -> list of buy orders (sorted: highest price first)
        self._buy_orders: Dict[str, List[Order]] = {}
        # symbol -> list of sell orders (sorted: lowest price first)
        self._sell_orders: Dict[str, List[Order]] = {}

    def place_order(self, order: Order) -> List[Trade]:
        """Place an order and try to match it immediately.
        Returns list of trades that were executed."""
        trades: List[Trade] = []

        if order.side == OrderSide.BUY.value:
            trades = self._match_buy(order)
            if order.remaining_quantity > 0:
                self._buy_orders.setdefault(order.symbol, [])
                self._buy_orders[order.symbol].append(order)
                self._buy_orders[order.symbol].sort(
                    key=lambda o: (-o.price, o.created_at)
                )
        else:
            trades = self._match_sell(order)
            if order.remaining_quantity > 0:
                self._sell_orders.setdefault(order.symbol, [])
                self._sell_orders[order.symbol].append(order)
                self._sell_orders[order.symbol].sort(
                    key=lambda o: (o.price, o.created_at)
                )

        logger.info(
            f"Order {order.id} ({order.side} {order.quantity} {order.symbol}): "
            f"{len(trades)} trades, remaining {order.remaining_quantity}"
        )
        return trades

    def _match_buy(self, buy_order: Order) -> List[Trade]:
        """Match a buy order against existing sell orders."""
        trades: List[Trade] = []
        sell_book = self._sell_orders.get(buy_order.symbol, [])

        remaining = buy_order.quantity
        matched_sells: List[int] = []

        for i, sell_order in enumerate(sell_book):
            if remaining <= 0:
                break
            if sell_order.status != OrderStatus.PENDING:
                continue
            # For limit orders, price must match
            if buy_order.price < sell_order.price:
                continue

            fill_qty = min(remaining, sell_order.remaining_quantity)
            if fill_qty <= 0:
                continue

            trade = Trade(
                buy_order_id=buy_order.id,
                sell_order_id=sell_order.id,
                symbol=buy_order.symbol,
                price=sell_order.price,
                quantity=fill_qty,
                total_value=round(fill_qty * sell_order.price, 2),
            )
            trades.append(trade)

            buy_order.fill(fill_qty)
            sell_order.fill(fill_qty)
            remaining -= fill_qty

            if sell_order.status == OrderStatus.FILLED:
                matched_sells.append(i)

        # Remove filled sell orders
        for i in reversed(matched_sells):
            sell_book.pop(i)

        self._sell_orders[buy_order.symbol] = sell_book
        return trades

    def _match_sell(self, sell_order: Order) -> List[Trade]:
        """Match a sell order against existing buy orders."""
        trades: List[Trade] = []
        buy_book = self._buy_orders.get(sell_order.symbol, [])

        remaining = sell_order.quantity
        matched_buys: List[int] = []

        for i, buy_order in enumerate(buy_book):
            if remaining <= 0:
                break
            if buy_order.status != OrderStatus.PENDING:
                continue
            if buy_order.price < sell_order.price:
                continue

            fill_qty = min(remaining, buy_order.remaining_quantity)
            if fill_qty <= 0:
                continue

            trade = Trade(
                buy_order_id=buy_order.id,
                sell_order_id=sell_order.id,
                symbol=sell_order.symbol,
                price=sell_order.price,
                quantity=fill_qty,
                total_value=round(fill_qty * sell_order.price, 2),
            )
            trades.append(trade)

            sell_order.fill(fill_qty)
            buy_order.fill(fill_qty)
            remaining -= fill_qty

            if buy_order.status == OrderStatus.FILLED:
                matched_buys.append(i)

        for i in reversed(matched_buys):
            buy_book.pop(i)

        self._buy_orders[sell_order.symbol] = buy_book
        return trades

    def get_order_book(self, symbol: str) -> dict:
        """Get current order book for a symbol."""
        buys = self._buy_orders.get(symbol, [])
        sells = self._sell_orders.get(symbol, [])
        return {
            "symbol": symbol,
            "bids": [
                {"price": o.price, "quantity": o.remaining_quantity, "orders": len(buys)}
                for o in buys[:5]
            ],
            "asks": [
                {"price": o.price, "quantity": o.remaining_quantity, "orders": len(sells)}
                for o in sells[:5]
            ],
        }
```

### File: `order-service/infrastructure/producer.py`

```python
"""Kafka event producer for Order Service."""

import json
import logging
from typing import Optional

from common.events import DomainEvent
from common.serializers import EventSerializer
from common.config import KafkaConfig

logger = logging.getLogger(__name__)


class KafkaOrderProducer:
    """Publishes order events to Kafka."""

    def __init__(self, config: KafkaConfig) -> None:
        self._config = config
        self._producer = None

    async def start(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._config.bootstrap_servers,
                acks=self._config.acks,
                compression_type=self._config.compression_type,
                linger_ms=self._config.linger_ms,
                batch_size=self._config.batch_size,
                enable_idempotence=self._config.enable_idempotence,
                value_serializer=lambda v: json.dumps(v, default=str).encode(),
            )
            await self._producer.start()
            logger.info("Kafka producer started")
        except ImportError:
            logger.warning("aiokafka not available. Using stub producer.")
            self._producer = None

    async def publish(self, event: DomainEvent, topic: str) -> None:
        """Publish a domain event to a Kafka topic."""
        if self._producer is None:
            logger.info(f"[STUB] Published {type(event).__name__} to {topic}")
            return

        payload = event.to_dict()
        key = event.aggregate_id.encode() if event.aggregate_id else None

        try:
            await self._producer.send(topic, key=key, value=payload)
            logger.debug(
                f"Published {type(event).__name__} to {topic} "
                f"[event_id={event.event_id[:8]}]"
            )
        except Exception as e:
            logger.error(f"Failed to publish event to {topic}: {e}")
            raise

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")
```

### File: `order-service/main.py`

```python
"""Order Service entry point — produces events."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from common.config import KafkaConfig, TradingServiceConfig
from common.events import OrderFilled, OrderPlaced, OrderSide, OrderType
from order_service.application.service import MatchingEngine
from order_service.domain.models import Order
from order_service.infrastructure.producer import KafkaOrderProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("order-service")


class OrderService:
    """Facade for order placement with event publishing."""

    def __init__(
        self,
        matching_engine: MatchingEngine,
        event_producer: KafkaOrderProducer,
    ) -> None:
        self._engine = matching_engine
        self._producer = event_producer

    async def place_order(
        self,
        user_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: int,
        order_type: str = "MARKET",
    ) -> dict:
        """Place an order and publish events."""
        order = Order(
            user_id=user_id,
            symbol=symbol,
            side=side.upper(),
            price=price,
            quantity=quantity,
            order_type=order_type.upper(),
        )

        # Publish OrderPlaced event
        placed_event = OrderPlaced(
            aggregate_id=order.id,
            user_id=user_id,
            symbol=symbol,
            side=OrderSide(side.upper()),
            order_type=OrderType(order_type.upper()),
            price=price,
            quantity=quantity,
            total_value=round(price * quantity, 2),
        )
        await self._producer.publish(placed_event, "order.events")

        # Execute matching
        trades = self._engine.place_order(order)

        # Publish OrderFilled events for each trade
        filled_events = []
        for trade in trades:
            filled_event = OrderFilled(
                aggregate_id=order.id,
                user_id=user_id,
                symbol=symbol,
                side=OrderSide(side.upper()),
                filled_price=trade.price,
                filled_quantity=trade.quantity,
                matched_order_id=(
                    trade.sell_order_id if side.upper() == "BUY"
                    else trade.buy_order_id
                ),
            )
            await self._producer.publish(filled_event, "order.events")
            filled_events.append(filled_event)

        return {
            "order_id": order.id,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "remaining_quantity": order.remaining_quantity,
            "trades": len(trades),
            "total_value": round(sum(t.total_value for t in trades), 2),
        }


async def main() -> None:
    """Demo: order matching with event-driven flow."""
    logger.info("=" * 60)
    logger.info("ORDER SERVICE — Event-Driven Trading Demo")
    logger.info("=" * 60)

    config = TradingServiceConfig(
        service_name="order-service",
        topics=("order.events",),
    )

    # Initialize
    engine = MatchingEngine()
    producer = KafkaOrderProducer(config.kafka)
    await producer.start()
    service = OrderService(engine, producer)

    # Demo: Place orders
    logger.info("\n📌 PLACING ORDERS...\n")

    # Sell orders (resting)
    sell_orders = [
        ("CUST-002", "VIC", "SELL", 86_000, 200),
        ("CUST-002", "VIC", "SELL", 87_000, 150),
        ("CUST-003", "VIC", "SELL", 85_500, 100),
    ]
    for user, sym, side, price, qty in sell_orders:
        result = await service.place_order(user, sym, side, price, qty, "LIMIT")
        logger.info(f"  [{side}] {user}: {qty} {sym} @ {price:,} — {result['status']}")

    logger.info("\n  Order book after sell orders:")
    book = engine.get_order_book("VIC")
    logger.info(f"  Bids: {book['bids']}")
    logger.info(f"  Asks: {book['asks']}")

    # Buy order that matches
    logger.info("\n📌 PLACING BUY ORDER (should match)...\n")
    result = await service.place_order("CUST-001", "VIC", "BUY", 86_000, 150, "LIMIT")
    logger.info(f"  [BUY] CUST-001: 150 VIC @ 86,000")
    logger.info(f"  Result: {result['status']} — filled {result['filled_quantity']}")

    logger.info("\n  Order book after matching:")
    book = engine.get_order_book("VIC")
    logger.info(f"  Bids: {book['bids']}")
    logger.info(f"  Asks: {book['asks']}")

    # Check order status
    logger.info("\n📌 ORDER STATUS:")
    # Note: In production, status would be queried from DB
    logger.info("  (order status persisted in event store)")

    await producer.stop()
    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

### File: `alert-engine/domain/alerts.py`

```python
"""Alert rules and configuration."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCategory(Enum):
    PRICE = "price"
    VOLUME = "volume"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    SYSTEM = "system"


@dataclass
class AlertRule:
    """A configurable alert rule."""
    id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity
    symbol: Optional[str] = None
    threshold: float = 0.0
    condition: str = ">"  # >, <, >=, <=, ==
    enabled: bool = True
    cooldown_seconds: int = 60  # Don't alert again within this window


@dataclass
class Alert:
    """An alert instance that was triggered."""
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    category: AlertCategory
    message: str
    symbol: str
    current_value: float
    threshold: float
    timestamp: str = ""
    user_id: str = ""
```

### File: `alert-engine/main.py`

```python
"""Alert Engine — consumes order and market events, triggers alerts."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional

from common.config import KafkaConfig, TradingServiceConfig
from common.events import EventType, OrderFilled, OrderPlaced, PriceUpdated
from common.serializers import EventSerializer
from alert_engine.domain.alerts import Alert, AlertCategory, AlertRule, AlertSeverity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ALERT: %(message)s",
)
logger = logging.getLogger("alert-engine")


class AlertEngine:
    """Consumes events and triggers alerts based on rules."""

    def __init__(self) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._last_triggered: Dict[str, datetime] = {}
        self._alerts: list = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Set up default alert rules."""
        rules = [
            AlertRule(
                id="price-spike-5pct",
                name="Price Spike > 5%",
                category=AlertCategory.PRICE,
                severity=AlertSeverity.WARNING,
                threshold=0.05,
                condition=">",
            ),
            AlertRule(
                id="large-trade-1b",
                name="Large Trade > 1B VND",
                category=AlertCategory.VOLUME,
                severity=AlertSeverity.CRITICAL,
                threshold=1_000_000_000,
                condition=">",
            ),
            AlertRule(
                id="consecutive-loss",
                name="Consecutive Loss Warning",
                category=AlertCategory.RISK,
                severity=AlertSeverity.WARNING,
                threshold=0.0,
                condition=">",
            ),
        ]
        for rule in rules:
            self._rules[rule.id] = rule

    async def handle_event(self, event_data: bytes) -> None:
        """Process an incoming event and check alert rules."""
        event = EventSerializer.deserialize(event_data)

        if isinstance(event, OrderPlaced):
            await self._check_order_alerts(event)
        elif isinstance(event, OrderFilled):
            await self._check_fill_alerts(event)
        elif isinstance(event, PriceUpdated):
            await self._check_price_alerts(event)

    async def _check_order_alerts(self, event: OrderPlaced) -> None:
        """Check if order triggers any alerts."""
        # Large order alert
        if event.total_value >= self._rules["large-trade-1b"].threshold:
            alert = Alert(
                rule_id="large-trade-1b",
                rule_name="Large Trade > 1B VND",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.VOLUME,
                message=(
                    f"🚨 GIAO DỊCH LỚN: {event.side.value} {event.quantity} "
                    f"{event.symbol} trị giá {event.total_value:,.0f} VND"
                ),
                symbol=event.symbol,
                current_value=event.total_value,
                threshold=1_000_000_000,
                timestamp=datetime.now().isoformat(),
                user_id=event.user_id,
            )
            self._trigger_alert(alert)

    async def _check_fill_alerts(self, event: OrderFilled) -> None:
        """Check if a fill triggers any alerts."""
        trade_value = event.filled_price * event.filled_quantity
        if trade_value >= 1_000_000_000:
            alert = Alert(
                rule_id="large-trade-1b",
                rule_name="Large Trade > 1B VND",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.VOLUME,
                message=(
                    f"🚨 LỆNH KHỚP LỚN: {event.side.value} {event.filled_quantity} "
                    f"{event.symbol} @ {event.filled_price:,.0f} "
                    f"(trị giá {trade_value:,.0f} VND)"
                ),
                symbol=event.symbol,
                current_value=trade_value,
                threshold=1_000_000_000,
                timestamp=datetime.now().isoformat(),
            )
            self._trigger_alert(alert)

    async def _check_price_alerts(self, event: PriceUpdated) -> None:
        """Check if price change triggers any alerts."""
        # This would compare against historical price in production
        logger.info(f"  Price update: {event.symbol} = {event.last_price:,.0f}")

    def _trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert with cooldown check."""
        now = datetime.now()
        last = self._last_triggered.get(alert.rule_id)

        if last and (now - last).total_seconds() < self._rules[alert.rule_id].cooldown_seconds:
            logger.debug(f"  Alert {alert.rule_id} suppressed (cooldown)")
            return

        self._last_triggered[alert.rule_id] = now
        self._alerts.append(alert)

        severity_icon = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
        }.get(alert.severity, "🔔")

        logger.info(f"{severity_icon} [{alert.severity.value}] {alert.message}")

    @property
    def triggered_alerts(self) -> list:
        return list(self._alerts)


class KafkaAlertConsumer:
    """Kafka consumer for the Alert Engine."""

    def __init__(self, config: KafkaConfig, engine: AlertEngine) -> None:
        self._config = config
        self._engine = engine
        self._consumer = None

    async def start(self) -> None:
        from aiokafka import AIOKafkaConsumer
        self._consumer = AIOKafkaConsumer(
            "order.events",
            "market.events",
            bootstrap_servers=self._config.bootstrap_servers,
            group_id="alert-engine",
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await self._consumer.start()
        logger.info("Alert Engine consumer started")

    async def consume(self) -> None:
        if not self._consumer:
            logger.error("Consumer not started")
            return

        try:
            async for msg in self._consumer:
                logger.debug(f"Received event: {msg.topic}/{msg.key}")
                await self._engine.handle_event(msg.value)
        except Exception as e:
            logger.error(f"Consumer error: {e}")

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()


async def main() -> None:
    config = KafkaConfig()
    engine = AlertEngine()
    consumer = KafkaAlertConsumer(config, engine)

    await consumer.start()
    logger.info("Alert Engine running. Waiting for events...")

    # In production, would run forever:
    # await consumer.consume()

    # For demo, simulate events
    await simulate_events(engine)

    await consumer.stop()


async def simulate_events(engine: AlertEngine) -> None:
    """Simulate events for demonstration."""
    from common.events import OrderPlaced, OrderSide, OrderType
    from common.serializers import EventSerializer

    logger.info("\n📌 SIMULATING EVENTS FOR ALERT TESTING\n")

    # Normal order (no alert)
    event1 = OrderPlaced(
        aggregate_id="ORD-001",
        user_id="CUST-001",
        symbol="VIC",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=85_000,
        quantity=100,
        total_value=8_500_000,
    )
    await engine.handle_event(EventSerializer.serialize(event1))

    # Large order (should trigger alert)
    event2 = OrderPlaced(
        aggregate_id="ORD-002",
        user_id="CUST-002",
        symbol="VNM",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        price=95_000,
        quantity=15_000,
        total_value=1_425_000_000,  # > 1B
    )
    await engine.handle_event(EventSerializer.serialize(event2))

    # Another normal order
    event3 = OrderPlaced(
        aggregate_id="ORD-003",
        user_id="CUST-003",
        symbol="HPG",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=28_000,
        quantity=500,
        total_value=14_000_000,
    )
    await engine.handle_event(EventSerializer.serialize(event3))

    logger.info(f"\n📊 TOTAL ALERTS TRIGGERED: {len(engine.triggered_alerts)}")
    for alert in engine.triggered_alerts:
        logger.info(f"  {alert.severity.value}: {alert.message}")


if __name__ == "__main__":
    asyncio.run(main())
```

### File: `notification-service/main.py`

```python
"""Notification Service — sends emails, SMS, and push notifications."""

import asyncio
import logging
from typing import Optional

from common.events import DomainEvent, EventType, OrderFilled, OrderPlaced, RiskBreach
from common.serializers import EventSerializer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] NOTIF: %(message)s",
)
logger = logging.getLogger("notification-service")


class NotificationService:
    """Sends notifications for various events."""

    def __init__(self) -> None:
        self._email_client = EmailClient()
        self._sms_client = SMSClient()

    async def handle_event(self, event_data: bytes) -> None:
        event = EventSerializer.deserialize(event_data)

        if isinstance(event, OrderPlaced):
            await self._handle_order_placed(event)
        elif isinstance(event, OrderFilled):
            await self._handle_order_filled(event)
        elif isinstance(event, RiskBreach):
            await self._handle_risk_breach(event)

    async def _handle_order_placed(self, event: OrderPlaced) -> None:
        message = (
            f"Lệnh {event.side.value} {event.quantity} CP {event.symbol} "
            f"giá {event.price:,.0f} VND đã được đặt thành công."
        )
        # In production, look up user email
        await self._email_client.send(
            to="user@example.com",
            subject=f"Xác nhận lệnh {event.symbol}",
            body=message,
        )
        logger.info(f"Email sent: {message}")

    async def _handle_order_filled(self, event: OrderFilled) -> None:
        message = (
            f"Lệnh {event.side.value} {event.symbol} "
            f"{event.filled_quantity}CP @ {event.filled_price:,.0f} VND "
            f"đã được khớp!"
        )
        await self._sms_client.send(
            to="+84123456789",
            message=message,
        )
        logger.info(f"SMS sent: {message}")

    async def _handle_risk_breach(self, event: RiskBreach) -> None:
        message = (
            f"CẢNH BÁO RỦI RO: Tài khoản {event.account_id} "
            f"vượt ngưỡng {event.breach_type} "
            f"(hiện tại: {event.current_value:,.0f}, ngưỡng: {event.threshold:,.0f})"
        )
        await self._email_client.send(
            to="risk@company.com",
            subject=f"🚨 Risk Breach: {event.breach_type}",
            body=message,
        )
        logger.info(f"Risk alert sent: {message}")


class EmailClient:
    """Stub email client (would use SendGrid/Mailgun in production)."""

    async def send(self, to: str, subject: str, body: str) -> None:
        await asyncio.sleep(0.05)  # Simulate network latency
        logger.debug(f"  [EMAIL] To: {to} | Subject: {subject}")


class SMSClient:
    """Stub SMS client (would use Twilio in production)."""

    async def send(self, to: str, message: str) -> None:
        await asyncio.sleep(0.03)  # Simulate latency
        logger.debug(f"  [SMS] To: {to} | Message: {message[:50]}...")


async def main() -> None:
    """Demo: simulate events and notifications."""
    logger.info("=" * 60)
    logger.info("NOTIFICATION SERVICE — Event Consumer Demo")
    logger.info("=" * 60)

    service = NotificationService()

    # Simulate events
    from common.events import OrderFilled, OrderPlaced, OrderSide, OrderType
    from common.serializers import EventSerializer

    events = [
        OrderPlaced(
            aggregate_id="ORD-001",
            user_id="CUST-001",
            symbol="VIC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            price=85_000,
            quantity=500,
            total_value=42_500_000,
        ),
        OrderFilled(
            aggregate_id="ORD-001",
            user_id="CUST-001",
            symbol="VIC",
            side=OrderSide.BUY,
            filled_price=85_000,
            filled_quantity=500,
            matched_order_id="ORD-002",
        ),
    ]

    for event in events:
        data = EventSerializer.serialize(event)
        await service.handle_event(data)

    logger.info("\n✅ Notifications sent successfully!")


if __name__ == "__main__":
    asyncio.run(main())
```

### File: `audit-service/main.py`

```python
"""Audit Service — records all events for compliance and reporting."""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from common.events import DomainEvent
from common.serializers import EventSerializer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] AUDIT: %(message)s",
)
logger = logging.getLogger("audit-service")


class AuditRecord:
    """An audit log entry."""
    def __init__(
        self,
        event: DomainEvent,
        received_at: Optional[datetime] = None,
    ) -> None:
        self.event_id = event.event_id
        self.event_type = event.event_type.value
        self.aggregate_id = event.aggregate_id
        self.timestamp = event.timestamp
        self.user_id = event.user_id
        self.trace_id = event.trace_id
        self.data = event.to_dict()
        self.received_at = received_at or datetime.now()


class AuditService:
    """Consumes and persists all domain events for audit trail."""

    def __init__(self) -> None:
        self._records: List[AuditRecord] = []

    async def handle_event(self, event_data: bytes) -> None:
        event = EventSerializer.deserialize(event_data)
        record = AuditRecord(event)
        self._records.append(record)
        logger.info(
            f"📝 AUDIT: {event.event_type.value} "
            f"[{event.aggregate_id[:8]}] "
            f"by user {event.user_id or 'system'}"
        )

    async def search(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        aggregate_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[AuditRecord]:
        """Search audit records."""
        results = self._records
        if event_type:
            results = [r for r in results if r.event_type == event_type]
        if user_id:
            results = [r for r in results if r.user_id == user_id]
        if aggregate_id:
            results = [r for r in results if r.aggregate_id == aggregate_id]
        return results[-limit:]

    @property
    def total_events(self) -> int:
        return len(self._records)


async def main() -> None:
    """Demo: audit service consuming events."""
    logger.info("=" * 60)
    logger.info("AUDIT SERVICE — Event Sink Demo")
    logger.info("=" * 60)

    service = AuditService()

    # Simulate events
    from common.events import OrderFilled, OrderPlaced, OrderSide, OrderType
    from common.serializers import EventSerializer

    events = [
        OrderPlaced(
            aggregate_id="ORD-001", user_id="CUST-001",
            symbol="VIC", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, price=85_000, quantity=200,
            total_value=17_000_000,
        ),
        OrderFilled(
            aggregate_id="ORD-001", user_id="CUST-001",
            symbol="VIC", side=OrderSide.BUY,
            filled_price=85_000, filled_quantity=200,
            matched_order_id="ORD-002",
        ),
        OrderPlaced(
            aggregate_id="ORD-003", user_id="CUST-002",
            symbol="VNM", side=OrderSide.SELL,
            order_type=OrderType.MARKET, price=95_000, quantity=1000,
            total_value=95_000_000,
        ),
    ]

    for event in events:
        data = EventSerializer.serialize(event)
        await service.handle_event(data)

    logger.info(f"\n📊 Total audit records: {service.total_events}")

    # Search demo
    results = await service.search(event_type="order.placed")
    logger.info(f"  Search 'order.placed': {len(results)} records")
    for r in results:
        logger.info(f"    - {r.event_type} | {r.aggregate_id} | {r.timestamp}")

    logger.info("\n✅ Audit completed!")


if __name__ == "__main__":
    asyncio.run(main())
```

### File: `docker-compose.yml` (trading system)

```yaml
version: "3.9"

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1

  order-service:
    build:
      context: ./order-service
    depends_on:
      - kafka
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092

  alert-engine:
    build:
      context: ./alert-engine
    depends_on:
      - kafka
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092

  notification-service:
    build:
      context: ./notification-service
    depends_on:
      - kafka
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092

  audit-service:
    build:
      context: ./audit-service
    depends_on:
      - kafka
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
```

---

## Kiểm thử

### File: `tests/test_matching_engine.py`

```python
"""Tests for the order matching engine."""

import pytest

from order_service.domain.models import Order
from order_service.application.service import MatchingEngine


@pytest.fixture
def engine() -> MatchingEngine:
    return MatchingEngine()


class TestMatchingEngine:

    def test_limit_buy_matches_sell(self, engine: MatchingEngine):
        """Buy order should match against existing sell orders."""
        sell = Order(symbol="VIC", side="SELL", price=86_000, quantity=100)
        engine.place_order(sell)

        buy = Order(symbol="VIC", side="BUY", price=86_000, quantity=50)
        trades = engine.place_order(buy)

        assert len(trades) == 1
        assert trades[0].quantity == 50
        assert trades[0].price == 86_000
        assert buy.status.name == "FILLED"

    def test_partial_fill(self, engine: MatchingEngine):
        """Partial fill when buy quantity exceeds sell."""
        sell = Order(symbol="VIC", side="SELL", price=85_000, quantity=100)
        engine.place_order(sell)

        buy = Order(symbol="VIC", side="BUY", price=85_000, quantity=150)
        trades = engine.place_order(buy)

        assert len(trades) == 1
        assert trades[0].quantity == 100
        assert buy.filled_quantity == 100
        assert buy.status.name == "PARTIALLY_FILLED"

    def test_no_match_if_price_too_low(self, engine: MatchingEngine):
        """Buy should not match if bid price is below ask."""
        sell = Order(symbol="VIC", side="SELL", price=86_000, quantity=100)
        engine.place_order(sell)

        buy = Order(symbol="VIC", side="BUY", price=85_000, quantity=50)
        trades = engine.place_order(buy)

        assert len(trades) == 0
        assert buy.status.name == "PENDING"
        assert buy.remaining_quantity == 50

    def test_market_buy_matches_lowest_ask(self, engine: MatchingEngine):
        """Market buy should match at the lowest sell price."""
        engine.place_order(Order(symbol="VIC", side="SELL", price=87_000, quantity=100))
        engine.place_order(Order(symbol="VIC", side="SELL", price=85_000, quantity=50))
        engine.place_order(Order(symbol="VIC", side="SELL", price=86_000, quantity=200))

        buy = Order(symbol="VIC", side="BUY", price=100_000, quantity=80)
        trades = engine.place_order(buy)

        assert len(trades) == 2
        assert trades[0].price == 85_000  # Match lowest first
        assert trades[0].quantity == 50
        assert trades[1].price == 86_000  # Then next lowest
        assert trades[1].quantity == 30

    def test_multiple_matches(self, engine: MatchingEngine):
        """Multiple sell orders matched by one buy."""
        for price in [85_000, 85_500, 86_000]:
            engine.place_order(Order(symbol="VIC", side="SELL", price=price, quantity=100))

        buy = Order(symbol="VIC", side="BUY", price=86_000, quantity=250)
        trades = engine.place_order(buy)

        assert len(trades) == 3
        assert sum(t.quantity for t in trades) == 250
        assert buy.status.name == "FILLED"

    def test_order_book_structure(self, engine: MatchingEngine):
        """Order book should show correct bids and asks."""
        engine.place_order(Order(symbol="VIC", side="SELL", price=87_000, quantity=100))
        engine.place_order(Order(symbol="VIC", side="BUY", price=85_000, quantity=50))

        book = engine.get_order_book("VIC")
        assert "bids" in book
        assert "asks" in book
        assert len(book["bids"]) == 1
        assert len(book["asks"]) == 1

    def test_cancel_order_prevents_matching(self, engine: MatchingEngine):
        """Cancelled order should not be matched."""
        sell = Order(symbol="VIC", side="SELL", price=85_000, quantity=100)
        engine.place_order(sell)
        sell.cancel()

        buy = Order(symbol="VIC", side="BUY", price=85_000, quantity=50)
        trades = engine.place_order(buy)

        assert len(trades) == 0
```

### File: `tests/test_event_serialization.py`

```python
"""Tests for event serialization/deserialization."""

import pytest
from datetime import datetime

from common.events import OrderFilled, OrderPlaced, OrderSide, OrderType
from common.serializers import EventSerializer


class TestEventSerializer:

    def test_serialize_deserialize_order_placed(self):
        original = OrderPlaced(
            event_id="EVT-001",
            aggregate_id="ORD-001",
            user_id="CUST-001",
            timestamp=datetime(2024, 6, 1, 10, 0, 0),
            symbol="VIC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=85_000,
            quantity=100,
            total_value=8_500_000,
        )

        data = EventSerializer.serialize(original)
        restored = EventSerializer.deserialize(data)

        assert isinstance(restored, OrderPlaced)
        assert restored.event_id == original.event_id
        assert restored.symbol == "VIC"
        assert restored.side == OrderSide.BUY
        assert restored.price == 85_000

    def test_serialize_deserialize_order_filled(self):
        original = OrderFilled(
            event_id="EVT-002",
            aggregate_id="ORD-001",
            symbol="VIC",
            side=OrderSide.SELL,
            filled_price=86_000,
            filled_quantity=50,
            matched_order_id="ORD-002",
        )

        data = EventSerializer.serialize(original)
        restored = EventSerializer.deserialize(data)

        assert isinstance(restored, OrderFilled)
        assert restored.filled_price == 86_000
        assert restored.filled_quantity == 50

    def test_round_trip_preserves_all_fields(self):
        original = OrderPlaced(
            aggregate_id="ORD-001",
            user_id="CUST-001",
            trace_id="TRACE-001",
            symbol="VNM",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            price=95_000,
            quantity=1000,
            total_value=95_000_000,
        )

        data = EventSerializer.serialize(original)
        restored = EventSerializer.deserialize(data)

        assert restored.aggregate_id == original.aggregate_id
        assert restored.user_id == original.user_id
        assert restored.trace_id == original.trace_id
        assert restored.price == original.price
        assert restored.quantity == original.quantity
```

---

## Khi nào dùng / Khi nào không

### ✅ Khi nào dùng Event-Driven Architecture

| Tình huống | Lý do |
|-----------|-------|
| **Real-time systems** (trading, IoT, monitoring) | Event xử lý tức thời, push-based |
| **Hệ thống nhiều service phải phối hợp** | Decoupling qua event bus |
| **Cần audit trail đầy đủ** | Event là nguồn sự thật |
| **Workflow dài, nhiều bước** | Saga pattern, mỗi step là một event |
| **Cần scale horizontal cho consumer** | Consumer scale độc lập, không ảnh hưởng producer |
| **Hệ thống notification/push** | Email, SMS, push — tất cả đều event-driven |
| **CDC (Change Data Capture)** | Capture DB changes thành event |

### ❌ Khi nào KHÔNG dùng

| Tình huống | Lý do | Alternative |
|-----------|-------|-------------|
| **Cần strong consistency tức thời** | Eventually consistency không đáp ứng | Layered + ACID transaction |
| **Hệ thống CRUD đơn giản** | Overhead của message broker không cần thiết | REST API, Layered |
| **Debugging/testing khó khăn** | Async flow khó trace | Request-response synchronous |
| **Complexity không justify** | EDA thêm nhiều moving parts | Monolith với observer pattern |
| **Team chưa có kinh nghiệm** | EDA learning curve cao | Start với synchronous, evolve sau |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Decoupling hoàn toàn**: Producer không biết consumer, consumer không biết producer | **Eventually consistency**: Dữ liệu không consistent tức thời, cần xử lý conflict |
| **Scalability**: Consumer scale độc lập, không ảnh hưởng producer | **Debugging khó**: Flow không đồng bộ, khó trace khi có lỗi |
| **Resilience**: Consumer fail không ảnh hưởng producer, event được replay | **Duplicate events**: At-least-once delivery → cần idempotent consumer |
| **Auditability**: Mọi event đều được ghi lại, có thể replay để rebuild state | **Event ordering**: Không đảm bảo thứ tự toàn cục, chỉ trong partition |
| **Extensibility**: Thêm consumer mới không cần sửa code cũ | **Schema evolution**: Event schema thay đổi → backward compatibility |
| **Performance**: Async processing không block request | **Complexity**: Message broker, schema registry, distributed tracing |
| **Real-time capability**: Push-based, phản ứng tức thời | **Latency**: Async có độ trễ cao hơn sync processing |

---

## Công cụ và Framework

### Message Brokers

| Công cụ | Đặc điểm | Use case |
|---------|----------|----------|
| **Apache Kafka** | High throughput, durable, partitioned, replayable | Enterprise event streaming — **recommended** |
| **RabbitMQ** | AMQP, reliable, flexible routing | Message queue, task distribution |
| **AWS SQS + SNS** | Managed, serverless, auto-scaling | Cloud-native, serverless apps |
| **NATS** | Lightweight, high performance, simple | Microservices, IoT, real-time |
| **Redis Pub/Sub** | In-memory, fast, ephemeral | Real-time chat, notifications |

### Event Processing

| Công cụ | Mục đích |
|---------|----------|
| **Kafka Streams** | Stream processing trong JVM |
| **Apache Flink** | Real-time stream processing, complex event processing |
| **Bytewax** | Python stream processing (Kafka → Python) |
| **Debezium** | Change Data Capture (DB → Kafka) |
| **Schema Registry** (Confluent) | Avro/Protobuf schema management |

### Frameworks

| Framework | Mục đích |
|-----------|----------|
| **FastAPI + aiokafka** | Python async event-driven microservices |
| **Spring Cloud Stream** | Java event-driven microservices |
| **NestJS + Kafka/RabbitMQ** | Node.js event-driven architecture |
| **Celery + Redis/RabbitMQ** | Python task queue (có thể dùng cho event processing) |

---

## Kiểm thử chiến lược

```python
# tests/conftest.py — shared fixtures

import asyncio
from typing import Generator
import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def kafka_producer():
    """Create a test Kafka producer (requires Kafka running)."""
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
    )
    await producer.start()
    yield producer
    await producer.stop()


@pytest.fixture
async def kafka_consumer():
    """Create a test Kafka consumer."""
    consumer = AIOKafkaConsumer(
        "test-topic",
        bootstrap_servers="localhost:9092",
        group_id="test-group",
        auto_offset_reset="earliest",
    )
    await consumer.start()
    yield consumer
    await consumer.stop()
```

### Testing Patterns for EDA

| Pattern | Mô tả | Code |
|---------|-------|------|
| **In-process event bus** | Test without Kafka — dùng in-memory event bus | `InMemoryEventBus` class |
| **Event store testing** | Verify events produced correctly | Assert event count, payload |
| **Consumer idempotency** | Send duplicate event, verify processed once | Send same event twice |
| **Timeout and retry** | Test consumer resilience | Mock slow responses |
| **Schema evolution** | Test consumer with old/new version events | Serialize with different versions |

---

## Kết luận

Event-Driven Architecture là một trong những kiến trúc quan trọng nhất trong kỷ nguyên real-time và distributed systems. Nó cho phép xây dựng hệ thống linh hoạt, scalable, và resilient — những phẩm chất mà kiến trúc request-response truyền thống không thể đạt được ở quy mô lớn.

### Best Practices

1. **Event là immutable** — không bao giờ sửa event đã publish
2. **Idempotent consumers** — duplicate event không gây hại
3. **At-least-once delivery** — chấp nhận duplicate, không chấp nhận mất event
4. **Event schema registry** — quản lý version và compatibility
5. **Distributed tracing** — trace_id xuyên suốt để debug
6. **Saga pattern** cho distributed transactions — không dùng 2PC
7. **Outbox pattern** — ghi event vào DB trước khi publish
8. **Dead letter queue** — event lỗi không được bỏ qua
9. **Monitoring alerting** — consumer lag, error rate, throughput
10. **Event replay** — có thể rebuild state từ event store

### Golden Rules

> 1. **Event là fact, không phải command.** Tên event ở thì quá khứ: `OrderPlaced`, không phải `PlaceOrder`.
> 2. **Một event, một trách nhiệm.** Không gộp nhiều sự kiện vào một event.
> 3. **Consumer tự quyết định tốc độ.** Không để producer dictate consumer speed.
> 4. **Design for failure.** Network, broker, consumer đều có thể fail.
> 5. **Event schema chỉ thêm, không xóa.** Backward compatibility là bắt buộc.

### Next Steps

Sau EDA, hãy tìm hiểu **Event Sourcing** — lưu trữ event làm nguồn sự thật duy nhất, và **CQRS** — tách command và query để tối ưu cả write và read. Hoặc quay lại **Hexagonal Architecture** để hiểu cách tổ chức code trong từng service sao cho testable và domain-centric.
