---
id: hexagonal-architecture
title: Hexagonal Architecture (Ports & Adapters)
sidebar_label: 🔷 Hexagonal Architecture
sidebar_position: 39
---

# Hexagonal Architecture (Ports & Adapters)

> *"Allow an application to equally be driven by users, programs, automated test or batch scripts, and to be developed and tested in isolation from its eventual run-time devices and databases."* — **Alistair Cockburn**, *người tạo ra Hexagonal Architecture*

**Hexagonal Architecture** (còn gọi là **Ports & Adapters Architecture** hay **Clean Architecture's spiritual predecessor**) là một kiểu kiến trúc phần mềm được Alistair Cockburn giới thiệu vào năm 2005. Ý tưởng cốt lõi là đặt **business logic** vào trung tâm (inside) và tất cả các tương tác với thế giới bên ngoài (database, web, message queue, UI) đều thông qua các **ports** (cổng) và **adapters** (bộ chuyển đổi) ở rìa (outside). Điều này tạo ra một hệ thống mà business logic hoàn toàn độc lập với infrastructure — có thể test, phát triển, và thay đổi mà không bị ảnh hưởng bởi công nghệ bên ngoài.

---

## Bài toán

### Vấn đề: Business logic bị "nhiễm" infrastructure

Hãy tưởng tượng bạn đang xây dựng **một hệ thống quản lý đơn hàng cho một chuỗi cửa hàng bán lẻ** — giống như The Gioi Di Dong hay Dien May Xanh. Hệ thống phải xử lý đơn hàng từ nhiều kênh: web, mobile app, cửa hàng vật lý (POS), và API từ đối tác (Shopee, Lazada). Mỗi kênh có định dạng dữ liệu khác nhau, giao thức khác nhau, và yêu cầu khác nhau.

Trong kiến trúc layered truyền thống, bạn thường thấy code như thế này:

```python
# VI PHẠM: Business logic phụ thuộc trực tiếp vào framework
from django.db import models
from django.core.mail import send_mail

class OrderService:
    def place_order(self, order_data):
        # Business logic xen lẫn với ORM
        order = OrderModel.objects.create(
            customer_id=order_data['customer_id'],
            total=order_data['total'],
            status='PENDING'
        )
        
        # Gọi trực tiếp email service
        send_mail(
            subject='Order Confirmation',
            message=f'Order {order.id} placed',
            from_email='shop@example.com',
            recipient_list=[order_data['email']]
        )
        
        # Log trực tiếp ra file
        logger.info(f'Order {order.id} created')
        
        return order
```

Vấn đề với cách tiếp cận này:

**1. Business logic gắn chặt với framework**: Service phụ thuộc vào Django ORM, Django email. Nếu bạn muốn chuyển từ Django sang FastAPI, bạn phải viết lại toàn bộ business logic — mặc dù business logic không hề thay đổi.

**2. Khó test**: Để test `place_order`, bạn cần database thật (Django test database), email server thật (hoặc mock phức tạp), và logger. Một unit test đơn giản trở thành integration test nặng nề.

**3. Công nghệ quyết định kiến trúc**: Bạn chọn Django → bạn phải dùng Django ORM, Django admin, Django templates. Business logic bị ép phải tuân theo cách của framework.

**4. Khó thay đổi công nghệ**: Muốn đổi từ PostgreSQL sang MongoDB? Bạn phải sửa tất cả model, service, và query. Muốn đổi từ REST sang GraphQL? Lại sửa toàn bộ controller.

**5. Violation of Dependency Inversion Principle**: Module cấp cao (business logic) phụ thuộc vào module cấp thấp (database, email). Đáng lẽ cả hai phải phụ thuộc vào abstraction.

### Hexagonal Architecture giải quyết vấn đề này như thế nào?

Hexagonal Architecture đảo ngược dependency: thay vì business logic phụ thuộc vào infrastructure, **infrastructure phụ thuộc vào business logic**. Cụ thể:

- **Domain (inside)**: Business logic thuần túy, không import gì từ framework hay infrastructure. Chỉ có plain Python objects, dataclasses, ABCs.
- **Ports (cổng)**: Interfaces được định nghĩa ở domain layer — đây là "hợp đồng" mà infrastructure phải tuân theo.
- **Adapters (bộ chuyển đổi)**: Implement ports ở infrastructure layer — đây là nơi framework, database, email, message queue sống.
- **Dependency Rule**: Domain không biết gì về infrastructure. Infrastructure biết domain qua ports.

Kết quả là bạn có thể:
- Test business logic mà không cần database, không cần web server
- Thay đổi database từ PostgreSQL sang MongoDB chỉ bằng cách viết adapter mới
- Thêm kênh bán hàng mới (API đối tác) chỉ bằng adapter mới
- Phát triển business logic trước, infrastructure sau (use-case driven)

---

## Nguyên lý thiết kế

### 1. Dependency Inversion Principle (DIP)

Đây là nguyên lý nền tảng của Hexagonal Architecture:

```
TRADITIONAL LAYERED:           HEXAGONAL (DIP applied):
Business → Data Access         Business ← Port (interface)
    ↓                               ↑
Data Access Layer               Data Access → Implements Port
    ↓
Database
```

Trong Hexagonal:
- **Business layer** định nghĩa **ports** (interfaces) cho những gì nó cần từ bên ngoài
- **Infrastructure** cung cấp **adapters** implement các ports đó
- Business layer **không import** bất kỳ thư viện infrastructure nào

### 2. Inside-Out Design

Không giống như layered architecture (bắt đầu từ database → lên business → UI), Hexagonal bắt đầu từ inside (domain) → out (infrastructure):

1. **Domain models** — entities, value objects
2. **Use cases (ports)** — business operations, repository interfaces
3. **Adapters** — implement ports cho từng công nghệ cụ thể

### 3. Port là "Hợp đồng" (Contract)

Có hai loại port:

- **Inbound ports (driving ports)**: API mà bên ngoài gọi vào hệ thống. Ví dụ: `CreateOrderUseCase`, `GetOrderQuery`.
- **Outbound ports (driven ports)**: API mà hệ thống gọi ra bên ngoài. Ví dụ: `OrderRepository`, `EmailSender`, `PaymentGateway`.

### 4. Adapter là "Cầu nối"

- **Inbound adapters (driving adapters)**: Controllers, CLI, message consumers — chuyển đổi input từ external format → domain call.
- **Outbound adapters (driven adapters)**: Repository implementations, email senders, message producers — chuyển đổi domain call → external action.

### 5. The Dependency Rule

> Source code dependencies can only point INWARD. Nothing in an inner circle can know about something in an outer circle.

- Domain (trong cùng): Không biết gì về bên ngoài
- Application (use cases): Chỉ biết domain, không biết infrastructure
- Infrastructure (ngoài cùng): Biết domain và application qua ports

### 6. Aggressive Testing Isolation

Vì business logic không phụ thuộc vào infrastructure, bạn có thể:
- **Unit test domain entities** 100% cô lập
- **Unit test use cases** với mock/stub adapters
- **Integration test adapters** riêng biệt
- **E2E test** full system với real adapters

---

## Cấu trúc chi tiết

### The Hexagon

```
                    ┌─────────────────────────────────┐
                    │        INBOUND ADAPTERS          │
                    │  (Controllers, CLI, Consumer)    │
                    └──────────┬──────────────────────┘
                               │ Gọi Inbound Ports
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                    APPLICATION LAYER                          │   │
│   │  ┌────────────────────────────────────────────────────────┐  │   │
│   │  │                    INBOUND PORTS                        │  │   │
│   │  │  CreateOrderUseCase (interface)                        │  │   │
│   │  │  CancelOrderUseCase (interface)                        │  │   │
│   │  │  GetOrderQuery (interface)                             │  │   │
│   │  └────────────────────────────────────────────────────────┘  │   │
│   │                                                               │   │
│   │  ┌────────────────────────────────────────────────────────┐  │   │
│   │  │                    USE CASE IMPLEMENTATIONS              │  │   │
│   │  │  CreateOrderService implements CreateOrderUseCase       │  │   │
│   │  │  CancelOrderService implements CancelOrderUseCase       │  │   │
│   │  └────────────────────────────────────────────────────────┘  │   │
│   │                                                               │   │
│   │  ┌────────────────────────────────────────────────────────┐  │   │
│   │  │                    DOMAIN LAYER                         │  │   │
│   │  │  Entities: Order, Customer, Product                    │  │   │
│   │  │  Value Objects: Money, Address, OrderItem              │  │   │
│   │  │  Domain Services: PricingService, DiscountPolicy       │  │   │
│   │  └────────────────────────────────────────────────────────┘  │   │
│   │                                                               │   │
│   │  ┌────────────────────────────────────────────────────────┐  │   │
│   │  │                    OUTBOUND PORTS                        │  │   │
│   │  │  OrderRepository (interface)                            │  │   │
│   │  │  PaymentGateway (interface)                             │  │   │
│   │  │  NotificationSender (interface)                         │  │   │
│   │  └────────────────────────────────────────────────────────┘  │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ Implement Outbound Ports
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     OUTBOUND ADAPTERS                                 │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  PostgreSQL      │  │  SendGrid        │  │  Stripe          │   │
│  │  OrderRepository │  │  EmailSender     │  │  PaymentGateway  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Redis Cache     │  │  Kafka Producer  │  │  S3 File Storage │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
ecommerce/
├── domain/
│   ├── __init__.py
│   ├── entities.py          # Core business entities
│   ├── value_objects.py     # Value objects (Money, Address)
│   ├── events.py            # Domain events
│   └── exceptions.py        # Business exceptions
├── application/
│   ├── __init__.py
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── inbound.py       # Use case interfaces
│   │   └── outbound.py      # Repository/Service interfaces
│   ├── services/
│   │   ├── __init__.py
│   │   ├── order_service.py # Use case implementation
│   │   └── payment_service.py
│   └── dto.py               # Data transfer objects
├── adapters/
│   ├── __init__.py
│   ├── inbound/
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py        # FastAPI/Flask routes
│   │   │   ├── serializers.py   # Request/Response models
│   │   │   └── middlewares.py
│   │   └── cli/
│   │       ├── __init__.py
│   │       └── commands.py      # Click/Argparse commands
│   └── outbound/
│       ├── __init__.py
│       ├── persistence/
│       │   ├── __init__.py
│       │   ├── models.py        # SQLAlchemy/ORM models
│       │   ├── repositories.py  # Repository implementations
│       │   └── migrations/
│       ├── email/
│       │   ├── __init__.py
│       │   └── sendgrid.py      # SendGrid adapter
│       ├── payment/
│       │   ├── __init__.py
│       │   └── stripe.py        # Stripe adapter
│       └── messaging/
│           ├── __init__.py
│           └── kafka_producer.py # Kafka adapter
├── config.py                    # Configuration
├── main.py                      # DI wiring + app entry
└── tests/
    ├── __init__.py
    ├── domain/
    │   └── test_entities.py
    ├── application/
    │   ├── test_order_service.py  # Mock adapters
    │   └── test_payment_service.py
    └── adapters/
        ├── test_repositories.py   # Real DB
        └── test_api.py           # HTTP tests
```

---

## Sơ đồ kiến trúc

```
                     HEXAGONAL ARCHITECTURE DIAGRAM
                     ==============================

                         ┌──────────────────┐
                         │  EXTERNAL        │
                         │  CLIENTS         │
                         │  (Web, Mobile,   │
                         │   API Clients)   │
                         └────────┬─────────┘
                                  │ HTTP / gRPC
                                  ▼
                 ┌──────────────────────────────────┐
                 │   INBOUND ADAPTER (API Layer)     │
                 │   FastAPI / Flask / Django        │
                 │   ┌────────────────────────────┐  │
                 │   │ Routes → Serializers →     │  │
                 │   │ → Calls Inbound Port        │  │
                 │   └────────────────────────────┘  │
                 └──────────────┬───────────────────┘
                                │ Gọi Inbound Port
                                ▼
   ╔══════════════════════════════════════════════════════════╗
   ║               APPLICATION LAYER (The Hexagon)             ║
   ║                                                          ║
   ║  ┌──────────────────────────────────────────────────────┐║
   ║  │  INBOUND PORTS (Use Case Interfaces)                 │║
   ║  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │║
   ║  │  │CreateOrder   │  │CancelOrder   │  │GetOrder   │  │║
   ║  │  │UseCase       │  │UseCase       │  │Query      │  │║
   ║  │  └──────────────┘  └──────────────┘  └───────────┘  │║
   ║  │                                                      │║
   ║  │  ┌──────────────────────────────────────────────────┐│║
   ║  │  │  USE CASE IMPLEMENTATIONS                        ││║
   ║  │  │  OrderService → implements CreateOrderUseCase   ││║
   ║  │  │                                 Uses             ││║
   ║  │  │  ┌──────────────┐  ┌──────────────┐  ┌────────┐ ││║
   ║  │  │  │ OrderRepo    │  │ PaymentGate   │  │ Notif  │ ││║
   ║  │  │  │ (Port)       │  │ (Port)        │  │ (Port) │ ││║
   ║  │  │  └──────────────┘  └──────────────┘  └────────┘ ││║
   ║  │  └──────────────────────────────────────────────────┘│║
   ║  │                                                      │║
   ║  │  ┌──────────────────────────────────────────────────┐│║
   ║  │  │  DOMAIN (cốt lõi, không phụ thuộc gì)            ││║
   ║  │  │  Order | Customer | Product | Money              ││║
   ║  │  │  OrderStatus | PaymentMethod                     ││║
   ║  │  └──────────────────────────────────────────────────┘│║
   ║  └──────────────────────────────────────────────────────┘║
   ╚══════════════════════╦═══════════════════════════════════╝
                          │ Implement Outbound Port
                          ▼
   ┌──────────────────────────────────────────────────────────┐
   │            OUTBOUND ADAPTERS (Infrastructure)            │
   │                                                          │
   │  ┌──────────────────┐  ┌──────────────────┐             │
   │  │  PostgreSQL      │  │  Stripe           │             │
   │  │  OrderRepository │  │  PaymentGateway   │             │
   │  │  (SQLAlchemy)    │  │  (Stripe SDK)     │             │
   │  └──────────────────┘  └──────────────────┘             │
   │  ┌──────────────────┐  ┌──────────────────┐             │
   │  │  SendGrid        │  │  Kafka            │             │
   │  │  EmailSender     │  │  EventPublisher   │             │
   │  └──────────────────┘  └──────────────────┘             │
   └──────────────────────────────────────────────────────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │  EXTERNAL SERVICES    │
               │  PostgreSQL | Stripe  │
               │  SendGrid | Kafka     │
               └──────────────────────┘

   DATA FLOW:
   Request → [API Adapter] → [Inbound Port] → [Use Case] → [Outbound Port] → [Outbound Adapter] → [External]
   Response ← [API Adapter] ← [Inbound Port] ← [Use Case] ← [Outbound Port] ← [Outbound Adapter] ← [External]

   DEPENDENCY DIRECTION:
   Adapters → Ports ← Use Cases → Domain
   (outside)  (contract) (logic)  (core)
```

---

## Ví dụ code hoàn chỉnh

### File: `domain/entities.py`

```python
"""Domain entities — pure business objects with no infrastructure dependencies."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional
from uuid import uuid4

from domain.value_objects import Address, Money, OrderItem
from domain.exceptions import (
    InsufficientInventoryError,
    InvalidOrderStateError,
    PaymentFailedError,
)


class OrderStatus(Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    PROCESSING = "PROCESSING"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    REFUNDED = "REFUNDED"


class PaymentMethod(Enum):
    CREDIT_CARD = "CREDIT_CARD"
    BANK_TRANSFER = "BANK_TRANSFER"
    COD = "COD"  # Cash on delivery
    MOMO = "MOMO"
    VNPAY = "VNPAY"


@dataclass
class Order:
    """Core domain entity — an order in the e-commerce system."""
    id: str = field(default_factory=lambda: str(uuid4()))
    customer_id: str = ""
    items: List[OrderItem] = field(default_factory=list)
    shipping_address: Optional[Address] = None
    payment_method: Optional[PaymentMethod] = None
    status: OrderStatus = OrderStatus.PENDING
    subtotal: Money = field(default_factory=lambda: Money(0))
    shipping_fee: Money = field(default_factory=lambda: Money(0))
    tax: Money = field(default_factory=lambda: Money(0))
    discount: Money = field(default_factory=lambda: Money(0))
    total: Money = field(default_factory=lambda: Money(0))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    paid_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    cancel_reason: str = ""
    version: int = 1

    def add_item(self, item: OrderItem) -> None:
        """Add an item to the order and recalculate totals."""
        if self.status != OrderStatus.PENDING:
            raise InvalidOrderStateError(
                f"Cannot add item to order in status {self.status.value}"
            )
        self.items.append(item)
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate order totals."""
        self.subtotal = sum(
            (item.price * item.quantity) for item in self.items
        )
        self.total = self.subtotal + self.shipping_fee + self.tax - self.discount

    def confirm(self) -> None:
        """Confirm the order (after payment)."""
        if self.status != OrderStatus.PENDING:
            raise InvalidOrderStateError(
                f"Cannot confirm order in status {self.status.value}"
            )
        if not self.items:
            raise InvalidOrderStateError("Cannot confirm empty order")
        if not self.payment_method:
            raise InvalidOrderStateError("Payment method not set")
        self.status = OrderStatus.CONFIRMED
        self.paid_at = datetime.now()
        self.updated_at = datetime.now()
        self.version += 1

    def ship(self) -> None:
        """Mark order as shipped."""
        if self.status != OrderStatus.CONFIRMED:
            raise InvalidOrderStateError(
                f"Cannot ship order in status {self.status.value}"
            )
        self.status = OrderStatus.SHIPPED
        self.updated_at = datetime.now()
        self.version += 1

    def deliver(self) -> None:
        """Mark order as delivered."""
        if self.status != OrderStatus.SHIPPED:
            raise InvalidOrderStateError(
                f"Cannot deliver order in status {self.status.value}"
            )
        self.status = OrderStatus.DELIVERED
        self.updated_at = datetime.now()
        self.version += 1

    def cancel(self, reason: str = "") -> None:
        """Cancel an order. Only PENDING or CONFIRMED orders can be cancelled."""
        if self.status not in (OrderStatus.PENDING, OrderStatus.CONFIRMED):
            raise InvalidOrderStateError(
                f"Cannot cancel order in status {self.status.value}"
            )
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.now()
        self.cancel_reason = reason
        self.updated_at = datetime.now()
        self.version += 1

    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self.items)

    @property
    def is_paid(self) -> bool:
        return self.paid_at is not None


@dataclass
class Customer:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    email: str = ""
    phone: str = ""
    default_address: Optional[Address] = None
    loyalty_points: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def add_points(self, points: int) -> None:
        self.loyalty_points += points


@dataclass
class Product:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    price: Money = field(default_factory=lambda: Money(0))
    sku: str = ""
    stock_quantity: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def reduce_stock(self, quantity: int) -> None:
        if quantity > self.stock_quantity:
            raise InsufficientInventoryError(
                product_id=self.id,
                requested=quantity,
                available=self.stock_quantity,
            )
        self.stock_quantity -= quantity
```

### File: `domain/value_objects.py`

```python
"""Value objects — immutable, no identity, compared by value."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional


@dataclass(frozen=True)
class Money:
    """Immutable monetary value."""
    amount: float = 0.0
    currency: str = "VND"

    def __post_init__(self) -> None:
        # Round to 2 decimal places
        rounded = Decimal(str(self.amount)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        object.__setattr__(self, "amount", float(rounded))

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {self.currency} and {other.currency}")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, multiplier: float) -> "Money":
        return Money(self.amount * multiplier, self.currency)

    def __rmul__(self, multiplier: float) -> "Money":
        return self.__mul__(multiplier)

    def __neg__(self) -> "Money":
        return Money(-self.amount, self.currency)

    def __repr__(self) -> str:
        return f"{self.amount:,.0f} {self.currency}"


@dataclass(frozen=True)
class Address:
    street: str
    ward: str
    district: str
    city: str
    country: str = "Vietnam"
    zip_code: str = ""

    def full_address(self) -> str:
        return f"{self.street}, {self.ward}, {self.district}, {self.city}"


@dataclass(frozen=True)
class OrderItem:
    product_id: str
    product_name: str
    price: Money
    quantity: int = 1
    sku: str = ""

    @property
    def total(self) -> Money:
        return self.price * self.quantity
```

### File: `domain/exceptions.py`

```python
"""Domain-specific exceptions with semantic meaning."""

from typing import Optional


class DomainError(Exception):
    """Base exception for all domain errors."""
    def __init__(self, message: str, code: str = "DOMAIN_ERROR") -> None:
        self.code = code
        super().__init__(message)


class InvalidOrderStateError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="INVALID_ORDER_STATE")


class InsufficientInventoryError(DomainError):
    def __init__(self, product_id: str, requested: int, available: int) -> None:
        self.product_id = product_id
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient stock for product {product_id}: "
            f"requested {requested}, available {available}",
            code="INSUFFICIENT_INVENTORY",
        )


class PaymentFailedError(DomainError):
    def __init__(self, order_id: str, reason: str, gateway: str = "unknown") -> None:
        self.order_id = order_id
        self.gateway = gateway
        super().__init__(
            f"Payment failed for order {order_id} via {gateway}: {reason}",
            code="PAYMENT_FAILED",
        )


class OrderNotFoundError(DomainError):
    def __init__(self, order_id: str) -> None:
        super().__init__(f"Order {order_id} not found", code="ORDER_NOT_FOUND")


class ProductNotFoundError(DomainError):
    def __init__(self, product_id: str) -> None:
        super().__init__(f"Product {product_id} not found", code="PRODUCT_NOT_FOUND")


class CustomerNotFoundError(DomainError):
    def __init__(self, customer_id: str) -> None:
        super().__init__(f"Customer {customer_id} not found", code="CUSTOMER_NOT_FOUND")
```

### File: `domain/events.py`

```python
"""Domain events — recorded side effects of domain operations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from domain.value_objects import Money


@dataclass
class DomainEvent:
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1


@dataclass
class OrderCreated(DomainEvent):
    order_id: str
    customer_id: str
    total: Money
    item_count: int


@dataclass
class OrderConfirmed(DomainEvent):
    order_id: str
    paid_at: datetime


@dataclass
class OrderShipped(DomainEvent):
    order_id: str
    tracking_number: str


@dataclass
class OrderDelivered(DomainEvent):
    order_id: str
    delivered_at: datetime


@dataclass
class OrderCancelled(DomainEvent):
    order_id: str
    reason: str
    refund_amount: Money
```

### File: `application/ports/inbound.py`

```python
"""Inbound ports (driving ports) — use case interfaces.
These define the API of the application core (inside)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from domain.entities import Order, OrderStatus
from domain.value_objects import Address, Money, OrderItem, PaymentMethod
from application.dto import CreateOrderInput, OrderDTO, PaymentInput


class CreateOrderUseCase(ABC):
    """Port: create a new order."""

    @abstractmethod
    def create_order(self, input_data: CreateOrderInput) -> OrderDTO:
        ...


class CancelOrderUseCase(ABC):
    """Port: cancel an existing order."""

    @abstractmethod
    def cancel_order(self, order_id: str, reason: str) -> OrderDTO:
        ...


class GetOrderQuery(ABC):
    """Port: query an order by ID."""

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[OrderDTO]:
        ...


class ListOrdersQuery(ABC):
    """Port: list orders by customer."""

    @abstractmethod
    def list_orders_by_customer(
        self, customer_id: str, skip: int = 0, limit: int = 20,
    ) -> List[OrderDTO]:
        ...


class ProcessPaymentUseCase(ABC):
    """Port: process payment for an order."""

    @abstractmethod
    def process_payment(self, input_data: PaymentInput) -> OrderDTO:
        ...
```

### File: `application/ports/outbound.py`

```python
"""Outbound ports (driven ports) — interfaces for external services.
These are defined BY the inside, implemented BY the outside."""

from abc import ABC, abstractmethod
from typing import List, Optional

from domain.entities import Customer, Order, Product
from domain.value_objects import Money


class OrderRepository(ABC):
    """Port for order persistence."""

    @abstractmethod
    def save(self, order: Order) -> Order:
        ...

    @abstractmethod
    def find_by_id(self, order_id: str) -> Optional[Order]:
        ...

    @abstractmethod
    def find_by_customer(
        self, customer_id: str, skip: int = 0, limit: int = 20,
    ) -> List[Order]:
        ...

    @abstractmethod
    def delete(self, order_id: str) -> None:
        ...


class ProductRepository(ABC):
    """Port for product persistence."""

    @abstractmethod
    def find_by_id(self, product_id: str) -> Optional[Product]:
        ...

    @abstractmethod
    def update_stock(self, product_id: str, quantity: int) -> None:
        ...


class CustomerRepository(ABC):
    """Port for customer persistence."""

    @abstractmethod
    def find_by_id(self, customer_id: str) -> Optional[Customer]:
        ...

    @abstractmethod
    def save(self, customer: Customer) -> Customer:
        ...


class PaymentGateway(ABC):
    """Port for payment processing."""

    @abstractmethod
    def charge(self, amount: Money, payment_method: str, metadata: dict) -> dict:
        ...

    @abstractmethod
    def refund(self, transaction_id: str, amount: Money) -> dict:
        ...


class NotificationSender(ABC):
    """Port for sending notifications."""

    @abstractmethod
    def send_order_confirmation(self, order_id: str, email: str) -> None:
        ...

    @abstractmethod
    def send_shipping_update(self, order_id: str, email: str, tracking: str) -> None:
        ...


class InventoryService(ABC):
    """Port for inventory management."""

    @abstractmethod
    def reserve_stock(self, order_id: str, items: list) -> bool:
        ...

    @abstractmethod
    def release_stock(self, order_id: str, items: list) -> None:
        ...


class EventPublisher(ABC):
    """Port for publishing domain events."""

    @abstractmethod
    def publish(self, event, topic: str) -> None:
        ...
```

### File: `application/dto.py`

```python
"""Data Transfer Objects — cross-boundary data structures."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from domain.value_objects import Address, Money, OrderItem, PaymentMethod


@dataclass
class CreateOrderInput:
    """Input DTO for order creation."""
    customer_id: str
    items: List[dict]  # [{"product_id": "...", "quantity": 2}, ...]
    shipping_address: Optional[dict] = None
    payment_method: Optional[str] = None


@dataclass
class PaymentInput:
    """Input DTO for payment processing."""
    order_id: str
    payment_method: str
    card_token: Optional[str] = None
    bank_code: Optional[str] = None


@dataclass
class OrderItemDTO:
    product_id: str
    product_name: str
    price: float
    quantity: int
    total: float

    @classmethod
    def from_domain(cls, item: OrderItem) -> "OrderItemDTO":
        return cls(
            product_id=item.product_id,
            product_name=item.product_name,
            price=item.price.amount,
            quantity=item.quantity,
            total=item.total.amount,
        )


@dataclass
class OrderDTO:
    """Output DTO — represents an order to the outside world."""
    id: str
    customer_id: str
    items: List[OrderItemDTO]
    status: str
    subtotal: float
    shipping_fee: float
    tax: float
    discount: float
    total: float
    created_at: str
    paid_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    item_count: int = 0

    @classmethod
    def from_domain(cls, order: Order) -> "OrderDTO":
        return cls(
            id=order.id,
            customer_id=order.customer_id,
            items=[OrderItemDTO.from_domain(item) for item in order.items],
            status=order.status.value,
            subtotal=order.subtotal.amount,
            shipping_fee=order.shipping_fee.amount,
            tax=order.tax.amount,
            discount=order.discount.amount,
            total=order.total.amount,
            item_count=order.item_count,
            created_at=order.created_at.isoformat(),
            paid_at=order.paid_at.isoformat() if order.paid_at else None,
            cancelled_at=order.cancelled_at.isoformat() if order.cancelled_at else None,
        )
```

### File: `application/services/order_service.py`

```python
"""Use case implementation — pure business logic with no infrastructure."""

import logging
from typing import List, Optional

from domain.entities import Order, OrderStatus, PaymentMethod
from domain.events import OrderCancelled, OrderConfirmed, OrderCreated
from domain.exceptions import (
    CustomerNotFoundError,
    InsufficientInventoryError,
    InvalidOrderStateError,
    OrderNotFoundError,
    PaymentFailedError,
    ProductNotFoundError,
)
from domain.value_objects import Address, Money, OrderItem
from application.dto import CreateOrderInput, OrderDTO, PaymentInput
from application.ports.inbound import (
    CancelOrderUseCase,
    CreateOrderUseCase,
    GetOrderQuery,
    ListOrdersQuery,
    ProcessPaymentUseCase,
)
from application.ports.outbound import (
    CustomerRepository,
    EventPublisher,
    InventoryService,
    NotificationSender,
    OrderRepository,
    PaymentGateway,
    ProductRepository,
)

logger = logging.getLogger(__name__)


class OrderService(CreateOrderUseCase, CancelOrderUseCase, GetOrderQuery, ListOrdersQuery):
    """Implements order-related use cases.
    Depends ONLY on abstractions (ports), not on concrete implementations."""

    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo: ProductRepository,
        customer_repo: CustomerRepository,
        notification_sender: NotificationSender,
        event_publisher: EventPublisher,
    ) -> None:
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._customer_repo = customer_repo
        self._notification_sender = notification_sender
        self._event_publisher = event_publisher

    def create_order(self, input_data: CreateOrderInput) -> OrderDTO:
        """Create a new order from input data."""

        # 1. Validate customer exists
        customer = self._customer_repo.find_by_id(input_data.customer_id)
        if not customer:
            raise CustomerNotFoundError(input_data.customer_id)

        # 2. Build order items from product data
        items = []
        for item_data in input_data.items:
            product = self._product_repo.find_by_id(item_data["product_id"])
            if not product:
                raise ProductNotFoundError(item_data["product_id"])
            if not product.is_active:
                raise ProductNotFoundError(f"Product {product.id} is not active")
            if product.stock_quantity < item_data["quantity"]:
                raise InsufficientInventoryError(
                    product_id=product.id,
                    requested=item_data["quantity"],
                    available=product.stock_quantity,
                )

            item = OrderItem(
                product_id=product.id,
                product_name=product.name,
                price=product.price,
                quantity=item_data["quantity"],
                sku=product.sku,
            )
            items.append(item)

            # Reduce stock
            product.reduce_stock(item.quantity)
            self._product_repo.update_stock(product.id, product.stock_quantity)

        # 3. Create domain entity
        order = Order(customer_id=input_data.customer_id, items=items)
        order._recalculate()

        # 4. Set payment method if provided
        if input_data.payment_method:
            try:
                order.payment_method = PaymentMethod(input_data.payment_method.upper())
            except ValueError:
                raise InvalidOrderStateError(
                    f"Invalid payment method: {input_data.payment_method}"
                )

        # 5. Set shipping address
        if input_data.shipping_address:
            addr_data = input_data.shipping_address
            order.shipping_address = Address(
                street=addr_data["street"],
                ward=addr_data["ward"],
                district=addr_data["district"],
                city=addr_data["city"],
            )

        # 6. Persist
        saved = self._order_repo.save(order)

        # 7. Send notification
        self._notification_sender.send_order_confirmation(
            order_id=saved.id, email=customer.email,
        )

        # 8. Publish event
        self._event_publisher.publish(
            OrderCreated(
                order_id=saved.id,
                customer_id=customer.id,
                total=saved.total,
                item_count=saved.item_count,
            ),
            topic="order.created",
        )

        logger.info(f"Order {saved.id} created for customer {customer.id}")
        return OrderDTO.from_domain(saved)

    def cancel_order(self, order_id: str, reason: str) -> OrderDTO:
        order = self._order_repo.find_by_id(order_id)
        if not order:
            raise OrderNotFoundError(order_id)

        order.cancel(reason)
        saved = self._order_repo.save(order)

        self._event_publisher.publish(
            OrderCancelled(
                order_id=order_id,
                reason=reason,
                refund_amount=order.total,
            ),
            topic="order.cancelled",
        )

        logger.info(f"Order {order_id} cancelled: {reason}")
        return OrderDTO.from_domain(saved)

    def get_order(self, order_id: str) -> Optional[OrderDTO]:
        order = self._order_repo.find_by_id(order_id)
        return OrderDTO.from_domain(order) if order else None

    def list_orders_by_customer(
        self, customer_id: str, skip: int = 0, limit: int = 20,
    ) -> List[OrderDTO]:
        orders = self._order_repo.find_by_customer(customer_id, skip, limit)
        return [OrderDTO.from_domain(o) for o in orders]


class PaymentService(ProcessPaymentUseCase):
    """Processes payments for orders."""

    def __init__(
        self,
        order_repo: OrderRepository,
        payment_gateway: PaymentGateway,
        notification_sender: NotificationSender,
        event_publisher: EventPublisher,
    ) -> None:
        self._order_repo = order_repo
        self._payment_gateway = payment_gateway
        self._notification_sender = notification_sender
        self._event_publisher = event_publisher

    def process_payment(self, input_data: PaymentInput) -> OrderDTO:
        order = self._order_repo.find_by_id(input_data.order_id)
        if not order:
            raise OrderNotFoundError(input_data.order_id)

        if order.status != OrderStatus.PENDING:
            raise InvalidOrderStateError(
                f"Cannot pay for order in status {order.status.value}"
            )

        # Call payment gateway
        try:
            result = self._payment_gateway.charge(
                amount=order.total,
                payment_method=input_data.payment_method,
                metadata={"order_id": order.id, "customer_id": order.customer_id},
            )
        except Exception as e:
            raise PaymentFailedError(
                order_id=order.id,
                reason=str(e),
                gateway="stripe",
            )

        if result.get("status") != "succeeded":
            raise PaymentFailedError(
                order_id=order.id,
                reason=result.get("error", "Payment declined"),
                gateway="stripe",
            )

        # Confirm order
        order.payment_method = PaymentMethod(input_data.payment_method.upper())
        order.confirm()
        saved = self._order_repo.save(order)

        # Notify
        customer = None  # Would fetch from repo
        self._notification_sender.send_order_confirmation(
            order_id=saved.id, email="customer@example.com",
        )

        # Publish event
        self._event_publisher.publish(
            OrderConfirmed(order_id=saved.id, paid_at=saved.paid_at),
            topic="order.confirmed",
        )

        return OrderDTO.from_domain(saved)
```

### File: `adapters/outbound/persistence/repositories.py`

```python
"""Outbound adapter: OrderRepository implementation using PostgreSQL + SQLAlchemy."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from domain.entities import Order, OrderItem, OrderStatus
from domain.value_objects import Address, Money, OrderItem as OrderItemVO
from application.ports.outbound import OrderRepository


class PostgresOrderRepository(OrderRepository):
    """Adapter implementing OrderRepository port with PostgreSQL."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, order: Order) -> Order:
        """Upsert an order into the database."""
        data = self._to_db(order)
        existing = self.find_by_id(order.id)

        if existing:
            # Update
            stmt = text("""
                UPDATE orders SET
                    status = :status, subtotal = :subtotal,
                    shipping_fee = :shipping_fee, tax = :tax,
                    discount = :discount, total = :total,
                    updated_at = :updated_at,
                    paid_at = :paid_at,
                    cancelled_at = :cancelled_at,
                    cancel_reason = :cancel_reason,
                    version = :version
                WHERE id = :id AND version = :old_version
            """)
            result = self._session.execute(
                stmt,
                {
                    "id": order.id,
                    "status": order.status.value,
                    "subtotal": order.subtotal.amount,
                    "shipping_fee": order.shipping_fee.amount,
                    "tax": order.tax.amount,
                    "discount": order.discount.amount,
                    "total": order.total.amount,
                    "updated_at": order.updated_at,
                    "paid_at": order.paid_at,
                    "cancelled_at": order.cancelled_at,
                    "cancel_reason": order.cancel_reason,
                    "version": order.version,
                    "old_version": order.version - 1,
                },
            )
            if result.rowcount == 0:
                raise ValueError(
                    f"Optimistic lock failed for order {order.id} "
                    f"(expected version {order.version - 1})"
                )
        else:
            stmt = text("""
                INSERT INTO orders (
                    id, customer_id, status,
                    subtotal, shipping_fee, tax, discount, total,
                    created_at, updated_at, paid_at, cancelled_at,
                    cancel_reason, version
                ) VALUES (
                    :id, :customer_id, :status,
                    :subtotal, :shipping_fee, :tax, :discount, :total,
                    :created_at, :updated_at, :paid_at, :cancelled_at,
                    :cancel_reason, :version
                )
            """)
            self._session.execute(stmt, data)

        # Save items (delete old, insert new)
        self._session.execute(
            text("DELETE FROM order_items WHERE order_id = :order_id"),
            {"order_id": order.id},
        )
        for i, item in enumerate(order.items):
            self._session.execute(
                text("""
                    INSERT INTO order_items (
                        order_id, product_id, product_name,
                        price, quantity, sku, sort_order
                    ) VALUES (
                        :order_id, :product_id, :product_name,
                        :price, :quantity, :sku, :sort_order
                    )
                """),
                {
                    "order_id": order.id,
                    "product_id": item.product_id,
                    "product_name": item.product_name,
                    "price": item.price.amount,
                    "quantity": item.quantity,
                    "sku": item.sku,
                    "sort_order": i,
                },
            )

        self._session.commit()
        return order

    def find_by_id(self, order_id: str) -> Optional[Order]:
        result = self._session.execute(
            text("SELECT * FROM orders WHERE id = :id"),
            {"id": order_id},
        )
        row = result.mappings().one_or_none()
        if not row:
            return None
        return self._from_db(row)

    def find_by_customer(
        self, customer_id: str, skip: int = 0, limit: int = 20,
    ) -> List[Order]:
        result = self._session.execute(
            text("""
                SELECT * FROM orders
                WHERE customer_id = :customer_id
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :skip
            """),
            {"customer_id": customer_id, "limit": limit, "skip": skip},
        )
        return [self._from_db(row) for row in result.mappings().all()]

    def delete(self, order_id: str) -> None:
        self._session.execute(
            text("DELETE FROM order_items WHERE order_id = :order_id"),
            {"order_id": order_id},
        )
        self._session.execute(
            text("DELETE FROM orders WHERE id = :id"),
            {"id": order_id},
        )
        self._session.commit()

    def _to_db(self, order: Order) -> Dict[str, Any]:
        return {
            "id": order.id,
            "customer_id": order.customer_id,
            "status": order.status.value,
            "subtotal": order.subtotal.amount,
            "shipping_fee": order.shipping_fee.amount,
            "tax": order.tax.amount,
            "discount": order.discount.amount,
            "total": order.total.amount,
            "created_at": order.created_at,
            "updated_at": order.updated_at,
            "paid_at": order.paid_at,
            "cancelled_at": order.cancelled_at,
            "cancel_reason": order.cancel_reason,
            "version": order.version,
        }

    def _from_db(self, row) -> Order:
        order = Order(
            id=row["id"],
            customer_id=row["customer_id"],
            status=OrderStatus(row["status"]),
            subtotal=Money(row["subtotal"]),
            shipping_fee=Money(row["shipping_fee"]),
            tax=Money(row["tax"]),
            discount=Money(row["discount"]),
            total=Money(row["total"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            paid_at=row.get("paid_at"),
            cancelled_at=row.get("cancelled_at"),
            cancel_reason=row.get("cancel_reason", ""),
            version=row["version"],
        )

        # Load items
        items_result = self._session.execute(
            text("""
                SELECT * FROM order_items
                WHERE order_id = :order_id
                ORDER BY sort_order
            """),
            {"order_id": order.id},
        )
        for item_row in items_result.mappings().all():
            item = OrderItemVO(
                product_id=item_row["product_id"],
                product_name=item_row["product_name"],
                price=Money(item_row["price"]),
                quantity=item_row["quantity"],
                sku=item_row.get("sku", ""),
            )
            order.items.append(item)

        return order
```

### File: `adapters/outbound/payment/stripe_adapter.py`

```python
"""Outbound adapter: Stripe payment gateway."""

import logging
from typing import Any, Dict

from domain.value_objects import Money
from application.ports.outbound import PaymentGateway

logger = logging.getLogger(__name__)


class StripePaymentGateway(PaymentGateway):
    """Adapter implementing PaymentGateway port using Stripe SDK."""

    def __init__(self, api_key: str, webhook_secret: str = "") -> None:
        self._api_key = api_key
        self._webhook_secret = webhook_secret
        self._client = None

    def _initialize(self) -> None:
        if self._client is None:
            try:
                import stripe
                stripe.api_key = self._api_key
                self._client = stripe
                logger.info("Stripe client initialized")
            except ImportError:
                logger.warning("Stripe SDK not available. Using stub.")
                self._client = "stub"

    def charge(self, amount: Money, payment_method: str, metadata: dict) -> dict:
        self._initialize()

        if self._client == "stub":
            logger.info(
                f"[STUB] Charging {amount} via {payment_method} "
                f"(metadata: {metadata})"
            )
            return {
                "status": "succeeded",
                "transaction_id": f"txn_stub_{metadata.get('order_id', 'unknown')}",
                "amount": amount.amount,
                "currency": amount.currency,
            }

        try:
            # Real Stripe API call
            intent = self._client.PaymentIntent.create(
                amount=int(amount.amount * 100),  # Convert to cents
                currency=amount.currency.lower(),
                payment_method=payment_method,
                metadata=metadata,
                confirm=True,
            )
            return {
                "status": "succeeded" if intent.status == "succeeded" else "failed",
                "transaction_id": intent.id,
                "amount": intent.amount / 100,
                "currency": intent.currency,
            }
        except Exception as e:
            logger.error(f"Stripe charge failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "transaction_id": "",
            }

    def refund(self, transaction_id: str, amount: Money) -> dict:
        self._initialize()

        if self._client == "stub":
            logger.info(f"[STUB] Refunding {amount} for transaction {transaction_id}")
            return {
                "status": "succeeded",
                "refund_id": f"ref_{transaction_id}",
                "amount": amount.amount,
            }

        try:
            refund = self._client.Refund.create(
                payment_intent=transaction_id,
                amount=int(amount.amount * 100),
            )
            return {
                "status": "succeeded" if refund.status == "succeeded" else "failed",
                "refund_id": refund.id,
                "amount": refund.amount / 100,
            }
        except Exception as e:
            logger.error(f"Stripe refund failed: {e}")
            return {"status": "failed", "error": str(e)}
```

### File: `adapters/inbound/api/serializers.py`

```python
"""Pydantic serializers for HTTP API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateOrderItemRequest(BaseModel):
    product_id: str
    quantity: int = Field(ge=1, le=1000)


class CreateOrderRequest(BaseModel):
    customer_id: str
    items: List[CreateOrderItemRequest] = Field(min_length=1)
    shipping_address: Optional[Dict[str, str]] = None
    payment_method: Optional[str] = None


class PaymentRequest(BaseModel):
    order_id: str
    payment_method: str
    card_token: Optional[str] = None


class OrderItemResponse(BaseModel):
    product_id: str
    product_name: str
    price: float
    quantity: int
    total: float


class OrderResponse(BaseModel):
    id: str
    customer_id: str
    items: List[OrderItemResponse]
    status: str
    subtotal: float
    shipping_fee: float
    tax: float
    discount: float
    total: float
    item_count: int
    created_at: str
    paid_at: Optional[str] = None
    cancelled_at: Optional[str] = None

    @classmethod
    def from_dto(cls, dto) -> "OrderResponse":
        return cls(
            id=dto.id,
            customer_id=dto.customer_id,
            items=dto.items,
            status=dto.status,
            subtotal=dto.subtotal,
            shipping_fee=dto.shipping_fee,
            tax=dto.tax,
            discount=dto.discount,
            total=dto.total,
            item_count=dto.item_count,
            created_at=dto.created_at,
            paid_at=dto.paid_at,
            cancelled_at=dto.cancelled_at,
        )
```

### File: `adapters/inbound/api/routes.py`

```python
"""FastAPI routes — inbound adapter for HTTP requests."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from application.dto import CreateOrderInput, PaymentInput
from application.ports.inbound import (
    CancelOrderUseCase,
    CreateOrderUseCase,
    GetOrderQuery,
    ListOrdersQuery,
    ProcessPaymentUseCase,
)
from domain.exceptions import DomainError
from adapters.inbound.api.serializers import (
    CreateOrderRequest,
    OrderResponse,
    PaymentRequest,
)

logger = logging.getLogger(__name__)


def create_order_router(
    create_order: CreateOrderUseCase,
    cancel_order: CancelOrderUseCase,
    get_order: GetOrderQuery,
    list_orders: ListOrdersQuery,
    process_payment: ProcessPaymentUseCase,
) -> APIRouter:
    """Factory: creates router with injected use cases."""
    router = APIRouter(prefix="/api/v1/orders", tags=["orders"])

    @router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
    async def create_order_endpoint(request: CreateOrderRequest):
        try:
            input_data = CreateOrderInput(
                customer_id=request.customer_id,
                items=[item.model_dump() for item in request.items],
                shipping_address=request.shipping_address,
                payment_method=request.payment_method,
            )
            result = create_order.create_order(input_data)
            return OrderResponse.from_dto(result)
        except DomainError as e:
            raise HTTPException(status_code=422, detail={"code": e.code, "message": str(e)})

    @router.get("/{order_id}", response_model=OrderResponse)
    async def get_order_endpoint(order_id: str):
        result = get_order.get_order(order_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        return OrderResponse.from_dto(result)

    @router.get("", response_model=List[OrderResponse])
    async def list_orders(
        customer_id: str,
        skip: int = 0,
        limit: int = 20,
    ):
        results = list_orders.list_orders_by_customer(customer_id, skip, limit)
        return [OrderResponse.from_dto(r) for r in results]

    @router.post("/{order_id}/cancel", response_model=OrderResponse)
    async def cancel_order_endpoint(order_id: str, reason: str = ""):
        try:
            result = cancel_order.cancel_order(order_id, reason)
            return OrderResponse.from_dto(result)
        except DomainError as e:
            raise HTTPException(
                status=400,
                detail={"code": e.code, "message": str(e)},
            )

    @router.post("/pay", response_model=OrderResponse)
    async def pay_order_endpoint(request: PaymentRequest):
        try:
            input_data = PaymentInput(
                order_id=request.order_id,
                payment_method=request.payment_method,
                card_token=request.card_token,
            )
            result = process_payment.process_payment(input_data)
            return OrderResponse.from_dto(result)
        except DomainError as e:
            raise HTTPException(status_code=402, detail={"code": e.code, "message": str(e)})

    return router
```

### File: `main.py`

```python
"""Application entry point — Dependency Injection wiring."""

import logging
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from adapters.inbound.api.routes import create_order_router
from adapters.outbound.payment.stripe_adapter import StripePaymentGateway
from adapters.outbound.persistence.repositories import PostgresOrderRepository
from application.ports.outbound import (
    CustomerRepository,
    EventPublisher,
    NotificationSender,
    PaymentGateway,
    ProductRepository,
)
from application.services.order_service import OrderService, PaymentService
from config import AppConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---- Stub adapters for demo (would be real in production) ----

class InMemoryProductRepository(ProductRepository):
    def __init__(self):
        self._products = {}

    def find_by_id(self, product_id):
        return self._products.get(product_id)

    def update_stock(self, product_id, quantity):
        if product_id in self._products:
            self._products[product_id].stock_quantity = quantity

    def add(self, product):
        self._products[product.id] = product


class InMemoryCustomerRepository(CustomerRepository):
    def __init__(self):
        self._customers = {}

    def find_by_id(self, customer_id):
        return self._customers.get(customer_id)

    def save(self, customer):
        self._customers[customer.id] = customer
        return customer


class StubNotificationSender(NotificationSender):
    def send_order_confirmation(self, order_id, email):
        logger.info(f"  📧 [EMAIL] Order {order_id} confirmed → sent to {email}")

    def send_shipping_update(self, order_id, email, tracking):
        logger.info(f"  📧 [EMAIL] Order {order_id} shipped (tracking: {tracking}) → {email}")


class StubEventPublisher(EventPublisher):
    def publish(self, event, topic):
        logger.info(f"  📢 [EVENT] {type(event).__name__} → {topic}")


# ---- Application Factory ----

def create_app(config: AppConfig) -> FastAPI:
    """Application factory with all dependency injection."""

    # 1. Infrastructure adapters
    product_repo = InMemoryProductRepository()
    customer_repo = InMemoryCustomerRepository()
    notification_sender = StubNotificationSender()
    event_publisher = StubEventPublisher()
    payment_gateway = StripePaymentGateway(api_key=config.stripe_api_key)

    # Use SQLite for demo
    engine = create_engine("sqlite:///ecommerce.db", echo=False)
    session_factory = sessionmaker(bind=engine)

    # Create tables
    with engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    subtotal REAL,
                    shipping_fee REAL,
                    tax REAL,
                    discount REAL,
                    total REAL,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    paid_at TIMESTAMP,
                    cancelled_at TIMESTAMP,
                    cancel_reason TEXT,
                    version INTEGER
                )
            """)
        )
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS order_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    product_name TEXT,
                    price REAL,
                    quantity INTEGER,
                    sku TEXT,
                    sort_order INTEGER
                )
            """)
        )

    session = session_factory()

    # 2. Application services (use cases)
    order_repo = PostgresOrderRepository(session)
    order_service = OrderService(
        order_repo=order_repo,
        product_repo=product_repo,
        customer_repo=customer_repo,
        notification_sender=notification_sender,
        event_publisher=event_publisher,
    )
    payment_service = PaymentService(
        order_repo=order_repo,
        payment_gateway=payment_gateway,
        notification_sender=notification_sender,
        event_publisher=event_publisher,
    )

    # 3. FastAPI app
    app = FastAPI(title="E-Commerce Hexagonal Architecture Demo")
    router = create_order_router(
        create_order=order_service,
        cancel_order=order_service,
        get_order=order_service,
        list_orders=order_service,
        process_payment=payment_service,
    )
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"status": "healthy", "architecture": "hexagonal"}

    return app, product_repo, customer_repo


# ---- Demo ----

def seed_data(product_repo, customer_repo):
    """Seed demo data."""
    from domain.entities import Customer, Product
    from domain.value_objects import Money

    customer = Customer(
        id="CUST-001",
        name="Nguyen Van A",
        email="nguyenvana@example.com",
        phone="0901234567",
    )
    customer_repo.save(customer)
    logger.info(f"✅ Created customer: {customer.name} ({customer.id})")

    products = [
        Product(
            id="PROD-001", name="iPhone 15 Pro Max", price=Money(34_990_000),
            sku="IP15PM-256", stock_quantity=50,
        ),
        Product(
            id="PROD-002", name="Samsung Galaxy S24 Ultra", price=Money(29_990_000),
            sku="S24U-256", stock_quantity=30,
        ),
        Product(
            id="PROD-003", name="AirPods Pro 2", price=Money(6_990_000),
            sku="APP2-USB", stock_quantity=100,
        ),
    ]
    for p in products:
        product_repo.add(p)
        logger.info(f"✅ Created product: {p.name} ({p.price})")

    return customer


def run_demo(app, order_service: OrderService, payment_service: PaymentService):
    """Run the demo order flow."""
    import uvicorn

    logger.info("\n" + "=" * 70)
    logger.info("HEXAGONAL ARCHITECTURE DEMO — E-Commerce Order Flow")
    logger.info("=" * 70)

    # 1. Create order
    logger.info("\n📌 USE CASE 1: Tạo đơn hàng mới\n")
    input_data = CreateOrderInput(
        customer_id="CUST-001",
        items=[
            {"product_id": "PROD-001", "quantity": 1},
            {"product_id": "PROD-003", "quantity": 2},
        ],
        payment_method="MOMO",
        shipping_address={
            "street": "123 Nguyễn Huệ",
            "ward": "Bến Nghé",
            "district": "Quận 1",
            "city": "Hồ Chí Minh",
        },
    )

    try:
        order_dto = order_service.create_order(input_data)
        logger.info(f"✅ Đơn hàng tạo thành công!")
        logger.info(f"   Mã đơn: {order_dto.id[:8]}")
        logger.info(f"   Số lượng: {order_dto.item_count} sản phẩm")
        logger.info(f"   Tổng tiền: {order_dto.total:,.0f} VND")
        logger.info(f"   Trạng thái: {order_dto.status}")

        for item in order_dto.items:
            logger.info(f"   - {item.product_name} x{item.quantity}: {item.total:,.0f} VND")

    except DomainError as e:
        logger.error(f"❌ Lỗi: {e}")

    # 2. Process payment
    logger.info("\n📌 USE CASE 2: Xử lý thanh toán\n")
    payment_input = PaymentInput(
        order_id=order_dto.id,
        payment_method="MOMO",
    )

    try:
        paid_dto = payment_service.process_payment(payment_input)
        logger.info(f"✅ Thanh toán thành công!")
        logger.info(f"   Mã đơn: {paid_dto.id[:8]}")
        logger.info(f"   Trạng thái: {paid_dto.status}")
        logger.info(f"   Đã thanh toán: {paid_dto.paid_at}")

    except DomainError as e:
        logger.error(f"❌ Lỗi thanh toán: {e}")

    # 3. Get order details
    logger.info("\n📌 USE CASE 3: Truy vấn đơn hàng\n")
    query_dto = order_service.get_order(order_dto.id)
    if query_dto:
        logger.info(f"✅ Thông tin đơn hàng {query_dto.id[:8]}:")
        logger.info(f"   Trạng thái: {query_dto.status}")
        logger.info(f"   Số item: {query_dto.item_count}")
        logger.info(f"   Tổng: {query_dto.total:,.0f} VND")
        logger.info(f"   Ngày tạo: {query_dto.created_at}")

    # 4. List orders
    logger.info("\n📌 USE CASE 4: Danh sách đơn hàng của khách hàng\n")
    orders = order_service.list_orders_by_customer("CUST-001")
    logger.info(f"   Khách hàng CUST-001 có {len(orders)} đơn hàng:")

    # 5. Cancel order
    logger.info("\n📌 USE CASE 5: Hủy đơn hàng\n")
    cancelled = order_service.cancel_order(
        order_dto.id,
        reason="Khách hàng yêu cầu hủy",
    )
    logger.info(f"✅ Đơn hàng đã hủy: {cancelled.id[:8]}")
    logger.info(f"   Lý do: {cancelled.cancel_reason}")
    logger.info(f"   Trạng thái: {cancelled.status}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Demo hoàn tất! Kiến trúc Hexagonal hoạt động chính xác.")
    logger.info("=" * 70)


if __name__ == "__main__":
    from config import AppConfig
    from sqlalchemy import text

    config = AppConfig()
    app, product_repo, customer_repo = create_app(config)
    customer = seed_data(product_repo, customer_repo)

    from application.services.order_service import OrderService, PaymentService
    order_service = app.state.get("order_service")
    payment_service = app.state.get("payment_service")

    # HACK: get services from app state or recreate
    engine = create_engine("sqlite:///ecommerce.db")
    session = sessionmaker(bind=engine)()
    order_repo = PostgresOrderRepository(session)
    from adapters.outbound.payment.stripe_adapter import StripePaymentGateway
    order_service = OrderService(
        order_repo=order_repo,
        product_repo=product_repo,
        customer_repo=customer_repo,
        notification_sender=StubNotificationSender(),
        event_publisher=StubEventPublisher(),
    )
    payment_service = PaymentService(
        order_repo=order_repo,
        payment_gateway=StripePaymentGateway(api_key="sk_test_demo"),
        notification_sender=StubNotificationSender(),
        event_publisher=StubEventPublisher(),
    )

    run_demo(app, order_service, payment_service)
```

### File: `config.py`

```python
"""Application configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    environment: str = "development"
    debug: bool = True

    # Database
    database_url: str = "sqlite:///ecommerce.db"

    # Payment
    stripe_api_key: str = "sk_test_demo"
    stripe_webhook_secret: str = ""

    # Email
    sendgrid_api_key: str = ""
    email_from: str = "shop@example.com"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
```

---

## Kiểm thử

### File: `tests/domain/test_entities.py`

```python
"""Unit tests for domain entities — no infrastructure needed."""

import pytest
from datetime import datetime

from domain.entities import Order, OrderStatus, PaymentMethod
from domain.exceptions import InvalidOrderStateError
from domain.value_objects import Address, Money, OrderItem


class TestOrder:

    def test_create_empty_order(self):
        order = Order(customer_id="CUST-001")
        assert order.status == OrderStatus.PENDING
        assert order.total.amount == 0.0
        assert order.item_count == 0

    def test_add_item_updates_total(self):
        order = Order(customer_id="CUST-001")
        item = OrderItem(
            product_id="P1", product_name="iPhone",
            price=Money(10_000_000), quantity=2,
        )
        order.add_item(item)
        assert order.item_count == 2
        assert order.subtotal.amount == 20_000_000
        assert order.total.amount == 20_000_000

    def test_add_multiple_items(self):
        order = Order(customer_id="CUST-001")
        items = [
            OrderItem("P1", "iPhone", Money(10_000_000), 1),
            OrderItem("P2", "iPad", Money(5_000_000), 2),
            OrderItem("P3", "AirPods", Money(2_000_000), 3),
        ]
        for item in items:
            order.add_item(item)

        assert order.item_count == 6
        assert order.subtotal.amount == 26_000_000  # 10M + 10M + 6M

    def test_confirm_order(self):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Test", Money(100_000), 1))
        order.payment_method = PaymentMethod.MOMO
        order.confirm()

        assert order.status == OrderStatus.CONFIRMED
        assert order.paid_at is not None
        assert order.is_paid

    def test_cancel_pending_order(self):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Test", Money(100_000), 1))
        order.cancel("User requested")
        assert order.status == OrderStatus.CANCELLED
        assert order.cancel_reason == "User requested"

    def test_cancel_after_confirm(self):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Test", Money(100_000), 1))
        order.payment_method = PaymentMethod.COD
        order.confirm()
        order.cancel("Customer changed mind")
        assert order.status == OrderStatus.CANCELLED

    def test_cannot_cancel_delivered_order(self):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Test", Money(100_000), 1))
        order.status = OrderStatus.DELIVERED

        with pytest.raises(InvalidOrderStateError):
            order.cancel()

    def test_cannot_confirm_without_items(self):
        order = Order(customer_id="CUST-001")
        with pytest.raises(InvalidOrderStateError, match="empty"):
            order.confirm()

    def test_cannot_add_item_after_confirmation(self):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Test", Money(100_000), 1))
        order.payment_method = PaymentMethod.MOMO
        order.confirm()

        with pytest.raises(InvalidOrderStateError):
            order.add_item(OrderItem("P2", "New", Money(50_000), 1))

    def test_order_lifecycle(self):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Test", Money(100_000), 1))
        order.payment_method = PaymentMethod.CREDIT_CARD
        order.confirm()
        order.ship()
        order.deliver()

        assert order.status == OrderStatus.DELIVERED
        assert order.version == 4  # Initial + confirm + ship + deliver

    def test_recalculate_on_add(self):
        order = Order(customer_id="CUST-001")
        order.shipping_fee = Money(30_000)
        order.tax = Money(10_000)

        order.add_item(OrderItem("P1", "Test", Money(100_000), 2))
        assert order.subtotal.amount == 200_000
        # total = subtotal + shipping + tax - discount
        expected = 200_000 + 30_000 + 10_000 - 0
        assert order.total.amount == expected

    def test_money_value_object(self):
        m1 = Money(100_000)
        m2 = Money(50_000)
        assert (m1 + m2).amount == 150_000
        assert (m1 - m2).amount == 50_000
        assert (m1 * 2).amount == 200_000
        assert m1.currency == "VND"
```

### File: `tests/application/test_order_service.py`

```python
"""Unit tests for use cases — adapters are mocked/stubbed."""

from unittest.mock import MagicMock
import pytest

from domain.entities import Customer, Order, OrderStatus, Product
from domain.exceptions import CustomerNotFoundError, ProductNotFoundError
from domain.value_objects import Address, Money, OrderItem
from application.dto import CreateOrderInput
from application.ports.outbound import (
    CustomerRepository,
    EventPublisher,
    NotificationSender,
    OrderRepository,
    PaymentGateway,
    ProductRepository,
)
from application.services.order_service import OrderService


@pytest.fixture
def mock_ports():
    """Create mock adapters for testing."""
    return {
        "order_repo": MagicMock(spec=OrderRepository),
        "product_repo": MagicMock(spec=ProductRepository),
        "customer_repo": MagicMock(spec=CustomerRepository),
        "notification_sender": MagicMock(spec=NotificationSender),
        "event_publisher": MagicMock(spec=EventPublisher),
    }


@pytest.fixture
def order_service(mock_ports) -> OrderService:
    return OrderService(
        order_repo=mock_ports["order_repo"],
        product_repo=mock_ports["product_repo"],
        customer_repo=mock_ports["customer_repo"],
        notification_sender=mock_ports["notification_sender"],
        event_publisher=mock_ports["event_publisher"],
    )


class TestCreateOrder:

    def test_successful_order_creation(self, order_service, mock_ports):
        """Happy path: create an order with valid data."""
        # Setup mocks
        mock_ports["customer_repo"].find_by_id.return_value = Customer(
            id="CUST-001", name="Test", email="test@example.com",
        )
        mock_ports["product_repo"].find_by_id.return_value = Product(
            id="PROD-001", name="iPhone", price=Money(10_000_000),
            stock_quantity=5,
        )
        mock_ports["order_repo"].save.return_value = Order(
            id="ORD-001", customer_id="CUST-001",
            items=[OrderItem("PROD-001", "iPhone", Money(10_000_000), 1)],
        )

        result = order_service.create_order(
            CreateOrderInput(
                customer_id="CUST-001",
                items=[{"product_id": "PROD-001", "quantity": 1}],
            )
        )

        assert result is not None
        assert result.id == "ORD-001"
        assert len(result.items) == 1
        mock_ports["order_repo"].save.assert_called_once()
        mock_ports["notification_sender"].send_order_confirmation.assert_called_once()
        mock_ports["event_publisher"].publish.assert_called_once()

    def test_customer_not_found(self, order_service, mock_ports):
        """Error: customer does not exist."""
        mock_ports["customer_repo"].find_by_id.return_value = None

        with pytest.raises(CustomerNotFoundError):
            order_service.create_order(
                CreateOrderInput(
                    customer_id="INVALID",
                    items=[{"product_id": "P1", "quantity": 1}],
                )
            )

    def test_product_not_found(self, order_service, mock_ports):
        """Error: product does not exist."""
        mock_ports["customer_repo"].find_by_id.return_value = Customer(
            id="CUST-001", name="Test", email="test@test.com",
        )
        mock_ports["product_repo"].find_by_id.return_value = None

        with pytest.raises(ProductNotFoundError):
            order_service.create_order(
                CreateOrderInput(
                    customer_id="CUST-001",
                    items=[{"product_id": "INVALID", "quantity": 1}],
                )
            )

    def test_insufficient_stock(self, order_service, mock_ports):
        """Error: not enough inventory."""
        mock_ports["customer_repo"].find_by_id.return_value = Customer(
            id="CUST-001", name="Test", email="test@test.com",
        )
        mock_ports["product_repo"].find_by_id.return_value = Product(
            id="PROD-001", name="Test", price=Money(100_000),
            stock_quantity=1,  # Only 1 in stock
        )

        with pytest.raises(Exception, match="Insufficient"):
            order_service.create_order(
                CreateOrderInput(
                    customer_id="CUST-001",
                    items=[{"product_id": "PROD-001", "quantity": 5}],
                )
            )

    def test_multiple_products(self, order_service, mock_ports):
        """Order with multiple products."""
        mock_ports["customer_repo"].find_by_id.return_value = Customer(
            id="CUST-001", name="Test", email="test@test.com",
        )

        def find_product(pid):
            products = {
                "P1": Product("P1", "iPhone", Money(10_000_000), stock_quantity=10),
                "P2": Product("P2", "iPad", Money(5_000_000), stock_quantity=10),
            }
            return products.get(pid)

        mock_ports["product_repo"].find_by_id.side_effect = find_product
        mock_ports["order_repo"].save.return_value = Order(
            id="ORD-002", customer_id="CUST-001",
            items=[
                OrderItem("P1", "iPhone", Money(10_000_000), 2),
                OrderItem("P2", "iPad", Money(5_000_000), 1),
            ],
        )

        result = order_service.create_order(
            CreateOrderInput(
                customer_id="CUST-001",
                items=[
                    {"product_id": "P1", "quantity": 2},
                    {"product_id": "P2", "quantity": 1},
                ],
            )
        )

        assert len(result.items) == 2
```

### File: `tests/adapters/test_repositories.py`

```python
"""Integration tests for persistence adapter — uses real SQLite."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from domain.entities import Order, OrderStatus
from domain.value_objects import Address, Money, OrderItem
from adapters.outbound.persistence.repositories import PostgresOrderRepository


@pytest.fixture
def db_session():
    """Create in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                subtotal REAL,
                shipping_fee REAL,
                tax REAL,
                discount REAL,
                total REAL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                paid_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                cancel_reason TEXT,
                version INTEGER
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                product_name TEXT,
                price REAL,
                quantity INTEGER,
                sku TEXT,
                sort_order INTEGER
            )
        """))

    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()


@pytest.fixture
def repo(db_session: Session) -> PostgresOrderRepository:
    return PostgresOrderRepository(db_session)


class TestPostgresOrderRepository:

    def test_save_and_find_by_id(self, repo: PostgresOrderRepository):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Product 1", Money(100_000), 2))

        saved = repo.save(order)
        assert saved.id == order.id

        found = repo.find_by_id(order.id)
        assert found is not None
        assert found.customer_id == "CUST-001"
        assert found.item_count == 2
        assert found.subtotal.amount == 200_000

    def test_find_by_id_not_found(self, repo):
        assert repo.find_by_id("NONEXIST") is None

    def test_save_updates_existing_order(self, repo):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Test", Money(100_000), 1))
        repo.save(order)

        # Update
        order.confirm()
        repo.save(order)

        found = repo.find_by_id(order.id)
        assert found.status == OrderStatus.CONFIRMED
        assert found.paid_at is not None

    def test_find_by_customer(self, repo):
        repo.save(Order(customer_id="CUST-001"))
        repo.save(Order(customer_id="CUST-001"))
        repo.save(Order(customer_id="CUST-002"))

        orders = repo.find_by_customer("CUST-001")
        assert len(orders) == 2

        orders2 = repo.find_by_customer("CUST-002")
        assert len(orders2) == 1

    def test_delete_order(self, repo):
        order = Order(customer_id="CUST-001")
        order.add_item(OrderItem("P1", "Test", Money(100_000), 1))
        repo.save(order)

        repo.delete(order.id)
        assert repo.find_by_id(order.id) is None
```

---

## Khi nào dùng / Khi nào không

### ✅ Khi nào dùng Hexagonal Architecture

| Tình huống | Lý do |
|-----------|-------|
| **Domain logic phức tạp** (finance, healthcare, trading) | Business logic là tài sản quý nhất, cần bảo vệ khỏi infrastructure |
| **Testability là ưu tiên hàng đầu** | Test business logic mà không cần infrastructure |
| **Nhiều infrastructure khác nhau** (SQL + NoSQL + cache + cloud) | Adapter cho mỗi technology, dễ swap |
| **DDD (Domain-Driven Design)** | Hexagonal là kiến trúc tự nhiên cho DDD |
| **Long-lived project** | Business logic sống lâu hơn framework |
| **Multiple delivery mechanisms** (API, CLI, batch, message consumer) | Cùng business logic cho nhiều inbound adapter |
| **Microservices** | Mỗi service có thể dùng Hexagonal nội bộ |

### ❌ Khi nào KHÔNG dùng

| Tình huống | Lý do | Alternative |
|-----------|-------|-------------|
| **Ứng dụng CRUD đơn giản** | Over-engineering, quá nhiều abstraction | Layered, MVC |
| **Prototype / MVP** | Cần tốc độ tối đa | Script, Flask/Django |
| **Team nhỏ, ít kinh nghiệm** | Learning curve cao | Start với layered, evolve khi cần |
| **Performance-critical path** | Abstraction layers thêm overhead | Direct implementation |
| **Data-centric application** | Logic chính là data, không phải domain | Data-driven architecture |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Domain isolation**: Business logic hoàn toàn độc lập với infrastructure | **Complexity**: Nhiều abstraction, interface, adapter hơn layered |
| **Testability**: Test domain/use cases mà không cần DB, network, framework | **Learning curve**: Khó hiểu hơn so với layered truyền thống |
| **Flexibility**: Swap database, UI, message broker chỉ bằng adapter mới | **More code**: Cần interface cho mọi port, adapter cho mọi implementation |
| **Dependency inversion đúng nghĩa**: Module cấp cao (domain) không phụ thuộc module cấp thấp (infra) | **Over-engineering risk**: Có thể quá abstract cho bài toán đơn giản |
| **Domain-driven**: Use case là trung tâm, không phải database | **Performance overhead**: Mỗi layer abstraction thêm một chút latency |
| **Framework độc lập**: Domain code không import framework | **Development speed chậm hơn initial phase** |
| **Dễ maintain lâu dài**: Business logic không bị ảnh hưởng bởi technology churn | **Debugging**: Stack trace dài hơn, nhiều indirection |

---

## Công cụ và Framework

### Inbound Adapters

| Loại | Công cụ | Mục đích |
|------|---------|----------|
| **API** | FastAPI / Flask / Django REST | REST endpoints |
| **GraphQL** | Strawberry / Graphene | GraphQL API |
| **CLI** | Click / Typer / Argparse | Command-line interface |
| **Message Consumer** | aiokafka / kombu / pika | Consume events from message broker |
| **gRPC** | grpcio | High-performance RPC |

### Outbound Adapters

| Loại | Công cụ | Mục đích |
|------|---------|----------|
| **Relational DB** | SQLAlchemy / asyncpg / psycopg2 | PostgreSQL, MySQL |
| **NoSQL** | motor / redis-py / cassandra-driver | MongoDB, Redis, Cassandra |
| **Payment** | Stripe / PayPal SDK | Payment processing |
| **Email** | SendGrid / Mailgun / SMTP | Email notifications |
| **SMS** | Twilio / Vonage | SMS notifications |
| **Message Broker** | aiokafka / kombu / aio-pika | Kafka, RabbitMQ |
| **File Storage** | boto3 (S3) / google-cloud-storage | Cloud file storage |
| **Cache** | redis-py / aioredis | Caching |

### Testing Tools

| Công cụ | Mục đích |
|---------|----------|
| **pytest** | Test framework |
| **pytest-asyncio** | Async test support |
| **unittest.mock** | Mock adapters for unit tests |
| **sqlite** | In-memory DB for repository tests |
| **Testcontainers** | Integration tests with real services (PostgreSQL, Kafka) |

---

## Kiểm thử chiến lược

### Hexagonal Test Pyramid

```
         /\
        /  \
       /    \
      / E2E  \         ← Full system with real adapters (5%)
     /  Tests  \
    /───────────\
   /  Adapter   \      ← Integration tests per adapter (15%)
  /    Tests      \
 /─────────────────\
/   Use Case Tests  \  ← Business logic with mock adapters (40%)
/────────────────────\
/   Domain Tests      \ ← Pure domain logic, no mocks (40%)
/──────────────────────\
```

### Testing Rules

1. **Domain tests**: 100% pure — no mocks, no infrastructure, no frameworks
2. **Use case tests**: Mock outbound adapters, test business logic + orchestration
3. **Adapter tests**: Real infrastructure (SQLite for DB, test container for Kafka)
4. **E2E tests**: Full system — API → use case → adapter → external

---

## Kết luận

Hexagonal Architecture là một trong những kiến trúc mạnh mẽ nhất để xây dựng các hệ thống có domain logic phức tạp, yêu cầu testability cao, và cần tồn tại lâu dài. Nó là nền tảng cho Clean Architecture của Robert C. Martin và là kiến trúc được khuyến nghị khi áp dụng Domain-Driven Design.

### Best Practices

1. **Domain trước, infrastructure sau** — thiết kế use case và entity trước, chọn công nghệ sau
2. **Ports ở application layer** — interface cho repository, gateway, publisher
3. **Adapters ở infrastructure** — implementation cụ thể, import framework ở đây
4. **DTOs cho cross-boundary communication** — không dùng domain entity trực tiếp ở adapter
5. **Dependency Injection qua constructor** — không dùng service locator
6. **Use case = một class** — mỗi use case là một class với single responsibility
7. **Domain entity có behavior** — không phải anemic model
8. **Value objects immutable** — frozen dataclasses
9. **Business exception có semantic** — không dùng generic Exception
10. **Aggressive testing ở domain layer** — đây là tài sản quý nhất

### Golden Rules

> 1. **Domain không import gì từ bên ngoài.** Không import framework, database driver, HTTP library.
> 2. **Port là của inside, không phải của outside.** Interface do domain định nghĩa, adapter implement.
> 3. **Dependency chỉ đi vào trong.** Adapter → Port → Use case → Domain.
> 4. **Framework là plugin.** Application không biết nó đang chạy trên framework nào.
> 5. **Test domain không cần mock.** Nếu bạn cần mock để test domain, bạn đang làm sai.

### Comparison: Hexagonal vs Other Architectures

| Tiêu chí | Hexagonal | Layered | Clean Architecture |
|----------|-----------|---------|-------------------|
| **Trọng tâm** | Ports & Adapters | Layer separation | Dependency Rule |
| **Domain isolation** | ✅ Rất cao | ❌ Thấp (phụ thuộc ORM) | ✅ Rất cao |
| **Testability** | ✅ Xuất sắc | ⚠️ Trung bình | ✅ Xuất sắc |
| **Complexity** | ⚠️ Cao | ✅ Thấp | ⚠️ Cao |
| **Flexibility** | ✅ Cao | ⚠️ Trung bình | ✅ Cao |
| **Popularity** | ⚠️ Trung bình | ✅ Rất cao | ⚠️ Trung bình |

### Next Steps

Sau Hexagonal Architecture, hãy tìm hiểu **Clean Architecture** — sự mở rộng của Hexagonal với nhiều layer concentric hơn. Hoặc **Domain-Driven Design (DDD)** — phương pháp luận thiết kế phần mềm dựa trên domain, mà Hexagonal là kiến trúc lý tưởng để implement.
