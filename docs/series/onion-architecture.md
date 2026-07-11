---
id: onion-architecture
title: Onion Architecture — Kiến trúc Hành tây
sidebar_label: Onion Architecture
sidebar_position: 40
---

# Onion Architecture — Kiến trúc Hành tây

> *"The fundamental principle of Onion Architecture is that the domain is the core — everything else wraps around it like layers of an onion."* — Jeffrey Palermo

---

## Tổng quan

**Onion Architecture** (Kiến trúc Hành tây) được Jeffrey Palermo giới thiệu lần đầu vào năm 2008 như một giải pháp thay thế cho kiến trúc分层 (layered architecture) truyền thống. Ý tưởng cốt lõi là **Domain** (miền nghiệp vụ) phải nằm ở trung tâm, hoàn toàn độc lập với cơ sở hạ tầng (database, UI, framework). Các layer bên ngoài giao tiếp với layer bên trong thông qua **inversion of control** (IoC) và **dependency injection** (DI).

Những người tiên phong và áp dụng thành công:
- **Jeffrey Palermo** — tác giả gốc, nguyên mẫu trong bài blog năm 2008
- **Robert C. Martin (Uncle Bob)** — ý tưởng tương tự trong *Clean Architecture* (2012)
- **Eric Evans** — Domain-Driven Design (2003) là nền tảng tư tưởng
- **Jimmy Bogard** — người sáng lập AutoMapper, áp dụng trong các hệ thống enterprise .NET

Trong thế giới Python, Onion Architecture được hiện thực qua các pattern như Repository Pattern, Unit of Work, Service Layer, và Dependency Injection container.

---

## Bài toán

### Sự phụ thuộc chết người vào infrastructure

Trong kiến trúc layered truyền thống (Presentation → Business → Data), layer Data (database) nằm ở đáy và Business layer phụ thuộc trực tiếp vào nó. Điều này tạo ra một vấn đề nan giải: khi bạn thay đổi database từ PostgreSQL sang MongoDB, hoặc thay đổi ORM từ SQLAlchemy sang Django ORM, bạn buộc phải viết lại toàn bộ Business layer. Business logic — thứ quyết định giá trị của hệ thống — lại phụ thuộc vào chi tiết kỹ thuật.

Hãy tưởng tượng bạn xây một ngôi nhà mà tường chịu lực lại gắn chặt vào lớp sơn trang trí. Muốn đổi màu sơn, bạn phải đập tường. Đó là hệ quả của việc để infrastructure chi phối domain.

### Khó kiểm thử (testability)

Khi business logic phụ thuộc trực tiếp vào database hoặc web framework, việc viết unit test trở nên cực kỳ khó khăn. Bạn phải mock hàng tá đối tượng infrastructure, test trở nên chậm và dễ vỡ (brittle). Một thay đổi nhỏ trong SQLAlchemy session có thể kéo theo hàng loạt test bị fail, dù business logic hoàn toàn không thay đổi.

### Khó bảo trì và mở rộng

Các hệ thống thiết kế theo layered architecture thường rơi vào tình trạng *big ball of mud* sau một thời gian phát triển. Business logic bị rò rỉ vào controller, repository bị lẫn với service, và không có ranh giới rõ ràng giữa các concern. Khi cần thêm một use case mới, developer phải đọc hàng nghìn dòng code để hiểu chỗ nào cần sửa.

### Vòng đời phát triển chậm

Vì mọi thứ đều phụ thuộc lẫn nhau, việc phát triển song song trở nên bất khả thi. Team database không thể làm việc độc lập với team business logic. Kết quả là tiến độ dự án chậm, và mỗi lần release là một lần căng thẳng.

---

## Nguyên lý thiết kế

### 1. Dependency Inversion Principle (DIP)

Các module cấp cao (domain) không được phụ thuộc vào module cấp thấp (infrastructure). Cả hai đều phải phụ thuộc vào abstraction (interface). Đây là chữ **D** trong SOLID.

Thay vì:
```python
class OrderService:
    def __init__(self):
        self.repo = PostgresOrderRepository()  # phụ thuộc trực tiếp
```

Chúng ta viết:
```python
class OrderService:
    def __init__(self, repo: OrderRepository):  # phụ thuộc vào abstraction
        self._repo = repo
```

### 2. Domain là trung tâm (Domain-centric)

Mọi business rule phải nằm trong Domain layer. Domain không import bất kỳ module nào từ infrastructure hay presentation. Nó hoàn toàn thuần khiết (pure Python).

### 3. Inversion of Control (IoC)

Layer bên ngoài gọi layer bên trong thông qua interface. Layer bên trong không biết gì về layer bên ngoài. Việc "cắm" implementation cụ thể được thực hiện ở tầng ứng dụng (composition root).

### 4. Boundaries qua Ports & Adapters

- **Port**: Interface định nghĩa contract (ví dụ: `OrderRepository` là port)
- **Adapter**: Implementation cụ thể (ví dụ: `PostgresOrderRepository` là adapter)

## Cấu trúc chi tiết

Onion Architecture có 4 layer chính, từ trong ra ngoài:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  (API, CLI, Web UI, GraphQL)                                │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Application Layer                         │  │
│  │  (Use Cases, DTOs, Application Services,              │  │
│  │   Unit of Work, CQRS handlers)                        │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │              Domain Layer                       │  │  │
│  │  │  (Entities, Value Objects, Aggregates,         │  │  │
│  │  │   Domain Events, Repository Interfaces,        │  │  │
│  │  │   Domain Services, Specifications)             │  │  │
│  │  │                                                 │  │  │
│  │  │  ┌───────────────────────────────────────────┐  │  │  │
│  │  │  │          Core / Model                     │  │  │  │
│  │  │  │   (Domain Primitives, Business Rules)     │  │  │  │
│  │  │  └───────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Infrastructure Layer                     │  │
│  │  (Database, File System, Message Queue, Email,       │  │
│  │   ORM, Cache, External APIs)                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Chi tiết từng layer

#### Domain Layer (Core)
- **Entities**: Đối tượng có identity, vòng đời (ví dụ: `Order`, `Customer`)
- **Value Objects**: Đối tượng bất biến, so sánh bằng value (ví dụ: `Money`, `Address`)
- **Aggregates**: Nhóm entities với transaction boundary
- **Domain Events**: Sự kiện trong domain (ví dụ: `OrderPlaced`)
- **Repository Interfaces**: Contract cho data access
- **Domain Services**: Business logic không thuộc entity nào

#### Application Layer
- **Use Cases / Application Services**: Điều phối business logic
- **DTOs**: Data Transfer Objects
- **Unit of Work**: Quản lý transaction
- **Application Events**: Sự kiện ứng dụng

#### Infrastructure Layer
- **Repository Implementations**: SQLAlchemy, Django ORM, MongoDB
- **Message Bus**: RabbitMQ, Kafka
- **File Storage**: S3, Local
- **Email**: SMTP, SendGrid

#### Presentation Layer
- **REST API**: FastAPI, Flask
- **CLI**: Click, Typer
- **Web UI**: React, Vue

---

## Sơ đồ kiến trúc

```
                    ┌─────────────────────────┐
                    │      Presentation       │
                    │  ┌───────────────────┐  │
                    │  │   Application     │  │
                    │  │  ┌─────────────┐  │  │
                    │  │  │   Domain    │  │  │
                    │  │  │ ┌─────────┐ │  │  │
                    │  │  │ │  Core   │ │  │  │
                    │  │  │ └─────────┘ │  │  │
                    │  │  └─────────────┘  │  │
                    │  └───────────────────┘  │
                    └─────────────────────────┘
                            ▲     │
                            │     │ Calls interface
                            │     ▼
                    ┌─────────────────────────┐
                    │     Infrastructure      │
                    │  (Implements interface) │
                    └─────────────────────────┘
```

**Luồng gọi điển hình:**

```
HTTP Request
    → Controller (Presentation)
        → OrderUseCase (Application)
            → OrderRepository interface (Domain)
                ← PostgresOrderRepository (Infrastructure)
            → Domain Entity logic
        ← Response DTO
    ← HTTP Response
```

---

## Ví dụ code hoàn chỉnh

Chúng ta sẽ xây dựng một hệ thống **xử lý đơn hàng thương mại điện tử** hoàn chỉnh.

### Cấu trúc project

```
ecommerce/
├── domain/
│   ├── __init__.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── order.py
│   │   ├── product.py
│   │   └── customer.py
│   ├── value_objects/
│   │   ├── __init__.py
│   │   ├── money.py
│   │   ├── address.py
│   │   └── order_status.py
│   ├── events/
│   │   ├── __init__.py
│   │   └── domain_events.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── discount_service.py
│   │   └── shipping_service.py
│   └── repositories/
│       ├── __init__.py
│       └── interfaces.py
├── application/
│   ├── __init__.py
│   ├── use_cases/
│   │   ├── __init__.py
│   │   ├── place_order.py
│   │   ├── cancel_order.py
│   │   └── get_order.py
│   ├── dto/
│   │   ├── __init__.py
│   │   └── order_dto.py
│   └── interfaces/
│       ├── __init__.py
│       └── unit_of_work.py
├── infrastructure/
│   ├── __init__.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── postgres_order_repo.py
│   │   └── postgres_product_repo.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── connection.py
│   └── unit_of_work/
│       ├── __init__.py
│       └── sqlalchemy_uow.py
├── presentation/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── order_controller.py
│   │   └── app.py
│   └── cli/
│       ├── __init__.py
│       └── order_cli.py
├── tests/
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── test_order.py
│   │   └── test_money.py
│   ├── application/
│   │   ├── __init__.py
│   │   └── test_place_order.py
│   └── infrastructure/
│       ├── __init__.py
│       └── test_repositories.py
├── pyproject.toml
└── requirements.txt
```

### File: domain/__init__.py

```python
"""
Domain layer - hoàn toàn không phụ thuộc vào infrastructure.
Đây là layer quan trọng nhất, chứa mọi business rule.
"""

from .entities.order import Order, OrderLine
from .entities.product import Product
from .entities.customer import Customer
from .value_objects.money import Money
from .value_objects.address import Address
from .value_objects.order_status import OrderStatus
from .events.domain_events import DomainEvent, OrderPlaced, OrderCancelled
from .services.discount_service import DiscountService
from .services.shipping_service import ShippingService
from .repositories.interfaces import OrderRepository, ProductRepository

__all__ = [
    "Order", "OrderLine", "Product", "Customer",
    "Money", "Address", "OrderStatus",
    "DomainEvent", "OrderPlaced", "OrderCancelled",
    "DiscountService", "ShippingService",
    "OrderRepository", "ProductRepository",
]
```

### File: domain/value_objects/money.py

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional


@dataclass(frozen=True)
class Money:
    """Value Object bất biến biểu diễn tiền tệ."""
    amount: Decimal
    currency: str = "VND"

    def __post_init__(self) -> None:
        if self.amount < Decimal("0"):
            raise ValueError("Số tiền không được âm")
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Mã tiền tệ phải gồm 3 ký tự")

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"Không thể cộng {self.currency} với {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"Không thể trừ {self.currency} với {other.currency}")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, multiplier: Decimal) -> Money:
        result = (self.amount * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return Money(result, self.currency)

    def __gt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError("Không thể so sánh khác loại tiền")
        return self.amount > other.amount

    def __lt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError("Không thể so sánh khác loại tiền")
        return self.amount < other.amount

    def __ge__(self, other: Money) -> bool:
        return self > other or self == other

    def __le__(self, other: Money) -> bool:
        return self < other or self == other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency

    def __hash__(self) -> int:
        return hash((self.amount, self.currency))

    def __str__(self) -> str:
        if self.currency == "VND":
            return f"{self.amount:,.0f}₫"
        return f"{self.currency} {self.amount:,.2f}"

    @classmethod
    def zero(cls, currency: str = "VND") -> Money:
        """Tạo số tiền 0."""
        return cls(Decimal("0"), currency)

    @classmethod
    def vnd(cls, amount: str | int | float | Decimal) -> Money:
        """Tạo tiền VND từ các kiểu dữ liệu khác nhau."""
        return cls(Decimal(str(amount)), "VND")
```

### File: domain/value_objects/order_status.py

```python
from __future__ import annotations

from enum import Enum, auto


class OrderStatus(Enum):
    """Trạng thái đơn hàng - định nghĩa rõ ràng các trạng thái business."""
    PENDING = auto()
    CONFIRMED = auto()
    PROCESSING = auto()
    SHIPPED = auto()
    DELIVERED = auto()
    CANCELLED = auto()
    REFUNDED = auto()

    def can_transition_to(self, new_status: OrderStatus) -> bool:
        """Kiểm tra xem có được phép chuyển sang trạng thái mới không."""
        allowed: dict[OrderStatus, set[OrderStatus]] = {
            OrderStatus.PENDING: {OrderStatus.CONFIRMED, OrderStatus.CANCELLED},
            OrderStatus.CONFIRMED: {OrderStatus.PROCESSING, OrderStatus.CANCELLED},
            OrderStatus.PROCESSING: {OrderStatus.SHIPPED, OrderStatus.CANCELLED},
            OrderStatus.SHIPPED: {OrderStatus.DELIVERED},
            OrderStatus.DELIVERED: {OrderStatus.REFUNDED},
            OrderStatus.CANCELLED: set(),
            OrderStatus.REFUNDED: set(),
        }
        return new_status in allowed.get(self, set())

    def __str__(self) -> str:
        return self.name.title()
```

### File: domain/value_objects/address.py

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Address:
    """Value Object bất biến biểu diễn địa chỉ."""
    street: str
    ward: str
    district: str
    city: str
    country: str = "Việt Nam"

    def __post_init__(self) -> None:
        if not self.street.strip():
            raise ValueError("Đường không được để trống")
        if not self.city.strip():
            raise ValueError("Thành phố không được để trống")

    def __str__(self) -> str:
        return f"{self.street}, {self.ward}, {self.district}, {self.city}, {self.country}"
```

### File: domain/events/domain_events.py

```python
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID, uuid4


T = TypeVar("T")


class DomainEvent(ABC):
    """Base class cho mọi domain event."""
    event_id: UUID
    occurred_at: datetime


@dataclass
class OrderPlaced(DomainEvent):
    """Sự kiện: đơn hàng được đặt."""
    order_id: UUID
    customer_id: UUID
    total: float
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderCancelled(DomainEvent):
    """Sự kiện: đơn hàng bị hủy."""
    order_id: UUID
    reason: str
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderShipped(DomainEvent):
    """Sự kiện: đơn hàng được gửi."""
    order_id: UUID
    tracking_code: str
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
```

### File: domain/entities/customer.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Optional

from domain.value_objects.address import Address
from domain.value_objects.money import Money


@dataclass
class Customer:
    """Entity khách hàng - có identity (id) và vòng đời."""
    name: str
    email: str
    phone: str
    address: Address
    id: UUID = field(default_factory=uuid4)
    loyalty_points: int = 0
    is_vip: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tên khách hàng không được để trống")
        if "@" not in self.email:
            raise ValueError("Email không hợp lệ")

    def add_loyalty_points(self, points: int) -> None:
        """Thêm điểm tích lũy."""
        if points < 0:
            raise ValueError("Điểm tích lũy không được âm")
        self.loyalty_points += points
        if self.loyalty_points >= 1000 and not self.is_vip:
            self.is_vip = True

    def get_vip_discount(self) -> float:
        """Lấy tỷ lệ giảm giá VIP."""
        if self.is_vip:
            return 0.10  # 10% cho VIP
        return 0.0
```

### File: domain/entities/product.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Optional

from domain.value_objects.money import Money


@dataclass
class Product:
    """Entity sản phẩm."""
    name: str
    price: Money
    sku: str
    stock_quantity: int
    id: UUID = field(default_factory=uuid4)
    description: str = ""
    is_active: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tên sản phẩm không được để trống")
        if self.stock_quantity < 0:
            raise ValueError("Tồn kho không được âm")

    def reduce_stock(self, quantity: int) -> None:
        """Giảm tồn kho khi bán."""
        if quantity <= 0:
            raise ValueError("Số lượng phải lớn hơn 0")
        if quantity > self.stock_quantity:
            raise ValueError(f"Không đủ hàng: chỉ còn {self.stock_quantity}")
        self.stock_quantity -= quantity

    def increase_stock(self, quantity: int) -> None:
        """Tăng tồn kho khi nhập."""
        if quantity <= 0:
            raise ValueError("Số lượng phải lớn hơn 0")
        self.stock_quantity += quantity

    def is_available(self, quantity: int = 1) -> bool:
        """Kiểm tra hàng còn đủ không."""
        return self.is_active and self.stock_quantity >= quantity
```

### File: domain/entities/order.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from domain.value_objects.money import Money
from domain.value_objects.order_status import OrderStatus


@dataclass
class OrderLine:
    """Một dòng trong đơn hàng - giá trị đã được snapshot tại thời điểm đặt."""
    product_id: UUID
    product_name: str
    quantity: int
    unit_price: Money
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Số lượng phải lớn hơn 0")

    @property
    def subtotal(self) -> Money:
        return self.unit_price * Decimal(str(self.quantity))

    def __str__(self) -> str:
        return f"{self.product_name} x{self.quantity} = {self.subtotal}"


@dataclass
class Order:
    """Aggregate Root: Đơn hàng - là trung tâm của bounded context."""
    customer_id: UUID
    order_lines: list[OrderLine]
    id: UUID = field(default_factory=uuid4)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    shipping_address: Optional[str] = None
    discount: Money = field(default_factory=lambda: Money.zero())
    _events: list = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.order_lines:
            raise ValueError("Đơn hàng phải có ít nhất một sản phẩm")

    @property
    def total_before_discount(self) -> Money:
        return sum(
            (line.subtotal for line in self.order_lines),
            start=Money.zero(),
        )

    @property
    def total(self) -> Money:
        return self.total_before_discount - self.discount

    @property
    def total_items(self) -> int:
        return sum(line.quantity for line in self.order_lines)

    def apply_discount(self, discount_amount: Money) -> None:
        """Áp dụng giảm giá cho đơn hàng."""
        if discount_amount > self.total_before_discount:
            raise ValueError("Giảm giá không thể lớn hơn tổng tiền hàng")
        self.discount = discount_amount

    def confirm(self) -> None:
        """Xác nhận đơn hàng."""
        self._transition_to(OrderStatus.CONFIRMED)

    def cancel(self, reason: str = "") -> None:
        """Hủy đơn hàng."""
        self._transition_to(OrderStatus.CANCELLED)

    def ship(self) -> None:
        """Gửi hàng."""
        self._transition_to(OrderStatus.SHIPPED)

    def deliver(self) -> None:
        """Xác nhận giao hàng thành công."""
        self._transition_to(OrderStatus.DELIVERED)

    def _transition_to(self, new_status: OrderStatus) -> None:
        """Transition logic - core business rule."""
        if not self.status.can_transition_to(new_status):
            raise ValueError(
                f"Không thể chuyển từ {self.status} sang {new_status}"
            )
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def collect_events(self) -> list:
        """Lấy và xóa các domain event."""
        events = list(self._events)
        self._events.clear()
        return events

    def __str__(self) -> str:
        return f"Order({self.id}, {self.status}, {self.total})"
```

### File: domain/repositories/interfaces.py

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from domain.entities.order import Order
from domain.entities.product import Product
from domain.entities.customer import Customer


class OrderRepository(ABC):
    """Port: Interface cho repository đơn hàng."""

    @abstractmethod
    def add(self, order: Order) -> None:
        """Thêm đơn hàng mới."""
        ...

    @abstractmethod
    def get_by_id(self, order_id: UUID) -> Optional[Order]:
        """Lấy đơn hàng theo ID."""
        ...

    @abstractmethod
    def update(self, order: Order) -> None:
        """Cập nhật đơn hàng."""
        ...

    @abstractmethod
    def delete(self, order_id: UUID) -> None:
        """Xóa đơn hàng."""
        ...

    @abstractmethod
    def list_by_customer(self, customer_id: UUID) -> list[Order]:
        """Lấy danh sách đơn hàng của khách."""
        ...


class ProductRepository(ABC):
    """Port: Interface cho repository sản phẩm."""

    @abstractmethod
    def add(self, product: Product) -> None:
        ...

    @abstractmethod
    def get_by_id(self, product_id: UUID) -> Optional[Product]:
        ...

    @abstractmethod
    def get_by_sku(self, sku: str) -> Optional[Product]:
        ...

    @abstractmethod
    def update(self, product: Product) -> None:
        ...

    @abstractmethod
    def list_available(self) -> list[Product]:
        ...


class CustomerRepository(ABC):
    """Port: Interface cho repository khách hàng."""

    @abstractmethod
    def add(self, customer: Customer) -> None:
        ...

    @abstractmethod
    def get_by_id(self, customer_id: UUID) -> Optional[Customer]:
        ...

    @abstractmethod
    def get_by_email(self, email: str) -> Optional[Customer]:
        ...
```

### File: domain/services/discount_service.py

```python
from __future__ import annotations

from decimal import Decimal

from domain.entities.order import Order
from domain.entities.customer import Customer
from domain.value_objects.money import Money


class DiscountService:
    """Domain Service: Xử lý logic giảm giá phức tạp."""

    def calculate_discount(self, order: Order, customer: Customer) -> Money:
        """Tính giảm giá dựa trên nhiều yếu tố."""
        total = order.total_before_discount
        discounts: list[Money] = []

        # Giảm giá theo cấp độ khách hàng
        vip_discount = total * Decimal(str(customer.get_vip_discount()))
        discounts.append(vip_discount)

        # Giảm giá theo số lượng
        if order.total_items >= 10:
            bulk_discount = total * Decimal("0.05")
            discounts.append(bulk_discount)

        # Giảm giá theo tổng giá trị
        if total.amount >= Decimal("10000000"):
            amount_discount = total * Decimal("0.03")
            discounts.append(amount_discount)

        # Tổng hợp giảm giá
        total_discount = sum(discounts, start=Money.zero())

        # Giới hạn giảm giá tối đa 30%
        max_discount = total * Decimal("0.30")
        if total_discount > max_discount:
            total_discount = max_discount

        return total_discount
```

### File: domain/services/shipping_service.py

```python
from __future__ import annotations

from decimal import Decimal

from domain.entities.order import Order
from domain.entities.customer import Customer
from domain.value_objects.money import Money


class ShippingService:
    """Domain Service: Tính phí vận chuyển."""

    FREE_SHIPPING_THRESHOLD = Money.vnd("500000")
    STANDARD_RATE = Money.vnd("30000")
    EXPRESS_RATE = Money.vnd("50000")

    def calculate_shipping(
        self,
        order: Order,
        customer: Customer,
        is_express: bool = False,
    ) -> Money:
        """Tính phí vận chuyển."""
        # Miễn phí cho đơn hàng lớn
        if order.total_before_discount >= self.FREE_SHIPPING_THRESHOLD:
            return Money.zero()

        # Miễn phí cho VIP
        if customer.is_vip:
            return Money.zero()

        return self.EXPRESS_RATE if is_express else self.STANDARD_RATE
```

### File: application/dto/order_dto.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass
class OrderLineDTO:
    """Data Transfer Object cho OrderLine."""
    product_id: str
    product_name: str
    quantity: int
    unit_price: float
    subtotal: float


@dataclass
class PlaceOrderInput:
    """Input DTO cho use case đặt hàng."""
    customer_id: UUID
    items: list[dict]
    shipping_address: str
    discount_code: Optional[str] = None


@dataclass
class OrderResponse:
    """Response DTO cho đơn hàng."""
    id: str
    customer_id: str
    status: str
    total: float
    total_before_discount: float
    discount: float
    items: list
    created_at: str
    shipping_address: str


@dataclass
class ErrorResponse:
    """Response DTO cho lỗi."""
    error: str
    code: str
    details: Optional[str] = None
```

### File: application/interfaces/unit_of_work.py

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class UnitOfWork(ABC):
    """Unit of Work pattern - quản lý transaction."""

    @abstractmethod
    def __enter__(self) -> UnitOfWork:
        ...

    @abstractmethod
    def __exit__(self, *args) -> None:
        ...

    @abstractmethod
    def commit(self) -> None:
        """Commit transaction."""
        ...

    @abstractmethod
    def rollback(self) -> None:
        """Rollback transaction."""
        ...
```

### File: application/use_cases/place_order.py

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from domain.entities.order import Order, OrderLine
from domain.entities.product import Product
from domain.value_objects.money import Money
from domain.value_objects.order_status import OrderStatus
from domain.events.domain_events import OrderPlaced
from domain.repositories.interfaces import OrderRepository, ProductRepository, CustomerRepository
from domain.services.discount_service import DiscountService
from domain.services.shipping_service import ShippingService
from application.dto.order_dto import PlaceOrderInput, OrderResponse, OrderLineDTO
from application.interfaces.unit_of_work import UnitOfWork


class PlaceOrderUseCase:
    """Use Case: Đặt hàng - điều phối business logic."""

    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo: ProductRepository,
        customer_repo: CustomerRepository,
        discount_service: DiscountService,
        shipping_service: ShippingService,
        uow: UnitOfWork,
    ):
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._customer_repo = customer_repo
        self._discount_service = discount_service
        self._shipping_service = shipping_service
        self._uow = uow

    def execute(self, input_dto: PlaceOrderInput) -> OrderResponse:
        """Thực thi use case đặt hàng."""
        with self._uow:
            # 1. Load customer
            customer = self._customer_repo.get_by_id(input_dto.customer_id)
            if customer is None:
                raise ValueError(f"Không tìm thấy khách hàng {input_dto.customer_id}")

            # 2. Tạo order lines từ input
            order_lines: list[OrderLine] = []
            for item in input_dto.items:
                product = self._product_repo.get_by_id(UUID(item["product_id"]))
                if product is None:
                    raise ValueError(f"Không tìm thấy sản phẩm {item['product_id']}")
                if not product.is_available(item["quantity"]):
                    raise ValueError(f"Sản phẩm {product.name} không đủ hàng")

                line = OrderLine(
                    product_id=product.id,
                    product_name=product.name,
                    quantity=item["quantity"],
                    unit_price=product.price,
                )
                order_lines.append(line)

                # Giảm tồn kho
                product.reduce_stock(item["quantity"])
                self._product_repo.update(product)

            # 3. Tạo order
            order = Order(
                customer_id=customer.id,
                order_lines=order_lines,
                shipping_address=input_dto.shipping_address,
            )

            # 4. Tính giảm giá
            discount = self._discount_service.calculate_discount(order, customer)
            if discount > Money.zero():
                order.apply_discount(discount)

            # 5. Thêm domain event
            order._events.append(OrderPlaced(
                order_id=order.id,
                customer_id=customer.id,
                total=float(order.total.amount),
            ))

            # 6. Lưu
            self._order_repo.add(order)
            customer.add_loyalty_points(order.total_items * 10)
            self._customer_repo.update(customer)

            # 7. Commit
            self._uow.commit()

            # 8. Trả về response
            return OrderResponse(
                id=str(order.id),
                customer_id=str(customer.id),
                status=order.status.name,
                total=float(order.total.amount),
                total_before_discount=float(order.total_before_discount.amount),
                discount=float(order.discount.amount),
                items=[
                    {
                        "product_name": line.product_name,
                        "quantity": line.quantity,
                        "unit_price": float(line.unit_price.amount),
                        "subtotal": float(line.subtotal.amount),
                    }
                    for line in order_lines
                ],
                created_at=order.created_at.isoformat(),
                shipping_address=order.shipping_address or "",
            )
```

### File: application/use_cases/cancel_order.py

```python
from __future__ import annotations

from uuid import UUID

from domain.repositories.interfaces import OrderRepository
from domain.events.domain_events import OrderCancelled
from application.dto.order_dto import OrderResponse
from application.interfaces.unit_of_work import UnitOfWork


class CancelOrderUseCase:
    """Use Case: Hủy đơn hàng."""

    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo,
        uow: UnitOfWork,
    ):
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._uow = uow

    def execute(self, order_id: UUID, reason: str = "") -> OrderResponse:
        """Thực thi hủy đơn."""
        with self._uow:
            order = self._order_repo.get_by_id(order_id)
            if order is None:
                raise ValueError(f"Không tìm thấy đơn hàng {order_id}")

            # Hoàn lại tồn kho
            for line in order.order_lines:
                product = self._product_repo.get_by_id(line.product_id)
                if product:
                    product.increase_stock(line.quantity)
                    self._product_repo.update(product)

            # Hủy đơn
            order.cancel(reason)
            order._events.append(OrderCancelled(
                order_id=order.id,
                reason=reason,
            ))

            self._order_repo.update(order)
            self._uow.commit()

            return OrderResponse(
                id=str(order.id),
                customer_id=str(order.customer_id),
                status=order.status.name,
                total=float(order.total.amount),
                total_before_discount=float(order.total_before_discount.amount),
                discount=float(order.discount.amount),
                items=[],
                created_at=order.created_at.isoformat(),
                shipping_address=order.shipping_address or "",
            )
```

### File: infrastructure/repositories/postgres_order_repo.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from domain.entities.order import Order
from domain.repositories.interfaces import OrderRepository


class PostgresOrderRepository(OrderRepository):
    """Adapter: PostgreSQL implementation của OrderRepository."""

    def __init__(self, session):
        self._session = session

    def add(self, order: Order) -> None:
        self._session.add(order)

    def get_by_id(self, order_id: UUID) -> Optional[Order]:
        return self._session.get(Order, order_id)

    def update(self, order: Order) -> None:
        self._session.merge(order)

    def delete(self, order_id: UUID) -> None:
        order = self.get_by_id(order_id)
        if order:
            self._session.delete(order)

    def list_by_customer(self, customer_id: UUID) -> list[Order]:
        return list(
            self._session.query(Order)
            .filter(Order.customer_id == customer_id)
            .all()
        )
```

### File: infrastructure/repositories/postgres_product_repo.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from domain.entities.product import Product
from domain.repositories.interfaces import ProductRepository


class PostgresProductRepository(ProductRepository):
    """Adapter: PostgreSQL implementation của ProductRepository."""

    def __init__(self, session):
        self._session = session

    def add(self, product: Product) -> None:
        self._session.add(product)

    def get_by_id(self, product_id: UUID) -> Optional[Product]:
        return self._session.get(Product, product_id)

    def get_by_sku(self, sku: str) -> Optional[Product]:
        return (
            self._session.query(Product)
            .filter(Product.sku == sku)
            .first()
        )

    def update(self, product: Product) -> None:
        self._session.merge(product)

    def list_available(self) -> list[Product]:
        return list(
            self._session.query(Product)
            .filter(Product.is_active == True)
            .all()
        )
```

### File: infrastructure/unit_of_work/sqlalchemy_uow.py

```python
from __future__ import annotations

from application.interfaces.unit_of_work import UnitOfWork


class SqlAlchemyUnitOfWork(UnitOfWork):
    """Adapter: SQLAlchemy Unit of Work."""

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._session = None

    def __enter__(self) -> UnitOfWork:
        self._session = self._session_factory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.rollback()
        self._session.close()

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

    @property
    def session(self):
        return self._session
```

### File: presentation/api/order_controller.py

```python
from __future__ import annotations

from uuid import UUID

import json


class OrderController:
    """Controller nhận request HTTP và gọi use case."""

    def __init__(self, place_order_uc, cancel_order_uc, get_order_uc):
        self._place_order = place_order_uc
        self._cancel_order = cancel_order_uc
        self._get_order = get_order_uc

    def handle_place_order(self, request_body: str) -> str:
        """POST /orders"""
        try:
            data = json.loads(request_body)
            input_dto = PlaceOrderInput(
                customer_id=UUID(data["customer_id"]),
                items=data["items"],
                shipping_address=data.get("shipping_address", ""),
                discount_code=data.get("discount_code"),
            )
            result = self._place_order.execute(input_dto)
            return json.dumps({"success": True, "data": result.__dict__}, ensure_ascii=False)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": "Internal server error"}, ensure_ascii=False)

    def handle_cancel_order(self, order_id: str, reason: str = "") -> str:
        """POST /orders/{id}/cancel"""
        try:
            result = self._cancel_order.execute(UUID(order_id), reason)
            return json.dumps({"success": True, "data": result.__dict__}, ensure_ascii=False)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
```

### File: presentation/api/app.py

```python
from __future__ import annotations

"""
FastAPI application - Composition Root.
Đây là nơi duy nhất DI container được cấu hình.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from domain.services.discount_service import DiscountService
from domain.services.shipping_service import ShippingService
from application.use_cases.place_order import PlaceOrderUseCase
from application.use_cases.cancel_order import CancelOrderUseCase
from application.use_cases.get_order import GetOrderUseCase
from infrastructure.repositories.postgres_order_repo import PostgresOrderRepository
from infrastructure.repositories.postgres_product_repo import PostgresProductRepository
from infrastructure.repositories.postgres_customer_repo import PostgresCustomerRepository
from infrastructure.unit_of_work.sqlalchemy_uow import SqlAlchemyUnitOfWork

# Composition Root
def create_app() -> FastAPI:
    app = FastAPI(title="Onion Architecture E-Commerce", version="1.0.0")

    # Infrastructure
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("postgresql://user:pass@localhost/ecommerce")
    session_factory = sessionmaker(bind=engine)
    uow = SqlAlchemyUnitOfWork(session_factory)

    order_repo = PostgresOrderRepository(session_factory)
    product_repo = PostgresProductRepository(session_factory)
    customer_repo = PostgresCustomerRepository(session_factory)

    # Domain Services
    discount_service = DiscountService()
    shipping_service = ShippingService()

    # Application Use Cases
    place_order_uc = PlaceOrderUseCase(
        order_repo, product_repo, customer_repo,
        discount_service, shipping_service, uow,
    )
    cancel_order_uc = CancelOrderUseCase(order_repo, product_repo, uow)
    get_order_uc = GetOrderUseCase(order_repo)

    # Controller
    controller = OrderController(place_order_uc, cancel_order_uc, get_order_uc)

    @app.post("/orders")
    async def create_order(request: dict):
        return controller.handle_place_order(json.dumps(request))

    @app.post("/orders/{order_id}/cancel")
    async def cancel_order(order_id: str):
        return controller.handle_cancel_order(order_id)

    return app


app = create_app()
```

### File: tests/domain/test_money.py

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from domain.value_objects.money import Money


class TestMoney:
    """Kiểm thử Value Object Money."""

    def test_create_money_success(self):
        money = Money(Decimal("100000"), "VND")
        assert money.amount == Decimal("100000")
        assert money.currency == "VND"

    def test_create_money_negative_raises_error(self):
        with pytest.raises(ValueError, match="không được âm"):
            Money(Decimal("-1000"))

    def test_create_money_invalid_currency(self):
        with pytest.raises(ValueError, match="3 ký tự"):
            Money(Decimal("1000"), "VNDD")

    def test_add_same_currency(self):
        a = Money(Decimal("10000"), "VND")
        b = Money(Decimal("20000"), "VND")
        assert (a + b) == Money(Decimal("30000"), "VND")

    def test_add_different_currency_raises_error(self):
        a = Money(Decimal("100"), "VND")
        b = Money(Decimal("50"), "USD")
        with pytest.raises(ValueError, match="Không thể cộng"):
            _ = a + b

    def test_money_immutability(self):
        money = Money(Decimal("50000"), "VND")
        with pytest.raises(AttributeError):
            money.amount = Decimal("100000")

    def test_money_zero(self):
        zero = Money.zero()
        assert zero.amount == Decimal("0")

    def test_vnd_convenience(self):
        money = Money.vnd("150000")
        assert money.currency == "VND"
        assert money.amount == Decimal("150000")

    def test_multiplication(self):
        money = Money(Decimal("1000"), "VND") * Decimal("5")
        assert money == Money(Decimal("5000"), "VND")

    def test_comparison(self):
        a = Money(Decimal("100"), "VND")
        b = Money(Decimal("200"), "VND")
        assert a < b
        assert b > a
        assert a <= b
        assert b >= a

    def test_str_vnd(self):
        money = Money(Decimal("1500000"), "VND")
        assert "1,500,000₫" in str(money)
```

### File: tests/domain/test_order.py

```python
from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from domain.entities.order import Order, OrderLine
from domain.value_objects.money import Money
from domain.value_objects.order_status import OrderStatus


class TestOrder:
    """Kiểm thử Domain Entity Order."""

    @pytest.fixture
    def valid_order(self):
        customer_id = uuid4()
        product_id = uuid4()
        lines = [
            OrderLine(
                product_id=product_id,
                product_name="Áo thun",
                quantity=2,
                unit_price=Money(Decimal("150000"), "VND"),
            )
        ]
        return Order(
            customer_id=customer_id,
            order_lines=lines,
        )

    def test_create_order_success(self, valid_order):
        assert valid_order.status == OrderStatus.PENDING
        assert valid_order.total_items == 2

    def test_create_order_empty_lines_raises_error(self):
        with pytest.raises(ValueError, match="ít nhất một sản phẩm"):
            Order(
                customer_id=uuid4(),
                order_lines=[],
            )

    def test_confirm_order(self, valid_order):
        valid_order.confirm()
        assert valid_order.status == OrderStatus.CONFIRMED

    def test_cancel_from_pending(self, valid_order):
        valid_order.cancel("Khách hủy")
        assert valid_order.status == OrderStatus.CANCELLED

    def test_cancel_from_confirmed(self, valid_order):
        valid_order.confirm()
        valid_order.cancel("Hết hàng")
        assert valid_order.status == OrderStatus.CANCELLED

    def test_invalid_transition(self, valid_order):
        valid_order.deliver()
        # Không thể chuyển từ DELIVERED -> CANCELLED
        with pytest.raises(ValueError, match="Không thể chuyển"):
            valid_order.cancel()

    def test_cannot_deliver_before_shipping(self, valid_order):
        with pytest.raises(ValueError, match="Không thể chuyển"):
            valid_order.deliver()

    def test_apply_discount(self, valid_order):
        discount = Money(Decimal("50000"), "VND")
        original_total = valid_order.total_before_discount
        valid_order.apply_discount(discount)
        assert valid_order.total == original_total - discount

    def test_discount_cannot_exceed_total(self, valid_order):
        huge_discount = Money(Decimal("999999999"), "VND")
        with pytest.raises(ValueError, match="Giảm giá không thể lớn hơn"):
            valid_order.apply_discount(huge_discount)

    def test_total_calculation(self, valid_order):
        total = valid_order.total_before_discount
        assert total == Money(Decimal("300000"), "VND")

    def test_orderline_subtotal(self):
        line = OrderLine(
            product_id=uuid4(),
            product_name="Test",
            quantity=3,
            unit_price=Money(Decimal("50000"), "VND"),
        )
        assert line.subtotal == Money(Decimal("150000"), "VND")
```

### File: tests/application/test_place_order.py

```python
from __future__ import annotations

from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from domain.entities.order import OrderLine
from domain.value_objects.money import Money
from domain.services.discount_service import DiscountService
from domain.services.shipping_service import ShippingService
from application.use_cases.place_order import PlaceOrderUseCase
from application.dto.order_dto import PlaceOrderInput


class TestPlaceOrderUseCase:
    """Kiểm thử Use Case đặt hàng."""

    @pytest.fixture
    def mock_repos(self):
        return {
            "order_repo": MagicMock(),
            "product_repo": MagicMock(),
            "customer_repo": MagicMock(),
        }

    @pytest.fixture
    def use_case(self, mock_repos):
        return PlaceOrderUseCase(
            order_repo=mock_repos["order_repo"],
            product_repo=mock_repos["product_repo"],
            customer_repo=mock_repos["customer_repo"],
            discount_service=DiscountService(),
            shipping_service=ShippingService(),
            uow=MagicMock(),
        )

    def test_place_order_success(self, use_case, mock_repos):
        # Setup
        customer_id = uuid4()
        product_id = uuid4()
        mock_repos["customer_repo"].get_by_id.return_value = MagicMock(
            id=customer_id,
            name="Nguyễn Văn A",
            email="test@test.com",
            phone="0909123456",
            address=MagicMock(),
            loyalty_points=0,
            is_vip=False,
        )
        mock_repos["product_repo"].get_by_id.return_value = MagicMock(
            id=product_id,
            name="Laptop Dell",
            price=Money.vnd("15000000"),
            stock_quantity=10,
            is_available=True,
            reduce_stock=MagicMock(),
            is_active=True,
        )

        input_dto = PlaceOrderInput(
            customer_id=customer_id,
            items=[{"product_id": str(product_id), "quantity": 1}],
            shipping_address="123 Nguyễn Huệ, Q1, HCM",
        )

        # Execute
        result = use_case.execute(input_dto)

        # Verify
        assert result.status == "PENDING"
        assert result.total_before_discount == 15000000.0
        mock_repos["order_repo"].add.assert_called_once()
        mock_repos["uow"].commit.assert_called_once()

    def test_place_order_customer_not_found(self, use_case, mock_repos):
        mock_repos["customer_repo"].get_by_id.return_value = None

        input_dto = PlaceOrderInput(
            customer_id=uuid4(),
            items=[],
            shipping_address="",
        )

        with pytest.raises(ValueError, match="Không tìm thấy khách hàng"):
            use_case.execute(input_dto)
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng | Khi nào không |
|---|---|
| Hệ thống có business logic phức tạp, nhiều rule | Ứng dụng CRUD đơn giản, ít nghiệp vụ |
| Cần testability cao, muốn test business logic độc lập | MVP, prototype cần ra mắt nhanh |
| Dự án dài hạn, nhiều team phát triển song song | Hệ thống nhỏ, 1-2 developer |
| Nhiều infrastructure thay đổi (đổi DB, message queue) | Ứng dụng không có infrastructure phức tạp |
| Cần maintain và phát triển trong 5-10 năm | Script đơn lẻ, batch job nhỏ |
| Tuân thủ DDD, cần bounded context rõ ràng | Dự án có domain đơn giản (ví dụ: blog, CMS cơ bản) |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Domain độc lập hoàn toàn khỏi infrastructure | Chi phí thiết lập ban đầu cao |
| Dễ dàng kiểm thử business logic | Nhiều boilerplate code |
| Dependency Injection giúp lỏng lẻo kết nối | Khó học đối với junior developer |
| Dễ thay đổi infrastructure | Over-engineering cho dự án nhỏ |
| Tuân thủ SOLID principles | Cần understanding về DIP và IoC |
| Các team có thể làm việc độc lập | Cần DI container hoặc factory |
| Bảo trì và mở rộng dễ dàng | Debug có thể phức tạp hơn |

---

## Công cụ và Framework

### Python
- **Dependency Injector** — DI container mạnh mẽ
- **SQLAlchemy 2.0** — ORM với repository pattern
- **Alembic** — Database migration
- **FastAPI** — Modern web framework
- **Pydantic** — Data validation và DTOs
- **pytest** — Testing framework
- **Click / Typer** — CLI applications
- **APScheduler** — Task scheduling

### .NET (nguồn gốc)
- **ASP.NET Core** — Web framework
- **Entity Framework Core** — ORM
- **AutoMapper** — Object mapping
- **MediatR** — CQRS implementation
- **FluentValidation** — Validation

### Java
- **Spring Boot** — Application framework
- **Spring Data JPA** — Repository pattern
- **MapStruct** — Object mapping
- **Axon Framework** — CQRS/Event Sourcing

---

## Kiểm thử

### Chiến lược kiểm thử trong Onion Architecture

Onion Architecture cho phép kiểm thử business logic mà không cần infrastructure thật:

```
Test Pyramid in Onion:
    ┌──────┐
    │ E2E  │  ← Few: full system tests
    ├──────┤
    │ Int  │  ← Some: integration with DB
    ├──────┤
    │ Unit │  ← Many: domain + use cases
    └──────┘
```

### File: tests/infrastructure/test_repositories.py

```python
from __future__ import annotations

"""
Integration test cho repository - cần database thật.
Dùng testcontainers hoặc SQLite in-memory.
"""

import pytest
from uuid import uuid4
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from domain.entities.product import Product
from domain.value_objects.money import Money
from infrastructure.repositories.postgres_product_repo import PostgresProductRepository


@pytest.fixture
def db_session():
    """SQLite in-memory cho test."""
    engine = create_engine("sqlite:///:memory:")
    # Tạo bảng ở đây
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def product_repo(db_session):
    return PostgresProductRepository(db_session)


class TestProductRepository:
    """Kiểm thử Product Repository."""

    def test_add_and_get_product(self, product_repo, db_session):
        product = Product(
            name="iPhone 15",
            price=Money(Decimal("25000000"), "VND"),
            sku="IP15-BLK-128",
            stock_quantity=50,
        )
        product_repo.add(product)
        db_session.flush()

        retrieved = product_repo.get_by_id(product.id)
        assert retrieved is not None
        assert retrieved.name == "iPhone 15"
        assert retrieved.sku == "IP15-BLK-128"
```

### File: tests/conftest.py

```python
from __future__ import annotations

"""
Pytest configuration với fixtures dùng chung.
"""

import pytest
from uuid import uuid4
from decimal import Decimal
from unittest.mock import MagicMock

from domain.entities.product import Product
from domain.entities.customer import Customer
from domain.value_objects.money import Money
from domain.value_objects.address import Address


@pytest.fixture
def sample_product():
    return Product(
        name="Laptop Dell XPS",
        price=Money(Decimal("35000000"), "VND"),
        sku="DELL-XPS-15",
        stock_quantity=10,
    )


@pytest.fixture
def sample_customer():
    return Customer(
        name="Nguyễn Văn An",
        email="an.nguyen@example.com",
        phone="0909123456",
        address=Address(
            street="123 Lê Lợi",
            ward="Bến Nghé",
            district="Quận 1",
            city="Hồ Chí Minh",
        ),
        loyalty_points=500,
    )
```

---

## Kết luận

Onion Architecture là một kiến trúc mạnh mẽ cho các hệ thống doanh nghiệp phức tạp. Nó buộc developer phải suy nghĩ về domain trước tiên, và infrastructure chỉ là chi tiết có thể thay thế.

### Best Practices

1.  **Domain First**: Luôn bắt đầu từ domain entities và business rules
2.  **Interface Segregation**: Repository interfaces nên nhỏ, chuyên biệt
3.  **Dependency Injection**: Inject dependency qua constructor, không dùng service locator
4.  **Composition Root**: Chỉ một nơi duy nhất được cấu hình DI
5.  **Testing Pyramid**: Unit test domain nhiều, integration test vừa phải
6.  **Don't Over-engineer**: Nếu ứng dụng chỉ là CRUD, layered architecture có thể đủ
7.  **Use a DI Container**: Dependency Injector (Python) hoặc built-in DI (FastAPI)

### Golden Rules

| Rule | Mô tả |
|---|---|
| **Domain không import infrastructure** | Domain layer không được import bất cứ thứ gì từ infrastructure |
| **Repository chỉ trả về Aggregate** | Không trả về ORM entity, chỉ trả về domain entity |
| **Use Case là đơn vị điều phối** | Mỗi use case một class, một method execute |
| **Unit of Work quản lý transaction** | Không dùng transaction trong repository |
| **Immutable Value Objects** | Value objects phải immutable, có `__hash__` |
| **Domain Events cho side effects** | Không gọi service trực tiếp, dùng event |

Onion Architecture không phải là silver bullet, nhưng nó là công cụ đắc lực khi bạn xây dựng hệ thống mà business logic là tài sản quý giá nhất.
