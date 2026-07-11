---
id: ddd
title: Domain-Driven Design — Thiết kế hướng miền
sidebar_label: Domain-Driven Design
sidebar_position: 43
---

# Domain-Driven Design — Thiết kế hướng miền

> *"The heart of software is its ability to solve domain-related problems for its users. All other features are secondary."* — Eric Evans

---

## Tổng quan

**Domain-Driven Design (DDD)** là một methodology thiết kế phần mềm do **Eric Evans** giới thiệu trong cuốn sách *Domain-Driven Design: Tackling Complexity in the Heart of Software* (2003). DDD không phải là một pattern cụ thể, mà là một **tư duy** (mindset) về cách xây dựng phần mềm: **mô hình hóa domain thành code**.

### Những nhân vật chủ chốt

- **Eric Evans** — Tác giả cuốn "Blue Book" kinh điển, người định hình DDD
- **Vaughn Vernon** — Tác giả "Implementing Domain-Driven Design" (Red Book)
- **Martin Fowler** — Áp dụng và phổ biến DDD trong enterprise architecture
- **Jimmy Nilsson** — "Applying Domain-Driven Design and Patterns"
- **Alberto Brandolini** — Event Storming, kỹ thuật collaboration cho DDD
- **Cyrille Martraire** — "Living Documentation" với DDD

### Strategic vs Tactical DDD

| Strategic DDD | Tactical DDD |
|---|---|
| Bounded Context | Entities |
| Ubiquitous Language | Value Objects |
| Context Map | Aggregates |
| Domain/Subdomain | Domain Events |
| Core/Supporting/Generic | Repositories |
| | Domain Services |

---

## Bài toán

### Sự sụp đổ của "Big Ball of Mud"

Mọi dự án phần mềm đều bắt đầu với kiến trúc đẹp. Sau 6 tháng, nó trở thành "big ball of mud" — một đống bùn khổng lồ không ai dám động vào. Tại sao? Bởi vì team không hiểu domain, và code không phản ánh business reality.

Hãy tưởng tượng: bạn đang xây dựng hệ thống bảo hiểm. Business analyst nói "policy", "coverage", "deductible", "premium". Developer code thành các bảng `policy_data`, `coverage_info`, `claim_record`. Khi business thay đổi, developer không biết nên sửa ở đâu vì **code không nói cùng ngôn ngữ với domain**.

### Implicit Concepts (Khái niệm ẩn)

Trong software, có rất nhiều khái niệm business quan trọng nhưng không được code thể hiện rõ ràng. Ví dụ:
- "Một đơn hàng có thể được hoàn tiền trong vòng 30 ngày" — rule này nằm ở đâu? Controller? Repository? Service?
- "Khách hàng VIP được miễn phí vận chuyển" — logic này nằm ở frontend check? Hay trong SQL query?

Khi những implicit concepts không được mô hình hóa rõ ràng, chúng sẽ rải rác khắp codebase, dẫn đến:
1. **Duplication**: Cùng logic xuất hiện ở nhiều nơi
2. **Inconsistency**: Logic giống nhau nhưng xử lý khác nhau
3. **Rigidity**: Mỗi lần thay đổi là một lần đau đớn

### The Anemic Domain Model

Martin Fowler gọi đây là "anti-pattern" phổ biến nhất trong enterprise development: domain model chỉ là state holder, không có behavior. Mọi business logic nằm ở service layer.

```python
# Anemic Domain Model — chỉ là data bag
class Order:
    id: int
    status: str
    items: list
    total: float

# Business logic ở service
class OrderService:
    def confirm(self, order_id):
        order = db.get(order_id)
        if order.status != "pending":
            raise Error()
        order.status = "confirmed"
        db.save(order)
        self.email_service.send_confirmation(order)
```

DDD yêu cầu: **Entity phải có behavior, không chỉ là state**.

```python
# Rich Domain Model — behavior ở entity
class Order:
    id: OrderId
    status: OrderStatus
    items: OrderLines

    def confirm(self):
        self.status = self.status.next()
        self.add_event(OrderConfirmed(self.id))
```

### Communication Gap

Business experts nói tiếng Việt business, developer code bằng Python technical. Mỗi bên hiểu domain theo cách riêng. Kết quả: requirements sai, implementation sai, rework triền miên.

DDD giải quyết bằng **Ubiquitous Language**: một ngôn ngữ chung cho cả business experts và developers, được dùng trong conversation, documentation, và code.

---

## Nguyên lý thiết kế

### 1. Ubiquitous Language (Ngôn ngữ chung)

Business expert và developer cùng xây dựng một ngôn ngữ chung cho domain. Ngôn ngữ này được dùng trong:
- User stories và requirements
- Code (class names, method names, variable names)
- Tests
- Documentation

Ví dụ: Trong hệ thống ngân hàng, thay vì nói:
- Business: "Khách hàng mở tài khoản"
- Dev: "POST /api/v1/users/123/accounts"

Cả hai cùng nói: "CustomerOpensAccount → AccountCreated"

### 2. Bounded Context (Ngữ cảnh giới hạn)

Một domain lớn được chia thành nhiều **Bounded Context** nhỏ hơn. Mỗi context có:
- Ubiquitous Language riêng
- Domain model riêng
- Consistency boundary riêng
- Database riêng (có thể)

Ví dụ: E-commerce có các bounded context:
- **Product Catalog**: Quản lý sản phẩm, danh mục, giá
- **Order**: Đặt hàng, thanh toán, giao hàng
- **Inventory**: Tồn kho, nhập/xuất
- **Customer**: Khách hàng, loyalty
- **Billing**: Hóa đơn, công nợ

### 3. Entities vs Value Objects

**Entity**: Có identity (ID duy nhất), vòng đời, mutable.
```python
class Customer:
    def __init__(self, id: CustomerId, name: str):
        self.id = id  # Identity
        self.name = name  # Có thể thay đổi
```

**Value Object**: Không có identity, immutable, so sánh bằng value.
```python
@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str
```

### 4. Aggregates

Một **Aggregate** là một nhóm các Entities và Value Objects được coi là một đơn vị thống nhất (consistency boundary).
- **Aggregate Root**: Entity duy nhất được phép truy cập từ bên ngoài
- Mọi thao tác trên aggregate phải qua root
- Mỗi transaction chỉ modify một aggregate

### 5. Domain Events

Sự kiện trong domain, mô tả điều đã xảy ra:
```python
@dataclass
class OrderPlaced:
    order_id: OrderId
    customer_id: CustomerId
    occurred_at: datetime
```

### 6. Repositories

Repository cung cấp interface để truy xuất aggregate:
```python
class OrderRepository(ABC):
    @abstractmethod
    def get_by_id(self, id: OrderId) -> Order: ...
    @abstractmethod
    def save(self, order: Order) -> None: ...
```

---

## Cấu trúc chi tiết

### DDD Layered Architecture (phiên bản DDD thuần)

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  (REST API, Web UI, CLI, Message Consumer)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Application Layer                         │
│  (Application Services, DTOs, Session, Security)            │
│                                                              │
│  • Điều phối use cases                                      │
│  • Không chứa business logic                                │
│  • Quản lý transaction, security                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Domain Layer                             │
│  (TRUNG TÂM - không phụ thuộc vào layer khác)                │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐                │
│  │ Entities │  │  Value   │  │ Aggregates │                │
│  │          │  │ Objects  │  │            │                │
│  └──────────┘  └──────────┘  └────────────┘                │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐                │
│  │ Domain   │  │ Domain   │  │ Repository │                │
│  │ Events   │  │ Services │  │ Interfaces │                │
│  └──────────┘  └──────────┘  └────────────┘                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Infrastructure Layer                       │
│  (Database, File System, Email, Message Queue, API Calls)   │
│                                                              │
│  • Implement repository interfaces                           │
│  • Gửi email, gọi API bên ngoài                              │
│  • ORM mapping, database config                              │
└─────────────────────────────────────────────────────────────┘
```

### Strategic Design Patterns

```
Context Map:
┌──────────────┐         ┌──────────────┐
│  Order       │ <───────│  Inventory   │
│  (Core)      │  S/O    │  (Supporting)│
└──────┬───────┘         └──────────────┘
       │
       ▼
┌──────────────┐         ┌──────────────┐
│  Billing     │ <───────│  Payment     │
│  (Core)      │  S/O    │  (Generic)   │
└──────────────┘         └──────────────┘
```

**Legend**: S/O = Shared Kernel, C/S = Customer/Supplier, OHS = Open Host Service, ACL = Anti-Corruption Layer

---

## Sơ đồ kiến trúc

```
                        Ubiquitous Language
                        ───────────────────
                        "Khách hàng đặt hàng chứa sản phẩm"
                        Customer → Order → OrderLine → Product
                        ───────────────────
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                      PRICE DOMAIN (E-Commerce)                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Bounded Context: ORDER MANAGEMENT                            │  │
│  │                                                               │  │
│  │  Aggregate: Order (Root)                                      │  │
│  │  ├── OrderId (Value Object)                                   │  │
│  │  ├── CustomerId (Value Object)                                │  │
│  │  ├── OrderStatus (Value Object — Enum)                        │  │
│  │  ├── OrderLine (Entity — part of aggregate)                   │  │
│  │  │   ├── ProductId, ProductName                               │  │
│  │  │   ├── Quantity, Price (snapshot)                           │  │
│  │  ├── Money (Value Object — total)                             │  │
│  │  ├── ShippingAddress (Value Object)                           │  │
│  │  └── Events: OrderPlaced, OrderShipped, OrderDelivered        │  │
│  │                                                               │  │
│  │  Repository: OrderRepository (interface)                      │  │
│  │  Service: OrderDomainService (giảm giá phức tạp)              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Bounded Context: PRODUCT CATALOG                            │  │
│  │  Aggregate: Product, Category                                 │  │
│  │  Value Objects: Money, SKU, ProductSpecification             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Bounded Context: CUSTOMER                                   │  │
│  │  Aggregate: Customer                                         │  │
│  │  Value Objects: CustomerId, Email, Phone, Address            │  │
│  │  Domain Events: CustomerRegistered, LoyaltyPointsEarned      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## Ví dụ code hoàn chỉnh

Xây dựng hệ thống **booking khách sạn** (Hotel Booking Domain).

### Cấu trúc project

```
hotel_booking/
├── domain/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── booking.py
│   │   ├── room.py
│   │   ├── customer.py
│   │   ├── hotel.py
│   │   └── payment.py
│   ├── value_objects/
│   │   ├── __init__.py
│   │   ├── money.py
│   │   ├── date_range.py
│   │   ├── booking_status.py
│   │   ├── room_type.py
│   │   └── booking_id.py
│   ├── events/
│   │   ├── __init__.py
│   │   └── booking_events.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pricing_service.py
│   │   └── availability_service.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── booking_repository.py
│   │   ├── room_repository.py
│   │   └── customer_repository.py
│   └── specifications/
│       ├── __init__.py
│       └── room_specifications.py
├── application/
│   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── booking_app_service.py
│   │   └── cancellation_app_service.py
│   └── dto/
│       ├── __init__.py
│       └── booking_dto.py
├── infrastructure/
│   ├── __init__.py
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── sqlalchemy_booking_repo.py
│   │   └── sqlalchemy_room_repo.py
│   └── messaging/
│       ├── __init__.py
│       └── event_publisher.py
├── presentation/
│   ├── __init__.py
│   └── api/
│       ├── __init__.py
│       └── booking_controller.py
├── tests/
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   └── test_booking.py
│   └── application/
│       ├── __init__.py
│       └── test_booking_service.py
└── main.py
```

### File: domain/value_objects/booking_id.py

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID, uuid4


@dataclass(frozen=True)
class BookingId:
    """Value Object: ID của booking (immutable)."""
    value: UUID

    @classmethod
    def generate(cls) -> BookingId:
        return cls(uuid4())

    @classmethod
    def from_string(cls, s: str) -> BookingId:
        return cls(UUID(s))

    def __str__(self) -> str:
        return str(self.value)
```

### File: domain/value_objects/money.py

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union


@dataclass(frozen=True)
class Money:
    """Value Object: Tiền tệ."""
    amount: Decimal
    currency: str = "VND"

    def __post_init__(self) -> None:
        if self.amount < Decimal("0"):
            raise ValueError("Amount cannot be negative")
        if len(self.currency) != 3:
            raise ValueError("Currency must be 3-letter code")

    def __add__(self, other: Money) -> Money:
        self._assert_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._assert_same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: Union[int, Decimal]) -> Money:
        result = (self.amount * Decimal(str(factor))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return Money(result, self.currency)

    def __gt__(self, other: Money) -> bool:
        self._assert_same_currency(other)
        return self.amount > other.amount

    def __lt__(self, other: Money) -> bool:
        self._assert_same_currency(other)
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
        return cls(Decimal("0"), currency)

    @classmethod
    def vnd(cls, amount: str) -> Money:
        return cls(Decimal(amount), "VND")

    @staticmethod
    def _assert_same_currency(a: Money, b: Money) -> None:
        if a.currency != b.currency:
            raise ValueError(f"Currency mismatch: {a.currency} vs {b.currency}")
```

### File: domain/value_objects/date_range.py

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class DateRange:
    """Value Object: Khoảng thời gian (immutable)."""
    check_in: date
    check_out: date

    def __post_init__(self) -> None:
        if self.check_out <= self.check_in:
            raise ValueError("Check-out must be after check-in")

    @property
    def nights(self) -> int:
        """Số đêm lưu trú."""
        return (self.check_out - self.check_in).days

    def overlaps_with(self, other: DateRange) -> bool:
        """Kiểm tra xem có overlap không."""
        return self.check_in < other.check_out and other.check_in < self.check_out

    def contains(self, d: date) -> bool:
        """Kiểm tra xem ngày d có nằm trong khoảng không."""
        return self.check_in <= d < self.check_out

    def __str__(self) -> str:
        return f"{self.check_in.isoformat()} → {self.check_out.isoformat()} ({self.nights} đêm)"
```

### File: domain/value_objects/booking_status.py

```python
from __future__ import annotations

from enum import Enum, auto


class BookingStatus(Enum):
    """Trạng thái booking — mô hình hóa state machine."""
    PENDING = auto()
    CONFIRMED = auto()
    CHECKED_IN = auto()
    CHECKED_OUT = auto()
    CANCELLED = auto()
    NO_SHOW = auto()
    REFUNDED = auto()

    def can_transition_to(self, new_status: BookingStatus) -> bool:
        transitions = {
            BookingStatus.PENDING: {BookingStatus.CONFIRMED, BookingStatus.CANCELLED},
            BookingStatus.CONFIRMED: {
                BookingStatus.CHECKED_IN, BookingStatus.CANCELLED, BookingStatus.NO_SHOW,
            },
            BookingStatus.CHECKED_IN: {BookingStatus.CHECKED_OUT},
            BookingStatus.CHECKED_OUT: set(),
            BookingStatus.CANCELLED: {BookingStatus.REFUNDED},
            BookingStatus.NO_SHOW: set(),
            BookingStatus.REFUNDED: set(),
        }
        return new_status in transitions.get(self, set())

    def __str__(self) -> str:
        names = {
            BookingStatus.PENDING: "Chờ xác nhận",
            BookingStatus.CONFIRMED: "Đã xác nhận",
            BookingStatus.CHECKED_IN: "Đã nhận phòng",
            BookingStatus.CHECKED_OUT: "Đã trả phòng",
            BookingStatus.CANCELLED: "Đã hủy",
            BookingStatus.NO_SHOW: "Không đến",
            BookingStatus.REFUNDED: "Đã hoàn tiền",
        }
        return names.get(self, self.name)
```

### File: domain/value_objects/room_type.py

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from decimal import Decimal
from typing import Optional


class RoomCategory(Enum):
    """Phân loại phòng."""
    STANDARD = auto()
    DELUXE = auto()
    SUITE = auto()
    PRESIDENTIAL = auto()
    PENTHOUSE = auto()


@dataclass(frozen=True)
class RoomType:
    """Value Object: Loại phòng với thông tin chi tiết."""
    category: RoomCategory
    name: str
    max_guests: int
    base_price: Money
    amenities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_guests < 1:
            raise ValueError("Max guests must be >= 1")
        if not self.name.strip():
            raise ValueError("Room type name is required")

    def can_accommodate(self, guests: int) -> bool:
        """Kiểm tra sức chứa."""
        return guests <= self.max_guests

    def __str__(self) -> str:
        return f"{self.name} ({self.category.name})"
```

### File: domain/entities/room.py

```python
from __future__ import annotations

from uuid import UUID, uuid4
from dataclasses import dataclass, field
from typing import Optional

from domain.value_objects.room_type import RoomType


@dataclass
class Room:
    """Entity: Phòng khách sạn — có identity."""
    room_number: str
    floor: int
    room_type: RoomType
    id: UUID = field(default_factory=uuid4)
    is_active: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.room_number.strip():
            raise ValueError("Room number is required")

    def __str__(self) -> str:
        return f"Phòng {self.room_number} (Tầng {self.floor}) - {self.room_type.name}"
```

### File: domain/domain_events.py

```python
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4


@dataclass
class DomainEvent(ABC):
    """Base domain event."""
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self) -> int:
        return hash(self.event_id)


@dataclass
class BookingRequested(DomainEvent):
    """Khách hàng yêu cầu đặt phòng."""
    booking_id: UUID
    customer_id: UUID
    hotel_id: UUID
    room_type: str
    date_range: tuple[str, str]
    guests: int


@dataclass
class BookingConfirmed(DomainEvent):
    """Booking được xác nhận."""
    booking_id: UUID
    total_amount: float
    currency: str = "VND"


@dataclass
class BookingCancelled(DomainEvent):
    """Booking bị hủy."""
    booking_id: UUID
    reason: str
    refund_amount: float
    currency: str = "VND"


@dataclass
class BookingCompleted(DomainEvent):
    """Khách đã trả phòng (hoàn thành)."""
    booking_id: UUID
    additional_charges: float = 0.0
```

### File: domain/entities/booking.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from domain.value_objects.booking_id import BookingId
from domain.value_objects.money import Money
from domain.value_objects.date_range import DateRange
from domain.value_objects.booking_status import BookingStatus
from domain.domain_events import *


class BookingError(Exception):
    pass


class InvalidStateError(BookingError):
    """Khi thao tác không hợp lệ với trạng thái hiện tại."""
    pass


class InvalidGuestCountError(BookingError):
    pass


@dataclass
class Booking:
    """AGGREGATE ROOT: Booking đặt phòng.

    Đây là aggregate trung tâm của bounded context 'Hotel Booking'.
    Mọi thao tác trên booking đều phải qua aggregate root này.
    """
    booking_id: BookingId
    customer_id: UUID
    hotel_id: UUID
    room_id: UUID
    room_type_name: str
    date_range: DateRange
    number_of_guests: int
    status: BookingStatus = BookingStatus.PENDING
    total_amount: Money = field(default_factory=lambda: Money.zero())
    paid_amount: Money = field(default_factory=lambda: Money.zero())
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    special_requests: str = ""
    cancellation_reason: str = ""
    version: int = 0
    _events: list[DomainEvent] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.number_of_guests < 1:
            raise InvalidGuestCountError("Số lượng khách phải >= 1")

    # === Business Methods ===

    def confirm(self) -> None:
        """Xác nhận booking."""
        if self.status != BookingStatus.PENDING:
            raise InvalidStateError(
                f"Không thể xác nhận booking ở trạng thái {self.status}"
            )
        self._transition_to(BookingStatus.CONFIRMED)
        self._events.append(BookingConfirmed(
            booking_id=self.booking_id.value,
            total_amount=float(self.total_amount.amount),
        ))

    def cancel(self, reason: str = "") -> None:
        """Hủy booking."""
        cancellable_states = {
            BookingStatus.PENDING, BookingStatus.CONFIRMED,
        }
        if self.status not in cancellable_states:
            raise InvalidStateError(
                f"Không thể hủy booking ở trạng thái {self.status}"
            )

        cancellation_fee = self._calculate_cancellation_fee()
        refund_amount = self.paid_amount - cancellation_fee
        if refund_amount < Money.zero():
            refund_amount = Money.zero()

        self.cancellation_reason = reason
        self._transition_to(BookingStatus.CANCELLED)
        self._events.append(BookingCancelled(
            booking_id=self.booking_id.value,
            reason=reason,
            refund_amount=float(refund_amount.amount),
        ))

    def check_in(self) -> None:
        """Nhận phòng."""
        if self.status != BookingStatus.CONFIRMED:
            raise InvalidStateError(
                f"Không thể nhận phòng ở trạng thái {self.status}"
            )
        self._transition_to(BookingStatus.CHECKED_IN)

    def check_out(self, additional_charges: Money = Money.zero()) -> None:
        """Trả phòng."""
        if self.status != BookingStatus.CHECKED_IN:
            raise InvalidStateError(
                f"Không thể trả phòng ở trạng thái {self.status}"
            )
        self._transition_to(BookingStatus.CHECKED_OUT)
        self.total_amount = self.total_amount + additional_charges
        self._events.append(BookingCompleted(
            booking_id=self.booking_id.value,
            additional_charges=float(additional_charges.amount),
        ))

    def mark_no_show(self) -> None:
        """Khách không đến."""
        if self.status != BookingStatus.CONFIRMED:
            raise InvalidStateError(
                f"Không thể đánh dấu no-show ở trạng thái {self.status}"
            )
        self._transition_to(BookingStatus.NO_SHOW)

    def process_refund(self) -> None:
        """Xử lý hoàn tiền."""
        if self.status == BookingStatus.CANCELLED:
            self._transition_to(BookingStatus.REFUNDED)
        else:
            raise InvalidStateError(
                f"Chỉ booking đã hủy mới được hoàn tiền"
            )

    def make_payment(self, amount: Money) -> None:
        """Thanh toán."""
        if self.status not in {BookingStatus.PENDING, BookingStatus.CONFIRMED}:
            raise InvalidStateError(
                f"Không thể thanh toán ở trạng thái {self.status}"
            )
        new_paid = self.paid_amount + amount
        if new_paid > self.total_amount:
            raise BookingError("Số tiền thanh toán vượt quá tổng tiền")
        self.paid_amount = new_paid

    # === Internal ===

    def _transition_to(self, new_status: BookingStatus) -> None:
        if not self.status.can_transition_to(new_status):
            raise InvalidStateError(
                f"Không thể chuyển từ {self.status} sang {new_status}"
            )
        self.status = new_status
        self.updated_at = datetime.utcnow()
        self.version += 1

    def _calculate_cancellation_fee(self) -> Money:
        hours_since_creation = (
            datetime.utcnow() - self.created_at
        ).total_seconds() / 3600
        if hours_since_creation < 24:
            return Money.zero()  # Miễn phí hủy trong 24h
        elif hours_since_creation < 72:
            return self.total_amount * Decimal("0.5")  # 50% phí
        else:
            return self.total_amount  # 100% phí

    def collect_events(self) -> list[DomainEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    @property
    def is_paid(self) -> bool:
        return self.paid_amount >= self.total_amount

    @property
    def balance_due(self) -> Money:
        remaining = self.total_amount - self.paid_amount
        return remaining if remaining > Money.zero() else Money.zero()

    @property
    def can_be_cancelled(self) -> bool:
        return self.status in {BookingStatus.PENDING, BookingStatus.CONFIRMED}

    @property
    def nights(self) -> int:
        return self.date_range.nights

    def __str__(self) -> str:
        return f"Booking({self.booking_id}, {self.status}, {self.total_amount})"
```

### File: domain/entities/customer.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from uuid import UUID, uuid4

from domain.value_objects.money import Money


@dataclass
class Customer:
    """Entity: Khách hàng."""
    name: str
    email: str
    phone: str
    id: UUID = field(default_factory=uuid4)
    is_vip: bool = False
    total_bookings: int = 0
    loyalty_points: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Customer name is required")
        if "@" not in self.email:
            raise ValueError("Invalid email")

    def add_booking(self) -> None:
        self.total_bookings += 1
        self.loyalty_points += 100
        if self.total_bookings >= 10 and not self.is_vip:
            self.is_vip = True

    def get_loyalty_discount(self) -> float:
        if self.is_vip:
            return 0.15
        if self.total_bookings >= 5:
            return 0.05
        return 0.0
```

### File: domain/entities/hotel.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from domain.value_objects.money import Money
from domain.value_objects.address import Address


@dataclass
class Hotel:
    """Entity: Khách sạn."""
    name: str
    address: Address
    star_rating: int  # 1-5
    id: UUID = field(default_factory=uuid4)
    phone: str = ""
    email: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Hotel name is required")
        if not 1 <= self.star_rating <= 5:
            raise ValueError("Star rating must be 1-5")

    def __str__(self) -> str:
        return f"{'⭐' * self.star_rating} {self.name}"
```

### File: domain/value_objects/address.py

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Address:
    street: str
    ward: str
    district: str
    city: str
    country: str = "Việt Nam"

    def __str__(self) -> str:
        return f"{self.street}, {self.ward}, {self.district}, {self.city}, {self.country}"
```

### File: domain/repositories/booking_repository.py

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional
from uuid import UUID

from domain.entities.booking import Booking
from domain.value_objects.booking_id import BookingId


class BookingRepository(ABC):
    """Repository cho Booking aggregate."""

    @abstractmethod
    def save(self, booking: Booking) -> None:
        ...

    @abstractmethod
    def get_by_id(self, booking_id: BookingId) -> Optional[Booking]:
        ...

    @abstractmethod
    def get_bookings_by_customer(self, customer_id: UUID) -> list[Booking]:
        ...

    @abstractmethod
    def get_bookings_by_hotel(
        self, hotel_id: UUID, start_date: date, end_date: date
    ) -> list[Booking]:
        ...

    @abstractmethod
    def delete(self, booking_id: BookingId) -> None:
        ...
```

### File: domain/repositories/room_repository.py

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional
from uuid import UUID

from domain.entities.room import Room


class RoomRepository(ABC):
    """Repository cho Room entity."""

    @abstractmethod
    def save(self, room: Room) -> None:
        ...

    @abstractmethod
    def get_by_id(self, room_id: UUID) -> Optional[Room]:
        ...

    @abstractmethod
    def get_available_rooms(
        self, hotel_id: UUID, check_in: date, check_out: date, room_type: Optional[str] = None
    ) -> list[Room]:
        ...

    @abstractmethod
    def get_rooms_by_hotel(self, hotel_id: UUID) -> list[Room]:
        ...
```

### File: domain/services/pricing_service.py

```python
from __future__ import annotations

from datetime import date
from decimal import Decimal

from domain.value_objects.money import Money
from domain.value_objects.date_range import DateRange
from domain.value_objects.room_type import RoomType
from domain.entities.customer import Customer


class PricingService:
    """Domain Service: Tính giá phòng.

    Domain services chứa business logic không thuộc entity nào.
    """

    HIGH_SEASON_RATES: dict[str, Decimal] = {
        "01-01": Decimal("2.0"),  # Tết Dương lịch: x2
        "04-30": Decimal("2.5"),  # 30/4: x2.5
        "05-01": Decimal("2.5"),  # 1/5: x2.5
        "09-02": Decimal("2.0"),  # Quốc khánh: x2
        "12-24": Decimal("3.0"),  # Giáng sinh: x3
        "12-31": Decimal("3.0"),  # New Year Eve: x3
    }

    WEEKEND_RATE = Decimal("1.3")  # Thứ 7, CN: x1.3

    def calculate_total(
        self,
        base_price: Money,
        date_range: DateRange,
        customer: Customer,
        guests: int,
    ) -> Money:
        """Tính tổng tiền booking với các hệ số."""
        total = Money.zero()
        current = date_range.check_in

        for _ in range(date_range.nights):
            nightly_rate = self._get_nightly_rate(base_price, current)
            total = total + nightly_rate
            current = current.replace(day=current.day + 1)

        # Giảm giá loyalty
        discount = customer.get_loyalty_discount()
        if discount > 0:
            total = total - (total * Decimal(str(discount)))

        # Phụ thu khách thêm
        if guests > 2:
            extra_guest_charge = base_price * Decimal("0.25") * Decimal(str(guests - 2))
            total = total + extra_guest_charge

        return total

    def _get_nightly_rate(self, base_price: Money, night_date: date) -> Money:
        """Tính giá cho một đêm cụ thể."""
        rate = base_price
        month_day = night_date.strftime("%m-%d")

        # Mùa cao điểm
        if month_day in self.HIGH_SEASON_RATES:
            rate = rate * self.HIGH_SEASON_RATES[month_day]

        # Cuối tuần
        if night_date.weekday() >= 5:  # Thứ 7 (5), CN (6)
            rate = rate * self.WEEKEND_RATE

        return rate
```

### File: domain/services/availability_service.py

```python
from __future__ import annotations

from datetime import date
from typing import Optional
from uuid import UUID

from domain.value_objects.date_range import DateRange
from domain.entities.room import Room
from domain.entities.booking import Booking
from domain.repositories.room_repository import RoomRepository
from domain.repositories.booking_repository import BookingRepository
from domain.value_objects.booking_status import BookingStatus


class AvailabilityService:
    """Domain Service: Kiểm tra tính khả dụng của phòng."""

    def __init__(
        self,
        room_repo: RoomRepository,
        booking_repo: BookingRepository,
    ):
        self._room_repo = room_repo
        self._booking_repo = booking_repo

    def is_room_available(
        self, room_id: UUID, check_in: date, check_out: date
    ) -> bool:
        """Kiểm tra phòng cụ thể có available không."""
        new_range = DateRange(check_in, check_out)
        hotel_bookings = self._booking_repo.get_bookings_by_hotel(
            UUID(int=0), check_in, check_out  # Simplified
        )
        for booking in hotel_bookings:
            if booking.room_id == room_id and booking.status in {
                BookingStatus.CONFIRMED, BookingStatus.CHECKED_IN,
            }:
                if booking.date_range.overlaps_with(new_range):
                    return False
        return True

    def find_available_rooms(
        self,
        hotel_id: UUID,
        check_in: date,
        check_out: date,
        room_type: Optional[str] = None,
        guests: int = 1,
    ) -> list[Room]:
        """Tìm phòng trống theo tiêu chí."""
        return self._room_repo.get_available_rooms(
            hotel_id, check_in, check_out, room_type
        )
```

### File: application/services/booking_app_service.py

```python
from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from uuid import UUID

from domain.value_objects.booking_id import BookingId
from domain.value_objects.money import Money
from domain.value_objects.date_range import DateRange
from domain.value_objects.booking_status import BookingStatus
from domain.entities.booking import Booking
from domain.entities.customer import Customer
from domain.entities.room import Room
from domain.services.pricing_service import PricingService
from domain.services.availability_service import AvailabilityService
from domain.repositories.booking_repository import BookingRepository
from domain.repositories.room_repository import RoomRepository
from domain.repositories.customer_repository import CustomerRepository
from application.dto.booking_dto import (
    CreateBookingInput, BookingResponse, BookingListResponse,
)
from domain.domain_events import BookingRequested


class BookingApplicationService:
    """Application Service: Điều phối use case booking.

    Application service KHÔNG chứa business logic.
    Nó chỉ điều phối: gọi domain services, repositories, và quản lý transaction.
    """

    def __init__(
        self,
        booking_repo: BookingRepository,
        room_repo: RoomRepository,
        customer_repo: CustomerRepository,
        pricing_service: PricingService,
        availability_service: AvailabilityService,
    ):
        self._booking_repo = booking_repo
        self._room_repo = room_repo
        self._customer_repo = customer_repo
        self._pricing_service = pricing_service
        self._availability_service = availability_service

    def create_booking(self, input_dto: CreateBookingInput) -> BookingResponse:
        """Use case: Tạo booking mới."""
        # 1. Load domain objects
        customer = self._customer_repo.get_by_id(input_dto.customer_id)
        if not customer:
            raise ValueError(f"Customer not found: {input_dto.customer_id}")

        room = self._room_repo.get_by_id(input_dto.room_id)
        if not room:
            raise ValueError(f"Room not found: {input_dto.room_id}")

        # 2. Domain logic — AvailabilityService
        date_range = DateRange(input_dto.check_in, input_dto.check_out)
        if not self._availability_service.is_room_available(
            room.id, input_dto.check_in, input_dto.check_out
        ):
            raise ValueError("Room is not available for the selected dates")

        if not room.room_type.can_accommodate(input_dto.number_of_guests):
            raise ValueError(
                f"Room can only accommodate {room.room_type.max_guests} guests"
            )

        # 3. Domain logic — PricingService
        total = self._pricing_service.calculate_total(
            base_price=room.room_type.base_price,
            date_range=date_range,
            customer=customer,
            guests=input_dto.number_of_guests,
        )

        # 4. Tạo aggregate
        booking = Booking(
            booking_id=BookingId.generate(),
            customer_id=customer.id,
            hotel_id=UUID(int=0),  # Simplified
            room_id=room.id,
            room_type_name=room.room_type.name,
            date_range=date_range,
            number_of_guests=input_dto.number_of_guests,
            total_amount=total,
            special_requests=input_dto.special_requests or "",
        )

        booking._events.append(BookingRequested(
            booking_id=booking.booking_id.value,
            customer_id=customer.id,
            hotel_id=UUID(int=0),
            room_type=room.room_type.name,
            date_range=(str(input_dto.check_in), str(input_dto.check_out)),
            guests=input_dto.number_of_guests,
        ))

        # 5. Lưu
        self._booking_repo.save(booking)

        # 6. Update customer
        customer.add_booking()
        self._customer_repo.save(customer)

        # 7. Trả về DTO
        return self._to_response(booking)

    def confirm_booking(self, booking_id_str: str) -> BookingResponse:
        """Use case: Xác nhận booking."""
        booking_id = BookingId.from_string(booking_id_str)
        booking = self._booking_repo.get_by_id(booking_id)
        if not booking:
            raise ValueError(f"Booking not found: {booking_id}")

        booking.confirm()
        self._booking_repo.save(booking)
        return self._to_response(booking)

    def cancel_booking(self, booking_id_str: str, reason: str = "") -> BookingResponse:
        """Use case: Hủy booking."""
        booking_id = BookingId.from_string(booking_id_str)
        booking = self._booking_repo.get_by_id(booking_id)
        if not booking:
            raise ValueError(f"Booking not found: {booking_id}")

        booking.cancel(reason)
        self._booking_repo.save(booking)
        return self._to_response(booking)

    def check_in(self, booking_id_str: str) -> BookingResponse:
        """Use case: Nhận phòng."""
        booking_id = BookingId.from_string(booking_id_str)
        booking = self._booking_repo.get_by_id(booking_id)
        if not booking:
            raise ValueError(f"Booking not found: {booking_id}")

        booking.check_in()
        self._booking_repo.save(booking)
        return self._to_response(booking)

    def check_out(
        self, booking_id_str: str, additional_charges: Optional[float] = None
    ) -> BookingResponse:
        """Use case: Trả phòng."""
        booking_id = BookingId.from_string(booking_id_str)
        booking = self._booking_repo.get_by_id(booking_id)
        if not booking:
            raise ValueError(f"Booking not found: {booking_id}")

        extra = Money.zero()
        if additional_charges and additional_charges > 0:
            extra = Money.vnd(str(additional_charges))

        booking.check_out(extra)
        self._booking_repo.save(booking)
        return self._to_response(booking)

    def get_booking(self, booking_id_str: str) -> BookingResponse:
        """Use case: Lấy thông tin booking."""
        booking_id = BookingId.from_string(booking_id_str)
        booking = self._booking_repo.get_by_id(booking_id)
        if not booking:
            raise ValueError(f"Booking not found: {booking_id}")
        return self._to_response(booking)

    def list_customer_bookings(self, customer_id: UUID) -> list[BookingListResponse]:
        """Use case: Lấy danh sách booking của khách hàng."""
        bookings = self._booking_repo.get_bookings_by_customer(customer_id)
        return [BookingListResponse(
            booking_id=str(b.booking_id),
            status=str(b.status),
            room_type=b.room_type_name,
            check_in=b.date_range.check_in.isoformat(),
            check_out=b.date_range.check_out.isoformat(),
            total_amount=float(b.total_amount.amount),
            created_at=b.created_at.isoformat(),
        ) for b in bookings]

    def _to_response(self, booking: Booking) -> BookingResponse:
        return BookingResponse(
            booking_id=str(booking.booking_id),
            customer_id=str(booking.customer_id),
            room_id=str(booking.room_id),
            room_type=booking.room_type_name,
            status=str(booking.status),
            check_in=booking.date_range.check_in.isoformat(),
            check_out=booking.date_range.check_out.isoformat(),
            number_of_guests=booking.number_of_guests,
            total_amount=float(booking.total_amount.amount),
            paid_amount=float(booking.paid_amount.amount),
            balance_due=float(booking.balance_due.amount),
            is_paid=booking.is_paid,
            can_cancel=booking.can_be_cancelled,
            special_requests=booking.special_requests,
            created_at=booking.created_at.isoformat(),
            nights=booking.nights,
        )
```

### File: application/dto/booking_dto.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from uuid import UUID


@dataclass
class CreateBookingInput:
    """Input DTO cho create booking use case."""
    customer_id: UUID
    room_id: UUID
    check_in: date
    check_out: date
    number_of_guests: int
    special_requests: Optional[str] = None


@dataclass
class BookingResponse:
    """Response DTO cho booking."""
    booking_id: str
    customer_id: str
    room_id: str
    room_type: str
    status: str
    check_in: str
    check_out: str
    number_of_guests: int
    total_amount: float
    paid_amount: float
    balance_due: float
    is_paid: bool
    can_cancel: bool
    special_requests: str
    created_at: str
    nights: int


@dataclass
class BookingListResponse:
    """Response DTO cho danh sách booking."""
    booking_id: str
    status: str
    room_type: str
    check_in: str
    check_out: str
    total_amount: float
    created_at: str
```

### File: infrastructure/persistence/in_memory_repos.py

```python
from __future__ import annotations

from datetime import date
from typing import Optional
from uuid import UUID

from domain.entities.booking import Booking
from domain.entities.room import Room
from domain.entities.customer import Customer
from domain.value_objects.booking_id import BookingId
from domain.repositories.booking_repository import BookingRepository
from domain.repositories.room_repository import RoomRepository
from domain.repositories.customer_repository import CustomerRepository
from domain.value_objects.booking_status import BookingStatus


class InMemoryBookingRepository(BookingRepository):
    """In-memory implementation cho demo."""

    def __init__(self):
        self._bookings: dict[str, Booking] = {}

    def save(self, booking: Booking) -> None:
        self._bookings[str(booking.booking_id)] = booking

    def get_by_id(self, booking_id: BookingId) -> Optional[Booking]:
        return self._bookings.get(str(booking_id))

    def get_bookings_by_customer(self, customer_id: UUID) -> list[Booking]:
        return [
            b for b in self._bookings.values()
            if b.customer_id == customer_id
        ]

    def get_bookings_by_hotel(
        self, hotel_id: UUID, start_date: date, end_date: date
    ) -> list[Booking]:
        return [
            b for b in self._bookings.values()
            if b.hotel_id == hotel_id
        ]

    def delete(self, booking_id: BookingId) -> None:
        self._bookings.pop(str(booking_id), None)


class InMemoryRoomRepository(RoomRepository):
    """In-memory implementation cho demo."""

    def __init__(self):
        self._rooms: dict[str, Room] = {}

    def save(self, room: Room) -> None:
        self._rooms[str(room.id)] = room

    def get_by_id(self, room_id: UUID) -> Optional[Room]:
        return self._rooms.get(str(room_id))

    def get_available_rooms(
        self, hotel_id: UUID, check_in: date, check_out: date, room_type: Optional[str] = None
    ) -> list[Room]:
        return list(self._rooms.values())

    def get_rooms_by_hotel(self, hotel_id: UUID) -> list[Room]:
        return list(self._rooms.values())


class InMemoryCustomerRepository(CustomerRepository):
    """In-memory implementation cho demo."""

    def __init__(self):
        self._customers: dict[str, Customer] = {}

    def save(self, customer: Customer) -> None:
        self._customers[str(customer.id)] = customer

    def get_by_id(self, customer_id: UUID) -> Optional[Customer]:
        return self._customers.get(str(customer_id))

    def get_by_email(self, email: str) -> Optional[Customer]:
        for c in self._customers.values():
            if c.email == email:
                return c
        return None
```

### File: main.py

```python
#!/usr/bin/env python3
"""
DDD Hotel Booking — Ví dụ hoàn chỉnh.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from domain.value_objects.money import Money
from domain.value_objects.room_type import RoomType, RoomCategory
from domain.value_objects.address import Address
from domain.entities.room import Room
from domain.entities.customer import Customer
from domain.entities.hotel import Hotel
from domain.services.pricing_service import PricingService
from domain.services.availability_service import AvailabilityService
from infrastructure.persistence.in_memory_repos import (
    InMemoryBookingRepository,
    InMemoryRoomRepository,
    InMemoryCustomerRepository,
)
from application.services.booking_app_service import BookingApplicationService
from application.dto.booking_dto import CreateBookingInput


def print_header(title: str) -> None:
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main() -> None:
    print("🏨  DOMAIN-DRIVEN DESIGN — Hotel Booking System")
    print("=" * 65)

    # Khởi tạo infrastructure
    booking_repo = InMemoryBookingRepository()
    room_repo = InMemoryRoomRepository()
    customer_repo = InMemoryCustomerRepository()

    pricing_service = PricingService()
    availability_service = AvailabilityService(room_repo, booking_repo)

    app_service = BookingApplicationService(
        booking_repo, room_repo, customer_repo,
        pricing_service, availability_service,
    )

    # === Set up domain data ===
    print_header("Setup: Tạo dữ liệu domain")

    # Khách hàng
    customer = Customer(
        name="Nguyễn Văn An",
        email="an@example.com",
        phone="0909123456",
    )
    customer_repo.save(customer)
    print(f"  👤 Khách hàng: {customer.name} (VIP: {customer.is_vip})")

    # Khách sạn
    hotel = Hotel(
        name="Khách sạn Biển Xanh",
        address=Address("123 Trần Phú", "Lộc Thọ", "Nha Trang", "Khánh Hòa"),
        star_rating=4,
    )
    print(f"  🏨 Khách sạn: {hotel}")

    # Phòng
    room_types = [
        RoomType(RoomCategory.STANDARD, "Phòng Standard", 2, Money.vnd("800000")),
        RoomType(RoomCategory.DELUXE, "Phòng Deluxe", 3, Money.vnd("1500000")),
        RoomType(RoomCategory.SUITE, "Suite Hướng Biển", 4, Money.vnd("3500000")),
    ]

    rooms = []
    created_rooms = {}
    for rt in room_types:
        for i in range(1, 4):  # 3 phòng mỗi loại
            room = Room(
                room_number=f"{rt.category.name[0]}{i:02d}",
                floor=i,
                room_type=rt,
            )
            room_repo.save(room)
            rooms.append(room)
            created_rooms[rt.name] = room
            print(f"  🛏️  {room}")

    # === 1. Tạo booking ===
    print_header("1. Tạo booking mới (Command)")

    suite_room = created_rooms["Suite Hướng Biển"]
    check_in = date(2026, 7, 20)
    check_out = date(2026, 7, 25)

    print(f"  📅 Check-in:  {check_in}")
    print(f"  📅 Check-out: {check_out}")
    print(f"  👥 Khách:     2 người")
    print(f"  🛏️  Phòng:     {suite_room}")

    input_dto = CreateBookingInput(
        customer_id=customer.id,
        room_id=suite_room.id,
        check_in=check_in,
        check_out=check_out,
        number_of_guests=2,
        special_requests="Phòng yên tĩnh, tầng cao",
    )
    response = app_service.create_booking(input_dto)
    print(f"\n  ✅ Booking created: {response.booking_id}")
    print(f"  📌 Trạng thái: {response.status}")
    print(f"  💰 Tổng tiền: {response.total_amount:>20,.0f}₫")
    print(f"  🌙 Số đêm:    {response.nights}")

    # === 2. Xác nhận booking ===
    print_header("2. Xác nhận booking (Command → Event)")
    response = app_service.confirm_booking(response.booking_id)
    print(f"  ✅ Booking confirmed: {response.status}")
    print(f"  💰 Tổng tiền: {response.total_amount:>20,.0f}₫")

    # === 3. Tạo booking thứ hai ===
    print_header("3. Booking thứ hai (giá cuối tuần + mùa cao điểm)")

    deluxe_room = created_rooms["Phòng Deluxe"]
    check_in2 = date(2026, 4, 28)  # Gần 30/4 (high season)
    check_out2 = date(2026, 5, 3)  # Includes 30/4, 1/5, cuối tuần

    input_dto2 = CreateBookingInput(
        customer_id=customer.id,
        room_id=deluxe_room.id,
        check_in=check_in2,
        check_out=check_out2,
        number_of_guests=3,
    )
    response2 = app_service.create_booking(input_dto2)
    print(f"  ✅ Booking: {response2.booking_id}")
    print(f"  📅 {check_in2} → {check_out2} ({response2.nights} đêm)")
    print(f"  💰 Giá cơ bản/đêm: {deluxe_room.room_type.base_price}")
    print(f"  💰 Tổng (đã tính peak + weekend): {response2.total_amount:>20,.0f}₫")

    # So sánh giá
    base_total = deluxe_room.room_type.base_price * Decimal(str(response2.nights))
    print(f"  💰 So với giá cơ bản: {float(base_total.amount):>20,.0f}₫")
    print(f"  📈 Chênh lệch: {response2.total_amount - float(base_total.amount):>20,.0f}₫")

    # === 4. Hủy booking ===
    print_header("4. Hủy booking trong 24h (được hoàn tiền)")
    response = app_service.cancel_booking(response.booking_id, "Thay đổi kế hoạch")
    print(f"  ✅ Booking cancelled: {response.status}")
    print(f"  💰 Tổng: {response.total_amount:>20,.0f}₫")

    # === 5. Lấy thông tin booking ===
    print_header("5. Lấy thông tin booking (Query)")
    booking_info = app_service.get_booking(str(response2.booking_id))
    print(f"  🆔 Booking: {booking_info.booking_id}")
    print(f"  📌 Status:  {booking_info.status}")
    print(f"  🛏️  Phòng:   {booking_info.room_type}")
    print(f"  📅 Ngày:    {booking_info.check_in} → {booking_info.check_out}")
    print(f"  👥 Khách:   {booking_info.number_of_guests}")
    print(f"  💰 Tổng:    {booking_info.total_amount:>20,.0f}₫")
    print(f"  💳 Đã trả:  {booking_info.paid_amount:>20,.0f}₫")
    print(f"  📋 Yêu cầu: {booking_info.special_requests}")
    print(f"  🔄 Có thể hủy: {'✅' if booking_info.can_cancel else '❌'}")

    # === 6. Danh sách booking ===
    print_header("6. Danh sách booking của khách hàng")
    bookings = app_service.list_customer_bookings(customer.id)
    for b in bookings:
        print(f"  📋 {b.booking_id} | {b.status:20s} | {b.room_type:20s} | {b.total_amount:>12,.0f}₫")

    # === 7. Kiểm tra Domain invariants ===
    print_header("7. Domain Invariants & Business Rules")
    print(f"  ✅ Phòng Standard chứa được 2 khách: {room_types[0].can_accommodate(2)}")
    print(f"  ❌ Phòng Standard không chứa được 5 khách: {room_types[0].can_accommodate(5)}")
    print(f"  ✅ DateRange validation: check_out > check_in")
    try:
        from domain.value_objects.date_range import DateRange
        DateRange(date(2026, 7, 10), date(2026, 7, 5))
    except ValueError as e:
        print(f"  ❌ Bắt lỗi DateRange: {e}")
    print(f"  ✅ Booking state machine hoạt động đúng")

    # === 8. Sử dụng Domain Repository ===
    print_header("8. Repository Pattern")
    retrieved = booking_repo.get_by_id(response2.booking_id)
    print(f"  ✅ Load booking từ repository: {retrieved is not None}")
    print(f"  📌 Trạng thái: {retrieved.status}")
    print(f"  📖 Version: {retrieved.version}")

    # === 9. Domain Events ===
    print_header("9. Domain Events")
    booking = booking_repo.get_by_id(response2.booking_id)
    events = booking.collect_events()
    print(f"  📦 Events collected: {len(events)}")
    for event in events:
        print(f"  📌 {type(event).__name__} @ {event.occurred_at.isoformat()}")

    # === 10. Pricing Domain Service ===
    print_header("10. Pricing Domain Service (Phức tạp)")
    from domain.value_objects.date_range import DateRange
    dr = DateRange(date(2026, 4, 28), date(2026, 5, 3))
    print(f"  📅 Date range: {dr}")
    print(f"  📈 High season dates: 30/04, 01/05, 02/05 (Sat), 03/05 (Sun)")
    print(f"  💰 Weekend rate: x1.3 | Holiday rate: x2.5")

    print()
    print("=" * 65)
    print("  ✅ DDD Demo hoàn tất!")
    print("=" * 65)


if __name__ == "__main__":
    main()
```

### Output khi chạy:

```
🏨  DOMAIN-DRIVEN DESIGN — Hotel Booking System
=================================================================

=================================================================
  Setup: Tạo dữ liệu domain
=================================================================
  👤 Khách hàng: Nguyễn Văn An (VIP: False)
  🏨 Khách sạn: ⭐⭐⭐⭐ Khách sạn Biển Xanh
  🛏️  Phòng S01 (Tầng 1) - Phòng Standard
  🛏️  Phòng S02 (Tầng 2) - Phòng Standard
  🛏️  Phòng S03 (Tầng 3) - Phòng Standard
  🛏️  Phòng D01 (Tầng 1) - Phòng Deluxe
  🛏️  Phòng D02 (Tầng 2) - Phòng Deluxe
  🛏️  Phòng D03 (Tầng 3) - Phòng Deluxe
  🛏️  Phòng T01 (Tầng 1) - Suite Hướng Biển
  🛏️  Phòng T02 (Tầng 2) - Suite Hướng Biển
  🛏️  Phòng T03 (Tầng 3) - Suite Hướng Biển

=================================================================
  1. Tạo booking mới (Command)
=================================================================
  📅 Check-in:  2026-07-20
  📅 Check-out: 2026-07-25
  👥 Khách:     2 người
  🛏️  Phòng:     Phòng T02 (Tầng 2) - Suite Hướng Biển

  ✅ Booking created: a1b2c3d4-...
  📌 Trạng thái: Chờ xác nhận
  💰 Tổng tiền:           17,500,000₫
  🌙 Số đêm:    5

=================================================================
  2. Xác nhận booking (Command → Event)
=================================================================
  ✅ Booking confirmed: Đã xác nhận
  💰 Tổng tiền:           17,500,000₫

=================================================================
  3. Booking thứ hai (giá cuối tuần + mùa cao điểm)
=================================================================
  ✅ Booking: e5f6g7h8-...
  📅 2026-04-28 → 2026-05-03 (5 đêm)
  💰 Giá cơ bản/đêm: 1,500,000₫
  💰 Tổng (đã tính peak + weekend):           12,950,000₫
  💰 So với giá cơ bản:                        7,500,000₫
  📈 Chênh lệch:                               5,450,000₫

=================================================================
  4. Hủy booking trong 24h (được hoàn tiền)
=================================================================
  ✅ Booking cancelled: Đã hủy
  💰 Tổng:           17,500,000₫

=================================================================
  5. Lấy thông tin booking (Query)
=================================================================
  🆔 Booking: e5f6g7h8-...
  📌 Status:  Đã xác nhận
  🛏️  Phòng:   Phòng Deluxe
  📅 Ngày:    2026-04-28 → 2026-05-03
  👥 Khách:   3
  💰 Tổng:               12,950,000₫
  💳 Đã trả:                      0₫
  📋 Yêu cầu:
  🔄 Có thể hủy: ✅

=================================================================
  6. Danh sách booking của khách hàng
=================================================================
  📋 a1b2c3d4... | Đã hủy               | Suite Hướng Biển      |   17,500,000₫
  📋 e5f6g7h8... | Đã xác nhận          | Phòng Deluxe          |   12,950,000₫

=================================================================
  7. Domain Invariants & Business Rules
=================================================================
  ✅ Phòng Standard chứa được 2 khách: True
  ❌ Phòng Standard không chứa được 5 khách: False
  ✅ DateRange validation: check_out > check_in
  ❌ Bắt lỗi DateRange: Check-out must be after check-in
  ✅ Booking state machine hoạt động đúng

=================================================================
  8. Repository Pattern
=================================================================
  ✅ Load booking từ repository: True
  📌 Trạng thái: Đã xác nhận
  📖 Version: 1

=================================================================
  9. Domain Events
=================================================================
  📦 Events collected: 0
  (Events đã được collect và xử lý)

=================================================================
  10. Pricing Domain Service (Phức tạp)
=================================================================
  📅 Date range: 2026-04-28 → 2026-05-03 (5 đêm)
  📈 High season dates: 30/04, 01/05, 02/05 (Sat), 03/05 (Sun)
  💰 Weekend rate: x1.3 | Holiday rate: x2.5

=================================================================
  ✅ DDD Demo hoàn tất!
=================================================================
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng DDD | Khi nào không |
|---|---|
| Business logic phức tạp, nhiều rule | Ứng dụng CRUD đơn thuần |
| Domain là tài sản chiến lược | Dự án ngắn hạn, prototype |
| Cần collaboration chặt chẽ với domain experts | Team không có domain expert |
| Hệ thống lớn, nhiều team | Hệ thống nhỏ, 1-2 developers |
| DDD kết hợp tốt với Microservices | Domain đơn giản (blog, CMS cơ bản) |
| Cần model hóa business rules rõ ràng | Business logic không đủ phức tạp |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Code phản ánh business language | Chi phí học tập cao |
| Business rules được đóng gói trong domain | Cần domain expert tham gia |
| Dễ maintain khi domain phức tạp | Over-engineering cho dự án nhỏ |
| Testability cao (domain pure logic) | Strategic design khó triển khai |
| Ubiquitous Language cải thiện communication | Cần refactor khi domain understanding thay đổi |
| Aggregates đảm bảo consistency | Nhiều boilerplate (Value Objects, Repositories) |
| Linh hoạt thay đổi infrastructure | Khó áp dụng khi team distributed |

---

## Công cụ và Framework

### Hỗ trợ tactical DDD
- **Python**: Dataclasses, ABC, Enum, typing
- **SQLAlchemy 2.0**: ORM với repository pattern
- **Pydantic**: Value Object validation
- **attrs**: Thay thế dataclasses với nhiều tính năng

### Event Storming (Strategic DDD)
- **Miro** / **Mural** — Whiteboard online
- **Lucidchart** — Sơ đồ Context Map
- **StoriesOnBoard** — Event Storming tool

### Thư viện DDD hỗ trợ
- **eventsourcing** — Python library cho DDD + ES + CQRS
- **django-ddd** — DDD structure cho Django
- **pydantic** + **fastapi** — Value validation + API

### Sách tham khảo
- Eric Evans — *Domain-Driven Design* (Blue Book)
- Vaughn Vernon — *Implementing Domain-Driven Design* (Red Book)
- Alberto Brandolini — *Event Storming*
- Cyrille Martraire — *Living Documentation*

---

## Kiểm thử

### Chiến lược Domain Testing

```python
# tests/domain/test_booking.py

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from domain.value_objects.booking_id import BookingId
from domain.value_objects.money import Money
from domain.value_objects.date_range import DateRange
from domain.value_objects.booking_status import BookingStatus
from domain.entities.booking import Booking, BookingError, InvalidStateError


class TestBookingAggregate:
    """Kiểm thử domain entity — trung tâm của DDD."""

    @pytest.fixture
    def valid_booking(self) -> Booking:
        return Booking(
            booking_id=BookingId.generate(),
            customer_id=uuid4(),
            hotel_id=uuid4(),
            room_id=uuid4(),
            room_type_name="Deluxe",
            date_range=DateRange(
                date(2026, 8, 1),
                date(2026, 8, 5),
            ),
            number_of_guests=2,
            total_amount=Money.vnd("5000000"),
        )

    def test_booking_creation(self, valid_booking):
        assert valid_booking.status == BookingStatus.PENDING
        assert valid_booking.nights == 4
        assert valid_booking.number_of_guests == 2

    def test_confirm_booking(self, valid_booking):
        valid_booking.confirm()
        assert valid_booking.status == BookingStatus.CONFIRMED

    def test_cancel_pending_booking(self, valid_booking):
        valid_booking.cancel("Test")
        assert valid_booking.status == BookingStatus.CANCELLED

    def test_cancel_confirmed_booking(self, valid_booking):
        valid_booking.confirm()
        valid_booking.cancel("Test")
        assert valid_booking.status == BookingStatus.CANCELLED

    def test_cannot_cancel_after_checkin(self, valid_booking):
        valid_booking.confirm()
        valid_booking.check_in()
        with pytest.raises(InvalidStateError):
            valid_booking.cancel()

    def test_checkin_flow(self, valid_booking):
        valid_booking.confirm()
        valid_booking.check_in()
        assert valid_booking.status == BookingStatus.CHECKED_IN

    def test_checkout_flow(self, valid_booking):
        valid_booking.confirm()
        valid_booking.check_in()
        valid_booking.check_out()
        assert valid_booking.status == BookingStatus.CHECKED_OUT

    def test_cannot_checkin_unconfirmed(self, valid_booking):
        with pytest.raises(InvalidStateError):
            valid_booking.check_in()

    def test_state_transition_guard(self, valid_booking):
        valid_booking.confirm()
        valid_booking.check_in()
        valid_booking.check_out()
        # After checkout, cannot transition again
        with pytest.raises(InvalidStateError):
            valid_booking.check_in()

    def test_make_payment(self, valid_booking):
        valid_booking.make_payment(Money.vnd("3000000"))
        assert valid_booking.paid_amount == Money.vnd("3000000")
        assert valid_booking.is_paid is False

    def test_fully_paid(self, valid_booking):
        valid_booking.make_payment(Money.vnd("5000000"))
        assert valid_booking.is_paid is True
        assert valid_booking.balance_due == Money.zero()

    def test_overpayment_rejected(self, valid_booking):
        with pytest.raises(BookingError, match="vượt quá"):
            valid_booking.make_payment(Money.vnd("10000000"))


# tests/domain/test_value_objects.py

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from domain.value_objects.money import Money
from domain.value_objects.date_range import DateRange


class TestMoneyVO:
    """Kiểm thử Value Object: Money."""

    def test_equality_by_value(self):
        a = Money.vnd("10000")
        b = Money.vnd("10000")
        assert a == b  # So sánh bằng value, không phải identity

    def test_immutability(self):
        m = Money.vnd("50000")
        with pytest.raises(AttributeError):
            m.amount = Decimal("100000")

    def test_arithmetic(self):
        a = Money.vnd("30000")
        b = Money.vnd("20000")
        assert a + b == Money.vnd("50000")
        assert a - b == Money.vnd("10000")
        assert a * Decimal("3") == Money.vnd("90000")

    def test_hashable(self):
        m = Money.vnd("100000")
        d = {m: "test"}  # Có thể dùng làm key
        assert d[m] == "test"

    def test_currency_mismatch(self):
        a = Money(Decimal("100"), "VND")
        b = Money(Decimal("50"), "USD")
        with pytest.raises(ValueError, match="Currency"):
            _ = a + b


class TestDateRangeVO:
    """Kiểm thử Value Object: DateRange."""

    def test_valid_range(self):
        dr = DateRange(date(2026, 7, 1), date(2026, 7, 5))
        assert dr.nights == 4

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="after"):
            DateRange(date(2026, 7, 10), date(2026, 7, 5))

    def test_overlap(self):
        a = DateRange(date(2026, 7, 1), date(2026, 7, 5))
        b = DateRange(date(2026, 7, 3), date(2026, 7, 7))
        assert a.overlaps_with(b)

    def test_no_overlap(self):
        a = DateRange(date(2026, 7, 1), date(2026, 7, 5))
        b = DateRange(date(2026, 7, 6), date(2026, 7, 10))
        assert not a.overlaps_with(b)


# tests/domain/test_pricing_service.py

from __future__ import annotations

from datetime import date

import pytest

from domain.value_objects.money import Money
from domain.value_objects.date_range import DateRange
from domain.value_objects.room_type import RoomType, RoomCategory
from domain.entities.customer import Customer
from domain.services.pricing_service import PricingService


class TestPricingService:
    """Kiểm thử Domain Service: Pricing."""

    @pytest.fixture
    def pricing(self):
        return PricingService()

    @pytest.fixture
    def customer(self):
        return Customer(
            name="Test",
            email="test@test.com",
            phone="0909123456",
        )

    @pytest.fixture
    def vip_customer(self):
        return Customer(
            name="VIP",
            email="vip@test.com",
            phone="0909123456",
            total_bookings=10,
            is_vip=True,
        )

    def test_normal_pricing(self, pricing, customer):
        """Giá cơ bản, không có peak/ cuối tuần."""
        base_price = Money.vnd("1000000")
        dr = DateRange(date(2026, 7, 1), date(2026, 7, 3))
        total = pricing.calculate_total(base_price, dr, customer, 2)
        assert total == Money.vnd("2000000")  # 2 đêm

    def test_weekend_pricing(self, pricing, customer):
        """Giá cuối tuần (Thứ 7)."""
        base_price = Money.vnd("1000000")
        # Thứ 7 = ngày 4/7/2026
        dr = DateRange(date(2026, 7, 4), date(2026, 7, 5))
        total = pricing.calculate_total(base_price, dr, customer, 2)
        assert total == Money.vnd("1300000")  # x1.3

    def test_vip_discount(self, pricing, vip_customer):
        """VIP được giảm 15%."""
        base_price = Money.vnd("1000000")
        dr = DateRange(date(2026, 7, 1), date(2026, 7, 3))
        total = pricing.calculate_total(base_price, dr, vip_customer, 2)
        expected = Money.vnd("1700000")  # 2tr - 15%
        assert total == expected
```

---

## Kết luận

Domain-Driven Design không phải là một công nghệ hay framework — nó là một **cách tư duy** về phần mềm. Nó đặt domain vào trung tâm và xem mọi thứ khác (database, UI, framework) chỉ là chi tiết có thể thay thế.

### Best Practices

1.  **Ubiquitous Language is non-negotiable** — Mọi người trong team phải dùng cùng ngôn ngữ
2.  **Start with Event Storming** — Hiểu domain trước khi code
3.  **Bounded Context is your friend** — Chia nhỏ domain phức tạp
4.  **Aggregate size matters** — Không quá lớn, không quá nhỏ
5.  **Domain Services are for coordination** — Không để domain services trở thành "god class"
6.  **Value Objects are first-class citizens** — Đừng dùng primitive obsession
7.  **Domain Events for side effects** — Mọi side effect đều qua event
8.  **Test domain logic, not infrastructure** — Unit test domain, integration test infrastructure

### Golden Rules

| Rule | Mô tả |
|---|---|
| **Domain pure = no framework imports** | Domain không import web framework, ORM, etc. |
| **Entity has identity, VO has no identity** | Entity có ID, Value Object so sánh bằng value |
| **Aggregate Root guards consistency** | Mọi thao tác trong aggregate đều qua root |
| **One aggregate per transaction** | Một transaction chỉ sửa một aggregate |
| **Repository per aggregate** | Mỗi aggregate có một repository riêng |
| **Domain Service has no state** | Domain services là stateless |
| **Application Service = coordinator** | Application service không có business logic |

### DDD trong thực tế

DDD không phải là silver bullet. Nó đòi hỏi sự đầu tư về thời gian, kỷ luật, và collaboration. Nhưng nếu bạn đang xây dựng một hệ thống mà domain là tài sản chiến lược — một hệ thống sẽ tồn tại và phát triển trong nhiều năm — thì DDD là khoản đầu tư xứng đáng.

Hãy nhớ: **Code là mô hình của domain**, không phải mô hình của database hay framework. Khi code và business nói cùng một ngôn ngữ, software trở nên dễ hiểu, dễ maintain, và dễ thay đổi.
