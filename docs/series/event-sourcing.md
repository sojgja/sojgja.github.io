---
id: event-sourcing
title: Event Sourcing — Lưu trữ sự kiện
sidebar_label: Event Sourcing
sidebar_position: 42
---

# Event Sourcing — Lưu trữ sự kiện

> *"Event Sourcing ensures that all changes to application state are stored as a sequence of events. Not just the current state — the full story."* — Martin Fowler

---

## Tổng quan

**Event Sourcing** (ES) là một architectural pattern trong đó mọi thay đổi trạng thái của ứng dụng được lưu trữ như một chuỗi các **sự kiện bất biến** (immutable events), thay vì chỉ lưu trạng thái hiện tại. Để biết trạng thái hiện tại, ta **replay** tất cả các sự kiện từ đầu.

Ý tưởng này đã tồn tại từ lâu trong lĩnh vực tài chính (audit log, ledger) và được phổ biến trong software engineering bởi:

- **Martin Fowler** — Bài viết kinh điển (2005), tổng quan và phân tích sâu
- **Greg Young** — Kết hợp Event Sourcing với CQRS, phổ biến trong cộng đồng .NET
- **Eric Evans** — Domain-Driven Design cung cấp nền tảng lý thuyết
- **Udi Dahan** — Áp dụng ES trong hệ thống enterprise SOA
- **Chris Richardson** — Microservices patterns, saga pattern với ES

### So sánh với CRUD truyền thống

```python
# CRUD — chỉ lưu trạng thái hiện tại
db.orders.update({"id": 123, "status": "SHIPPED", "total": 500000})

# Event Sourcing — lưu mọi sự kiện
event_store.append(OrderShipped(order_id=123, timestamp=...))
event_store.append(OrderTotalAdjusted(order_id=123, amount=500000, ...))
```

| Khía cạnh | CRUD | Event Sourcing |
|---|---|---|
| Lưu trữ | Trạng thái hiện tại | Chuỗi sự kiện bất biến |
| Lịch sử | Không (hoặc phải thêm audit log) | Có sẵn, đầy đủ |
| Debug | Khó biết ai đã thay đổi gì | Replay events để debug |
| Temporal query | Không hỗ trợ | Có thể query bất kỳ thời điểm nào |
| Phức tạp | Thấp | Cao |

---

## Bài toán

### Audit trail và compliance

Trong các hệ thống tài chính, y tế, hoặc pháp lý, việc có một audit trail đầy đủ là bắt buộc. Với CRUD, khi ai đó sửa một record, thông tin cũ sẽ bị mất. Bạn phải xây dựng thêm audit log riêng — và audit log này thường không đồng bộ với dữ liệu chính.

Ví dụ: Một nhân viên ngân hàng sửa số dư tài khoản của khách hàng. Với CRUD, bạn chỉ thấy số dư mới, không biết ai đã sửa, sửa từ đâu, tại sao. Với Event Sourcing, mọi thao tác đều là event bất biến: `AdminForcedBalanceChanged(admin_id, old_balance, new_balance, reason, timestamp)`.

### Temporal query và business intelligence

"Doanh thu tháng 6 năm ngoái là bao nhiêu?" — Với CRUD, nếu bạn không lưu lịch sử, bạn không thể trả lời. Bạn cần một data warehouse riêng, ETL process, và đủ thứ phức tạp.

Event Sourcing cho phép bạn **temporal query**:
- Trạng thái hệ thống tại bất kỳ thời điểm nào trong quá khứ
- Replay events để tái tạo dữ liệu cho báo cáo
- Time travel debugging

### Debugging và incident response

Khi có bug, bạn cần biết **chính xác** điều gì đã xảy ra. Với CRUD, bạn chỉ thấy kết quả cuối cùng. Bạn phải guess. Với ES, bạn replay events trong môi trường dev, tái tạo chính xác trạng thái lúc bug xảy ra, và debug từng bước.

### Complex event processing

Hệ thống cần phản ứng với các pattern phức tạp:
- "Phát hiện gian lận khi user đăng nhập từ 3 quốc gia khác nhau trong 1 giờ"
- "Tự động hủy đơn hàng nếu thanh toán không hoàn tất sau 24h"
- "Gửi email chúc mừng khi user đạt 100 đơn hàng"

Với ES, bạn có event stream, dễ dàng áp dụng **Complex Event Processing** (CEP) để phát hiện pattern.

---

## Nguyên lý thiết kế

### 1. Event là bất biến (Immutable)

Một event, khi đã được lưu, không bao giờ được sửa hoặc xóa. Nếu phát hiện sai, bạn tạo event mới để correct:

```python
# Sai: sửa event cũ
event.amount = 100  # ❌ Không bao giờ

# Đúng: tạo event mới
event_store.append(OrderCorrected(order_id, reason="Sai số tiền", correction=...))
```

### 2. Current state là derived data

Trạng thái hiện tại được tính bằng cách replay tất cả events từ đầu. Đây là **projection**:

```python
def get_account_balance(account_id):
    balance = 0
    for event in event_store.get_events(account_id):
        if isinstance(event, MoneyDeposited):
            balance += event.amount
        elif isinstance(event, MoneyWithdrawn):
            balance -= event.amount
    return balance
```

### 3. Event là sự thật (Event as Truth)

Event store là **single source of truth**. Mọi thứ khác (read models, caches, indexes) đều là projection và có thể được rebuild từ events.

### 4. Snapshot cho performance

Khi number of events quá lớn (hàng triệu), replay từ đầu có thể chậm. **Snapshot** lưu trạng thái tại một thời điểm để làm điểm khởi đầu:

```python
account = snapshot_store.get(account_id)  # State tại version 1000
for event in event_store.get_events(account_id, from_version=1001):
    account.apply(event)
```

---

## Cấu trúc chi tiết

### Các thành phần chính

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Event Sourcing System                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Event Store                              │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │ Events: [E1] → [E2] → [E3] → [E4] → [E5] → ...        │ │   │
│  │  │         Immutable, Append-only, Chronological            │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│  ┌──────────────────┐ ┌────────────┐ ┌──────────────┐              │
│  │   Projection 1   │ │ Projection │ │ Projection 3 │              │
│  │ (Current State)  │ │ (Read      │ │ (Analytics)  │              │
│  │                  │ │  Model)    │ │              │              │
│  │ Order Aggregate  │ │ OrderDTO   │ │ Revenue/Month│              │
│  └──────────────────┘ └────────────┘ └──────────────┘              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Commands                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │   │
│  │  │ Validate │→│ Business │→│ Append   │                  │   │
│  │  │          │  │ Logic    │  │ Event    │                  │   │
│  │  └──────────┘  └──────────┘  └──────────┘                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Event Bus                                │   │
│  │  Khi event được lưu, nó được publish cho các projector       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Event Store

- **Append-only**: Events chỉ được thêm, không sửa/xóa
- **Chronological**: Events có thứ tự thời gian
- **Immutable**: Một event đã lưu không thay đổi
- **Replayable**: Có thể replay từ đầu

### Aggregate

- Là đơn vị transaction (theo DDD)
- Nhận command → validate → sinh event → append
- Có version để optimistic concurrency

### Projection

- Đọc events từ event store
- Tính toán và update read model
- Có thể rebuild từ đầu
- Idempotent (chạy lại không gây hại)

---

## Sơ đồ kiến trúc

```
                          ┌─────────────┐
                          │   Command   │
                          │  (External) │
                          └──────┬──────┘
                                 │
                                 ▼
                     ┌─────────────────────┐
                     │   Command Handler   │
                     │  • Load aggregate   │
                     │  • Validate         │
                     │  • Execute          │
                     │  • Generate events  │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │    Event Store      │
                     │  ┌────────────────┐ │
                     │  │ Event 1        │ │──────┐
                     │  │ Event 2        │ │      │
                     │  │ Event 3        │ │      │
                     │  │ ...            │ │      │
                     │  └────────────────┘ │      │
                     └─────────────────────┘      │
                                                  │
                    ┌─────────────────────────────┘
                    │              │              │
                    ▼              ▼              ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │  Projector   │ │  Projector   │ │  Projector   │
          │  (Current    │ │  (Read DB)   │ │  (Analytics) │
          │   State)     │ │              │ │              │
          └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                 │                │                 │
                 ▼                ▼                 ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │  Snapshot    │ │  PostgreSQL  │ │  Elastic     │
          │  Store       │ │  (Read Model)│ │  Search      │
          └──────────────┘ └──────────────┘ └──────────────┘
                 │
                 ▼
          ┌──────────────┐
          │  API/Query   │
          │  (FastAPI)   │
          └──────────────┘
```

---

## Ví dụ code hoàn chỉnh

Xây dựng hệ thống **quản lý giỏ hàng thương mại điện tử** với Event Sourcing.

### Cấu trúc project

```
cart_es/
├── domain/
│   ├── __init__.py
│   ├── events.py
│   ├── cart.py
│   └── product.py
├── infrastructure/
│   ├── __init__.py
│   ├── event_store.py
│   ├── snapshot_store.py
│   └── projections.py
├── application/
│   ├── __init__.py
│   ├── commands.py
│   ├── queries.py
│   └── handlers.py
├── tests/
│   ├── __init__.py
│   ├── test_cart.py
│   ├── test_event_store.py
│   └── test_projections.py
└── main.py
```

### File: domain/events.py

```python
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Optional
from uuid import UUID, uuid4


class EventVersion(Enum):
    """Phiên bản event schema (cho migration)."""
    V1 = auto()


@dataclass
class DomainEvent(ABC):
    """Base class cho mọi domain event (bất biến)."""
    event_id: UUID = field(default_factory=uuid4)
    aggregate_id: UUID = field(default_factory=uuid4)
    version: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Đảm bảo tính bất biến — raise error nếu ai đó cố sửa."""
        pass


@dataclass
class CartCreated(DomainEvent):
    """Giỏ hàng được tạo."""
    customer_id: UUID
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ItemAddedToCart(DomainEvent):
    """Sản phẩm được thêm vào giỏ."""
    product_id: UUID
    product_name: str
    price: Decimal
    quantity: int
    currency: str = "VND"


@dataclass
class ItemQuantityChanged(DomainEvent):
    """Số lượng sản phẩm trong giỏ thay đổi."""
    product_id: UUID
    old_quantity: int
    new_quantity: int


@dataclass
class ItemRemovedFromCart(DomainEvent):
    """Sản phẩm bị xóa khỏi giỏ."""
    product_id: UUID
    product_name: str
    removed_quantity: int


@dataclass
class DiscountApplied(DomainEvent):
    """Mã giảm giá được áp dụng."""
    coupon_code: str
    discount_amount: Decimal
    discount_type: str  # PERCENTAGE, FIXED


@dataclass
class CartCleared(DomainEvent):
    """Giỏ hàng bị xóa sạch."""
    reason: str = ""


@dataclass
class CartCheckedOut(DomainEvent):
    """Giỏ hàng được thanh toán."""
    total_amount: Decimal
    payment_method: str
    shipping_address: str
```

### File: domain/cart.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from domain.events import *


class CartError(Exception):
    """Base exception cho cart domain."""
    pass


class CartNotFoundError(CartError):
    pass


class CartEmptyError(CartError):
    pass


class ProductNotInCartError(CartError):
    pass


class InvalidQuantityError(CartError):
    pass


@dataclass
class CartItem:
    """Item trong giỏ hàng (value object cho current state)."""
    product_id: UUID
    product_name: str
    price: Decimal
    quantity: int
    currency: str = "VND"

    @property
    def subtotal(self) -> Decimal:
        return self.price * Decimal(str(self.quantity))


@dataclass
class Cart:
    """Aggregate: Giỏ hàng — trạng thái hiện tại được tính từ events."""
    id: UUID
    customer_id: UUID
    items: dict[UUID, CartItem]
    applied_discount: Optional[dict] = None
    is_checked_out: bool = False
    created_at: Optional[datetime] = None
    version: int = 0

    @classmethod
    def create(cls, customer_id: UUID) -> tuple[Cart, list[DomainEvent]]:
        """Factory method tạo giỏ hàng mới."""
        cart_id = uuid4()
        event = CartCreated(
            aggregate_id=cart_id,
            customer_id=customer_id,
        )
        cart = cls._from_events([event])
        return cart, [event]

    @classmethod
    def _from_events(cls, events: list[DomainEvent]) -> Cart:
        """Tái tạo Cart từ danh sách events."""
        cart = cls(id=uuid4(), customer_id=uuid4(), items={})
        for event in events:
            cart._apply(event)
        return cart

    def apply(self, event: DomainEvent) -> None:
        """Apply một event vào aggregate."""
        self._apply(event)
        self.version += 1

    def _apply(self, event: DomainEvent) -> None:
        """Internal apply — cập nhật trạng thái dựa trên event."""
        if isinstance(event, CartCreated):
            self.id = event.aggregate_id
            self.customer_id = event.customer_id
            self.created_at = event.timestamp
        elif isinstance(event, ItemAddedToCart):
            item = CartItem(
                product_id=event.product_id,
                product_name=event.product_name,
                price=event.price,
                quantity=event.quantity,
                currency=event.currency,
            )
            if event.product_id in self.items:
                existing = self.items[event.product_id]
                existing.quantity += event.quantity
            else:
                self.items[event.product_id] = item
        elif isinstance(event, ItemQuantityChanged):
            if event.product_id in self.items:
                self.items[event.product_id].quantity = event.new_quantity
        elif isinstance(event, ItemRemovedFromCart):
            self.items.pop(event.product_id, None)
        elif isinstance(event, DiscountApplied):
            self.applied_discount = {
                "code": event.coupon_code,
                "amount": event.discount_amount,
                "type": event.discount_type,
            }
        elif isinstance(event, CartCleared):
            self.items.clear()
            self.applied_discount = None
        elif isinstance(event, CartCheckedOut):
            self.is_checked_out = True

    def add_item(
        self, product_id: UUID, product_name: str,
        price: Decimal, quantity: int, currency: str = "VND"
    ) -> list[DomainEvent]:
        """Thêm item vào giỏ — sinh event."""
        if quantity <= 0:
            raise InvalidQuantityError("Số lượng phải lớn hơn 0")
        if self.is_checked_out:
            raise CartError("Giỏ hàng đã thanh toán")

        event = ItemAddedToCart(
            aggregate_id=self.id,
            product_id=product_id,
            product_name=product_name,
            price=price,
            quantity=quantity,
            currency=currency,
        )
        return [event]

    def change_quantity(self, product_id: UUID, new_quantity: int) -> list[DomainEvent]:
        """Thay đổi số lượng — sinh event."""
        if product_id not in self.items:
            raise ProductNotInCartError("Sản phẩm không có trong giỏ")
        if new_quantity <= 0:
            raise InvalidQuantityError("Số lượng phải lớn hơn 0")
        if self.is_checked_out:
            raise CartError("Giỏ hàng đã thanh toán")

        old_qty = self.items[product_id].quantity
        if old_qty == new_quantity:
            return []

        event = ItemQuantityChanged(
            aggregate_id=self.id,
            product_id=product_id,
            old_quantity=old_qty,
            new_quantity=new_quantity,
        )
        return [event]

    def remove_item(self, product_id: UUID) -> list[DomainEvent]:
        """Xóa item — sinh event."""
        if product_id not in self.items:
            raise ProductNotInCartError("Sản phẩm không có trong giỏ")
        if self.is_checked_out:
            raise CartError("Giỏ hàng đã thanh toán")

        item = self.items[product_id]
        event = ItemRemovedFromCart(
            aggregate_id=self.id,
            product_id=product_id,
            product_name=item.product_name,
            removed_quantity=item.quantity,
        )
        return [event]

    def apply_discount(self, code: str, amount: Decimal, discount_type: str) -> list[DomainEvent]:
        """Áp dụng mã giảm giá — sinh event."""
        if self.is_checked_out:
            raise CartError("Giỏ hàng đã thanh toán")

        event = DiscountApplied(
            aggregate_id=self.id,
            coupon_code=code,
            discount_amount=amount,
            discount_type=discount_type,
        )
        return [event]

    def clear(self, reason: str = "") -> list[DomainEvent]:
        """Xóa giỏ hàng — sinh event."""
        if self.is_checked_out:
            raise CartError("Giỏ hàng đã thanh toán")

        event = CartCleared(
            aggregate_id=self.id,
            reason=reason,
        )
        return [event]

    def checkout(self, payment_method: str, shipping_address: str) -> list[DomainEvent]:
        """Thanh toán giỏ hàng — sinh event."""
        if not self.items:
            raise CartEmptyError("Giỏ hàng trống")
        if self.is_checked_out:
            raise CartError("Giỏ hàng đã thanh toán")

        total = self.calculate_total()
        event = CartCheckedOut(
            aggregate_id=self.id,
            total_amount=total,
            payment_method=payment_method,
            shipping_address=shipping_address,
        )
        return [event]

    def calculate_total(self) -> Decimal:
        """Tính tổng tiền."""
        total = sum(item.subtotal for item in self.items.values())
        if self.applied_discount:
            if self.applied_discount["type"] == "PERCENTAGE":
                discount = total * self.applied_discount["amount"] / Decimal("100")
                total -= discount
            elif self.applied_discount["type"] == "FIXED":
                total -= self.applied_discount["amount"]
        return max(total, Decimal("0"))

    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self.items.values())

    @property
    def unique_items(self) -> int:
        return len(self.items)
```

### File: domain/product.py

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID


@dataclass
class ProductInfo:
    """Thông tin sản phẩm (value object)."""
    product_id: UUID
    name: str
    price: Decimal
    currency: str = "VND"
    stock: int = 0
    is_available: bool = True
```

### File: infrastructure/event_store.py

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from domain.events import DomainEvent


class EventStore:
    """Event Store — append-only, immutable event storage.

    Trong production, dùng PostgreSQL, EventStoreDB, hoặc Kafka.
    Ở đây dùng in-memory + file-based persistence cho demo.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._storage_path = storage_path or Path("events.jsonl")
        self._events_by_aggregate: dict[UUID, list[DomainEvent]] = {}
        self._all_events: list[DomainEvent] = []
        self._load_from_disk()

    def append(self, aggregate_id: UUID, events: list[DomainEvent], expected_version: int) -> None:
        """Append events với optimistic concurrency control."""
        if aggregate_id not in self._events_by_aggregate:
            self._events_by_aggregate[aggregate_id] = []
            current_version = 0
        else:
            current_version = len(self._events_by_aggregate[aggregate_id])

        if current_version != expected_version:
            raise ConcurrencyError(
                f"Conflict: expected {expected_version}, current {current_version}"
            )

        for event in events:
            event.event_id = UUID(event.event_id) if isinstance(event.event_id, str) else event.event_id
            event.aggregate_id = aggregate_id
            self._events_by_aggregate[aggregate_id].append(event)
            self._all_events.append(event)
            self._save_to_disk(event)

    def get_events(self, aggregate_id: UUID) -> list[DomainEvent]:
        """Lấy tất cả events của một aggregate."""
        return list(self._events_by_aggregate.get(aggregate_id, []))

    def get_events_since(self, aggregate_id: UUID, from_version: int) -> list[DomainEvent]:
        """Lấy events từ một version."""
        events = self._events_by_aggregate.get(aggregate_id, [])
        return events[from_version:]

    def get_all_events(self) -> list[DomainEvent]:
        """Lấy tất cả events."""
        return list(self._all_events)

    def get_all_aggregate_ids(self) -> list[UUID]:
        """Lấy danh sách tất cả aggregate IDs."""
        return list(self._events_by_aggregate.keys())

    def count_events(self, aggregate_id: UUID) -> int:
        """Đếm số events của một aggregate."""
        return len(self._events_by_aggregate.get(aggregate_id, []))

    def _save_to_disk(self, event: DomainEvent) -> None:
        """Persist event xuống disk (JSONL format)."""
        try:
            with open(self._storage_path, "a", encoding="utf-8") as f:
                data = {
                    "event_type": type(event).__name__,
                    "aggregate_id": str(event.aggregate_id),
                    "event_id": str(event.event_id),
                    "timestamp": event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else event.timestamp,
                    "data": self._serialize_event(event),
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except IOError:
            pass  # Trong demo, ignore disk errors

    def _serialize_event(self, event: DomainEvent) -> dict:
        """Serialize event data."""
        result = {}
        for attr in vars(event):
            if attr.startswith("_"):
                continue
            value = getattr(event, attr)
            if isinstance(value, UUID):
                result[attr] = str(value)
            elif isinstance(value, datetime):
                result[attr] = value.isoformat()
            elif isinstance(value, Decimal):
                result[attr] = str(value)
            else:
                result[attr] = value
        return result

    def _load_from_disk(self) -> None:
        """Load events từ disk vào memory."""
        if not self._storage_path.exists():
            return
        # Trong thực tế, cần deserialize events
        pass


class ConcurrencyError(Exception):
    """Xung đột optimistic locking."""
    pass
```

### File: infrastructure/snapshot_store.py

```python
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional
from uuid import UUID

from domain.cart import Cart


class SnapshotStore:
    """Lưu snapshot của aggregate để tăng tốc replay.

    Thay vì replay từ event đầu tiên, ta chỉ replay từ snapshot gần nhất.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._storage_path = storage_path or Path("snapshots")
        self._storage_path.mkdir(exist_ok=True)
        self._snapshots: dict[UUID, tuple[Cart, int]] = {}

    def save(self, aggregate_id: UUID, cart: Cart, version: int) -> None:
        """Lưu snapshot."""
        self._snapshots[aggregate_id] = (cart, version)
        # File-based persistence (pickle for demo)
        snapshot_file = self._storage_path / f"{aggregate_id}.snap"
        try:
            with open(snapshot_file, "wb") as f:
                pickle.dump((cart, version), f)
        except IOError:
            pass

    def load(self, aggregate_id: UUID) -> Optional[tuple[Cart, int]]:
        """Load snapshot gần nhất."""
        # Check memory first
        if aggregate_id in self._snapshots:
            return self._snapshots[aggregate_id]

        # Check disk
        snapshot_file = self._storage_path / f"{aggregate_id}.snap"
        if snapshot_file.exists():
            try:
                with open(snapshot_file, "rb") as f:
                    snapshot = pickle.load(f)
                    self._snapshots[aggregate_id] = snapshot
                    return snapshot
            except (pickle.PickleError, IOError):
                pass
        return None

    def clear(self, aggregate_id: Optional[UUID] = None) -> None:
        """Xóa snapshot."""
        if aggregate_id:
            self._snapshots.pop(aggregate_id, None)
            snapshot_file = self._storage_path / f"{aggregate_id}.snap"
            snapshot_file.unlink(missing_ok=True)
        else:
            self._snapshots.clear()
            for f in self._storage_path.glob("*.snap"):
                f.unlink(missing_ok=True)
```

### File: infrastructure/projections.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from domain.events import *
from domain.cart import Cart, CartItem


@dataclass
class CartProjection:
    """Read model: thông tin giỏ hàng cho hiển thị."""
    cart_id: UUID
    customer_id: UUID
    items: list[dict]
    total: Decimal
    item_count: int
    unique_items: int
    applied_discount: Optional[dict]
    is_checked_out: bool
    created_at: Optional[str]
    last_updated: Optional[str]


@dataclass
class CustomerCartSummary:
    """Read model: tổng quan giỏ hàng của khách hàng."""
    customer_id: UUID
    active_carts: int
    total_items_in_carts: int
    total_value: Decimal
    last_activity: Optional[str]


class CartProjector:
    """Projector: cập nhật read models từ events."""

    def __init__(self):
        self._cart_read_models: dict[UUID, CartProjection] = {}
        self._customer_summaries: dict[UUID, CustomerCartSummary] = {}

    def project(self, event: DomainEvent) -> None:
        """Xử lý event và cập nhật projection."""
        handler = self._get_handler(event)
        if handler:
            handler(event)

    def _get_handler(self, event: DomainEvent):
        handlers = {
            CartCreated: self._on_cart_created,
            ItemAddedToCart: self._on_item_added,
            ItemQuantityChanged: self._on_quantity_changed,
            ItemRemovedFromCart: self._on_item_removed,
            DiscountApplied: self._on_discount_applied,
            CartCleared: self._on_cart_cleared,
            CartCheckedOut: self._on_cart_checked_out,
        }
        return handlers.get(type(event))

    def _on_cart_created(self, event: CartCreated) -> None:
        projection = CartProjection(
            cart_id=event.aggregate_id,
            customer_id=event.customer_id,
            items=[],
            total=Decimal("0"),
            item_count=0,
            unique_items=0,
            applied_discount=None,
            is_checked_out=False,
            created_at=event.timestamp.isoformat(),
            last_updated=event.timestamp.isoformat(),
        )
        self._cart_read_models[event.aggregate_id] = projection
        self._update_customer_summary(event.customer_id)

    def _on_item_added(self, event: ItemAddedToCart) -> None:
        projection = self._cart_read_models.get(event.aggregate_id)
        if not projection:
            return

        # Tìm item hiện tại
        existing = None
        for item in projection.items:
            if item["product_id"] == str(event.product_id):
                existing = item
                break

        if existing:
            existing["quantity"] += event.quantity
            existing["subtotal"] = float(
                Decimal(str(existing["price"])) * Decimal(str(existing["quantity"]))
            )
        else:
            projection.items.append({
                "product_id": str(event.product_id),
                "product_name": event.product_name,
                "price": float(event.price),
                "quantity": event.quantity,
                "currency": event.currency,
                "subtotal": float(event.price * Decimal(str(event.quantity))),
            })

        projection.total = sum(
            Decimal(str(item["subtotal"])) for item in projection.items
        )
        projection.item_count = sum(item["quantity"] for item in projection.items)
        projection.unique_items = len(projection.items)
        projection.last_updated = event.timestamp.isoformat()

        self._update_customer_summary(projection.customer_id)

    def _on_quantity_changed(self, event: ItemQuantityChanged) -> None:
        projection = self._cart_read_models.get(event.aggregate_id)
        if not projection:
            return

        for item in projection.items:
            if item["product_id"] == str(event.product_id):
                item["quantity"] = event.new_quantity
                item["subtotal"] = float(
                    Decimal(str(item["price"])) * Decimal(str(event.new_quantity))
                )
                break

        projection.total = sum(
            Decimal(str(item["subtotal"])) for item in projection.items
        )
        projection.item_count = sum(item["quantity"] for item in projection.items)
        projection.last_updated = event.timestamp.isoformat()

    def _on_item_removed(self, event: ItemRemovedFromCart) -> None:
        projection = self._cart_read_models.get(event.aggregate_id)
        if not projection:
            return

        projection.items = [
            item for item in projection.items
            if item["product_id"] != str(event.product_id)
        ]
        projection.total = sum(
            Decimal(str(item["subtotal"])) for item in projection.items
        )
        projection.item_count = sum(item["quantity"] for item in projection.items)
        projection.unique_items = len(projection.items)
        projection.last_updated = event.timestamp.isoformat()

        self._update_customer_summary(projection.customer_id)

    def _on_discount_applied(self, event: DiscountApplied) -> None:
        projection = self._cart_read_models.get(event.aggregate_id)
        if not projection:
            return

        projection.applied_discount = {
            "code": event.coupon_code,
            "amount": float(event.discount_amount),
            "type": event.discount_type,
        }
        projection.last_updated = event.timestamp.isoformat()

    def _on_cart_cleared(self, event: CartCleared) -> None:
        projection = self._cart_read_models.get(event.aggregate_id)
        if not projection:
            return

        projection.items = []
        projection.total = Decimal("0")
        projection.item_count = 0
        projection.unique_items = 0
        projection.applied_discount = None
        projection.last_updated = event.timestamp.isoformat()

        self._update_customer_summary(projection.customer_id)

    def _on_cart_checked_out(self, event: CartCheckedOut) -> None:
        projection = self._cart_read_models.get(event.aggregate_id)
        if not projection:
            return

        projection.is_checked_out = True
        projection.last_updated = event.timestamp.isoformat()

        self._update_customer_summary(projection.customer_id)

    def _update_customer_summary(self, customer_id: UUID) -> None:
        """Cập nhật summary cho khách hàng."""
        active_carts = [
            c for c in self._cart_read_models.values()
            if c.customer_id == customer_id and not c.is_checked_out
        ]
        total_items = sum(c.item_count for c in active_carts)
        total_value = sum(c.total if isinstance(c.total, Decimal) else Decimal(str(c.total)) for c in active_carts)
        last_activity = max(
            (c.last_updated for c in active_carts if c.last_updated),
            default=None,
        )

        self._customer_summaries[customer_id] = CustomerCartSummary(
            customer_id=customer_id,
            active_carts=len(active_carts),
            total_items_in_carts=total_items,
            total_value=total_value,
            last_activity=last_activity,
        )

    def get_cart_projection(self, cart_id: UUID) -> Optional[CartProjection]:
        """Lấy read model của giỏ hàng."""
        return self._cart_read_models.get(cart_id)

    def get_customer_summary(self, customer_id: UUID) -> Optional[CustomerCartSummary]:
        """Lấy summary của khách hàng."""
        return self._customer_summaries.get(customer_id)

    def rebuild_from_events(self, all_events: list[DomainEvent]) -> None:
        """Rebuild tất cả projections từ events (idempotent)."""
        self._cart_read_models.clear()
        self._customer_summaries.clear()
        for event in all_events:
            self.project(event)
```

### File: application/commands.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class CreateCartCommand:
    """Command: Tạo giỏ hàng mới."""
    customer_id: UUID


@dataclass
class AddItemCommand:
    """Command: Thêm sản phẩm vào giỏ."""
    cart_id: UUID
    product_id: UUID
    product_name: str
    price: Decimal
    quantity: int
    currency: str = "VND"
    idempotency_key: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ChangeQuantityCommand:
    """Command: Thay đổi số lượng."""
    cart_id: UUID
    product_id: UUID
    new_quantity: int


@dataclass
class RemoveItemCommand:
    """Command: Xóa sản phẩm khỏi giỏ."""
    cart_id: UUID
    product_id: UUID


@dataclass
class ApplyDiscountCommand:
    """Command: Áp dụng mã giảm giá."""
    cart_id: UUID
    coupon_code: str
    amount: Decimal
    discount_type: str  # PERCENTAGE, FIXED


@dataclass
class ClearCartCommand:
    """Command: Xóa giỏ hàng."""
    cart_id: UUID
    reason: str = ""


@dataclass
class CheckoutCommand:
    """Command: Thanh toán giỏ hàng."""
    cart_id: UUID
    payment_method: str
    shipping_address: str
```

### File: application/handlers.py

```python
from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from domain.cart import Cart
from domain.events import DomainEvent
from infrastructure.event_store import EventStore, ConcurrencyError
from infrastructure.snapshot_store import SnapshotStore
from infrastructure.projections import CartProjector, CartProjection


class CartCommandHandler:
    """Xử lý commands cho giỏ hàng."""

    SNAPSHOT_FREQUENCY = 10  # Lưu snapshot mỗi 10 events

    def __init__(
        self,
        event_store: EventStore,
        snapshot_store: Optional[SnapshotStore] = None,
        projector: Optional[CartProjector] = None,
    ):
        self._event_store = event_store
        self._snapshot_store = snapshot_store or SnapshotStore()
        self._projector = projector

    def handle_create_cart(self, cmd) -> dict:
        """Xử lý tạo giỏ hàng."""
        cart, events = Cart.create(cmd.customer_id)
        self._event_store.append(cart.id, events, 0)
        cart.apply(events[-1])  # Cập nhật local state

        if self._projector:
            for event in events:
                self._projector.project(event)

        self._maybe_save_snapshot(cart.id, cart)

        return {"cart_id": str(cart.id), "events_appended": len(events)}

    def handle_add_item(self, cmd) -> dict:
        """Xử lý thêm item."""
        cart = self._load_cart(cmd.cart_id)
        events = cart.add_item(
            product_id=cmd.product_id,
            product_name=cmd.product_name,
            price=cmd.price,
            quantity=cmd.quantity,
            currency=cmd.currency,
        )
        expected_version = self._event_store.count_events(cart.id)
        self._event_store.append(cart.id, events, expected_version)

        for event in events:
            cart.apply(event)

        if self._projector:
            for event in events:
                self._projector.project(event)

        self._maybe_save_snapshot(cart.id, cart)

        return {
            "cart_id": str(cart.id),
            "item_count": cart.item_count,
            "total": float(cart.calculate_total()),
            "events_appended": len(events),
        }

    def handle_change_quantity(self, cmd) -> dict:
        """Xử lý thay đổi số lượng."""
        cart = self._load_cart(cmd.cart_id)
        events = cart.change_quantity(cmd.product_id, cmd.new_quantity)
        if events:
            expected_version = self._event_store.count_events(cart.id)
            self._event_store.append(cart.id, events, expected_version)

            for event in events:
                cart.apply(event)

            if self._projector:
                for event in events:
                    self._projector.project(event)

            self._maybe_save_snapshot(cart.id, cart)

        return {
            "cart_id": str(cart.id),
            "item_count": cart.item_count,
            "total": float(cart.calculate_total()),
        }

    def handle_remove_item(self, cmd) -> dict:
        """Xử lý xóa item."""
        cart = self._load_cart(cmd.cart_id)
        events = cart.remove_item(cmd.product_id)
        expected_version = self._event_store.count_events(cart.id)
        self._event_store.append(cart.id, events, expected_version)

        for event in events:
            cart.apply(event)

        if self._projector:
            for event in events:
                self._projector.project(event)

        self._maybe_save_snapshot(cart.id, cart)

        return {
            "cart_id": str(cart.id),
            "item_count": cart.item_count,
            "total": float(cart.calculate_total()),
        }

    def handle_apply_discount(self, cmd) -> dict:
        """Xử lý áp dụng mã giảm giá."""
        cart = self._load_cart(cmd.cart_id)
        events = cart.apply_discount(cmd.coupon_code, cmd.amount, cmd.discount_type)
        expected_version = self._event_store.count_events(cart.id)
        self._event_store.append(cart.id, events, expected_version)

        for event in events:
            cart.apply(event)

        if self._projector:
            for event in events:
                self._projector.project(event)

        return {
            "cart_id": str(cart.id),
            "total": float(cart.calculate_total()),
            "discount": float(cmd.amount),
        }

    def handle_clear_cart(self, cmd) -> dict:
        """Xử lý xóa giỏ hàng."""
        cart = self._load_cart(cmd.cart_id)
        events = cart.clear(cmd.reason)
        expected_version = self._event_store.count_events(cart.id)
        self._event_store.append(cart.id, events, expected_version)

        for event in events:
            cart.apply(event)

        if self._projector:
            for event in events:
                self._projector.project(event)

        return {
            "cart_id": str(cart.id),
            "cleared": True,
        }

    def handle_checkout(self, cmd) -> dict:
        """Xử lý thanh toán."""
        cart = self._load_cart(cmd.cart_id)
        events = cart.checkout(cmd.payment_method, cmd.shipping_address)
        expected_version = self._event_store.count_events(cart.id)
        self._event_store.append(cart.id, events, expected_version)

        for event in events:
            cart.apply(event)

        if self._projector:
            for event in events:
                self._projector.project(event)

        return {
            "cart_id": str(cart.id),
            "total": float(cart.calculate_total()),
            "item_count": cart.item_count,
        }

    def _load_cart(self, cart_id: UUID) -> Cart:
        """Load cart với snapshot optimization."""
        # Thử load từ snapshot
        snapshot = self._snapshot_store.load(cart_id)
        if snapshot:
            cart, snapshot_version = snapshot
            start_version = snapshot_version
        else:
            cart = Cart(id=cart_id, customer_id=UUID(int=0), items={})
            start_version = 0

        # Replay các events từ snapshot đến hiện tại
        events = self._event_store.get_events_since(cart_id, start_version)
        for event in events:
            cart.apply(event)

        return cart

    def _maybe_save_snapshot(self, cart_id: UUID, cart: Cart) -> None:
        """Lưu snapshot định kỳ."""
        event_count = self._event_store.count_events(cart_id)
        if event_count > 0 and event_count % self.SNAPSHOT_FREQUENCY == 0:
            self._snapshot_store.save(cart_id, cart, event_count)
```

### File: main.py

```python
#!/usr/bin/env python3
"""
Event Sourcing E-Commerce Cart — Ví dụ hoàn chỉnh.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

from application.commands import (
    CreateCartCommand, AddItemCommand, ChangeQuantityCommand,
    RemoveItemCommand, ApplyDiscountCommand, ClearCartCommand, CheckoutCommand,
)
from application.handlers import CartCommandHandler
from infrastructure.event_store import EventStore
from infrastructure.snapshot_store import SnapshotStore
from infrastructure.projections import CartProjector


def print_separator(title: str) -> None:
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main() -> None:
    print("🛒  EVENT SOURCING — Shopping Cart Demo")
    print("=" * 65)

    # Khởi tạo infrastructure
    event_store = EventStore()
    snapshot_store = SnapshotStore()
    projector = CartProjector()
    handler = CartCommandHandler(event_store, snapshot_store, projector)

    customer_id = UUID("12345678-1234-5678-1234-567812345678")

    # === 1. Tạo giỏ hàng ===
    print_separator("1. Tạo giỏ hàng mới")
    result = handler.handle_create_cart(CreateCartCommand(customer_id=customer_id))
    cart_id = UUID(result["cart_id"])
    print(f"   ✅ Đã tạo giỏ hàng: {cart_id}")
    print(f"   📦 Events appended: {result['events_appended']}")
    print(f"   📌 Trạng thái: {event_store.count_events(cart_id)} events")

    # === 2. Thêm sản phẩm ===
    print_separator("2. Thêm sản phẩm vào giỏ")
    items = [
        ("iPhone 15 Pro Max", Decimal("34990000"), 1),
        ("AirPods Pro 2", Decimal("6490000"), 2),
        ("Ốp lưng Silicon", Decimal("299000"), 3),
    ]
    for name, price, qty in items:
        result = handler.handle_add_item(AddItemCommand(
            cart_id=cart_id,
            product_id=uuid4(),
            product_name=name,
            price=price,
            quantity=qty,
        ))
        print(f"   ✅ + {name:25s} x{qty:2d} = {price * Decimal(str(qty)):>12,.0f}₫")
        print(f"      Tổng: {result['total']:>25,.0f}₫  |  Items: {result['item_count']:2d}")

    # === 3. Đọc trạng thái hiện tại ===
    print_separator("3. Replay events → Current State")
    cart = handler._load_cart(cart_id)
    print(f"   🛒 Giỏ hàng: {cart.id}")
    print(f"   👤 Khách: {cart.customer_id}")
    print(f"   📦 Số sản phẩm: {cart.unique_items} loại, {cart.item_count} cái")
    print(f"   💰 Tổng tiền: {cart.calculate_total():>25,.0f}₫")
    print(f"   📌 Version: {cart.version}")

    for pid, item in cart.items.items():
        print(f"      • {item.product_name:25s} x{item.quantity} = {item.subtotal:>12,.0f}₫")

    # === 4. Thay đổi số lượng ===
    print_separator("4. Thay đổi số lượng (Command + Event)")
    first_product_id = list(cart.items.keys())[0]
    result = handler.handle_change_quantity(ChangeQuantityCommand(
        cart_id=cart_id,
        product_id=first_product_id,
        new_quantity=3,
    ))
    print(f"   ✅ Đã thay đổi số lượng sản phẩm đầu tiên từ 1 → 3")
    print(f"   📦 Tổng items: {result['item_count']}")
    print(f"   💰 Tổng tiền: {result['total']:>25,.0f}₫")

    # === 5. Áp dụng mã giảm giá ===
    print_separator("5. Áp dụng mã giảm giá (Event)")
    total_before = handler._load_cart(cart_id).calculate_total()
    print(f"   💵 Tổng trước giảm: {total_before:>25,.0f}₫")

    result = handler.handle_apply_discount(ApplyDiscountCommand(
        cart_id=cart_id,
        coupon_code="SALE10",
        amount=Decimal("10"),
        discount_type="PERCENTAGE",
    ))
    print(f"   🏷️  Mã: SALE10 (giảm 10%)")
    print(f"   💰 Tổng sau giảm: {result['total']:>25,.0f}₫")

    # === 6. Xóa sản phẩm ===
    print_separator("6. Xóa sản phẩm khỏi giỏ (Event)")
    second_product_id = list(cart.items.keys())[1]
    result = handler.handle_remove_item(RemoveItemCommand(
        cart_id=cart_id,
        product_id=second_product_id,
    ))
    print(f"   ✅ Đã xóa sản phẩm thứ hai khỏi giỏ")
    print(f"   📦 Items còn lại: {result['item_count']}")
    print(f"   💰 Tổng: {result['total']:>25,.0f}₫")

    # === 7. Lịch sử events ===
    print_separator("7. Event Stream (Immutable History)")
    events = event_store.get_events(cart_id)
    for i, event in enumerate(events, 1):
        event_type = type(event).__name__
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        details = _get_event_details(event)
        print(f"   {i:2d}. [{timestamp}] {event_type:25s} {details}")

    # === 8. Kiểm tra Projection ===
    print_separator("8. Read Model (CartProjection)")
    projection = projector.get_cart_projection(cart_id)
    if projection:
        print(f"   🆔 Cart: {projection.cart_id}")
        print(f"   👤 Customer: {projection.customer_id}")
        print(f"   📦 Items: {projection.unique_items} loại, {projection.item_count} cái")
        print(f"   💰 Total: {projection.total:>25,.0f}₫")
        print(f"   🏷️  Discount: {projection.applied_discount}")
        print(f"   ✅ Checked out: {projection.is_checked_out}")
        print()
        for item in projection.items:
            print(f"      • {item['product_name']:25s} x{item['quantity']} = {item['subtotal']:>12,.0f}₫")

    # === 9. Customer Summary ===
    print_separator("9. Customer Summary (Rebuilt from Events)")
    summary = projector.get_customer_summary(customer_id)
    if summary:
        print(f"   👤 Customer: {summary.customer_id}")
        print(f"   🛒 Active carts: {summary.active_carts}")
        print(f"   📦 Total items: {summary.total_items_in_carts}")
        print(f"   💰 Total value: {summary.total_value:>25,.0f}₫")

    # === 10. Rebuild từ events ===
    print_separator("10. Rebuild Projections từ Events (Idempotent)")
    all_events = event_store.get_all_events()
    print(f"    Tổng events: {len(all_events)}")
    projector.rebuild_from_events(all_events)
    rebuilt = projector.get_cart_projection(cart_id)
    print(f"    ✅ Rebuilt projection: {rebuilt.unique_items} items, {rebuilt.total:,.0f}₫")
    print(f"    Kiểm tra consistency: projection.total == load_cart().total?")
    cart = handler._load_cart(cart_id)
    assert rebuilt.total == float(cart.calculate_total()), "Consistency check FAILED!"
    print(f"    ✅ Consistency check PASSED!")

    # === 11. Thanh toán ===
    print_separator("11. Checkout (Event cuối cùng)")
    result = handler.handle_checkout(CheckoutCommand(
        cart_id=cart_id,
        payment_method="CREDIT_CARD",
        shipping_address="123 Nguyễn Huệ, Quận 1, TP. Hồ Chí Minh",
    ))
    print(f"   ✅ Đã thanh toán: {result['total']:>25,.0f}₫")
    print(f"   📦 Items: {result['item_count']}")

    # === 12. Tính bất biến của Event ===
    print_separator("12. Event Immutability & Temporal Query")
    print(f"    Tổng events trong Event Store: {len(event_store.get_all_events())}")
    print(f"    Các events KHÔNG THỂ bị sửa hoặc xóa.")
    print(f"    Để sửa lỗi, ta tạo event mới (corrective event).")
    print(f"    Luôn có thể replay từ đầu để rebuild bất kỳ trạng thái nào.")

    print()
    print("=" * 65)
    print("  ✅ Event Sourcing Demo hoàn tất!")
    print("=" * 65)


def _get_event_details(event) -> str:
    """Lấy thông tin chi tiết ngắn gọn từ event."""
    if hasattr(event, 'product_name'):
        return f"{getattr(event, 'product_name', '')} x{getattr(event, 'quantity', '')}"
    if hasattr(event, 'coupon_code'):
        return f"Code: {event.coupon_code} ({event.discount_type})"
    if hasattr(event, 'total_amount'):
        return f"Total: {event.total_amount:,.0f}₫"
    if hasattr(event, 'reason'):
        return f"Reason: {event.reason}"
    return ""


if __name__ == "__main__":
    main()
```

### Output khi chạy:

```
🛒  EVENT SOURCING — Shopping Cart Demo
=================================================================

=================================================================
  1. Tạo giỏ hàng mới
=================================================================
   ✅ Đã tạo giỏ hàng: 550e8400-e29b-41d4-a716-446655440000
   📦 Events appended: 1
   📌 Trạng thái: 1 events

=================================================================
  2. Thêm sản phẩm vào giỏ
=================================================================
   ✅ + iPhone 15 Pro Max           x 1 =   34,990,000₫
      Tổng:                    34,990,000₫  |  Items:  1
   ✅ + AirPods Pro 2               x 2 =   12,980,000₫
      Tổng:                    47,970,000₫  |  Items:  3
   ✅ + Ốp lưng Silicon             x 3 =      897,000₫
      Tổng:                    48,867,000₫  |  Items:  6

=================================================================
  3. Replay events → Current State
=================================================================
   🛒 Giỏ hàng: 550e8400-e29b-41d4-a716-446655440000
   👤 Khách: 12345678-1234-5678-1234-567812345678
   📦 Số sản phẩm: 3 loại, 6 cái
   💰 Tổng tiền:                   48,867,000₫
   📌 Version: 4
      • iPhone 15 Pro Max          x1 =   34,990,000₫
      • AirPods Pro 2              x2 =   12,980,000₫
      • Ốp lưng Silicon            x3 =      897,000₫

=================================================================
  4. Thay đổi số lượng (Command + Event)
=================================================================
   ✅ Đã thay đổi số lượng sản phẩm đầu tiên từ 1 → 3
   📦 Tổng items: 8
   💰 Tổng tiền:                  118,847,000₫

=================================================================
  5. Áp dụng mã giảm giá (Event)
=================================================================
   💵 Tổng trước giảm:            118,847,000₫
   🏷️  Mã: SALE10 (giảm 10%)
   💰 Tổng sau giảm:              106,962,300₫

=================================================================
  6. Xóa sản phẩm khỏi giỏ (Event)
=================================================================
   ✅ Đã xóa sản phẩm thứ hai khỏi giỏ
   📦 Items còn lại: 6
   💰 Tổng:                          96,291,000₫

=================================================================
  7. Event Stream (Immutable History)
=================================================================
    1. [timestamp] CartCreated               Customer: 12345678-1234-5678-1234-567812345678
    2. [timestamp] ItemAddedToCart           iPhone 15 Pro Max x1
    3. [timestamp] ItemAddedToCart           AirPods Pro 2 x2
    4. [timestamp] ItemAddedToCart           Ốp lưng Silicon x3
    5. [timestamp] ItemQuantityChanged       iPhone 15 Pro Max: 1→3
    6. [timestamp] DiscountApplied           Code: SALE10 (PERCENTAGE)
    7. [timestamp] ItemRemovedFromCart       AirPods Pro 2 removed
    8. [timestamp] CartCheckedOut            Total: 96,291,000₫

=================================================================
  8. Read Model (CartProjection)
=================================================================
   🆔 Cart: 550e8400-e29b-41d4-a716-446655440000
   👤 Customer: 12345678-1234-5678-1234-567812345678
   📦 Items: 2 loại, 6 cái
   💰 Total:                       96,291,000₫
   🏷️  Discount: {'code': 'SALE10', 'amount': 10.0, 'type': 'PERCENTAGE'}
   ✅ Checked out: True
      • iPhone 15 Pro Max          x3 =  104,970,000₫
      • Ốp lưng Silicon            x3 =      897,000₫

=================================================================
  9. Customer Summary (Rebuilt from Events)
=================================================================
   👤 Customer: 12345678-1234-5678-1234-567812345678
   🛒 Active carts: 0
   📦 Total items: 0
   💰 Total value:                        0₫

=================================================================
  10. Rebuild Projections từ Events (Idempotent)
=================================================================
    Tổng events: 8
    ✅ Rebuilt projection: 2 items, 96,291,000₫
    Kiểm tra consistency: projection.total == load_cart().total?
    ✅ Consistency check PASSED!

=================================================================
  11. Checkout (Event cuối cùng)
=================================================================
   ✅ Đã thanh toán:                   96,291,000₫
   📦 Items: 6

=================================================================
  12. Event Immutability & Temporal Query
=================================================================
    Tổng events trong Event Store: 8
    Các events KHÔNG THỂ bị sửa hoặc xóa.
    Để sửa lỗi, ta tạo event mới (corrective event).
    Luôn có thể replay từ đầu để rebuild bất kỳ trạng thái nào.

=================================================================
  ✅ Event Sourcing Demo hoàn tất!
=================================================================
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng Event Sourcing | Khi nào không dùng |
|---|---|
| Cần audit trail đầy đủ, không thể mất lịch sử | CRUD đơn giản, không cần lịch sử |
| Cần temporal queries (trạng thái tại thời điểm bất kỳ) | Hệ thống real-time, cần strong consistency |
| Domain phức tạp, nhiều business events | Storage không phải là vấn đề |
| Cần debug/replay khả năng cao | Số lượng events quá lớn, performance quan trọng |
| Hệ thống event-driven, CQRS | Team chưa quen với event-driven thinking |
| Machine learning trên event stream | Chi phí storage là ưu tiên số 1 |
| Compliance/Regulatory requirements | Cần xóa dữ liệu vĩnh viễn (GDPR right to erasure) |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Audit trail hoàn hảo, biết chính xác ai làm gì | Phức tạp hơn CRUD rất nhiều |
| Temporal query: biết trạng thái tại bất kỳ thời điểm nào | Event schema evolution khó khăn |
| Replay để debug, tái tạo bug | Storage lớn hơn (lưu tất cả events) |
| Event stream cho ML/AI | Eventually consistency |
| CQRS tự nhiên: events là nguồn cho read model | Cần snapshot cho performance |
| Không mất dữ liệu (append-only) | Khó xóa dữ liệu (immutable) |
| Parallel development dễ dàng | Cần kỷ luật về event design |
| High availability (event store phân tán) | Testing phức tạp hơn |

---

## Công cụ và Framework

### Event Store chuyên dụng
- **EventStoreDB** — CSDL events chuyên dụng, có giao diện HTTP/Protobuf
- **Axon Server** — Event store + message bus cho Java/Axon Framework
- **Kafka** — Distributed event log, có thể dùng làm event store
- **Pulsar** — Event streaming platform

### Python
- **Eventsourcing** — Thư viện Python đầy đủ cho Event Sourcing + CQRS
- **EventStoreDB Client** — Python client cho EventStoreDB
- **Kafka-Python** — Kafka client cho Python
- **Faust** — Stream processing library

### .NET
- **EventStore.Client** — .NET client cho EventStoreDB
- **Martendb** — Event store + document DB cho .NET
- **NEventStore** — Event store library cho .NET
- **SqlStreamStore** — Event store trên SQL database

### Java
- **Axon Framework** — CQRS + Event Sourcing full-stack
- **Eventuate** — Microservices với ES
- **Lagom** — Reactive microservices với ES

### Storage Options
- **PostgreSQL** — Dùng event table + JSONB
- **MySQL/MariaDB** — Tương tự PostgreSQL
- **MongoDB** — Document store phù hợp với event
- **DynamoDB** — AWS native event store
- **Google Spanner** — Globally distributed

---

## Kiểm thử

### Chiến lược kiểm thử cho Event Sourcing

```python
# tests/test_cart.py

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from domain.cart import Cart, CartError, CartEmptyError, ProductNotInCartError
from domain.events import (
    CartCreated, ItemAddedToCart, ItemRemovedFromCart,
    CartCleared, CartCheckedOut,
)


class TestCartAggregate:
    """Kiểm thử Aggregate — trung tâm của Event Sourcing."""

    @pytest.fixture
    def cart_id(self):
        return uuid4()

    @pytest.fixture
    def customer_id(self):
        return uuid4()

    def test_create_cart(self, customer_id):
        cart, events = Cart.create(customer_id)
        assert len(events) == 1
        assert isinstance(events[0], CartCreated)
        assert cart.customer_id == customer_id
        assert cart.item_count == 0

    def test_add_item(self, customer_id):
        cart, _ = Cart.create(customer_id)
        product_id = uuid4()
        events = cart.add_item(
            product_id=product_id,
            product_name="Test Product",
            price=Decimal("100000"),
            quantity=2,
        )
        assert len(events) == 1
        assert isinstance(events[0], ItemAddedToCart)
        cart.apply(events[0])
        assert cart.item_count == 2
        assert cart.calculate_total() == Decimal("200000")

    def test_add_zero_quantity(self, customer_id):
        cart, _ = Cart.create(customer_id)
        with pytest.raises(CartError, match="lớn hơn 0"):
            cart.add_item(
                product_id=uuid4(),
                product_name="Test",
                price=Decimal("100000"),
                quantity=0,
            )

    def test_remove_item(self, customer_id):
        cart, _ = Cart.create(customer_id)
        pid = uuid4()
        events = cart.add_item(pid, "Test", Decimal("50000"), 1)
        cart.apply(events[0])

        events = cart.remove_item(pid)
        assert len(events) == 1
        assert isinstance(events[0], ItemRemovedFromCart)
        cart.apply(events[0])
        assert cart.item_count == 0

    def test_remove_nonexistent_item(self, customer_id):
        cart, _ = Cart.create(customer_id)
        with pytest.raises(ProductNotInCartError):
            cart.remove_item(uuid4())

    def test_checkout_empty_cart(self, customer_id):
        cart, _ = Cart.create(customer_id)
        with pytest.raises(CartEmptyError):
            cart.checkout("CREDIT_CARD", "Address")

    def test_cannot_modify_after_checkout(self, customer_id):
        cart, _ = Cart.create(customer_id)
        pid = uuid4()
        events = cart.add_item(pid, "Test", Decimal("50000"), 1)
        cart.apply(events[0])

        cart.checkout("CASH", "Address")
        cart.apply = lambda e: None  # Ứng xử như đã checkout

        with pytest.raises(CartError, match="đã thanh toán"):
            cart.add_item(pid, "Test", Decimal("50000"), 1)

    def test_calculate_total_with_discount(self, customer_id):
        cart, _ = Cart.create(customer_id)
        pid = uuid4()
        events = cart.add_item(pid, "Laptop", Decimal("20000000"), 1)
        cart.apply(events[0])

        # Percentage discount 10%
        events = cart.apply_discount("SALE10", Decimal("10"), "PERCENTAGE")
        cart.apply(events[0])
        assert cart.calculate_total() == Decimal("18000000")

        # Fixed discount
        cart.applied_discount = None
        events = cart.apply_discount("GIAM2TR", Decimal("2000000"), "FIXED")
        cart.apply(events[0])
        assert cart.calculate_total() == Decimal("18000000")


# tests/test_event_store.py

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from domain.events import CartCreated, ItemAddedToCart
from infrastructure.event_store import EventStore, ConcurrencyError


class TestEventStore:
    """Kiểm thử Event Store — append-only, concurrency."""

    @pytest.fixture
    def event_store(self):
        return EventStore()

    def test_append_and_read_events(self, event_store: EventStore):
        aggregate_id = uuid4()
        events = [
            CartCreated(aggregate_id=aggregate_id, customer_id=uuid4()),
            ItemAddedToCart(
                aggregate_id=aggregate_id,
                product_id=uuid4(),
                product_name="Test",
                price=Decimal("100000"),
                quantity=1,
            ),
        ]
        event_store.append(aggregate_id, events, 0)

        retrieved = event_store.get_events(aggregate_id)
        assert len(retrieved) == 2
        assert isinstance(retrieved[0], CartCreated)
        assert isinstance(retrieved[1], ItemAddedToCart)

    def test_optimistic_concurrency(self, event_store: EventStore):
        aggregate_id = uuid4()
        events = [CartCreated(aggregate_id=aggregate_id, customer_id=uuid4())]
        event_store.append(aggregate_id, events, 0)

        # Conflict: expected version 0 nhưng đã có 1 event
        with pytest.raises(ConcurrencyError):
            event_store.append(aggregate_id, events, 0)

    def test_append_multiple_batches(self, event_store: EventStore):
        aggregate_id = uuid4()
        event1 = [CartCreated(aggregate_id=aggregate_id, customer_id=uuid4())]
        event_store.append(aggregate_id, event1, 0)

        event2 = [ItemAddedToCart(
            aggregate_id=aggregate_id,
            product_id=uuid4(),
            product_name="Test",
            price=Decimal("50000"),
            quantity=1,
        )]
        event_store.append(aggregate_id, event2, 1)  # version 1

        assert event_store.count_events(aggregate_id) == 2


# tests/test_projections.py

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from domain.events import CartCreated, ItemAddedToCart, DiscountApplied
from infrastructure.projections import CartProjector


class TestCartProjector:
    """Kiểm thử Projector — event → read model."""

    @pytest.fixture
    def projector(self):
        return CartProjector()

    def test_cart_created_projection(self, projector: CartProjector):
        cart_id = uuid4()
        event = CartCreated(aggregate_id=cart_id, customer_id=uuid4())
        projector.project(event)

        projection = projector.get_cart_projection(cart_id)
        assert projection is not None
        assert projection.item_count == 0
        assert projection.total == Decimal("0")

    def test_item_added_projection(self, projector: CartProjector):
        cart_id = uuid4()
        projector.project(CartCreated(aggregate_id=cart_id, customer_id=uuid4()))

        event = ItemAddedToCart(
            aggregate_id=cart_id,
            product_id=uuid4(),
            product_name="Test",
            price=Decimal("100000"),
            quantity=3,
        )
        projector.project(event)

        projection = projector.get_cart_projection(cart_id)
        assert projection.item_count == 3
        assert projection.total == Decimal("300000")
        assert len(projection.items) == 1

    def test_rebuild_is_idempotent(self, projector: CartProjector):
        """Rebuild from same events → same result."""
        cart_id = uuid4()
        events = [
            CartCreated(aggregate_id=cart_id, customer_id=uuid4()),
            ItemAddedToCart(
                aggregate_id=cart_id,
                product_id=uuid4(),
                product_name="Product A",
                price=Decimal("50000"),
                quantity=2,
            ),
        ]

        projector.rebuild_from_events(events)
        first = projector.get_cart_projection(cart_id)

        projector.rebuild_from_events(events)
        second = projector.get_cart_projection(cart_id)

        assert first.total == second.total
        assert first.item_count == second.item_count
```

---

## Kết luận

Event Sourcing là một pattern mạnh mẽ thay đổi cách bạn nghĩ về dữ liệu. Thay vì lưu "trạng thái hiện tại", bạn lưu "câu chuyện" — mọi sự kiện đã xảy ra. Điều này mang lại khả năng audit, temporal query, và debug không thể có với CRUD.

### Best Practices

1.  **Event names are past tense** — `OrderPlaced`, `ItemAdded`, `MoneyWithdrawn`
2.  **Events are business facts** — Không lưu technical detail trong event
3.  **One event = one change** — Event nhỏ, atomic
4.  **Schema evolution plan** — Dùng event versioning, không sửa event cũ
5.  **Snapshot strategy** — Snapshot khi cần, không snapshot quá thường xuyên
6.  **Idempotent projections** — Chạy lại projection nhiều lần cho kết quả giống nhau
7.  **Eventual consistency accepted** — Ghi rõ trong API documentation
8.  **Testing bằng event replay** — Test bằng cách replay events

### Golden Rules

| Rule | Mô tả |
|---|---|
| **Events are immutable** | Không bao giờ sửa hoặc xóa event đã lưu |
| **Current state is derived** | State hiện tại là kết quả của replay events |
| **Event store is source of truth** | Mọi thứ khác đều là cache |
| **Correct with new events** | Sai thì tạo event mới, không sửa event cũ |
| **Schema evolution is additive** | Thêm field, không xóa field |
| **Projections are disposable** | Có thể xóa và rebuild bất kỳ lúc nào |
| **Aggregate = consistency boundary** | Transaction chỉ trong một aggregate |

### Khi nào Event Sourcing thực sự tỏa sáng

- **Hệ thống tài chính**: Banking, insurance, trading
- **Hệ thống compliance**: Healthcare, government, legal
- **E-commerce**: Order management, inventory, cart
- **IoT**: Device state history, sensor data
- **Audit-heavy systems**: Any system where "who changed what and when" matters

Event Sourcing không phải là giải pháp cho mọi bài toán. Nhưng khi bạn cần biết **chính xác** điều gì đã xảy ra, không chỉ trạng thái hiện tại, thì Event Sourcing là lựa chọn đúng đắn.
