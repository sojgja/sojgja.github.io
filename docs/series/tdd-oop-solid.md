---
id: tdd-oop-solid
title: TDD với OOP và SOLID
sidebar_label: 🔴 TDD & SOLID
sidebar_position: 58
---

# TDD với OOP và SOLID

> *"TDD is not about testing. TDD is about design. The tests are just a means to an end."* — **Uncle Bob Martin**

Một trong những sức mạnh lớn nhất của TDD là nó **dẫn dắt thiết kế** theo hướng OOP tốt và tuân thủ SOLID một cách tự nhiên. Khi bạn viết test trước, bạn buộc phải suy nghĩ về interface từ góc nhìn client — và điều đó tự động đẩy bạn về phía thiết kế có trách nhiệm đơn lẻ, dependency injection, và abstraction đúng đắn.

## TDD → Dependency Injection → Testability

### Vấn đề: Code khó test

Hãy bắt đầu với một ví dụ điển hình — code không được thiết kế cho testability:

```python
# src/order_processor.py — KHÔNG testable
import sqlite3
import smtplib
from datetime import datetime


class OrderProcessor:
    def process_order(self, order_id: str) -> dict:
        # Tự tạo connection — hardcoded dependency
        conn = sqlite3.connect("production.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
        order = cursor.fetchone()

        # Business logic
        total = order[2] * order[3]
        if total > 100:
            total *= 0.9  # 10% discount

        # Tự gửi email
        smtp = smtplib.SMTP("smtp.company.com")
        smtp.sendmail(
            "orders@company.com",
            "customer@example.com",
            f"Order {order_id}: ${total}"
        )

        # Tự update database
        cursor.execute("UPDATE orders SET total = ? WHERE id = ?", (total, order_id))
        conn.commit()

        return {"order_id": order_id, "total": total}
```

**Vấn đề với code này**:
1. Không thể test business logic riêng — phải có database thật
2. Không thể test discount logic — phải có order thật trong DB
3. Gửi email thật mỗi lần chạy test
4. Không thể mock/stub dependencies
5. Vi phạm SRP — một class làm quá nhiều việc

### Bước 1: TDD phát hiện vấn đề

Khi viết test trước, bạn sẽ thấy ngay code này không thể test được:

```python
# tests/test_order_processor.py
from src.order_processor import OrderProcessor


def test_apply_discount_for_large_orders():
    processor = OrderProcessor()
    # ❌ Làm sao để test? Cần DB thật có order với total > 100?
    # ❌ Làm sao tránh gửi email thật?
    # ❌ Làm sao kiểm tra discount logic mà không ảnh hưởng đến DB?
    pass  # Impossible to test!
```

### Bước 2: TDD buộc Dependency Injection

Vì không thể test code hiện tại, TDD buộc bạn phải **tái cấu trúc** — inject dependencies:

```python
# src/order_processor.py — TDD-driven design
from typing import Protocol
from dataclasses import dataclass


@dataclass
class Order:
    id: str
    customer_email: str
    item_price: float
    quantity: int
    total: float = 0.0


class OrderRepository(Protocol):
    """Abstraction cho database — tuân thủ DIP."""
    def find_by_id(self, order_id: str) -> Order | None: ...
    def update_total(self, order_id: str, total: float) -> None: ...


class EmailService(Protocol):
    """Abstraction cho email service."""
    def send_order_confirmation(self, email: str, order_id: str, total: float) -> None: ...


class DiscountPolicy(Protocol):
    """Abstraction cho discount logic — tuân thủ OCP."""
    def apply(self, total: float) -> float: ...


class OrderProcessor:
    def __init__(
        self,
        repository: OrderRepository,
        email_service: EmailService,
        discount_policy: DiscountPolicy,
    ):
        self._repository = repository
        self._email_service = email_service
        self._discount_policy = discount_policy

    def process_order(self, order_id: str) -> dict:
        order = self._repository.find_by_id(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        total = order.item_price * order.quantity
        total = self._discount_policy.apply(total)

        self._repository.update_total(order_id, total)
        self._email_service.send_order_confirmation(
            order.customer_email, order_id, total
        )

        return {"order_id": order_id, "total": total}
```

Bây giờ mọi dependency đều được inject — testable!

### Bước 3: Test với Fake/Mock

```python
# tests/conftest.py
import pytest
from dataclasses import dataclass
from typing import Optional
from src.order_processor import (
    Order, OrderRepository, EmailService, DiscountPolicy
)


class FakeOrderRepository:
    """Fake — in-memory implementation."""
    def __init__(self):
        self._orders: dict[str, Order] = {}

    def seed(self, order: Order) -> None:
        self._orders[order.id] = order

    def find_by_id(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def update_total(self, order_id: str, total: float) -> None:
        if order_id in self._orders:
            self._orders[order_id].total = total


class SpyEmailService:
    """Spy — ghi lại email đã gửi."""
    def __init__(self):
        self.sent_emails: list[tuple[str, str, float]] = []

    def send_order_confirmation(self, email: str, order_id: str, total: float) -> None:
        self.sent_emails.append((email, order_id, total))


class FixedDiscountPolicy:
    """Stub — discount cố định cho test."""
    def __init__(self, rate: float = 0.0):
        self._rate = rate

    def apply(self, total: float) -> float:
        return total * (1 - self._rate)


@pytest.fixture
def repo():
    return FakeOrderRepository()


@pytest.fixture
def email_spy():
    return SpyEmailService()


@pytest.fixture
def no_discount():
    return FixedDiscountPolicy(rate=0.0)


@pytest.fixture
def ten_percent_discount():
    return FixedDiscountPolicy(rate=0.1)


@pytest.fixture
def processor(repo, email_spy, no_discount):
    return OrderProcessor(repo, email_spy, no_discount)


@pytest.fixture
def sample_order():
    return Order(
        id="ORD-001",
        customer_email="customer@example.com",
        item_price=50.0,
        quantity=3,
    )
```

```python
# tests/test_order_processor.py
import pytest
from src.order_processor import OrderProcessor


class TestOrderProcessor:
    def test_process_order_calculates_total(self, processor, repo, sample_order):
        repo.seed(sample_order)
        result = processor.process_order("ORD-001")
        assert result["total"] == 150.0

    def test_process_order_applies_discount(
        self, repo, email_spy, ten_percent_discount, sample_order
    ):
        repo.seed(sample_order)
        processor = OrderProcessor(repo, email_spy, ten_percent_discount)
        result = processor.process_order("ORD-001")
        assert result["total"] == 135.0  # 150 - 10%

    def test_process_order_sends_email(
        self, processor, repo, sample_order
    ):
        repo.seed(sample_order)
        processor.process_order("ORD-001")
        assert len(email_spy.sent_emails) == 1
        assert email_spy.sent_emails[0] == (
            "customer@example.com", "ORD-001", 150.0
        )

    def test_process_order_updates_repository(
        self, processor, repo, sample_order
    ):
        repo.seed(sample_order)
        processor.process_order("ORD-001")
        assert repo.find_by_id("ORD-001").total == 150.0

    def test_process_order_raises_on_missing_order(self, processor, repo):
        with pytest.raises(ValueError, match="ORD-999 not found"):
            processor.process_order("ORD-999")
```

## SOLID áp dụng qua TDD

Hãy xem TDD dẫn dắt từng nguyên lý SOLID như thế nào.

### SRP — Single Responsibility Principle

**Vấn đề**: Khi viết test cho một class, nếu test quá phức tạp hoặc có nhiều "reason to change", class đó đang vi phạm SRP.

**TDD dẫn dắt**:

```python
# ❌ SRP violation detected by TDD
class InvoiceService:
    def generate(self, order_id: str) -> bytes:
        # Fetch order
        # Calculate totals
        # Apply taxes
        # Generate PDF
        # Send email
        # Save to database
        pass

# Test cho class này sẽ rất phức tạp:
def test_invoice_service():
    # Cần mock DB, PDF generator, email service, tax calculator...
    # Quá nhiều thứ — dấu hiệu SRP violation!

# ✅ SRP — tách thành nhiều class, mỗi class một responsibility
class InvoiceCalculator:
    def calculate(self, order: Order) -> InvoiceData: ...

class PdfGenerator:
    def generate(self, data: InvoiceData) -> bytes: ...

class InvoiceSender:
    def send(self, pdf: bytes, email: str) -> None: ...

# Test đơn giản hơn nhiều:
def test_invoice_calculator():
    calc = InvoiceCalculator()
    result = calc.calculate(sample_order)
    assert result.total == 150.0
```

**Dấu hiệu nhận biết SRP violation qua TDD**:

| Dấu hiệu trong test | Vấn đề SRP |
|---------------------|------------|
| Test setup quá dài (>10 dòng) | Class có quá nhiều dependency |
| Cần nhiều mock objects | Class làm quá nhiều việc |
| Test có nhiều assert khác loại | Nhiều responsibility trong một method |
| Mỗi lần requirement thay đổi, test cũng thay đổi | Class có nhiều "reason to change" |

### OCP — Open/Closed Principle

**Vấn đề**: Mỗi khi thêm behavior mới, bạn phải sửa code cũ — vi phạm OCP.

**TDD dẫn dắt**: Viết test cho behavior mới trước — behavior cũ vẫn pass (code cũ không cần sửa).

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class InvoiceData:
    items: List[dict]
    subtotal: float
    tax: float
    total: float


class TaxCalculator(ABC):
    """Abstraction cho phép thêm tax strategy mới (OCP)."""
    @abstractmethod
    def calculate(self, subtotal: float) -> float: ...


class VATTaxCalculator(TaxCalculator):
    """VAT 10%."""
    def calculate(self, subtotal: float) -> float:
        return subtotal * 0.10


class NoTaxCalculator(TaxCalculator):
    """No tax — cho special zones."""
    def calculate(self, subtotal: float) -> float:
        return 0.0


class InvoiceCalculator:
    def __init__(self, tax_calculator: TaxCalculator):
        self._tax = tax_calculator

    def calculate(self, items: List[dict]) -> InvoiceData:
        subtotal = sum(item["price"] * item["qty"] for item in items)
        tax = self._tax.calculate(subtotal)
        total = subtotal + tax
        return InvoiceData(items=items, subtotal=subtotal, tax=tax, total=total)
```

Test cho behavior cũ vẫn pass khi thêm behavior mới:

```python
# tests/test_invoice_calculator.py
import pytest


class TestVATInvoice:
    @pytest.fixture
    def calc(self):
        return InvoiceCalculator(VATTaxCalculator())

    def test_calculates_vat(self, calc):
        items = [{"price": 100.0, "qty": 2}]
        result = calc.calculate(items)
        assert result.subtotal == 200.0
        assert result.tax == 20.0  # 10% VAT
        assert result.total == 220.0


# Behavior mới — không sửa code cũ!
class TestTaxFreeZoneInvoice:
    @pytest.fixture
    def calc(self):
        return InvoiceCalculator(NoTaxCalculator())

    def test_no_tax_for_special_zone(self, calc):
        items = [{"price": 100.0, "qty": 2}]
        result = calc.calculate(items)
        assert result.subtotal == 200.0
        assert result.tax == 0.0
        assert result.total == 200.0
```

### LSP — Liskov Substitution Principle

**Vấn đề**: Subclass thay đổi behavior của base class một cách không mong đợi — test fail.

**TDD dẫn dắt**: Viết test contract cho base type, chạy với mọi implementation:

```python
class Repository(ABC):
    """Contract — tất cả implementation phải tuân theo."""
    @abstractmethod
    def save(self, entity: dict) -> None:
        """Lưu entity. Ném TypeError nếu entity không hợp lệ."""
        ...

    @abstractmethod
    def find_by_id(self, id: str) -> dict | None:
        """Tìm entity theo id. Trả về None nếu không tìm thấy."""
        ...


# Base test — chạy với MỌI implementation
class RepositoryContractTest:
    """Test contract — đảm bảo LSP."""
    @pytest.fixture
    def repo(self) -> Repository:
        raise NotImplementedError("Subclass must provide repo")

    def test_save_and_find(self, repo):
        repo.save({"id": "1", "name": "Alice"})
        result = repo.find_by_id("1")
        assert result is not None
        assert result["name"] == "Alice"

    def test_find_nonexistent_returns_none(self, repo):
        result = repo.find_by_id("nonexistent")
        assert result is None

    def test_save_raises_on_invalid_entity(self, repo):
        with pytest.raises(TypeError):
            repo.save(None)

    def test_save_updates_existing(self, repo):
        repo.save({"id": "1", "name": "Alice"})
        repo.save({"id": "1", "name": "Bob"})
        result = repo.find_by_id("1")
        assert result["name"] == "Bob"


# Concrete test — kế thừa contract test
class TestInMemoryRepository(RepositoryContractTest):
    @pytest.fixture
    def repo(self):
        return InMemoryRepository()


class TestSQLiteRepository(RepositoryContractTest):
    @pytest.fixture
    def repo(self):
        return SQLiteRepository(":memory:")
```

Nếu `TestSQLiteRepository` fail contract test → LSP violation — implementation không thay thế được cho base type.

### ISP — Interface Segregation Principle

**Vấn đề**: Interface quá lớn — implementation phải implement methods không dùng đến, test phải mock methods không cần.

**TDD dẫn dắt**: Nếu test cần mock quá nhiều methods không liên quan, interface đang vi phạm ISP.

```python
# ❌ ISP violation — "fat interface"
class BigOrderService(Protocol):
    def process_order(self, order_id: str) -> dict: ...
    def send_email(self, to: str, body: str) -> None: ...
    def generate_invoice(self, order_id: str) -> bytes: ...
    def update_inventory(self, product_id: str, qty: int) -> None: ...
    def calculate_shipping(self, address: str) -> float: ...
    def apply_coupon(self, code: str) -> float: ...

# Test phải mock tất cả:
def test_process_order():
    service = Mock(spec=BigOrderService)  # Phải mock 6 methods!
    ...

# ✅ ISP — interface nhỏ, chuyên biệt
class OrderProcessor(Protocol):
    def process_order(self, order_id: str) -> dict: ...

class EmailSender(Protocol):
    def send_email(self, to: str, body: str) -> None: ...

class InvoiceGenerator(Protocol):
    def generate_invoice(self, order_id: str) -> bytes: ...

class InventoryManager(Protocol):
    def update_inventory(self, product_id: str, qty: int) -> None: ...

class ShippingCalculator(Protocol):
    def calculate_shipping(self, address: str) -> float: ...

class CouponService(Protocol):
    def apply_coupon(self, code: str) -> float: ...

# Test chỉ mock những gì cần:
def test_process_order():
    processor = Mock(spec=OrderProcessor)  # Chỉ 1 method!
    ...
```

### DIP — Dependency Inversion Principle

**Vấn đề**: Module cấp cao phụ thuộc trực tiếp vào module cấp thấp — không thể test riêng, không thể thay đổi implementation.

**TDD dẫn dắt**: Khi viết test, bạn inject mock/fake → tự nhiên phụ thuộc vào abstraction (DIP):

```python
# ❌ DIP violation — high-level depends on low-level directly
class ReportService:
    def __init__(self):
        self._db = PostgreSQLConnection()  # Hardcoded!

    def generate_report(self) -> dict:
        data = self._db.query("SELECT * FROM sales")
        return {"total": sum(row["amount"] for row in data)}

# Không thể test nếu không có PostgreSQL!

# ✅ DIP — inject abstraction
class Database(Protocol):
    def query(self, sql: str) -> list[dict]: ...

class ReportService:
    def __init__(self, database: Database):
        self._db = database  # Phụ thuộc abstraction!

    def generate_report(self) -> dict:
        data = self._db.query("SELECT * FROM sales")
        return {"total": sum(row["amount"] for row in data)}

# Dễ dàng test với Fake:
class FakeDatabase:
    def __init__(self):
        self._data: list[dict] = []

    def seed(self, data: list[dict]) -> None:
        self._data = data

    def query(self, sql: str) -> list[dict]:
        return self._data

def test_report_service():
    db = FakeDatabase()
    db.seed([{"amount": 100}, {"amount": 200}, {"amount": 300}])
    service = ReportService(db)

    result = service.generate_report()
    assert result["total"] == 600
```

## Design Patterns sinh ra từ TDD

TDD tự nhiên dẫn đến một số design patterns:

### Strategy Pattern

Khi bạn inject discount policy, tax calculator, shipping calculator — bạn đang dùng Strategy pattern!

```python
class ShippingStrategy(ABC):
    @abstractmethod
    def calculate(self, weight: float, distance: float) -> float: ...

class StandardShipping(ShippingStrategy):
    def calculate(self, weight: float, distance: float) -> float:
        return weight * distance * 0.1

class ExpressShipping(ShippingStrategy):
    def calculate(self, weight: float, distance: float) -> float:
        return weight * distance * 0.3 + 10.0  # Flat $10 premium

class OrderService:
    def __init__(self, shipping: ShippingStrategy):
        self._shipping = shipping

    def checkout(self, order: dict) -> dict:
        shipping_cost = self._shipping.calculate(
            order["weight"], order["distance"]
        )
        return {**order, "shipping": shipping_cost}
```

### Factory Pattern

Khi test cần tạo objects phức tạp, bạn tạo factory:

```python
class TestDataFactory:
    @staticmethod
    def create_order(**overrides) -> Order:
        defaults = {
            "id": "ORD-DEFAULT",
            "customer_id": "CUST-DEFAULT",
            "items": [],
            "shipping_address": "123 Default St",
            "status": "pending",
        }
        defaults.update(overrides)
        return Order(**defaults)

    @staticmethod
    def create_customer(**overrides) -> Customer:
        defaults = {
            "id": "CUST-DEFAULT",
            "name": "Default Customer",
            "email": "default@test.com",
            "tier": "standard",
        }
        defaults.update(overrides)
        return Customer(**defaults)
```

### Template Method Pattern

Khi có nhiều test với cùng flow nhưng implementation khác nhau:

```python
class BaseOrderTest:
    """Template Method pattern cho test."""
    @pytest.fixture
    def order(self):
        return self.create_sample_order()

    def create_sample_order(self) -> Order:
        raise NotImplementedError

    def test_order_total(self, order):
        assert order.calculate_total() > 0

    def test_order_items(self, order):
        assert len(order.items) > 0


class TestPhysicalOrder(BaseOrderTest):
    def create_sample_order(self) -> Order:
        return PhysicalOrder(items=[...], shipping_address="...")


class TestDigitalOrder(BaseOrderTest):
    def create_sample_order(self) -> Order:
        return DigitalOrder(items=[...], download_link="...")
```

## Architectural Patterns cho Testability

### Hexagonal Architecture (Ports & Adapters)

TDD tự nhiên dẫn đến Hexagonal Architecture — business logic ở trung tâm, infrastructure ở rìa:

```python
# Domain — business logic (core)
from dataclasses import dataclass
from typing import Protocol
from decimal import Decimal


@dataclass
class Money:
    amount: Decimal
    currency: str

    def __add__(self, other: "Money") -> "Money":
        assert self.currency == other.currency
        return Money(self.amount + other.amount, self.currency)


# Port — abstraction (inbound)
class OrderService(Protocol):
    def place_order(self, customer_id: str, items: list[dict]) -> dict: ...


# Port — abstraction (outbound)
class PaymentGateway(Protocol):
    def charge(self, amount: Money, token: str) -> dict: ...


class InventorySystem(Protocol):
    def reserve(self, product_id: str, quantity: int) -> bool: ...


# Adapter — domain implementation
class OrderServiceImpl:
    def __init__(self, payment: PaymentGateway, inventory: InventorySystem):
        self._payment = payment
        self._inventory = inventory

    def place_order(self, customer_id: str, items: list[dict]) -> dict:
        total = Money(Decimal("0.00"), "USD")
        for item in items:
            total += Money(Decimal(str(item["price"])), "USD")
            if not self._inventory.reserve(item["id"], item["qty"]):
                raise ValueError(f"Insufficient inventory for {item['id']}")

        result = self._payment.charge(total, "tok_visa")
        return {"status": result["status"], "total": str(total.amount)}
```

Test domain core hoàn toàn độc lập với infrastructure:

```python
class FakePaymentGateway:
    def __init__(self):
        self.charges = []

    def charge(self, amount: Money, token: str) -> dict:
        self.charges.append((amount, token))
        return {"status": "success", "transaction_id": "TXN-MOCK"}


class FakeInventorySystem:
    def __init__(self):
        self._available = {}

    def set_available(self, product_id: str, qty: int):
        self._available[product_id] = qty

    def reserve(self, product_id: str, quantity: int) -> bool:
        avail = self._available.get(product_id, 0)
        if quantity <= avail:
            self._available[product_id] = avail - quantity
            return True
        return False
```

## Refactoring cho Testability

### Extract Interface

Khi một class cụ thể khó test, extract interface:

```python
# Before — khó test
class StripePayment:
    def charge(self, amount: float, token: str) -> dict:
        import stripe
        result = stripe.Charge.create(amount=int(amount * 100), source=token)
        return {"status": "success", "id": result.id}

# After — extract interface
class PaymentProcessor(Protocol):
    def charge(self, amount: float, token: str) -> dict: ...

class StripePayment(PaymentProcessor):
    def charge(self, amount: float, token: str) -> dict:
        import stripe
        result = stripe.Charge.create(amount=int(amount * 100), source=token)
        return {"status": "success", "id": result.id}

# Inject abstraction
class CheckoutService:
    def __init__(self, payment: PaymentProcessor):
        self._payment = payment

# Test with mock
def test_checkout():
    payment = Mock(spec=PaymentProcessor)
    payment.charge.return_value = {"status": "success"}
    service = CheckoutService(payment)
    ...
```

### Break Method Overload

Khi một method quá lớn và khó test, tách thành nhiều method nhỏ:

```python
# Before — khó test
def process(self, order_id: str) -> dict:
    order = self._db.find(order_id)
    total = sum(item.price * item.qty for item in order.items)
    if order.customer.tier == "vip":
        total *= 0.9
    if total > 1000:
        total *= 0.95
    tax = total * 0.08
    final = total + tax
    self._db.update(order_id, final)
    self._email.send(order.customer.email, final)
    return {"total": final}

# After — dễ test từng phần
def process(self, order_id: str) -> dict:
    order = self._fetch_order(order_id)
    total = self._calculate_total(order)
    self._save_total(order_id, total)
    self._notify_customer(order, total)
    return {"total": total}

def _fetch_order(self, order_id: str) -> Order:
    return self._db.find(order_id)

def _calculate_total(self, order: Order) -> float:
    subtotal = self._calculate_subtotal(order)
    subtotal = self._apply_vip_discount(order, subtotal)
    subtotal = self._apply_bulk_discount(subtotal)
    tax = self._calculate_tax(subtotal)
    return subtotal + tax

def _calculate_subtotal(self, order: Order) -> float:
    return sum(item.price * item.qty for item in order.items)

def _apply_vip_discount(self, order: Order, total: float) -> float:
    return total * 0.9 if order.customer.tier == "vip" else total

def _apply_bulk_discount(self, total: float) -> float:
    return total * 0.95 if total > 1000 else total

def _calculate_tax(self, total: float) -> float:
    return total * 0.08
```

## Anti-patterns: Test-induced damage

Marco Emrich cảnh báo về "Test-Induced Damage" — thiết kế tồi do TDD áp dụng sai:

| Anti-pattern | Biểu hiện | Giải pháp |
|-------------|-----------|-----------|
| **Mock Everything** | Mock cả những thứ không cần | Chỉ mock boundary (DB, API, filesystem) |
| **Testing Private Methods** | Test implementation details | Test behavior public, private là implementation |
| **Over-abstraction** | Interface cho mọi thứ | Interface only khi cần thay thế implementation |
| **Test-infected Design** | Thiết kế chỉ để dễ test | TDD design phải phục vụ production, không chỉ test |
| **Giant Fixtures** | Fixture quá lớn, khó hiểu | Dùng Test Data Builder |

## Tổng kết

TDD và SOLID có mối quan hệ cộng sinh:

| SOLID | Cách TDD dẫn dắt |
|-------|------------------|
| **SRP** | Test khó viết → class làm quá nhiều → tách nhỏ |
| **OCP** | Test behavior mới → code cũ không cần sửa → abstraction |
| **LSP** | Contract test → mọi implementation phải thay thế được |
| **ISP** | Mock quá nhiều method → interface quá lớn → tách nhỏ |
| **DIP** | Không thể test → inject dependency qua abstraction |

Khi bạn TDD đúng, bạn sẽ tự nhiên có thiết kế tuân thủ SOLID — không cần cố gắng áp dụng SOLID một cách cứng nhắc. Trang tiếp theo chúng ta sẽ áp dụng tất cả vào một dự án thực tế.

## Tài liệu tham khảo

- Robert C. Martin, *"Clean Architecture: A Craftsman's Guide to Software Structure and Design"* (2017)
- Steve Freeman & Nat Pryce, *"Growing Object-Oriented Software Guided by Tests"* (2009)
- Marco Emrich, *"Test-Induced Damage"* (2013) — https://www.slideshare.net/MarcoEmrich/testinduced-damage
- Alistair Cockburn, *"Hexagonal Architecture"* (2005)
