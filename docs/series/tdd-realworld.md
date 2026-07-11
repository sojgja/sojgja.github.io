---
id: tdd-realworld
title: TDD trong dự án thực tế
sidebar_label: 🔴 TDD Real-World
sidebar_position: 59
---

# TDD trong dự án thực tế

> *"In real projects, TDD is not a luxury — it's a survival mechanism."* — **Michael Feathers**

Trong bài này, chúng ta xây dựng một hệ thống **E-commerce Order Processing** hoàn chỉnh bằng TDD. Đây là dự án mô phỏng real-world với multiple modules, database mocking, payment gateway integration, và business logic phức tạp.

## Kiến trúc Project

```
ecommerce/
├── src/
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models.py          # Domain entities
│   │   ├── value_objects.py    # Value objects
│   │   ├── services.py         # Domain services
│   │   └── exceptions.py       # Domain exceptions
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── repositories.py     # Repository interfaces
│   │   └── gateways.py         # External service interfaces
│   └── application/
│       ├── __init__.py
│       ├── order_service.py    # Application service
│       ├── payment_service.py  # Payment orchestration
│       └── inventory_service.py # Inventory management
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   └── test_services.py
│   ├── application/
│   │   ├── __init__.py
│   │   └── test_order_service.py
│   └── integration/
│       ├── __init__.py
│       └── test_full_flow.py
└── pyproject.toml
```

## Bước 1: Domain Layer

### Value Objects

```python
# src/domain/value_objects.py
from dataclasses import dataclass
from decimal import Decimal
from typing import List


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "USD"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Money amount cannot be negative")

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

    def __mul__(self, factor: Decimal) -> "Money":
        return Money((self.amount * factor).quantize(Decimal("0.01")), self.currency)

    def __gt__(self, other: "Money") -> bool:
        return self.amount > other.amount

    def __repr__(self) -> str:
        return f"{self.amount:.2f} {self.currency}"
```

Tests (TDD — viết trước code):

```python
# tests/domain/test_value_objects.py
import pytest
from decimal import Decimal
from src.domain.value_objects import Money


class TestMoney:
    def test_create_money_with_valid_amount(self):
        money = Money(Decimal("100.00"))
        assert money.amount == Decimal("100.00")
        assert money.currency == "USD"

    def test_create_money_with_different_currency(self):
        money = Money(Decimal("50.00"), "EUR")
        assert money.currency == "EUR"

    def test_raises_on_negative_amount(self):
        with pytest.raises(ValueError, match="Money amount cannot be negative"):
            Money(Decimal("-10.00"))

    def test_add_two_money_objects(self):
        a = Money(Decimal("100.00"))
        b = Money(Decimal("50.00"))
        result = a + b
        assert result.amount == Decimal("150.00")

    def test_add_raises_on_different_currencies(self):
        usd = Money(Decimal("100.00"), "USD")
        eur = Money(Decimal("50.00"), "EUR")
        with pytest.raises(ValueError, match="Cannot add different currencies"):
            _ = usd + eur

    def test_multiply_by_factor(self):
        money = Money(Decimal("100.00"))
        result = money * Decimal("0.9")
        assert result.amount == Decimal("90.00")

    def test_compare_greater_than(self):
        a = Money(Decimal("200.00"))
        b = Money(Decimal("100.00"))
        assert a > b
        assert not b > a

    def test_money_is_immutable(self):
        money = Money(Decimal("100.00"))
        with pytest.raises(AttributeError):
            money.amount = Decimal("200.00")  # frozen dataclass
```

### Domain Entities

```python
# src/domain/models.py
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from .value_objects import Money
from .exceptions import (
    InsufficientInventoryError,
    InvalidOrderStateError,
    PaymentDeclinedError,
)


@dataclass
class Product:
    id: str
    name: str
    price: Money
    weight_kg: float
    category: str
    is_digital: bool = False


@dataclass
class OrderItem:
    product: Product
    quantity: int

    @property
    def subtotal(self) -> Money:
        return self.product.price * Decimal(str(self.quantity))


@dataclass
class Customer:
    id: str
    name: str
    email: str
    tier: str = "standard"  # standard, vip, premium

    @property
    def discount_rate(self) -> Decimal:
        rates = {"standard": Decimal("0.0"), "vip": Decimal("0.1"), "premium": Decimal("0.2")}
        return rates.get(self.tier, Decimal("0.0"))


class OrderStatus:
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class Order:
    id: str
    customer: Customer
    items: List[OrderItem]
    status: str = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    shipping_address: Optional[str] = None
    _total: Optional[Money] = None

    @property
    def subtotal(self) -> Money:
        if not self.items:
            return Money(Decimal("0.00"))
        total = self.items[0].subtotal
        for item in self.items[1:]:
            total = total + item.subtotal
        return total

    @property
    def discount(self) -> Money:
        return self.subtotal * self.customer.discount_rate

    @property
    def total(self) -> Money:
        if self._total is None:
            self._total = self.subtotal - self.discount
        return self._total

    def confirm(self) -> None:
        if self.status != OrderStatus.PENDING:
            raise InvalidOrderStateError(
                f"Cannot confirm order in state: {self.status}"
            )
        self.status = OrderStatus.CONFIRMED

    def cancel(self) -> None:
        if self.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
            raise InvalidOrderStateError(
                f"Cannot cancel order in state: {self.status}"
            )
        self.status = OrderStatus.CANCELLED
```

### Domain Exceptions

```python
# src/domain/exceptions.py
class DomainError(Exception):
    """Base domain exception."""
    pass


class InsufficientInventoryError(DomainError):
    def __init__(self, product_id: str, requested: int, available: int):
        self.product_id = product_id
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient inventory for {product_id}: "
            f"requested {requested}, available {available}"
        )


class InvalidOrderStateError(DomainError):
    pass


class PaymentDeclinedError(DomainError):
    def __init__(self, transaction_id: str, reason: str):
        self.transaction_id = transaction_id
        self.reason = reason
        super().__init__(f"Payment declined: {reason} (tx: {transaction_id})")
```

Tests cho Order entity:

```python
# tests/domain/test_models.py
import pytest
from decimal import Decimal
from src.domain.models import (
    Product, OrderItem, Customer, Order, OrderStatus
)
from src.domain.value_objects import Money
from src.domain.exceptions import InvalidOrderStateError


@pytest.fixture
def sample_product():
    return Product(
        id="PROD-001",
        name="Wireless Mouse",
        price=Money(Decimal("25.00")),
        weight_kg=0.2,
        category="electronics",
    )


@pytest.fixture
def sample_customer():
    return Customer(
        id="CUST-001",
        name="Alice Nguyen",
        email="alice@example.com",
        tier="vip",
    )


@pytest.fixture
def sample_order(sample_product, sample_customer):
    item = OrderItem(product=sample_product, quantity=3)
    return Order(
        id="ORD-001",
        customer=sample_customer,
        items=[item],
    )


class TestOrder:
    def test_subtotal_calculation(self, sample_order, sample_product):
        expected = sample_product.price * Decimal("3")
        assert sample_order.subtotal == expected

    def test_discount_for_vip_customer(self, sample_order):
        # VIP discount = 10%
        expected_discount = sample_order.subtotal * Decimal("0.1")
        assert sample_order.discount == expected_discount

    def test_total_after_discount(self, sample_order):
        expected = sample_order.subtotal - sample_order.discount
        assert sample_order.total == expected

    def test_standard_customer_no_discount(self, sample_product):
        customer = Customer(id="CUST-002", name="Bob", email="bob@test.com", tier="standard")
        item = OrderItem(product=sample_product, quantity=1)
        order = Order(id="ORD-002", customer=customer, items=[item])
        assert order.discount == Money(Decimal("0.00"))

    def test_confirm_pending_order(self, sample_order):
        assert sample_order.status == OrderStatus.PENDING
        sample_order.confirm()
        assert sample_order.status == OrderStatus.CONFIRMED

    def test_cannot_confirm_non_pending_order(self, sample_order):
        sample_order.confirm()
        with pytest.raises(InvalidOrderStateError, match="Cannot confirm"):
            sample_order.confirm()

    def test_cancel_pending_order(self, sample_order):
        sample_order.cancel()
        assert sample_order.status == OrderStatus.CANCELLED

    def test_cannot_cancel_shipped_order(self, sample_order):
        sample_order.confirm()
        sample_order.status = OrderStatus.SHIPPED
        with pytest.raises(InvalidOrderStateError, match="Cannot cancel"):
            sample_order.cancel()

    def test_total_is_cached_after_calculation(self, sample_order):
        total1 = sample_order.total
        total2 = sample_order.total
        assert total1 is total2  # Same cached object
```

## Bước 2: Port Interfaces

```python
# src/ports/repositories.py
from typing import Protocol, Optional
from src.domain.models import Order, Product, Customer


class OrderRepository(Protocol):
    def save(self, order: Order) -> None: ...
    def find_by_id(self, order_id: str) -> Optional[Order]: ...
    def find_by_customer(self, customer_id: str) -> list[Order]: ...
    def update_status(self, order_id: str, status: str) -> None: ...


class ProductRepository(Protocol):
    def find_by_id(self, product_id: str) -> Optional[Product]: ...
    def update_stock(self, product_id: str, quantity: int) -> None: ...
    def get_stock(self, product_id: str) -> int: ...


class CustomerRepository(Protocol):
    def find_by_id(self, customer_id: str) -> Optional[Customer]: ...
```

```python
# src/ports/gateways.py
from typing import Protocol
from src.domain.value_objects import Money


class PaymentGateway(Protocol):
    def charge(self, amount: Money, token: str) -> dict: ...
    def refund(self, transaction_id: str, amount: Money) -> dict: ...


class EmailService(Protocol):
    def send_order_confirmation(self, email: str, order_id: str) -> None: ...
    def send_shipping_notification(self, email: str, order_id: str, tracking: str) -> None: ...


class InventorySystem(Protocol):
    def reserve(self, product_id: str, quantity: int) -> bool: ...
    def release(self, product_id: str, quantity: int) -> None: ...
```

## Bước 3: Application Services

```python
# src/application/order_service.py
from decimal import Decimal
from typing import Optional
from src.domain.models import Order, OrderItem, Customer, Product
from src.domain.value_objects import Money
from src.domain.exceptions import (
    InsufficientInventoryError,
    PaymentDeclinedError,
    DomainError,
)
from src.ports.repositories import OrderRepository, ProductRepository, CustomerRepository
from src.ports.gateways import PaymentGateway, EmailService, InventorySystem


class OrderService:
    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo: ProductRepository,
        customer_repo: CustomerRepository,
        payment_gateway: PaymentGateway,
        email_service: EmailService,
        inventory_system: InventorySystem,
    ):
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._customer_repo = customer_repo
        self._payment = payment_gateway
        self._email = email_service
        self._inventory = inventory_system

    def create_order(
        self,
        customer_id: str,
        items: list[dict],
        shipping_address: Optional[str] = None,
    ) -> Order:
        customer = self._customer_repo.find_by_id(customer_id)
        if customer is None:
            raise ValueError(f"Customer {customer_id} not found")

        order_items = []
        for item_data in items:
            product = self._product_repo.find_by_id(item_data["product_id"])
            if product is None:
                raise ValueError(f"Product {item_data['product_id']} not found")
            order_items.append(OrderItem(product=product, quantity=item_data["quantity"]))

        order_id = f"ORD-{hash(str(items)) % 100000:05d}"
        order = Order(
            id=order_id,
            customer=customer,
            items=order_items,
            shipping_address=shipping_address,
        )
        self._order_repo.save(order)
        return order

    def place_order(self, order_id: str, payment_token: str) -> dict:
        order = self._order_repo.find_by_id(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        # Reserve inventory
        for item in order.items:
            available = self._product_repo.get_stock(item.product.id)
            if available < item.quantity:
                raise InsufficientInventoryError(
                    item.product.id, item.quantity, available
                )
            self._inventory.reserve(item.product.id, item.quantity)

        # Process payment
        try:
            payment_result = self._payment.charge(order.total, payment_token)
        except Exception as e:
            raise PaymentDeclinedError("unknown", str(e))

        if payment_result.get("status") != "success":
            self._release_inventory(order)
            raise PaymentDeclinedError(
                payment_result.get("transaction_id", "unknown"),
                payment_result.get("reason", "unknown error"),
            )

        # Confirm order
        order.confirm()
        self._order_repo.update_status(order_id, Order.STATUS_CONFIRMED)
        self._email.send_order_confirmation(order.customer.email, order_id)

        return {
            "order_id": order_id,
            "total": str(order.total.amount),
            "transaction_id": payment_result.get("transaction_id"),
            "status": "confirmed",
        }

    def _release_inventory(self, order: Order) -> None:
        for item in order.items:
            self._inventory.release(item.product.id, item.quantity)
```

## Bước 4: Test Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
from decimal import Decimal
from unittest.mock import Mock, create_autospec
from src.domain.models import Product, Customer, OrderItem, Order
from src.domain.value_objects import Money
from src.ports.repositories import OrderRepository, ProductRepository, CustomerRepository
from src.ports.gateways import PaymentGateway, EmailService, InventorySystem


@pytest.fixture
def sample_product():
    return Product(
        id="PROD-001",
        name="Wireless Mouse",
        price=Money(Decimal("25.00")),
        weight_kg=0.2,
        category="electronics",
    )


@pytest.fixture
def sample_product2():
    return Product(
        id="PROD-002",
        name="Mechanical Keyboard",
        price=Money(Decimal("89.99")),
        weight_kg=1.2,
        category="electronics",
    )


@pytest.fixture
def sample_customer():
    return Customer(
        id="CUST-001",
        name="Alice Nguyen",
        email="alice@example.com",
        tier="vip",
    )


@pytest.fixture
def mock_order_repo():
    return create_autospec(OrderRepository)


@pytest.fixture
def mock_product_repo(sample_product, sample_product2):
    repo = create_autospec(ProductRepository)
    repo.find_by_id.side_effect = lambda pid: {
        "PROD-001": sample_product,
        "PROD-002": sample_product2,
    }.get(pid)
    repo.get_stock.return_value = 100
    return repo


@pytest.fixture
def mock_customer_repo(sample_customer):
    repo = create_autospec(CustomerRepository)
    repo.find_by_id.return_value = sample_customer
    return repo


@pytest.fixture
def mock_payment_gateway():
    gateway = create_autospec(PaymentGateway)
    gateway.charge.return_value = {
        "status": "success",
        "transaction_id": "TXN-MOCK-001",
    }
    return gateway


@pytest.fixture
def mock_email_service():
    return create_autospec(EmailService)


@pytest.fixture
def mock_inventory_system():
    inventory = create_autospec(InventorySystem)
    inventory.reserve.return_value = True
    return inventory


@pytest.fixture
def order_service(
    mock_order_repo,
    mock_product_repo,
    mock_customer_repo,
    mock_payment_gateway,
    mock_email_service,
    mock_inventory_system,
):
    from src.application.order_service import OrderService
    return OrderService(
        order_repo=mock_order_repo,
        product_repo=mock_product_repo,
        customer_repo=mock_customer_repo,
        payment_gateway=mock_payment_gateway,
        email_service=mock_email_service,
        inventory_system=mock_inventory_system,
    )


@pytest.fixture
def sample_order(sample_customer, sample_product):
    return Order(
        id="ORD-001",
        customer=sample_customer,
        items=[OrderItem(product=sample_product, quantity=2)],
        shipping_address="123 Main St",
    )
```

## Bước 5: Application Service Tests

```python
# tests/application/test_order_service.py
import pytest
from decimal import Decimal
from unittest.mock import ANY
from src.domain.value_objects import Money
from src.domain.exceptions import (
    InsufficientInventoryError,
    PaymentDeclinedError,
)


class TestCreateOrder:
    def test_create_order_successfully(self, order_service, mock_customer_repo,
                                       mock_product_repo, mock_order_repo, sample_customer):
        result = order_service.create_order(
            customer_id="CUST-001",
            items=[{"product_id": "PROD-001", "quantity": 2}],
        )
        assert result.customer == sample_customer
        assert len(result.items) == 1
        assert result.items[0].quantity == 2
        mock_order_repo.save.assert_called_once()

    def test_create_order_raises_on_unknown_customer(self, order_service, mock_customer_repo):
        mock_customer_repo.find_by_id.return_value = None
        with pytest.raises(ValueError, match="Customer CUST-999 not found"):
            order_service.create_order(
                customer_id="CUST-999",
                items=[{"product_id": "PROD-001", "quantity": 1}],
            )

    def test_create_order_raises_on_unknown_product(self, order_service, mock_product_repo):
        mock_product_repo.find_by_id.return_value = None
        with pytest.raises(ValueError, match="Product PROD-999 not found"):
            order_service.create_order(
                customer_id="CUST-001",
                items=[{"product_id": "PROD-999", "quantity": 1}],
            )


class TestPlaceOrder:
    def test_place_order_successfully(self, order_service, mock_order_repo,
                                      mock_payment_gateway, mock_email_service,
                                      sample_order):
        mock_order_repo.find_by_id.return_value = sample_order

        result = order_service.place_order("ORD-001", "tok_visa")

        assert result["status"] == "confirmed"
        assert result["transaction_id"] == "TXN-MOCK-001"
        mock_payment_gateway.charge.assert_called_once_with(
            sample_order.total, "tok_visa"
        )
        mock_email_service.send_order_confirmation.assert_called_once_with(
            "alice@example.com", "ORD-001"
        )

    def test_place_order_fails_on_insufficient_inventory(
        self, order_service, mock_order_repo, mock_inventory_system, sample_order
    ):
        mock_order_repo.find_by_id.return_value = sample_order
        mock_inventory_system.reserve.return_value = False
        mock_product_repo = order_service._product_repo
        mock_product_repo.get_stock.return_value = 0

        with pytest.raises(InsufficientInventoryError):
            order_service.place_order("ORD-001", "tok_visa")

    def test_place_order_fails_on_payment_declined(
        self, order_service, mock_order_repo, mock_payment_gateway, sample_order
    ):
        mock_order_repo.find_by_id.return_value = sample_order
        mock_payment_gateway.charge.return_value = {
            "status": "failure",
            "reason": "insufficient_funds",
            "transaction_id": "TXN-BAD",
        }

        with pytest.raises(PaymentDeclinedError, match="insufficient_funds"):
            order_service.place_order("ORD-001", "tok_bad")

    def test_place_order_releases_inventory_on_payment_failure(
        self, order_service, mock_order_repo, mock_payment_gateway,
        mock_inventory_system, sample_order
    ):
        mock_order_repo.find_by_id.return_value = sample_order
        mock_payment_gateway.charge.return_value = {
            "status": "failure",
            "reason": "insufficient_funds",
        }

        with pytest.raises(PaymentDeclinedError):
            order_service.place_order("ORD-001", "tok_bad")

        # Inventory phải được release
        assert mock_inventory_system.release.call_count == 1

    def test_place_order_raises_on_unknown_order(self, order_service, mock_order_repo):
        mock_order_repo.find_by_id.return_value = None
        with pytest.raises(ValueError, match="Order ORD-999 not found"):
            order_service.place_order("ORD-999", "tok_visa")

    def test_place_order_updates_status_to_confirmed(
        self, order_service, mock_order_repo, sample_order
    ):
        mock_order_repo.find_by_id.return_value = sample_order
        order_service.place_order("ORD-001", "tok_visa")
        mock_order_repo.update_status.assert_called_once_with(
            "ORD-001", "confirmed"
        )

    def test_place_order_handles_gateway_exception(
        self, order_service, mock_order_repo, mock_payment_gateway, sample_order
    ):
        mock_order_repo.find_by_id.return_value = sample_order
        mock_payment_gateway.charge.side_effect = ConnectionError("API timeout")

        with pytest.raises(PaymentDeclinedError, match="API timeout"):
            order_service.place_order("ORD-001", "tok_visa")
```

## Bước 6: Fake Repositories cho Integration Tests

```python
# tests/integration/conftest.py
import pytest
from datetime import datetime
from decimal import Decimal
from typing import Optional
from src.domain.models import Order, Product, Customer, OrderItem, OrderStatus
from src.domain.value_objects import Money


class FakeOrderRepository:
    """In-memory OrderRepository for integration tests."""
    def __init__(self):
        self._orders: dict[str, Order] = {}

    def save(self, order: Order) -> None:
        self._orders[order.id] = order

    def find_by_id(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def find_by_customer(self, customer_id: str) -> list[Order]:
        return [o for o in self._orders.values() if o.customer.id == customer_id]

    def update_status(self, order_id: str, status: str) -> None:
        if order_id in self._orders:
            self._orders[order_id].status = status

    def clear(self) -> None:
        self._orders.clear()


class FakeProductRepository:
    """In-memory ProductRepository with stock tracking."""
    def __init__(self):
        self._products: dict[str, Product] = {}
        self._stock: dict[str, int] = {}

    def add_product(self, product: Product, stock: int = 0) -> None:
        self._products[product.id] = product
        self._stock[product.id] = stock

    def find_by_id(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def update_stock(self, product_id: str, quantity: int) -> None:
        self._stock[product_id] = quantity

    def get_stock(self, product_id: str) -> int:
        return self._stock.get(product_id, 0)


class FakePaymentGateway:
    """Simulated payment gateway — controllable for tests."""
    def __init__(self):
        self._should_fail = False
        self._failure_reason = ""
        self.charges = []

    def set_failure(self, reason: str = "card_declined") -> None:
        self._should_fail = True
        self._failure_reason = reason

    def charge(self, amount: Money, token: str) -> dict:
        self.charges.append((amount, token))
        if self._should_fail:
            return {
                "status": "failure",
                "reason": self._failure_reason,
                "transaction_id": "TXN-FAIL",
            }
        return {
            "status": "success",
            "transaction_id": f"TXN-{len(self.charges):05d}",
        }

    def refund(self, transaction_id: str, amount: Money) -> dict:
        return {"status": "success", "refund_id": f"REF-{transaction_id}"}


class FakeEmailService:
    """In-memory email tracking for tests."""
    def __init__(self):
        self.sent_emails = []

    def send_order_confirmation(self, email: str, order_id: str) -> None:
        self.sent_emails.append(("confirmation", email, order_id))

    def send_shipping_notification(self, email: str, order_id: str, tracking: str) -> None:
        self.sent_emails.append(("shipping", email, order_id, tracking))


class FakeInventorySystem:
    """In-memory inventory tracking."""
    def __init__(self):
        self._reserved: dict[str, int] = {}

    def reserve(self, product_id: str, quantity: int) -> bool:
        current = self._reserved.get(product_id, 0)
        self._reserved[product_id] = current + quantity
        return True

    def release(self, product_id: str, quantity: int) -> None:
        current = self._reserved.get(product_id, 0)
        self._reserved[product_id] = max(0, current - quantity)

    def get_reserved(self, product_id: str) -> int:
        return self._reserved.get(product_id, 0)
```

## Bước 7: Full Flow Integration Test

```python
# tests/integration/test_full_flow.py
import pytest
from decimal import Decimal
from src.application.order_service import OrderService
from src.domain.exceptions import PaymentDeclinedError, InsufficientInventoryError
from .conftest import (
    FakeOrderRepository, FakeProductRepository, FakePaymentGateway,
    FakeEmailService, FakeInventorySystem,
)


@pytest.fixture
def real_order_service(sample_product, sample_product2, sample_customer):
    order_repo = FakeOrderRepository()
    product_repo = FakeProductRepository()
    product_repo.add_product(sample_product, stock=10)
    product_repo.add_product(sample_product2, stock=5)

    customer_repo = FakeCustomerRepository()
    customer_repo.add_customer(sample_customer)

    payment = FakePaymentGateway()
    email = FakeEmailService()
    inventory = FakeInventorySystem()

    return (
        OrderService(order_repo, product_repo, customer_repo, payment, email, inventory),
        order_repo, product_repo, payment, email, inventory,
    )


class TestFullOrderFlow:
    def test_complete_happy_path(self, real_order_service):
        service, order_repo, product_repo, payment, email, inventory = real_order_service

        # Create order
        order = service.create_order(
            customer_id="CUST-001",
            items=[
                {"product_id": "PROD-001", "quantity": 2},
                {"product_id": "PROD-002", "quantity": 1},
            ],
            shipping_address="123 Main St",
        )
        assert order.id is not None
        assert len(order.items) == 2

        # Place order
        result = service.place_order(order.id, "tok_visa")
        assert result["status"] == "confirmed"
        assert result["transaction_id"].startswith("TXN-")

        # Verify inventory reserved
        updated_order = order_repo.find_by_id(order.id)
        assert updated_order.status == "confirmed"

        # Verify email sent
        assert len(email.sent_emails) == 1
        assert email.sent_emails[0][0] == "confirmation"

    def test_payment_failure_releases_inventory(self, real_order_service):
        service, order_repo, product_repo, payment, email, inventory = real_order_service
        payment.set_failure("insufficient_funds")

        order = service.create_order(
            customer_id="CUST-001",
            items=[{"product_id": "PROD-001", "quantity": 2}],
        )

        with pytest.raises(PaymentDeclinedError):
            service.place_order(order.id, "tok_bad")

        # Inventory released
        assert inventory.get_reserved("PROD-001") == 0

    def test_insufficient_inventory(self, real_order_service):
        service, order_repo, product_repo, payment, email, inventory = real_order_service

        # Set low stock
        product_repo.update_stock("PROD-001", 1)

        order = service.create_order(
            customer_id="CUST-001",
            items=[{"product_id": "PROD-001", "quantity": 5}],
        )

        with pytest.raises(InsufficientInventoryError):
            service.place_order(order.id, "tok_visa")

    def test_multiple_orders_same_customer(self, real_order_service):
        service, order_repo, product_repo, payment, email, inventory = real_order_service

        order1 = service.create_order(
            customer_id="CUST-001",
            items=[{"product_id": "PROD-001", "quantity": 1}],
        )
        order2 = service.create_order(
            customer_id="CUST-001",
            items=[{"product_id": "PROD-002", "quantity": 2}],
        )

        result1 = service.place_order(order1.id, "tok_visa")
        result2 = service.place_order(order2.id, "tok_visa")

        assert result1["status"] == "confirmed"
        assert result2["status"] == "confirmed"

        customer_orders = order_repo.find_by_customer("CUST-001")
        assert len(customer_orders) == 2
```

## Bước 8: Running Tests

```text
$ pytest -v --tb=short
============================= test session starts =============================
collected 24 items

tests/domain/test_value_objects.py::TestMoney::test_create_money_with_valid_amount PASSED
tests/domain/test_value_objects.py::TestMoney::test_create_money_with_different_currency PASSED
tests/domain/test_value_objects.py::TestMoney::test_raises_on_negative_amount PASSED
tests/domain/test_value_objects.py::TestMoney::test_add_two_money_objects PASSED
tests/domain/test_value_objects.py::TestMoney::test_add_raises_on_different_currencies PASSED
tests/domain/test_value_objects.py::TestMoney::test_multiply_by_factor PASSED
tests/domain/test_value_objects.py::TestMoney::test_compare_greater_than PASSED
tests/domain/test_value_objects.py::TestMoney::test_money_is_immutable PASSED
tests/domain/test_models.py::TestOrder::test_subtotal_calculation PASSED
tests/domain/test_models.py::TestOrder::test_discount_for_vip_customer PASSED
tests/domain/test_models.py::TestOrder::test_total_after_discount PASSED
tests/domain/test_models.py::TestOrder::test_standard_customer_no_discount PASSED
tests/domain/test_models.py::TestOrder::test_confirm_pending_order PASSED
tests/domain/test_models.py::TestOrder::test_cannot_confirm_non_pending_order PASSED
tests/domain/test_models.py::TestOrder::test_cancel_pending_order PASSED
tests/domain/test_models.py::TestOrder::test_cannot_cancel_shipped_order PASSED
tests/domain/test_models.py::TestOrder::test_total_is_cached_after_calculation PASSED
tests/application/test_order_service.py::TestCreateOrder::test_create_order_successfully PASSED
tests/application/test_order_service.py::TestCreateOrder::test_create_order_raises_on_unknown_customer PASSED
tests/application/test_order_service.py::TestCreateOrder::test_create_order_raises_on_unknown_product PASSED
tests/application/test_order_service.py::TestPlaceOrder::test_place_order_successfully PASSED
tests/application/test_order_service.py::TestPlaceOrder::test_place_order_fails_on_insufficient_inventory PASSED
tests/application/test_order_service.py::TestPlaceOrder::test_place_order_fails_on_payment_declined PASSED
tests/application/test_order_service.py::TestPlaceOrder::test_place_order_releases_inventory_on_payment_failure PASSED
tests/application/test_order_service.py::TestPlaceOrder::test_place_order_raises_on_unknown_order PASSED
tests/application/test_order_service.py::TestPlaceOrder::test_place_order_updates_status_to_confirmed PASSED
tests/application/test_order_service.py::TestPlaceOrder::test_place_order_handles_gateway_exception PASSED
tests/integration/test_full_flow.py::TestFullOrderFlow::test_complete_happy_path PASSED
tests/integration/test_full_flow.py::TestFullOrderFlow::test_payment_failure_releases_inventory PASSED
tests/integration/test_full_flow.py::TestFullOrderFlow::test_insufficient_inventory PASSED
tests/integration/test_full_flow.py::TestFullOrderFlow::test_multiple_orders_same_customer PASSED

============================== 31 passed in 0.15s ==============================
```

## Design Decisions qua TDD

### Tại sao dùng Fake thay vì Mock cho integration tests?

| Tiêu chí | Mock | Fake |
|----------|------|------|
| Setup complexity | Thấp — vài dòng | Trung bình — cần implementation |
| Behavior fidelity | Thấp — chỉ trả về giá trị | Cao — có real behavior |
| Maintenance | Test phụ thuộc vào mock setup | Test giống production flow |
| Speed | Cao | Cao (in-memory) |
| Confidence | Trung bình | Cao |

Quy tắc: **Domain logic → Mock boundaries. Integration flow → Fake infrastructures.**

### Cấu trúc test theo layers

```
tests/
├── domain/     # Pure business logic — không cần mock
├── application/ # Orchestration — mock ports
└── integration/ # Full flow — fake infrastructures
```

Test domain nhanh nhất, test integration chậm hơn một chút nhưng cho confidence cao hơn.

## Lessons Learned từ dự án thực tế

### 1. Phân tách rõ Domain vs Application

**Domain layer** chứa pure business logic — không phụ thuộc vào infrastructure. Dễ test, không cần mock.

**Application layer** orchestrate domain objects với external systems. Test với mock/fake.

### 2. Value Objects là công cụ TDD mạnh

`Money`, `OrderItem.subtotal`, `Customer.discount_rate` — mỗi value object là cơ hội để test behavior nhỏ, isolated. Khi bạn tìm thấy một "primitive obsession" (dùng `float` cho tiền, `str` cho email), hãy tạo value object — TDD sẽ dễ hơn.

### 3. Edge cases phải được test từng cái một

```python
@pytest.mark.parametrize("amount,expected", [
    (Decimal("0.00"), True),   # Zero
    (Decimal("0.01"), True),   # Smallest positive
    (Decimal("999999.99"), True),  # Large
])
def test_money_creation_boundaries(self, amount, expected):
    if expected:
        money = Money(amount)
        assert money.amount == amount
```

### 4. Fixtures nên được tổ chức theo layer

Mỗi layer có conftest riêng:
- `tests/conftest.py` — fixtures dùng chung
- `tests/domain/conftest.py` — domain-specific fixtures
- `tests/application/conftest.py` — mock services
- `tests/integration/conftest.py` — fake implementations

## Kết luận

Dự án e-commerce này minh họa TDD trong real-world setting: multiple layers, external dependencies, complex business logic. Key takeaways:

1. **Start with domain layer**: Pure logic, không cần mock, test chạy nhanh
2. **Use ports/interfaces**: DIP cho phép thay thế implementation dễ dàng
3. **Fakes cho integration, Mocks cho unit tests**: Mỗi loại có mục đích riêng
4. **Test behavior, not implementation**: Cho phép refactor tự do

Trang cuối cùng sẽ giới thiệu các kỹ thuật TDD nâng cao: legacy code, property-based testing, BDD, và async TDD.

## Tài liệu tham khảo

- Eric Evans, *"Domain-Driven Design: Tackling Complexity in the Heart of Software"* (2003)
- Vaughn Vernon, *"Implementing Domain-Driven Design"* (2013)
- Alistair Cockburn, *"Hexagonal Architecture"* (2005)
- Gerard Meszaros, *"xUnit Test Patterns"* (2007)
