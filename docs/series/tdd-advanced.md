---
id: tdd-advanced
title: TDD Nâng cao
sidebar_label: 🔴 TDD Nâng cao
sidebar_position: 60
---

# TDD Nâng cao

> *"The most important thing about legacy code is: it's code that works. The most dangerous thing about legacy code is: it's code that nobody understands."* — **Michael Feathers**

Bài này dành cho những ai đã thành thạo TDD cơ bản và muốn áp dụng vào các tình huống khó: legacy code không có test, property-based testing, BDD, và code bất đồng bộ (async).

## Legacy Code và Characterization Tests

### Bài toán: Legacy code không có test

Legacy code là code đã tồn tại, thường không có test, khó hiểu, và không ai dám sửa. Michael Feathers định nghĩa legacy code đơn giản là **"code without tests"**.

```python
# legacy/price_engine.py — KHÔNG được sửa trước khi có test!
def calculate_price(base_price, customer_type, quantity, is_weekend, coupon_code):
    """Tính giá bán dựa trên nhiều yếu tố.

    WARNING: Code này đã chạy production 5 năm. Không ai hiểu hết.
    """
    price = base_price * quantity

    if customer_type == "vip":
        if quantity > 10:
            price *= 0.75
        else:
            price *= 0.9
    elif customer_type == "wholesale":
        price *= 0.8
        if quantity > 50:
            price *= 0.95
    elif customer_type == "employee":
        if is_weekend:
            price *= 0.85
        else:
            price *= 0.7
    else:
        if quantity > 5:
            price *= 0.95

    if coupon_code:
        if coupon_code.startswith("SAVE"):
            price -= 10
        elif coupon_code.startswith("BIG"):
            price *= 0.5
        elif coupon_code == "FREE":
            price = 0

    if is_weekend and price > 100:
        price *= 1.1  # Weekend surcharge

    return max(0, price)
```

### Characterization Tests — "Test để hiểu"

Characterization test (còn gọi là Golden Master test) là kỹ thuật viết test cho legacy code **mà không cần hiểu nó**. Bạn chạy code với nhiều input, ghi lại output, và viết test assert output đó.

```python
# tests/legacy/test_price_engine.py
import pytest
from legacy.price_engine import calculate_price


class TestPriceEngineCharacterization:
    """Characterization tests — capture current behavior."""

    @pytest.mark.parametrize("base_price,customer_type,quantity,is_weekend,coupon,expected", [
        # Regular customer
        (100, "regular", 1, False, None, 100),
        (100, "regular", 6, False, None, 95),    # 5% discount for qty > 5
        (100, "regular", 10, False, None, 950),   # 100*10*0.95

        # VIP customer
        (100, "vip", 1, False, None, 90),          # 10% off
        (100, "vip", 11, False, None, 825),         # 25% off for qty > 10

        # Wholesale
        (100, "wholesale", 10, False, None, 800),   # 20% off
        (100, "wholesale", 51, False, None, 3800),   # 20% + 5% = 24% off

        # Employee
        (100, "employee", 1, False, None, 70),      # 30% off weekday
        (100, "employee", 1, True, None, 85),       # 15% off weekend

        # Coupons
        (100, "regular", 1, False, "SAVE10", 90),   # SAVE = -10
        (100, "regular", 1, False, "BIG50", 50),    # BIG = 50% off
        (100, "regular", 1, False, "FREE", 0),       # FREE = $0

        # Weekend surcharge
        (200, "regular", 1, True, None, 220),        # +10% surcharge

        # Edge cases
        (100, "regular", 1, False, "SAVE20", 90),   # min 0, price=90
        (5, "regular", 1, False, None, 5),           # small price
        (0, "regular", 1, False, None, 0),           # zero price
    ])
    def test_characterize_price_calculation(
        self, base_price, customer_type, quantity, is_weekend, coupon, expected
    ):
        result = calculate_price(base_price, customer_type, quantity, is_weekend, coupon)
        assert result == expected, (
            f"calculate_price({base_price}, '{customer_type}', {quantity}, "
            f"{is_weekend}, {coupon}) = {result}, expected {expected}"
        )

    def test_characterize_more_edge_cases(self):
        """Capture behaviors that might be bugs."""
        # Negative? Let's see what happens
        result = calculate_price(-100, "regular", 1, False, None)
        assert result == 0  # max(0, -100) = 0

        # Unknown customer type
        result = calculate_price(100, "unknown", 1, False, None)
        assert result == 100  # Falls through to else? No discount?

        # Free coupon with employee
        result = calculate_price(100, "employee", 1, False, "FREE")
        assert result == 0  # FREE trumps all discounts
```

### TDD với Legacy Code: The Seam Technique

Michael Feathers giới thiệu khái niệm **Seam** — nơi bạn có thể can thiệp vào code mà không cần sửa nó.

```python
# legacy/email_sender.py — không thể test vì gửi email thật
import smtplib


class EmailSender:
    def send_welcome(self, email: str, name: str) -> None:
        server = smtplib.SMTP("smtp.company.com")  # Hardcoded!
        server.sendmail(
            "welcome@company.com",
            email,
            f"Subject: Welcome {name}!\n\nThank you for joining!"
        )
        server.quit()
```

**Bước 1 — Tìm Seam**: Tạo subclass để override method gọi SMTP:

```python
# tests/legacy/test_email_sender.py
from unittest.mock import patch, MagicMock
from legacy.email_sender import EmailSender


class TestableEmailSender(EmailSender):
    def __init__(self):
        super().__init__()
        self.last_server = None

    def _create_smtp_server(self, host: str):
        self.last_server = MagicMock()
        return self.last_server


class TestEmailSender:
    def test_send_welcome_uses_correct_parameters(self):
        sender = TestableEmailSender()
        sender.send_welcome("alice@example.com", "Alice")

        # Verify via the seam
        server = sender.last_server
        assert server is not None
        server.sendmail.assert_called_once()
        args = server.sendmail.call_args
        assert args[0][0] == "welcome@company.com"
        assert args[0][1] == "alice@example.com"
        assert "Welcome Alice" in args[0][2]
```

**Bước 2 — Thêm test, refactor dần**: Sau khi characterization tests pass, mới bắt đầu refactor.

## Property-Based Testing với Hypothesis

### Tại sao cần Property-Based Testing?

Example-based testing kiểm tra **specific cases**. Property-based testing kiểm tra **universal properties** — những tính chất luôn đúng với mọi input.

```python
# Example-based: chỉ test 3 cases
@pytest.mark.parametrize("s", ["", "a", "abc"])
def test_reverse_example_based(s):
    assert reverse(reverse(s)) == s

# Property-based: test property với HÀNG TRĂM random inputs
from hypothesis import given, strategies as st

@given(st.text())
def test_reverse_property(s):
    assert reverse(reverse(s)) == s  # Double reverse = identity
```

### Cài đặt Hypothesis

```bash
pip install hypothesis
```

### Chiến lược (Strategies) cơ bản

```python
from hypothesis import given, strategies as st, assume
from hypothesis import HealthCheck, settings


# Basic types
@given(st.integers())
def test_absolute_value_non_negative(x):
    assert abs(x) >= 0


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_floats(x):
    assert isinstance(x, float)


@given(st.text(max_size=100))
def test_strings(s):
    assert isinstance(s, str)


# Composed strategies
@given(st.lists(st.integers(min_value=0, max_value=1000)))
def test_sum_positive_list(lst):
    assert sum(lst) >= 0


@given(st.dictionaries(
    keys=st.text(max_size=10),
    values=st.integers(min_value=0, max_value=100),
))
def test_dict_values_non_negative(d):
    assert all(v >= 0 for v in d.values())


# Custom strategies
UserStrategy = st.fixed_dictionaries({
    "name": st.text(min_size=1, max_size=50),
    "age": st.integers(min_value=0, max_value=120),
    "email": st.emails(),
    "is_active": st.booleans(),
})


@given(UserStrategy)
def test_user_validation(user):
    assert len(user["name"]) > 0
    assert 0 <= user["age"] <= 120
```

### Property-Based TDD cho String Calculator

Áp dụng Hypothesis cho String Calculator từ bài 2:

```python
from hypothesis import given, strategies as st, assume


class TestStringCalculatorProperty:
    """Property-based tests for String Calculator."""

    @given(st.integers(min_value=0, max_value=1000))
    def test_single_number_returns_itself(self, n):
        calc = StringCalculator()
        assert calc.add(str(n)) == n

    @given(st.lists(
        st.integers(min_value=0, max_value=1000),
        min_size=1, max_size=10,
    ))
    def test_sum_is_commutative(self, numbers):
        calc = StringCalculator()
        input_str = ",".join(str(n) for n in numbers)
        result = calc.add(input_str)
        assert result == sum(numbers)

    @given(st.lists(
        st.integers(min_value=0, max_value=1000),
        min_size=1, max_size=10,
    ))
    def test_adding_zero_does_not_change_sum(self, numbers):
        """Identity property."""
        calc = StringCalculator()
        original = ",".join(str(n) for n in numbers)
        with_zero = original + ",0"
        assert calc.add(with_zero) == calc.add(original)

    @given(st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=1, max_size=5,
    ), st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=1, max_size=5,
    ))
    def test_sum_is_associative(self, first_half, second_half):
        """(a+b)+c == a+(b+c)"""
        calc = StringCalculator()
        all_nums = first_half + second_half
        input_all = ",".join(str(n) for n in all_nums)
        result_all = calc.add(input_all)
        assert result_all == sum(first_half) + sum(second_half)

    @given(st.text(max_size=10))
    def test_empty_string_returns_zero(self, s):
        """Empty string always returns 0."""
        calc = StringCalculator()
        if s == "":
            assert calc.add("") == 0
        else:
            # Skip non-empty for this property
            assume(False)
```

### Tìm Bug với Hypothesis

Hypothesis nổi tiếng với khả năng tìm ra edge cases mà bạn không nghĩ tới:

```python
from hypothesis import given, strategies as st, assume
from decimal import Decimal


class BankAccount:
    def __init__(self, balance: Decimal = Decimal("0")):
        self._balance = balance

    def deposit(self, amount: Decimal) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    def withdraw(self, amount: Decimal) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

    @property
    def balance(self) -> Decimal:
        return self._balance


class TestBankAccountProperty:
    @given(st.integers(min_value=1, max_value=10000))
    def test_deposit_increases_balance(self, amount):
        account = BankAccount()
        account.deposit(Decimal(str(amount)))
        assert account.balance == Decimal(str(amount))

    @given(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    )
    def test_deposit_and_withdraw_returns_to_original(self, deposit_amt, withdraw_amt):
        """Balance should be unchanged if we deposit X and withdraw X."""
        assume(deposit_amt == withdraw_amt)  # Only test when equal
        account = BankAccount()
        account.deposit(Decimal(str(deposit_amt)))
        account.withdraw(Decimal(str(withdraw_amt)))
        assert account.balance == Decimal("0")

    @given(
        st.integers(min_value=1, max_value=10000),
        st.integers(min_value=1, max_value=10000),
    )
    def test_balance_never_negative_after_withdraw(self, initial, withdraw_amt):
        """Balance should never go negative."""
        account = BankAccount(Decimal(str(initial)))
        assume(withdraw_amt <= initial)  # Valid withdrawal
        account.withdraw(Decimal(str(withdraw_amt)))
        assert account.balance >= 0
```

### Custom Strategies cho Domain Objects

```python
from hypothesis import strategies as st
from decimal import Decimal
from src.domain.value_objects import Money


# Custom strategy for Money
MoneyStrategy = st.builds(
    Money,
    amount=st.decimals(
        min_value=Decimal("0.01"),
        max_value=Decimal("999999.99"),
        places=2,
    ),
    currency=st.just("USD"),
)


@given(MoneyStrategy, MoneyStrategy)
def test_money_addition_commutative(a: Money, b: Money):
    assert a + b == b + a


@given(MoneyStrategy, MoneyStrategy, MoneyStrategy)
def test_money_addition_associative(a: Money, b: Money, c: Money):
    assert (a + b) + c == a + (b + c)


@given(MoneyStrategy, st.decimals(
    min_value=Decimal("0.5"),
    max_value=Decimal("2.0"),
    places=2,
))
def test_money_multiplication(money: Money, factor: Decimal):
    result = money * factor
    assert isinstance(result, Money)
    expected = Money(
        (money.amount * factor).quantize(Decimal("0.01")),
        money.currency,
    )
    assert result == expected
```

### Shrinking — Sức mạnh của Hypothesis

Khi Hypothesis tìm ra bug, nó tự động **shrink** (thu nhỏ) input về trường hợp đơn giản nhất:

```text
Falsifying example: test_money_creation(
    amount=Decimal('-0.01'),  # Shrunk from -12345.67!
)
```

Đây là killer feature — thay vì bạn phải debug với input phức tạp, Hypothesis đưa ra **minimum failing case**.

### Settings và Filtering

```python
from hypothesis import settings, given, assume
import hypothesis.strategies as st


# Tăng độ phủ
@settings(max_examples=1000)  # Mặc định 100
@given(st.integers())
def test_with_more_examples(x):
    assert abs(x) >= 0


# Filter dữ liệu không hợp lệ
@given(st.integers())
def test_even_number_property(x):
    assume(x % 2 == 0)  # Chỉ chạy với số chẵn
    assert x % 2 == 0


# Suppress health checks cho slow tests
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(), max_size=1000))
def test_large_lists(lst):
    assert len(lst) <= 1000
```

## BDD với pytest-bdd

Behavior-Driven Development (BDD) là extension của TDD — viết test bằng ngôn ngữ tự nhiên (Gherkin) để business stakeholders có thể đọc và hiểu.

### Cài đặt

```bash
pip install pytest-bdd
```

### Feature File (Gherkin)

```gherkin
# tests/features/order.feature
Feature: Order Processing
  As a customer
  I want to place orders
  So that I can purchase products

  Background:
    Given a VIP customer "Alice Nguyen" with email "alice@example.com"
    And a product "Wireless Mouse" costs $25.00
    And the product has 10 items in stock

  Scenario: Customer places a simple order
    Given Alice creates an order with 2 "Wireless Mouse"
    When she places the order with payment token "tok_visa"
    Then the order should be confirmed
    And the total should be $45.00  # 10% VIP discount applied
    And a confirmation email should be sent to "alice@example.com"

  Scenario: Order fails due to insufficient inventory
    Given Alice creates an order with 20 "Wireless Mouse"
    When she places the order with payment token "tok_visa"
    Then the order should fail with "Insufficient inventory" error

  Scenario: VIP discount is applied correctly
    Given Alice creates an order with 3 "Wireless Mouse"
    When she calculates the total
    Then the subtotal should be $75.00
    And the discount should be $7.50  # 10% VIP discount
    And the total should be $67.50

  Scenario Outline: Bulk discount for wholesale customers
    Given a wholesale customer "Bob" with email "bob@example.com"
    And a product "Mechanical Keyboard" costs $89.99
    When <quantity> keyboards are added to the order
    And the total is calculated
    Then the total should be <expected>

    Examples:
      | quantity | expected |
      | 1        | 71.99    |  # 20% wholesale discount
      | 10       | 719.92   |  # 20% wholesale discount
      | 100      | 6839.24  |  # 20% + 5% bulk discount
```

### Step Definitions

```python
# tests/step_defs/test_order_steps.py
from pytest_bdd import given, when, then, parsers, scenarios
from decimal import Decimal
from src.domain.models import Product, Customer, Order, OrderItem
from src.domain.value_objects import Money
from src.application.order_service import OrderService
from tests.integration.conftest import (
    FakeOrderRepository, FakeProductRepository, FakeCustomerRepository,
    FakePaymentGateway, FakeEmailService, FakeInventorySystem,
)

# Load scenarios từ feature file
scenarios("../features/order.feature")


# Shared state
@pytest.fixture
def context():
    """Shared context giữa các steps."""
    return {}


@pytest.fixture
def repositories():
    order_repo = FakeOrderRepository()
    product_repo = FakeProductRepository()
    customer_repo = FakeCustomerRepository()
    payment = FakePaymentGateway()
    email = FakeEmailService()
    inventory = FakeInventorySystem()
    return order_repo, product_repo, customer_repo, payment, email, inventory


@pytest.fixture
def service(repositories):
    order_repo, product_repo, customer_repo, payment, email, inventory = repositories
    return OrderService(order_repo, product_repo, customer_repo, payment, email, inventory)


# Background steps
@given(parsers.parse('a VIP customer "{name}" with email "{email}"'))
def a_vip_customer(repositories, context, name, email):
    _, _, customer_repo, _, _, _ = repositories
    customer = Customer(
        id=f"CUST-{hash(name) % 1000:03d}",
        name=name,
        email=email,
        tier="vip",
    )
    customer_repo.add_customer(customer)
    context["customer"] = customer


@given(parsers.parse('a product "{name}" costs ${price}'))
def a_product(repositories, context, name, price):
    _, product_repo, _, _, _, _ = repositories
    product = Product(
        id=f"PROD-{hash(name) % 1000:03d}",
        name=name,
        price=Money(Decimal(price)),
        weight_kg=0.5,
        category="electronics",
    )
    product_repo.add_product(product, stock=10)
    context["product"] = product


@given(parsers.parse('the product has {stock:d} items in stock'))
def product_stock(repositories, context, stock):
    _, product_repo, _, _, _, _ = repositories
    product = context["product"]
    product_repo.update_stock(product.id, stock)


@given(parsers.parse('a wholesale customer "{name}" with email "{email}"'))
def a_wholesale_customer(repositories, context, name, email):
    _, _, customer_repo, _, _, _ = repositories
    customer = Customer(
        id=f"CUST-{hash(name) % 1000:03d}",
        name=name,
        email=email,
        tier="wholesale",
    )
    customer_repo.add_customer(customer)
    context["customer"] = customer


# When steps
@when(parsers.parse('{name} creates an order with {qty:d} "{product_name}"'))
def create_order(service, context, name, qty, product_name):
    customer = context["customer"]
    product = context["product"]
    try:
        order = service.create_order(
            customer_id=customer.id,
            items=[{"product_id": product.id, "quantity": qty}],
        )
        context["order"] = order
        context["error"] = None
    except Exception as e:
        context["error"] = str(e)
        context["order"] = None


@when(parsers.parse('she places the order with payment token "{token}"'))
def place_order(service, context, token):
    order = context.get("order")
    if order is None:
        return
    try:
        result = service.place_order(order.id, token)
        context["result"] = result
        context["error"] = None
    except Exception as e:
        context["error"] = str(e)
        context["result"] = None


@when(parsers.parse('{qty:d} keyboards are added to the order'))
def add_keyboards(service, context, qty):
    customer = context["customer"]
    product = context["product"]
    order = service.create_order(
        customer_id=customer.id,
        items=[{"product_id": product.id, "quantity": qty}],
    )
    context["order"] = order


@when("she calculates the total")
def calculate_total(service, context):
    pass  # Total already calculated in Order


@when("the total is calculated")
def the_total_is_calculated(service, context):
    pass  # Total already calculated in Order


# Then steps
@then("the order should be confirmed")
def order_confirmed(context):
    assert context["result"]["status"] == "confirmed"


@then(parsers.parse('the total should be ${expected}'))
def check_total(context, expected):
    order = context["order"]
    assert str(order.total.amount) == expected


@then(parsers.parse('a confirmation email should be sent to "{email}"'))
def check_email(repositories, context, email):
    _, _, _, _, email_service, _ = repositories
    sent = email_service.sent_emails
    assert any(
        e[0] == "confirmation" and e[1] == email
        for e in sent
    )


@then(parsers.parse('the order should fail with "{error_msg}" error'))
def order_failed(context, error_msg):
    assert context["error"] is not None
    assert error_msg.lower() in context["error"].lower()


@then(parsers.parse('the subtotal should be ${expected}'))
def check_subtotal(context, expected):
    order = context["order"]
    assert str(order.subtotal.amount) == expected


@then(parsers.parse('the discount should be ${expected}'))
def check_discount(context, expected):
    order = context["order"]
    assert str(order.discount.amount) == expected
```

### Chạy BDD Tests

```text
$ pytest -v --tb=short tests/step_defs/

============================= test session starts =============================
collected 5 items

tests/step_defs/test_order_steps.py::test_order_processing[customer_places_a_simple_order] PASSED
tests/step_defs/test_order_steps.py::test_order_processing[order_fails_due_to_insufficient_inventory] PASSED
tests/step_defs/test_order_steps.py::test_order_processing[VIP_discount_is_applied_correctly] PASSED
tests/step_defs/test_order_steps.py::test_order_processing[Bulk_discount_for_wholesale_customers_qty_1] PASSED
tests/step_defs/test_order_steps.py::test_order_processing[Bulk_discount_for_wholesale_customers_qty_100] PASSED

============================== 5 passed in 0.25s ==============================
```

## TDD với Async Code

Python 3.10+ có `asyncio` built-in. TDD với async code đòi hỏi các công cụ đặc biệt.

### Async Test với pytest-asyncio

```bash
pip install pytest-asyncio
```

```python
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # Tự động detect async tests
```

### Ví dụ: Async Repository

```python
# src/async_repository.py
from typing import Protocol, Optional
import asyncio


class AsyncUserRepository(Protocol):
    async def find_by_id(self, user_id: str) -> Optional[dict]: ...
    async def save(self, user: dict) -> None: ...


class AsyncUserService:
    def __init__(self, repo: AsyncUserRepository):
        self._repo = repo

    async def get_user_display_name(self, user_id: str) -> str:
        user = await self._repo.find_by_id(user_id)
        if user is None:
            return "Unknown User"
        return f"{user['first_name']} {user['last_name']}"

    async def create_user(self, first_name: str, last_name: str, email: str) -> dict:
        user = {
            "id": f"USER-{hash(email) % 10000:04d}",
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
        }
        await self._repo.save(user)
        return user
```

### TDD Cycle với Async

```python
# tests/async/test_user_service.py
import pytest
from unittest.mock import AsyncMock, create_autospec
from src.async_repository import AsyncUserRepository, AsyncUserService


@pytest.fixture
def mock_repo():
    return create_autospec(AsyncUserRepository)


@pytest.fixture
def service(mock_repo):
    return AsyncUserService(mock_repo)


class TestAsyncUserService:
    async def test_get_display_name_for_existing_user(self, service, mock_repo):
        # Arrange
        mock_repo.find_by_id.return_value = {
            "first_name": "Alice",
            "last_name": "Nguyen",
            "email": "alice@example.com",
        }

        # Act
        result = await service.get_user_display_name("USER-001")

        # Assert
        assert result == "Alice Nguyen"
        mock_repo.find_by_id.assert_called_once_with("USER-001")

    async def test_get_display_name_for_missing_user(self, service, mock_repo):
        mock_repo.find_by_id.return_value = None

        result = await service.get_user_display_name("USER-999")

        assert result == "Unknown User"

    async def test_create_user_invokes_save(self, service, mock_repo):
        result = await service.create_user("Bob", "Smith", "bob@example.com")

        assert result["first_name"] == "Bob"
        assert result["last_name"] == "Smith"
        assert result["email"] == "bob@example.com"
        mock_repo.save.assert_called_once_with(result)
```

### Async Fake Repository

```python
# tests/async/conftest.py
import pytest
from typing import Optional


class FakeAsyncUserRepository:
    def __init__(self):
        self._users: dict[str, dict] = {}

    def add_user(self, user: dict) -> None:
        self._users[user["id"]] = user

    async def find_by_id(self, user_id: str) -> Optional[dict]:
        await asyncio.sleep(0.01)  # Simulate async I/O
        return self._users.get(user_id)

    async def save(self, user: dict) -> None:
        await asyncio.sleep(0.01)
        self._users[user["id"]] = user


class TestAsyncUserIntegration:
    @pytest.fixture
    def fake_repo(self):
        repo = FakeAsyncUserRepository()
        repo.add_user({
            "id": "USER-001",
            "first_name": "Alice",
            "last_name": "Nguyen",
            "email": "alice@example.com",
        })
        return repo

    async def test_full_async_flow(self, fake_repo):
        service = AsyncUserService(fake_repo)

        # Get existing user
        name = await service.get_user_display_name("USER-001")
        assert name == "Alice Nguyen"

        # Create new user
        created = await service.create_user("Bob", "Smith", "bob@example.com")
        assert created["id"] == "USER-0010"  # Based on hash

        # Verify new user exists
        name = await service.get_user_display_name(created["id"])
        assert name == "Bob Smith"
```

### Async Mocking with side_effect

```python
class TestAsyncErrorHandling:
    async def test_repository_error_is_propagated(self, service, mock_repo):
        mock_repo.find_by_id.side_effect = ConnectionError("Database timeout")

        with pytest.raises(ConnectionError, match="Database timeout"):
            await service.get_user_display_name("USER-001")

    async def test_retry_on_transient_error(self, service, mock_repo):
        # Fail first 2 calls, succeed on 3rd
        mock_repo.find_by_id.side_effect = [
            ConnectionError("timeout"),
            ConnectionError("timeout"),
            {"first_name": "Alice", "last_name": "Nguyen"},
        ]

        result = await self._retry_get_user(service, "USER-001")
        assert result == "Alice Nguyen"
        assert mock_repo.find_by_id.call_count == 3

    async def _retry_get_user(self, service, user_id, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await service.get_user_display_name(user_id)
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
```

### Property-Based Testing với Async

```python
import pytest
from hypothesis import given, strategies as st, settings
import asyncio


class TestAsyncProperty:
    @given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    async def test_create_and_retrieve_user(self, first_name, last_name):
        """Property: created user can always be retrieved."""
        repo = FakeAsyncUserRepository()
        service = AsyncUserService(repo)

        created = await service.create_user(first_name, last_name, f"{first_name}@test.com")
        name = await service.get_user_display_name(created["id"])

        assert name == f"{first_name} {last_name}"

    @given(st.text())
    async def test_missing_user_returns_unknown(self, user_id):
        """Property: non-existent ID returns 'Unknown User'."""
        assume(len(user_id) > 0)  # Skip empty
        repo = FakeAsyncUserRepository()
        service = AsyncUserService(repo)

        result = await service.get_user_display_name(user_id)
        assert result == "Unknown User"
```

## Advanced Test Patterns

### Test với Context Managers

```python
import contextlib
from typing import Generator
from unittest.mock import patch


@contextlib.contextmanager
def mock_env_variable(key: str, value: str) -> Generator:
    """Context manager để mock environment variables."""
    import os
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            del os.environ[key]
        else:
            os.environ[key] = old_value


def test_uses_environment_variable():
    with mock_env_variable("API_KEY", "test-key-123"):
        service = ExternalApiService()
        assert service.api_key == "test-key-123"
```

### Test với Temporary Files

```python
import tempfile
from pathlib import Path


def test_file_processing():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,age\nAlice,30\nBob,25\n")
        f.flush()
        temp_path = Path(f.name)

    try:
        processor = CsvProcessor(temp_path)
        result = processor.process()
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == "30"
    finally:
        temp_path.unlink()
```

### Test với Time

```python
from unittest.mock import patch
from datetime import datetime, timezone


class TimeBasedGreeting:
    def greet(self) -> str:
        hour = datetime.now(timezone.utc).hour
        if hour < 12:
            return "Good morning"
        elif hour < 18:
            return "Good afternoon"
        return "Good evening"


class TestTimeBasedGreeting:
    @patch("src.service.datetime")
    def test_morning_greeting(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
        greeter = TimeBasedGreeting()
        assert greeter.greet() == "Good morning"

    @patch("src.service.datetime")
    def test_evening_greeting(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc)
        greeter = TimeBasedGreeting()
        assert greeter.greet() == "Good evening"
```

## Mutation Testing

Mutation testing kiểm tra chất lượng test bằng cách thay đổi code (tạo mutation) và xem test có phát hiện không.

```bash
pip install mutmut
```

```bash
# Chạy mutation testing
mutmut run --paths-to-mutate src/
```

Output:

```text
<mutmut> Starting mutation testing
<mutmut> 32 mutations were generated
<mutmut> 28 mutants survived  (87.5% coverage)
<mutmut> 4 mutants killed by tests
<mutmut> The full list can be found at html/index.html
```

Nếu mutation **survived** (test vẫn pass dù code bị thay đổi), test của bạn chưa đủ mạnh. Ví dụ:

```python
# Original
def is_adult(age):
    return age >= 18

# Mutation — thay >= bằng >
def is_adult(age):
    return age > 18  # Bug! 18-year-old not considered adult

# Nếu test không phát hiện → mutation survived!
def test_is_adult():
    assert is_adult(20) is True
    assert is_adult(15) is False
    # ❌ Thiếu test cho age == 18!

# Fix: thêm test boundary
def test_is_adult_boundary():
    assert is_adult(18) is True  # Nếu mutation, test này fail!
```

## Tổng kết Series

| Bài | Nội dung | Kỹ thuật chính |
|-----|----------|---------------|
| **1. Giới thiệu** | Lịch sử, nguyên lý, ROI | Red-Green-Refactor, 3 Laws |
| **2. Cơ bản** | String Calculator | Baby steps, Fake It, Triangulation |
| **3. Patterns** | Test patterns, doubles | Mock, Stub, Fake, Spy, FIRST |
| **4. OOP & SOLID** | Thiết kế testable | DIP, SRP, Hexagonal Architecture |
| **5. Real-world** | E-commerce system | Fakes, Integration tests, Layers |
| **6. Nâng cao** | Legacy, Hypothesis, BDD, Async | Characterization, Property-based, Gherkin |

## Tài liệu tham khảo

- Michael Feathers, *"Working Effectively with Legacy Code"* (2004)
- David R. MacIver, *"Hypothesis: Property-Based Testing for Python"*
- pytest-bdd documentation: https://pytest-bdd.readthedocs.io/
- *"Mutation Testing"* — https://mutmut.readthedocs.io/
- *"pytest-asyncio"* — https://pytest-asyncio.readthedocs.io/
