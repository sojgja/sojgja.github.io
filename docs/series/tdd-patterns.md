---
id: tdd-patterns
title: TDD Patterns và Kỹ thuật
sidebar_label: 🔴 TDD Patterns
sidebar_position: 57
---

# TDD Patterns và Kỹ thuật

> *"Patterns are not just solutions to problems — they are insights into the nature of the design space."* — **Christopher Alexander**

TDD có một bộ patterns và kỹ thuật phong phú được đúc kết qua hơn 20 năm thực hành. Bài này sẽ trình bày một catalog có hệ thống các pattern quan trọng nhất, từ test design patterns đến test doubles, và các nguyên lý viết test tốt.

## Test Patterns

### 1. Assert First (Khung xanh)

Viết assertion **trước** — không phải test action hay setup. Bắt đầu từ kết quả mong đợi, sau đó suy ngược ra các bước cần thiết.

**Cách thực hành**:

```python
# Bước 1: Viết assertion (kết quả mong đợi)
def test_calculate_discount():
    # ???
    assert final_price == 85.0

# Bước 2: Viết action (hành vi cần kiểm tra)
def test_calculate_discount():
    final_price = calculator.calculate(order)
    assert final_price == 85.0

# Bước 3: Viết setup (chuẩn bị dữ liệu)
def test_calculate_discount():
    order = Order(total=100.0, customer_type="vip")
    calculator = DiscountCalculator()
    final_price = calculator.calculate(order)
    assert final_price == 85.0
```

**Lợi ích**: Tư duy từ kết quả → nguyên nhân, giúp bạn tập trung vào behavior mong muốn thay vì implementation details.

### 2. Triangulation (Tam giác hóa)

Viết nhiều test với các giá trị khác nhau để buộc code phải tổng quát hóa, thay vì chỉ trả về giá trị cứng.

**Ví dụ**:

```python
# RED — single test
def test_add_returns_4_for_2_plus_2(self):
    calc = Calculator()
    assert calc.add(2, 2) == 4

# GREEN — fake it
def add(self, a, b):
    return 4  # Cứng — pass test này nhưng chưa tổng quát

# RED — test thứ hai (triangulation)
def test_add_returns_5_for_2_plus_3(self):
    calc = Calculator()
    assert calc.add(2, 3) == 5

# Bây giờ code phải tổng quát:
def add(self, a, b):
    return a + b
```

**Khi nào dùng**: Khi bạn muốn buộc implementation phải tổng quát hóa mà không cần thiết kế trước. Càng nhiều test với các giá trị khác nhau, code càng tổng quát.

### 3. One to Many (Một đến Nhiều)

Bắt đầu với một object, sau đó mở rộng ra collection — viết test cho một trường hợp đơn giản trước, sau đó tổng quát hóa.

```python
# RED — test với một item
def test_total_price_for_single_item(self):
    cart = ShoppingCart()
    cart.add_item("Apple", 2.5, 3)  # 3 apples at $2.5 each
    assert cart.total() == 7.5

# GREEN — code tối thiểu
class ShoppingCart:
    def __init__(self):
        self._items = []

    def add_item(self, name, price, quantity):
        self._items = [(name, price, quantity)]

    def total(self):
        return self._items[0][1] * self._items[0][2]

# RED — test với nhiều items
def test_total_price_for_multiple_items(self):
    cart = ShoppingCart()
    cart.add_item("Apple", 2.5, 3)
    cart.add_item("Banana", 1.0, 5)
    assert cart.total() == 12.5  # 7.5 + 5.0

# GREEN — tổng quát hóa
class ShoppingCart:
    def __init__(self):
        self._items = []

    def add_item(self, name, price, quantity):
        self._items.append((name, price, quantity))

    def total(self):
        return sum(price * qty for _, price, qty in self._items)
```

### 4. Fake It (Làm giả)

Khi gặp một test khó, hãy trả về giá trị cứng trước, sau đó refactor dần. Fake It là cách để giữ chu kỳ RED-GREEN nhanh.

```python
# RED — test tính thuế phức tạp
def test_complex_tax_calculation(self):
    calc = TaxCalculator()
    result = calc.calculate(amount=1000, location="NY", customer_type="retail")
    assert result == 88.75

# GREEN — Fake It
def calculate(self, amount, location, customer_type):
    return 88.75  # Cứng, nhưng test xanh ngay!

# Sau đó: thêm test khác, dần tổng quát hóa
```

**Khi nào dùng**: Khi implementation phức tạp, bạn muốn pass test nhanh để có baseline, sau đó refactor dần. Đây là kỹ thuật cực kỳ hữu ích khi làm việc với legacy code.

### 5. Obvious Implementation (Implement rõ ràng)

Khi solution quá đơn giản, không cần Fake It — hãy implement ngay.

```python
def test_absolute_value_of_positive_number(self):
    assert abs(5) == 5

def test_absolute_value_of_negative_number(self):
    assert abs(-5) == 5

# Obvious implementation
def abs(x):
    return x if x >= 0 else -x
```

**Khi nào dùng**: Khi bạn biết chắc chắn implementation sẽ như thế nào và nó đủ đơn giản. Fake It trong trường hợp này chỉ là lãng phí thời gian.

### 6. Test Data Builders

Pattern này giải quyết vấn đề setup phức tạp. Thay vì viết fixtures dài dòng, dùng Builder pattern để tạo test data.

```python
# tests/builders.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OrderItem:
    product_id: str
    name: str
    price: float
    quantity: int


class OrderBuilder:
    def __init__(self):
        self.order_id = "ORD-001"
        self.items: List[OrderItem] = []
        self.customer_id = "CUST-001"
        self.shipping_address = "123 Main St"

    def with_item(self, product_id="PROD-1", name="Widget",
                  price=10.0, quantity=1) -> "OrderBuilder":
        self.items.append(OrderItem(product_id, name, price, quantity))
        return self

    def with_customer(self, customer_id: str) -> "OrderBuilder":
        self.customer_id = customer_id
        return self

    def build(self) -> "Order":
        return Order(
            order_id=self.order_id,
            items=self.items,
            customer_id=self.customer_id,
            shipping_address=self.shipping_address,
        )


# Sử dụng trong test:
def test_order_total_with_builder(self):
    order = (OrderBuilder()
             .with_item(price=10.0, quantity=2)
             .with_item(price=5.0, quantity=3)
             .with_customer("VIP-001")
             .build())
    assert order.total() == 35.0
```

### 7. TDD Cycle Patterns (RGR Rhythms)

Có ba nhịp độ khác nhau trong TDD:

| Nhịp | Thời gian | Mô tả |
|------|-----------|-------|
| **Micro-cycle** | 30-60 giây | Một test → code → xanh → refactor nhẹ |
| **Mini-cycle** | 5-15 phút | Hoàn thành một behavior/use case |
| **Macro-cycle** | 1-4 giờ | Hoàn thành một feature/story |

Mỗi cycle đều tuân theo Red-Green-Refactor, chỉ khác scale.

## Test Doubles (Test tương tự)

Test doubles là các object thay thế cho dependency thật trong test. Gerard Meszaros phân loại 5 loại:

| Loại | Mô tả | Có behavior không? | Có verify không? |
|------|-------|--------------------|------------------|
| **Dummy** | Được truyền vào nhưng không bao giờ dùng | Không | Không |
| **Fake** | Implementation đơn giản, hoạt động thật | Có | Không |
| **Stub** | Trả về giá trị định trước | Có (cứng) | Không |
| **Spy** | Ghi lại cách nó được gọi | Có | Có |
| **Mock** | Định nghĩa trước expectation, tự verify | Có | Có (built-in) |

### Dummy

Dummy là object không có behavior — chỉ để thỏa mãn tham số.

```python
from typing import Protocol


class Logger(Protocol):
    def log(self, message: str) -> None: ...


class DummyLogger:
    """Dummy — không làm gì cả."""
    def log(self, message: str) -> None:
        pass


class EmailService:
    def __init__(self, logger: Logger):
        self._logger = logger

    def send(self, to: str, body: str) -> None:
        # send logic
        self._logger.log(f"Email sent to {to}")


def test_email_service_uses_dummy_logger():
    logger = DummyLogger()  # Dummy — không dùng trong test này
    service = EmailService(logger)
    # Chỉ test send, không test log
    service.send("test@example.com", "Hello")
    # Không assert gì về logger — nó là dummy
```

### Fake

Fake là implementation "thật" nhưng đơn giản, thường dùng trong-memory thay vì database thật.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class User:
    id: str
    name: str
    email: str


class UserRepository(ABC):
    @abstractmethod
    def save(self, user: User) -> None: ...
    @abstractmethod
    def find_by_id(self, user_id: str) -> User | None: ...
    @abstractmethod
    def find_by_email(self, email: str) -> User | None: ...


class FakeUserRepository(UserRepository):
    """Fake — in-memory implementation, behavior thật."""
    def __init__(self):
        self._users: dict[str, User] = {}

    def save(self, user: User) -> None:
        self._users[user.id] = user

    def find_by_id(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def find_by_email(self, email: str) -> User | None:
        for user in self._users.values():
            if user.email == email:
                return user
        return None


# Test dùng Fake (chạy nhanh, không cần DB)
def test_register_user():
    repo = FakeUserRepository()
    service = RegistrationService(repo)
    service.register("John", "john@example.com")
    user = repo.find_by_email("john@example.com")
    assert user is not None
    assert user.name == "John"
```

### Stub

Stub trả về giá trị định trước cho các câu hỏi cụ thể.

```python
from typing import Protocol
from datetime import date


class Clock(Protocol):
    def today(self) -> date: ...


class StubClock:
    """Stub — luôn trả về một ngày cố định."""
    def __init__(self, fixed_date: date):
        self._fixed = fixed_date

    def today(self) -> date:
        return self._fixed


class DiscountService:
    def __init__(self, clock: Clock):
        self._clock = clock

    def get_discount_rate(self) -> float:
        today = self._clock.today()
        if today.month == 12 and today.day == 25:
            return 0.50  # 50% off on Christmas
        return 0.0


def test_christmas_discount():
    clock = StubClock(date(2024, 12, 25))
    service = DiscountService(clock)
    assert service.get_discount_rate() == 0.50

def test_regular_day_discount():
    clock = StubClock(date(2024, 6, 1))
    service = DiscountService(clock)
    assert service.get_discount_rate() == 0.0
```

### Spy

Spy ghi lại tất cả tương tác để test có thể kiểm tra sau.

```python
class SpyLogger:
    """Spy — ghi lại tất cả log messages."""
    def __init__(self):
        self.messages: list[str] = []
        self.call_count = 0

    def log(self, message: str) -> None:
        self.messages.append(message)
        self.call_count += 1


class OrderProcessor:
    def __init__(self, logger):
        self._logger = logger

    def process(self, order_id: str) -> None:
        # process order...
        self._logger.log(f"Processing order {order_id}")
        self._logger.log(f"Order {order_id} completed")


def test_processor_logs_correctly():
    logger = SpyLogger()
    processor = OrderProcessor(logger)
    processor.process("ORD-001")
    assert logger.call_count == 2
    assert logger.messages[0] == "Processing order ORD-001"
    assert logger.messages[1] == "Order ORD-001 completed"
```

### Mock

Mock định nghĩa trước expectation và tự verify. Trong Python, dùng `unittest.mock`:

```python
from unittest.mock import Mock, create_autospec


class PaymentGateway:
    def charge(self, amount: float, token: str) -> dict:
        """Charge a payment. Returns {'status': 'success', 'transaction_id': ...}."""
        ...


class OrderService:
    def __init__(self, gateway: PaymentGateway):
        self._gateway = gateway

    def checkout(self, order_id: str, amount: float, token: str) -> dict:
        # validate order, apply discounts, etc.
        result = self._gateway.charge(amount, token)
        if result["status"] == "success":
            return {"order_id": order_id, "transaction_id": result["transaction_id"]}
        return {"order_id": order_id, "error": "Payment failed"}


def test_checkout_uses_gateway():
    gateway = Mock(spec=PaymentGateway)
    gateway.charge.return_value = {
        "status": "success",
        "transaction_id": "TXN-001",
    }
    service = OrderService(gateway)
    result = service.checkout("ORD-001", 100.0, "tok_visa")

    gateway.charge.assert_called_once_with(100.0, "tok_visa")
    assert result["order_id"] == "ORD-001"
    assert result["transaction_id"] == "TXN-001"

def test_checkout_payment_failure():
    gateway = create_autospec(PaymentGateway)
    gateway.charge.return_value = {"status": "failure", "reason": "insufficient_funds"}
    service = OrderService(gateway)
    result = service.checkout("ORD-001", 200.0, "tok_bad")

    assert result["error"] == "Payment failed"
```

### Khi nào dùng loại nào?

```python
# Dùng Dummy khi parameter bắt buộc nhưng không dùng
def test_something():
    user_repo = DummyUserRepository()  # Không dùng trong test này
    service = MyService(user_repo, email_service=RealEmailService())

# Dùng Fake khi cần behavior thật nhưng đơn giản
def test_with_fake_db():
    repo = FakeUserRepository()  # Giống DB thật, nhưng in-memory
    repo.save(User(id="1", name="Alice"))
    assert repo.find_by_id("1").name == "Alice"

# Dùng Stub khi cần trả về giá trị cố định
def test_with_stub():
    clock = StubClock(date(2024, 1, 1))  # Luôn là ngày 1/1
    assert service.calculate(clock) == expected

# Dùng Spy khi cần kiểm tra side effects sau khi gọi
def test_with_spy():
    email_spy = SpyEmailService()
    service.register_user(email_spy, ...)
    assert email_spy.sent_count == 1
    assert "Welcome" in email_spy.last_subject

# Dùng Mock khi cần định nghĩa expectation trước
def test_with_mock():
    email_mock = Mock(spec=EmailService)
    email_mock.send.return_value = True
    service.register_user(email_mock, ...)
    email_mock.send.assert_called_once()
```

## FIRST Principles

Tim Ottinger đưa ra 5 nguyên lý viết test tốt — viết tắt là **FIRST**:

| Chữ | Nguyên lý | Mô tả | Ví dụ vi phạm |
|-----|-----------|-------|---------------|
| **F** | **Fast** (Nhanh) | Test phải chạy nhanh — dưới 100ms mỗi test | Gọi API thật, đọc file, query DB |
| **I** | **Isolated** (Cô lập) | Test không phụ thuộc lẫn nhau | Dùng shared state, global variable |
| **R** | **Repeatable** (Lặp lại) | Kết quả giống nhau mọi lần chạy | Phụ thuộc vào thời gian, random, network |
| **S** | **Self-validating** (Tự kiểm tra) | Pass/Fail rõ ràng, không cần inspect thủ công | Test in ra output rồi phải đọc bằng mắt |
| **T** | **Timely** (Đúng lúc) | Viết test đúng lúc — trước code | Viết test sau khi code 2 tuần |

### Fast

```python
# ❌ SLOW — gọi API thật
def test_user_info():
    response = requests.get("https://api.example.com/users/1")
    assert response.status_code == 200
    assert response.json()["name"] == "Alice"

# ✅ FAST — mock HTTP call
def test_user_info():
    with patch("requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"name": "Alice"}
        service = UserService()
        result = service.get_user(1)
        assert result["name"] == "Alice"
```

### Isolated

```python
# ❌ NOT ISOLATED — dùng shared list
users = []  # global state

def test_create_user():
    users.append({"id": 1, "name": "Alice"})
    assert len(users) == 1

def test_create_another_user():
    users.append({"id": 2, "name": "Bob"})
    assert len(users) == 2  # FAILS! users đã có 1 phần tử từ test trước

# ✅ ISOLATED — mỗi test tự tạo dữ liệu
def test_create_user():
    repo = FakeUserRepository()
    repo.save(User(id="1", name="Alice"))
    assert repo.find_by_id("1").name == "Alice"

def test_create_another_user():
    repo = FakeUserRepository()
    repo.save(User(id="2", name="Bob"))
    assert repo.find_by_id("2").name == "Bob"
```

### Repeatable

```python
# ❌ NOT REPEATABLE — phụ thuộc vào thời gian thực
def test_good_morning_greeting():
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12:
        assert greet() == "Good morning!"
    else:
        assert greet() != "Good morning!"  # Mơ hồ!

# ✅ REPEATABLE — inject clock
def test_good_morning_greeting():
    clock = StubClock(datetime(2024, 1, 1, 8, 0))
    greeter = Greeter(clock)
    assert greeter.greet() == "Good morning!"

def test_good_evening_greeting():
    clock = StubClock(datetime(2024, 1, 1, 19, 0))
    greeter = Greeter(clock)
    assert greeter.greet() == "Good evening!"
```

### Self-validating

```python
# ❌ NOT SELF-VALIDATING — phải đọc output
def test_report():
    result = generate_report()
    print(f"Report: {result}")  # Phải đọc bằng mắt!
    # Không có assert!

# ✅ SELF-VALIDATING
def test_report():
    result = generate_report()
    assert len(result) > 0
    assert "Total: $" in result
    assert "Date: 2024-" in result
```

### Timely

```python
# ❌ NOT TIMELY — test viết sau khi code 1 tháng
# Developer: "Tôi nhớ là hàm này trả về list..."
# (viết test dựa trên trí nhớ — dễ sai)

# ✅ TIMELY — test viết TRƯỚC code
# Test định nghĩa behavior → code implement → test pass
```

## Parameterized Tests (Test tham số hóa)

Một trong những kỹ thuật mạnh nhất của pytest — viết một test chạy với nhiều bộ dữ liệu:

```python
import pytest


# Cách 1: @pytest.mark.parametrize
@pytest.mark.parametrize("input_str,expected", [
    ("", 0),
    ("1", 1),
    ("1,2", 3),
    ("1,2,3,4,5", 15),
    ("1\n2,3", 6),
])
def test_string_calculator(input_str, expected):
    calc = StringCalculator()
    assert calc.add(input_str) == expected


# Cách 2: Nhiều tham số
@pytest.mark.parametrize("a,b,expected", [
    (1, 1, 2),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
    (1.5, 2.5, 4.0),
])
def test_add(a, b, expected):
    assert Calculator().add(a, b) == expected


# Cách 3: Kết hợp với fixture
class TestOrderCalculation:
    @pytest.fixture
    def order(self):
        return OrderBuilder().build()

    @pytest.mark.parametrize("discount,shipping,expected", [
        (0.0, 5.0, 105.0),
        (0.1, 0.0, 90.0),
        (0.2, 10.0, 90.0),
    ])
    def test_final_total(self, order, discount, shipping, expected):
        result = calculate_final(order, discount, shipping)
        assert result == expected
```

## Test Organization Patterns

### Arrange-Act-Assert (AAA)

Pattern cơ bản nhất để tổ chức test:

```python
def test_order_total_with_discount():
    # Arrange — setup dữ liệu
    order = OrderBuilder().with_item(price=50.0, quantity=2).build()
    calculator = TotalCalculator(discount_rate=0.1)

    # Act — gọi hành vi cần test
    total = calculator.calculate(order)

    # Assert — kiểm tra kết quả
    assert total == 90.0  # 100 - 10% discount
```

### Given-When-Then (BDD style)

```python
def test_vip_customer_gets_priority_shipping():
    # Given
    customer = Customer(type="vip")
    order = Order(customer=customer, items=[...])

    # When
    shipping = ShippingService().calculate_shipping(order)

    # Then
    assert shipping.priority is True
    assert shipping.cost == 0.0  # Free shipping for VIP
```

### Four-Phase Test

Mở rộng của AAA với cleanup phase:

```python
def test_database_operation():
    # Setup
    db = FakeDatabase()
    db.seed([User(id="1", name="Alice")])

    # Exercise
    result = db.find_by_name("Alice")

    # Verify
    assert result.name == "Alice"

    # Teardown (có thể dùng fixture với yield)
    db.clear()
```

## Pytest Fixtures nâng cao

```python
# tests/conftest.py
import pytest
from datetime import date
from typing import Generator


@pytest.fixture
def stub_clock() -> Generator:
    """Fixture cung cấp StubClock với ngày mặc định."""
    from src.clock import StubClock
    from datetime import date
    clock = StubClock(date(2024, 1, 15))
    yield clock


@pytest.fixture
def fake_user_repo() -> FakeUserRepository:
    """Fixture cung cấp FakeUserRepository sạch."""
    return FakeUserRepository()


@pytest.fixture
def mock_payment_gateway() -> Mock:
    """Fixture cung cấp Mock PaymentGateway."""
    from unittest.mock import Mock
    gateway = Mock(spec=PaymentGateway)
    gateway.charge.return_value = {"status": "success", "transaction_id": "TXN-MOCK"}
    return gateway


@pytest.fixture
def order_service(
    fake_user_repo: FakeUserRepository,
    mock_payment_gateway: Mock,
) -> OrderService:
    """Fixture tích hợp — inject tất cả dependencies."""
    return OrderService(
        user_repo=fake_user_repo,
        payment_gateway=mock_payment_gateway,
    )
```

## Pattern: Test Factory Method

Khi cần tạo nhiều object phức tạp trong test:

```python
class TestUserFactory:
    @staticmethod
    def create_user(overrides: dict = None) -> User:
        defaults = {
            "id": "default-id",
            "name": "Default User",
            "email": "default@example.com",
            "role": "member",
            "is_active": True,
        }
        if overrides:
            defaults.update(overrides)
        return User(**defaults)

    def test_user_creation(self):
        user = self.create_user({"name": "Alice"})
        assert user.name == "Alice"
        assert user.email == "default@example.com"  # Giữ default

    def test_vip_user(self):
        user = self.create_user({"role": "vip", "email": "vip@example.com"})
        assert user.role == "vip"
        assert user.email == "vip@example.com"
```

## Tổng kết

Catalog patterns trong bài này là công cụ để bạn áp dụng TDD hiệu quả hơn:

| Pattern | Mục đích | Khi nào dùng |
|---------|----------|--------------|
| **Assert First** | Suy nghĩ từ kết quả | Khi bắt đầu test mới |
| **Triangulation** | Buộc tổng quát hóa | Khi Fake It quá lâu |
| **One to Many** | Từ đơn giản đến phức tạp | Collection behavior |
| **Fake It** | Pass test nhanh | Implementation phức tạp |
| **Obvious Impl** | Implement trực tiếp | Solution rõ ràng |
| **AAA** | Cấu trúc test | Mọi test |
| **Test Doubles** | Thay thế dependency | Khi có external dependency |
| **FIRST** | Nguyên lý thiết kế test | Review test quality |

Trang tiếp theo chúng ta sẽ kết hợp TDD với OOP và SOLID để thấy testability dẫn dắt thiết kế tốt hơn như thế nào.

## Tài liệu tham khảo

- Gerard Meszaros, *"xUnit Test Patterns"* (2007) — Kinh thánh về test patterns
- Martin Fowler, *"Mocks Aren't Stubs"* (2007) — https://martinfowler.com/articles/mocksArentStubs.html
- Steve Freeman & Nat Pryce, *"Growing Object-Oriented Software Guided by Tests"* (2009)
- Tim Ottinger, *"FIRST Principles of Test-Driven Development"*
