---
id: humble-object
title: Humble Object
sidebar_label: 🎯 Humble Object
sidebar_position: 2
---

# Humble Object

<!-- truncate -->

## Nguồn gốc và định nghĩa

Humble Object được Gerard Meszaros đặt tên trong *xUnit Test Patterns* (2007), dựa trên ý tưởng từ pattern "Humble Dialog Box" của UnitTest thời kỳ đầu. Martin Fowler sau đó phổ biến khái niệm này trên bliki của ông.

Định nghĩa gốc:

> *"The Humble Object pattern is a way to separate the behavior that is hard to test from the behavior that is easy to test, by moving the hard-to-test behavior into a separate object that is so simple that it can be ignored during testing."*

Hay ngắn gọn hơn:

> **Extract business logic from hard-to-test classes into easy-to-test classes.**

Hard-to-test classes là những class phụ thuộc vào infrastructure: database, network, filesystem, framework base classes, global state. Easy-to-test classes là pure functions hoặc pure data transformations không có side effect.

---

## Cơ sở lý thuyết

### Coupling và Testability

Một class khó test khi nó có **efferent coupling** cao — càng nhiều dependency, càng khó khởi tạo trong isolation. Humble Object giảm coupling bằng cách:

- **Tách biệt business logic khỏi infrastructure code**
- **Biến business logic thành pure functions** (deterministic, no side effects)
- **Để lại "glue code" tối thiểu** trong class framework

### Liên hệ với SOLID

| Nguyên lý | Cách Humble Object áp dụng |
|-----------|--------------------------|
| **S**ingle Responsibility | Humble object chỉ làm nhiệm vụ orchestration; pure class chịu trách nhiệm logic |
| **O**pen/Closed | Pure functions có thể mở rộng bằng composition mà không sửa đổi |
| **L**iskov Substitution | Humble object là thin wrapper, không vi phạm contract |
| **I**nterface Segregation | Pure functions định nghĩa interface nhỏ, rõ ràng |
| **D**ependency Inversion | Humble object phụ thuộc vào pure abstraction, không phải framework cụ thể |

### Testability Heuristic

Một heuristic đơn giản: **nếu bạn cần chạy database để test một hàm tính toán, bạn đang làm sai.**

```python
# Dễ test — pure function
def calculate_tax(amount: float, rate: float) -> float:
    return amount * rate

# Khó test — phụ thuộc ORM
def calculate_tax(self, product_id):
    product = self.env["product.product"].browse(product_id)
    return product.price * product.tax_category.rate
```

Heuristic này dẫn đến một quy tắc quan trọng: **hãy đặt câu hỏi "tôi có thể test hàm này mà không cần khởi tạo framework không?"** Nếu câu trả lời là không, bạn cần Humble Object.

---

## Các dạng Humble Object

Humble Object không chỉ là một pattern — nó là một **họ patterns** với nhiều biến thể:

### 1. Humble Function (dạng đơn giản nhất)

Tách một method thành pure function + thin wrapper.

```python
# Pure function
def validate_order(items: list, max_total: float) -> list[str]:
    errors = []
    total = sum(item["price"] * item["qty"] for item in items)
    if total > max_total:
        errors.append(f"Total ${total} exceeds limit ${max_total}")
    for item in items:
        if item.get("qty", 0) <= 0:
            errors.append(f"Item {item['name']} has invalid quantity")
    return errors

# Humble wrapper (framework-dependent)
class OrderView(APIView):
    def post(self, request):
        errors = validate_order(request.data["items"], request.user.credit_limit)
        if errors:
            return Response({"errors": errors}, status=400)
        return Response({"status": "ok"})
```

### 2. Humble Class (dạng có state)

Khi logic cần state, ta tách thành một class pure Python thuần túy.

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class CartCalculator:
    items: list
    shipping_zip: str
    customer_tier: str

    def subtotal(self) -> Decimal:
        return sum(item["price"] * item["qty"] for item in self.items)

    def tax(self) -> Decimal:
        rate = 0.10 if self.shipping_zip.startswith("9") else 0.08
        return self.subtotal() * rate

    def shipping(self) -> Decimal:
        weight = sum(item["weight"] * item["qty"] for item in self.items)
        if self.customer_tier == "premium":
            return Decimal("0")
        return Decimal("5.99") if weight < 10 else Decimal("12.99")

    def total(self) -> Decimal:
        return self.subtotal() + self.tax() + self.shipping()
```

### 3. Humble Module (dạng hệ thống con)

Khi một nhóm functions có liên quan với nhau được tổ chức thành module.

```
core/
├── pricing/
│   ├── __init__.py        # Public API: calculate_price()
│   ├── discounts.py       # Chiết khấu
│   ├── taxes.py           # Thuế
│   ├── fees.py            # Phí
│   └── tests/
├── inventory/
│   ├── __init__.py
│   ├── allocation.py
│   ├── reservation.py
│   └── tests/
└── shipping/
    ├── __init__.py
    ├── rates.py
    ├── zones.py
    └── tests/
```

### 4. Humble Service (dạng hexagonal architecture)

Khi core logic giao tiếp với bên ngoài qua ports (interfaces).

```python
# core/ports.py — Port definitions (pure abstraction)
from abc import ABC, abstractmethod

class PaymentGateway(ABC):
    @abstractmethod
    def charge(self, amount: Decimal, token: str) -> str:
        ...

class InventoryService(ABC):
    @abstractmethod
    def reserve(self, sku: str, qty: int) -> bool:
        ...

class NotificationService(ABC):
    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> None:
        ...

# core/checkout.py — Business logic (pure, depends on abstractions)
def process_order(
    cart: Cart,
    payment: PaymentGateway,
    inventory: InventoryService,
    notifier: NotificationService,
) -> OrderResult:
    for item in cart.items:
        if not inventory.reserve(item.sku, item.qty):
            return OrderResult.failed(f"Out of stock: {item.sku}")
    txn_id = payment.charge(cart.total(), cart.payment_token)
    notifier.send_email(cart.email, "Order confirmed", f"Txn: {txn_id}")
    return OrderResult.success(txn_id)

# adapters/stripe_payment.py — Adapter implementation (humble)
from core.ports import PaymentGateway
import stripe

class StripePayment(PaymentGateway):
    def charge(self, amount: Decimal, token: str) -> str:
        charge = stripe.Charge.create(amount=int(amount * 100), currency="usd", source=token)
        return charge.id
```

---

## Ví dụ chi tiết qua nhiều ngữ cảnh

### Django REST Framework

**Trước** — Logic trong view:

```python
from rest_framework.views import APIView
from rest_framework.response import Response

class OrderView(APIView):
    def post(self, request):
        items = request.data.get("items", [])
        promo = request.data.get("promo_code")

        # Business logic trộn lẫn với request/response handling
        total = 0
        for item in items:
            total += item["price"] * item["qty"]

        discount = 0
        if promo:
            promos = PromoCode.objects.filter(code=promo, active=True)
            if promos.exists() and total > promos[0].min_total:
                discount = total * promos[0].rate
            else:
                return Response({"error": "Invalid promo"}, status=400)

        final = total - discount
        order = Order.objects.create(total=final, ...)
        return Response({"order_id": order.id, "total": final})
```

**Sau** — Áp dụng Humble Object:

```python
# core/pricing.py
from dataclasses import dataclass

@dataclass
class ItemInput:
    price: float
    qty: int

@dataclass
class PromoInput:
    code: str
    rate: float
    min_total: float

@dataclass
class OrderCalculation:
    items: list[ItemInput]
    promo: PromoInput | None

    def subtotal(self) -> float:
        return sum(i.price * i.qty for i in self.items)

    def apply_promo(self) -> float:
        sub = self.subtotal()
        if not self.promo or sub < self.promo.min_total:
            return 0.0
        return sub * self.promo.rate

    def final_total(self) -> float:
        return self.subtotal() - self.apply_promo()


# core/tests/test_pricing.py
import unittest
from core.pricing import OrderCalculation, ItemInput, PromoInput

class TestOrderCalculation(unittest.TestCase):
    def test_subtotal_sums_all_items(self):
        calc = OrderCalculation(
            items=[ItemInput(10, 2), ItemInput(5, 3)],
            promo=None,
        )
        self.assertEqual(calc.subtotal(), 35.0)

    def test_promo_applied_when_above_minimum(self):
        calc = OrderCalculation(
            items=[ItemInput(100, 1)],
            promo=PromoInput("SAVE10", 0.1, 50),
        )
        self.assertEqual(calc.apply_promo(), 10.0)

    def test_promo_not_applied_below_minimum(self):
        calc = OrderCalculation(
            items=[ItemInput(10, 2)],
            promo=PromoInput("SAVE10", 0.1, 50),
        )
        self.assertEqual(calc.apply_promo(), 0.0)

    def test_final_total_with_promo(self):
        calc = OrderCalculation(
            items=[ItemInput(100, 1)],
            promo=PromoInput("SAVE10", 0.1, 50),
        )
        self.assertEqual(calc.final_total(), 90.0)

    def test_final_total_without_promo(self):
        calc = OrderCalculation(
            items=[ItemInput(100, 1)],
            promo=None,
        )
        self.assertEqual(calc.final_total(), 100.0)


# orders/views.py — Humble view
from rest_framework.views import APIView
from rest_framework.response import Response
from core.pricing import OrderCalculation, ItemInput, PromoInput
from orders.models import Order, PromoCode

class OrderView(APIView):
    def post(self, request):
        items = [ItemInput(i["price"], i["qty"]) for i in request.data["items"]]

        promo = None
        if code := request.data.get("promo_code"):
            promos = PromoCode.objects.filter(code=code, active=True)
            if promos.exists():
                p = promos[0]
                promo = PromoInput(p.code, p.rate, p.min_total)

        calc = OrderCalculation(items, promo)
        total = calc.final_total()

        if promo and calc.apply_promo() == 0:
            return Response({"error": "Promo not applicable"}, status=400)

        order = Order.objects.create(total=total, ...)
        return Response({"order_id": order.id, "total": total})
```

### FastAPI

```python
# core/risk.py
from decimal import Decimal

def should_approve_loan(
    credit_score: int,
    annual_income: Decimal,
    loan_amount: Decimal,
    existing_debt: Decimal,
) -> tuple[bool, str]:
    dti = (existing_debt + loan_amount) / annual_income  # debt-to-income ratio

    if credit_score < 600:
        return False, "Credit score too low"
    if dti > Decimal("0.43"):
        return False, "Debt-to-income ratio exceeds limit"
    if loan_amount > annual_income * Decimal("0.5"):
        return False, "Loan amount exceeds 50% of income"

    if credit_score >= 740:
        return True, "Approved with best rate"
    return True, "Approved with standard rate"


# api/routes.py — Humble route
from fastapi import APIRouter, HTTPException
from core.risk import should_approve_loan

router = APIRouter()

@router.post("/loans/approve")
def approve_loan(req: LoanRequest):
    approved, reason = should_approve_loan(
        credit_score=req.credit_score,
        annual_income=req.annual_income,
        loan_amount=req.loan_amount,
        existing_debt=req.existing_debt,
    )
    if not approved:
        raise HTTPException(status_code=400, detail=reason)
    return {"status": "approved", "reason": reason}
```

### Odoo (ví dụ từ stock module)

```python
# core/stock_availability.py
from typing import Any

def qty_available(quant: Any) -> float:
    return quant.quantity - quant.reserved_quantity

def availability_by_tracking(
    tracking: str, quants: list[Any]
) -> list[float]:
    if tracking == "none":
        return [sum(qty_available(q) for q in quants)]

    lots: dict[int | str, float] = {"untracked": 0.0}
    for q in quants:
        if q.lot_id is None:
            lots["untracked"] += qty_available(q)
        else:
            lots[q.lot_id] = lots.get(q.lot_id, 0.0) + qty_available(q)
    return list(lots.values())


# models/stock_quant.py — Humble model
from odoo import models
from odoo.tools import float_compare
from core.stock_availability import availability_by_tracking

class StockQuant(models.Model):
    _inherit = "stock.quant"

    def _get_available_quantity(self, product_id, location_id, lot_id=None,
                                package_id=None, owner_id=None, strict=False,
                                allow_negative=False):
        self = self.sudo()
        rounding = product_id.uom_id.rounding
        quants = self._gather(product_id, location_id, lot_id=lot_id,
                              package_id=package_id, owner_id=owner_id,
                              strict=strict)

        quantities = availability_by_tracking(product_id.tracking, quants)

        if not allow_negative:
            quantities = [
                q for q in quantities
                if float_compare(q, 0.0, precision_rounding=rounding) > 0
            ]
        return sum(quantities)
```

---

## Mối quan hệ với các Architectural Patterns

### Humble Object vs Hexagonal Architecture (Ports & Adapters)

Humble Object là **trường hợp đặc biệt** của Hexagonal Architecture với quy mô nhỏ hơn:

| Tiêu chí | Humble Object | Hexagonal Architecture |
|----------|--------------|----------------------|
| Phạm vi | Một class / function | Toàn bộ ứng dụng |
| Core | Pure function / dataclass | Domain layer |
| Boundary | Framework class | Port interface |
| Adapter | Không cần (gọi trực tiếp) | Adapter implementation |

### Humble Object vs MVC / MVP

- **MVC**: View hiển thị dữ liệu, Controller xử lý input, Model chứa logic
- **MVP**: Presenter chứa logic presentation, View là interface
- **Humble Object trong MVP**: View là humble — nó chỉ render, không chứa logic. Presenter chứa logic và có thể test dễ dàng.

```python
# MVP với Humble View

# View interface (humble)
class IOrderView(ABC):
    @abstractmethod
    def show_order(self, order_data: dict) -> None: ...
    @abstractmethod
    def show_error(self, message: str) -> None: ...

# Presenter (testable)
class OrderPresenter:
    def __init__(self, view: IOrderView, calculator: CartCalculator):
        self.view = view
        self.calculator = calculator

    def on_checkout(self, items: list, promo: str | None):
        calc = CartCalculator(items, promo)
        if errors := calc.validate():
            self.view.show_error(errors[0])
            return
        self.view.show_order(calc.to_dict())

# Test presenter dễ dàng
class MockView(IOrderView):
    def __init__(self):
        self.data = None
        self.error = None
    def show_order(self, data): self.data = data
    def show_error(self, msg): self.error = msg

def test_presenter_shows_error_for_invalid_order():
    view = MockView()
    presenter = OrderPresenter(view, CartCalculator)
    presenter.on_checkout([{"qty": -1}], None)
    assert view.error is not None
```

### Humble Object vs Domain-Driven Design

Humble Object ánh xạ trực tiếp vào DDD:

- **Domain Service** → Pure function trong core layer
- **Value Object** → Dataclass thuần túy
- **Entity** → Có thể là pure hoặc humble tùy persistence strategy
- **Repository** → Port interface (abstraction), adapter (humble implementation)
- **Application Service** → Humble object điều phối

---

## Chiến lược test toàn diện

### Test Pyramid với Humble Object

```
         /\
        /E2E\
       /------\
      /Integration\
     /--------------\
    /   Unit Tests   \   ← Humble Object đưa nhiều test lên đây
   /--------------------\
  / Manual / Exploratory \
 /------------------------\
```

Humble Object cho phép **đẩy phần lớn test lên tầng Unit Test** — nhanh, rẻ, đáng tin cậy.

### Property-Based Testing

Với pure functions, ta có thể dùng property-based testing (Hypothesis):

```python
from hypothesis import given, strategies as st
from core.pricing import calculate_discount

@given(
    tier=st.sampled_from(["vip", "regular", "none"]),
    total=st.floats(min_value=0, max_value=10000),
)
def test_discount_never_exceeds_total(tier, total):
    discount = calculate_discount(tier, total)
    assert 0 <= discount <= total

@given(
    total=st.floats(min_value=0, max_value=10000),
)
def test_regular_never_gets_discount(total):
    assert calculate_discount("regular", total) == 0.0

@given(
    total=st.floats(min_value=1000, max_value=10000),
)
def test_vip_above_threshold_gets_exact_20_percent(total):
    assert calculate_discount("vip", total) == total * 0.2
```

### Mutation Testing

Với pure functions, mutation testing cho kết quả chính xác hơn vì không có side effect gây nhiễu:

```bash
pip install mutmut
mutmut run --paths-to-mutate core/
```

Nếu mutation pass, test của bạn chưa đủ mạnh.

### Contract Testing (cho Humble Service dạng Ports/Adapters)

```python
# tests/contracts.py
class PaymentGatewayContract(ABC):
    @abstractmethod
    def create_gateway(self) -> PaymentGateway: ...

    def test_successful_charge(self):
        gw = self.create_gateway()
        result = gw.charge(Decimal("10.00"), "valid_token")
        assert result.startswith("txn_")

    def test_failed_charge(self):
        gw = self.create_gateway()
        with pytest.raises(PaymentError):
            gw.charge(Decimal("10.00"), "invalid_token")

class TestStripePayment(PaymentGatewayContract):
    def create_gateway(self):
        return StripePayment(api_key="sk_test_...")
```

---

## Áp dụng trong thực tế

### Legacy Code Migration

Khi đưa Humble Object vào codebase cũ, dùng **Sprout Method** và **Sprout Class**:

```python
# Bước 1: Sprout Method — thêm pure function bên cạnh code cũ
def calculate_shipping_cost(weight: float, zone: str) -> float:
    rates = {"A": 5.0, "B": 7.5, "C": 12.0}
    return weight * rates.get(zone, 15.0)

# Bước 2: Gọi từ code cũ, dần dần thay thế
class Order(models.Model):
    def compute_shipping(self):
        # ... logic cũ phức tạp ...
        # Thay thế dần:
        if hasattr(self, "_use_new_shipping"):
            return calculate_shipping_cost(self.weight, self.zone)
        # ... logic cũ ...

# Bước 3: Khi đã test đủ, xóa code cũ
class Order(models.Model):
    def compute_shipping(self):
        return calculate_shipping_cost(self.weight, self.zone)
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e core/
      - run: python -m pytest core/tests/ --cov=core --cov-report=xml
      # Chạy trong < 10s, không cần database, không cần framework

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
    steps:
      - run: python -m pytest tests/integration/
      # Chạy ít test hơn, chậm hơn, nhưng cần thiết
```

### Test Coverage Thresholds

Với Humble Object, bạn có thể đặt ngưỡng coverage riêng:

```ini
# .coveragerc
[run]
source = core, models

[report]
# Core layer phải đạt coverage cao
show_missing = True
precision = 2

[report:core]
fail_under = 95

[report:models]
fail_under = 70  # Humble layer có thể thấp hơn
```

---

## Khi nào KHÔNG nên dùng Humble Object

### 1. Logic quá đơn giản

```python
# Không cần tách — quá đơn giản
class UserView(APIView):
    def get(self, request):
        return Response({"count": User.objects.count()})
```

Việc tách `count_users()` là over-engineering.

### 2. Pure function không mang lại lợi ích

Khi method chỉ là ORM query đơn thuần, không có logic rẽ nhánh:

```python
# Không cần tách — không có business logic
class ProductView(APIView):
    def get(self, request, pk):
        product = get_object_or_404(Product, pk=pk)
        return Response(ProductSerializer(product).data)
```

### 3. Hiệu suất quan trọng hơn testability

Trong hệ thống real-time, việc tách layer có thể thêm overhead không đáng có. Cân nhắc trade-off.

### 4. Prototype / MVP

Trong giai đoạn khám phá, việc áp dụng Humble Object quá sớm làm chậm tốc độ. Hãy refactor khi product-market fit đã được xác nhận.

---

## So sánh với các giải pháp khác

| Giải pháp | Độ khó | Test speed | Phụ thuộc framework | Bảo trì |
|-----------|--------|-----------|-------------------|---------|
| **Humble Object** | Thấp | Rất nhanh | Không | Dễ |
| **Dependency Injection** | Trung bình | Nhanh | Có thể có | Trung bình |
| **Mocking** | Thấp (ban đầu) | Trung bình | Có | Khó (mock coupling) |
| **Integration Test** | Thấp | Chậm | Có | Dễ (ít code hơn) |
| **Test Containers** | Cao | Chậm | Có (thật) | Trung bình |

Humble Object thường kết hợp tốt nhất với **Dependency Injection** ở tầng Adapter và **Mocking** ở tầng Port interface.

---

## Kết luận

Humble Object không phải là pattern cao siêu. Nó chỉ đơn giản là: **đừng đặt logic trong framework class nếu bạn muốn test nó.**

Nhưng sức mạnh của nó nằm ở tính phổ quát:

- Áp dụng được ở mọi quy mô — từ function nhỏ đến toàn bộ hệ thống
- Kết hợp được với mọi architecture — MVC, MVP, Hexagonal, DDD
- Không yêu cầu thư viện hay tool đặc biệt
- Có thể áp dụng dần dần vào legacy code
- Test chạy nhanh — khuyến khích developer viết test nhiều hơn

Triết lý đằng sau Humble Object cũng chính là triết lý của software engineering tốt:

> *Separate what changes from what stays the same. Separate what's hard to test from what's easy to test. Separate what's complex from what's simple.*

Bắt đầu từ việc nhỏ — lần tới khi bạn viết một method trong model/view/controller, hãy tự hỏi: **"Phần nào ở đây là business logic? Và tôi có thể tách nó ra để test mà không cần framework không?"**

Đó là bước đầu tiên để viết code testable hơn, sạch hơn, và đáng tin cậy hơn.
