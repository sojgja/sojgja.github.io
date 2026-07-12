---
id: adapter
title: Adapter
sidebar_label: 🔌 Adapter
sidebar_position: 7
---

# Adapter

> "Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces." — Erich Gamma, *Design Patterns: Elements of Reusable Object-Oriented Software*

Bạn đã bao giờ mua một cái adapter chuyển đổi đầu cắm điện kiểu Mỹ sang châu Âu chưa? Bạn không thể sửa ổ điện, cũng không thể sửa đầu cắm. Bạn cần một "bộ trung gian" — một adapter. Pattern hôm nay cũng làm điều tương tự, nhưng với code.

## Bài toán chi tiết

Một công ty thương mại điện tử đang vận hành hệ thống xử lý thanh toán được xây dựng cách đây 5 năm với class `LegacyPaymentProcessor`. Hệ thống này được tích hợp sâu vào hàng trăm module khác nhau: từ giỏ hàng, đơn hàng, đến module hoàn tiền và báo cáo tài chính. Interface của nó rất đơn giản: method `process_payment(order_id: str, amount: float) -> bool`. Toàn bộ codebase đều gọi vào method này, và mọi thứ đều hoạt động ổn định trong nhiều năm.

Khi công ty mở rộng thị trường ra quốc tế, họ quyết định tích hợp cổng thanh toán Stripe — một giải pháp hiện đại hỗ trợ đa tiền tệ, webhook, và bảo mật PCI DSS. Tuy nhiên, Stripe cung cấp interface hoàn toàn khác: `StripeClient.charge(amount_cents: int, currency: str, source: str, description: str) -> StripeResponse`. Method này yêu cầu số tiền tính bằng cent, mã tiền tệ ISO 4217, và token nguồn thanh toán. Nó trả về một object phức tạp chứa transaction ID, status, và nhiều trường metadata.

Vấn đề xuất hiện ngay lập tức. Đội ngũ phát triển không thể sửa interface của Stripe vì đó là thư viện bên thứ ba. Họ cũng không thể sửa tất cả các chỗ gọi `process_payment` vì có hàng trăm điểm tích hợp, mỗi điểm lại nằm trong các module khác nhau do nhiều team quản lý. Việc sửa từng chỗ một vừa tốn thời gian, vừa rủi ro cao vì có thể gây lỗi lan truyền. Một giải pháp kế thừa đơn thuần cũng không khả thi, vì `LegacyPaymentProcessor` và `StripeClient` không có quan hệ cha-con về mặt ngữ nghĩa.

Nếu không có giải pháp phù hợp, công ty buộc phải viết một lớp trung gian phức tạp kết hợp cả hai hệ thống song song, hoặc tồi tệ hơn là dùng câu lệnh `if...else` để phân nhánh logic — một giải pháp không bền vững và khó bảo trì khi số lượng cổng thanh toán tăng lên.

Tôi đã thấy cảnh này nhiều lần. Cứ mỗi lần thêm một cổng thanh toán mới, code lại phình ra thêm một đoạn `if`. Rồi một ngày đẹp trời, bạn có 50 `if` và chẳng ai dám đụng vào.

## Giải pháp với Pattern

Adapter Pattern giải quyết triệt để vấn đề này bằng cách đóng vai trò như một "bộ chuyển đổi" nằm giữa client và lớp không tương thích. Nó giữ nguyên interface mà client đang phụ thuộc vào, nhưng bên trong chuyển hướng các lời gọi đến interface thực tế của thư viện mới. Client hoàn toàn không biết mình đang tương tác với adapter hay với implementation cũ — đây là sức mạnh của tính đa hình (polymorphism).

Cụ thể, Adapter Pattern gồm bốn thành phần chính:
- **Target**: Interface mà client đang sử dụng (ví dụ `process_payment`).
- **Client**: Code hiện có phụ thuộc vào Target interface.
- **Adaptee**: Class không tương thích cần tích hợp (Stripe, PayPal, v.v.).
- **Adapter**: Lớp trung gian implement Target interface và ủy quyền (delegate) các lời gọi cho Adaptee, đồng thời thực hiện mọi chuyển đổi dữ liệu cần thiết.

Trong trường hợp thanh toán, `StripeAdapter` giữ nguyên method `process_payment(order_id, amount)` nhưng bên trong gọi `stripe.charge()` với các tham số được chuyển đổi: từ VND sang cent, từ order_id sang description, v.v. Nhờ vậy, toàn bộ codebase không cần thay đổi dù backend thanh toán đã thay đổi hoàn toàn.

## Phân tích thiết kế

Adapter Pattern vận hành dựa trên nguyên lý **Single Responsibility Principle** (SRP): mỗi adapter chỉ chịu trách nhiệm chuyển đổi giữa hai interface cụ thể. Nó cũng tuân thủ **Open/Closed Principle** (OCP): hệ thống có thể thêm cổng thanh toán mới mà không cần sửa code client, chỉ cần viết thêm adapter mới.

Có hai biến thể chính của Adapter:
- **Object Adapter** (dùng composition): Adapter chứa một tham chiếu đến Adaptee và dùng delegation. Đây là biến thể linh hoạt hơn, hoạt động với mọi subclass của Adaptee, và được khuyến khích sử dụng trong hầu hết trường hợp.
- **Class Adapter** (dùng đa kế thừa): Adapter kế thừa cả Target và Adaptee. Biến thể này phù hợp khi cần ghi đè (override) một số hành vi của Adaptee. Tuy nhiên, Python hỗ trợ đa kế thừa nên có thể áp dụng, nhưng composition vẫn được ưu tiên hơn để tránh diamond problem.

**Khi KHÔNG nên dùng Adapter:**
- Khi có thể sửa trực tiếp interface của Adaptee (đơn giản hơn nhiều).
- Khi hầu hết các method của Adaptee đều không cần thiết — lúc đó adapter trở nên phức tạp và khó bảo trì, nên cân nhắc Facade pattern.
- Khi interface của Target và Adaptee quá khác biệt, dẫn đến adapter quá dài và vi phạm SRP.

**Trade-offs:**
- Adapter làm tăng độ phức tạp của hệ thống vì thêm một lớp gián tiếp.
- Hiệu năng giảm nhẹ do có thêm một lớp gọi hàm (method delegation), nhưng trong thực tế mức ảnh hưởng không đáng kể.
- Debugging khó hơn vì phải theo dõi qua nhiều lớp.

## Ví dụ code hoàn chỉnh

### Cách làm sai: Dùng if-else và sửa client

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import json
import time


class Currency(Enum):
    VND = "vnd"
    USD = "usd"
    EUR = "eur"


@dataclass
class PaymentResult:
    success: bool
    transaction_id: str
    message: str
    amount_processed: float


# --- Legacy System ---
class LegacyPaymentProcessor:
    """Hệ thống thanh toán cũ — chỉ hỗ trợ VND."""

    def process_payment(self, order_id: str, amount: float) -> bool:
        print(f"[Legacy] Processing order {order_id} for {amount} VND...")
        # Giả lập xử lý
        time.sleep(0.05)
        return True


# --- Third-party library (Stripe) ---
class StripeClient:
    """Thư viện Stripe mới — interface hoàn toàn khác."""

    def charge(self, amount_cents: int, currency: str, source: str) -> dict:
        print(f"[Stripe] Charging {amount_cents} {currency} from {source}")
        time.sleep(0.05)
        return {
            "id": f"ch_{int(time.time())}",
            "status": "succeeded",
            "amount": amount_cents,
            "currency": currency,
        }

    def refund(self, charge_id: str) -> dict:
        print(f"[Stripe] Refunding charge {charge_id}")
        return {"status": "refunded"}


# --- PAYMENT PROCESSOR — WRONG WAY ---
class PaymentService:
    """Client bị ô nhiễm bởi logic phân nhánh cổng thanh toán."""

    def __init__(self, use_stripe: bool = False):
        self._use_stripe = use_stripe
        self._legacy = LegacyPaymentProcessor()
        self._stripe = StripeClient()

    def process_order(self, order_id: str, amount: float, currency: str = "VND") -> bool:
        if self._use_stripe:
            # Client phải tự chuyển đổi — vi phạm SRP
            amount_cents = int(amount * 100) if currency == "USD" else int(amount)
            if currency == "VND":
                amount_cents = int(amount)
            elif currency == "USD":
                amount_cents = int(amount * 100)
            else:
                amount_cents = int(amount)
            result = self._stripe.charge(amount_cents, currency, "tok_visa")
            return result["status"] == "succeeded"
        else:
            return self._legacy.process_payment(order_id, amount)
```

### Cách đúng: Dùng Adapter Pattern

```python
# --- Common Interface (Target) ---
class PaymentProcessor(ABC):
    """Interface chung mà toàn bộ hệ thống phụ thuộc vào."""

    @abstractmethod
    def process_payment(self, order_id: str, amount: float) -> PaymentResult:
        ...

    @abstractmethod
    def refund_payment(self, transaction_id: str) -> PaymentResult:
        ...


# --- Concrete Implementations ---
class LegacyPaymentAdapter(PaymentProcessor):
    """Adapter cho LegacyPaymentProcessor — giữ nguyên interface cũ."""

    def __init__(self) -> None:
        self._legacy = LegacyPaymentProcessor()

    def process_payment(self, order_id: str, amount: float) -> PaymentResult:
        success = self._legacy.process_payment(order_id, amount)
        return PaymentResult(
            success=success,
            transaction_id=f"legacy_{order_id}",
            message="Processed via legacy system",
            amount_processed=amount,
        )

    def refund_payment(self, transaction_id: str) -> PaymentResult:
        print(f"[Legacy] Refunding transaction {transaction_id}")
        return PaymentResult(
            success=True,
            transaction_id=transaction_id,
            message="Refunded via legacy system",
            amount_processed=0.0,
        )


class StripeAdapter(PaymentProcessor):
    """Adapter chuyển đổi từ Stripe interface sang PaymentProcessor interface."""

    def __init__(self, api_key: str, currency: Currency = Currency.VND) -> None:
        self._stripe = StripeClient()
        self._api_key = api_key
        self._currency = currency

    def process_payment(self, order_id: str, amount: float) -> PaymentResult:
        amount_cents = self._to_cents(amount)
        currency_code = self._currency.value
        try:
            response = self._stripe.charge(amount_cents, currency_code, "tok_visa")
            return PaymentResult(
                success=response["status"] == "succeeded",
                transaction_id=response["id"],
                message=f"Stripe charge {response['status']}",
                amount_processed=amount,
            )
        except Exception as exc:
            return PaymentResult(
                success=False,
                transaction_id="",
                message=f"Stripe error: {exc}",
                amount_processed=0.0,
            )

    def refund_payment(self, transaction_id: str) -> PaymentResult:
        try:
            response = self._stripe.refund(transaction_id)
            return PaymentResult(
                success=True,
                transaction_id=transaction_id,
                message=f"Refund {response['status']}",
                amount_processed=0.0,
            )
        except Exception as exc:
            return PaymentResult(
                success=False,
                transaction_id=transaction_id,
                message=f"Refund error: {exc}",
                amount_processed=0.0,
            )

    def _to_cents(self, amount: float) -> int:
        if self._currency == Currency.USD:
            return int(amount * 100)
        return int(amount)


class PayPalAdapter(PaymentProcessor):
    """Adapter cho PayPal — minh họa khả năng mở rộng."""

    def __init__(self, client_id: str, secret: str) -> None:
        self._client_id = client_id
        self._secret = secret
        print(f"[PayPal] Initialized with client_id={client_id[:8]}...")

    def process_payment(self, order_id: str, amount: float) -> PaymentResult:
        # Giả lập PayPal API call
        print(f"[PayPal] Creating payment for order {order_id}, amount={amount} USD")
        time.sleep(0.05)
        return PaymentResult(
            success=True,
            transaction_id=f"paypal_{order_id}_{int(time.time())}",
            message="PayPal payment approved",
            amount_processed=amount,
        )

    def refund_payment(self, transaction_id: str) -> PaymentResult:
        print(f"[PayPal] Refunding {transaction_id}")
        return PaymentResult(
            success=True,
            transaction_id=transaction_id,
            message="PayPal refund completed",
            amount_processed=0.0,
        )


# --- Client — hoàn toàn không thay đổi ---
class OrderService:
    """Dịch vụ đơn hàng — không cần biết backend thanh toán là gì."""

    def __init__(self, payment_processor: PaymentProcessor) -> None:
        self._payment_processor = payment_processor

    def checkout(self, order_id: str, total: float) -> None:
        result = self._payment_processor.process_payment(order_id, total)
        if result.success:
            print(f"Order {order_id} paid successfully. TX: {result.transaction_id}")
        else:
            print(f"Order {order_id} FAILED: {result.message}")


# --- Usage ---
if __name__ == "__main__":
    # Legacy system
    legacy = LegacyPaymentAdapter()
    service = OrderService(legacy)
    service.checkout("ORD-001", 150000.0)

    # Chuyển sang Stripe — không sửa OrderService
    stripe = StripeAdapter(api_key="sk_test_xxx", currency=Currency.VND)
    service = OrderService(stripe)
    service.checkout("ORD-002", 250000.0)

    # PayPal
    paypal = PayPalAdapter(client_id="test_client", secret="test_secret")
    service = OrderService(paypal)
    service.checkout("ORD-003", 99.99)
```

## Sơ đồ UML

```
┌──────────────┐        ┌──────────────────────┐
│    Client    │        │   «interface»        │
│ (OrderService)│──────▶│  PaymentProcessor    │
└──────────────┘        │──────────────────────│
                        │+ process_payment()   │
                        │+ refund_payment()    │
                        └────────┬─────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
        ┌───────────┴───┐ ┌──────┴──────┐ ┌───┴──────────┐
        │LegacyPayment  │ │StripeAdapter│ │PayPalAdapter │
        │   Adapter     │ │             │ │              │
        │───────────────│ │─────────────│ │──────────────│
        │- legacy:      │ │- stripe:    │ │- client_id   │
        │  LegacyPayment│ │  StripeClient│ │- secret      │
        │  Processor    │ │- currency   │ │              │
        └───────┬───────┘ └──────┬──────┘ └──────────────┘
                │                │
        ┌───────┴───────┐ ┌──────┴──────┐
        │LegacyPayment  │ │StripeClient │
        │  Processor    │ │(Adaptee)    │
        │───────────────│ │─────────────│
        │+ process_     │ │+ charge()   │
        │  payment()    │ │+ refund()   │
        └───────────────┘ └─────────────┘
```

## So sánh với Pattern liên quan

**Adapter vs Bridge**: Bridge được thiết kế từ đầu để tách abstraction khỏi implementation, trong khi Adapter ra đời sau để hàn gắn sự không tương thích giữa các hệ thống có sẵn. Bridge là giải pháp chủ động (proactive design), Adapter là giải pháp đối phó (reactive integration). Bridge thường dùng composition ngay từ đầu, còn Adapter thường được thêm vào sau khi hệ thống đã hoạt động.

**Adapter vs Facade**: Cả hai đều là wrapper, nhưng mục đích khác nhau. Adapter chuyển đổi interface này sang interface khác nhằm đạt được sự tương thích, trong khi Facade đơn giản hóa một interface phức tạp thành một interface dễ dùng hơn. Facade có thể wrap nhiều class cùng lúc, còn Adapter thường chỉ wrap một class. Facade không yêu cầu interface phải tương thích — nó tạo ra interface mới hoàn toàn.

**Adapter vs Decorator**: Cả hai đều dùng wrapper và delegation, nhưng Decorator thêm hành vi mới trong khi Adapter chuyển đổi interface. Decorator không thay đổi interface — nó implement cùng interface với đối tượng gốc. Adapter bắt buộc phải thay đổi interface.

## Ứng dụng thực tế

**1. Django ORM — Database backend adapter**: Django sử dụng adapter pattern thông qua `DatabaseWrapper` để hỗ trợ nhiều backend database khác nhau (PostgreSQL, MySQL, SQLite, Oracle). Mỗi backend implement cùng một interface nhưng chuyển đổi các query thành SQL tương ứng. Ví dụ:

```python
# django/db/backends/postgresql/base.py
class DatabaseWrapper(BaseDatabaseWrapper):
    def get_new_connection(self, conn_params):
        return psycopg2.connect(**conn_params)
```

**2. Zope Component Architecture**: Hệ thống adapter registry mạnh mẽ trong Zope cho phép đăng ký adapter động. Khi cần một interface cụ thể, framework tự động tìm và áp dụng adapter phù hợp:

```python
from zope.interface import Interface, implementer
from zope.component import getAdapter

class IStripePayment(Interface):
    pass

@implementer(IStripePayment)
class StripeAdapter:
    def __init__(self, context):
        self.context = context
```

**3. Python's `socket` module**: `makefile()` method trả về một file-like object từ socket — đây là adapter chuyển từ socket interface sang file interface, cho phép dùng `read()`, `write()` thay vì `send()`, `recv()`:

```python
import socket
s = socket.socket()
s.connect(('example.com', 80))
# Adapter: socket → file-like
f = s.makefile('rw')
f.write(b'GET / HTTP/1.0\r\n\r\n')
response = f.read()
```

**4. Square Peg Round Hole — logging adapter**: Thư viện `loguru` cung cấp adapter để tích hợp với logging chuẩn của Python, cho phép chuyển đổi giữa hai hệ thống logging:

```python
from loguru import logger
import logging

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger.log(record.levelname, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0)
```

## Kiểm thử

```python
import pytest
from unittest.mock import patch, MagicMock
from adapter import (
    PaymentProcessor, StripeAdapter, LegacyPaymentAdapter,
    PayPalAdapter, OrderService, Currency, PaymentResult,
)


class TestLegacyPaymentAdapter:
    def setup_method(self) -> None:
        self.adapter = LegacyPaymentAdapter()

    def test_process_payment_success(self) -> None:
        result = self.adapter.process_payment("ORD-001", 100000.0)
        assert result.success is True
        assert result.transaction_id == "legacy_ORD-001"
        assert result.amount_processed == 100000.0

    def test_refund_payment(self) -> None:
        result = self.adapter.refund_payment("legacy_001")
        assert result.success is True
        assert result.amount_processed == 0.0


class TestStripeAdapter:
    def setup_method(self) -> None:
        self.adapter = StripeAdapter(api_key="sk_test", currency=Currency.VND)

    @patch.object(StripeAdapter, "_stripe")
    def test_process_payment_stripe_called(self, mock_stripe) -> None:
        mock_stripe.charge.return_value = {
            "id": "ch_123", "status": "succeeded", "amount": 50000, "currency": "vnd"
        }
        result = self.adapter.process_payment("ORD-002", 50000.0)
        assert result.success is True
        assert result.transaction_id == "ch_123"
        mock_stripe.charge.assert_called_once_with(50000, "vnd", "tok_visa")

    @patch.object(StripeAdapter, "_stripe")
    def test_process_payment_failure(self, mock_stripe) -> None:
        mock_stripe.charge.side_effect = Exception("Card declined")
        result = self.adapter.process_payment("ORD-003", 100.0)
        assert result.success is False
        assert "Card declined" in result.message


class TestOrderService:
    def test_checkout_calls_processor(self) -> None:
        mock_processor = MagicMock(spec=PaymentProcessor)
        mock_processor.process_payment.return_value = PaymentResult(
            success=True, transaction_id="tx_001", message="OK", amount_processed=100.0
        )
        service = OrderService(mock_processor)
        service.checkout("ORD-001", 100.0)
        mock_processor.process_payment.assert_called_once_with("ORD-001", 100.0)

    def test_adapter_polymorphism(self) -> None:
        """Verify different adapters work via common interface."""
        adapters: list[PaymentProcessor] = [
            LegacyPaymentAdapter(),
            StripeAdapter(api_key="test", currency=Currency.USD),
            PayPalAdapter(client_id="test", secret="test"),
        ]
        for adapter in adapters:
            result = adapter.process_payment("TEST", 100.0)
            assert isinstance(result, PaymentResult), f"Failed for {type(adapter).__name__}"
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Tuân thủ SRP và OCP — thêm tích hợp mới không sửa code cũ | Tăng độ phức tạp do thêm lớp gián tiếp |
| Tái sử dụng code tối đa — không cần viết lại logic client | Có thể che giấu lỗi nếu adapter chuyển đổi sai |
| Giảm coupling — client không phụ thuộc vào implementation cụ thể | Debugging khó khăn hơn do nhiều lớp wrapper |
| Dễ dàng thay thế thư viện bên thứ ba | Nếu interface quá khác biệt, adapter trở nên phức tạp |
| Cho phép tích hợp legacy system với modern system | Performance overhead nhẹ do delegation |

---

## Kết luận

Adapter Pattern là giải pháp bất di bất dịch cho vấn đề tích hợp hệ thống. Hãy áp dụng Adapter khi bạn đang đối mặt với interface không tương thích và không thể (hoặc không muốn) sửa một trong hai phía. Pattern này đặc biệt hữu dụng trong kiến trúc microservice, nơi các dịch vụ giao tiếp qua API với interface khác nhau.

**Nguyên tắc vàng**: Nếu bạn thấy code client chứa `if gateway == "stripe": ... elif gateway == "paypal": ...`, đó là dấu hiệu chắc chắn bạn cần Adapter. Hãy tạo interface chung và một adapter cho mỗi gateway để giữ cho client luôn sạch sẽ và tập trung vào business logic.

Như Alan Perlis từng nói: *"A language that doesn't affect the way you think about programming is not worth knowing."* Adapter Pattern thay đổi cách bạn nghĩ về sự tương thích — thay vì ép buộc các hệ thống phải giống nhau, hãy xây cầu nối giữa chúng.

---

*Trân trọng!*
