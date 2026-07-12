---
id: factory-method
title: Factory Method
sidebar_label: 🏭 Factory Method
sidebar_position: 3
---

# Factory Method

> *"Define an interface for creating an object, but let subclasses decide which class to instantiate. Factory Method lets a class defer instantiation to subclasses."* — Gang of Four, *Design Patterns: Elements of Reusable Object-Oriented Software*, 1994.

Bạn có bao giờ cảm thấy mệt mỏi với những câu lệnh `if/elif` lằng nhằng để chọn class cần khởi tạo? Tôi thì có — rất nhiều lần.

**Factory Method** thuộc nhóm **Creational Patterns**. Nó cung cấp một interface để tạo object, nhưng cho phép subclass quyết định class cụ thể nào được khởi tạo. Nghe có vẻ đơn giản, nhưng sức mạnh của nó nằm ở chỗ: **chuyển trách nhiệm khởi tạo từ client code sang subclass**. Đây chính là **Dependency Inversion Principle** trong thực tế — module cấp cao không phụ thuộc vào module cấp thấp, cả hai đều phụ thuộc vào abstraction.

## Bài toán chi tiết

Hãy tưởng tượng bạn đang phát triển một **hệ thống thanh toán trực tuyến** cho một sàn thương mại điện tử đa quốc gia. Ban đầu, sàn chỉ hỗ trợ thanh toán qua **thẻ tín dụng** (Visa, MasterCard). Logic xử lý thanh toán được viết trực tiếp trong class `OrderProcessor`:

```python
class OrderProcessor:
    def process_payment(self, order, card_number, expiry, cvv):
        payment = CreditCardPayment(card_number, expiry, cvv)
        payment.charge(order.total)
```

Mọi thứ đều ổn... cho đến khi sàn mở rộng ra các thị trường mới. Khách hàng ở châu Âu yêu cầu **PayPal**, khách hàng ở châu Á yêu cầu **VNPay** và **WeChat Pay**, khách hàng doanh nghiệp yêu cầu **chuyển khoản ngân hàng**. Mỗi phương thức thanh toán có:
- Cách xác thực khác nhau (OAuth token, mã QR, chữ ký số).
- Cách tính phí khác nhau (phần trăm, fixed fee, hoặc miễn phí).
- Cách xử lý hoàn tiền khác nhau.
- Cách xử lý lỗi khác nhau (timeout, insufficient funds, fraud detection).

Nếu tiếp tục dùng cách cũ, `OrderProcessor` sẽ phình to với hàng tá điều kiện `if/elif`:

```python
class OrderProcessor:
    def process_payment(self, order, method, **kwargs):
        if method == "credit_card":
            payment = CreditCardPayment(kwargs["card_number"], ...)
        elif method == "paypal":
            payment = PayPalPayment(kwargs["token"], ...)
        elif method == "vnpay":
            payment = VNPayPayment(kwargs["qr_code"], ...)
        elif method == "bank_transfer":
            payment = BankTransferPayment(kwargs["account"], ...)
        # ... thêm một elif nữa mỗi khi có phương thức thanh toán mới
        payment.charge(order.total)
```

Tôi từng chứng kiến một dự án với hơn 20 `elif` trong một hàm. Bạn biết kết cục không? Mỗi lần thêm method mới là một lần cầu nguyện — "xin đừng hỏng chỗ cũ."

Vấn đề với cách tiếp cận này:
1. **Vi phạm Open/Closed Principle**: Mỗi lần thêm phương thức thanh toán mới, bạn phải sửa class `OrderProcessor`.
2. **Rủi ro regression**: Sửa một dòng trong `process_payment` có thể làm hỏng toàn bộ logic thanh toán.
3. **Khó kiểm thử**: Không thể test riêng lẻ từng payment method mà không khởi tạo cả `OrderProcessor`.
4. **Code trùng lặp**: Logic khởi tạo (gọi constructor, cấu hình tham số) bị lặp lại ở nhiều nơi trong codebase.

## Giải pháp với Pattern

Factory Method giải quyết triệt để bằng cách tách **phần khởi tạo object** ra khỏi **phần sử dụng object**. Nghe như chia tay vậy — mỗi bên một trách nhiệm.

Cụ thể:

1. **Product interface** (`PaymentMethod`): Định nghĩa interface chung cho tất cả các phương thức thanh toán.
2. **Concrete Products** (`CreditCardPayment`, `PayPalPayment`, ...): Implement cụ thể từng phương thức.
3. **Creator interface** (`PaymentFactory`): Định nghĩa factory method `create_payment()`.
4. **Concrete Creators** (`CreditCardFactory`, `PayPalFactory`, ...): Override factory method để tạo product tương ứng.

Khi cần thêm phương thức thanh toán mới (ví dụ: `CryptoPayment`), bạn chỉ cần:
1. Tạo class `CryptoPayment` implements `PaymentMethod`.
2. Tạo class `CryptoFactory` implements `PaymentFactory`.
3. **Không động gì đến code đang chạy.**

Client code (`OrderProcessor`) chỉ làm việc với interface `PaymentFactory` — nó không biết và không cần biết class cụ thể nào được tạo ra. Điều này đảm bảo **loose coupling** giữa client và concrete classes.

## Phân tích thiết kế

**OOP Principles áp dụng:**

- **Open/Closed Principle**: Hệ thống mở cho việc mở rộng (thêm payment method), đóng cho việc sửa đổi (không sửa code cũ).
- **Dependency Inversion Principle**: Client và product đều phụ thuộc vào abstraction (`PaymentFactory`, `PaymentMethod`).
- **Single Responsibility Principle**: Mỗi class chỉ làm một việc: Product xử lý business logic, Creator quản lý việc tạo product.
- **Liskov Substitution Principle**: Bất kỳ ConcreteProduct nào cũng có thể thay thế Product interface — client không bị ảnh hưởng.

**Trade-offs và rủi ro:**

- **Complexity tăng**: Thay vì một class, bạn có cả một hệ thống class (Product interface, Creator interface, nhiều Concrete classes). Với ứng dụng nhỏ, đây là over-engineering.
- **Parallel class hierarchy**: Factory Method thường tạo ra hai hệ thống class song song (Product hierarchy và Creator hierarchy), làm tăng số lượng class đáng kể.
- **Không linh hoạt với tham số động**: Nếu việc chọn class phụ thuộc vào runtime parameters phức tạp, Factory Method có thể không đủ — cần Abstract Factory hoặc Prototype.

**Khi nào KHÔNG nên dùng Factory Method:**

- Khi chỉ có một loại product duy nhất và không có kế hoạch mở rộng.
- Khi việc khởi tạo object quá đơn giản (chỉ gọi `ClassName()` không tham số).
- Khi bạn không muốn tạo thêm class cho mỗi product type (chi phí bảo trì quá cao).
- Khi có thể dùng **Simple Factory** (một class factory duy nhất với method tĩnh) hoặc **lazy function** thay vì pattern đầy đủ.

## Ví dụ code hoàn chỉnh

### Cách làm sai (Violating OCP)

```python
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional
import hashlib
import hmac
import time


class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class Order:
    order_id: str
    total: Decimal
    currency: str
    customer_email: str


class PaymentResult:
    def __init__(self, success: bool, transaction_id: Optional[str] = None,
                 message: str = "", status: PaymentStatus = PaymentStatus.PENDING):
        self.success = success
        self.transaction_id = transaction_id
        self.message = message
        self.status = status


class OrderProcessor:
    """Cách làm sai: mỗi lần thêm payment method mới, phải sửa class này."""

    def process_payment(self, order: Order, method: str, **kwargs) -> PaymentResult:
        if method == "credit_card":
            # Xác thực credit card
            card_number = kwargs.get("card_number")
            expiry = kwargs.get("expiry")
            cvv = kwargs.get("cvv")
            if not card_number or not expiry or not cvv:
                return PaymentResult(False, message="Thiếu thông tin thẻ")
            # Gọi API cổng thanh toán
            print(f"[CreditCard] Đang xử lý {order.total} {order.currency}")
            # ... logic riêng của credit card
            return PaymentResult(True, transaction_id=f"CC-{order.order_id}")

        elif method == "paypal":
            token = kwargs.get("token")
            if not token:
                return PaymentResult(False, message="Thiếu PayPal token")
            print(f"[PayPal] Đang xử lý {order.total} {order.currency}")
            # ... logic riêng của PayPal (OAuth, v.v.)
            return PaymentResult(True, transaction_id=f"PP-{order.order_id}")

        elif method == "vnpay":
            qr_code = kwargs.get("qr_code")
            if not qr_code:
                return PaymentResult(False, message="Thiếu mã QR")
            print(f"[VNPay] Đang xử lý {order.total} {order.currency}")
            # ... logic riêng của VNPay
            return PaymentResult(True, transaction_id=f"VN-{order.order_id}")

        elif method == "bank_transfer":
            account_number = kwargs.get("account_number")
            bank_code = kwargs.get("bank_code")
            if not account_number or not bank_code:
                return PaymentResult(False, message="Thiếu thông tin tài khoản")
            print(f"[BankTransfer] Đang xử lý {order.total} {order.currency}")
            # ... logic riêng của bank transfer
            return PaymentResult(True, transaction_id=f"BT-{order.order_id}")

        else:
            return PaymentResult(False, message=f"Phương thức thanh toán không hỗ trợ: {method}")


# Sử dụng — mọi thứ đều trong một class
processor = OrderProcessor()
order = Order("ORD-001", Decimal("250000"), "VND", "customer@example.com")
result = processor.process_payment(order, "credit_card", card_number="4111111111111111",
                                    expiry="12/26", cvv="123")
print(f"Kết quả: {result.success}, Transaction: {result.transaction_id}")
```

### Refactored với Factory Method

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Optional, Protocol
import secrets


# ============== DOMAIN MODELS ==============

class PaymentStatus(Enum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REFUNDED = auto()
    PARTIALLY_REFUNDED = auto()


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    VND = "VND"
    JPY = "JPY"


@dataclass
class Order:
    order_id: str
    total: Decimal
    currency: Currency
    customer_email: str
    description: str = ""


@dataclass
class Transaction:
    transaction_id: str
    order_id: str
    amount: Decimal
    currency: Currency
    status: PaymentStatus
    gateway_reference: str = ""
    error_message: str = ""


# ============== PRODUCT: PAYMENT METHOD ==============

class PaymentMethod(ABC):
    """Abstract Product — interface chung cho mọi phương thức thanh toán."""

    @abstractmethod
    def validate(self) -> bool:
        """Kiểm tra thông tin thanh toán có hợp lệ không."""
        ...

    @abstractmethod
    def charge(self, order: Order) -> Transaction:
        """Thực hiện giao dịch thanh toán."""
        ...

    @abstractmethod
    def refund(self, transaction: Transaction) -> Transaction:
        """Hoàn tiền cho giao dịch."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Tên hiển thị của phương thức thanh toán."""
        ...


# ============== CONCRETE PRODUCTS ==============

@dataclass
class CreditCardInfo:
    card_number: str
    expiry_month: int
    expiry_year: int
    cvv: str
    cardholder_name: str


class CreditCardPayment(PaymentMethod):
    """Thanh toán bằng thẻ tín dụng — tích hợp cổng Stripe."""

    def __init__(self, card_info: CreditCardInfo) -> None:
        self.card_info = card_info

    def _mask_card(self) -> str:
        """Che số thẻ, chỉ hiện 4 số cuối."""
        return f"****{self.card_info.card_number[-4:]}"

    def validate(self) -> bool:
        if len(self.card_info.card_number) != 16:
            return False
        if not self.card_info.card_number.isdigit():
            return False
        if self.card_info.expiry_year < 2024:
            return False
        if len(self.card_info.cvv) != 3 or not self.card_info.cvv.isdigit():
            return False
        return True

    def charge(self, order: Order) -> Transaction:
        # Mô phỏng gọi Stripe API
        tx_id = f"CC-{order.order_id}-{secrets.token_hex(4).upper()}"
        print(f"[{self.name()}] {order.total} {order.currency.value} "
              f"trên thẻ {self._mask_card()} — THÀNH CÔNG")
        return Transaction(
            transaction_id=tx_id,
            order_id=order.order_id,
            amount=order.total,
            currency=order.currency,
            status=PaymentStatus.COMPLETED,
            gateway_reference=f"stripe_ch_{secrets.token_hex(8)}",
        )

    def refund(self, transaction: Transaction) -> Transaction:
        return Transaction(
            transaction_id=f"RF-{transaction.transaction_id}",
            order_id=transaction.order_id,
            amount=transaction.amount,
            currency=transaction.currency,
            status=PaymentStatus.REFUNDED,
        )

    def name(self) -> str:
        return "Credit Card"


class PayPalPayment(PaymentMethod):
    """Thanh toán qua PayPal — OAuth 2.0 + REST API."""

    def __init__(self, oauth_token: str, email: str) -> None:
        self.oauth_token = oauth_token
        self.email = email

    def validate(self) -> bool:
        # Kiểm tra token với PayPal API
        if not self.oauth_token.startswith("PAYPAL-"):
            return False
        if "@" not in self.email:
            return False
        return True

    def charge(self, order: Order) -> Transaction:
        tx_id = f"PP-{order.order_id}-{secrets.token_hex(4).upper()}"
        fee = order.total * Decimal("0.029") + Decimal("0.30")  # PayPal fee: 2.9% + $0.30
        print(f"[{self.name()}] {order.total} {order.currency.value} "
              f"từ {self.email} — Phí: {fee:.2f} {order.currency.value}")
        return Transaction(
            transaction_id=tx_id,
            order_id=order.order_id,
            amount=order.total - fee,
            currency=order.currency,
            status=PaymentStatus.COMPLETED,
            gateway_reference=f"paypal_capt_{secrets.token_hex(8)}",
        )

    def refund(self, transaction: Transaction) -> Transaction:
        # PayPal refund mất phí 2.5% nếu refund sau 180 ngày
        return Transaction(
            transaction_id=f"RF-{transaction.transaction_id}",
            order_id=transaction.order_id,
            amount=transaction.amount * Decimal("0.975"),
            currency=transaction.currency,
            status=PaymentStatus.REFUNDED,
        )

    def name(self) -> str:
        return "PayPal"


class VNPayPayment(PaymentMethod):
    """Thanh toán VNPay — QR Code + checksum HMAC."""

    def __init__(self, qr_data: str, bank_code: str = "") -> None:
        self.qr_data = qr_data
        self.bank_code = bank_code

    def validate(self) -> bool:
        if not self.qr_data:
            return False
        # Kiểm tra checksum QR
        return len(self.qr_data) >= 10

    def charge(self, order: Order) -> Transaction:
        tx_id = f"VN-{order.order_id}-{secrets.token_hex(4).upper()}"
        print(f"[{self.name()}] {order.total} {order.currency.value} "
              f"mã QR: {self.qr_data[:8]}... — THÀNH CÔNG")
        return Transaction(
            transaction_id=tx_id,
            order_id=order.order_id,
            amount=order.total,
            currency=order.currency,
            status=PaymentStatus.COMPLETED,
            gateway_reference=f"vnpay_{secrets.token_hex(8)}",
        )

    def refund(self, transaction: Transaction) -> Transaction:
        return Transaction(
            transaction_id=f"RF-{transaction.transaction_id}",
            order_id=transaction.order_id,
            amount=transaction.amount,
            currency=transaction.currency,
            status=PaymentStatus.REFUNDED,
        )

    def name(self) -> str:
        return "VNPay"


# ============== CREATOR: FACTORY ==============

class PaymentFactory(ABC):
    """Abstract Creator — interface cho factory method."""

    @abstractmethod
    def create_payment(self, **kwargs) -> PaymentMethod:
        """Factory Method — tạo PaymentMethod từ tham số đầu vào."""
        ...

    def process_payment(self, order: Order, **kwargs) -> Transaction:
        """Template method — quy trình xử lý thanh toán chung."""
        payment = self.create_payment(**kwargs)

        if not payment.validate():
            return Transaction(
                transaction_id="",
                order_id=order.order_id,
                amount=order.total,
                currency=order.currency,
                status=PaymentStatus.FAILED,
                error_message=f"Thông tin {payment.name()} không hợp lệ",
            )

        return payment.charge(order)


# ============== CONCRETE CREATORS ==============

class CreditCardFactory(PaymentFactory):
    def create_payment(self, **kwargs) -> PaymentMethod:
        card_info = CreditCardInfo(
            card_number=kwargs["card_number"],
            expiry_month=kwargs["expiry_month"],
            expiry_year=kwargs["expiry_year"],
            cvv=kwargs["cvv"],
            cardholder_name=kwargs.get("cardholder_name", ""),
        )
        return CreditCardPayment(card_info)


class PayPalFactory(PaymentFactory):
    def create_payment(self, **kwargs) -> PaymentMethod:
        return PayPalPayment(
            oauth_token=kwargs["token"],
            email=kwargs["email"],
        )


class VNPayFactory(PaymentFactory):
    def create_payment(self, **kwargs) -> PaymentMethod:
        return VNPayPayment(
            qr_data=kwargs["qr_data"],
            bank_code=kwargs.get("bank_code", ""),
        )


# ============== CLIENT CODE ==============

class PaymentService:
    """Client — chỉ làm việc với abstraction, không biết class cụ thể."""

    def __init__(self) -> None:
        self._factories: dict[str, PaymentFactory] = {
            "credit_card": CreditCardFactory(),
            "paypal": PayPalFactory(),
            "vnpay": VNPayFactory(),
        }

    def register_factory(self, name: str, factory: PaymentFactory) -> None:
        """Đăng ký factory method mới — mở rộng mà không sửa code cũ."""
        self._factories[name] = factory

    def pay(self, order: Order, method: str, **kwargs) -> Transaction:
        factory = self._factories.get(method)
        if factory is None:
            return Transaction(
                transaction_id="",
                order_id=order.order_id,
                amount=order.total,
                currency=order.currency,
                status=PaymentStatus.FAILED,
                error_message=f"Phương thức '{method}' không được hỗ trợ",
            )
        return factory.process_payment(order, **kwargs)


# ========== SỬ DỤNG THỰC TẾ ==========

if __name__ == "__main__":
    service = PaymentService()

    order = Order(
        order_id="ORD-2024-0422",
        total=Decimal("1250000"),
        currency=Currency.VND,
        customer_email="nguyen.van.a@example.com",
        description="MacBook Air M3 15-inch",
    )

    # Thanh toán bằng Credit Card
    tx1 = service.pay(
        order,
        "credit_card",
        card_number="4111111111111111",
        expiry_month=12,
        expiry_year=2026,
        cvv="123",
        cardholder_name="Nguyen Van A",
    )
    print(f"  -> {tx1.transaction_id} | Status: {tx1.status.name}")

    # Thanh toán bằng PayPal
    tx2 = service.pay(
        order,
        "paypal",
        token="PAYPAL-abc123def456",
        email="nguyen.van.a@example.com",
    )
    print(f"  -> {tx2.transaction_id} | Status: {tx2.status.name} | Amount: {tx2.amount}")

    # Thanh toán bằng VNPay
    tx3 = service.pay(
        order,
        "vnpay",
        qr_data="VNPQR20240422120000",
        bank_code="NCB",
    )
    print(f"  -> {tx3.transaction_id} | Status: {tx3.status.name}")

    # Mở rộng: thêm Bank Transfer mà không sửa PaymentService
    from dataclasses import dataclass

    @dataclass
    class BankInfo:
        account_number: str
        account_name: str
        bank_name: str

    class BankTransferPayment(PaymentMethod):
        def __init__(self, bank_info: BankInfo) -> None:
            self.bank_info = bank_info

        def validate(self) -> bool:
            return len(self.bank_info.account_number) >= 8

        def charge(self, order: Order) -> Transaction:
            tx_id = f"BT-{order.order_id}-{secrets.token_hex(4).upper()}"
            print(f"[{self.name()}] {order.total} {order.currency.value} "
                  f"vào TK {self.bank_info.account_number}")
            return Transaction(
                transaction_id=tx_id, order_id=order.order_id,
                amount=order.total, currency=order.currency,
                status=PaymentStatus.PROCESSING,  # Bank transfer cần xác nhận thủ công
            )

        def refund(self, transaction: Transaction) -> Transaction:
            return Transaction(
                transaction_id=f"RF-{transaction.transaction_id}",
                order_id=transaction.order_id,
                amount=transaction.amount,
                currency=transaction.currency,
                status=PaymentStatus.PENDING,
            )

        def name(self) -> str:
            return "Bank Transfer"

    class BankTransferFactory(PaymentFactory):
        def create_payment(self, **kwargs) -> PaymentMethod:
            bank_info = BankInfo(
                account_number=kwargs["account_number"],
                account_name=kwargs["account_name"],
                bank_name=kwargs["bank_name"],
            )
            return BankTransferPayment(bank_info)

    # Đăng ký factory mới — không sửa một dòng code cũ nào
    service.register_factory("bank_transfer", BankTransferFactory())

    tx4 = service.pay(
        order,
        "bank_transfer",
        account_number="1234567890",
        account_name="Nguyen Van A",
        bank_name="Vietcombank",
    )
    print(f"  -> {tx4.transaction_id} | Status: {tx4.status.name}")
```

## Sơ đồ UML

```
┌──────────────────────────────────────────────────────────────┐
│                     «interface»                              │
│                     PaymentFactory                           │
├──────────────────────────────────────────────────────────────┤
│ + create_payment(**kwargs): PaymentMethod      «factory»     │
│ + process_payment(order, **kwargs): Transaction «template»   │
└──────────────────────────────────────────────────────────────┘
            ▲                              ▲
            │                              │
┌───────────┴──────────────┐    ┌──────────┴──────────────────┐
│    CreditCardFactory     │    │     PayPalFactory            │
├──────────────────────────┤    ├─────────────────────────────┤
│ + create_payment()       │    │ + create_payment()          │
│   -> CreditCardPayment   │    │   -> PayPalPayment          │
└──────────────────────────┘    └─────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  «interface» PaymentMethod                   │
├──────────────────────────────────────────────────────────────┤
│ + validate(): bool                                           │
│ + charge(order: Order) -> Transaction                        │
│ + refund(transaction: Transaction) -> Transaction            │
│ + name(): str                                                │
└──────────────────────────────────────────────────────────────┘
            ▲                    ▲                    ▲
            │                    │                    │
┌───────────┴──────┐  ┌─────────┴────────┐  ┌───────┴──────────┐
│CreditCardPayment │  │ PayPalPayment    │  │  VNPayPayment    │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ - card_info      │  │ - oauth_token    │  │ - qr_data        │
│ + validate()     │  │ - email          │  │ - bank_code      │
│ + charge()       │  │ + validate()     │  │ + validate()     │
│ + refund()       │  │ + charge()       │  │ + charge()       │
└──────────────────┘  └──────────────────┘  └──────────────────┘

┌─────────────────────────────────────┐
│        PaymentService (Client)      │
├─────────────────────────────────────┤
│ - factories: dict[str, PaymentFactory] │
│ + register_factory(name, factory)   │
│ + pay(order, method, **kwargs)      │
└─────────────────────────────────────┘
```

## So sánh với Pattern liên quan

| Pattern | Điểm giống | Điểm khác biệt chính |
|---------|-----------|---------------------|
| **Abstract Factory** | Đều tạo object thông qua interface | Factory Method tạo *một* object, Abstract Factory tạo *một họ* object liên quan. Factory Method dùng inheritance, Abstract Factory dùng composition. |
| **Simple Factory** | Đều tập trung logic khởi tạo | Simple Factory không phải GoF pattern. Nó dùng một method tĩnh duy nhất (thường có `if/elif`) — không thể mở rộng bằng subclass. Factory Method dùng polymorphism và inheritance để mở rộng. |
| **Builder** | Đều tạo object phức tạp | Factory Method tập trung vào việc *chọn class nào để instantiate*. Builder tập trung vào *cách xây dựng object từng bước*. Factory Method thường trả về product ngay lập tức, Builder có thể trả về product sau nhiều bước. |

**Khi nào chọn Factory Method thay vì Simple Factory?**
- Khi bạn cần mở rộng (thêm product mới) mà không sửa code cũ.
- Khi việc chọn product phụ thuộc vào context (platform, config, user role) — mỗi context có một Concrete Creator riêng.
- Khi bạn sử dụng framework và muốn cho phép người dùng override cách tạo object.

**Khi nào chọn Abstract Factory thay vì Factory Method?**
- Khi system cần tạo *nhiều loại product* khác nhau (chair + table + sofa) và các product này phải tương thích với nhau.
- Khi bạn muốn đảm bảo các product trong cùng một "family" được sử dụng cùng nhau.

## Ứng dụng thực tế

### 1. Django Forms — `form_class` attribute

Django dùng Factory Method pattern trong class-based views. View gọi `get_form()` — factory method — để tạo form instance:

```python
from django.views.generic.edit import FormView
from django import forms

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()
    message = forms.CharField(widget=forms.Textarea)

class ContactView(FormView):
    template_name = "contact.html"
    form_class = ContactForm  # Factory Method — có thể override ở subclass
    success_url = "/thanks/"

    def form_valid(self, form):
        # form đã được tạo bởi factory method in the parent class
        form.send_email()
        return super().form_valid(form)


# Mở rộng: không cần sửa ContactView
class PremiumContactForm(ContactForm):
    priority = forms.ChoiceField(choices=[("low", "Low"), ("high", "High")])

class PremiumContactView(ContactView):
    form_class = PremiumContactForm  # Override Factory Method
```

### 2. SQLAlchemy — Engine creation

SQLAlchemy tách creation qua `create_engine()` — factory method cho phép chọn database backend:

```python
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Factory Method pattern — cùng interface, backend khác nhau
engine: Engine = create_engine("postgresql://user:pass@localhost/db")
engine: Engine = create_engine("mysql+pymysql://user:pass@localhost/db")
engine: Engine = create_engine("sqlite:///local.db")
# Mỗi câu lệnh tạo ra Engine implementation khác nhau (PGDialect, MySQLDialect, SQLiteDialect)
```

### 3. Pytest — Fixture Factory

Pytest fixture có thể dùng Factory Method pattern để tạo dữ liệu test linh hoạt:

```python
import pytest
from typing import Callable

@pytest.fixture
def make_order() -> Callable:
    """Factory Method fixture — tạo Order với các tham số khác nhau."""
    def _make_order(total: float = 100.0, currency: str = "USD") -> Order:
        return Order(
            order_id=f"ORD-{secrets.token_hex(4).upper()}",
            total=Decimal(str(total)),
            currency=Currency(currency),
            customer_email="test@example.com",
        )
    return _make_order

# Sử dụng fixture factory
def test_payment_processing(make_order):
    small_order = make_order(total=50.0)
    large_order = make_order(total=50000.0)
    # ...
```

### 4. AWS SDK — `boto3.client()` factory

AWS SDK dùng factory để tạo client cho các service khác nhau:

```python
import boto3

# Factory Method pattern
s3_client = boto3.client("s3", region_name="ap-southeast-1")
dynamodb_client = boto3.client("dynamodb", region_name="us-west-2")
sqs_client = boto3.client("sqs", region_name="eu-west-1")

# Cùng interface, implementation khác nhau
s3_client.list_buckets()
dynamodb_client.list_tables()
sqs_client.list_queues()
```

## Kiểm thử

Factory Method giúp việc kiểm thử trở nên dễ dàng hơn nhờ khả năng mock factory:

```python
import pytest
from unittest.mock import MagicMock, patch
from decimal import Decimal
import secrets


@pytest.fixture
def sample_order() -> Order:
    return Order(
        order_id=f"ORD-{secrets.token_hex(4).upper()}",
        total=Decimal("500000"),
        currency=Currency.VND,
        customer_email="test@example.com",
    )


class TestCreditCardPayment:
    """Kiểm thử Concrete Product."""

    def test_valid_card(self):
        payment = CreditCardPayment(CreditCardInfo(
            card_number="4111111111111111",
            expiry_month=12,
            expiry_year=2026,
            cvv="123",
            cardholder_name="Test User",
        ))
        assert payment.validate() is True

    def test_invalid_card_short_number(self):
        payment = CreditCardPayment(CreditCardInfo(
            card_number="1234", expiry_month=12, expiry_year=2026,
            cvv="123", cardholder_name="Test",
        ))
        assert payment.validate() is False

    def test_expired_card(self):
        payment = CreditCardPayment(CreditCardInfo(
            card_number="4111111111111111", expiry_month=12,
            expiry_year=2020, cvv="123", cardholder_name="Test",
        ))
        assert payment.validate() is False

    def test_charge_creates_transaction(self, sample_order):
        payment = CreditCardPayment(CreditCardInfo(
            card_number="4111111111111111", expiry_month=12,
            expiry_year=2026, cvv="123", cardholder_name="Test",
        ))
        tx = payment.charge(sample_order)
        assert tx.status == PaymentStatus.COMPLETED
        assert tx.transaction_id.startswith("CC-")
        assert tx.amount == sample_order.total


class TestPaymentFactory:
    """Kiểm thử Factory Method — mock product."""

    def test_factory_creates_correct_product_type(self):
        factory = CreditCardFactory()
        payment = factory.create_payment(
            card_number="4111111111111111",
            expiry_month=12,
            expiry_year=2026,
            cvv="123",
        )
        assert isinstance(payment, CreditCardPayment)

    def test_factory_invalid_params_raises_error(self):
        factory = CreditCardFactory()
        with pytest.raises(KeyError):
            factory.create_payment(wrong_param="123")

    def test_process_payment_invalid_returns_failed_transaction(self, sample_order):
        # Dùng mock để giả lập validate() trả về False
        mock_payment = MagicMock(spec=PaymentMethod)
        mock_payment.validate.return_value = False
        mock_payment.name.return_value = "Mock"

        factory = CreditCardFactory()
        with patch.object(factory, "create_payment", return_value=mock_payment):
            tx = factory.process_payment(sample_order, card_number="bad")
            assert tx.status == PaymentStatus.FAILED
            assert "không hợp lệ" in tx.error_message


class TestPaymentService:
    """Kiểm thử Client — mock factory."""

    def test_pay_with_unknown_method(self, sample_order):
        service = PaymentService()
        tx = service.pay(sample_order, "unknown_method")
        assert tx.status == PaymentStatus.FAILED

    def test_pay_calls_factory_process_payment(self, sample_order):
        mock_factory = MagicMock(spec=PaymentFactory)
        mock_tx = Transaction(
            transaction_id="TEST-001", order_id=sample_order.order_id,
            amount=sample_order.total, currency=sample_order.currency,
            status=PaymentStatus.COMPLETED,
        )
        mock_factory.process_payment.return_value = mock_tx

        service = PaymentService()
        service.register_factory("test", mock_factory)
        tx = service.pay(sample_order, "test")

        assert tx == mock_tx
        mock_factory.process_payment.assert_called_once_with(sample_order)

    def test_open_closed_principle(self):
        """Thêm factory mới không ảnh hưởng factory cũ."""
        service = PaymentService()
        original_count = len(service._factories)

        mock_factory = MagicMock(spec=PaymentFactory)
        service.register_factory("new_method", mock_factory)

        assert len(service._factories) == original_count + 1
        # Factory cũ vẫn hoạt động
        assert "credit_card" in service._factories
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| **Open/Closed Principle**: Thêm product mới không cần sửa code cũ | **Complexity tăng**: Cần tạo nhiều class hơn (Product interface, Creator interface + implementations) |
| **Single Responsibility**: Tách khởi tạo khỏi business logic | **Parallel hierarchy**: Mỗi Product thường cần một Creator tương ứng — nhân đôi số lượng class |
| **Loose coupling**: Client chỉ phụ thuộc vào interface, không phải class cụ thể | **Khó đơn giản hóa**: Nếu chỉ có một product, Factory Method là over-engineering |
| **Dễ kiểm thử**: Có thể mock Product và Creator độc lập | **Constructor tham số**: Không dễ dàng truyền tham số động qua interface thống nhất |
| **Reusability**: Factory Method có thể được tái sử dụng ở nhiều nơi | **Học curve**: Developer mới có thể thấy khó hiểu với indirect creation |
| **Consistency**: Mọi product được tạo theo cùng một quy trình | **Static methods không mở rộng được**: Nếu dùng @staticmethod làm factory |
| **Lazy initialization**: Có thể kết hợp với lazy loading | **Phân tán logic**: Đôi khi logic khởi tạo phức tạp nằm rải rác ở nhiều factory |

---

## Kết luận

Factory Method là một trong những pattern quan trọng nhất của GoF — tôi dám nói nó là nền tảng cho hầu hết các framework hiện đại. **Golden rule**: Hãy dùng Factory Method khi bạn thấy mình đang dùng `if/elif` để chọn class cụ thể dựa trên một điều kiện nào đó, và điều kiện này có thể thay đổi hoặc mở rộng trong tương lai.

Như Warren Buffett từng nói: *"Risk comes from not knowing what you're doing."* — rủi ro đến từ việc không biết mình đang làm gì. Factory Method giúp bạn biết chính xác ai chịu trách nhiệm tạo object nào.

Pattern này đặc biệt hữu ích khi:
1. Bạn xây dựng một thư viện/framework và muốn cho phép người dùng mở rộng.
2. Bạn muốn test một class độc lập với các dependency phức tạp của nó.
3. Bạn cần kiểm soát việc tái sử dụng object (pooling, caching).
4. Bạn muốn chuẩn hóa cách tạo object trong toàn bộ codebase.

Hãy nhớ: Factory Method không phải là việc *thay thế `new` bằng factory*, mà là việc *cho phép subclass quyết định class nào được tạo*. Nếu factory của bạn không có subclass nào override nó, thì đó chỉ là **Simple Factory** — không sai, nhưng không tận dụng được sức mạnh của pattern.

---

*Trân trọng!*
