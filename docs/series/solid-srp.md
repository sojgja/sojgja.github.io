---
id: solid-srp
title: S — Single Responsibility Principle
sidebar_label: S — Single Responsibility
sidebar_position: 26
---

# S — Single Responsibility Principle

> *"A class should have only one reason to change."* — **Robert C. Martin, *Agile Software Development, Principles, Patterns, and Practices*, 2002**

Single Responsibility Principle (SRP) là nguyên lý đầu tiên và cũng là nguyên lý dễ hiểu nhất trong SOLID, nhưng nghịch lý thay lại là một trong những nguyên lý khó áp dụng đúng nhất. Robert C. Martin định nghĩa "trách nhiệm" (responsibility) là "một lý do để thay đổi" (a reason to change). Nếu một class có nhiều hơn một lý do để bị sửa đổi — ví dụ, thay đổi logic nghiệp vụ, thay đổi format hiển thị, thay đổi cơ chế lưu trữ, thay đổi giao thức gửi thông báo — thì class đó đang làm quá nhiều việc. Hậu quả là mỗi lần sửa một trách nhiệm đều có nguy cơ làm hỏng những trách nhiệm khác, và việc test trở nên cực kỳ phức tạp vì phải mock quá nhiều dependency không liên quan.

## Bài toán chi tiết: Hệ thống xử lý đơn hàng thương mại điện tử

Công ty X phát triển một hệ thống thương mại điện tử với module xử lý đơn hàng. Đội ngũ kỹ thuật viết class `OrderProcessor` với chức năng: nhận đơn hàng, tính toán chiết khấu dựa trên loại khách hàng, tạo hóa đơn PDF, lưu đơn hàng vào PostgreSQL, gửi email xác nhận đến khách hàng, gửi thông báo đến kho hàng qua WebSocket, và cập nhật số lượng tồn kho. Ban đầu mọi thứ đều ổn — class này có vẻ tiện lợi vì tập trung mọi thao tác liên quan đến đơn hàng vào một chỗ. Sau 3 tháng, sản phẩm phát triển và có thêm yêu cầu: thay đổi template email, chuyển từ PostgreSQL sang MongoDB, thêm tính năng gửi SMS thay vì email cho đơn hàng COD, tích hợp thêm cổng thanh toán mới. Mỗi yêu cầu mới đều buộc developer phải mở file `OrderProcessor` và sửa nó. Sau 6 tháng, class này dài hơn 2000 dòng với hàng chục method, 15 dependency được inject, và trở thành nỗi ám ảnh — không ai dám refactor vì sợ ảnh hưởng đến toàn bộ hệ thống. Bug xuất hiện liên miên: một lần sửa template email vô tình làm hỏng logic ghi nhận tồn kho, một lần khác thay đổi cấu trúc database query gây lỗi tính chiết khấu.

Vấn đề càng trở nên trầm trọng hơn khi đội ngũ phát triển mở rộng. Năm developer cùng làm việc trên cùng một file, thường xuyên conflict khi merge. Testing cũng là cơn ác mộng — `OrderProcessor` cần kết nối database thật, email server thật, WebSocket server thật. Mỗi lần chạy test mất 15 phút, và kết quả không đáng tin cậy vì phụ thuộc vào môi trường. Code coverage luôn ở mức thấp vì viết test cho class quá khổ là bất khả thi. Đây là "Big Ball of Mud" — kiến trúc "cục bùn" mà các chuyên gia phần mềm cảnh báo. Và tất cả bắt nguồn từ việc vi phạm SRP ngay từ những dòng code đầu tiên.

## Phân tích vấn đề

Root cause của vấn đề không phải là class có nhiều dòng code, mà là class có **nhiều hơn một lý do để thay đổi**. Mỗi actor — bộ phận kinh doanh, bộ phận vận hành, bộ phận kế toán, bộ phận IT — đều có những yêu cầu thay đổi khác nhau tác động lên cùng một class. Khi một class phục vụ nhiều actor, thay đổi từ actor này sẽ ảnh hưởng đến actor khác. Cụ thể:

1. **Vi phạm Separation of Concerns**: Logic nghiệp vụ (tính toán, chiết khấu) bị trộn lẫn với infrastructure (database, email, WebSocket) và presentation (PDF, template).
2. **Coupling cao**: Class `OrderProcessor` phụ thuộc trực tiếp vào các implementation cụ thể — PostgreSQL driver, thư viện tạo PDF, SMTP client.
3. **Khả năng test kém**: Để test logic tính chiết khấu, bạn phải mock database, email service, WebSocket — mặc dù chúng không liên quan gì đến logic đó.
4. **Vi phạm DRY một cách tinh vi**: Các phần code phụ trách các trách nhiệm khác nhau không thể tái sử dụng ở nơi khác. Nếu một module khác cũng cần gửi email, nó không thể dùng code từ `OrderProcessor` vì code đó bị trộn lẫn với logic đơn hàng.
5. **Khó mở rộng**: Mỗi tính năng mới phải được nhồi nhét vào class đã quá tải. Thay vì plugin hóa, bạn phải sửa code hiện tại.

**Code smells điển hình** của vi phạm SRP bao gồm: `Manager`, `Processor`, `Util`, `Helper` trong tên class; constructor có hơn 4-5 tham số; class có các method không chia sẻ chung dữ liệu (không liên quan đến nhau); xuất hiện comment phân cách "// --- Database methods ---" và "// --- Email methods ---" trong cùng một file; một method duy nhất thực hiện nhiều bước không cùng mức abstraction (ví dụ: gọi API, parse JSON, ghi file, gửi notification).

## Giải pháp: Tách class theo từng trách nhiệm

Giải pháp là áp dụng SRP triệt để: **mỗi class chỉ làm một việc và làm việc đó thật tốt**. Chúng ta tách `OrderProcessor` thành các class riêng biệt, mỗi class có một lý do duy nhất để thay đổi:

1. **`Order`** — Domain model thuần túy, chỉ chứa dữ liệu và business logic cốt lõi (tính tổng, áp dụng chiết khấu).
2. **`OrderRepository`** — Chịu trách nhiệm persistence (lưu/truy xuất đơn hàng từ database). Thay đổi khi cơ chế lưu trữ thay đổi.
3. **`InvoiceGenerator`** — Tạo hóa đơn (PDF/HTML). Thay đổi khi format hóa đơn thay đổi.
4. **`EmailNotifier`** — Gửi email. Thay đổi khi email template hoặc email provider thay đổi.
5. **`InventoryService`** — Cập nhật tồn kho. Thay đổi khi logic kho hàng thay đổi.
6. **`OrderProcessor`** — Orchestrator (Facade pattern), phối hợp các service trên. Class này VẪN có lý do để thay đổi — khi quy trình xử lý đơn hàng thay đổi — nhưng đó là MỘT lý do duy nhất.

## Ví dụ code hoàn chỉnh

### VIOLATION — Vi phạm SRP

```python
# order_processor_violation.py
from __future__ import annotations
import json
import smtplib
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class Customer:
    id: str
    name: str
    email: str
    tier: str  # 'gold', 'silver', 'bronze'


@dataclass
class OrderItem:
    product_id: str
    name: str
    quantity: int
    unit_price: float


@dataclass
class Order:
    id: str
    customer: Customer
    items: list[OrderItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class OrderProcessor:
    """
    VIOLATION: Class này có 5+ lý do để thay đổi:
    1. Thay đổi logic chiết khấu
    2. Thay đổi format hóa đơn
    3. Thay đổi database
    4. Thay đổi email template/provider
    5. Thay đổi WebSocket/protocol
    """

    def __init__(self, db_connection_string: str, smtp_server: str,
                 smtp_port: int, ws_url: str) -> None:
        self.db_string = db_connection_string
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.ws_url = ws_url

    def process_order(self, order: Order) -> dict[str, Any]:
        # Bước 1: Tính toán
        discount = self._calculate_discount(order)
        total = sum(item.quantity * item.unit_price for item in order.items)
        final_total = total - discount

        # Bước 2: Tạo hóa đơn PDF (dùng template HTML đơn giản)
        invoice_html = f"""
        <html><body>
        <h1>HÓA ĐƠN #{order.id}</h1>
        <p>Khách hàng: {order.customer.name}</p>
        <table>
        """
        for item in order.items:
            invoice_html += f"<tr><td>{item.name}</td><td>{item.quantity}</td><td>{item.unit_price}</td></tr>"
        invoice_html += f"<tr><td colspan='2'>Tổng</td><td>{final_total}</td></tr>"
        invoice_html += "</table></body></html>"

        # Bước 3: Lưu vào database (PostgreSQL)
        import psycopg2  # type: ignore
        conn = psycopg2.connect(self.db_string)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO orders (id, customer_id, total, status) VALUES (%s, %s, %s, 'confirmed')",
            (order.id, order.customer.id, final_total),
        )
        conn.commit()
        cur.close()
        conn.close()

        # Bước 4: Gửi email
        import smtplib
        from email.mime.text import MIMEText  # type: ignore
        msg = MIMEText(f"Cảm ơn {order.customer.name}, đơn hàng #{order.id} đã được xác nhận!")
        msg['Subject'] = f'Xác nhận đơn hàng #{order.id}'
        msg['To'] = order.customer.email
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.send_message(msg)

        # Bước 5: Cập nhật tồn kho
        for item in order.items:
            self._update_inventory(item.product_id, -item.quantity)

        return {'order_id': order.id, 'total': final_total, 'status': 'confirmed'}

    def _calculate_discount(self, order: Order) -> float:
        if order.customer.tier == 'gold':
            return sum(item.quantity * item.unit_price for item in order.items) * 0.15
        elif order.customer.tier == 'silver':
            return sum(item.quantity * item.unit_price for item in order.items) * 0.10
        return 0.0

    def _update_inventory(self, product_id: str, quantity: int) -> None:
        # Giả lập gọi API inventory service
        print(f"📦 Cập nhật tồn kho: {product_id}, số lượng: {quantity}")


# Sử dụng
customer = Customer(id='C001', name='Nguyễn Văn A', email='a@example.com', tier='gold')
order = Order(id='ORD-001', customer=customer, items=[
    OrderItem(product_id='P001', name='Laptop', quantity=1, unit_price=15000000),
    OrderItem(product_id='P002', name='Chuột', quantity=2, unit_price=500000),
])
processor = OrderProcessor(
    db_connection_string='postgresql://localhost:5432/shop',
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    ws_url='ws://warehouse:8080',
)
result = processor.process_order(order)
print(f'Kết quả: {result}')
```

### REFACTORED — Tuân thủ SRP

```python
# ─── domain/order.py ───
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List


@dataclass(frozen=True)
class OrderItem:
    product_id: str
    name: str
    quantity: int
    unit_price: Decimal


@dataclass
class Order:
    order_id: str
    customer_id: str
    customer_email: str
    customer_name: str
    customer_tier: str
    items: List[OrderItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def subtotal(self) -> Decimal:
        return sum(
            Decimal(item.quantity) * item.unit_price
            for item in self.items
        )

    def apply_discount(self, rate: Decimal) -> Decimal:
        return self.subtotal() * rate


# ─── services/discount_service.py ───
from __future__ import annotations
from abc import ABC, abstractmethod
from decimal import Decimal


class DiscountPolicy(ABC):
    @abstractmethod
    def get_discount_rate(self, customer_tier: str) -> Decimal:
        ...


class DefaultDiscountPolicy(DiscountPolicy):
    _RATES: dict[str, Decimal] = {
        'gold': Decimal('0.15'),
        'silver': Decimal('0.10'),
        'bronze': Decimal('0.05'),
        'regular': Decimal('0.0'),
    }

    def get_discount_rate(self, customer_tier: str) -> Decimal:
        return self._RATES.get(customer_tier, Decimal('0.0'))


# ─── services/order_repository.py ───
from __future__ import annotations
from abc import ABC, abstractmethod


class OrderRepository(ABC):
    @abstractmethod
    def save(self, order: Order, total: Decimal) -> None:
        ...

    @abstractmethod
    def find_by_id(self, order_id: str) -> Order | None:
        ...


class PostgresOrderRepository(OrderRepository):
    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string

    def save(self, order: Order, total: Decimal) -> None:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(self._connection_string)
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO orders (id, customer_id, total, status) VALUES (%s, %s, %s, 'confirmed')",
                (order.order_id, order.customer_id, str(total)),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()

    def find_by_id(self, order_id: str) -> Order | None:
        ...


# ─── services/invoice_generator.py ───
from __future__ import annotations
from abc import ABC, abstractmethod
from decimal import Decimal


class InvoiceData:
    def __init__(self, order: Order, total: Decimal) -> None:
        self.order_id: str = order.order_id
        self.customer_name: str = order.customer_name
        self.items: List[OrderItem] = order.items
        self.total: Decimal = total


class InvoiceGenerator(ABC):
    @abstractmethod
    def generate(self, data: InvoiceData) -> str:
        ...


class HtmlInvoiceGenerator(InvoiceGenerator):
    def generate(self, data: InvoiceData) -> str:
        rows = ''.join(
            f"<tr><td>{item.name}</td><td>{item.quantity}</td>"
            f"<td>{item.unit_price:,} VND</td></tr>"
            for item in data.items
        )
        return f"""<html><body>
<h1>HÓA ĐƠN #{data.order_id}</h1>
<p>Khách hàng: {data.customer_name}</p>
<table border="1">{rows}
<tr><td colspan="2"><b>Tổng</b></td><td><b>{data.total:,} VND</b></td></tr>
</table>
<p>Cảm ơn quý khách!</p>
</body></html>"""


# ─── services/notifier.py ───
from __future__ import annotations
from abc import ABC, abstractmethod


class Notifier(ABC):
    @abstractmethod
    def send(self, recipient: str, subject: str, body: str) -> None:
        ...


class EmailNotifier(Notifier):
    def __init__(self, smtp_server: str, smtp_port: int) -> None:
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send(self, recipient: str, subject: str, body: str) -> None:
        import smtplib
        from email.mime.text import MIMEText  # type: ignore
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['To'] = recipient
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.send_message(msg)


class ConsoleNotifier(Notifier):
    def send(self, recipient: str, subject: str, body: str) -> None:
        print(f"[NOTIFICATION] To: {recipient} | Subject: {subject} | Body: {body[:50]}...")


# ─── services/inventory_service.py ───
from __future__ import annotations
from decimal import Decimal


class InventoryService:
    def adjust_stock(self, product_id: str, quantity: int) -> None:
        # Gọi REST API hoặc message queue để cập nhật
        print(f"📦 Inventory adjusted: {product_id}, delta={quantity}")


# ─── orchestration/order_processor.py ───
from __future__ import annotations
from decimal import Decimal


class OrderProcessor:
    """
    Chỉ có MỘT lý do để thay đổi: quy trình xử lý đơn hàng thay đổi.
    """

    def __init__(
        self,
        repository: OrderRepository,
        discount: DiscountPolicy,
        invoice: InvoiceGenerator,
        notifier: Notifier,
        inventory: InventoryService,
    ) -> None:
        self._repository = repository
        self._discount = discount
        self._invoice = invoice
        self._notifier = notifier
        self._inventory = inventory

    def process(self, order: Order) -> dict[str, str]:
        rate = self._discount.get_discount_rate(order.customer_tier)
        total: Decimal = order.subtotal() * (Decimal('1.0') - rate)

        self._repository.save(order, total)

        invoice_data = InvoiceData(order, total)
        invoice_html = self._invoice.generate(invoice_data)

        self._notifier.send(
            recipient=order.customer_email,
            subject=f'Xác nhận đơn hàng #{order.order_id}',
            body=f"Cảm ơn {order.customer_name}, đơn hàng #{order.order_id} đã được xác nhận!",
        )

        for item in order.items:
            self._inventory.adjust_stock(item.product_id, -item.quantity)

        return {'order_id': order.order_id, 'total': str(total), 'status': 'confirmed'}
```

## Dấu hiệu nhận biết vi phạm SRP

- **Tên class chung chung**: `OrderManager`, `DataProcessor`, `UtilHelper`, `SystemHandler` — đây là dấu hiệu kinh điển cho thấy class đang làm quá nhiều việc. Một class có tên rõ ràng thường chỉ gồm một noun và một verb.
- **Constructor có quá nhiều tham số**: Nếu constructor của bạn có 5+ tham số, rất có thể class đang phụ thuộc vào quá nhiều thứ và do đó có quá nhiều trách nhiệm. Dấu hiệu này cũng có thể chỉ ra vi phạm DIP, nhưng thường đi kèm với SRP.
- **Method có nhiều hơn một cấp độ abstraction**: Một method vừa gọi API HTTP, vừa tính toán business logic, vừa ghi log, vừa gửi email — đó là dấu hiệu rõ ràng. Một method tốt chỉ có một cấp độ abstraction duy nhất.
- **Comment phân cách trong class**: Khi bạn thấy comment kiểu `# ─── Database methods ───` và `# ─── Email methods ───` trong cùng một class, đó là lúc cần tách class.
- **Class có quá nhiều method public**: Trên 10-15 method public thường là dấu hiệu class đang làm quá nhiều việc.
- **Khó viết unit test**: Nếu bạn phải mock 5-6 dependency để test một method đơn giản, class đó chắc chắn vi phạm SRP.
- **Một thay đổi nhỏ buộc bạn sửa nhiều test**: Khi bạn thay đổi một behavior của class và phải sửa 20 test files, có thể class đó có quá nhiều trách nhiệm.

## Kiểm thử

```python
# test_order_processor.py
from __future__ import annotations
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch, call
import pytest  # type: ignore
from domain.order import Order, OrderItem
from services.discount_service import DefaultDiscountPolicy
from services.invoice_generator import HtmlInvoiceGenerator, InvoiceData
from services.notifier import ConsoleNotifier
from services.inventory_service import InventoryService
from orchestration.order_processor import OrderProcessor


@pytest.fixture
def sample_order() -> Order:
    return Order(
        order_id='ORD-001',
        customer_id='C001',
        customer_name='Nguyễn Văn A',
        customer_email='a@example.com',
        customer_tier='gold',
        items=[
            OrderItem(product_id='P001', name='Laptop', quantity=1, unit_price=Decimal('15000000')),
            OrderItem(product_id='P002', name='Mouse', quantity=2, unit_price=Decimal('500000')),
        ],
    )


def test_discount_policy_gold_tier() -> None:
    policy = DefaultDiscountPolicy()
    rate = policy.get_discount_rate('gold')
    assert rate == Decimal('0.15')


def test_discount_policy_unknown_tier() -> None:
    policy = DefaultDiscountPolicy()
    rate = policy.get_discount_rate('platinum')
    assert rate == Decimal('0.0')


def test_order_processor_full_flow(sample_order: Order) -> None:
    # Arrange: mock tất cả dependencies
    mock_repo = Mock(spec=OrderRepository)
    mock_discount = Mock(spec=DefaultDiscountPolicy)
    mock_invoice = Mock(spec=HtmlInvoiceGenerator)
    mock_notifier = Mock(spec=ConsoleNotifier)
    mock_inventory = Mock(spec=InventoryService)

    mock_discount.get_discount_rate.return_value = Decimal('0.15')
    mock_invoice.generate.return_value = '<html>Invoice</html>'

    processor = OrderProcessor(
        repository=mock_repo,
        discount=mock_discount,
        invoice=mock_invoice,
        notifier=mock_notifier,
        inventory=mock_inventory,
    )

    # Act
    result = processor.process(sample_order)

    # Assert
    assert result['status'] == 'confirmed'
    assert result['order_id'] == 'ORD-001'

    # Verify từng dependency được gọi đúng
    mock_discount.get_discount_rate.assert_called_once_with('gold')
    mock_repo.save.assert_called_once()
    mock_invoice.generate.assert_called_once()
    mock_notifier.send.assert_called_once()
    assert mock_notifier.send.call_args[1]['recipient'] == 'a@example.com'
    assert mock_inventory.adjust_stock.call_count == 2  # 2 items


def test_order_repository_persistence() -> None:
    # Integration test với mock database
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch('psycopg2.connect', return_value=mock_conn):
        repo = PostgresOrderRepository('postgresql://localhost/test')
        order = Order(
            order_id='ORD-002',
            customer_id='C002',
            customer_name='Test',
            customer_email='test@test.com',
            customer_tier='bronze',
        )
        repo.save(order, Decimal('100000'))

    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_conn.close.assert_called_once()


@pytest.mark.parametrize('tier,expected_rate', [
    ('gold', Decimal('0.15')),
    ('silver', Decimal('0.10')),
    ('bronze', Decimal('0.05')),
    ('regular', Decimal('0.0')),
    ('', Decimal('0.0')),
])
def test_discount_policy_parameterized(tier: str, expected_rate: Decimal) -> None:
    policy = DefaultDiscountPolicy()
    assert policy.get_discount_rate(tier) == expected_rate
```

## Ứng dụng thực tế

1. **FastAPI — Route Handlers và Service Layer**: FastAPI khuyến khích tách route handler (chỉ nhận request/trả response) khỏi business logic (service layer). Một file `routes/users.py` chỉ chứa endpoint definitions, trong khi `services/user_service.py` chứa toàn bộ logic nghiệp vụ. Repository pattern trong SQLAlchemy cũng tuân thủ SRP — mỗi repository chỉ làm việc với đúng một entity.

2. **Django — Fat Models vs Service Layer**: Django thường mắc bẫy "fat models" — models chứa cả business logic, validation, email sending, permission checking. Giải pháp là dùng Service Layer pattern: models chỉ chứa data mapping và business logic cốt lõi, trong khi services xử lý orchestration. Django REST Framework cũng khuyến khích tách Serializers (chuyển đổi dữ liệu) khỏi Views (xử lý request).

3. **Clean Architecture trong thực tế**: Một dự án e-commerce tại Việt Nam đã áp dụng SRP triệt để bằng cách chia codebase thành 4 layer: Domain (entities, value objects), Application (use cases, DTOs), Infrastructure (repositories, email, payment gateway interfaces), và Presentation (controllers, serializers). Mỗi layer chỉ thay đổi vì một lý do duy nhất, giúp giảm thời gian fix bug từ 3 ngày xuống còn 4 giờ.

4. **Notification System**: Một hệ thống notification tích hợp email, SMS, push notification, Slack — mỗi channel là một class riêng biệt tuân thủ SRP. Khi cần thêm channel mới (Telegram, Zalo), chỉ cần tạo class mới implements Notifier interface, không cần sửa bất kỳ code nào của các channel cũ.

## Liên hệ với Pattern

- **Facade Pattern**: `OrderProcessor` trong ví dụ refactored chính là một Facade — nó cung cấp interface đơn giản cho một subsystem phức tạp. Facade giúp duy trì SRP cho các class con bằng cách đóng vai trò orchestration.
- **Strategy Pattern**: `DiscountPolicy` là Strategy — cho phép thay đổi thuật toán tính chiết khấu mà không ảnh hưởng đến OrderProcessor.
- **Repository Pattern**: `OrderRepository` là Repository — abstract hóa persistence layer, giúp business logic không phụ thuộc vào cơ chế lưu trữ cụ thể.
- **Observer/Event Pattern**: Khi có nhiều hành động cần thực hiện sau khi xử lý đơn hàng (gửi email, cập nhật inventory, thông báo analytics), Event pattern kết hợp SRP giúp mỗi listener chỉ làm một việc.

## Ưu và nhược điểm

| Tiêu chí | Trước (vi phạm SRP) | Sau (tuân thủ SRP) |
|----------|---------------------|-------------------|
| **Số class** | 1 class "khủng long" | 6+ class nhỏ, mỗi class chuyên biệt |
| **Số dòng code/class** | 2000+ dòng, rất khó đọc | 50-200 dòng, dễ đọc và maintain |
| **Lý do thay đổi** | 5+ lý do (vi phạm SRP) | 1 lý do duy nhất mỗi class |
| **Khả năng test** | Rất khó (cần DB, SMTP thật) | Dễ dàng (mock từng dependency) |
| **Thời gian chạy test** | 15 phút (phụ thuộc hạ tầng) | 5-10 giây (unit test thuần túy) |
| **Rủi ro refactor** | Rất cao (sửa một chỗ hỏng cả hệ thống) | Thấp (mỗi class độc lập) |
| **Khả năng tái sử dụng** | Thấp (code bị trộn lẫn) | Cao (mỗi class có thể dùng độc lập) |
| **Số lượng file cần mở khi thêm tính năng** | 1 file (nhưng sửa rất nhiều) | 1-2 file mới, không sửa file cũ |
| **Dependency coupling** | Cao (phụ thuộc implementation cụ thể) | Thấp (phụ thuộc abstraction) |
| **Số lượng conflict khi merge** | Nhiều (nhiều dev cùng sửa một file) | Ít (mỗi dev sửa file riêng) |
| **Chi phí maintenance (6 tháng)** | Rất cao (technical debt tích lũy) | Thấp (code sạch, dễ bảo trì) |

## Kết luận

SRP không phải là "một class chỉ có một method" hay "một file chỉ có 100 dòng". Đó là việc đảm bảo mỗi class chỉ có một lý do duy nhất để thay đổi — tức là chỉ phục vụ một actor duy nhất trong hệ thống. Trong thực tế, áp dụng SRP đúng cách đòi hỏi sự tinh tế: không nên tách quá nhỏ (over-engineering) cũng không nên gộp quá lớn (big ball of mud). Một heuristic hữu ích: hãy tưởng tượng class của bạn là một file trong repository. Nếu file đó thường xuyên bị thay đổi vì những lý do khác nhau từ những bộ phận khác nhau của doanh nghiệp, hãy mạnh dạn tách nó ra. Hãy nhớ: "gộp code dễ, tách code mới khó" — luôn bắt đầu với thiết kế tuân thủ SRP ngay từ đầu, bởi vì refactor một monolith về sau còn tốn kém gấp nhiều lần so với thiết kế đúng từ những dòng code đầu tiên.
