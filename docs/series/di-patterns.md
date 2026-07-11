---
id: di-patterns
title: 3 hình thức Dependency Injection
sidebar_label: 💉 Hình thức DI
sidebar_position: 32
---

# 3 hình thức Dependency Injection

> *"There are three main styles of dependency injection: Constructor Injection, Setter Injection, and Interface Injection. The question is which to choose, and when."* — **Martin Fowler, "Inversion of Control Containers and the Dependency Injection Pattern", 2004**

Martin Fowler đã phân loại DI thành 3 hình thức chính trong bài viết kinh điển năm 2004: Constructor Injection, Setter Injection, và Interface Injection (thường gọi là Method Injection trong thực tế). Mỗi hình thức có ưu nhược điểm riêng và phù hợp với những tình huống khác nhau. Hiểu rõ sự khác biệt và cách chọn lựa giữa chúng là kỹ năng quan trọng để áp dụng DI hiệu quả. Trong thực tế, bạn sẽ thường xuyên sử dụng cả ba — đôi khi trong cùng một class — tùy vào bản chất của từng dependency. Bài viết này sẽ đi sâu vào từng hình thức với ví dụ code thực tế, so sánh chi tiết, và hướng dẫn chọn lựa.

## 1. Constructor Injection

Constructor Injection là hình thức inject dependency qua tham số của constructor (`__init__` trong Python). Dependency được set **một lần duy nhất** khi object được tạo và **bất biến** (immutable) trong suốt vòng đời của object.

### Đặc điểm

- Dependency được khai báo trong signature của constructor — thể hiện rõ ràng class cần gì để hoạt động.
- Dependency là bắt buộc — object không thể hoạt động nếu thiếu.
- Dependency không thể thay đổi sau khi object được tạo (immutable).
- Không có state "invalid" — object luôn sẵn sàng hoạt động ngay sau khi khởi tạo.

### Ví dụ chi tiết: Hệ thống xử lý ảnh

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol
from pathlib import Path


class ImageFormat(Enum):
    JPEG = 'jpeg'
    PNG = 'png'
    WEBP = 'webp'


@dataclass(frozen=True)
class Image:
    data: bytes
    format: ImageFormat
    width: int
    height: int
    metadata: dict[str, str] = field(default_factory=dict)


# ─── Abstractions ───

class StorageBackend(Protocol):
    """Lưu trữ ảnh — có thể là local disk, S3, GCS, ..."""

    def save(self, path: str, data: bytes) -> None: ...

    def load(self, path: str) -> bytes: ...


class ImageProcessor(Protocol):
    """Xử lý ảnh — resize, compress, filter, ..."""

    def process(self, image: Image, **options: object) -> Image: ...


class ImageValidator(Protocol):
    """Kiểm tra ảnh trước khi xử lý."""

    def validate(self, image: Image) -> None: ...


# ─── Concrete Implementations ───

class LocalDiskStorage:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    def save(self, path: str, data: bytes) -> None:
        full_path = self._base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        print(f"💾 Saved to {full_path}")

    def load(self, path: str) -> bytes:
        return (self._base_path / path).read_bytes()


class S3Storage:
    def __init__(self, bucket: str, region: str, access_key: str, secret_key: str) -> None:
        self._bucket = bucket
        self._region = region
        self._access_key = access_key
        self._secret_key = secret_key

    def save(self, path: str, data: bytes) -> None:
        # import boto3
        print(f"☁️  Uploaded s3://{self._bucket}/{path}")
        pass

    def load(self, path: str) -> bytes:
        print(f"☁️  Downloaded s3://{self._bucket}/{path}")
        return b''


class ResizeProcessor:
    def process(self, image: Image, **options: object) -> Image:
        width = options.get('width', image.width)
        height = options.get('height', image.height)
        # Giả lập resize
        print(f"🖼️  Resized {image.width}x{image.height} → {width}x{height}")
        return Image(
            data=image.data,
            format=image.format,
            width=int(width),
            height=int(height),
            metadata={'resized': f'{width}x{height}'},
        )


class SizeValidator:
    MAX_SIZE = 10 * 1024 * 1024  # 10MB

    def validate(self, image: Image) -> None:
        if len(image.data) > self.MAX_SIZE:
            raise ValueError(f"Image too large: {len(image.data)} bytes > {self.MAX_SIZE}")


# ─── Service using Constructor Injection ───

class ImageService:
    """
    Constructor Injection: tất cả dependency được inject qua __init__.
    Không có setter, không có optional dependency — mọi thứ rõ ràng.
    """

    def __init__(
        self,
        storage: StorageBackend,
        processor: ImageProcessor,
        validator: ImageValidator,
        max_dimension: int = 2048,  # default value — vẫn injection nhưng có default
    ) -> None:
        self._storage = storage
        self._processor = processor
        self._validator = validator
        self._max_dimension = max_dimension

    def upload_and_process(self, image: Image, user_id: str) -> dict[str, object]:
        self._validator.validate(image)

        processed = self._processor.process(
            image, width=self._max_dimension, height=self._max_dimension
        )

        path = f"users/{user_id}/{id(image)}.{image.format.value}"
        self._storage.save(path, processed.data)

        return {
            'path': path,
            'width': processed.width,
            'height': processed.height,
        }


# ─── Sử dụng ───
storage = LocalDiskStorage(Path("/data/images"))
processor = ResizeProcessor()
validator = SizeValidator()
service = ImageService(storage, processor, validator)

# Inject implementation khác — không sửa ImageService
cloud_service = ImageService(S3Storage("my-bucket", "us-east-1", "ak", "sk"), processor, validator)
```

### Đặc điểm của Constructor Injection

**Ưu điểm**:
- **Immutability**: Dependency không thay đổi — giảm bug do trạng thái không mong muốn.
- **Tường minh**: Constructor signature cho biết chính xác class cần gì.
- **Type-safe**: Python type hints + mypy/pyright có thể kiểm tra.
- **Không có invalid state**: Object luôn có đủ dependency ngay khi khởi tạo.
- **Dễ test**: Mock dễ dàng, inject trực tiếp.

**Nhược điểm**:
- Constructor dài nếu có nhiều dependency (có thể là dấu hiệu vi phạm SRP).
- Không phù hợp cho dependency optional (phải dùng `None` default hoặc sentinel).
- Với Python, không có compile-time check — lỗi thiếu dependency chỉ phát hiện runtime.

## 2. Setter Injection

Setter Injection inject dependency qua các setter method, thường là sau khi object đã được khởi tạo với constructor không tham số. Hình thức này cho phép thay đổi dependency trong suốt vòng đời của object.

### Ví dụ: API Client với optional caching

```python
from __future__ import annotations
from typing import Protocol, Any
from dataclasses import dataclass
import json
import time


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int) -> None: ...
    def invalidate(self, key: str) -> None: ...


class RateLimiter(Protocol):
    def acquire(self, key: str) -> bool: ...
    def release(self, key: str) -> None: ...


class ApiClient:
    """
    Setter Injection: cache và rate_limiter là optional.
    Client có thể hoạt động mà không cần chúng.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip('/')
        self._api_key = api_key
        self._cache: CacheBackend | None = None          # Optional
        self._rate_limiter: RateLimiter | None = None     # Optional
        self._timeout: float = 30.0
        self._retry_count: int = 3

    # ─── Setter Injection ───

    def set_cache(self, cache: CacheBackend) -> None:
        """Optional dependency — client vẫn hoạt động nếu không có cache."""
        self._cache = cache

    def set_rate_limiter(self, limiter: RateLimiter) -> None:
        """Optional dependency — chỉ dùng khi cần rate limiting."""
        self._rate_limiter = limiter

    def set_timeout(self, timeout: float) -> None:
        """Có thể thay đổi timeout runtime."""
        self._timeout = timeout

    # ─── Core method ───

    def get(self, endpoint: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        cache_key = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"

        # Check cache trước
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached  # type: ignore

        # Rate limiting
        if self._rate_limiter is not None and not self._rate_limiter.acquire(endpoint):
            raise RuntimeError(f"Rate limit exceeded for {endpoint}")

        # Thực hiện request
        import urllib.request
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        if params:
            query = '&'.join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"

        req = urllib.request.Request(url, headers={'Authorization': f'Bearer {self._api_key}'})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())

        # Lưu cache
        if self._cache is not None:
            self._cache.set(cache_key, data, ttl=300)

        return data


class RedisCache:
    def __init__(self, host: str = 'localhost', port: int = 6379) -> None:
        self._host = host
        self._port = port

    def get(self, key: str) -> Any | None:
        print(f"🔍 Redis GET {key}")
        return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        print(f"💾 Redis SET {key} (TTL={ttl}s)")
        pass

    def invalidate(self, key: str) -> None:
        print(f"🗑️ Redis DEL {key}")


# ─── Sử dụng ───
client = ApiClient("https://api.example.com", "sk-xxxx")

# Cache là optional — chỉ set nếu cần
if USE_CACHE:
    client.set_cache(RedisCache())

# Rate limiter là optional
if USE_RATE_LIMIT:
    client.set_rate_limiter(MyRateLimiter())

data = client.get("/users", {"page": "1"})
```

### Đặc điểm của Setter Injection

**Ưu điểm**:
- **Optional dependencies**: Phù hợp cho dependency có thể không cần (cache, logging, metrics).
- **Runtime reconfiguration**: Có thể thay đổi dependency khi đang chạy (ví dụ: hot-swap cache backend).
- **Constructor gọn**: Constructor chỉ chứa required params.
- **Dễ inherit**: Subclass có thể override setter để inject implementation khác.

**Nhược điểm**:
- **Không immutable**: Dependency có thể bị thay đổi bất kỳ lúc nào — khó debug.
- **Invalid state**: Object có thể tồn tại mà không có dependency cần thiết — lỗi runtime.
- **Không rõ ràng**: Không thể biết object cần gì từ constructor signature — phải đọc documentation.
- **Thread-safety**: Setter injection không thread-safe — cần synchronization nếu dùng trong concurrent context.
- **Dễ quên set**: Developer có thể tạo object mà quên set dependency — lỗi chỉ xuất hiện ở runtime.

## 3. Method Injection (Parameter Injection)

Method Injection inject dependency trực tiếp qua tham số của method. Dependency không được lưu trữ — chỉ dùng trong phạm vi method đó.

### Ví dụ: Order processing with temporary dependencies

```python
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True)
class Order:
    order_id: str
    items: list[tuple[str, int, Decimal]]  # (product_id, qty, price)
    customer_email: str


class DiscountCalculator(Protocol):
    def calculate(self, order: Order, coupon_code: str | None) -> Decimal: ...


class ReceiptRenderer(Protocol):
    def render(self, order: Order, total: Decimal, discount: Decimal) -> str: ...


class OrderService:
    def __init__(self, discount: DiscountCalculator) -> None:
        # Constructor Injection cho required dependencies
        self._discount = discount

    def process_order(
        self,
        order: Order,
        payment_gateway: PaymentGateway,  # Method Injection — chỉ cần khi process
        renderer: ReceiptRenderer | None = None,  # Method Injection optional
    ) -> dict[str, object]:
        """
        Method Injection: payment_gateway và renderer được inject qua tham số.
        Chúng chỉ cần trong method này, không cần lưu lại.
        """
        total = sum(qty * price for _, qty, price in order.items)
        discount = self._discount.calculate(order, None)
        final_total = total - discount

        # Payment gateway được inject — chỉ dùng trong method
        result = payment_gateway.charge(order.order_id, final_total)

        # Renderer optional — chỉ render nếu được cung cấp
        receipt_html = None
        if renderer is not None:
            receipt_html = renderer.render(order, final_total, discount)

        return {
            'order_id': order.order_id,
            'charged': result.success,
            'total': str(final_total),
            'receipt': receipt_html,
        }


class StripePaymentGateway:
    def charge(self, order_id: str, amount: Decimal) -> PaymentResult:
        print(f"💳 Charging {amount} VND for order {order_id} via Stripe")
        return PaymentResult(success=True, transaction_id=f"txn_{order_id}")
```

### Khi nào dùng Method Injection?

- **Temporary dependency**: Dependency chỉ cần trong một method, không cần lưu lại. Ví dụ: `send_email(email_service, to, subject)` — email service chỉ cần khi gửi.
- **Depends on method parameter**: Dependency thay đổi tùy theo tham số đầu vào. Ví dụ: `export(report, format='pdf', renderer=PDFRenderer())` — renderer thay đổi theo format.
- **Cross-cutting concerns (thỉnh thoảng)**: Nếu một method cụ thể cần logging/auditing, có thể inject logger vào method thay vì constructor.
- **Strategy pattern**: Method nhận strategy làm tham số — `sort(data, comparator=NaturalComparator())`.

## So sánh chi tiết

| Tiêu chí | Constructor Injection | Setter Injection | Method Injection |
|----------|---------------------|-----------------|-----------------|
| **Thời điểm inject** | Khi tạo object | Sau khi tạo object | Khi gọi method |
| **Dependency lưu trữ** | Có — field của class | Có — field của class | Không — chỉ dùng trong method |
| **Tính bắt buộc** | Bắt buộc | Có thể optional | Tùy method |
| **Immutability** | ✅ Immutable | ❌ Mutable | ✅ Immutable (trong method) |
| **Tính tường minh** | ✅ Rất rõ ràng | ⚠️ Phải đọc setter | ⚠️ Phải đọc method signature |
| **Testability** | ✅ Dễ nhất (mock qua constructor) | ✅ Dễ (setter cho mock) | ✅ Dễ (mock qua tham số) |
| **Thay đổi runtime** | ❌ Không | ✅ Có | ✅ Có |
| **Thread-safe** | ✅ Có | ❌ Không | ✅ Có |
| **Rủi ro quên inject** | Thấp — lỗi ngay khi tạo | Cao — lỗi runtime muộn | Thấp — lỗi khi gọi |
| **Constructor complexity** | Cao nếu nhiều dependency | Thấp | Thấp |
| **Phù hợp cho** | Required, immutable deps | Optional, replaceable deps | Temporary, method-scope deps |
| **Tần suất sử dụng** | ✅ 80% trường hợp | ⚠️ 15% trường hợp | ⚠️ 5% trường hợp |

## Khi nào dùng hình thức nào?

**Luôn ưu tiên Constructor Injection** vì những lý do:
- Dependency rõ ràng — ai đọc code cũng hiểu class cần gì.
- Object luôn valid — không thể tạo object thiếu dependency.
- Immutable — tránh cả đống bug do thay đổi state.
- Dễ test nhất — chỉ cần tạo object với mock.

**Dùng Setter Injection khi**:
- Dependency là optional (cache, logger, metrics).
- Cần thay đổi dependency runtime (hot-reload config, A/B testing).
- Có circular dependency (dù nên tránh).

**Dùng Method Injection khi**:
- Dependency chỉ cần trong method cụ thể.
- Dependency thay đổi theo từng lần gọi method (strategy pattern).
- Dependency là tạm thời, không cần lưu lại (transaction context, request context).

## Nguyên tắc vàng

> **"Khi nghi ngờ, hãy dùng Constructor Injection. Nếu bạn thực sự cần một hình thức khác, hãy chắc chắn rằng bạn đã cân nhắc đủ."**

Trong thực tế, một class có thể kết hợp cả ba hình thức:
- Constructor Injection cho required dependencies.
- Setter Injection cho optional dependencies.
- Method Injection cho temporary/context-specific dependencies.

Ví dụ:

```python
class ReportService:
    # Constructor — required deps
    def __init__(self, db: Database, storage: StorageBackend) -> None:
        self._db = db
        self._storage = storage

    # Setter — optional deps
    def set_logger(self, logger: Logger) -> None:
        self._logger = logger

    # Method — temporary deps
    def generate_report(
        self,
        report_type: str,
        renderer: ReportRenderer,  # Method injection
        format: str = 'pdf',
    ) -> str:
        data = self._db.query(...)
        if self._logger:
            self._logger.info(f"Generating {report_type} report")
        output = renderer.render(data, format)
        path = self._storage.save(...)
        return path
```

Kết luận: không có hình thức DI nào là "tốt nhất" tuyệt đối. Mỗi hình thức có chỗ đứng riêng. Constructor Injection là lựa chọn mặc định — hãy bắt đầu với nó, và chỉ dùng Setter/Method Injection khi có lý do chính đáng.
