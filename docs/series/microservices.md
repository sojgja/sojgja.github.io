---
id: microservices
title: Microservices Architecture
sidebar_label: 🔬 Microservices
sidebar_position: 37
---

# Microservices Architecture

> *"Microservices are small, autonomous services that work together."* — **Sam Newman**, *Building Microservices*

**Microservices Architecture** là một phương pháp kiến trúc phần mềm nơi một ứng dụng được cấu thành từ nhiều dịch vụ nhỏ, độc lập, mỗi dịch vụ chạy trong tiến trình riêng và giao tiếp qua các giao thức nhẹ (thường là HTTP/REST hoặc message queue). Mỗi microservice được xây dựng xoay quanh một **business capability** cụ thể, có thể được triển khai, mở rộng, và bảo trì độc lập với các service khác. Đây là một trong những kiến trúc có ảnh hưởng nhất trong thập kỷ qua — được áp dụng bởi Netflix, Amazon, Spotify, Uber, và hầu hết các công ty công nghệ hàng đầu thế giới.

---

## Bài toán

### Vấn đề: Khi monolith trở thành gánh nặng

Hãy tưởng tượng bạn đang làm việc cho **một nền tảng thương mại điện tử lớn** — giống như Shopee, Tiki, hay Lazada. Hệ thống bắt đầu như một monolith: một codebase duy nhất, deploy như một ứng dụng duy nhất. Khi công ty còn nhỏ (vài chục nghìn user, đội ngũ 5-10 developers), monolith hoạt động hoàn hảo. Nhưng khi công ty phát triển, monolith bắt đầu bộc lộ những vấn đề nghiêm trọng:

**1. Development bottleneck**: Một monolith với 2 triệu dòng code, 200 developers làm việc trên cùng một repository. Merge conflict là nỗi ám ảnh hàng ngày. Mỗi pull request mất 3-5 ngày để review vì phải hiểu toàn bộ hệ thống. Feature branch có thể sống sót hàng tuần trước khi được merge.

**2. Scaling inefficiency**: Module search (CPU-bound) và module payment (I/O-bound) có nhu cầu scaling hoàn toàn khác nhau. Với monolith, bạn phải scale toàn bộ ứng dụng — kể cả những phần không cần thiết. Bạn trả tiền cho server chạy module forgot-password trong khi module search đang quá tải.

**3. Technology lock-in**: Bạn muốn thử nghiệm một công nghệ mới? Muốn dùng Python cho AI recommendation service, Go cho real-time chat, Java cho payment processing? Với monolith, bạn chỉ có một lựa chọn. Công nghệ bạn chọn năm 2018 (PHP + MySQL + Memcached) vẫn là công nghệ bạn phải dùng năm 2025.

**4. Reliability cascade**: Bug trong module product listing (hiển thị sản phẩm) làm sập toàn bộ ứng dụng — kể cả module checkout và payment. Một lỗi nhỏ trong phần không quan trọng có thể kéo sập toàn bộ hệ thống, gây thiệt hại hàng triệu đô la.

**5. Scaling the team is hard**: Quy tắc của Conway: "Hệ thống sẽ phản ánh cấu trúc giao tiếp của tổ chức." Một monolith với 200 developers không thể hoạt động hiệu quả. Giao tiếp giữa các team trở nên hỗn loạn. Không ai thực sự "sở hữu" module nào. Trách nhiệm mờ nhạt.

### Giải pháp từ Microservices

Microservices giải quyết tất cả các vấn đề trên bằng cách chia nhỏ monolith thành nhiều service độc lập, mỗi service:

- **Owned by một team** (thường 3-8 developers) — team có toàn quyền quyết định về công nghệ, database, cách triển khai
- **Single responsibility** — mỗi service làm một việc và làm tốt việc đó
- **Independently deployable** — team có thể deploy service của mình mà không cần phối hợp với team khác
- **Independently scalable** — mỗi service có thể scale theo nhu cầu riêng
- **Failure isolated** — service A sập không kéo theo service B sập

Hãy xem xét một ví dụ cụ thể. Khi user tìm kiếm "iPhone 15" trên nền tảng thương mại điện tử, hành trình request đi qua nhiều service:

```
User → API Gateway → 
  1. Search Service (tìm kiếm sản phẩm)
  2. Product Service (lấy thông tin chi tiết)
  3. Inventory Service (kiểm tra tồn kho)
  4. Pricing Service (tính giá + khuyến mãi)
  5. Recommendation Service (sản phẩm gợi ý)
  6. Review Service (đánh giá, xếp hạng)
  7. User Service (thông tin user, loyalty points)
```

Mỗi service này do một team riêng phụ trách, có database riêng, có thể scale riêng, và có thể fail riêng mà không ảnh hưởng đến các service khác.

---

## Nguyên lý thiết kế

### 1. Single Responsibility per Service

Mỗi microservice nên tập trung vào một **business capability** duy nhất. Tiêu chí xác định ranh giới service:

- **Data ownership**: Service sở hữu dữ liệu của nó. Không share database giữa các service.
- **Business domain**: Xoay quanh một bounded context trong DDD (Domain-Driven Design)
- **Change reason**: Service chỉ thay đổi vì một lý do duy nhất — thay đổi trong business capability của nó
- **Team size**: Service vừa đủ để một team (3-8 người) có thể hiểu và maintain

### 2. Decentralized Data Management

Mỗi microservice có **database riêng** (Database-per-Service pattern). Điều này khác hoàn toàn với monolith nơi mọi module đều dùng chung một database. Hệ quả:

- **Data consistency**: Không có ACID transaction xuyên service. Phải dùng eventual consistency và saga pattern.
- **Data duplication**: Một số dữ liệu có thể được duplicate giữa các service (ví dụ: Product Service có thông tin sản phẩm, Search Service cũng có index riêng).
- **Data synchronization**: Dùng event-driven để đồng bộ dữ liệu giữa các service.

### 3. Communication Patterns

Có hai mô hình giao tiếp chính:

**Synchronous (Request-Response)**:
- HTTP/REST hoặc gRPC
- Đơn giản, quen thuộc
- Vấn đề: Coupling giữa các service, cascading failure
- Dùng khi: Cần response ngay lập tức, query data

**Asynchronous (Event-Driven)**:
- Message queue (RabbitMQ, Kafka)
- Decoupling cao, chịu lỗi tốt
- Vấn đề: Eventually consistency, debugging khó
- Dùng khi: Cần broadcast event, eventual consistency chấp nhận được

### 4. API Gateway Pattern

Một gateway duy nhất đứng trước tất cả microservices, đảm nhận:

- **Routing**: Định tuyến request đến đúng service
- **Authentication**: Xác thực tập trung
- **Rate limiting**: Giới hạn request
- **Load balancing**: Phân phối request
- **Aggregation**: Gọi nhiều service và aggregate response
- **Protocol translation**: REST → gRPC, HTTP → WebSocket

### 5. Service Discovery

Trong môi trường container (Kubernetes, Docker Swarm), service có thể được tạo và hủy động. Cần cơ chế discovery:

- **Client-side discovery**: Client (service consumer) tự query service registry
- **Server-side discovery**: Load balancer (API Gateway) biết vị trí của service

### 6. Resilience Patterns

| Pattern | Mô tả | Library |
|---------|-------|---------|
| **Circuit Breaker** | Ngăn gọi service đang lỗi, fail fast | Hystrix, resilience4j, pybreaker |
| **Retry with Backoff** | Thử lại khi lỗi tạm thời, có exponential backoff | tenacity (Python) |
| **Bulkhead** | Cô lập tài nguyên (thread pool riêng cho mỗi service) | ThreadPoolExecutor riêng |
| **Timeout** | Giới hạn thời gian chờ response | asyncio timeout, HTTP timeout |
| **Fallback** | Trả về default/dummy data khi service lỗi | Cache, default value |

### 7. Observability

Với nhiều service phân tán, observability là bắt buộc:

- **Distributed Tracing**: Theo dõi request xuyên suốt các service (Jaeger, Zipkin)
- **Centralized Logging**: Tất cả log về một chỗ (ELK Stack: Elasticsearch, Logstash, Kibana)
- **Metrics**: Prometheus + Grafana cho monitoring
- **Health Checks**: Endpoint `/health` mỗi service, readiness + liveness probes
- **SLA Monitoring**: Uptime, latency, error rate theo service

---

## Cấu trúc chi tiết

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                  │
│              Web App │ Mobile App │ Third-Party API                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Auth    │ │  Route   │ │  Rate    │ │  Cache   │ │  Log     │  │
│  │  Middle. │ │  Engine  │ │  Limiter │ │          │ │  Middle. │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
         ┌───────────────────────┼──────────────────────┐
         ▼                       ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Product        │  │   Order          │  │   User           │
│   Service        │  │   Service        │  │   Service        │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ - Product CRUD   │  │ - Cart mgmt      │  │ - Auth (JWT)     │
│ - Category mgmt  │  │ - Checkout       │  │ - Profile mgmt   │
│ - Search index   │  │ - Payment        │  │ - Address mgmt   │
│                  │  │ - Shipping       │  │ - Loyalty points  │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ Database: MySQL  │  │ Database: Postgres│  │ Database: MongoDB│
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                      │                      │
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MESSAGE BROKER (Kafka / RabbitMQ)                   │
│  Topics: product.created │ order.placed │ payment.completed │ ...     │
└─────────────────────────────────────────────────────────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Notification   │  │   Analytics      │  │   Recommendation │
│   Service        │  │   Service        │  │   Service        │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ - Email          │  │ - Event tracking │  │ - ML model       │
│ - SMS            │  │ - Report gen     │  │ - Product rec    │
│ - Push notif.    │  │ - Real-time dash │  │ - Personalization│
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Service Structure (per service)

```
product-service/
├── Dockerfile                 # Container build
├── requirements.txt           # Dependencies
├── pyproject.toml             # Python project config
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py            # HTTP endpoint tests
│   ├── test_service.py        # Business logic tests
│   └── test_repository.py     # Data access tests
├── src/
│   ├── __init__.py
│   ├── main.py                # FastAPI app + startup
│   ├── config.py              # Service configuration
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models.py          # Domain entities
│   │   ├── events.py          # Domain events
│   │   └── exceptions.py      # Service exceptions
│   ├── application/
│   │   ├── __init__.py
│   │   ├── service.py         # Use cases
│   │   └── interfaces.py      # Ports (repositories, messaging)
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── api/               # HTTP controllers
│   │   │   ├── __init__.py
│   │   │   ├── routes.py
│   │   │   ├── serializers.py
│   │   │   └── middlewares.py
│   │   ├── persistence/       # Database implementation
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   ├── models.py      # ORM models
│   │   │   └── repositories.py
│   │   └── messaging/         # Kafka/RabbitMQ implementation
│   │       ├── __init__.py
│   │       ├── producer.py
│   │       └── consumer.py
│   └── schemas/
│       ├── __init__.py
│       └── requests.py        # Pydantic request/response models
```

---

## Sơ đồ kiến trúc

```
                    E-COMMERCE MICROSERVICES ARCHITECTURE
                    =====================================

                          ┌──────────────┐
                          │   EXTERNAL    │
                          │   CLIENTS     │
                          │(Web, Mobile,  │
                          │  3rd Party)   │
                          └──────┬───────┘
                                 │ HTTPS
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Port: 8000  │  Auth: JWT │  Rate: 1000/min │  CORS: *      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Routes: /api/v1/products  → product-service:5001                   │
│          /api/v1/orders    → order-service:5002                     │
│          /api/v1/users     → user-service:5003                      │
│          /api/v1/search    → search-service:5004                    │
└────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Product Service │  │  Order Service  │  │  User Service   │
│  :5001          │  │  :5002          │  │  :5003          │
├────────────────┤  ├────────────────┤  ├────────────────┤
│                │  │                │  │                │
│  Products API   │  │  Orders API    │  │  Users API     │
│  Categories API │  │  Cart API      │  │  Auth API      │
│  Inventory API  │  │  Payment API   │  │  Address API   │
│                │  │  Shipping API   │  │  Loyalty API   │
│  ┌──────────┐  │  │  ┌──────────┐  │  │  ┌──────────┐  │
│  │PostgreSQL│  │  │  │PostgreSQL│  │  │  │ MongoDB  │  │
│  └──────────┘  │  │  └──────────┘  │  │  └──────────┘  │
└────────────────┘  └────────────────┘  └────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │ Async Events
                              ▼
              ┌─────────────────────────────┐
              │   KAFKA / NATS / RABBITMQ    │
              │                             │
              │  Topics:                    │
              │  order.placed               │
              │  payment.confirmed          │
              │  product.updated            │
              │  user.registered            │
              └─────────────────────────────┘
                              │
          ┌───────────────────┼──────────────┐
          ▼                   ▼              ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Notification   │  │  Analytics     │  │  Search        │
│ Service        │  │  Service       │  │  Service       │
├────────────────┤  ├────────────────┤  ├────────────────┤
│ Email (SendGrid)│  │  ELK Stack    │  │  Elasticsearch │
│ SMS (Twilio)   │  │  Prometheus    │  │  Index Builder │
│ Push (Firebase)│  │  Grafana       │  │  Auto-complete │
└────────────────┘  └────────────────┘  └────────────────┘
```

---

## Ví dụ code hoàn chỉnh

### File: `src/domain/models.py`

```python
"""Domain models — pure business entities with no infrastructure dependency."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import List, Optional
from uuid import uuid4


class ProductStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONTINUED = "discontinued"
    PENDING_REVIEW = "pending_review"


class ProductCategory(Enum):
    ELECTRONICS = "electronics"
    FASHION = "fashion"
    HOME = "home"
    SPORTS = "sports"
    BOOKS = "books"
    FOOD = "food"


class InventoryStatus(Enum):
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"


@dataclass
class Money:
    """Value object for monetary values."""
    amount: float
    currency: str = "VND"

    def __post_init__(self) -> None:
        self.amount = round(self.amount, 2)

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot subtract different currencies")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, multiplier: float) -> "Money":
        return Money(self.amount * multiplier, self.currency)


@dataclass
class Image:
    url: str
    alt_text: str = ""
    is_primary: bool = False


@dataclass
class Product:
    """Core domain entity."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    category: ProductCategory = ProductCategory.ELECTRONICS
    status: ProductStatus = ProductStatus.ACTIVE
    price: Money = field(default_factory=lambda: Money(0))
    images: List[Image] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def update_price(self, new_price: Money) -> None:
        if new_price.amount <= 0:
            raise ValueError("Price must be positive")
        self.price = new_price
        self.updated_at = datetime.now()
        self.version += 1

    def add_image(self, image: Image) -> None:
        if image.is_primary:
            for img in self.images:
                img.is_primary = False
        self.images.append(image)
        self.updated_at = datetime.now()
        self.version += 1

    def deactivate(self) -> None:
        self.status = ProductStatus.INACTIVE
        self.updated_at = datetime.now()
        self.version += 1


@dataclass
class Inventory:
    product_id: str
    quantity: int
    reserved_quantity: int = 0
    status: InventoryStatus = InventoryStatus.IN_STOCK
    warehouse_code: str = "WH-001"
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    @property
    def available_quantity(self) -> int:
        return self.quantity - self.reserved_quantity

    def reserve(self, quantity: int) -> bool:
        if quantity <= self.available_quantity:
            self.reserved_quantity += quantity
            self._update_status()
            return True
        return False

    def release(self, quantity: int) -> None:
        self.reserved_quantity = max(0, self.reserved_quantity - quantity)
        self._update_status()

    def confirm(self, quantity: int) -> None:
        if quantity <= self.quantity:
            self.quantity -= quantity
            self.reserved_quantity -= quantity
            self._update_status()

    def _update_status(self) -> None:
        if self.quantity <= 0:
            self.status = InventoryStatus.OUT_OF_STOCK
        elif self.available_quantity <= 5:
            self.status = InventoryStatus.LOW_STOCK
        else:
            self.status = InventoryStatus.IN_STOCK
```

### File: `src/domain/events.py`

```python
"""Domain events for inter-service communication."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4


@dataclass
class DomainEvent:
    """Base class for all domain events."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProductCreated(DomainEvent):
    product_id: str
    name: str
    category: str
    price: float


@dataclass
class ProductUpdated(DomainEvent):
    product_id: str
    changes: Dict[str, Any]
    version: int


@dataclass
class ProductDeactivated(DomainEvent):
    product_id: str
    reason: str = ""


@dataclass
class InventoryChanged(DomainEvent):
    product_id: str
    old_quantity: int
    new_quantity: int
    change_type: str  # "reserved" | "confirmed" | "restocked"
```

### File: `src/domain/exceptions.py`

```python
"""Domain-specific exceptions."""

from typing import Optional


class ProductNotFoundError(Exception):
    def __init__(self, product_id: str) -> None:
        self.product_id = product_id
        super().__init__(f"Product {product_id} not found")


class InsufficientInventoryError(Exception):
    def __init__(self, product_id: str, requested: int, available: int) -> None:
        self.product_id = product_id
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient inventory for product {product_id}: "
            f"requested {requested}, available {available}"
        )


class ProductNotActiveError(Exception):
    def __init__(self, product_id: str, status: str) -> None:
        super().__init__(f"Product {product_id} is not active (status: {status})")


class ValidationError(Exception):
    def __init__(self, message: str, field: Optional[str] = None) -> None:
        self.field = field
        super().__init__(message)
```

### File: `src/application/interfaces.py`

```python
"""Ports (interfaces) defined in the application layer."""

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

from src.domain.models import Inventory, Product
from src.domain.events import DomainEvent


class IProductRepository(ABC):
    """Port for product data access."""

    @abstractmethod
    async def find_by_id(self, product_id: str) -> Optional[Product]:
        ...

    @abstractmethod
    async def find_by_category(self, category: str, skip: int = 0, limit: int = 20) -> List[Product]:
        ...

    @abstractmethod
    async def search(self, query: str, skip: int = 0, limit: int = 20) -> List[Product]:
        ...

    @abstractmethod
    async def save(self, product: Product) -> Product:
        ...

    @abstractmethod
    async def delete(self, product_id: str) -> None:
        ...


class IInventoryRepository(ABC):
    """Port for inventory data access."""

    @abstractmethod
    async def find_by_product_id(self, product_id: str) -> Optional[Inventory]:
        ...

    @abstractmethod
    async def save(self, inventory: Inventory) -> Inventory:
        ...


class IEventPublisher(ABC):
    """Port for publishing domain events to message broker."""

    @abstractmethod
    async def publish(self, event: DomainEvent, topic: str) -> None:
        ...
```

### File: `src/application/service.py`

```python
"""Application services — implements use cases for the Product domain."""

import logging
from typing import List, Optional

from src.domain.models import (
    Image, Inventory, InventoryStatus, Money, Product,
    ProductCategory, ProductStatus,
)
from src.domain.events import (
    InventoryChanged,
    ProductCreated,
    ProductDeactivated,
    ProductUpdated,
)
from src.domain.exceptions import (
    InsufficientInventoryError,
    ProductNotFoundError,
    ProductNotActiveError,
    ValidationError,
)
from src.application.interfaces import IEventPublisher, IInventoryRepository, IProductRepository

logger = logging.getLogger(__name__)


class ProductService:
    """Core product use cases."""

    def __init__(
        self,
        product_repo: IProductRepository,
        inventory_repo: IInventoryRepository,
        event_publisher: IEventPublisher,
    ) -> None:
        self._product_repo = product_repo
        self._inventory_repo = inventory_repo
        self._event_publisher = event_publisher

    async def create_product(
        self,
        name: str,
        description: str,
        category: str,
        price: float,
        tags: Optional[List[str]] = None,
        attributes: Optional[dict] = None,
    ) -> Product:
        """Create a new product with initial inventory."""

        # Validate
        if not name or len(name.strip()) < 3:
            raise ValidationError("Product name must be at least 3 characters", "name")
        if price <= 0:
            raise ValidationError("Price must be positive", "price")

        try:
            category_enum = ProductCategory(category)
        except ValueError:
            raise ValidationError(f"Invalid category: {category}", "category")

        # Create domain entity
        product = Product(
            name=name.strip(),
            description=description.strip(),
            category=category_enum,
            price=Money(price),
            tags=tags or [],
            attributes=attributes or {},
        )

        # Persist
        saved = await self._product_repo.save(product)

        # Initialize inventory
        inventory = Inventory(product_id=saved.id, quantity=0)
        await self._inventory_repo.save(inventory)

        # Publish event
        await self._event_publisher.publish(
            ProductCreated(
                product_id=saved.id,
                name=saved.name,
                category=saved.category.value,
                price=saved.price.amount,
            ),
            topic="product.created",
        )

        logger.info(f"Product created: {saved.id} - {saved.name}")
        return saved

    async def update_product(
        self, product_id: str, updates: dict,
    ) -> Product:
        """Update product details."""
        product = await self._product_repo.find_by_id(product_id)
        if not product:
            raise ProductNotFoundError(product_id)

        # Track changes for event
        changes = {}

        if "name" in updates:
            product.name = updates["name"]
            changes["name"] = updates["name"]
        if "description" in updates:
            product.description = updates["description"]
            changes["description"] = updates["description"]
        if "price" in updates:
            product.update_price(Money(updates["price"]))
            changes["price"] = updates["price"]
        if "tags" in updates:
            product.tags = updates["tags"]
            changes["tags"] = updates["tags"]
        if "attributes" in updates:
            product.attributes = updates["attributes"]
            changes["attributes"] = updates["attributes"]

        product.version += 1
        saved = await self._product_repo.save(product)

        if changes:
            await self._event_publisher.publish(
                ProductUpdated(
                    product_id=product_id,
                    changes=changes,
                    version=saved.version,
                ),
                topic="product.updated",
            )

        return saved

    async def get_product(self, product_id: str) -> Product:
        """Get product by ID."""
        product = await self._product_repo.find_by_id(product_id)
        if not product:
            raise ProductNotFoundError(product_id)
        return product

    async def search_products(self, query: str, skip: int = 0, limit: int = 20) -> List[Product]:
        """Search products by name/description."""
        return await self._product_repo.search(query, skip, limit)

    async def deactivate_product(self, product_id: str, reason: str = "") -> Product:
        """Deactivate a product."""
        product = await self._product_repo.find_by_id(product_id)
        if not product:
            raise ProductNotFoundError(product_id)

        product.deactivate()
        saved = await self._product_repo.save(product)

        await self._event_publisher.publish(
            ProductDeactivated(product_id=product_id, reason=reason),
            topic="product.deactivated",
        )
        return saved


class InventoryService:
    """Inventory management use cases."""

    def __init__(
        self,
        inventory_repo: IInventoryRepository,
        product_repo: IProductRepository,
        event_publisher: IEventPublisher,
    ) -> None:
        self._inventory_repo = inventory_repo
        self._product_repo = product_repo
        self._event_publisher = event_publisher

    async def add_stock(self, product_id: str, quantity: int) -> Inventory:
        """Add stock to inventory."""
        if quantity <= 0:
            raise ValidationError("Quantity must be positive")

        inventory = await self._inventory_repo.find_by_product_id(product_id)
        if not inventory:
            # Create inventory if not exists
            inventory = Inventory(product_id=product_id, quantity=0)

        old_qty = inventory.quantity
        inventory.quantity += quantity
        inventory._update_status()
        inventory.version += 1

        saved = await self._inventory_repo.save(inventory)

        await self._event_publisher.publish(
            InventoryChanged(
                product_id=product_id,
                old_quantity=old_qty,
                new_quantity=saved.quantity,
                change_type="restocked",
            ),
            topic="inventory.changed",
        )
        return saved

    async def reserve_inventory(self, product_id: str, quantity: int) -> bool:
        """Reserve inventory for an order."""
        inventory = await self._inventory_repo.find_by_product_id(product_id)
        if not inventory:
            raise InsufficientInventoryError(product_id, quantity, 0)

        if not inventory.reserve(quantity):
            raise InsufficientInventoryError(
                product_id, quantity, inventory.available_quantity,
            )

        await self._inventory_repo.save(inventory)

        await self._event_publisher.publish(
            InventoryChanged(
                product_id=product_id,
                old_quantity=inventory.quantity,
                new_quantity=inventory.quantity - quantity,
                change_type="reserved",
            ),
            topic="inventory.changed",
        )
        return True

    async def confirm_inventory(self, product_id: str, quantity: int) -> Inventory:
        """Confirm reserved inventory after payment."""
        inventory = await self._inventory_repo.find_by_product_id(product_id)
        if not inventory:
            raise InventoryNotFoundError(product_id)

        old_qty = inventory.quantity
        inventory.confirm(quantity)
        inventory.version += 1

        saved = await self._inventory_repo.save(inventory)

        await self._event_publisher.publish(
            InventoryChanged(
                product_id=product_id,
                old_quantity=old_qty,
                new_quantity=saved.quantity,
                change_type="confirmed",
            ),
            topic="inventory.changed",
        )
        return saved

    async def get_inventory(self, product_id: str) -> Inventory:
        """Get current inventory for a product."""
        inventory = await self._inventory_repo.find_by_product_id(product_id)
        if not inventory:
            raise ProductNotFoundError(product_id)
        return inventory


class InventoryNotFoundError(Exception):
    def __init__(self, product_id: str) -> None:
        super().__init__(f"Inventory not found for product {product_id}")
```

### File: `src/infrastructure/persistence/database.py`

```python
"""Database configuration for the Product Service."""

from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    username: str = "product_user"
    password: str = "product_pass"
    database: str = "product_db"
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class DatabaseSession:
    """Manages async database sessions."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._engine = create_async_engine(
            config.url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False,
        )

    async def create_tables(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncSession:
        return self._session_factory()

    async def close(self) -> None:
        await self._engine.dispose()
```

### File: `src/infrastructure/persistence/models.py`

```python
"""SQLAlchemy models for Product Service."""

from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime,
    Text, JSON, Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.domain.models import ProductCategory, ProductStatus
from src.infrastructure.persistence.database import Base


class ProductModel(Base):
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    category: Mapped[ProductCategory] = mapped_column(
        SAEnum(ProductCategory), nullable=False, index=True,
    )
    status: Mapped[ProductStatus] = mapped_column(
        SAEnum(ProductStatus), nullable=False, default=ProductStatus.ACTIVE,
    )
    price_amount: Mapped[float] = mapped_column(Float, nullable=False)
    price_currency: Mapped[str] = mapped_column(String(3), nullable=False, default="VND")
    images: Mapped[str] = mapped_column(JSON, nullable=True, default=list)
    attributes: Mapped[str] = mapped_column(JSON, nullable=True, default=dict)
    tags: Mapped[str] = mapped_column(JSON, nullable=True, default=list)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class InventoryModel(Base):
    __tablename__ = "inventory"

    product_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reserved_quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[InventoryStatus] = mapped_column(
        SAEnum(InventoryStatus), nullable=False, default=InventoryStatus.IN_STOCK,
    )
    warehouse_code: Mapped[str] = mapped_column(String(10), nullable=False, default="WH-001")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


from src.domain.models import InventoryStatus  # noqa: E402, F811
```

### File: `src/infrastructure/persistence/repositories.py`

```python
"""Repository implementations using SQLAlchemy."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src.domain.models import Image, Inventory, Money, Product
from src.application.interfaces import IInventoryRepository, IProductRepository
from src.infrastructure.persistence.models import InventoryModel, ProductModel


class SQLAlchemyProductRepository(IProductRepository):
    """Product repository backed by SQLAlchemy + PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_id(self, product_id: str) -> Optional[Product]:
        result = await self._session.execute(
            select(ProductModel).where(ProductModel.id == product_id)
        )
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None

    async def find_by_category(
        self, category: str, skip: int = 0, limit: int = 20,
    ) -> List[Product]:
        result = await self._session.execute(
            select(ProductModel)
            .where(ProductModel.category == ProductCategory(category))
            .offset(skip).limit(limit)
        )
        return [self._to_domain(row) for row in result.scalars().all()]

    async def search(self, query: str, skip: int = 0, limit: int = 20) -> List[Product]:
        search_pattern = f"%{query}%"
        result = await self._session.execute(
            select(ProductModel)
            .where(
                or_(
                    ProductModel.name.ilike(search_pattern),
                    ProductModel.description.ilike(search_pattern),
                    ProductModel.tags[search_pattern].astext.isnot(None),
                )
            )
            .offset(skip).limit(limit)
        )
        return [self._to_domain(row) for row in result.scalars().all()]

    async def save(self, product: Product) -> Product:
        model = ProductModel(
            id=product.id,
            name=product.name,
            description=product.description,
            category=product.category,
            status=product.status,
            price_amount=product.price.amount,
            price_currency=product.price.currency,
            images=[{"url": img.url, "alt_text": img.alt_text, "is_primary": img.is_primary}
                    for img in product.images],
            attributes=product.attributes,
            tags=product.tags,
            version=product.version,
            created_at=product.created_at,
            updated_at=product.updated_at,
        )
        # Upsert logic
        await self._session.merge(model)
        await self._session.commit()
        return product

    async def delete(self, product_id: str) -> None:
        await self._session.execute(
            ProductModel.__table__.delete().where(ProductModel.id == product_id)
        )
        await self._session.commit()

    def _to_domain(self, model: ProductModel) -> Product:
        images = [
            Image(
                url=img["url"],
                alt_text=img.get("alt_text", ""),
                is_primary=img.get("is_primary", False),
            )
            for img in (model.images or [])
        ]
        return Product(
            id=model.id,
            name=model.name,
            description=model.description or "",
            category=model.category,
            status=model.status,
            price=Money(model.price_amount, model.price_currency),
            images=images,
            attributes=model.attributes or {},
            tags=model.tags or [],
            created_at=model.created_at,
            updated_at=model.updated_at,
            version=model.version,
        )


class SQLAlchemyInventoryRepository(IInventoryRepository):
    """Inventory repository backed by SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_product_id(self, product_id: str) -> Optional[Inventory]:
        result = await self._session.execute(
            select(InventoryModel).where(InventoryModel.product_id == product_id)
        )
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None

    async def save(self, inventory: Inventory) -> Inventory:
        model = InventoryModel(
            product_id=inventory.product_id,
            quantity=inventory.quantity,
            reserved_quantity=inventory.reserved_quantity,
            status=inventory.status,
            warehouse_code=inventory.warehouse_code,
            version=inventory.version,
            updated_at=inventory.updated_at,
        )
        await self._session.merge(model)
        await self._session.commit()
        return inventory

    def _to_domain(self, model: InventoryModel) -> Inventory:
        return Inventory(
            product_id=model.product_id,
            quantity=model.quantity,
            reserved_quantity=model.reserved_quantity,
            status=model.status,
            warehouse_code=model.warehouse_code,
            updated_at=model.updated_at,
            version=model.version,
        )
```

### File: `src/infrastructure/messaging/producer.py`

```python
"""Kafka event publisher implementation."""

import json
import logging
from typing import Optional

from src.domain.events import DomainEvent
from src.application.interfaces import IEventPublisher

logger = logging.getLogger(__name__)


class KafkaEventPublisher(IEventPublisher):
    """Publish domain events to Kafka topics."""

    def __init__(self, bootstrap_servers: str = "localhost:9092") -> None:
        self._bootstrap_servers = bootstrap_servers
        self._producer = None

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode(),
            )
            await self._producer.start()
            logger.info("Kafka producer connected")
        except ImportError:
            logger.warning("aiokafka not installed. Using stub producer.")
            self._producer = None

    async def publish(self, event: DomainEvent, topic: str) -> None:
        if self._producer is None:
            logger.info(f"[STUB] Published {type(event).__name__} to {topic}")
            return

        payload = {
            "event_type": type(event).__name__,
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "data": {
                k: v for k, v in event.__dict__.items()
                if k not in ("event_id", "timestamp")
            },
        }
        await self._producer.send(topic, value=payload)
        logger.debug(f"Published {type(event).__name__} to {topic}")

    async def close(self) -> None:
        if self._producer:
            await self._producer.stop()
```

### File: `src/infrastructure/api/routes.py`

```python
"""FastAPI routes for Product Service."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.domain.exceptions import (
    InsufficientInventoryError,
    ProductNotFoundError,
    ValidationError,
)
from src.application.service import ProductService, InventoryService
from src.infrastructure.api.serializers import (
    CreateProductRequest,
    InventoryResponse,
    ProductResponse,
    UpdateInventoryRequest,
    UpdateProductRequest,
)

logger = logging.getLogger(__name__)


def create_product_router(
    product_service: ProductService,
    inventory_service: InventoryService,
) -> APIRouter:
    """Factory function to create the router with dependencies."""
    router = APIRouter(prefix="/api/v1/products", tags=["products"])

    @router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
    async def create_product(request: CreateProductRequest):
        try:
            product = await product_service.create_product(
                name=request.name,
                description=request.description,
                category=request.category,
                price=request.price,
                tags=request.tags,
                attributes=request.attributes,
            )
            return ProductResponse.from_domain(product)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @router.get("/{product_id}", response_model=ProductResponse)
    async def get_product(product_id: str):
        try:
            product = await product_service.get_product(product_id)
            return ProductResponse.from_domain(product)
        except ProductNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("", response_model=List[ProductResponse])
    async def search_products(
        q: str = Query("", description="Search query"),
        skip: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),
    ):
        products = await product_service.search_products(q, skip, limit)
        return [ProductResponse.from_domain(p) for p in products]

    @router.patch("/{product_id}", response_model=ProductResponse)
    async def update_product(product_id: str, request: UpdateProductRequest):
        try:
            updates = {k: v for k, v in request.model_dump().items() if v is not None}
            product = await product_service.update_product(product_id, updates)
            return ProductResponse.from_domain(product)
        except ProductNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def deactivate_product(product_id: str):
        try:
            await product_service.deactivate_product(product_id)
        except ProductNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("/{product_id}/inventory", response_model=InventoryResponse)
    async def get_inventory(product_id: str):
        try:
            inventory = await inventory_service.get_inventory(product_id)
            return InventoryResponse.from_domain(inventory)
        except ProductNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.patch("/{product_id}/inventory", response_model=InventoryResponse)
    async def update_inventory(product_id: str, request: UpdateInventoryRequest):
        try:
            if request.quantity is not None:
                inventory = await inventory_service.add_stock(product_id, request.quantity)
            else:
                raise HTTPException(status_code=422, detail="Missing quantity")
            return InventoryResponse.from_domain(inventory)
        except ProductNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except InsufficientInventoryError as e:
            raise HTTPException(status_code=409, detail=str(e))

    return router
```

### File: `src/infrastructure/api/serializers.py`

```python
"""Pydantic request/response models for Product Service API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.domain.models import Inventory, Product


class CreateProductRequest(BaseModel):
    name: str = Field(..., min_length=3, max_length=200)
    description: str = ""
    category: str = Field(..., description="Product category slug")
    price: float = Field(..., gt=0, description="Price in VND")
    tags: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None


class UpdateProductRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    tags: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None


class UpdateInventoryRequest(BaseModel):
    quantity: Optional[int] = Field(None, gt=0)


class ProductResponse(BaseModel):
    id: str
    name: str
    description: str
    category: str
    status: str
    price: float
    currency: str
    tags: List[str]
    attributes: Dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, product: Product) -> "ProductResponse":
        return cls(
            id=product.id,
            name=product.name,
            description=product.description,
            category=product.category.value,
            status=product.status.value,
            price=product.price.amount,
            currency=product.price.currency,
            tags=product.tags,
            attributes=product.attributes,
            version=product.version,
            created_at=product.created_at,
            updated_at=product.updated_at,
        )


class InventoryResponse(BaseModel):
    product_id: str
    quantity: int
    reserved_quantity: int
    available_quantity: int
    status: str
    warehouse_code: str

    @classmethod
    def from_domain(cls, inventory: Inventory) -> "InventoryResponse":
        return cls(
            product_id=inventory.product_id,
            quantity=inventory.quantity,
            reserved_quantity=inventory.reserved_quantity,
            available_quantity=inventory.available_quantity,
            status=inventory.status.value,
            warehouse_code=inventory.warehouse_code,
        )
```

### File: `src/main.py`

```python
"""Product Service entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import ServiceConfig
from src.infrastructure.api.routes import create_product_router
from src.infrastructure.messaging.producer import KafkaEventPublisher
from src.infrastructure.persistence.database import DatabaseConfig, DatabaseSession
from src.infrastructure.persistence.repositories import (
    SQLAlchemyInventoryRepository,
    SQLAlchemyProductRepository,
)
from src.application.service import InventoryService, ProductService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup: initialize dependencies
    config = ServiceConfig.from_env()

    db_session = DatabaseSession(config.database)
    await db_session.create_tables()

    event_publisher = KafkaEventPublisher(config.kafka_bootstrap_servers)
    await event_publisher.connect()

    session = await db_session.get_session()
    product_repo = SQLAlchemyProductRepository(session)
    inventory_repo = SQLAlchemyInventoryRepository(session)

    product_service = ProductService(product_repo, inventory_repo, event_publisher)
    inventory_service = InventoryService(inventory_repo, product_repo, event_publisher)

    # Store in app state
    app.state.product_service = product_service
    app.state.inventory_service = inventory_service
    app.state.db_session = db_session
    app.state.event_publisher = event_publisher

    logger.info("Product Service started")

    yield

    # Shutdown
    await event_publisher.close()
    await db_session.close()
    logger.info("Product Service stopped")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Product Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Wire up routes
    router = create_product_router(
        product_service=app.state.product_service,
        inventory_service=app.state.inventory_service,
    )
    app.include_router(router)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "product-service"}

    return app


app = create_app()
```

### File: `docker-compose.yml`

```yaml
version: "3.9"

services:
  # API Gateway
  gateway:
    image: nginx:latest
    ports:
      - "8000:80"
    volumes:
      - ./gateway/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - product-service
      - order-service
      - user-service

  # Product Service
  product-service:
    build:
      context: ./product-service
      dockerfile: Dockerfile
    ports:
      - "5001:8000"
    environment:
      - DB_HOST=product-db
      - DB_PORT=5432
      - DB_USER=product_user
      - DB_PASS=product_pass
      - DB_NAME=product_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - product-db
      - kafka

  product-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: product_db
      POSTGRES_USER: product_user
      POSTGRES_PASSWORD: product_pass
    volumes:
      - product_data:/var/lib/postgresql/data

  # Order Service
  order-service:
    build:
      context: ./order-service
      dockerfile: Dockerfile
    ports:
      - "5002:8000"
    environment:
      - DB_HOST=order-db
      - DB_PORT=5432
      - DB_USER=order_user
      - DB_PASS=order_pass
      - DB_NAME=order_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - order-db
      - kafka

  order-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: order_db
      POSTGRES_USER: order_user
      POSTGRES_PASSWORD: order_pass
    volumes:
      - order_data:/var/lib/postgresql/data

  # Message Broker
  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

volumes:
  product_data:
  order_data:
```

---

## Kiểm thử

### File: `tests/test_services.py`

```python
"""Unit tests for Product and Inventory services."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.domain.models import Inventory, InventoryStatus, Money, Product, ProductCategory
from src.domain.exceptions import (
    InsufficientInventoryError,
    ProductNotFoundError,
    ValidationError,
)
from src.application.service import InventoryService, ProductService


@pytest.fixture
def mock_repos():
    return {
        "product_repo": AsyncMock(),
        "inventory_repo": AsyncMock(),
        "event_publisher": AsyncMock(),
    }


@pytest.fixture
def product_service(mock_repos) -> ProductService:
    return ProductService(
        product_repo=mock_repos["product_repo"],
        inventory_repo=mock_repos["inventory_repo"],
        event_publisher=mock_repos["event_publisher"],
    )


@pytest.fixture
def inventory_service(mock_repos) -> InventoryService:
    return InventoryService(
        inventory_repo=mock_repos["inventory_repo"],
        product_repo=mock_repos["product_repo"],
        event_publisher=mock_repos["event_publisher"],
    )


class TestProductService:

    @pytest.mark.asyncio
    async def test_create_product_success(self, product_service, mock_repos):
        mock_repos["product_repo"].save.return_value = Product(
            id="PROD-001",
            name="iPhone 15",
            price=Money(25_000_000),
            category=ProductCategory.ELECTRONICS,
        )
        mock_repos["inventory_repo"].save.return_value = Inventory(
            product_id="PROD-001", quantity=0,
        )

        result = await product_service.create_product(
            name="iPhone 15",
            description="Latest Apple phone",
            category="electronics",
            price=25_000_000,
        )

        assert result.name == "iPhone 15"
        assert result.price.amount == 25_000_000
        mock_repos["product_repo"].save.assert_awaited_once()
        mock_repos["event_publisher"].publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_product_invalid_name(self, product_service):
        with pytest.raises(ValidationError) as exc:
            await product_service.create_product(
                name="AB", description="", category="electronics", price=1000,
            )
        assert "name" in str(exc.value.field)

    @pytest.mark.asyncio
    async def test_create_product_invalid_category(self, product_service):
        with pytest.raises(ValidationError) as exc:
            await product_service.create_product(
                name="Test", description="", category="invalid_cat", price=1000,
            )
        assert "category" in str(exc.value.field)

    @pytest.mark.asyncio
    async def test_get_product_not_found(self, product_service, mock_repos):
        mock_repos["product_repo"].find_by_id.return_value = None

        with pytest.raises(ProductNotFoundError):
            await product_service.get_product("INVALID")

    @pytest.mark.asyncio
    async def test_update_product_price(self, product_service, mock_repos):
        product = Product(
            id="PROD-001", name="Test", price=Money(10_000),
            category=ProductCategory.ELECTRONICS,
        )
        mock_repos["product_repo"].find_by_id.return_value = product
        mock_repos["product_repo"].save.return_value = product

        result = await product_service.update_product("PROD-001", {"price": 15_000})

        assert result.price.amount == 15_000
        assert result.version == 2  # Version bumped


class TestInventoryService:

    @pytest.mark.asyncio
    async def test_add_stock(self, inventory_service, mock_repos):
        inventory = Inventory(product_id="PROD-001", quantity=10)
        mock_repos["inventory_repo"].find_by_product_id.return_value = inventory
        mock_repos["inventory_repo"].save.return_value = inventory

        result = await inventory_service.add_stock("PROD-001", 5)

        assert result.quantity == 15

    @pytest.mark.asyncio
    async def test_reserve_inventory_success(self, inventory_service, mock_repos):
        inventory = Inventory(product_id="PROD-001", quantity=10)
        mock_repos["inventory_repo"].find_by_product_id.return_value = inventory
        mock_repos["inventory_repo"].save.return_value = inventory

        result = await inventory_service.reserve_inventory("PROD-001", 3)
        assert result is True

    @pytest.mark.asyncio
    async def test_reserve_inventory_insufficient(self, inventory_service, mock_repos):
        inventory = Inventory(product_id="PROD-001", quantity=2)
        mock_repos["inventory_repo"].find_by_product_id.return_value = inventory

        with pytest.raises(InsufficientInventoryError) as exc:
            await inventory_service.reserve_inventory("PROD-001", 10)
        assert exc.value.requested == 10
        assert exc.value.available == 2

    @pytest.mark.asyncio
    async def test_confirm_inventory(self, inventory_service, mock_repos):
        inventory = Inventory(product_id="PROD-001", quantity=10, reserved_quantity=3)
        mock_repos["inventory_repo"].find_by_product_id.return_value = inventory
        mock_repos["inventory_repo"].save.return_value = inventory

        result = await inventory_service.confirm_inventory("PROD-001", 3)

        assert result.quantity == 7
        assert result.reserved_quantity == 0
```

### File: `tests/test_api.py`

```python
"""Integration tests for Product Service API."""

from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient

from src.domain.models import Inventory, Money, Product, ProductCategory, ProductStatus
from src.main import create_app


@pytest.fixture
def mock_services():
    """Patch service dependencies with mocks."""
    with patch("src.main.ProductService") as mock_ps, \
         patch("src.main.InventoryService") as mock_is:
        yield mock_ps, mock_is


@pytest.fixture
async def client(mock_services):
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


class TestProductAPI:

    @pytest.mark.asyncio
    async def test_create_product(self, client, mock_services):
        mock_ps, _ = mock_services
        mock_ps.return_value.create_product.return_value = Product(
            id="PROD-001",
            name="iPhone 15",
            description="Latest iPhone",
            category=ProductCategory.ELECTRONICS,
            status=ProductStatus.ACTIVE,
            price=Money(25_000_000),
            tags=["apple", "iphone"],
        )

        response = await client.post("/api/v1/products", json={
            "name": "iPhone 15",
            "description": "Latest iPhone",
            "category": "electronics",
            "price": 25_000_000,
        })

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "iPhone 15"
        assert data["price"] == 25_000_000

    @pytest.mark.asyncio
    async def test_create_product_validation_error(self, client):
        response = await client.post("/api/v1/products", json={
            "name": "AB",  # Too short
            "category": "electronics",
            "price": -100,  # Negative price
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_product_not_found(self, client, mock_services):
        mock_ps, _ = mock_services
        mock_ps.return_value.get_product.side_effect = ProductNotFoundError("INVALID")

        response = await client.get("/api/v1/products/INVALID")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_search_products(self, client, mock_services):
        mock_ps, _ = mock_services
        mock_ps.return_value.search_products.return_value = [
            Product(id="P1", name="iPhone", price=Money(20_000_000), category=ProductCategory.ELECTRONICS),
            Product(id="P2", name="iPad", price=Money(15_000_000), category=ProductCategory.ELECTRONICS),
        ]

        response = await client.get("/api/v1/products?q=iphone")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
```

---

## Khi nào dùng / Khi nào không

### ✅ Khi nào dùng Microservices

| Tình huống | Lý do |
|-----------|-------|
| **Hệ thống lớn, nhiều team** (>20 developers) | Conway's Law: tổ chức phản ánh kiến trúc |
| **Scale theo chiều ngang (horizontal scaling)** | Mỗi service scale độc lập, optimal resource |
| **Polyglot technology stack** | Service khác nhau dùng công nghệ khác nhau |
| **Frequent deployments** | Mỗi service deploy độc lập, không ảnh hưởng |
| **High availability requirements** | Failure isolation, không cascade |
| **Product complexity cao** | Nhiều business domain phức tạp |
| **Cloud-native / Container** | Kubernetes, Docker ecosystem |

### ❌ Khi nào KHÔNG dùng

| Tình huống | Lý do | Alternative |
|-----------|-------|-------------|
| **Ứng dụng nhỏ (< 10k users)** | Overhead >> benefit | Monolith, Layered |
| **Team nhỏ (< 5 developers)** | Mỗi người phải hiểu quá nhiều service | Modular Monolith |
| **Time-to-market gấp** | Microservices chậm hơn ban đầu | Start monolith, split later |
| **Domain đơn giản (CRUD)** | Không đủ phức tạp để justify | Layered Architecture |
| **Data consistency nghiêm ngặt** | Distributed transactions rất khó | Monolith, Event Sourcing |
| **Tổ chức chưa sẵn sàng** | Microservices yêu cầu DevOps mature | Đầu tư DevOps trước |
| **Legacy system** | Refactor monolith → microservices rủi ro | Strangler Fig pattern |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Independently deployable**: Deploy service mà không ảnh hưởng đến service khác | **Distributed complexity**: Network latency, partial failure, distributed tracing |
| **Independent scaling**: Scale từng service theo nhu cầu riêng | **Data consistency**: Eventually consistency, sagas, no ACID across services |
| **Technology diversity**: Dùng đúng công nghệ cho đúng việc | **Operations overhead**: Nhiều service = nhiều monitoring, logging, CI/CD |
| **Small codebase**: Mỗi service nhỏ, dễ hiểu, dễ maintain | **Debugging khó**: Lỗi có thể nằm ở bất kỳ service nào |
| **Team autonomy**: Team ownership, ít phối hợp với team khác | **Latency**: Gọi service qua network chậm hơn in-process call |
| **Fault isolation**: Service A lỗi không kéo sập Service B | **Duplicate code**: Cross-cutting concerns phải implement ở mỗi service |
| **Better scalability**: Horizontal scaling dễ dàng | **Testing phức tạp**: Integration test, contract test, E2E test |
| **Polyglot persistence**: Mỗi service chọn DB phù hợp | **Database per service**: Join query giữa các service không khả thi |

---

## Công cụ và Framework

### Service Development

| Công cụ | Mục đích |
|---------|----------|
| **FastAPI** (Python) | Async, type-safe, tự động docs |
| **Spring Boot** (Java) | Enterprise microservices, ecosystem lớn |
| **Go kit / Fiber** (Go) | High-performance, light-weight |
| **Express.js / NestJS** (Node.js) | Nhanh, event-loop based |

### API Gateway

| Công cụ | Mục đích |
|---------|----------|
| **Kong** | API gateway mạnh mẽ, plugin ecosystem |
| **Traefik** | Cloud-native, tự động service discovery |
| **NGINX** | Reverse proxy, load balancing |
| **AWS API Gateway** | Managed service, serverless |

### Service Mesh

| Công cụ | Mục đích |
|---------|----------|
| **Istio** | Service mesh chuẩn, nhiều tính năng |
| **Linkerd** | Nhẹ hơn Istio, dễ cài đặt |
| **Consul Connect** | HashiCorp ecosystem |

### Message Broker

| Công cụ | Mục đích |
|---------|----------|
| **Apache Kafka** | High-throughput, durable, streaming — **recommended** |
| **RabbitMQ** | Message queue truyền thống, reliable |
| **NATS** | Light-weight, high-performance |
| **AWS SQS/SNS** | Managed, serverless |

### Observability

| Công cụ | Mục đích |
|---------|----------|
| **Prometheus + Grafana** | Metrics và monitoring |
| **ELK Stack** (Elasticsearch, Logstash, Kibana) | Centralized logging |
| **Jaeger / Zipkin** | Distributed tracing |
| **Datadog / New Relic** | Managed APM |

### Container & Orchestration

| Công cụ | Mục đích |
|---------|----------|
| **Docker** | Container runtime |
| **Kubernetes** | Container orchestration |
| **Docker Compose** | Local development |
| **Helm** | Kubernetes package manager |

---

## Kiểm thử chiến lược

### Test Pyramid cho Microservices

```
         /\
        /  \
       /    \
      / E2E  \         ← End-to-End tests (5%)
     /  Tests  \
    /───────────\
   /  Contract   \      ← Contract tests (15%)
  /    Tests      \
 /─────────────────\
/   Service Tests   \   ← Service/Integration tests (30%)
/────────────────────\
/    Unit Tests       \  ← Unit tests (50%)
/──────────────────────\
```

### Testing Patterns

| Pattern | Mô tả | Thư viện |
|---------|-------|----------|
| **Consumer-Driven Contract** | API contract giữa services | Pact |
| **Service Virtualization** | Mock external services | WireMock |
| **Chaos Engineering** | Thử nghiệm resilience | Chaos Monkey |
| **Property-Based Testing** | Random input generation | Hypothesis |
| **Performance Testing** | Load test endpoints | locust, k6 |

### Pytest Configuration

```python
# pytest.ini
[pytest]
asyncio_mode = auto
testpaths = tests
markers =
    asyncio: asyncio-based tests
    integration: integration tests requiring external services
    slow: slow tests (> 5 seconds)

# conftest.py bổ sung
@pytest.fixture(scope="session")
def docker_services():
    """Start required services (Kafka, Postgres) for integration tests."""
    import subprocess
    subprocess.run(["docker-compose", "up", "-d", "kafka", "product-db"])
    yield
    subprocess.run(["docker-compose", "down"])
```

---

## Kết luận

Microservices Architecture là một bước tiến quan trọng trong kiến trúc phần mềm, giải quyết những vấn đề mà monolith không thể xử lý ở quy mô lớn. Tuy nhiên, nó không phải là "silver bullet" — nó đi kèm với distributed complexity, operational overhead, và yêu cầu tổ chức trưởng thành.

### Best Practices

1. **Start với monolith**, split khi thực sự cần (Strangler Fig pattern)
2. **Database per service** — không share database giữa các service
3. **API-first design** — contract trước, implementation sau
4. **Saga pattern** cho distributed transactions (không dùng 2PC)
5. **Idempotent APIs** — cho phép retry an toàn
6. **Circuit breaker** mọi external call
7. **Health check endpoints** cho mỗi service
8. **Centralized logging và tracing** — observability là bắt buộc
9. **Automated CI/CD** — deploy thường xuyên, deploy an toàn
10. **Team ownership** — mỗi service có team chịu trách nhiệm

### Golden Rules

> 1. **Mỗi service làm một việc và làm tốt việc đó** — nếu service có >3 lý do để thay đổi, hãy split.
> 2. **Không share code giữa services qua shared library** — dùng contract (API/event) thay vì code.
> 3. **Eventual consistency là mặc định** — chấp nhận data không consistent tức thời.
> 4. **Automate mọi thứ** — CI/CD, testing, deployment, monitoring.
> 5. **Design for failure** — mọi external call đều có thể fail, hãy chuẩn bị cho điều đó.

### Next Steps

Sau Microservices, hãy đọc **Event-Driven Architecture** — mô hình giao tiếp bất đồng bộ là trái tim của microservices. Hoặc **Hexagonal Architecture** để hiểu cách tổ chức code trong từng service sao cho testable và maintainable.
