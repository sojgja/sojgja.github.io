---
id: component-based
title: Component-Based Architecture
sidebar_label: 🏗️ Component-Based Architecture
sidebar_position: 54
---

# Component-Based Architecture

> **Component-Based Architecture (CBA)** — *"A software engineering paradigm that emphasizes the decomposition of systems into independent, reusable, replaceable, and composable components, each encapsulating a set of related functionality with well-defined interfaces."* — Clemens Szyperski, 1997

## Tổng quan

Component-Based Architecture (CBA) là một kiến trúc phần mềm trong đó hệ thống được xây dựng từ các **component** độc lập, có thể tái sử dụng, thay thế được, và có thể kết hợp với nhau thông qua các interface được định nghĩa rõ ràng. Mỗi component là một đơn vị triển khai độc lập, chứa cả code lẫn dữ liệu (hoặc reference đến dữ liệu) cho một chức năng cụ thể.

Khác với Object-Oriented Programming (OOP) nơi đơn vị cơ bản là **object** (instance của class), CBA lấy **component** làm đơn vị kiến trúc cơ bản. Một component thường chứa nhiều class, resources, configuration, và đôi khi là toàn bộ một subsystem.

**Nguồn gốc và lịch sử:**
- **1986 — Brad Cox**: Khái niệm "Software IC" — phần mềm như chip điện tử, có thể cắm ghép
- **1990s — Microsoft COM/DCOM**: Component Object Model — binary standard cho component interaction
- **1996 — Sun JavaBeans**: Component model cho Java
- **1998 — CORBA Component Model (CCM)**: OMG standard cho distributed components
- **2000s — Enterprise JavaBeans (EJB)**: Component model cho enterprise Java
- **2000s — .NET Components**: Assembly, NuGet packages
- **2010s — Microservices**: Kiến trúc component ở cấp độ service
- **2020s — Web Components**: W3C standard cho reusable UI components

**Những người tiên phong:**

| Tên | Đóng góp |
|-----|----------|
| **Brad Cox** | "Software IC" concept — nền tảng cho CBA |
| **Clemens Szyperski** | Định nghĩa chuẩn về Component Software |
| **Grady Booch** | UML — component diagram notation |
| **Robert C. Martin** | SOLID principles — nền tảng cho component design |
| **Ivar Jacobson** | Component-Based Software Engineering (CBSE) |

**Phân biệt với các khái niệm liên quan:**

| Khái niệm | So với Component |
|-----------|------------------|
| **Object (OOP)** | Component CHỨA nhiều objects. Component là deployment unit, object là runtime unit. |
| **Module** | Module là logical grouping. Component là physical deployment unit (JAR, DLL, package) |
| **Microservice** | Microservice là component triển khai độc lập qua network. Component có thể in-process. |
| **Package** | Package là distribution unit. Component là runtime unit + deployment unit. |

## Bài toán

### Vấn đề 1: Tái sử dụng code thất bại

Một tập đoàn tài chính có 5 sản phẩm phần mềm khác nhau: Internet Banking, Mobile Banking, Core Banking, CRM, và Risk Management. Mỗi sản phẩm đều cần xử lý "User Authentication" và "Transaction Logging". Nhưng mỗi đội implement riêng:
- Team A dùng bcrypt + PostgreSQL
- Team B dùng scrypt + MongoDB
- Team C dùng SHA-256 + file log

Kết quả: code trùng lặp, security inconsistency, khó audit, lãng phí 2000+ giờ dev mỗi năm. CBA giải quyết bằng cách đóng gói Authentication và Logging thành **shared components**. Mỗi sản phẩm chỉ cần khai báo dependency, import component, configure, và dùng. Component được maintain bởi một team riêng, security best practices được áp dụng một lần.

### Vấn đề 2: "Big ball of mud" — dependency hell

Kiến trúc monolithic truyền thống dẫn đến **circular dependency** và **tight coupling**:
- `AuthService` import `UserService`
- `UserService` import `OrderService`
- `OrderService` import `AuthService` (xác thực đơn hàng)

Kết quả: không thể test riêng module nào, không thể deploy riêng module nào, không thể hiểu system structure. Mỗi lần thay đổi đều sợ side effect.

CBA giải quyết bằng **component graph** với directed acyclic dependencies:
- Component A → Component B (A depends on B)
- Không có circular dependency
- Mỗi component có interface rõ ràng, implementation thay thế được

### Vấn đề 3: Deployment không linh hoạt

Một hệ thống ERP có 50+ modules. Mỗi lần bug fix nhỏ ở module "Inventory", phải redeploy toàn bộ ERP (200MB WAR/React build). Thời gian deploy: 30 phút, downtime: 5 phút. Nếu deploy 2 lần/tuần → 10 phút downtime/tuần → 8+ giờ downtime/năm.

Với CBA, mỗi component được deploy độc lập:
- Fix Inventory component → chỉ deploy Inventory component (5MB)
- Downtime: 0 (hot-reload hoặc rolling update)
- Risk: thấp (chỉ ảnh hưởng Inventory)

### Vấn đề 4: Third-party integration không kiểm soát

Công ty bảo hiểm tích hợp với 20+ đối tác (Bảo Việt, Bảo Minh, PVI, PTI, ...). Mỗi đối tác có API khác nhau, format khác nhau (SOAP, REST, custom XML). Nếu code tích hợp lẫn trong core business logic, khi một đối tác thay đổi API, phải sửa core code, test toàn bộ, redeploy toàn bộ.

CBA tách mỗi tích hợp thành một **adapter component**:
- Core business logic chỉ giao tiếp qua interface `InsuranceProvider`
- Mỗi đối tác có component riêng implement interface đó
- Đối tác thay đổi → chỉ sửa component của đối tác đó
- Thêm đối tác mới → thêm component mới, không sửa core

### Vấn đề 5: Testing khó do tight coupling

Trong monolithic, để test `processOrder()`, bạn cần:
- Database thật hoặc mock phức tạp
- Payment gateway sandbox
- Email server
- Shipping API

Setup mất 30 phút, test chạy 10 phút. Developer ngại viết test.

CBA cho phép **component isolation testing**:
- Mỗi component test riêng với mock của dependencies
- Component có thể test trong isolation (hexagonal testing)
- Integration test: chỉ test component interaction, không phải implementation

## Nguyên lý thiết kế

### 1. Component Contract — Interface trước, Implementation sau

Mỗi component có:
- **Provided Interface** — service component cung cấp cho component khác
- **Required Interface** — service component cần từ component khác

```
┌──────────────┐         ┌──────────────┐
│  Component A │ ──uses──▶  Component B │
│              │         │              │
│  ┌────────┐  │         │  ┌────────┐  │
│  │Provided│  │         │  │Provided│  │
│  │Interface│  │         │  │Interface│  │
│  └────────┘  │         │  └────────┘  │
│  ┌────────┐  │         │  ┌────────┐  │
│  │Required│  │         │  │Required│  │
│  └────────┘  │         │  └────────┘  │
└──────────────┘         └──────────────┘
```

### 2. Design by Contract (DbC)

Bertrand Meyer's Design by Contract:
- **Precondition**: Điều kiện để gọi component method (input validation)
- **Postcondition**: Đảm bảo sau khi method chạy xong (output guarantee)
- **Invariant**: Điều kiện luôn đúng trong suốt vòng đời component

### 3. Component Composition

Component được kết hợp theo ba cách:

| Pattern | Mô tả | Ví dụ |
|---------|-------|-------|
| **Sequential** | Component A gọi B, B gọi C | Pipeline, workflow |
| **Hierarchical** | Component chứa component con | UI component tree |
| **Peer-to-peer** | Các component cùng level, gọi nhau qua interface | Microservices |

### 4. Component Granularity

| Cấp độ | Kích thước | Ví dụ |
|--------|-----------|-------|
| **Fine-grained** | 1-5 classes | Logger, Validator, Mapper |
| **Medium-grained** | 5-20 classes | UserService, PaymentProcessor |
| **Coarse-grained** | 20+ classes | Authentication Component, Reporting Component |
| **Very coarse** | Whole subsystem | ERP Module, CRM System |

Nguyên tắc: Component đủ nhỏ để dễ hiểu, đủ lớn để có ý nghĩa business.

### 5. Component Discovery

Component có thể được khám phá:
- **Static**: Import trực tiếp (import package)
- **Dynamic**: Plugin/SPI discovery
- **Registry-based**: Component registry (Spring context, IoC container)
- **Service discovery**: DNS, Consul, Eureka (cho distributed components)

## Cấu trúc chi tiết

### Component Structure

Một component thường có cấu trúc:

```
component-name/
├── api/                   # Provided interfaces + DTOs
│   ├── __init__.py
│   ├── interfaces.py       # Abstract interfaces (SPI)
│   └── models.py           # Data transfer objects, value objects
├── internal/               # Implementation (private)
│   ├── __init__.py
│   ├── impl.py             # Implementation của interfaces
│   ├── repository.py       # Data access
│   └── validators.py       # Business validation
├── spi/                    # Required interfaces (what this component needs)
│   ├── __init__.py
│   └── required_interfaces.py
├── config/                 # Component configuration
│   ├── __init__.py
│   └── settings.py
├── exceptions/             # Component-specific exceptions
│   └── __init__.py
├── tests/                  # Component tests
│   ├── unit/
│   └── integration/
├── pyproject.toml          # Package metadata + dependencies
└── README.md
```

### Core Components (E-Commerce Example)

| Component | Provided Interface | Dependencies | Responsibility |
|-----------|-------------------|--------------|---------------|
| **UserComponent** | IUserService | — | User CRUD, authentication, authorization |
| **ProductComponent** | IProductService | — | Product catalog, inventory, pricing |
| **OrderComponent** | IOrderService | IUserService, IProductService | Order lifecycle management |
| **PaymentComponent** | IPaymentService | — | Payment gateway integration |
| **ShippingComponent** | IShippingService | IOrderService | Shipping calculation, logistics |
| **NotificationComponent** | INotificationService | IUserComponent | Email, SMS, push notifications |
| **AnalyticsComponent** | IAnalyticsService | IOrderService, IProductService | Reporting, BI |

### Dependency Graph

```
UserComponent        ProductComponent
    │                      │
    └──────────┬───────────┘
               │
         OrderComponent
          │         │
          │         │
    PaymentComp   NotificationComp
          │
      ShippingComp
```

## Sơ đồ kiến trúc (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    COMPONENT-BASED ARCHITECTURE                           │
│                                                                           │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  UserComponent   │    │ ProductComponent  │    │  ConfigComponent │   │
│  │  ──────────────  │    │ ────────────────  │    │ ───────────────  │   │
│  │  □ IUserService  │    │ □ IProductService │    │ □ IConfigService │   │
│  │  ──────────────  │    │ ────────────────  │    │ ───────────────  │   │
│  │  + register()    │    │ + getProduct()    │    │ + getConfig()    │   │
│  │  + authenticate()│    │ + search()        │    │ + updateConfig() │   │
│  │  + authorize()   │    │ + updateStock()   │    │                  │   │
│  └────────┬─────────┘    └────────┬──────────┘    └──────────────────┘   │
│           │                       │                                       │
│           └───────────┬───────────┘                                       │
│                       │                                                   │
│           ┌───────────▼────────────┐    ┌──────────────────┐              │
│           │   OrderComponent       │    │ NotificationComp │              │
│           │   ──────────────────   │    │ ───────────────  │              │
│           │   □ IOrderService      │◄───│ □ INotifyService │              │
│           │   ──────────────────   │    │ ───────────────  │              │
│           │   + createOrder()      │    │ + sendEmail()    │              │
│           │   + processPayment()   │    │ + sendSMS()      │              │
│           │   + cancelOrder()      │    │ + sendPush()     │              │
│           └────────────────────────┘    └──────────────────┘              │
│                       │                                                   │
│           ┌───────────▼────────────┐    ┌──────────────────┐              │
│           │   PaymentComponent     │    │  ShippingComp    │              │
│           │   ──────────────────   │    │ ───────────────  │              │
│           │   □ IPaymentService    │    │ □ IShippingSvc   │              │
│           │   ──────────────────   │    │ ───────────────  │              │
│           │   + processPayment()   │    │ + calculate()    │              │
│           │   + refund()         │    │ + createShip()   │              │
│           │   + validateCard()    │    │ + trackShipment() │              │
│           └────────────────────────┘    └──────────────────┘              │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    COMPOSITION ROOT                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │ DI Container │  │ Component    │  │ Wiring      │             │   │
│  │  │ (Autofac,    │  │ Registry     │  │ (Assembler) │             │   │
│  │  │  Spring)     │  │              │  │              │             │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

## Ví dụ code hoàn chỉnh

### Cách làm sai: Monolithic tightly-coupled

```python
from __future__ import annotations
from typing import Any


class MonolithicOrderProcessor:
    """Mọi thứ trong một class — tightly coupled, không thể test riêng."""

    def process_order(self, order_id: str, user_id: str, items: list[dict], payment_info: dict) -> dict:
        # User validation
        user = self._get_user_from_db(user_id)
        if not user or user["status"] != "active":
            return {"error": "User not found or inactive"}

        # Product validation
        total = 0
        for item in items:
            product = self._get_product_from_db(item["product_id"])
            if not product or product["stock"] < item["quantity"]:
                return {"error": f"Product {item['product_id']} out of stock"}
            total += product["price"] * item["quantity"]

        # Payment
        payment_result = self._call_payment_gateway(payment_info, total)
        if payment_result.get("status") != "success":
            return {"error": "Payment failed"}

        # Create order in DB
        order = self._save_order_to_db(order_id, user_id, items, total, payment_result)

        # Send email
        self._send_email(user["email"], "Order Confirmed", f"Your order {order_id} is confirmed")

        return {"order_id": order_id, "total": total, "status": "confirmed"}

    def _get_user_from_db(self, user_id: str) -> dict | None:
        return {"id": user_id, "name": "Test", "email": "test@test.com", "status": "active"}

    def _get_product_from_db(self, product_id: str) -> dict | None:
        return {"id": product_id, "name": "Product", "price": 100.0, "stock": 10}

    def _call_payment_gateway(self, payment_info: dict, amount: float) -> dict:
        return {"status": "success", "transaction_id": "TXN123"}

    def _save_order_to_db(self, order_id: str, user_id: str, items: list, total: float, payment: dict) -> dict:
        return {"order_id": order_id, "user_id": user_id, "total": total}

    def _send_email(self, to: str, subject: str, body: str) -> None:
        pass
```

### Cách làm đúng: Component-Based Architecture

```python
from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, TypeVar
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections import defaultdict
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# 1. CROSS-CUTTING TYPES
# ======================================================================

class ComponentState(Enum):
    CREATED = auto()
    INITIALIZED = auto()
    STARTED = auto()
    STOPPED = auto()
    FAILED = auto()


@dataclass
class Address:
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "VN"


@dataclass
class Money:
    amount: float
    currency: str = "VND"

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"Currency mismatch: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __mul__(self, factor: float) -> Money:
        return Money(self.amount * factor, self.currency)


@dataclass
class OrderItem:
    product_id: str
    name: str
    price: Money
    quantity: int

    @property
    def subtotal(self) -> Money:
        return self.price * self.quantity


@dataclass
class Order:
    order_id: str
    user_id: str
    items: list[OrderItem]
    shipping_address: Address
    subtotal: Money = field(default_factory=lambda: Money(0))
    shipping_fee: Money = field(default_factory=lambda: Money(0))
    tax: Money = field(default_factory=lambda: Money(0))
    discount: Money = field(default_factory=lambda: Money(0))
    total: Money = field(default_factory=lambda: Money(0))
    status: str = "pending"
    payment_id: str = ""
    tracking_id: str = ""
    created_at: float = field(default_factory=time.time)

    def calculate_total(self) -> None:
        self.total = self.subtotal + self.shipping_fee + self.tax - self.discount


class OrderStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


# ======================================================================
# 2. COMPONENT MODEL — COMPOSITION ROOT
# ======================================================================

T = TypeVar("T")


class ComponentRegistry:
    """Central registry — quản lý component instances và lifecycle."""

    def __init__(self) -> None:
        self._instances: dict[type, Any] = {}
        self._named_instances: dict[str, Any] = {}
        self._states: dict[str, ComponentState] = {}
        self._lock = threading.RLock()

    def register(self, interface: type, implementation: Any, name: str = "") -> None:
        with self._lock:
            self._instances[interface] = implementation
            if name:
                self._named_instances[name] = implementation
            impl_name = name or interface.__name__
            self._states[impl_name] = ComponentState.CREATED
            logger.debug("Registered: %s → %s", interface.__name__, type(implementation).__name__)

    def resolve(self, interface: type, name: str = "") -> Any:
        with self._lock:
            if name:
                return self._named_instances.get(name)
            return self._instances.get(interface)

    def resolve_all(self, interface: type) -> list[Any]:
        with self._lock:
            return [impl for iface, impl in self._instances.items()
                    if issubclass(iface, interface) or iface == interface]

    def set_state(self, name: str, state: ComponentState) -> None:
        with self._lock:
            self._states[name] = state

    def get_state(self, name: str) -> ComponentState:
        return self._states.get(name, ComponentState.CREATED)

    def initialize_all(self) -> None:
        for name, impl in list(self._named_instances.items()) + \
                          [(t.__name__, i) for t, i in self._instances.items()
                           if t.__name__ not in self._named_instances]:
            if hasattr(impl, "initialize"):
                try:
                    impl.initialize(self)
                    self.set_state(name, ComponentState.INITIALIZED)
                    logger.info("Initialized component: %s", name)
                except Exception as e:
                    self.set_state(name, ComponentState.FAILED)
                    logger.error("Failed to initialize component %s: %s", name, e)

    def start_all(self) -> None:
        for name, impl in list(self._named_instances.items()) + \
                          [(t.__name__, i) for t, i in self._instances.items()
                           if t.__name__ not in self._named_instances]:
            if hasattr(impl, "start"):
                try:
                    impl.start()
                    self.set_state(name, ComponentState.STARTED)
                    logger.info("Started component: %s", name)
                except Exception as e:
                    logger.error("Failed to start component %s: %s", name, e)

    def stop_all(self) -> None:
        for name, impl in list(self._named_instances.items()) + \
                          [(t.__name__, i) for t, i in self._instances.items()
                           if t.__name__ not in self._named_instances]:
            if hasattr(impl, "stop"):
                try:
                    impl.stop()
                    self.set_state(name, ComponentState.STOPPED)
                    logger.info("Stopped component: %s", name)
                except Exception as e:
                    logger.error("Failed to stop component %s: %s", name, e)


class Component(ABC):
    """Abstract base cho mọi component — có lifecycle."""

    @abstractmethod
    def initialize(self, registry: ComponentRegistry) -> None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


# ======================================================================
# 3. COMPONENT: USER COMPONENT
# ======================================================================

@dataclass
class User:
    user_id: str
    email: str
    name: str
    role: str = "customer"  # customer, admin, manager
    is_active: bool = True
    created_at: float = field(default_factory=time.time)


class IUserService(Protocol):
    """Provided interface của UserComponent."""

    def register(self, email: str, name: str, password: str) -> User: ...
    def authenticate(self, email: str, password: str) -> User | None: ...
    def get_user(self, user_id: str) -> User | None: ...
    def get_users_by_role(self, role: str) -> list[User]: ...
    def deactivate_user(self, user_id: str) -> bool: ...


class UserComponent(Component):
    """User management component."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._email_index: dict[str, User] = {}
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self, registry: ComponentRegistry) -> None:
        logger.info("UserComponent initializing...")
        registry.register(IUserService, self)
        # Seed data
        self._users["user_001"] = User("user_001", "alice@example.com", "Alice", "customer")
        self._users["user_002"] = User("user_002", "bob@example.com", "Bob", "customer")
        self._users["user_003"] = User("user_003", "admin@example.com", "Admin", "admin")
        for u in self._users.values():
            self._email_index[u.email] = u
        self._initialized = True
        logger.info("UserComponent initialized with %d users", len(self._users))

    def start(self) -> None:
        logger.info("UserComponent started")

    def stop(self) -> None:
        with self._lock:
            self._users.clear()
            self._email_index.clear()
        logger.info("UserComponent stopped")

    def register(self, email: str, name: str, password: str) -> User:
        with self._lock:
            if email in self._email_index:
                raise ValueError(f"Email already registered: {email}")
            user_id = f"user_{int(time.time())}"
            user = User(user_id=user_id, email=email, name=name)
            self._users[user_id] = user
            self._email_index[email] = user
            logger.info("Registered user: %s (%s)", name, email)
            return user

    def authenticate(self, email: str, password: str) -> User | None:
        if email in self._email_index:
            user = self._email_index[email]
            if user.is_active:
                logger.info("Authenticated user: %s", email)
                return user
        return None

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def get_users_by_role(self, role: str) -> list[User]:
        return [u for u in self._users.values() if u.role == role]

    def deactivate_user(self, user_id: str) -> bool:
        user = self._users.get(user_id)
        if user:
            user.is_active = False
            logger.info("Deactivated user: %s", user_id)
            return True
        return False


# ======================================================================
# 4. COMPONENT: PRODUCT CATALOG
# ======================================================================

@dataclass
class Product:
    product_id: str
    name: str
    description: str
    price: Money
    category: str
    stock: int
    is_active: bool = True


class IProductService(Protocol):
    """Provided interface của ProductComponent."""

    def get_product(self, product_id: str) -> Product | None: ...
    def search_products(self, query: str, category: str = "") -> list[Product]: ...
    def update_stock(self, product_id: str, quantity: int) -> bool: ...
    def get_products_by_category(self, category: str) -> list[Product]: ...
    def add_product(self, product: Product) -> None: ...


class ProductComponent(Component):
    """Product catalog component."""

    def __init__(self) -> None:
        self._products: dict[str, Product] = {}
        self._lock = threading.RLock()

    def initialize(self, registry: ComponentRegistry) -> None:
        logger.info("ProductComponent initializing...")
        registry.register(IProductService, self)
        # Seed data
        self._products = {
            "prod_001": Product("prod_001", "iPhone 15 Pro", "Latest Apple phone",
                                Money(27990000), "Electronics", 50),
            "prod_002": Product("prod_002", "MacBook Pro 16", "Apple laptop M3 Pro",
                                Money(59990000), "Electronics", 20),
            "prod_003": Product("prod_003", "AirPods Pro", "Wireless earbuds",
                                Money(5490000), "Electronics", 100),
            "prod_004": Product("prod_004", "Samsung Galaxy S24", "Android flagship",
                                Money(21990000), "Electronics", 30),
            "prod_005": Product("prod_005", "Sony WH-1000XM5", "Noise cancelling headphones",
                                Money(7990000), "Electronics", 40),
            "prod_006": Product("prod_006", "Nike Air Max", "Running shoes",
                                Money(3500000), "Fashion", 60),
            "prod_007": Product("prod_007", "Levi's 501", "Classic jeans",
                                Money(1200000), "Fashion", 80),
        }
        logger.info("ProductComponent initialized with %d products", len(self._products))

    def start(self) -> None:
        logger.info("ProductComponent started")

    def stop(self) -> None:
        with self._lock:
            self._products.clear()
        logger.info("ProductComponent stopped")

    def get_product(self, product_id: str) -> Product | None:
        return self._products.get(product_id)

    def search_products(self, query: str, category: str = "") -> list[Product]:
        query = query.lower()
        results = []
        with self._lock:
            for product in self._products.values():
                if (query in product.name.lower() or query in product.description.lower()):
                    if not category or product.category == category:
                        results.append(product)
        return results

    def update_stock(self, product_id: str, quantity: int) -> bool:
        product = self._products.get(product_id)
        if product and product.stock >= quantity:
            product.stock -= quantity
            logger.info("Stock updated: %s (-%d, remaining=%d)", product_id, quantity, product.stock)
            return True
        return False

    def get_products_by_category(self, category: str) -> list[Product]:
        return [p for p in self._products.values() if p.category == category]

    def add_product(self, product: Product) -> None:
        with self._lock:
            self._products[product.product_id] = product


# ======================================================================
# 5. COMPONENT: PAYMENT PROCESSING
# ======================================================================

@dataclass
class PaymentTransaction:
    transaction_id: str
    order_id: str
    amount: Money
    method: str  # credit_card, bank_transfer, momo, vnpay
    status: str  # pending, success, failed, refunded
    gateway_response: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class IPaymentService(Protocol):
    """Provided interface của PaymentComponent."""

    def process_payment(self, order_id: str, amount: Money, method: str, payment_info: dict[str, Any]) -> PaymentTransaction: ...
    def refund_payment(self, transaction_id: str) -> PaymentTransaction: ...
    def get_transaction(self, transaction_id: str) -> PaymentTransaction | None: ...
    def get_available_methods(self) -> list[str]: ...


class PaymentComponent(Component):
    """Payment processing component."""

    def __init__(self) -> None:
        self._transactions: dict[str, PaymentTransaction] = {}
        self._lock = threading.RLock()
        self._payment_methods = ["credit_card", "bank_transfer", "vnpay", "momo"]

    def initialize(self, registry: ComponentRegistry) -> None:
        logger.info("PaymentComponent initializing...")
        registry.register(IPaymentService, self)
        logger.info("PaymentComponent initialized")

    def start(self) -> None:
        logger.info("PaymentComponent started")

    def stop(self) -> None:
        with self._lock:
            self._transactions.clear()
        logger.info("PaymentComponent stopped")

    def process_payment(self, order_id: str, amount: Money, method: str, payment_info: dict[str, Any]) -> PaymentTransaction:
        if method not in self._payment_methods:
            raise ValueError(f"Unsupported payment method: {method}")
        txn_id = f"TXN{int(time.time())}"
        logger.info("Processing payment: order=%s method=%s amount=%.0f", order_id, method, amount.amount)
        txn = PaymentTransaction(
            transaction_id=txn_id,
            order_id=order_id,
            amount=amount,
            method=method,
            status="success",
            gateway_response={"code": "00", "message": "Approved"},
        )
        with self._lock:
            self._transactions[txn_id] = txn
        logger.info("Payment success: %s (%.0f %s)", txn_id, amount.amount, amount.currency)
        return txn

    def refund_payment(self, transaction_id: str) -> PaymentTransaction:
        txn = self._transactions.get(transaction_id)
        if txn is None:
            raise ValueError(f"Transaction not found: {transaction_id}")
        if txn.status != "success":
            raise ValueError(f"Cannot refund transaction with status: {txn.status}")
        txn.status = "refunded"
        logger.info("Payment refunded: %s", transaction_id)
        return txn

    def get_transaction(self, transaction_id: str) -> PaymentTransaction | None:
        return self._transactions.get(transaction_id)

    def get_available_methods(self) -> list[str]:
        return list(self._payment_methods)


# ======================================================================
# 6. COMPONENT: SHIPPING
# ======================================================================

@dataclass
class ShippingQuote:
    method: str
    fee: Money
    estimated_days: int
    provider: str


@dataclass
class Shipment:
    tracking_id: str
    order_id: str
    method: str
    status: str
    address: Address
    created_at: float = field(default_factory=time.time)


class IShippingService(Protocol):
    """Provided interface của ShippingComponent."""

    def get_quotes(self, items: list[OrderItem], address: Address) -> list[ShippingQuote]: ...
    def create_shipment(self, order_id: str, method: str, address: Address) -> Shipment: ...
    def track_shipment(self, tracking_id: str) -> dict[str, Any]: ...


class ShippingComponent(Component):
    """Shipping management component."""

    def __init__(self) -> None:
        self._shipments: dict[str, Shipment] = {}
        self._lock = threading.RLock()

    def initialize(self, registry: ComponentRegistry) -> None:
        logger.info("ShippingComponent initializing...")
        registry.register(IShippingService, self)
        logger.info("ShippingComponent initialized")

    def start(self) -> None:
        logger.info("ShippingComponent started")

    def stop(self) -> None:
        with self._lock:
            self._shipments.clear()
        logger.info("ShippingComponent stopped")

    def get_quotes(self, items: list[OrderItem], address: Address) -> list[ShippingQuote]:
        total_weight = sum(item.quantity for item in items)
        return [
            ShippingQuote("standard", Money(total_weight * 15000), 3, "GHN"),
            ShippingQuote("express", Money(total_weight * 30000), 1, "GHN Express"),
            ShippingQuote("economy", Money(total_weight * 10000), 5, "ViettelPost"),
        ]

    def create_shipment(self, order_id: str, method: str, address: Address) -> Shipment:
        tracking_id = f"SHIP{int(time.time())}"
        with self._lock:
            shipment = Shipment(tracking_id, order_id, method, "created", address)
            self._shipments[tracking_id] = shipment
        logger.info("Shipment created: %s for order %s (%s)", tracking_id, order_id, method)
        return shipment

    def track_shipment(self, tracking_id: str) -> dict[str, Any]:
        shipment = self._shipments.get(tracking_id)
        if not shipment:
            raise ValueError(f"Shipment not found: {tracking_id}")
        return {
            "tracking_id": tracking_id,
            "order_id": shipment.order_id,
            "status": shipment.status,
            "method": shipment.method,
            "estimated_delivery": "3-5 business days",
        }


# ======================================================================
# 7. COMPONENT: NOTIFICATION
# ======================================================================

@dataclass
class Notification:
    notification_id: str
    user_id: str
    channel: str  # email, sms, push, in_app
    title: str
    message: str
    status: str = "pending"
    sent_at: float | None = None


class INotificationService(Protocol):
    """Provided interface của NotificationComponent."""

    def send_email(self, user_id: str, subject: str, body: str) -> Notification: ...
    def send_sms(self, user_id: str, message: str) -> Notification: ...
    def send_in_app(self, user_id: str, title: str, message: str) -> Notification: ...
    def get_user_notifications(self, user_id: str, limit: int = 10) -> list[Notification]: ...


class NotificationComponent(Component):
    """Notification management component."""

    def __init__(self) -> None:
        self._notifications: dict[str, Notification] = {}
        self._lock = threading.RLock()

    def initialize(self, registry: ComponentRegistry) -> None:
        logger.info("NotificationComponent initializing...")
        registry.register(INotificationService, self)

        # Required interface: need IUserService to get user email
        self._user_service: IUserService | None = None
        # We'll resolve it lazily
        logger.info("NotificationComponent initialized")

    def start(self) -> None:
        logger.info("NotificationComponent started")

    def stop(self) -> None:
        with self._lock:
            self._notifications.clear()
        logger.info("NotificationComponent stopped")

    def _resolve_user_service(self, registry: ComponentRegistry) -> None:
        if self._user_service is None:
            self._user_service = registry.resolve(IUserService)

    def send_email(self, user_id: str, subject: str, body: str) -> Notification:
        notif_id = f"NOTIF{int(time.time())}"
        notif = Notification(notif_id, user_id, "email", subject, body, "sent", time.time())
        with self._lock:
            self._notifications[notif_id] = notif
        logger.info("[EMAIL] User=%s | Subject=%s", user_id, subject)
        return notif

    def send_sms(self, user_id: str, message: str) -> Notification:
        notif_id = f"NOTIF{int(time.time())}"
        notif = Notification(notif_id, user_id, "sms", "SMS", message, "sent", time.time())
        with self._lock:
            self._notifications[notif_id] = notif
        logger.info("[SMS] User=%s | Message=%s...", user_id, message[:30])
        return notif

    def send_in_app(self, user_id: str, title: str, message: str) -> Notification:
        notif_id = f"NOTIF{int(time.time())}"
        notif = Notification(notif_id, user_id, "in_app", title, message, "sent", time.time())
        with self._lock:
            self._notifications[notif_id] = notif
        return notif

    def get_user_notifications(self, user_id: str, limit: int = 10) -> list[Notification]:
        with self._lock:
            user_notifs = [n for n in self._notifications.values() if n.user_id == user_id]
            return sorted(user_notifs, key=lambda n: n.sent_at or 0, reverse=True)[:limit]


# ======================================================================
# 8. COMPONENT: ORDER PROCESSING (CORE BUSINESS)
# ======================================================================

class IOrderService(Protocol):
    """Provided interface của OrderComponent."""

    def create_order(self, user_id: str, items: list[dict[str, Any]], shipping_address: Address,
                     payment_method: str, payment_info: dict[str, Any]) -> Order: ...
    def get_order(self, order_id: str) -> Order | None: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_user_orders(self, user_id: str) -> list[Order]: ...


class OrderComponent(Component):
    """Core business component — phối hợp các component khác."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._lock = threading.RLock()
        self._product_service: IProductService | None = None
        self._payment_service: IPaymentService | None = None
        self._shipping_service: IShippingService | None = None
        self._notification_service: INotificationService | None = None
        self._user_service: IUserService | None = None

    def initialize(self, registry: ComponentRegistry) -> None:
        logger.info("OrderComponent initializing...")
        registry.register(IOrderService, self)

        # Resolve dependencies from registry
        self._product_service = registry.resolve(IProductService)
        self._payment_service = registry.resolve(IPaymentService)
        self._shipping_service = registry.resolve(IShippingService)
        self._notification_service = registry.resolve(INotificationService)
        self._user_service = registry.resolve(IUserService)

        if not all([self._product_service, self._payment_service, self._shipping_service,
                    self._notification_service, self._user_service]):
            logger.warning("OrderComponent: some dependencies not resolved")

        logger.info("OrderComponent initialized")

    def start(self) -> None:
        logger.info("OrderComponent started")

    def stop(self) -> None:
        with self._lock:
            self._orders.clear()
        logger.info("OrderComponent stopped")

    def create_order(self, user_id: str, items_data: list[dict[str, Any]],
                     shipping_address: Address, payment_method: str,
                     payment_info: dict[str, Any]) -> Order:
        """Tạo đơn hàng — phối hợp các components."""
        logger.info("=" * 60)
        logger.info("Creating order for user=%s with %d items", user_id, len(items_data))

        # 1. Validate user
        user = self._user_service.get_user(user_id) if self._user_service else None
        if not user or not user.is_active:
            raise ValueError(f"User not found or inactive: {user_id}")
        logger.info("User validated: %s (%s)", user.name, user.role)

        # 2. Validate products and build OrderItem list
        order_items: list[OrderItem] = []
        for item_data in items_data:
            product = self._product_service.get_product(item_data["product_id"]) if self._product_service else None
            if not product or not product.is_active:
                raise ValueError(f"Product not found: {item_data['product_id']}")
            if product.stock < item_data.get("quantity", 1):
                raise ValueError(f"Product {product.name} out of stock (available: {product.stock})")
            order_items.append(OrderItem(
                product_id=product.product_id,
                name=product.name,
                price=product.price,
                quantity=item_data.get("quantity", 1),
            ))
            # Reserve stock
            if self._product_service:
                self._product_service.update_stock(product.product_id, item_data.get("quantity", 1))
        logger.info("Products validated: %d items", len(order_items))

        # 3. Calculate subtotal
        subtotal = Money(sum(
            item.price.amount * item.quantity for item in order_items
        ))
        logger.info("Subtotal: %.0f", subtotal.amount)

        # 4. Calculate shipping
        shipping_fee = Money(0)
        shipping_method = "standard"
        if self._shipping_service:
            quotes = self._shipping_service.get_quotes(order_items, shipping_address)
            if quotes:
                shipping_method = quotes[0].method
                shipping_fee = quotes[0].fee
                logger.info("Shipping: %s (%.0f)", shipping_method, shipping_fee.amount)

        # 5. Calculate tax (simplified: 8% VAT)
        tax = Money(subtotal.amount * 0.08)
        logger.info("Tax (8%%): %.0f", tax.amount)

        # 6. Create order
        order_id = f"ORD{int(time.time())}_{user_id[:4]}"
        order = Order(
            order_id=order_id,
            user_id=user_id,
            items=order_items,
            shipping_address=shipping_address,
            subtotal=subtotal,
            shipping_fee=shipping_fee,
            tax=tax,
        )
        order.calculate_total()
        logger.info("Total: %.0f", order.total.amount)

        # 7. Process payment
        if self._payment_service:
            txn = self._payment_service.process_payment(order_id, order.total, payment_method, payment_info)
            order.payment_id = txn.transaction_id
            order.status = "paid"
            logger.info("Payment processed: %s", txn.transaction_id)
        else:
            order.status = "pending_payment"

        # 8. Create shipment
        if self._shipping_service and order.status == "paid":
            shipment = self._shipping_service.create_shipment(order_id, shipping_method, shipping_address)
            order.tracking_id = shipment.tracking_id
            order.status = "processing"
            logger.info("Shipment created: %s", shipment.tracking_id)

        # 9. Send notifications
        if self._notification_service:
            self._notification_service.send_email(
                user_id, f"Order {order_id} Confirmed",
                f"Your order has been confirmed. Total: {order.total.amount} {order.total.currency}. "
                f"Tracking: {order.tracking_id or 'N/A'}.",
            )
            self._notification_service.send_in_app(
                user_id, "Order Confirmed",
                f"Order {order_id} confirmed. Track your shipment with ID: {order.tracking_id}",
            )

        # 10. Store order
        with self._lock:
            self._orders[order_id] = order
        logger.info("Order %s created successfully!", order_id)
        return order

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status in ("pending", "paid"):
            order.status = "cancelled"
            logger.info("Order cancelled: %s", order_id)
            return True
        return False

    def get_user_orders(self, user_id: str) -> list[Order]:
        return [o for o in self._orders.values() if o.user_id == user_id]


# ======================================================================
# 9. COMPONENT: ANALYTICS (EXTENSION)
# ======================================================================

class IAnalyticsService(Protocol):
    def get_sales_summary(self, days: int = 7) -> dict[str, Any]: ...
    def get_popular_products(self, limit: int = 5) -> list[dict[str, Any]]: ...


class AnalyticsComponent(Component):
    """Analytics component — extension component."""

    def __init__(self) -> None:
        self._order_service: IOrderService | None = None

    def initialize(self, registry: ComponentRegistry) -> None:
        logger.info("AnalyticsComponent initializing...")
        registry.register(IAnalyticsService, self)
        self._order_service = registry.resolve(IOrderService)
        logger.info("AnalyticsComponent initialized")

    def start(self) -> None:
        logger.info("AnalyticsComponent started")

    def stop(self) -> None:
        logger.info("AnalyticsComponent stopped")

    def get_sales_summary(self, days: int = 7) -> dict[str, Any]:
        if not self._order_service:
            return {"error": "Order service not available"}
        # In production: query from database
        return {
            "total_orders": 0,
            "total_revenue": 0,
            "average_order_value": 0,
            "period_days": days,
        }

    def get_popular_products(self, limit: int = 5) -> list[dict[str, Any]]:
        return []


# ======================================================================
# 10. COMPOSITION ROOT — SYSTEM ASSEMBLER
# ======================================================================

class SystemAssembler:
    """Composition root — khởi tạo và wiring tất cả components."""

    def __init__(self) -> None:
        self.registry = ComponentRegistry()
        self._components: list[Component] = []

    def build(self) -> ComponentRegistry:
        """Build the component graph."""
        logger.info("=== Building Component System ===")

        # 1. Create all components
        components = [
            UserComponent(),
            ProductComponent(),
            PaymentComponent(),
            ShippingComponent(),
            NotificationComponent(),
            OrderComponent(),
            AnalyticsComponent(),
        ]

        # 2. Initialize (each component registers its interface + resolves dependencies)
        for component in components:
            component.initialize(self.registry)
            self._components.append(component)

        # 3. Start all
        self.registry.start_all()

        logger.info("=== System built with %d components ===", len(components))
        return self.registry

    def shutdown(self) -> None:
        logger.info("=== Shutting down system ===")
        self.registry.stop_all()
        logger.info("=== System shut down ===")

    def get_component_states(self) -> dict[str, str]:
        return {name: state.name for name, state in self.registry._states.items()}


# ======================================================================
# 11. MAIN — SIMULATION
# ======================================================================

def main() -> None:
    logger.info("=== Component-Based Architecture: E-Commerce Platform ===")

    # Build system
    assembler = SystemAssembler()
    registry = assembler.build()

    # Get services
    user_service = registry.resolve(IUserService)
    product_service = registry.resolve(IProductService)
    order_service = registry.resolve(IOrderService)
    notification_service = registry.resolve(INotificationService)

    if not all([user_service, product_service, order_service]):
        logger.error("Required services not available!")
        return

    # Show available products
    logger.info("\n=== Available Products ===")
    products = product_service.search_products("")
    for p in products:
        logger.info("  %s | %s | %.0f VND | Stock: %d", p.product_id, p.name, p.price.amount, p.stock)

    # Create orders
    logger.info("\n=== Creating Orders ===")

    order1 = order_service.create_order(
        user_id="user_001",
        items_data=[
            {"product_id": "prod_001", "quantity": 1},
            {"product_id": "prod_003", "quantity": 2},
        ],
        shipping_address=Address("123 Nguyễn Huệ", "HCM", "HCM", "70000"),
        payment_method="credit_card",
        payment_info={"card_number": "4111-1111-1111-1111", "cvv": "123"},
    )

    order2 = order_service.create_order(
        user_id="user_002",
        items_data=[{"product_id": "prod_002", "quantity": 1}],
        shipping_address=Address("456 Lê Lợi", "Hà Nội", "HN", "10000"),
        payment_method="momo",
        payment_info={"phone": "0909123456"},
    )

    # Query orders
    logger.info("\n=== User Orders ===")
    for uid in ["user_001", "user_002"]:
        orders = order_service.get_user_orders(uid)
        user = user_service.get_user(uid)
        logger.info("\n%s's Orders:", user.name if user else uid)
        for o in orders:
            logger.info("  %s | Status: %s | Total: %.0f VND | Items: %d",
                        o.order_id, o.status, o.total.amount, len(o.items))

    # Try cancelling
    order_service.cancel_order("ORD_cancel_test")
    logger.info("\n=== Component System Stats ===")
    states = assembler.get_component_states()
    for name, state in states.items():
        if "Component" in name:
            logger.info("  %s: %s", name, state)

    # Shutdown
    assembler.shutdown()
    logger.info("=== Component-Based Architecture Demo Complete ===")


if __name__ == "__main__":
    main()
```

## Khi nào dùng / Khi nào không

| Khi nào dùng | Khi nào không |
|--------------|---------------|
| Hệ thống lớn, nhiều module, cần phân chia rõ ràng | Hệ thống nhỏ (< 10k LOC) — overhead không đáng |
| Nhiều sản phẩm dùng chung components | Toàn bộ system chỉ có 1-2 tính năng |
| Cần tái sử dụng code trên nhiều projects | Composition overhead là không chấp nhận được |
| Team phân tán, mỗi team maintain component riêng | Circular dependency tự nhiên trong domain |
| Cần thay thế implementation (A/B testing) | Yêu cầu performance tối đa (in-process call > IPC) |
| Third-party integration cần pluggable | Component interface thay đổi liên tục |
| Multi-product strategy (platform thinking) | Hệ thống batch processing đơn giản |

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Tái sử dụng cao**: Component dùng lại trên nhiều sản phẩm | **Overhead thiết kế ban đầu lớn**: Cần thiết kế contract cẩn thận |
| **Thay thế được**: Implementation swap dễ dàng qua interface | **Complexity**: Component graph có thể rất phức tạp |
| **Triển khai độc lập**: Mỗi component deploy riêng | **Versioning**: Component dependency version management khó |
| **Testing dễ dàng**: Mock component, isolation test | **Performance**: Interface abstraction có overhead (dispatch) |
| **Parallel development**: Team làm việc độc lập | **Component granularity khó**: Quá nhỏ → nhiều interfaces; quá lớn → monolithic |
| **System comprehension**: Component diagram dễ hiểu | **Circular dependency detection**: Cần tool hỗ trợ |
| **Language agnostic**: Component có thể viết bằng ngôn ngữ khác | **Distribution overhead**: Nếu component distributed (cần serialization) |

## Công cụ và Framework

| Tên | Loại | Ngôn ngữ | Ghi chú |
|-----|------|----------|---------|
| **Spring IOC / Spring Boot** | Framework | Java | Component scanning, DI, AOP |
| **Google Guice** | DI Framework | Java | Lightweight dependency injection |
| **Autofac** | DI Container | .NET | .NET component lifecycle |
| **Unity Container** | DI Container | .NET | Microsoft's DI container |
| **Python inject** | Library | Python | Lightweight DI |
| **dependency-injector** | Library | Python | Python DI framework |
| **Angular DI** | Framework | TypeScript | Client-side component DI |
| **React Components** | Library | JavaScript | UI component model |
| **Web Components** | W3C Standard | JavaScript | Browser-native component model |
| **OSGi** | Module System | Java | Dynamic component loading |
| **.NET MAUI / WPF** | UI Framework | .NET | XAML component model |
| **Flutter Widgets** | UI Framework | Dart | Widget component tree |

## Kiểm thử

Component-based architecture yêu cầu testing ở nhiều cấp độ: unit test cho component riêng, integration test cho component interaction, end-to-end test cho toàn hệ thống.

```python
from __future__ import annotations
import pytest
from typing import Any, Protocol


class TestUserComponent:
    def test_register_user(self, user_component_initialized: IUserService) -> None:
        user = user_component_initialized.register("new@test.com", "New User", "pass123")
        assert user.email == "new@test.com"
        assert user.name == "New User"
        assert user.is_active is True
        assert user.role == "customer"

    def test_register_duplicate_email(self, user_component_initialized: IUserService) -> None:
        with pytest.raises(ValueError, match="already registered"):
            user_component_initialized.register("alice@example.com", "Alice Again", "pass")

    def test_authenticate_valid(self, user_component_initialized: IUserService) -> None:
        user = user_component_initialized.authenticate("alice@example.com", "pass")
        assert user is not None
        assert user.user_id == "user_001"

    def test_authenticate_inactive(self, user_component_initialized: IUserService) -> None:
        user_component_initialized.deactivate_user("user_001")
        user = user_component_initialized.authenticate("alice@example.com", "pass")
        assert user is None

    def test_get_user(self, user_component_initialized: IUserService) -> None:
        user = user_component_initialized.get_user("user_001")
        assert user is not None
        assert user.name == "Alice"

    def test_get_users_by_role(self, user_component_initialized: IUserService) -> None:
        admins = user_component_initialized.get_users_by_role("admin")
        assert len(admins) == 1
        assert admins[0].email == "admin@example.com"

    def test_deactivate_user(self, user_component_initialized: IUserService) -> None:
        result = user_component_initialized.deactivate_user("user_002")
        assert result is True
        user = user_component_initialized.get_user("user_002")
        assert user is not None
        assert user.is_active is False


class TestProductComponent:
    def test_get_product(self, product_component_initialized: IProductService) -> None:
        product = product_component_initialized.get_product("prod_001")
        assert product is not None
        assert product.name == "iPhone 15 Pro"
        assert product.price.amount == 27990000

    def test_search_products(self, product_component_initialized: IProductService) -> None:
        results = product_component_initialized.search_products("iphone")
        assert len(results) >= 1
        assert all("iphone" in r.name.lower() for r in results)

    def test_update_stock_success(self, product_component_initialized: IProductService) -> None:
        result = product_component_initialized.update_stock("prod_001", 10)
        assert result is True
        product = product_component_initialized.get_product("prod_001")
        assert product is not None
        assert product.stock == 40  # originally 50 - 10

    def test_update_stock_insufficient(self, product_component_initialized: IProductService) -> None:
        result = product_component_initialized.update_stock("prod_001", 999)
        assert result is False

    def test_get_products_by_category(self, product_component_initialized: IProductService) -> None:
        fashion = product_component_initialized.get_products_by_category("Fashion")
        assert len(fashion) == 2
        assert all(p.category == "Fashion" for p in fashion)


class TestPaymentComponent:
    def test_process_payment(self, payment_component_initialized: IPaymentService) -> None:
        txn = payment_component_initialized.process_payment(
            "ORD001", Money(100000), "credit_card", {"number": "4111-1111"}
        )
        assert txn.status == "success"
        assert txn.amount.amount == 100000
        assert txn.order_id == "ORD001"

    def test_refund_payment(self, payment_component_initialized: IPaymentService) -> None:
        txn = payment_component_initialized.process_payment("ORD002", Money(50000), "momo", {})
        refunded = payment_component_initialized.refund_payment(txn.transaction_id)
        assert refunded.status == "refunded"

    def test_unsupported_method(self, payment_component_initialized: IPaymentService) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            payment_component_initialized.process_payment("ORD003", Money(100), "bitcoin", {})

    def test_get_available_methods(self, payment_component_initialized: IPaymentService) -> None:
        methods = payment_component_initialized.get_available_methods()
        assert "credit_card" in methods
        assert "momo" in methods
        assert "vnpay" in methods


class TestShippingComponent:
    def test_get_quotes(self, shipping_component_initialized: IShippingService) -> None:
        items = [OrderItem("p1", "Product", Money(100000), 2)]
        address = Address("123 Street", "City", "ST", "12345")
        quotes = shipping_component_initialized.get_quotes(items, address)
        assert len(quotes) == 3
        assert quotes[0].fee.amount > 0

    def test_create_shipment(self, shipping_component_initialized: IShippingService) -> None:
        address = Address("456 Road", "Town", "ST", "67890")
        shipment = shipping_component_initialized.create_shipment("ORD001", "express", address)
        assert shipment.tracking_id.startswith("SHIP")
        assert shipment.order_id == "ORD001"

    def test_track_shipment(self, shipping_component_initialized: IShippingService) -> None:
        address = Address("789 Ave", "City", "ST", "11111")
        shipment = shipping_component_initialized.create_shipment("ORD002", "standard", address)
        tracking = shipping_component_initialized.track_shipment(shipment.tracking_id)
        assert tracking["order_id"] == "ORD002"
        assert "estimated_delivery" in tracking


class TestOrderComponent:
    def test_create_order_success(self, system_with_components: IOrderService) -> None:
        order = system_with_components.create_order(
            user_id="user_001",
            items_data=[{"product_id": "prod_001", "quantity": 1}],
            shipping_address=Address("123 Main", "City", "ST", "12345"),
            payment_method="credit_card",
            payment_info={"number": "4111-1111"},
        )
        assert order.order_id.startswith("ORD")
        assert order.status in ("paid", "processing")
        assert len(order.items) == 1
        assert order.payment_id != ""

    def test_create_order_insufficient_stock(self, system_with_components: IOrderService) -> None:
        with pytest.raises(ValueError):
            system_with_components.create_order(
                user_id="user_001",
                items_data=[{"product_id": "prod_001", "quantity": 9999}],
                shipping_address=Address("123 Main", "City", "ST", "12345"),
                payment_method="credit_card",
                payment_info={},
            )

    def test_create_order_inactive_user(self, system_with_components: IOrderService) -> None:
        with pytest.raises(ValueError):
            system_with_components.create_order(
                user_id="nonexistent",
                items_data=[{"product_id": "prod_001", "quantity": 1}],
                shipping_address=Address("123 Main", "City", "ST", "12345"),
                payment_method="credit_card",
                payment_info={},
            )

    def test_cancel_order(self, system_with_components: IOrderService) -> None:
        order = system_with_components.create_order(
            user_id="user_001",
            items_data=[{"product_id": "prod_003", "quantity": 1}],
            shipping_address=Address("123 Main", "City", "ST", "12345"),
            payment_method="credit_card",
            payment_info={},
        )
        result = system_with_components.cancel_order(order.order_id)
        assert result is True
        cancelled = system_with_components.get_order(order.order_id)
        assert cancelled is not None
        assert cancelled.status == "cancelled"

    def test_get_user_orders(self, system_with_components: IOrderService) -> None:
        system_with_components.create_order(
            user_id="user_001",
            items_data=[{"product_id": "prod_001", "quantity": 1}],
            shipping_address=Address("123 Main", "City", "ST", "12345"),
            payment_method="credit_card",
            payment_info={},
        )
        orders = system_with_components.get_user_orders("user_001")
        assert len(orders) >= 1
        assert all(o.user_id == "user_001" for o in orders)


class TestComponentLifecycle:
    def test_component_states(self) -> None:
        assembler = SystemAssembler()
        registry = assembler.build()
        states = assembler.get_component_states()
        for name, state in states.items():
            assert state in ("STARTED", "INITIALIZED"), f"{name} is {state}"
        assembler.shutdown()
        for name, state in assembler.get_component_states().items():
            assert state in ("STOPPED", "FAILED"), f"{name} is {state} after shutdown"

    def test_component_dependency_resolution(self) -> None:
        assembler = SystemAssembler()
        registry = assembler.build()
        # All core services should be resolvable
        assert registry.resolve(IUserService) is not None
        assert registry.resolve(IProductService) is not None
        assert registry.resolve(IPaymentService) is not None
        assert registry.resolve(IShippingService) is not None
        assert registry.resolve(INotificationService) is not None
        assert registry.resolve(IOrderService) is not None
        assembler.shutdown()

    def test_component_wiring(self) -> None:
        """OrderComponent should have its dependencies wired correctly."""
        assembler = SystemAssembler()
        registry = assembler.build()
        order_component = registry.resolve(IOrderService)
        if order_component:
            # Should be able to create an order (which requires all dependencies)
            order = order_component.create_order(
                user_id="user_001",
                items_data=[{"product_id": "prod_001", "quantity": 1}],
                shipping_address=Address("Test", "City", "ST", "12345"),
                payment_method="credit_card",
                payment_info={},
            )
            assert order is not None
        assembler.shutdown()


class TestNotificationComponent:
    def test_send_email(self, notification_component_initialized: INotificationService) -> None:
        notif = notification_component_initialized.send_email("user_001", "Test", "Hello")
        assert notif.status == "sent"
        assert notif.channel == "email"

    def test_get_user_notifications(self, notification_component_initialized: INotificationService) -> None:
        notification_component_initialized.send_email("user_001", "Subject", "Body")
        notification_component_initialized.send_sms("user_001", "SMS Body")
        notifs = notification_component_initialized.get_user_notifications("user_001")
        assert len(notifs) == 2

    def test_notifications_scoped_to_user(self, notification_component_initialized: INotificationService) -> None:
        notification_component_initialized.send_email("user_001", "For Alice", "Hello")
        notification_component_initialized.send_email("user_002", "For Bob", "Hi")
        alice_notifs = notification_component_initialized.get_user_notifications("user_001")
        assert len(alice_notifs) == 1
        assert alice_notifs[0].title == "For Alice"


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def user_component() -> UserComponent:
    return UserComponent()


@pytest.fixture
def user_component_initialized(user_component: UserComponent) -> IUserService:
    registry = ComponentRegistry()
    user_component.initialize(registry)
    registry.start_all()
    return registry.resolve(IUserService)


@pytest.fixture
def product_component() -> ProductComponent:
    return ProductComponent()


@pytest.fixture
def product_component_initialized(product_component: ProductComponent) -> IProductService:
    registry = ComponentRegistry()
    product_component.initialize(registry)
    registry.start_all()
    return registry.resolve(IProductService)


@pytest.fixture
def payment_component() -> PaymentComponent:
    return PaymentComponent()


@pytest.fixture
def payment_component_initialized(payment_component: PaymentComponent) -> IPaymentService:
    registry = ComponentRegistry()
    payment_component.initialize(registry)
    registry.start_all()
    return registry.resolve(IPaymentService)


@pytest.fixture
def shipping_component() -> ShippingComponent:
    return ShippingComponent()


@pytest.fixture
def shipping_component_initialized(shipping_component: ShippingComponent) -> IShippingService:
    registry = ComponentRegistry()
    shipping_component.initialize(registry)
    registry.start_all()
    return registry.resolve(IShippingService)


@pytest.fixture
def notification_component() -> NotificationComponent:
    return NotificationComponent()


@pytest.fixture
def notification_component_initialized(notification_component: NotificationComponent) -> INotificationService:
    registry = ComponentRegistry()
    notification_component.initialize(registry)
    registry.start_all()
    return registry.resolve(INotificationService)


@pytest.fixture
def system_with_components() -> IOrderService:
    assembler = SystemAssembler()
    registry = assembler.build()
    return registry.resolve(IOrderService)
```

## Kết luận

Component-Based Architecture là một trong những kiến trúc nền tảng nhất trong software engineering. Bằng cách phân chia hệ thống thành các component độc lập với interface rõ ràng, CBA mang lại khả năng tái sử dụng, thay thế, và mở rộng vượt trội. Đây là kiến trúc cơ sở cho hầu hết các hệ thống enterprise hiện đại — từ Spring Framework, Angular, React đến microservices.

**Best Practices:**
- **Thiết kế interface trước, implementation sau** (Contract-First Development)
- **Component granularity**: Một component nên giải quyết một business capability. Nếu component có nhiều hơn 7±2 classes, xem xét tách.
- **Dependency direction**: Component cấp cao không phụ thuộc component cấp thấp — cả hai phụ thuộc abstraction (DIP).
- **Stable dependencies**: Component core không phụ thuộc component concrete — inject qua interface.
- **Component testability**: Mỗi component phải testable trong isolation với mock dependencies.
- **Versioning strategy**: Component API phải versioned (SemVer) — breaking change = major version.
- **Document component contract**: Precondition, postcondition, invariant cho mỗi public method.

**Golden Rules:**
1. Component là deployment unit, không phải design unit. Interface là design unit.
2. Package theo component, không theo layer (by feature, not by layer).
3. Component không được có circular dependency — dùng Dependency Inversion để break cycle.
4. Component phải có interface riêng — không export internal classes.
5. Component registry (IoC container) là composition root — chỉ dùng ở application entry point.
6. Component nên stateless hoặc state được quản lý rõ ràng.
7. Component contract phải được test như một phần của CI/CD pipeline.
