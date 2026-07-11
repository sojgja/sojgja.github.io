---
id: microkernel
title: Microkernel Architecture (Plugin)
sidebar_label: 🏗️ Microkernel Architecture
sidebar_position: 53
---

# Microkernel Architecture (Plugin)

> **Microkernel Architecture** — *"A minimal core system provides essential services, while additional capabilities are delivered through plug-in modules that can be independently developed, tested, and deployed."* — OS Design Principle, 1980s

## Tổng quan

Microkernel Architecture, còn gọi là **Plugin Architecture**, là một kiến trúc phần mềm trong đó một **core system** nhỏ gọn (microkernel) cung cấp các chức năng tối thiểu cần thiết, và các tính năng bổ sung được triển khai dưới dạng **plugin modules** độc lập. Core system định nghĩa các **extension points** (điểm mở rộng) — interface chuẩn mà mọi plugin phải tuân theo.

Khái niệm microkernel bắt nguồn từ lĩnh vực hệ điều hành. **Mach kernel** (Carnegie Mellon University, 1985) là một trong những microkernel đầu tiên, chỉ cung cấp quản lý task, IPC, và memory management — mọi thứ khác (file system, network stack, device drivers) đều chạy ở user space. **QNX** và **Minix** theo cùng triết lý. Ngày nay, **macOS** dùng XNU kernel (hybrid, nhưng ảnh hưởng bởi microkernel). **Google's Fuchsia** dùng Zircon microkernel.

**Những người tiên phong:**

| Tên | Đóng góp |
|-----|----------|
| **Richard Rashid** | Mach microkernel (CMU) |
| **Andrew S. Tanenbaum** | Minix microkernel |
| **Dan Dodge** | QNX microkernel (real-time OS) |
| **Erich Gamma** | Eclipse plugin architecture (OSGi) |
| **Kent Beck** | Plugin architecture cho JUnit |
| **Martin Fowler** | Định nghĩa "Plugin Architecture" trong enterprise patterns |

**Ứng dụng hiện đại:**

| Lĩnh vực | Ví dụ |
|----------|-------|
| **IDE** | VS Code (extensions), Eclipse (plugins), IntelliJ (plugins) |
| **CI/CD** | Jenkins (plugins), GitHub Actions (actions) |
| **Web Browser** | Chrome extensions, Firefox add-ons |
| **Game Engine** | Unity (packages), Unreal Engine (plugins) |
| **E-Commerce** | WooCommerce (plugins), Magento (modules) |
| **CMS** | WordPress (plugins), Drupal (modules) |
| **Build Tool** | Webpack (loaders/plugins), Vite (plugins), Babel (plugins) |
| **Logging** | Log4j (appenders), SLF4J (bindings) |

## Bài toán

### Vấn đề 1: Core system phồng to, khó bảo trì

Một nền tảng thương mại điện tử SaaS phục vụ 500+ merchant. Mỗi merchant có yêu cầu tích hợp riêng: merchant A cần tính thuế theo luật Việt Nam, merchant B cần tích hợp với ShipStation, merchant C cần custom payment gateway, merchant D cần ERP sync. Nếu tất cả các tính năng này đều nằm trong core codebase, codebase sẽ phình lên hàng triệu dòng, mỗi bản deploy phải test tất cả — không khả thi.

Microkernel giải quyết: Core chỉ cung cấp order management, product catalog, user management cơ bản. Mỗi tính năng riêng là một plugin: `tax-vn-plugin`, `shipping-shipstation-plugin`, `payment-vnpay-plugin`, `erp-sync-plugin`. Merchant chỉ cài plugin họ cần.

### Vấn đề 2: Third-party developers cần mở rộng hệ thống

Một nền tảng phân tích dữ liệu (analytics platform) muốn cho phép cộng đồng phát triển các visualization plugin — biểu đồ mới, dashboard widget, data source connector. Nếu core team phải tự viết tất cả, họ không thể theo kịp nhu cầu đa dạng của users.

Plugin architecture cho phép:
- **ISV (Independent Software Vendor)** viết plugin
- **Community** đóng góp open-source plugin
- **Khách hàng** tự viết plugin nội bộ

### Vấn đề 3: Release cycle conflict

Core system cần stable, release chậm (quarterly), testing kỹ lưỡng. Plugin cần release nhanh (weekly) để đáp ứng thị trường. Nếu tất cả trong cùng codebase, hoặc core bị chậm theo plugin, hoặc plugin bị chậm theo core.

Với microkernel, core và plugin có release cycle riêng:
- Core: Release 2.0 → API ổn định 6 tháng
- Plugin A: Release 1.5 mỗi tuần
- Plugin B: Release 3.2 mỗi tháng

### Vấn đề 4: Customization cho từng khách hàng

Doanh nghiệp lớn (enterprise) thường yêu cầu customization sâu. Nếu core hỗ trợ mọi customization, core sẽ phức tạp và chậm. Nếu từ chối, mất khách hàng.

Microkernel cho phép:
- **Core** — generic, reusable, cho mọi khách hàng
- **Enterprise plugins** — custom cho từng khách hàng
- **Marketplace** — ecosystem plugin cho tất cả

## Nguyên lý thiết kế

### 1. Core System (Microkernel)

Core chỉ chứa:
- **Extension points** — interface/abstract class cho plugin
- **Plugin registry** — quản lý lifecycle (load, init, start, stop)
- **Essential services** — logging, configuration, security, event bus
- **Plugin communication** — cách các plugin giao tiếp với nhau và với core

Core không chứa business logic cụ thể — chỉ chứa infrastructure để plugin hoạt động.

### 2. Extension Points — SPI (Service Provider Interface)

Mỗi extension point là một abstract class hoặc protocol:
- **Stable contract**: Interface ít thay đổi (backward compatible)
- **Versioned**: Extension point có version (v1, v2)
- **Documented**: Mỗi method có spec rõ ràng

Ví dụ extension points:
- `PaymentPlugin`: `process_payment(order, amount) → PaymentResult`
- `ShippingPlugin`: `calculate_shipping(order) → float`
- `TaxPlugin`: `calculate_tax(order) → float`
- `NotificationPlugin`: `send_notification(user, message)`
- `ReportPlugin`: `generate_report(start, end) → Report`

### 3. Plugin Lifecycle

Mọi plugin đều trải qua các phase:

```
DISCOVERED → LOADED → INITIALIZED → STARTED → STOPPED → UNLOADED
```

| Phase | Mô tả | Error Handling |
|-------|-------|---------------|
| **DISCOVERED** | Core tìm thấy plugin (filesystem, database) | Bỏ qua nếu lỗi discovery |
| **LOADED** | Plugin binary/code được load vào memory | Log error, continue |
| **INITIALIZED** | Plugin khởi tạo internal state | Plugin bị disable |
| **STARTED** | Plugin sẵn sàng xử lý request | Retry với backoff |
| **STOPPED** | Plugin tạm dừng (maintenance) | Graceful shutdown |
| **UNLOADED** | Plugin bị gỡ | Cleanup resources |

### 4. Isolation

Plugin phải được cách ly khỏi core và plugin khác:
- **ClassLoader isolation** (Java) — mỗi plugin có classloader riêng
- **Process isolation** — mỗi plugin chạy process riêng (microservices)
- **Thread isolation** — plugin chạy trong thread pool riêng
- **Namespace isolation** — plugin không thể access internal core API

### 5. Plugin Communication

Plugin không giao tiếp trực tiếp:
- **Event bus**: Core publish event, plugin subscribe
- **Service registry**: Plugin A gọi service của Plugin B qua core
- **Data pipeline**: Output của plugin này là input của plugin kia

### 6. Versioning và Dependency

- **Plugin dependency**: Plugin A cần Plugin B version >= 2.0
- **API versioning**: Extension point có version, plugin chỉ rõ API version nó dùng
- **Semantic versioning**: Core dùng SemVer cho API

## Cấu trúc chi tiết

### Core Components

| Component | Responsibility |
|-----------|---------------|
| **PluginRegistry** | Quản lý danh sách plugin đã discover |
| **PluginLoader** | Load plugin từ filesystem / database |
| **PluginManager** | Lifecycle management (init, start, stop) |
| **ExtensionPointRegistry** | Đăng ký extension points |
| **EventBus** | Plugin giao tiếp qua event |
| **ConfigurationService** | Cấu hình cho core và plugin |
| **SecurityManager** | Kiểm tra quyền của plugin |
| **DependencyResolver** | Giải quyết dependency giữa các plugin |

### Plugin Structure

Mỗi plugin có:
```
plugin-name/
├── manifest.yaml          # Metadata: name, version, author, dependencies
├── plugin.py              # Plugin class (implement Plugin interface)
├── requirements.txt       # Dependencies
├── static/                # Static assets (templates, images)
└── config/                # Default configuration
```

### Extension Points Example (E-Commerce Platform)

| Extension Point | Interface | Example Plugins |
|----------------|-----------|-----------------|
| `PaymentPlugin` | `process(order) → TransactionResult` | VNPay, MoMo, Stripe, PayPal |
| `ShippingPlugin` | `calculate(order) → ShippingQuote` | GiaoHangNhanh, GHTK, ViettelPost |
| `TaxPlugin` | `calculate(order) → TaxResult` | VAT VN, GST SG, TaxJar |
| `NotificationPlugin` | `notify(event) → None` | Email, SMS, Slack, Zalo |
| `DiscountPlugin` | `apply(order) → DiscountResult` | FlashSale, Bundle, Loyalty |
| `ReportPlugin` | `generate(from, to) → Report` | Sales, Tax, Inventory |
| `AuthPlugin` | `authenticate(credentials) → User` | OAuth2, SSO, LDAP, OTP |
| `ExportPlugin` | `export(data, format) → bytes` | PDF, Excel, CSV, JSON |

## Sơ đồ kiến trúc (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     MICROKERNEL / PLUGIN ARCHITECTURE                     │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  ┌─ ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ┐  │
│  │   PLUGIN 1        PLUGIN 2        PLUGIN 3        PLUGIN N     │  │
│  │  │ [Payment]       [Shipping]      [Tax]           [Export]    │  │
│  │   ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐   │  │
│  │  │ │Plugin A││     │Plugin B││     │Plugin C││     │Plugin N││  │  │
│  │   │manifest│      │manifest│      │manifest│      │manifest│   │  │
│  │  │ │ & code ││    │ │ & code ││    │ │ & code ││    │ │ & code ││  │  │
│  │   └────────┘      └────────┘      └────────┘      └────────┘   │  │
│  │  └─ ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ┘  │  │
│  │           │              │              │              │          │  │
│  │           └──────────────┼──────────────┼──────────────┘          │  │
│  │                          │              │                          │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                    MICROKERNEL (CORE)                       │  │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │  │
│  │  │  │ Plugin   │  │ Plugin   │  │ Event    │  │ Config   │   │  │  │
│  │  │  │ Registry │  │ Manager  │  │ Bus      │  │ Service  │   │  │  │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │  │
│  │  │  │ Security │  │ Extension│  │ Dependency│  │ Plugin   │   │  │  │
│  │  │  │ Manager  │  │ Points   │  │ Resolver  │  │ Loader   │   │  │  │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │               APPLICATION DOMAIN                             │  │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │  │
│  │  │  │  Order   │  │ Product  │  │  User    │  │  Cart    │   │  │  │
│  │  │  │  Service │  │  Service │  │  Service │  │  Service │   │  │  │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                     EXTERNAL SYSTEMS                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │
│  │  │ Database │  │  Queue   │  │  Cache   │  │  Search  │         │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Ví dụ code hoàn chỉnh

### Cách làm sai: Monolithic với tất cả payment methods

```python
from __future__ import annotations
from typing import Any


class MonolithicPaymentProcessor:
    """Monolith: mỗi lần thêm payment method phải sửa class này."""

    def process_payment(self, method: str, order: dict[str, Any], amount: float) -> dict[str, Any]:
        if method == "vnpay":
            return {"status": "ok", "txn_id": "VNP123"}
        elif method == "momo":
            return {"status": "ok", "txn_id": "MOMO456"}
        elif method == "stripe":
            return {"status": "ok", "txn_id": "STR789"}
        elif method == "paypal":
            return {"status": "ok", "txn_id": "PYP789"}
        # Thêm method mới → phải sửa file này, test lại toàn bộ
        raise ValueError(f"Unknown payment method: {method}")
```

### Cách làm đúng: Microkernel Architecture

```python
from __future__ import annotations
import os
import json
import time
import threading
import logging
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
from enum import Enum, auto
from abc import ABC, abstractmethod
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# 1. ENUMS & DOMAIN TYPES
# ======================================================================

class PluginState(Enum):
    DISCOVERED = auto()
    LOADED = auto()
    INITIALIZED = auto()
    STARTED = auto()
    STOPPED = auto()
    ERROR = auto()


class EventPriority(Enum):
    LOW = 0
    NORMAL = 50
    HIGH = 100


@dataclass
class PluginManifest:
    """Plugin metadata — thông tin từ manifest.json/yaml."""
    id: str
    name: str
    version: str
    author: str = ""
    description: str = ""
    entry_point: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)  # plugin_id → version
    min_core_version: str = "1.0.0"
    extension_points: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """Event cho plugin communication."""
    type: str
    data: dict[str, Any]
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL


@dataclass
class Order:
    """Domain model — đơn hàng."""
    order_id: str
    user_id: str
    items: list[dict[str, Any]]
    subtotal: float
    shipping_fee: float = 0.0
    tax: float = 0.0
    discount: float = 0.0
    total: float = 0.0
    payment_method: str = ""
    shipping_method: str = ""
    status: str = "pending"

    def calculate_total(self) -> float:
        self.total = self.subtotal + self.shipping_fee + self.tax - self.discount
        return self.total


# ======================================================================
# 2. EXTENSION POINTS (SPI)
# ======================================================================

class PaymentPlugin(ABC):
    """Extension point cho payment processing."""

    @abstractmethod
    def get_payment_method(self) -> str:
        ...

    @abstractmethod
    def process_payment(self, order: Order, amount: float) -> dict[str, Any]:
        ...

    @abstractmethod
    def refund(self, transaction_id: str, amount: float) -> dict[str, Any]:
        ...

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate configuration — override nếu cần."""
        return []


class ShippingPlugin(ABC):
    """Extension point cho shipping calculation."""

    @abstractmethod
    def get_shipping_method(self) -> str:
        ...

    @abstractmethod
    def calculate_shipping(self, order: Order) -> float:
        ...

    @abstractmethod
    def create_shipment(self, order: Order) -> dict[str, Any]:
        ...


class TaxPlugin(ABC):
    """Extension point cho tax calculation."""

    @abstractmethod
    def get_tax_name(self) -> str:
        ...

    @abstractmethod
    def calculate_tax(self, order: Order) -> float:
        ...


class NotificationPlugin(ABC):
    """Extension point cho notification."""

    @abstractmethod
    def get_channel_name(self) -> str:
        ...

    @abstractmethod
    def send(self, recipient: str, subject: str, message: str) -> bool:
        ...


class ExportPlugin(ABC):
    """Extension point cho data export."""

    @abstractmethod
    def get_format_name(self) -> str:
        ...

    @abstractmethod
    def export(self, data: list[dict[str, Any]], options: dict[str, Any] = None) -> bytes:
        ...


# ======================================================================
# 3. PLUGIN SYSTEM — CORE MICROKERNEL
# ======================================================================

class PluginWrapper:
    """Wrapper cho mỗi plugin instance — quản lý lifecycle."""

    def __init__(self, manifest: PluginManifest, plugin_instance: Any) -> None:
        self.manifest = manifest
        self.instance = plugin_instance
        self.state = PluginState.DISCOVERED
        self.config: dict[str, Any] = {}
        self.error: str | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Plugin({self.manifest.id}@{self.manifest.version}, {self.state.name})"


class ExtensionPointRegistry:
    """Đăng ký và tra cứu extension points."""

    def __init__(self) -> None:
        self._extensions: dict[type, dict[str, tuple[str, Any]]] = {}  # SPI → {method_name: (plugin_id, instance)}

    def register(self, spi_type: type, plugin_id: str, instance: Any) -> None:
        if spi_type not in self._extensions:
            self._extensions[spi_type] = {}
        # Register all methods in the SPI
        for name, method in inspect.getmembers(spi_type, predicate=inspect.isfunction):
            if not name.startswith("_"):
                self._extensions[spi_type][name] = (plugin_id, instance)

    def unregister(self, plugin_id: str) -> None:
        for spi_type in list(self._extensions.keys()):
            to_remove = [k for k, v in self._extensions[spi_type].items() if v[0] == plugin_id]
            for k in to_remove:
                del self._extensions[spi_type][k]
            if not self._extensions[spi_type]:
                del self._extensions[spi_type]

    def get_plugins_for_spi(self, spi_type: type) -> list[tuple[str, Any]]:
        """Get all plugin instances implementing an SPI."""
        plugins: dict[str, Any] = {}
        for method_name, (plugin_id, instance) in self._extensions.get(spi_type, {}).items():
            plugins[plugin_id] = instance
        return list(plugins.items())

    def get_plugin_method(self, spi_type: type, plugin_id: str, method: str) -> Callable | None:
        entry = self._extensions.get(spi_type, {}).get(method)
        if entry and entry[0] == plugin_id:
            return getattr(entry[1], method, None)
        return None


class EventBus:
    """Plugin communication — publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [c for c in self._subscribers[event_type] if c != callback]

    def publish(self, event: Event) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(event.type, []))
            wildcard = list(self._subscribers.get("*", []))
        all_callbacks = callbacks + wildcard
        all_callbacks.sort(key=lambda c: getattr(c, "_priority", EventPriority.NORMAL.value), reverse=True)

        for callback in all_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Event handler error for %s: %s", event.type, e)

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()


class PluginLoader:
    """Load plugin từ filesystem."""

    def __init__(self, plugin_dirs: list[str]) -> None:
        self._plugin_dirs = plugin_dirs
        self._loaded_modules: dict[str, Any] = {}

    def discover_manifests(self) -> list[PluginManifest]:
        manifests: list[PluginManifest] = []
        for plugin_dir in self._plugin_dirs:
            base_path = Path(plugin_dir)
            if not base_path.exists():
                logger.warning("Plugin directory not found: %s", plugin_dir)
                continue
            for item in base_path.iterdir():
                if item.is_dir():
                    manifest = self._load_manifest(item)
                    if manifest:
                        manifests.append(manifest)
        return manifests

    def _load_manifest(self, plugin_path: Path) -> PluginManifest | None:
        manifest_file = plugin_path / "manifest.json"
        if not manifest_file.exists():
            manifest_file = plugin_path / "manifest.yaml"
        if not manifest_file.exists():
            return None
        try:
            with open(manifest_file, encoding="utf-8") as f:
                if manifest_file.suffix == ".json":
                    data = json.load(f)
                else:
                    import yaml
                    data = yaml.safe_load(f)
            return PluginManifest(
                id=data.get("id", plugin_path.name),
                name=data.get("name", plugin_path.name),
                version=data.get("version", "0.0.1"),
                author=data.get("author", ""),
                description=data.get("description", ""),
                entry_point=data.get("entry_point", f"{plugin_path.name}.plugin"),
                dependencies=data.get("dependencies", {}),
                min_core_version=data.get("min_core_version", "1.0.0"),
                extension_points=data.get("extension_points", []),
                config_schema=data.get("config_schema", {}),
            )
        except Exception as e:
            logger.error("Failed to load manifest from %s: %s", plugin_path, e)
            return None

    def load_plugin(self, manifest: PluginManifest, plugin_path: Path) -> Any | None:
        try:
            module_path = manifest.entry_point.rsplit(".", 1)
            if len(module_path) != 2:
                raise ValueError(f"Invalid entry_point: {manifest.entry_point}")
            module_name, class_name = module_path

            spec = importlib.util.spec_from_file_location(
                module_name,
                plugin_path / f"{module_name}.py",
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {plugin_path / module_name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._loaded_modules[manifest.id] = module

            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class()
            return plugin_instance
        except Exception as e:
            logger.error("Failed to load plugin %s: %s", manifest.id, e)
            return None


class PluginManager:
    """Quản lý lifecycle của tất cả plugins."""

    def __init__(self, loader: PluginLoader, extension_registry: ExtensionPointRegistry, event_bus: EventBus) -> None:
        self._loader = loader
        self._extension_registry = extension_registry
        self._event_bus = event_bus
        self._plugins: dict[str, PluginWrapper] = {}
        self._lock = threading.RLock()
        self._startup_callbacks: list[Callable[[PluginWrapper], None]] = []
        self._shutdown_callbacks: list[Callable[[PluginWrapper], None]] = []

    def discover_and_load(self) -> list[PluginWrapper]:
        """Discover manifests → load plugin instances → return wrappers."""
        manifests = self._loader.discover_manifests()
        loaded: list[PluginWrapper] = []
        for manifest in manifests:
            plugin_dir = self._find_plugin_dir(manifest.id)
            if plugin_dir is None:
                logger.warning("Plugin %s not found on disk", manifest.id)
                continue
            instance = self._loader.load_plugin(manifest, plugin_dir)
            if instance:
                wrapper = PluginWrapper(manifest, instance)
                wrapper.state = PluginState.LOADED
                self._plugins[manifest.id] = wrapper
                loaded.append(wrapper)
                logger.info("Loaded plugin: %s v%s", manifest.name, manifest.version)
            else:
                wrapper = PluginWrapper(manifest, None)
                wrapper.state = PluginState.ERROR
                wrapper.error = "Failed to load"
                self._plugins[manifest.id] = wrapper
        return loaded

    def initialize_all(self) -> None:
        """Gọi init cho tất cả plugins."""
        for plugin_id, wrapper in self._plugins.items():
            if wrapper.state == PluginState.LOADED and wrapper.instance:
                try:
                    if hasattr(wrapper.instance, "initialize"):
                        wrapper.instance.initialize(self._extension_registry)
                    wrapper.state = PluginState.INITIALIZED
                    logger.info("Initialized plugin: %s", plugin_id)
                except Exception as e:
                    wrapper.state = PluginState.ERROR
                    wrapper.error = str(e)
                    logger.error("Failed to init plugin %s: %s", plugin_id, e)

    def start_all(self) -> None:
        """Start tất cả plugins."""
        for plugin_id, wrapper in self._plugins.items():
            if wrapper.state == PluginState.INITIALIZED and wrapper.instance:
                try:
                    if hasattr(wrapper.instance, "start"):
                        wrapper.instance.start(self._event_bus)
                    wrapper.state = PluginState.STARTED
                    logger.info("Started plugin: %s", plugin_id)
                    for cb in self._startup_callbacks:
                        cb(wrapper)
                except Exception as e:
                    wrapper.state = PluginState.ERROR
                    wrapper.error = str(e)
                    logger.error("Failed to start plugin %s: %s", plugin_id, e)

    def stop_all(self) -> None:
        """Stop tất cả plugins."""
        for plugin_id, wrapper in list(self._plugins.items()):
            if wrapper.state == PluginState.STARTED and wrapper.instance:
                try:
                    if hasattr(wrapper.instance, "stop"):
                        wrapper.instance.stop()
                    wrapper.state = PluginState.STOPPED
                    for cb in self._shutdown_callbacks:
                        cb(wrapper)
                    logger.info("Stopped plugin: %s", plugin_id)
                except Exception as e:
                    logger.error("Failed to stop plugin %s: %s", plugin_id, e)

    def get_plugin(self, plugin_id: str) -> PluginWrapper | None:
        return self._plugins.get(plugin_id)

    def get_plugins_by_state(self, state: PluginState) -> list[PluginWrapper]:
        return [p for p in self._plugins.values() if p.state == state]

    def get_all_plugins(self) -> list[PluginWrapper]:
        return list(self._plugins.values())

    def on_startup(self, callback: Callable[[PluginWrapper], None]) -> None:
        self._startup_callbacks.append(callback)

    def on_shutdown(self, callback: Callable[[PluginWrapper], None]) -> None:
        self._shutdown_callbacks.append(callback)

    def _find_plugin_dir(self, plugin_id: str) -> Path | None:
        for plugin_dir in self._loader._plugin_dirs:
            path = Path(plugin_dir) / plugin_id
            if path.exists():
                return path
        return None


# ======================================================================
# 4. MICROKERNEL (CORE)
# ======================================================================

class Microkernel:
    """Core system — tích hợp tất cả components."""

    def __init__(self, plugin_dirs: list[str] | None = None) -> None:
        self.plugin_dirs = plugin_dirs or ["plugins"]
        self.event_bus = EventBus()
        self.extension_registry = ExtensionPointRegistry()
        self.plugin_loader = PluginLoader(self.plugin_dirs)
        self.plugin_manager = PluginManager(self.plugin_loader, self.extension_registry, self.event_bus)
        self._running = False

    def initialize(self) -> None:
        """Khởi động core: discover → load → init → start plugins."""
        logger.info("Microkernel initializing...")
        self.plugin_manager.discover_and_load()
        self.plugin_manager.initialize_all()
        self.plugin_manager.start_all()

        # In-memory plugin registration fallback (for demo)
        self._register_demo_plugins()

        self._running = True
        logger.info("Microkernel initialized with %d plugins", len(self.plugin_manager.get_all_plugins()))

    def _register_demo_plugins(self) -> None:
        """Đăng ký demo plugins (trong trường hợp không có thư mục plugins)."""
        if not self.plugin_manager.get_plugins_by_state(PluginState.STARTED):
            logger.info("No external plugins found. Registering built-in demo plugins.")
            self.plugin_manager._plugins = {}
            self.extension_registry = ExtensionPointRegistry()

            # Register VNPay Payment
            vnpay = VNPayPlugin()
            vnpay_wrapper = self._demo_register("vnpay", "VNPay Payment", "1.0.0", vnpay)

            # Register Momo Payment
            momo = MoMoPaymentPlugin()
            self._demo_register("momo", "MoMo Payment", "1.2.0", momo)

            # Register GiaoHangNhanh Shipping
            ghn = GiaoHangNhanhShippingPlugin()
            self._demo_register("ghn", "GiaoHangNhanh", "2.0.1", ghn)

            # Register VAT Tax
            vat = VATTaxPlugin()
            self._demo_register("vat_vn", "VAT Vietnam", "1.0.0", vat)

            # Register Email Notification
            email = EmailNotificationPlugin()
            self._demo_register("email_notif", "Email Notifications", "1.1.0", email)

            # Register SMS Notification
            sms = SMSNotificationPlugin()
            self._demo_register("sms_notif", "SMS Notifications", "1.0.0", sms)

            # Register Excel Export
            excel = ExcelExportPlugin()
            self._demo_register("excel_export", "Excel Export", "1.0.0", excel)

    def _demo_register(self, plugin_id: str, name: str, version: str, instance: Any) -> PluginWrapper:
        manifest = PluginManifest(
            id=plugin_id, name=name, version=version,
            entry_point="demo", author="Built-in",
        )
        wrapper = PluginWrapper(manifest, instance)
        wrapper.state = PluginState.INITIALIZED

        # Đăng ký extension points
        if isinstance(instance, PaymentPlugin):
            self.extension_registry.register(PaymentPlugin, plugin_id, instance)
        if isinstance(instance, ShippingPlugin):
            self.extension_registry.register(ShippingPlugin, plugin_id, instance)
        if isinstance(instance, TaxPlugin):
            self.extension_registry.register(TaxPlugin, plugin_id, instance)
        if isinstance(instance, NotificationPlugin):
            self.extension_registry.register(NotificationPlugin, plugin_id, instance)
        if isinstance(instance, ExportPlugin):
            self.extension_registry.register(ExportPlugin, plugin_id, instance)

        # Đăng ký event handlers
        if hasattr(instance, "handle_event"):
            self.event_bus.subscribe("*", instance.handle_event)

        wrapper.state = PluginState.STARTED
        self.plugin_manager._plugins[plugin_id] = wrapper
        logger.info("Registered demo plugin: %s v%s", name, version)
        return wrapper

    def shutdown(self) -> None:
        logger.info("Microkernel shutting down...")
        self.plugin_manager.stop_all()
        self.event_bus.clear()
        self._running = False
        logger.info("Microkernel shut down")

    def get_plugin(self, plugin_id: str) -> PluginWrapper | None:
        return self.plugin_manager.get_plugin(plugin_id)

    def get_payment_plugin(self, method: str) -> PaymentPlugin | None:
        for plugin_id, instance in self.extension_registry.get_plugins_for_spi(PaymentPlugin):
            if instance.get_payment_method() == method:
                return instance
        return None

    def get_shipping_plugin(self, method: str) -> ShippingPlugin | None:
        for plugin_id, instance in self.extension_registry.get_plugins_for_spi(ShippingPlugin):
            if instance.get_shipping_method() == method:
                return instance
        return None

    def publish_event(self, event_type: str, data: dict[str, Any], source: str = "core") -> None:
        event = Event(type=event_type, data=data, source=source)
        self.event_bus.publish(event)


# ======================================================================
# 5. DEMO PLUGINS
# ======================================================================

class VNPayPlugin(PaymentPlugin):
    def get_payment_method(self) -> str:
        return "vnpay"

    def process_payment(self, order: Order, amount: float) -> dict[str, Any]:
        logger.info("VNPay processing payment for order %s: %.0f VND", order.order_id, amount)
        return {
            "status": "success",
            "transaction_id": f"VNP{int(time.time())}",
            "amount": amount,
            "method": "vnpay",
            "message": "Payment processed via VNPay",
        }

    def refund(self, transaction_id: str, amount: float) -> dict[str, Any]:
        return {"status": "success", "transaction_id": transaction_id, "refund_amount": amount}


class MoMoPaymentPlugin(PaymentPlugin):
    def get_payment_method(self) -> str:
        return "momo"

    def process_payment(self, order: Order, amount: float) -> dict[str, Any]:
        logger.info("MoMo processing payment for order %s: %.0f VND", order.order_id, amount)
        return {
            "status": "success",
            "transaction_id": f"MOMO{int(time.time())}",
            "amount": amount,
            "method": "momo",
            "message": "Payment processed via MoMo",
        }

    def refund(self, transaction_id: str, amount: float) -> dict[str, Any]:
        return {"status": "success", "transaction_id": transaction_id, "refund_amount": amount}


class StripePaymentPlugin(PaymentPlugin):
    def get_payment_method(self) -> str:
        return "stripe"

    def process_payment(self, order: Order, amount: float) -> dict[str, Any]:
        logger.info("Stripe processing payment for order %s: %.2f USD", order.order_id, amount)
        return {
            "status": "success",
            "transaction_id": f"STR{int(time.time())}",
            "amount": amount,
            "method": "stripe",
            "message": "Payment processed via Stripe",
        }

    def refund(self, transaction_id: str, amount: float) -> dict[str, Any]:
        return {"status": "success", "transaction_id": transaction_id, "refund_amount": amount}


class GiaoHangNhanhShippingPlugin(ShippingPlugin):
    def get_shipping_method(self) -> str:
        return "ghn"

    def calculate_shipping(self, order: Order) -> float:
        base_rate = 20000  # 20k VND base
        weight_charge = len(order.items) * 5000  # 5k per item
        total = base_rate + weight_charge
        logger.info("GHN shipping for %s: %.0f VND", order.order_id, total)
        return total

    def create_shipment(self, order: Order) -> dict[str, Any]:
        return {
            "status": "created",
            "tracking_id": f"GHN{int(time.time())}",
            "estimated_delivery": "2-3 days",
        }


class ViettelPostShippingPlugin(ShippingPlugin):
    def get_shipping_method(self) -> str:
        return "viettelpost"

    def calculate_shipping(self, order: Order) -> float:
        base_rate = 15000
        weight_charge = len(order.items) * 4000
        total = base_rate + weight_charge
        logger.info("ViettelPost shipping for %s: %.0f VND", order.order_id, total)
        return total

    def create_shipment(self, order: Order) -> dict[str, Any]:
        return {
            "status": "created",
            "tracking_id": f"VTP{int(time.time())}",
            "estimated_delivery": "3-5 days",
        }


class VATTaxPlugin(TaxPlugin):
    def get_tax_name(self) -> str:
        return "vat_vn"

    def calculate_tax(self, order: Order) -> float:
        vat_rate = 0.10  # 10% VAT
        tax = order.subtotal * vat_rate
        logger.info("VAT tax for %s: %.0f VND (10%%)", order.order_id, tax)
        return tax


class EmailNotificationPlugin(NotificationPlugin):
    def get_channel_name(self) -> str:
        return "email"

    def send(self, recipient: str, subject: str, message: str) -> bool:
        logger.info("[EMAIL] To: %s | Subject: %s | Body: %s...", recipient, subject, message[:50])
        return True

    def handle_event(self, event: Event) -> None:
        if event.type in ("order.created", "payment.received", "order.shipped"):
            order_data = event.data.get("order", {})
            user_email = event.data.get("user_email", "unknown@example.com")
            self.send(
                recipient=user_email,
                subject=f"Order {order_data.get('order_id', '')} - {event.type}",
                message=f"Your order status has been updated to {event.type}.",
            )


class SMSNotificationPlugin(NotificationPlugin):
    def get_channel_name(self) -> str:
        return "sms"

    def send(self, recipient: str, subject: str, message: str) -> bool:
        logger.info("[SMS] To: %s | Message: %s", recipient, message[:50])
        return True


class ExcelExportPlugin(ExportPlugin):
    def get_format_name(self) -> str:
        return "excel"

    def export(self, data: list[dict[str, Any]], options: dict[str, Any] = None) -> bytes:
        logger.info("Exporting %d records to Excel format", len(data))
        import io
        output = io.BytesIO()
        output.write(f"Exported {len(data)} records\n".encode())
        for row in data:
            output.write(f"{json.dumps(row, ensure_ascii=False)}\n".encode())
        return output.getvalue()


# ======================================================================
# 6. ORDER SERVICE — CORE BUSINESS LOGIC
# ======================================================================

class OrderService:
    """Core business logic — dùng plugin qua extension points."""

    def __init__(self, kernel: Microkernel) -> None:
        self._kernel = kernel

    def create_order(self, user_id: str, items: list[dict[str, Any]], payment_method: str,
                     shipping_method: str, user_email: str = "") -> Order:
        """Tạo đơn hàng — gọi plugin để tính shipping, tax, payment."""
        order = Order(
            order_id=f"ORD{int(time.time())}_{user_id[:4]}",
            user_id=user_id,
            items=items,
            subtotal=sum(item["price"] * item.get("quantity", 1) for item in items),
        )
        logger.info("=" * 60)
        logger.info("Creating order %s for user %s", order.order_id, user_id)
        logger.info("Subtotal: %.0f VND", order.subtotal)

        # Tính shipping qua plugin
        shipping_plugin = self._kernel.get_shipping_plugin(shipping_method)
        if shipping_plugin:
            order.shipping_fee = shipping_plugin.calculate_shipping(order)
            order.shipping_method = shipping_method
            logger.info("Shipping (%s): %.0f VND", shipping_method, order.shipping_fee)
        else:
            logger.warning("No shipping plugin found for: %s", shipping_method)

        # Tính thuế qua plugin (lấy plugin tax đầu tiên)
        tax_plugins = self._kernel.extension_registry.get_plugins_for_spi(TaxPlugin)
        if tax_plugins:
            _, tax_plugin = tax_plugins[0]
            order.tax = tax_plugin.calculate_tax(order)
            logger.info("Tax: %.0f VND", order.tax)

        # Tính total
        order.calculate_total()
        logger.info("Total: %.0f VND", order.total)

        # Xử lý payment qua plugin
        payment_plugin = self._kernel.get_payment_plugin(payment_method)
        if payment_plugin:
            result = payment_plugin.process_payment(order, order.total)
            order.payment_method = payment_method
            order.status = "paid" if result["status"] == "success" else "payment_failed"
            logger.info("Payment (%s): %s (txn=%s)", payment_method, result["status"], result.get("transaction_id", ""))
        else:
            logger.warning("No payment plugin found for: %s", payment_method)

        # Publish event
        self._kernel.publish_event("order.created", {
            "order": {
                "order_id": order.order_id,
                "total": order.total,
                "status": order.status,
            },
            "user_email": user_email,
        })

        return order

    def get_available_payment_methods(self) -> list[str]:
        return [instance.get_payment_method()
                for _, instance in self._kernel.extension_registry.get_plugins_for_spi(PaymentPlugin)]

    def get_available_shipping_methods(self) -> list[str]:
        return [instance.get_shipping_method()
                for _, instance in self._kernel.extension_registry.get_plugins_for_spi(ShippingPlugin)]


# ======================================================================
# 7. PLUGIN SYSTEM ADMIN
# ======================================================================

class PluginAdmin:
    """Admin utilities cho plugin management."""

    def __init__(self, kernel: Microkernel) -> None:
        self._kernel = kernel

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {
                "id": p.manifest.id,
                "name": p.manifest.name,
                "version": p.manifest.version,
                "state": p.state.name,
                "error": p.error,
            }
            for p in self._kernel.plugin_manager.get_all_plugins()
        ]

    def disable_plugin(self, plugin_id: str) -> bool:
        wrapper = self._kernel.plugin_manager.get_plugin(plugin_id)
        if wrapper and wrapper.state == PluginState.STARTED:
            self._kernel.plugin_manager._extension_registry.unregister(plugin_id)
            wrapper.state = PluginState.STOPPED
            logger.info("Plugin disabled: %s", plugin_id)
            return True
        return False

    def enable_plugin(self, plugin_id: str) -> bool:
        wrapper = self._kernel.plugin_manager.get_plugin(plugin_id)
        if wrapper and wrapper.state == PluginState.STOPPED:
            if hasattr(wrapper.instance, "start"):
                wrapper.instance.start(self._kernel.event_bus)
            wrapper.state = PluginState.STARTED
            # Re-register extension points
            for spi_type in [PaymentPlugin, ShippingPlugin, TaxPlugin, NotificationPlugin, ExportPlugin]:
                if isinstance(wrapper.instance, spi_type):
                    self._kernel.extension_registry.register(spi_type, plugin_id, wrapper.instance)
            logger.info("Plugin enabled: %s", plugin_id)
            return True
        return False

    def reload_plugin(self, plugin_id: str) -> bool:
        """Reload plugin (simulate)."""
        self.disable_plugin(plugin_id)
        self.enable_plugin(plugin_id)
        return True


# ======================================================================
# 8. MAIN — SIMULATION
# ======================================================================

def main() -> None:
    logger.info("=== Microkernel Architecture: E-Commerce Platform ===")

    # Initialize microkernel
    kernel = Microkernel(plugin_dirs=["plugins"])
    kernel.initialize()

    # Create order service
    order_service = OrderService(kernel)
    admin = PluginAdmin(kernel)

    # Show available plugins
    logger.info("\n=== Available Plugins ===")
    for p in admin.list_plugins():
        logger.info("  %s v%s [%s]", p["name"], p["version"], p["state"])
    logger.info("\nPayment methods: %s", order_service.get_available_payment_methods())
    logger.info("Shipping methods: %s", order_service.get_available_shipping_methods())

    # Create orders với các plugin khác nhau
    logger.info("\n=== Creating Orders ===")

    order1 = order_service.create_order(
        user_id="user_001",
        items=[
            {"name": "iPhone 15 Pro", "price": 27990000, "quantity": 1},
            {"name": "AirPods Pro", "price": 5490000, "quantity": 2},
        ],
        payment_method="vnpay",
        shipping_method="ghn",
        user_email="alice@example.com",
    )

    order2 = order_service.create_order(
        user_id="user_002",
        items=[{"name": "MacBook Pro 16", "price": 59990000, "quantity": 1}],
        payment_method="momo",
        shipping_method="viettelpost",
        user_email="bob@example.com",
    )

    # Dynamic plugin registration
    logger.info("\n=== Dynamic Plugin: Adding Stripe ===")
    stripe = StripePaymentPlugin()
    kernel._demo_register("stripe_live", "Stripe Payment", "1.5.0", stripe)
    logger.info("New payment methods: %s", order_service.get_available_payment_methods())

    order3 = order_service.create_order(
        user_id="user_003",
        items=[{"name": "Domain License", "price": 999000, "quantity": 1}],
        payment_method="stripe",
        shipping_method="ghn",
        user_email="charlie@example.com",
    )

    # Disable a plugin
    logger.info("\n=== Disable VNPay Plugin ===")
    admin.disable_plugin("vnpay")
    logger.info("Payment methods after disable: %s", order_service.get_available_payment_methods())

    # Try to use disabled plugin
    order4 = order_service.create_order(
        user_id="user_004",
        items=[{"name": "Test Item", "price": 100000, "quantity": 1}],
        payment_method="vnpay",  # Should fail gracefully
        shipping_method="ghn",
    )

    # Re-enable
    logger.info("\n=== Re-enable VNPay ===")
    admin.enable_plugin("vnpay")
    logger.info("Payment methods: %s", order_service.get_available_payment_methods())

    # Export
    logger.info("\n=== Export Plugin ===")
    export_plugins = kernel.extension_registry.get_plugins_for_spi(ExportPlugin)
    if export_plugins:
        _, excel_plugin = export_plugins[0]
        export_data = [
            {"order_id": order1.order_id, "total": order1.total, "status": order1.status},
            {"order_id": order2.order_id, "total": order2.total, "status": order2.status},
        ]
        result = excel_plugin.export(export_data)
        logger.info("Exported %d bytes", len(result))

    # Summary
    logger.info("\n=== Plugin System Summary ===")
    for p in admin.list_plugins():
        logger.info("  [%s] %s v%s by %s", p["state"], p["name"], p["version"], "Built-in")

    logger.info("\n=== Microkernel Demo Complete ===")
    kernel.shutdown()


if __name__ == "__main__":
    main()
```

## Khi nào dùng / Khi nào không

| Khi nào dùng | Khi nào không |
|--------------|---------------|
| Hệ thống cần mở rộng bởi third-party | Số lượng tính năng nhỏ, cố định |
| Product cần customization cho từng khách hàng | Core system quá nhỏ, không đáng để tách plugin |
| Release cycle khác nhau giữa core và tính năng | Performance-critical (plugin overhead) |
| Cần ecosystem / marketplace | Đội ngũ nhỏ, không đủ resources cho plugin architecture |
| Nhiều integration với external systems | Tất cả integration đều giống nhau (cùng pattern) |
| Open source project với community contributions | Hệ thống real-time (microseconds latency) |

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Extensibility**: Thêm tính năng bằng plugin, không sửa core | **Complexity**: Plugin system complex — loading, isolation, versioning |
| **Isolation**: Plugin lỗi không ảnh hưởng core | **Performance**: Plugin communication overhead (IPC, serialization) |
| **Independent deployment**: Mỗi plugin release riêng | **Dependency hell**: Plugin A cần plugin B version X |
| **Ecosystem**: Third-party developers đóng góp | **API stability**: Extension point thay đổi → break plugins |
| **Customization**: Mỗi khách hàng có plugin riêng | **Testing complexity**: Cần test core + từng plugin + combinations |
| **Technology diversity**: Plugin viết bằng ngôn ngữ khác | **Distribution**: Cần plugin registry, update mechanism |
| **Gradual adoption**: Thêm plugin dần dần | **Security risks**: Plugin có thể chứa malicious code |

## Công cụ và Framework

| Tên | Loại | Ngôn ngữ | Ghi chú |
|-----|------|----------|---------|
| **OSGi (Equinox, Felix)** | Framework | Java | Enterprise plugin system chuẩn |
| **Java SPI (ServiceLoader)** | Built-in | Java | JDK built-in plugin mechanism |
| **Python setuptools entry_points** | Built-in | Python | Plugin discovery qua package metadata |
| **Stevedore** | Library | Python | Python plugin management |
| **Pluggy** | Library | Python | Plugin framework (dùng bởi pytest) |
| **Yapsy** | Library | Python | Yet Another Plugin System |
| **Go Plugin** | Built-in | Go | `plugin` package (Go 1.8+) |
| **.NET MEF (Managed Extensibility Framework)** | Built-in | .NET | Plugin framework cho .NET |
| **Webpack / Rollup** | Build tool | JS | Plugin system cho build pipeline |
| **WordPress Plugin API** | Platform | PHP | Lớn nhất: 60k+ plugins |

## Kiểm thử

Testing microkernel architecture gồm: (1) core testing, (2) plugin testing, (3) integration testing, (4) plugin isolation testing.

```python
from __future__ import annotations
import pytest
import tempfile
import json
from pathlib import Path
from typing import Any


class TestMicrokernel:
    def test_kernel_initialization(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        assert kernel._running is True
        assert len(kernel.plugin_manager.get_all_plugins()) > 0
        kernel.shutdown()

    def test_kernel_shutdown(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        kernel.shutdown()
        assert kernel._running is False
        assert all(p.state in (PluginState.STOPPED, PluginState.ERROR)
                   for p in kernel.plugin_manager.get_all_plugins())

    def test_plugin_registration(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        plugins = kernel.plugin_manager.get_all_plugins()
        assert any(p.manifest.id == "vnpay" for p in plugins)
        assert any(p.manifest.id == "momo" for p in plugins)
        kernel.shutdown()

    def test_extension_point_registry(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        payment_plugins = kernel.extension_registry.get_plugins_for_spi(PaymentPlugin)
        assert len(payment_plugins) >= 2
        kernel.shutdown()


class TestPluginManager:
    def test_discover_no_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loader = PluginLoader([tmp])
            manifests = loader.discover_manifests()
            assert len(manifests) == 0

    def test_discover_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_dir = Path(tmp) / "test_plugin"
            plugin_dir.mkdir()
            manifest = {
                "id": "test_plugin",
                "name": "Test Plugin",
                "version": "1.0.0",
                "entry_point": "test_plugin.plugin",
                "author": "Test",
            }
            (plugin_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
            (plugin_dir / "test_plugin.py").write_text("class plugin: pass")
            loader = PluginLoader([tmp])
            manifests = loader.discover_manifests()
            assert len(manifests) == 1
            assert manifests[0].id == "test_plugin"
            assert manifests[0].version == "1.0.0"

    def test_plugin_lifecycle(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()

        vnpay = kernel.plugin_manager.get_plugin("vnpay")
        assert vnpay is not None
        assert vnpay.state == PluginState.STARTED

        kernel.plugin_manager.stop_all()
        assert vnpay.state == PluginState.STOPPED

        kernel.shutdown()

    def test_plugin_error_handling(self) -> None:
        """Plugin lỗi không ảnh hưởng đến plugin khác."""
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        # Simulate error handling
        error_wrapper = PluginWrapper(
            PluginManifest(id="bad_plugin", name="Bad", version="0.1", entry_point="bad"),
            None,
        )
        error_wrapper.state = PluginState.ERROR
        kernel.plugin_manager._plugins["bad_plugin"] = error_wrapper

        assert len(kernel.plugin_manager.get_plugins_by_state(PluginState.ERROR)) == 1
        assert len(kernel.plugin_manager.get_plugins_by_state(PluginState.STARTED)) >= 6
        kernel.shutdown()


class TestExtensionPoints:
    def test_payment_plugins(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        vnpay = kernel.get_payment_plugin("vnpay")
        assert vnpay is not None
        assert vnpay.get_payment_method() == "vnpay"

        momo = kernel.get_payment_plugin("momo")
        assert momo is not None
        assert momo.get_payment_method() == "momo"

        unknown = kernel.get_payment_plugin("bitcoin")
        assert unknown is None

    def test_shipping_plugins(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        ghn = kernel.get_shipping_plugin("ghn")
        assert ghn is not None
        assert ghn.get_shipping_method() == "ghn"


class TestOrderService:
    def test_create_order_with_vnpay(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        service = OrderService(kernel)
        order = service.create_order(
            user_id="test_user",
            items=[{"name": "Item", "price": 100000, "quantity": 1}],
            payment_method="vnpay",
            shipping_method="ghn",
        )
        assert order.order_id.startswith("ORD")
        assert order.subtotal == 100000
        assert order.payment_method == "vnpay"
        assert order.shipping_method == "ghn"
        assert order.status == "paid"
        assert order.total > 0
        kernel.shutdown()

    def test_create_order_with_momo(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        service = OrderService(kernel)
        order = service.create_order(
            user_id="test_user_2",
            items=[{"name": "Laptop", "price": 15000000, "quantity": 1}],
            payment_method="momo",
            shipping_method="viettelpost",
        )
        assert order.payment_method == "momo"
        assert order.status == "paid"
        kernel.shutdown()

    def test_unknown_payment_method_graceful(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        service = OrderService(kernel)
        order = service.create_order(
            user_id="test_user_3",
            items=[{"name": "Test", "price": 50000, "quantity": 1}],
            payment_method="unknown_gateway",
            shipping_method="ghn",
        )
        # Order created, just no payment processed
        assert order.payment_method == ""
        kernel.shutdown()

    def test_available_methods(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        service = OrderService(kernel)
        payments = service.get_available_payment_methods()
        assert "vnpay" in payments
        assert "momo" in payments
        shipping = service.get_available_shipping_methods()
        assert "ghn" in shipping
        assert "viettelpost" in shipping
        kernel.shutdown()


class TestEventBus:
    def test_publish_subscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.event", handler)
        bus.publish(Event(type="test.event", data={"key": "value"}))
        assert len(received) == 1
        assert received[0].type == "test.event"
        assert received[0].data["key"] == "value"

    def test_wildcard_subscription(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("*", handler)
        bus.publish(Event(type="any.event", data={}))
        bus.publish(Event(type="another.event", data={}))
        assert len(received) == 2

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test", handler)
        bus.unsubscribe("test", handler)
        bus.publish(Event(type="test", data={}))
        assert len(received) == 0

    def test_event_to_plugins(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        kernel.publish_event("order.created", {
            "order": {"order_id": "TEST001"},
            "user_email": "test@example.com",
        })
        # Plugin should handle without error
        kernel.shutdown()


class TestPluginAdmin:
    def test_list_plugins(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        admin = PluginAdmin(kernel)
        plugins = admin.list_plugins()
        assert len(plugins) >= 6
        kernel.shutdown()

    def test_disable_enable_plugin(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        admin = PluginAdmin(kernel)
        result = admin.disable_plugin("vnpay")
        assert result is True
        vnpay = kernel.plugin_manager.get_plugin("vnpay")
        assert vnpay is not None
        assert vnpay.state == PluginState.STOPPED

        result = admin.enable_plugin("vnpay")
        assert result is True
        assert vnpay.state == PluginState.STARTED
        kernel.shutdown()

    def test_disable_unknown_plugin(self) -> None:
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        admin = PluginAdmin(kernel)
        result = admin.disable_plugin("non_existent")
        assert result is False
        kernel.shutdown()


class TestPluginIsolation:
    def test_plugin_error_does_not_affect_other_plugins(self) -> None:
        """One plugin's failure shouldn't crash other plugins."""
        kernel = Microkernel(plugin_dirs=[])
        kernel.initialize()
        # Force an error in one plugin's process_payment
        momo = kernel.get_payment_plugin("momo")
        if momo:
            original = momo.process_payment
            def broken(order: Order, amount: float) -> dict[str, Any]:
                raise RuntimeError("Simulated failure")
            momo.process_payment = broken

        # Other payment should still work
        service = OrderService(kernel)
        order = service.create_order(
            user_id="test_isolation",
            items=[{"name": "Test", "price": 100000, "quantity": 1}],
            payment_method="vnpay",  # This should still work
            shipping_method="ghn",
        )
        assert order.status == "paid"
        assert order.payment_method == "vnpay"
        kernel.shutdown()
```

## Kết luận

Microkernel (Plugin) Architecture là kiến trúc lý tưởng cho các sản phẩm cần mở rộng, tùy biến, và có ecosystem. Nó cho phép core system nhỏ gọn, ổn định, trong khi các tính năng mới được phát triển độc lập dưới dạng plugin.

**Best Practices:**
- **Thiết kế extension points cẩn thận**: API ổn định, backward compatible, versioned.
- **Plugin isolation**: Plugin không được truy cập internal core API. Dùng interface/abstract class rõ ràng.
- **Plugin discovery linh hoạt**: Filesystem scanning, database registry, or service discovery.
- **Dependency management**: Plugin manifest phải khai báo dependencies và version constraints.
- **Testing plugin isolation**: Plugin không được ảnh hưởng đến core hay plugin khác khi fail.
- **Security sandboxing**: Kiểm tra plugin signature, giới hạn resources, validate config.
- **Graceful degradation**: Core vẫn hoạt động khi plugin bị lỗi hoặc không có.

**Golden Rules:**
1. Core không bao giờ import plugin — plugin import core (IoC / Dependency Inversion).
2. Mỗi extension point là một interface nhỏ, focused (Interface Segregation).
3. Plugin manifest chứa tất cả metadata — core không cần load plugin để biết thông tin.
4. Plugin lifecycle phải rõ ràng: load → init → start → stop → unload.
5. Giữ API backward compatibility — breaking change = phiên bản major mới.
6. Plugin communication chỉ qua Event Bus — không gọi trực tiếp.
7. Luôn có cách disable plugin an toàn mà không restart core.
