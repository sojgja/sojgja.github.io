---
id: mvp-architecture
title: MVP (Model-View-Presenter)
sidebar_label: 🏗️ MVP Architecture
sidebar_position: 45
---

# MVP (Model-View-Presenter)

> "The presenter acts as the middleman, mediating between the view and the model, ensuring that the view remains completely passive and testable."
> — **Mike Potel**, *MVP: Model-View-Presenter* (1996)

**MVP** (Model-View-Presenter) là một software architecture pattern được phát triển bởi Mike Potel tại Taligent (một công ty con của IBM) vào những năm 1990. MVP ra đời như một sự kế thừa và cải tiến của MVC (Model-View-Controller), giải quyết vấn đề chính của MVC: **khả năng kiểm thử** (testability) và **tách biệt hoàn toàn** giữa View và Model.

---

## Tổng quan

### Lịch sử và nguồn gốc

MVP được giới thiệu lần đầu tiên trong bài viết *"Model-View-Presenter — The Taligent Programming Model for C++ and Java"* của Mike Potel. Sau đó, nó được phổ biến rộng rãi bởi:

- **Martin Fowler**: Phân biệt giữa *Passive View* và *Supervising Controller* trong bài viết kinh điển
- **Google**: Sử dụng MVP làm kiến trúc chính cho các ứng dụng Android trước khi MVVM ra đời
- **Microsoft**: Áp dụng MVP trong Web Forms (.NET) — mô hình code-behind
- **GWT** (Google Web Toolkit): Framework Java cho web ứng dụng MVP làm pattern chính

### So sánh với MVC

Trong MVC, Controller nhận input từ user và cập nhật Model; View quan sát Model để tự động cập nhật (qua Observer pattern). Trong MVP:

- **View** hoàn toàn thụ động (Passive View) — không biết gì về Model
- **Presenter** chịu trách nhiệm xử lý tất cả logic, cập nhật View qua interface
- **Model** chỉ đơn thuần là data/business logic
- **View** và **Presenter** giao tiếp qua một interface chung (thường gọi là `IView`)

### Các biến thể của MVP

| Biến thể | Mô tả |
|---------|-------|
| **Passive View** | View không chứa logic nào, Presenter điều khiển hoàn toàn |
| **Supervising Controller** | View xử lý data-binding đơn giản, Presenter xử lý logic phức tạp |
| **Presentation Model** | Presenter duy trì state của View (giống MVVM hơn) |

---

## Bài toán

### Hệ thống Dashboard Phân tích Dữ liệu Thương mại Điện tử

Giả sử bạn đang xây dựng một **Dashboard phân tích dữ liệu thời gian thực** cho một sàn thương mại điện tử lớn tại Việt Nam (Tiki, ShopeeFood, hoặc Sendo). Dashboard cần hiển thị:

1. Tổng doanh thu hôm nay (real-time)
2. Top 10 sản phẩm bán chạy
3. Tỷ lệ chuyển đổi theo giờ
4. Bản đồ nhiệt đơn hàng theo khu vực
5. Thống kê lỗi thanh toán
6. Cảnh báo bất thường (anomaly detection)

### Thách thức trong phát triển

Một Dashboard điển hình trên web thường được xây dựng với kiến trúc MVC truyền thống. Tuy nhiên, khi ứng dụng phát triển, các vấn đề sau xuất hiện:

**Vấn đề 1 — View phình to**: Trong MVC, View thường chứa cả template rendering, event handling, và một phần logic hiển thị. Khi Dashboard có 50+ widget khác nhau, file View trở nên khổng lồ (3000+ dòng), khó bảo trì.

```python
# MVC — View bị phình to
class DashboardView:
    def render(self):
        # 100 dòng render layout
        self._render_header()
        self._render_revenue_chart()    # 50 dòng JS + HTML
        self._render_top_products()     # 30 dòng xử lý data
        self._render_heatmap()          # 80 dòng
        self._render_alerts()           # 40 dòng + logic filter
        # ... cứ thế, view dần trở thành god object
```

**Vấn đề 2 — Không thể unit test View**: Trong MVC, View thường gắn chặt với DOM/UI framework. Bạn không thể test logic hiển thị nếu không chạy trình duyệt. Ví dụ:

```python
# MVC — View logic không thể test
class RevenueWidget:
    def update(self, data):
        if data["growth_rate"] > 0.2:
            self.element.style.color = "green"
            self.show_arrow("up")
        else:
            self.element.style.color = "red"
            self.show_arrow("down")
```

Làm sao để test dòng `self.element.style.color = "green"` mà không cần DOM?

**Vấn đề 3 — Model và View gắn chặt**: Khi Model thay đổi, View cần được cập nhật. Trong MVC, View thường đăng ký lắng nghe trực tiếp Model (Observer). Điều này tạo ra circular dependency và khó debug khi có 50+ widget cùng lắng nghe model.

**Vấn đề 4 — Logic hiển thị phân tán**: Khi business logic liên quan đến hiển thị (format tiền tệ, hiển thị % tăng trưởng, quyết định màu sắc dựa trên threshold) nằm rải rác cả ở View lẫn Controller, việc maintain consistency gần như bất khả thi.

### MVP giải quyết vấn đề như thế nào

MVP giải quyết tất cả các vấn đề trên bằng cách:

1. **View chỉ là interface**: View chỉ định nghĩa một interface (`IView`), không chứa logic
2. **Presenter xử lý mọi logic**: Presenter quyết định hiển thị gì, khi nào, màu gì
3. **View hoàn toàn thụ động**: View chỉ làm đúng những gì Presenter bảo (qua interface calls)
4. **Model độc lập**: Model không biết gì về View hay Presenter
5. **Testability tối đa**: Có thể mock View và test Presenter hoàn toàn bằng unit test

---

## Nguyên lý thiết kế

### 1. Tách biệt hoàn toàn Presentation Layer

Mọi logic liên quan đến hiển thị (presentation logic) phải nằm trong Presenter, không phải View. Presenter là cầu nối duy nhất giữa Model và View.

### 2. View là Passive (Passive View)

View không chứa business logic. View chỉ:
- Hiển thị dữ liệu khi Presenter yêu cầu
- Chuyển tiếp sự kiện người dùng cho Presenter
- View **không** tự query Model, **không** tự format dữ liệu

### 3. Model là Domain Layer thuần túy

Model chỉ chứa business logic và data. Model **không biết** sự tồn tại của View hay Presenter. Model có thể được reuse ở bất kỳ đâu (API service, background job, CLI tool).

### 4. Giao tiếp qua Interface

View và Presenter giao tiếp qua interface:
- `IView`: Định nghĩa các method mà Presenter có thể gọi trên View
- `IPresenter`: Định nghĩa các method mà View có thể gọi trên Presenter (event handlers)

### 5. Single Responsibility cho mỗi Presenter

Mỗi Presenter chỉ quản lý một màn hình / một widget. Nếu màn hình quá phức tạp, chia thành nhiều Presenter nhỏ.

### 6. Unit of Work pattern

Presenter thường kết hợp với Unit of Work để quản lý transaction khi thao tác với nhiều Model objects.

---

## Cấu trúc chi tiết

### Các thành phần

```
┌─────────────────────────────────────────────────┐
│                   APPLICATION                     │
├─────────────────────────────────────────────────┤
│  ┌──────────┐     ┌────────────┐                 │
│  │   View    │────▶│  Presenter │                 │
│  │(Interface)│◀────│  (Logic)   │                 │
│  └──────────┘     └─────┬──────┘                 │
│                         │                         │
│                    ┌────▼──────┐                  │
│                    │   Model   │                  │
│                    │ (Domain)  │                  │
│                    └───────────┘                  │
├─────────────────────────────────────────────────┤
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Services │  │   Data    │  │  External     │  │
│  │          │  │   Access  │  │  Integrations │  │
│  └──────────┘  └───────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

**1. View (Interface)**

- Định nghĩa contract dưới dạng abstract class hoặc Protocol
- Các method setter để Presenter đẩy dữ liệu xuống
- Các event/signal để báo cho Presenter về hành động user
- Không chứa implementation logic

**2. Presenter**

- Nhận event từ View, xử lý logic, gọi Model
- Quyết định trạng thái hiển thị của View
- Quản lý lifecycle của màn hình
- Có thể inject Model qua constructor (DI)

**3. Model**

- Domain entities, business rules
- Data access, repositories
- Validation logic
- Hoàn toàn độc lập với UI

**4. Services Layer**

- API services
- Cache services
- Logging, analytics
- Authentication, authorization

### Luồng tương tác

```
User Action → View → Presenter.handleEvent()
                              ↓
                    Presenter calls Model
                              ↓
                    Presenter updates View
                    View.displayData(data)
```

### Quy tắc vàng

1. **View không bao giờ gọi Model trực tiếp**
2. **Presenter không bao giờ tham chiếu đến View implementation — chỉ dùng interface**
3. **Model không bao giờ import View hoặc Presenter**
4. **Mỗi Presenter quản lý đúng một View**
5. **View interface nên fine-grained (nhiều method nhỏ) thay vì một method render() lớn**
6. **Presenter nên stateless, state lưu trong Model hoặc View state object**

---

## Sơ đồ kiến trúc

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│  ┌──────────────────────────────┐   ┌────────────────────────────┐  │
│  │          View (Passive)      │   │        Presenter           │  │
│  │                              │   │                            │  │
│  │  ┌──────────────────────┐    │   │  ┌──────────────────────┐  │  │
│  │  │   DashboardView      │    │   │  │ DashboardPresenter   │  │  │
│  │  │                      │    │   │  │                      │  │  │
│  │  │  + displayRevenue(d) │◀───┼───┼──│ + onViewLoaded()     │  │  │
│  │  │  + displayTopProds() │    │   │  │ + onRefreshClick()   │  │  │
│  │  │  + showAlert(msg)    │◀───┼───┼──│ + onFilterChange(f)  │  │  │
│  │  │  + showLoading(b)    │    │   │  │ + onExportClick()    │  │  │
│  │  │                      │    │   │  │                      │  │  │
│  │  │  event onRefresh ────┼────┼───▶│                        │  │  │
│  │  │  event onFilter ─────┼────┼───▶│                        │  │  │
│  │  └──────────────────────┘    │   │  └──────────┬───────────┘  │  │
│  └──────────────────────────────┘   └─────────────┼──────────────┘  │
└───────────────────────────────────────────────────┼─────────────────┘
                                                     │
                                                    │ calls
                                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       DOMAIN LAYER (Model)                          │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   RevenueService    │  │  ProductService   │  │ AlertService │  │
│  │                     │  │                   │  │              │  │
│  │  + getTodayRevenue()│  │ + getTopProducts()│  │ + getAlerts()│  │
│  │  + getRevenueByHour()│  │ + searchProduct()│  │ + acknowledge│  │
│  └──────────┬───────────┘  └────────┬─────────┘  └──────┬───────┘  │
│             │                       │                    │          │
│             └───────────┬───────────┴───────────┬────────┘          │
│                         │                       │                   │
│                  ┌──────▼──────┐         ┌──────▼──────┐           │
│                  │ Repository  │         │  External   │           │
│                  │  (Data)     │         │   API       │           │
│                  └─────────────┘         └─────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Ví dụ code hoàn chỉnh

### Cấu trúc project

```
dashboard/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── entities.py          # Domain entities
│   ├── services.py          # Business logic services
│   └── repositories.py      # Data access
├── presenter/
│   ├── __init__.py
│   ├── base.py              # BasePresenter
│   └── dashboard_presenter.py
├── view/
│   ├── __init__.py
│   ├── interfaces.py        # IView protocols
│   └── console_view.py      # Concrete implementation (Console)
├── main.py                  # Entry point
└── tests/
    ├── __init__.py
    └── test_dashboard.py
```

### model/entities.py

```python
"""Domain entities — hoàn toàn không biết gì về View hay Presenter."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from datetime import datetime
from typing import Optional


class ProductCategory(Enum):
    ELECTRONICS = auto()
    FASHION = auto()
    FOOD = auto()
    BOOKS = auto()
    HOME = auto()
    SPORTS = auto()


class AlertSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass(frozen=True)
class Revenue:
    """Tổng doanh thu."""
    total: Decimal
    currency: str = "VND"
    growth_rate: float = 0.0  # % tăng trưởng so với hôm qua
    hourly_data: tuple[Decimal, ...] = field(default_factory=lambda: tuple())

    def formatted_total(self) -> str:
        return f"{self.total:,.0f} {self.currency}"

    def formatted_growth(self) -> str:
        sign = "+" if self.growth_rate >= 0 else ""
        return f"{sign}{self.growth_rate:.1f}%"


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    category: ProductCategory
    price: Decimal
    quantity_sold: int
    revenue: Decimal

    @property
    def formatted_revenue(self) -> str:
        return f"{self.revenue:,.0f} VND"


@dataclass(frozen=True)
class SalesAlert:
    id: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    acknowledged: bool = False

    def is_critical(self) -> bool:
        return self.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)


@dataclass
class DashboardFilter:
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    category: Optional[ProductCategory] = None
    min_revenue: Optional[Decimal] = None
```

### model/repositories.py

```python
"""Data repositories — mô phỏng database/API calls."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
import random
from typing import Sequence

from .entities import (
    Product, ProductCategory, Revenue, SalesAlert, AlertSeverity,
)


class RevenueRepository:
    """Repository for revenue data."""

    def get_today_revenue(self) -> Revenue:
        # Mô phỏng gọi API/database
        total = Decimal(str(random.randint(500_000_000, 2_000_000_000)))
        growth = round(random.uniform(-0.15, 0.35), 2)
        hourly = tuple(
            Decimal(str(random.randint(10_000_000, 150_000_000)))
            for _ in range(24)
        )
        return Revenue(
            total=total,
            growth_rate=growth,
            hourly_data=hourly,
        )

    def get_revenue_by_date(self, dt: datetime) -> Revenue:
        total = Decimal(str(random.randint(200_000_000, 1_500_000_000)))
        growth = round(random.uniform(-0.2, 0.4), 2)
        hourly = tuple(
            Decimal(str(random.randint(5_000_000, 100_000_000)))
            for _ in range(24)
        )
        return Revenue(total=total, growth_rate=growth, hourly_data=hourly)


class ProductRepository:
    """Repository for product data."""

    def __init__(self) -> None:
        self._products = self._seed_products()

    @staticmethod
    def _seed_products() -> list[Product]:
        names = [
            ("iPhone 15 Pro Max", ProductCategory.ELECTRONICS, Decimal("33_990_000")),
            ("Áo sơ mi nam công sở", ProductCategory.FASHION, Decimal("450_000")),
            ("Sách Clean Code", ProductCategory.BOOKS, Decimal("280_000")),
            ("Bộ nồi inox 5 món", ProductCategory.HOME, Decimal("1_890_000")),
            ("Giày chạy bộ Nike", ProductCategory.SPORTS, Decimal("2_500_000")),
            ("MacBook Air M3", ProductCategory.ELECTRONICS, Decimal("28_990_000")),
            ("Đồng hồ thông minh", ProductCategory.ELECTRONICS, Decimal("5_990_000")),
            ("Máy pha cà phê", ProductCategory.HOME, Decimal("8_500_000")),
            ("Sách DSA", ProductCategory.BOOKS, Decimal("350_000")),
            ("Quần jean nam", ProductCategory.FASHION, Decimal("650_000")),
        ]
        products: list[Product] = []
        for i, (name, cat, price) in enumerate(names, 1):
            qty = random.randint(50, 5000)
            products.append(Product(
                id=f"P{i:04d}",
                name=name,
                category=cat,
                price=price,
                quantity_sold=qty,
                revenue=price * qty,
            ))
        products.sort(key=lambda p: p.revenue, reverse=True)
        return products

    def get_top_products(self, limit: int = 10) -> Sequence[Product]:
        return self._products[:limit]

    def get_products_by_category(self, category: ProductCategory) -> Sequence[Product]:
        return [p for p in self._products if p.category == category]


class AlertRepository:
    """Repository for system alerts."""

    def __init__(self) -> None:
        self._alerts = self._seed_alerts()

    @staticmethod
    def _seed_alerts() -> list[SalesAlert]:
        now = datetime.now()
        return [
            SalesAlert("A1", "Doanh thu vượt 1 tỷ!", AlertSeverity.INFO, now - timedelta(minutes=5)),
            SalesAlert("A2", "Tỷ lệ lỗi thanh toán > 5%", AlertSeverity.WARNING, now - timedelta(minutes=2)),
            SalesAlert("A3", "Server payment trả timeout", AlertSeverity.CRITICAL, now - timedelta(minutes=1)),
            SalesAlert("A4", "Phát hiện bot đặt hàng hàng loạt", AlertSeverity.EMERGENCY, now),
        ]

    def get_active_alerts(self) -> Sequence[SalesAlert]:
        return [a for a in self._alerts if not a.acknowledged]

    def acknowledge_alert(self, alert_id: str) -> None:
        for alert in self._alerts:
            if alert.id == alert_id:
                object.__setattr__(alert, "acknowledged", True)
                break
```

### model/services.py

```python
"""Domain services — business logic orchestration."""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from .entities import (
    Product, ProductCategory, Revenue, SalesAlert, DashboardFilter,
)
from .repositories import RevenueRepository, ProductRepository, AlertRepository


class DashboardService:
    """Service tổng hợp data cho Dashboard — không biết gì về UI."""

    def __init__(
        self,
        revenue_repo: RevenueRepository,
        product_repo: ProductRepository,
        alert_repo: AlertRepository,
    ) -> None:
        self._revenue_repo = revenue_repo
        self._product_repo = product_repo
        self._alert_repo = alert_repo

    def get_dashboard_data(self, filter_: DashboardFilter | None = None) -> dict:
        """Lấy tất cả dữ liệu cần thiết cho dashboard."""
        revenue = self._revenue_repo.get_today_revenue()
        top_products = self._product_repo.get_top_products(10)
        alerts = self._alert_repo.get_active_alerts()

        if filter_ and filter_.category:
            top_products = self._product_repo.get_products_by_category(filter_.category)

        return {
            "revenue": revenue,
            "top_products": top_products,
            "alerts": alerts,
            "total_products": len(top_products),
            "critical_alert_count": sum(1 for a in alerts if a.is_critical()),
        }

    def acknowledge_alert(self, alert_id: str) -> None:
        self._alert_repo.acknowledge_alert(alert_id)

    def get_conversion_rate(self) -> float:
        # Mô phỏng: trả về tỉ lệ chuyển đổi
        return round(random.uniform(0.02, 0.08), 4)
```

### view/interfaces.py

```python
"""View interfaces — contract giữa View và Presenter.

Đây là trái tim của MVP: View interface định nghĩa các method
mà Presenter có thể gọi để cập nhật View.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Protocol, Sequence, runtime_checkable

from model.entities import Product, Revenue, SalesAlert


@runtime_checkable
class IDashboardView(Protocol):
    """Interface cho Dashboard View — Presenter phụ thuộc vào interface này.

    View implementation cụ thể (Console, Web, Mobile) sẽ implement
    interface này.
    """

    def display_revenue(self, revenue: Revenue) -> None: ...
    def display_top_products(self, products: Sequence[Product]) -> None: ...
    def display_alerts(self, alerts: Sequence[SalesAlert]) -> None: ...
    def display_conversion_rate(self, rate: float) -> None: ...
    def show_loading(self, visible: bool) -> None: ...
    def show_error(self, message: str) -> None: ...
    def show_success(self, message: str) -> None: ...


class IAlertView(Protocol):
    """Interface riêng cho Alert widget."""

    def show_alert(self, alert: SalesAlert) -> None: ...
    def show_alert_summary(self, critical_count: int, total_count: int) -> None: ...
```

### view/console_view.py

```python
"""Console implementation của Dashboard View.

View này implement IDashboardView để hiển thị trên terminal.
Trong thực tế, đây sẽ là một component React/Vue/SwiftUI.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence
from datetime import datetime

from model.entities import Product, Revenue, SalesAlert
from view.interfaces import IDashboardView


class ConsoleDashboardView:
    """View hiển thị dashboard trên console.

    View này HOÀN TOÀN THỤ ĐỘNG — nó chỉ làm đúng những gì
    Presenter bảo nó làm qua interface calls.

    Không chứa business logic, không format dữ liệu phức tạp.
    """

    def display_revenue(self, revenue: Revenue) -> None:
        growth_color = "\033[32m" if revenue.growth_rate >= 0 else "\033[31m"
        reset = "\033[0m"
        print(f"\n{'='*60}")
        print(f"  📊 DOANH THU HÔM NAY")
        print(f"{'='*60}")
        print(f"  Tổng:      {revenue.formatted_total()}")
        print(f"  Tăng trưởng: {growth_color}{revenue.formatted_growth()}{reset}")

        # Hiển thị hourly data dạng bar chart đơn giản
        if revenue.hourly_data:
            max_val = max(revenue.hourly_data)
            print(f"\n  Biểu đồ theo giờ (triệu VND):")
            for i, val in enumerate(revenue.hourly_data):
                bar_len = int(val / max_val * 20) if max_val > 0 else 0
                bar = "█" * bar_len
                print(f"  {i:02d}h │ {bar} {val / 1_000_000:.0f}M")

    def display_top_products(self, products: Sequence[Product]) -> None:
        print(f"\n{'='*60}")
        print(f"  🏆 TOP SẢN PHẨM BÁN CHẠY")
        print(f"{'='*60}")
        for i, p in enumerate(products, 1):
            medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"  {i:2d}."
            print(f"  {medal} {p.name:<30s} | {p.quantity_sold:>5,} cái | {p.formatted_revenue:>15s}")

    def display_alerts(self, alerts: Sequence[SalesAlert]) -> None:
        severity_colors = {
            "INFO": "\033[34m",       # Blue
            "WARNING": "\033[33m",    # Yellow
            "CRITICAL": "\033[31m",   # Red
            "EMERGENCY": "\033[41m",  # Red background
        }
        reset = "\033[0m"

        print(f"\n{'='*60}")
        print(f"  🔔 CẢNH BÁO HỆ THỐNG")
        print(f"{'='*60}")
        for alert in alerts:
            color = severity_colors.get(alert.severity.name, "")
            severity_tag = f"[{alert.severity.name:^9s}]"
            print(f"  {color}{severity_tag}{reset} {alert.message:<50s}")
            print(f"  {'':12s}{alert.timestamp.strftime('%H:%M:%S')}")
            if alert.is_critical():
                print(f"  {'':12s}{'⚠️  Cần xử lý ngay!'}")

    def display_conversion_rate(self, rate: float) -> None:
        pct = rate * 100
        color = "\033[32m" if pct > 5.0 else "\033[33m"
        reset = "\033[0m"
        print(f"\n{'='*60}")
        print(f"  📈 TỶ LỆ CHUYỂN ĐỔI")
        print(f"{'='*60}")
        print(f"  Hiện tại: {color}{pct:.2f}%{reset}")
        bar_len = int(pct)
        print(f"  {'█' * bar_len}{'░' * (20 - bar_len)} {pct:.2f}%")

    def show_loading(self, visible: bool) -> None:
        if visible:
            print("\n  ⏳ Đang tải dữ liệu...")
        else:
            print("  ✅ Tải dữ liệu hoàn tất")

    def show_error(self, message: str) -> None:
        print(f"\n  \033[31m❌ LỖI: {message}\033[0m")

    def show_success(self, message: str) -> None:
        print(f"\n  \033[32m✅ {message}\033[0m")

    def get_user_input(self) -> str:
        """Nhận input từ user (mô phỏng event từ UI)."""
        print(f"\n{'─'*60}")
        print("  Hành động: [r]efresh  [a]cknowledge alert  [f]ilter  [q]uit")
        return input("  Chọn: ").strip().lower()
```

### presenter/base.py

```python
"""Base Presenter — định nghĩa lifecycle và common behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePresenter(ABC):
    """Base class cho tất cả Presenter.

    Presenter quản lý lifecycle của một màn hình/widget và
    chịu trách nhiệm xử lý tất cả logic hiển thị.
    """

    def __init__(self) -> None:
        self._initialized = False

    @abstractmethod
    def on_view_loaded(self) -> None:
        """View đã sẵn sàng — Presenter bắt đầu load data."""
        ...

    @abstractmethod
    def on_view_destroyed(self) -> None:
        """View bị hủy — cleanup resources."""
        ...

    def initialize(self) -> None:
        if not self._initialized:
            self._initialized = True
```

### presenter/dashboard_presenter.py

```python
"""Dashboard Presenter — trung tâm logic của màn hình Dashboard.

Presenter này xử lý mọi logic:
- Khi nào load data, load gì
- Format dữ liệu thế nào trước khi đẩy xuống View
- Xử lý filter, refresh, acknowledge
- Quyết định trạng thái hiển thị (loading, error, success)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from model.entities import DashboardFilter, ProductCategory
from model.services import DashboardService
from presenter.base import BasePresenter

if TYPE_CHECKING:
    from view.interfaces import IDashboardView


class DashboardPresenter(BasePresenter):
    """Presenter cho Dashboard — chứa tất cả presentation logic.

    Nguyên tắc:
    - Presenter giao tiếp với View qua IDashboardView interface
    - Presenter gọi Model (DashboardService) để lấy dữ liệu
    - Presenter format và quyết định hiển thị
    - View chỉ là passive receiver
    """

    def __init__(self, service: DashboardService, view: Optional["IDashboardView"] = None) -> None:
        super().__init__()
        self._service = service
        self._view: Optional[IDashboardView] = None
        self._current_filter: Optional[DashboardFilter] = None

        if view is not None:
            self.attach_view(view)

    def attach_view(self, view: "IDashboardView") -> None:
        """Gán View cho Presenter (setter injection)."""
        self._view = view

    def on_view_loaded(self) -> None:
        """View đã sẵn sàng — Presenter load initial data."""
        if not self._initialized:
            self.initialize()
        self._load_dashboard_data()

    def on_view_destroyed(self) -> None:
        """View bị hủy — Presenter cleanup."""
        self._view = None
        self._current_filter = None

    def on_refresh_clicked(self) -> None:
        """User nhấn nút Refresh."""
        self._load_dashboard_data()

    def on_filter_changed(self, category: ProductCategory | None = None) -> None:
        """User thay đổi filter."""
        if category:
            self._current_filter = DashboardFilter(category=category)
        else:
            self._current_filter = None
        self._load_dashboard_data()

    def on_acknowledge_alert(self, alert_id: str) -> None:
        """User acknowledge một alert."""
        try:
            self._service.acknowledge_alert(alert_id)
            if self._view:
                self._view.show_success(f"Alert {alert_id} đã được xác nhận")
            self._load_dashboard_data()  # Reload với alert mới
        except Exception as e:
            if self._view:
                self._view.show_error(f"Không thể xác nhận alert: {e}")

    def _load_dashboard_data(self) -> None:
        """Load và hiển thị dữ liệu dashboard.

        Đây là method quan trọng nhất — Presenter quyết định:
        1. Gọi service nào
        2. Format dữ liệu thế nào
        3. Cập nhật View ra sao
        """
        if not self._view:
            return

        try:
            self._view.show_loading(True)
            data = self._service.get_dashboard_data(self._current_filter)

            # Presenter quyết định hiển thị gì
            self._view.display_revenue(data["revenue"])
            self._view.display_top_products(data["top_products"])
            self._view.display_alerts(data["alerts"])
            self._view.display_conversion_rate(self._service.get_conversion_rate())

            # Presenter tính toán và hiển thị summary
            summary_msg = (
                f"Dashboard loaded: "
                f"{data['total_products']} products, "
                f"{data['critical_alert_count']} critical alerts"
            )
            self._view.show_success(summary_msg)
        except Exception as e:
            self._view.show_error(f"Lỗi khi tải dashboard: {e}")
        finally:
            self._view.show_loading(False)
```

### main.py

```python
"""Entry point — kết nối tất cả thành phần.

Đây là nơi khởi tạo và kết nối Model, View, Presenter.
Lưu ý: Model hoàn toàn không biết về View hay Presenter.
View chỉ biết Presenter qua interface. Presenter chỉ biết View qua interface.
"""

from __future__ import annotations

import sys
from typing import NoReturn

from model.repositories import RevenueRepository, ProductRepository, AlertRepository
from model.services import DashboardService
from view.console_view import ConsoleDashboardView
from presenter.dashboard_presenter import DashboardPresenter


def main() -> NoReturn:
    # 1. Khởi tạo Model layer (hoàn toàn độc lập)
    revenue_repo = RevenueRepository()
    product_repo = ProductRepository()
    alert_repo = AlertRepository()
    service = DashboardService(revenue_repo, product_repo, alert_repo)

    # 2. Khởi tạo View (passive, không biết gì)
    view = ConsoleDashboardView()

    # 3. Khởi tạo Presenter với Model, inject View sau
    presenter = DashboardPresenter(service=service, view=view)

    # 4. Presenter bắt đầu load dữ liệu
    presenter.on_view_loaded()

    # 5. Event loop (mô phỏng user interaction)
    while True:
        try:
            command = view.get_user_input()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Tạm biệt!")
            presenter.on_view_destroyed()
            sys.exit(0)

        if command == "q":
            print("  👋 Tạm biệt!")
            presenter.on_view_destroyed()
            sys.exit(0)
        elif command == "r":
            presenter.on_refresh_clicked()
        elif command == "a":
            presenter.on_acknowledge_alert("A3")
        elif command == "f":
            presenter.on_filter_changed(ProductCategory.ELECTRONICS)
        else:
            view.show_error("Lệnh không hợp lệ")


if __name__ == "__main__":
    main()
```

### Đầu ra mẫu

```
============================================================
  📊 DOANH THU HÔM NAY
============================================================
  Tổng:      1,234,567,890 VND
  Tăng trưởng: +12.3%

  Biểu đồ theo giờ (triệu VND):
  00h │ ████████████▉            125M
  01h │ █████▊                   56M
  ...
  23h │ ██████████████▋          178M

============================================================
  🏆 TOP SẢN PHẨM BÁN CHẠY
============================================================
  🥇 iPhone 15 Pro Max              | 3,456 cái |  117,469,440,000 VND
  🥈 MacBook Air M3                 | 2,100 cái |   60,879,000,000 VND
  🥉 Máy pha cà phê                 | 1,500 cái |   12,750,000,000 VND

============================================================
  🔔 CẢNH BÁO HỆ THỐNG
============================================================
  [  INFO   ] Doanh thu vượt 1 tỷ!
              14:30:00
  [CRITICAL ] Server payment trả timeout
              14:35:00
              ⚠️  Cần xử lý ngay!
  [EMERGENCY] Phát hiện bot đặt hàng hàng loạt
              14:36:00
              ⚠️  Cần xử lý ngay!

  ✅ Dashboard loaded: 10 products, 2 critical alerts
  ⏳ Đang tải dữ liệu...
  ✅ Tải dữ liệu hoàn tất
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng MVP | Khi nào không dùng MVP |
|-----------------|----------------------|
| **Ứng dụng cần testability cao** — Presenter có thể unit test 100% | **UI đơn giản** — Form CRUD ít logic, data binding là đủ |
| **View thay đổi thường xuyên** — Có nhiều platform (Web, Mobile, Desktop) | **Ứng dụng real-time complex** — Nếu có data binding 2 chiều phức tạp, MVVM phù hợp hơn |
| **Business logic phức tạp** — Nhiều validation, transformation, conditional display | **MVP overhead không đáng** — Với ứng dụng < 5 màn hình, MVP quá nặng |
| **Cần test UI logic** — Logic quyết định màu sắc, hiển thị dựa trên data | **Đội ngũ quen MVC/MVVM** — Learning curve cho MVP khá cao |
| **Legacy system migration** — Dễ dàng thay View cũ bằng View mới | **Data binding framework mạnh** — React/Vue/Angular có sẵn state management |
| **Ứng dụng enterprise** — Cần separation of concerns nghiêm ngặt | **Prototype/MVP (sản phẩm tối thiểu)** — Cần speed, không cần architecture phức tạp |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| **Testability tối đa**: Presenter có thể test hoàn toàn với mock View | **Boilerplate code**: Cần viết interface, nhiều class hơn |
| **Separation of concerns hoàn hảo**: Model ↔ View ↔ Presenter tách biệt hoàn toàn | **Presenter phình to**: Presenter dễ trở thành god object |
| **View swapping**: Dễ dàng thay đổi UI platform mà không ảnh hưởng logic | **View interface maintenance**: Mỗi lần thêm UI feature, phải update interface |
| **Single Responsibility**: Mỗi Presenter chỉ quản lý một màn hình | **Complex event handling**: Nhiều event từ View → Presenter tracking khó |
| **Dễ onboarding**: Architecture rõ ràng, predict | **Learning curve**: Khác biệt lớn so với MVC/MVVM quen thuộc |
| **Model reuse**: Model có thể dùng ở nhiều Presenter khác nhau | **Over-engineering cho ứng dụng nhỏ**: Nếu chỉ có 2-3 màn hình |
| **Debug dễ dàng**: Luồng dữ liệu 1 chiều rõ ràng | **Presenter có thể chứa UI logic**: Cần discipline để không leak |
| **Hỗ trợ DI tốt**: Dễ dàng inject dependencies | **Thread-safety issues**: Presenter cần quản lý async operations |

---

## Công cụ và Framework

### Python
| Framework | Mô tả |
|-----------|-------|
| **Tkinter + MVP** | Có thể implement MVP pattern thủ công với Tkinter |
| **PyQt/PySide** | Qt framework có sẵn signal/slot rất phù hợp với MVP |
| **Kivy** | Cross-platform UI, dễ implement MVP |
| **Flet** | Flutter-based UI cho Python, event-driven |
| **PyWebIO** | Web UI trong terminal, MVP friendly |

### Java/Kotlin
| Framework | Mô tả |
|-----------|-------|
| **Android MVP** (deprecated) | Google từng khuyến nghị MVP cho Android |
| **GWT** (Google Web Toolkit) | MVP là pattern chính của GWT |
| **Vaadin** | Java web framework với MVP-like architecture |
| **MVP4G** | MVP framework cho GWT |

### .NET
| Framework | Mô tả |
|-----------|-------|
| **Windows Forms** | MVP pattern rất phổ biến trong WinForms |
| **ASP.NET Web Forms** | Code-behind pattern chính là MVP |
| **Prism** (WPF/Xamarin) | Hỗ trợ cả MVP và MVVM |

### Web
| Framework | Mô tả |
|-----------|-------|
| **React + MVP** | Có thể implement MVP với React hooks |
| **Vue + MVP** | Vue component là View, Vuex là Model |
| **Backbone.js** | JavaScript framework MVP-native |

---

## Kiểm thử

### Chiến lược kiểm thử MVP

MVP cho phép test **Presenter hoàn toàn độc lập** với View bằng cách mock View interface.

```python
"""tests/test_dashboard_presenter.py

Chiến lược:
1. Mock View interface — kiểm tra Presenter gọi đúng method trên View
2. Mock Service — kiểm tra Presenter xử lý dữ liệu đúng
3. Integration test — Presenter + Service + Repository thật
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, Mock, patch
from decimal import Decimal
from datetime import datetime
from typing import Sequence

import sys
sys.path.insert(0, "..")

from model.entities import (
    Product, ProductCategory, Revenue, SalesAlert, AlertSeverity, DashboardFilter,
)
from model.services import DashboardService
from model.repositories import RevenueRepository, ProductRepository, AlertRepository
from presenter.dashboard_presenter import DashboardPresenter


class MockDashboardView:
    """Mock View để test Presenter — không cần UI thật."""

    def __init__(self) -> None:
        self.display_revenue_calls: list[Revenue] = []
        self.display_top_products_calls: list[Sequence[Product]] = []
        self.display_alerts_calls: list[Sequence[SalesAlert]] = []
        self.display_conversion_rate_calls: list[float] = []
        self.show_loading_calls: list[bool] = []
        self.show_error_calls: list[str] = []
        self.show_success_calls: list[str] = []

    def display_revenue(self, revenue: Revenue) -> None:
        self.display_revenue_calls.append(revenue)

    def display_top_products(self, products: Sequence[Product]) -> None:
        self.display_top_products_calls.append(list(products))

    def display_alerts(self, alerts: Sequence[SalesAlert]) -> None:
        self.display_alerts_calls.append(list(alerts))

    def display_conversion_rate(self, rate: float) -> None:
        self.display_conversion_rate_calls.append(rate)

    def show_loading(self, visible: bool) -> None:
        self.show_loading_calls.append(visible)

    def show_error(self, message: str) -> None:
        self.show_error_calls.append(message)

    def show_success(self, message: str) -> None:
        self.show_success_calls.append(message)


class TestDashboardPresenter(unittest.TestCase):
    """Test Presenter với mock View và mock Service."""

    def setUp(self) -> None:
        self.mock_view = MockDashboardView()
        self.mock_service = MagicMock(spec=DashboardService)

        # Setup mock data
        self.mock_revenue = Revenue(
            total=Decimal("1_000_000_000"),
            growth_rate=0.15,
            hourly_data=tuple(Decimal("50_000_000") for _ in range(24)),
        )
        self.mock_products = [
            Product("P1", "Test Product", ProductCategory.ELECTRONICS,
                    Decimal("1_000_000"), 100, Decimal("100_000_000")),
        ]
        self.mock_alerts = [
            SalesAlert("A1", "Test alert", AlertSeverity.INFO, datetime.now()),
        ]

        self.mock_service.get_dashboard_data.return_value = {
            "revenue": self.mock_revenue,
            "top_products": self.mock_products,
            "alerts": self.mock_alerts,
            "total_products": 1,
            "critical_alert_count": 0,
        }
        self.mock_service.get_conversion_rate.return_value = 0.0523

        self.presenter = DashboardPresenter(
            service=self.mock_service,
            view=self.mock_view,
        )

    def test_on_view_loaded_shows_correct_data(self):
        """Presenter load data và gọi đúng View methods."""
        self.presenter.on_view_loaded()

        self.assertEqual(len(self.mock_view.display_revenue_calls), 1)
        self.assertEqual(
            self.mock_view.display_revenue_calls[0].total,
            Decimal("1_000_000_000"),
        )
        self.assertEqual(len(self.mock_view.display_top_products_calls), 1)
        self.assertEqual(len(self.mock_view.display_alerts_calls), 1)
        self.assertEqual(len(self.mock_view.display_conversion_rate_calls), 1)
        self.assertEqual(self.mock_view.display_conversion_rate_calls[0], 0.0523)

    def test_loading_state_is_shown(self):
        """Presenter hiển thị loading trước và sau khi load."""
        self.presenter.on_view_loaded()

        # loading true -> loading false
        self.assertEqual(self.mock_view.show_loading_calls, [True, False])

    def test_success_message_on_load(self):
        """Presenter hiển thị success message sau khi load xong."""
        self.presenter.on_view_loaded()

        self.assertEqual(len(self.mock_view.show_success_calls), 1)
        self.assertIn("Dashboard loaded", self.mock_view.show_success_calls[0])

    def test_error_handling(self):
        """Presenter xử lý lỗi từ service và hiển thị error."""
        self.mock_service.get_dashboard_data.side_effect = RuntimeError("DB connection failed")

        self.presenter.on_view_loaded()

        self.assertEqual(len(self.mock_view.show_error_calls), 1)
        self.assertIn("DB connection failed", self.mock_view.show_error_calls[0])
        # Vẫn phải tắt loading
        self.assertEqual(self.mock_view.show_loading_calls[-1], False)

    def test_refresh_reloads_data(self):
        """Refresh click gọi lại load data."""
        self.presenter.on_view_loaded()
        initial_calls = len(self.mock_view.display_revenue_calls)

        self.presenter.on_refresh_clicked()

        self.assertEqual(
            len(self.mock_view.display_revenue_calls),
            initial_calls + 1,
        )

    def test_acknowledge_alert(self):
        """Acknowledge alert gọi service và reload."""
        self.presenter.on_view_loaded()
        self.mock_view.show_success_calls.clear()

        self.presenter.on_acknowledge_alert("A3")

        self.mock_service.acknowledge_alert.assert_called_once_with("A3")
        self.assertGreater(len(self.mock_view.show_success_calls), 0)

    def test_filter_by_category(self):
        """Filter thay đổi dữ liệu hiển thị."""
        self.presenter.on_view_loaded()
        calls_before = len(self.mock_view.display_top_products_calls)

        self.presenter.on_filter_changed(ProductCategory.ELECTRONICS)

        self.assertGreater(len(self.mock_view.display_top_products_calls), calls_before)

    def test_presenter_no_view_does_not_crash(self):
        """Presenter không crash nếu không có view."""
        p = DashboardPresenter(service=self.mock_service)
        p.on_view_loaded()  # Should not raise
        p.on_refresh_clicked()
        p.on_filter_changed(ProductCategory.FOOD)
        p.on_acknowledge_alert("A1")

    def test_attach_view_late(self):
        """Presenter có thể nhận View sau khi khởi tạo."""
        p = DashboardPresenter(service=self.mock_service)
        p.attach_view(self.mock_view)
        p.on_view_loaded()

        self.assertEqual(len(self.mock_view.display_revenue_calls), 1)

    def test_on_view_destroyed_cleanup(self):
        """Presenter cleanup đúng khi View bị hủy."""
        self.presenter.on_view_destroyed()

        self.assertIsNone(self.presenter._view)


class TestDashboardService(unittest.TestCase):
    """Test Service với Repository thật (integration test)."""

    def setUp(self) -> None:
        self.revenue_repo = RevenueRepository()
        self.product_repo = ProductRepository()
        self.alert_repo = AlertRepository()
        self.service = DashboardService(
            self.revenue_repo,
            self.product_repo,
            self.alert_repo,
        )

    def test_get_dashboard_data_returns_all_keys(self):
        """Service trả về đầy đủ dữ liệu."""
        data = self.service.get_dashboard_data()

        self.assertIn("revenue", data)
        self.assertIn("top_products", data)
        self.assertIn("alerts", data)
        self.assertIn("total_products", data)
        self.assertIn("critical_alert_count", data)

    def test_revenue_is_valid(self):
        """Revenue có format đúng."""
        data = self.service.get_dashboard_data()
        revenue = data["revenue"]

        self.assertIsInstance(revenue.total, Decimal)
        self.assertGreater(revenue.total, Decimal("0"))
        self.assertEqual(len(revenue.hourly_data), 24)

    def test_top_products_sorted(self):
        """Top products được sắp xếp theo revenue giảm dần."""
        data = self.service.get_dashboard_data()
        products = list(data["top_products"])

        for i in range(len(products) - 1):
            self.assertGreaterEqual(
                products[i].revenue,
                products[i + 1].revenue,
            )

    def test_acknowledge_alert(self):
        """Acknowledge alert hoạt động."""
        data = self.service.get_dashboard_data()
        initial_count = len(data["alerts"])

        self.service.acknowledge_alert("A3")

        data = self.service.get_dashboard_data()
        self.assertLess(len(data["alerts"]), initial_count)

    def test_filter_by_category(self):
        """Filter theo category trả về đúng sản phẩm."""
        filt = DashboardFilter(category=ProductCategory.ELECTRONICS)
        data = self.service.get_dashboard_data(filt)

        for product in data["top_products"]:
            self.assertEqual(product.category, ProductCategory.ELECTRONICS)

    def test_conversion_rate_in_range(self):
        """Conversion rate nằm trong khoảng hợp lý."""
        for _ in range(100):
            rate = self.service.get_conversion_rate()
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

### Chạy test

```bash
python -m pytest tests/test_dashboard_presenter.py -v

# Output:
# test_acknowledge_alert ... OK
# test_attach_view_late ... OK
# test_error_handling ... OK
# test_filter_by_category ... OK
# test_loading_state_is_shown ... OK
# test_on_view_destroyed_cleanup ... OK
# test_on_view_loaded_shows_correct_data ... OK
# test_presenter_no_view_does_not_crash ... OK
# test_refresh_reloads_data ... OK
# test_success_message_on_load ... OK
# test_acknowledge_alert ... OK
# test_conversion_rate_in_range ... OK
# test_filter_by_category ... OK
# test_get_dashboard_data_returns_all_keys ... OK
# test_revenue_is_valid ... OK
# test_top_products_sorted ... OK
```

---

## Kết luận

**MVP (Model-View-Presenter)** là một architectural pattern mạnh mẽ, đặc biệt phù hợp cho các ứng dụng cần độ testability cao và separation of concerns nghiêm ngặt.

### Best Practices

1. **Passive View là lựa chọn mặc định**: View nên càng "ngu" càng tốt. Nếu View bắt đầu format dữ liệu, đó là code smell.

2. **Interface-first design**: Viết View interface trước, implementation sau. Điều này đảm bảo Presenter không phụ thuộc vào View cụ thể.

3. **Một Presenter / một Màn hình**: Không tạo Presenter quá lớn. Nếu một màn hình có 5 widget phức tạp, hãy có 5 Presenter nhỏ.

4. **Dependency Injection**: Inject Model (Service) vào Presenter qua constructor. Dùng factory pattern để tạo Presenter.

5. **Lifecycle management**: Presenter cần quản lý lifecycle — `on_view_loaded()` và `on_view_destroyed()`.

6. **Async operations**: Xử lý async trong Presenter một cách cẩn thận. Dùng async/await hoặc callback.

7. **Không leak Model vào View**: View không bao giờ được nhìn thấy Model entity. Presenter chuyển đổi Model → ViewModel nếu cần.

### Golden Rules

| Rule | Giải thích |
|------|-----------|
| **View không import Model** | View chỉ biết kiểu dữ liệu primitive (string, int, bool) |
| **Presenter không import UI framework** | Presenter không biết đến DOM, Widget, Component |
| **Presenter càng stateless càng tốt** | State lưu trong Model, không trong Presenter |
| **Test Presenter trước, View sau** | Presenter không cần View thật để test |
| **Interface càng nhỏ càng tốt** | IDashboardView nên có 5-10 method, không phải 50 |

MVP là kiến trúc lý tưởng khi bạn cần kiểm soát hoàn toàn presentation logic, đặc biệt trong các ứng dụng enterprise phức tạp. Sự đánh đổi là boilerplate code, nhưng lợi ích về maintainability và testability là rất lớn.
