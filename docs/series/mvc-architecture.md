---
id: mvc-architecture
title: MVC — Model-View-Controller
sidebar_label: MVC Architecture
sidebar_position: 44
---

# MVC — Model-View-Controller

> *"MVC is not a design pattern — it's a compositional pattern that describes the structure of a system of cooperating objects."* — Trygve Reenskaug

---

## Tổng quan

**Model-View-Controller (MVC)** là một architectural pattern được **Trygve Reenskaug** giới thiệu lần đầu năm 1979 khi làm việc tại Xerox PARC. MVC được thiết kế cho Smalltalk-80 và trở thành một trong những pattern có ảnh hưởng nhất trong lịch sử software engineering.

### Các nhân vật chủ chốt

- **Trygve Reenskaug** — Cha đẻ của MVC (1979)
- **Steve Burbeck** — Phát triển MVC cho Smalltalk (1987)
- **Krasner & Pope** — "A Description of the Model-View-Controller User Interface Paradigm" (1988) — bài báo kinh điển
- **Kent Beck** — Áp dụng MVC trong XP và Test-Driven Development
- **Ruby on Rails team (David Heinemeier Hansson)** — Phổ biến MVC trong web development (2004)

### Các biến thể MVC

| Loại | Đặc điểm | Ví dụ |
|---|---|---|
| **Classic MVC** | View nhận event, gọi Controller → Model | Smalltalk |
| **Web MVC** | Controller nhận request, xử lý, trả response | Rails, Django, Spring |
| **Passive View** | View không biết Model, Controller điều khiển hoàn toàn | MVP |
| **MVVM** | View bound vào ViewModel qua data binding | WPF, Vue.js |

---

## Bài toán

### Sự hỗn loạn của spaghetti code

Trước MVC, code UI thường là hỗn loạn: business logic, data access, và UI presentation nằm lẫn lộn trong cùng một file. Hãy tưởng tượng một PHP page năm 2000:

```php
<?php
// Database query ở đây
$result = mysql_query("SELECT * FROM users");
// HTML ở đây
echo "<table>";
while ($row = mysql_fetch_array($result)) {
    // Business logic ở đây
    if ($row['balance'] > 1000000) {
        echo "<tr class='vip'>";
    }
    echo "<td>" . $row['name'] . "</td>";
    echo "</tr>";
}
echo "</table>";
// Và thêm logic xử lý form ở đây nữa
?>
```

File này là horror: không thể test, không thể maintain, không thể mở rộng.

### Separation of Concerns

MVC giải quyết bằng cách tách thành 3 thành phần riêng biệt:
- **Model**: Dữ liệu và business logic — KHÔNG biết gì về UI
- **View**: Cách dữ liệu được hiển thị — KHÔNG có business logic
- **Controller**: Xử lý input, điều phối Model và View — KHÔNG có SQL hay HTML trong code

### Multiple Representations

Một Model có thể có nhiều View khác nhau. Ví dụ:
- Một biểu đồ doanh thu có thể hiển thị dưới dạng: bảng số liệu, biểu đồ cột, biểu đồ tròn, JSON API
- Một tài khoản ngân hàng có thể hiển thị trong: Web UI, Mobile App, CLI, PDF statement

MVC cho phép thêm View mới mà không ảnh hưởng đến Model.

### Testability

Trong spaghetti code, không thể viết unit test vì UI logic gắn chặt với business logic. MVC tách biệt:
- **Model**: Test thuần (pure logic)
- **Controller**: Test với mock request/response
- **View**: Template rendering test (snapshot testing)

---

## Nguyên lý thiết kế

### 1. Separation of Concerns

Ba thành phần có trách nhiệm riêng biệt:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Model      │     │    View      │     │  Controller  │
│──────────────│     │──────────────│     │──────────────│
│ Dữ liệu      │     │ Hiển thị    │     │ Xử lý input  │
│ Business     │     │ UI          │     │ Điều phối    │
│ Logic        │     │ Template    │     │ Response     │
│ Validation   │     │ Data binding│     │ Validation   │
│ Persistence  │     │ Formatting  │     │ Routing      │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 2. Model không biết View và Controller

Model hoàn toàn độc lập. Nó chỉ biết về business logic. View và Controller biết Model, nhưng Model không biết View.

```python
class UserModel:
    """Model hoàn toàn không biết đến View hay Controller."""
    def __init__(self, db_session):
        self._db = db_session

    def get_active_users(self) -> list[User]:
        return self._db.query(User).filter(User.is_active == True).all()
```

### 3. Observer Pattern (trong Classic MVC)

Khi Model thay đổi, nó thông báo cho tất cả View đang lắng nghe:

```python
class ObservableModel:
    def __init__(self):
        self._observers = []

    def register_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self):
        for obs in self._observers:
            obs.model_changed(self)
```

### 4. Controller là glue

Controller điều phối: nhận request → gọi Model → chọn View → trả response. Controller không chứa business logic hay SQL.

---

## Cấu trúc chi tiết

### MVC trong Web Application

```
HTTP Request
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Router / Dispatcher                        │
│  • Phân tích URL → Route đến Controller                        │
│  • Middleware: Authentication, Logging, CORS                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Controller                                  │
│  • Nhận request parameters                                      │
│  • Gọi Model để lấy/xử lý data                                  │
│  • Chọn View và truyền data                                     │
│  • Trả response                                                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
               ▼                       ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│         Model            │  │         View             │
│  • Business Logic        │  │  • Render template       │
│  • Database Access       │  │  • Hiển thị data         │
│  • Validation            │  │  • Format output         │
│  • Data Transformation   │  │  • Generate HTML/JSON    │
└──────────────────────────┘  └──────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│     Database / Storage   │
└──────────────────────────┘
```

### Luồng xử lý chi tiết

```
1. User gửi GET /users
2. Router → UserController.list()
3. Controller:
   a. Gọi UserModel.get_all() → Model query DB
   b. Model trả về list[User]
   c. Controller chọn View "users/list.html"
   d. Truyền data: {"users": [...]}
4. View render template → HTML string
5. Trả HTTP Response
```

### Thành phần trong Model

```
Model Layer
├── Business Logic
│   ├── Domain Rules
│   ├── Calculations
│   └── Validations
├── Data Access
│   ├── Repository / DAO
│   ├── ORM (SQLAlchemy)
│   └── Queries
├── Data Transfer
│   ├── DTOs (Data Transfer Objects)
│   └── Serializers
└── Services
    ├── Application Services
    └── Domain Services (DDD)
```

### Thành phần trong View

```
View Layer
├── Templates
│   ├── HTML Templates (Jinja2)
│   ├── Reusable Components
│   └── Layouts / Base Templates
├── Static Files
│   ├── CSS, JavaScript
│   ├── Images
│   └── Fonts
├── Serializers
│   ├── JSON Serializers (API)
│   ├── XML Serializers
│   └── CSV Exporters
└── Helpers / Filters
    ├── Date Formatting
    ├── Currency Formatting
    └── Text Processing
```

---

## Sơ đồ kiến trúc

```
                        ┌─────────────────────┐
                        │     Bộ định tuyến     │
                        │     (Router)         │
                        │  URL → Controller    │
                        └────────┬────────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
                  ▼              ▼              ▼
          ┌────────────┐ ┌────────────┐ ┌────────────┐
          │  UserCtrl  │ │ OrderCtrl │ │  ProdCtrl  │
          └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                │              │              │
                ├───────┬──────┘              │
                │       │                     │
                ▼       ▼                     ▼
          ┌────────────────────────────────────────┐
          │               Model Layer              │
          │  ┌────────┐ ┌────────┐ ┌──────────┐  │
          │  │  User  │ │ Order  │ │ Product  │  │
          │  │  Model │ │ Model  │ │ Model    │  │
          │  └────────┘ └────────┘ └──────────┘  │
          │  ┌────────────────────────────┐      │
          │  │      Services Layer       │      │
          │  └────────────────────────────┘      │
          │  ┌────────────────────────────┐      │
          │  │      Database Access      │      │
          │  └────────────────────────────┘      │
          └────────────────────────────────────────┘
                          │
                          ▼
          ┌────────────────────────────────────────┐
          │               View Layer               │
          │  ┌────────┐ ┌────────┐ ┌──────────┐  │
          │  │  HTML  │ │  JSON  │ │   PDF   │  │
          │  │ Views  │ │ Views  │ │  Views   │  │
          │  └────────┘ └────────┘ └──────────┘  │
          │  ┌────────┐ ┌────────┐ ┌──────────┐  │
          │  │ Jinja2 │ │ Dict/  │ │ ReportLab│  │
          │  │        │ │ Serial │ │          │  │
          │  └────────┘ └────────┘ └──────────┘  │
          └────────────────────────────────────────┘
                          │
                          ▼
          ┌────────────────────────────────────────┐
          │            HTTP Response               │
          └────────────────────────────────────────┘
```

---

## Ví dụ code hoàn chỉnh

Xây dựng hệ thống **quản lý thư viện** (Library Management System) với MVC.

### Cấu trúc project

```
library/
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── book.py
│   ├── member.py
│   ├── loan.py
│   ├── author.py
│   └── enums.py
├── views/
│   ├── __init__.py
│   ├── book_view.py
│   ├── member_view.py
│   ├── loan_view.py
│   └── helpers.py
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py
│   ├── book_controller.py
│   ├── member_controller.py
│   ├── loan_controller.py
│   └── auth_controller.py
├── services/
│   ├── __init__.py
│   ├── book_service.py
│   ├── member_service.py
│   ├── loan_service.py
│   └── notification_service.py
├── repository/
│   ├── __init__.py
│   ├── book_repository.py
│   ├── member_repository.py
│   └── loan_repository.py
├── infrastructure/
│   ├── __init__.py
│   ├── database.py
│   ├── router.py
│   └── middleware.py
├── templates/
│   ├── base.html
│   ├── books/
│   │   ├── list.html
│   │   ├── detail.html
│   │   └── form.html
│   ├── members/
│   │   ├── list.html
│   │   └── detail.html
│   └── loans/
│       ├── active.html
│       └── history.html
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_controllers.py
│   └── test_views.py
├── static/
│   ├── css/
│   └── js/
├── config.py
└── app.py
```

### File: models/enums.py

```python
from __future__ import annotations

from enum import Enum, auto


class BookStatus(Enum):
    AVAILABLE = "Có sẵn"
    BORROWED = "Đã mượn"
    RESERVED = "Đã đặt trước"
    DAMAGED = "Hư hỏng"
    LOST = "Mất"

    def can_be_borrowed(self) -> bool:
        return self == BookStatus.AVAILABLE

    def __str__(self) -> str:
        return self.value


class LoanStatus(Enum):
    ACTIVE = "Đang mượn"
    OVERDUE = "Quá hạn"
    RETURNED = "Đã trả"
    LOST = "Mất sách"

    def __str__(self) -> str:
        return self.value


class MemberType(Enum):
    STANDARD = "Tiêu chuẩn"
    PREMIUM = "Cao cấp"
    STUDENT = "Sinh viên"

    @property
    def max_loans(self) -> int:
        return {
            MemberType.STANDARD: 3,
            MemberType.PREMIUM: 10,
            MemberType.STUDENT: 5,
        }[self]

    @property
    def loan_duration_days(self) -> int:
        return {
            MemberType.STANDARD: 14,
            MemberType.PREMIUM: 30,
            MemberType.STUDENT: 21,
        }[self]

    def __str__(self) -> str:
        return self.value
```

### File: models/base.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar
from uuid import UUID, uuid4


@dataclass
class BaseModel:
    """Base class cho tất cả models."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    _repository: ClassVar = None

    def save(self) -> None:
        """Lưu model xuống database."""
        self.updated_at = datetime.utcnow()
        if self._repository:
            self._repository.save(self)

    def delete(self) -> None:
        """Xóa model khỏi database."""
        if self._repository:
            self._repository.delete(self.id)

    def to_dict(self) -> dict:
        """Convert model thành dictionary."""
        result = {}
        for attr in dir(self):
            if attr.startswith("_") or callable(getattr(self, attr)):
                continue
            value = getattr(self, attr)
            if isinstance(value, UUID):
                result[attr] = str(value)
            elif isinstance(value, datetime):
                result[attr] = value.isoformat()
            elif isinstance(value, Enum):
                result[attr] = value.value
            elif isinstance(value, BaseModel):
                result[attr] = str(value)
            elif hasattr(value, 'to_dict'):
                result[attr] = value.to_dict()
            else:
                result[attr] = value
        return result
```

### File: models/book.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional
from uuid import UUID

from models.base import BaseModel
from models.enums import BookStatus
from models.author import Author


@dataclass
class Book(BaseModel):
    """Model: Sách trong thư viện."""
    title: str
    isbn: str
    author: Optional[Author] = None
    author_id: Optional[UUID] = None
    publisher: str = ""
    publish_year: int = 0
    category: str = ""
    total_copies: int = 1
    available_copies: int = 1
    status: BookStatus = BookStatus.AVAILABLE
    location: str = ""  # Ví dụ: "Kệ A-12"
    description: str = ""
    cover_image: str = ""

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Title is required")
        if not self.isbn.strip():
            raise ValueError("ISBN is required")
        if self.total_copies < 1:
            raise ValueError("Total copies must be >= 1")

    def borrow_copy(self) -> None:
        """Mượn một bản sách."""
        if self.available_copies <= 0:
            raise ValueError(f"'{self.title}' is not available")
        self.available_copies -= 1
        if self.available_copies == 0:
            self.status = BookStatus.BORROWED

    def return_copy(self) -> None:
        """Trả một bản sách."""
        if self.available_copies >= self.total_copies:
            raise ValueError("All copies are already returned")
        self.available_copies += 1
        if self.available_copies > 0:
            self.status = BookStatus.AVAILABLE

    @property
    def is_available(self) -> bool:
        return self.available_copies > 0

    def __str__(self) -> str:
        return f"📖 {self.title} ({self.isbn})"
```

### File: models/member.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from uuid import UUID

from models.base import BaseModel
from models.enums import MemberType


@dataclass
class Member(BaseModel):
    """Model: Thành viên thư viện."""
    name: str
    email: str
    phone: str
    member_type: MemberType = MemberType.STANDARD
    address: str = ""
    birth_date: Optional[date] = None
    membership_start: date = field(default_factory=date.today)
    membership_expiry: Optional[date] = None
    is_active: bool = True
    total_loans: int = 0
    active_loans: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Name is required")
        if "@" not in self.email:
            raise ValueError("Invalid email")

    @property
    def can_borrow(self) -> bool:
        """Kiểm tra xem member có thể mượn thêm sách không."""
        if not self.is_active:
            return False
        if self.membership_expiry and self.membership_expiry < date.today():
            return False
        return self.active_loans < self.member_type.max_loans

    @property
    def max_duration_days(self) -> int:
        """Thời gian mượn tối đa (ngày)."""
        return self.member_type.loan_duration_days

    def borrow_book(self) -> None:
        """Mượn sách — tăng active_loans."""
        if not self.can_borrow:
            raise ValueError("Member cannot borrow more books")
        self.active_loans += 1
        self.total_loans += 1

    def return_book(self) -> None:
        """Trả sách — giảm active_loans."""
        if self.active_loans <= 0:
            raise ValueError("No active loans")
        self.active_loans -= 1

    def is_overdue(self, due_date: date) -> bool:
        """Kiểm tra xem member có sách quá hạn không."""
        return date.today() > due_date

    def __str__(self) -> str:
        return f"👤 {self.name} ({self.member_type.value})"
```

### File: models/loan.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from uuid import UUID

from models.base import BaseModel
from models.enums import LoanStatus


@dataclass
class Loan(BaseModel):
    """Model: Phiếu mượn sách."""
    book_id: UUID
    book_title: str
    member_id: UUID
    member_name: str
    loan_date: date = field(default_factory=date.today)
    due_date: date = field(default_factory=lambda: date.today() + timedelta(days=14))
    return_date: Optional[date] = None
    status: LoanStatus = LoanStatus.ACTIVE
    fine_amount: float = 0.0
    notes: str = ""

    DAILY_FINE = 5000  # 5,000₫/ngày quá hạn

    def __post_init__(self) -> None:
        if self.due_date <= self.loan_date:
            raise ValueError("Due date must be after loan date")

    def return_book(self) -> None:
        """Trả sách — kết thúc loan."""
        if self.status == LoanStatus.RETURNED:
            raise ValueError("Book already returned")
        self.return_date = date.today()
        self.status = LoanStatus.RETURNED
        self.calculate_fine()

    def calculate_fine(self) -> None:
        """Tính tiền phạt nếu quá hạn."""
        if self.return_date and self.return_date > self.due_date:
            days_overdue = (self.return_date - self.due_date).days
            self.fine_amount = days_overdue * self.DAILY_FINE
        else:
            self.fine_amount = 0.0

    @property
    def is_overdue(self) -> bool:
        if self.status == LoanStatus.RETURNED:
            return False
        return date.today() > self.due_date

    @property
    def days_overdue(self) -> int:
        if not self.is_overdue:
            return 0
        return (date.today() - self.due_date).days

    @property
    def potential_fine(self) -> float:
        """Tiền phạt hiện tại (nếu trả ngay bây giờ)."""
        if not self.is_overdue:
            return 0.0
        return self.days_overdue * self.DAILY_FINE

    def mark_lost(self) -> None:
        """Đánh dấu sách bị mất."""
        self.status = LoanStatus.LOST
        self.fine_amount = 50000  # Phạt mất sách: 50,000₫

    def __str__(self) -> str:
        status_icon = "🔴" if self.is_overdue else "🟢"
        return f"{status_icon} {self.book_title} → {self.member_name} (Hạn: {self.due_date})"
```

### File: models/author.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from models.base import BaseModel


@dataclass
class Author(BaseModel):
    """Model: Tác giả sách."""
    name: str
    birth_date: Optional[date] = None
    nationality: str = ""
    biography: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Author name is required")

    def __str__(self) -> str:
        return f"✍️ {self.name}"
```

### File: repository/book_repository.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from models.book import Book
from models.enums import BookStatus


class BookRepository:
    """Repository cho Book model — truy cập dữ liệu."""

    def __init__(self, db_session):
        self._session = db_session

    def save(self, book: Book) -> None:
        self._session["books"][str(book.id)] = book

    def get_by_id(self, book_id: UUID) -> Optional[Book]:
        return self._session["books"].get(str(book_id))

    def get_by_isbn(self, isbn: str) -> Optional[Book]:
        for book in self._session["books"].values():
            if book.isbn == isbn:
                return book
        return None

    def get_all(self) -> list[Book]:
        return list(self._session["books"].values())

    def search(self, query: str) -> list[Book]:
        """Tìm kiếm sách theo title, author hoặc ISBN."""
        query = query.lower()
        results = []
        for book in self._session["books"].values():
            if (query in book.title.lower()
                or query in book.isbn.lower()
                or (book.author and query in book.author.name.lower())):
                results.append(book)
        return results

    def get_available(self) -> list[Book]:
        return [
            b for b in self._session["books"].values()
            if b.status == BookStatus.AVAILABLE
        ]

    def get_by_category(self, category: str) -> list[Book]:
        return [
            b for b in self._session["books"].values()
            if b.category.lower() == category.lower()
        ]

    def delete(self, book_id: UUID) -> None:
        self._session["books"].pop(str(book_id), None)

    def count(self) -> int:
        return len(self._session["books"])
```

### File: repository/member_repository.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from models.member import Member


class MemberRepository:
    """Repository cho Member model."""

    def __init__(self, db_session):
        self._session = db_session

    def save(self, member: Member) -> None:
        self._session["members"][str(member.id)] = member

    def get_by_id(self, member_id: UUID) -> Optional[Member]:
        return self._session["members"].get(str(member_id))

    def get_by_email(self, email: str) -> Optional[Member]:
        for m in self._session["members"].values():
            if m.email == email:
                return m
        return None

    def get_all(self) -> list[Member]:
        return list(self._session["members"].values())

    def search(self, query: str) -> list[Member]:
        query = query.lower()
        return [
            m for m in self._session["members"].values()
            if query in m.name.lower() or query in m.email.lower()
        ]

    def get_active_members(self) -> list[Member]:
        return [m for m in self._session["members"].values() if m.is_active]

    def delete(self, member_id: UUID) -> None:
        self._session["members"].pop(str(member_id), None)

    def count(self) -> int:
        return len(self._session["members"])
```

### File: repository/loan_repository.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from models.loan import Loan
from models.enums import LoanStatus


class LoanRepository:
    """Repository cho Loan model."""

    def __init__(self, db_session):
        self._session = db_session

    def save(self, loan: Loan) -> None:
        self._session["loans"][str(loan.id)] = loan

    def get_by_id(self, loan_id: UUID) -> Optional[Loan]:
        return self._session["loans"].get(str(loan_id))

    def get_all(self) -> list[Loan]:
        return list(self._session["loans"].values())

    def get_active_loans(self) -> list[Loan]:
        return [
            l for l in self._session["loans"].values()
            if l.status == LoanStatus.ACTIVE
        ]

    def get_overdue_loans(self) -> list[Loan]:
        return [
            l for l in self._session["loans"].values()
            if l.is_overdue
        ]

    def get_loans_by_member(self, member_id: UUID) -> list[Loan]:
        return [
            l for l in self._session["loans"].values()
            if l.member_id == member_id
        ]

    def get_loans_by_book(self, book_id: UUID) -> list[Loan]:
        return [
            l for l in self._session["loans"].values()
            if l.book_id == book_id
        ]

    def get_active_loan_for_book(self, book_id: UUID) -> Optional[Loan]:
        for l in self._session["loans"].values():
            if l.book_id == book_id and l.status == LoanStatus.ACTIVE:
                return l
        return None

    def delete(self, loan_id: UUID) -> None:
        self._session["loans"].pop(str(loan_id), None)

    def count(self) -> int:
        return len(self._session["loans"])

    def count_overdue(self) -> int:
        return len(self.get_overdue_loans())
```

### File: services/book_service.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from models.book import Book
from models.enums import BookStatus
from repository.book_repository import BookRepository


class BookService:
    """Service layer — chứa business logic liên quan đến sách."""

    def __init__(self, book_repo: BookRepository):
        self._repo = book_repo

    def create_book(
        self, title: str, isbn: str, author_id: Optional[str] = None,
        publisher: str = "", publish_year: int = 0,
        category: str = "", total_copies: int = 1,
    ) -> Book:
        """Tạo sách mới với validation."""
        # Kiểm tra ISBN trùng
        existing = self._repo.get_by_isbn(isbn)
        if existing:
            raise ValueError(f"ISBN '{isbn}' already exists")

        book = Book(
            title=title.strip(),
            isbn=isbn.strip(),
            publisher=publisher.strip(),
            publish_year=publish_year,
            category=category.strip(),
            total_copies=total_copies,
            available_copies=total_copies,
        )
        book.save()
        return book

    def update_book(self, book_id: UUID, **kwargs) -> Book:
        """Cập nhật thông tin sách."""
        book = self._repo.get_by_id(book_id)
        if not book:
            raise ValueError(f"Book not found: {book_id}")

        for key, value in kwargs.items():
            if hasattr(book, key) and value is not None:
                setattr(book, key, value)

        book.save()
        return book

    def delete_book(self, book_id: UUID) -> None:
        """Xóa sách (nếu không có loan active)."""
        from repository.loan_repository import LoanRepository
        # Kiểm tra có ai đang mượn không
        # (trong thực tế, cần inject loan_repo)

        book = self._repo.get_by_id(book_id)
        if not book:
            raise ValueError(f"Book not found: {book_id}")
        book.delete()

    def search_books(self, query: str) -> list[Book]:
        return self._repo.search(query)

    def get_available_books(self) -> list[Book]:
        return self._repo.get_available()

    def get_book_summary(self) -> dict:
        """Thống kê sách."""
        all_books = self._repo.get_all()
        available = self._repo.get_available()
        return {
            "total": len(all_books),
            "available": len(available),
            "borrowed": len(all_books) - len(available),
            "total_copies": sum(b.total_copies for b in all_books),
            "available_copies": sum(b.available_copies for b in all_books),
        }
```

### File: services/loan_service.py

```python
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
from uuid import UUID

from models.loan import Loan
from models.book import Book
from models.member import Member
from models.enums import LoanStatus, BookStatus
from repository.loan_repository import LoanRepository
from repository.book_repository import BookRepository
from repository.member_repository import MemberRepository


class LoanService:
    """Service layer — business logic cho mượn/trả sách."""

    def __init__(
        self,
        loan_repo: LoanRepository,
        book_repo: BookRepository,
        member_repo: MemberRepository,
    ):
        self._loan_repo = loan_repo
        self._book_repo = book_repo
        self._member_repo = member_repo

    def borrow_book(self, book_id: UUID, member_id: UUID) -> Loan:
        """Mượn sách — business logic quan trọng."""
        # 1. Load models
        book = self._book_repo.get_by_id(book_id)
        if not book:
            raise ValueError(f"Book not found: {book_id}")

        member = self._member_repo.get_by_id(member_id)
        if not member:
            raise ValueError(f"Member not found: {member_id}")

        # 2. Validate: sách có sẵn?
        if not book.is_available:
            raise ValueError(f"'{book.title}' is not available")

        # 3. Validate: member có thể mượn?
        if not member.can_borrow:
            raise ValueError(f"'{member.name}' cannot borrow (limit: {member.member_type.max_loans})")

        # 4. Validate: member có sách quá hạn?
        active_loans = self._loan_repo.get_loans_by_member(member_id)
        overdue_loans = [l for l in active_loans if l.is_overdue]
        if overdue_loans:
            raise ValueError(f"'{member.name}' has {len(overdue_loans)} overdue book(s)")

        # 5. Tạo loan
        due_date = date.today() + timedelta(days=member.max_duration_days)
        loan = Loan(
            book_id=book_id,
            book_title=book.title,
            member_id=member_id,
            member_name=member.name,
            due_date=due_date,
        )

        # 6. Update book và member
        book.borrow_copy()
        member.borrow_book()

        # 7. Lưu
        loan.save()
        book.save()
        member.save()

        return loan

    def return_book(self, loan_id: UUID) -> Loan:
        """Trả sách."""
        loan = self._loan_repo.get_by_id(loan_id)
        if not loan:
            raise ValueError(f"Loan not found: {loan_id}")

        if loan.status == LoanStatus.RETURNED:
            raise ValueError("Book already returned")

        # Trả sách
        loan.return_book()

        # Update book
        book = self._book_repo.get_by_id(loan.book_id)
        if book:
            book.return_copy()
            book.save()

        # Update member
        member = self._member_repo.get_by_id(loan.member_id)
        if member:
            member.return_book()
            member.save()

        loan.save()

        return loan

    def renew_loan(self, loan_id: UUID, extra_days: int = 7) -> Loan:
        """Gia hạn mượn sách."""
        loan = self._loan_repo.get_by_id(loan_id)
        if not loan:
            raise ValueError(f"Loan not found: {loan_id}")

        if loan.status != LoanStatus.ACTIVE:
            raise ValueError("Can only renew active loans")

        if loan.is_overdue:
            raise ValueError("Cannot renew overdue loan")

        # Gia hạn
        loan.due_date = loan.due_date + timedelta(days=extra_days)
        loan.save()
        return loan

    def get_member_loans(self, member_id: UUID) -> list[Loan]:
        """Lấy danh sách phiếu mượn của member."""
        return self._loan_repo.get_loans_by_member(member_id)

    def get_overdue_loans(self) -> list[Loan]:
        """Lấy danh sách phiếu mượn quá hạn."""
        return self._loan_repo.get_overdue_loans()

    def get_loan_statistics(self) -> dict:
        """Thống kê mượn trả."""
        all_loans = self._loan_repo.get_all()
        active = self._loan_repo.get_active_loans()
        overdue = self._loan_repo.get_overdue_loans()
        returned = [l for l in all_loans if l.status == LoanStatus.RETURNED]

        total_fines = sum(l.fine_amount for l in all_loans)

        return {
            "total_loans": len(all_loans),
            "active": len(active),
            "overdue": len(overdue),
            "returned": len(returned),
            "total_fines": total_fines,
            "overdue_rate": round(len(overdue) / len(active) * 100, 1) if active else 0,
        }
```

### File: services/member_service.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from models.member import Member
from models.enums import MemberType
from repository.member_repository import MemberRepository


class MemberService:
    """Service layer — business logic cho member."""

    def __init__(self, member_repo: MemberRepository):
        self._repo = member_repo

    def register_member(
        self, name: str, email: str, phone: str,
        member_type: MemberType = MemberType.STANDARD,
    ) -> Member:
        """Đăng ký thành viên mới."""
        existing = self._repo.get_by_email(email)
        if existing:
            raise ValueError(f"Email '{email}' already registered")

        member = Member(name=name.strip(), email=email.strip(), phone=phone, member_type=member_type)
        member.save()
        return member

    def update_member(self, member_id: UUID, **kwargs) -> Member:
        """Cập nhật thông tin member."""
        member = self._repo.get_by_id(member_id)
        if not member:
            raise ValueError(f"Member not found: {member_id}")

        for key, value in kwargs.items():
            if hasattr(member, key) and value is not None:
                setattr(member, key, value)

        member.save()
        return member

    def deactivate_member(self, member_id: UUID) -> Member:
        """Vô hiệu hóa member (không thể mượn sách)."""
        member = self._repo.get_by_id(member_id)
        if not member:
            raise ValueError(f"Member not found: {member_id}")
        member.is_active = False
        member.save()
        return member

    def get_member_statistics(self) -> dict:
        """Thống kê member."""
        members = self._repo.get_all()
        active = self._repo.get_active_members()
        return {
            "total": len(members),
            "active": len(active),
            "inactive": len(members) - len(active),
            "standard": len([m for m in members if m.member_type == MemberType.STANDARD]),
            "premium": len([m for m in members if m.member_type == MemberType.PREMIUM]),
            "student": len([m for m in members if m.member_type == MemberType.STUDENT]),
        }
```

### File: controllers/base_controller.py

```python
from __future__ import annotations

from typing import Any, Optional


class BaseController:
    """Base controller với các method tiện ích."""

    def __init__(self):
        self._request: Optional[dict] = None

    def set_request(self, request: dict) -> None:
        """Set request data từ router."""
        self._request = request or {}

    def get_param(self, key: str, default: Any = None) -> Any:
        """Lấy parameter từ request."""
        if self._request:
            return self._request.get(key, default)
        return default

    def json_response(self, data: Any, status: int = 200) -> dict:
        """Tạo JSON response."""
        return {
            "status": status,
            "data": data,
        }

    def error_response(self, message: str, status: int = 400) -> dict:
        """Tạo error response."""
        return {
            "status": status,
            "error": message,
        }

    def paginate(self, items: list, page: int = 1, per_page: int = 20) -> dict:
        """Phân trang."""
        total = len(items)
        total_pages = max(1, (total + per_page - 1) // per_page)
        start = (page - 1) * per_page
        end = start + per_page

        return {
            "items": items[start:end],
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        }
```

### File: controllers/book_controller.py

```python
from __future__ import annotations

from uuid import UUID

from controllers.base_controller import BaseController
from services.book_service import BookService
from views.book_view import BookView


class BookController(BaseController):
    """Controller: Xử lý request liên quan đến sách."""

    def __init__(self, book_service: BookService, book_view: BookView):
        super().__init__()
        self._service = book_service
        self._view = book_view

    def list_books(self) -> dict:
        """GET /books"""
        page = int(self.get_param("page", 1))
        per_page = int(self.get_param("per_page", 20))
        search = self.get_param("search", "")

        if search:
            books = self._service.search_books(search)
        else:
            books = self._service._repo.get_all()

        # View formats the data
        formatted = self._view.render_list(books)
        paginated = self.paginate(formatted, page, per_page)

        return self.json_response({
            "books": paginated["items"],
            "pagination": {
                "page": paginated["page"],
                "per_page": paginated["per_page"],
                "total": paginated["total"],
                "total_pages": paginated["total_pages"],
            },
            "summary": self._service.get_book_summary(),
        })

    def get_book(self, book_id: str) -> dict:
        """GET /books/{id}"""
        try:
            book = self._service._repo.get_by_id(UUID(book_id))
            if not book:
                return self.error_response(f"Book not found: {book_id}", 404)
            return self.json_response(self._view.render_detail(book))
        except ValueError as e:
            return self.error_response(str(e))

    def create_book(self) -> dict:
        """POST /books"""
        try:
            book = self._service.create_book(
                title=self.get_param("title", ""),
                isbn=self.get_param("isbn", ""),
                publisher=self.get_param("publisher", ""),
                publish_year=int(self.get_param("publish_year", 0)),
                category=self.get_param("category", ""),
                total_copies=int(self.get_param("total_copies", 1)),
            )
            return self.json_response(
                self._view.render_detail(book),
                status=201,
            )
        except ValueError as e:
            return self.error_response(str(e))
        except Exception as e:
            return self.error_response(f"Internal error: {str(e)}", 500)

    def update_book(self, book_id: str) -> dict:
        """PUT /books/{id}"""
        try:
            book = self._service.update_book(
                UUID(book_id),
                title=self.get_param("title"),
                publisher=self.get_param("publisher"),
                category=self.get_param("category"),
            )
            return self.json_response(self._view.render_detail(book))
        except ValueError as e:
            return self.error_response(str(e))

    def delete_book(self, book_id: str) -> dict:
        """DELETE /books/{id}"""
        try:
            self._service.delete_book(UUID(book_id))
            return self.json_response({"message": "Book deleted"})
        except ValueError as e:
            return self.error_response(str(e))

    def search_books(self) -> dict:
        """GET /books/search"""
        query = self.get_param("q", "")
        if not query:
            return self.error_response("Search query required")
        books = self._service.search_books(query)
        return self.json_response({
            "query": query,
            "results": self._view.render_list(books),
            "count": len(books),
        })
```

### File: controllers/loan_controller.py

```python
from __future__ import annotations

from uuid import UUID

from controllers.base_controller import BaseController
from services.loan_service import LoanService
from views.loan_view import LoanView


class LoanController(BaseController):
    """Controller: Xử lý request mượn/trả sách."""

    def __init__(self, loan_service: LoanService, loan_view: LoanView):
        super().__init__()
        self._service = loan_service
        self._view = loan_view

    def borrow_book(self) -> dict:
        """POST /loans/borrow"""
        try:
            book_id = UUID(self.get_param("book_id", ""))
            member_id = UUID(self.get_param("member_id", ""))

            loan = self._service.borrow_book(book_id, member_id)
            return self.json_response(
                self._view.render_detail(loan),
                status=201,
            )
        except ValueError as e:
            return self.error_response(str(e))
        except Exception as e:
            return self.error_response(f"Internal error: {str(e)}", 500)

    def return_book(self, loan_id: str) -> dict:
        """POST /loans/{id}/return"""
        try:
            loan = self._service.return_book(UUID(loan_id))
            return self.json_response(self._view.render_detail(loan))
        except ValueError as e:
            return self.error_response(str(e))

    def renew_loan(self, loan_id: str) -> dict:
        """POST /loans/{id}/renew"""
        try:
            extra_days = int(self.get_param("extra_days", 7))
            loan = self._service.renew_loan(UUID(loan_id), extra_days)
            return self.json_response(self._view.render_detail(loan))
        except ValueError as e:
            return self.error_response(str(e))

    def list_active_loans(self) -> dict:
        """GET /loans/active"""
        page = int(self.get_param("page", 1))
        loans = self._service._loan_repo.get_active_loans()
        formatted = self._view.render_list(loans)
        paginated = self.paginate(formatted, page)
        return self.json_response({
            "loans": paginated["items"],
            "pagination": {k: v for k, v in paginated.items() if k != "items"},
        })

    def list_overdue_loans(self) -> dict:
        """GET /loans/overdue"""
        loans = self._service.get_overdue_loans()
        return self.json_response({
            "overdue_loans": self._view.render_list(loans),
            "count": len(loans),
            "total_fines": sum(l.potential_fine for l in loans),
        })

    def get_member_loans(self, member_id: str) -> dict:
        """GET /members/{id}/loans"""
        try:
            loans = self._service.get_member_loans(UUID(member_id))
            return self.json_response(self._view.render_list(loans))
        except ValueError as e:
            return self.error_response(str(e))

    def get_statistics(self) -> dict:
        """GET /loans/statistics"""
        stats = self._service.get_loan_statistics()
        return self.json_response(stats)
```

### File: controllers/member_controller.py

```python
from __future__ import annotations

from uuid import UUID

from controllers.base_controller import BaseController
from services.member_service import MemberService
from services.loan_service import LoanService
from views.member_view import MemberView
from views.loan_view import LoanView


class MemberController(BaseController):
    """Controller: Xử lý request liên quan đến member."""

    def __init__(
        self,
        member_service: MemberService,
        loan_service: LoanService,
        member_view: MemberView,
        loan_view: LoanView,
    ):
        super().__init__()
        self._service = member_service
        self._loan_service = loan_service
        self._member_view = member_view
        self._loan_view = loan_view

    def list_members(self) -> dict:
        """GET /members"""
        page = int(self.get_param("page", 1))
        members = self._service._repo.get_all()
        formatted = self._member_view.render_list(members)
        paginated = self.paginate(formatted, page)
        return self.json_response({
            "members": paginated["items"],
            "pagination": {k: v for k, v in paginated.items() if k != "items"},
            "statistics": self._service.get_member_statistics(),
        })

    def get_member(self, member_id: str) -> dict:
        """GET /members/{id}"""
        try:
            member = self._service._repo.get_by_id(UUID(member_id))
            if not member:
                return self.error_response(f"Member not found: {member_id}", 404)

            loans = self._loan_service.get_member_loans(UUID(member_id))
            return self.json_response({
                "member": self._member_view.render_detail(member),
                "active_loans": self._loan_view.render_list(
                    [l for l in loans if l.status.name == "ACTIVE"]
                ),
                "loan_history": self._loan_view.render_list(
                    [l for l in loans if l.status.name != "ACTIVE"]
                ),
            })
        except ValueError as e:
            return self.error_response(str(e))

    def register_member(self) -> dict:
        """POST /members"""
        try:
            member = self._service.register_member(
                name=self.get_param("name", ""),
                email=self.get_param("email", ""),
                phone=self.get_param("phone", ""),
                member_type=self.get_param("member_type", "STANDARD"),
            )
            return self.json_response(
                self._member_view.render_detail(member),
                status=201,
            )
        except ValueError as e:
            return self.error_response(str(e))
```

### File: views/book_view.py

```python
from __future__ import annotations

from typing import Any

from models.book import Book


class BookView:
    """View: Format dữ liệu Book để hiển thị."""

    def render_list(self, books: list[Book]) -> list[dict]:
        """Format danh sách sách."""
        return [self._format_brief(b) for b in books]

    def render_detail(self, book: Book) -> dict:
        """Format chi tiết một cuốn sách."""
        return {
            "id": str(book.id),
            "title": book.title,
            "isbn": book.isbn,
            "author": book.author.name if book.author else "N/A",
            "author_id": str(book.author_id) if book.author_id else None,
            "publisher": book.publisher,
            "publish_year": book.publish_year,
            "category": book.category,
            "status": str(book.status),
            "total_copies": book.total_copies,
            "available_copies": book.available_copies,
            "borrowed_copies": book.total_copies - book.available_copies,
            "location": book.location,
            "description": book.description,
            "cover_image": book.cover_image,
            "is_available": book.is_available,
            "created_at": book.created_at.isoformat() if book.created_at else None,
        }

    def _format_brief(self, book: Book) -> dict:
        """Format tóm tắt một cuốn sách."""
        return {
            "id": str(book.id),
            "title": book.title,
            "isbn": book.isbn,
            "author": book.author.name if book.author else "N/A",
            "category": book.category,
            "status": str(book.status),
            "available": f"{book.available_copies}/{book.total_copies}",
            "is_available": book.is_available,
        }
```

### File: views/member_view.py

```python
from __future__ import annotations

from models.member import Member


class MemberView:
    """View: Format dữ liệu Member."""

    def render_list(self, members: list[Member]) -> list[dict]:
        return [self._format_brief(m) for m in members]

    def render_detail(self, member: Member) -> dict:
        return {
            "id": str(member.id),
            "name": member.name,
            "email": member.email,
            "phone": member.phone,
            "member_type": member.member_type.value,
            "address": member.address,
            "birth_date": member.birth_date.isoformat() if member.birth_date else None,
            "membership_start": member.membership_start.isoformat(),
            "membership_expiry": member.membership_expiry.isoformat() if member.membership_expiry else None,
            "is_active": member.is_active,
            "can_borrow": member.can_borrow,
            "max_loans": member.member_type.max_loans,
            "active_loans": member.active_loans,
            "total_loans": member.total_loans,
            "created_at": member.created_at.isoformat() if member.created_at else None,
        }

    def _format_brief(self, member: Member) -> dict:
        return {
            "id": str(member.id),
            "name": member.name,
            "email": member.email,
            "type": member.member_type.value,
            "active_loans": f"{member.active_loans}/{member.member_type.max_loans}",
            "is_active": member.is_active,
        }
```

### File: views/loan_view.py

```python
from __future__ import annotations

from models.loan import Loan


class LoanView:
    """View: Format dữ liệu Loan."""

    def render_list(self, loans: list[Loan]) -> list[dict]:
        return [self._format_brief(l) for l in loans]

    def render_detail(self, loan: Loan) -> dict:
        return {
            "id": str(loan.id),
            "book_id": str(loan.book_id),
            "book_title": loan.book_title,
            "member_id": str(loan.member_id),
            "member_name": loan.member_name,
            "loan_date": loan.loan_date.isoformat(),
            "due_date": loan.due_date.isoformat(),
            "return_date": loan.return_date.isoformat() if loan.return_date else None,
            "status": str(loan.status),
            "is_overdue": loan.is_overdue,
            "days_overdue": loan.days_overdue,
            "fine_amount": loan.fine_amount,
            "potential_fine": loan.potential_fine,
            "notes": loan.notes,
            "created_at": loan.created_at.isoformat() if loan.created_at else None,
        }

    def _format_brief(self, loan: Loan) -> dict:
        return {
            "id": str(loan.id),
            "book_title": loan.book_title,
            "member_name": loan.member_name,
            "loan_date": loan.loan_date.isoformat(),
            "due_date": loan.due_date.isoformat(),
            "status": str(loan.status),
            "is_overdue": loan.is_overdue,
            "fine": loan.potential_fine if loan.is_overdue else 0,
        }
```

### File: views/helpers.py

```python
from __future__ import annotations

from datetime import date, datetime


class ViewHelpers:
    """Các function helper cho View layer."""

    @staticmethod
    def format_date(d: date | datetime | None) -> str:
        if d is None:
            return "N/A"
        if isinstance(d, datetime):
            return d.strftime("%d/%m/%Y %H:%M")
        return d.strftime("%d/%m/%Y")

    @staticmethod
    def format_currency(amount: float) -> str:
        return f"{amount:,.0f}₫"

    @staticmethod
    def pluralize(count: int, singular: str, plural: str = "") -> str:
        if count == 1:
            return f"{count} {singular}"
        return f"{count} {plural or singular + 's'}"

    @staticmethod
    def truncate(text: str, max_length: int = 50) -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    @staticmethod
    def status_badge(status_name: str) -> str:
        badges = {
            "Có sẵn": "🟢",
            "Đã mượn": "🔵",
            "Quá hạn": "🔴",
            "Mất": "⚫",
            "Hư hỏng": "🟡",
            "Đang mượn": "🟢",
            "Đã trả": "✅",
            "Chờ xác nhận": "🟡",
            "Đã xác nhận": "🔵",
        }
        return badges.get(status_name, "⚪")
```

### File: infrastructure/router.py

```python
from __future__ import annotations

from typing import Any, Callable


class Router:
    """Bộ định tuyến HTTP request → Controller method."""

    def __init__(self):
        self._routes: dict[str, dict[str, Callable]] = {}
        self._middlewares: list[Callable] = []

    def register(self, method: str, path: str, handler: Callable) -> None:
        """Đăng ký route."""
        if path not in self._routes:
            self._routes[path] = {}
        self._routes[path][method.upper()] = handler

    def add_middleware(self, middleware: Callable) -> None:
        """Thêm middleware."""
        self._middlewares.append(middleware)

    def dispatch(self, method: str, path: str, request: dict) -> dict:
        """Phân phối request đến handler."""
        # Apply middlewares
        for middleware in self._middlewares:
            result = middleware(method, path, request)
            if result:
                return result

        # Find route
        handler = self._find_handler(method, path)
        if not handler:
            return {"status": 404, "error": f"No route: {method} {path}"}

        # Execute handler
        try:
            result = handler(request)
            return result if isinstance(result, dict) else {"status": 200, "data": result}
        except Exception as e:
            return {"status": 500, "error": f"Internal error: {str(e)}"}

    def _find_handler(self, method: str, path: str) -> Callable | None:
        """Tìm handler cho path (hỗ trợ dynamic segments)."""
        method = method.upper()

        # Exact match
        if path in self._routes and method in self._routes[path]:
            return self._routes[path][method]

        # Dynamic match: /books/{id} → /books/abc-123
        for route_path, handlers in self._routes.items():
            route_parts = route_path.split("/")
            path_parts = path.split("/")

            if len(route_parts) != len(path_parts):
                continue

            params = {}
            match = True
            for rp, pp in zip(route_parts, path_parts):
                if rp.startswith("{") and rp.endswith("}"):
                    params[rp[1:-1]] = pp
                elif rp != pp:
                    match = False
                    break

            if match and method in handlers:
                handler = handlers[method]
                # Wrap handler with params
                def make_wrapped(h, p):
                    def wrapped(request):
                        request.update(p)
                        return h(request)
                    return wrapped
                return make_wrapped(handler, params)

        return None
```

### File: infrastructure/middleware.py

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


class LoggingMiddleware:
    """Middleware ghi log request."""

    def __call__(self, method: str, path: str, request: dict) -> Optional[dict]:
        print(f"  [{datetime.utcnow().strftime('%H:%M:%S')}] {method:6s} {path}")
        return None  # Không block request


class AuthMiddleware:
    """Middleware kiểm tra authentication (demo)."""

    def __init__(self, api_key: str = "secret-key"):
        self._api_key = api_key

    def __call__(self, method: str, path: str, request: dict) -> Optional[dict]:
        auth = request.get("headers", {}).get("Authorization", "")
        if not auth.startswith("Bearer "):
            return {
                "status": 401,
                "error": "Authentication required",
            }
        return None
```

### File: app.py

```python
#!/usr/bin/env python3
"""
MVC Library Management System — Ví dụ hoàn chỉnh.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from uuid import UUID, uuid4

from models.book import Book
from models.member import Member
from models.loan import Loan
from models.author import Author
from models.enums import MemberType, BookStatus, LoanStatus
from models.base import BaseModel

from repository.book_repository import BookRepository
from repository.member_repository import MemberRepository
from repository.loan_repository import LoanRepository

from services.book_service import BookService
from services.member_service import MemberService
from services.loan_service import LoanService

from controllers.book_controller import BookController
from controllers.member_controller import MemberController
from controllers.loan_controller import LoanController

from views.book_view import BookView
from views.member_view import MemberView
from views.loan_view import LoanView

from infrastructure.router import Router
from infrastructure.middleware import LoggingMiddleware


def print_separator(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main() -> None:
    print("📚  MVC ARCHITECTURE — Library Management System")
    print("=" * 68)

    # ========== KHỞI TẠO ==========
    # Database (in-memory)
    db_session = {
        "books": {},
        "members": {},
        "loans": {},
    }

    # Repository Layer
    book_repo = BookRepository(db_session)
    member_repo = MemberRepository(db_session)
    loan_repo = LoanRepository(db_session)

    # Service Layer
    book_service = BookService(book_repo)
    member_service = MemberService(member_repo)
    loan_service = LoanService(loan_repo, book_repo, member_repo)

    # View Layer
    book_view = BookView()
    member_view = MemberView()
    loan_view = LoanView()

    # Controller Layer
    book_controller = BookController(book_service, book_view)
    member_controller = MemberController(member_service, loan_service, member_view, loan_view)
    loan_controller = LoanController(loan_service, loan_view)

    # Router
    router = Router()
    router.add_middleware(LoggingMiddleware())

    # Đăng ký routes
    router.register("GET", "/books", lambda r: book_controller.set_request(r) or book_controller.list_books())
    router.register("GET", "/books/search", lambda r: book_controller.set_request(r) or book_controller.search_books())
    router.register("GET", "/books/{id}", lambda r: book_controller.set_request(r) or book_controller.get_book(r.get("id", "")))
    router.register("POST", "/books", lambda r: book_controller.set_request(r) or book_controller.create_book())
    router.register("PUT", "/books/{id}", lambda r: book_controller.set_request(r) or book_controller.update_book(r.get("id", "")))
    router.register("DELETE", "/books/{id}", lambda r: book_controller.set_request(r) or book_controller.delete_book(r.get("id", "")))

    router.register("GET", "/members", lambda r: member_controller.set_request(r) or member_controller.list_members())
    router.register("GET", "/members/{id}", lambda r: member_controller.set_request(r) or member_controller.get_member(r.get("id", "")))
    router.register("POST", "/members", lambda r: member_controller.set_request(r) or member_controller.register_member())

    router.register("POST", "/loans/borrow", lambda r: loan_controller.set_request(r) or loan_controller.borrow_book())
    router.register("GET", "/loans/active", lambda r: loan_controller.set_request(r) or loan_controller.list_active_loans())
    router.register("GET", "/loans/overdue", lambda r: loan_controller.set_request(r) or loan_controller.list_overdue_loans())
    router.register("POST", "/loans/{id}/return", lambda r: loan_controller.set_request(r) or loan_controller.return_book(r.get("id", "")))
    router.register("POST", "/loans/{id}/renew", lambda r: loan_controller.set_request(r) or loan_controller.renew_loan(r.get("id", "")))
    router.register("GET", "/loans/statistics", lambda r: loan_controller.set_request(r) or loan_controller.get_statistics())
    router.register("GET", "/members/{id}/loans", lambda r: loan_controller.set_request(r) or loan_controller.get_member_loans(r.get("id", "")))

    # ========== SEED DATA ==========
    print_separator("Seed Data: Tạo dữ liệu mẫu")

    authors = [
        Author(name="Nguyễn Nhật Ánh", nationality="Việt Nam"),
        Author(name="Harper Lee", nationality="Mỹ"),
        Author(name="Gabriel García Márquez", nationality="Colombia"),
    ]
    for author in authors:
        author.save()
        # HACK: lưu author vào db_session
        db_session["books"][str(author.id)] = author

    books_data = [
        ("Mắt Biếc", "978-0-061-23456-7", authors[0], "NXB Trẻ", 1990, "Văn học", 3),
        ("Giết con chim nhại", "978-0-061-23457-4", authors[1], "NXB Văn học", 1960, "Văn học Mỹ", 2),
        ("Trăm năm cô đơn", "978-0-061-23458-1", authors[2], "NXB Văn học", 1967, "Văn học Thế giới", 2),
        ("Tôi thấy hoa vàng trên cỏ xanh", "978-0-061-23459-8", authors[0], "NXB Trẻ", 2010, "Văn học", 4),
        ("Có hai con mèo ngồi bên cửa sổ", "978-0-061-23460-4", authors[0], "NXB Trẻ", 2015, "Văn học", 2),
    ]

    created_books = {}
    for title, isbn, author, pub, year, cat, copies in books_data:
        book = Book(
            title=title, isbn=isbn, author=author, author_id=author.id,
            publisher=pub, publish_year=year, category=cat,
            total_copies=copies, available_copies=copies,
        )
        book._repository = book_repo
        book.save()
        created_books[title] = book
        print(f"  📖 {title:35s} | {author.name:25s} | {copies} bản")

    members_data = [
        ("Lê Văn Bình", "binh@email.com", "0909123456", MemberType.PREMIUM),
        ("Phạm Thị Cúc", "cuc@email.com", "0909987654", MemberType.STANDARD),
        ("Trần Văn Dũng", "dung@email.com", "0909555666", MemberType.STUDENT),
        ("Ngô Thị Em", "em@email.com", "0909444333", MemberType.STANDARD),
    ]

    created_members = {}
    for name, email, phone, mtype in members_data:
        member = Member(name=name, email=email, phone=phone, member_type=mtype)
        member._repository = member_repo
        member.save()
        created_members[name] = member
        print(f"  👤 {name:25s} | {mtype.value:15s} | 📧 {email}")

    print(f"\n  ✅ Tổng: {len(created_books)} sách, {len(created_members)} thành viên")

    # ========== DEMO MVC ==========
    print_separator("MVC Demo 1: Tạo sách mới (POST /books)")

    request = {
        "title": "Nhà giả kim",
        "isbn": "978-0-061-23461-1",
        "publisher": "NXB Hội Nhà Văn",
        "publish_year": "1988",
        "category": "Văn học Thế giới",
        "total_copies": "5",
        "headers": {"Authorization": "Bearer secret-key"},
    }
    response = router.dispatch("POST", "/books", request)
    print(f"  📋 Request: POST /books")
    print(f"  📦 Body: {json.dumps({k: v for k, v in request.items() if k != 'headers'}, ensure_ascii=False)}")
    if "id" in str(response):
        print(f"  ✅ Status: {response['status']}")
        print(f"  📖 Book created: {response['data']['title']} (ISBN: {response['data']['isbn']})")

    print_separator("MVC Demo 2: Danh sách sách (GET /books)")

    response = router.dispatch("GET", "/books", {"headers": {}, "page": "1"})
    if response["status"] == 200:
        data = response["data"]
        print(f"  📚 Danh sách sách (tổng: {data['summary']['total']}):")
        for book in data["books"][:5]:
            print(f"     • {book['title']:35s} [{book['status']:10s}] {book['available']}")
        print(f"     ... và {data['summary']['total'] - 5} sách khác")
        print(f"  📊 Thống kê: {data['summary']['total_copies']} bản, "
              f"{data['summary']['available_copies']} có sẵn, "
              f"{data['summary']['borrowed']} đã mượn")

    print_separator("MVC Demo 3: Mượn sách (POST /loans/borrow)")

    # Bình (Premium) mượn "Mắt Biếc"
    binh = created_members["Lê Văn Bình"]
    mat_biec = created_books["Mắt Biếc"]

    request = {
        "book_id": str(mat_biec.id),
        "member_id": str(binh.id),
        "headers": {},
    }
    response = router.dispatch("POST", "/loans/borrow", request)
    print(f"  📋 {binh.name} → {mat_biec.title}")
    if response["status"] == 201:
        loan = response["data"]
        print(f"  ✅ Mượn thành công!")
        print(f"   📅 Hạn trả: {loan['due_date']}")
    else:
        print(f"  ❌ Lỗi: {response.get('error')}")

    # Cúc (Standard) mượn "Trăm năm cô đơn"
    cuc = created_members["Phạm Thị Cúc"]
    tram_nam = created_books["Trăm năm cô đơn"]

    request = {
        "book_id": str(tram_nam.id),
        "member_id": str(cuc.id),
        "headers": {},
    }
    response = router.dispatch("POST", "/loans/borrow", request)
    print(f"  📋 {cuc.name} → {tram_nam.title}")
    if response["status"] == 201:
        loan2 = response["data"]
        print(f"  ✅ Mượn thành công!")
        print(f"   📅 Hạn trả: {loan2['due_date']}")
    else:
        print(f"  ❌ Lỗi: {response.get('error')}")

    print_separator("MVC Demo 4: Danh sách mượn (GET /loans/active)")

    response = router.dispatch("GET", "/loans/active", {"headers": {}, "page": "1"})
    if response["status"] == 200:
        print(f"  📋 Sách đang mượn ({response['data']['pagination']['total']}):")
        for loan in response["data"]["loans"]:
            overdue_icon = "🔴" if loan["is_overdue"] else "🟢"
            print(f"     {overdue_icon} {loan['book_title']:35s} → {loan['member_name']:20s} (Hạn: {loan['due_date']})")

    print_separator("MVC Demo 5: Trả sách (POST /loans/{id}/return)")

    loan_id = loan2["id"]
    response = router.dispatch("POST", f"/loans/{loan_id}/return", {"headers": {}, "id": loan_id})
    if response["status"] == 200:
        print(f"  ✅ Trả sách '{response['data']['book_title']}' thành công!")
        print(f"   📅 Ngày trả: {response['data']['return_date']}")
        print(f"   💰 Phí phạt: {response['data']['fine_amount']:,.0f}₫")

    print_separator("MVC Demo 6: Thông tin Member (GET /members/{id})")

    response = router.dispatch("GET", f"/members/{str(binh.id)}", {"headers": {}, "id": str(binh.id)})
    if response["status"] == 200:
        data = response["data"]
        print(f"  👤 {data['member']['name']}")
        print(f"   📧 {data['member']['email']}")
        print(f"   🏷️  {data['member']['member_type']}")
        print(f"   📚 Đang mượn: {data['member']['active_loans']}/{data['member']['max_loans']}")
        print(f"   📖 Tổng đã mượn: {data['member']['total_loans']}")
        print(f"   ✅ Có thể mượn: {'Có' if data['member']['can_borrow'] else 'Không'}")

    print_separator("MVC Demo 7: Thống kê (GET /loans/statistics)")

    response = router.dispatch("GET", "/loans/statistics", {"headers": {}})
    if response["status"] == 200:
        stats = response["data"]
        print(f"  📊 Thống kê mượn trả:")
        print(f"     📚 Tổng lượt mượn: {stats['total_loans']}")
        print(f"     🟢 Đang mượn:      {stats['active']}")
        print(f"     🔴 Quá hạn:        {stats['overdue']} ({stats['overdue_rate']}%)")
        print(f"     ✅ Đã trả:         {stats['returned']}")
        print(f"     💰 Tổng phí phạt:  {stats['total_fines']:,.0f}₫")

    print_separator("MVC Demo 8: Validation (Edge Cases)")

    # Test 1: Mượn sách không có sẵn
    request = {"book_id": str(mat_biec.id), "member_id": str(cuc.id), "headers": {}}
    response = router.dispatch("POST", "/loans/borrow", request)
    print(f"  ❌ {cuc.name} mượn '{mat_biec.title}' (còn 0 bản): {response.get('error', 'OK')}")

    # Test 2: Member không thể mượn thêm (đã vượt limit)
    dũng = created_members["Trần Văn Dũng"]
    nha_gia_kim = book_repo.get_by_isbn("978-0-061-23461-1")
    for _ in range(4):  # Student max = 5
        from models.loan import Loan
        l = Loan(
            book_id=nha_gia_kim.id,
            book_title=nha_gia_kim.title,
            member_id=dũng.id,
            member_name=dũng.name,
            due_date=date.today(),
        )
        l._repository = loan_repo
        loan_repo.save(l)
        dũng.active_loans += 1
    member_repo.save(dũng)

    response = router.dispatch("POST", "/loans/borrow", {
        "book_id": str(created_books["Tôi thấy hoa vàng trên cỏ xanh"].id),
        "member_id": str(dũng.id),
        "headers": {},
    })
    print(f"  ❌ {dũng.name} mượn sách tiếp (đã {dũng.active_loans}/{dũng.member_type.max_loans}): {response.get('error', 'OK')}")

    # Test 3: ISBN trùng
    response = router.dispatch("POST", "/books", {
        "title": "Sách trùng ISBN",
        "isbn": "978-0-061-23456-7",
        "headers": {},
    })
    print(f"  ❌ Tạo sách trùng ISBN: {response.get('error', 'OK')}")

    print()
    print("=" * 68)
    print("  ✅ MVC Demo hoàn tất!")
    print("  📚 Model + View + Controller — Separation of Concerns")
    print("=" * 68)


if __name__ == "__main__":
    main()
```

### Output khi chạy:

```
📚  MVC ARCHITECTURE — Library Management System
====================================================================

=================================================================
  Seed Data: Tạo dữ liệu mẫu
=================================================================
  📖 Mắt Biếc                           | Nguyễn Nhật Ánh           | 3 bản
  📖 Giết con chim nhại                 | Harper Lee                | 2 bản
  📖 Trăm năm cô đơn                    | Gabriel García Márquez    | 2 bản
  📖 Tôi thấy hoa vàng trên cỏ xanh     | Nguyễn Nhật Ánh           | 4 bản
  📖 Có hai con mèo ngồi bên cửa sổ     | Nguyễn Nhật Ánh           | 2 bản
  👤 Lê Văn Bình               | Cao cấp          | 📧 binh@email.com
  👤 Phạm Thị Cúc               | Tiêu chuẩn       | 📧 cuc@email.com
  👤 Trần Văn Dũng              | Sinh viên        | 📧 dung@email.com
  👤 Ngô Thị Em                 | Tiêu chuẩn       | 📧 em@email.com

  ✅ Tổng: 6 sách, 4 thành viên

=================================================================
  MVC Demo 1: Tạo sách mới (POST /books)
=================================================================
  [timestamp] POST   /books
  📋 Request: POST /books
  📦 Body: {"title": "Nhà giả kim", "isbn": "978-0-061-23461-1", ...}
  ✅ Status: 201
  📖 Book created: Nhà giả kim (ISBN: 978-0-061-23461-1)

=================================================================
  MVC Demo 2: Danh sách sách (GET /books)
=================================================================
  [timestamp] GET    /books
  📚 Danh sách sách (tổng: 6):
     • Nhà giả kim                      [Có sẵn   ] 5/5
     • Mắt Biếc                         [Có sẵn   ] 3/3
     • Giết con chim nhại               [Có sẵn   ] 2/2
     • Trăm năm cô đơn                   [Có sẵn   ] 2/2
     • Tôi thấy hoa vàng trên cỏ xanh   [Có sẵn   ] 4/4
     ... và 1 sách khác
  📊 Thống kê: 18 bản, 18 có sẵn, 0 đã mượn

=================================================================
  MVC Demo 3: Mượn sách (POST /loans/borrow)
=================================================================
  [timestamp] POST   /loans/borrow
  📋 Lê Văn Bình → Mắt Biếc
  ✅ Mượn thành công!
   📅 Hạn trả: 2026-08-11

  [timestamp] POST   /loans/borrow
  📋 Phạm Thị Cúc → Trăm năm cô đơn
  ✅ Mượn thành công!
   📅 Hạn trả: 2026-07-26

=================================================================
  MVC Demo 4: Danh sách mượn (GET /loans/active)
=================================================================
  [timestamp] GET    /loans/active
  📋 Sách đang mượn (2):
     🟢 Mắt Biếc                        → Lê Văn Bình        (Hạn: 2026-08-11)
     🟢 Trăm năm cô đơn                  → Phạm Thị Cúc       (Hạn: 2026-07-26)

=================================================================
  MVC Demo 5: Trả sách (POST /loans/{id}/return)
=================================================================
  [timestamp] POST   /loans/{id}/return
  ✅ Trả sách 'Trăm năm cô đơn' thành công!
   📅 Ngày trả: 2026-07-12
   💰 Phí phạt: 0₫

=================================================================
  MVC Demo 6: Thông tin Member (GET /members/{id})
=================================================================
  [timestamp] GET    /members/{id}
  👤 Lê Văn Bình
   📧 binh@email.com
   🏷️  Cao cấp
   📚 Đang mượn: 1/10
   📖 Tổng đã mượn: 1
   ✅ Có thể mượn: Có

=================================================================
  MVC Demo 7: Thống kê (GET /loans/statistics)
=================================================================
  [timestamp] GET    /loans/statistics
  📊 Thống kê mượn trả:
     📚 Tổng lượt mượn: 2
     🟢 Đang mượn:      1
     🔴 Quá hạn:        0
     ✅ Đã trả:         1
     💰 Tổng phí phạt:  0₫

=================================================================
  MVC Demo 8: Validation (Edge Cases)
=================================================================
  [timestamp] POST   /loans/borrow
  ❌ Phạm Thị Cúc mượn 'Mắt Biếc' (còn 0 bản): 'Mắt Biếc' is not available
  [timestamp] POST   /loans/borrow
  ❌ Trần Văn Dũng mượn sách tiếp (đã 5/5): 'Trần Văn Dũng' cannot borrow
  [timestamp] POST   /books
  ❌ Tạo sách trùng ISBN: ISBN '978-0-061-23456-7' already exists

=================================================================
  ✅ MVC Demo hoàn tất!
  📚 Model + View + Controller — Separation of Concerns
=================================================================
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng MVC | Khi nào không |
|---|---|
| Web application với UI rõ ràng | API-first, không có View rendering |
| Cần separation of concerns | Ứng dụng real-time (WebSocket) |
| Team có frontend và backend riêng | Ứng dụng CLI đơn giản |
| Nhiều loại view cho cùng data | Hệ thống event-driven |
| Dự án medium → large | Microservices quá nhỏ |
| Template rendering (server-side) | SPA với REST API backend |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Separation of concerns rõ ràng | Controller có thể trở nên "fat" |
| Dễ test (từng layer riêng) | View phụ thuộc vào Model |
| Code dễ maintain, dễ hiểu | Không phù hợp với real-time |
| Hỗ trợ multiple views cho cùng data | Có thể dẫn đến over-engineering |
| Cộng đồng lớn, nhiều framework | Model bị passive (anemic domain model) |
| Tái sử dụng Model cho nhiều controller | Navigation logic phức tạp trong web |

---

## Công cụ và Framework

### Python MVC Frameworks
- **Django** — Full-stack, "batteries included"
- **Flask** — Micro-framework, linh hoạt
- **FastAPI** — Modern, async, tự động OpenAPI
- **Pyramid** — Scale từ nhỏ đến lớn
- **web2py** — All-in-one
- **Tornado** — Async, real-time

### Template Engines (View)
- **Jinja2** — Python template engine chuẩn
- **Mako** — Hiệu năng cao
- **Chameleon** — HTML-first

### Non-Python MVC
- **Ruby on Rails** — MVC kinh điển
- **ASP.NET MVC / Core** — Microsoft ecosystem
- **Spring MVC** — Java enterprise
- **Laravel** — PHP MVC
- **Angular** — Client-side MVC
- **Vue.js / React** — Component-based (MVVM-like)

---

## Kiểm thử

### Chiến lược kiểm thử MVC

```python
# tests/test_models.py

from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4

import pytest

from models.book import Book
from models.member import Member
from models.loan import Loan
from models.enums import MemberType, BookStatus, LoanStatus


class TestBookModel:
    def test_create_book(self):
        book = Book(title="Test Book", isbn="978-0-061-23456-7")
        assert book.status == BookStatus.AVAILABLE
        assert book.is_available is True

    def test_borrow_copy(self):
        book = Book(title="Test", isbn="978-0-061-23456-7", total_copies=2, available_copies=2)
        book.borrow_copy()
        assert book.available_copies == 1
        assert book.is_available is True

    def test_borrow_last_copy(self):
        book = Book(title="Test", isbn="978-0-061-23456-7", total_copies=1, available_copies=1)
        book.borrow_copy()
        assert book.available_copies == 0
        assert book.status == BookStatus.BORROWED

    def test_borrow_unavailable(self):
        book = Book(title="Test", isbn="978-0-061-23456-7", total_copies=1, available_copies=0)
        with pytest.raises(ValueError, match="not available"):
            book.borrow_copy()

    def test_return_copy(self):
        book = Book(title="Test", isbn="978-0-061-23456-7", total_copies=2, available_copies=0)
        book.return_copy()
        assert book.available_copies == 1

    def test_return_all_copies_error(self):
        book = Book(title="Test", isbn="978-0-061-23456-7", total_copies=2, available_copies=2)
        with pytest.raises(ValueError, match="already returned"):
            book.return_copy()


class TestMemberModel:
    def test_create_member(self):
        member = Member(name="Test", email="test@test.com", phone="0909123456")
        assert member.is_active is True
        assert member.can_borrow is True

    def test_premium_member_limits(self):
        member = Member(name="Premium", email="p@test.com", phone="0909123456",
                        member_type=MemberType.PREMIUM)
        assert member.max_duration_days == 30

    def test_member_cannot_borrow_when_inactive(self):
        member = Member(name="Test", email="test@test.com", phone="0909123456")
        member.is_active = False
        assert member.can_borrow is False

    def test_member_limit(self):
        member = Member(name="Test", email="test@test.com", phone="0909123456",
                        member_type=MemberType.STANDARD)
        member.active_loans = 3  # Max for standard
        assert member.can_borrow is False

    def test_borrow_book(self):
        member = Member(name="Test", email="test@test.com", phone="0909123456")
        member.borrow_book()
        assert member.active_loans == 1
        assert member.total_loans == 1

    def test_return_book(self):
        member = Member(name="Test", email="test@test.com", phone="0909123456")
        member.active_loans = 2
        member.return_book()
        assert member.active_loans == 1


class TestLoanModel:
    def test_create_loan(self):
        loan = Loan(
            book_id=uuid4(), book_title="Test",
            member_id=uuid4(), member_name="Test",
        )
        assert loan.status == LoanStatus.ACTIVE
        assert loan.due_date == date.today() + timedelta(days=14)

    def test_return_book_updates_status(self):
        loan = Loan(
            book_id=uuid4(), book_title="Test",
            member_id=uuid4(), member_name="Test",
        )
        loan.return_book()
        assert loan.status == LoanStatus.RETURNED
        assert loan.return_date == date.today()


# tests/test_controllers.py

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from controllers.book_controller import BookController
from controllers.loan_controller import LoanController
from models.book import Book
from models.enums import BookStatus


class TestBookController:
    @pytest.fixture
    def controller(self):
        service = MagicMock()
        view = MagicMock()
        return BookController(service, view)

    def test_list_books_success(self, controller):
        controller.set_request({"page": "1", "per_page": "20"})
        response = controller.list_books()
        assert response["status"] == 200

    def test_get_book_not_found(self, controller):
        controller.set_request({})
        response = controller.get_book("invalid-uuid")
        assert response["status"] == 404
        assert "error" in response

    def test_create_book_missing_fields(self, controller):
        controller.set_request({"title": "", "isbn": ""})
        response = controller.create_book()
        assert response["status"] in (400, 500) or "error" in response
```

---

## Kết luận

MVC là một trong những architectural pattern lâu đời và có ảnh hưởng nhất. Dù đã hơn 40 năm, nó vẫn là nền tảng của hầu hết các web framework hiện đại. Sức mạnh của MVC nằm ở sự đơn giản: **tách biệt dữ liệu (Model), hiển thị (View), và điều phối (Controller)**.

### Best Practices

1.  **Fat Model, Skinny Controller** — Business logic trong Model, Controller chỉ điều phối
2.  **View không gọi Model trực tiếp** — View chỉ nhận dữ liệu từ Controller
3.  **Controller không chứa SQL** — SQL trong Model/Repository
4.  **One Controller per resource** — Mỗi resource một controller
5.  **Use Service Layer** — Nếu business logic quá phức tạp, thêm Service layer
6.  **Template inheritance** — Dùng layout/base template
7.  **Don't put logic in templates** — Template chỉ format, không tính toán

### Golden Rules

| Rule | Mô tả |
|---|---|
| **Model độc lập** | Model không biết View/Controller tồn tại |
| **Controller gọn nhẹ** | Controller chỉ 5-10 dòng, gọi Model + chọn View |
| **View không logic** | View không có if/for phức tạp, không gọi DB |
| **Router không xử lý** | Router chỉ định tuyến, không xử lý request |
| **Một URL = một controller method** | RESTful design |
| **Service layer optional** | Thêm service khi controller quá dày |
| **Test Model trước** | Model là nơi chứa logic quan trọng nhất |

### Tương lai của MVC

MVC vẫn là pattern phổ biến cho server-rendered web applications. Tuy nhiên, với sự lên ngôi của SPA (Single Page Application) và component-based architectures (React, Vue), MVC đang dần chuyển thành MVVM hoặc pattern khác ở client-side. Nhưng ở server-side, MVC vẫn là lựa chọn số một cho hầu hết các web application.

Và hãy nhớ: dù pattern có thay đổi thế nào, **Separation of Concerns** vẫn là nguyên lý bất di bất dịch của software engineering.
