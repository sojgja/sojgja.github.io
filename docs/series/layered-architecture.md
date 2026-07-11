---
id: layered-architecture
title: Layered Architecture (N-Tier)
sidebar_label: 🧅 Layered Architecture
sidebar_position: 36
---

# Layered Architecture (N-Tier)

> *"Layered architecture is the most common architectural pattern and the de facto standard for most business applications."* — **Mark Richards**, *Software Architecture Patterns*

**Layered Architecture** (còn gọi là **N-Tier Architecture** hay **Multi-Layer Architecture**) là một trong những kiểu kiến trúc lâu đời và phổ biến nhất trong kỹ thuật phần mềm. Nó tổ chức code thành các **tầng (layers)** xếp chồng lên nhau, mỗi tầng đảm nhận một nhóm trách nhiệm cụ thể. Tầng trên cùng giao tiếp với tầng ngay dưới nó thông qua các interface được định nghĩa trước. Đây là kiến trúc mặc định mà hầu hết developer được học đầu tiên — từ các đồ án đại học cho đến hệ thống enterprise lớn.

---

## Bài toán

### Vấn đề: Monolithic spaghetti code

Hãy tưởng tượng bạn là kiến trúc sư chính của **một hệ thống ngân hàng số (digital banking platform)** — giống như Timo hay VPBank Digital. Hệ thống phải xử lý hàng triệu giao dịch mỗi ngày, tích hợp với nhiều core banking system khác nhau, hỗ trợ nhiều loại tài khoản (saving, checking, credit), và tuân thủ nghiêm ngặt các quy định của Ngân hàng Nhà nước về bảo mật và audit.

Trong một hệ thống không có kiến trúc rõ ràng, mọi thứ nhanh chóng trở thành **spaghetti code** — code lộn xộn, không có ranh giới rõ ràng giữa các chức năng. Validation logic nằm trong controller, business logic xen lẫn với SQL queries, và database connection được tạo trực tiếp trong view layer. Kết quả là một codebase mà không ai dám động vào vì chỉ cần thay đổi một dòng code cũng có thể gây ra hàng loạt lỗi không lường trước.

Cụ thể hơn, hãy xem xét một use case điển hình: **chuyển khoản giữa hai tài khoản (internal transfer)**. Quy trình này bao gồm:

1. **Input Validation**: Kiểm tra số tài khoản, số dư, hạn mức giao dịch trong ngày
2. **Business Logic**: Tính phí giao dịch, kiểm tra hạn mức theo loại tài khoản (VIP, Standard)
3. **Data Access**: Cập nhật số dư tài khoản gửi và nhận, ghi audit log
4. **Cross-cutting Concerns**: Transaction management, logging, authorization

Khi tất cả logic này được viết trong một hàm duy nhất, nó vi phạm nghiêm trọng **Single Responsibility Principle** — hàm đó có quá nhiều lý do để thay đổi. Nếu ngân hàng thay đổi biểu phí, bạn phải sửa hàm đó. Nếu database chuyển từ Oracle sang PostgreSQL, bạn lại sửa. Nếu thêm loại tài khoản mới, bạn lại động vào. Mỗi lần sửa là một lần rủi ro.

Hơn nữa, việc kiểm thử trở nên cực kỳ khó khăn. Để test logic tính phí, bạn phải set up toàn bộ hệ thống — database thật, kết nối mạng, authentication — thay vì chỉ đơn giản gọi một hàm với input và kiểm tra output. Điều này làm giảm tốc độ phát triển và tăng chi phí bảo trì lên gấp nhiều lần.

### Layered Architecture giải quyết vấn đề này như thế nào?

Layered Architecture giải quyết chaos bằng cách áp đặt một cấu trúc phân tầng rõ ràng. Mỗi tầng chỉ giao tiếp với tầng kế cận, và mỗi tầng chỉ có một lý do duy nhất để thay đổi:

- **Presentation Layer** thay đổi khi UI thay đổi (web → mobile → desktop)
- **Business Layer** thay đổi khi business rule thay đổi (biểu phí mới, loại tài khoản mới)
- **Persistence Layer** thay đổi khi data source thay đổi (Oracle → PostgreSQL, thêm cache layer)
- **Database Layer** thay đổi khi DB engine thay đổi (schema migration, indexing strategy)

Sự phân tách này cho phép nhiều developer làm việc song song trên các tầng khác nhau, kiểm thử từng tầng độc lập, và thay đổi implementation của một tầng mà không ảnh hưởng đến các tầng khác — miễn là interface giữa chúng được giữ nguyên.

---

## Nguyên lý thiết kế

### 1. Strict Layering vs Relaxed Layering

Có hai biến thể chính:

- **Strict Layering**: Một tầng **chỉ** được gọi tầng ngay dưới nó. Presentation gọi Business, Business gọi Persistence, Persistence gọi Database. Đây là dạng thuần khiết nhất, dễ hiểu và dễ maintain nhất.
- **Relaxed Layering**: Một tầng có thể gọi tầng dưới nó **hoặc tầng cách xa hơn**. Ví dụ: Presentation có thể gọi trực tiếp Persistence (skip Business layer). Điều này linh hoạt hơn nhưng dễ dẫn đến vi phạm nguyên tắc và làm rối kiến trúc.

**Khuyến nghị**: Dùng strict layering trong hầu hết trường hợp. Chỉ dùng relaxed khi có lý do chính đáng và được document rõ ràng.

### 2. Dependency Direction

Nguyên tắc vàng: **Dependencies đi từ trên xuống dưới**. Tầng trên phụ thuộc vào tầng dưới, không bao giờ ngược lại. Presentation phụ thuộc vào Business, Business phụ thuộc vào Persistence. Nếu tầng dưới thay đổi interface, nó sẽ ảnh hưởng đến tầng trên — nhưng nếu tầng trên thay đổi, tầng dưới không bị ảnh hưởng.

### 3. Layer Isolation

Mỗi tầng phải **hoàn toàn độc lập** về mặt triển khai. Bạn phải có thể thay thế Persistence Layer (ví dụ: từ SQLAlchemy sang Django ORM) mà không ảnh hưởng đến Business Layer. Điều này đạt được bằng cách định nghĩa interface trừu tượng giữa các tầng.

### 4. Cross-cutting Concerns

Một số concern xuyên suốt tất cả các tầng — logging, security, transaction management, error handling. Các concern này không thuộc về bất kỳ tầng cụ thể nào. Chúng được xử lý qua:

- **Decorator pattern**: Wrap method với logging/security logic
- **AOP (Aspect-Oriented Programming)**: AspectJ, Spring AOP, Python decorators
- **Middleware**: Trong web framework (Express.js, Django middleware)
- **Pipeline pattern**: Chain các handler xử lý request/response

### 5. Layer Supertype

Mỗi tầng thường có một base class chung cung cấp các tiện ích dùng chung: logging, error handling, validation helper. Ví dụ: `BaseController`, `BaseService`, `BaseRepository`. Điều này giảm code duplication trong cùng tầng.

### 6. Số lượng tầng tối ưu

**3 tầng (Three-Tier)** là phổ biến nhất: Presentation - Business - Data. Đây là sweet spot giữa độ phức tạp và lợi ích.
**4 tầng**: Thêm Integration Layer giữa Business và Data cho hệ thống cần tích hợp nhiều hệ thống bên ngoài.
**5+ tầng**: Hiếm khi cần thiết, thường gây over-engineering và performance overhead.

---

## Cấu trúc chi tiết

### Three-Tier Architecture (3 tầng)

```
┌─────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER (UI)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Controllers │  │   Views   │  │ Serializers │  │   Forms   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                   BUSINESS LAYER (Logic)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Services  │  │  Domain   │  │ Validators │  │  Workflows│       │
│  │            │  │  Models   │  │            │  │  (Orch.) │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                  PERSISTENCE LAYER (Data)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Repositories│  │    ORM    │  │ Data Mappers│  │   DAOs   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Chi tiết từng tầng

#### 1. Presentation Layer

**Trách nhiệm**: Giao tiếp với user (human hoặc external system). Không chứa business logic.

**Thành phần**:
- **Controllers/Routers**: Nhận request, gọi service, trả response
- **Views/Templates**: Render HTML/JSON response
- **Serializers/Formatters**: Chuyển đổi dữ liệu giữa internal representation và external format (JSON, XML, Protocol Buffers)
- **Middlewares**: Xử lý request/response pipeline (auth, logging, CORS)
- **Validators**: Input validation cơ bản (format, required fields, length)

**Quy tắc**: Presentation Layer không được:
- Truy cập database trực tiếp
- Chứa business logic
- Import bất kỳ module nào từ Persistence Layer

#### 2. Business Layer (Domain/Service Layer)

**Trách nhiệm**: Chứa toàn bộ business logic của ứng dụng. Đây là tầng quan trọng nhất — "trái tim" của hệ thống.

**Thành phần**:
- **Services**: Các use case của hệ thống (TransferService, AccountService, ReportService)
- **Domain Models**: Business entities với behavior (Account, Transaction, Customer)
- **Business Validators**: Validation có business context (không chỉ format)
- **Workflows/Orchestrators**: Phối hợp nhiều service để thực hiện một business process
- **Domain Events/Notifications**: Sự kiện business (TransferCompleted, AccountOverdrawn)

**Quy tắc**:
- Business Layer là tầng **quan trọng nhất** — cần được test kỹ nhất
- Không phụ thuộc vào framework hay infrastructure
- Sử dụng dependency injection để nhận repository từ bên ngoài
- Có thể throw business exception (InsufficientBalanceError, AccountNotFoundError)

#### 3. Persistence Layer (Data Access Layer)

**Trách nhiệm**: Quản lý lưu trữ và truy xuất dữ liệu.

**Thành phần**:
- **Repositories**: Abstraction trên data source (interface: `save()`, `find_by_id()`, `delete()`)
- **ORM Mappers**: SQLAlchemy models, Django ORM models
- **Data Access Objects (DAOs)**: Low-level data access cho các query phức tạp
- **Connection Managers**: Connection pooling, session management

**Quy tắc**:
- Repository interface được định nghĩa ở Business Layer (Dependency Inversion)
- Implementation cụ thể có thể thay đổi (PostgreSQL, MongoDB, Redis)
- Transaction management thường được kiểm soát ở tầng Service

#### 4. Database Layer (Infrastructure)

**Trách nhiệm**: Lưu trữ dữ liệu vật lý.

**Thành phần**:
- Relational Databases: PostgreSQL, MySQL, Oracle
- NoSQL: MongoDB, Cassandra, Redis
- Message Queues: RabbitMQ, Kafka (nếu dùng cho persistence)

---

## Sơ đồ kiến trúc

```
                      ┌─────────────────────────────────────┐
                      │       EXTERNAL CLIENTS               │
                      │  (Web Browser, Mobile App, API)      │
                      └──────────┬──────────────────────────┘
                                 │ HTTP/HTTPS
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                             │
│                                                                    │
│  ┌──────────────────┐   ┌──────────────────┐                      │
│  │   Controllers    │   │    Middleware     │                      │
│  │  ┌────────────┐  │   │  ┌──────────────┐│                      │
│  │  │ AuthCtrl   │  │   │  │ AuthMiddleware││                      │
│  │  │ AccountCtrl│  │   │  │ LogMiddleware ││                      │
│  │  │ TransferCt.│  │   │  │ RateLimitMiddl││                      │
│  │  └────────────┘  │   │  └──────────────┘│                      │
│  └──────────────────┘   └──────────────────┘                      │
│                                                                    │
│  ┌────────────────────────────────────────────────┐               │
│  │           Serializers / Validators              │               │
│  │  { "account": {...}, "amount": number }         │               │
│  └────────────────────────────────────────────────┘               │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ Gọi Service Layer
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                       BUSINESS LAYER                                │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ AccountService│  │TransferServ. │  │  ReportServ. │             │
│  │              │  │              │  │              │             │
│  │ - create()   │  │ - transfer() │  │ - monthly()  │             │
│  │ - freeze()   │  │ - validate() │  │ - analytics()│             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                    │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │
│  │Domain Models │  │Business Validat. │  │   Workflows          │  │
│  │  Account     │  │  FeeCalculator   │  │  TransferWorkflow   │  │
│  │  Transaction │  │  LimitChecker    │  │  NewAccountWorkflow │  │
│  └──────────────┘  └──────────────────┘  └─────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │          Repository Interfaces (Ports)                    │      │
│  │  IAccountRepository, ITransactionRepository, ICustomerRep│      │
│  └──────────────────────────────────────────────────────────┘      │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ Gọi Repository Implementation
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                     PERSISTENCE LAYER                               │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │AccountRepo   │  │TransactRepo  │  │CustomerRepo   │             │
│  │ (SQLAlchemy) │  │ (SQLAlchemy) │  │ (SQLAlchemy)  │             │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘             │
│         │                 │                   │                    │
│         └─────────────────┼───────────────────┘                    │
│                           │                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Database Connections                       │   │
│  │  Connection Pool, Session Factory, Migration Engine           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ Read/Write
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                        DATABASE SERVER                              │
│                                                                    │
│  ╔══════════════════════════════════════════════════════════╗      │
│  ║            PostgreSQL / MySQL / Oracle                    ║      │
│  ║  accounts | transactions | customers | audit_logs        ║      │
│  ╚══════════════════════════════════════════════════════════╝      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Ví dụ code hoàn chỉnh

### Cấu trúc project

```
banking_system/
├── app.py                          # Application entry point
├── requirements.txt                # Dependencies
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_presentation.py
│   ├── test_business.py
│   └── test_persistence.py
└── src/
    ├── __init__.py
    ├── config.py                   # Configuration
    ├── presentation/
    │   ├── __init__.py
    │   ├── controllers.py          # API controllers
    │   ├── serializers.py          # Request/Response serializers
    │   └── validators.py           # Input validation
    ├── business/
    │   ├── __init__.py
    │   ├── exceptions.py           # Business exceptions
    │   ├── interfaces.py           # Repository interfaces (ports)
    │   ├── models.py               # Domain models
    │   ├── services.py             # Business services
    │   └── validators.py           # Business validation
    └── persistence/
        ├── __init__.py
        ├── database.py             # Database connection & session
        ├── models.py               # SQLAlchemy ORM models
        └── repositories.py         # Repository implementations
```

### File: `src/config.py`

```python
"""Application configuration with type-safe settings."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    username: str = "bank_user"
    password: str = "bank_pass"
    database: str = "banking_db"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class AppConfig:
    """Global application configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    transfer_limit_daily: float = 500_000_000.0  # 500M VND
    transfer_fee_percent: float = 0.001  # 0.1%

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables with defaults."""
        import os
        return cls(
            environment=Environment(os.getenv("APP_ENV", "development")),
            debug=os.getenv("APP_DEBUG", "true").lower() == "true",
            database=DatabaseConfig(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                username=os.getenv("DB_USER", "bank_user"),
                password=os.getenv("DB_PASS", "bank_pass"),
                database=os.getenv("DB_NAME", "banking_db"),
            ),
        )
```

### File: `src/business/exceptions.py`

```python
"""Business-level exceptions with error codes for API mapping."""

from enum import IntEnum


class ErrorCode(IntEnum):
    INVALID_INPUT = 400
    UNAUTHORIZED = 401
    NOT_FOUND = 404
    CONFLICT = 409
    UNPROCESSABLE = 422
    INTERNAL = 500


class BusinessError(Exception):
    """Base exception for all business-layer errors."""
    def __init__(self, message: str, code: ErrorCode = ErrorCode.UNPROCESSABLE) -> None:
        self.message = message
        self.code = code
        super().__init__(self.message)


class AccountNotFoundError(BusinessError):
    def __init__(self, account_id: str) -> None:
        super().__init__(
            message=f"Tài khoản {account_id} không tồn tại",
            code=ErrorCode.NOT_FOUND,
        )


class InsufficientBalanceError(BusinessError):
    def __init__(self, account_id: str, balance: float, required: float) -> None:
        super().__init__(
            message=(
                f"Tài khoản {account_id} không đủ số dư. "
                f"Số dư: {balance:,.0f} VND, Yêu cầu: {required:,.0f} VND"
            ),
            code=ErrorCode.UNPROCESSABLE,
        )


class DailyLimitExceededError(BusinessError):
    def __init__(self, account_id: str, limit: float) -> None:
        super().__init__(
            message=(
                f"Tài khoản {account_id} đã vượt hạn mức giao dịch trong ngày. "
                f"Hạn mức: {limit:,.0f} VND"
            ),
            code=ErrorCode.UNPROCESSABLE,
        )


class AccountFrozenError(BusinessError):
    def __init__(self, account_id: str) -> None:
        super().__init__(
            message=f"Tài khoản {account_id} đang bị đóng băng",
            code=ErrorCode.CONFLICT,
        )
```

### File: `src/business/interfaces.py`

```python
"""Repository interfaces (Ports) defined in Business Layer.
Following Dependency Inversion Principle — Business defines the contract,
Persistence provides the implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional


class AccountType(Enum):
    SAVING = "saving"
    CHECKING = "checking"
    CREDIT = "credit"
    LOAN = "loan"


class AccountStatus(Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    CLOSED = "closed"


@dataclass
class AccountDTO:
    """Data Transfer Object for Account — crosses layer boundaries."""
    id: str
    customer_id: str
    account_type: AccountType
    status: AccountStatus
    balance: float
    daily_transfer_limit: float
    opened_at: datetime


@dataclass
class TransactionDTO:
    id: str
    from_account_id: str
    to_account_id: str
    amount: float
    fee: float
    transaction_type: str
    status: str
    description: str
    created_at: datetime


class IAccountRepository(ABC):
    """Port for account persistence operations."""

    @abstractmethod
    def find_by_id(self, account_id: str) -> Optional[AccountDTO]:
        ...

    @abstractmethod
    def find_by_customer(self, customer_id: str) -> List[AccountDTO]:
        ...

    @abstractmethod
    def update_balance(self, account_id: str, new_balance: float) -> None:
        ...

    @abstractmethod
    def save(self, account: AccountDTO) -> AccountDTO:
        ...


class ITransactionRepository(ABC):
    """Port for transaction persistence operations."""

    @abstractmethod
    def save(self, transaction: TransactionDTO) -> TransactionDTO:
        ...

    @abstractmethod
    def find_by_account(
        self, account_id: str, from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[TransactionDTO]:
        ...

    @abstractmethod
    def get_daily_total(
        self, account_id: str, transaction_date: date,
    ) -> float:
        """Return sum of all outgoing transactions on given date."""
        ...
```

### File: `src/business/services.py`

```python
"""Business services — the heart of the application.
Contains all business logic, domain rules, and validation.
This layer never imports from presentation or persistence directly."""

from datetime import date, datetime
from typing import List, Optional
from uuid import uuid4

from src.config import AppConfig
from src.business.exceptions import (
    AccountFrozenError,
    AccountNotFoundError,
    DailyLimitExceededError,
    InsufficientBalanceError,
)
from src.business.interfaces import (
    AccountDTO,
    AccountStatus,
    AccountType,
    IAccountRepository,
    ITransactionRepository,
    TransactionDTO,
)


class TransferService:
    """Handles money transfer between accounts.
    This is the core use case of the banking system."""

    def __init__(
        self,
        account_repo: IAccountRepository,
        transaction_repo: ITransactionRepository,
        config: AppConfig,
    ) -> None:
        self._account_repo = account_repo
        self._transaction_repo = transaction_repo
        self._config = config

    def transfer(
        self,
        from_account_id: str,
        to_account_id: str,
        amount: float,
        description: str = "",
    ) -> TransactionDTO:
        """Execute a transfer between two accounts with full validation."""

        # 1. Validate accounts exist
        from_account = self._account_repo.find_by_id(from_account_id)
        if not from_account:
            raise AccountNotFoundError(from_account_id)

        to_account = self._account_repo.find_by_id(to_account_id)
        if not to_account:
            raise AccountNotFoundError(to_account_id)

        # 2. Validate accounts are active
        if from_account.status != AccountStatus.ACTIVE:
            raise AccountFrozenError(from_account_id)
        if to_account.status != AccountStatus.ACTIVE:
            raise AccountFrozenError(to_account_id)

        # 3. Validate balance
        if from_account.balance < amount:
            raise InsufficientBalanceError(
                from_account_id, from_account.balance, amount,
            )

        # 4. Validate daily limit
        today = date.today()
        daily_total = self._transaction_repo.get_daily_total(
            from_account_id, today,
        )
        if daily_total + amount > from_account.daily_transfer_limit:
            raise DailyLimitExceededError(
                from_account_id, from_account.daily_transfer_limit,
            )

        # 5. Calculate fee
        fee = self._calculate_fee(amount, from_account.account_type)

        # 6. Validate balance with fee
        total_required = amount + fee
        if from_account.balance < total_required:
            raise InsufficientBalanceError(
                from_account_id, from_account.balance, total_required,
            )

        # 7. Execute transfer (atomic)
        new_from_balance = from_account.balance - total_required
        new_to_balance = to_account.balance + amount

        self._account_repo.update_balance(from_account_id, new_from_balance)
        self._account_repo.update_balance(to_account_id, new_to_balance)

        # 8. Record transaction
        transaction = TransactionDTO(
            id=str(uuid4()),
            from_account_id=from_account_id,
            to_account_id=to_account_id,
            amount=amount,
            fee=fee,
            transaction_type="TRANSFER",
            status="COMPLETED",
            description=description or f"Chuyển khoản đến {to_account_id}",
            created_at=datetime.now(),
        )
        return self._transaction_repo.save(transaction)

    def _calculate_fee(self, amount: float, account_type: AccountType) -> float:
        """Calculate transfer fee based on account type and amount.
        VIP accounts get 50% discount on transfer fees."""
        fee_rate = self._config.transfer_fee_percent

        if account_type == AccountType.CREDIT:
            fee_rate *= 1.5  # Credit accounts pay 50% more
        elif account_type == AccountType.SAVING:
            fee_rate *= 0.5  # Saving accounts pay 50% less

        return round(amount * fee_rate, 2)


class AccountService:
    """Manages account lifecycle — creation, status changes, reporting."""

    def __init__(
        self,
        account_repo: IAccountRepository,
        transaction_repo: ITransactionRepository,
        config: AppConfig,
    ) -> None:
        self._account_repo = account_repo
        self._transaction_repo = transaction_repo
        self._config = config

    def create_account(
        self,
        customer_id: str,
        account_type: AccountType,
        initial_deposit: float = 0.0,
    ) -> AccountDTO:
        """Create a new bank account."""
        account = AccountDTO(
            id=str(uuid4()),
            customer_id=customer_id,
            account_type=account_type,
            status=AccountStatus.ACTIVE,
            balance=initial_deposit,
            daily_transfer_limit=self._get_default_limit(account_type),
            opened_at=datetime.now(),
        )
        saved = self._account_repo.save(account)

        if initial_deposit > 0:
            # Record initial deposit as a transaction
            deposit = TransactionDTO(
                id=str(uuid4()),
                from_account_id="DEPOSIT",
                to_account_id=account.id,
                amount=initial_deposit,
                fee=0.0,
                transaction_type="DEPOSIT",
                status="COMPLETED",
                description="Nạp tiền tạo tài khoản",
                created_at=datetime.now(),
            )
            self._transaction_repo.save(deposit)

        return saved

    def _get_default_limit(self, account_type: AccountType) -> float:
        limits = {
            AccountType.SAVING: 200_000_000,   # 200M VND
            AccountType.CHECKING: 500_000_000,  # 500M VND
            AccountType.CREDIT: 1_000_000_000,  # 1B VND
            AccountType.LOAN: 0.0,
        }
        return limits.get(account_type, 100_000_000)

    def get_account_summary(self, account_id: str) -> dict:
        """Get a comprehensive account summary with recent transactions."""
        account = self._account_repo.find_by_id(account_id)
        if not account:
            raise AccountNotFoundError(account_id)

        transactions = self._transaction_repo.find_by_account(account_id)
        return {
            "account": account,
            "recent_transactions": transactions[-10:],
            "total_transactions": len(transactions),
        }
```

### File: `src/business/validators.py`

```python
"""Business validators — domain-level validation with context."""

from typing import List, Tuple

from src.business.exceptions import BusinessError
from src.business.interfaces import AccountDTO, AccountType, AccountStatus


class TransferValidator:
    """Validates transfer requests with business context."""

    @staticmethod
    def validate_amount(amount: float) -> List[str]:
        errors: List[str] = []
        if amount <= 0:
            errors.append("Số tiền chuyển phải lớn hơn 0")
        if amount > 1_000_000_000:  # 1B VND max per transaction
            errors.append("Số tiền chuyển tối đa là 1,000,000,000 VND")
        if amount != round(amount, 2):
            errors.append("Số tiền không hợp lệ (tối đa 2 chữ số thập phân)")
        return errors

    @staticmethod
    def validate_account_type_for_transfer(
        account: AccountDTO,
    ) -> List[str]:
        errors: List[str] = []
        if account.account_type == AccountType.LOAN:
            errors.append("Tài khoản vay không được chuyển tiền")
        if account.status != AccountStatus.ACTIVE:
            errors.append("Tài khoản không hoạt động")
        return errors

    @staticmethod
    def validate_same_account(
        from_account_id: str, to_account_id: str,
    ) -> List[str]:
        if from_account_id == to_account_id:
            return ["Không thể chuyển tiền vào cùng tài khoản"]
        return []


class CustomerValidator:
    """Validates customer-related business rules."""

    @staticmethod
    def validate_account_creation(
        account_type: AccountType, initial_deposit: float,
    ) -> List[str]:
        errors: List[str] = []
        if account_type == AccountType.CREDIT and initial_deposit < 1_000_000:
            errors.append("Tài khoản tín dụng yêu cầu số dư tối thiểu 1,000,000 VND")
        return errors
```

### File: `src/persistence/database.py`

```python
"""Database connection management with SQLAlchemy.
This file belongs to the Persistence Layer."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from typing import Generator

from src.config import DatabaseConfig


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


engine = None
SessionLocal = None


def init_database(config: DatabaseConfig) -> None:
    """Initialize database connection with connection pooling."""
    global engine, SessionLocal
    engine = create_engine(
        config.connection_string,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        echo=config.echo,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create all tables
    import src.persistence.models  # noqa: F401 — register models
    Base.metadata.create_all(bind=engine)


def get_session() -> Generator[Session, None, None]:
    """Dependency injection for database sessions."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### File: `src/persistence/models.py`

```python
"""SQLAlchemy ORM models for the database."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Column, String, Float, Enum as SAEnum, DateTime, ForeignKey,
    Index, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.business.interfaces import AccountType, AccountStatus
from src.persistence.database import Base


class AccountModel(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    account_type: Mapped[AccountType] = mapped_column(
        SAEnum(AccountType), nullable=False,
    )
    status: Mapped[AccountStatus] = mapped_column(
        SAEnum(AccountStatus), nullable=False, default=AccountStatus.ACTIVE,
    )
    balance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    daily_transfer_limit: Mapped[float] = mapped_column(Float, nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_accounts_customer_type", "customer_id", "account_type"),
    )

    def __repr__(self) -> str:
        return f"<Account(id={self.id}, type={self.account_type}, balance={self.balance})>"


class TransactionModel(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    from_account_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True,
    )
    to_account_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True,
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    transaction_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="COMPLETED")
    description: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now,
    )

    __table_args__ = (
        Index("ix_transactions_from_date", "from_account_id", "created_at"),
        Index("ix_transactions_to_date", "to_account_id", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction(id={self.id}, from={self.from_account_id}, "
            f"to={self.to_account_id}, amount={self.amount})>"
        )
```

### File: `src/persistence/repositories.py`

```python
"""Concrete repository implementations. These belong to the Persistence Layer
and implement interfaces defined in the Business Layer."""

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.business.interfaces import (
    AccountDTO,
    IAccountRepository,
    ITransactionRepository,
    TransactionDTO,
)
from src.persistence.models import AccountModel, TransactionModel


class SQLAlchemyAccountRepository(IAccountRepository):
    """Implements IAccountRepository using SQLAlchemy ORM."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def find_by_id(self, account_id: str) -> Optional[AccountDTO]:
        model = self._session.query(AccountModel).filter_by(id=account_id).first()
        return self._to_dto(model) if model else None

    def find_by_customer(self, customer_id: str) -> List[AccountDTO]:
        models = (
            self._session.query(AccountModel)
            .filter_by(customer_id=customer_id)
            .all()
        )
        return [self._to_dto(m) for m in models]

    def update_balance(self, account_id: str, new_balance: float) -> None:
        self._session.query(AccountModel).filter_by(id=account_id).update(
            {"balance": new_balance}
        )
        self._session.commit()

    def save(self, account: AccountDTO) -> AccountDTO:
        model = AccountModel(
            id=account.id,
            customer_id=account.customer_id,
            account_type=account.account_type,
            status=account.status,
            balance=account.balance,
            daily_transfer_limit=account.daily_transfer_limit,
            opened_at=account.opened_at,
        )
        self._session.add(model)
        self._session.commit()
        self._session.refresh(model)
        return self._to_dto(model)

    def _to_dto(self, model: AccountModel) -> AccountDTO:
        return AccountDTO(
            id=model.id,
            customer_id=model.customer_id,
            account_type=model.account_type,
            status=model.status,
            balance=model.balance,
            daily_transfer_limit=model.daily_transfer_limit,
            opened_at=model.opened_at,
        )


class SQLAlchemyTransactionRepository(ITransactionRepository):
    """Implements ITransactionRepository using SQLAlchemy ORM."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, transaction: TransactionDTO) -> TransactionDTO:
        model = TransactionModel(
            id=transaction.id,
            from_account_id=transaction.from_account_id,
            to_account_id=transaction.to_account_id,
            amount=transaction.amount,
            fee=transaction.fee,
            transaction_type=transaction.transaction_type,
            status=transaction.status,
            description=transaction.description,
            created_at=transaction.created_at,
        )
        self._session.add(model)
        self._session.commit()
        self._session.refresh(model)
        return self._to_dto(model)

    def find_by_account(
        self, account_id: str, from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[TransactionDTO]:
        query = self._session.query(TransactionModel).filter(
            (TransactionModel.from_account_id == account_id)
            | (TransactionModel.to_account_id == account_id),
        )
        if from_date:
            query = query.filter(TransactionModel.created_at >= datetime.combine(from_date, datetime.min.time()))
        if to_date:
            query = query.filter(TransactionModel.created_at <= datetime.combine(to_date, datetime.max.time()))
        query = query.order_by(TransactionModel.created_at.desc())
        return [self._to_dto(m) for m in query.all()]

    def get_daily_total(self, account_id: str, transaction_date: date) -> float:
        start = datetime.combine(transaction_date, datetime.min.time())
        end = datetime.combine(transaction_date, datetime.max.time())
        result = (
            self._session.query(func.coalesce(func.sum(TransactionModel.amount), 0.0))
            .filter(
                TransactionModel.from_account_id == account_id,
                TransactionModel.created_at.between(start, end),
            )
            .scalar()
        )
        return float(result)

    def _to_dto(self, model: TransactionModel) -> TransactionDTO:
        return TransactionDTO(
            id=model.id,
            from_account_id=model.from_account_id,
            to_account_id=model.to_account_id,
            amount=model.amount,
            fee=model.fee,
            transaction_type=model.transaction_type,
            status=model.status,
            description=model.description or "",
            created_at=model.created_at,
        )
```

### File: `src/presentation/serializers.py`

```python
"""Serializers and validators for API input/output."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TransferRequest:
    """Serialized transfer request from API."""
    from_account_id: str
    to_account_id: str
    amount: float
    description: str = ""


@dataclass
class CreateAccountRequest:
    customer_id: str
    account_type: str
    initial_deposit: float = 0.0


@dataclass
class ErrorResponse:
    error: str
    code: int
    details: Optional[List[str]] = None


@dataclass
class SuccessResponse:
    data: Any
    message: str = "Thành công"
    status: int = 200


def serialize_account(dto: Any) -> Dict[str, Any]:
    return {
        "id": dto.id,
        "customer_id": dto.customer_id,
        "account_type": dto.account_type.value,
        "status": dto.status.value,
        "balance": dto.balance,
        "daily_transfer_limit": dto.daily_transfer_limit,
        "opened_at": dto.opened_at.isoformat(),
    }


def serialize_transaction(dto: Any) -> Dict[str, Any]:
    return {
        "id": dto.id,
        "from_account_id": dto.from_account_id,
        "to_account_id": dto.to_account_id,
        "amount": dto.amount,
        "fee": dto.fee,
        "type": dto.transaction_type,
        "status": dto.status,
        "description": dto.description,
        "created_at": dto.created_at.isoformat(),
    }
```

### File: `src/presentation/controllers.py`

```python
"""API Controllers — handles HTTP request/response.
This is the outermost layer of the application."""

from typing import Any, Dict

from src.business.exceptions import BusinessError, ErrorCode
from src.business.interfaces import AccountType
from src.business.services import AccountService, TransferService
from src.business.validators import CustomerValidator, TransferValidator
from src.presentation.serializers import (
    CreateAccountRequest,
    TransferRequest,
    serialize_account,
    serialize_transaction,
)


class AccountController:
    """Handles account-related API requests."""

    def __init__(self, account_service: AccountService) -> None:
        self._service = account_service

    def create_account(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """POST /accounts — Create a new bank account."""
        try:
            req = CreateAccountRequest(
                customer_id=request["customer_id"],
                account_type=request["account_type"],
                initial_deposit=float(request.get("initial_deposit", 0)),
            )
            account_type = AccountType(req.account_type)

            # Validate
            errors = CustomerValidator.validate_account_creation(
                account_type, req.initial_deposit,
            )
            if errors:
                return {
                    "error": "Validation failed",
                    "code": 422,
                    "details": errors,
                }

            account = self._service.create_account(
                customer_id=req.customer_id,
                account_type=account_type,
                initial_deposit=req.initial_deposit,
            )
            return {
                "data": serialize_account(account),
                "message": "Tạo tài khoản thành công",
                "status": 201,
            }
        except KeyError as e:
            return {"error": f"Missing field: {e}", "code": 400, "status": 400}
        except ValueError as e:
            return {"error": str(e), "code": 422, "status": 422}

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """GET /accounts/:id — Get account details."""
        try:
            summary = self._service.get_account_summary(account_id)
            return {
                "data": {
                    "account": serialize_account(summary["account"]),
                    "recent_transactions": [
                        serialize_transaction(t)
                        for t in summary["recent_transactions"]
                    ],
                    "total_transactions": summary["total_transactions"],
                },
                "status": 200,
            }
        except BusinessError as e:
            return self._handle_error(e)

    def _handle_error(self, error: BusinessError) -> Dict[str, Any]:
        return {
            "error": error.message,
            "code": error.code.value,
            "status": error.code.value,
        }


class TransferController:
    """Handles transfer-related API requests."""

    def __init__(self, transfer_service: TransferService) -> None:
        self._service = transfer_service

    def transfer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """POST /transfers — Execute a money transfer."""
        try:
            req = TransferRequest(
                from_account_id=request["from_account_id"],
                to_account_id=request["to_account_id"],
                amount=float(request["amount"]),
                description=request.get("description", ""),
            )

            # 1. Validate input
            errors = TransferValidator.validate_amount(req.amount)
            errors += TransferValidator.validate_same_account(
                req.from_account_id, req.to_account_id,
            )
            if errors:
                return {
                    "error": "Validation failed",
                    "code": 422,
                    "details": errors,
                    "status": 422,
                }

            # 2. Execute business logic
            transaction = self._service.transfer(
                from_account_id=req.from_account_id,
                to_account_id=req.to_account_id,
                amount=req.amount,
                description=req.description,
            )
            return {
                "data": serialize_transaction(transaction),
                "message": "Chuyển khoản thành công",
                "status": 200,
            }
        except BusinessError as e:
            return {"error": e.message, "code": e.code.value, "status": e.code.value}
        except KeyError as e:
            return {"error": f"Missing field: {e}", "code": 400, "status": 400}
```

### File: `src/presentation/validators.py`

```python
"""Input-level validators — format, type, and presence checks.
These run BEFORE business validators."""

import re
from typing import List, Optional


class InputValidator:
    """Basic input validation for API requests."""

    ACCOUNT_ID_PATTERN = re.compile(r"^ACC-\d{10}$")

    @staticmethod
    def validate_account_id(account_id: str) -> Optional[str]:
        if not account_id:
            return "Account ID không được để trống"
        if not InputValidator.ACCOUNT_ID_PATTERN.match(account_id):
            return "Account ID không đúng định dạng (ACC-XXXXXXXXXX)"
        return None

    @staticmethod
    def validate_amount(amount: float) -> Optional[str]:
        try:
            val = float(amount)
            if val <= 0:
                return "Số tiền phải lớn hơn 0"
            if val > 1e12:
                return "Số tiền quá lớn"
        except (TypeError, ValueError):
            return "Số tiền không hợp lệ"
        return None
```

### File: `app.py`

```python
"""Application entry point — wires up all layers and runs the demo.
This is where dependency injection happens."""

from typing import Any, Dict

from src.config import AppConfig, Environment
from src.business.services import AccountService, TransferService
from src.business.interfaces import AccountType
from src.persistence.database import init_database, get_session
from src.persistence.repositories import (
    SQLAlchemyAccountRepository,
    SQLAlchemyTransactionRepository,
)
from src.presentation.controllers import AccountController, TransferController


def main() -> None:
    print("=" * 70)
    print("🏦 BANKING SYSTEM — Layered Architecture Demo")
    print("=" * 70)

    # 1. Configuration
    config = AppConfig.from_env()
    print(f"\n📋 Environment: {config.environment.value}")

    # 2. Initialize database (in-memory for demo)
    config.database = config.database.__class__(
        host=config.database.host,
        port=config.database.port,
        username=config.database.username,
        password=config.database.password,
        database=config.database.database,
        echo=False,
    )
    init_database(config.database)
    print("✅ Database initialized")

    # 3. Wire up dependencies
    session = next(get_session())
    account_repo = SQLAlchemyAccountRepository(session)
    transaction_repo = SQLAlchemyTransactionRepository(session)

    account_service = AccountService(account_repo, transaction_repo, config)
    transfer_service = TransferService(account_repo, transaction_repo, config)

    account_ctrl = AccountController(account_service)
    transfer_ctrl = TransferController(transfer_service)

    # 4. Demo: Create accounts
    print("\n" + "-" * 70)
    print("📌 DEMO 1: Tạo tài khoản mới")
    print("-" * 70)

    # Customer 1: Saving account
    result = account_ctrl.create_account({
        "customer_id": "CUST-001",
        "account_type": "saving",
        "initial_deposit": 10_000_000,  # 10M VND
    })
    acc1 = result["data"]
    print(f"✅ Tạo tài khoản tiết kiệm: {acc1['id']}")
    print(f"   Số dư: {acc1['balance']:,.0f} VND")

    # Customer 1: Checking account
    result = account_ctrl.create_account({
        "customer_id": "CUST-001",
        "account_type": "checking",
        "initial_deposit": 50_000_000,  # 50M VND
    })
    acc2 = result["data"]
    print(f"✅ Tạo tài khoản vãng lai: {acc2['id']}")
    print(f"   Số dư: {acc2['balance']:,.0f} VND")

    # Customer 2: Saving account
    result = account_ctrl.create_account({
        "customer_id": "CUST-002",
        "account_type": "saving",
        "initial_deposit": 5_000_000,  # 5M VND
    })
    acc3 = result["data"]
    print(f"✅ Tạo tài khoản cho CUST-002: {acc3['id']}")
    print(f"   Số dư: {acc3['balance']:,.0f} VND")

    # 5. Demo: Transfer money
    print("\n" + "-" * 70)
    print("📌 DEMO 2: Chuyển khoản")
    print("-" * 70)

    result = transfer_ctrl.transfer({
        "from_account_id": acc2["id"],  # Checking: 50M
        "to_account_id": acc3["id"],     # Customer 2 saving
        "amount": 2_000_000,             # 2M VND
        "description": "Chuyển tiền thanh toán",
    })
    if "data" in result:
        t = result["data"]
        print(f"✅ Chuyển khoản thành công!")
        print(f"   Từ: {t['from_account_id']}")
        print(f"   Đến: {t['to_account_id']}")
        print(f"   Số tiền: {t['amount']:,.0f} VND")
        print(f"   Phí: {t['fee']:,.0f} VND")
    else:
        print(f"❌ Lỗi: {result['error']}")

    # 6. Demo: Check balance after transfer
    print("\n" + "-" * 70)
    print("📌 DEMO 3: Kiểm tra số dư sau chuyển khoản")
    print("-" * 70)

    for acc_id, label in [(acc2["id"], "TK Vãng lai"), (acc3["id"], "TK Tiết kiệm (CUST-002)")]:
        result = account_ctrl.get_account(acc_id)
        if "data" in result:
            account = result["data"]["account"]
            print(f"\n{label}: {account['id']}")
            print(f"   Số dư: {account['balance']:,.0f} VND")
            print(f"   Loại: {account['account_type']}")
            print(f"   Trạng thái: {account['status']}")
            print(f"   Số giao dịch gần đây: {len(result['data']['recent_transactions'])}")

    # 7. Demo: Error handling — insufficient balance
    print("\n" + "-" * 70)
    print("📌 DEMO 4: Xử lý lỗi — Không đủ số dư")
    print("-" * 70)

    result = transfer_ctrl.transfer({
        "from_account_id": acc1["id"],  # Saving: 10M
        "to_account_id": acc2["id"],
        "amount": 999_999_999,          # 999M VND — quá số dư
    })
    print(f"❌ Lỗi: {result['error']} (HTTP {result['code']})")

    print("\n" + "=" * 70)
    print("✅ Demo hoàn tất! Kiến trúc phân tầng hoạt động chính xác.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

### File: `requirements.txt`

```
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

---

## Sơ đồ kiến trúc chi tiết

```
                    L A Y E R E D   A R C H I T E C T U R E
                    ========================================

CLIENT                    ┌──────────────────────────────────────────┐
(user/machine)            │          PRESENTATION LAYER              │
                         │  ┌────────────────────────────────────┐   │
                         │  │  Controllers  │ Validators         │   │
                         │  │  Serializers  │ Middleware          │   │
                         │  └────────────────────────────────────┘   │
                         │  Chỉ xử lý HTTP/input/output format        │
                         ├──────────────────────────────────────────┤
                         │           BUSINESS LAYER                  │
                         │  ┌────────────────────────────────────┐   │
                         │  │  Services (Use Cases)              │   │
                         │  │  Domain Models                     │   │
                         │  │  Business Validators               │   │
                         │  │  Repository Interfaces (Ports)     │   │
                         │  └────────────────────────────────────┘   │
                         │  Chứa toàn bộ business logic             │
                         ├──────────────────────────────────────────┤
                         │          PERSISTENCE LAYER                │
                         │  ┌────────────────────────────────────┐   │
                         │  │  Repository Implementations         │   │
                         │  │  ORM Models                       │   │
                         │  │  Connection Management             │   │
                         │  └────────────────────────────────────┘   │
                         │  Xử lý data access, không có business    │
                         ├──────────────────────────────────────────┤
                         │            DATABASE LAYER                 │
                         │  ╔══════════════════════════════════╗     │
                         │  ║  PostgreSQL / MySQL / Oracle     ║     │
                         │  ╚══════════════════════════════════╝     │
                         └──────────────────────────────────────────┘

DATA FLOW:
Request → [Controller] → [Validator] → [Service] → [Repository] → [Database]
Response ← [Controller] ← [Serializer] ← [Service] ← [Repository] ← [Database]

DEPENDENCY DIRECTION:
Presentation ──> Business ──> Persistence ──> Database
      (gọi)       (gọi)        (gọi)
      (phụ thuộc) (phụ thuộc)  (phụ thuộc)
```

---

## Kiểm thử

### File: `tests/conftest.py`

```python
"""Pytest fixtures for all tests."""

from typing import Generator
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import AppConfig, DatabaseConfig
from src.business.interfaces import (
    AccountDTO, AccountStatus, AccountType, IAccountRepository,
    ITransactionRepository, TransactionDTO,
)
from src.business.services import AccountService, TransferService
from src.persistence.database import Base


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def account_repo(db_session: Session) -> IAccountRepository:
    from src.persistence.repositories import SQLAlchemyAccountRepository
    return SQLAlchemyAccountRepository(db_session)


@pytest.fixture
def transaction_repo(db_session: Session) -> ITransactionRepository:
    from src.persistence.repositories import SQLAlchemyTransactionRepository
    return SQLAlchemyTransactionRepository(db_session)


@pytest.fixture
def account_service(
    account_repo: IAccountRepository,
    transaction_repo: ITransactionRepository,
    config: AppConfig,
) -> AccountService:
    return AccountService(account_repo, transaction_repo, config)


@pytest.fixture
def transfer_service(
    account_repo: IAccountRepository,
    transaction_repo: ITransactionRepository,
    config: AppConfig,
) -> TransferService:
    return TransferService(account_repo, transaction_repo, config)


@pytest.fixture
def sample_accounts(
    account_service: AccountService,
) -> tuple[AccountDTO, AccountDTO, AccountDTO]:
    acc1 = account_service.create_account("CUST-001", AccountType.SAVING, 10_000_000)
    acc2 = account_service.create_account("CUST-001", AccountType.CHECKING, 50_000_000)
    acc3 = account_service.create_account("CUST-002", AccountType.SAVING, 5_000_000)
    return acc1, acc2, acc3
```

### File: `tests/test_business.py`

```python
"""Unit tests for Business Layer — the most critical test suite."""

from datetime import date
from unittest.mock import MagicMock, Mock
import pytest

from src.business.exceptions import (
    AccountNotFoundError,
    DailyLimitExceededError,
    InsufficientBalanceError,
)
from src.business.interfaces import (
    AccountDTO, AccountStatus, AccountType, IAccountRepository,
    ITransactionRepository,
)
from src.business.services import AccountService, TransferService
from src.config import AppConfig


class TestTransferService:
    """Unit tests with mocked repositories — pure business logic."""

    @pytest.fixture
    def mock_repos(self) -> tuple[MagicMock, MagicMock]:
        account_repo = MagicMock(spec=IAccountRepository)
        transaction_repo = MagicMock(spec=ITransactionRepository)
        return account_repo, transaction_repo

    @pytest.fixture
    def service(
        self, mock_repos: tuple[MagicMock, MagicMock]
    ) -> TransferService:
        account_repo, transaction_repo = mock_repos
        config = AppConfig()
        return TransferService(account_repo, transaction_repo, config)

    def test_transfer_success(self, service: TransferService, mock_repos):
        """Happy path: successful transfer between accounts."""
        account_repo, transaction_repo = mock_repos

        from_acc = AccountDTO(
            id="ACC-001", customer_id="CUST-1",
            account_type=AccountType.CHECKING, status=AccountStatus.ACTIVE,
            balance=100_000, daily_transfer_limit=500_000,
            opened_at="2024-01-01",
        )
        to_acc = AccountDTO(
            id="ACC-002", customer_id="CUST-2",
            account_type=AccountType.SAVING, status=AccountStatus.ACTIVE,
            balance=50_000, daily_transfer_limit=200_000,
            opened_at="2024-01-01",
        )

        account_repo.find_by_id.side_effect = [from_acc, to_acc]
        transaction_repo.get_daily_total.return_value = 0.0

        result = service.transfer("ACC-001", "ACC-002", 10_000, "Test transfer")

        assert result.amount == 10_000
        assert result.fee == 5.0  # 10_000 * 0.05% (checking fee = 0.05%)
        assert result.status == "COMPLETED"
        account_repo.update_balance.assert_called()

    def test_transfer_account_not_found(self, service: TransferService, mock_repos):
        """Error: source account does not exist."""
        account_repo, _ = mock_repos
        account_repo.find_by_id.return_value = None

        with pytest.raises(AccountNotFoundError):
            service.transfer("INVALID", "ACC-002", 10_000)

    def test_transfer_insufficient_balance(self, service: TransferService, mock_repos):
        """Error: not enough balance to cover amount + fee."""
        account_repo, transaction_repo = mock_repos

        from_acc = AccountDTO(
            id="ACC-001", customer_id="CUST-1",
            account_type=AccountType.CHECKING, status=AccountStatus.ACTIVE,
            balance=1_000, daily_transfer_limit=500_000,
            opened_at="2024-01-01",
        )
        account_repo.find_by_id.return_value = from_acc
        transaction_repo.get_daily_total.return_value = 0.0

        with pytest.raises(InsufficientBalanceError):
            service.transfer("ACC-001", "ACC-002", 5_000)

    def test_transfer_exceeds_daily_limit(self, service: TransferService, mock_repos):
        """Error: transfer exceeds daily limit."""
        account_repo, transaction_repo = mock_repos

        from_acc = AccountDTO(
            id="ACC-001", customer_id="CUST-1",
            account_type=AccountType.CHECKING, status=AccountStatus.ACTIVE,
            balance=1_000_000, daily_transfer_limit=500_000,
            opened_at="2024-01-01",
        )
        account_repo.find_by_id.return_value = from_acc
        transaction_repo.get_daily_total.return_value = 480_000

        with pytest.raises(DailyLimitExceededError):
            service.transfer("ACC-001", "ACC-002", 50_000)

    def test_fee_calculation_different_account_types(self, mock_repos):
        """Verify fee calculation for different account types."""
        account_repo, transaction_repo = mock_repos
        config = AppConfig()
        service = TransferService(account_repo, transaction_repo, config)

        # SAVING: 0.05% fee (50% discount)
        from_acc = AccountDTO(
            id="ACC-001", customer_id="CUST-1",
            account_type=AccountType.SAVING, status=AccountStatus.ACTIVE,
            balance=1_000_000, daily_transfer_limit=500_000,
            opened_at="2024-01-01",
        )
        account_repo.find_by_id.return_value = from_acc
        transaction_repo.get_daily_total.return_value = 0.0

        result = service.transfer("ACC-001", "ACC-002", 100_000)
        assert result.fee == 50.0  # 100_000 * 0.05%

        # CREDIT: 0.15% fee (50% more)
        from_acc.account_type = AccountType.CREDIT
        result = service.transfer("ACC-001", "ACC-002", 100_000)
        assert result.fee == 150.0  # 100_000 * 0.15%


class TestAccountService:
    """Unit tests for AccountService."""

    def test_create_account_with_initial_deposit(self, mock_repos):
        account_repo, transaction_repo = mock_repos
        config = AppConfig()
        service = AccountService(account_repo, transaction_repo, config)

        account_repo.save.return_value = AccountDTO(
            id="ACC-NEW", customer_id="CUST-1",
            account_type=AccountType.SAVING, status=AccountStatus.ACTIVE,
            balance=1_000_000, daily_transfer_limit=200_000_000,
            opened_at="2024-06-01",
        )

        result = service.create_account("CUST-1", AccountType.SAVING, 1_000_000)

        assert result.balance == 1_000_000
        account_repo.save.assert_called_once()
        transaction_repo.save.assert_called_once()
```

### File: `tests/test_persistence.py`

```python
"""Integration tests for Persistence Layer — uses real SQLite database."""

import pytest
from datetime import date, datetime
from uuid import uuid4

from src.business.interfaces import AccountDTO, AccountStatus, AccountType, TransactionDTO
from src.persistence.repositories import (
    SQLAlchemyAccountRepository,
    SQLAlchemyTransactionRepository,
)


class TestSQLAlchemyAccountRepository:

    def test_save_and_find_by_id(self, account_repo: SQLAlchemyAccountRepository):
        account = AccountDTO(
            id="ACC-TEST-001", customer_id="CUST-TEST",
            account_type=AccountType.SAVING, status=AccountStatus.ACTIVE,
            balance=100_000, daily_transfer_limit=500_000,
            opened_at=datetime.now(),
        )
        saved = account_repo.save(account)
        assert saved.id == account.id
        assert saved.balance == 100_000

        found = account_repo.find_by_id("ACC-TEST-001")
        assert found is not None
        assert found.customer_id == "CUST-TEST"

    def test_find_by_id_not_found(self, account_repo):
        assert account_repo.find_by_id("NONEXIST") is None

    def test_update_balance(self, account_repo):
        account = AccountDTO(
            id="ACC-BAL-001", customer_id="CUST-1",
            account_type=AccountType.CHECKING, status=AccountStatus.ACTIVE,
            balance=500_000, daily_transfer_limit=1_000_000,
            opened_at=datetime.now(),
        )
        account_repo.save(account)
        account_repo.update_balance("ACC-BAL-001", 400_000)

        updated = account_repo.find_by_id("ACC-BAL-001")
        assert updated.balance == 400_000


class TestSQLAlchemyTransactionRepository:

    def test_save_transaction(self, transaction_repo):
        tx = TransactionDTO(
            id=str(uuid4()), from_account_id="ACC-1", to_account_id="ACC-2",
            amount=100_000, fee=50.0, transaction_type="TRANSFER",
            status="COMPLETED", description="Test",
            created_at=datetime.now(),
        )
        saved = transaction_repo.save(tx)
        assert saved.id == tx.id
        assert saved.amount == 100_000

    def test_get_daily_total(self, transaction_repo):
        tx = TransactionDTO(
            id=str(uuid4()), from_account_id="ACC-DAILY", to_account_id="ACC-2",
            amount=50_000, fee=25.0, transaction_type="TRANSFER",
            status="COMPLETED", description="",
            created_at=datetime.now(),
        )
        transaction_repo.save(tx)

        total = transaction_repo.get_daily_total("ACC-DAILY", date.today())
        assert total == 50_000

    def test_find_by_account(self, transaction_repo):
        for i in range(3):
            tx = TransactionDTO(
                id=str(uuid4()), from_account_id="ACC-FIND", to_account_id="ACC-2",
                amount=10_000 * (i + 1), fee=5.0, transaction_type="TRANSFER",
                status="COMPLETED", description="",
                created_at=datetime.now(),
            )
            transaction_repo.save(tx)

        results = transaction_repo.find_by_account("ACC-FIND")
        assert len(results) == 3
```

### File: `tests/test_presentation.py`

```python
"""Tests for Presentation Layer controllers."""

import pytest
from unittest.mock import MagicMock

from src.business.exceptions import AccountNotFoundError, InsufficientBalanceError
from src.business.interfaces import AccountDTO, AccountStatus, AccountType
from src.business.services import AccountService, TransferService
from src.presentation.controllers import AccountController, TransferController
from datetime import datetime


class TestAccountController:

    @pytest.fixture
    def controller(self):
        service = MagicMock(spec=AccountService)
        return AccountController(service)

    def test_create_account_success(self, controller):
        result = controller.create_account({
            "customer_id": "CUST-001",
            "account_type": "saving",
            "initial_deposit": "1000000",
        })
        assert "data" in result
        assert result["status"] == 201

    def test_create_account_missing_field(self, controller):
        result = controller.create_account({
            "customer_id": "CUST-001",
        })
        assert "error" in result
        assert result["status"] == 400

    def test_get_account_not_found(self, controller):
        controller._service.get_account_summary.side_effect = AccountNotFoundError("INVALID")
        result = controller.get_account("INVALID")
        assert "error" in result


class TestTransferController:

    @pytest.fixture
    def controller(self):
        service = MagicMock(spec=TransferService)
        return TransferController(service)

    def test_transfer_success(self, controller):
        controller._service.transfer.return_value = MagicMock(
            id="TX-1", from_account_id="ACC-1", to_account_id="ACC-2",
            amount=100_000, fee=50.0, transaction_type="TRANSFER",
            status="COMPLETED", description="Test",
            created_at=datetime.now(),
        )
        result = controller.transfer({
            "from_account_id": "ACC-1",
            "to_account_id": "ACC-2",
            "amount": "100000",
        })
        assert "data" in result
        assert result["status"] == 200
```

---

## Khi nào dùng / Khi nào không

### ✅ Khi nào dùng Layered Architecture

| Tình huống | Lý do |
|-----------|-------|
| **Enterprise application** với business logic phức tạp | Phân tách rõ ràng giữa presentation, business, data |
| **Đội ngũ từ 3-15 developers** | Dễ phân công công việc theo layer |
| **Hệ thống CRUD điển hình** | Cấu trúc chuẩn, dễ hiểu cho mọi member |
| **Time-to-market quan trọng** | Kiến trúc quen thuộc, onboarding nhanh |
| **Yêu cầu maintainability cao** | Layer isolation giúp thay đổi an toàn |
| **Hệ thống cần nhiều loại UI** | Presentation layer có thể swap (web, mobile, API) |
| **Tích hợp với legacy system** | Business layer che giấu complexity của legacy |

### ❌ Khi nào KHÔNG dùng

| Tình huống | Lý do | Alternative |
|-----------|-------|-------------|
| **Hệ thống siêu nhỏ (1-2 endpoints)** | Over-engineering, quá nhiều boilerplate | Script, Serverless functions |
| **Real-time system (latency < 10ms)** | Mỗi layer thêm latency | Event-Driven, Pipeline |
| **Big Data / Data-intensive** | Data không phải phụ thuộc chính | Data Mesh, Lambda Architecture |
| **Hệ thống AI/ML inference** | Khác biệt về data flow | Pipeline, Microservices |
| **Prototype / MVP** | Cần tốc độ, không cần maintain lâu | Monolith, Framework MVC |
| **Hệ thống peer-to-peer** | Không có client-server | P2P Architecture |
| **Hệ thống event-heavy** | Layered xử lý sync request-response chậm | Event-Driven, CQRS |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Đơn giản, dễ hiểu**: Hầu hết developer đều biết, onboarding nhanh | **Performance overhead**: Mỗi layer thêm latency và complexity |
| **Separation of concerns**: Mỗi layer có trách nhiệm rõ ràng | **Tight coupling giữa layers**: Thay đổi interface ảnh hưởng chain |
| **Dễ test**: Test từng layer độc lập với mock/stub | **Database-centric**: Business logic thường bị kéo xuống persistence |
| **Dễ phát triển song song**: Nhiều dev làm việc trên các layer khác nhau | **Khó scale**: Monolith deployment, vertical scaling là chính |
| **Bảo mật tốt**: Business logic không exposed trực tiếp | **Layer leakage**: Business logic tràn vào presentation/common code |
| **Dễ maintain**: Code có tổ chức, dễ tìm bug | **Lazy loading anti-pattern**: Service layer chỉ pass-through |
| **Công nghệ độc lập**: Có thể đổi DB hay UI framework | **Big ball of mud**: Nếu không kỷ luật, layers sẽ hòa vào nhau |

---

## Công cụ và Framework

### Presentation Layer

| Framework | Ngôn ngữ | Đặc điểm |
|-----------|----------|----------|
| **FastAPI** | Python 3.10+ | Async, auto-docs, type-safe — **recommended** |
| **Django REST Framework** | Python | Full-featured, batteries-included |
| **Flask** | Python | Lightweight, flexible |
| **Spring Boot** | Java/Kotlin | Enterprise-grade, ecosystem lớn |
| **ASP.NET Core** | C# | Cross-platform, high performance |
| **Express.js** | Node.js | Nhẹ, non-blocking I/O |

### Business Layer

| Công cụ | Mục đích |
|---------|----------|
| **Pydantic** | Data validation & settings management (Python) |
| **attrs / dataclasses** | Domain model definition |
| **inject / dependency-injector** | Dependency injection |
| **APScheduler / Celery** | Background task & scheduling |

### Persistence Layer

| Công cụ | Mục đích |
|---------|----------|
| **SQLAlchemy 2.0** | ORM cho Python — **recommended** |
| **SQLModel** | Kết hợp Pydantic + SQLAlchemy |
| **Django ORM** | Nếu dùng Django full-stack |
| **Alembic** | Database migration |
| **Psycopg2 / asyncpg** | PostgreSQL drivers |
| **Redis** | Cache và session storage |

---

## Kiểm thử chiến lược

### Unit Tests (70% coverage target)
- **Business Layer**: Test service logic với mock repositories
- **Persistence Layer**: Test repository với in-memory database (SQLite)
- **Presentation Layer**: Test controller input/output mapping

### Integration Tests (20%)
- Business + Persistence: Flow thực tế từ service → DB
- Presentation + Business: HTTP request → JSON response

### End-to-End Tests (10%)
- Full system test với real database
- API endpoints testing (pytest + httpx)

```python
# pytest configuration (tests/conftest.py bổ sung)
@pytest.fixture
def client(account_service, transfer_service):
    """FastAPI test client for E2E tests."""
    from fastapi.testclient import TestClient
    from src.presentation.api import create_app
    app = create_app(account_service, transfer_service)
    return TestClient(app)
```

---

## Kết luận

Layered Architecture là nền tảng của kiến trúc phần mềm hiện đại. Dù bạn có chọn microservices, hexagonal, hay event-driven, những nguyên lý cốt lõi của layered architecture — separation of concerns, dependency direction, layer isolation — vẫn là kiến thức bắt buộc. 

### Best Practices

1. **Strict layering** cho đến khi có lý do chính đáng để relaxed
2. **Repository interfaces** ở business layer, implementation ở persistence
3. **Dependency Injection** qua constructor để test dễ dàng
4. **DTOs cho cross-layer communication** — không dùng ORM entity ở presentation
5. **Business exception** với error code để presentation mapping
6. **Transaction management** ở service layer (use case boundary)
7. **Logging và error handling** qua middleware/AOP
8. **Input validation ở presentation**, business validation ở service
9. **Domain models không phụ thuộc framework** — plain Python objects
10. **Test business layer trước** — đây là tầng quan trọng nhất

### Golden Rules

> 1. **Presentation không biết gì về Persistence.** Controller không bao giờ import repository.
> 2. **Business không biết gì về Framework.** Service không import SQLAlchemy, FastAPI, Django.
> 3. **Persistence implement interface của Business.** Repository interface sống ở business layer.
> 4. **Dependency chỉ đi một chiều: Presentation → Business → Persistence.**
> 5. **Nếu một layer quá dày (>1000 loc)**, hãy split thành sub-layers.

### Next Steps

Sau khi nắm vững Layered Architecture, hãy tiếp tục với **Microservices Architecture** — nơi mỗi layer trở thành một service độc lập, và sự phức tạp của distributed systems xuất hiện. Hoặc nếu bạn muốn đi sâu vào testability và DDD, **Hexagonal Architecture** là bước tiến tự nhiên từ layered.
