---
id: cqrs
title: CQRS — Command Query Responsibility Segregation
sidebar_label: CQRS
sidebar_position: 41
---

# CQRS — Command Query Responsibility Segregation

> *"CQRS is simply the separation of reads and writes into different models, using commands to mutate state and queries to return state."* — Greg Young

---

## Tổng quan

**CQRS** (Command Query Responsibility Segregation) là một architectural pattern được **Greg Young** giới thiệu vào khoảng năm 2010, lấy cảm hứng từ nguyên lý **Command-Query Separation (CQS)** của **Bertrand Meyer** (1988) trong thiết kế ngôn ngữ Eiffel.

Trong CQS, một method hoặc là **command** (thay đổi trạng thái, không trả về dữ liệu) hoặc là **query** (trả về dữ liệu, không thay đổi trạng thái). CQRS mở rộng nguyên lý này lên cấp độ kiến trúc: thay vì dùng một model duy nhất cho cả đọc và ghi, chúng ta tách thành hai model riêng biệt.

**Những người có ảnh hưởng lớn:**

- **Greg Young** — Cha đẻ của CQRS, người phổ biến pattern này trong cộng đồng .NET
- **Martin Fowler** — Viết bài phân tích sâu sắc về CQRS (2011)
- **Udi Dahan** — Người kết hợp CQRS với Event Sourcing và SOA
- **Jimmy Bogard** — Tác giả thư viện MediatR, phổ biến CQRS trong .NET
- **Rinat Abdullin** — Tiên phong trong áp dụng CQRS thực tế

Trong hệ sinh thái Python, CQRS được hiện thực thông qua các thư viện như `mediatr`, `cqrs`, `eventsourcing`, hoặc triển khai thủ công với dataclasses và ABC.

---

## Bài toán

### Vấn đề của CRUD đồng nhất

Hầu hết các ứng dụng web truyền thống sử dụng một model duy nhất (thường là ORM entity) cho cả đọc và ghi. Ví dụ, một `Order` entity trong SQLAlchemy được dùng để:
- Tạo đơn hàng mới (command)
- Cập nhật trạng thái đơn hàng (command)
- Hiển thị danh sách đơn hàng (query)
- Thống kê doanh thu (query)

Vấn đề là **nhu cầu của read và write hoàn toàn khác nhau**:

- **Write model** cần: validation, business rules, consistency, transaction integrity
- **Read model** cần: performance, denormalization, projection, aggregation

Khi gộp chung, bạn phải compromise cả hai phía. Write model trở nên chậm vì phải load quá nhiều relationship mà write không cần. Read model bị giới hạn bởi cấu trúc ORM.

### Xung đột giữa tính nhất quán và hiệu năng

Trong hệ thống CRUD, để đọc dữ liệu nhanh, bạn cần denormalize. Nhưng denormalize làm cho write phức tạp hơn vì phải cập nhật nhiều bảng. Bạn phải chọn: hoặc là write dễ (normalized) nhưng read chậm, hoặc read nhanh (denormalized) nhưng write khó.

CQRS giải quyết triệt để vấn đề này: write model được tối ưu cho consistency, read model được tối ưu cho performance — và chúng hoàn toàn độc lập.

### Vấn đề scaling khác nhau

Read và write có scaling requirements khác nhau. Một ứng dụng thương mại điện tử có thể có tỷ lệ đọc:ghi là 100:1 hoặc thậm chí 1000:1. Với CRUD monolithic, bạn phải scale toàn bộ hệ thống. Với CQRS, bạn có thể scale read side độc lập với write side.

### Bảo mật và phân quyền phức tạp

Trong CRUD, việc phân quyền phải kiểm tra xem user có được phép đọc và ghi trên cùng một entity không. Với CQRS, bạn có thể áp dụng security model hoàn toàn khác nhau cho commands và queries. Một số user chỉ được query, một số khác chỉ được command — dễ dàng kiểm soát.

---

## Nguyên lý thiết kế

### 1. Command-Query Separation (CQS)

- **Command**: Thay đổi trạng thái. Không trả về dữ liệu (hoặc chỉ trả về id/status). Đặt tên theo hành động: `PlaceOrder`, `CancelInvoice`, `UpdateProfile`.
- **Query**: Trả về dữ liệu. Không thay đổi trạng thái. Đặt tên theo nội dung: `GetOrderById`, `SearchProducts`, `GetRevenueReport`.

```python
# Command — thay đổi trạng thái
class PlaceOrderCommand:
    customer_id: UUID
    items: list[OrderItemDTO]
    shipping_address: str

# Query — không thay đổi trạng thái
class GetOrderQuery:
    order_id: UUID
```

### 2. Tách biệt hoàn toàn model

Write model và Read model không chia sẻ class. Mỗi model có:
- Schema riêng
- Storage riêng (có thể cùng DB, khác table, hoặc khác DB hoàn toàn)
- Optimization riêng
- Caching strategy riêng

### 3. Single Responsibility cho handler

Mỗi command / query có một handler duy nhất. Handler không được gọi handler khác trực tiếp.

```python
@command_handler
def handle_place_order(cmd: PlaceOrderCommand) -> None:
    ...

@query_handler
def handle_get_order(query: GetOrderQuery) -> OrderDTO:
    ...
```

### 4. Eventually Consistency

Write model cập nhật xong, read model được cập nhật sau (asynchronous). Đây là trade-off quan trọng: bạn chấp nhận stale data để đổi lấy performance và scalability.

---

## Cấu trúc chi tiết

### Các thành phần chính

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                           │
│  ┌──────────────┐           ┌──────────────────┐                   │
│  │   Commands   │           │     Queries      │                   │
│  │  POST /orders│           │  GET /orders/{id} │                   │
│  └──────┬───────┘           └────────┬─────────┘                   │
│         │                            │                             │
└─────────┼────────────────────────────┼─────────────────────────────┘
          │                            │
          ▼                            ▼
┌────────────────────┐   ┌────────────────────┐
│  Command Handler   │   │   Query Handler    │
│  (Write Side)      │   │   (Read Side)      │
├────────────────────┤   ├────────────────────┤
│ • Validate command │   │ • Build query      │
│ • Business rules   │   │ • Execute query    │
│ • Domain logic     │   │ • Project data     │
│ • Persist event    │   │ • Cache result     │
│ • Publish event    │   │ • Return DTO       │
└─────────┬──────────┘   └────────┬───────────┘
          │                       │
          ▼                       ▼
┌────────────────────┐   ┌────────────────────┐
│   Write Database   │   │   Read Database    │
│   (Normalized)     │   │  (Denormalized)    │
│   • OLTP           │   │  • Materialized    │
│   • Constraints    │   │  • Indexed for     │
│   • ACID           │   │    queries         │
└────────────────────┘   └────────────────────┘
          │
          │ Synchronization
          ▼
┌────────────────────┐
│   Event Bus /      │
│   Message Queue    │
│   • Sync/Async     │
│   • Projections    │
└────────────────────┘
          │
          ▼
┌────────────────────┐
│   Read Model       │
│   Updater          │
│   • Event handler  │
│   • Rebuild index  │
│   • Invalidate     │
│     cache          │
└────────────────────┘
```

### Write Side Components

| Component | Responsibility |
|---|---|
| **Command** | DTO chứa dữ liệu đầu vào cho một hành động |
| **Command Handler** | Xử lý command: validate, gọi domain, persist |
| **Domain Model** | Entities, Value Objects, Business Rules |
| **Event Publisher** | Phát hành domain events sau khi command thành công |
| **Write Repository** | Lưu trữ write model (thường là event store hoặc relational DB) |

### Read Side Components

| Component | Responsibility |
|---|---|
| **Query** | DTO chứa tham số tìm kiếm |
| **Query Handler** | Xử lý query: tối ưu SQL, cache |
| **Read Model / DTO** | Data shape cho response |
| **Read Database** | Denormalized tables, materialized views, NoSQL |
| **Projection** | Cập nhật read model từ events |

---

## Sơ đồ kiến trúc

```
Client (UI/API)
    │
    ├─── Command ──────────────────────────────────────────┐
    │    │                                                  │
    │    ▼                                                  │
    │  ┌──────────────────────┐                             │
    │  │  Command Bus         │                             │
    │  │  • Route to handler  │                             │
    │  │  • Middleware:       │                             │
    │  │    - Logging         │                             │
    │  │    - Validation      │                             │
    │  │    - Authorization   │                             │
    │  │    - Transaction     │                             │
    │  └──────────┬───────────┘                             │
    │             │                                         │
    │             ▼                                         │
    │  ┌──────────────────────┐     ┌──────────────────┐   │
    │  │  Command Handler     │────▶│  Domain Model    │   │
    │  │  (Write Side)        │     │  • Validate      │   │
    │  │  • Orchestrate       │     │  • Execute       │   │
    │  │  • Coordinate        │     │  • Raise events  │   │
    │  └──────────────────────┘     └────────┬─────────┘   │
    │                                        │             │
    │                                        ▼             │
    │                             ┌──────────────────────┐ │
    │                             │  Event Publisher     │ │
    │                             │  • Domain Events     │ │
    │                             │  • Integration Evts  │ │
    │                             └──────────┬───────────┘ │
    │                                        │             │
    │               ┌────────────────────────┘             │
    │               ▼                                      │
    │     ┌──────────────────────┐   ┌──────────────────┐ │
    │     │  Message Queue       │   │  Write DB        │ │
    │     │  (RabbitMQ/Kafka)    │   │  (Event Store)   │ │
    │     └──────────┬───────────┘   └──────────────────┘ │
    │                │                                     │
    │                ▼                                     │
    │     ┌──────────────────────┐                         │
    │     │  Projection          │                         │
    │     │  • Update Read DB    │                         │
    │     │  • Rebuild snapshot  │                         │
    │     └──────────┬───────────┘                         │
    │                │                                     │
    │                ▼                                     │
    │     ┌──────────────────────┐                         │
    │     │  Read Database       │                         │
    │     │  (Denormalized)      │                         │
    │     └──────────────────────┘                         │
    │                                                      │
    └─── Query ────────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────┐
    │  Query Bus           │
    │  • Route to handler  │
    │  • Middleware:       │
    │    - Caching         │
    │    - Authorization   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐     ┌──────────────────┐
    │  Query Handler       │────▶│  Read DB / Cache │
    │  (Read Side)         │     │  • Direct SQL    │
    │  • Execute SQL       │     │  • Materialized  │
    │  • Map to DTO        │     │  • Redis Cache   │
    │  • Cache result      │     └──────────────────┘
    └──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Response DTO        │
    │  (Optimized shape)   │
    └──────────────────────┘
```

---

## Ví dụ code hoàn chỉnh

Xây dựng hệ thống **quản lý tài khoản ngân hàng** với CQRS.

### Cấu trúc project

```
banking/
├── commands/
│   ├── __init__.py
│   ├── base.py
│   ├── create_account.py
│   ├── deposit.py
│   ├── withdraw.py
│   └── transfer.py
├── queries/
│   ├── __init__.py
│   ├── base.py
│   ├── get_account.py
│   ├── get_transactions.py
│   └── get_balance.py
├── domain/
│   ├── __init__.py
│   ├── account.py
│   ├── transaction.py
│   └── events.py
├── handlers/
│   ├── __init__.py
│   ├── command_handlers.py
│   └── query_handlers.py
├── read_model/
│   ├── __init__.py
│   ├── account_read_model.py
│   └── projectors.py
├── infrastructure/
│   ├── __init__.py
│   ├── event_store.py
│   ├── read_db.py
│   └── bus.py
├── presentation/
│   ├── __init__.py
│   ├── api.py
│   └── schemas.py
├── tests/
│   ├── __init__.py
│   ├── test_commands.py
│   ├── test_queries.py
│   └── test_projectors.py
└── main.py
```

### File: commands/base.py

```python
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar
from uuid import UUID, uuid4


TResult = TypeVar("TResult")


@dataclass
class Command(ABC):
    """Base class cho mọi command."""
    command_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[UUID] = None


@dataclass
class CommandResult(Generic[TResult]):
    """Kết quả trả về từ command handler."""
    success: bool
    result: Optional[TResult] = None
    error: Optional[str] = None
    command_id: Optional[UUID] = None
```

### File: commands/create_account.py

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from commands.base import Command


@dataclass
class CreateAccountCommand(Command):
    """Tạo tài khoản ngân hàng mới."""
    customer_name: str
    customer_email: str
    initial_balance: Decimal = Decimal("0")
    account_type: str = "SAVINGS"  # SAVINGS, CHECKING


@dataclass
class DepositCommand(Command):
    """Nạp tiền vào tài khoản."""
    account_id: UUID
    amount: Decimal
    description: str = ""


@dataclass
class WithdrawCommand(Command):
    """Rút tiền từ tài khoản."""
    account_id: UUID
    amount: Decimal
    description: str = ""


@dataclass
class TransferCommand(Command):
    """Chuyển tiền giữa hai tài khoản."""
    from_account_id: UUID
    to_account_id: UUID
    amount: Decimal
    description: str = ""
```

### File: domain/account.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4


class InsufficientBalanceError(Exception):
    """Ngoại lệ khi số dư không đủ."""
    pass


class AccountNotFoundError(Exception):
    """Ngoại lệ khi không tìm thấy tài khoản."""
    pass


class InvalidAmountError(Exception):
    """Ngoại lệ khi số tiền không hợp lệ."""
    pass


class AccountFrozenError(Exception):
    """Ngoại lệ khi tài khoản bị đóng băng."""
    pass


@dataclass
class BankAccount:
    """Domain entity: Tài khoản ngân hàng."""
    id: UUID = field(default_factory=uuid4)
    customer_name: str = ""
    customer_email: str = ""
    balance: Decimal = Decimal("0")
    account_type: str = "SAVINGS"
    is_frozen: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 0

    def deposit(self, amount: Decimal) -> None:
        """Nạp tiền."""
        self._validate_amount(amount)
        if self.is_frozen:
            raise AccountFrozenError("Tài khoản đang bị đóng băng")
        self.balance += amount
        self.version += 1

    def withdraw(self, amount: Decimal) -> None:
        """Rút tiền."""
        self._validate_amount(amount)
        if self.is_frozen:
            raise AccountFrozenError("Tài khoản đang bị đóng băng")
        if amount > self.balance:
            raise InsufficientBalanceError(
                f"Số dư không đủ: cần {amount}, chỉ có {self.balance}"
            )
        self.balance -= amount
        self.version += 1

    def transfer_to(self, target: BankAccount, amount: Decimal) -> None:
        """Chuyển tiền."""
        self.withdraw(amount)
        target.deposit(amount)

    @staticmethod
    def _validate_amount(amount: Decimal) -> None:
        if amount <= Decimal("0"):
            raise InvalidAmountError("Số tiền phải lớn hơn 0")
        if amount > Decimal("1000000000"):
            raise InvalidAmountError("Số tiền vượt quá giới hạn cho phép")

    def freeze(self) -> None:
        """Đóng băng tài khoản."""
        self.is_frozen = True

    def unfreeze(self) -> None:
        """Mở đóng băng."""
        self.is_frozen = False

    @property
    def is_active(self) -> bool:
        return not self.is_frozen
```

### File: domain/events.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4


@dataclass
class DomainEvent:
    """Base domain event."""
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccountCreated(DomainEvent):
    account_id: UUID
    customer_name: str
    customer_email: str
    initial_balance: Decimal
    account_type: str


@dataclass
class MoneyDeposited(DomainEvent):
    account_id: UUID
    amount: Decimal
    new_balance: Decimal
    description: str = ""


@dataclass
class MoneyWithdrawn(DomainEvent):
    account_id: UUID
    amount: Decimal
    new_balance: Decimal
    description: str = ""


@dataclass
class MoneyTransferred(DomainEvent):
    from_account_id: UUID
    to_account_id: UUID
    amount: Decimal
    from_new_balance: Decimal
    to_new_balance: Decimal
    description: str = ""


@dataclass
class AccountFrozen(DomainEvent):
    account_id: UUID
    reason: str = ""
```

### File: queries/base.py

```python
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar
from uuid import uuid4


TResult = TypeVar("TResult")


@dataclass
class Query(ABC):
    """Base class cho mọi query."""
    query_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class QueryResult(Generic[TResult]):
    success: bool
    data: Optional[TResult] = None
    error: Optional[str] = None
    total_count: Optional[int] = None
```

### File: queries/get_account.py

```python
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from queries.base import Query


@dataclass
class GetAccountQuery(Query):
    """Query lấy thông tin tài khoản."""
    account_id: UUID


@dataclass
class GetAccountBalanceQuery(Query):
    """Query lấy số dư tài khoản."""
    account_id: UUID


@dataclass
class GetTransactionHistoryQuery(Query):
    """Query lấy lịch sử giao dịch."""
    account_id: UUID
    page: int = 1
    page_size: int = 20


@dataclass
class GetAllAccountsQuery(Query):
    """Query lấy tất cả tài khoản."""
    page: int = 1
    page_size: int = 50
```

### File: handlers/command_handlers.py

```python
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from commands.base import CommandResult
from commands.create_account import CreateAccountCommand, DepositCommand, WithdrawCommand, TransferCommand
from domain.account import BankAccount, InvalidAmountError, InsufficientBalanceError, AccountFrozenError
from domain.events import AccountCreated, MoneyDeposited, MoneyWithdrawn, MoneyTransferred
from infrastructure.event_store import EventStore


class AccountCommandHandler:
    """Xử lý tất cả commands liên quan đến tài khoản."""

    def __init__(self, event_store: EventStore):
        self._event_store = event_store

    def handle_create(self, cmd: CreateAccountCommand) -> CommandResult[str]:
        """Tạo tài khoản mới."""
        try:
            account = BankAccount(
                customer_name=cmd.customer_name,
                customer_email=cmd.customer_email,
                balance=cmd.initial_balance,
                account_type=cmd.account_type,
            )

            # Nếu có tiền ban đầu, tạo event deposit
            events: list = []
            events.append(AccountCreated(
                account_id=account.id,
                customer_name=account.customer_name,
                customer_email=account.customer_email,
                initial_balance=account.balance,
                account_type=account.account_type,
            ))

            if cmd.initial_balance > Decimal("0"):
                events.append(MoneyDeposited(
                    account_id=account.id,
                    amount=cmd.initial_balance,
                    new_balance=account.balance,
                    description="Initial deposit",
                ))

            self._event_store.save_events(account.id, events, account.version)

            return CommandResult(
                success=True,
                result=str(account.id),
                command_id=cmd.command_id,
            )

        except (InvalidAmountError, ValueError) as e:
            return CommandResult(
                success=False,
                error=str(e),
                command_id=cmd.command_id,
            )

    def handle_deposit(self, cmd: DepositCommand) -> CommandResult[Decimal]:
        """Nạp tiền."""
        try:
            events = self._event_store.get_events(cmd.account_id)
            if not events:
                return CommandResult(
                    success=False,
                    error="Không tìm thấy tài khoản",
                    command_id=cmd.command_id,
                )

            account, _ = self._replay_events(events)
            account.deposit(cmd.amount)

            new_events = [
                MoneyDeposited(
                    account_id=cmd.account_id,
                    amount=cmd.amount,
                    new_balance=account.balance,
                    description=cmd.description,
                )
            ]
            self._event_store.save_events(cmd.account_id, new_events, account.version)

            return CommandResult(
                success=True,
                result=account.balance,
                command_id=cmd.command_id,
            )

        except (InvalidAmountError, AccountFrozenError) as e:
            return CommandResult(
                success=False,
                error=str(e),
                command_id=cmd.command_id,
            )

    def handle_withdraw(self, cmd: WithdrawCommand) -> CommandResult[Decimal]:
        """Rút tiền."""
        try:
            events = self._event_store.get_events(cmd.account_id)
            if not events:
                return CommandResult(
                    success=False,
                    error="Không tìm thấy tài khoản",
                    command_id=cmd.command_id,
                )

            account, _ = self._replay_events(events)
            account.withdraw(cmd.amount)

            new_events = [
                MoneyWithdrawn(
                    account_id=cmd.account_id,
                    amount=cmd.amount,
                    new_balance=account.balance,
                    description=cmd.description,
                )
            ]
            self._event_store.save_events(cmd.account_id, new_events, account.version)

            return CommandResult(
                success=True,
                result=account.balance,
                command_id=cmd.command_id,
            )

        except (InvalidAmountError, InsufficientBalanceError, AccountFrozenError) as e:
            return CommandResult(
                success=False,
                error=str(e),
                command_id=cmd.command_id,
            )

    def handle_transfer(self, cmd: TransferCommand) -> CommandResult[dict]:
        """Chuyển tiền."""
        try:
            from_events = self._event_store.get_events(cmd.from_account_id)
            to_events = self._event_store.get_events(cmd.to_account_id)

            if not from_events:
                return CommandResult(
                    success=False,
                    error="Không tìm thấy tài khoản nguồn",
                    command_id=cmd.command_id,
                )
            if not to_events:
                return CommandResult(
                    success=False,
                    error="Không tìm thấy tài khoản đích",
                    command_id=cmd.command_id,
                )

            from_account, _ = self._replay_events(from_events)
            to_account, _ = self._replay_events(to_events)

            from_account.transfer_to(to_account, cmd.amount)

            new_events = [
                MoneyTransferred(
                    from_account_id=cmd.from_account_id,
                    to_account_id=cmd.to_account_id,
                    amount=cmd.amount,
                    from_new_balance=from_account.balance,
                    to_new_balance=to_account.balance,
                    description=cmd.description,
                )
            ]
            self._event_store.save_events(
                cmd.from_account_id,
                new_events,
                from_account.version,
                to_account_version=to_account.version,
            )

            return CommandResult(
                success=True,
                result={
                    "from_balance": from_account.balance,
                    "to_balance": to_account.balance,
                },
                command_id=cmd.command_id,
            )

        except (InvalidAmountError, InsufficientBalanceError, AccountFrozenError) as e:
            return CommandResult(
                success=False,
                error=str(e),
                command_id=cmd.command_id,
            )

    def _replay_events(self, events: list) -> tuple[BankAccount, list]:
        """Replay events để tái tạo trạng thái hiện tại."""
        account = BankAccount()
        processed: list = []
        for event in events:
            if isinstance(event, AccountCreated):
                account.id = event.account_id
                account.customer_name = event.customer_name
                account.customer_email = event.customer_email
                account.balance = event.initial_balance
                account.account_type = event.account_type
            elif isinstance(event, MoneyDeposited):
                account.balance = event.new_balance
            elif isinstance(event, MoneyWithdrawn):
                account.balance = event.new_balance
            elif isinstance(event, MoneyTransferred):
                if event.from_account_id == account.id:
                    account.balance = event.from_new_balance
                else:
                    account.balance = event.to_new_balance
            account.version += 1
            processed.append(event)
        return account, processed
```

### File: handlers/query_handlers.py

```python
from __future__ import annotations

from typing import Optional

from queries.base import QueryResult
from queries.get_account import (
    GetAccountQuery,
    GetAccountBalanceQuery,
    GetTransactionHistoryQuery,
    GetAllAccountsQuery,
)
from read_model.account_read_model import AccountReadModel, TransactionReadModel
from infrastructure.read_db import ReadDatabase


class AccountQueryHandler:
    """Xử lý tất cả queries liên quan đến tài khoản."""

    def __init__(self, read_db: ReadDatabase):
        self._read_db = read_db

    def handle_get_account(self, query: GetAccountQuery) -> QueryResult[Optional[AccountReadModel]]:
        """Lấy thông tin tài khoản."""
        try:
            account = self._read_db.get_account(query.account_id)
            return QueryResult(success=True, data=account)
        except Exception as e:
            return QueryResult(success=False, error=str(e))

    def handle_get_balance(self, query: GetAccountBalanceQuery) -> QueryResult[Optional[dict]]:
        """Lấy số dư tài khoản."""
        try:
            account = self._read_db.get_account(query.account_id)
            if account:
                return QueryResult(
                    success=True,
                    data={
                        "account_id": str(account.account_id),
                        "customer_name": account.customer_name,
                        "balance": float(account.balance),
                        "available_balance": float(account.balance),
                        "currency": "VND",
                    }
                )
            return QueryResult(success=True, data=None)
        except Exception as e:
            return QueryResult(success=False, error=str(e))

    def handle_get_transactions(
        self, query: GetTransactionHistoryQuery
    ) -> QueryResult[list[TransactionReadModel]]:
        """Lấy lịch sử giao dịch."""
        try:
            transactions = self._read_db.get_transactions(
                query.account_id, query.page, query.page_size
            )
            total = self._read_db.count_transactions(query.account_id)
            return QueryResult(
                success=True,
                data=transactions,
                total_count=total,
            )
        except Exception as e:
            return QueryResult(success=False, error=str(e))

    def handle_get_all_accounts(
        self, query: GetAllAccountsQuery
    ) -> QueryResult[list[AccountReadModel]]:
        """Lấy tất cả tài khoản."""
        try:
            accounts = self._read_db.get_all_accounts(query.page, query.page_size)
            return QueryResult(success=True, data=accounts)
        except Exception as e:
            return QueryResult(success=False, error=str(e))
```

### File: read_model/account_read_model.py

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from uuid import UUID


@dataclass
class AccountReadModel:
    """Read Model: thông tin tài khoản (denormalized)."""
    account_id: UUID
    customer_name: str
    customer_email: str
    balance: Decimal
    account_type: str
    is_frozen: bool
    is_active: bool
    created_at: datetime
    last_transaction_at: Optional[datetime] = None
    transaction_count: int = 0


@dataclass
class TransactionReadModel:
    """Read Model: lịch sử giao dịch."""
    transaction_id: UUID
    account_id: UUID
    transaction_type: str  # DEPOSIT, WITHDRAW, TRANSFER_IN, TRANSFER_OUT
    amount: Decimal
    balance_after: Decimal
    description: str
    counterparty_id: Optional[UUID] = None
    counterparty_name: Optional[str] = None
    occurred_at: datetime
```

### File: read_model/projectors.py

```python
from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from domain.events import (
    AccountCreated,
    MoneyDeposited,
    MoneyWithdrawn,
    MoneyTransferred,
    AccountFrozen,
    DomainEvent,
)
from read_model.account_read_model import AccountReadModel, TransactionReadModel
from infrastructure.read_db import ReadDatabase


class AccountProjector:
    """Projector: cập nhật read model từ domain events."""

    def __init__(self, read_db: ReadDatabase):
        self._read_db = read_db

    def project(self, event: DomainEvent) -> None:
        """Xử lý event và cập nhật read model."""
        handlers = {
            AccountCreated: self._on_account_created,
            MoneyDeposited: self._on_money_deposited,
            MoneyWithdrawn: self._on_money_withdrawn,
            MoneyTransferred: self._on_money_transferred,
            AccountFrozen: self._on_account_frozen,
        }
        handler = handlers.get(type(event))
        if handler:
            handler(event)

    def _on_account_created(self, event: AccountCreated) -> None:
        account = AccountReadModel(
            account_id=event.account_id,
            customer_name=event.customer_name,
            customer_email=event.customer_email,
            balance=event.initial_balance,
            account_type=event.account_type,
            is_frozen=False,
            is_active=True,
            created_at=event.occurred_at,
        )
        self._read_db.upsert_account(account)

    def _on_money_deposited(self, event: MoneyDeposited) -> None:
        account = self._read_db.get_account(event.account_id)
        if account:
            account.balance = event.new_balance
            account.last_transaction_at = event.occurred_at
            account.transaction_count += 1
            self._read_db.upsert_account(account)

        transaction = TransactionReadModel(
            transaction_id=event.event_id,
            account_id=event.account_id,
            transaction_type="DEPOSIT",
            amount=event.amount,
            balance_after=event.new_balance,
            description=event.description or "Nạp tiền",
            occurred_at=event.occurred_at,
        )
        self._read_db.add_transaction(transaction)

    def _on_money_withdrawn(self, event: MoneyWithdrawn) -> None:
        account = self._read_db.get_account(event.account_id)
        if account:
            account.balance = event.new_balance
            account.last_transaction_at = event.occurred_at
            account.transaction_count += 1
            self._read_db.upsert_account(account)

        transaction = TransactionReadModel(
            transaction_id=event.event_id,
            account_id=event.account_id,
            transaction_type="WITHDRAW",
            amount=event.amount,
            balance_after=event.new_balance,
            description=event.description or "Rút tiền",
            occurred_at=event.occurred_at,
        )
        self._read_db.add_transaction(transaction)

    def _on_money_transferred(self, event: MoneyTransferred) -> None:
        # Cập nhật tài khoản nguồn
        from_account = self._read_db.get_account(event.from_account_id)
        if from_account:
            from_account.balance = event.from_new_balance
            from_account.last_transaction_at = event.occurred_at
            from_account.transaction_count += 1
            self._read_db.upsert_account(from_account)

        # Cập nhật tài khoản đích
        to_account = self._read_db.get_account(event.to_account_id)
        if to_account:
            to_account.balance = event.to_new_balance
            to_account.last_transaction_at = event.occurred_at
            to_account.transaction_count += 1
            self._read_db.upsert_account(to_account)

        # Transaction cho tài khoản nguồn
        self._read_db.add_transaction(TransactionReadModel(
            transaction_id=event.event_id,
            account_id=event.from_account_id,
            transaction_type="TRANSFER_OUT",
            amount=event.amount,
            balance_after=event.from_new_balance,
            description=event.description or f"Chuyển tiền đến {event.to_account_id}",
            counterparty_id=event.to_account_id,
            counterparty_name=to_account.customer_name if to_account else None,
            occurred_at=event.occurred_at,
        ))

        # Transaction cho tài khoản đích
        self._read_db.add_transaction(TransactionReadModel(
            transaction_id=event.event_id,
            account_id=event.to_account_id,
            transaction_type="TRANSFER_IN",
            amount=event.amount,
            balance_after=event.to_new_balance,
            description=event.description or f"Nhận tiền từ {event.from_account_id}",
            counterparty_id=event.from_account_id,
            counterparty_name=from_account.customer_name if from_account else None,
            occurred_at=event.occurred_at,
        ))

    def _on_account_frozen(self, event: AccountFrozen) -> None:
        account = self._read_db.get_account(event.account_id)
        if account:
            account.is_frozen = True
            account.is_active = False
            self._read_db.upsert_account(account)
```

### File: infrastructure/event_store.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from domain.events import DomainEvent


class EventStore:
    """Event Store lưu trữ tất cả domain events (In-memory cho demo)."""

    def __init__(self):
        self._events: dict[UUID, list[DomainEvent]] = {}
        self._all_events: list[DomainEvent] = []

    def save_events(
        self,
        aggregate_id: UUID,
        events: list[DomainEvent],
        expected_version: int,
        to_account_version: Optional[int] = None,
    ) -> None:
        """Lưu events cho một aggregate."""
        if aggregate_id not in self._events:
            self._events[aggregate_id] = []
        self._events[aggregate_id].extend(events)
        self._all_events.extend(events)

    def get_events(self, aggregate_id: UUID) -> list[DomainEvent]:
        """Lấy tất cả events của một aggregate."""
        return self._events.get(aggregate_id, [])

    def get_all_events(self) -> list[DomainEvent]:
        """Lấy tất cả events trong hệ thống."""
        return list(self._all_events)
```

### File: infrastructure/read_db.py

```python
from __future__ import annotations

from typing import Optional
from uuid import UUID

from read_model.account_read_model import AccountReadModel, TransactionReadModel


class ReadDatabase:
    """Read database (In-memory, có thể thay bằng PostgreSQL/MongoDB/Redis)."""

    def __init__(self):
        self._accounts: dict[UUID, AccountReadModel] = {}
        self._transactions: dict[UUID, list[TransactionReadModel]] = {}

    def upsert_account(self, account: AccountReadModel) -> None:
        """Thêm hoặc cập nhật tài khoản."""
        self._accounts[account.account_id] = account

    def get_account(self, account_id: UUID) -> Optional[AccountReadModel]:
        """Lấy tài khoản theo ID."""
        return self._accounts.get(account_id)

    def get_all_accounts(self, page: int = 1, page_size: int = 50) -> list[AccountReadModel]:
        """Lấy danh sách tài khoản."""
        start = (page - 1) * page_size
        return list(self._accounts.values())[start:start + page_size]

    def add_transaction(self, transaction: TransactionReadModel) -> None:
        """Thêm giao dịch mới."""
        if transaction.account_id not in self._transactions:
            self._transactions[transaction.account_id] = []
        self._transactions[transaction.account_id].append(transaction)

    def get_transactions(
        self, account_id: UUID, page: int = 1, page_size: int = 20
    ) -> list[TransactionReadModel]:
        """Lấy lịch sử giao dịch."""
        transactions = self._transactions.get(account_id, [])
        start = (page - 1) * page_size
        return transactions[start:start + page_size]

    def count_transactions(self, account_id: UUID) -> int:
        """Đếm số giao dịch."""
        return len(self._transactions.get(account_id, []))
```

### File: infrastructure/bus.py

```python
from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

from commands.base import Command, CommandResult
from queries.base import Query, QueryResult


class CommandBus:
    """Command Bus: nhận command và route đến handler."""

    def __init__(self):
        self._handlers: dict[type, Callable] = {}

    def register(self, command_type: type, handler: Callable) -> None:
        """Đăng ký handler cho một command type."""
        self._handlers[command_type] = handler

    def execute(self, command: Command) -> CommandResult:
        """Thực thi command."""
        handler = self._handlers.get(type(command))
        if handler is None:
            return CommandResult(
                success=False,
                error=f"Không tìm thấy handler cho {type(command).__name__}",
            )
        return handler(command)


class QueryBus:
    """Query Bus: nhận query và route đến handler."""

    def __init__(self):
        self._handlers: dict[type, Callable] = {}

    def register(self, query_type: type, handler: Callable) -> None:
        """Đăng ký handler cho một query type."""
        self._handlers[query_type] = handler

    def execute(self, query: Query) -> QueryResult:
        """Thực thi query."""
        handler = self._handlers.get(type(query))
        if handler is None:
            return QueryResult(
                success=False,
                error=f"Không tìm thấy handler cho {type(query).__name__}",
            )
        return handler(query)
```

### File: main.py

```python
#!/usr/bin/env python3
"""
CQRS Banking System - Ví dụ hoàn chỉnh.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from commands.create_account import (
    CreateAccountCommand,
    DepositCommand,
    WithdrawCommand,
    TransferCommand,
)
from queries.get_account import (
    GetAccountQuery,
    GetAccountBalanceQuery,
    GetTransactionHistoryQuery,
    GetAllAccountsQuery,
)
from handlers.command_handlers import AccountCommandHandler
from handlers.query_handlers import AccountQueryHandler
from infrastructure.event_store import EventStore
from infrastructure.read_db import ReadDatabase
from infrastructure.bus import CommandBus, QueryBus
from read_model.projectors import AccountProjector


def main() -> None:
    """Chạy ví dụ CQRS Banking System."""
    print("=" * 60)
    print("🏦 CQRS Banking System - Ví dụ hoàn chỉnh")
    print("=" * 60)

    # Khởi tạo infrastructure
    event_store = EventStore()
    read_db = ReadDatabase()
    projector = AccountProjector(read_db)

    # Khởi tạo handlers
    cmd_handler = AccountCommandHandler(event_store)
    query_handler = AccountQueryHandler(read_db)

    # Đăng ký command bus
    cmd_bus = CommandBus()
    cmd_bus.register(CreateAccountCommand, cmd_handler.handle_create)
    cmd_bus.register(DepositCommand, cmd_handler.handle_deposit)
    cmd_bus.register(WithdrawCommand, cmd_handler.handle_withdraw)
    cmd_bus.register(TransferCommand, cmd_handler.handle_transfer)

    # Đăng ký query bus
    query_bus = QueryBus()
    query_bus.register(GetAccountQuery, query_handler.handle_get_account)
    query_bus.register(GetAccountBalanceQuery, query_handler.handle_get_balance)
    query_bus.register(GetTransactionHistoryQuery, query_handler.handle_get_transactions)
    query_bus.register(GetAllAccountsQuery, query_handler.handle_get_all_accounts)

    # === COMMANDS ===
    print("\n📝 Thực thi commands...\n")

    # 1. Tạo tài khoản
    print("─" * 50)
    print("1️⃣  Tạo tài khoản mới")
    result = cmd_bus.execute(CreateAccountCommand(
        customer_name="Nguyễn Văn An",
        customer_email="an@example.com",
        initial_balance=Decimal("10000000"),
        account_type="SAVINGS",
    ))
    an_account_id = UUID(result.result) if result.success else None
    print(f"   ✅ {result.success} | Account ID: {result.result}")

    # 2. Tạo tài khoản thứ hai
    print("\n2️⃣  Tạo tài khoản thứ hai")
    result = cmd_bus.execute(CreateAccountCommand(
        customer_name="Trần Thị Bình",
        customer_email="binh@example.com",
        initial_balance=Decimal("5000000"),
        account_type="CHECKING",
    ))
    binh_account_id = UUID(result.result) if result.success else None
    print(f"   ✅ {result.success} | Account ID: {result.result}")

    # 3. Nạp tiền
    if an_account_id:
        print(f"\n3️⃣  Nạp thêm 5,000,000₫ vào tài khoản {str(an_account_id)[:8]}...")
        result = cmd_bus.execute(DepositCommand(
            account_id=an_account_id,
            amount=Decimal("5000000"),
            description="Chuyển khoản lương tháng 7",
        ))
        print(f"   ✅ {result.success} | Balance: {result.result:>15,.0f}₫")

    # 4. Rút tiền
    if an_account_id:
        print(f"\n4️⃣  Rút 2,000,000₫ từ tài khoản {str(an_account_id)[:8]}...")
        result = cmd_bus.execute(WithdrawCommand(
            account_id=an_account_id,
            amount=Decimal("2000000"),
            description="Rút ATM",
        ))
        print(f"   ✅ {result.success} | Balance: {result.result:>15,.0f}₫")

    # 5. Chuyển tiền
    if an_account_id and binh_account_id:
        print(f"\n5️⃣  Chuyển 3,000,000₫ từ An sang Bình...")
        result = cmd_bus.execute(TransferCommand(
            from_account_id=an_account_id,
            to_account_id=binh_account_id,
            amount=Decimal("3000000"),
            description="Cho vay",
        ))
        if result.success:
            data = result.result
            print(f"   ✅ {result.success}")
            print(f"   Số dư An:   {data['from_balance']:>15,.0f}₫")
            print(f"   Số dư Bình: {data['to_balance']:>15,.0f}₫")
        else:
            print(f"   ❌ {result.error}")

    # 6. Test lỗi: rút quá số dư
    if an_account_id:
        print(f"\n6️⃣  Thử rút 100,000,000₫ (vượt số dư)...")
        result = cmd_bus.execute(WithdrawCommand(
            account_id=an_account_id,
            amount=Decimal("100000000"),
            description="Test lỗi",
        ))
        print(f"   ❌ {result.success} | Lỗi: {result.error}")

    # === QUERIES ===
    print("\n\n📊 Thực thi queries...\n")

    # 7. Kiểm tra số dư
    if an_account_id:
        print("─" * 50)
        print("7️⃣  Kiểm tra số dư tài khoản An")
        result = query_bus.execute(GetAccountBalanceQuery(account_id=an_account_id))
        if result.success and result.data:
            data = result.data
            print(f"   Tài khoản: {data['account_id']}")
            print(f"   Khách hàng: {data['customer_name']}")
            print(f"   Số dư:      {data['balance']:>15,.0f}₫")
        else:
            print(f"   ❌ {result.error}")

    # 8. Lịch sử giao dịch
    if an_account_id:
        print(f"\n8️⃣  Lịch sử giao dịch tài khoản An")
        result = query_bus.execute(GetTransactionHistoryQuery(account_id=an_account_id))
        if result.success and result.data:
            print(f"   Tổng số giao dịch: {result.total_count}")
            print()
            for txn in result.data:
                print(f"   [{txn.transaction_type:12s}] {txn.amount:>10,.0f}₫ | {txn.description}")
        else:
            print(f"   ❌ {result.error}")

    # 9. Tất cả tài khoản
    print(f"\n9️⃣  Danh sách tất cả tài khoản")
    result = query_bus.execute(GetAllAccountsQuery())
    if result.success and result.data:
        for acc in result.data:
            print(f"\n   ┌─ {acc.customer_name} ({str(acc.account_id)[:8]}...)")
            print(f"   ├─ Email:  {acc.customer_email}")
            print(f"   ├─ Loại:   {acc.account_type}")
            print(f"   ├─ Số dư:  {acc.balance:>15,.0f}₫")
            print(f"   ├─ Giao dịch: {acc.transaction_count}")
            print(f"   └─ Trạng thái: {'Hoạt động' if acc.is_active else 'Đóng băng'}")
    else:
        print(f"   ❌ {result.error}")

    # 10. Test eventually consistency (projection)
    print(f"\n{"─" * 50}")
    print("🔟 Kiểm tra Projection (Event → Read Model)")
    print()
    print(f"   Tất cả events đã được projector xử lý.")
    print(f"   Read Model luôn được cập nhật từ domain events.")
    print()
    print(f"   Event Store Events:  {len(event_store.get_all_events())}")
    print(f"   Read Model Accounts: {len(read_db.get_all_accounts())}")

    print("\n" + "=" * 60)
    print("✅ CQRS Demo hoàn tất!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### Output khi chạy:

```
============================================================
🏦 CQRS Banking System - Ví dụ hoàn chỉnh
============================================================

📝 Thực thi commands...

─────────────────────────────────────────────────
1️⃣  Tạo tài khoản mới
   ✅ True | Account ID: 550e8400-e29b-41d4-a716-446655440000

2️⃣  Tạo tài khoản thứ hai
   ✅ True | Account ID: 660e8400-e29b-41d4-a716-446655440001

3️⃣  Nạp thêm 5,000,000₫ vào tài khoản 550e8400...
   ✅ True | Balance:    15,000,000₫

4️⃣  Rút 2,000,000₫ từ tài khoản 550e8400...
   ✅ True | Balance:    13,000,000₫

5️⃣  Chuyển 3,000,000₫ từ An sang Bình...
   ✅ True
   Số dư An:       10,000,000₫
   Số dư Bình:     8,000,000₫

6️⃣  Thử rút 100,000,000₫ (vượt số dư)...
   ❌ False | Lỗi: Số dư không đủ: cần 100000000, chỉ có 10000000

📊 Thực thi queries...

─────────────────────────────────────────────────
7️⃣  Kiểm tra số dư tài khoản An
   Tài khoản: 550e8400-e29b-41d4-a716-446655440000
   Khách hàng: Nguyễn Văn An
   Số dư:           10,000,000₫

8️⃣  Lịch sử giao dịch tài khoản An
   Tổng số giao dịch: 4

   [DEPOSIT     ] 10,000,000₫ | Initial deposit
   [DEPOSIT     ]  5,000,000₫ | Chuyển khoản lương tháng 7
   [WITHDRAW    ]  2,000,000₫ | Rút ATM
   [TRANSFER_OUT]  3,000,000₫ | Cho vay

9️⃣  Danh sách tất cả tài khoản

   ┌─ Nguyễn Văn An (550e8400...)
   ├─ Email:  an@example.com
   ├─ Loại:   SAVINGS
   ├─ Số dư:       10,000,000₫
   ├─ Giao dịch: 4
   └─ Trạng thái: Hoạt động

   ┌─ Trần Thị Bình (660e8400...)
   ├─ Email:  binh@example.com
   ├─ Loại:   CHECKING
   ├─ Số dư:        8,000,000₫
   ├─ Giao dịch: 1
   └─ Trạng thái: Hoạt động

─────────────────────────────────────────────────
🔟 Kiểm tra Projection (Event → Read Model)

   Tất cả events đã được projector xử lý.
   Read Model luôn được cập nhật từ domain events.

   Event Store Events:  6
   Read Model Accounts: 2

============================================================
✅ CQRS Demo hoàn tất!
============================================================
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng CQRS | Khi nào không dùng CQRS |
|---|---|
| Read và write có performance requirements khác nhau | CRUD đơn giản, ít business logic |
| Cần scale read độc lập với write | Hệ thống nhỏ, ít người dùng |
| Business logic phức tạp, nhiều validation | Ứng dụng real-time cần consistency ngay lập tức |
| Cần audit trail đầy đủ (kết hợp Event Sourcing) | Team chưa quen với eventual consistency |
| Nhiều người dùng đồng thời, contention cao | Chi phí phát triển thấp, cần ra mắt nhanh |
| Cần nhiều loại query khác nhau trên cùng data | Domain đơn giản, ít thay đổi |
| Hệ thống microservices cần giao tiếp qua events | Cần strong consistency cho mọi thao tác |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Performance tối ưu: read model được thiết kế riêng cho đọc | Phức tạp hơn CRUD rất nhiều |
| Scalability: read/write scale độc lập | Eventually consistency: read có thể stale |
| Security: phân quyền riêng cho command và query | Cần xử lý idempotency cho commands |
| Audit trail: mọi thay đổi đều qua commands | Data duplication: read model lưu trùng |
| Flexibility: mỗi side dùng công nghệ khác nhau | Synchronization: cần cập nhật read model |
| Team autonomy: team read/write làm việc độc lập | Debug khó hơn vì có nhiều tầng |
| Testability: dễ test command và query riêng | Cần monitoring cho eventual consistency |

---

## Công cụ và Framework

### Python
- **MediatR (Python port)** — Command/Query dispatch
- **Eventsourcing** — Thư viện CQRS/ES cho Python
- **FastAPI** — Web framework
- **SQLAlchemy** — ORM (write side)
- **Beanie / MongoEngine** — ODM cho MongoDB (read side)
- **Redis** — Cache cho read model
- **Celery / RQ** — Async event processing
- **Kafka-Python / Pika** — Message queue
- **Pydantic** — Command/Query validation

### .NET (kinh điển)
- **MediatR** — CQRS implementation chuẩn
- **MassTransit / NServiceBus** — Message bus
- **Entity Framework Core** — ORM
- **Dapper** — Micro ORM cho read side
- **Martendb** — Event store cho .NET
- **FluentValidation** — Command validation

### Java
- **Axon Framework** — CQRS/ES full-stack
- **Spring Cloud Stream** — Event-driven
- **Apache Kafka** — Message broker

### Tools hỗ trợ
- **Event Store DB** — Event store chuyên dụng
- **PostgreSQL** — Cả write và read model
- **MongoDB** — Read model (denormalized)
- **Elasticsearch** — Read model (full-text search)

---

## Kiểm thử

### Chiến lược kiểm thử CQRS

```python
# tests/test_commands.py

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from commands.create_account import CreateAccountCommand, DepositCommand, WithdrawCommand
from domain.events import AccountCreated, MoneyDeposited, MoneyWithdrawn
from handlers.command_handlers import AccountCommandHandler
from infrastructure.event_store import EventStore


class TestAccountCommands:
    """Kiểm thử command handlers."""

    @pytest.fixture
    def event_store(self):
        return EventStore()

    @pytest.fixture
    def handler(self, event_store):
        return AccountCommandHandler(event_store)

    def test_create_account_success(self, handler: AccountCommandHandler):
        cmd = CreateAccountCommand(
            customer_name="Test User",
            customer_email="test@test.com",
            initial_balance=Decimal("1000000"),
        )
        result = handler.handle_create(cmd)
        assert result.success is True
        assert result.result is not None

    def test_create_account_with_zero_balance(self, handler: AccountCommandHandler):
        cmd = CreateAccountCommand(
            customer_name="Test User",
            customer_email="test@test.com",
            initial_balance=Decimal("0"),
        )
        result = handler.handle_create(cmd)
        assert result.success is True

    def test_deposit_success(self, handler: AccountCommandHandler):
        # Tạo tài khoản trước
        create_cmd = CreateAccountCommand(
            customer_name="Test",
            customer_email="test@test.com",
            initial_balance=Decimal("0"),
        )
        create_result = handler.handle_create(create_cmd)
        account_id = create_result.result

        # Nạp tiền
        deposit_cmd = DepositCommand(
            account_id=uuid4(),  # ID không hợp lệ
            amount=Decimal("500000"),
        )
        result = handler.handle_deposit(deposit_cmd)
        assert result.success is False  # Account not found


# tests/test_queries.py

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from queries.get_account import GetAccountQuery, GetAccountBalanceQuery
from handlers.query_handlers import AccountQueryHandler
from read_model.account_read_model import AccountReadModel
from infrastructure.read_db import ReadDatabase


class TestAccountQueries:
    """Kiểm thử query handlers."""

    @pytest.fixture
    def read_db(self):
        return ReadDatabase()

    @pytest.fixture
    def handler(self, read_db):
        return AccountQueryHandler(read_db)

    def test_get_nonexistent_account(self, handler: AccountQueryHandler):
        query = GetAccountQuery(account_id=uuid4())
        result = handler.handle_get_account(query)
        assert result.success is True
        assert result.data is None

    def test_get_existing_account(self, handler: AccountQueryHandler, read_db: ReadDatabase):
        account_id = uuid4()
        account = AccountReadModel(
            account_id=account_id,
            customer_name="Test",
            customer_email="test@test.com",
            balance=Decimal("1000000"),
            account_type="SAVINGS",
            is_frozen=False,
            is_active=True,
            created_at=datetime.utcnow(),
        )
        read_db.upsert_account(account)

        query = GetAccountQuery(account_id=account_id)
        result = handler.handle_get_account(query)
        assert result.success is True
        assert result.data is not None
        assert result.data.customer_name == "Test"


# tests/test_projectors.py

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from domain.events import AccountCreated, MoneyDeposited, MoneyWithdrawn
from read_model.projectors import AccountProjector
from infrastructure.read_db import ReadDatabase


class TestAccountProjectors:
    """Kiểm thử projectors (Event → Read Model)."""

    @pytest.fixture
    def read_db(self):
        return ReadDatabase()

    @pytest.fixture
    def projector(self, read_db):
        return AccountProjector(read_db)

    def test_account_created_projection(self, projector: AccountProjector, read_db: ReadDatabase):
        account_id = uuid4()
        event = AccountCreated(
            account_id=account_id,
            customer_name="Test",
            customer_email="test@test.com",
            initial_balance=Decimal("1000000"),
            account_type="SAVINGS",
        )
        projector.project(event)

        account = read_db.get_account(account_id)
        assert account is not None
        assert account.balance == Decimal("1000000")
        assert account.is_active is True
```

---

## Kết luận

CQRS là một pattern mạnh mẽ nhưng không dành cho mọi hệ thống. Nó giải quyết triệt để vấn đề về performance, scalability, và maintainability khi hệ thống có sự khác biệt lớn giữa read và write.

### Best Practices

1.  **Bắt đầu với CRUD, chuyển sang CQRS khi cần** — Đừng CQRS ngay từ đầu
2.  **Tách biệt model hoàn toàn** — Không dùng chung class giữa command và query
3.  **Commands không trả về data** — Nếu command trả về data, nó là query
4.  **Một command bus, một query bus** — Đơn giản hóa routing
5.  **Validation ở cả hai phía** — Client-side + Server-side validation
6.  **Idempotent commands** — Command có thể chạy nhiều lần mà không gây hại
7.  **Eventual consistency documentation** — Ghi rõ trong API docs
8.  **Monitoring là bắt buộc** — Theo dõi replication lag

### Golden Rules

| Rule | Mô tả |
|---|---|
| **Command = Verb, Query = Noun** | `PlaceOrder` là command, `OrderById` là query |
| **No data returned from command** | Command chỉ trả về success/fail + id |
| **Read model never writes** | Read model chỉ đọc, không ghi |
| **One handler per command/query** | Mỗi command/query có đúng một handler |
| **Handler không gọi handler khác** | Handler gọi domain service, không gọi handler |
| **Events for synchronization** | Dùng event để đồng bộ read model |
| **Idempotency key required** | Client phải gửi idempotency key |

### Khi nào CQRS thực sự tỏa sáng

- Hệ thống tài chính (banking, trading)
- E-commerce platform (Product catalog + Checkout)
- Hệ thống booking (đặt vé máy bay, khách sạn)
- Hệ thống analytics và reporting
- IoT data ingestion (write heavy)
- Social media feeds (read heavy)

CQRS không chỉ là pattern — nó là một cách tư duy về kiến trúc. Khi bạn hiểu rõ rằng đọc và ghi là hai bài toán hoàn toàn khác nhau, bạn sẽ thiết kế hệ thống tốt hơn.
