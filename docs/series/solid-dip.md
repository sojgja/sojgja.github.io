---
id: solid-dip
title: D — Dependency Inversion Principle
sidebar_label: D — Dependency Inversion
sidebar_position: 30
---

# D — Dependency Inversion Principle

> *"A. High-level modules should not depend on low-level modules. Both should depend on abstractions. B. Abstractions should not depend on details. Details should depend on abstractions."* — **Robert C. Martin, *Agile Software Development, Principles, Patterns, and Practices*, 2002**

Dependency Inversion Principle (DIP) là nguyên lý cuối cùng và sâu sắc nhất trong SOLID — cũng là nguyên lý có ảnh hưởng lớn nhất đến kiến trúc phần mềm hiện đại. Robert C. Martin đã tổng hợp DIP từ những ý tưởng của Ivar Jacobson (Object-Oriented Software Engineering) và Bertrand Meyer, nhưng ông đã đưa nó lên một tầm cao mới khi kết hợp với Dependency Injection. DIP thường bị nhầm lẫn với Dependency Injection (DI) — nhưng thực ra DIP là *nguyên lý* (cái gì), còn DI là *kỹ thuật* (làm thế nào). DIP khác với Inversion of Control (IoC): IoC là nguyên lý tổng quát "framework gọi code của bạn", DIP là một trường hợp cụ thể về cách tổ chức dependency.

## Bài toán chi tiết: Hệ thống xử lý giao dịch tài chính

Công ty A phát triển module xử lý giao dịch tài chính. Module cấp cao `TransactionProcessor` xử lý các giao dịch — validate, kiểm tra số dư, thực hiện chuyển tiền, ghi log. Nó phụ thuộc trực tiếp vào module cấp thấp: `PostgresDatabase` để lưu giao dịch, `Log4jLogger` để ghi log, `SMTPEmailService` để gửi xác nhận, `InternalFraudCheckAPI` để kiểm tra gian lận. Mỗi dependency đều là concrete class — được khởi tạo trực tiếp trong constructor.

```python
class TransactionProcessor:
    def __init__(self):
        self.db = PostgresDatabase("localhost", 5432, "banking")      # Concrete
        self.logger = Log4jLogger("transaction.log")                  # Concrete
        self.email = SMTPEmailService("smtp.bank.com", 587)           # Concrete
        self.fraud = InternalFraudCheckAPI("https://fraud.internal")  # Concrete
```

Mọi thứ hoạt động tốt trong 2 năm đầu. Sau đó:
1. **Chuyển database**: Ngân hàng quyết định chuyển từ PostgreSQL sang Oracle vì lý do licensing. `TransactionProcessor` phải được sửa để dùng `OracleDatabase` thay vì `PostgresDatabase`.
2. **Đổi logger**: Team bảo mật yêu cầu log phải được gửi đến centralized logging system (ELK stack) thay vì file local. Lại sửa `TransactionProcessor`.
3. **Thêm fraud detection**: Thay vì gọi internal API, họ muốn tích hợp service bên thứ ba (Riskified). Lại sửa.
4. **Unit testing**: Không thể test `TransactionProcessor` nếu không có PostgreSQL và internal fraud API thật. Mỗi lần chạy test mất 10 phút để setup infrastructure.
5. **A/B testing**: Muốn thử nghiệm hai email service khác nhau (SendGrid vs SES) để so sánh delivery rate. Không thể vì email service bị hardcode.

Sau 3 năm, `TransactionProcessor` là class được sửa nhiều nhất trong codebase. Mỗi lần infrastructure thay đổi, class này phải thay đổi. Đây là dấu hiệu kinh điển của vi phạm DIP: **module cấp cao (business logic) phụ thuộc vào module cấp thấp (infrastructure)**, dẫn đến business logic bị "nhiễm" các chi tiết kỹ thuật. Hệ quả là chi phí bảo trì tăng vọt, code không thể test, và mỗi thay đổi infrastructure đều rủi ro cao. Cuối cùng, team phải dành 2 tháng để tái cấu trúc toàn bộ module, áp dụng DIP và Clean Architecture.

## Phân tích vấn đề

Root cause: **module cấp cao (business logic) phụ thuộc trực tiếp vào module cấp thấp (infrastructure)**. "Cấp cao" và "cấp thấp" ở đây không phải về phân cấp trong code, mà về mức độ trừu tượng và lý do thay đổi. Business logic thay đổi vì lý do nghiệp vụ; infrastructure thay đổi vì lý do kỹ thuật. Khi chúng phụ thuộc trực tiếp, một thay đổi kỹ thuật có thể kéo theo thay đổi nghiệp vụ — và ngược lại.

Cụ thể hơn, vi phạm DIP dẫn đến:

1. **Tight coupling**: Business logic và infrastructure không thể phát triển độc lập. Thay đổi database driver (Postgres → Oracle) ảnh hưởng đến code xử lý giao dịch.
2. **Khó test**: Không thể test business logic mà không có infrastructure thật. Phải chạy PostgreSQL, email server, fraud API — chậm, không ổn định, không parallelizable.
3. **Không thể thay thế**: Không thể switch từ SMTP sang SendGrid, từ PostgreSQL sang MongoDB, từ file log sang ELK mà không sửa business code.
4. **Vi phạm SRP (hệ quả)**: Class vừa chứa business logic vừa chứa infrastructure orchestration.
5. **Khó mở rộng**: Thêm tính năng mới (ví dụ: gửi SMS thay vì email) buộc phải sửa business logic.

**Code smells**: Constructor khởi tạo concrete class (dùng `new`/constructor của class khác trực tiếp); Hard-coded connection strings, API keys, file paths; Class có tên chứa implementation detail (`PostgresTransactionProcessor`, `SMTPEmailService`); Không thể test business logic mà không có database thật; Dependency injection thủ công (manual wiring) lan tràn.

## Giải pháp: Dependency Inversion + Dependency Injection

Giải pháp có hai bước:

**Bước 1 — Đảo ngược dependency**: Định nghĩa abstraction (interface/abstract class) cho mỗi module cấp thấp. `TransactionProcessor` (cấp cao) phụ thuộc vào `Database` (abstraction), không phải `PostgresDatabase` (concrete). `PostgresDatabase` (cấp thấp) implement `Database` (abstraction). Như vậy cả cấp cao và cấp thấp đều phụ thuộc vào abstraction.

**Bước 2 — Dependency Injection**: Abstraction được "inject" từ bên ngoài vào module cấp cao, thay vì module cấp cao tự tạo. Injection có thể qua constructor (phổ biến nhất), setter, hoặc method parameter.

Kết quả: `TransactionProcessor` không còn khởi tạo database nữa — nó nhận `Database` qua constructor. Khi cần đổi database, chỉ cần inject implementation khác. Khi test, inject mock implementation. Business logic hoàn toàn độc lập với infrastructure.

## Ví dụ code hoàn chỉnh

### VIOLATION — Vi phạm DIP

```python
# transaction_violation.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
import json
import logging


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    from_account: str
    to_account: str
    amount: Decimal
    currency: str
    timestamp: datetime = datetime.now()


class Log4jLogger:
    """Concrete logger — ghi ra file."""

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        logging.basicConfig(filename=file_path, level=logging.INFO)

    def info(self, message: str) -> None:
        logging.info(message)

    def error(self, message: str) -> None:
        logging.error(message)

    def warn(self, message: str) -> None:
        logging.warning(message)


class PostgresDatabase:
    """Concrete database connection — PostgreSQL-specific."""

    def __init__(self, host: str, port: int, db_name: str) -> None:
        self._host = host
        self._port = port
        self._db_name = db_name
        self._connection: Any = None

    def connect(self) -> None:
        import psycopg2  # type: ignore
        self._connection = psycopg2.connect(
            host=self._host, port=self._port, dbname=self._db_name
        )
        print(f"Connected to PostgreSQL at {self._host}:{self._port}/{self._db_name}")

    def insert_transaction(self, txn: Transaction) -> str:
        if self._connection is None:
            self.connect()
        cur = self._connection.cursor()
        cur.execute(
            "INSERT INTO transactions (id, from_acct, to_acct, amount, currency) VALUES (%s, %s, %s, %s, %s)",
            (txn.transaction_id, txn.from_account, txn.to_account,
             str(txn.amount), txn.currency),
        )
        self._connection.commit()
        cur.close()
        return txn.transaction_id

    def get_balance(self, account_id: str) -> Decimal:
        if self._connection is None:
            self.connect()
        cur = self._connection.cursor()
        cur.execute("SELECT balance FROM accounts WHERE id = %s", (account_id,))
        row = cur.fetchone()
        cur.close()
        return Decimal(str(row[0])) if row else Decimal('0')


class SMTPEmailService:
    """Concrete email service — SMTP-specific."""

    def __init__(self, smtp_host: str, smtp_port: int) -> None:
        self._host = smtp_host
        self._port = smtp_port

    def send_confirmation(self, to: str, txn: Transaction) -> None:
        import smtplib
        from email.mime.text import MIMEText  # type: ignore
        msg = MIMEText(
            f"Giao dịch #{txn.transaction_id}: {txn.amount} {txn.currency} "
            f"từ {txn.from_account} đến {txn.to_account}"
        )
        msg['Subject'] = f'Xác nhận giao dịch #{txn.transaction_id}'
        msg['To'] = to
        with smtplib.SMTP(self._host, self._port) as server:
            server.send_message(msg)


class InternalFraudCheckAPI:
    """Concrete fraud check — internal API."""

    def __init__(self, api_url: str) -> None:
        self._api_url = api_url

    def check(self, txn: Transaction) -> dict[str, Any]:
        import urllib.request
        import json
        data = json.dumps({
            'amount': str(txn.amount),
            'from': txn.from_account,
            'to': txn.to_account,
        }).encode()
        req = urllib.request.Request(self._api_url, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())


class TransactionProcessor:
    """
    VIOLATION DIP: Module cấp cao phụ thuộc trực tiếp vào module cấp thấp.
    Mỗi lần thay đổi infrastructure phải sửa class này.
    """

    def __init__(self) -> None:
        # ❌ Phụ thuộc trực tiếp vào concrete classes
        self._logger = Log4jLogger("transactions.log")
        self._db = PostgresDatabase("localhost", 5432, "banking")
        self._email = SMTPEmailService("smtp.bank.com", 587)
        self._fraud = InternalFraudCheckAPI("https://fraud.internal/api/check")

        self._db.connect()

    def process(self, txn: Transaction, user_email: str) -> dict[str, object]:
        self._logger.info(f"Processing transaction {txn.transaction_id}")

        # Check fraud
        fraud_result = self._fraud.check(txn)
        if fraud_result.get('risk_score', 0) > 0.8:
            self._logger.warn(f"Fraud detected for {txn.transaction_id}")
            return {'status': 'rejected', 'reason': 'Fraud detected'}

        # Check balance
        balance = self._db.get_balance(txn.from_account)
        if balance < txn.amount:
            self._logger.error(f"Insufficient balance for {txn.transaction_id}")
            return {'status': 'rejected', 'reason': 'Insufficient balance'}

        # Execute
        self._db.insert_transaction(txn)
        self._logger.info(f"Transaction {txn.transaction_id} completed")

        # Notify
        self._email.send_confirmation(user_email, txn)

        return {'status': 'completed', 'transaction_id': txn.transaction_id}
```

### REFACTORED — Tuân thủ DIP

```python
# ─── domain/transaction.py ───
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    from_account: str
    to_account: str
    amount: Decimal
    currency: str
    timestamp: datetime = datetime.now()


# ─── interfaces/txn_interfaces.py ───
from __future__ import annotations
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol


class Logger(Protocol):
    """Abstraction cho logging."""

    def info(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...

    def warn(self, message: str) -> None: ...


class TransactionRepository(Protocol):
    """Abstraction cho lưu trữ giao dịch."""

    def insert(self, txn: Transaction) -> str: ...

    def get_balance(self, account_id: str) -> Decimal: ...


class Notifier(Protocol):
    """Abstraction cho gửi thông báo."""

    def send_confirmation(self, recipient: str, txn: Transaction) -> None: ...


class FraudDetector(Protocol):
    """Abstraction cho kiểm tra gian lận."""

    def check(self, txn: Transaction) -> FraudResult: ...


@dataclass(frozen=True)
class FraudResult:
    risk_score: float  # 0.0 - 1.0
    is_fraud: bool = False
    details: str = ''


# ─── infrastructure/postgres_repository.py ───
from __future__ import annotations
from decimal import Decimal
from typing import Any


class PostgresTransactionRepository:
    """Concrete implementation — có thể thay bằng MongoTransactionRepository."""

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._connection: Any = None

    def _ensure_connected(self) -> None:
        if self._connection is None:
            import psycopg2  # type: ignore
            self._connection = psycopg2.connect(self._connection_string)

    def insert(self, txn: Transaction) -> str:
        self._ensure_connected()
        cur = self._connection.cursor()
        cur.execute(
            "INSERT INTO transactions (id, from_acct, to_acct, amount, currency) "
            "VALUES (%s, %s, %s, %s, %s)",
            (txn.transaction_id, txn.from_account, txn.to_account,
             str(txn.amount), txn.currency),
        )
        self._connection.commit()
        cur.close()
        return txn.transaction_id

    def get_balance(self, account_id: str) -> Decimal:
        self._ensure_connected()
        cur = self._connection.cursor()
        cur.execute("SELECT balance FROM accounts WHERE id = %s", (account_id,))
        row = cur.fetchone()
        cur.close()
        return Decimal(str(row[0])) if row else Decimal('0')


# ─── infrastructure/file_logger.py ───
from __future__ import annotations
import logging


class FileLogger:
    """Có thể thay bằng CloudLogger, ELKLogger, ConsoleLogger, ..."""

    def __init__(self, file_path: str, level: int = logging.INFO) -> None:
        logging.basicConfig(filename=file_path, level=level)
        self._logger = logging.getLogger(__name__)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def warn(self, message: str) -> None:
        self._logger.warning(message)


# ─── infrastructure/smtp_notifier.py ───
from __future__ import annotations


class SmtpNotifier:
    """Có thể thay bằng SendGridNotifier, SESNotifier, ..."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port

    def send_confirmation(self, recipient: str, txn: Transaction) -> None:
        import smtplib
        from email.mime.text import MIMEText  # type: ignore
        body = (f"Giao dịch #{txn.transaction_id}: {txn.amount} {txn.currency} "
                f"từ {txn.from_account} đến {txn.to_account}")
        msg = MIMEText(body)
        msg['Subject'] = f'Xác nhận giao dịch #{txn.transaction_id}'
        msg['To'] = recipient
        with smtplib.SMTP(self._host, self._port) as server:
            server.send_message(msg)


# ─── infrastructure/fraud_check_service.py ───
from __future__ import annotations
from typing import Any
import json


class ThirdPartyFraudDetector:
    """Có thể thay bằng InternalFraudAPI, RuleBasedFraudDetector, ..."""

    def __init__(self, api_key: str, endpoint: str = "https://fraud.riskified.com/check") -> None:
        self._api_key = api_key
        self._endpoint = endpoint

    def check(self, txn: Transaction) -> FraudResult:
        import urllib.request
        payload = json.dumps({
            'amount': str(txn.amount),
            'from': txn.from_account,
            'to': txn.to_account,
            'currency': txn.currency,
        }).encode()
        req = urllib.request.Request(
            self._endpoint, data=payload,
            headers={'Content-Type': 'application/json', 'X-API-Key': self._api_key},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            return FraudResult(
                risk_score=data.get('risk_score', 0.0),
                is_fraud=data.get('is_fraud', False),
                details=data.get('reason', ''),
            )
        except Exception as e:
            return FraudResult(risk_score=0.0, details=f"API error: {e}")


# ─── application/transaction_processor.py ───
from __future__ import annotations


class TransactionProcessor:
    """
    Tuân thủ DIP: phụ thuộc vào abstraction (Protocol), không phụ thuộc vào concrete.
    Cả lớp này và các infrastructure classes đều phụ thuộc vào abstraction.
    """

    def __init__(
        self,
        repository: TransactionRepository,
        logger: Logger,
        notifier: Notifier,
        fraud_detector: FraudDetector,
    ) -> None:
        self._repo = repository
        self._logger = logger
        self._notifier = notifier
        self._fraud = fraud_detector

    def process(self, txn: Transaction, user_email: str) -> dict[str, object]:
        self._logger.info(f"Processing transaction {txn.transaction_id}")

        # 1. Fraud check
        fraud_result = self._fraud.check(txn)
        if fraud_result.risk_score > 0.8:
            self._logger.warn(f"Fraud detected: {txn.transaction_id} "
                              f"(risk={fraud_result.risk_score}, {fraud_result.details})")
            return {'status': 'rejected', 'reason': 'Fraud detected'}

        # 2. Balance check
        balance = self._repo.get_balance(txn.from_account)
        if balance < txn.amount:
            self._logger.error(f"Insufficient balance: {txn.from_account} "
                               f"has {balance}, needs {txn.amount}")
            return {'status': 'rejected', 'reason': 'Insufficient balance'}

        # 3. Execute
        self._repo.insert(txn)
        self._logger.info(f"Transaction {txn.transaction_id} completed successfully")

        # 4. Notify
        self._notifier.send_confirmation(user_email, txn)

        return {
            'status': 'completed',
            'transaction_id': txn.transaction_id,
            'amount': str(txn.amount),
            'currency': txn.currency,
        }


# ─── application/wiring.py ───
from __future__ import annotations

# Dependency Injection Wiring — có thể dùng DI container cho production

def create_production_processor() -> TransactionProcessor:
    return TransactionProcessor(
        repository=PostgresTransactionRepository(
            "postgresql://user:pass@localhost:5432/banking"
        ),
        logger=FileLogger("/var/log/banking/transactions.log"),
        notifier=SmtpNotifier("smtp.bank.com", 587),
        fraud_detector=ThirdPartyFraudDetector(api_key="sk_live_xxxxx"),
    )


def create_local_processor() -> TransactionProcessor:
    """Dùng cho development — dễ dàng thay đổi implementation."""
    return TransactionProcessor(
        repository=PostgresTransactionRepository(
            "postgresql://dev:dev@localhost:5432/banking_dev"
        ),
        logger=FileLogger("transactions.log"),
        notifier=SmtpNotifier("smtp.mailtrap.io", 2525),
        fraud_detector=ThirdPartyFraudDetector(
            api_key="sk_test_yyyyy",
            endpoint="https://sandbox.fraud.test/check",
        ),
    )


# ─── main.py ───
from __future__ import annotations

processor = create_production_processor()

txn = Transaction(
    transaction_id="TXN-001",
    from_account="ACC-1001",
    to_account="ACC-2001",
    amount=Decimal('5000000'),
    currency="VND",
)

result = processor.process(txn, "user@example.com")
print(f"Result: {result}")
```

## Dấu hiệu nhận biết vi phạm DIP

- **Constructor khởi tạo concrete class**: `self.db = PostgresDatabase(...)` thay vì nhận abstraction từ bên ngoài.
- **Dùng Singleton/Static method để lấy dependency**: `Database.getInstance()`, `LoggerFactory.getLogger()` — dấu hiệu Service Locator pattern (thường vi phạm DIP).
- **Hard-coded connection string, API key, file path trong business class**: Business logic không nên biết infrastructure details.
- **Class import và sử dụng thư viện infrastructure trực tiếp**: Business logic import `psycopg2`, `boto3`, `redis` — dấu hiệu vi phạm.
- **Không thể test business logic nếu không có infrastructure thật**: Nếu bạn phải chạy Docker container để chạy unit test, business logic của bạn đang vi phạm DIP.
- **Base class chứa implementation details**: Abstract class có method concrete, field concrete — làm rò rỉ chi tiết.
- **High coupling giữa business và infrastructure layers**: Một thay đổi trong database schema kéo theo thay đổi trong business logic.

## Kiểm thử

```python
# test_transaction_processor.py
from __future__ import annotations
from decimal import Decimal
from unittest.mock import Mock, MagicMock, call, ANY
import pytest  # type: ignore
from domain.transaction import Transaction
from interfaces.txn_interfaces import TransactionRepository, Logger, Notifier, FraudDetector, FraudResult
from application.transaction_processor import TransactionProcessor


@pytest.fixture
def sample_txn() -> Transaction:
    return Transaction(
        transaction_id="TXN-TEST-001",
        from_account="ACC-TEST-101",
        to_account="ACC-TEST-202",
        amount=Decimal('1000000'),
        currency="VND",
    )


class TestTransactionProcessor:
    """Sử dụng mock — không cần DB, email, fraud API thật."""

    @pytest.fixture
    def mock_repo(self) -> Mock:
        repo = Mock(spec=TransactionRepository)
        repo.get_balance.return_value = Decimal('5000000')
        repo.insert.return_value = "TXN-TEST-001"
        return repo

    @pytest.fixture
    def mock_logger(self) -> Mock:
        return Mock(spec=Logger)

    @pytest.fixture
    def mock_notifier(self) -> Mock:
        return Mock(spec=Notifier)

    @pytest.fixture
    def mock_fraud_safe(self) -> Mock:
        fraud = Mock(spec=FraudDetector)
        fraud.check.return_value = FraudResult(risk_score=0.1, is_fraud=False)
        return fraud

    @pytest.fixture
    def mock_fraud_risky(self) -> Mock:
        fraud = Mock(spec=FraudDetector)
        fraud.check.return_value = FraudResult(risk_score=0.95, is_fraud=True, details="Suspicious pattern")
        return fraud

    def test_successful_transaction(
        self, sample_txn: Transaction,
        mock_repo: Mock, mock_logger: Mock,
        mock_notifier: Mock, mock_fraud_safe: Mock,
    ) -> None:
        processor = TransactionProcessor(mock_repo, mock_logger, mock_notifier, mock_fraud_safe)
        result = processor.process(sample_txn, "user@test.com")

        assert result['status'] == 'completed'
        assert result['transaction_id'] == 'TXN-TEST-001'

        # Verify interaction với từng dependency
        mock_fraud_safe.check.assert_called_once_with(sample_txn)
        mock_repo.get_balance.assert_called_once_with('ACC-TEST-101')
        mock_repo.insert.assert_called_once_with(sample_txn)
        mock_logger.info.assert_called()
        mock_notifier.send_confirmation.assert_called_once_with('user@test.com', sample_txn)

    def test_fraud_detected(
        self, sample_txn: Transaction,
        mock_repo: Mock, mock_logger: Mock,
        mock_notifier: Mock, mock_fraud_risky: Mock,
    ) -> None:
        processor = TransactionProcessor(mock_repo, mock_logger, mock_notifier, mock_fraud_risky)
        result = processor.process(sample_txn, "user@test.com")

        assert result['status'] == 'rejected'
        assert result['reason'] == 'Fraud detected'

        # Nếu fraud detected, không insert, không gửi email
        mock_repo.insert.assert_not_called()
        mock_notifier.send_confirmation.assert_not_called()
        mock_logger.warn.assert_called_once()

    def test_insufficient_balance(
        self, sample_txn: Transaction,
        mock_repo: Mock, mock_logger: Mock,
        mock_notifier: Mock, mock_fraud_safe: Mock,
    ) -> None:
        mock_repo.get_balance.return_value = Decimal('500000')  # < amount 1,000,000
        processor = TransactionProcessor(mock_repo, mock_logger, mock_notifier, mock_fraud_safe)
        result = processor.process(sample_txn, "user@test.com")

        assert result['status'] == 'rejected'
        assert result['reason'] == 'Insufficient balance'
        mock_repo.insert.assert_not_called()
        mock_notifier.send_confirmation.assert_not_called()
        mock_logger.error.assert_called_once()

    def test_fraud_api_failure_graceful(
        self, sample_txn: Transaction,
        mock_repo: Mock, mock_logger: Mock,
        mock_notifier: Mock,
    ) -> None:
        """Khi fraud API fail, transaction vẫn được xử lý (dùng risk_score mặc định 0)."""
        mock_fraud_fail = Mock(spec=FraudDetector)
        mock_fraud_fail.check.return_value = FraudResult(risk_score=0.0, details="API unavailable")

        processor = TransactionProcessor(mock_repo, mock_logger, mock_notifier, mock_fraud_fail)
        result = processor.process(sample_txn, "user@test.com")

        assert result['status'] == 'completed'

    def test_dependency_injection_enables_easy_testing(
        self, sample_txn: Transaction,
    ) -> None:
        """
        DIP cho phép inject các implementation khác nhau.
        Trong test, inject mock — dễ dàng, nhanh chóng.
        Trong production, inject real implementation.
        """
        # Test với implementation giả lập inline
        class FakeRepo:
            def insert(self, txn: Transaction) -> str:
                return txn.transaction_id

            def get_balance(self, account_id: str) -> Decimal:
                return Decimal('9999999')

        class FakeLogger:
            def info(self, msg: str) -> None: print(f"[INFO] {msg}")
            def error(self, msg: str) -> None: print(f"[ERROR] {msg}")
            def warn(self, msg: str) -> None: print(f"[WARN] {msg}")

        class FakeNotifier:
            def send_confirmation(self, recipient: str, txn: Transaction) -> None:
                print(f"Would send email to {recipient} about {txn.transaction_id}")

        class FakeFraud:
            def check(self, txn: Transaction) -> FraudResult:
                return FraudResult(risk_score=0.0)

        processor = TransactionProcessor(FakeRepo(), FakeLogger(), FakeNotifier(), FakeFraud())
        result = processor.process(sample_txn, "user@test.com")
        assert result['status'] == 'completed'


class TestDIPFlexibility:
    """Kiểm tra khả năng thay đổi implementation nhờ DIP."""

    def test_different_implementations_same_behavior(self, sample_txn: Transaction) -> None:
        """Với DIP, thay đổi implementation không ảnh hưởng đến business logic."""

        def test_with_repo(repo) -> None:
            logger = Mock(spec=Logger)
            notifier = Mock(spec=Notifier)
            fraud = Mock(spec=FraudDetector)
            fraud.check.return_value = FraudResult(risk_score=0.0)

            processor = TransactionProcessor(repo, logger, notifier, fraud)
            result = processor.process(sample_txn, "user@test.com")
            assert result['status'] == 'completed'

        # Test với PostgresTransactionRepository (mocked)
        with patch('infrastructure.postgres_repository.psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            mock_cursor.fetchone.return_value = (Decimal('5000000'),)

            pg_repo = PostgresTransactionRepository("postgresql://test:test@localhost/test")
            test_with_repo(pg_repo)

        # Test với InMemoryRepository (không cần database)
        class InMemoryRepo:
            def __init__(self) -> None:
                self._balances: dict[str, Decimal] = {'ACC-TEST-101': Decimal('5000000')}

            def insert(self, txn: Transaction) -> str:
                return txn.transaction_id

            def get_balance(self, account_id: str) -> Decimal:
                return self._balances.get(account_id, Decimal('0'))

        test_with_repo(InMemoryRepo())
```

## Ứng dụng thực tế

1. **Clean Architecture của Robert C. Martin**: DIP là trụ cột của Clean Architecture. Dependency rule: các layer ngoài cùng (infrastructure, UI) phụ thuộc vào layer trong (business logic), business logic không biết gì về bên ngoài. Use cases (application layer) định nghĩa interfaces (output ports), infrastructure implement chúng (output adapters). Đây là DIP ở quy mô kiến trúc.

2. **FastAPI + SQLAlchemy + Repository Pattern**: FastAPI route handler gọi `UserService`, `UserService` nhận `UserRepository` (abstraction) qua constructor. `UserRepository` được implement bởi `SQLAlchemyUserRepository` (dùng ORM) hoặc `InMemoryUserRepository` (cho test). Khi chuyển từ SQLAlchemy sang async SQLAlchemy, chỉ cần implement lại repository, không cần sửa service.

3. **Django — Dao/Repository Pattern (trái với Django philosophy)**: Django khuyến khích dùng active record pattern (`User.objects.create()`), nhưng các dự án lớn thường thêm repository layer để áp dụng DIP. `UserRepository` (abstraction) được inject vào service. Có thể dùng `DjangoUserRepository` cho production, `MockUserRepository` cho test.

4. **Payment Gateway Integration**: Hệ thống thanh toán có `PaymentGateway` interface. `VNPayGateway`, `MoMoGateway`, `StripeGateway`, `PayPalGateway` implement interface này. `PaymentService` không biết gateway cụ thể nào đang dùng — nó chỉ gọi `gateway.process(amount)`. Gateway implementation được inject từ config hoặc DI container.

5. **Testability là lợi ích số một**: Một công ty FinTech tại Việt Nam áp dụng DIP cho toàn bộ hệ thống core banking dẫn đến giảm thời gian chạy regression test từ 6 giờ xuống còn 12 phút, nhờ khả năng mock tất cả infrastructure dependencies. Velocity tăng 3x vì developer không cần chờ infrastructure layer.

## Liên hệ với Pattern

- **Dependency Injection (DI)**: Kỹ thuật implement DIP. Dependency được inject từ bên ngoài (qua constructor, setter, method parameter) thay vì tự tạo bên trong.
- **Service Locator**: Anti-pattern so với DIP. Service Locator là một global registry mà class gọi để lấy dependency — class vẫn biết về Service Locator (vẫn phụ thuộc), và dependency không rõ ràng từ constructor. DIP + Constructor Injection ưu việt hơn vì dependency được khai báo tường minh.
- **Abstract Factory**: Dùng để tạo dependency mà không vi phạm DIP. Factory trả về abstraction, client không cần biết concrete class nào được tạo.
- **Proxy / Decorator**: Wrapper quanh interface — cho phép thêm behavior (caching, logging, retry) mà không ảnh hưởng đến business logic. Business logic phụ thuộc vào interface, proxy implement interface đó.
- **Strategy Pattern**: Cho phép thay đổi algorithm bằng cách inject Strategy khác — business logic không biết strategy cụ thể.
- **Plugin Architecture**: DIP ở cấp độ cao nhất. Ứng dụng định nghĩa plugin interface, các plugin implement và được load động. Apply cho mọi loại extension.

## Ưu và nhược điểm

| Tiêu chí | Trước (vi phạm DIP) | Sau (tuân thủ DIP) |
|----------|---------------------|-------------------|
| **Phụ thuộc** | Business → Infrastructure (concrete) | Business → Abstraction ← Infrastructure |
| **Khả năng test** | Kém — phải có infrastructure thật | Tuyệt vời — mock dễ dàng |
| **Thời gian chạy test** | Hàng chục phút | Vài giây |
| **Độc lập giữa layers** | Không — thay đổi infra ảnh hưởng business | Có — thay đổi infra không ảnh hưởng business |
| **Khả năng thay thế** | Khó — phải sửa business code | Dễ — inject implementation khác |
| **Complexity** | Ít file hơn (không interface) | Nhiều file hơn (interface + implementation) |
| **Setup ban đầu** | Nhanh — khởi tạo trực tiếp | Chậm hơn — phải thiết kế abstraction |
| **Runtime flexibility** | Thấp — gắn cứng | Cao — có thể swap implementation runtime |
| **Dependency graph rõ ràng** | Không — dependency ẩn trong constructor | Có — dependency khai báo qua type hints |
| **Phù hợp cho** | Prototype, script nhỏ | Production, hệ thống lớn > 5-10 services |
| **Dependency Injection Container cần?** | Không | Có thể — giúp quản lý wiring khi scale |

## Kết luận

DIP là nguyên lý quan trọng nhất trong SOLID vì nó thay đổi hoàn toàn cách bạn nghĩ về kiến trúc phần mềm. Thay vì để business logic phụ thuộc vào infrastructure, bạn "đảo ngược" dependency: cả hai đều phụ thuộc vào abstraction do business logic định nghĩa. Điều này làm cho business logic trở thành "core" của hệ thống — độc lập, có thể test, có thể phát triển song song với infrastructure. Lời khuyên thực tế:

1. **Bắt đầu từ test**: Viết test cho business logic trước. Test sẽ cho bạn biết abstraction nào cần thiết.
2. **Interface thuộc về consumer**: Interface `TransactionRepository` thuộc về `TransactionProcessor` (business logic), không thuộc về `PostgresTransactionRepository` (infrastructure). Đặt interface cùng package với consumer.
3. **Đừng DIP hóa mọi thứ**: Nếu một dependency hiếm khi thay đổi (ví dụ: `datetime.now()`), không cần abstract hóa nó. DIP có cost — code phức tạp hơn, nhiều file hơn.
4. **Dùng Protocol/ABC**: Python typing.Protocol (structural subtyping) linh hoạt hơn ABC (nominal subtyping). Protocol cho phép "duck typing" ở level type checking.
5. **Wiring ở entry point**: Việc kết nối concrete classes với nhau (wiring) chỉ nên xảy ra ở entry point của ứng dụng (main, module config, DI container). Business logic không nên tự wiring.

DIP kết hợp với DI tạo nền tảng cho những kiến trúc mạnh mẽ như Hexagonal Architecture (Ports and Adapters), Clean Architecture, và Onion Architecture. Khi bạn nắm vững DIP, bạn không chỉ viết code tốt hơn — bạn đang thiết kế hệ thống có thể tồn tại và phát triển qua nhiều năm thay đổi về công nghệ và yêu cầu kinh doanh.
