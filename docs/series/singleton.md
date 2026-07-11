---
id: singleton
title: Singleton
sidebar_label: 🥇 Singleton
sidebar_position: 2
---

# Singleton

> *"Ensure a class only has one instance, and provide a global point of access to it."* — Gang of Four, *Design Patterns: Elements of Reusable Object-Oriented Software*, 1994.

**Singleton** thuộc nhóm **Creational Patterns**, giải quyết vấn đề giới hạn số lượng instance của một class xuống **chính xác một** và cung cấp một điểm truy cập toàn cục duy nhất đến instance đó. Đây là pattern gây tranh cãi nhất trong GoF — vừa cực kỳ hữu ích trong một số tình huống, vừa bị chỉ trích nặng nề vì vi phạm **Single Responsibility Principle** và tạo ra **global state** khó kiểm soát.

## Bài toán chi tiết

Hãy tưởng tượng bạn đang xây dựng một nền tảng **xử lý giao dịch tài chính thời gian thực** cho một công ty fintech. Hệ thống bao gồm hàng trăm microservice: dịch vụ xác thực người dùng, dịch vụ quản lý tài khoản, dịch vụ ghi nhận giao dịch, dịch vụ báo cáo, và dịch vụ cảnh báo gian lận. Tất cả các dịch vụ này đều cần ghi log vào cùng một hệ thống tập trung — vừa để audit, vừa để debug, vừa để phục vụ compliance.

Mỗi dịch vụ, mỗi class, mỗi module đều có thể tạo ra một **Logger instance** riêng. Nếu không có cơ chế kiểm soát, hệ thống có thể đồng thời tồn tại hàng nghìn đối tượng Logger, mỗi đối tượng mở một kết nối riêng đến file log, đến hệ thống gửi log tập trung (như ELK, Splunk, hoặc Graylog), và chiếm giữ tài nguyên hệ thống một cách lãng phí.

Vấn đề trở nên nghiêm trọng hơn khi bạn cần đảm bảo **tính nhất quán của dữ liệu log**. Nếu mỗi Logger có buffer riêng, thứ tự các log entry có thể bị xáo trộn, hoặc tệ hơn — log từ cùng một luồng xử lý lại bị chia cắt thành nhiều file khác nhau. Khi cơ quan thuế yêu cầu audit trail đầy đủ cho một giao dịch cụ thể, bạn không thể trả lời: "log nằm rải rác ở 15 file khác nhau."

Thử nghiệm với cách tiếp cận ngây thơ: truyền Logger instance qua constructor hoặc qua tham số function. Cách này buộc mọi class trong toàn bộ codebase phải nhận Logger làm dependency — dẫn đến **telescoping constructor**: `def __init__(self, db, cache, logger, config, ...)`. Hàng trăm class phải sửa đổi chỉ vì một class muốn ghi log. Đây là vi phạm **Law of Demeter** và tạo ra sự phụ thuộc cực kỳ khó chịu. Sử dụng biến toàn cục `global_logger` thì dễ nhưng phá hủy khả năng kiểm thử và không kiểm soát được khởi tạo.

## Giải pháp với Pattern

Singleton giải quyết triệt để bài toán này bằng cách đảm bảo class `Logger` chỉ có **một instance duy nhất** trong toàn bộ vòng đời ứng dụng. Instance này được khởi tạo **lazy** — chỉ khi có yêu cầu lần đầu tiên — và được lưu trữ ở một biến class (class-level variable). Mọi lời gọi `Logger()` sau đó đều trả về chính instance đã tồn tại.

Cơ chế hoạt động:

1. **Private constructor** (mô phỏng qua `__new__`): Ngăn không cho bên ngoài dùng `Logger()` tạo instance mới một cách tùy tiện.
2. **Static accessor** (phương thức class `get_instance` hoặc override `__new__`): Cung cấp điểm truy cập duy nhất.
3. **Lazy initialization**: Instance chỉ được tạo khi thực sự cần, tiết kiệm tài nguyên khởi động.
4. **Thread safety**: Đảm bảo trong môi trường đa luồng, không có hai thread nào cùng tạo instance.

Với Singleton, module A ghi log "User login", module B ghi log "Transaction processed" — cả hai đều đi qua cùng một instance, cùng một buffer, cùng một kết nối đến hệ thống log tập trung. Thứ tự log được bảo toàn, tài nguyên được tiết kiệm, và dependency injection không còn là ác mộng.

## Phân tích thiết kế

Singleton là một pattern mang nhiều tranh cãi. Về mặt OOP, nó vi phạm **Single Responsibility Principle** vì class Singleton vừa quản lý business logic riêng, vừa quản lý vòng đời instance của chính nó. Nó cũng tạo ra **global state** — thứ bị coi là anti-pattern trong thiết kế phần mềm hiện đại vì khó kiểm thử, khó debug, và khó mở rộng.

**Trade-offs cần cân nhắc:**

- **Ưu điểm về hiệu năng**: Khởi tạo một lần, dùng mãi mãi. Tiết kiệm bộ nhớ và thời gian khởi tạo lặp lại.
- **Nhược điểm về kiểm thử**: Unit test không thể dễ dàng mock Singleton instance. Cần thêm cơ chế reset instance giữa các test case.
- **Đa luồng**: Cần xử lý double-checked locking hoặc sử dụng cơ chế thread-safe của Python (như `threading.Lock`).
- **Tight coupling**: Code gọi Singleton trực tiếp tạo sự phụ thuộc cứng vào class cụ thể, khó thay thế implementation.

**Khi nào KHÔNG nên dùng Singleton:**

- Khi bạn cần nhiều instance khác nhau trong các context khác nhau (ví dụ: một logger cho mỗi user session).
- Khi bạn muốn unit test nghiêm ngặt — global state là kẻ thù của kiểm thử.
- Khi class có thể phát triển thành nhiều biến thể (cần subclassing).
- Khi ứng dụng của bạn là serverless (AWS Lambda, Google Cloud Functions) — mỗi invocation có thể chạy trên container khác nhau.
- Khi bạn có thể dùng dependency injection container để quản lý vòng đời instance thay vì tự quản lý.

Trong thực tế hiện đại, nhiều framework khuyến khích dùng **DI container** (như Spring, Google Guice) hoặc **Monostate pattern** (các object khác nhau nhưng chia sẻ cùng trạng thái) thay vì Singleton thuần túy.

## Ví dụ code hoàn chỉnh

### Cách làm sai (Naive approach)

```python
import logging
import threading
import time
from typing import Optional


class Logger:
    """Cách làm sai: mỗi nơi tạo một Logger riêng, không kiểm soát."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handler = logging.FileHandler(f"app_{name}.log")
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        # Mỗi lần tạo Logger mới, lại mở một file handle mới
        print(f"[Logger] Khởi tạo logger '{name}' — mở file handle mới")

    def info(self, message: str) -> None:
        print(f"[{self.name}] INFO: {message}")
        # Ghi vào file...


class TransactionService:
    """Dịch vụ giao dịch — mỗi instance tạo logger riêng."""

    def __init__(self) -> None:
        self.logger = Logger("transaction")

    def process(self, tx_id: str) -> None:
        self.logger.info(f"Process transaction {tx_id}")


class AuditService:
    """Dịch vụ kiểm toán — cũng tạo logger riêng."""

    def __init__(self) -> None:
        self.logger = Logger("audit")

    def audit(self, tx_id: str) -> None:
        self.logger.info(f"Audit transaction {tx_id}")


# Sử dụng — mỗi service tạo logger riêng, không chia sẻ
tx_service = TransactionService()
audit_service = AuditService()
tx_service.process("TXN-001")
audit_service.audit("TXN-001")
# Output: hai Logger riêng biệt, hai file handle riêng biệt, lãng phí tài nguyên.
```

### Refactored với Singleton

```python
import threading
import logging
from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class LogLevel(Enum):
    """Các cấp độ log được định nghĩa rõ ràng."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass(frozen=True)
class LogRecord:
    """Một record log bất biến — đảm bảo tính nhất quán."""
    timestamp: datetime
    level: LogLevel
    module: str
    message: str
    trace_id: Optional[str] = None


class LogSink(ABC):
    """Abstract sink: nơi log được ghi đến — có thể là file, console, network."""

    @abstractmethod
    def write(self, record: LogRecord) -> None: ...


class ConsoleSink(LogSink):
    """Ghi log ra console với màu sắc."""

    COLORS = {
        LogLevel.ERROR: "\033[91m",  # Red
        LogLevel.WARNING: "\033[93m",  # Yellow
        LogLevel.INFO: "\033[94m",  # Blue
        LogLevel.DEBUG: "\033[90m",  # Grey
        LogLevel.CRITICAL: "\033[91;1m",  # Bold Red
    }
    RESET = "\033[0m"

    def write(self, record: LogRecord) -> None:
        color = self.COLORS.get(record.level, self.RESET)
        trace = f" [{record.trace_id}]" if record.trace_id else ""
        print(
            f"{color}{record.timestamp.isoformat()} | "
            f"{record.level.name:8} | {record.module:15} | "
            f"{record.message}{trace}{self.RESET}"
        )


class FileSink(LogSink):
    """Ghi log ra file với cơ chế buffer và rotation."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._buffer: list[LogRecord] = []
        self._lock = threading.Lock()

    def write(self, record: LogRecord) -> None:
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= 10:  # Flush mỗi 10 records
                self._flush()

    def _flush(self) -> None:
        with open(self.filepath, "a", encoding="utf-8") as f:
            for record in self._buffer:
                trace = f" [{record.trace_id}]" if record.trace_id else ""
                f.write(
                    f"{record.timestamp.isoformat()} | "
                    f"{record.level.name:8} | {record.module:15} | "
                    f"{record.message}{trace}\n"
                )
        self._buffer.clear()

    def flush(self) -> None:
        with self._lock:
            self._flush()


class AppLogger:
    """
    Singleton Logger — một instance duy nhất trong toàn bộ ứng dụng.
    Thread-safe với double-checked locking.
    """

    _instance: Optional["AppLogger"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "AppLogger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, min_level: LogLevel = LogLevel.INFO) -> None:
        if getattr(self, "_initialized", False):
            return  # Đã khởi tạo — không khởi tạo lại
        self._min_level = min_level
        self._sinks: list[LogSink] = []
        self._initialized = True

    def add_sink(self, sink: LogSink) -> "AppLogger":
        """Thêm sink mới (fluent API)."""
        self._sinks.append(sink)
        return self

    def _log(self, level: LogLevel, module: str, message: str, trace_id: Optional[str] = None) -> None:
        if level.value < self._min_level.value:
            return
        record = LogRecord(
            timestamp=datetime.now(),
            level=level,
            module=module,
            message=message,
            trace_id=trace_id,
        )
        for sink in self._sinks:
            sink.write(record)

    def info(self, module: str, message: str, trace_id: Optional[str] = None) -> None:
        self._log(LogLevel.INFO, module, message, trace_id)

    def error(self, module: str, message: str, trace_id: Optional[str] = None) -> None:
        self._log(LogLevel.ERROR, module, message, trace_id)

    def warning(self, module: str, message: str, trace_id: Optional[str] = None) -> None:
        self._log(LogLevel.WARNING, module, message, trace_id)

    def debug(self, module: str, message: str, trace_id: Optional[str] = None) -> None:
        self._log(LogLevel.DEBUG, module, message, trace_id)

    def flush(self) -> None:
        for sink in self._sinks:
            if hasattr(sink, "flush"):
                sink.flush()

    @classmethod
    def reset(cls) -> None:
        """Chỉ dùng cho testing — reset instance."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.flush()
            cls._instance = None


# ========== SỬ DỤNG THỰC TẾ ==========

# --- Khởi tạo Singleton lần đầu (và cũng là lần duy nhất) ---
logger = AppLogger(min_level=LogLevel.INFO)
logger.add_sink(ConsoleSink()).add_sink(FileSink("app.log"))


class PaymentProcessor:
    """Xử lý thanh toán — dùng chung logger Singleton."""

    def process_payment(self, user_id: str, amount: float, tx_id: str) -> bool:
        logger.info(
            module="PaymentProcessor",
            message=f"Bắt đầu xử lý thanh toán: user={user_id}, amount={amount}",
            trace_id=tx_id,
        )
        # Logic xử lý thanh toán...
        if amount > 10000:
            logger.warning(
                module="PaymentProcessor",
                message=f"Giao dịch lớn cần xác nhận thêm: {amount}",
                trace_id=tx_id,
            )
        logger.info(
            module="PaymentProcessor",
            message=f"Hoàn tất thanh toán: {tx_id}",
            trace_id=tx_id,
        )
        return True


class FraudDetector:
    """Phát hiện gian lận — dùng chính logger Singleton."""

    def analyze(self, tx_id: str, amount: float, user_id: str) -> str:
        logger.info(
            module="FraudDetector",
            message=f"Phân tích giao dịch: {tx_id}",
            trace_id=tx_id,
        )
        if amount > 50000:
            logger.error(
                module="FraudDetector",
                message=f"PHÁT HIỆN GIAN LẬN TIỀM NĂNG: {tx_id}",
                trace_id=tx_id,
            )
            return "flagged"
        logger.debug(
            module="FraudDetector",
            message=f"Giao dịch an toàn: {tx_id}",
            trace_id=tx_id,
        )
        return "safe"


# Minh họa Singleton hoạt động
logger1 = AppLogger()
logger2 = AppLogger()
print(f"logger1 is logger2: {logger1 is logger2}")  # True
print(f"logger1._sinks is logger2._sinks: {logger1._sinks is logger2._sinks}")  # True

# Các service dùng chung một logger
processor = PaymentProcessor()
detector = FraudDetector()

tx_id = "TXN-2024-001"
processor.process_payment("user_42", 15000.0, tx_id)
result = detector.analyze(tx_id, 15000.0, "user_42")
print(f"Kết quả phân tích: {result}")
```

## Sơ đồ UML

```
┌────────────────────────────────────────────────────────┐
│                    <<class>> AppLogger                 │
├────────────────────────────────────────────────────────┤
│ - _instance: Optional[AppLogger]                      │
│ - _lock: Lock                                         │
│ - _initialized: bool                                  │
│ - _min_level: LogLevel                                │
│ - _sinks: list[LogSink]                               │
├────────────────────────────────────────────────────────┤
│ + __new__() -> AppLogger                              │
│ + __init__(min_level: LogLevel)                       │
│ + add_sink(sink: LogSink) -> AppLogger                │
│ + info(module, message, trace_id)                     │
│ + error(module, message, trace_id)                    │
│ + warning(module, message, trace_id)                  │
│ + debug(module, message, trace_id)                    │
│ + flush()                                             │
│ + reset() [classmethod]                               │
└────────────────────────────────────────────────────────┘
            ▲
            │  Singleton — chỉ một instance duy nhất
            │
            ▼
┌──────────────────────────────┐
│     «interface» LogSink    │
├──────────────────────────────┤
│ + write(record: LogRecord)   │
└──────────────────────────────┘
            ▲
            │
    ┌───────┴──────────┐
    │                  │
┌──────────┐    ┌──────────┐
│ConsoleSink│    │ FileSink │
├──────────┤    ├──────────┤
│+ write() │    │+ write() │
└──────────┘    │+ flush() │
                └──────────┘

┌──────────────────────────────────┐
│   @dataclass(frozen) LogRecord   │
├──────────────────────────────────┤
│ + timestamp: datetime            │
│ + level: LogLevel                │
│ + module: str                    │
│ + message: str                   │
│ + trace_id: Optional[str]        │
└──────────────────────────────────┘
```

## So sánh với Pattern liên quan

| Pattern | Điểm giống | Điểm khác biệt chính |
|---------|-----------|---------------------|
| **Monostate** | Cùng chia sẻ trạng thái | Monostate dùng class variable để chia sẻ, cho phép nhiều instance nhưng cùng state. Singleton chỉ cho phép một instance. Monostate dễ kiểm thử hơn nhưng có thể gây nhầm lẫn. |
| **Factory Method** | Đều kiểm soát việc tạo object | Factory Method tập trung vào việc *quyết định class nào được instantiate*, trong khi Singleton tập trung vào *số lượng instance*. Có thể kết hợp: dùng Factory Method để tạo Singleton instance. |
| **Dependency Injection (DI)** | Đều giải quyết vấn đề global access | DI container quản lý vòng đời object (có thể singleton scope) nhưng không bắt buộc global point of access. DI linh hoạt hơn, dễ kiểm thử hơn, và không tạo global state. Singleton thường được coi là "service locator anti-pattern" khi dùng quá mức. |

**Khi nào chọn Singleton thay vì DI container?**
- Dự án nhỏ, không dùng framework có DI.
- Cần giải pháp đơn giản, không muốn thêm dependency.
- System library, utility code — nơi không kiểm soát được lifecycle của ứng dụng.

**Khi nào chọn DI container thay vì Singleton?**
- Dự án lớn, nhiều module, cần kiểm thử rộng rãi.
- Cần thay đổi implementation theo môi trường (dev/staging/prod).
- Muốn kiểm soát fine-grained lifecycle (singleton, prototype, request scope).

## Ứng dụng thực tế

### 1. Python `logging` module — module-level singleton

Thư viện chuẩn `logging` của Python quản lý logger theo tên. `logging.getLogger(name)` trả về cùng một instance nếu gọi với cùng tên:

```python
import logging

# Logger gốc — singleton implicitly
root_logger = logging.getLogger()
# Hai lần gọi với cùng tên — cùng instance
logger_a = logging.getLogger("my_app")
logger_b = logging.getLogger("my_app")
assert logger_a is logger_b  # True

# Cấu hình một lần
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger_a.addHandler(handler)
logger_a.setLevel(logging.INFO)

# Dùng ở module khác — vẫn là cùng logger
logger_b.info("Hello từ logger_b — thực chất là logger_a!")
```

### 2. SQLAlchemy — Engine Singleton

SQLAlchemy tạo `Engine` object rất tốn kém (kết nối pool, dialect, reflection). Nên dùng Singleton pattern cho Engine:

```python
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_engine: Engine | None = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            "postgresql://user:pass@localhost/mydb",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine
```

### 3. Django — AppConfig và cache

Django sử dụng Singleton cho `AppConfig` — mỗi app chỉ có một config instance. Tương tự, Django cache framework cũng dùng Singleton pattern qua `django.core.cache.caches`:

```python
from django.core.cache import caches

# cache['default'] luôn trả về cùng một instance
default_cache = caches["default"]
other_ref = caches["default"]
assert default_cache is other_ref  # True
```

### 4. Redis client connection pool

Các Redis client thường khuyến khích Singleton pattern để tái sử dụng connection pool:

```python
import redis.asyncio as aioredis
from typing import Optional

_redis_pool: Optional[aioredis.Redis] = None

async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.Redis.from_url(
            "redis://localhost:6379/0",
            max_connections=20,
            decode_responses=True,
        )
    return _redis_pool
```

## Kiểm thử

Testing Singleton cần cẩn thận vì global state tồn tại xuyên suốt các test case:

```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset Singleton trước mỗi test để tránh side-effects."""
    AppLogger.reset()
    yield
    AppLogger.reset()


def test_singleton_guarantee():
    """Đảm bảo chỉ một instance duy nhất."""
    logger_a = AppLogger()
    logger_b = AppLogger()
    assert logger_a is logger_b


def test_multiple_calls_return_same_instance():
    """Gọi AppLogger() nhiều lần phải trả về cùng instance."""
    instances = [AppLogger() for _ in range(100)]
    assert all(instances[0] is inst for inst in instances)


def test_initialization_only_once():
    """__init__ chỉ chạy một lần, lần sau bỏ qua."""
    logger = AppLogger(min_level=LogLevel.DEBUG)
    assert logger._min_level == LogLevel.DEBUG

    # Lần gọi thứ hai — __init__ bỏ qua vì _initialized = True
    same_logger = AppLogger(min_level=LogLevel.ERROR)
    assert same_logger._min_level == LogLevel.DEBUG  # Vẫn là DEBUG


def test_log_level_filtering():
    """Log dưới min_level không được ghi."""
    logger = AppLogger(min_level=LogLevel.WARNING)
    mock_sink = MagicMock(spec=LogSink)
    logger.add_sink(mock_sink)

    logger.debug("test", "debug message")
    logger.info("test", "info message")
    logger.warning("test", "warning message")
    logger.error("test", "error message")

    # debug và info bị filter, warning và error được ghi
    assert mock_sink.write.call_count == 2


def test_thread_safety():
    """Kiểm tra thread-safety: 100 thread cùng truy cập Singleton."""
    import concurrent.futures

    def get_logger() -> int:
        logger = AppLogger()
        return id(logger)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(get_logger) for _ in range(100)]
        results = [f.result() for f in futures]

    # Tất cả đều trả về cùng id
    assert len(set(results)) == 1


def test_sink_isolation():
    """Các sink hoạt động độc lập."""
    logger = AppLogger()
    sink_a = MagicMock(spec=LogSink)
    sink_b = MagicMock(spec=LogSink)
    logger.add_sink(sink_a).add_sink(sink_b)

    logger.info("test", "message")

    sink_a.write.assert_called_once()
    sink_b.write.assert_called_once()


def test_reset():
    """Reset Singleton — tạo instance mới."""
    old_id = id(AppLogger())
    AppLogger.reset()
    new_id = id(AppLogger())
    assert old_id != new_id
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| **Kiểm soát instance**: Đảm bảo chỉ một instance duy nhất, tránh lãng phí tài nguyên | **Vi phạm SRP**: Class vừa quản lý business logic vừa quản lý vòng đời instance |
| **Lazy initialization**: Instance chỉ được tạo khi cần, tiết kiệm thời gian khởi động | **Global state**: Khó kiểm thử, khó debug, tạo implicit coupling |
| **Global access point**: Mọi nơi đều có thể truy cập dễ dàng, không cần DI | **Khó mở rộng**: Singleton và subclassing không kết hợp tốt |
| **Thread-safe**: Có thể implement thread-safe một lần ở một chỗ | **Testability**: Mock Singleton khó hơn so với DI. Cần cơ chế reset đặc biệt |
| **Tiết kiệm bộ nhớ**: Giảm số lượng object không cần thiết | **Tight coupling**: Code gọi Singleton trực tiếp thay vì qua interface |
| **Consistency**: Dữ liệu tập trung, nhất quán | **Concurrency**: Nếu không xử lý đúng, có thể gặp race condition |
| **Dễ dùng**: Không cần framework phức tạp, chỉ vài dòng code | **Violates Open/Closed**: Khó thay đổi behavior mà không sửa class gốc |
| **Namespace sạch**: Một điểm truy cập duy nhất thay vì biến toàn cục lộn xộn | **Parallel testing**: Các test chạy song song dễ ảnh hưởng lẫn nhau qua global state |

## Kết luận

Singleton là một pattern "con dao hai lưỡi" — cực kỳ hữu ích khi dùng đúng chỗ, nhưng nguy hiểm khi lạm dụng. **Golden rule**: Chỉ dùng Singleton cho những class thực sự cần **duy nhất một instance** về mặt bản chất, như: Logger, Configuration Manager, Connection Pool, Hardware Interface. Đừng dùng Singleton chỉ vì "lười truyền dependency" hoặc "muốn truy cập nhanh từ mọi nơi".

Trong kiến trúc hiện đại, hãy ưu tiên **Dependency Injection** và để framework quản lý vòng đời object. Singleton chỉ nên xuất hiện ở tầng infrastructure (logging, caching, config) — không bao giờ ở tầng business logic. Khi bạn viết `class DatabaseConnectionPool` và tự hỏi "có nên Singleton không?", hãy tự hỏi ngược lại: "Tại sao tôi muốn có nhiều hơn một connection pool?" Nếu câu trả lời là "không có lý do chính đáng" — thì Singleton là lựa chọn đúng.

Cuối cùng, hãy nhớ: **Singleton là pattern kiểm soát số lượng, không phải pattern kiểm soát truy cập**. Nếu bạn muốn kiểm soát truy cập, hãy dùng Proxy hoặc Facade. Nếu bạn muốn kiểm soát số lượng instance linh hoạt hơn (tối đa N instance), hãy dùng Pool pattern.
