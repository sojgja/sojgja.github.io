---
id: singleton
title: Singleton
sidebar_label: 🥇 Singleton
sidebar_position: 2
---

# Singleton

> *"Đảm bảo một class chỉ có một instance duy nhất, và cung cấp một điểm truy cập toàn cục đến instance đó."*
> — **Gang of Four**, *Design Patterns* (1994)

**Singleton** thuộc nhóm **Creational Patterns**. Pattern này giải quyết vấn đề: làm sao để class của bạn chỉ có **đúng một** instance trong toàn bộ vòng đời ứng dụng, và làm sao để mọi nơi đều truy cập được instance đó?

Đây là pattern gây tranh cãi nhất trong GoF — vừa cực kỳ hữu ích, vừa bị chỉ trích nặng nề. Có người yêu, có người ghét. Cá nhân tôi cho rằng, Singleton cũng như con dao, dùng đúng chỗ thì lợi, dùng sai thì hại.

---

## Bài toán chi tiết

Hãy tưởng tượng bạn đang xây dựng một **hệ thống xử lý giao dịch tài chính thời gian thực** cho một công ty fintech. Hệ thống của bạn có hàng trăm microservice: xác thực người dùng, quản lý tài khoản, ghi nhận giao dịch, báo cáo, cảnh báo gian lận. Tất cả đều cần ghi log vào cùng một chỗ.

Vấn đề đầu tiên: mỗi service tạo một Logger riêng. Không có cơ chế kiểm soát, hệ thống có thể có hàng nghìn Logger, mỗi cái mở một kết nối riêng đến file log, đến ELK, Splunk — tài nguyên tiêu tốn một cách lãng phí.

Vấn đề thứ hai: tính nhất quán của dữ liệu log. Mỗi Logger có buffer riêng, thứ tự log entry bị xáo trộn, log từ cùng một luồng xử lý bị chia cắt thành nhiều file. Khi cơ quan thuế yêu cầu audit trail, bạn không thể trả lời: "log nằm rải rác ở 15 file khác nhau."

Cách tiếp cận ngây thơ: truyền Logger qua constructor hoặc tham số function. Cách này buộc mọi class phải nhận Logger làm dependency — dẫn đến **telescoping constructor**: `def __init__(self, db, cache, logger, config, ...)`. Hàng trăm class phải sửa chỉ vì muốn ghi log.

---

## Giải pháp với Singleton

Singleton giải quyết triệt để bài toán này. Class `Logger` chỉ có **một instance duy nhất**. Instance được khởi tạo **lazy** — chỉ khi có yêu cầu lần đầu — và được lưu ở biến class. Mọi lời gọi `Logger()` sau đó đều trả về chính instance đó.

Cơ chế hoạt động:

1. **Private constructor** (qua `__new__`): Ngăn tạo instance mới tùy tiện
2. **Static accessor**: Điểm truy cập duy nhất
3. **Lazy initialization**: Chỉ tạo khi cần
4. **Thread safety**: Đa luồng không tạo hai instance

Với Singleton, module A ghi log "User login", module B ghi log "Transaction processed" — cả hai qua cùng một instance, cùng một buffer, cùng một kết nối. Thứ tự log được bảo toàn, tài nguyên được tiết kiệm.

---

## Phân tích thiết kế

Singleton là pattern "dao hai lưỡi". Nó vi phạm **Single Responsibility Principle**: class vừa quản lý business logic, vừa quản lý vòng đời instance. Nó tạo ra **global state** — thứ bị coi là anti-pattern trong thiết kế hiện đại.

**Khi nào KHÔNG nên dùng Singleton:**

- Cần nhiều instance khác nhau (một logger cho mỗi user session)
- Muốn unit test nghiêm ngặt — global state là kẻ thù của kiểm thử
- Class có thể phát triển thành nhiều biến thể (cần subclassing)
- Ứng dụng serverless (Lambda, Cloud Functions) — mỗi invocation chạy container khác
- Có thể dùng DI container thay vì tự quản lý

Trong thực tế hiện đại, nhiều framework khuyến khích dùng **DI container** (Spring, Google Guice) hoặc **Monostate pattern** thay vì Singleton thuần túy.

---

## Code hoàn chỉnh

### Cách làm sai

```python
import logging
from typing import Optional


class Logger:
    """Cách sai: mỗi nơi tạo một Logger riêng, không kiểm soát."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handler = logging.FileHandler(f"app_{name}.log")
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        print(f"[Logger] Khởi tạo logger '{name}' — mở file handle mới")

    def info(self, message: str) -> None:
        print(f"[{self.name}] INFO: {message}")


class TransactionService:
    def __init__(self) -> None:
        self.logger = Logger("transaction")

    def process(self, tx_id: str) -> None:
        self.logger.info(f"Process transaction {tx_id}")


class AuditService:
    def __init__(self) -> None:
        self.logger = Logger("audit")

    def audit(self, tx_id: str) -> None:
        self.logger.info(f"Audit transaction {tx_id}")
```

Vấn đề: hai service tạo hai Logger riêng biệt, hai file handle, lãng phí tài nguyên.

### Refactored với Singleton

```python
import threading
from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass(frozen=True)
class LogRecord:
    timestamp: datetime
    level: LogLevel
    module: str
    message: str
    trace_id: Optional[str] = None


class LogSink(ABC):
    @abstractmethod
    def write(self, record: LogRecord) -> None: ...


class ConsoleSink(LogSink):
    COLORS = {
        LogLevel.ERROR: "\033[91m",
        LogLevel.WARNING: "\033[93m",
        LogLevel.INFO: "\033[94m",
        LogLevel.DEBUG: "\033[90m",
        LogLevel.CRITICAL: "\033[91;1m",
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
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._buffer: list[LogRecord] = []
        self._lock = threading.Lock()

    def write(self, record: LogRecord) -> None:
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= 10:
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
            return
        self._min_level = min_level
        self._sinks: list[LogSink] = []
        self._initialized = True

    def add_sink(self, sink: LogSink) -> "AppLogger":
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


# Minh họa Singleton hoạt động
logger = AppLogger(min_level=LogLevel.INFO)
logger.add_sink(ConsoleSink()).add_sink(FileSink("app.log"))


class PaymentProcessor:
    def process_payment(self, user_id: str, amount: float, tx_id: str) -> bool:
        logger.info(module="PaymentProcessor", message=f"Xử lý thanh toán: user={user_id}", trace_id=tx_id)
        if amount > 10000:
            logger.warning(module="PaymentProcessor", message=f"Giao dịch lớn: {amount}", trace_id=tx_id)
        logger.info(module="PaymentProcessor", message=f"Hoàn tất: {tx_id}", trace_id=tx_id)
        return True


class FraudDetector:
    def analyze(self, tx_id: str, amount: float) -> str:
        logger.info(module="FraudDetector", message=f"Phân tích: {tx_id}", trace_id=tx_id)
        if amount > 50000:
            logger.error(module="FraudDetector", message=f"PHÁT HIỆN GIAN LẬN: {tx_id}", trace_id=tx_id)
            return "flagged"
        return "safe"


# Tất cả service dùng chung một logger
logger1 = AppLogger()
logger2 = AppLogger()
print(f"logger1 is logger2: {logger1 is logger2}")  # True
```

---

## Ứng dụng thực tế

### Python `logging` module

```python
import logging

logger_a = logging.getLogger("my_app")
logger_b = logging.getLogger("my_app")
assert logger_a is logger_b  # True — cùng instance
```

### SQLAlchemy Engine

```python
from sqlalchemy import create_engine

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine("postgresql://...")
    return _engine
```

### Django cache

```python
from django.core.cache import caches
default_cache = caches["default"]  # Luôn cùng instance
```

---

## So sánh với Pattern liên quan

| Pattern | Giống | Khác |
|---------|-------|------|
| **Monostate** | Cùng chia sẻ trạng thái | Nhiều instance, cùng state. Dễ kiểm thử hơn |
| **Factory Method** | Kiểm soát tạo object | Factory Method quyết định *class nào*, Singleton quyết định *bao nhiêu instance* |
| **DI Container** | Giải quyết global access | DI linh hoạt, dễ test, không tạo global state |

---

## Kết luận

Singleton là pattern "dao hai lưỡi" — cực kỳ hữu ích khi dùng đúng chỗ, nguy hiểm khi lạm dụng.

**Golden rule**: Chỉ dùng Singleton cho những class thực sự cần **duy nhất một instance** về mặt bản chất: Logger, Configuration Manager, Connection Pool, Hardware Interface. Đừng dùng Singleton chỉ vì lười truyền dependency.

Trong kiến trúc hiện đại, ưu tiên **Dependency Injection** để framework quản lý vòng đời object. Singleton chỉ nên ở tầng infrastructure (logging, caching, config) — không bao giờ ở tầng business logic.

Khi bạn viết `class DatabaseConnectionPool` và tự hỏi "có nên Singleton không?", hãy tự hỏi ngược lại: "Tại sao tôi muốn có nhiều hơn một connection pool?" Nếu câu trả lời là "không có lý do chính đáng" — thì Singleton là lựa chọn đúng.

Cuối cùng, hãy nhớ: Singleton kiểm soát **số lượng**, không phải **truy cập**. Muốn kiểm soát truy cập, dùng Proxy hoặc Facade.

---
*Trân trọng!*
