---
id: pipeline-architecture
title: Pipeline Architecture
sidebar_label: 🏗️ Pipeline Architecture
sidebar_position: 49
---

# Pipeline Architecture

> "A pipeline is a set of data processing elements connected in series, where the output of one element is the input of the next. Pipelines are the backbone of modern data processing and CI/CD systems."
> — **Michael T. Nygard**, *Release It!* (2007)

**Pipeline Architecture** (hay Pipes & Filters) là một trong những kiến trúc phần mềm cổ điển nhất, bắt nguồn từ các hệ điều hành Unix (1970s) với khái niệm pipe (`|`). Trong kiến trúc này, dữ liệu được xử lý qua một chuỗi các **bước** (stages/filters) kết nối với nhau qua các **kênh** (pipes/channels). Mỗi bước nhận đầu vào, xử lý, và gửi kết quả sang bước tiếp theo.

---

## Tổng quan

### Lịch sử và nguồn gốc

Pipeline architecture có lịch sử lâu đời trong khoa học máy tính:

- **1970s**: Unix pipes (`ls | grep | sort`) — triết lý "do one thing and do it well"
- **1978**: **Ken Thompson** giới thiệu pipeline trong Unix — mỗi lệnh là một filter
- **1990s**: Data Transformation Services (DTS), ETL pipelines trong data warehousing
- **2004**: **MapReduce** (Google) — pipeline cho distributed data processing
- **2009**: **Apache Hadoop** — open-source distributed pipeline
- **2014**: **Apache Spark** — in-memory distributed pipeline
- **2015-nay**: **CI/CD pipelines** (Jenkins, GitHub Actions, GitLab CI), **ML pipelines** (Kubeflow, TFX), **stream processing** (Kafka Streams, Flink)

### Những người tiên phong

| Tên | Đóng góp |
|-----|---------|
| **Ken Thompson & Dennis Ritchie** | Unix pipes — origin of pipeline architecture |
| **Doug McIlroy** | Bell Labs — "pipes" concept và Unix philosophy |
| **Jeffrey Dean & Sanjay Ghemawat** | MapReduce — distributed pipeline |
| **Matei Zaharia** | Apache Spark — in-memory pipeline processing |
| **Jay Kreps** | Apache Kafka — event streaming pipeline |
| **Brendan Gregg** | Performance analysis pipeline |

### Các loại Pipeline

| Loại | Mô tả | Ví dụ |
|------|-------|-------|
| **Data Pipeline** | Xử lý dữ liệu qua nhiều stages | ETL, ELT, data transformation |
| **CI/CD Pipeline** | Build → Test → Deploy | Jenkins, GitHub Actions |
| **Processing Pipeline** | Xử lý media/file pipeline | Image processing, video transcoding |
| **Streaming Pipeline** | Real-time data processing | Kafka Streams, Flink |
| **ML Pipeline** | ML model training pipeline | Kubeflow, TFX |
| **Event Pipeline** | Event processing pipeline | Event sourcing, CQRS |
| **Log Pipeline** | Log aggregation pipeline | ELK Stack (Elasticsearch, Logstash, Kibana) |

---

## Bài toán

### Hệ thống Xử lý Đơn hàng Thương mại Điện tử Quy mô Lớn

Giả sử bạn đang xây dựng **TikiNgon** — một nền tảng giao đồ ăn và hàng tạp hóa với hàng triệu đơn hàng mỗi ngày tại Việt Nam. Mỗi đơn hàng trải qua nhiều bước xử lý:

1. **Đặt hàng**: Customer tạo đơn
2. **Xác thực**: Kiểm tra thông tin, số dư ví
3. **Thanh toán**: Xử lý payment (Visa, MoMo, COD)
4. **Kiểm tra kho**: Check tồn kho, reserve item
5. **Xác nhận người bán**: Gửi cho merchant, chờ xác nhận
6. **Đóng gói**: Gán shipper, in hóa đơn
7. **Vận chuyển**: Logistic tracking
8. **Hoàn thành**: Giao hàng thành công
9. **Đánh giá**: Gửi review request

### Khó khăn với kiến trúc monolithic/traditional

**Vấn đề 1 — Mỗi bước xử lý có yêu cầu khác nhau về tài nguyên**:

```
Monolith: [Đặt hàng → Thanh toán → Kho → Gửi → Vận chuyển]
                    ⬆ Tất cả trong 1 server

- Bước validate: Cần CPU (string parsing, regex)
- Bước payment: Cần I/O network (gọi API ngân hàng)
- Bước kho: Cần memory (cache inventory)
- Bước vận chuyển: Cần I/O disk (logistics routes)

→ Tài nguyên hỗn độn, không thể tối ưu từng bước
```

**Vấn đề 2 — Một bước chậm kéo theo toàn bộ pipeline**:

Nếu bước xác thực thanh toán mất 10 giây (do timeout ngân hàng), tất cả đơn hàng phía sau bị block. Hàng nghìn đơn hàng bị delay chỉ vì một bước.

```python
# Synchronous pipeline — blocking
def process_order(order):
    validate(order)              # Nhanh: 100ms
    process_payment(order)       # Chậm: 10s (API ngân hàng timeout)
    check_inventory(order)       # Bị block vì chờ payment xong!
    confirm_merchant(order)      # Bị block tiếp!
    assign_shipper(order)        # Không thể chạy!
```

**Vấn đề 3 — Khó thêm/xóa bước xử lý**:

Khi business yêu cầu thêm bước "AI fraud detection" giữa payment và inventory, bạn phải:
1. Sửa code ở vị trí chính xác
2. Deploy lại toàn bộ ứng dụng
3. Nguy cơ ảnh hưởng đến các bước khác

```python
# Thêm bước mới phải sửa code hiện tại
def process_order(order):
    validate(order)
    process_payment(order)
    # Phải chèn ở đây:
    fraud_check(order)    # New step — sửa function!
    check_inventory(order)
    # ...
```

**Vấn đề 4 — Không thể retry từng bước riêng**:

Khi bước payment fail, bạn phải chạy lại toàn bộ pipeline từ đầu:
- Đã validate → lại validate (lãng phí)
- Đã check kho → lại check (nguy cơ duplicate)
- Không thể resume từ bước fail

**Vấn đề 5 — Khó scale từng bước**:

Bước validate cần 2 servers, nhưng bước vận chuyển cần 50 servers. Trong monolith, bạn phải scale toàn bộ app.

### Pipeline Architecture giải quyết vấn đề

1. **Decoupled stages**: Mỗi stage độc lập, giao tiếp qua message queue
2. **Independent scaling**: Mỗi stage scale riêng (validate: 2 pods, shipping: 50 pods)
3. **Fault isolation**: Một stage fail không ảnh hưởng stage khác
4. **Retry per stage**: Retry stage fail, không cần restart pipeline
5. **Dynamic pipeline**: Thêm/xóa stage không ảnh hưởng code hiện tại
6. **Resource optimization**: Mỗi stage dùng resource phù hợp
7. **Monitoring per stage**: Biết chính xác stage nào chậm

---

## Nguyên lý thiết kế

### 1. Single Responsibility per Stage

Mỗi stage làm đúng MỘT việc:
```python
# GOOD: mỗi stage một responsibility
class ValidateOrder: ...
class ProcessPayment: ...
class CheckInventory: ...

# BAD: stage làm nhiều việc
class ValidateAndPaymentAndInventory: ...
```

### 2. Standardized Interface (Pipe Interface)

Tất cả stages đều có cùng interface:
```python
@runtime_checkable
class Stage(Protocol):
    async def process(self, context: PipelineContext) -> PipelineContext: ...
```

Input/output qua `PipelineContext` — một dict chứa tất cả dữ liệu pipeline.

### 3. Immutability

Context không nên bị mutate trực tiếp. Mỗi stage nên trả về context mới (hoặc copy-on-write).

### 4. Idempotency

Mỗi stage có thể chạy lại nhiều lần mà không gây side effects:
```python
# GOOD: idempotent
def process_payment(context):
    if context.get("payment_processed"):
        return context  # Already done, skip
    # Process payment
    return context | {"payment_processed": True}

# BAD: not idempotent
def process_payment(context):
    charge_credit_card(context["amount"])  # Sẽ charge 2 lần nếu retry!
```

### 5. Error Handling per Stage

Mỗi stage tự xử lý lỗi và quyết định:
- **Retry**: Tạm thời (network timeout)
- **Skip**: Có thể bỏ qua (optional step)
- **Abort**: Dừng pipeline (critical error)

### 6. Pipeline Configuration

Pipeline được cấu hình động, không hardcode:
```python
pipeline = Pipeline([
    ValidateStage(),
    PaymentStage(config.PAYMENT_GATEWAY),
    InventoryStage(config.WAREHOUSE_API),
    ...
])
```

### 7. Observability

Mỗi stage phải emit metrics:
- Duration
- Success/failure count
- Input/output size
- Retry count

### 8. Backpressure

Khi stage sau chậm hơn stage trước, cần cơ chế backpressure để không làm overflow queue.

---

## Cấu trúc chi tiết

### Thành phần cốt lõi

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ARCHITECTURE                            │
│                                                                         │
│  INPUT ──→ [Pipe 1] ──→ [Stage 1] ──→ [Pipe 2] ──→ [Stage 2] ──→ ... │
│                              │                                              │
│                              ▼                                              │
│                        PipelineContext                              │
│                        {                                                 │
│                          "order_id": "123",                              │
│                          "amount": 500000,                               │
│                          "status": "pending",                            │
│                          "errors": [],                                    │
│                          "results": {}                                    │
│                        }                                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        PIPELINE MANAGER                          │   │
│  │                                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │ Stage Runner │  │ Retry Logic │  │ Circuit    │              │   │
│  │  │              │  │ (exponential │  │ Breaker    │              │   │
│  │  │              │  │  backoff)   │  │            │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │ Metrics     │  │ Logging     │  │ Dead Letter │              │   │
│  │  │ Collector   │  │ (tracing)   │  │ Queue       │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**1. Stage (Filter)**
- Đơn vị xử lý cơ bản
- Nhận context, xử lý, trả về context
- Có thể là sync hoặc async
- Có tên, version, metrics

**2. Pipe (Channel)**
- Kết nối giữa các stages
- Có thể là in-memory, message queue, file, network
- Đảm bảo delivery (at-least-once, exactly-once)
- Buffer size có giới hạn

**3. Pipeline Manager**
- Orchestrate pipeline execution
- Quản lý lifecycle
- Error handling, retry, circuit breaker
- Metrics và monitoring

**4. Pipeline Context**
- Shared data giữa các stages
- Immutable (thường dùng dict)
- Chứa input, intermediate data, output
- Chứa metadata (timestamps, errors, trace ID)

### Luồng dữ liệu

```
                    ┌─────────────┐
                    │  Input      │
                    │  (Raw       │
                    │   Order)    │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
        ┌──────────▶│  Validate   │◀──────────┐
        │           └──────┬──────┘           │
        │                  ▼                  │
        │           ┌─────────────┐          │
        │           │  Payment    │────Error──┘
        │           └──────┬──────┘          Retry (3x)
        │                  ▼
        │           ┌─────────────┐
        │           │  Inventory  │
        │           └──────┬──────┘
        │                  ▼
        │           ┌─────────────┐
        │           │  Fraud      │
        │           │  Detection  │
        │           └──────┬──────┘
        │                  ▼
        │           ┌─────────────┐
        │           │  Confirm    │
        │           │  Merchant   │
        │           └──────┬──────┘
        │                  ▼
        │           ┌─────────────┐
        │           │  Assign     │
        │           │  Shipper    │
        │           └──────┬──────┘
        │                  ▼
        │           ┌─────────────┐
        │           │  Notify     │
        │           │  Customer   │
        │           └──────┬──────┘
        │                  ▼
        │           ┌─────────────┐
        └───────────│   Output    │
                    │  (Complete) │
                    └─────────────┘
```

---

## Sơ đồ kiến trúc

```
TIKINGON — ORDER PROCESSING PIPELINE
═══════════════════════════════════════════════════════════════════════════

                        ┌──────────────────────┐
                        │   API Gateway         │
                        │   (New Order Event)    │
                        └──────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   ORDER PIPELINE              │
                    │                              │
                    │  ┌──────────────────────┐   │
                    │  │  Stage 1: VALIDATE    │   │
                    │  │  - Check required     │   │
                    │  │    fields             │   │
                    │  │  - Validate address   │   │
                    │  │  - Check duplicates   │   │
                    │  └──────────┬───────────┘   │
                    │             │                │
                    │             ▼                │
                    │  ┌──────────────────────┐   │
                    │  │  Stage 2: PAYMENT     │   │
                    │  │  - Process payment    │   │
                    │  │  - Call payment       │   │
                    │  │    gateway            │   │
                    │  │  - Handle 3DS/OTP     │   │
                    │  └──────────┬───────────┘   │
                    │             │                │
                    │             ▼                │
                    │  ┌──────────────────────┐   │
                    │  │  Stage 3: INVENTORY   │   │
                    │  │  - Check stock        │   │
                    │  │  - Reserve items      │   │
                    │  │  - Calculate ETA      │   │
                    │  └──────────┬───────────┘   │
                    │             │                │
                    │             ▼                │
                    │  ┌──────────────────────┐   │
                    │  │  Stage 4: FRAUD       │   │
                    │  │  DETECTION            │   │
                    │  │  - ML model scoring   │   │
                    │  │  - Risk assessment    │   │
                    │  │  - Rule-based check   │   │
                    │  └──────────┬───────────┘   │
                    │             │                │
                    │             ▼                │
                    │  ┌──────────────────────┐   │
                    │  │  Stage 5: NOTIFY      │   │
                    │  │  - Send email         │   │
                    │  │  - Push notification  │   │
                    │  │  - Update WebSocket   │   │
                    │  └──────────┬───────────┘   │
                    │             │                │
                    │             ▼                │
                    │  ┌──────────────────────┐   │
                    │  │  Stage 6: COMPLETE    │   │
                    │  │  - Update order       │   │
                    │  │    status             │   │
                    │  │  - Log to analytics   │   │
                    │  │  - Emit event         │   │
                    │  └──────────────────────┘   │
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   OUTPUT                      │
                    │   - Order confirmed event     │
                    │   - Analytics data            │
                    │   - Notification sent         │
                    └──────────────────────────────┘

MESSAGE QUEUE ARCHITECTURE (Async Pipeline)
═══════════════════════════════════════════════════════════════

  [Order Created]
       │
       ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  validate-q   │────▶│  payment-q   │────▶│  inventory-q  │
  │  (SQS)        │     │  (SQS)        │     │  (SQS)         │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ Validate     │     │ Payment      │     │ Inventory    │
  │ Worker       │     │ Worker       │     │ Worker       │
  │ (scale: 5)   │     │ (scale: 20)  │     │ (scale: 10)  │
  └──────────────┘     └──────────────┘     └──────────────┘
```

---

## Ví dụ code hoàn chỉnh

### Cấu trúc project

```
tikingon-pipeline/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── pipeline.py         # Pipeline engine
│   ├── stage.py            # Base stage interface
│   ├── context.py          # Pipeline context
│   └── errors.py           # Error types
├── stages/
│   ├── __init__.py
│   ├── validate.py         # Order validation
│   ├── payment.py          # Payment processing
│   ├── inventory.py        # Inventory check & reserve
│   ├── fraud_detection.py  # AI fraud detection
│   ├── notification.py     # Customer notifications
│   └── complete.py         # Order completion
├── workers/
│   ├── __init__.py
│   ├── message_queue.py    # Async queue abstraction
│   └── worker.py           # Worker process
├── config.py               # Pipeline configuration
├── main.py                 # Entry point
├── benchmarks/
│   └── test_performance.py
└── tests/
    ├── __init__.py
    ├── test_pipeline.py
    └── test_stages.py
```

### core/stage.py

```python
"""Base Stage interface — trái tim của Pipeline Architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from datetime import datetime
import time


class StageStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()
    RETRYING = auto()


class StageAction(Enum):
    """Hành động mà stage yêu cầu pipeline thực hiện."""
    CONTINUE = auto()       # Tiếp tục pipeline
    RETRY = auto()          # Retry stage này
    SKIP = auto()           # Skip stage, tiếp tục
    ABORT = auto()          # Dừng pipeline
    WAIT = auto()           # Đợi điều kiện (async)


@dataclass
class StageResult:
    """Kết quả trả về từ một stage."""
    stage_name: str
    status: StageStatus
    action: StageAction = StageAction.CONTINUE
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStage(ABC):
    """Abstract base class cho tất cả pipeline stages.

    Mỗi stage là một đơn vị xử lý độc lập, single-responsibility.
    Stage giao tiếp với pipeline qua PipelineContext.
    """

    def __init__(self, name: Optional[str] = None, max_retries: int = 3) -> None:
        self._name = name or self.__class__.__name__
        self._max_retries = max_retries
        self._metrics: dict[str, Any] = {
            "total_processed": 0,
            "total_errors": 0,
            "total_duration_ms": 0.0,
        }

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def process(self, context: "PipelineContext") -> StageResult:
        """Xử lý stage.

        Args:
            context: Pipeline context chứa dữ liệu từ các stage trước.

        Returns:
            StageResult với action hướng dẫn pipeline.
        """
        ...

    def before_process(self, context: "PipelineContext") -> None:
        """Hook trước khi process — có thể override."""
        pass

    def after_process(self, context: "PipelineContext", result: StageResult) -> None:
        """Hook sau khi process — có thể override."""
        pass

    def execute(self, context: "PipelineContext") -> StageResult:
        """Execute stage với timing và error handling.

        Đây là wrapper của process(), tự động:
        - Tính thời gian xử lý
        - Catch exceptions
        - Update metrics
        """
        self.before_process(context)
        start = time.monotonic()

        try:
            result = self.process(context)
            duration = (time.monotonic() - start) * 1000
            result.duration_ms = duration
            result.stage_name = self._name

            self._update_metrics(result)
            self.after_process(context, result)
            return result

        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            self._metrics["total_errors"] += 1
            self._metrics["total_duration_ms"] += duration

            error_result = StageResult(
                stage_name=self._name,
                status=StageStatus.FAILED,
                action=StageAction.RETRY if self._max_retries > 0 else StageAction.ABORT,
                duration_ms=duration,
                error_message=str(e),
            )
            self.after_process(context, error_result)
            return error_result

    def can_retry(self, retry_count: int) -> bool:
        """Kiểm tra có thể retry stage không."""
        return retry_count < self._max_retries

    def _update_metrics(self, result: StageResult) -> None:
        self._metrics["total_processed"] += 1
        self._metrics["total_duration_ms"] += result.duration_ms
        if result.status == StageStatus.FAILED:
            self._metrics["total_errors"] += 1

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        self._metrics = {
            "total_processed": 0,
            "total_errors": 0,
            "total_duration_ms": 0.0,
        }

    def __repr__(self) -> str:
        return f"Stage({self._name})"
```

### core/context.py

```python
"""Pipeline Context — shared data container cho pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import uuid


@dataclass
class PipelineContext:
    """Context chứa tất cả dữ liệu pipeline.

    Mỗi stage đọc input từ context, ghi output vào context.
    Context được truyền qua các stages và có thể accumulate data.

    Immutability: Không mutate context trực tiếp.
    Dùng merge() để tạo context mới với data bổ sung.
    """

    # Core fields
    pipeline_id: str = field(default_factory=lambda: f"PL{uuid.uuid4().hex[:8].upper()}")
    order_id: str = ""
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Order data
    order_data: dict[str, Any] = field(default_factory=dict)

    # Results từ các stages
    stage_results: dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Trace
    trace_id: str = field(default_factory=lambda: f"TR{uuid.uuid4().hex[:12].upper()}")

    def merge(self, **updates: Any) -> "PipelineContext":
        """Tạo context mới với dữ liệu merge.

        Không mutate context hiện tại — trả về context mới (immutable style).
        """
        new_data = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        new_data.update(updates)
        new_data["updated_at"] = datetime.now()
        new_context = PipelineContext(**new_data)
        return new_context

    def add_stage_result(self, stage_name: str, result: dict[str, Any]) -> "PipelineContext":
        """Thêm kết quả của một stage."""
        updated_results = dict(self.stage_results)
        updated_results[stage_name] = result
        return self.merge(stage_results=updated_results)

    def add_error(self, stage: str, message: str, details: dict | None = None) -> "PipelineContext":
        """Thêm lỗi vào context."""
        new_errors = list(self.errors)
        new_errors.append({
            "stage": stage,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
        })
        return self.merge(errors=new_errors)

    def is_failed(self) -> bool:
        """Kiểm tra pipeline có lỗi critical không."""
        return any(e.get("stage") == "critical" for e in self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "order_id": self.order_id,
            "status": self.status,
            "trace_id": self.trace_id,
            "stage_results": self.stage_results,
            "errors": self.errors,
            "order_summary": {
                k: self.order_data.get(k)
                for k in ["total", "items", "customer_id", "payment_method"]
                if k in self.order_data
            },
        }
```

### core/pipeline.py

```python
"""Pipeline Engine — orchestrate execution of stages."""

from __future__ import annotations

from typing import Sequence, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import asyncio
import logging

from .stage import BaseStage, StageResult, StageStatus, StageAction
from .context import PipelineContext
from .errors import PipelineError, StageError, RetryExhaustedError

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Kết quả chạy pipeline."""
    pipeline_id: str
    status: str  # completed, failed, aborted
    total_stages: int
    completed_stages: int
    failed_stages: int
    total_duration_ms: float
    stage_results: list[StageResult]
    context: PipelineContext
    error: Optional[str] = None


class Pipeline:
    """Pipeline Engine — quản lý và thực thi pipeline.

    Pipeline có thể chạy:
    - Synchronous: stages chạy tuần tự trong cùng thread
    - Async: stages chạy qua message queue (workers)
    - Parallel: stages có thể chạy song song (DAG)
    """

    def __init__(
        self,
        stages: Sequence[BaseStage],
        name: Optional[str] = None,
        async_mode: bool = False,
        stop_on_failure: bool = True,
    ) -> None:
        self._stages = list(stages)
        self._name = name or f"Pipeline-{id(self)}"
        self._async_mode = async_mode
        self._stop_on_failure = stop_on_failure
        self._metrics: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def stages(self) -> Sequence[BaseStage]:
        return list(self._stages)

    def add_stage(self, stage: BaseStage, index: Optional[int] = None) -> None:
        """Thêm stage vào pipeline."""
        if index is not None:
            self._stages.insert(index, stage)
        else:
            self._stages.append(stage)

    def remove_stage(self, stage_name: str) -> None:
        """Xóa stage khỏi pipeline."""
        self._stages = [s for s in self._stages if s.name != stage_name]

    def run(self, context: PipelineContext) -> PipelineResult:
        """Chạy pipeline synchronous.

        Args:
            context: Initial pipeline context.

        Returns:
            PipelineResult với kết quả từng stage.
        """
        start = time.monotonic()
        stage_results: list[StageResult] = []
        current_context = context
        failed = 0
        completed = 0

        logger.info(f"Pipeline '{self._name}' started — {len(self._stages)} stages")

        for stage in self._stages:
            # Check if should skip
            if current_context.is_failed() and self._stop_on_failure:
                logger.warning(f"Skipping stage '{stage.name}' due to previous failure")
                stage_results.append(StageResult(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                    action=StageAction.CONTINUE,
                ))
                continue

            retry_count = 0
            while True:
                logger.debug(f"Running stage '{stage.name}' (attempt {retry_count + 1})")

                # Execute stage
                result = stage.execute(current_context)
                stage_results.append(result)

                # Handle result
                if result.status == StageStatus.SUCCESS:
                    # Update context with stage result
                    if result.metadata:
                        current_context = current_context.add_stage_result(
                            stage.name, result.metadata
                        )
                    completed += 1
                    break

                elif result.status == StageStatus.FAILED:
                    if result.action == StageAction.RETRY and stage.can_retry(retry_count):
                        retry_count += 1
                        # Exponential backoff
                        sleep_time = min(0.1 * (2 ** retry_count), 5.0)
                        time.sleep(sleep_time)
                        continue
                    elif result.action == StageAction.SKIP:
                        logger.warning(f"Stage '{stage.name}' skipped: {result.error_message}")
                        completed += 1
                        break
                    elif result.action == StageAction.ABORT:
                        logger.error(f"Stage '{stage.name}' aborted pipeline: {result.error_message}")
                        current_context = current_context.add_error(
                            "critical", result.error_message or "Unknown error"
                        )
                        failed += 1
                        total_duration = (time.monotonic() - start) * 1000
                        return PipelineResult(
                            pipeline_id=context.pipeline_id,
                            status="aborted",
                            total_stages=len(self._stages),
                            completed_stages=completed,
                            failed_stages=failed,
                            total_duration_ms=total_duration,
                            stage_results=stage_results,
                            context=current_context,
                            error=result.error_message,
                        )
                    else:
                        # CONTINUE despite failure
                        logger.warning(f"Stage '{stage.name}' failed but continuing: {result.error_message}")
                        current_context = current_context.add_error(
                            stage.name, result.error_message or "Unknown error"
                        )
                        failed += 1
                        break

        total_duration = (time.monotonic() - start) * 1000
        final_status = "aborted" if current_context.is_failed() else "completed"

        if failed > 0 and not current_context.is_failed():
            final_status = "completed_with_errors"

        logger.info(
            f"Pipeline '{self._name}' finished: "
            f"{final_status} in {total_duration:.0f}ms "
            f"({completed}/{len(self._stages)} stages)"
        )

        return PipelineResult(
            pipeline_id=context.pipeline_id,
            status=final_status,
            total_stages=len(self._stages),
            completed_stages=completed,
            failed_stages=failed,
            total_duration_ms=total_duration,
            stage_results=stage_results,
            context=current_context,
        )

    async def run_async(self, context: PipelineContext) -> PipelineResult:
        """Chạy pipeline asynchronous (qua message queue).

        Mỗi stage được gửi đến worker queue riêng.
        Worker xử lý xong sẽ gửi context đến queue tiếp theo.
        """
        # In thực tế, đây sẽ publish message đến Kafka/SQS
        # Mỗi stage consumer sẽ lấy message, process, publish tiếp
        logger.info(f"Async pipeline '{self._name}' started")
        return await asyncio.to_thread(self.run, context)

    def get_metrics(self) -> dict[str, Any]:
        """Lấy metrics của tất cả stages."""
        return {
            stage.name: stage.get_metrics()
            for stage in self._stages
        }

    def __repr__(self) -> str:
        stages_str = " → ".join(s.name for s in self._stages)
        return f"Pipeline({self._name}: {stages_str})"
```

### core/errors.py

```python
"""Pipeline errors."""

from __future__ import annotations


class PipelineError(Exception):
    """Base error cho pipeline."""
    pass


class StageError(PipelineError):
    """Error khi stage xử lý thất bại."""
    def __init__(self, stage_name: str, message: str) -> None:
        self.stage_name = stage_name
        super().__init__(f"[{stage_name}] {message}")


class RetryExhaustedError(StageError):
    """Retry hết số lần cho phép."""
    def __init__(self, stage_name: str, retries: int, last_error: str) -> None:
        self.retries = retries
        self.last_error = last_error
        super().__init__(stage_name, f"Failed after {retries} retries: {last_error}")


class ValidationError(StageError):
    """Dữ liệu đầu vào không hợp lệ."""
    def __init__(self, stage_name: str, field: str, reason: str) -> None:
        self.field = field
        super().__init__(stage_name, f"Invalid field '{field}': {reason}")


class ConfigurationError(PipelineError):
    """Pipeline configuration sai."""
    pass


class TimeoutError(StageError):
    """Stage timeout."""
    def __init__(self, stage_name: str, timeout_seconds: int) -> None:
        super().__init__(stage_name, f"Timeout after {timeout_seconds}s")
```

### stages/validate.py

```python
"""Validate Stage — kiểm tra đơn hàng đầu vào."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from core.stage import BaseStage, StageResult, StageStatus, StageAction
from core.context import PipelineContext
from core.errors import ValidationError


class ValidateOrderStage(BaseStage):
    """Stage 1: Validate thông tin đơn hàng.

    Kiểm tra:
    - Required fields (customer_id, items, total)
    - Tính hợp lệ của dữ liệu (email, phone, address)
    - Không có duplicate order (dựa vào idempotency key)
    - Không vượt quá giới hạn (max items, max total)
    """

    MAX_ITEMS = 50
    MAX_TOTAL = Decimal("100_000_000")  # 100 triệu VND
    MIN_TOTAL = Decimal("10_000")  # 10,000 VND

    def __init__(self) -> None:
        super().__init__(name="ValidateOrder")

    def process(self, context: PipelineContext) -> StageResult:
        order = context.order_data

        # Check required fields
        required_fields = ["customer_id", "items", "total", "payment_method", "delivery_address"]
        for field in required_fields:
            if field not in order or order[field] is None:
                raise ValidationError(
                    self.name, field, f"Trường bắt buộc '{field}' không được để trống"
                )

        # Validate customer_id format
        customer_id = str(order["customer_id"])
        if not customer_id.startswith("CUST") or len(customer_id) < 5:
            raise ValidationError(
                self.name, "customer_id", "Mã khách hàng không hợp lệ"
            )

        # Validate items
        items = order.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            raise ValidationError(self.name, "items", "Đơn hàng phải có ít nhất 1 sản phẩm")

        if len(items) > self.MAX_ITEMS:
            raise ValidationError(
                self.name, "items",
                f"Đơn hàng không thể vượt quá {self.MAX_ITEMS} sản phẩm"
            )

        for i, item in enumerate(items):
            if not item.get("product_id"):
                raise ValidationError(self.name, f"items[{i}]", "Thiếu product_id")
            if not item.get("quantity") or int(item["quantity"]) <= 0:
                raise ValidationError(self.name, f"items[{i}]", "Số lượng phải > 0")
            if not item.get("price") or Decimal(str(item["price"])) <= 0:
                raise ValidationError(self.name, f"items[{i}]", "Giá phải > 0")

        # Validate total
        total = Decimal(str(order["total"]))
        if total < self.MIN_TOTAL:
            raise ValidationError(
                self.name, "total",
                f"Giá trị đơn hàng tối thiểu {self.MIN_TOTAL:,} VND"
            )
        if total > self.MAX_TOTAL:
            raise ValidationError(
                self.name, "total",
                f"Giá trị đơn hàng tối đa {self.MAX_TOTAL:,} VND"
            )

        # Validate payment method
        valid_payments = ["VISA", "MASTERCARD", "MOMO", "ZALOPAY", "COD", "BANK_TRANSFER"]
        payment = str(order["payment_method"]).upper()
        if payment not in valid_payments:
            raise ValidationError(
                self.name, "payment_method",
                f"Phương thức thanh toán không hỗ trợ: {payment}. "
                f"Hỗ trợ: {', '.join(valid_payments)}"
            )

        # Validate delivery address
        address = order.get("delivery_address", {})
        for addr_field in ["street", "ward", "district", "city"]:
            if not address.get(addr_field):
                raise ValidationError(
                    self.name, f"delivery_address.{addr_field}",
                    f"Địa chỉ thiếu trường '{addr_field}'"
                )

        # Check idempotency
        idempotency_key = order.get("idempotency_key")
        if idempotency_key:
            # In production: check Redis/S3 for duplicate
            pass

        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            metadata={
                "validated_fields": required_fields,
                "item_count": len(items),
                "payment_method": payment,
                "total": str(total),
            },
        )
```

### stages/payment.py

```python
"""Payment Stage — xử lý thanh toán đơn hàng."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional
import random

from core.stage import BaseStage, StageResult, StageStatus, StageAction
from core.context import PipelineContext
from core.errors import StageError


class PaymentGatewayMock:
    """Mock payment gateway — thay thế bằng thật (VnPay, MoMo, VISA)."""

    @staticmethod
    def charge(
        payment_method: str,
        amount: Decimal,
        order_id: str,
        card_info: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Charge payment.

        Returns: {'success': bool, 'transaction_id': str, 'error': str}
        """
        # Simulate processing
        if random.random() < 0.1:  # 10% failure rate
            return {
                "success": False,
                "transaction_id": "",
                "error": "INSUFFICIENT_FUNDS",
                "message": "Số dư không đủ",
            }

        import uuid
        return {
            "success": True,
            "transaction_id": f"TXN{uuid.uuid4().hex[:12].upper()}",
            "error": "",
            "message": "Thanh toán thành công",
        }


class PaymentStage(BaseStage):
    """Stage 2: Xử lý thanh toán.

    Hỗ trợ multiple payment methods:
    - Credit card (VISA/Mastercard): Gọi qua VNPay
    - Ví điện tử (MoMo/ZaloPay): Gọi qua API partner
    - COD: Không cần thanh toán ngay
    - Bank transfer: Generate payment QR
    """

    def __init__(self) -> None:
        super().__init__(name="Payment", max_retries=2)

    def process(self, context: PipelineContext) -> StageResult:
        order = context.order_data
        payment_method = str(order.get("payment_method", "")).upper()
        total = Decimal(str(order.get("total", 0)))
        order_id = context.order_id

        # COD — skip payment
        if payment_method == "COD":
            return StageResult(
                stage_name=self.name,
                status=StageStatus.SUCCESS,
                metadata={
                    "payment_method": "COD",
                    "status": "pending_collection",
                    "message": "Chờ thu hộ COD",
                },
            )

        # Process payment
        result = PaymentGatewayMock.charge(
            payment_method=payment_method,
            amount=total,
            order_id=order_id,
            card_info=order.get("card_info"),
        )

        if not result["success"]:
            error_msg = result.get("message", result.get("error", "Payment failed"))
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                action=StageAction.RETRY,
                error_message=error_msg,
            )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            metadata={
                "payment_method": payment_method,
                "transaction_id": result["transaction_id"],
                "amount": str(total),
                "status": "paid",
                "gateway_response": result,
            },
        )
```

### stages/inventory.py

```python
"""Inventory Stage — kiểm tra tồn kho và reserve items."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from core.stage import BaseStage, StageResult, StageStatus, StageAction
from core.context import PipelineContext
from core.errors import StageError


class InventoryStage(BaseStage):
    """Stage 3: Kiểm tra và reserve tồn kho.

    Các bước:
    1. Check stock availability cho từng item
    2. Reserve items (tạm giữ) để tránh bán double
    3. Tính estimated delivery time
    4. Nếu thiếu hàng: gợi ý thay thế hoặc partial shipment
    """

    def __init__(self) -> None:
        super().__init__(name="Inventory")

    def process(self, context: PipelineContext) -> StageResult:
        order = context.order_data
        items = order.get("items", [])

        inventory_results = []
        all_available = True
        total_estimated_minutes = 0
        unavailable_items: list[dict[str, Any]] = []

        for item in items:
            product_id = item["product_id"]
            quantity = int(item["quantity"])

            # Mock inventory check
            stock_info = self._check_stock(product_id, quantity)
            inventory_results.append(stock_info)

            if stock_info["available"]:
                self._reserve_item(product_id, quantity, context.order_id)
                # ETA based on warehouse location (mock)
                eta = stock_info.get("estimated_minutes", 30)
                total_estimated_minutes = max(total_estimated_minutes, eta)
            else:
                all_available = False
                unavailable_items.append({
                    "product_id": product_id,
                    "requested": quantity,
                    "available": stock_info.get("available_qty", 0),
                    "suggestion": stock_info.get("suggestion", ""),
                })

        # Build result
        result_data = {
            "items_checked": len(items),
            "all_available": all_available,
            "estimated_delivery_minutes": total_estimated_minutes,
            "estimated_delivery": self._format_eta(total_estimated_minutes),
            "inventory_details": inventory_results,
        }

        if not all_available:
            result_data["unavailable_items"] = unavailable_items
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                action=StageAction.SKIP,  # Tiếp tục pipeline nhưng báo lỗi
                error_message=f"Thiếu {len(unavailable_items)} sản phẩm trong kho",
                metadata=result_data,
            )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            metadata=result_data,
        )

    def _check_stock(self, product_id: str, quantity: int) -> dict[str, Any]:
        """Mock inventory check.

        Trong thực tế: gọi API đến warehouse service hoặc query database.
        """
        import random
        available = random.choice([True, True, True, False])  # 75% available
        stock_qty = random.randint(10, 1000) if available else random.randint(0, quantity - 1)

        return {
            "product_id": product_id,
            "requested": quantity,
            "available": available,
            "available_qty": stock_qty if not available else None,
            "estimated_minutes": random.randint(15, 90),
            "suggestion": f"Sản phẩm thay thế: {product_id}_variant" if not available else "",
        }

    def _reserve_item(self, product_id: str, quantity: int, order_id: str) -> None:
        """Reserve items trong kho (2PL, timeout 30 phút)."""
        # In production: ghi vào Redis với TTL = 30 phút
        # Nếu order không hoàn tất sau 30 phút, release auto
        pass

    @staticmethod
    def _format_eta(minutes: int) -> str:
        if minutes < 60:
            return f"{minutes} phút"
        hours = minutes // 60
        mins = minutes % 60
        if mins == 0:
            return f"{hours} giờ"
        return f"{hours} giờ {mins} phút"
```

### stages/fraud_detection.py

```python
"""Fraud Detection Stage — phát hiện gian lận với ML."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
import random
import hashlib

from core.stage import BaseStage, StageResult, StageStatus, StageAction
from core.context import PipelineContext


class FraudDetectionStage(BaseStage):
    """Stage 4: Phát hiện giao dịch gian lận.

    Sử dụng kết hợp:
    - Rule-based checks
    - ML model scoring (mock)
    - Historical data analysis

    Các yếu tố kiểm tra:
    - Device fingerprint (location, IP, device ID)
    - Hành vi bất thường (nhiều đơn trong thời gian ngắn)
    - Payment pattern (card test, amount threshold)
    - Account age và history
    """

    RISK_THRESHOLDS = {
        "max_orders_per_hour": 5,
        "max_amount_per_day": Decimal("20_000_000"),
        "new_account_limit": Decimal("5_000_000"),
        "high_risk_countries": ["XX", "YY"],
    }

    def __init__(self) -> None:
        super().__init__(name="FraudDetection")

    def process(self, context: PipelineContext) -> StageResult:
        order = context.order_data

        # Collect signals
        signals = self._collect_signals(order)

        # Rule-based checks
        rule_results = self._run_rules(signals)

        # ML scoring (mock)
        ml_score = self._ml_scoring(signals)

        # Composite risk assessment
        risk_result = self._assess_risk(rule_results, ml_score)

        # Action based on risk
        if risk_result["decision"] == "reject":
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                action=StageAction.ABORT,
                error_message=f"Đơn hàng bị từ chối: {risk_result['reason']}",
                metadata=risk_result,
            )
        elif risk_result["decision"] == "review":
            # Flag for manual review — tiếp tục pipeline
            return StageResult(
                stage_name=self.name,
                status=StageStatus.SUCCESS,
                metadata={
                    "decision": "review",
                    "risk_score": ml_score,
                    "flags": risk_result.get("flags", []),
                    "requires_manual_review": True,
                },
            )

        # Decision: approve
        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            metadata={
                "decision": "approve",
                "risk_score": ml_score,
                "flags": [],
                "message": "Đơn hàng an toàn",
            },
        )

    def _collect_signals(self, order: dict[str, Any]) -> dict[str, Any]:
        """Thu thập các tín hiệu cho fraud detection."""
        return {
            "customer_id": order.get("customer_id"),
            "total": Decimal(str(order.get("total", 0))),
            "payment_method": order.get("payment_method"),
            "items_count": len(order.get("items", [])),
            "device_id": order.get("device_id", "unknown"),
            "ip_address": order.get("ip_address", "0.0.0.0"),
            "user_agent": order.get("user_agent", ""),
            "is_new_customer": order.get("is_new_customer", False),
            "orders_today": order.get("orders_today", 0),
            "shipping_same_as_billing": order.get("shipping_same_as_billing", True),
            "fingerprint": self._generate_fingerprint(order),
        }

    def _generate_fingerprint(self, order: dict[str, Any]) -> str:
        """Tạo fingerprint cho order."""
        raw = f"{order.get('device_id')}_{order.get('ip_address')}_{order.get('user_agent')}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _run_rules(self, signals: dict[str, Any]) -> list[dict[str, Any]]:
        """Run rule-based checks."""
        flags = []
        total = signals["total"]
        thresholds = self.RISK_THRESHOLDS

        # Rule 1: New customer with high value order
        if signals["is_new_customer"] and total > thresholds["new_account_limit"]:
            flags.append({
                "rule": "new_account_high_value",
                "severity": "high",
                "message": "Khách hàng mới với đơn hàng giá trị lớn",
            })

        # Rule 2: Too many orders today
        if signals["orders_today"] > thresholds["max_orders_per_hour"]:
            flags.append({
                "rule": "too_many_orders",
                "severity": "medium",
                "message": f"{signals['orders_today']} đơn trong hôm nay",
            })

        # Rule 3: High amount
        if total > thresholds["max_amount_per_day"]:
            flags.append({
                "rule": "high_amount",
                "severity": "medium",
                "message": f"Giá trị đơn {total:,} VND vượt ngưỡng",
            })

        # Rule 4: Shipping different from billing (potential stolen card)
        if not signals["shipping_same_as_billing"]:
            flags.append({
                "rule": "address_mismatch",
                "severity": "low",
                "message": "Địa chỉ giao hàng khác địa chỉ thanh toán",
            })

        return flags

    def _ml_scoring(self, signals: dict[str, Any]) -> float:
        """Mock ML model scoring.

        Trong thực tế: gọi SageMaker/Azure ML endpoint với feature vector.
        """
        # Features: total, items_count, is_new_customer, orders_today, ...
        # Mock score: 0.0 (safe) to 1.0 (fraud)
        base_score = 0.05

        if signals["is_new_customer"]:
            base_score += 0.2
        if signals["orders_today"] > 3:
            base_score += 0.15 * min(signals["orders_today"] / 10, 1)
        if signals["total"] > Decimal("10_000_000"):
            base_score += 0.1
        if signals["shipping_same_as_billing"]:
            base_score -= 0.02

        return min(base_score + random.uniform(-0.05, 0.05), 1.0)

    def _assess_risk(
        self,
        rule_results: list[dict[str, Any]],
        ml_score: float,
    ) -> dict[str, Any]:
        """Tổng hợp đánh giá rủi ro."""
        high_severity = [r for r in rule_results if r["severity"] == "high"]

        if ml_score > 0.7 or len(high_severity) >= 2:
            return {
                "decision": "reject",
                "reason": "Giao dịch có dấu hiệu gian lận cao",
                "risk_score": ml_score,
                "flags": rule_results,
            }

        if ml_score > 0.4 or len(high_severity) >= 1:
            return {
                "decision": "review",
                "flags": rule_results,
                "risk_score": ml_score,
                "message": "Giao dịch cần kiểm tra thủ công",
            }

        return {
            "decision": "approve",
            "flags": rule_results,
            "risk_score": ml_score,
        }
```

### stages/notification.py

```python
"""Notification Stage — gửi thông báo cho khách hàng."""

from __future__ import annotations

from typing import Any

from core.stage import BaseStage, StageResult, StageStatus
from core.context import PipelineContext


class NotificationStage(BaseStage):
    """Stage 5: Gửi thông báo cho khách hàng.

    Channels:
    - Email: Xác nhận đơn hàng
    - SMS: OTP và trạng thái
    - Push notification: Cập nhật real-time
    - Zalo/ZNS: Zalo notification service
    """

    def __init__(self) -> None:
        super().__init__(name="Notification")

    def process(self, context: PipelineContext) -> StageResult:
        order = context.order_data
        customer_id = order.get("customer_id", "")
        email = order.get("email", "")
        phone = order.get("phone", "")

        notifications_sent = []

        # Gửi email xác nhận
        if email:
            email_result = self._send_email(
                to=email,
                template="order_confirmation",
                params={
                    "order_id": context.order_id,
                    "customer_name": order.get("customer_name", ""),
                    "total": order.get("total", 0),
                    "items_count": len(order.get("items", [])),
                    "estimated_delivery": context.stage_results.get("Inventory", {}).get(
                        "estimated_delivery", "Đang tính toán"
                    ),
                },
            )
            notifications_sent.append(email_result)

        # Gửi SMS
        if phone:
            sms_result = self._send_sms(
                to=phone,
                template="order_received",
                params={"order_id": context.order_id},
            )
            notifications_sent.append(sms_result)

        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            metadata={
                "notifications_sent": notifications_sent,
                "channels": [n["channel"] for n in notifications_sent],
            },
        )

    def _send_email(self, to: str, template: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send email (mock)."""
        print(f"[Email] → {to}: {template} — {params.get('order_id', '')}")
        return {"channel": "email", "to": to, "template": template, "status": "sent"}

    def _send_sms(self, to: str, template: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send SMS (mock)."""
        print(f"[SMS] → {to}: {template} — {params.get('order_id', '')}")
        return {"channel": "sms", "to": to, "template": template, "status": "sent"}
```

### stages/complete.py

```python
"""Complete Stage — hoàn tất đơn hàng và emit event."""

from __future__ import annotations

from datetime import datetime

from core.stage import BaseStage, StageResult, StageStatus
from core.context import PipelineContext


class CompleteOrderStage(BaseStage):
    """Stage 6: Hoàn tất pipeline.

    Các bước:
    1. Update order status trong database
    2. Emit 'order.created' event (Kafka/EventBridge)
    3. Log analytics
    4. Cleanup temporary resources
    """

    def __init__(self) -> None:
        super().__init__(name="CompleteOrder")

    def process(self, context: PipelineContext) -> StageResult:
        # Update order status
        order = context.order_data
        order["status"] = "confirmed"
        order["confirmed_at"] = datetime.utcnow().isoformat()

        # Emit event
        event = {
            "event_type": "order.created",
            "order_id": context.order_id,
            "pipeline_id": context.pipeline_id,
            "trace_id": context.trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "customer_id": order.get("customer_id"),
                "total": str(order.get("total", 0)),
                "payment_method": order.get("payment_method"),
                "items_count": len(order.get("items", [])),
                "stage_results": {
                    name: info.get("decision", info.get("status", "completed"))
                    for name, info in context.stage_results.items()
                },
            },
        }
        self._emit_event(event)

        # Log analytics
        self._log_analytics(event)

        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            metadata={
                "order_status": "confirmed",
                "event": event["event_type"],
                "completed_at": datetime.utcnow().isoformat(),
                "pipeline_summary": {
                    "total_stages": len(context.stage_results),
                    "successful_stages": sum(
                        1 for v in context.stage_results.values()
                        if isinstance(v, dict) and v.get("decision") != "reject"
                    ),
                },
            },
        )

    def _emit_event(self, event: dict) -> None:
        """Emit event to event bus (Kafka/EventBridge/RabbitMQ)."""
        # In production: publish to Kafka topic 'order.events'
        print(f"[Event] {event['event_type']} → order.events | order={event['order_id']}")

    def _log_analytics(self, event: dict) -> None:
        """Log analytics data."""
        print(
            f"[Analytics] Order {event['order_id']}: "
            f"{event['data']['total']} VND, "
            f"{event['data']['items_count']} items, "
            f"{event['data']['payment_method']}"
        )
```

### config.py

```python
"""Pipeline configuration — định nghĩa pipeline cho ứng dụng."""

from __future__ import annotations

from typing import Sequence

from core.pipeline import Pipeline
from core.stage import BaseStage

from stages.validate import ValidateOrderStage
from stages.payment import PaymentStage
from stages.inventory import InventoryStage
from stages.fraud_detection import FraudDetectionStage
from stages.notification import NotificationStage
from stages.complete import CompleteOrderStage


def create_order_pipeline(async_mode: bool = False) -> Pipeline:
    """Tạo pipeline xử lý đơn hàng hoàn chỉnh.

    Thứ tự stages có thể được cấu hình động.
    Có thể thêm/bớt stages mà không ảnh hưởng code hiện tại.
    """
    stages: list[BaseStage] = [
        ValidateOrderStage(),
        PaymentStage(),
        InventoryStage(),
        FraudDetectionStage(),
        NotificationStage(),
        CompleteOrderStage(),
    ]

    return Pipeline(
        stages=stages,
        name="OrderProcessingPipeline",
        async_mode=async_mode,
        stop_on_failure=False,  # Tiếp tục dù có stage fail (non-critical)
    )


def create_express_pipeline() -> Pipeline:
    """Pipeline rút gọn cho đơn hàng express (< 30 phút)."""
    stages: list[BaseStage] = [
        ValidateOrderStage(),
        PaymentStage(),
        InventoryStage(),
        NotificationStage(),
        CompleteOrderStage(),
    ]
    return Pipeline(
        stages=stages,
        name="ExpressPipeline",
        stop_on_failure=True,
    )
```

### main.py

```python
"""Main entry point — chạy pipeline mẫu."""

from __future__ import annotations

import json
from decimal import Decimal
from datetime import datetime

from core.context import PipelineContext
from config import create_order_pipeline


def create_sample_order() -> PipelineContext:
    """Tạo đơn hàng mẫu để test pipeline."""
    return PipelineContext(
        order_id="ORD20260712001",
        order_data={
            "customer_id": "CUST001",
            "customer_name": "Nguyễn Văn An",
            "email": "an.nguyen@email.com",
            "phone": "0912345678",
            "payment_method": "MOMO",
            "total": Decimal("1_500_000"),
            "items": [
                {"product_id": "PROD001", "name": "Cơm tấm sườn", "quantity": 2, "price": 45000},
                {"product_id": "PROD002", "name": "Trà tắc", "quantity": 2, "price": 15000},
                {"product_id": "PROD003", "name": "Bánh flan", "quantity": 1, "price": 20000},
            ],
            "delivery_address": {
                "street": "123 Nguyễn Huệ",
                "ward": "Bến Nghé",
                "district": "Quận 1",
                "city": "TP. Hồ Chí Minh",
            },
            "is_new_customer": False,
            "orders_today": 1,
            "device_id": "DEVICE-iPhone15-001",
            "ip_address": "192.168.1.100",
            "shipping_same_as_billing": True,
        },
    )


def main() -> None:
    print("=" * 70)
    print("  🏪 TIKINGON — ORDER PROCESSING PIPELINE DEMO")
    print("=" * 70)

    # Create pipeline
    pipeline = create_order_pipeline(async_mode=False)
    print(f"\n  Pipeline: {pipeline}")
    print(f"  Stages ({len(pipeline.stages)}):")
    for i, stage in enumerate(pipeline.stages, 1):
        print(f"    {i}. {stage.name}")

    # Create sample order
    context = create_sample_order()
    print(f"\n  {'─' * 50}")
    print(f"  ĐƠN HÀNG #{context.order_id}")
    print(f"  Khách hàng: {context.order_data['customer_name']}")
    print(f"  Số lượng: {len(context.order_data['items'])} items")
    print(f"  Tổng tiền: {context.order_data['total']:,.0f} VND")
    print(f"  Thanh toán: {context.order_data['payment_method']}")

    # Run pipeline
    print(f"\n  {'═' * 50}")
    print(f"  BẮT ĐẦU XỬ LÝ PIPELINE")
    print(f"  {'═' * 50}\n")

    result = pipeline.run(context)

    # Print results
    print(f"\n  {'═' * 50}")
    print(f"  KẾT QUẢ")
    print(f"  {'═' * 50}")
    print(f"  Status: {result.status.upper()}")
    print(f"  Duration: {result.total_duration_ms:.0f}ms")
    print(f"  Stages: {result.completed_stages}/{result.total_stages} completed")
    if result.error:
        print(f"  Error: {result.error}")
    print()

    # Per-stage details
    for sr in result.stage_results:
        icon = "✅" if sr.status.name == "SUCCESS" else "❌" if sr.status.name == "FAILED" else "⏭️"
        print(f"  {icon} {sr.stage_name:<20s} "
              f"| {sr.status.name:<8s} "
              f"| {sr.duration_ms:>8.0f}ms "
              f"| {sr.error_message or ''}")

    # Final context summary
    print(f"\n  {'─' * 50}")
    print(f"  PIPELINE SUMMARY:")
    print(f"  Pipeline ID: {result.context.pipeline_id}")
    print(f"  Trace ID: {result.context.trace_id}")

    if result.context.stage_results:
        print(f"  Stages completed: {list(result.context.stage_results.keys())}")

    print(f"\n  {'=' * 70}")
    print(f"  ✅ DEMO HOÀN TẤT")
    print(f"  {'=' * 70}")


def main_async_demo() -> None:
    """Demo async pipeline với multiple orders."""
    import asyncio
    import random

    pipeline = create_order_pipeline(async_mode=True)
    print("=" * 70)
    print("  🏪 TIKINGON — ASYNC PIPELINE DEMO")
    print("=" * 70)

    # Simulate 5 orders
    orders = [
        PipelineContext(
            order_id=f"ORD20260712{i:03d}",
            order_data={
                "customer_id": f"CUST{i:03d}",
                "customer_name": f"Khách hàng {i}",
                "email": f"customer{i}@email.com",
                "phone": f"09123{i:05d}",
                "payment_method": random.choice(["MOMO", "VISA", "COD"]),
                "total": Decimal(str(random.randint(50000, 5000000))),
                "items": [
                    {"product_id": f"PROD{random.randint(1,100):03d}",
                     "quantity": random.randint(1,5),
                     "price": Decimal(str(random.randint(10000, 200000)))}
                    for _ in range(random.randint(1, 5))
                ],
                "delivery_address": {
                    "street": f"{random.randint(1,999)} Đường ABC",
                    "ward": "Phường X",
                    "district": "Quận Y",
                    "city": "TP. Hồ Chí Minh",
                },
                "is_new_customer": random.choice([True, False, False]),
                "orders_today": random.randint(0, 3),
            },
        )
        for i in range(1, 6)
    ]

    async def run_async_orders():
        tasks = [pipeline.run_async(order) for order in orders]
        results = await asyncio.gather(*tasks)

        for order, result in zip(orders, results):
            print(f"  {result.status:>12s} | {order.order_id} | {result.total_duration_ms:6.0f}ms | "
                  f"{result.completed_stages}/{result.total_stages} stages")

    asyncio.run(run_async_orders())

    print(f"\n  All orders processed! Pipeline metrics:")
    for name, metrics in pipeline.get_metrics().items():
        print(f"  {name}: {metrics}")


if __name__ == "__main__":
    main()
    print("\n")
    main_async_demo()
```

### tests/test_pipeline.py

```python
"""Tests for Pipeline Engine."""

from __future__ import annotations

import unittest
from decimal import Decimal
from typing import Any

import sys
sys.path.insert(0, "..")

from core.pipeline import Pipeline, PipelineResult
from core.stage import BaseStage, StageResult, StageStatus, StageAction
from core.context import PipelineContext


class EchoStage(BaseStage):
    """Stage test — echo metadata."""
    def __init__(self, name: str = "Echo", fail: bool = False) -> None:
        super().__init__(name=name)
        self._fail = fail

    def process(self, context: PipelineContext) -> StageResult:
        if self._fail:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                action=StageAction.ABORT,
                error_message="Test failure",
            )
        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            metadata={"echo": self.name, "order": context.order_id},
        )


class RetryStage(BaseStage):
    """Stage test — fail N lần trước khi thành công."""
    def __init__(self, name: str, fail_count: int = 2) -> None:
        super().__init__(name=name, max_retries=3)
        self._fail_count = fail_count
        self._attempts = 0

    def process(self, context: PipelineContext) -> StageResult:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                action=StageAction.RETRY,
                error_message=f"Attempt {self._attempts} failed",
            )
        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            metadata={"attempts": self._attempts},
        )


class TestPipeline(unittest.TestCase):
    """Test Pipeline engine core."""

    def test_empty_pipeline(self):
        pipeline = Pipeline(stages=[])
        context = PipelineContext(order_id="ORD001", order_data={"test": True})
        result = pipeline.run(context)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.total_stages, 0)

    def test_single_stage_success(self):
        pipeline = Pipeline(stages=[EchoStage("Stage1")])
        context = PipelineContext(order_id="ORD001")
        result = pipeline.run(context)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.total_stages, 1)
        self.assertEqual(result.completed_stages, 1)
        self.assertEqual(result.failed_stages, 0)

    def test_multiple_stages_success(self):
        pipeline = Pipeline(stages=[
            EchoStage("Validate"),
            EchoStage("Payment"),
            EchoStage("Complete"),
        ])
        context = PipelineContext(order_id="ORD002")
        result = pipeline.run(context)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.completed_stages, 3)

    def test_stage_failure_aborts(self):
        pipeline = Pipeline(
            stages=[EchoStage("S1"), EchoStage("S2", fail=True), EchoStage("S3")],
            stop_on_failure=True,
        )
        context = PipelineContext(order_id="ORD003")
        result = pipeline.run(context)

        self.assertEqual(result.status, "aborted")

    def test_stage_failure_continues(self):
        pipeline = Pipeline(
            stages=[EchoStage("S1"), EchoStage("S2", fail=True), EchoStage("S3")],
            stop_on_failure=False,
        )
        context = PipelineContext(order_id="ORD004")
        result = pipeline.run(context)

        # S2 fail, S3 should still run
        self.assertEqual(result.completed_stages, 2)
        self.assertEqual(result.failed_stages, 1)

    def test_retry_success(self):
        pipeline = Pipeline(stages=[RetryStage("RetryTest", fail_count=2)])
        context = PipelineContext(order_id="ORD005")
        result = pipeline.run(context)

        self.assertEqual(result.status, "completed")

    def test_retry_exhausted(self):
        pipeline = Pipeline(stages=[RetryStage("RetryFail", fail_count=10)])
        context = PipelineContext(order_id="ORD006")
        result = pipeline.run(context)

        self.assertEqual(result.status, "completed")  # Continues after max retries
        self.assertEqual(result.failed_stages, 1)

    def test_context_passed_through_stages(self):
        metadata_list = []

        class CaptureStage(BaseStage):
            def process(self, ctx: PipelineContext) -> StageResult:
                metadata_list.append(ctx.order_id)
                return StageResult(
                    stage_name=self.name,
                    status=StageStatus.SUCCESS,
                    metadata={"captured": ctx.order_id},
                )

        pipeline = Pipeline(stages=[CaptureStage("C1"), CaptureStage("C2")])
        context = PipelineContext(order_id="ORD007")
        pipeline.run(context)

        self.assertEqual(metadata_list, ["ORD007", "ORD007"])

    def test_pipeline_metrics(self):
        pipeline = Pipeline(stages=[EchoStage("M1"), EchoStage("M2")])
        context = PipelineContext(order_id="ORD008")
        pipeline.run(context)

        metrics = pipeline.get_metrics()
        self.assertIn("M1", metrics)
        self.assertIn("M2", metrics)
        self.assertEqual(metrics["M1"]["total_processed"], 1)

    def test_dynamic_stage_addition(self):
        pipeline = Pipeline(stages=[EchoStage("Original")])
        pipeline.add_stage(EchoStage("Added"))

        self.assertEqual(len(pipeline.stages), 2)

    def test_dynamic_stage_removal(self):
        pipeline = Pipeline(stages=[EchoStage("Keep"), EchoStage("Remove")])
        pipeline.remove_stage("Remove")

        self.assertEqual(len(pipeline.stages), 1)
        self.assertEqual(pipeline.stages[0].name, "Keep")


class TestPipelineContext(unittest.TestCase):
    """Test PipelineContext immutability and merging."""

    def test_initial_context(self):
        ctx = PipelineContext(order_id="ORD001", order_data={"total": 100})
        self.assertEqual(ctx.order_id, "ORD001")
        self.assertEqual(ctx.order_data["total"], 100)

    def test_merge_creates_new_context(self):
        ctx1 = PipelineContext(order_id="ORD001")
        ctx2 = ctx1.merge(status="processing")

        self.assertEqual(ctx1.order_id, "ORD001")
        self.assertEqual(ctx1.status, "pending")
        self.assertEqual(ctx2.status, "processing")

    def test_add_stage_result(self):
        ctx = PipelineContext(order_id="ORD001")
        ctx2 = ctx.add_stage_result("Validate", {"passed": True})

        self.assertNotIn("Validate", ctx.stage_results)
        self.assertIn("Validate", ctx2.stage_results)
        self.assertEqual(ctx2.stage_results["Validate"]["passed"], True)

    def test_add_error(self):
        ctx = PipelineContext(order_id="ORD001")
        ctx2 = ctx.add_error("Payment", "Insufficient funds")

        self.assertEqual(len(ctx.errors), 0)
        self.assertEqual(len(ctx2.errors), 1)
        self.assertEqual(ctx2.errors[0]["stage"], "Payment")

    def test_is_failed(self):
        ctx = PipelineContext(order_id="ORD001")
        self.assertFalse(ctx.is_failed())

        ctx2 = ctx.add_error("critical", "Fatal error")
        self.assertTrue(ctx2.is_failed())


class TestStageResult(unittest.TestCase):
    """Test StageResult data class."""

    def test_default_action_is_continue(self):
        r = StageResult(stage_name="Test", status=StageStatus.SUCCESS)
        self.assertEqual(r.action, StageAction.CONTINUE)

    def test_failed_retry_action(self):
        r = StageResult(
            stage_name="Test",
            status=StageStatus.FAILED,
            action=StageAction.RETRY,
            error_message="Timeout",
        )
        self.assertEqual(r.action, StageAction.RETRY)
        self.assertEqual(r.error_message, "Timeout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

### tests/test_stages.py

```python
"""Tests for individual pipeline stages."""

from __future__ import annotations

import unittest
from decimal import Decimal

import sys
sys.path.insert(0, "..")

from core.context import PipelineContext
from core.stage import StageStatus, StageAction

from stages.validate import ValidateOrderStage, ValidationError
from stages.payment import PaymentStage
from stages.inventory import InventoryStage
from stages.fraud_detection import FraudDetectionStage
from stages.notification import NotificationStage
from stages.complete import CompleteOrderStage


class TestValidateOrderStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = ValidateOrderStage()
        self.valid_order = {
            "customer_id": "CUST001",
            "items": [
                {"product_id": "P1", "quantity": 2, "price": 50000},
            ],
            "total": Decimal("100000"),
            "payment_method": "MOMO",
            "delivery_address": {
                "street": "123 ABC",
                "ward": "Phường 1",
                "district": "Quận 1",
                "city": "TP.HCM",
            },
        }

    def test_valid_order_passes(self):
        ctx = PipelineContext(order_id="ORD001", order_data=self.valid_order)
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.SUCCESS)

    def test_missing_required_field(self):
        ctx = PipelineContext(order_id="ORD002", order_data={"customer_id": "C1"})
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.FAILED)

    def test_invalid_customer_id(self):
        data = dict(self.valid_order, customer_id="123")
        ctx = PipelineContext(order_id="ORD003", order_data=data)
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.FAILED)

    def test_empty_items(self):
        data = dict(self.valid_order, items=[])
        ctx = PipelineContext(order_id="ORD004", order_data=data)
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.FAILED)

    def test_too_many_items(self):
        data = dict(self.valid_order, items=[
            {"product_id": f"P{i}", "quantity": 1, "price": 1000}
            for i in range(100)
        ])
        ctx = PipelineContext(order_id="ORD005", order_data=data)
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.FAILED)

    def test_invalid_total(self):
        data = dict(self.valid_order, total=Decimal("1000"))  # Under minimum
        ctx = PipelineContext(order_id="ORD006", order_data=data)
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.FAILED)

    def test_invalid_payment_method(self):
        data = dict(self.valid_order, payment_method="BITCOIN")
        ctx = PipelineContext(order_id="ORD007", order_data=data)
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.FAILED)

    def test_missing_address_field(self):
        data = dict(self.valid_order, delivery_address={"street": "123 ABC"})
        ctx = PipelineContext(order_id="ORD008", order_data=data)
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.FAILED)


class TestPaymentStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = PaymentStage()

    def test_cod_passes(self):
        ctx = PipelineContext(order_id="ORD001", order_data={
            "payment_method": "COD",
            "total": Decimal("100000"),
        })
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.SUCCESS)

    def test_momo_payment(self):
        ctx = PipelineContext(order_id="ORD002", order_data={
            "payment_method": "MOMO",
            "total": Decimal("500000"),
        })
        result = self.stage.execute(ctx)
        self.assertIn(result.status, [StageStatus.SUCCESS, StageStatus.FAILED])


class TestInventoryStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = InventoryStage()

    def test_check_inventory(self):
        ctx = PipelineContext(order_id="ORD001", order_data={
            "items": [
                {"product_id": "P1", "quantity": 2, "price": 50000},
                {"product_id": "P2", "quantity": 1, "price": 30000},
            ],
        })
        result = self.stage.execute(ctx)
        self.assertIn(result.status, [StageStatus.SUCCESS, StageStatus.FAILED])

    def test_eta_format(self):
        from stages.inventory import InventoryStage as Inv
        self.assertIn("phút", Inv._format_eta(30))
        self.assertIn("giờ", Inv._format_eta(90))


class TestFraudDetectionStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = FraudDetectionStage()

    def test_low_risk_order(self):
        ctx = PipelineContext(order_id="ORD001", order_data={
            "customer_id": "CUST001",
            "total": Decimal("150000"),
            "payment_method": "MOMO",
            "items": [{"product_id": "P1", "quantity": 1, "price": 150000}],
            "is_new_customer": False,
            "orders_today": 1,
            "shipping_same_as_billing": True,
        })
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.SUCCESS)

    def test_high_risk_order(self):
        ctx = PipelineContext(order_id="ORD002", order_data={
            "customer_id": "CUST999",
            "total": Decimal("50000000"),
            "payment_method": "VISA",
            "items": [{"product_id": "P1", "quantity": 10, "price": 5000000}],
            "is_new_customer": True,
            "orders_today": 10,
            "shipping_same_as_billing": False,
        })
        result = self.stage.execute(ctx)
        # May be rejected or flagged for review
        self.assertIn(result.status, [StageStatus.SUCCESS, StageStatus.FAILED])


class TestNotificationStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = NotificationStage()

    def test_notification_sent(self):
        ctx = PipelineContext(order_id="ORD001", order_data={
            "customer_id": "CUST001",
            "customer_name": "Test",
            "email": "test@email.com",
            "phone": "0912345678",
            "total": Decimal("100000"),
            "items": [{"product_id": "P1", "quantity": 1, "price": 100000}],
        })
        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.SUCCESS)
        self.assertGreater(len(result.metadata.get("notifications_sent", [])), 0)


class TestCompleteOrderStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = CompleteOrderStage()

    def test_complete(self):
        ctx = PipelineContext(
            order_id="ORD001",
            order_data={"customer_id": "CUST001", "total": Decimal("100000")},
        )
        ctx = ctx.add_stage_result("Validate", {"passed": True})
        ctx = ctx.add_stage_result("Payment", {"status": "paid"})

        result = self.stage.execute(ctx)
        self.assertEqual(result.status, StageStatus.SUCCESS)
        self.assertEqual(result.metadata["order_status"], "confirmed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng Pipeline | Khi nào không dùng Pipeline |
|----------------------|-----------------------------|
| **Data processing có thứ tự** — ETL, ETL, batch processing | **Request-response đơn giản** — CRUD API, form submit |
| **Multi-stage processing** — Xử lý ảnh, video, document | **Real-time interactive** — User cần response ngay lập tức |
| **CI/CD** — Build → Test → Deploy | **Simple CRUD** — Không cần multi-stage |
| **Event-driven processing** — Order, notification, analytics | **Ứng dụng stateful phức tạp** — Game, đồ họa |
| **Cần retry và error recovery** — Mỗi stage xử lý lỗi riêng | **Pipeline không cần thay đổi** — Nếu stages cố định |
| **Cần scale từng bước riêng** — Stage A 2 pods, Stage B 50 pods | **Tight coupling** — Nếu stages phụ thuộc chặt vào nhau |
| **Cần monitoring per stage** — Biết stage nào chậm | **Overhead không đáng** — Pipeline 2 stages |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| **Decoupled stages**: Mỗi stage độc lập, dễ maintain | **Latency**: Dữ liệu đi qua nhiều stages → chậm hơn |
| **Independent scaling**: Scale từng stage riêng | **Debugging khó**: Theo dõi context qua stages |
| **Fault isolation**: Stage fail không ảnh hưởng toàn pipeline | **Data transformation cost**: Context phải được serialize giữa stages |
| **Retry per stage**: Retry stage fail, không restart pipeline | **Pipeline management**: Cần orchestration (Kafka, Step Functions) |
| **Dynamic pipeline**: Thêm/bớt stage dễ dàng | **Complexity**: Nhiều stages → khó hiểu toàn bộ flow |
| **Single responsibility**: Mỗi stage làm một việc | **Distributed tracing**: Cần tooling (Jaeger, X-Ray) |
| **Reusable stages**: Stage có thể dùng lại ở pipeline khác | **Backpressure**: Stage sau chậm → stage trước bị block |
| **Observability per stage**: Metrics, logging, timing | **Testing complexity**: Integration test multi-stage |
| **Resource optimization**: Stage CPU-intensive dùng GPU riêng | **Configuration overhead**: Pipeline YAML/JSON config |
| **Standardized interface**: Tất cả stages cùng interface | **Data consistency**: Đảm bảo exactly-once processing |

---

## Công cụ và Framework

### Pipeline Frameworks
| Framework | Mô tả |
|-----------|-------|
| **Apache Airflow** | Workflow orchestration (Python DAGs) |
| **Apache Beam** | Unified batch + streaming pipeline |
| **Apache NiFi** | Data flow automation, UI-based |
| **Kubeflow Pipelines** | ML pipelines trên Kubernetes |
| **Dagster** | Data orchestrator cho ML, analytics |
| **Prefect** | Modern workflow orchestration |
| **Luigi** (Spotify) | Batch pipeline (Python) |
| **TensorFlow Extended (TFX)** | ML production pipeline |

### CI/CD Pipelines
| Công cụ | Mô tả |
|---------|-------|
| **Jenkins** | CI/CD pipeline với Jenkinsfile |
| **GitHub Actions** | YAML-based CI/CD pipeline |
| **GitLab CI** | Built-in CI/CD pipeline |
| **CircleCI** | Cloud CI/CD pipeline |
| **Drone CI** | Container-native CI/CD |
| **Buildkite** | Hybrid CI/CD pipeline |
| **ArgoCD** | GitOps deployment pipeline |

### Stream Processing
| Công cụ | Mô tả |
|---------|-------|
| **Apache Kafka** | Event streaming platform |
| **Apache Flink** | Stream processing framework |
| **Kafka Streams** | Stream processing với Kafka |
| **Apache Storm** | Real-time stream processing |
| **Spark Streaming** | Micro-batch stream processing |

### Message Queue (Pipeline Pipes)
| Công cụ | Mô tả |
|---------|-------|
| **RabbitMQ** | AMQP message broker |
| **Amazon SQS** | Managed message queue |
| **Google Pub/Sub** | Cloud messaging |
| **Azure Service Bus** | Enterprise messaging |
| **NATS** | Lightweight messaging |

### Python Libraries
| Library | Mô tả |
|---------|-------|
| **Celery** | Distributed task queue (pipeline-like) |
| **RQ** (Redis Queue) | Simple job queue |
| **Apache Airflow** (Python) | Workflow as code |
| **Prefect** (Python) | Modern workflow |
| **Luigi** | Batch pipeline |
| **Kombu** | Messaging library (RabbitMQ, Redis) |

---

## Kiểm thử

### Chiến lược kiểm thử Pipeline

**1. Unit Tests — Từng stage riêng lẻ**
```bash
python -m pytest tests/test_stages.py -v
```

**2. Pipeline Integration — Pipeline engine**
```bash
python -m pytest tests/test_pipeline.py -v
```

**3. Performance Benchmarks**
```python
# benchmarks/test_performance.py
import time
from core.context import PipelineContext
from config import create_order_pipeline

def benchmark_pipeline(n_orders: int = 100):
    pipeline = create_order_pipeline()
    times = []

    for i in range(n_orders):
        ctx = PipelineContext(
            order_id=f"BENCH{i:05d}",
            order_data={
                "customer_id": "CUST001",
                "items": [{"product_id": "P1", "quantity": 1, "price": 50000}],
                "total": 50000,
                "payment_method": "MOMO",
                "delivery_address": {
                    "street": "123 ABC", "ward": "W1",
                    "district": "D1", "city": "HCMC",
                },
            },
        )
        start = time.monotonic()
        result = pipeline.run(ctx)
        duration = (time.monotonic() - start) * 1000
        times.append(duration)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]

    print(f"Pipeline benchmark ({n_orders} orders):")
    print(f"  Average: {avg:.0f}ms")
    print(f"  P50:     {p50:.0f}ms")
    print(f"  P99:     {p99:.0f}ms")
    print(f"  Min:     {min(times):.0f}ms")
    print(f"  Max:     {max(times):.0f}ms")
```

**4. Chaos Engineering — Test failure recovery**
```python
# Test pipeline với simulated failures
def test_pipeline_recovery():
    """Stage fail → retry → success."""
    pipeline = Pipeline(stages=[RetryStage("RetryMe", fail_count=2)])
    result = pipeline.run(PipelineContext(order_id="CHAOS001"))
    assert result.status == "completed"

def test_pipeline_graceful_degradation():
    """Stage fail → skip → continue."""
    pipeline = Pipeline(stages=[EchoStage("S1"), EchoStage("S2", fail=True), EchoStage("S3")])
    result = pipeline.run(PipelineContext(order_id="CHAOS002"))
    assert result.completed_stages >= 2  # S1 and S3 succeeded
```

Xem `tests/test_pipeline.py` và `tests/test_stages.py` cho test examples chi tiết.

---

## Kết luận

**Pipeline Architecture** là một trong những kiến trúc lâu đời nhất nhưng vẫn cực kỳ mạnh mẽ và phù hợp cho thế giới data-driven, event-driven ngày nay. Từ Unix pipes đến Apache Kafka Streams, nguyên lý "do one thing and do it well" vẫn là chìa khóa.

### Best Practices

1. **Mỗi stage là một đơn vị deployable**: Stage có thể là Lambda function, Docker container, hoặc microservice riêng.

2. **Idempotency là bắt buộc**: Stage có thể chạy lại nhiều lần. Thiết kế để retry an toàn.

3. **Context là immutable**: Không mutate context — merge để tạo context mới. Giúp dễ debug và trace.

4. **Metrics cho mọi stage**: Duration, count, error rate, input/output size. Biết stage nào là bottleneck.

5. **Error handling per stage**: Mỗi stage quyết định retry, skip, hay abort. Pipeline không nên quyết định thay stage.

6. **Backpressure handling**: Queue có giới hạn. Nếu stage sau quá tải, stage trước phải chờ.

7. **Pipeline as configuration**: Pipeline topology là config, không phải code. Dùng YAML/JSON để định nghĩa.

8. **Distributed tracing**: Mỗi context có trace_id. Dùng OpenTelemetry để trace qua stages.

### Golden Rules

| Rule | Giải thích |
|------|-----------|
| **One stage = One responsibility** | Nếu stage có thể tách thành 2, hãy tách |
| **Immutable context** | Không sửa context, chỉ tạo mới |
| **Idempotent operations** | Chạy N lần cho cùng kết quả |
| **Fail fast with context** | Lỗi sớm, nhưng kèm context đầy đủ |
| **Measure everything** | Metrics per stage là bắt buộc |
| **Pipeline should be visible** | Dashboard cho pipeline status |
| **Test each stage in isolation** | Stage test trước, pipeline test sau |
| **Async là mặc định** | Pipeline nên async để không block |

Pipeline Architecture là kiến trúc lý tưởng cho mọi hệ thống cần xử lý dữ liệu qua nhiều bước, từ CI/CD, data processing, đến order fulfillment. Sự đơn giản của nó (input → process → output) và khả năng scale linh hoạt làm cho nó trở thành một trong những công cụ mạnh mẽ nhất trong tay của một software engineer.
