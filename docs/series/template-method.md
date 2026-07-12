---
id: template-method
title: Template Method
sidebar_label: 📋 Template Method
sidebar_position: 23
---

# Template Method

> "Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure."
> — **GoF**, *Design Patterns* (1994)

Có bao giờ bạn cảm thấy mình đang viết đi viết lại cùng một cấu trúc code, chỉ khác nhau vài dòng lẻ tẻ? Bạn copy nguyên một hàm từ class này sang class khác, đổi vài tham số, rồi tự nhủ "lần sau sẽ refactor"? Tôi cũng từng như vậy đấy.

**Template Method** là một behavioral pattern — nó định nghĩa **khung** (skeleton) của một thuật toán trong base class, để subclass cài đặt chi tiết các bước mà không thay đổi cấu trúc tổng thể. Pattern này còn được gọi là **Hollywood Principle**: "Don't call us, we'll call you." — nghe như một ông trùm Hollywood nói với diễn viên vậy.

---

## Bài toán chi tiết

Hãy tưởng tượng bạn đang xây dựng **hệ thống trích xuất và báo cáo dữ liệu** (Data Report Generator) cho một công ty tài chính. Hệ thống cần tạo báo cáo từ nhiều nguồn dữ liệu khác nhau và xuất ra nhiều định dạng:

**Quy trình chung cho mọi báo cáo:**
1. **Kết nối nguồn dữ liệu** (Connect)
2. **Trích xuất dữ liệu thô** (Extract)
3. **Chuyển đổi và làm sạch** (Transform)
4. **Tính toán các chỉ số** (Compute)
5. **Định dạng báo cáo** (Format)
6. **Xuất file** (Export)
7. **Đóng kết nối** (Cleanup)

**Các loại báo cáo khác nhau:**

| Báo cáo | Nguồn | Transform | Định dạng |
|---------|-------|-----------|-----------|
| Doanh thu ngày | PostgreSQL | Tính tổng theo ngày, loại bỏ giao dịch test | Excel |
| Rủi ro tín dụng | MongoDB + Redis | Tính score, phân loại rủi ro | PDF |
| Giao dịch real-time | Kafka Stream | Lọc, windowing, aggregate | JSON |
| Thuế cuối năm | CSV + API Thuế | Mapping mã số thuế | XML |

Bạn biết cảm giác khi nhìn vào code và thấy nó lặp đi lặp lại chứ? Cách tiếp cập ngây thơ là viết từng class riêng biệt... và thế là bạn có một núi code trùng lặp. Mỗi class đều có cấu trúc giống hệt nhau: connect → extract → transform → compute → format → export → close. Vi phạm DRY. Không nhất quán trong error handling. Và quan trọng nhất — **khó thay đổi quy trình chung**.

---

## Giải pháp với Pattern

Template Method định nghĩa **khung thuật toán** (template method) trong base class và cho phép subclass override các bước cụ thể:

- **AbstractClass** (`ReportGenerator`): Chứa template method `generate()` và khai báo các abstract method cho các bước
- **ConcreteClass** (`DailyRevenueReport`, `RiskReport`): Override các bước cụ thể

Template method gọi các method theo thứ tự cố định. Subclass chỉ override những method cần thay đổi, các method mặc định (hook) có thể giữ nguyên. **Đơn giản mà hiệu quả.**

---

## Phân tích thiết kế

### Nguyên lý OOP được áp dụng

- **Inheritance**: Subclass kế thừa template method từ base class
- **Polymorphism**: Template method gọi abstract method → behavior được quyết định bởi subclass
- **Hollywood Principle**: Base class gọi method của subclass, không phải ngược lại
- **Open/Closed Principle**: Thêm báo cáo mới = thêm subclass, không sửa base class

### Hook Methods

Template Method cung cấp các **hook** — method mặc định (không abstract) mà subclass có thể override hoặc không:

```python
def should_validate(self) -> bool:
    return True  # hook — mặc định có validate

def pre_process(self, ctx) -> None:
    pass  # hook — mặc định không làm gì
```

Hook cho phép subclass can thiệp vào thuật toán mà không phải override toàn bộ template method. **Như một cánh cửa phụ — chỉ mở khi cần.**

### Trade-offs

1. **Hạn chế của inheritance**: Template Method dùng inheritance. Nếu base class thay đổi, tất cả subclass có thể bị ảnh hưởng.
2. **Khó theo dõi flow**: Với deep inheritance hierarchy, khó biết method nào được gọi ở bước nào.
3. **Rigid structure**: Template method định nghĩa thứ tự cố định. Nếu subclass muốn thay đổi thứ tự bước, không thể.
4. **Violates Liskov Substitution**: Nếu subclass override template method, pattern bị phá vỡ.

### Khi nào KHÔNG dùng

- Khi thuật toán có thể thay đổi thứ tự bước → dùng Strategy
- Khi các bước không dùng chung cấu trúc → dùng function riêng lẻ
- Khi chỉ có 1-2 subclass → chi phí tạo abstract class không đáng

---

## Ví dụ code hoàn chỉnh

### Cách sai: Copy-paste code

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json


@dataclass
class Transaction:
    id: str
    amount: float
    currency: str
    status: str
    created_at: datetime
    customer_id: str
    description: str


class NaiveReportService:
    """Cách sai: viết lại toàn bộ logic mỗi loại report"""

    def generate_daily_revenue(self, date: datetime) -> Dict:
        print("[CONNECT] Kết nối PostgreSQL...")
        transactions = self._mock_transactions()

        print("[VALIDATE] Lọc giao dịch test...")
        valid = [t for t in transactions if t.status != "test"]

        print("[COMPUTE] Tính doanh thu...")
        total = sum(t.amount for t in valid if t.status == "completed")
        by_hour = {}
        for t in valid:
            hour = t.created_at.hour
            by_hour[hour] = by_hour.get(hour, 0) + t.amount

        print("[FORMAT] Tạo JSON report...")
        report = {
            "type": "DAILY_REVENUE",
            "date": date.isoformat(),
            "total": total,
            "transaction_count": len(valid),
            "by_hour": by_hour,
        }

        print("[EXPORT] Lưu file...")
        with open(f"revenue_{date.date()}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        print("[CLOSE] Đóng kết nối...")
        return report

    def generate_risk_report(self) -> Dict:
        print("[CONNECT] Kết nối MongoDB...")
        transactions = self._mock_transactions()

        print("[VALIDATE] Lọc giao dịch rủi ro...")
        risky = [t for t in transactions if t.amount > 10_000_000]

        print("[COMPUTE] Tính risk score...")
        total_exposure = sum(t.amount for t in risky)

        print("[FORMAT] Tạo PDF report...")
        report = {
            "type": "RISK_REPORT",
            "generated_at": datetime.now().isoformat(),
            "total_risky_transactions": len(risky),
            "total_exposure": total_exposure,
        }

        print("[EXPORT] Gửi email...")
        print("[CLOSE] Đóng kết nối...")
        return report

    def _mock_transactions(self) -> List[Transaction]:
        return [
            Transaction("T1", 5_000_000, "VND", "completed",
                        datetime.now() - timedelta(hours=2), "C1", ""),
            Transaction("T2", 20_000_000, "VND", "completed",
                        datetime.now() - timedelta(hours=1), "C2", ""),
            Transaction("T3", 1_000, "VND", "test",
                        datetime.now(), "C3", "TEST"),
        ]
```

### Cách đúng: Template Method Pattern

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
from pathlib import Path


# ============================================================
# Domain Models
# ============================================================
@dataclass
class Transaction:
    id: str
    amount: float
    currency: str
    status: str
    created_at: datetime
    customer_id: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportContext:
    """Context object chứa dữ liệu xuyên suốt quy trình"""
    source_connection: Any = None
    raw_data: List[Transaction] = field(default_factory=list)
    cleaned_data: List[Transaction] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[Path] = None
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None


# ============================================================
# Abstract Class with Template Method
# ============================================================
class ReportGenerator(ABC):
    """Base class với template method generate()"""

    @property
    @abstractmethod
    def report_name(self) -> str:
        pass

    def generate(self, output_path: Optional[Path] = None) -> ReportContext:
        """Template method — khung cố định của thuật toán tạo báo cáo"""
        ctx = ReportContext(output_path=output_path)
        try:
            self.pre_process(ctx)          # Hook
            self.connect(ctx)              # Abstract
            self.extract(ctx)              # Abstract
            if self.should_validate():     # Hook
                self.validate(ctx)
            self.transform(ctx)            # Abstract
            self.compute_metrics(ctx)      # Abstract
            self.format_output(ctx)        # Abstract
            self.export(ctx)               # Abstract
            self.post_process(ctx)         # Hook
        except Exception as e:
            self.handle_error(ctx, e)      # Hook
            raise
        finally:
            self.cleanup(ctx)              # Hook
        ctx.finished_at = datetime.now()
        self.log_completion(ctx)
        return ctx

    # --- Abstract methods ---
    @abstractmethod
    def connect(self, ctx: ReportContext) -> None:
        pass

    @abstractmethod
    def extract(self, ctx: ReportContext) -> None:
        pass

    @abstractmethod
    def transform(self, ctx: ReportContext) -> None:
        pass

    @abstractmethod
    def compute_metrics(self, ctx: ReportContext) -> None:
        pass

    @abstractmethod
    def format_output(self, ctx: ReportContext) -> None:
        pass

    @abstractmethod
    def export(self, ctx: ReportContext) -> None:
        pass

    # --- Hook methods (có thể override) ---
    def pre_process(self, ctx: ReportContext) -> None:
        print(f"[{self.report_name}] Bắt đầu...")

    def post_process(self, ctx: ReportContext) -> None:
        pass

    def should_validate(self) -> bool:
        return True

    def validate(self, ctx: ReportContext) -> None:
        ctx.cleaned_data = [
            t for t in ctx.raw_data
            if t.status != "test" and t.amount > 0
        ]

    def handle_error(self, ctx: ReportContext, error: Exception) -> None:
        print(f"[ERROR] {self.report_name}: {error}")

    def cleanup(self, ctx: ReportContext) -> None:
        if ctx.source_connection:
            ctx.source_connection = None

    def log_completion(self, ctx: ReportContext) -> None:
        duration = (ctx.finished_at - ctx.started_at).total_seconds()
        print(f"[COMPLETE] {self.report_name} trong {duration:.2f}s")


# ============================================================
# Concrete Class 1: Daily Revenue Report
# ============================================================
class DailyRevenueReport(ReportGenerator):
    """Báo cáo doanh thu ngày — PostgreSQL → JSON"""

    def __init__(self, report_date: Optional[datetime] = None):
        self._date = report_date or datetime.now()

    @property
    def report_name(self) -> str:
        return f"DAILY_REVENUE_{self._date.strftime('%Y%m%d')}"

    def connect(self, ctx: ReportContext) -> None:
        print(f"[CONNECT] PostgreSQL cho ngày {self._date.date()}")
        ctx.source_connection = "postgresql://localhost:5432/analytics"

    def extract(self, ctx: ReportContext) -> None:
        print("[EXTRACT] Đọc giao dịch trong ngày...")
        ctx.raw_data = self._mock_transactions(30)

    def transform(self, ctx: ReportContext) -> None:
        print("[TRANSFORM] Nhóm dữ liệu theo giờ...")

    def should_validate(self) -> bool:
        return True

    def compute_metrics(self, ctx: ReportContext) -> None:
        data = ctx.cleaned_data or ctx.raw_data
        print("[COMPUTE] Tính doanh thu...")
        completed = [t for t in data if t.status == "completed"]
        total = sum(t.amount for t in completed)
        by_hour: Dict[int, float] = {}
        for t in completed:
            h = t.created_at.hour
            by_hour[h] = by_hour.get(h, 0) + t.amount
        ctx.metrics = {
            "report_type": "DAILY_REVENUE",
            "date": self._date.isoformat(),
            "total_revenue": total,
            "total_orders": len(completed),
            "avg_order_value": total / len(completed) if completed else 0,
            "revenue_by_hour": by_hour,
            "peak_hour": max(by_hour, key=by_hour.get) if by_hour else None,
        }

    def format_output(self, ctx: ReportContext) -> None:
        print("[FORMAT] JSON output")

    def export(self, ctx: ReportContext) -> None:
        path = ctx.output_path or Path(f"reports/{self.report_name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ctx.metrics, f, indent=2, default=str, ensure_ascii=False)
        print(f"[EXPORT] Đã lưu {path}")

    @staticmethod
    def _mock_transactions(count: int) -> List[Transaction]:
        import random
        txns = []
        base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(count):
            txns.append(Transaction(
                id=f"TXN-{i:04d}",
                amount=random.uniform(50_000, 5_000_000),
                currency="VND",
                status=random.choice(["completed", "completed", "completed", "failed", "test"]),
                created_at=base + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59)),
                customer_id=f"CUST-{random.randint(1, 100):04d}",
                description=f"Order #{i}"
            ))
        return txns


# ============================================================
# Concrete Class 2: Risk Report
# ============================================================
class RiskReport(ReportGenerator):
    """Báo cáo rủi ro tín dụng — MongoDB → PDF (mô phỏng)"""

    @property
    def report_name(self) -> str:
        return "RISK_REPORT"

    def connect(self, ctx: ReportContext) -> None:
        print("[CONNECT] MongoDB cluster...")
        ctx.source_connection = "mongodb://mongos:27017/loans"

    def extract(self, ctx: ReportContext) -> None:
        print("[EXTRACT] Đọc khoản vay...")
        ctx.raw_data = [
            Transaction("R1", 100_000_000, "VND", "active",
                        datetime.now() - timedelta(days=30), "C1", "Mortgage"),
            Transaction("R2", 500_000_000, "VND", "overdue",
                        datetime.now() - timedelta(days=90), "C2", "Business"),
            Transaction("R3", 20_000_000, "VND", "active",
                        datetime.now() - timedelta(days=15), "C3", "Personal"),
            Transaction("R4", 1_000_000_000, "VND", "default",
                        datetime.now() - timedelta(days=365), "C4", "Mortgage"),
        ]

    def should_validate(self) -> bool:
        return False

    def transform(self, ctx: ReportContext) -> None:
        print("[TRANSFORM] Tính risk score...")
        for txn in ctx.raw_data:
            score = self._calculate_risk(txn)
            txn.metadata["risk_score"] = score
            txn.metadata["risk_level"] = self._risk_level(score)

    def compute_metrics(self, ctx: ReportContext) -> None:
        data = ctx.cleaned_data or ctx.raw_data
        print("[COMPUTE] Tổng hợp rủi ro...")
        high = [t for t in data if t.metadata.get("risk_level") == "HIGH"]
        medium = [t for t in data if t.metadata.get("risk_level") == "MEDIUM"]
        low = [t for t in data if t.metadata.get("risk_level") == "LOW"]
        ctx.metrics = {
            "report_type": "RISK_REPORT",
            "generated_at": datetime.now().isoformat(),
            "total_exposure": sum(t.amount for t in data),
            "high_risk_count": len(high),
            "high_risk_exposure": sum(t.amount for t in high),
            "medium_risk_count": len(medium),
            "low_risk_count": len(low),
        }

    def format_output(self, ctx: ReportContext) -> None:
        print("[FORMAT] PDF layout")

    def export(self, ctx: ReportContext) -> None:
        path = ctx.output_path or Path("reports/risk_report.pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ctx.metrics, f, indent=2)
        print(f"[EXPORT] Đã lưu {path}")

    @staticmethod
    def _calculate_risk(txn: Transaction) -> float:
        score = 0.0
        if txn.amount > 500_000_000:
            score += 40
        if txn.status == "default":
            score += 50
        elif txn.status == "overdue":
            score += 30
        days_outstanding = (datetime.now() - txn.created_at).days
        if days_outstanding > 180:
            score += 20
        return min(score, 100)

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 70:
            return "HIGH"
        if score >= 40:
            return "MEDIUM"
        return "LOW"


# ============================================================
# Concrete Class 3: Export Report with custom hook
# ============================================================
class CSVExportReport(ReportGenerator):
    """Báo cáo xuất CSV — ghi đè hook để skip validate và format"""

    def __init__(self, filename: str):
        self._filename = filename

    @property
    def report_name(self) -> str:
        return f"CSV_EXPORT_{self._filename}"

    def connect(self, ctx: ReportContext) -> None:
        print("[CONNECT] File system")
        ctx.source_connection = Path("data")

    def extract(self, ctx: ReportContext) -> None:
        print("[EXTRACT] Đọc file CSV...")
        ctx.raw_data = self._mock_transactions(5)

    def should_validate(self) -> bool:
        return False  # Không validate, export raw data

    def transform(self, ctx: ReportContext) -> None:
        pass  # Không transform

    def compute_metrics(self, ctx: ReportContext) -> None:
        print("[COMPUTE] Export toàn bộ dữ liệu")
        ctx.metrics = {"data": [t.__dict__ for t in ctx.raw_data]}

    def format_output(self, ctx: ReportContext) -> None:
        print("[FORMAT] CSV format")
        lines = ["id,amount,currency,status,created_at,customer_id"]
        for t in ctx.raw_data:
            lines.append(f"{t.id},{t.amount},{t.currency},{t.status},{t.created_at.isoformat()},{t.customer_id}")
        ctx.metrics["csv_lines"] = lines

    def export(self, ctx: ReportContext) -> None:
        path = ctx.output_path or Path(f"exports/{self._filename}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ctx.metrics["csv_lines"]))
        print(f"[EXPORT] Đã lưu CSV tại {path}")

    @staticmethod
    def _mock_transactions(count: int) -> List[Transaction]:
        import random
        txns = []
        base = datetime.now()
        for i in range(count):
            txns.append(Transaction(
                id=f"CSV-{i:04d}", amount=random.uniform(10_000, 1_000_000),
                currency="VND", status=random.choice(["active", "completed"]),
                created_at=base - timedelta(days=i),
                customer_id=f"CUST-{random.randint(1, 50):04d}",
                description=f"CSV row {i}"
            ))
        return txns


# ============================================================
# Usage
# ============================================================
def main() -> None:
    print("=" * 60)
    print("BÁO CÁO DOANH THU NGÀY")
    print("=" * 60)
    revenue_report = DailyRevenueReport()
    ctx1 = revenue_report.generate(Path("output/revenue_today.json"))
    print(f"Tổng doanh thu: {ctx1.metrics['total_revenue']:,.0f} VND")
    print(f"Giờ cao điểm: {ctx1.metrics['peak_hour']}h")

    print("\n" + "=" * 60)
    print("BÁO CÁO RỦI RO TÍN DỤNG")
    print("=" * 60)
    risk_report = RiskReport()
    ctx2 = risk_report.generate(Path("output/risk_report.json"))
    print(f"Tổng dư nợ: {ctx2.metrics['total_exposure']:,.0f} VND")
    print(f"Giao dịch rủi ro cao: {ctx2.metrics['high_risk_count']}")

    print("\n" + "=" * 60)
    print("XUẤT CSV")
    print("=" * 60)
    csv_report = CSVExportReport("transactions_2024")
    ctx3 = csv_report.generate(Path("output/transactions.csv"))
    print(f"Số dòng xuất: {len(ctx3.metrics['csv_lines']) - 1}")


if __name__ == "__main__":
    main()
```

---

## Sơ đồ UML

```
┌───────────────────────────────┐
│  ReportGenerator (Abstract)   │
├───────────────────────────────┤
│ # connect(ctx)                │
│ # extract(ctx)                │
│ # transform(ctx)              │
│ # compute_metrics(ctx)        │
│ # format_output(ctx)          │
│ # export(ctx)                 │
│ # pre_process(ctx)    [hook]  │
│ # post_process(ctx)   [hook]  │
│ # validate(ctx)       [hook]  │
│ # cleanup(ctx)        [hook]  │
├───────────────────────────────┤
│ + generate(output_path)       │ ← Template Method
└───────────────────────────────┘
           ▲          ▲              ▲
           │          │              │
  ┌────────┴──┐  ┌───┴────────┐  ┌──┴────────────┐
  │ Daily     │  │ RiskReport │  │ CSVExport      │
  │ Revenue   │  │            │  │ Report         │
  │ Report    │  │            │  │                │
  ├───────────┤  ├────────────┤  ├────────────────┤
  │+connect() │  │+connect()  │  │+connect()      │
  │+extract() │  │+extract()  │  │+extract()      │
  │+compute() │  │+transform()│  │+shouldValidate │
  │+export()  │  │+compute()  │  │ = False        │
  └───────────┘  └────────────┘  └────────────────┘
```

---

## So sánh với Pattern liên quan

### 1. Template Method vs Strategy

| Tiêu chí | Template Method | Strategy |
|----------|----------------|----------|
| Cơ chế | Inheritance (subclass override) | Composition (delegate) |
| Thay đổi runtime | ❌ Không (compile time) | ✅ Có (set strategy mới) |
| Số lượng thuật toán | Một, các bước khác nhau | Nhiều, độc lập |
| Base class có code? | ✅ Rất nhiều (template method + hooks) | ❌ Chỉ interface |
| Khi nào dùng | Các variant có chung cấu trúc lớn | Nhiều thuật toán hoán đổi được |

**Kết hợp**: Dùng Template Method để định nghĩa khung, Strategy để implement các bước chi tiết. Pattern này gọi là **Strategy Pattern trong Template Method**.

### 2. Template Method vs Factory Method

| Tiêu chí | Template Method | Factory Method |
|----------|----------------|----------------|
| Mục đích | Định nghĩa thuật toán | Tạo đối tượng |
| Hook type | Functional hooks (bước của thuật toán) | Creational hooks (factory method) |
| Template method có thể chứa | Factory method để tạo object | - |

**Mối quan hệ**: Factory Method thường được gọi từ Template Method. Ví dụ: `Document.open()` (template) gọi `createDocument()` (factory method).

### 3. Template Method vs Bridge

Bridge pattern có intent tách abstraction khỏi implementation. Template Method là một dạng "implementation inheritance" (kế thừa để mở rộng).

| Tiêu chí | Template Method | Bridge |
|----------|----------------|--------|
| Relationship | IS-A (inheritance) | HAS-A (composition) |
| Cấp độ | Method-level | Architecture-level |
| Tính linh hoạt | Thấp hơn | Cao hơn (runtime swap) |

---

## Ứng dụng thực tế

### 1. Django Class-Based Views (CBV)

Django CBV là ví dụ kinh điển của Template Method pattern:

```python
# django/views/generic/base.py
class View:
    def dispatch(self, request, *args, **kwargs):
        if request.method.lower() in self.http_method_names:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
        else:
            handler = self.http_method_not_allowed
        return handler(request, *args, **kwargs)

class TemplateView(TemplateResponseMixin, View):
    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        return self.render_to_response(context)

# Cách sử dụng: subclass và override các method cần thiết
class MyView(TemplateView):
    template_name = "my_template.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["extra_data"] = self.get_extra()
        return context
```

### 2. Python threading.Thread

```python
import threading

class MyWorker(threading.Thread):
    def run(self):  # Template method
        print("Kết nối database...")
        data = self.fetch_data()
        self.process(data)
        print("Đóng kết nối...")

    def fetch_data(self):
        raise NotImplementedError

    def process(self, data):
        raise NotImplementedError

# Sử dụng
class DataImportWorker(MyWorker):
    def fetch_data(self):
        return ["item1", "item2", "item3"]

    def process(self, data):
        for item in data:
            print(f"Xử lý {item}")
```

### 3. ORM Lifecycle Hooks (SQLAlchemy/Django ORM)

```python
# Django Model.save() là template method
# django/db/models/base.py
class Model:
    def save(self, *args, **kwargs):
        self.full_clean(exclude=exclude, ...)       # validate
        self.save_base(...)                           # actual save
        self._post_save(...)                          # post-process

    def full_clean(self, exclude=None, ...):
        self.clean_fields(exclude=exclude)            # Hook
        self.clean()                                  # Hook
        self.validate_unique()                        # Hook

class MyModel(Model):
    def clean(self):
        if self.price < 0:
            raise ValidationError("Giá không thể âm")
```

### 4. Quy trình ETL (Apache Airflow)

```python
from airflow.operators.python import PythonOperator
from airflow.models import DAG

# Template Method trong data pipeline
class ETLPipeline:
    def run(self):
        self.extract()
        self.transform()
        self.load()

    def extract(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

class UserAnalyticsETL(ETLPipeline):
    def extract(self):
        return db.query("SELECT * FROM users")

    def transform(self):
        # Làm sạch và chuyển đổi
        pass

    def load(self):
        # Load vào data warehouse
        pass
```

---

## Kiểm thử

```python
import unittest
from pathlib import Path
import tempfile
import shutil


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_daily_revenue_generate(self):
        """DailyRevenueReport tạo báo cáo thành công"""
        report = DailyRevenueReport()
        output = Path(self.temp_dir) / "revenue.json"
        ctx = report.generate(output)
        self.assertIsNotNone(ctx.metrics)
        self.assertIn("total_revenue", ctx.metrics)
        self.assertIn("revenue_by_hour", ctx.metrics)
        self.assertGreater(ctx.metrics["total_orders"], 0)
        self.assertTrue(output.exists())

    def test_risk_report_generate(self):
        """RiskReport tính đúng risk level"""
        report = RiskReport()
        output = Path(self.temp_dir) / "risk.json"
        ctx = report.generate(output)
        self.assertEqual(ctx.metrics["report_type"], "RISK_REPORT")
        self.assertIn("high_risk_count", ctx.metrics)
        self.assertIn("total_exposure", ctx.metrics)

    def test_csv_export_format(self):
        """CSVExportReport xuất đúng định dạng CSV"""
        report = CSVExportReport("test_export")
        output = Path(self.temp_dir) / "test.csv"
        ctx = report.generate(output)
        csv_lines = ctx.metrics["csv_lines"]
        self.assertTrue(csv_lines[0].startswith("id,amount"))  # Header
        self.assertGreater(len(csv_lines), 1)  # Có dữ liệu

        # Kiểm tra nội dung file
        with open(output, "r") as f:
            content = f.read()
        self.assertIn("CSV-0000", content)

    def test_template_method_flow(self):
        """Template method gọi đúng thứ tự các bước"""
        class CallTrackingReport(ReportGenerator):
            def __init__(self):
                self.call_order = []

            @property
            def report_name(self):
                return "TRACKING"

            def connect(self, ctx):
                self.call_order.append("connect")

            def extract(self, ctx):
                self.call_order.append("extract")

            def transform(self, ctx):
                self.call_order.append("transform")

            def compute_metrics(self, ctx):
                self.call_order.append("compute")

            def format_output(self, ctx):
                self.call_order.append("format")

            def export(self, ctx):
                self.call_order.append("export")

        report = CallTrackingReport()
        report.generate()
        expected = ["connect", "extract", "transform", "compute", "format", "export"]
        # pre_process, should_validate, validate, post_process, cleanup, log_completion
        for step in expected:
            self.assertIn(step, report.call_order)

    def test_validate_hook_removes_test_transactions(self):
        """Hook validate mặc định loại bỏ giao dịch test"""
        report = DailyRevenueReport()
        ctx = ReportContext()
        ctx.raw_data = [
            Transaction("1", 100_000, "VND", "test", datetime.now(), "C1", ""),
            Transaction("2", 200_000, "VND", "completed", datetime.now(), "C2", ""),
            Transaction("3", -1000, "VND", "completed", datetime.now(), "C3", ""),
        ]
        report.validate(ctx)
        self.assertEqual(len(ctx.cleaned_data), 1)

    def test_should_validate_hook(self):
        """RiskReport override should_validate = False"""
        report = RiskReport()
        self.assertFalse(report.should_validate())

    def test_error_handling(self):
        """handle_error được gọi khi có exception"""
        class BrokenReport(ReportGenerator):
            @property
            def report_name(self):
                return "BROKEN"

            def connect(self, ctx):
                raise ValueError("Connection failed")

            def extract(self, ctx): pass
            def transform(self, ctx): pass
            def compute_metrics(self, ctx): pass
            def format_output(self, ctx): pass
            def export(self, ctx): pass

        report = BrokenReport()
        with self.assertRaises(ValueError):
            report.generate()

    def test_cleanup_always_called(self):
        """Cleanup được gọi ngay cả khi có lỗi"""
        class CleanupTrackingReport(ReportGenerator):
            def __init__(self):
                self.cleaned_up = False

            @property
            def report_name(self):
                return "CLEANUP_TEST"

            def connect(self, ctx): pass
            def extract(self, ctx): raise RuntimeError("Boom")
            def transform(self, ctx): pass
            def compute_metrics(self, ctx): pass
            def format_output(self, ctx): pass
            def export(self, ctx): pass

            def cleanup(self, ctx):
                self.cleaned_up = True

        report = CleanupTrackingReport()
        with self.assertRaises(RuntimeError):
            report.generate()
        self.assertTrue(report.cleaned_up)


if __name__ == "__main__":
    unittest.main()
```

---

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Giảm trùng lặp**: Cấu trúc thuật toán viết một lần | **Inheritance rigid**: Khó thay đổi cấu trúc base |
| **Dễ bảo trì**: Sửa template method = sửa tất cả | **Violates LSP**: Nếu subclass override template method |
| **Dễ mở rộng**: Thêm lớp mới = override vài method | **Nhiều class**: Mỗi biến thể một subclass |
| **Kiểm soát**: Base class kiểm soát kiến trúc tổng thể | **Khó debug**: Flow qua nhiều lớp, khó theo dõi |
| **Hollywood Principle**: Base gọi subclass, tránh duplicate | **Deep hierarchy**: Nếu kế thừa nhiều cấp, khó quản lý |

---

## Kết luận

Tôi đã thấy Template Method xuất hiện ở khắp mọi nơi — từ Django CBV, threading.Thread, đến Airflow DAG. Nó là một trong những pattern đơn giản nhất nhưng cực kỳ hữu ích. Pattern này dạy chúng ta một bài học quan trọng: **đừng viết lại cấu trúc, hãy định nghĩa khung và để phần chi tiết cho subclass**.

Như một người thợ mộc lành nghề — ông ta không đẽo từng cái bàn từ đầu mỗi lần. Ông ta có một cái khung, một quy trình. Mỗi cái bàn chỉ khác nhau ở chi tiết trang trí. Đó chính xác là những gì Template Method làm cho code của bạn.

### Khi nào áp dụng

- ✅ Các class có cùng cấu trúc thuật toán, chỉ khác nhau vài bước
- ✅ Muốn tránh copy-paste code với cùng một flow
- ✅ Framework code — bạn viết base class, để người dùng override
- ✅ Các bước có thứ tự cố định và ít thay đổi

### Golden Rules

1. **Template method nên là `final`/không override được**: Trong Python, không có `final` keyword nhưng hãy coi template method là bất khả xâm phạm.
2. **Abstract method cho bước bắt buộc, hook cho bước tùy chọn**: Phân biệt rõ điều gì subclass phải implement và điều gì có thể override.
3. **Hook nên có default implementation hợp lý**: `should_validate()` mặc định trả `True`, `cleanup()` mặc định không làm gì.
4. **Giữ template method ngắn**: Nếu template method dài hơn 20 dòng, hãy xem xét tách thành nhiều method nhỏ hơn.
5. **Dùng Strategy khi cần linh hoạt hơn**: Nếu các bước cần thay đổi runtime hoặc thứ tự, Strategy pattern phù hợp hơn.

---

*Trân trọng!*
