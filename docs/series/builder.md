---
id: builder
title: Builder
sidebar_label: 🔨 Builder
sidebar_position: 5
---

# Builder

> *"Separate the construction of a complex object from its representation so that the same construction process can create different representations."* — Gang of Four, *Design Patterns: Elements of Reusable Object-Oriented Software*, 1994.

Có bao giờ bạn nhìn vào một constructor với 20 tham số và tự hỏi: "Cái quái gì đang xảy ra ở đây vậy?" Tôi thì có — và tôi ghét nó.

**Builder** thuộc nhóm **Creational Patterns**, tách quá trình xây dựng một object phức tạp ra khỏi biểu diễn của nó. Pattern này cho phép cùng một quy trình xây dựng (construction process) tạo ra nhiều biểu diễn (representations) khác nhau. **Điểm mạnh cốt lõi** của Builder nằm ở chỗ nó kiểm soát được **từng bước** của quá trình xây dựng — điều mà Factory Method và Abstract Factory không làm được.

## Bài toán chi tiết

Giả sử bạn đang xây dựng một **hệ thống tạo báo cáo tài chính tự động** cho một ngân hàng đầu tư. Mỗi ngày, hệ thống phải sinh ra hàng trăm báo cáo khác nhau: báo cáo giao dịch nội bộ, báo cáo cho khách hàng, báo cáo cho cơ quan thuế, và báo cáo kiểm toán nội bộ. Mỗi loại báo cáo có cấu trúc phức tạp với nhiều phần: header (logo, tên báo cáo, ngày giờ, mã tham chiếu), summary section (tổng quan tài chính, con số chính, các KPI), transaction table (danh sách giao dịch), charts section (biểu đồ phân tích), và footer (chữ ký số, disclaimer, QR code).

Mỗi báo cáo có thể được xuất ra **nhiều định dạng**: PDF (cho khách hàng), Excel (cho nội bộ), HTML (cho web dashboard), và JSON (cho API). Mỗi định dạng có cách render khác nhau — PDF cần layout chính xác từng pixel, Excel cần cell references và formulas, HTML cần CSS và responsive design.

Cách tiếp cận ngây thơ ban đầu là dùng một constructor khổng lồ với 20+ tham số. Điều này dẫn đến **telescoping constructor** — người dùng phải nhớ thứ tự và ý nghĩa của từng tham số. Object có thể được tạo với state không hợp lệ (thiếu header, thiếu footer). Cùng loại báo cáo nhưng khác format phải viết lại toàn bộ constructor.

**Kinh nghiệm của tôi**: Mỗi khi thấy một constructor có hơn 5 tham số, đó là dấu hiệu bạn đang cần Builder.

## Giải pháp với Pattern

Builder tách quá trình xây dựng thành các bước riêng biệt, mỗi bước là một method có tên rõ ràng. Pattern gồm: **Product** (`Report`), **Builder interface** (`ReportBuilder`), **Concrete Builders** (`PDFReportBuilder`, `HTMLReportBuilder`, `JSONReportBuilder`), và **Director** (`ReportDirector`) điều khiển quy trình xây dựng.

Với Builder: fluent interface cho phép method chaining, có thể xây dựng immutable object (build xong không sửa được), Director hoặc Builder đảm bảo object được xây dựng đầy đủ trước khi trả về, và cùng một process có thể tạo ra nhiều representation khác nhau.

## Phân tích thiết kế

**OOP Principles áp dụng:**

- **Single Responsibility Principle**: Product không biết cách xây dựng nó, Builder không biết business logic của Product.
- **Open/Closed Principle**: Thêm định dạng output mới (CSV) không cần sửa Director hay Builder cũ.
- **Encapsulation**: Product không cần constructor công khai — chỉ Builder mới biết cách tạo Product.

**Trade-offs:**

- **So với Factory Method**: Factory Method tạo object "một lần" — gọi factory, nhận product. Builder tạo object "từng bước" — gọi method A, method B, build(). Builder cần nhiều code hơn nhưng linh hoạt hơn.
- **Code duplication**: Các Builder thường có logic trùng nhau — cần base builder hoặc mixin.
- **Memory**: Builder phải giữ intermediate state — có thể tốn bộ nhớ nếu Product quá lớn.

**Khi nào KHÔNG nên dùng Builder:**

- Khi object chỉ có 2-3 thuộc tính — constructor thông thường hoặc dataclass là đủ.
- Khi object không có nhiều biểu diễn khác nhau — Factory Method đơn giản hơn.
- Khi client cần kiểm soát chặt thứ tự các bước — Director khó quản lý mọi permutation.

## Ví dụ code hoàn chỉnh

### Cách làm sai (Telescoping Constructor)

```python
from dataclasses import dataclass
from typing import Any
from enum import Enum


class ReportFormat(Enum):
    PDF = "pdf"
    EXCEL = "xlsx"
    HTML = "html"
    JSON = "json"


@dataclass
class FinancialReport:
    """Cách sai: constructor khổng lồ, khó dùng, khó đọc."""
    logo_path: str
    report_name: str
    created_at: str
    reference_code: str
    total_revenue: float
    total_expenses: float
    net_profit: float
    kpis: dict
    transactions: list
    columns: list
    chart_types: list
    chart_data: dict
    signature: str
    disclaimer: str
    qr_code: str
    compliance_text: str
    output_format: str


report = FinancialReport(
    "logo_bank.png",
    "Báo cáo giao dịch tháng 4/2024",
    "2024-04-22T10:30:00",
    "REF-BANK-0422",
    15000000000.0,
    8000000000.0,
    7000000000.0,
    {"ROE": 0.15, "ROA": 0.08},
    [],
    [],
    ["bar", "pie"],
    {},
    "Nguyen Van A",
    "Báo cáo này chỉ mang tính tham khảo...",
    "QR-BANK-0422",
    "Basel III compliant",
    "pdf",
)
```

### Refactored với Builder Pattern

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Optional
from enum import Enum
from datetime import datetime
import json


class ReportFormat(Enum):
    PDF = "pdf"
    EXCEL = "xlsx"
    HTML = "html"
    JSON = "json"


class PageSize(Enum):
    A4 = "A4"
    LETTER = "LETTER"
    LEGAL = "LEGAL"


@dataclass(frozen=True)
class KPI:
    name: str
    value: float
    unit: str = ""
    trend: str = "stable"


@dataclass(frozen=True)
class Transaction:
    id: str
    date: str
    description: str
    amount: float
    currency: str = "VND"
    category: str = "other"


@dataclass(frozen=True)
class ChartConfig:
    chart_type: str
    title: str
    data: dict[str, list[float]]
    labels: list[str]


class Report:
    """Product — được xây dựng bởi Builder."""

    def __init__(self) -> None:
        self._sections: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def set_section(self, name: str, content: Any) -> None:
        self._sections[name] = content

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_section(self, name: str) -> Any:
        return self._sections.get(name)

    def has_section(self, name: str) -> bool:
        return name in self._sections

    def validate(self) -> bool:
        required = {"header", "summary", "transactions", "footer"}
        missing = required - set(self._sections.keys())
        if missing:
            raise ValueError(f"Thiếu sections: {missing}")
        return True

    @property
    def sections(self) -> dict[str, Any]:
        return dict(self._sections)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def export(self) -> dict[str, Any]:
        return {"metadata": self._metadata, "sections": self._sections}


class ReportBuilder(ABC):
    """Abstract Builder."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def build_header(self, logo_path: str, report_name: str,
                     created_at: str, reference_code: str) -> None: ...

    @abstractmethod
    def build_summary(self, revenue: float, expenses: float,
                      profit: float, kpis: list[KPI]) -> None: ...

    @abstractmethod
    def add_transaction(self, transaction: Transaction) -> None: ...

    @abstractmethod
    def set_transaction_columns(self, columns: list[str]) -> None: ...

    @abstractmethod
    def add_chart(self, chart: ChartConfig) -> None: ...

    @abstractmethod
    def build_footer(self, signature: str, disclaimer: str,
                     qr_code: str = "", compliance: str = "") -> None: ...

    @abstractmethod
    def get_result(self) -> Report: ...


class PDFReportBuilder(ReportBuilder):
    """Concrete Builder — tạo Report cho PDF output."""

    def __init__(self, page_size: PageSize = PageSize.A4) -> None:
        self._page_size = page_size
        self._report: Optional[Report] = None

    def reset(self) -> None:
        self._report = Report()

    def _check(self) -> None:
        if self._report is None:
            self.reset()

    def build_header(self, logo_path: str, report_name: str,
                     created_at: str, reference_code: str) -> None:
        self._check()
        self._report.set_section("header", {
            "logo": logo_path, "report_name": report_name,
            "created_at": created_at, "reference_code": reference_code,
            "page_size": self._page_size.value,
        })
        self._report.set_metadata("report_name", report_name)
        self._report.set_metadata("created_at", created_at)
        self._report.set_metadata("reference_code", reference_code)

    def build_summary(self, revenue: float, expenses: float,
                      profit: float, kpis: list[KPI]) -> None:
        self._check()
        self._report.set_section("summary", {
            "revenue": revenue, "expenses": expenses,
            "profit": profit, "kpis": [asdict(k) for k in kpis],
            "net_margin": profit / revenue if revenue else 0,
        })

    def add_transaction(self, transaction: Transaction) -> None:
        self._check()
        trans = self._report.get_section("transactions")
        if trans is None:
            trans = {"rows": []}
            self._report.set_section("transactions", trans)
        trans["rows"].append(asdict(transaction))

    def set_transaction_columns(self, columns: list[str]) -> None:
        self._check()
        trans = self._report.get_section("transactions")
        if trans is None:
            trans = {"rows": []}
            self._report.set_section("transactions", trans)
        trans["columns"] = columns

    def add_chart(self, chart: ChartConfig) -> None:
        self._check()
        charts = self._report.get_section("charts")
        if charts is None:
            charts = {"charts": []}
            self._report.set_section("charts", charts)
        charts["charts"].append(asdict(chart))

    def build_footer(self, signature: str, disclaimer: str,
                     qr_code: str = "", compliance: str = "") -> None:
        self._check()
        self._report.set_section("footer", {
            "signature": signature, "disclaimer": disclaimer,
            "qr_code": qr_code, "compliance": compliance,
        })

    def get_result(self) -> Report:
        if self._report is None:
            raise RuntimeError("Chưa gọi reset()")
        self._report.validate()
        report = self._report
        self._report = None
        return report


class HTMLReportBuilder(ReportBuilder):
    def __init__(self) -> None:
        self._report: Optional[Report] = None

    def reset(self) -> None:
        self._report = Report()

    def _check(self) -> None:
        if self._report is None:
            self.reset()

    def build_header(self, logo_path: str, report_name: str,
                     created_at: str, reference_code: str) -> None:
        self._check()
        html = f"<header><img src='{logo_path}'/><h1>{report_name}</h1><p>{created_at} | {reference_code}</p></header>"
        self._report.set_section("header", {"html": html})

    def build_summary(self, revenue: float, expenses: float,
                      profit: float, kpis: list[KPI]) -> None:
        self._check()
        rows = "".join(f"<tr><td>{k.name}</td><td>{k.value:.2f}</td></tr>" for k in kpis)
        html = f"<section><h2>Tong quan</h2><p>Doanh thu: {revenue:,.0f}</p><p>Chi phi: {expenses:,.0f}</p><p>Loi nhuan: {profit:,.0f}</p><table>{rows}</table></section>"
        self._report.set_section("summary", {"html": html})

    def add_transaction(self, transaction: Transaction) -> None:
        self._check()
        trans = self._report.get_section("transactions")
        if trans is None:
            trans = {"rows_html": "", "count": 0}
            self._report.set_section("transactions", trans)
        trans["rows_html"] += f"<tr><td>{transaction.date}</td><td>{transaction.description}</td><td>{transaction.amount:,.0f}</td></tr>"
        trans["count"] += 1

    def set_transaction_columns(self, columns: list[str]) -> None:
        self._check()
        trans = self._report.get_section("transactions")
        if trans is None:
            trans = {"rows_html": "", "count": 0}
            self._report.set_section("transactions", trans)
        trans["header_html"] = "<tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>"

    def add_chart(self, chart: ChartConfig) -> None:
        self._check()
        charts = self._report.get_section("charts")
        if charts is None:
            charts = {"html": ""}
            self._report.set_section("charts", charts)
        charts["html"] += f"<div><h4>{chart.title}</h4><canvas></canvas></div>"

    def build_footer(self, signature: str, disclaimer: str,
                     qr_code: str = "", compliance: str = "") -> None:
        self._check()
        html = f"<footer><p>Ky boi: {signature}</p><p>{disclaimer}</p></footer>"
        self._report.set_section("footer", {"html": html})

    def get_result(self) -> Report:
        if self._report is None:
            raise RuntimeError("Chua goi reset()")
        self._report.validate()
        report = self._report
        self._report = None
        return report


class JSONReportBuilder(ReportBuilder):
    def __init__(self) -> None:
        self._report: Optional[Report] = None

    def reset(self) -> None:
        self._report = Report()

    def _check(self) -> None:
        if self._report is None:
            self.reset()

    def build_header(self, logo_path: str, report_name: str,
                     created_at: str, reference_code: str) -> None:
        self._check()
        self._report.set_section("header", {
            "logo_url": logo_path, "name": report_name,
            "created_at": created_at, "ref": reference_code,
        })
        self._report.set_metadata("report_name", report_name)

    def build_summary(self, revenue: float, expenses: float,
                      profit: float, kpis: list[KPI]) -> None:
        self._check()
        self._report.set_section("summary", {
            "revenue": revenue, "expenses": expenses,
            "profit": profit, "kpis": [asdict(k) for k in kpis],
        })

    def add_transaction(self, transaction: Transaction) -> None:
        self._check()
        trans = self._report.get_section("transactions")
        if trans is None:
            trans = {"items": []}
            self._report.set_section("transactions", trans)
        trans["items"].append(asdict(transaction))

    def set_transaction_columns(self, columns: list[str]) -> None:
        self._check()
        trans = self._report.get_section("transactions")
        if trans is None:
            trans = {"items": []}
            self._report.set_section("transactions", trans)
        trans["columns"] = columns

    def add_chart(self, chart: ChartConfig) -> None:
        self._check()
        charts = self._report.get_section("charts")
        if charts is None:
            charts = {"items": []}
            self._report.set_section("charts", charts)
        charts["items"].append(asdict(chart))

    def build_footer(self, signature: str, disclaimer: str,
                     qr_code: str = "", compliance: str = "") -> None:
        self._check()
        self._report.set_section("footer", {
            "signed_by": signature, "disclaimer": disclaimer,
            "qr_code": qr_code, "compliance": compliance,
        })

    def get_result(self) -> Report:
        if self._report is None:
            raise RuntimeError("Chua goi reset()")
        self._report.validate()
        report = self._report
        self._report = None
        return report


class ReportDirector:
    """Director — dieu khien quy trinh xay dung."""

    def __init__(self, builder: ReportBuilder) -> None:
        self._builder = builder

    @property
    def builder(self) -> ReportBuilder:
        return self._builder

    @builder.setter
    def builder(self, builder: ReportBuilder) -> None:
        self._builder = builder

    def build_daily_report(self, date: str, branch: str) -> Report:
        self._builder.reset()
        self._builder.build_header(
            logo_path="logo_bank.png",
            report_name=f"Bao cao giao dich ngay {date} - Chi nhanh {branch}",
            created_at=datetime.now().isoformat(),
            reference_code=f"DR-{date}-{branch[:3].upper()}",
        )
        self._builder.build_summary(revenue=0.0, expenses=0.0, profit=0.0, kpis=[])
        self._builder.set_transaction_columns(["Ngay", "Mo ta", "So tien", "Loai"])
        self._builder.build_footer(
            signature="Giam doc chi nhanh",
            disclaimer="Bao cao noi bo",
        )
        return self._builder.get_result()

    def build_monthly_report(self, month: int, year: int,
                             transactions: list[Transaction],
                             kpis: list[KPI],
                             charts: list[ChartConfig]) -> Report:
        self._builder.reset()
        self._builder.build_header(
            logo_path="logo_bank.png",
            report_name=f"Bao cao tai chinh thang {month}/{year}",
            created_at=datetime.now().isoformat(),
            reference_code=f"MR-{year}{month:02d}",
        )
        revenue = sum(t.amount for t in transactions if t.amount > 0)
        expenses = sum(t.amount for t in transactions if t.amount < 0) * -1
        profit = revenue - expenses
        self._builder.build_summary(revenue=revenue, expenses=expenses,
                                     profit=profit, kpis=kpis)
        self._builder.set_transaction_columns(["Ngay", "Mo ta", "So tien", "Tien te", "Danh muc"])
        for tx in transactions:
            self._builder.add_transaction(tx)
        for chart in charts:
            self._builder.add_chart(chart)
        self._builder.build_footer(
            signature="Tong giam doc",
            disclaimer="Bao cao da duoc kiem toan noi bo.",
            qr_code=f"QR-MR-{year}{month:02d}",
            compliance="Basel III compliant",
        )
        return self._builder.get_result()


# Su dung thuc te
if __name__ == "__main__":
    transactions = [
        Transaction("T1", "2024-04-01", "Thu tu khach hang A", 500000000, category="revenue"),
        Transaction("T2", "2024-04-02", "Chi tra nha cung cap B", -200000000, category="expense"),
        Transaction("T3", "2024-04-03", "Thu tu khach hang C", 350000000, category="revenue"),
    ]
    kpis = [KPI("ROE", 0.15, "%", "up"), KPI("ROA", 0.08, "%", "stable")]
    charts = [ChartConfig("bar", "Doanh thu", {"doanh thu": [500, 0, 350]}, ["01/04", "02/04", "03/04"])]

    pdf_builder = PDFReportBuilder()
    director = ReportDirector(pdf_builder)
    pdf_report = director.build_monthly_report(4, 2024, transactions, kpis, charts)
    print(f"[PDF] {pdf_report}")

    html_builder = HTMLReportBuilder()
    director.builder = html_builder
    html_report = director.build_monthly_report(4, 2024, transactions, kpis, charts)
    print(f"[HTML] {html_report}")

    json_builder = JSONReportBuilder()
    director.builder = json_builder
    json_report = director.build_monthly_report(4, 2024, transactions, kpis, charts)
    print(f"[JSON] Sections: {list(json_report.sections.keys())}")
```

## So do UML

```
+-----------------------------------------------+
|              «interface»                     |
|              ReportBuilder                     |
+-----------------------------------------------+
| + reset()                                      |
| + build_header(logo, name, created, ref)       |
| + build_summary(revenue, expenses, profit, kpi)|
| + add_transaction(transaction)                 |
| + set_transaction_columns(columns)             |
| + add_chart(chart)                             |
| + build_footer(sig, disc, qr, comp)            |
| + get_result() -> Report                       |
+-----------------------------------------------+
          ^              ^              ^
          |              |              |
+---------+------+ +----+-------+ +----+----------+
|PDFReportBuilder| |HTMLBuilder| | JSONBuilder    |
+----------------+ +-----------+ +----------------+
| - page_size    | | - report  | | - report       |
| - report       | | + methods | | + methods      |
+----------------+ +-----------+ +----------------+
          ^
          |  su dung
+---------+----------+
|  ReportDirector    |
+--------------------+
| - builder          |
+--------------------+
| + build_daily_rpt  |
| + build_monthly    |
+--------------------+

+--------------------------+
|         Report           |
+--------------------------+
| - _sections: dict        |
| - _metadata: dict        |
+--------------------------+
| + set_section()          |
| + set_metadata()         |
| + get_section()          |
| + validate()             |
| + export()               |
+--------------------------+
```

## So sanh voi Pattern lien quan

| Pattern | Diem giong | Diem khac biet chinh |
|---------|-----------|---------------------|
| **Factory Method** | Deu tao object qua interface | Factory tao object *mot lan*, Builder tao *tung buoc*. Factory khong co Director. Builder co the co nhieu buoc optional. |
| **Abstract Factory** | Deu tao family objects | Abstract Factory tao *nhieu loai* object khac nhau, Builder tao *mot* object phuc tap. Builder co the dung Abstract Factory de tao cac thanh phan con. |
| **Prototype** | Deu tao object phuc tap | Prototype clone object co san, Builder xay dung tu dau. Prototype phu hop khi object hien co gan giong object can tao. |

## Ung dung thuc te

### 1. Django QuerySet — Fluent Builder

```python
from django.db import models

# Builder pattern — moi method tra ve QuerySet moi
qs = (
    User.objects
    .filter(is_active=True)
    .select_related("profile")
    .prefetch_related("orders")
    .order_by("-date_joined")
    .only("id", "email")
)

# Cung builder process, nhieu representation
active_users = qs[:10]
count = qs.count()
exists = qs.exists()
```

### 2. Pandas Method Chaining

```python
import pandas as pd

df = (
    pd.read_csv("transactions.csv")
    .query("amount > 0")
    .assign(tax=lambda x: x.amount * 0.1)
    .groupby("category")
    .agg({"amount": "sum", "tax": "mean"})
    .reset_index()
    .sort_values("amount", ascending=False)
)
```

### 3. SQLAlchemy Query Builder

```python
from sqlalchemy import select
from sqlalchemy.orm import Session

# Builder pattern trong SQL construction
query = (
    select(User)
    .where(User.is_active == True)
    .order_by(User.created_at.desc())
    .limit(10)
    .offset(0)
)

# Cung query builder, nhieu cach thuc thi
result = session.execute(query).scalars().all()
count = session.execute(select(func.count()).select_from(query.subquery())).scalar()
```

## Kiem thu

```python
import pytest
from unittest.mock import MagicMock
from datetime import datetime


class TestReportBuilder:
    def test_pdf_builder_creates_valid_report(self):
        builder = PDFReportBuilder()
        # Reset duoc goi tu dong qua _check, nhung goi explicit cho ro
        builder.build_header("logo.png", "Test", "2024-01-01", "REF-001")
        builder.build_summary(1000, 400, 600, [KPI("ROE", 0.15)])
        builder.add_transaction(Transaction("T1", "2024-01-01", "Test", 500))
        builder.set_transaction_columns(["Date", "Desc", "Amount"])
        builder.build_footer("Signature", "Disclaimer")

        report = builder.get_result()
        assert report.has_section("header")
        assert report.has_section("summary")
        assert report.has_section("transactions")
        assert report.has_section("footer")

    def test_missing_required_section_raises_error(self):
        builder = PDFReportBuilder()
        builder.build_header("logo.png", "Test", "2024-01-01", "REF")
        # Khong goi build_summary va build_footer
        with pytest.raises(ValueError, match="Thieu sections"):
            builder.get_result()

    def test_builder_isolation(self):
        builder_a = PDFReportBuilder()
        builder_b = PDFReportBuilder()

        builder_a.build_header("logo_a.png", "Report A", "2024-01-01", "REF-A")
        builder_a.build_summary(100, 50, 50, [])
        builder_a.build_footer("Sign A", "Disc A")

        builder_b.build_header("logo_b.png", "Report B", "2024-01-02", "REF-B")
        builder_b.build_summary(200, 100, 100, [])
        builder_b.build_footer("Sign B", "Disc B")

        assert builder_a.get_result().metadata["report_name"] == "Report A"
        assert builder_b.get_result().metadata["report_name"] == "Report B"

    def test_json_builder_output_structure(self):
        builder = JSONReportBuilder()
        builder.build_header("logo.png", "JSON Report", "2024-01-01", "J-REF")
        builder.build_summary(500, 200, 300, [KPI("Test", 1.0)])
        builder.build_footer("Sign", "Disc")

        report = builder.get_result()
        export = report.export()
        assert "sections" in export
        assert export["sections"]["header"]["name"] == "JSON Report"

    def test_multiple_calls_to_get_result_raises_error(self):
        builder = PDFReportBuilder()
        builder.build_header("logo.png", "Test", "2024-01-01", "REF")
        builder.build_summary(100, 50, 50, [])
        builder.build_footer("Sign", "Disc")
        builder.get_result()

        with pytest.raises(RuntimeError):
            builder.get_result()


class TestReportDirector:
    def test_same_process_different_representations(self):
        pdf_builder = PDFReportBuilder()
        html_builder = HTMLReportBuilder()

        director = ReportDirector(pdf_builder)
        pdf_report = director.build_daily_report("2024-01-01", "HN")

        director.builder = html_builder
        html_report = director.build_daily_report("2024-01-01", "HN")

        assert pdf_report.has_section("header")
        assert html_report.has_section("header")
        assert "page_size" in pdf_report.get_section("header")
        assert "html" in html_report.get_section("header")

    def test_monthly_report_calculates_correctly(self):
        transactions = [
            Transaction("T1", "2024-04-01", "Thu", 1000000),
            Transaction("T2", "2024-04-02", "Chi", -500000),
        ]
        builder = PDFReportBuilder()
        director = ReportDirector(builder)
        report = director.build_monthly_report(4, 2024, transactions, [], [])

        summary = report.get_section("summary")
        assert summary["revenue"] == 1000000
        assert summary["expenses"] == 500000
        assert summary["profit"] == 500000

## Uu va nhuoc diem

| Uu diem | Nhuoc diem |
|---------|-----------|
| **Kiem soat tung buoc**: Xay dung object phuc tap theo tung buoc ro rang | **Code quantity**: Can nhieu class (Builder interface + nhieu ConcreteBuilders) |
| **Tai su dung**: Cung process xay dung cho nhieu representation | **Complexity**: Pattern co the qua phuc tap cho object don gian |
| **Single Responsibility**: Tach construction khoi business logic | **Mutable builder**: Builder thuong co state — can reset giua cac lan build |
| **Fluent interface**: Code goi ro rang, de doc, de maintain | **Director rigid**: Director co the qua cung nhac neu process thay doi |
| **Validation**: Co the validate truoc khi tra ve product | **Thread-safety**: Builder khong an toan cho da luong |
| **Immutability**: De dang tao immutable product | **Memory overhead**: Builder luu intermediate state |
| **Open/Closed**: Them representation khong sua code cu | **Learning curve**: Developer moi can hieu Builder + Director relationship |

---

## Ket luan

Builder la pattern ly tuong khi ban can xay dung object phuc tap voi nhieu buoc va nhieu bieu dien. **Golden rule**: Neu constructor cua ban co 5+ tham so (dac biet la optional parameters), hoac neu object co the duoc tao theo nhieu cach khac nhau — do la dau hieu ban can Builder.

Tôi thường nói với các bạn trong team: *"Một constructor dài quá 5 tham số là một constructor cần được giải cứu."* Builder chính là đội cứu hộ đó.

Hay nho su khac biet quan trong:
- **Factory Method / Abstract Factory**: Tao object *ngay lap tuc* (goi factory, nhan product).
- **Builder**: Tao object *qua nhieu buoc* (goi method A, method B, build).

Builder dac biet manh khi ket hop voi **Fluent Interface** (method chaining) va **Director** (dong goi quy trinh xay dung). Trong thuc te, ban thuong thay Builder trong:
- **Query builders**: SQL, Elasticsearch, Django ORM.
- **Document builders**: PDF, HTML, Excel, Word.
- **Configuration builders**: Dockerfile, Kubernetes manifests, CI/CD pipelines.

Khong phai luc nao cung can Director. Neu client code co the tu dieu khien cac buoc (va muon su linh hoat do), chi can Builder voi fluent interface la du. Director chi thuc su can khi ban muon chuan hoa quy trinh xay dung.

---

*Trân trọng!*
