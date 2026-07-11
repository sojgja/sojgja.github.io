import os

path = r'F:\git\sojgja.github.io\docs\series\builder.md'

content = '''
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
'''

with open(path, 'a', encoding='utf-8') as f:
    f.write(content)

print(f'Part 2 appended. File size: {os.path.getsize(path)} bytes')
