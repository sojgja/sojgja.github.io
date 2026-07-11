import os

path = r'F:\git\sojgja.github.io\docs\series\builder.md'

content = '''
## So do UML

```
+-----------------------------------------------+
|              <<interface>>                     |
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
'''

with open(path, 'a', encoding='utf-8') as f:
    f.write(content)

print(f'Part 3 appended. File size: {os.path.getsize(path)} bytes')
