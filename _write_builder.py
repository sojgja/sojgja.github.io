import os

path = r'F:\git\sojgja.github.io\docs\series\builder.md'

content = r'''---
id: builder
title: Builder
sidebar_label: \U0001f528 Builder
sidebar_position: 5
---

# Builder

> *"Separate the construction of a complex object from its representation so that the same construction process can create different representations."* — Gang of Four, *Design Patterns: Elements of Reusable Object-Oriented Software*, 1994.

**Builder** thuộc nhóm **Creational Patterns**, tách quá trình xây dựng một object phức tạp ra khỏi biểu diễn của nó. Pattern này cho phép cùng một quy trình xây dựng (construction process) tạo ra nhiều biểu diễn (representations) khác nhau. Điểm mạnh cốt lõi của Builder nằm ở chỗ nó kiểm soát được **từng bước** của quá trình xây dựng — điều mà Factory Method và Abstract Factory không làm được.
'''

content += r'''
## Bài toán chi tiết

Giả sử bạn đang xây dựng một **hệ thống tạo báo cáo tài chính tự động** cho một ngân hàng đầu tư. Mỗi ngày, hệ thống phải sinh ra hàng trăm báo cáo khác nhau: báo cáo giao dịch nội bộ, báo cáo cho khách hàng, báo cáo cho cơ quan thuế, và báo cáo kiểm toán nội bộ. Mỗi loại báo cáo có cấu trúc phức tạp với nhiều phần: header (logo, tên báo cáo, ngày giờ, mã tham chiếu), summary section (tổng quan tài chính, con số chính, các KPI), transaction table (danh sách giao dịch), charts section (biểu đồ phân tích), và footer (chữ ký số, disclaimer, QR code).

Mỗi báo cáo có thể được xuất ra **nhiều định dạng**: PDF (cho khách hàng), Excel (cho nội bộ), HTML (cho web dashboard), và JSON (cho API). Mỗi định dạng có cách render khác nhau — PDF cần layout chính xác từng pixel, Excel cần cell references và formulas, HTML cần CSS và responsive design.

Cách tiếp cận ngây thơ ban đầu là dùng một constructor khổng lồ với 20+ tham số. Điều này dẫn đến **telescoping constructor** — người dùng phải nhớ thứ tự và ý nghĩa của từng tham số. Object có thể được tạo với state không hợp lệ (thiếu header, thiếu footer). Cùng loại báo cáo nhưng khác format phải viết lại toàn bộ constructor.
'''

content += r'''
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
'''

content += r'''
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
'''

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Part 1 written: {os.path.getsize(path)} bytes')
