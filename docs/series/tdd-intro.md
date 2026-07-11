---
id: tdd-intro
title: TDD — Giới thiệu và Nguyên lý
sidebar_label: 🔴 Giới thiệu TDD
sidebar_position: 55
---

# TDD — Giới thiệu và Nguyên lý

> *"Clean code that works — in 30-minute cycles."* — **Kent Beck, "Test-Driven Development: By Example", 2002**

Test-Driven Development (TDD) là một kỹ thuật phát triển phần mềm trong đó bạn viết test **trước** khi viết code production. Không phải "test sau khi code" — mà là test dẫn dắt thiết kế. Kent Beck, người đã "tái khám phá" và hệ thống hóa TDD vào đầu những năm 2000, mô tả nó như một kỹ thuật để đạt được "code sạch mà chạy được" thông qua các chu kỳ ngắn lặp đi lặp lại. TDD không chỉ là về testing — nó là về **thiết kế phần mềm**. Test trong TDD không phải mục đích cuối cùng, mà là công cụ để đạt được thiết kế tốt hơn, code an toàn hơn, và quy trình làm việc hiệu quả hơn.

## Lịch sử của TDD

TDD không ra đời trong chân không. Nó là kết quả của nhiều thập kỷ tiến hóa trong tư duy về chất lượng phần mềm:

| Thời kỳ | Sự kiện | Tác động |
|---------|---------|----------|
| **1968** | NATO Software Engineering Conference | Lần đầu tiên "software crisis" được thừa nhận — phần mềm quá phức tạp, lỗi quá nhiều |
| **1970s** | Dijkstra và kiểm chứng hình thức | Ý tưởng viết specification trước code, dùng toán học để chứng minh tính đúng đắn |
| **1980s** | Extreme Programming (XP) khởi nguồn | Kent Beck làm việc với Ward Cunningham, giới thiệu khái niệm "test-first" trong dự án Chrysler C3 |
| **1999** | *Extreme Programming Explained* — Kent Beck | Test-first là một trong 12 practice cốt lõi của XP |
| **2002** | *Test-Driven Development: By Example* — Kent Beck | TDD chính thức được định nghĩa như một kỹ thuật độc lập |
| **2003** | JUnit ra đời (Beck & Gamma) | Framework testing cho Java, tạo nền tảng cho xUnit family |
| **2005** | pyTest ra đời | Python testing framework, dần trở thành tiêu chuẩn cho Python TDD |
| **2010s** | TDD được áp dụng rộng rãi | Google, Microsoft, Amazon áp dụng TDD ở quy mô lớn |
| **2020s** | TDD trong kỷ nguyên AI | TDD vẫn là best practice — AI hỗ trợ viết test nhưng TDD cycle vẫn do developer kiểm soát |

Kent Beck thường kể câu chuyện về dự án Chrysler Comprehensive Compensation (C3) năm 1996 — dự án payroll system đầu tiên áp dụng XP và test-first. Ông nhận thấy rằng viết test trước không chỉ giúp giảm bug, mà còn **thay đổi cách thiết kế code**. Khi bạn viết test trước, bạn buộc phải suy nghĩ về interface từ góc nhìn của người dùng (client code), dẫn đến thiết kế đơn giản và trực quan hơn.

## Red-Green-Refactor: Vòng đời cơ bản của TDD

TDD xoay quanh một chu kỳ 3 bước cực kỳ đơn giản nhưng sâu sắc:

### 🔴 Red: Viết một test bị fail

Trước khi viết bất kỳ code production nào, bạn viết một test kiểm tra hành vi mong muốn. Test này phải **fail** (đèn đỏ) vì code chưa tồn tại.

```python
# test_calculator.py — Bước RED
from calculator import Calculator

def test_add_returns_sum_of_two_numbers():
    calc = Calculator()
    assert calc.add(2, 3) == 5
```

Khi chạy test này (chưa có class `Calculator`), pytest báo lỗi `ModuleNotFoundError` hoặc `AttributeError`. Đây là RED — test fail đúng như kỳ vọng.

### 🟢 Green: Viết code tối thiểu để test pass

Viết **code tối thiểu** — không hơn, không kém — chỉ đủ để test chuyển sang xanh.

```python
# calculator.py — Bước GREEN
class Calculator:
    def add(self, a, b):
        return a + b
```

Chạy lại test: `PASSED` — đèn xanh.

### 🔄 Refactor: Cải thiện code giữ nguyên hành vi

Làm sạch code: đổi tên biến, tách method, thêm type hint, loại bỏ duplication. Test vẫn xanh trong suốt quá trình.

```python
# calculator.py — Bước REFACTOR
from typing import Union

Number = Union[int, float]

class Calculator:
    def add(self, a: Number, b: Number) -> Number:
        return a + b
```

Chu kỳ này lặp lại cho từng tính năng nhỏ, từng test case, từng hành vi.

## Ba Luật của TDD (Uncle Bob)

Robert C. Martin (Uncle Bob) đã đúc kết TDD thành 3 luật bất di bất dịch:

1. **Luật 1 — Không viết code production cho đến khi có test fail**: Nếu test chưa fail, không được viết production code. Điều này đảm bảo mọi dòng code đều có lý do tồn tại.

2. **Luật 2 — Không viết nhiều hơn một test fail trong một lần**: Mỗi lần chỉ viết đủ test để fail — không viết cả loạt test rồi mới code. Chu kỳ phải ngắn, thường 30-60 giây.

3. **Luật 3 — Không viết nhiều code production hơn mức cần để pass test đang fail**: Code tối thiểu. Nếu test chỉ kiểm tra `add(2, 3) == 5`, bạn chỉ viết `return 2 + 3` cũng được, dù biết nó chưa tổng quát. Test tiếp theo sẽ buộc bạn tổng quát hóa.

Ba luật này đảm bảo test và production code phát triển song song, không có dòng code nào không được test, và thiết kế được dẫn dắt bởi nhu cầu thực tế của test.

## Tại sao TDD lại quan trọng?

### 1. Giảm mật độ bug một cách căn bản

Một nghiên cứu của Microsoft Research (2017) trên 4,000+ developer cho thấy TDD giảm mật độ bug từ 40-80% so với phát triển truyền thống. Lý do không chỉ vì có nhiều test hơn, mà vì **viết test trước buộc bạn hiểu rõ yêu cầu trước khi code**. Hầu hết bugs đến từ việc hiểu sai requirement, không phải từ lỗi cú pháp.

### 2. Thiết kế tốt hơn tự nhiên

Khi viết test trước, bạn đứng ở góc nhìn của **client code**. Bạn muốn API như thế nào? Nếu API khó dùng trong test, nó sẽ khó dùng trong production. TDD tự nhiên dẫn đến:
- **Single Responsibility**: Class khó test nếu có quá nhiều responsibility
- **Dependency Injection**: Khó test nếu class tự tạo dependency — bạn sẽ inject chúng
- **Interface segregation**: Interface lớn khó mock — bạn sẽ tách nhỏ

### 3. Regression safety net

Bạn có thể refactor mà không sợ hỏng gì. Một codebase được TDD có bộ test toàn diện cho phép bạn thay đổi kiến trút mạnh dạn. Google báo cáo rằng các team áp dụng TDD có tốc độ refactor nhanh hơn 2-3 lần so với team không áp dụng, vì họ không sợ "đụng vào code cũ".

### 4. Documentation sống

Test là documentation khả thi — nó luôn đồng bộ với code (vì nếu không, nó sẽ fail). Một developer mới vào dự án có thể đọc test để hiểu "class này làm gì", "method này mong đợi input gì", "edge cases nào được xử lý".

### 5. Giảm chi phí bảo trì

| Giai đoạn | Không TDD | Với TDD |
|-----------|-----------|---------|
| Development | Nhanh hơn ban đầu | Chậm hơn 15-30% |
| Testing/QA | Testing manual, nhiều bug | QA tập trung vào integration test |
| Production | Hotfix gấp, regression | Hotfix an toàn, test bảo vệ |
| Maintenance | Code cứng nhắc, sợ sửa | Code linh hoạt, dám refactor |

Trong dài hạn, TDD tiết kiệm 20-50% tổng chi phí bảo trì (theo nghiên cứu của Nagappan et al., IEEE 2008).

## ROI của TDD: Phân tích chi phí - lợi ích

### Chi phí của TDD

- **Chậm hơn 15-30% ở giai đoạn đầu**: Viết test trước mất thời gian. Một feature đơn giản có thể mất 30 phút thay vì 15.
- **Học curve**: Developer mới học TDD thường mất 2-4 tuần để thành thạo, trong thời gian đó tốc độ có thể giảm 40-50%.
- **Bảo trì test**: Khi requirement thay đổi, test phải được cập nhật. Đây là chi phí thực.

### Lợi ích của TDD

- **Giảm 40-90% bug trong production** (Bhat & Nagappan, IEEE 2006)
- **Giảm 50-80% thời gian debug**: Bug được phát hiện trong vài phút thay vì vài ngày
- **Giảm 35-50% chi phí bảo trì**: Code dễ thay đổi hơn
- **Tăng 25-40% năng suất dài hạn**: Dù ban đầu chậm, nhưng sau 6-12 tháng, tốc độ phát triển cao hơn hẳn

### Khi nào TDD không có ROI?

- **Prototype/Proof-of-concept**: Bạn chưa biết mình đang làm gì. TDD có thể làm chậm quá trình khám phá.
- **Script một lần**: Code chạy một lần rồi bỏ.
- **Data exploration**: Phân tích dữ liệu ad-hoc, notebook.
- **UI/UX exploration**: Khi chưa biết giao diện sẽ như thế nào.

Nhưng ngay cả trong những trường hợp này, nếu script trở thành production code (và nó thường xảy ra), bạn sẽ ước mình đã viết test từ đầu.

## Những hiểu lầm phổ biến về TDD

### "TDD là về testing"

Sai. TDD là về **thiết kế**, không phải testing. Test chỉ là công cụ. Mục tiêu của TDD là tạo ra thiết kế tốt hơn thông qua feedback loop ngắn. Kent Beck nói rõ: *"TDD is not about testing — it's about design."*

### "TDD làm chậm development"

Đúng ở ngắn hạn, sai ở dài hạn. Giống như việc đổ bê tông móng nhà — nó làm chậm quá trình xây nhà ở ngày đầu tiên, nhưng nếu không có nó, bạn sẽ không có nhà để ở sau 5 năm. Code không có test là **nợ kỹ thuật** với lãi suất kép — càng để lâu càng đắt.

### "TDD chỉ dùng cho greenfield projects"

Sai. Characterization tests (viết test cho legacy code trước khi refactor) là một trong những ứng dụng mạnh mẽ nhất của TDD. Michael Feathers dành cả cuốn *Working Effectively with Legacy Code* để nói về kỹ thuật này.

### "TDD đảm bảo không có bug"

Sai. TDD chỉ đảm bảo code làm đúng những gì test nói. Nếu test sai, code cũng sai. TDD không kiểm tra được:
- Integration issues
- Performance problems
- Security vulnerabilities
- Missing requirements

### "TDD cần phải viết test cho mọi thứ"

Không. TDD hướng dẫn bạn viết test cho **hành vi**, không phải implementation. Bạn không cần test getter/setter, private methods, hay implementation details. Test hành vi public — những gì class hứa hẹn với thế giới bên ngoài.

## TDD vs Các phương pháp testing khác

| Khía cạnh | TDD | Test-last (viết test sau) | No testing |
|-----------|-----|--------------------------|------------|
| **Thời điểm viết test** | Trước code | Sau code | Không viết |
| **Mục đích chính** | Thiết kế | Xác nhận | — |
| **Coverage** | Rất cao (80-95%+) | Phụ thuộc vào kỷ luật | 0% |
| **Chất lượng thiết kế** | Thường tốt hơn | Phụ thuộc | Không kiểm soát |
| **Tốc độ ban đầu** | Chậm | Nhanh | Rất nhanh |
| **Tốc độ dài hạn** | Nhanh | Chậm dần | Rất chậm (khi code lớn) |
| **Refactor safety** | Cao | Thấp | Không |

## Cài đặt môi trường TDD Python

Trước khi bắt đầu series này, hãy đảm bảo bạn đã cài đặt môi trường:

```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows

# Cài đặt pytest và các plugin
pip install pytest pytest-mock hypothesis pytest-bdd

# Kiểm tra
pytest --version
# pytest 8.x
```

File `pyproject.toml` khuyến nghị:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

## Kiến trúc project cho TDD

Trong series này, chúng ta sử dụng cấu trúc thư mục chuẩn:

```
project/
├── src/
│   ├── __init__.py
│   └── tdd_demo/
│       ├── __init__.py
│       └── ... (production code)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── ... (test files)
├── pyproject.toml
└── README.md
```

## Kết luận

TDD không phải là một kỹ thuật testing — nó là một **phương pháp thiết kế phần mềm** sử dụng test làm công cụ dẫn dắt. Nó đòi hỏi kỷ luật, kiên nhẫn, và thực hành. Nhưng lợi ích — code sạch, thiết kế tốt, bộ test an toàn, khả năng refactor mạnh dạn — là những thứ mà không phương pháp nào khác có thể mang lại một cách hệ thống. Trong các bài tiếp theo, chúng ta sẽ đi sâu vào từng khía cạnh của TDD: từ Red-Green-Refactor cơ bản, patterns và kỹ thuật, kết hợp với OOP và SOLID, cho đến áp dụng TDD trong dự án thực tế và các kỹ thuật nâng cao.

## Tài liệu tham khảo

- Kent Beck, *"Test-Driven Development: By Example"* (2002)
- Robert C. Martin, *"Clean Code: A Handbook of Agile Software Craftsmanship"* (2008)
- Michael Feathers, *"Working Effectively with Legacy Code"* (2004)
- Steve Freeman & Nat Pryce, *"Growing Object-Oriented Software Guided by Tests"* (2009)
- Gerard Meszaros, *"xUnit Test Patterns"* (2007)
- Nagappan et al., *"Realizing quality improvement through test driven development: results and experiences of four industrial teams"*, Empirical Software Engineering, 2008
- Bhat & Nagappan, *"Evaluating the efficacy of test-driven development"*, IEEE ISESE, 2006
