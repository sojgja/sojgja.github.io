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
| **2006** | Nghiên cứu của Microsoft Research | Bằng chứng thực nghiệm đầu tiên về hiệu quả của TDD trong công nghiệp |
| **2010s** | TDD được áp dụng rộng rãi | Google, Microsoft, Amazon áp dụng TDD ở quy mô lớn |
| **2020s** | TDD trong kỷ nguyên AI | TDD vẫn là best practice — AI hỗ trợ viết test nhưng TDD cycle vẫn do developer kiểm soát |

Kent Beck thường kể câu chuyện về dự án Chrysler Comprehensive Compensation (C3) năm 1996 — dự án payroll system đầu tiên áp dụng XP và test-first. Ông nhận thấy rằng viết test trước không chỉ giúp giảm bug, mà còn **thay đổi cách thiết kế code**. Khi bạn viết test trước, bạn buộc phải suy nghĩ về interface từ góc nhìn của người dùng (client code), dẫn đến thiết kế đơn giản và trực quan hơn.

## Red-Green-Refactor: Vòng đời cơ bản của TDD

TDD xoay quanh một chu kỳ 3 bước cực kỳ đơn giản nhưng sâu sắc:

### 🔴 Red: Viết một test bị fail

Trước khi viết bất kỳ code production nào, bạn viết một test kiểm tra hành vi mong đợi. Test này phải **fail** (đèn đỏ) vì code chưa tồn tại.

```python
# test_calculator.py — Bước RED
from calculator import Calculator


def test_add_returns_sum_of_two_numbers():
    calc = Calculator()
    assert calc.add(2, 3) == 5
```

Khi chạy test này (chưa có class `Calculator`), pytest báo lỗi `ModuleNotFoundError` hoặc `AttributeError`:

```text
$ pytest test_calculator.py -v
============================= test session starts =============================
collected 0 items / 1 error

=================================== ERRORS ====================================
_________________ ERROR collecting test_calculator.py __________________
ImportError while importing 'test_calculator': No module named 'calculator'
=========================== 1 error in 0.12s ============================
```

Đây là RED — test fail đúng như kỳ vọng.

### 🟢 Green: Viết code tối thiểu để test pass

Viết **code tối thiểu** — không hơn, không kém — chỉ đủ để test chuyển sang xanh.

```python
# calculator.py — Bước GREEN
class Calculator:
    def add(self, a, b):
        return a + b
```

Chạy lại test:

```text
$ pytest test_calculator.py -v
============================= test session starts =============================
collected 1 item

test_calculator.py::test_add_returns_sum_of_two_numbers PASSED

============================== 1 passed in 0.02s ==============================
```

GREEN — đèn xanh.

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

Chạy lại test — vẫn xanh:

```text
$ pytest test_calculator.py -v
============================= test session starts =============================
collected 1 item

test_calculator.py::test_add_returns_sum_of_two_numbers PASSED

============================== 1 passed in 0.03s ==============================
```

Chu kỳ này lặp lại cho từng tính năng nhỏ, từng test case, từng hành vi. Mỗi chu kỳ thường kéo dài 30 giây đến 2 phút. Nếu chu kỳ của bạn dài hơn 5 phút, hãy chia nhỏ test — bạn đang làm sai TDD.

## Ba Luật của TDD (Uncle Bob)

Robert C. Martin (Uncle Bob) đã đúc kết TDD thành 3 luật bất di bất dịch trong cuốn *"Clean Code: A Handbook of Agile Software Craftsmanship"* (2008):

### Luật 1 — Không viết code production cho đến khi có test fail

Nếu test chưa fail, không được viết production code. Điều này đảm bảo mọi dòng code đều có lý do tồn tại. Không có "Tôi biết sau này tôi sẽ cần function này" — bạn chỉ viết code khi có test yêu cầu nó.

**Vi phạm luật 1**:
```python
# ❌ Vi phạm — code viết trước test
class Calculator:
    def subtract(self, a, b):
        return a - b  # Chưa có test nào yêu cầu! YAGNI violation

# ✅ Đúng — test trước
def test_subtract_returns_difference():
    calc = Calculator()
    assert calc.subtract(5, 3) == 2

class Calculator:
    def subtract(self, a, b):
        return a - b  # Chỉ viết sau khi có test fail
```

### Luật 2 — Không viết nhiều hơn một test fail trong một lần

Mỗi lần chỉ viết đủ test để fail — không viết cả loạt test rồi mới code. Chu kỳ phải ngắn, thường 30-60 giây.

```python
# ❌ Vi phạm — viết 3 test cùng lúc
def test_add_positive():
    assert Calculator().add(2, 3) == 5

def test_add_negative():
    assert Calculator().add(-2, -3) == -5

def test_add_zero():
    assert Calculator().add(0, 5) == 5

# ✅ Đúng — từng test một:
# Cycle 1: viết test_add_positive → code → pass → refactor
# Cycle 2: viết test_add_negative → code → pass → refactor
# Cycle 3: viết test_add_zero → code → pass → refactor
```

### Luật 3 — Không viết nhiều code production hơn mức cần để pass test đang fail

Code tối thiểu. Nếu test chỉ kiểm tra `add(2, 3) == 5`, bạn chỉ viết `return 2 + 3` cũng được, dù biết nó chưa tổng quát. Test tiếp theo sẽ buộc bạn tổng quát hóa.

```python
# Test 1
def test_add_2_plus_3():
    assert Calculator().add(2, 3) == 5

# ❌ Vi phạm luật 3 — code quá nhiều
def add(self, a, b):
    return a + b  # OK, nhưng viết sớm hơn mức cần

# ✅ Đúng — code tối thiểu
def add(self, a, b):
    return 2 + 3  # Cứng, chỉ pass test 1!

# Test 2 — buộc tổng quát hóa
def test_add_2_plus_4():
    assert Calculator().add(2, 4) == 6

# Bây giờ mới viết:
def add(self, a, b):
    return a + b  # Tổng quát
```

Ba luật này đảm bảo test và production code phát triển song song, không có dòng code nào không được test, và thiết kế được dẫn dắt bởi nhu cầu thực tế của test.

## Tại sao TDD lại quan trọng?

### 1. Giảm mật độ bug một cách căn bản

Một nghiên cứu của Microsoft Research (2008) trên 4 đội industrial team cho thấy mật độ bug giảm từ 40-90% khi áp dụng TDD so với phát triển truyền thống. Lý do không chỉ vì có nhiều test hơn, mà vì **viết test trước buộc bạn hiểu rõ yêu cầu trước khi code**. Hầu hết bugs đến từ việc hiểu sai requirement, không phải từ lỗi cú pháp.

| Nghiên cứu | Giảm bug | Năm |
|------------|----------|-----|
| Microsoft Research (Nagappan et al.) | 40-90% | 2008 |
| IBM (Bhat & Nagappan) | 40-60% | 2006 |
| University of Oulu (Ikonen et al.) | 60-80% | 2011 |
| Case study tại Sabre Airline Solutions | 40% | 2015 |

### 2. Thiết kế tốt hơn tự nhiên

Khi viết test trước, bạn đứng ở góc nhìn của **client code**. Bạn muốn API như thế nào? Nếu API khó dùng trong test, nó sẽ khó dùng trong production. TDD tự nhiên dẫn đến:

- **Single Responsibility Principle**: Class khó test nếu có quá nhiều responsibility
- **Dependency Injection**: Khó test nếu class tự tạo dependency — bạn sẽ inject chúng
- **Interface Segregation**: Interface lớn khó mock — bạn sẽ tách nhỏ
- **Law of Demeter**: Chuỗi gọi `a.b().c().d()` khó mock — bạn sẽ giảm coupling

### 3. Regression Safety Net

Bạn có thể refactor mà không sợ hỏng gì. Một codebase được TDD có bộ test toàn diện cho phép bạn thay đổi kiến trúc mạnh dạn. Google báo cáo rằng các team áp dụng TDD có tốc độ refactor nhanh hơn 2-3 lần so với team không áp dụng, vì họ không sợ "đụng vào code cũ".

### 4. Documentation sống

Test là documentation khả thi — nó luôn đồng bộ với code (vì nếu không, nó sẽ fail). Một developer mới vào dự án có thể đọc test để hiểu "class này làm gì", "method này mong đợi input gì", "edge cases nào được xử lý".

```python
# Test là documentation sống:
def test_withdraw_reduces_balance():
    account = BankAccount(Decimal("100"))
    account.withdraw(Decimal("30"))
    assert account.balance == Decimal("70")

def test_withdraw_raises_on_insufficient_funds():
    account = BankAccount(Decimal("50"))
    with pytest.raises(ValueError, match="Insufficient funds"):
        account.withdraw(Decimal("100"))

def test_withdraw_raises_on_negative_amount():
    account = BankAccount(Decimal("100"))
    with pytest.raises(ValueError, match="Amount must be positive"):
        account.withdraw(Decimal("-10"))
```

Developer mới đọc 3 test này biết ngay:
- `withdraw` giảm balance khi hợp lệ
- Không cho withdraw nếu balance không đủ
- Không cho withdraw số âm

### 5. Giảm chi phí bảo trì

Chi phí sửa bug tăng theo thời gian phát hiện:

| Giai đoạn phát hiện bug | Chi phí relative |
|------------------------|------------------|
| Trong khi code (TDD) | 1x |
| Code review | 5x |
| QA testing | 15x |
| Staging | 50x |
| Production | 100x+ |

TDD phát hiện bug ở giai đoạn sớm nhất (khi bạn vừa viết code), giảm chi phí sửa lỗi từ 50-100 lần.

### 6. Tăng tốc độ phát triển dài hạn

Nghịch lý của TDD: **ban đầu chậm hơn, nhưng về sau nhanh hơn nhiều**.

```text
Tốc độ phát triển (LOC/developer/day)
^
|   Không TDD: Khởi đầu nhanh, nhưng giảm dần do technical debt
|   /
|  /
| /
|/____________________________________> Thời gian

|   Với TDD: Khởi đầu chậm, nhưng duy trì được tốc độ
|            \
|             \
|              \
|               \______________________________> Thời gian
```

## ROI của TDD: Phân tích chi phí - lợi ích

### Chi phí của TDD

- **Chậm hơn 15-30% ở giai đoạn đầu**: Viết test trước mất thời gian. Một feature đơn giản có thể mất 30 phút thay vì 15.
- **Học curve**: Developer mới học TDD thường mất 2-4 tuần để thành thạo, trong thời gian đó tốc độ có thể giảm 40-50%.
- **Bảo trì test**: Khi requirement thay đổi, test phải được cập nhật. Đây là chi phí thực.

### Lợi ích của TDD

| KPI | Không TDD | Với TDD | Cải thiện |
|-----|-----------|---------|-----------|
| Mật độ bug (bugs/KLOC) | 2-10 | 0.5-2 | 60-80% |
| Thời gian debug trung bình | 2-4 giờ | 10-30 phút | 80-90% |
| Thời gian cho feature mới | Giảm dần (technical debt) | Ổn định | 25-40% nhanh hơn sau 6 tháng |
| Chi phí bảo trì | Cao (20-40% tổng budget) | Thấp (10-20% tổng budget) | 50% |
| Onboarding developer mới | 2-4 tháng | 1-2 tháng | 50% |

### Khi nào TDD không có ROI?

- **Prototype/Proof-of-concept**: Bạn chưa biết mình đang làm gì. TDD có thể làm chậm quá trình khám phá.
- **Script một lần**: Code chạy một lần rồi bỏ.
- **Data exploration**: Phân tích dữ liệu ad-hoc, notebook.
- **UI/UX exploration**: Khi chưa biết giao diện sẽ như thế nào.

Nhưng ngay cả trong những trường hợp này, nếu script trở thành production code (và nó thường xảy ra), bạn sẽ ước mình đã viết test từ đầu.

### Khi nào TDD đặc biệt có ROI?

- **Critical systems**: Banking, healthcare, aerospace, autonomous driving
- **Long-lived projects**: Sản phẩm tồn tại 5+ năm
- **Large teams**: 10+ developers trên cùng codebase
- **Frequently changing requirements**: Cần refactor liên tục
- **Open source libraries**: Nhiều người dùng, cần stability

## Những hiểu lầm phổ biến về TDD

### "TDD là về testing"

Sai. TDD là về **thiết kế**, không phải testing. Test chỉ là công cụ. Mục tiêu của TDD là tạo ra thiết kế tốt hơn thông qua feedback loop ngắn. Kent Beck nói rõ: *"TDD is not about testing — it's about design."*

### "TDD làm chậm development"

Đúng ở ngắn hạn, sai ở dài hạn. Giống như việc đổ bê tông móng nhà — nó làm chậm quá trình xây nhà ở ngày đầu tiên, nhưng nếu không có nó, bạn sẽ không có nhà để ở sau 5 năm. Code không có test là **nợ kỹ thuật** với lãi suất kép — càng để lâu càng đắt.

### "TDD chỉ dùng cho greenfield projects"

Sai. Characterization tests (viết test cho legacy code trước khi refactor) là một trong những ứng dụng mạnh mẽ nhất của TDD. Michael Feathers dành cả cuốn *"Working Effectively with Legacy Code"* để nói về kỹ thuật này.

### "TDD đảm bảo không có bug"

Sai. TDD chỉ đảm bảo code làm đúng những gì test nói. Nếu test sai, code cũng sai. TDD không kiểm tra được:
- Integration issues (việc kết nối các module với nhau)
- Performance problems (TDD không phải performance testing)
- Security vulnerabilities (TDD không thay thế security audit)
- Missing requirements (bạn không biết mình không biết)

### "TDD cần phải viết test cho mọi thứ"

Không. TDD hướng dẫn bạn viết test cho **hành vi**, không phải implementation. Bạn không cần test getter/setter, private methods, hay implementation details. Test hành vi public — những gì class hứa hẹn với thế giới bên ngoài.

```python
class User:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def can_access(self, resource: str) -> bool:
        """Behavior cần test: quyền truy cập resource."""
        return self._has_permission(resource)

    def _has_permission(self, resource: str) -> bool:
        """Private implementation — KHÔNG cần test trực tiếp."""
        permissions = {"admin": ["*"], "user": ["read"]}
        return resource in permissions.get("admin", [])

    @property
    def name_upper(self) -> str:
        """Derived property — cần test nếu logic phức tạp."""
        return self._name.upper()


# ✅ Test behavior
def test_user_can_access_admin_resources():
    user = User("Alice")
    assert user.can_access("admin_panel")

# ❌ Không cần test — quá đơn giản
# def test_name_property():
#     user = User("Alice")
#     assert user.name == "Alice"

# ❌ Không test private methods
# def test_has_permission():
#     user = User("Alice")
#     assert user._has_permission("admin")
```

### "Viết test trước là không tự nhiên"

Tất nhiên — nó là một kỹ năng cần học. Giống như học lái xe số sàn — ban đầu bạn phải nghĩ về từng thao tác. Nhưng sau một thời gian, nó trở thành phản xạ. Sau 2-4 tuần thực hành TDD đều đặn, hầu hết developer đều cảm thấy "không tự nhiên khi viết code trước test".

### "TDD không áp dụng được cho Python vì dynamic typing"

Hoàn toàn sai. Python là ngôn ngữ tuyệt vời cho TDD nhờ:
- pytest — một trong những testing frameworks tốt nhất
- Duck typing — dễ mock, dễ tạo fake objects
- Dynamic nature — dễ dùng patch, monkeypatch
- REPL — debug nhanh khi test fail

## TDD vs Các phương pháp testing khác

| Khía cạnh | TDD | Test-last (viết test sau) | No testing |
|-----------|-----|--------------------------|------------|
| **Thời điểm viết test** | Trước code | Sau code | Không viết |
| **Mục đích chính** | Thiết kế | Xác nhận | — |
| **Coverage** | Rất cao (80-95%+) | Phụ thuộc vào kỷ luật (thường 30-60%) | 0% |
| **Chất lượng thiết kế** | Thường tốt hơn | Phụ thuộc | Không kiểm soát |
| **Tốc độ ban đầu** | Chậm | Nhanh | Rất nhanh |
| **Tốc độ dài hạn** | Nhanh | Chậm dần | Rất chậm (khi code lớn) |
| **Refactor safety** | Cao | Thấp | Không |
| **Documentation** | Luôn đồng bộ | Lỗi thời dần | Không có |
| **Debug time** | Vài phút | Vài giờ | Vài ngày |

### TDD vs Traditional Testing

```python
# Test-last approach
def calculate_discount(price, tier):
    if tier == "vip":
        return price * 0.9
    elif tier == "wholesale":
        return price * 0.8
    return price

# Viết test sau — thường bị skip vì "no time"
# def test_calculate_discount():
#     assert calculate_discount(100, "vip") == 90

# TDD approach
# 1. Viết test TRƯỚC
def test_vip_discount():
    assert calculate_discount(100, "vip") == 90

# 2. Viết code tối thiểu
def calculate_discount(price, tier):
    return price * 0.9  # Cứng — chỉ pass vip

# 3. Thêm test cho wholesale
def test_wholesale_discount():
    assert calculate_discount(100, "wholesale") == 80

# 4. Mở rộng code
def calculate_discount(price, tier):
    if tier == "vip":
        return price * 0.9
    elif tier == "wholesale":
        return price * 0.8
    return price
```

## Cài đặt môi trường TDD Python

### Cài đặt cơ bản

```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows

# Cài đặt pytest và các plugin
pip install pytest pytest-mock hypothesis pytest-bdd pytest-asyncio mutmut

# Kiểm tra
pytest --version
# pytest 8.x
```

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "tdd-workshop"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"
filterwarnings = ["error"]

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
show_missing = true
fail_under = 80
```

### Cấu trúc thư mục

Trong series này, chúng ta sử dụng cấu trúc thư mục chuẩn (còn gọi là "flat layout" hoặc "src layout"):

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
├── requirements.txt
└── README.md
```

### Các lệnh cơ bản

```bash
# Chạy tất cả test
pytest

# Chạy với verbose output
pytest -v

# Chạy specific test file
pytest tests/test_calculator.py

# Chạy test với keyword filter
pytest -k "discount"

# Chạy và dừng ở test fail đầu tiên
pytest -x

# Chạy test kèm coverage
pip install pytest-cov
pytest --cov=src

# Chạy test song song
pip install pytest-xdist
pytest -n auto
```

## Những câu hỏi thường gặp (FAQs)

### "Làm sao để viết test cho UI?"

UI testing có thể dùng:
- **Selenium/Playwright** cho browser tests
- **pytest-qt** cho PyQt/PySide
- **Tkinter testing** với mô phỏng sự kiện

Tuy nhiên, nguyên tắc TDD vẫn giữ: tập trung vào business logic, UI càng mỏng càng tốt. Dùng pattern như **Model-View-Presenter** hoặc **MVVM** để tách UI khỏi logic.

### "Làm sao để viết test cho database?"

```python
# Dùng Fake Repository (in-memory) thay vì DB thật
class FakeUserRepository:
    def __init__(self):
        self._users = {}

    def save(self, user):
        self._users[user.id] = user

    def find_by_id(self, user_id):
        return self._users.get(user_id)

# Dùng test database cho integration tests
@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()
```

### "Làm sao để viết test cho external APIs?"

```python
from unittest.mock import patch, Mock

@patch("requests.post")
def test_api_call(mock_post):
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {"status": "success"}
    )
    result = my_service.call_api()
    assert result["status"] == "success"
```

### "Làm sao để viết test cho random data?"

```python
# 1. Inject random generator
class DiceRoller:
    def __init__(self, random_gen=None):
        self._random = random_gen or random

    def roll(self):
        return self._random.randint(1, 6)

# Test với seed
def test_dice_roller_with_seed():
    fixed_random = random.Random(42)  # Seed cố định
    roller = DiceRoller(fixed_random)
    results = [roller.roll() for _ in range(1000)]
    assert 1 <= min(results) <= 6
    assert 1 <= max(results) <= 6

# 2. Dùng Hypothesis property-based testing
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=6))
def test_dice_range(roll):
    assert 1 <= roll <= 6
```

## Kết luận

TDD không phải là một kỹ thuật testing — nó là một **phương pháp thiết kế phần mềm** sử dụng test làm công cụ dẫn dắt. Nó đòi hỏi kỷ luật, kiên nhẫn, và thực hành. Nhưng lợi ích — code sạch, thiết kế tốt, bộ test an toàn, khả năng refactor mạnh dạn — là những thứ mà không phương pháp nào khác có thể mang lại một cách hệ thống.

Tóm lại, TDD là:
- 🔴 **Red**: Viết test fail trước
- 🟢 **Green**: Viết code tối thiểu để pass
- 🔄 **Refactor**: Cải thiện code với test xanh

Trong các bài tiếp theo, chúng ta sẽ đi sâu vào từng khía cạnh của TDD:
1. **Bài 2**: TDD Cơ bản — String Calculator step-by-step
2. **Bài 3**: TDD Patterns — Test doubles, FIRST principles
3. **Bài 4**: TDD & SOLID — Testable OOP design
4. **Bài 5**: TDD Real-world — E-commerce system
5. **Bài 6**: TDD Nâng cao — Legacy, Hypothesis, BDD, Async

## Tài liệu tham khảo

- Kent Beck, *"Test-Driven Development: By Example"* (2002) — Cuốn sách kinh điển, nền tảng của TDD
- Robert C. Martin, *"Clean Code: A Handbook of Agile Software Craftsmanship"* (2008) — 3 laws of TDD
- Robert C. Martin, *"The Clean Coder: A Code of Conduct for Professional Programmers"* (2011) — Professionalism trong TDD
- Michael Feathers, *"Working Effectively with Legacy Code"* (2004) — TDD cho legacy code
- Steve Freeman & Nat Pryce, *"Growing Object-Oriented Software Guided by Tests"* (2009) — TDD kết hợp OOP
- Gerard Meszaros, *"xUnit Test Patterns"* (2007) — Catalog đầy đủ về test patterns
- Nagappan et al., *"Realizing quality improvement through test driven development: results and experiences of four industrial teams"*, Empirical Software Engineering, 2008 — Nghiên cứu về hiệu quả TDD
- Bhat & Nagappan, *"Evaluating the efficacy of test-driven development"*, IEEE ISESE, 2006 — Nghiên cứu thực nghiệm đầu tiên về TDD
- Martin Fowler, *"Mocks Aren't Stubs"* (2007) — https://martinfowler.com/articles/mocksArentStubs.html
- Roy Osherove, *"The Art of Unit Testing"* (2013) — Unit testing practices
- Emily Bache, *"The Coding Dojo Handbook"* (2013) — TDD katas
