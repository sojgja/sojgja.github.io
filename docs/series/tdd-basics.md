---
id: tdd-basics
title: TDD Cơ bản — Red-Green-Refactor
sidebar_label: 🔴 TDD Cơ bản
sidebar_position: 56
---

# TDD Cơ bản — Red-Green-Refactor

> *"The secret to getting good at TDD is to do it. A lot. On small things. Until it becomes reflex."* — **Kent Beck**

Bài này là một TDD session **thực chiến** — chúng ta sẽ xây dựng một **String Calculator** từ đầu đến cuối, theo đúng chu kỳ Red-Green-Refactor, từng bước một. Mỗi bước đều có code, test, và kết quả chạy. Bạn có thể chạy theo trên máy của mình.

## Bài toán: String Calculator

String Calculator là một classic TDD kata (bài tập luyện) do Roy Osherove phổ biển. Yêu cầu:

1. Calculator nhận một string gồm các số phân cách bởi dấu phẩy, trả về tổng
2. Hỗ trợ số lượng số không giới hạn
3. Hỗ trợ newline làm delimiter
4. Hỗ trợ custom delimiter (dạng `//[delimiter]\n[numbers]`)
5. Ném exception khi gặp số âm — liệt kê tất cả số âm trong message
6. Bỏ qua số >1000
7. Delimiter có thể dài hơn 1 ký tự
8. Hỗ trợ multiple delimiters

Chúng ta sẽ implement từng yêu cầu một, theo TDD cycle.

## Cấu trúc project

```
string_calculator/
├── src/
│   ├── __init__.py
│   └── calculator.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_calculator.py
└── pyproject.toml
```

## Bước 1: RED — Test cơ bản đầu tiên

Chúng ta chưa có file `calculator.py`. Hãy viết test trước:

```python
# tests/test_calculator.py
from src.calculator import StringCalculator


class TestStringCalculator:
    def test_add_returns_0_for_empty_string(self):
        calc = StringCalculator()
        assert calc.add("") == 0
```

Chạy test:

```text
$ pytest -v
============================= test session starts =============================
collected 1 item

tests/test_calculator.py FAILED                                          [100%]

================================== FAILURES ===================================
___________ TestStringCalculator.test_add_returns_0_for_empty_string __________
...
ModuleNotFoundError: No module named 'src.calculator'
```

RED — như mong đợi.

## Bước 2: GREEN — Code tối thiểu

```python
# src/calculator.py
class StringCalculator:
    def add(self, numbers: str) -> int:
        return 0
```

Chạy lại:

```text
$ pytest -v
============================= test session starts =============================
collected 1 item

tests/test_calculator.py PASSED                                          [100%]

============================== 1 passed in 0.02s ==============================
```

GREEN. Code tối thiểu — chỉ `return 0` — vì test chưa yêu cầu gì hơn.

## Bước 3: RED — Test với một số

```python
    def test_add_returns_the_number_itself_for_single_number(self):
        calc = StringCalculator()
        assert calc.add("1") == 1
```

Chạy:

```text
$ pytest -v
...
FAILED — assert 0 == 1
```

RED.

## Bước 4: GREEN — Mở rộng code

```python
class StringCalculator:
    def add(self, numbers: str) -> int:
        if numbers == "":
            return 0
        return int(numbers)
```

Chạy: GREEN. Cả 2 test pass.

## Bước 5: REFACTOR

Code vẫn đơn giản, chưa cần refactor. Hãy thêm type hints:

```python
class StringCalculator:
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        return int(numbers)
```

Test vẫn xanh.

## Bước 6: RED — Test với hai số

```python
    def test_add_returns_sum_for_two_numbers(self):
        calc = StringCalculator()
        assert calc.add("1,2") == 3
```

RED: `ValueError: invalid literal for int() with base 10: '1,2'`

## Bước 7: GREEN — Parse delimiter

```python
class StringCalculator:
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        parts = numbers.split(",")
        return sum(int(n) for n in parts)
```

GREEN.

## Bước 8: RED — Nhiều số

```python
    def test_add_returns_sum_for_multiple_numbers(self):
        calc = StringCalculator()
        assert calc.add("1,2,3,4,5") == 15
```

GREEN ngay — code hiện tại đã xử lý số lượng không giới hạn. Đây là TDD: bạn chỉ viết test cho behavior, code tự động pass nếu đã đúng.

## Bước 9: RED — Newline làm delimiter

```python
    def test_add_supports_newline_as_delimiter(self):
        calc = StringCalculator()
        assert calc.add("1\n2,3") == 6
```

RED: `1\n2,3` không được split bởi dấu phẩy.

## Bước 10: GREEN — Hỗ trợ newline

```python
import re

class StringCalculator:
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        parts = re.split(r"[,\n]", numbers)
        return sum(int(n) for n in parts)
```

GREEN.

## Bước 11: REFACTOR

Tách việc parse delimiter ra method riêng:

```python
import re
from typing import List


class StringCalculator:
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        parts = self._split_numbers(numbers)
        return sum(int(n) for n in parts)

    def _split_numbers(self, numbers: str) -> List[str]:
        return re.split(r"[,\n]", numbers)
```

Test vẫn xanh:

```text
$ pytest -v
============================= test session starts =============================
collected 4 items

tests/test_calculator.py::TestStringCalculator::test_add_returns_0_for_empty_string PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_the_number_itself_for_single_number PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_sum_for_two_numbers PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_newline_as_delimiter PASSED

============================== 4 passed in 0.03s ==============================
```

## Bước 12: RED — Custom delimiter

```python
    def test_add_supports_custom_delimiter(self):
        calc = StringCalculator()
        assert calc.add("//;\n1;2") == 3
```

RED.

## Bước 13: GREEN — Parse custom delimiter

```python
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        delimiter = ","
        if numbers.startswith("//"):
            delimiter, numbers = numbers[2], numbers[4:]
        parts = re.split(f"[{delimiter}\n]", numbers)
        return sum(int(n) for n in parts)
```

GREEN.

## Bước 14: REFACTOR

Tách logic xử lý delimiter:

```python
class StringCalculator:
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        numbers, delimiters = self._parse_delimiters(numbers)
        parts = self._split_numbers(numbers, delimiters)
        return sum(self._to_int(n) for n in parts)

    def _parse_delimiters(self, numbers: str):
        if numbers.startswith("//"):
            delimiter = numbers[2]
            return numbers[4:], delimiter
        return numbers, ","

    def _split_numbers(self, numbers: str, delimiter: str) -> List[str]:
        return re.split(f"[{delimiter}\n]", numbers)

    @staticmethod
    def _to_int(n: str) -> int:
        return int(n)
```

Test vẫn xanh.

## Bước 15: RED — Số âm bị cấm

```python
    def test_add_raises_on_negative_numbers(self):
        calc = StringCalculator()
        with pytest.raises(ValueError, match="negatives not allowed: -1"):
            calc.add("1,-1,2")
```

RED.

## Bước 16: GREEN — Validate số âm

```python
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        numbers, delimiters = self._parse_delimiters(numbers)
        parts = self._split_numbers(numbers, delimiters)
        values = [self._to_int(n) for n in parts]
        self._check_negatives(values)
        return sum(values)

    @staticmethod
    def _check_negatives(values: List[int]) -> None:
        negatives = [v for v in values if v < 0]
        if negatives:
            raise ValueError(f"negatives not allowed: {', '.join(str(n) for n in negatives)}")
```

GREEN.

## Bước 17: RED — Nhiều số âm

```python
    def test_add_raises_on_multiple_negatives(self):
        calc = StringCalculator()
        with pytest.raises(ValueError, match="negatives not allowed: -2, -5"):
            calc.add("-2,3,-5")
```

GREEN ngay — code đã xử lý từ bước 16.

## Bước 18: RED — Bỏ qua số >1000

```python
    def test_add_ignores_numbers_over_1000(self):
        calc = StringCalculator()
        assert calc.add("1001,2") == 2
```

RED.

## Bước 19: GREEN — Filter số >1000

```python
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        numbers, delimiters = self._parse_delimiters(numbers)
        parts = self._split_numbers(numbers, delimiters)
        values = [self._to_int(n) for n in parts if self._to_int(n) <= 1000]
        self._check_negatives(values)
        return sum(values)
```

GREEN.

## Bước 20: REFACTOR

Tối ưu — tránh gọi `_to_int` hai lần:

```python
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        numbers, delimiters = self._parse_delimiters(numbers)
        parts = self._split_numbers(numbers, delimiters)
        values = [self._to_int(n) for n in parts]
        self._check_negatives(values)
        return sum(v for v in values if v <= 1000)
```

Test vẫn xanh:

```text
$ pytest -v
============================= test session starts =============================
collected 8 items

tests/test_calculator.py::TestStringCalculator::test_add_returns_0_for_empty_string PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_the_number_itself_for_single_number PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_sum_for_two_numbers PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_newline_as_delimiter PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_custom_delimiter PASSED
tests/test_calculator.py::TestStringCalculator::test_add_raises_on_negative_numbers PASSED
tests/test_calculator.py::TestStringCalculator::test_add_raises_on_multiple_negatives PASSED
tests/test_calculator.py::TestStringCalculator::test_add_ignores_numbers_over_1000 PASSED

============================== 8 passed in 0.04s ==============================
```

## Bước 21: RED — Delimiter dài hơn 1 ký tự

```python
    def test_add_supports_long_delimiter(self):
        calc = StringCalculator()
        assert calc.add("//[***]\n1***2***3") == 6
```

RED.

## Bước 22: GREEN — Hỗ trợ delimiter dài

```python
import re
from typing import List, Tuple


class StringCalculator:
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        numbers, delimiters = self._parse_delimiters(numbers)
        parts = self._split_numbers(numbers, delimiters)
        values = [self._to_int(n) for n in parts]
        self._check_negatives(values)
        return sum(v for v in values if v <= 1000)

    def _parse_delimiters(self, numbers: str) -> Tuple[str, str]:
        default = ","
        if not numbers.startswith("//"):
            return numbers, default
        if numbers[2] == "[":
            delimiter = numbers[3:numbers.index("]")]
            return numbers[numbers.index("]") + 2:], delimiter
        delimiter = numbers[2]
        return numbers[4:], delimiter

    def _split_numbers(self, numbers: str, delimiter: str) -> List[str]:
        escaped = re.escape(delimiter)
        pattern = f"{escaped}|\n"
        return re.split(pattern, numbers)

    @staticmethod
    def _to_int(n: str) -> int:
        return int(n)

    @staticmethod
    def _check_negatives(values: List[int]) -> None:
        negatives = [v for v in values if v < 0]
        if negatives:
            msg = ", ".join(str(n) for n in negatives)
            raise ValueError(f"negatives not allowed: {msg}")
```

GREEN.

## Bước 23: RED — Nhiều custom delimiters

```python
    def test_add_supports_multiple_custom_delimiters(self):
        calc = StringCalculator()
        assert calc.add("//[*][%]\n1*2%3") == 6
```

RED.

## Bước 24: GREEN — Multiple delimiters

```python
import re
from typing import List, Tuple


class StringCalculator:
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        numbers, delimiters = self._parse_delimiters(numbers)
        parts = self._split_numbers(numbers, delimiters)
        values = [self._to_int(n) for n in parts]
        self._check_negatives(values)
        return sum(v for v in values if v <= 1000)

    def _parse_delimiters(self, numbers: str) -> Tuple[str, List[str]]:
        if not numbers.startswith("//"):
            return numbers, [","]
        delimiters = re.findall(r"\[([^\]]+)\]", numbers[2:])
        if delimiters:
            return numbers[numbers.rindex("]") + 2:], delimiters
        return numbers[4:], [numbers[2]]

    def _split_numbers(self, numbers: str, delimiters: List[str]) -> List[str]:
        escaped = [re.escape(d) for d in delimiters]
        pattern = "|".join(escaped) + r"|\n"
        return re.split(pattern, numbers)

    @staticmethod
    def _to_int(n: str) -> int:
        return int(n)

    @staticmethod
    def _check_negatives(values: List[int]) -> None:
        negatives = [v for v in values if v < 0]
        if negatives:
            msg = ", ".join(str(n) for n in negatives)
            raise ValueError(f"negatives not allowed: {msg}")
```

GREEN:

```text
$ pytest -v
============================= test session starts =============================
collected 10 items

tests/test_calculator.py::TestStringCalculator::test_add_returns_0_for_empty_string PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_the_number_itself_for_single_number PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_sum_for_two_numbers PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_newline_as_delimiter PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_custom_delimiter PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_long_delimiter PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_multiple_custom_delimiters PASSED
tests/test_calculator.py::TestStringCalculator::test_add_raises_on_negative_numbers PASSED
tests/test_calculator.py::TestStringCalculator::test_add_raises_on_multiple_negatives PASSED
tests/test_calculator.py::TestStringCalculator::test_add_ignores_numbers_over_1000 PASSED

============================== 10 passed in 0.05s ==============================
```

## Bước 25: REFACTOR cuối cùng

Hoàn thiện code với type hints đầy đủ:

```python
"""
String Calculator — TDD Kata implementation.

Supports:
- Comma and newline delimiters
- Custom single and multi-character delimiters
- Multiple delimiters
- Negative number validation
- Ignoring numbers > 1000
"""

import re
from typing import List, Tuple


class StringCalculator:
    """A calculator that parses and sums numbers from a string."""

    DEFAULT_DELIMITERS = [","]

    def add(self, numbers: str) -> int:
        """Add numbers from a formatted string.

        Args:
            numbers: Formatted string of numbers.

        Returns:
            Sum of parsed numbers.

        Raises:
            ValueError: If negative numbers are found.
        """
        if not numbers:
            return 0
        numbers, delimiters = self._parse_delimiters(numbers)
        parts = self._split_numbers(numbers, delimiters)
        values = [self._to_int(n) for n in parts]
        self._check_negatives(values)
        return sum(v for v in values if v <= 1000)

    def _parse_delimiters(self, numbers: str) -> Tuple[str, List[str]]:
        if not numbers.startswith("//"):
            return numbers, self.DEFAULT_DELIMITERS.copy()
        delimiters = re.findall(r"\[([^\]]+)\]", numbers[2:])
        if delimiters:
            header_end = numbers.rindex("]") + 2
            return numbers[header_end:], delimiters
        return numbers[4:], [numbers[2]]

    def _split_numbers(self, numbers: str, delimiters: List[str]) -> List[str]:
        escaped = [re.escape(d) for d in delimiters]
        pattern = "|".join(escaped) + r"|\n"
        return re.split(pattern, numbers)

    @staticmethod
    def _to_int(n: str) -> int:
        if n == "":
            return 0
        return int(n)

    @staticmethod
    def _check_negatives(values: List[int]) -> None:
        negatives = [v for v in values if v < 0]
        if negatives:
            msg = ", ".join(str(n) for n in negatives)
            raise ValueError(f"negatives not allowed: {msg}")
```

Test vẫn xanh. Code sạch, type-safe, tested toàn diện.

## Tổng kết: String Calculator hoàn chỉnh

Toàn bộ test file:

```python
# tests/test_calculator.py
import pytest
from src.calculator import StringCalculator


class TestStringCalculator:
    """Test suite for StringCalculator using TDD approach."""

    def test_add_returns_0_for_empty_string(self):
        calc = StringCalculator()
        assert calc.add("") == 0

    def test_add_returns_the_number_itself_for_single_number(self):
        calc = StringCalculator()
        assert calc.add("1") == 1

    def test_add_returns_sum_for_two_numbers(self):
        calc = StringCalculator()
        assert calc.add("1,2") == 3

    def test_add_returns_sum_for_multiple_numbers(self):
        calc = StringCalculator()
        assert calc.add("1,2,3,4,5") == 15

    def test_add_supports_newline_as_delimiter(self):
        calc = StringCalculator()
        assert calc.add("1\n2,3") == 6

    def test_add_supports_custom_delimiter(self):
        calc = StringCalculator()
        assert calc.add("//;\n1;2") == 3

    def test_add_supports_long_delimiter(self):
        calc = StringCalculator()
        assert calc.add("//[***]\n1***2***3") == 6

    def test_add_supports_multiple_custom_delimiters(self):
        calc = StringCalculator()
        assert calc.add("//[*][%]\n1*2%3") == 6

    def test_add_raises_on_negative_numbers(self):
        calc = StringCalculator()
        with pytest.raises(ValueError, match="negatives not allowed: -1"):
            calc.add("1,-1,2")

    def test_add_raises_on_multiple_negatives(self):
        calc = StringCalculator()
        with pytest.raises(ValueError, match="negatives not allowed: -2, -5"):
            calc.add("-2,3,-5")

    def test_add_ignores_numbers_over_1000(self):
        calc = StringCalculator()
        assert calc.add("1001,2") == 2

    def test_add_handles_1000_as_valid(self):
        calc = StringCalculator()
        assert calc.add("1000,2") == 1002
```

## Kết quả cuối cùng

```text
$ pytest -v --tb=short
============================= test session starts =============================
collected 12 items

tests/test_calculator.py::TestStringCalculator::test_add_returns_0_for_empty_string PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_the_number_itself_for_single_number PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_sum_for_two_numbers PASSED
tests/test_calculator.py::TestStringCalculator::test_add_returns_sum_for_multiple_numbers PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_newline_as_delimiter PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_custom_delimiter PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_long_delimiter PASSED
tests/test_calculator.py::TestStringCalculator::test_add_supports_multiple_custom_delimiters PASSED
tests/test_calculator.py::TestStringCalculator::test_add_raises_on_negative_numbers PASSED
tests/test_calculator.py::TestStringCalculator::test_add_raises_on_multiple_negatives PASSED
tests/test_calculator.py::TestStringCalculator::test_add_ignores_numbers_over_1000 PASSED
tests/test_calculator.py::TestStringCalculator::test_add_handles_1000_as_valid PASSED

============================== 12 passed in 0.06s ==============================
```

## Những bài học từ String Calculator

### Lesson 1: Baby steps

Mỗi bước trong TDD session này chỉ thay đổi một lượng nhỏ code. Test nhỏ → code nhỏ → refactor nhỏ. Chu kỳ 30-60 giây. Nếu test của bạn mất 10 phút để viết và code mất 30 phút để pass — bạn đang làm sai. Hãy chia nhỏ.

### Lesson 2: Code tối thiểu

Ở bước 1, chúng ta chỉ viết `return 0`. Dù biết rằng sau này sẽ phức tạp hơn, nhưng nguyên tắc TDD là chỉ viết code cho test hiện tại. Không dự đoán tương lai. YAGNI (You Ain't Gonna Need It).

### Lesson 3: Test behavior, not implementation

Các test của chúng ta kiểm tra hành vi của `add()`, không kiểm tra method private `_split_numbers()`. Điều này cho phép refactor tự do — chúng ta có thể thay đổi implementation mà không cần sửa test.

### Lesson 4: Refactor là bước bắt buộc

Sau mỗi GREEN, chúng ta dừng lại để refactor. Nếu không có bước này, code sẽ trở nên lộn xộn sau nhiều iteration. Refactor với test xanh là an toàn.

## Thực hành thêm: Bài tập tự làm

Hãy tự implement các bài toán sau bằng TDD, không đọc solution:

1. **FizzBuzz**: Nhập n, in ra list từ 1 đến n, thay số chia hết cho 3 bằng "Fizz", cho 5 bằng "Buzz", cho cả 3 và 5 bằng "FizzBuzz"
2. **Fibonacci**: Tính số Fibonacci thứ n (cả đệ quy và không đệ quy)
3. **Roman Numerals**: Chuyển đổi giữa số thập phân và số La Mã
4. **Shopping Cart**: Tính tổng tiền giỏ hàng với thuế, giảm giá, phí ship

Mỗi bài đều phải tuân thủ chu kỳ Red-Green-Refactor. Đừng viết code trước khi có test fail!

## Kết luận

String Calculator là bài tập TDD kinh điển vì nó đủ nhỏ để hoàn thành trong 20-30 phút nhưng đủ phức tạp để thể hiện mọi khía cạnh của TDD cycle. Qua 25+ bước, chúng ta đã thấy:
- Test dẫn dắt thiết kế
- Code chỉ được viết khi có test yêu cầu
- Mỗi tính năng được thêm vào từng bước nhỏ
- Refactor được thực hiện sau mỗi GREEN

Trang tiếp theo sẽ giới thiệu các TDD patterns và kỹ thuật nâng cao hơn.

## Tài liệu tham khảo

- Roy Osherove, *"The Art of Unit Testing"* (2013)
- TDD Kata: *"String Calculator"* — https://osherove.com/tdd-kata-1
- Emily Bache, *"The Coding Dojo Handbook"* (2013)
