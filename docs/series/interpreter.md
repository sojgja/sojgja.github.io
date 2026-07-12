---
id: interpreter
title: Interpreter
sidebar_label: 📖 Interpreter
sidebar_position: 16
---

# Interpreter

> **Interpreter** — *"Given a language, define a representation for its grammar along with an interpreter that uses the representation to interpret sentences in the language."* — GoF, 1994

## Bài toán chi tiết

Hãy tưởng tượng bạn xây một hệ thống lọc dữ liệu IoT. Hàng ngàn cảm biến gửi dữ liệu về: nhiệt độ, độ ẩm, áp suất, rung động. Kỹ sư vận hành cần tạo các bộ lọc như: `(temperature > 85 AND pressure > 100) OR (vibration > 5.0 AND NOT zone == "safe")`.

**Vấn đề đầu tiên:** Hard-code các rule này vào nguồn là không khả thi. Rule thay đổi hàng ngày, cần được nhập từ UI và lưu trong database. Dùng `eval()` là cực kỳ nguy hiểm — một lỗ hổng bảo mật chết người.

**Vấn đề thứ hai:** Mở rộng ngữ pháp. Hệ thống cần hỗ trợ thêm operator mới: `contains`, `between`, `regex_match`. Với parser thủ công, thêm operator yêu cầu sửa logic ở nhiều chỗ.

**Vấn đề thứ ba:** Hiệu năng. Hàng triệu data point mỗi giây, mỗi data point kiểm tra với hàng trăm rule. AST interpreter phải đủ nhanh.

**Vấn đề cuối cùng:** Validation. Người dùng nhập rule qua UI — cần kiểm tra ngay tại client xem rule có hợp lệ không. Interpreter pattern hỗ trợ `validate()` riêng biệt với `interpret()`.

## Giải pháp với Pattern

Interpreter pattern định nghĩa ngữ pháp bằng cách biểu diễn mỗi production rule thành một class. Có hai loại: **TerminalExpression** (lá — không chứa expression con) và **NonterminalExpression** (nút — chứa expression con). Client xây dựng AST từ chuỗi đầu vào bằng parser (thường là recursive descent parser), sau đó gọi `interpret(context)` trên root node.

**Cấu trúc pattern:**
- **AbstractExpression** (Expression): interface với `interpret(context)`.
- **TerminalExpression**: implement interpret dựa trên dữ liệu context (ví dụ: kiểm tra sensor value).
- **NonterminalExpression**: `AndExpression`, `OrExpression`, `NotExpression`, `ComparisonExpression`.
- **Context**: chứa dữ liệu đầu vào (sensor reading) được truyền vào interpret.
- **Parser**: xây AST từ chuỗi đầu vào.

**Pattern này giải quyết:**
- **Safety**: Không dùng eval. Mỗi node trong AST là object an toàn.
- **Extensibility**: Thêm operator mới bằng class mới.
- **Composability**: Expression lồng nhau vô hạn.
- **Validation**: Dễ dàng thêm `validate()` kiểm tra type, range, operator.

## Phân tích thiết kế

**OOP Principles:**
- **Composite pattern con**: Interpreter về cấu trúc giống Composite — cây với leaf và non-leaf.
- **Single Responsibility (SRP)**: Mỗi class chỉ implement đúng một production rule.
- **Open/Closed (OCP)**: Thêm operator mới = thêm class mới, không sửa class cũ.
- **Recursive Composition**: Dùng đệ quy — elegance của pattern.

**Trade-offs:**
- **Class explosion**: Ngữ pháp phức tạp (10+ rules) dẫn đến quá nhiều class.
- **Performance**: AST interpreter chậm hơn code compiled.
- **Maintenance**: Grammar thay đổi thường xuyên = chi phí bảo trì cao.

**Khi không nên dùng:**
- Ngữ pháp quá phức tạp (hàng trăm rule) — dùng parser generator (ANTLR, Lark).
- Cần hiệu năng cực cao — compile rule thành Python function.
- Ngữ pháp đơn giản (1–2 operator) — `if-else` đủ dùng.

## Ví dụ code hoàn chỉnh

### Cách làm sai: Dùng eval

```python
from __future__ import annotations
from typing import Any
import operator


class UnsafeRuleEngine:
    """Dùng eval — cực kỳ nguy hiểm, không kiểm soát được."""

    def evaluate(self, rule: str, data: dict[str, Any]) -> bool:
        # Ví dụ rule: "(data['temperature'] > 85 and data['pressure'] > 100)"
        # Nguy cơ: eval("__import__('os').system('rm -rf /')") — RCE!
        return bool(eval(rule, {"data": data, "__builtins__": {}}))

    # Không thể extend operator, không thể debug, không thể validate
```

### Cách làm đúng: Interpreter Pattern

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import re
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


# --- Context ---

@dataclass
class SensorData:
    """Dữ liệu từ cảm biến — context cho interpret."""
    temperature: float = 25.0
    pressure: float = 100.0
    vibration: float = 0.5
    humidity: float = 50.0
    zone: str = "safe"
    status: str = "normal"
    timestamp: int = 0

    def get(self, field: str) -> Any:
        return getattr(self, field, None)


# --- Abstract Expression ---

class Expression(ABC):
    """Interface cho mọi node trong AST."""

    @abstractmethod
    def interpret(self, context: SensorData) -> bool:
        ...

    def validate(self) -> list[str]:
        """Kiểm tra tính hợp lệ của expression tree."""
        return []


# --- Terminal Expressions ---

class FieldExpression(Expression):
    """Lấy giá trị của một field từ context."""
    def __init__(self, field_name: str) -> None:
        self.field_name = field_name

    def interpret(self, context: SensorData) -> bool:
        raise TypeError("FieldExpression không thể interpret trực tiếp — dùng trong comparison")

    def __repr__(self) -> str:
        return f"Field({self.field_name})"


class LiteralExpression(Expression):
    """Giá trị hằng số."""
    def __init__(self, value: Any) -> None:
        self.value = value

    def interpret(self, context: SensorData) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"Literal({self.value})"


# --- Comparison Operators ---

class ComparisonOp(Enum):
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    CONTAINS = "contains"
    MATCHES = "matches"
    BETWEEN = "between"


class ComparisonExpression(Expression):
    """So sánh field với literal hoặc field khác."""
    def __init__(
        self,
        field: str,
        op: ComparisonOp,
        value: Any,
    ) -> None:
        self.field = field
        self.op = op
        self.value = value

    def interpret(self, context: SensorData) -> bool:
        field_val = context.get(self.field)
        if field_val is None:
            logger.warning(f"Field '{self.field}' not found in context")
            return False

        try:
            match self.op:
                case ComparisonOp.GT:
                    return field_val > self.value
                case ComparisonOp.GE:
                    return field_val >= self.value
                case ComparisonOp.LT:
                    return field_val < self.value
                case ComparisonOp.LE:
                    return field_val <= self.value
                case ComparisonOp.EQ:
                    return field_val == self.value
                case ComparisonOp.NE:
                    return field_val != self.value
                case ComparisonOp.CONTAINS:
                    return str(self.value) in str(field_val)
                case ComparisonOp.MATCHES:
                    return bool(re.search(str(self.value), str(field_val)))
                case ComparisonOp.BETWEEN:
                    if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                        raise ValueError("BETWEEN requires [min, max]")
                    return self.value[0] <= field_val <= self.value[1]
                case _:
                    raise ValueError(f"Unknown operator: {self.op}")
        except TypeError as e:
            logger.error(f"Type error comparing {field_val} {self.op.value} {self.value}: {e}")
            return False

    def validate(self) -> list[str]:
        errors = []
        op_name = self.op.value if isinstance(self.op, ComparisonOp) else str(self.op)
        if self.op == ComparisonOp.BETWEEN:
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                errors.append(f"BETWEEN requires [min, max], got {self.value}")
        return errors

    def __repr__(self) -> str:
        return f"{self.field} {self.op.value} {self.value}"


# --- Logical Operators (Nonterminal Expressions) ---

class AndExpression(Expression):
    """Logical AND."""
    def __init__(self, left: Expression, right: Expression) -> None:
        self.left = left
        self.right = right

    def interpret(self, context: SensorData) -> bool:
        return self.left.interpret(context) and self.right.interpret(context)

    def validate(self) -> list[str]:
        return self.left.validate() + self.right.validate()

    def __repr__(self) -> str:
        return f"({self.left} AND {self.right})"


class OrExpression(Expression):
    """Logical OR."""
    def __init__(self, left: Expression, right: Expression) -> None:
        self.left = left
        self.right = right

    def interpret(self, context: SensorData) -> bool:
        return self.left.interpret(context) or self.right.interpret(context)

    def validate(self) -> list[str]:
        return self.left.validate() + self.right.validate()

    def __repr__(self) -> str:
        return f"({self.left} OR {self.right})"


class NotExpression(Expression):
    """Logical NOT."""
    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def interpret(self, context: SensorData) -> bool:
        return not self.expr.interpret(context)

    def validate(self) -> list[str]:
        return self.expr.validate()

    def __repr__(self) -> str:
        return f"NOT ({self.expr})"


# --- XorExpression (mở rộng) ---

class XorExpression(Expression):
    """Logical XOR — ví dụ mở rộng ngữ pháp."""
    def __init__(self, left: Expression, right: Expression) -> None:
        self.left = left
        self.right = right

    def interpret(self, context: SensorData) -> bool:
        return self.left.interpret(context) != self.right.interpret(context)

    def __repr__(self) -> str:
        return f"({self.left} XOR {self.right})"


# --- Parser (Recursive Descent) ---

class RuleParser:
    """Parser chuyển chuỗi rule thành AST."""

    def __init__(self) -> None:
        self._tokens: list[str] = []
        self._pos: int = 0

    def parse(self, rule_string: str) -> Expression:
        """Public API: parse rule string → Expression tree."""
        self._tokenize(rule_string)
        self._pos = 0
        ast = self._parse_or()
        if self._pos < len(self._tokens):
            raise SyntaxError(f"Unexpected token: {self._tokens[self._pos:]}")
        return ast

    def _tokenize(self, s: str) -> None:
        """Tách chuỗi thành token list."""
        s = s.replace("(", " ( ").replace(")", " ) ")
        tokens = []
        for t in s.split():
            t = t.strip()
            if t:
                tokens.append(t)
        self._tokens = tokens

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected: str | None = None) -> str:
        token = self._tokens[self._pos]
        if expected and token != expected:
            raise SyntaxError(f"Expected '{expected}', got '{token}'")
        self._pos += 1
        return token

    def _parse_or(self) -> Expression:
        left = self._parse_and()
        while self._peek() == "OR":
            self._consume("OR")
            right = self._parse_and()
            left = OrExpression(left, right)
        return left

    def _parse_and(self) -> Expression:
        left = self._parse_not()
        while self._peek() == "AND":
            self._consume("AND")
            right = self._parse_not()
            left = AndExpression(left, right)
        return left

    def _parse_not(self) -> Expression:
        if self._peek() == "NOT":
            self._consume("NOT")
            return NotExpression(self._parse_atom())
        # XOR: a XOR b
        left = self._parse_atom()
        if self._peek() == "XOR":
            self._consume("XOR")
            right = self._parse_atom()
            return XorExpression(left, right)
        return left

    def _parse_atom(self) -> Expression:
        token = self._peek()
        if token is None:
            raise SyntaxError("Unexpected end of expression")

        if token == "(":
            self._consume("(")
            expr = self._parse_or()
            self._consume(")")
            return expr

        # So sánh: field op value
        return self._parse_comparison()

    def _parse_comparison(self) -> Expression:
        field = self._consume()
        op_token = self._peek()

        op_map = {
            ">": ComparisonOp.GT,
            ">=": ComparisonOp.GE,
            "<": ComparisonOp.LT,
            "<=": ComparisonOp.LE,
            "==": ComparisonOp.EQ,
            "!=": ComparisonOp.NE,
            "contains": ComparisonOp.CONTAINS,
            "matches": ComparisonOp.MATCHES,
            "between": ComparisonOp.BETWEEN,
        }

        if op_token not in op_map:
            raise SyntaxError(f"Expected operator, got '{op_token}'")

        op = op_map[op_token]
        self._consume(op_token)

        if op == ComparisonOp.BETWEEN:
            self._consume("[")
            low = float(self._consume())
            self._consume(",")
            high = float(self._consume())
            self._consume("]")
            return ComparisonExpression(field, op, [low, high])

        value_token = self._consume()
        try:
            value = int(value_token)
        except ValueError:
            try:
                value = float(value_token)
            except ValueError:
                value = value_token.strip('"').strip("'")

        return ComparisonExpression(field, op, value)


# --- Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = RuleParser()

    rules = [
        "temperature > 85 AND pressure > 100",
        "vibration > 5.0 OR zone == unsafe",
        "NOT (status == error) AND temperature between [20, 80]",
        "(temperature > 100 AND pressure > 200) OR zone == critical",
        "zone contains safe AND humidity > 30",
        "status matches err.*",
    ]

    data = SensorData(
        temperature=90.0,
        pressure=150.0,
        vibration=2.0,
        humidity=45.0,
        zone="safe-zone",
        status="error_404",
    )

    for rule_str in rules:
        try:
            ast = parser.parse(rule_str)
            result = ast.interpret(data)
            errors = ast.validate()
            print(f"  Rule: {rule_str}")
            print(f"  AST:  {ast}")
            print(f"  Valid: {errors if errors else 'OK'}")
            print(f"  Result: {'✅ PASS' if result else '❌ FAIL'}")
            print()
        except (SyntaxError, ValueError) as e:
            print(f"  ⚠️ Error parsing '{rule_str}': {e}\n")
```

## Sơ đồ UML

```
┌──────────────────────────┐
│      Expression (ABC)    │
│──────────────────────────│
│ + interpret(ctx): bool   │
│ + validate(): list[str]  │
└────────────┬─────────────┘
             │
    ┌────────┼───────────┬──────────────┐
    │        │           │              │
    │  ┌─────┴─────┐  ┌──┴──────┐  ┌───┴────────┐
    │  │ Terminal  │  │ Nonterm │  │Comparison  │
    │  │ Expression│  │Expression│  │Expression  │
    │  └─────┬─────┘  └──┬──────┘  └────────────┘
    │        │           │
 ┌──┴────┐ ┌─┴───┐  ┌───┴────┐  ┌────┴───┐  ┌───────┐
 │Field  │ │Lit  │  │AndExpr │  │OrExpr  │  │NotExpr│
 │Expr   │ │Expr │  │        │  │        │  │       │
 └───────┘ └─────┘  └────────┘  └────────┘  └───────┘

┌────────────────────────┐
│    RuleParser          │
│────────────────────────│
│ - tokens: list[str]    │
│ - pos: int             │
│────────────────────────│
│ + parse(s): Expression │
│ - _parse_or(): Expr    │
│ - _parse_and(): Expr   │
│ - _parse_not(): Expr   │
│ - _parse_atom(): Expr  │
└────────────────────────┘

┌────────────────┐
│  SensorData    │
│  (Context)     │
│────────────────│
│ temperature    │
│ pressure       │
│ vibration      │
│ zone           │
│ ...            │
└────────────────┘
```

## So sánh với Pattern liên quan

**1. Composite Pattern:**

Interpreter và Composite có cấu trúc giống nhau — cây với leaf và composite node. Nhưng mục đích khác xa: Composite dùng để xử lý collection đồng nhất, Interpreter dùng để tính toán kết quả boolean từ AST. Interpreter có `interpret(context)` nhận context parameter; Composite thường không có context.

**2. Strategy Pattern:**

Strategy thay đổi thuật toán trong runtime. Interpreter có thể dùng Strategy bên trong: mỗi operator (GT, LT, CONTAINS) có thể là một Strategy object thay vì enum với switch. Như vậy, thêm operator mới không cần sửa class `ComparisonExpression`.

**3. Visitor Pattern:**

Visitor thường kết hợp với Interpreter để tách operations khỏi AST. Thay vì mỗi Expression có `interpret()`, bạn dùng Visitor để implement interpret, validate, optimize, print riêng biệt. Với grammar phức tạp, nên tách bằng Visitor.

## Ứng dụng thực tế

**1. Django ORM Query (filter):**

Django ORM dùng interpreter để biến đổi `Q` objects thành SQL WHERE clause. Mỗi `Q` object là expression node, `Q.AND` và `Q.OR` là nonterminal. Bạn dùng Django hàng ngày mà không nhận ra pattern này đấy!

```python
from django.db.models import Q

# Django Q object = Interpreter pattern
query = Q(price__gt=100) & (Q(category="electronics") | Q(category="books"))
# Biểu diễn AST: And(price_gt(100), Or(category_eq("electronics"), category_eq("books")))
```

**2. SQLAlchemy Core Expression Language:**

SQLAlchemy xây dựng SQL query bằng Python expression — mỗi column, operator, function là expression node.

```python
from sqlalchemy import select, and_, or_, text

# SQLAlchemy expression = Interpreter pattern
stmt = select(users).where(
    and_(
        users.c.age > 18,
        or_(
            users.c.country == "VN",
            users.c.country == "US"
        )
    )
)
```

**3. ANTLR / Lark Parser Generators:**

Các parser generator implement Interpreter pattern ở mức cao hơn: grammar file định nghĩa production rules, tool sinh ra parser.

```python
# Lark: Python parser generator
from lark import Lark, Transformer

grammar = """
    rule: comparison ("AND" comparison)*
    comparison: FIELD OP VALUE
    FIELD: /[a-zA-Z_]+/
    OP: ">" | "<" | "==" | ">=" | "<=" | "contains"
    VALUE: NUMBER | ESCAPED_STRING
    %import common.NUMBER
    %import common.ESCAPED_STRING
"""

parser = Lark(grammar, start="rule")
tree = parser.parse('temperature > 85 AND pressure > 100')
```

**4. Regular Expression Engine:**

Mọi regex engine về bản chất là Interpreter pattern: pattern `[a-z]+@[a-z]+\.com` được parse thành AST gồm các node: `Group`, `Range`, `Quantifier`, `Concat`. Engine interpret AST trên input string.

## Kiểm thử

```python
import pytest


class TestInterpreterPattern:
    def setup_method(self):
        self.parser = RuleParser()
        self.data = SensorData(
            temperature=90.0,
            pressure=150.0,
            vibration=2.0,
            humidity=45.0,
            zone="safe-zone",
            status="error_404",
        )

    def test_comparison_gt(self):
        ast = self.parser.parse("temperature > 85")
        assert ast.interpret(self.data) is True

    def test_comparison_gt_false(self):
        ast = self.parser.parse("temperature > 95")
        assert ast.interpret(self.data) is False

    def test_and_expression(self):
        ast = self.parser.parse("temperature > 85 AND pressure > 100")
        assert ast.interpret(self.data) is True

    def test_and_expression_false(self):
        ast = self.parser.parse("temperature > 85 AND vibration > 10")
        assert ast.interpret(self.data) is False

    def test_or_expression(self):
        ast = self.parser.parse("vibration > 5.0 OR zone == safe-zone")
        assert ast.interpret(self.data) is True

    def test_not_expression(self):
        ast = self.parser.parse("NOT (status == error)")
        assert ast.interpret(self.data) is False

    def test_nested_parentheses(self):
        ast = self.parser.parse("(temperature > 80 AND pressure > 100) OR zone == critical")
        assert ast.interpret(self.data) is True

    def test_contains_operator(self):
        ast = self.parser.parse('zone contains safe')
        assert ast.interpret(self.data) is True

    def test_matches_operator(self):
        ast = self.parser.parse('status matches err.*')
        assert ast.interpret(self.data) is True

    def test_between_operator(self):
        ast = self.parser.parse("temperature between [20, 80]")
        assert ast.interpret(self.data) is False  # 90 > 80

        ast2 = self.parser.parse("humidity between [30, 60]")
        assert ast2.interpret(self.data) is True  # 45 ∈ [30, 60]

    def test_xor_expression(self):
        ast = self.parser.parse("temperature > 80 XOR vibration > 3.0")
        assert ast.interpret(self.data) is True  # True XOR False = True

    def test_validate_between_invalid(self):
        """Validation phát hiện BETWEEN thiếu range."""
        expr = ComparisonExpression("temp", ComparisonOp.BETWEEN, 100)
        errors = expr.validate()
        assert len(errors) > 0

    def test_parse_syntax_error(self):
        with pytest.raises(SyntaxError):
            self.parser.parse("temperature >> 85")

    def test_invalid_field(self):
        ast = self.parser.parse("nonexistent > 10")
        assert ast.interpret(self.data) is False

    def test_empty_rule_raises_error(self):
        with pytest.raises(SyntaxError):
            self.parser.parse("")
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| An toàn: không dùng eval — không RCE | Class explosion khi ngữ pháp phức tạp |
| Dễ mở rộng ngữ pháp (thêm class) | Hiệu năng kém hơn code compiled |
| Cấu trúc cây rõ ràng, dễ debug | Khó bảo trì khi grammar thay đổi thường xuyên |
| Hỗ trợ validation riêng biệt | Parser phức tạp nếu có operator precedence |
| Tách biệt grammar khỏi interpreter logic | Với grammar lớn, nên dùng ANTLR/Lark |
| Type safety hơn eval | Recursion depth có thể gây stack overflow |

---

## Kết luận

**Interpreter pattern là giải pháp tuyệt vời cho các ngôn ngữ đơn giản, ổn định, có cấu trúc đệ quy.** Dùng nó khi bạn cần parsing và thực thi user-defined expressions một cách an toàn, có thể mở rộng operator, và muốn kiểm soát AST chặt chẽ.

Tôi đã thấy pattern này xuất hiện trong business rule engine, DSL cho IoT, filter/sort query, và template engine. Nó mạnh mẽ — nhưng cũng dễ bị lạm dụng.

Những gì tôi muốn bạn nhớ:
1. Chỉ dùng khi grammar **nhỏ và ổn định** (< 20 production rules). Lớn hơn thì dùng ANTLR hoặc Lark.
2. Tách **Parser** khỏi **Interpreter** — parser build AST, interpreter walk AST.
3. Luôn implement `validate()` để kiểm tra AST trước khi interpret.
4. Cân nhắc **Visitor pattern** nếu cần nhiều thao tác trên AST (interpret, optimize, print).
5. Với hiệu năng cao, **compile** AST thành function — dùng `code` module hoặc transpile sang SQL.

---
*Trân trọng!*
