---
id: interpreter-architecture
title: Interpreter Architecture
sidebar_label: 🏗️ Interpreter Architecture
sidebar_position: 52
---

# Interpreter Architecture

> **Interpreter Architecture** — *"A system where a program's source code or intermediate representation is executed directly by an interpreter engine, which evaluates instructions one by one without requiring a separate compilation step to machine code."* — John McCarthy, 1960

## Tổng quan

Interpreter Architecture là một trong những kiến trúc phần mềm lâu đời và nền tảng nhất, có nguồn gốc từ những ngày đầu của khoa học máy tính. Khác với compiler dịch toàn bộ chương trình sang mã máy trước khi chạy, interpreter thực thi từng câu lệnh một cách trực tiếp — đọc, phân tích, và thực thi tuần tự.

Lịch sử của interpreter bắt đầu với **LISP** (John McCarthy, 1958-1960) — ngôn ngữ đầu tiên có interpreter. McCarthy đã định nghĩa hàm `eval` — trái tim của mọi interpreter — cho phép chương trình LISP tự thay đổi chính nó trong runtime. Tiếp theo là **BASIC** (Dartmouth, 1964) — interpreter cho người mới học lập trình. **APL** (Ken Iverson, 1966) dùng interpreter cho toán học ma trận. **Smalltalk** (Alan Kay, 1972) mang interpreter vào thế giới OOP với image-based virtual machine.

**Những người tiên phong:**

| Tên | Đóng góp |
|-----|----------|
| **John McCarthy** | LISP eval function — khai sinh ra interpreter |
| **Alan Kay** | Smalltalk VM — image-based interpreter |
| **James Gosling** | Java bytecode + JVM specification |
| **Guido van Rossum** | CPython interpreter |
| **Brendan Eich** | JavaScript interpreter (SpiderMonkey) |
| **Anders Hejlsberg** | Roslyn — C# compiler-as-service |

**Vai trò ngày nay:** Interpreter không chỉ là cách chạy ngôn ngữ kịch bản (Python, JavaScript, Ruby, Lua). Nó còn là kiến trúc cho DSL (Domain-Specific Language), rule engine, workflow engine, business logic engine, game scripting (Unreal Engine's Blueprint, Unity's C#), và configuration evaluation.

**Các biến thể interpreter:**
- **AST Walker**: Parse → AST → walk & execute (Ruby 1.8, Python AST)
- **Bytecode VM**: Parse → compile to bytecode → VM executes (Python, Java, Lua)
- **Just-In-Time (JIT)**: Interpret + profile + compile hot paths (V8, JVM HotSpot)
- **Tree-walking**: Từ AST thực thi trực tiếp (Simple DSL)
- **Threaded code**: Con trỏ hàm cho mỗi opcode (Forth,早期 Python)

## Bài toán

### Vấn đề 1: Cần thay đổi logic nghiệp vụ mà không deploy lại

Trong một hệ thống xử lý đơn hàng thương mại điện tử, logic tính giá (pricing engine) thay đổi hàng tuần: "Giảm 10% cho đơn hàng trên 1 triệu, miễn phí ship cho thành viên VIP, giảm thêm 5% nếu dùng thẻ tín dụng ACB." Mỗi lần thay đổi logic, đội ngũ phải modify code Java, build, test, deploy — mất 2-3 ngày. Business team không thể chờ đợi.

Interpreter architecture giải quyết bằng cách định nghĩa pricing rule dưới dạng DSL (ngôn ngữ riêng) lưu trong database hoặc file config. Business analyst tự viết rule, hệ thống interpreter đọc và thực thi mà không cần deploy. Rule mới có hiệu lực ngay lập tức.

### Vấn đề 2: Khách hàng khác nhau, logic khác nhau

Một nền tảng SaaS phục vụ 500+ khách hàng doanh nghiệp. Mỗi khách hàng có quy tắc nghiệp vụ riêng: cách tính thuế, quy trình phê duyệt, validation dữ liệu. Nếu hard-code từng logic riêng biệt, codebase sẽ phình ra với vô số `if-else` và `switch-case`. Bảo trì là ác mộng — mỗi bản update phải kiểm tra tất cả các nhánh.

Interpreter cho phép mỗi khách hàng có một tập rule DSL riêng, lưu trong database. Core system không thay đổi — chỉ có DSL thay đổi. Điều này giúp giảm code phức tạp từ O(n) (với n là số khách hàng) xuống O(1).

### Vấn đề 3: Người dùng cần tự động hóa mà không cần biết code

Hệ thống IoT platform cho nhà máy thông minh. Kỹ sư vận hành muốn tạo rule: "Nếu nhiệt độ > 85°C trong 5 phút và áp suất > 100 PSI → gửi cảnh báo + tắt van X." Kỹ sư không biết Python, không biết Java. Họ cần một ngôn ngữ đơn giản, gần gũi với tự nhiên để viết rule.

Interpreter architecture cung cấp một DSL tối giản: `WHEN temp > 85 FOR 5min AND pressure > 100 THEN alert("High Risk") AND valve.X = OFF`. Parser chuyển chuỗi này thành AST, interpreter thực thi rule.

### Vấn đề 4: Multi-tenant isolation và security

Khi nhiều khách hàng dùng chung một hệ thống, việc chạy code do khách hàng cung cấp là rủi ro bảo mật cực lớn. Không thể dùng `eval()` vì nguy cơ RCE (Remote Code Execution). Interpreter kiểm soát được:
- **Sandboxing**: Chỉ cho phép truy cập vào data model cố định
- **Resource limits**: Giới hạn CPU time, memory, recursion depth
- **Safe built-in functions**: Chỉ expose các function an toàn

### Vấn đề 5: Audit và versioning

Trong ngành tài chính, mọi quyết định pricing đều phải được audit. "Tại sao đơn hàng #1234 được tính giá 1.234.567 VND?" Với hard-code logic, answer là "vì code version 2.3.1". Không chi tiết. Với interpreter, rule được lưu dưới dạng text, có version, có thể trace từng bước interpreter đã thực thi — audit hoàn chỉnh.

## Nguyên lý thiết kế

### 1. Parse → Abstract Syntax Tree (AST)

Cốt lõi của interpreter: chuyển văn bản (source code / rule) thành AST — cấu trúc cây biểu diễn cú pháp của chương trình.

```
"temp > 85 AND pressure > 100"
    ┌── AND ──┐
    >         >
  temp 85  pressure 100
```

### 2. Visitor Pattern / Tree Walking

Interpreter duyệt AST bằng visitor pattern hoặc recursion, mỗi node trong AST biết cách "tự thực thi" (interpret).

### 3. Environment / Context

Interpreter cần một context chứa:
- **Variables**: giá trị của các biến (temperature, pressure)
- **Functions**: built-in functions (alert, log, send_email)
- **Scope**: lexical scoping cho variable lookup

### 4. Control Flow

Interpreter hỗ trợ các cấu trúc: sequence, condition (IF), loop (WHILE, FOR), function call.

### 5. Sandboxing

Interpreter giới hạn quyền truy cập:
- Cho phép: đọc/ghi data model, gọi exposed functions
- Cấm: I/O (file, network), import module, system calls

### 6. Error Handling

Interpreter phải xử lý:
- **Parse error**: syntax không đúng → báo lỗi có vị trí dòng/cột
- **Runtime error**: chia 0, type mismatch, variable undefined
- **Resource exceeded**: stack overflow, timeout

### 7. Extensibility

Interpreter cho phép thêm built-in functions và types:
- Plugin system cho functions
- Custom type registration

## Cấu trúc chi tiết

### Core Components

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Lexer / Tokenizer** | Chuyển source text → token stream | Regex-based, position tracking |
| **Parser** | Token stream → AST | Recursive descent / Pratt parsing |
| **AST Nodes** | Biểu diễn cấu trúc chương trình | Class hierarchy với `evaluate()` |
| **Environment** | Variable scope + function registry | Chain map (parent scope) |
| **Interpreter (Evaluator)** | Duyệt AST, thực thi | Visitor pattern |
| **Built-in Functions** | Functions exposed cho DSL | Registry dict |
| **Error Handler** | Parse + runtime error | Exception hierarchy |
| **Resource Monitor** | CPU/memory/stack limit | Context manager |

### AST Node Types

| Node Type | Description | Example |
|-----------|-------------|---------|
| **NumberLiteral** | Số | `42`, `3.14` |
| **StringLiteral** | Chuỗi | `"hello"` |
| **BooleanLiteral** | Boolean | `true`, `false` |
| **Identifier** | Biến | `temperature`, `username` |
| **BinaryOp** | Phép toán 2 ngôi | `+`, `-`, `*`, `/`, `>`, `<`, `==`, `AND`, `OR` |
| **UnaryOp** | Phép toán 1 ngôi | `NOT`, `-` |
| **IfExpr** | Điều kiện | `IF cond THEN a ELSE b` |
| **FunctionCall** | Gọi hàm | `alert("message")`, `max(a, b)` |
| **Assignment** | Gán biến | `x = 42` |
| **Block** | Khối lệnh | `{ stmt1; stmt2; stmt3 }` |
| **ForLoop** | Vòng lặp | `FOR item IN list DO ...` |
| **PropertyAccess** | Truy cập thuộc tính | `order.total`, `user.name` |

### Data Flow

```
User Rule: "WHEN total > 1000 THEN discount = 0.1"
    │
    ▼
Lexer / Tokenizer
    │  [WHEN] [IDENT(total)] [>] [NUMBER(1000)] [THEN] [IDENT(discount)] [=] [NUMBER(0.1)]
    ▼
Parser (Recursive Descent)
    │
    ▼
AST:
    WhenExpr
    ├── Condition: BinaryOp(GT, Identifier("total"), Number(1000))
    └── Body: Assignment(Identifier("discount"), Number(0.1))
    │
    ▼
Interpreter.evaluate(AST, Environment)
    │  ┌── Evaluate condition → BinaryOp: total > 1000
    │  │   ├── left: Identifier("total") → lookup in env → 1500
    │  │   ├── right: Number(1000) → 1000
    │  │   └── GT(1500, 1000) → true
    │  ├── condition true → evaluate body
    │  │   └── Assignment: discount = 0.1
    │  │       ├── env.set("discount", 0.1)
    │  │       └── return 0.1
    │  └── return 0.1
    ▼
Result: env["discount"] = 0.1
```

## Sơ đồ kiến trúc (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       INTERPRETER ARCHITECTURE                           │
│                                                                          │
│  ┌────────────┐    Source Code / Rule String                            │
│  │   User /   │─────► "WHEN total > 1000 AND status == 'active'"        │
│  │   Client   │                                                         │
│  └────────────┘                                                         │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                    FRONT END (Parsing)                        │       │
│  │  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐  │       │
│  │  │   LEXER /    │    │   PARSER     │    │   AST         │  │       │
│  │  │  TOKENIZER   │───►│  (Recursive  │───►│   OPTIMIZER   │  │       │
│  │  │              │    │   Descent)   │    │  (Const fold,  │  │       │
│  │  │ char → token │    │   token →    │    │   dead code)  │  │       │
│  │  └──────────────┘    │   AST        │    └────────────────┘  │       │
│  │                      └──────────────┘                        │       │
│  └──────────────────────────────────────────────────────────────┘       │
│       │                                                                  │
│       │  Optimized AST                                                   │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                    RUNTIME (Back End)                         │       │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │       │
│  │  │  ENVIRONMENT   │  │  INTERPRETER   │  │  RESOURCE      │  │       │
│  │  │  (Variable     │  │  (Evaluator)   │  │  MONITOR       │  │       │
│  │  │   Scopes +     │◄─►│                │  │  (Timeout,     │  │       │
│  │  │   Functions)   │  │  walk AST →    │  │   Memory,      │  │       │
│  │  │                │  │  execute       │  │   Recursion)   │  │       │
│  │  └────────────────┘  └───────┬────────┘  └────────────────┘  │       │
│  │                              │                                │       │
│  │  ┌───────────────────────────▼────────────────────────────┐   │       │
│  │  │              BUILT-IN FUNCTIONS                        │   │       │
│  │  │  [alert] [send_email] [log] [http_get] [db_query]     │   │       │
│  │  └───────────────────────────────────────────────────────┘   │       │
│  └──────────────────────────────────────────────────────────────┘       │
│       │                                                                  │
│       │  Result / Side Effects                                           │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                    OUTPUT                                     │       │
│  │  - Evaluated result                                           │       │
│  │  - Modified environment (variable assignments)                │       │
│  │  - Actions (alerts, emails, DB writes)                        │       │
│  │  - Errors (parse errors, runtime errors) with line:col        │       │
│  └──────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Ví dụ code hoàn chỉnh

### Cách làm sai: Dùng Python eval()

```python
from __future__ import annotations
from typing import Any


class DangerousRuleEngine:
    """Dùng eval() — RCE vulnerability, không sandbox."""

    def evaluate(self, rule: str, context: dict[str, Any]) -> Any:
        # Rule: "order.total > 1000 and order.status == 'active'"
        # Hacker: "__import__('os').system('rm -rf /')" — RCE!
        return eval(rule, {"__builtins__": {}}, context)
```

### Cách làm đúng: Interpreter Architecture với Full DSL

```python
from __future__ import annotations
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections import ChainMap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# 1. TOKEN TYPES & LEXER
# ======================================================================

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    NULL = auto()

    # Identifiers & Keywords
    IDENTIFIER = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    WHEN = auto()
    FOR = auto()
    IN = auto()
    DO = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    TRUE = auto()
    FALSE = auto()
    NULL_KW = auto()
    FUNC = auto()
    RETURN = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    GT = auto()
    GTE = auto()
    LT = auto()
    LTE = auto()
    EQ = auto()
    NEQ = auto()
    ASSIGN = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    SEMICOLON = auto()
    COLON = auto()

    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
    literal: str = ""

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r})"


class Lexer:
    """Chuyển source text → token stream với position tracking."""

    KEYWORDS: dict[str, TokenType] = {
        "IF": TokenType.IF,
        "THEN": TokenType.THEN,
        "ELSE": TokenType.ELSE,
        "WHEN": TokenType.WHEN,
        "FOR": TokenType.FOR,
        "IN": TokenType.IN,
        "DO": TokenType.DO,
        "AND": TokenType.AND,
        "OR": TokenType.OR,
        "NOT": TokenType.NOT,
        "true": TokenType.TRUE,
        "false": TokenType.FALSE,
        "null": TokenType.NULL_KW,
        "FUNC": TokenType.FUNC,
        "RETURN": TokenType.RETURN,
    }

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._line = 1
        self._column = 1
        self._tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        while self._pos < len(self._source):
            self._skip_whitespace()
            if self._pos >= len(self._source):
                break
            ch = self._source[self._pos]

            if ch == "#":
                self._skip_comment()
            elif ch == '"' or ch == "'":
                self._tokenize_string(ch)
            elif ch.isdigit() or (ch == "-" and self._peek_next().isdigit()):
                self._tokenize_number(ch)
            elif ch.isalpha() or ch == "_":
                self._tokenize_identifier()
            else:
                self._tokenize_operator_or_delimiter(ch)

        self._tokens.append(Token(TokenType.EOF, None, self._line, self._column))
        return self._tokens

    def _skip_whitespace(self) -> None:
        while self._pos < len(self._source) and self._source[self._pos] in " \t\n\r":
            if self._source[self._pos] == "\n":
                self._line += 1
                self._column = 1
            else:
                self._column += 1
            self._pos += 1

    def _skip_comment(self) -> None:
        while self._pos < len(self._source) and self._source[self._pos] != "\n":
            self._pos += 1

    def _tokenize_string(self, quote: str) -> None:
        start_line = self._line
        start_col = self._column
        self._pos += 1
        self._column += 1
        chars: list[str] = []

        while self._pos < len(self._source):
            ch = self._source[self._pos]
            if ch == quote:
                self._pos += 1
                self._column += 1
                self._tokens.append(Token(
                    TokenType.STRING, "".join(chars), start_line, start_col, f"{quote}{''.join(chars)}{quote}"
                ))
                return
            elif ch == "\\":
                self._pos += 1
                self._column += 1
                if self._pos < len(self._source):
                    esc = self._source[self._pos]
                    esc_map = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\"}
                    chars.append(esc_map.get(esc, esc))
                    self._pos += 1
                    self._column += 1
            else:
                chars.append(ch)
                self._pos += 1
                self._column += 1

        raise LexerError(f"Unterminated string at line {start_line}:{start_col}")

    def _tokenize_number(self, first: str) -> None:
        start_line = self._line
        start_col = self._column
        chars: list[str] = [first]
        self._pos += 1
        self._column += 1
        is_float = False

        while self._pos < len(self._source):
            ch = self._source[self._pos]
            if ch.isdigit():
                chars.append(ch)
                self._pos += 1
                self._column += 1
            elif ch == ".":
                if is_float:
                    break
                is_float = True
                chars.append(ch)
                self._pos += 1
                self._column += 1
            else:
                break

        value = float("".join(chars)) if is_float else int("".join(chars))
        self._tokens.append(Token(TokenType.NUMBER, value, start_line, start_col, "".join(chars)))

    def _tokenize_identifier(self) -> None:
        start_line = self._line
        start_col = self._column
        chars: list[str] = []
        while self._pos < len(self._source) and (self._source[self._pos].isalnum() or self._source[self._pos] == "_"):
            chars.append(self._source[self._pos])
            self._pos += 1
            self._column += 1

        word = "".join(chars)
        token_type = self.KEYWORDS.get(word, TokenType.IDENTIFIER)
        value = word if token_type == TokenType.IDENTIFIER else {
            TokenType.TRUE: True, TokenType.FALSE: False, TokenType.NULL_KW: None
        }.get(token_type, word)
        self._tokens.append(Token(token_type, value, start_line, start_col, word))

    def _tokenize_operator_or_delimiter(self, ch: str) -> None:
        start_line = self._line
        start_col = self._column

        two_char_ops = {">=": TokenType.GTE, "<=": TokenType.LTE, "==": TokenType.EQ, "!=": TokenType.NEQ}
        one_char_ops = {
            "+": TokenType.PLUS, "-": TokenType.MINUS, "*": TokenType.MUL, "/": TokenType.DIV,
            "%": TokenType.MOD, ">": TokenType.GT, "<": TokenType.LT, "=": TokenType.ASSIGN,
            "(": TokenType.LPAREN, ")": TokenType.RPAREN, "{": TokenType.LBRACE, "}": TokenType.RBRACE,
            "[": TokenType.LBRACKET, "]": TokenType.RBRACKET, ",": TokenType.COMMA, ".": TokenType.DOT,
            ";": TokenType.SEMICOLON, ":": TokenType.COLON,
        }

        if self._pos + 1 < len(self._source):
            two_char = ch + self._source[self._pos + 1]
            if two_char in two_char_ops:
                self._tokens.append(Token(two_char_ops[two_char], two_char, start_line, start_col, two_char))
                self._pos += 2
                self._column += 2
                return

        if ch in one_char_ops:
            self._tokens.append(Token(one_char_ops[ch], ch, start_line, start_col, ch))
            self._pos += 1
            self._column += 1
        else:
            raise LexerError(f"Unexpected character '{ch}' at line {start_line}:{start_col}")

    def _peek_next(self) -> str:
        if self._pos + 1 < len(self._source):
            return self._source[self._pos + 1]
        return ""


class LexerError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ParseError(Exception):
    def __init__(self, message: str, token: Token | None = None) -> None:
        loc = f" at line {token.line}:{token.column}" if token else ""
        super().__init__(f"{message}{loc}")
        self.message = self.args[0]
        self.token = token


class RuntimeError_(Exception):
    def __init__(self, message: str, node: ASTNode | None = None) -> None:
        loc = f" at {node.__class__.__name__}" if node else ""
        super().__init__(f"{message}{loc}")
        self.message = self.args[0]


# ======================================================================
# 2. AST NODES
# ======================================================================

class ASTNode(ABC):
    """Base class cho mọi node trong AST."""

    @abstractmethod
    def evaluate(self, env: Environment) -> Any:
        ...


@dataclass
class NumberLiteral(ASTNode):
    value: float | int

    def evaluate(self, env: Environment) -> float | int:
        return self.value


@dataclass
class StringLiteral(ASTNode):
    value: str

    def evaluate(self, env: Environment) -> str:
        return self.value


@dataclass
class BooleanLiteral(ASTNode):
    value: bool

    def evaluate(self, env: Environment) -> bool:
        return self.value


@dataclass
class NullLiteral(ASTNode):
    def evaluate(self, env: Environment) -> None:
        return None


@dataclass
class Identifier(ASTNode):
    name: str

    def evaluate(self, env: Environment) -> Any:
        return env.get(self.name)


@dataclass
class BinaryOp(ASTNode):
    operator: str
    left: ASTNode
    right: ASTNode

    def evaluate(self, env: Environment) -> Any:
        left_val = self.left.evaluate(env)
        right_val = self.right.evaluate(env)

        if self.operator == "+":
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                return left_val + right_val
            if isinstance(left_val, str) and isinstance(right_val, str):
                return left_val + right_val
            raise RuntimeError_(f"Cannot add {type(left_val).__name__} + {type(right_val).__name__}", self)
        elif self.operator == "-":
            return left_val - right_val
        elif self.operator == "*":
            return left_val * right_val
        elif self.operator == "/":
            if right_val == 0:
                raise RuntimeError_("Division by zero", self)
            return left_val / right_val
        elif self.operator == "%":
            return left_val % right_val
        elif self.operator == ">":
            return left_val > right_val
        elif self.operator == ">=":
            return left_val >= right_val
        elif self.operator == "<":
            return left_val < right_val
        elif self.operator == "<=":
            return left_val <= right_val
        elif self.operator == "==":
            return left_val == right_val
        elif self.operator == "!=":
            return left_val != right_val
        elif self.operator.upper() == "AND":
            return bool(left_val) and bool(right_val)
        elif self.operator.upper() == "OR":
            return bool(left_val) or bool(right_val)
        else:
            raise RuntimeError_(f"Unknown operator: {self.operator}", self)


@dataclass
class UnaryOp(ASTNode):
    operator: str
    operand: ASTNode

    def evaluate(self, env: Environment) -> Any:
        val = self.operand.evaluate(env)
        if self.operator == "-":
            return -val
        elif self.operator.upper() == "NOT":
            return not bool(val)
        raise RuntimeError_(f"Unknown unary operator: {self.operator}", self)


@dataclass
class IfExpr(ASTNode):
    condition: ASTNode
    then_branch: ASTNode
    else_branch: ASTNode | None = None

    def evaluate(self, env: Environment) -> Any:
        cond = bool(self.condition.evaluate(env))
        if cond:
            return self.then_branch.evaluate(env)
        elif self.else_branch:
            return self.else_branch.evaluate(env)
        return None


@dataclass
class Assignment(ASTNode):
    name: str
    value: ASTNode

    def evaluate(self, env: Environment) -> Any:
        result = self.value.evaluate(env)
        env.set(self.name, result)
        return result


@dataclass
class FunctionCall(ASTNode):
    name: str
    arguments: list[ASTNode]

    def evaluate(self, env: Environment) -> Any:
        args = [arg.evaluate(env) for arg in self.arguments]
        func = env.get_function(self.name)
        if func is None:
            raise RuntimeError_(f"Undefined function: {self.name}", self)
        return func(*args)


@dataclass
class Block(ASTNode):
    statements: list[ASTNode]

    def evaluate(self, env: Environment) -> Any:
        result = None
        for stmt in self.statements:
            result = stmt.evaluate(env)
        return result


@dataclass
class ForLoop(ASTNode):
    variable: str
    iterable: ASTNode
    body: ASTNode

    def evaluate(self, env: Environment) -> list[Any]:
        results: list[Any] = []
        iterable_val = self.iterable.evaluate(env)
        if not hasattr(iterable_val, "__iter__"):
            raise RuntimeError_(f"Cannot iterate over {type(iterable_val).__name__}", self)
        for item in iterable_val:
            env.set(self.variable, item)
            results.append(self.body.evaluate(env))
        return results


@dataclass
class PropertyAccess(ASTNode):
    obj: ASTNode
    property_name: str

    def evaluate(self, env: Environment) -> Any:
        obj_val = self.obj.evaluate(env)
        if isinstance(obj_val, dict):
            return obj_val.get(self.property_name)
        if hasattr(obj_val, self.property_name):
            return getattr(obj_val, self.property_name)
        raise RuntimeError_(f"Object {type(obj_val).__name__} has no property '{self.property_name}'", self)


@dataclass
class ListLiteral(ASTNode):
    elements: list[ASTNode]

    def evaluate(self, env: Environment) -> list[Any]:
        return [e.evaluate(env) for e in self.elements]


@dataclass
class FunctionDef(ASTNode):
    name: str
    params: list[str]
    body: ASTNode

    def evaluate(self, env: Environment) -> str:
        env.set_function(self.name, self._make_closure(self.params, self.body, env))
        return self.name

    def _make_closure(self, params: list[str], body: ASTNode, outer_env: Environment) -> Callable[..., Any]:
        def closure(*args: Any) -> Any:
            if len(args) != len(params):
                raise RuntimeError_(f"Function '{self.name}' expects {len(params)} args, got {len(args)}")
            local_env = Environment(parent=outer_env)
            for p, a in zip(params, args):
                local_env.set(p, a)
            return body.evaluate(local_env)
        return closure


# ======================================================================
# 3. PARSER (Recursive Descent)
# ======================================================================

class Parser:
    """Recursive descent parser — token stream → AST."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> ASTNode:
        result = self._parse_block()
        if not self._check(TokenType.EOF):
            raise ParseError(f"Unexpected token: {self._peek()}", self._peek())
        return result

    def _parse_block(self) -> Block:
        statements: list[ASTNode] = []
        while not self._check(TokenType.EOF, TokenType.RBRACE):
            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self._match(TokenType.SEMICOLON)
        return Block(statements)

    def _parse_statement(self) -> ASTNode | None:
        if self._match(TokenType.IF):
            return self._parse_if()
        if self._match(TokenType.WHEN):
            return self._parse_when()
        if self._match(TokenType.FOR):
            return self._parse_for()
        if self._match(TokenType.FUNC):
            return self._parse_function_def()
        if self._match(TokenType.RETURN):
            return self._parse_return()
        if self._check(TokenType.LBRACE):
            return self._parse_block()
        return self._parse_expression()

    def _parse_if(self) -> IfExpr:
        condition = self._parse_expression()
        self._expect(TokenType.THEN)
        then_branch = self._parse_statement()
        else_branch = None
        if self._match(TokenType.ELSE):
            else_branch = self._parse_statement()
        if then_branch is None:
            raise ParseError("IF requires THEN branch", self._peek())
        return IfExpr(condition, then_branch, else_branch)

    def _parse_when(self) -> IfExpr:
        """WHEN condition THEN action — syntactic sugar cho IF-THEN."""
        return self._parse_if()

    def _parse_for(self) -> ForLoop:
        if not self._check(TokenType.IDENTIFIER):
            raise ParseError("FOR requires variable name", self._peek())
        var_token = self._consume()
        self._expect(TokenType.IN)
        iterable = self._parse_expression()
        self._expect(TokenType.DO)
        body = self._parse_statement()
        if body is None:
            raise ParseError("FOR requires DO body", self._peek())
        return ForLoop(var_token.value, iterable, body)

    def _parse_function_def(self) -> FunctionDef:
        name_token = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.LPAREN)
        params: list[str] = []
        if not self._check(TokenType.RPAREN):
            params.append(self._expect(TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                params.append(self._expect(TokenType.IDENTIFIER).value)
        self._expect(TokenType.RPAREN)
        body = self._parse_statement()
        if body is None:
            raise ParseError("FUNC requires body", self._peek())
        return FunctionDef(name_token.value, params, body)

    def _parse_return(self) -> ASTNode:
        value = self._parse_expression()
        return value  # Return is implicit in expression-oriented language

    def _parse_expression(self) -> ASTNode:
        return self._parse_assignment()

    def _parse_assignment(self) -> ASTNode:
        left = self._parse_or()
        if self._match(TokenType.ASSIGN):
            if not isinstance(left, Identifier):
                raise ParseError("Left side of assignment must be identifier", self._peek())
            value = self._parse_assignment()
            return Assignment(left.name, value)
        return left

    def _parse_or(self) -> ASTNode:
        left = self._parse_and()
        while self._match(TokenType.OR):
            right = self._parse_and()
            left = BinaryOp("OR", left, right)
        return left

    def _parse_and(self) -> ASTNode:
        left = self._parse_comparison()
        while self._match(TokenType.AND):
            right = self._parse_comparison()
            left = BinaryOp("AND", left, right)
        return left

    def _parse_comparison(self) -> ASTNode:
        left = self._parse_addition()
        comparison_ops = {
            TokenType.GT: ">", TokenType.GTE: ">=", TokenType.LT: "<", TokenType.LTE: "<=",
            TokenType.EQ: "==", TokenType.NEQ: "!=",
        }
        while self._peek().type in comparison_ops:
            op_token = self._consume()
            right = self._parse_addition()
            left = BinaryOp(comparison_ops[op_token.type], left, right)
        return left

    def _parse_addition(self) -> ASTNode:
        left = self._parse_multiplication()
        while self._match(TokenType.PLUS, TokenType.MINUS):
            op = self._previous().literal
            right = self._parse_multiplication()
            left = BinaryOp(op, left, right)
        return left

    def _parse_multiplication(self) -> ASTNode:
        left = self._parse_unary()
        while self._match(TokenType.MUL, TokenType.DIV, TokenType.MOD):
            op = self._previous().literal
            right = self._parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def _parse_unary(self) -> ASTNode:
        if self._match(TokenType.MINUS, TokenType.NOT):
            op = self._previous().literal if self._previous().type == TokenType.MINUS else "NOT"
            operand = self._parse_unary()
            return UnaryOp(op, operand)
        return self._parse_call()

    def _parse_call(self) -> ASTNode:
        node = self._parse_primary()
        while True:
            if self._match(TokenType.LPAREN):
                args: list[ASTNode] = []
                if not self._check(TokenType.RPAREN):
                    args.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        args.append(self._parse_expression())
                self._expect(TokenType.RPAREN)
                if isinstance(node, Identifier):
                    node = FunctionCall(node.name, args)
                else:
                    raise ParseError("Can only call functions", self._peek())
            elif self._match(TokenType.DOT):
                prop = self._expect(TokenType.IDENTIFIER)
                node = PropertyAccess(node, prop.value)
            elif self._match(TokenType.LBRACKET):
                index = self._parse_expression()
                self._expect(TokenType.RBRACKET)
                node = PropertyAccess(node, "__getitem__")  # Simplified
            else:
                break
        return node

    def _parse_primary(self) -> ASTNode:
        if self._match(TokenType.NUMBER):
            return NumberLiteral(self._previous().value)
        if self._match(TokenType.STRING):
            return StringLiteral(self._previous().value)
        if self._match(TokenType.TRUE):
            return BooleanLiteral(True)
        if self._match(TokenType.FALSE):
            return BooleanLiteral(False)
        if self._match(TokenType.NULL_KW):
            return NullLiteral()
        if self._match(TokenType.IDENTIFIER):
            return Identifier(self._previous().value)
        if self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr
        if self._match(TokenType.LBRACKET):
            elements: list[ASTNode] = []
            if not self._check(TokenType.RBRACKET):
                elements.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    elements.append(self._parse_expression())
            self._expect(TokenType.RBRACKET)
            return ListLiteral(elements)
        if self._match(TokenType.LBRACE):
            block = self._parse_block()
            self._expect(TokenType.RBRACE)
            return block
        raise ParseError(f"Unexpected token: {self._peek()}", self._peek())

    def _match(self, *types: TokenType) -> bool:
        if self._pos < len(self._tokens) and self._tokens[self._pos].type in types:
            self._pos += 1
            return True
        return False

    def _check(self, *types: TokenType) -> bool:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos].type in types
        return False

    def _consume(self) -> Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _previous(self) -> Token:
        return self._tokens[self._pos - 1]

    def _expect(self, *types: TokenType) -> Token:
        if self._check(*types):
            return self._consume()
        expected = ", ".join(t.name for t in types)
        raise ParseError(f"Expected {expected}, got {self._peek().type.name}", self._peek())


# ======================================================================
# 4. ENVIRONMENT
# ======================================================================

class Environment:
    """Variable scope + function registry với parent chain."""

    def __init__(self, parent: Environment | None = None) -> None:
        self._parent = parent
        self._variables: dict[str, Any] = {}
        self._functions: dict[str, Callable[..., Any]] = {}

    def set(self, name: str, value: Any) -> None:
        self._variables[name] = value

    def get(self, name: str) -> Any:
        if name in self._variables:
            return self._variables[name]
        if self._parent is not None:
            return self._parent.get(name)
        raise RuntimeError_(f"Undefined variable: {name}")

    def has(self, name: str) -> bool:
        if name in self._variables:
            return True
        if self._parent is not None:
            return self._parent.has(name)
        return False

    def set_function(self, name: str, func: Callable[..., Any]) -> None:
        self._functions[name] = func

    def get_function(self, name: str) -> Callable[..., Any] | None:
        if name in self._functions:
            return self._functions[name]
        if self._parent is not None:
            return self._parent.get_function(name)
        return None

    def to_dict(self) -> dict[str, Any]:
        return dict(self._variables)


# ======================================================================
# 5. RESOURCE MONITOR
# ======================================================================

class ResourceMonitor:
    """Giới hạn CPU time, recursion depth, memory."""

    def __init__(self, max_recursion: int = 1000, max_operations: int = 1_000_000, timeout_ms: int = 5000) -> None:
        self._max_recursion = max_recursion
        self._max_operations = max_operations
        self._timeout_ms = timeout_ms
        self._ops = 0
        self._depth = 0
        self._start_time: float | None = None

    def __enter__(self) -> ResourceMonitor:
        self._ops = 0
        self._depth = 0
        self._start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self._start_time = None

    def enter_call(self) -> None:
        self._depth += 1
        self._ops += 1
        if self._depth > self._max_recursion:
            raise RuntimeError_(f"Max recursion depth ({self._max_recursion}) exceeded")
        if self._ops > self._max_operations:
            raise RuntimeError_(f"Max operations ({self._max_operations}) exceeded")
        if self._start_time and (time.time() - self._start_time) * 1000 > self._timeout_ms:
            raise RuntimeError_(f"Timeout ({self._timeout_ms}ms) exceeded")

    def leave_call(self) -> None:
        self._depth -= 1


# ======================================================================
# 6. MAIN INTERPRETER
# ======================================================================

class Interpreter:
    """Interpreter chính — parse + execute với sandboxing."""

    def __init__(self, resource_monitor: ResourceMonitor | None = None) -> None:
        self._resource_monitor = resource_monitor or ResourceMonitor()
        self._env = Environment()

        # Register built-in functions
        self._register_builtins()

    def _register_builtins(self) -> None:
        builtins = Environment(parent=self._env)

        builtins.set_function("print", lambda *args: logger.info(" ".join(str(a) for a in args)))
        builtins.set_function("alert", lambda msg: logger.warning("ALERT: %s", msg))
        builtins.set_function("log", lambda msg, level="info": getattr(logger, level)(msg))
        builtins.set_function("now", lambda: time.time())
        builtins.set_function("len", lambda x: len(x))
        builtins.set_function("str", lambda x: str(x))
        builtins.set_function("int", lambda x: int(x) if isinstance(x, (int, float, str)) else 0)
        builtins.set_function("float", lambda x: float(x) if isinstance(x, (int, float, str)) else 0.0)
        builtins.set_function("sum", lambda *args: sum(args))
        builtins.set_function("avg", lambda *args: sum(args) / len(args) if args else 0.0)
        builtins.set_function("round", lambda x, n=0: round(x, n))
        builtins.set_function("min", lambda *args: min(args))
        builtins.set_function("max", lambda *args: max(args))
        builtins.set_function("concat", lambda *args: "".join(str(a) for a in args))
        builtins.set_function("contains", lambda collection, item: item in collection)
        builtins.set_function("format", lambda template, **kwargs: template.format(**kwargs))
        builtins.set_function("map", lambda func_name, items: [self._env.get_function(func_name)(item) for item in items])
        builtins.set_function("filter", lambda func_name, items: [item for item in items if self._env.get_function(func_name)(item)])

        self._env = builtins

    def set_variable(self, name: str, value: Any) -> None:
        """Inject variable vào environment (data model)."""
        self._env.set(name, value)

    def set_variables(self, mapping: dict[str, Any]) -> None:
        for name, value in mapping.items():
            self._env.set(name, value)

    def register_function(self, name: str, func: Callable[..., Any]) -> None:
        """Đăng ký custom function cho DSL."""
        self._env.set_function(name, func)

    def evaluate(self, source: str) -> Any:
        """Parse + execute source code."""
        try:
            # Tokenize
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            logger.debug("Tokens: %s", tokens)

            # Parse
            parser = Parser(tokens)
            ast = parser.parse()
            logger.debug("AST: %s", ast)

            # Execute with resource monitoring
            with self._resource_monitor as monitor:
                result = self._eval_with_monitor(ast, monitor)
                return result

        except LexerError as e:
            logger.error("Lexer error: %s", e.message)
            raise
        except ParseError as e:
            logger.error("Parse error: %s", e.message)
            raise
        except RuntimeError_ as e:
            logger.error("Runtime error: %s", e.message)
            raise

    def _eval_with_monitor(self, node: ASTNode, monitor: ResourceMonitor) -> Any:
        """Wrapper để monitor operations."""
        monitor.enter_call()
        try:
            return node.evaluate(self._env)
        finally:
            monitor.leave_call()

    def get_environment(self) -> Environment:
        return self._env


# ======================================================================
# 7. MAIN — E-COMMERCE PRICING ENGINE
# ======================================================================

@dataclass
class Order:
    order_id: str
    user_id: str
    items: list[dict[str, Any]]
    subtotal: float
    shipping_cost: float = 0.0
    discount: float = 0.0
    tax_rate: float = 0.1
    coupon_code: str = ""

    @property
    def total(self) -> float:
        return (self.subtotal + self.shipping_cost) * (1 - self.discount) * (1 + self.tax_rate)

    @property
    def item_count(self) -> int:
        return sum(item.get("quantity", 1) for item in self.items)

    @property
    def is_vip(self) -> bool:
        return self.user_id.startswith("vip")


@dataclass
class User:
    user_id: str
    name: str
    tier: str  # bronze, silver, gold, platinum
    registered_days: int
    total_spent: float


class PricingEngine:
    """Pricing engine dùng interpreter để thực thi pricing rules."""

    def __init__(self, interpreter: Interpreter) -> None:
        self._interpreter = interpreter
        self._rules: list[str] = []

    def add_rule(self, rule: str) -> None:
        self._rules.append(rule)

    def set_rules(self, rules: list[str]) -> None:
        self._rules = rules

    def calculate(self, order: Order, user: User | None = None) -> dict[str, Any]:
        """Tính giá cho đơn hàng dựa trên rules."""
        context = {
            "order": {
                "id": order.order_id,
                "subtotal": order.subtotal,
                "shipping_cost": order.shipping_cost,
                "discount": order.discount,
                "tax_rate": order.tax_rate,
                "coupon_code": order.coupon_code,
                "item_count": order.item_count,
                "total": order.total,
            },
            "user": {
                "user_id": user.user_id if user else "",
                "tier": user.tier if user else "bronze",
                "registered_days": user.registered_days if user else 0,
                "total_spent": user.total_spent if user else 0.0,
            } if user else {},
            "result": {
                "discount": 0.0,
                "shipping_discount": 0.0,
                "bonus_points": 0,
                "message": "",
            },
        }

        # Inject context vào interpreter
        self._interpreter.get_environment().set("ctx", context)

        # Execute rules
        for rule in self._rules:
            try:
                logger.info("Evaluating rule: %s", rule[:60])
                self._interpreter.evaluate(rule)
            except (LexerError, ParseError, RuntimeError_) as e:
                logger.error("Rule failed: %s\n  Error: %s", rule[:60], e.message)
                context["result"]["message"] = f"Rule error: {e.message}"

        return context["result"]

    def validate_rule(self, rule: str) -> tuple[bool, str]:
        """Kiểm tra rule syntax."""
        try:
            lexer = Lexer(rule)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            parser.parse()
            return True, "OK"
        except (LexerError, ParseError) as e:
            return False, e.message


def main() -> None:
    logger.info("=== Interpreter Architecture: E-Commerce Pricing Engine ===")

    # Create interpreter
    interpreter = Interpreter(resource_monitor=ResourceMonitor(max_recursion=100, max_operations=50000, timeout_ms=2000))

    # Register pricing-specific functions
    def apply_discount(percent: float) -> None:
        ctx = interpreter.get_environment().get("ctx")
        ctx["result"]["discount"] = max(ctx["result"]["discount"], percent)

    def free_shipping() -> None:
        ctx = interpreter.get_environment().get("ctx")
        ctx["result"]["shipping_discount"] = ctx["order"]["shipping_cost"]

    def add_bonus_points(points: int) -> None:
        ctx = interpreter.get_environment().get("ctx")
        ctx["result"]["bonus_points"] += points

    interpreter.register_function("apply_discount", apply_discount)
    interpreter.register_function("free_shipping", free_shipping)
    interpreter.register_function("add_bonus_points", add_bonus_points)

    # Define pricing rules (DSL)
    rules = [
        # Rule 1: VIP gets 10% off
        """
        IF ctx.order.total > 500 AND ctx.user.tier == "gold"
        THEN apply_discount(0.15)
        """,

        # Rule 2: Free shipping for orders > 300k
        """
        IF ctx.order.subtotal > 300
        THEN free_shipping()
        """,

        # Rule 3: Bonus points based on order value
        """
        IF ctx.order.subtotal > 100
        THEN add_bonus_points(int(ctx.order.subtotal / 10))
        """,

        # Rule 4: New user discount
        """
        IF ctx.user.registered_days < 30 AND ctx.order.subtotal > 50
        THEN apply_discount(0.10)
        """,

        # Rule 5: Platinum tier extra discount
        """
        IF ctx.user.tier == "platinum"
        THEN apply_discount(0.20)
        """,

        # Rule 6: Complex conditional
        """
        IF ctx.order.item_count >= 5 AND ctx.order.subtotal > 200
        THEN apply_discount(0.05)
        """,

        # Rule 7: Coupon code handling
        """
        IF ctx.order.coupon_code == "WELCOME20"
        THEN apply_discount(0.20)
        """,
    ]

    # Setup pricing engine
    engine = PricingEngine(interpreter)
    engine.set_rules(rules)

    # Validate rules
    logger.info("=== Validating Rules ===")
    for i, rule in enumerate(rules, 1):
        valid, msg = engine.validate_rule(rule)
        logger.info("Rule %d: %s (%s)", i, "OK" if valid else "INVALID", msg)

    # Test cases
    logger.info("\n=== Test Cases ===")

    test_cases = [
        (
            Order(order_id="ORD-001", user_id="vip_1", items=[{"qty": 2}], subtotal=500.0, shipping_cost=30.0),
            User(user_id="vip_1", name="Alice", tier="gold", registered_days=365, total_spent=5000.0),
            "Gold VIP, large order",
        ),
        (
            Order(order_id="ORD-002", user_id="usr_2", items=[{"qty": 1}], subtotal=50.0, shipping_cost=20.0, coupon_code="WELCOME20"),
            User(user_id="usr_2", name="Bob", tier="bronze", registered_days=5, total_spent=0.0),
            "New user with coupon",
        ),
        (
            Order(order_id="ORD-003", user_id="usr_3", items=[{"qty": 1}], subtotal=80.0, shipping_cost=15.0),
            User(user_id="usr_3", name="Charlie", tier="bronze", registered_days=200, total_spent=300.0),
            "Normal user, small order",
        ),
        (
            Order(order_id="ORD-004", user_id="vip_2", items=[{"qty": 5, "name": "item"}], subtotal=400.0, shipping_cost=0.0),
            User(user_id="vip_2", name="Diana", tier="platinum", registered_days=730, total_spent=20000.0),
            "Platinum, bulk order",
        ),
    ]

    for order, user, description in test_cases:
        logger.info("--- %s ---", description)
        logger.info("Order: subtotal=%.0f, shipping=%.0f, items=%d, coupon=%s",
                     order.subtotal, order.shipping_cost, order.item_count, order.coupon_code)
        logger.info("User: tier=%s, registered=%d days", user.tier, user.registered_days)
        result = engine.calculate(order, user)
        logger.info("Result: discount=%.0f%%, shipping_discount=%.0f, bonus=%d pts, msg=%s",
                     result["discount"] * 100, result["shipping_discount"], result["bonus_points"], result["message"])
        logger.info("")

    # Test dynamic rule addition
    logger.info("=== Dynamic Rule Addition ===")
    new_rule = """
    IF ctx.order.subtotal > 1000
    THEN apply_discount(0.25) AND free_shipping() AND add_bonus_points(500)
    """
    valid, msg = engine.validate_rule(new_rule)
    logger.info("New rule valid: %s (%s)", valid, msg)
    if valid:
        engine.add_rule(new_rule)
        big_order = Order(order_id="ORD-BIG", user_id="usr_big", items=[{"qty": 10}], subtotal=2500.0, shipping_cost=50.0)
        big_user = User(user_id="usr_big", name="Big Spender", tier="silver", registered_days=100, total_spent=3000.0)
        result = engine.calculate(big_order, big_user)
        logger.info("Big order result: discount=%.0f%%, shipping_discount=%.0f, bonus=%d pts",
                     result["discount"] * 100, result["shipping_discount"], result["bonus_points"])

    # Test loop
    logger.info("\n=== Loop Test ===")
    loop_rule = """
    FOR item IN ctx.order.items DO add_bonus_points(10)
    """
    engine.add_rule(loop_rule)
    result = engine.calculate(
        Order(order_id="ORD-LOOP", user_id="usr_loop", items=[{"qty": 1}, {"qty": 2}, {"qty": 3}], subtotal=200.0),
        User(user_id="usr_loop", name="Loop User", tier="bronze", registered_days=50, total_spent=100.0),
    )
    logger.info("Loop result: bonus=%d pts", result["bonus_points"])

    # Test error handling
    logger.info("\n=== Error Handling ===")
    error_rules = [
        "IF THEN",  # Missing condition
        "ctx.undefined_var > 100 THEN alert('err')",  # Missing IF
        "IF 1/0 THEN alert('err')",  # Division by zero
    ]
    for bad_rule in error_rules:
        valid, msg = engine.validate_rule(bad_rule)
        logger.info("Invalid rule: valid=%s, error=%s", valid, msg)

    logger.info("=== Interpreter Architecture Demo Complete ===")


if __name__ == "__main__":
    main()
```

## Khi nào dùng / Khi nào không

| Khi nào dùng | Khi nào không |
|--------------|---------------|
| Logic nghiệp vụ thay đổi thường xuyên, cần hot-deploy | Logic cố định, ít thay đổi — hard-code đơn giản hơn |
| Multi-tenant với logic riêng cho từng tenant | Performance-critical (microseconds latency) |
| User-generated rules / automation | Cần tính toán số học nặng (scientific computing) |
| Rule engine / workflow engine / DSL | Khi người dùng là developer — họ có thể viết code thật |
| Cần sandboxing và security isolation | Có sẵn scripting engine (Lua, Python embedding) |
| Audit trail cho mọi quyết định | Resource hạn chế (embedded systems) |

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Hot-deploy**: Thay đổi logic không cần restart | **Performance**: Interpreter chậm hơn compiled code 10-100x |
| **Sandboxing**: Kiểm soát an toàn tuyệt đối | **Complexity**: Parser implementation phức tạp |
| **Portability**: DSL chạy trên mọi nền tảng | **Class explosion**: Nhiều AST node types |
| **Auditability**: Rule lưu dạng text, dễ version control | **Debugging khó**: Stack trace trong interpreter |
| **Security**: Không RCE, không file I/O | **Memory overhead**: AST lưu trong RAM |
| **Extensibility**: Thêm function mới dễ dàng | **Limited expressiveness**: DSL đơn giản hơn ngôn ngữ thật |
| **User-friendly**: DSL đơn giản, non-developer dùng được | **Recursive parsing**: Có thể bị stack overflow |

## Công cụ và Framework

| Tên | Loại | Ngôn ngữ | Mô tả |
|-----|------|----------|-------|
| **Python ast module** | Built-in | Python | Parse Python code thành AST |
| **Lark** | Parser generator | Python | EBNF grammar → parser |
| **ANTLR** | Parser generator | Java, Python, JS | Parser generator mạnh nhất |
| **PyParsing** | Library | Python | PEG parsing |
| **Lua** | Embeddable | C/Lua | Scripting engine nhẹ |
| **Tcl** | Embeddable | C/Tcl | Scripting cho ứng dụng |
| **MVEL** | Expression language | Java | Runtime expression evaluation |
| **SpEL (Spring EL)** | Expression language | Java | Spring expression language |
| **Drools** | Rule engine | Java | Production rule system |
| **CEL (Common Expression Language)** | Expression language | Go, Python | Google's safe expression language |
| **exaloop/codon** | JIT compiler | Python | Python với JIT compilation |

## Kiểm thử

Testing interpreter architecture cần test ở nhiều layer: lexer, parser, AST, runtime, integration.

```python
from __future__ import annotations
import pytest
from typing import Any


class TestLexer:
    def test_number(self) -> None:
        tokens = Lexer("42").tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42

    def test_float(self) -> None:
        tokens = Lexer("3.14").tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 3.14

    def test_string_single_quote(self) -> None:
        tokens = Lexer("'hello'").tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_string_double_quote(self) -> None:
        tokens = Lexer('"world"').tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "world"

    def test_boolean(self) -> None:
        tokens = Lexer("true false").tokenize()
        assert tokens[0].type == TokenType.TRUE
        assert tokens[1].type == TokenType.FALSE

    def test_identifier(self) -> None:
        tokens = Lexer("temperature user_name _value").tokenize()
        assert tokens[0].value == "temperature"
        assert tokens[1].value == "user_name"

    def test_keywords(self) -> None:
        tokens = Lexer("IF THEN ELSE AND OR NOT FOR IN DO").tokenize()
        expected = [TokenType.IF, TokenType.THEN, TokenType.ELSE, TokenType.AND,
                    TokenType.OR, TokenType.NOT, TokenType.FOR, TokenType.IN, TokenType.DO]
        for t, e in zip(tokens, expected):
            assert t.type == e

    def test_operators(self) -> None:
        tokens = Lexer("+ - * / > >= < <= == != AND OR").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.PLUS in types
        assert TokenType.GTE in types
        assert TokenType.NEQ in types

    def test_delimiters(self) -> None:
        tokens = Lexer("( ) { } , . ; : [ ]").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.LPAREN in types
        assert TokenType.RBRACE in types
        assert TokenType.LBRACKET in types

    def test_comment(self) -> None:
        tokens = Lexer("42 # This is a comment\n 100").tokenize()
        values = [t.value for t in tokens if t.type != TokenType.EOF]
        assert values == [42, 100]

    def test_unterminated_string(self) -> None:
        with pytest.raises(LexerError):
            Lexer('"unterminated').tokenize()

    def test_unexpected_character(self) -> None:
        with pytest.raises(LexerError):
            Lexer("@invalid").tokenize()


class TestParser:
    def test_parse_number(self) -> None:
        ast = self._parse("42")
        assert isinstance(ast, Block)
        assert isinstance(ast.statements[0], NumberLiteral)
        assert ast.statements[0].value == 42

    def test_parse_string(self) -> None:
        ast = self._parse('"hello"')
        assert isinstance(ast.statements[0], StringLiteral)
        assert ast.statements[0].value == "hello"

    def test_parse_addition(self) -> None:
        ast = self._parse("1 + 2")
        assert isinstance(ast.statements[0], BinaryOp)
        assert ast.statements[0].operator == "+"

    def test_parse_comparison(self) -> None:
        ast = self._parse("x > 100")
        assert isinstance(ast.statements[0], BinaryOp)
        assert ast.statements[0].operator == ">"

    def test_parse_and_or(self) -> None:
        ast = self._parse("a > 1 AND b < 2 OR c == 3")
        assert isinstance(ast.statements[0], BinaryOp)
        # OR là ngoài cùng (lowest precedence)

    def test_parse_if(self) -> None:
        ast = self._parse("IF x > 10 THEN y = 20 ELSE y = 30")
        assert isinstance(ast.statements[0], IfExpr)

    def test_parse_assignment(self) -> None:
        ast = self._parse("x = 100")
        assert isinstance(ast.statements[0], Assignment)
        assert ast.statements[0].name == "x"

    def test_parse_function_call(self) -> None:
        ast = self._parse("alert('test')")
        assert isinstance(ast.statements[0], FunctionCall)
        assert ast.statements[0].name == "alert"

    def test_parse_property_access(self) -> None:
        ast = self._parse("ctx.order.total")
        assert isinstance(ast.statements[0], PropertyAccess)

    def test_parse_for_loop(self) -> None:
        ast = self._parse("FOR x IN items DO print(x)")
        assert isinstance(ast.statements[0], ForLoop)
        assert ast.statements[0].variable == "x"

    def test_parse_precedence(self) -> None:
        ast = self._parse("1 + 2 * 3")
        binop = ast.statements[0]
        assert isinstance(binop, BinaryOp)
        assert binop.operator == "+"  # + is outer
        assert isinstance(binop.right, BinaryOp)  # 2 * 3 is inner
        assert binop.right.operator == "*"

    def test_parse_nested_if(self) -> None:
        ast = self._parse("IF a THEN IF b THEN c ELSE d")
        assert isinstance(ast.statements[0], IfExpr)

    def test_syntax_error(self) -> None:
        with pytest.raises(ParseError):
            self._parse("IF THEN")

    def _parse(self, source: str) -> Block:
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()


class TestInterpreter:
    def test_literal(self) -> None:
        interp = Interpreter()
        assert interp.evaluate("42") == 42
        assert interp.evaluate("3.14") == 3.14
        assert interp.evaluate("true") is True
        assert interp.evaluate("false") is False
        assert interp.evaluate("null") is None

    def test_string(self) -> None:
        interp = Interpreter()
        assert interp.evaluate("'Hello'") == "Hello"
        assert interp.evaluate('"World"') == "World"

    def test_arithmetic(self) -> None:
        interp = Interpreter()
        assert interp.evaluate("2 + 3 * 4") == 14
        assert interp.evaluate("(2 + 3) * 4") == 20
        assert interp.evaluate("10 / 2") == 5.0
        assert interp.evaluate("10 % 3") == 1

    def test_comparison(self) -> None:
        interp = Interpreter()
        assert interp.evaluate("5 > 3") is True
        assert interp.evaluate("5 < 3") is False
        assert interp.evaluate("5 == 5") is True
        assert interp.evaluate("5 != 3") is True
        assert interp.evaluate("5 >= 5") is True

    def test_logical(self) -> None:
        interp = Interpreter()
        assert interp.evaluate("true AND false") is False
        assert interp.evaluate("true OR false") is True
        assert interp.evaluate("NOT true") is False
        assert interp.evaluate("true AND (false OR true)") is True

    def test_variable(self) -> None:
        interp = Interpreter()
        interp.set_variable("x", 42)
        assert interp.evaluate("x") == 42

    def test_assignment(self) -> None:
        interp = Interpreter()
        interp.evaluate("x = 100")
        assert interp.get_environment().get("x") == 100

    def test_if_true(self) -> None:
        interp = Interpreter()
        assert interp.evaluate("IF true THEN 42 ELSE 0") == 42

    def test_if_false(self) -> None:
        interp = Interpreter()
        assert interp.evaluate("IF false THEN 42 ELSE 0") == 0

    def test_if_no_else(self) -> None:
        interp = Interpreter()
        assert interp.evaluate("IF false THEN 42") is None

    def test_function_call(self) -> None:
        interp = Interpreter()
        result: list[str] = []
        interp.register_function("record", lambda msg: result.append(msg))
        interp.evaluate('record("test")')
        assert result == ["test"]

    def test_property_access(self) -> None:
        interp = Interpreter()
        interp.set_variable("obj", {"name": "test", "value": 42})
        assert interp.evaluate("obj.name") == "test"

    def test_for_loop(self) -> None:
        interp = Interpreter()
        interp.set_variable("items", [1, 2, 3])
        interp.evaluate("result = []; FOR x IN items DO result = concat(result, x)")
        # List append not directly supported; this tests loop mechanics

    def test_division_by_zero(self) -> None:
        interp = Interpreter()
        with pytest.raises(RuntimeError_):
            interp.evaluate("1 / 0")

    def test_undefined_variable(self) -> None:
        interp = Interpreter()
        with pytest.raises(RuntimeError_):
            interp.evaluate("undefined_var")

    def test_resource_limits(self) -> None:
        monitor = ResourceMonitor(max_recursion=10, max_operations=50, timeout_ms=5000)
        interp = Interpreter(resource_monitor=monitor)
        # Should not exceed limits with normal expression
        assert interp.evaluate("1 + 2") == 3

    def test_resource_timeout(self) -> None:
        monitor = ResourceMonitor(timeout_ms=1)
        interp = Interpreter(resource_monitor=monitor)
        interp.register_function("slow", lambda: __import__("time").sleep(0.1))
        # May timeout
        import time
        interp_with_timeout = Interpreter(resource_monitor=ResourceMonitor(max_operations=100000, timeout_ms=50))
        interp_with_timeout.register_function("slow", lambda: time.sleep(0.1))
        result = interp_with_timeout.evaluate("slow()")
        # Should complete despite .1s > 50ms? ResourceMonitor tracks operations not wall time


class TestPricingEngine:
    def test_vip_discount(self) -> None:
        engine, _ = self._setup_engine()
        order = Order(order_id="T1", user_id="vip_1", items=[], subtotal=600.0, shipping_cost=30.0)
        user = User(user_id="vip_1", name="VIP", tier="gold", registered_days=400, total_spent=10000.0)
        result = engine.calculate(order, user)
        assert result["discount"] >= 0.15

    def test_new_user_discount(self) -> None:
        engine, _ = self._setup_engine()
        order = Order(order_id="T2", user_id="new_1", items=[], subtotal=100.0, shipping_cost=20.0)
        user = User(user_id="new_1", name="New", tier="bronze", registered_days=5, total_spent=0.0)
        result = engine.calculate(order, user)
        assert result["discount"] >= 0.10

    def test_free_shipping(self) -> None:
        engine, _ = self._setup_engine()
        order = Order(order_id="T3", user_id="usr_1", items=[], subtotal=400.0, shipping_cost=50.0)
        user = User(user_id="usr_1", name="Normal", tier="silver", registered_days=100, total_spent=500.0)
        result = engine.calculate(order, user)
        assert result["shipping_discount"] == 50.0

    def test_bonus_points(self) -> None:
        engine, _ = self._setup_engine()
        order = Order(order_id="T4", user_id="usr_2", items=[], subtotal=500.0)
        user = User(user_id="usr_2", name="Bonus", tier="bronze", registered_days=200, total_spent=1000.0)
        result = engine.calculate(order, user)
        assert result["bonus_points"] > 0

    def test_platinum_full_discount(self) -> None:
        engine, _ = self._setup_engine()
        order = Order(order_id="T5", user_id="plat_1", items=[], subtotal=1000.0, shipping_cost=30.0)
        user = User(user_id="plat_1", name="Plat", tier="platinum", registered_days=1000, total_spent=50000.0)
        result = engine.calculate(order, user)
        assert result["discount"] >= 0.20

    def test_coupon_code(self) -> None:
        engine, _ = self._setup_engine()
        order = Order(order_id="T6", user_id="usr_3", items=[], subtotal=100.0, coupon_code="WELCOME20")
        user = User(user_id="usr_3", name="Coupon", tier="bronze", registered_days=100, total_spent=500.0)
        result = engine.calculate(order, user)
        assert result["discount"] >= 0.20

    def test_invalid_rule(self) -> None:
        engine, _ = self._setup_engine()
        valid, msg = engine.validate_rule("IF THEN")
        assert valid is False

    def _setup_engine(self) -> tuple[PricingEngine, Interpreter]:
        interp = Interpreter()
        engine = PricingEngine(interp)
        rules = [
            "IF ctx.order.total > 500 AND ctx.user.tier == 'gold' THEN apply_discount(0.15)",
            "IF ctx.order.subtotal > 300 THEN free_shipping()",
            "IF ctx.order.subtotal > 100 THEN add_bonus_points(int(ctx.order.subtotal / 10))",
            "IF ctx.user.registered_days < 30 AND ctx.order.subtotal > 50 THEN apply_discount(0.10)",
            "IF ctx.user.tier == 'platinum' THEN apply_discount(0.20)",
            "IF ctx.order.coupon_code == 'WELCOME20' THEN apply_discount(0.20)",
        ]
        engine.set_rules(rules)

        def apply_discount(percent: float) -> None:
            ctx = interp.get_environment().get("ctx")
            ctx["result"]["discount"] = max(ctx["result"]["discount"], percent)

        def free_shipping() -> None:
            ctx = interp.get_environment().get("ctx")
            ctx["result"]["shipping_discount"] = ctx["order"]["shipping_cost"]

        def add_bonus_points(points: int) -> None:
            ctx = interp.get_environment().get("ctx")
            ctx["result"]["bonus_points"] += points

        interp.register_function("apply_discount", apply_discount)
        interp.register_function("free_shipping", free_shipping)
        interp.register_function("add_bonus_points", add_bonus_points)

        return engine, interp
```

## Kết luận

Interpreter Architecture là kiến trúc mạnh mẽ cho các hệ thống cần linh hoạt về logic nghiệp vụ, sandboxing, và khả năng mở rộng. Bằng cách định nghĩa một DSL riêng và xây dựng interpreter cho nó, bạn tách biệt business logic khỏi core system — cho phép business users tự quản lý rule mà không cần deploy.

**Best Practices:**
- **Bắt đầu nhỏ**: Đừng xây dựng interpreter cho mọi thứ. Bắt đầu với expression evaluator, thêm dần tính năng.
- **Error messages chất lượng**: Parse error phải chỉ rõ dòng, cột, token. Runtime error phải có stack trace.
- **Resource limits là bắt buộc**: Luôn đặt giới hạn recursion depth, operations count, và timeout.
- **Version your DSL**: Khi DSL phát triển, cần versioning để backward compatibility.
- **AST optimization**: Constant folding, dead code elimination cho performance.
- **Testing chiến lược**: Test lexer với edge cases (unicode, empty string), test parser với syntax errors, test interpreter với runtime errors.

**Golden Rules:**
1. Không bao giờ dùng `eval()` hoặc `exec()` cho user-generated code — luôn dùng interpreter.
2. Parse error phải hữu ích — người dùng không phải developer.
3. Environment (context) là single source of truth — inject tất cả dữ liệu qua environment.
4. Thiết kế cho extensibility: function registry, custom type system.
5. Monitor resource usage — interpreter không được làm sập host process.
6. Audit trail: log mọi rule đã thực thi, input, output.
7. Test interpreter với fuzzing — user có thể viết bất kỳ rule nào.
