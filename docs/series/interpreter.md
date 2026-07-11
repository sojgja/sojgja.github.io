---
id: interpreter
title: Interpreter
sidebar_label: 📖 Interpreter
sidebar_position: 16
---

# Interpreter

**Interpreter** định nghĩa ngữ pháp (grammar) cho một ngôn ngữ đơn giản và cung cấp interpreter để diễn giải các câu trong ngôn ngữ đó.

## Bài toán

Ứng dụng cần hỗ trợ **tìm kiếm nâng cao** với các biểu thức: `"Java AND Python"`, `"Design OR Pattern"`, `"NOT JavaScript"`, `"(Java AND Spring) OR Node"`. Viết parser cho các expression phức tạp rất khó nếu dùng if-else.

## Giải pháp

Interpreter biểu diễn mỗi rule ngữ pháp thành một class, xây cây cú pháp (AST) và diễn giải.

```python
from abc import ABC, abstractmethod

class Expression(ABC):
    @abstractmethod
    def interpret(self, context: str) -> bool:
        pass

class TerminalExpression(Expression):
    def __init__(self, word: str):
        self.word = word

    def interpret(self, context: str) -> bool:
        return self.word.lower() in context.lower()

class OrExpression(Expression):
    def __init__(self, expr1: Expression, expr2: Expression):
        self.expr1 = expr1
        self.expr2 = expr2

    def interpret(self, context: str) -> bool:
        return self.expr1.interpret(context) or self.expr2.interpret(context)

class AndExpression(Expression):
    def __init__(self, expr1: Expression, expr2: Expression):
        self.expr1 = expr1
        self.expr2 = expr2

    def interpret(self, context: str) -> bool:
        return self.expr1.interpret(context) and self.expr2.interpret(context)

class NotExpression(Expression):
    def __init__(self, expr: Expression):
        self.expr = expr

    def interpret(self, context: str) -> bool:
        return not self.expr.interpret(context)

# Sử dụng
java = TerminalExpression('Java')
spring = TerminalExpression('Spring')
js = TerminalExpression('JavaScript')

query = AndExpression(
    OrExpression(java, spring),
    NotExpression(js)
)

text = 'Tôi học Java và Spring Boot'
print(query.interpret(text))  # True
```

## Khi nào dùng

- Cần diễn giải một ngôn ngữ đơn giản
- Grammar ổn định và không quá phức tạp
- Cần xây dựng cây cú pháp

## Thực tế

- Django ORM filter: `User.objects.filter(name__contains='John')`
- SQL parser
- Template engine (Jinja2, Django template)
- Regular expression engine
