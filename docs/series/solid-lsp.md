---
id: solid-lsp
title: L — Liskov Substitution Principle
sidebar_label: L — Liskov Substitution
sidebar_position: 28
---

# L — Liskov Substitution Principle

> **"Objects of a superclass should be replaceable with objects of its subclasses without breaking the system."** — Barbara Liskov

Subclass phải có thể **thay thế** base class mà không gây ra lỗi hay hành vi bất thường.

## Bài toán: Lớp `Rectangle` và `Square`

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_width(self, w):
        self.width = w

    def set_height(self, h):
        self.height = h

    def area(self):
        return self.width * self.height
```

Kế thừa: `Square extends Rectangle`:

```python
class Square(Rectangle):
    def set_width(self, w):
        self.width = w
        self.height = w  # Vuông nên width = height

    def set_height(self, h):
        self.width = h
        self.height = h  # Vuông nên width = height
```

Nhìn có vẻ hợp lý. Nhưng hãy xem điều gì xảy ra:

```python
def test_area(rect: Rectangle):
    rect.set_width(5)
    rect.set_height(4)
    expected = 20  # 5 * 4
    actual = rect.area()
    assert actual == expected, f'Expected {expected}, got {actual}'

test_area(Rectangle(0, 0))  # ✅ OK: 20 = 20
test_area(Square(0, 0))     # ❌ FAIL: 16 != 20 (vì set_width(5) set height lên 5 luôn)
```

**Vấn đề:** `Square` không thể thay thế `Rectangle` được. Hành vi của `set_width` khác nhau.

## Giải pháp đúng: Không dùng kế thừa sai

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self): pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

# ✅ Cả hai đều thay thế được Shape
def print_area(shape: Shape):
    print(f'Diện tích: {shape.area()}')

print_area(Rectangle(5, 4))  # 20
print_area(Square(5))        # 25
```

## Ví dụ thực tế: Persistence

Vi phạm LSP:

```python
class UserRepository:
    def save(self, user): pass
    def update(self, user): pass
    def delete(self, user_id): pass

class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self.data = {}

    def save(self, user):
        self.data[user.id] = user

    def update(self, user):
        raise NotImplementedError("InMemory không hỗ trợ update!")  # ❌

    def delete(self, user_id):
        del self.data[user_id]
```

✅ Đúng LSP:

```python
from abc import ABC, abstractmethod

class ReadOnlyRepository(ABC):
    @abstractmethod
    def get_all(self): pass

class WriteRepository(ABC):
    @abstractmethod
    def save(self, entity): pass

class UserReadRepo(ReadOnlyRepository):
    def get_all(self):
        return ['user1', 'user2']

class UserWriteRepo(WriteRepository):
    def save(self, user):
        print(f'💾 Lưu user: {user}')
```

## Dấu hiệu nhận biết vi phạm LSP

- Subclass **override** method và ném exception "Not supported"
- Subclass **bỏ qua** method (empty implementation)
- Subclass thay đổi **hành vi** của method so với base class
- Dùng `isinstance()` để kiểm tra type và xử lý riêng từng subclass

## Kết luận

LSP không phải là "kế thừa là xấu", mà là "kế thừa đúng cách". Nếu subclass không thay thế được base class trong mọi tình huống, thì đó không phải là kế thừa — đó là kế thừa sai.
