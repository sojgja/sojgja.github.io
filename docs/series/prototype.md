---
id: prototype
title: Prototype
sidebar_label: 🧬 Prototype
sidebar_position: 6
---

# Prototype

**Prototype** tạo object mới bằng cách **clone** một object hiện có (prototype) thay vì gọi constructor.

## Bài toán

Ứng dụng chỉnh sửa hình ảnh có các shape: Circle, Rectangle. Mỗi shape có nhiều thuộc tính (`color`, `border_width`, `opacity`, ...). Khi người dùng vẽ shape mới, bạn phải khởi tạo từ constructor với đầy đủ tham số. Nếu người dùng duplicate một shape đã chỉnh sửa, bạn phải đọc từng thuộc tính rồi tạo mới.

## Giải pháp

Prototype clone object hiện có, giữ nguyên toàn bộ trạng thái. Không cần constructor phức tạp.

```python
import copy
from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self):
        self.color = 'black'
        self.border_width = 1

    @abstractmethod
    def clone(self):
        pass

class Circle(Shape):
    def __init__(self):
        super().__init__()
        self.radius = 10

    def clone(self):
        return copy.deepcopy(self)

# Sử dụng
original = Circle()
original.color = 'red'
original.radius = 25

cloned = original.clone()
print(cloned.color)        # red
print(cloned.radius)       # 25
print(type(cloned).__name__)  # Circle
print(cloned is original)  # False — khác object
```

## Khi nào dùng

- Khởi tạo object tốn kém tài nguyên
- Cần nhiều object chỉ khác nhau vài thuộc tính
- Object có quá nhiều thuộc tính, không muốn khởi tạo từ đầu

## Thực tế

- `copy.copy()` / `copy.deepcopy()` trong Python
- Django Model khởi tạo từ instance có sẵn
- Prototype pattern trong game (spawn quái vật từ prototype)
