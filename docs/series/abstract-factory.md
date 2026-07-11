---
id: abstract-factory
title: Abstract Factory
sidebar_label: 🏭🏭 Abstract Factory
sidebar_position: 4
---

# Abstract Factory

**Abstract Factory** cung cấp interface để tạo các **họ object** có liên quan mà không cần chỉ định class cụ thể.

## Bài toán

Ứng dụng furniture của bạn cần hỗ trợ nhiều **phong cách**: Hiện đại (Modern) và Cổ điển (Victorian). Mỗi phong cách có bộ sản phẩm riêng: Ghế (`Chair`), Bàn (`Table`), Tủ (`Sofa`). Nếu dùng `ModernChair()`, `VictorianChair()` rải rác, code sẽ rất khó bảo trì.

## Giải pháp

Abstract Factory nhóm các sản phẩm liên quan thành một **họ** (family). Mỗi factory cụ thể tạo ra một họ sản phẩm thống nhất.

```python
from abc import ABC, abstractmethod

class Chair(ABC):
    @abstractmethod
    def sit(self): pass

class Table(ABC):
    @abstractmethod
    def place(self): pass

class ModernChair(Chair):
    def sit(self):
        return '💺 Ngồi ghế hiện đại — đơn giản, tinh tế'

class ModernTable(Table):
    def place(self):
        return '🪑 Bàn hiện đại — mặt kính trong suốt'

class VictorianChair(Chair):
    def sit(self):
        return '👑 Ngồi ghế cổ điển — chạm khắc tinh xảo'

class VictorianTable(Table):
    def place(self):
        return '🪵 Bàn cổ điển — gỗ óc chó nặng trịch'

class FurnitureFactory(ABC):
    @abstractmethod
    def create_chair(self) -> Chair: pass
    @abstractmethod
    def create_table(self) -> Table: pass

class ModernFurnitureFactory(FurnitureFactory):
    def create_chair(self):
        return ModernChair()
    def create_table(self):
        return ModernTable()

class VictorianFurnitureFactory(FurnitureFactory):
    def create_chair(self):
        return VictorianChair()
    def create_table(self):
        return VictorianTable()

# Sử dụng
def setup_room(factory: FurnitureFactory):
    chair = factory.create_chair()
    table = factory.create_table()
    print(chair.sit())
    print(table.place())

setup_room(ModernFurnitureFactory())
# 💺 Ngồi ghế hiện đại — đơn giản, tinh tế
# 🪑 Bàn hiện đại — mặt kính trong suốt
```

## Khi nào dùng

- Hệ thống cần làm việc với nhiều họ sản phẩm khác nhau
- Muốn đảm bảo các sản phẩm trong cùng họ được dùng cùng nhau
- Muốn thêm họ sản phẩm mới mà không sửa code client

## Thực tế

- UI toolkit hỗ trợ nhiều theme: Material, Fluent
- Cross-platform app (Windows factory, macOS factory)
- Database driver families (PostgreSQL factory, MySQL factory)
