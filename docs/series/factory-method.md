---
id: factory-method
title: Factory Method
sidebar_label: 🏭 Factory Method
sidebar_position: 3
---

# Factory Method

**Factory Method** định nghĩa một interface để tạo object, nhưng để subclass quyết định class nào được instantiate.

## Bài toán

Bạn viết ứng dụng quản lý **vận chuyển** hàng hóa. Ban đầu chỉ có vận chuyển bằng **xe tải**. Khách hàng mới yêu cầu thêm **tàu thủy**. Code hiện tại dùng `Truck()` khắp nơi — bạn phải sửa từng chỗ.

## Giải pháp

Factory Method tách việc **khởi tạo** ra khỏi **logic nghiệp vụ**. Thêm phương thức vận chuyển mới chỉ cần tạo class mới — không sửa code cũ.

```python
from abc import ABC, abstractmethod

class Transport(ABC):
    @abstractmethod
    def deliver(self):
        pass

class Truck(Transport):
    def deliver(self):
        return '🚚 Giao hàng bằng đường bộ'

class Ship(Transport):
    def deliver(self):
        return '🚢 Giao hàng bằng đường biển'

class Logistics(ABC):
    @abstractmethod
    def create_transport(self) -> Transport:
        pass

    def plan_delivery(self):
        transport = self.create_transport()
        return transport.deliver()

class RoadLogistics(Logistics):
    def create_transport(self):
        return Truck()

class SeaLogistics(Logistics):
    def create_transport(self):
        return Ship()

# Sử dụng
app = RoadLogistics()
print(app.plan_delivery())  # 🚚 Giao hàng bằng đường bộ

app2 = SeaLogistics()
print(app2.plan_delivery())  # 🚢 Giao hàng bằng đường biển
```

## Khi nào dùng

- Không biết trước class chính xác của object sẽ tạo
- Muốn cho phép mở rộng bằng cách thêm subclass
- Muốn giảm sự phụ thuộc giữa code gọi và implementation cụ thể

## Thực tế

- Django Form factory, ModelForm
- SQLAlchemy engine creation
- Framework hooks cho phép override cách tạo object
