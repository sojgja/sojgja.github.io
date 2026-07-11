---
id: decorator
title: Decorator
sidebar_label: 🎄 Decorator
sidebar_position: 10
---

# Decorator

**Decorator** cho phép thêm hành vi mới vào object một cách linh hoạt bằng cách "bọc" nó trong các wrapper object.

## Bài toán

Ứng dụng coffee shop: có `Espresso`, `Latte`, `Cappuccino`. Khách hàng có thể thêm topping: sữa, đường, kem, caramel, ... Nếu dùng kế thừa: `EspressoWithMilk`, `EspressoWithMilkAndSugar`, ... số class tăng vô hạn.

## Giải pháp

Decorator "bọc" coffee gốc bằng các topping. Mỗi topping là một decorator có cùng interface, thêm chi phí và mô tả vào kết quả.

```python
from abc import ABC, abstractmethod

class Coffee(ABC):
    @abstractmethod
    def cost(self): pass

    @abstractmethod
    def description(self): pass

class Espresso(Coffee):
    def cost(self):
        return 25000

    def description(self):
        return 'Espresso'

class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee

    def cost(self):
        return self._coffee.cost()

    def description(self):
        return self._coffee.description()

class MilkDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 5000

    def description(self):
        return self._coffee.description() + ' + Sữa'

class CaramelDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 8000

    def description(self):
        return self._coffee.description() + ' + Caramel'

# Sử dụng
coffee = Espresso()
coffee = MilkDecorator(coffee)
coffee = CaramelDecorator(coffee)

print(f'{coffee.description()}: {coffee.cost()} VND')
# Espresso + Sữa + Caramel: 38000 VND
```

## Khi nào dùng

- Cần thêm/xóa hành vi ở runtime
- Không muốn dùng kế thừa (quá nhiều class, behavior tĩnh)
- Cần kết hợp nhiều behavior linh hoạt

## Thực tế

- Python decorator `@staticmethod`, `@property`, `@login_required`
- Django middleware stack (request → auth → session → csrf → view)
- I/O wrapping: `BufferedReader(GzipReader(FileReader(path)))`
