---
id: strategy
title: Strategy
sidebar_label: 🧠 Strategy
sidebar_position: 22
---

# Strategy

**Strategy** định nghĩa họ các thuật toán, đóng gói từng thuật toán và làm cho chúng hoán đổi cho nhau. Strategy cho phép thuật toán thay đổi độc lập với client sử dụng nó.

## Bài toán

App **xử lý ảnh** hỗ trợ nhiều bộ lọc: sepia, grayscale, negative, blur, ... Số lượng filter ngày càng tăng. Nếu viết tất cả trong một class với if-else, class sẽ phình to và mỗi lần thêm filter phải sửa code cũ.

## Giải pháp

Strategy đóng gói mỗi filter thành một class riêng. ImageProcessor nhận filter qua constructor — không cần biết filter cụ thể.

```python
from abc import ABC, abstractmethod

class FilterStrategy(ABC):
    @abstractmethod
    def apply(self, image: str) -> str:
        pass

class SepiaFilter(FilterStrategy):
    def apply(self, image: str) -> str:
        return f'{image} → [Sepia: ấm vàng]'

class GrayscaleFilter(FilterStrategy):
    def apply(self, image: str) -> str:
        return f'{image} → [Grayscale: đen trắng]'

class NegativeFilter(FilterStrategy):
    def apply(self, image: str) -> str:
        return f'{image} → [Negative: đảo màu]'

class ImageProcessor:
    def __init__(self, strategy: FilterStrategy = None):
        self._strategy = strategy

    def set_strategy(self, strategy: FilterStrategy):
        self._strategy = strategy

    def process(self, image: str) -> str:
        if not self._strategy:
            return f'{image} → [Không filter]'
        return self._strategy.apply(image)

# Sử dụng
processor = ImageProcessor()

photo = 'Ảnh ở bãi biển'
print(processor.process(photo))  # Không filter

processor.set_strategy(SepiaFilter())
print(processor.process(photo))  # Sepia

processor.set_strategy(NegativeFilter())
print(processor.process(photo))  # Negative
```

## Bài toán thực tế hơn: Tính phí vận chuyển

```python
class ShippingStrategy(ABC):
    @abstractmethod
    def calculate(self, weight: float) -> float:
        pass

class ExpressShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 30000  # 30k/kg

class StandardShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 15000 + 20000  # 15k/kg + phí base

class EconomyShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 8000  # 8k/kg

class Order:
    def __init__(self, weight: float, strategy: ShippingStrategy):
        self.weight = weight
        self.strategy = strategy

    def shipping_cost(self) -> float:
        return self.strategy.calculate(self.weight)

order = Order(2.5, ExpressShipping())
print(f'Phí ship: {order.shipping_cost():,.0f} VND')  # 75,000 VND
```

## Khi nào dùng

- Có nhiều thuật toán cho cùng một tác vụ
- Cần hoán đổi thuật toán tại runtime
- Muốn tránh class explosion với kế thừa

## Thực tế

- Django authentication backends
- Serializer/Deserializer formats (JSON, XML, YAML)
- Compression algorithms (Gzip, Zip, Bz2)
- Route calculation (xe máy, ô tô, đi bộ)
