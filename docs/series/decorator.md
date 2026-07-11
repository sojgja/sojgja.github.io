---
id: decorator
title: Decorator
sidebar_label: 🎄 Decorator
sidebar_position: 10
---

# Decorator

> "Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality." — Erich Gamma, *Design Patterns: Elements of Reusable Object-Oriented Software*

## Bài toán chi tiết

Một công ty khởi nghiệp trong lĩnh vực foodtech đang xây dựng hệ thống đặt đồ uống trực tuyến. Trung tâm của hệ thống là một module tính giá cho các loại đồ uống với topping. Một ly cà phê có thể kết hợp với nhiều loại topping khác nhau: sữa tươi (5,000đ), caramel (8,000đ), kem tươi (10,000đ), siro vanilla (6,000đ), bột cacao (4,000đ), và nhiều hơn nữa. Khách hàng có thể chọn bất kỳ tổ hợp topping nào, với số lượng tùy ý.

Ban đầu, các kỹ sư dùng kế thừa để giải quyết. Họ tạo ra `Espresso`, `Latte`, `Cappuccino` là các class nền tảng. Sau đó, họ tạo ra `EspressoWithMilk`, `EspressoWithMilkAndCaramel`, `LatteWithWhippedCream`, v.v. Mỗi tổ hợp là một class riêng. Với 5 loại đồ uống nền và 10 loại topping, số class có thể lên tới 5 × 2¹⁰ = 5.120 tổ hợp — một con số khủng khiếp. Rõ ràng, kế thừa là bất khả thi.

Đội ngũ chuyển sang giải pháp "thông minh" hơn: một class `Beverage` khổng lồ với một danh sách topping và câu lệnh if-else trong method `get_cost()`:

```python
def get_cost(self):
    cost = self.base_cost
    for topping in self.toppings:
        if topping == "milk": cost += 5000
        elif topping == "caramel": cost += 8000
        # ... thêm topping mới phải sửa đây
```

Giải pháp này vi phạm Open/Closed Principle một cách nghiêm trọng: mỗi lần thêm topping mới, phải sửa class `Beverage`. Hơn nữa, logic nghiệp vụ như giảm giá theo combo (mua 3 topping được giảm 10%), topping theo mùa (chỉ có vào mùa hè), và giới hạn số lượng topping tối đa đều phải nhồi nhét vào một class duy nhất — class này trở thành "god object" với hàng nghìn dòng code, cực kỳ khó bảo trì và kiểm thử.

Vấn đề cốt lõi: hành vi (topping) cần được thêm vào một cách linh hoạt tại runtime (khi khách chọn), với số lượng và tổ hợp không xác định trước. Kế thừa không giải quyết được vì nó tĩnh (compile-time). Cần một giải pháp động.

## Giải pháp với Pattern

Decorator Pattern giải quyết vấn đề này bằng cách cho phép "bọc" (wrap) một đối tượng gốc trong các lớp wrapper, mỗi lớp thêm một hành vi mới. Tất cả đều chia sẻ cùng một interface, nên client không biết mình đang tương tác với đối tượng gốc hay với một chuỗi decorator. Đây là một dạng đệ quy lồng nhau: decorator này có thể chứa decorator khác, và cứ thế tạo thành một "stack".

Cấu trúc Decorator gồm:
- **Component**: Interface chung cho cả đối tượng gốc và decorator. Trong ví dụ, đây là `Beverage`.
- **ConcreteComponent**: Đối tượng gốc có hành vi cơ sở — `Espresso`, `Latte`.
- **Decorator (abstract)**: Lớp trừu tượng implement Component và chứa một tham chiếu đến Component khác. Tất cả method của Decorator đều ủy quyền (delegate) cho component được wrap.
- **ConcreteDecorator**: Các lớp decorator cụ thể override một số method để thêm hành vi — `MilkDecorator`, `CaramelDecorator`.

Khi client gọi `get_cost()` trên một đối tượng đã qua nhiều lớp wrap, mỗi decorator cộng thêm chi phí của mình vào kết quả từ lớp bên trong, tạo thành một chuỗi xử lý. Số lượng và tổ hợp decorator là vô hạn, hoàn toàn do client quyết định tại runtime.

## Phân tích thiết kế

Decorator Pattern tuân thủ nghiêm ngặt **Single Responsibility Principle**: mỗi decorator chỉ chịu trách nhiệm thêm đúng một hành vi cụ thể. Nó cũng tuân thủ **Open/Closed Principle**: có thể thêm decorator mới mà không sửa code hiện có. Bên cạnh đó, **Favor composition over inheritance** là nguyên lý nền tảng — decorator dùng composition (wrap) thay vì kế thừa để mở rộng hành vi.

Một điểm quan trọng: Decorator thay đổi hành vi (behavior) của đối tượng, không phải interface. Đây là điểm khác biệt với Adapter (thay đổi interface) và Proxy (kiểm soát truy cập).

**Khi KHÔNG nên dùng Decorator:**
- Khi số lượng decorator cố định và nhỏ — kế thừa hoặc strategy pattern đơn giản hơn.
- Khi decorator cần truy cập vào internal state của đối tượng gốc — vi phạm encapsulation.
- Khi thứ tự decorator không quan trọng — lúc đó dùng danh sách đơn giản hơn.
- Khi decorator thay đổi interface (thêm method mới) — Decorator giữ nguyên interface.

**Trade-offs:**
- Tạo nhiều object nhỏ (mỗi decorator là một object) — có thể gây memory overhead.
- Debugging khó vì stack trace sâu và lồng nhau.
- Thứ tự decorator có thể ảnh hưởng đến kết quả — cần tài liệu rõ ràng.
- Configuration phức tạp — khởi tạo đối tượng với nhiều lớp wrap khó đọc.

## Ví dụ code hoàn chỉnh

### Cách làm sai: God object với if-else

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto


class ToppingType(Enum):
    MILK = "milk"
    CARAMEL = "caramel"
    WHIPPED_CREAM = "whipped_cream"
    VANILLA_SYRUP = "vanilla_syrup"
    COCOA_POWDER = "cocoa_powder"
    CINNAMON = "cinnamon"
    HAZELNUT = "hazelnut"


TOPPING_PRICES = {
    ToppingType.MILK: 5000,
    ToppingType.CARAMEL: 8000,
    ToppingType.WHIPPED_CREAM: 10000,
    ToppingType.VANILLA_SYRUP: 6000,
    ToppingType.COCOA_POWDER: 4000,
    ToppingType.CINNAMON: 3000,
    ToppingType.HAZELNUT: 7000,
}


class BadBeverage:
    """GOD OBJECT — vi phạm SRP và OCP nghiêm trọng."""

    BASE_PRICES = {
        "espresso": 25000,
        "latte": 35000,
        "cappuccino": 30000,
        "mocha": 40000,
        "matcha": 38000,
    }

    def __init__(self, base_name: str) -> None:
        self._base_name = base_name
        self._base_cost = self.BASE_PRICES.get(base_name, 20000)
        self._toppings: list[ToppingType] = []

    def add_topping(self, topping: ToppingType) -> None:
        self._toppings.append(topping)

    def get_cost(self) -> int:
        cost = self._base_cost
        for topping in self._toppings:
            # Mỗi lần thêm topping phải sửa đây — OCP violation!
            if topping == ToppingType.MILK:
                cost += 5000
            elif topping == ToppingType.CARAMEL:
                cost += 8000
            elif topping == ToppingType.WHIPPED_CREAM:
                cost += 10000
            elif topping == ToppingType.VANILLA_SYRUP:
                cost += 6000
            elif topping == ToppingType.COCOA_POWDER:
                cost += 4000
            # ...
        return cost

    def get_description(self) -> str:
        desc = f"{self._base_name.title()}"
        for topping in self._toppings:
            desc += f" + {topping.value}"
        return desc
```

### Cách đúng: Decorator Pattern

```python
# --- Component Interface ---
class Beverage:
    """Interface chung cho đồ uống và decorator."""

    def get_cost(self) -> int:
        raise NotImplementedError

    def get_description(self) -> str:
        raise NotImplementedError

    def get_calories(self) -> int:
        raise NotImplementedError

    def get_size_ml(self) -> int:
        raise NotImplementedError


# --- Concrete Components ---
class Espresso(Beverage):
    def __init__(self, size_ml: int = 250) -> None:
        self._size = size_ml

    def get_cost(self) -> int:
        return 25000

    def get_description(self) -> str:
        return f"Espresso ({self._size}ml)"

    def get_calories(self) -> int:
        return 5

    def get_size_ml(self) -> int:
        return self._size


class Latte(Beverage):
    def __init__(self, size_ml: int = 350) -> None:
        self._size = size_ml

    def get_cost(self) -> int:
        return 35000

    def get_description(self) -> str:
        return f"Latte ({self._size}ml)"

    def get_calories(self) -> int:
        return 180

    def get_size_ml(self) -> int:
        return self._size


class MatchaLatte(Beverage):
    def __init__(self, size_ml: int = 350) -> None:
        self._size = size_ml

    def get_cost(self) -> int:
        return 38000

    def get_description(self) -> str:
        return f"Matcha Latte ({self._size}ml)"

    def get_calories(self) -> int:
        return 160

    def get_size_ml(self) -> int:
        return self._size


# --- Abstract Decorator ---
class BeverageDecorator(Beverage):
    """Base decorator — giữ reference đến Beverage được wrap."""

    def __init__(self, beverage: Beverage) -> None:
        self._beverage = beverage

    def get_cost(self) -> int:
        return self._beverage.get_cost()

    def get_description(self) -> str:
        return self._beverage.get_description()

    def get_calories(self) -> int:
        return self._beverage.get_calories()

    def get_size_ml(self) -> int:
        return self._beverage.get_size_ml()


# --- Concrete Decorators ---
class MilkDecorator(BeverageDecorator):
    def get_cost(self) -> int:
        return self._beverage.get_cost() + 5000

    def get_description(self) -> str:
        return self._beverage.get_description() + " + Sữa tươi"

    def get_calories(self) -> int:
        return self._beverage.get_calories() + 120  # Whole milk


class CaramelDecorator(BeverageDecorator):
    def __init__(self, beverage: Beverage, extra_shot: bool = False) -> None:
        super().__init__(beverage)
        self._extra_shot = extra_shot

    def get_cost(self) -> int:
        extra = 3000 if self._extra_shot else 0
        return self._beverage.get_cost() + 8000 + extra

    def get_description(self) -> str:
        desc = self._beverage.get_description() + " + Caramel"
        if self._extra_shot:
            desc += " (extra shot)"
        return desc

    def get_calories(self) -> int:
        return self._beverage.get_calories() + 95


class WhippedCreamDecorator(BeverageDecorator):
    def get_cost(self) -> int:
        return self._beverage.get_cost() + 10000

    def get_description(self) -> str:
        return self._beverage.get_description() + " + Kem tươi"

    def get_calories(self) -> int:
        return self._beverage.get_calories() + 150


class VanillaSyrupDecorator(BeverageDecorator):
    def __init__(self, beverage: Beverage, pumps: int = 1) -> None:
        super().__init__(beverage)
        self._pumps = max(1, min(4, pumps))

    def get_cost(self) -> int:
        return self._beverage.get_cost() + 6000 * self._pumps

    def get_description(self) -> str:
        return f"{self._beverage.get_description()} + Vanilla ({self._pumps}pumps)"

    def get_calories(self) -> int:
        return self._beverage.get_calories() + 20 * self._pumps


class CinnamonDecorator(BeverageDecorator):
    def get_cost(self) -> int:
        return self._beverage.get_cost() + 3000

    def get_description(self) -> str:
        return self._beverage.get_description() + " + Bột quế"

    def get_calories(self) -> int:
        return self._beverage.get_calories() + 5


# --- Service Layer: Xây dựng đồ uống từ config ---
class Barista:
    """Barista xây dựng đồ uống với decorator chain."""

    TOPPING_MAP = {
        "milk": MilkDecorator,
        "caramel": CaramelDecorator,
        "whipped_cream": WhippedCreamDecorator,
        "vanilla": VanillaSyrupDecorator,
        "cinnamon": CinnamonDecorator,
    }

    @classmethod
    def build_beverage(cls, base: str, toppings: list[dict], size: int = 350) -> Beverage:
        bases = {
            "espresso": Espresso(size),
            "latte": Latte(size),
            "matcha": MatchaLatte(size),
        }
        beverage = bases.get(base)
        if beverage is None:
            raise ValueError(f"Unknown base: {base}")

        for topping_config in toppings:
            name = topping_config.get("name", "")
            decorator_cls = cls.TOPPING_MAP.get(name)
            if decorator_cls is None:
                raise ValueError(f"Unknown topping: {name}")
            kwargs = {k: v for k, v in topping_config.items() if k != "name"}
            beverage = decorator_cls(beverage, **kwargs)

        return beverage


# --- Usage ---
if __name__ == "__main__":
    # Cách 1: Thủ công (manual chaining)
    print("=== Manual Order ===")
    my_coffee: Beverage = Espresso(250)
    my_coffee = MilkDecorator(my_coffee)
    my_coffee = CaramelDecorator(my_coffee, extra_shot=True)
    my_coffee = WhippedCreamDecorator(my_coffee)
    print(f"{my_coffee.get_description()}: {my_coffee.get_cost()}đ, {my_coffee.get_calories()}kcal")

    # Cách 2: Tự động qua Barista
    print("\n=== Barista Order ===")
    order = Barista.build_beverage("latte", [
        {"name": "vanilla", "pumps": 2},
        {"name": "cinnamon"},
    ], size=500)
    print(f"{order.get_description()}: {order.get_cost()}đ, {order.get_calories()}kcal")

    # Cách 3: Nhiều loại đồ uống
    print("\n=== Multiple Orders ===")
    orders = [
        Espresso(),
        Barista.build_beverage("matcha", [{"name": "milk"}, {"name": "whipped_cream"}]),
        Barista.build_beverage("latte", [{"name": "caramel"}, {"name": "milk"}, {"name": "vanilla", "pumps": 1}]),
    ]
    for i, bev in enumerate(orders, 1):
        print(f"#{i}: {bev.get_description()} → {bev.get_cost()}đ")
```

## Sơ đồ UML

```
┌──────────────────────────────┐
│         Beverage             │
│       (Component)            │
│──────────────────────────────│
│+ get_cost()→int              │
│+ get_description()→str       │
│+ get_calories()→int          │
│+ get_size_ml()→int           │
└──────────┬───────────────────┘
           │
     ┌─────┴──────┬──────────────────────────────┐
     │            │                              │
┌────┴────┐ ┌─────┴──────┐     ┌────────────────┴─────────────────┐
│ Espresso│ │   Latte    │     │   BeverageDecorator (abstract)   │
│(Concrete│ │ (Concrete  │     │──────────────────────────────────│
│Component)│ │ Component) │     │ # _beverage: Beverage            │
└─────────┘ └────────────┘     └────────────────┬─────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────┐
                    │                           │                    │
          ┌─────────┴──────┐          ┌─────────┴──────┐  ┌────────┴─────────┐
          │ MilkDecorator  │          │CaramelDecorator│  │WhippedCream      │
          │(Concrete       │          │(Concrete       │  │Decorator         │
          │ Decorator)     │          │ Decorator)     │  │(Concrete Decor.) │
          │────────────────│          │────────────────│  │──────────────────│
          │+ get_cost()    │          │+ extra_shot    │  │+ get_cost()      │
          │  → super+5000  │          │+ get_cost()    │  │  → super+10000   │
          │+ get_desc()    │          │  → super+8000  │  └──────────────────┘
          │  → super +     │          │+ get_desc()    │
          │  " + Sữa tươi" │          │  → super +     │
          └────────────────┘          │  " + Caramel"  │
                                      └────────────────┘
```

## So sánh với Pattern liên quan

**Decorator vs Composite**: Cả hai đều dùng cấu trúc đệ quy và chung interface. Decorator thường wrap một component duy nhất (single child) để thêm hành vi, tạo thành chuỗi (chain). Composite quản lý nhiều children (multiple children) để tạo cấu trúc cây. Decorator bổ sung, Composite tập hợp. Hai pattern có thể kết hợp: Decorator có thể wrap một Composite node.

**Decorator vs Strategy**: Strategy thay thế toàn bộ thuật toán bên trong một đối tượng, trong khi Decorator thêm hành vi vào bên ngoài. Strategy dùng composition để ủy quyền, Decorator cũng dùng composition nhưng theo kiểu wrapper chain. Strategy ảnh hưởng đến cách thức hoạt động của method, Decorator chỉ thêm vào kết quả trước/sau.

**Decorator vs Proxy**: Cả hai đều wrap đối tượng và giữ nguyên interface. Proxy kiểm soát truy cập (lazy loading, protection, logging), Decorator thêm hành vi. Proxy tạo ra đối tượng thay thế với mục đích quản lý vòng đời của đối tượng thật; Decorator không quan tâm đến vòng đời, chỉ thêm chức năng.

## Ứng dụng thực tế

**1. Python Standard Library — `@property`, `@staticmethod`, `@classmethod`**: Đây là những decorator tích hợp sẵn của Python. `@property` biến method thành attribute, `@staticmethod` biến method thành static method — tất cả đều thêm hành vi vào function gốc:

```python
class Order:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    @property  # Decorator — thêm hành vi: gọi không cần ()
    def total(self) -> float:
        return sum(item["price"] * item["qty"] for item in self._items)

    @staticmethod
    def validate_item(item: dict) -> bool:
        return "price" in item and "qty" in item
```

**2. Django Middleware**: Middleware trong Django là một chuỗi decorator điển hình. Mỗi middleware nhận request, xử lý, có thể gọi middleware tiếp theo, và xử lý response trên đường về. Đây là decorator pattern ở cấp độ framework:

```python
# Django middleware như decorator chain
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
]

# Tương đương với:
def view(request):
    return HttpResponse("Hello")

view = SecurityMiddleware(SessionMiddleware(CommonMiddleware(AuthMiddleware(view))))
```

**3. I/O Stream trong Java và Python**: Java I/O dùng decorator rộng rãi. Python `io.BufferedReader` wrap `io.RawIOBase` để thêm buffering:

```python
import io
import gzip

# Chuỗi decorator trong I/O
raw = io.FileIO("data.txt", "r")           # Component gốc
buffered = io.BufferedReader(raw)            # Decorator: thêm buffer
# Trong thực tế:
with gzip.open("data.txt.gz", "rt") as f:   # gzip.open là decorator chain
    data = f.read()                          # Gzip → BufferedReader → FileIO
```

**4. Click/Argparse — CLI Framework**: Thư viện `click` dùng decorator để xây dựng CLI:

```python
import click

@click.command()          # Decorator
@click.option('--count', default=1)  # Decorator thêm option
@click.option('--name', prompt='Your name')  # Decorator thêm prompt
def hello(count, name):
    for _ in range(count):
        print(f"Hello {name}!")
```

## Kiểm thử

```python
import pytest
from decorator import (
    Beverage, Espresso, Latte, MatchaLatte,
    MilkDecorator, CaramelDecorator, WhippedCreamDecorator,
    VanillaSyrupDecorator, CinnamonDecorator, Barista,
)


class TestBaseBeverages:
    def test_espresso_cost(self) -> None:
        espresso = Espresso()
        assert espresso.get_cost() == 25000
        assert "Espresso" in espresso.get_description()

    def test_latte_cost(self) -> None:
        latte = Latte()
        assert latte.get_cost() == 35000

    def test_matcha_cost(self) -> None:
        matcha = MatchaLatte()
        assert matcha.get_cost() == 38000


class TestDecorators:
    def test_milk_decorator(self) -> None:
        beverage: Beverage = Espresso()
        beverage = MilkDecorator(beverage)
        assert beverage.get_cost() == 30000  # 25000 + 5000
        assert "Sữa tươi" in beverage.get_description()

    def test_multi_decorator_chain(self) -> None:
        beverage: Beverage = Espresso()
        beverage = MilkDecorator(beverage)
        beverage = CaramelDecorator(beverage, extra_shot=True)
        beverage = WhippedCreamDecorator(beverage)
        assert beverage.get_cost() == 25000 + 5000 + 11000 + 10000  # 51000

    def test_vanilla_pumps_affect_cost(self) -> None:
        beverage: Beverage = Espresso()
        single = VanillaSyrupDecorator(beverage, pumps=1)
        double = VanillaSyrupDecorator(beverage, pumps=2)
        assert double.get_cost() - single.get_cost() == 6000

    def test_decorator_chain_description(self) -> None:
        beverage: Beverage = Latte()
        beverage = MilkDecorator(beverage)
        beverage = CinnamonDecorator(beverage)
        parts = beverage.get_description().split(" + ")
        assert parts[0] == "Latte (350ml)"
        assert "Sữa tươi" in parts[1]
        assert "Bột quế" in parts[2]


class TestBarista:
    def test_build_from_config(self) -> None:
        beverage = Barista.build_beverage("latte", [
            {"name": "vanilla", "pumps": 2},
            {"name": "cinnamon"},
        ])
        assert isinstance(beverage, Beverage)
        assert beverage.get_cost() > 35000
        assert "Vanilla" in beverage.get_description()

    def test_unknown_base_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown base"):
            Barista.build_beverage("invalid", [])

    def test_unknown_topping_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown topping"):
            Barista.build_beverage("espresso", [{"name": "alien_topping"}])


class TestLiskovSubstitution:
    def test_decorator_is_substitutable(self) -> None:
        """Decorator giữ nguyên interface — LSP."""
        beverage: Beverage = Espresso()
        beverage = MilkDecorator(CaramelDecorator(beverage))
        # Có thể dùng ở mọi nơi Beverage được chấp nhận
        assert isinstance(beverage, Beverage)
        assert beverage.get_cost() > 0
        assert beverage.get_calories() > 0
        assert beverage.get_size_ml() > 0
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Linh hoạt vô hạn — tổ hợp hành vi không giới hạn ở runtime | Tạo nhiều object nhỏ — memory overhead |
| Tuân thủ SRP và OCP — mỗi decorator chỉ thêm một hành vi | Debugging khó — stack trace phức tạp |
| Tránh class explosion — không cần subclass cho mọi tổ hợp | Thứ tự decorator quan trọng — có thể gây lỗi logic |
| Kết hợp nhiều decorator dễ dàng — tạo chuỗi phức tạp | Khởi tạo đối tượng dài dòng (verbose instantiation) |
| Có thể thêm/bớt decorator ở runtime | Decorator phụ thuộc vào interface của component |
| Giữ nguyên interface — client không bị ảnh hưởng | Không phù hợp khi cần thêm method mới |

## Kết luận

Decorator Pattern là giải pháp tối ưu cho bài toán thêm hành vi động vào đối tượng mà không làm phình to class hierarchy. Nó đặc biệt hữu ích trong các hệ thống mà tổ hợp hành vi là không xác định trước và cần thay đổi linh hoạt dựa trên input người dùng — như đồ uống với topping, middleware stack, I/O stream, hay tính năng sản phẩm.

**Nguyên tắc vàng**: Hãy dùng Decorator khi bạn cần thêm các "lớp" hành vi chồng lên nhau và mỗi lớp độc lập với nhau. Nếu bạn thấy mình đang viết class `BaseWithFeatureAAndFeatureBAndFeatureC`, đó là lúc cần Decorator. Hãy nhớ: mỗi decorator chỉ nên làm đúng một việc, và thứ tự các decorator quan trọng — hãy tài liệu hóa điều này.
