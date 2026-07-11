---
id: builder
title: Builder
sidebar_label: 🔨 Builder
sidebar_position: 5
---

# Builder

**Builder** tách việc xây dựng một object phức tạp khỏi biểu diễn của nó, cho phép cùng một quá trình tạo ra các biểu diễn khác nhau.

## Bài toán

Bạn viết constructor cho class `Pizza` với 15 tham số: `size`, `crust`, `cheese`, `sauce`, `toppings`, ... Code gọi trông như thế này:

```python
pizza = Pizza('large', 'thin', True, False, ['pepperoni'], True, False, 'bbq', ...)
```

Thứ tự tham số dễ nhầm, khó đọc, khó thêm option mới.

## Giải pháp

Builder cho phép xây dựng object từng bước, mỗi bước là một method rõ ràng.

```python
class Pizza:
    def __init__(self):
        self.size = 'medium'
        self.cheese = False
        self.toppings = []

class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()

    def set_size(self, size):
        self.pizza.size = size
        return self

    def add_cheese(self):
        self.pizza.cheese = True
        return self

    def add_topping(self, topping):
        self.pizza.toppings.append(topping)
        return self

    def build(self):
        return self.pizza

my_pizza = (PizzaBuilder()
    .set_size('large')
    .add_cheese()
    .add_topping('pepperoni')
    .add_topping('mushroom')
    .build())

print(my_pizza.size)      # large
print(my_pizza.toppings)  # ['pepperoni', 'mushroom']
```

## Khi nào dùng

- Object có nhiều tham số tùy chọn
- Quá trình khởi tạo gồm nhiều bước
- Muốn tạo nhiều biến thể khác nhau từ cùng một process

## Thực tế

- Django `QuerySet` builder: `User.objects.filter(...).order_by(...).select_related(...)`
- `requests` API: `session.prepare_request(...)`
- HTML form builder trong Flask/WTForms
