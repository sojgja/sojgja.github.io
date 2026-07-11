---
id: visitor
title: Visitor
sidebar_label: 🚶 Visitor
sidebar_position: 24
---

# Visitor

**Visitor** cho phép thêm các thao tác mới vào một nhóm object mà không sửa class của chúng.

## Bài toán

Ứng dụng **xuất hóa đơn** với nhiều loại sản phẩm: Book, Electronics, Clothing. Mỗi loại có cách tính thuế khác nhau (sách 5%, điện tử 10%, quần áo 8%). Bạn cũng cần xuất JSON và XML. Nếu thêm method vào mỗi class (`calculate_tax()`, `to_json()`, `to_xml()`), class sẽ phình to và vi phạm Single Responsibility.

## Giải pháp

Visitor tách thao tác khỏi object. Mỗi thao tác là một Visitor riêng.

```python
from abc import ABC, abstractmethod

class Product(ABC):
    @abstractmethod
    def accept(self, visitor): pass

class Book(Product):
    def __init__(self, price):
        self.price = price

    def accept(self, visitor):
        return visitor.visit_book(self)

class Electronics(Product):
    def __init__(self, price):
        self.price = price

    def accept(self, visitor):
        return visitor.visit_electronics(self)

class Clothing(Product):
    def __init__(self, price):
        self.price = price

    def accept(self, visitor):
        return visitor.visit_clothing(self)

class Visitor(ABC):
    @abstractmethod
    def visit_book(self, product: Book): pass

    @abstractmethod
    def visit_electronics(self, product: Electronics): pass

    @abstractmethod
    def visit_clothing(self, product: Clothing): pass

class TaxVisitor(Visitor):
    def visit_book(self, product: Book):
        return product.price * 0.05  # 5%

    def visit_electronics(self, product: Electronics):
        return product.price * 0.10  # 10%

    def visit_clothing(self, product: Clothing):
        return product.price * 0.08  # 8%

class InvoiceVisitor(Visitor):
    def visit_book(self, product: Book):
        return f'📚 Sách: {product.price:,} VND (Thuế: {product.price * 0.05:,.0f})'

    def visit_electronics(self, product: Electronics):
        return f'💻 Điện tử: {product.price:,} VND (Thuế: {product.price * 0.10:,.0f})'

    def visit_clothing(self, product: Clothing):
        return f'👕 Quần áo: {product.price:,} VND (Thuế: {product.price * 0.08:,.0f})'

# Sử dụng
products = [
    Book(100000),
    Electronics(5000000),
    Clothing(300000),
]

tax_calc = TaxVisitor()
invoice = InvoiceVisitor()

total_tax = 0
for product in products:
    total_tax += product.accept(tax_calc)
    print(product.accept(invoice))

print(f'\n💰 Tổng thuế: {total_tax:,.0f} VND')
# 📚 Sách: 100,000 VND (Thuế: 5,000)
# 💻 Điện tử: 5,000,000 VND (Thuế: 500,000)
# 👕 Quần áo: 300,000 VND (Thuế: 24,000)
# 
# 💰 Tổng thuế: 529,000 VND
```

## Khi nào dùng

- Muốn thêm thao tác vào object hierarchy ổn định
- Các thao tác không liên quan đến nhau
- Không muốn sửa class gốc

## Thực tế

- Python `ast.NodeVisitor` (walk AST tree)
- Django template engine: render visitor
- Compiler: semantic analysis, code generation đều là visitor
