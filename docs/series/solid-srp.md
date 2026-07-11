---
id: solid-srp
title: S — Single Responsibility Principle
sidebar_label: S — Single Responsibility
sidebar_position: 26
---

# S — Single Responsibility Principle

> **"A class should have only one reason to change."** — Robert C. Martin

Một class chỉ nên có **một trách nhiệm duy nhất**. Nếu có nhiều hơn, class sẽ có nhiều lý do để thay đổi, và thay đổi một trách nhiệm có thể ảnh hưởng đến trách nhiệm khác.

## Bài toán: Class "thiên tài toàn năng"

```python
class Order:
    def __init__(self, items):
        self.items = items

    def calculate_total(self):
        return sum(item['price'] * item['qty'] for item in self.items)

    def print_invoice(self):
        total = self.calculate_total()
        print(f'{"="*30}')
        print(f'HÓA ĐƠN')
        print(f'{"="*30}')
        for item in self.items:
            print(f'{item["name"]:20} x{item["qty"]} {item["price"]:>8,} VND')
        print(f'{"="*30}')
        print(f'Tổng: {total:>26,} VND')

    def save_to_db(self):
        print(f'💾 Lưu hóa đơn {id(self)} vào database...')

    def send_email(self, email):
        print(f'📧 Gửi hóa đơn đến {email}...')
```

**Vấn đề:** Class `Order` làm quá nhiều việc: tính toán, in ấn, lưu DB, gửi email. Mỗi khi thay đổi format in, hoặc đổi database, hoặc thay email service — đều phải sửa class này.

## Giải pháp: Tách thành nhiều class, mỗi class một trách nhiệm

```python
class Order:
    def __init__(self, items):
        self.items = items

    def calculate_total(self):
        return sum(item['price'] * item['qty'] for item in self.items)


class InvoicePrinter:
    def print(self, order: Order):
        total = order.calculate_total()
        print(f'{"="*30}')
        print('HÓA ĐƠN')
        print(f'{"="*30}')
        for item in order.items:
            print(f'{item["name"]:20} x{item["qty"]} {item["price"]:>8,} VND')
        print(f'{"="*30}')
        print(f'Tổng: {total:>26,} VND')


class OrderRepository:
    def save(self, order: Order):
        print(f'💾 Lưu hóa đơn {id(order)} vào database...')


class EmailService:
    def send_invoice(self, order: Order, email: str):
        print(f'📧 Gửi hóa đơn đến {email}...')
```

## Lợi ích

| Trước (vi phạm SRP) | Sau (đúng SRP) |
|---------------------|-----------------|
| 1 class, 4 lý do thay đổi | 4 class, mỗi class 1 lý do |
| Sửa format in → chạm vào DB | Sửa `InvoicePrinter` — không ảnh hưởng gì khác |
| Khó test (phải mock DB, email) | Test từng class độc lập, dễ dàng |
| Thêm tính năng (SMS, PDF) phải sửa Order | Thêm class mới — không sửa code cũ |

## Dấu hiệu nhận biết vi phạm SRP

- Class có nhiều method không liên quan đến nhau
- Một method có nhiều hơn 1 cấp độ abstraction
- Class có quá nhiều dependencies (tham số constructor nhiều)
- Khó đặt tên cho class (tên chung chung như `OrderManager`, `Util`, `Helper`)

## Kết luận

SRP là nguyên lý **đơn giản nhất nhưng khó áp dụng nhất**. Bí quyết: nếu bạn cảm thấy một class "có gì đó sai", hãy thử tách nó ra. Một class tốt thường có tên rõ ràng và method dễ hiểu.
