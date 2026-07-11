---
id: adapter
title: Adapter
sidebar_label: 🔌 Adapter
sidebar_position: 7
---

# Adapter

**Adapter** chuyển đổi interface của một class thành interface khác mà client mong đợi, giúp các class không tương thích có thể làm việc cùng nhau.

## Bài toán

Hệ thống thanh toán cũ dùng class `LegacyPayment` với method `make_payment(amount)`. Bạn mua một thư viện thanh toán mới `StripePayment` với method `charge(amount, currency)`. Hàng trăm chỗ trong code đang gọi `make_payment`. Bạn không thể sửa thư viện, cũng không muốn sửa hàng trăm chỗ.

## Giải pháp

Adapter "bọc" thư viện mới bằng interface cũ.

```python
class LegacyPayment:
    def make_payment(self, amount):
        return f'✅ Thanh toán {amount} VND'

class StripePayment:
    def charge(self, amount, currency):
        return f'✅ Stripe: {amount} {currency}'

class StripeAdapter(LegacyPayment):
    def __init__(self, stripe):
        self.stripe = stripe

    def make_payment(self, amount):
        return self.stripe.charge(amount, 'VND')

# Sử dụng — client không thay đổi
def process_order(payment, amount):
    print(payment.make_payment(amount))

process_order(LegacyPayment(), 100000)
process_order(StripeAdapter(StripePayment()), 200000)
```

## Khi nào dùng

- Cần tích hợp thư viện/API cũ với code mới
- Interface không tương thích nhưng không thể sửa
- Muốn tái sử dụng class hiện có với interface khác

## Thực tế

- Tích hợp nhiều payment gateway (Stripe, PayPal, VNPay) chung interface
- Database driver adapter (SQLite, PostgreSQL, MySQL)
- API versioning adapter
