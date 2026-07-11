---
id: state
title: State
sidebar_label: ⚡ State
sidebar_position: 21
---

# State

**State** cho phép object thay đổi hành vi khi trạng thái bên trong thay đổi. Object sẽ "đổi class" khi đổi state.

## Bài toán

Ứng dụng **đặt đồ ăn** có các trạng thái: `Pending` (chờ xác nhận), `Confirmed` (đã xác nhận), `Preparing` (đang nấu), `Delivering` (đang giao), `Delivered` (đã giao), `Cancelled` (đã hủy). Mỗi trạng thái có hành vi khác nhau khi gọi `next()` hoặc `cancel()`. Nếu dùng if-else với enum, code dễ sai và khó mở rộng.

## Giải pháp

Mỗi trạng thái là một class riêng. Order chuyển trạng thái bằng cách thay đổi current state.

```python
from abc import ABC, abstractmethod

class OrderState(ABC):
    @abstractmethod
    def next(self, order): pass

    @abstractmethod
    def cancel(self, order): pass

class PendingState(OrderState):
    def next(self, order):
        print('✅ Đơn hàng đã được xác nhận')
        order.state = ConfirmedState()

    def cancel(self, order):
        print('❌ Đơn hàng đã hủy')
        order.state = CancelledState()

class ConfirmedState(OrderState):
    def next(self, order):
        print('👨‍🍳 Đang nấu...')
        order.state = PreparingState()

    def cancel(self, order):
        print('❌ Đơn hàng đã hủy (hoàn tiền)')
        order.state = CancelledState()

class PreparingState(OrderState):
    def next(self, order):
        print('🚚 Đang giao hàng...')
        order.state = DeliveringState()

    def cancel(self, order):
        print('❌ Đã nấu rồi — không thể hủy')

class DeliveringState(OrderState):
    def next(self, order):
        print('✅ Đã giao hàng thành công!')
        order.state = DeliveredState()

    def cancel(self, order):
        print('❌ Đang giao — không thể hủy')

class DeliveredState(OrderState):
    def next(self, order):
        print('📦 Đơn hàng đã giao xong')

    def cancel(self, order):
        print('❌ Đã giao rồi — không thể hủy')

class CancelledState(OrderState):
    def next(self, order):
        print('❌ Đơn hàng đã hủy — không thể tiếp tục')

    def cancel(self, order):
        print('❌ Đã hủy rồi')

class Order:
    def __init__(self):
        self.state = PendingState()

    def next(self):
        self.state.next(self)

    def cancel(self):
        self.state.cancel(self)

# Sử dụng
order = Order()
order.next()    # ✅ Đơn hàng đã được xác nhận
order.next()    # 👨‍🍳 Đang nấu...
order.cancel()  # ❌ Đã nấu rồi — không thể hủy
order.next()    # 🚚 Đang giao hàng...
order.next()    # ✅ Đã giao hàng thành công!
```

## Khi nào dùng

- Object có nhiều trạng thái, mỗi trạng thái có hành vi riêng
- Code có nhiều if/switch kiểm tra trạng thái
- Các trạng thái chuyển đổi phức tạp

## Thực tế

- Django `FSM` (Finite State Machine)
- UI component states: loading → ready → error
- Game character states: idle → running → jumping → falling
- HTTP request lifecycle
