---
id: mediator
title: Mediator
sidebar_label: 🤝 Mediator
sidebar_position: 18
---

# Mediator

**Mediator** giảm sự phụ thuộc lộn xộn giữa các object bằng cách đặt logic giao tiếp vào một object trung gian (mediator).

## Bài toán

Ứng dụng **chat room**: User A gửi tin nhắn → User B, C, D nhận được. Nếu không có Mediator, mỗi User phải biết tất cả User khác và gửi trực tiếp — tạo ra kết nối N×N.

## Giải pháp

Mediator (ChatRoom) làm trung gian. User chỉ giao tiếp với Mediator.

```python
class ChatRoom:
    def __init__(self):
        self.users = {}

    def register(self, user):
        self.users[user.name] = user
        user.chat_room = self

    def send(self, message, from_name, to_name=None):
        if to_name:
            # Private message
            user = self.users.get(to_name)
            if user:
                user.receive(message, from_name)
        else:
            # Broadcast
            for name, user in self.users.items():
                if name != from_name:
                    user.receive(message, from_name)

class User:
    def __init__(self, name):
        self.name = name
        self.chat_room = None

    def send(self, message, to=None):
        self.chat_room.send(message, self.name, to)

    def receive(self, message, from_name):
        print(f'📩 [{self.name}] nhận từ {from_name}: {message}')

# Sử dụng
room = ChatRoom()
alice = User('Alice')
bob = User('Bob')
charlie = User('Charlie')

room.register(alice)
room.register(bob)
room.register(charlie)

alice.send('Chào mọi người!')
bob.send('Chào Alice!', 'Alice')
# 📩 [Bob] nhận từ Alice: Chào mọi người!
# 📩 [Charlie] nhận từ Alice: Chào mọi người!
# 📩 [Alice] nhận từ Bob: Chào Alice!
```

## Khi nào dùng

- Nhiều object giao tiếp phức tạp, khó bảo trì
- Muốn tái sử dụng logic giao tiếp
- Cần centralized control cho giao tiếp

## Thực tế

- Django signal dispatcher
- Flask `current_app` — mediator cho các service
- Air traffic control (phi công ↔ controller ↔ phi công)
