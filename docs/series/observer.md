---
id: observer
title: Observer
sidebar_label: 👀 Observer
sidebar_position: 20
---

# Observer

**Observer** định nghĩa cơ chế subscribe (đăng ký) để nhiều object có thể theo dõi và phản ứng khi một object khác thay đổi trạng thái.

## Bài toán

Kênh YouTube có 1 triệu subscriber. Khi có video mới, kênh phải thông báo cho tất cả subscriber. Cách tồi: kênh gọi API từng subscriber — chậm, dễ lỗi, khó mở rộng. Cách tốt: subscriber tự đăng ký, kênh chỉ cần notify.

## Giải pháp

Observer tách kênh (subject) khỏi subscriber (observer). Subject quản lý danh sách observer và thông báo khi có thay đổi.

```python
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, video_title: str): pass

class Subject(ABC):
    @abstractmethod
    def attach(self, observer: Observer): pass

    @abstractmethod
    def notify(self, video_title: str): pass

class YouTubeChannel(Subject):
    def __init__(self, name: str):
        self.name = name
        self._subscribers = []

    def attach(self, observer: Observer):
        self._subscribers.append(observer)

    def upload(self, title: str):
        print(f'📹 {self.name} đã đăng: {title}')
        self.notify(title)

    def notify(self, video_title: str):
        for sub in self._subscribers:
            sub.update(video_title)

class Subscriber(Observer):
    def __init__(self, name: str):
        self.name = name

    def update(self, video_title: str):
        print(f'🔔 {self.name} nhận thông báo: "{video_title}"')

# Sử dụng
channel = YouTubeChannel('Lập Trình Viên')

alice = Subscriber('Alice')
bob = Subscriber('Bob')
charlie = Subscriber('Charlie')

channel.attach(alice)
channel.attach(bob)
channel.attach(charlie)

channel.upload('Observer Pattern trong Python')
# 📹 Lập Trình Viên đã đăng: Observer Pattern trong Python
# 🔔 Alice nhận thông báo: "Observer Pattern trong Python"
# 🔔 Bob nhận thông báo: "Observer Pattern trong Python"
# 🔔 Charlie nhận thông báo: "Observer Pattern trong Python"
```

## Khi nào dùng

- Một object thay đổi cần thông báo cho nhiều object khác
- Không biết trước số lượng object cần thông báo
- Cần loose coupling giữa subject và observer

## Thực tế

- Django signals (`post_save`, `pre_delete`)
- Flask `app.before_request`, `app.after_request`
- `asyncio` event loop
- JavaScript event listeners (`element.addEventListener`)
