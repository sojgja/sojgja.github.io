---
id: command
title: Command
sidebar_label: 🎮 Command
sidebar_position: 15
---

# Command

**Command** đóng gói một request thành một object độc lập, cho phép tham số hóa client với các request khác nhau, hỗ trợ queue, undo/redo.

## Bài toán

Remote điều khiển TV có 10 nút bấm. Mỗi nút gọi một method khác nhau: `turn_on()`, `turn_off()`, `volume_up()`, ... Nếu code cứng, mỗi lần thêm nút mới phải sửa class Remote.

## Giải pháp

Command đóng gói mỗi hành động thành một object. Remote không biết hành động cụ thể — nó chỉ gọi `execute()`.

```python
from abc import ABC, abstractmethod

class TV:
    def turn_on(self):
        print('📺 TV bật')

    def turn_off(self):
        print('📺 TV tắt')

    def volume_up(self):
        print('🔊 Tăng âm lượng')

class Command(ABC):
    @abstractmethod
    def execute(self): pass

class TurnOnCommand(Command):
    def __init__(self, tv: TV):
        self.tv = tv

    def execute(self):
        self.tv.turn_on()

class TurnOffCommand(Command):
    def __init__(self, tv: TV):
        self.tv = tv

    def execute(self):
        self.tv.turn_off()

class Remote:
    def __init__(self):
        self.buttons = {}

    def set_command(self, button, command):
        self.buttons[button] = command

    def press(self, button):
        self.buttons[button].execute()

# Sử dụng
tv = TV()
remote = Remote()
remote.set_command('on', TurnOnCommand(tv))
remote.set_command('off', TurnOffCommand(tv))

remote.press('on')   # 📺 TV bật
remote.press('off')  # 📺 TV tắt
```

## Mở rộng: Undo

```python
class VolumeUpCommand(Command):
    def __init__(self, tv: TV):
        self.tv = tv
        self.prev_volume = tv.volume if hasattr(tv, 'volume') else 0

    def execute(self):
        self.tv.volume_up()

    def undo(self):
        self.tv.volume = self.prev_volume
```

## Khi nào dùng

- Cần tham số hóa object với hành động
- Cần queue, log, undo/redo
- Cần tách người gọi và người thực hiện

## Thực tế

- Django signals (các action được đóng gói)
- Celery task queue
- Git (commit, checkout, revert là command)
