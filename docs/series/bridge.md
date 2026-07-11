---
id: bridge
title: Bridge
sidebar_label: 🌉 Bridge
sidebar_position: 8
---

# Bridge

**Bridge** tách abstraction khỏi implementation, cho phép chúng thay đổi độc lập với nhau.

## Bài toán

Bạn xây dựng ứng dụng điều khiển thiết bị từ xa. Có 2 loại remote (Basic, Advanced) và 2 loại thiết bị (TV, Radio). Nếu dùng kế thừa:

```
Remote → BasicRemote → BasicTVRemote, BasicRadioRemote
       → AdvancedRemote → AdvancedTVRemote, AdvancedRadioRemote
```

Mỗi lần thêm remote mới hoặc thiết bị mới, số class tăng theo cấp số nhân. Đây là "class explosion".

## Giải pháp

Bridge tách thành 2 nhánh: **Abstraction** (remote) và **Implementation** (device). Mỗi nhánh phát triển độc lập.

```python
from abc import ABC, abstractmethod

# Implementation
class Device(ABC):
    def __init__(self):
        self.volume = 50

    @abstractmethod
    def get_type(self): pass

class TV(Device):
    def get_type(self):
        return '📺 TV'

class Radio(Device):
    def get_type(self):
        return '📻 Radio'

# Abstraction
class Remote:
    def __init__(self, device: Device):
        self.device = device

    def volume_up(self):
        self.device.volume += 10
        print(f'{self.device.get_type()} volume: {self.device.volume}')

class AdvancedRemote(Remote):
    def mute(self):
        self.device.volume = 0
        print(f'{self.device.get_type()} đã tắt tiếng')

# Sử dụng — kết hợp bất kỳ
tv = TV()
remote = AdvancedRemote(tv)
remote.volume_up()  # 📺 TV volume: 60
remote.mute()       # 📺 TV đã tắt tiếng

radio = Radio()
basic = Remote(radio)
basic.volume_up()   # 📻 Radio volume: 60
```

## Khi nào dùng

- Muốn tránh class explosion với kế thừa đa chiều
- Cần thay đổi abstraction và implementation độc lập
- Implementation được chọn tại runtime

## Thực tế

- Logger: format (JSON, text) × output (file, console, cloud)
- GUI: Window type × Platform renderer
- Database ORM: Model × Engine backend
