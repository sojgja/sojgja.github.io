---
id: facade
title: Facade
sidebar_label: 🏢 Facade
sidebar_position: 11
---

# Facade

**Facade** cung cấp một interface đơn giản hóa cho một subsystem phức tạp.

## Bài toán

Bạn xây dựng hệ thống **smart home** với nhiều subsystem: `LightSystem`, `ACSystem`, `SecuritySystem`, `MusicSystem`. Để "rời nhà", người dùng phải:
1. Tắt hết đèn (`light.turn_off_all()`)
2. Tắt điều hòa (`ac.turn_off()`)
3. Bật báo động (`security.arm()`)
4. Tắt nhạc (`music.stop()`)

Client phải biết tất cả subsystem và thứ tự gọi — phức tạp và dễ thiếu bước.

## Giải pháp

Facade cung cấp method `leave_home()` che giấu toàn bộ logic bên trong.

```python
class LightSystem:
    def turn_off_all(self):
        print('💡 Tắt hết đèn')

class ACSystem:
    def turn_off(self):
        print('❄️ Tắt điều hòa')

class SecuritySystem:
    def arm(self):
        print('🔒 Bật báo động')

class SmartHomeFacade:
    def __init__(self):
        self.lights = LightSystem()
        self.ac = ACSystem()
        self.security = SecuritySystem()

    def leave_home(self):
        self.lights.turn_off_all()
        self.ac.turn_off()
        self.security.arm()
        print('🏠 Nhà đã an toàn. Tạm biệt!')

# Client
home = SmartHomeFacade()
home.leave_home()
# 💡 Tắt hết đèn
# ❄️ Tắt điều hòa
# 🔒 Bật báo động
# 🏠 Nhà đã an toàn. Tạm biệt!
```

## Khi nào dùng

- Hệ thống có nhiều subsystem phức tạp
- Muốn giảm coupling giữa client và subsystem
- Cần một entry point đơn giản cho tính năng phổ biến

## Thực tế

- Django REST Framework: `ViewSet` là facade cho serializer, authentication, permissions
- `requests.get()` — facade cho urllib, connection pool
- Docker Compose — facade cho Docker CLI
