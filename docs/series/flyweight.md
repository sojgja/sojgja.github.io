---
id: flyweight
title: Flyweight
sidebar_label: 🪶 Flyweight
sidebar_position: 12
---

# Flyweight

**Flyweight** giảm bộ nhớ bằng cách chia sẻ trạng thái chung giữa nhiều object, thay vì mỗi object giữ riêng.

## Bài toán

Game bắn gà có 10.000 viên đạn trên màn hình. Mỗi viên đạn lưu: `x`, `y`, `speed`, `color`, `texture`. `color` và `texture` giống nhau cho cùng loại đạn, nhưng mỗi viên đều lưu riêng → tốn bộ nhớ.

## Giải pháp

Flyweight tách trạng thái thành 2 loại:
- **Intrinsic** (dùng chung): `color`, `texture` — đặt trong BulletType
- **Extrinsic** (riêng): `x`, `y`, `speed` — đặt trong Bullet

```python
class BulletType:
    def __init__(self, color, texture):
        self.color = color
        self.texture = texture

    def render(self, x, y, speed):
        print(f'🎯 Bắn đạn {self.color} tại ({x},{y}) tốc độ {speed}')

class BulletTypeFactory:
    _types = {}

    @classmethod
    def get_type(cls, color, texture):
        key = f'{color}_{texture}'
        if key not in cls._types:
            cls._types[key] = BulletType(color, texture)
        return cls._types[key]

class Bullet:
    def __init__(self, x, y, speed, bullet_type: BulletType):
        self.x = x
        self.y = y
        self.speed = speed
        self.type = bullet_type  # Flyweight — dùng chung

    def render(self):
        self.type.render(self.x, self.y, self.speed)

# Sử dụng
red_type = BulletTypeFactory.get_type('red', 'fire.png')
blue_type = BulletTypeFactory.get_type('blue', 'ice.png')

bullets = [
    Bullet(10, 20, 5, red_type),
    Bullet(30, 40, 7, red_type),
    Bullet(50, 60, 6, blue_type),
    # 10.000 viên...
]

print(f'Kiểu đạn đã tạo: {len(BulletTypeFactory._types)}')  # 2
```

## Khi nào dùng

- Ứng dụng tạo rất nhiều object tương tự
- Bộ nhớ là vấn đề
- Object có thể tách intrinsic và extrinsic state

## Thực tế

- String interning (Python tự động intern chuỗi nhỏ)
- Cache font/glyph trong text editor
- Django model `__slots__` optimization
