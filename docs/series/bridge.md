---
id: bridge
title: Bridge
sidebar_label: 🌉 Bridge
sidebar_position: 8
---

# Bridge

> "Decouple an abstraction from its implementation so that the two can vary independently." — Erich Gamma, *Design Patterns: Elements of Reusable Object-Oriented Software*

Có bao giờ bạn cảm thấy code của mình đang phình to không kiểm soát? Tôi cũng từng ở trong hoàn cảnh đó...

## Bài toán chi tiết

Tôi muốn kể cho bạn một câu chuyện. Một công ty IoT đang xây dựng hệ thống điều khiển thiết bị thông minh cho tòa nhà văn phòng. Họ có hai loại thiết bị chính: đèn LED (`SmartLight`) và điều hòa nhiệt độ (`SmartAC`). Ban đầu, mỗi thiết bị chỉ có một remote điều khiển đơn giản (`BasicRemote`) với các chức năng bật/tắt và tăng/giảm mức độ.

Các kỹ sư — những người rất thông minh — quyết định dùng kế thừa. Họ tạo ra `BasicLightRemote` và `BasicACRemote`, mỗi class kế thừa từ `BasicRemote`. Mọi thứ hoạt động tốt... cho đến khi khách hàng yêu cầu thêm remote cao cấp (`AdvancedRemote`) với chức năng hẹn giờ, cảm biến, và kịch bản tự động. Lúc này, team phải tạo thêm `AdvancedLightRemote` và `AdvancedACRemote`.

Bạn biết chuyện gì xảy ra tiếp theo không? Khi hệ thống mở rộng lên 10 loại thiết bị và 5 loại remote, số class cần tạo là 10 × 5 = **50 class**. Đây chính là "class explosion" — một vấn đề kinh điển của kế thừa cứng nhắc khi có hai chiều biến thiên độc lập. Mỗi khi thêm một thiết bị mới, team phải tạo thêm 5 class remote mới. Mỗi khi thêm một loại remote mới, họ phải implement lại cho 10 thiết bị.

Hậu quả? Codebase phình to không kiểm soát. Thay đổi logic chung yêu cầu sửa hàng chục class. Nguy cơ lỗi tăng theo cấp số nhân. Các class bị trùng lặp code nghiêm trọng vì cùng một chức năng "hẹn giờ" phải implement riêng rẽ trong từng class remote-thiết bị. Nghe quen không? Tôi cá là bạn đã từng gặp tình huống này rồi.

## Giải pháp với Pattern

Bridge Pattern giải quyết vấn đề này bằng cách tách abstraction (remote) khỏi implementation (thiết bị) thành hai hệ thống phân cấp riêng biệt, kết nối với nhau qua composition. Thay vì N × M class, giờ đây chỉ cần **N + M class**. Abstraction chứa một tham chiếu đến implementation interface, và mọi lời gọi từ abstraction đều được ủy quyền cho implementation hiện tại.

Cụ thể, Bridge gồm bốn thành phần:
- **Abstraction**: Interface cấp cao định nghĩa các method điều khiển. Trong ví dụ, đây là `RemoteControl`.
- **RefinedAbstraction**: Mở rộng abstraction cơ bản — `AdvancedRemoteControl` thêm các method như `schedule()`, `mute()`, `boost()`.
- **Implementor**: Interface cho các thiết bị — `Device` với các method `power_on()`, `power_off()`, `set_level()`, `get_level()`.
- **ConcreteImplementor**: Implement cụ thể cho từng thiết bị — `SmartLight`, `SmartAC`, `SmartBlind`.

Mỗi remote (abstraction) có thể kết hợp với bất kỳ thiết bị (implementation) nào thông qua composition. Khi cần thêm thiết bị mới, chỉ cần tạo một `ConcreteImplementor` mới. Khi cần remote mới, chỉ cần tạo một `RefinedAbstraction` mới. **Hai chiều phát triển hoàn toàn độc lập.**

## Phân tích thiết kế

Bridge Pattern là một ví dụ điển hình của nguyên tắc **Favor composition over inheritance**. Thay vì kế thừa để có được hành vi (dẫn đến class explosion), composition cho phép kết hợp linh hoạt tại runtime. Pattern này cũng tuân thủ **Single Responsibility Principle**: abstraction tập trung vào logic điều khiển cấp cao, implementation tập trung vào chi tiết tương tác với phần cứng.

Một điểm quan trọng mà tôi muốn bạn nhớ: Bridge thường bị nhầm với Adapter. Rất nhiều người mắc sai lầm này. Điểm khác biệt cốt lõi: Bridge được thiết kế từ đầu (*design time*) để cho phép hai thành phần biến thiên độc lập, trong khi Adapter được thêm vào sau (*integration time*) để làm cho hai hệ thống có sẵn tương thích với nhau.

**Khi KHÔNG nên dùng Bridge:**
- Khi chỉ có một implementation duy nhất và không có kế hoạch mở rộng — Bridge chỉ thêm độ phức tạp vô ích.
- Khi abstraction và implementation có mối quan hệ cố định, một-một — lúc đó kế thừa đơn giản hơn nhiều.
- Khi client cần truy cập trực tiếp vào implementation (ví dụ: gọi method đặc thù của thiết bị).

**Trade-offs:**
- Tăng độ phức tạp ban đầu của thiết kế do phải xác định và tách biệt hai chiều biến thiên.
- Hiệu năng giảm nhẹ do có thêm một lớp gián tiếp (delegation từ abstraction sang implementation).
- Khó áp dụng retrofit — Bridge cần được thiết kế từ đầu, khó thêm vào hệ thống hiện có.

## Ví dụ code hoàn chỉnh

### Cách làm sai: Class explosion với kế thừa

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import time


class DeviceType(Enum):
    LIGHT = auto()
    AC = auto()
    BLIND = auto()
    SPEAKER = auto()


# --- WRONG: Mỗi tổ hợp remote × device là một class ---
class BasicLightRemote:
    def __init__(self) -> None:
        self._power: bool = False
        self._brightness: int = 50

    def toggle_power(self) -> str:
        self._power = not self._power
        return f"Light {'ON' if self._power else 'OFF'}"
    
    def brightness_up(self) -> str:
        self._brightness = min(100, self._brightness + 10)
        return f"Brightness: {self._brightness}"

    def brightness_down(self) -> str:
        self._brightness = max(0, self._brightness - 10)
        return f"Brightness: {self._brightness}"


class BasicACRemote:
    def __init__(self) -> None:
        self._power: bool = False
        self._temperature: int = 26

    def toggle_power(self) -> str:
        self._power = not self._power
        return f"AC {'ON' if self._power else 'OFF'}"

    def temperature_up(self) -> str:
        self._temperature = min(30, self._temperature + 1)
        return f"Temperature: {self._temperature}°C"

    def temperature_down(self) -> str:
        self._temperature = max(16, self._temperature - 1)
        return f"Temperature: {self._temperature}°C"


# Hãy tưởng tượng: AdvancedLightRemote, AdvancedACRemote,
# BasicSpeakerRemote, AdvancedSpeakerRemote, v.v. — vô hạn!
```

### Cách đúng: Bridge Pattern

```python
# --- Implementation Interface (Device) ---
class Device(ABC):
    """Implementor — interface chung cho tất cả thiết bị."""

    def __init__(self) -> None:
        self._power: bool = False
        self._level: int = 50

    @abstractmethod
    def device_type(self) -> str:
        ...

    def power_on(self) -> None:
        self._power = True
        print(f"{self.device_type()} turned ON")

    def power_off(self) -> None:
        self._power = False
        print(f"{self.device_type()} turned OFF")

    def is_powered(self) -> bool:
        return self._power

    @abstractmethod
    def set_level(self, level: int) -> None:
        ...

    @abstractmethod
    def get_level(self) -> int:
        ...

    @abstractmethod
    def get_status(self) -> dict:
        ...


# --- Concrete Implementors ---
class SmartLight(Device):
    def __init__(self, light_id: str, location: str) -> None:
        super().__init__()
        self._light_id = light_id
        self._location = location
        self._brightness: int = 50
        self._color_temp: int = 4000  # Kelvin

    def device_type(self) -> str:
        return f"SmartLight [{self._location}]"

    def set_level(self, level: int) -> None:
        self._brightness = max(0, min(100, level))
        print(f"{self.device_type()} brightness set to {self._brightness}%")

    def get_level(self) -> int:
        return self._brightness

    def set_color_temperature(self, kelvin: int) -> None:
        self._color_temp = max(2700, min(6500, kelvin))
        print(f"{self.device_type()} color temp: {self._color_temp}K")

    def get_status(self) -> dict:
        return {
            "id": self._light_id,
            "type": "light",
            "power": self._power,
            "brightness": self._brightness,
            "color_temp": self._color_temp,
        }


class SmartAC(Device):
    def __init__(self, ac_id: str, room: str) -> None:
        super().__init__()
        self._ac_id = ac_id
        self._room = room
        self._temperature: int = 26
        self._mode: str = "cool"

    def device_type(self) -> str:
        return f"SmartAC [{self._room}]"

    def set_level(self, level: int) -> None:
        self._temperature = max(16, min(30, level))
        print(f"{self.device_type()} set to {self._temperature}°C")

    def get_level(self) -> int:
        return self._temperature

    def set_mode(self, mode: str) -> None:
        if mode in ("cool", "heat", "fan", "dry"):
            self._mode = mode
            print(f"{self.device_type()} mode: {mode}")

    def get_status(self) -> dict:
        return {
            "id": self._ac_id,
            "type": "ac",
            "power": self._power,
            "temperature": self._temperature,
            "mode": self._mode,
        }


# --- Abstraction ---
class RemoteControl:
    """Abstraction cơ bản — điều khiển thiết bị ở mức đơn giản."""

    def __init__(self, device: Device) -> None:
        self._device = device

    def toggle_power(self) -> None:
        if self._device.is_powered():
            self._device.power_off()
        else:
            self._device.power_on()

    def level_up(self) -> None:
        current = self._device.get_level()
        self._device.set_level(current + 10)

    def level_down(self) -> None:
        current = self._device.get_level()
        self._device.set_level(current - 10)

    def show_status(self) -> dict:
        return self._device.get_status()


# --- Refined Abstraction ---
class AdvancedRemoteControl(RemoteControl):
    """Mở rộng abstraction với nhiều tính năng hơn."""

    def mute(self) -> None:
        """Tắt thiết bị (tương đương power off)."""
        if self._device.is_powered():
            self._device.power_off()
            print("Device muted (powered off)")

    def schedule(self, hour: int, minute: int, action: str) -> None:
        """Hẹn giờ cho thiết bị."""
        time_str = f"{hour:02d}:{minute:02d}"
        print(f"Scheduled '{action}' at {time_str} for {self._device.device_type()}")

    def boost(self) -> None:
        """Đẩy thiết bị lên mức tối đa."""
        self._device.set_level(100)
        print(f"{self._device.device_type()} boosted to maximum!")

    def scene(self, scene_name: str) -> None:
        """Kích hoạt kịch bản — áp dụng nhiều thiết lập cùng lúc."""
        scenes = {
            "movie": {"device": self._device, "level": 20},
            "party": {"device": self._device, "level": 80},
            "sleep": {"device": self._device, "level": 5},
        }
        if scene_name in scenes:
            config = scenes[scene_name]
            config["device"].set_level(config["level"])
            print(f"Scene '{scene_name}' activated for {self._device.device_type()}")
        else:
            print(f"Unknown scene: {scene_name}")


# --- Usage ---
if __name__ == "__main__":
    # Thiết bị cụ thể
    living_room_light = SmartLight("L-001", "Living Room")
    bedroom_ac = SmartAC("AC-001", "Bedroom")

    # Kết hợp remote với thiết bị qua Bridge
    basic_remote = RemoteControl(living_room_light)
    advanced_remote = AdvancedRemoteControl(bedroom_ac)

    # Điều khiển
    basic_remote.toggle_power()
    basic_remote.level_up()
    basic_remote.level_up()
    print(basic_remote.show_status())

    advanced_remote.toggle_power()
    advanced_remote.boost()
    advanced_remote.schedule(22, 30, "power_off")
    advanced_remote.scene("movie")
    print(advanced_remote.show_status())

    # Tính linh hoạt: gán remote khác cho cùng thiết bị
    print("\n--- Reusing remote with different device ---")
    light_advanced = AdvancedRemoteControl(living_room_light)
    light_advanced.scene("movie")
```

## Sơ đồ UML

```
       ┌─────────────────────────┐
       │     RemoteControl       │──────┐
       │  (Abstraction)          │      │
       │─────────────────────────│      │
       │ # _device: Device       │      │
       │─────────────────────────│      │
       │ + toggle_power()        │      │
       │ + level_up()            │      │
       │ + level_down()          │      │
       │ + show_status()         │      │
       └──────────┬──────────────┘      │
                  │                     │
       ┌──────────┴──────────────┐      │
       │AdvancedRemoteControl    │      │
       │ (RefinedAbstraction)    │      │
       │─────────────────────────│      │
       │ + mute()                │      │
       │ + schedule()            │      │
       │ + boost()               │      │
       │ + scene()               │      │
       └─────────────────────────┘      │
                                         │
                ┌────────────────────────┘
                │
       ┌────────┴─────────────────┐
       │   «interface»            │
       │      Device              │
       │  (Implementor)           │
       │──────────────────────────│
       │ + power_on()             │
       │ + power_off()            │
       │ + is_powered()           │
       │ + set_level(int)         │
       │ + get_level() → int      │
       │ + get_status() → dict    │
       └──────────┬───────────────┘
                  │
         ┌────────┼────────┐
         │        │        │
   ┌─────┴──┐ ┌──┴────┐ ┌─┴─────┐
   │Smart   │ │Smart  │ │Smart  │
   │ Light  │ │  AC   │ │ Blinds│
   │(Concrete│ │(Concr.)│ │(Concr)│
   │Implem.)│ │       │ │       │
   └────────┘ └───────┘ └───────┘
```

## So sánh với Pattern liên quan

**Bridge vs Adapter**: Đây là hai pattern dễ nhầm lẫn nhất. Tôi nhắc lại lần nữa: Bridge được thiết kế chủ động từ đầu (*design time*) để tách abstraction và implementation, cho phép cả hai phát triển độc lập. Adapter được thêm vào một cách thụ động (*integration time*) để làm cho hai class không tương thích có thể làm việc với nhau. Bridge thường phức tạp hơn vì nó định nghĩa hai hệ thống phân cấp hoàn chỉnh, trong khi Adapter chỉ thêm một lớp duy nhất.

**Bridge vs Strategy**: Cả hai đều dùng composition và delegation. Strategy tập trung vào việc thay đổi thuật toán (cách thức thực hiện một hành vi), còn Bridge tập trung vào việc tách abstraction khỏi implementation để chúng biến thiên độc lập. Strategy thường thay đổi hành vi của một context duy nhất, còn Bridge cho phép cả abstraction lẫn implementation đều có thể thay đổi.

**Bridge vs Abstract Factory**: Hai pattern thường đi cùng nhau. Abstract Factory có thể được dùng để tạo ra các bộ (family) đối tượng Bridge cụ thể. Ví dụ: một `RemoteFactory` có thể tạo ra `RemoteControl` kết hợp với `SmartLight`, hoặc `AdvancedRemoteControl` kết hợp với `SmartAC`.

## Ứng dụng thực tế

**1. Django Rest Framework — Renderers and Parsers**: DRF sử dụng Bridge pattern để tách View (abstraction) khỏi serialization format (implementation). Một view có thể kết hợp với bất kỳ renderer nào (JSON, XML, YAML, HTML) và parser nào thông qua composition:

```python
# rest_framework/views.py
class APIView(View):
    renderer_classes = api_settings.DEFAULT_RENDERER_CLASSES
    parser_classes = api_settings.DEFAULT_PARSER_CLASSES

    def dispatch(self, request, *args, **kwargs):
        # Renderer và parser được chọn dựa trên request
        self.renderer = self.get_renderer()
        self.parser = self.get_parser()
```

**2. JDBC (Java Database Connectivity)**: Bridge pattern kinh điển trong Java. `java.sql.DriverManager` và `java.sql.Connection` là abstraction, các driver cụ thể (MySQL Driver, PostgreSQL Driver) là implementation. Người dùng chỉ cần load driver, và JDBC tự động quản lý việc kết nối:

```python
# Tương tự trong Python với DB-API
import psycopg2  # Concrete implementor
import sqlite3   # Concrete implementor

# Abstraction — cùng một interface
conn = psycopg2.connect(database="test")  # hoặc
conn = sqlite3.connect("test.db")
```

**3. GUI Framework (Qt, Tkinter)**: Các framework GUI sử dụng Bridge để tách platform-independent API (abstraction) khỏi platform-specific rendering (implementation). Qt QWindow abstraction làm việc với các QPlatformWindow implementor khác nhau cho Windows, macOS, Linux:

```cpp
// Qt internals — Bridge Pattern
class QWindow {  // Abstraction
    QPlatformWindow *d;  // Implementor
};
// QWindowsWindow, QXcbWindow, QCocoaWindow: Concrete Implementors
```

**4. Logging framework**: SLF4J (Java) / structlog (Python) là Bridge giữa logging API và logging backend:

```python
import structlog

# structlog là abstraction, có thể gắn với nhiều backend:
structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
# hoặc
structlog.configure(logger_factory=structlog.PrintLoggerFactory())
```

## Kiểm thử

```python
import pytest
from unittest.mock import MagicMock, PropertyMock
from bridge import (
    Device, SmartLight, SmartAC,
    RemoteControl, AdvancedRemoteControl,
)


class TestSmartLight:
    def setup_method(self) -> None:
        self.light = SmartLight("L-001", "Kitchen")

    def test_initial_state(self) -> None:
        assert self.light.device_type() == "SmartLight [Kitchen]"
        assert self.light.is_powered() is False
        assert self.light.get_level() == 50

    def test_set_level_clamps(self) -> None:
        self.light.set_level(150)
        assert self.light.get_level() == 100
        self.light.set_level(-50)
        assert self.light.get_level() == 0


class TestSmartAC:
    def test_set_level_converts_to_temperature(self) -> None:
        ac = SmartAC("AC-001", "Living")
        ac.set_level(22)
        assert ac.get_level() == 22

    def test_invalid_mode_ignored(self) -> None:
        ac = SmartAC("AC-001", "Living")
        ac.set_mode("invalid_mode")
        # Mode không thay đổi vì không hợp lệ
        assert ac._mode == "cool"


class TestRemoteControl:
    def test_toggle_power(self) -> None:
        device = MagicMock(spec=Device)
        device.is_powered.return_value = False
        remote = RemoteControl(device)
        remote.toggle_power()
        device.power_on.assert_called_once()

    def test_level_up(self) -> None:
        device = MagicMock(spec=Device)
        device.get_level.return_value = 50
        remote = RemoteControl(device)
        remote.level_up()
        device.set_level.assert_called_once_with(60)


class TestAdvancedRemoteControl:
    def test_boost_sets_level_to_100(self) -> None:
        device = MagicMock(spec=Device)
        remote = AdvancedRemoteControl(device)
        remote.boost()
        device.set_level.assert_called_once_with(100)

    def test_mute_turns_off_if_on(self) -> None:
        device = MagicMock(spec=Device)
        device.is_powered.return_value = True
        remote = AdvancedRemoteControl(device)
        remote.mute()
        device.power_off.assert_called_once()

    def test_mute_does_nothing_if_off(self) -> None:
        device = MagicMock(spec=Device)
        device.is_powered.return_value = False
        remote = AdvancedRemoteControl(device)
        remote.mute()
        device.power_off.assert_not_called()


class TestBridgeIntegration:
    def test_any_remote_with_any_device(self) -> None:
        """Bridge cho phép kết hợp bất kỳ remote với bất kỳ thiết bị."""
        light = SmartLight("L-001", "Test")
        ac = SmartAC("AC-001", "Test")

        # Mọi tổ hợp đều hợp lệ
        combos = [
            (RemoteControl, light),
            (RemoteControl, ac),
            (AdvancedRemoteControl, light),
            (AdvancedRemoteControl, ac),
        ]
        for remote_cls, device in combos:
            remote = remote_cls(device)
            remote.toggle_power()  # Không exception
            assert device.is_powered() is True
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Loại bỏ hoàn toàn class explosion — N + M thay vì N × M | Tăng độ phức tạp thiết kế ban đầu |
| Abstraction và implementation phát triển độc lập | Khó retrofit vào hệ thống có sẵn |
| Tuân thủ SRP — mỗi class chỉ có một lý do để thay đổi | Performance overhead do delegation |
| Linh hoạt ở runtime — có thể thay đổi implementation | Cần hiểu rõ hai chiều biến thiên |
| Dễ mở rộng — thêm thiết bị hoặc remote không ảnh hưởng nhau | Client code dài hơn do phải khởi tạo cả hai thành phần |
| Tái sử dụng code ở cả hai phía | Không phù hợp với hệ thống đơn giản, ít biến thiên |

---

Bridge Pattern là công cụ mạnh mẽ để quản lý sự phức tạp khi hệ thống có hai hoặc nhiều chiều biến thiên độc lập. Nó chuyển từ quan hệ "is-a" (kế thừa) sang "has-a" (composition), giúp codebase linh hoạt và dễ bảo trì hơn rất nhiều. Như câu nói: "Đổ mồ hôi trên sân tập để không đổ máu trên chiến trường" — hãy thiết kế ngay từ đầu, đừng đợi đến khi class explosion xảy ra.

**Nguyên tắc vàng**: Hãy dùng Bridge khi bạn thấy mình đang tạo ra các class với tên kết hợp như `XWithY`, `AdvancedXForSpecialY`. Đó là dấu hiệu của class explosion. Hãy dừng lại, xác định hai chiều biến thiên, tách chúng thành hai hệ thống phân cấp riêng, và kết nối bằng composition. Bạn sẽ giảm được 80% số lượng class và tăng gấp đôi khả năng mở rộng của hệ thống.

---
*Trân trọng!*
