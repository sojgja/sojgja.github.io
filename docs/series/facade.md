---
id: facade
title: Facade
sidebar_label: 🏢 Facade
sidebar_position: 11
---

# Facade

> "Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use." — Erich Gamma, *Design Patterns: Elements of Reusable Object-Oriented Software*

Bạn có bao giờ phải gọi 20 API khác nhau chỉ để làm một việc đơn giản? Tôi thì có, và nó ám ảnh tôi đến tận bây giờ...

## Bài toán chi tiết

Tôi muốn kể cho bạn câu chuyện về một công ty bất động sản đang phát triển hệ thống **Smart Building** — quản lý tòa nhà văn phòng 50 tầng. Hệ thống này tích hợp với hàng chục subsystem khác nhau: hệ thống chiếu sáng (LuxControl), HVAC (HeatVentAc), báo cháy (FireAlertPro), an ninh (SecureGate), thang máy (ElevatorBrain), quản lý năng lượng (PowerOptimizer)... Mỗi subsystem có interface riêng, protocol riêng (MQTT, HTTP, WebSocket, Modbus), và yêu cầu khởi tạo phức tạp.

Bạn thử tưởng tượng xem: khi một nhân viên bảo trì muốn thực hiện thao tác "kích hoạt chế độ khẩn cấp", anh ta phải làm tuần tự hàng chục bước:
1. Gọi `lux_control.set_emergency_lighting(True)` với API key
2. Gọi `hvac.shutdown_all()` qua MQTT topic `building/hvac/emergency`
3. Gọi `fire_alert.silence_alarm()` — nhưng chỉ nếu alarm đang kêu
4. Gọi `secure_gate.unlock_all_exits()` với master token
5. Gọi `elevator.brain.send_command("ground_floor_all")` qua gRPC
6. Gọi `power_optimizer.cut_non_essential()` qua REST API
7. Và hơn chục bước khác nữa...

Nếu mỗi subsystem thay đổi API (ví dụ: nâng cấp firmware), tất cả các client phải cập nhật theo. Đây là một ví dụ điển hình của **tight coupling** — client phụ thuộc quá nhiều vào chi tiết implementation của subsystem. Và bạn biết không? Khi có sự cố thực sự (cháy thật), nhân viên bảo trì không thể nhớ hết 20 bước — một sai sót nhỏ cũng có thể dẫn đến hậu quả nghiêm trọng. **Cần một giải pháp đơn giản hóa.**

## Giải pháp với Pattern

Facade Pattern cung cấp một interface đơn giản, thống nhất che giấu hoàn toàn sự phức tạp của các subsystem bên dưới. Thay vì client phải tương tác với 10 subsystem khác nhau, **client chỉ cần tương tác với một class Facade duy nhất.** Facade biết cách phối hợp các subsystem với nhau và xử lý mọi chi tiết kỹ thuật.

Cấu trúc Facade gồm:
- **Facade**: Lớp duy nhất mà client tương tác — cung cấp các method đơn giản, có ý nghĩa nghiệp vụ như `activate_emergency_mode()`, `leave_building()`, `optimize_energy()`.
- **Subsystem Classes**: Các class phức tạp của từng hệ thống — client không bao giờ gọi trực tiếp.
- **Client**: Chỉ phụ thuộc vào Facade, hoàn toàn không biết subsystem tồn tại.

Điểm mạnh: **Facade không ngăn cản client truy cập subsystem khi cần thiết** — nó chỉ cung cấp một "lối tắt" cho các tác vụ phổ biến. Subsystem vẫn có thể được gọi trực tiếp nếu client cần kiểm soát chi tiết.

## Phân tích thiết kế

Facade Pattern thể hiện rõ nguyên lý **Law of Demeter** (nguyên tắc ít biết nhất): client chỉ nên biết đến Facade, không cần biết subsystem. Nó cũng giảm **coupling** giữa client và subsystem — khi subsystem thay đổi, chỉ Facade cần cập nhật, client không bị ảnh hưởng. Đây là một hình thức của **information hiding**.

**Phân biệt với các pattern tương tự:**
- Facade vs Adapter: Facade đơn giản hóa interface, Adapter chuyển đổi interface.
- Facade vs Mediator: Mediator quản lý giao tiếp giữa các object (hai chiều), Facade cung cấp interface đơn giản hóa cho client (một chiều).
- Facade vs Proxy: Proxy kiểm soát truy cập đến một đối tượng duy nhất, Facade đơn giản hóa truy cập đến cả một subsystem.

**Khi KHÔNG nên dùng Facade:**
- Khi client cần kiểm soát chi tiết từng subsystem — Facade trở thành bottleneck.
- Khi subsystem đã đơn giản — Facade chỉ thêm độ phức tạp vô ích.
- Khi Facade trở thành "god object" chứa quá nhiều trách nhiệm.
- Khi performance là critical — Facade thêm một lớp gọi hàm.

**Trade-offs:**
- Facade có thể trở thành single point of failure — nếu nó hỏng, cả hệ thống ngừng hoạt động.
- Facade có tendency trở nên phình to khi quá nhiều use case được thêm vào.
- Khó test subsystem riêng lẻ nếu mọi thứ đều qua Facade.
- Có thể che giấu các vấn đề của subsystem (khó debug).

## Ví dụ code hoàn chỉnh

### Cách làm sai: Client gọi trực tiếp subsystem

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol
import time


class AlarmStatus(Enum):
    ACTIVE = auto()
    SILENT = auto()
    DISABLED = auto()


class PowerZone(Enum):
    ESSENTIAL = "essential"
    NON_ESSENTIAL = "non_essential"
    EMERGENCY = "emergency"


# --- Complex subsystems ---
class LightingSystem:
    """Hệ thống chiếu sáng — giao tiếp qua REST API."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._lights: dict[str, bool] = {}

    def authenticate(self, api_key: str) -> bool:
        return api_key == self._api_key

    def set_zone_brightness(self, zone: str, level: int) -> str:
        time.sleep(0.05)
        self._lights[zone] = level > 0
        return f"Zone {zone} brightness set to {level}%"

    def emergency_lighting(self, active: bool) -> str:
        return f"Emergency lighting {'ON' if active else 'OFF'}"


class HVACSystem:
    """HVAC — giao tiếp qua MQTT."""

    def __init__(self, mqtt_topic: str) -> None:
        self._topic = mqtt_topic

    def shutdown(self) -> str:
        time.sleep(0.05)
        return f"HVAC shutdown via {self._topic}"

    def set_temperature(self, zone: str, temp: float) -> str:
        return f"Zone {zone} set to {temp}°C"

    def get_status(self) -> dict:
        return {"status": "running", "temp": 25.0}


class SecuritySystem:
    """Hệ thống an ninh — giao tiếp qua gRPC."""

    def __init__(self, master_token: str) -> None:
        self._token = master_token
        self._armed = False

    def arm(self, token: str) -> str:
        if token == self._token:
            self._armed = True
            return "Security ARMED"
        return "Authentication failed"

    def disarm(self, token: str) -> str:
        if token == self._token:
            self._armed = False
            return "Security DISARMED"
        return "Authentication failed"

    def unlock_all_exits(self, token: str) -> str:
        if token == self._token:
            return "All exits UNLOCKED"
        return "Authentication failed"


class FireSystem:
    """Hệ thống báo cháy — Modbus protocol."""

    def __init__(self) -> None:
        self._alarm = AlarmStatus.DISABLED

    def trigger_test(self) -> str:
        self._alarm = AlarmStatus.ACTIVE
        return "Fire drill ACTIVATED"

    def silence(self) -> str:
        if self._alarm == AlarmStatus.ACTIVE:
            self._alarm = AlarmStatus.SILENT
            return "Alarm SILENCED"
        return "No active alarm to silence"

    def reset(self) -> str:
        self._alarm = AlarmStatus.DISABLED
        return "System RESET"

    def get_alarm_status(self) -> AlarmStatus:
        return self._alarm


# Client code — phức tạp, dễ sai, phụ thuộc vào mọi subsystem
lights = LightingSystem(api_key="KEY_123")
hvac = HVACSystem(mqtt_topic="building/hvac")
security = SecuritySystem(master_token="MASTER_SECRET")
fire = FireSystem()

# Client phải biết:
# 1. Thứ tự gọi
# 2. API key / token
# 3. Protocol cụ thể
# 4. Xử lý lỗi từng subsystem
```

### Cách đúng: Facade Pattern

```python
# --- Facade ---
class SmartBuildingFacade:
    """Facade đơn giản hóa toàn bộ Smart Building system."""

    def __init__(
        self,
        lighting_api_key: str,
        mqtt_topic: str,
        security_token: str,
    ) -> None:
        self._lighting = LightingSystem(lighting_api_key)
        self._hvac = HVACSystem(mqtt_topic)
        self._security = SecuritySystem(security_token)
        self._fire = FireSystem()
        self._emergency_active = False

    def activate_emergency_mode(self) -> list[str]:
        """Kích hoạt chế độ khẩn cấp — một method thay cho 20 bước."""
        logs: list[str] = []
        try:
            logs.append(self._fire.silence())
            logs.append(self._lighting.emergency_lighting(True))
            logs.append(self._lighting.set_zone_brightness("all", 100))
            logs.append(self._hvac.shutdown())
            logs.append(self._security.unlock_all_exits("MASTER_SECRET"))
            logs.append("[SYSTEM] Emergency mode activated — all safe")
            self._emergency_active = True
        except Exception as exc:
            logs.append(f"[ERROR] Emergency mode failed: {exc}")
        return logs

    def deactivate_emergency_mode(self) -> list[str]:
        """Thoát chế độ khẩn cấp — khôi phục hệ thống."""
        logs: list[str] = []
        try:
            logs.append(self._lighting.emergency_lighting(False))
            logs.append(self._lighting.set_zone_brightness("all", 50))
            logs.append(self._fire.reset())
            logs.append("[SYSTEM] Emergency mode deactivated")
            self._emergency_active = False
        except Exception as exc:
            logs.append(f"[ERROR] Deactivation failed: {exc}")
        return logs

    def leave_building(self) -> list[str]:
        """Chuẩn bị khi rời tòa nhà."""
        logs: list[str] = []
        logs.append(self._lighting.set_zone_brightness("all", 10))
        logs.append(self._hvac.set_temperature("all", 18.0))
        logs.append(self._security.arm("MASTER_SECRET"))
        logs.append("[SYSTEM] Building is secure. Goodbye!")
        return logs

    def start_workday(self) -> list[str]:
        """Chuẩn bị tòa nhà cho ngày làm việc mới."""
        logs: list[str] = []
        logs.append(self._security.disarm("MASTER_SECRET"))
        logs.append(self._lighting.set_zone_brightness("floor_1", 80))
        logs.append(self._lighting.set_zone_brightness("floor_2", 80))
        logs.append(self._hvac.set_temperature("floor_1", 24.0))
        logs.append(self._hvac.set_temperature("floor_2", 24.0))
        logs.append("[SYSTEM] Building ready for work!")
        return logs

    def run_fire_drill(self) -> list[str]:
        """Mô phỏng cháy — kiểm tra toàn bộ hệ thống."""
        logs: list[str] = []
        logs.append(self._fire.trigger_test())
        logs.extend(self.activate_emergency_mode())
        logs.append("[DRILL] Fire drill completed successfully")
        return logs

    def get_building_status(self) -> dict:
        return {
            "hvac": self._hvac.get_status(),
            "emergency": self._emergency_active,
            "fire_alarm": self._fire.get_alarm_status().name,
        }


# --- Client — đơn giản, gọn nhẹ ---
class BuildingManager:
    """Người quản lý tòa nhà — chỉ cần biết Facade."""

    def __init__(self, facade: SmartBuildingFacade) -> None:
        self._facade = facade

    def handle_emergency(self) -> None:
        print("🚨 EMERGENCY! Activating protocols...")
        for log in self._facade.activate_emergency_mode():
            print(f"  → {log}")

    def end_of_day(self) -> None:
        print("\n🌙 End of day — securing building...")
        for log in self._facade.leave_building():
            print(f"  → {log}")

    def morning_setup(self) -> None:
        print("\n☀️ Good morning! Preparing building...")
        for log in self._facade.start_workday():
            print(f"  → {log}")


# --- Usage ---
if __name__ == "__main__":
    facade = SmartBuildingFacade(
        lighting_api_key="KEY_123",
        mqtt_topic="building/hvac",
        security_token="MASTER_SECRET",
    )
    manager = BuildingManager(facade)

    manager.morning_setup()
    manager.handle_emergency()
    manager.end_of_day()

    # Nếu cần, client vẫn có thể truy cập subsystem trực tiếp
    print("\n=== Direct HVAC Access ===")
    print(facade._hvac.get_status())
```

## Sơ đồ UML

```
┌─────────────────────────────────────────────────────┐
│                   Client                            │
│              (BuildingManager)                      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              SmartBuildingFacade                    │
│─────────────────────────────────────────────────────│
│ - _lighting: LightingSystem                         │
│ - _hvac: HVACSystem                                 │
│ - _security: SecuritySystem                         │
│ - _fire: FireSystem                                 │
│─────────────────────────────────────────────────────│
│ + activate_emergency_mode() → list[str]             │
│ + deactivate_emergency_mode() → list[str]           │
│ + leave_building() → list[str]                      │
│ + start_workday() → list[str]                       │
│ + run_fire_drill() → list[str]                      │
│ + get_building_status() → dict                      │
└──────┬─────────────┬──────────────┬─────────────────┘
       │             │              │
       ▼             ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐
│Lighting  │ │ HVAC     │ │ Security     │ │ Fire     │
│ System   │ │ System   │ │ System       │ │ System   │
│──────────│ │──────────│ │──────────────│ │──────────│
│+ set_zone│ │+ shutdown│ │+ arm()       │ │+ trigger │
│ brightness││+ setTemp │ │+ disarm()    │ │+ silence │
│+ emerge- │ │+ getStat │ │+ unlockAll() │ │+ reset() │
│ ncyLight │ └──────────┘ └──────────────┘ └──────────┘
└──────────┘
```

## So sánh với Pattern liên quan

**Facade vs Mediator**: Cả hai đều quản lý giao tiếp giữa nhiều class. Mediator quản lý giao tiếp hai chiều giữa các colleague object — các colleague không biết nhau, mọi giao tiếp đều qua Mediator. Facade cung cấp interface một chiều từ client đến subsystem — subsystem vẫn có thể giao tiếp trực tiếp với nhau. Facade đơn giản hóa việc sử dụng, Mediator đơn giản hóa việc giao tiếp giữa các thành phần.

**Facade vs Adapter**: Adapter chuyển đổi interface — client mong đợi interface A, thực tế có interface B, Adapter làm cầu nối. Facade không chuyển đổi interface — nó cung cấp interface mới, đơn giản hơn cho một nhóm subsystem. Adapter wrap một class, Facade wrap nhiều class.

**Facade vs Proxy**: Proxy cung cấp cùng interface với đối tượng thật để kiểm soát truy cập (lazy loading, protection). Facade cung cấp interface khác, đơn giản hơn. Proxy thay thế một đối tượng, Facade che giấu cả một hệ thống.

## Ứng dụng thực tế

**1. `requests` library — HTTP làm việc dễ dàng**: Thư viện `requests` là một Facade điển hình cho các module HTTP phức tạp của Python (`http.client`, `urllib3`, socket). Thay vì phải quản lý connection pool, SSL handshake, redirect handling, encoding, header parsing..., người dùng chỉ cần gọi `requests.get()`:

```python
import requests

# Với requests — Facade
response = requests.get("https://api.github.com", timeout=5)
print(response.json())

# Không có requests — phải tự làm mọi thứ
import http.client
conn = http.client.HTTPSConnection("api.github.com")
conn.request("GET", "/")
response = conn.getresponse()
data = response.read()
conn.close()
```

**2. Django Class-Based Views**: `CreateView`, `ListView` là Facade cho toàn bộ quy trình xử lý request: authentication, authorization, form rendering, validation, database save, response generation. Lập trình viên chỉ cần khai báo `model` và `fields`, mọi thứ khác đã được Facade xử lý:

```python
from django.views.generic import CreateView
from .models import Article

# Facade — một class thay cho 50 dòng logic
class ArticleCreateView(CreateView):
    model = Article
    fields = ['title', 'content']
    template_name = 'article_form.html'
```

**3. Docker Compose**: `docker-compose up` là Facade cho hàng loạt lệnh Docker: build image, tạo network, mount volume, start container theo đúng thứ tự, health check. Thay vì gõ 20 lệnh `docker`, người dùng chỉ cần một file YAML và một lệnh:

```yaml
# docker-compose.yml — Facade cho Docker commands
version: '3'
services:
  web:
    build: .
    ports: ["8000:8000"]
  db:
    image: postgres:15
```

**4. ORM (SQLAlchemy, Django ORM)**: SQLAlchemy's `session.commit()` là Facade cho toàn bộ quy trình flush → transaction → commit → close connection, bao gồm xử lý lỗi, rollback, connection pooling. Lập trình viên không cần biết chi tiết giao tiếp với database.

## Kiểm thử

```python
import pytest
from unittest.mock import MagicMock, patch
from facade import (
    SmartBuildingFacade, BuildingManager,
    LightingSystem, HVACSystem, SecuritySystem, FireSystem,
)


class TestSmartBuildingFacade:
    def setup_method(self) -> None:
        self.facade = SmartBuildingFacade(
            lighting_api_key="KEY",
            mqtt_topic="topic",
            security_token="TOKEN",
        )

    def test_emergency_mode_activates_all_systems(self) -> None:
        logs = self.facade.activate_emergency_mode()
        assert len(logs) == 6
        assert "Emergency mode activated" in logs[-1]
        assert self.facade._emergency_active is True

    def test_emergency_mode_includes_shutdown(self) -> None:
        logs = self.facade.activate_emergency_mode()
        shutdown_logs = [l for l in logs if "shutdown" in l.lower() or "SHUTDOWN" in l]
        assert len(shutdown_logs) >= 1

    def test_deactivate_resets_system(self) -> None:
        self.facade.activate_emergency_mode()
        logs = self.facade.deactivate_emergency_mode()
        assert "Emergency mode deactivated" in logs[-1]
        assert self.facade._emergency_active is False

    def test_leave_building_arms_security(self) -> None:
        logs = self.facade.leave_building()
        assert "ARMED" in str(logs) or "secure" in logs[-1].lower()

    def test_fire_drill_runs_emergency(self) -> None:
        logs = self.facade.run_fire_drill()
        assert "Fire drill" in logs[0]
        assert "completed" in logs[-1]


class TestBuildingManager:
    def test_manager_handles_emergency(self, capsys) -> None:
        facade = SmartBuildingFacade("KEY", "topic", "TOKEN")
        manager = BuildingManager(facade)
        manager.handle_emergency()
        captured = capsys.readouterr()
        assert "EMERGENCY" in captured.out


class TestSubsystemsStillAccessible:
    def test_direct_subsystem_access(self) -> None:
        """Facade không ngăn cản truy cập trực tiếp subsystem."""
        facade = SmartBuildingFacade("KEY", "topic", "TOKEN")
        hvac_status = facade._hvac.get_status()
        assert hvac_status["status"] == "running"


class TestFacadeWithMocks:
    def test_facade_delegates_to_subsystems(self) -> None:
        mock_lighting = MagicMock(spec=LightingSystem)
        mock_hvac = MagicMock(spec=HVACSystem)
        mock_security = MagicMock(spec=SecuritySystem)
        mock_fire = MagicMock(spec=FireSystem)

        # Inject mocks
        facade = SmartBuildingFacade.__new__(SmartBuildingFacade)
        facade._lighting = mock_lighting
        facade._hvac = mock_hvac
        facade._security = mock_security
        facade._fire = mock_fire
        facade._emergency_active = False

        facade.activate_emergency_mode()
        mock_fire.silence.assert_called_once()
        mock_lighting.emergency_lighting.assert_called_once_with(True)
        mock_hvac.shutdown.assert_called_once()
        mock_security.unlock_all_exits.assert_called_once()
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Giảm coupling — client không phụ thuộc vào subsystem | Có thể trở thành god object nếu không kiểm soát |
| Đơn giản hóa API — một method cho một use case phức tạp | Che giấu lỗi — subsystem lỗi khó debug |
| Tách biệt concerns — subsystem thay đổi không ảnh hưởng client | Facade có thể trở thành bottleneck về hiệu năng |
| Dễ test — chỉ cần test Facade thay vì từng subsystem | Nếu client cần quyền kiểm soát chi tiết, Facade gây khó chịu |
| Giảm dependency — thay đổi subsystem chỉ ảnh hưởng Facade | Thêm một lớp gián tiếp |
| Tái cấu trúc subsystem dễ dàng | Khó maintain nếu subsystem quá nhiều và thay đổi thường xuyên |

---

Facade Pattern là giải pháp tuyệt vời khi bạn cần cung cấp một interface đơn giản cho người dùng cuối, trong khi vẫn giữ được sự linh hoạt và phức tạp của hệ thống bên trong. Nó đặc biệt hữu ích trong các hệ thống tích hợp nhiều thư viện, API, hoặc service — nơi người dùng không cần biết chi tiết kỹ thuật. Như câu nói: **"Đổ mồ hôi trên sân tập, đừng đổ máu trên chiến trường"** — hãy dành thời gian xây dựng Facade tốt, để client code của bạn đơn giản và an toàn.

**Nguyên tắc vàng**: Facade nên được thiết kế theo use case của client, không phải theo cấu trúc của subsystem. Hãy tự hỏi: "Người dùng muốn làm gì?" thay vì "Subsystem có gì?". Một Facade tốt là một Facade mà người dùng có thể hoàn thành tác vụ phức tạp chỉ với một method call — giống như nút "Tự động" trên máy giặt vậy.

---
*Trân trọng!*
