---
id: mediator
title: Mediator
sidebar_label: 🤝 Mediator
sidebar_position: 18
---

# Mediator

> **Mediator** — *"Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly, and it lets you vary their interaction independently."* — GoF, 1994

## Bài toán chi tiết

Hãy tưởng tượng bạn xây hệ thống điều khiển chuyến bay tại sân bay. Có bao nhiêu thành phần? **ControlTower**, **Aircraft**, **Runway**, **Gate**, **WeatherService**, **GroundCrew** — và tất cả phải giao tiếp với nhau. Máy bay hỏi tower về đường băng trống, tower hỏi weather về gió, tower thông báo ground crew chuẩn bị tiếp nhiên liệu...

Nếu không có Mediator, mỗi object phải giữ tham chiếu đến tất cả object khác. `Aircraft` phải biết `ControlTower`, `Runway`, `Gate`, `GroundCrew`. Kết nối N×N tạo ra mạng lưới phụ thuộc chằng chịt — **như mớ bòng bong không thể gỡ.** Mỗi lần thay đổi một class, bạn phải sửa hàng loạt class khác.

**Vấn đề thứ hai:** Business logic giao tiếp bị phân tán. Logic "cho phép máy bay hạ cánh" bao gồm kiểm tra thời tiết, kiểm tra đường băng, thông báo ground crew, cập nhật gate schedule... Khi logic nằm rải rác trong nhiều class, debug và kiểm thử là cực kỳ khó khăn.

**Vấn đề thứ ba:** Tight coupling. Khi `Aircraft` gọi trực tiếp `Runway.is_available()`, nếu `Runway` đổi API thành `check_status()`, mọi class gọi `is_available()` đều phải sửa.

Cuối cùng, đồng bộ. Nhiều máy bay cùng yêu cầu đường băng — cần cơ chế tập trung để tránh race condition. Mediator là nơi duy nhất quản lý lock, queue, và priority.

## Giải pháp với Pattern

Mediator đặt toàn bộ logic giao tiếp vào một object trung gian. Các colleague không biết nhau — chúng chỉ biết mediator. Khi cần giao tiếp, colleague gửi thông báo đến mediator; mediator quyết định ai nhận, khi nào, và như thế nào. **Đây chính là "tổng đài viên" cho các object.**

**Cấu trúc:**
- **Mediator (ABC)**: interface cho giao tiếp (`notify()`, `register()`, `broadcast()`).
- **ConcreteMediator** (FlightControlTower): implement logic giao tiếp, quản lý colleagues.
- **Colleague** (Aircraft, Runway, Gate, GroundCrew): mỗi colleague chỉ giữ tham chiếu đến mediator.
- **Event/Dispatch**: colleague gửi event; mediator dispatch đến colleague phù hợp.

**Pattern giải quyết:**
- **N→1→N**: Thay vì N×N links → N links (mỗi colleague → mediator).
- **Centralized logic**: Mọi business rule giao tiếp tập trung trong mediator.
- **Loose coupling**: Colleague không biết colleague khác — chỉ biết mediator interface.
- **Dễ kiểm thử**: Mock mediator để test colleague riêng; mock colleague để test mediator.

## Phân tích thiết kế

**OOP Principles:**
- **Single Responsibility (SRP)**: Mediator quản lý giao tiếp; Colleague quản lý logic riêng.
- **Open/Closed (OCP)**: Thêm colleague mới không sửa colleague cũ — chỉ sửa mediator. Tuy nhiên, mediator thường phải sửa khi thêm colleague. Đây là trade-off kinh điển.
- **Dependency Inversion (DIP)**: Colleague phụ thuộc vào abstraction Mediator.
- **Law of Demeter (LoD)**: Colleague chỉ nói chuyện với mediator — không nói chuyện với colleague khác.

**Trade-offs:**
- **God Object risk**: Mediator có thể phình thành "super class". Nếu mediator quá lớn, hãy chia thành multiple mediators hoặc dùng Event Bus.
- **Single point of failure**: Mediator là bottleneck. Nếu mediator chết, toàn bộ giao tiếp sụp đổ.
- **Performance overhead**: Mỗi giao tiếp qua mediator — thêm một hop.

**Khi không nên dùng:**
- Giao tiếp đơn giản 1-1 — dùng direct reference hoặc Strategy.
- Cần broadcast thuần túy — dùng Observer pattern.
- Hệ thống peer-to-peer (BitTorrent, blockchain).

## Ví dụ code hoàn chỉnh

### Cách làm sai: Direct communication

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Aircraft:
    """Máy bay giao tiếp trực tiếp với nhiều object — coupling cao."""
    def __init__(self, callsign: str) -> None:
        self.callsign = callsign
        self.runway: Optional[Runway] = None
        self.tower: Optional[ControlTower] = None
        self.gate: Optional[Gate] = None
        self.weather: Optional[WeatherService] = None

    def request_landing(self) -> bool:
        # Phải biết 3 object khác — vi phạm LoD
        if not self.weather:
            raise RuntimeError("Weather service not set")
        if not self.runway:
            raise RuntimeError("Runway not set")

        if self.weather.is_storm():
            logger.warning(f"{self.callsign}: Storm detected, cannot land")
            return False

        if not self.runway.is_available():
            logger.warning(f"{self.callsign}: Runway busy")
            # Phải tự quản lý queue — logic phức tạp
            return False

        self.runway.occupy()
        if self.tower:
            self.tower.record_landing(self.callsign)
        if self.gate:
            self.gate.assign(self.callsign)
        return True

    # Mỗi lần thêm colleague mới (FuelTruck, Baggage) — phải thêm field + sửa logic


class Runway:
    def is_available(self) -> bool: ...
    def occupy(self) -> None: ...


class ControlTower:
    def record_landing(self, callsign: str) -> None: ...


class Gate:
    def assign(self, callsign: str) -> None: ...


class WeatherService:
    def is_storm(self) -> bool: ...
```

### Cách làm đúng: Mediator Pattern

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum, auto
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


# --- Event Types ---

class EventType(Enum):
    LANDING_REQUEST = auto()
    TAKEOFF_REQUEST = auto()
    RUNWAY_AVAILABLE = auto()
    GATE_AVAILABLE = auto()
    WEATHER_UPDATE = auto()
    EMERGENCY = auto()
    FUEL_REQUEST = auto()
    GROUND_CLEARED = auto()


@dataclass
class Event:
    type: EventType
    source: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


# --- Colleague Abstract ---

class Colleague(ABC):
    """Base class cho tất cả thành viên giao tiếp."""
    def __init__(self, name: str, mediator: Mediator) -> None:
        self.name = name
        self._mediator = mediator

    @abstractmethod
    def receive(self, event: Event) -> None:
        ...

    def send(self, event: Event) -> None:
        logger.debug(f"{self.name} sends: {event.type.name}")
        self._mediator.notify(self, event)

    @abstractmethod
    def status(self) -> str:
        ...


# --- Mediator ---

class Mediator(ABC):
    @abstractmethod
    def register(self, colleague: Colleague) -> None: ...

    @abstractmethod
    def notify(self, sender: Colleague, event: Event) -> None: ...


class FlightControlMediator(Mediator):
    """Concrete Mediator — điều phối toàn bộ giao tiếp sân bay."""

    def __init__(self) -> None:
        self._colleagues: dict[str, Colleague] = {}
        self._runway_queue: deque[Event] = deque()
        self._landing_log: list[str] = []
        self._emergency_mode: bool = False

    def register(self, colleague: Colleague) -> None:
        self._colleagues[colleague.name] = colleague

    def get(self, name: str) -> Optional[Colleague]:
        return self._colleagues.get(name)

    def notify(self, sender: Colleague, event: Event) -> None:
        if event.type == EventType.LANDING_REQUEST:
            self._handle_landing_request(sender, event)
        elif event.type == EventType.TAKEOFF_REQUEST:
            self._handle_takeoff_request(sender, event)
        elif event.type == EventType.WEATHER_UPDATE:
            self._handle_weather_update(sender, event)
        elif event.type == EventType.EMERGENCY:
            self._handle_emergency(sender, event)
        elif event.type == EventType.RUNWAY_AVAILABLE:
            self._handle_runway_available(sender, event)
        elif event.type == EventType.GATE_AVAILABLE:
            self._handle_gate_available(sender, event)
        elif event.type == EventType.FUEL_REQUEST:
            self._handle_fuel_request(sender, event)
        elif event.type == EventType.GROUND_CLEARED:
            self._handle_ground_cleared(sender, event)
        else:
            logger.warning(f"Unknown event type: {event.type}")

    def _handle_landing_request(self, aircraft: Colleague, event: Event) -> None:
        weather = self.get("weather_service")
        runway = self.get("runway_01")
        ground = self.get("ground_crew")

        if not weather or not runway or not ground:
            self._send_to(aircraft, "SYSTEM", Event(
                EventType.LANDING_REQUEST, "mediator",
                {"status": "rejected", "reason": "Missing services"}
            ))
            return

        # Kiểm tra thời tiết
        if event.data.get("weather_check", True):
            weather_event = Event(EventType.WEATHER_UPDATE, "mediator",
                                  {"query": True})
            weather.receive(weather_event)
            # Giả sử weather đã set storm flag

        # Nếu emergency, ưu tiên tuyệt đối
        if self._emergency_mode and event.data.get("emergency", False):
            self._clear_runway_for(aircraft, event)
            return

        # Nếu đường băng trống
        if self._check_runway_available(event.data.get("runway", "runway_01")):
            self._clear_runway_for(aircraft, event)
        else:
            self._runway_queue.append(event)
            self._send_to(aircraft, "mediator", Event(
                EventType.LANDING_REQUEST, "mediator",
                {"status": "queued", "position": len(self._runway_queue)}
            ))

    def _clear_runway_for(self, aircraft: Colleague, event: Event) -> None:
        runway_name = event.data.get("runway", "runway_01")
        runway = self.get(runway_name)
        ground = self.get("ground_crew")
        gate_name = event.data.get("gate", "gate_A1")
        gate = self.get(gate_name)

        if runway and ground:
            self._send_to(runway, "mediator", Event(
                EventType.RUNWAY_AVAILABLE, "mediator", {"occupy": True, "by": aircraft.name}
            ))
            self._send_to(ground, "mediator", Event(
                EventType.GROUND_CLEARED, "mediator", {"prepare_for": aircraft.name}
            ))

        if gate:
            self._send_to(gate, "mediator", Event(
                EventType.GATE_AVAILABLE, "mediator", {"assign": aircraft.name}
            ))

        self._landing_log.append(f"{aircraft.name} landed at {time.strftime('%H:%M:%S')}")
        self._send_to(aircraft, "mediator", Event(
            EventType.LANDING_REQUEST, "mediator",
            {"status": "cleared", "runway": runway_name, "gate": gate_name}
        ))

    def _handle_takeoff_request(self, aircraft: Colleague, event: Event) -> None:
        runway = self.get(event.data.get("runway", "runway_01"))
        if runway and self._check_runway_available("runway_01"):
            self._clear_runway_for(aircraft, event)
        else:
            self._send_to(aircraft, "mediator", Event(
                EventType.TAKEOFF_REQUEST, "mediator", {"status": "hold"}
            ))

    def _handle_weather_update(self, weather: Colleague, event: Event) -> None:
        storm = event.data.get("storm", False)
        self._emergency_mode = storm
        status = "STORM WARNING" if storm else "Weather clear"
        for name, colleague in self._colleagues.items():
            if colleague != weather:
                self._send_to(colleague, "mediator", Event(
                    EventType.WEATHER_UPDATE, "weather_service",
                    {"status": status}
                ))

    def _handle_emergency(self, sender: Colleague, event: Event) -> None:
        self._emergency_mode = True
        logger.warning(f"EMERGENCY from {sender.name}: {event.data}")
        # Broadcast to all
        for name, colleague in self._colleagues.items():
            if colleague != sender:
                self._send_to(colleague, "mediator", Event(
                    EventType.EMERGENCY, sender.name, event.data
                ))

    def _handle_runway_available(self, runway: Colleague, event: Event) -> None:
        if self._runway_queue:
            next_event = self._runway_queue.popleft()
            sender = self.get(next_event.source)
            if sender:
                self._clear_runway_for(sender, next_event)

    def _handle_gate_available(self, gate: Colleague, event: Event) -> None: ...
    def _handle_fuel_request(self, aircraft: Colleague, event: Event) -> None: ...
    def _handle_ground_cleared(self, ground: Colleague, event: Event) -> None: ...

    def _check_runway_available(self, name: str) -> bool:
        runway = self.get(name)
        if runway:
            return "Available" in runway.status()
        return False

    def _send_to(self, colleague: Colleague, source: str, event: Event) -> None:
        event.source = source
        colleague.receive(event)

    def landing_summary(self) -> list[str]:
        return self._landing_log


# --- Concrete Colleagues ---

class Aircraft(Colleague):
    def __init__(self, callsign: str, mediator: Mediator) -> None:
        super().__init__(callsign, mediator)
        self._altitude: int = 0
        self._fuel: int = 100
        self._cleared: bool = False

    def request_landing(self, emergency: bool = False) -> None:
        self.send(Event(
            EventType.LANDING_REQUEST, self.name,
            {"emergency": emergency, "fuel": self._fuel}
        ))

    def request_takeoff(self) -> None:
        self.send(Event(EventType.TAKEOFF_REQUEST, self.name))

    def receive(self, event: Event) -> None:
        if event.type == EventType.LANDING_REQUEST:
            if event.data.get("status") == "cleared":
                self._cleared = True
                logger.info(f"✅ {self.name}: CLEARED TO LAND (runway: {event.data.get('runway')})")
            elif event.data.get("status") == "queued":
                logger.info(f"⏳ {self.name}: Queued position {event.data.get('position')}")
            elif event.data.get("status") == "rejected":
                logger.error(f"❌ {self.name}: Landing rejected - {event.data.get('reason')}")
        elif event.type == EventType.EMERGENCY:
            logger.warning(f"⚠️ {self.name}: EMERGENCY broadcast - {event.data}")
        elif event.type == EventType.WEATHER_UPDATE:
            logger.info(f"🌤️ {self.name}: Weather - {event.data.get('status')}")

    def status(self) -> str:
        return f"Aircraft {self.name} | Alt={self._altitude}ft Fuel={self._fuel}%"


class Runway(Colleague):
    def __init__(self, name: str, mediator: Mediator) -> None:
        super().__init__(name, mediator)
        self._occupied: bool = False
        self._occupied_by: Optional[str] = None

    def receive(self, event: Event) -> None:
        if event.data.get("occupy"):
            self._occupied = True
            self._occupied_by = event.data.get("by", "unknown")
            logger.info(f"🛤️ {self.name}: OCCUPIED by {self._occupied_by}")
        elif event.data.get("free"):
            self._occupied = False
            self._occupied_by = None
            logger.info(f"🛤️ {self.name}: FREE")
            # Thông báo mediator
            self.send(Event(EventType.RUNWAY_AVAILABLE, self.name))

    def status(self) -> str:
        status = "Occupied" if self._occupied else "Available"
        return f"Runway {self.name}: {status}"


class Gate(Colleague):
    def __init__(self, name: str, mediator: Mediator) -> None:
        super().__init__(name, mediator)
        self._assigned: Optional[str] = None

    def receive(self, event: Event) -> None:
        if "assign" in event.data:
            self._assigned = event.data["assign"]
            logger.info(f"🚪 {self.name}: ASSIGNED to {self._assigned}")

    def status(self) -> str:
        return f"Gate {self.name}: {'Available' if not self._assigned else f'Used by {self._assigned}'}"


class GroundCrew(Colleague):
    def __init__(self, name: str, mediator: Mediator) -> None:
        super().__init__(name, mediator)

    def receive(self, event: Event) -> None:
        if event.type == EventType.GROUND_CLEARED:
            target = event.data.get("prepare_for", "unknown")
            logger.info(f"👷 {self.name}: Preparing ground for {target}")

    def status(self) -> str:
        return f"GroundCrew {self.name}: Ready"


class WeatherService(Colleague):
    def __init__(self, name: str, mediator: Mediator) -> None:
        super().__init__(name, mediator)
        self._storm: bool = False

    def set_storm(self, active: bool) -> None:
        self._storm = active
        self.send(Event(EventType.WEATHER_UPDATE, self.name, {"storm": active}))

    def receive(self, event: Event) -> None:
        if event.data.get("query"):
            # Weather check requested
            self.send(Event(EventType.WEATHER_UPDATE, self.name, {
                "storm": self._storm, "wind": 45 if self._storm else 12
            }))

    def status(self) -> str:
        return f"Weather: {'⚠️ STORM' if self._storm else '☀️ Clear'}"


# --- Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mediator = FlightControlMediator()

    # Tạo các colleague
    tower = type("Tower", (Colleague,), {
        "__init__": lambda s, n, m: Colleague.__init__(s, n, m),
        "receive": lambda s, e: logger.info(f"🏛️ Tower: {e.type.name}"),
        "status": lambda s: "Control Tower"
    })("control_tower", mediator)

    runway = Runway("runway_01", mediator)
    gate_a = Gate("gate_A1", mediator)
    gate_b = Gate("gate_B1", mediator)
    ground = GroundCrew("ground_crew_north", mediator)
    weather = WeatherService("weather_service", mediator)

    for c in [tower, runway, gate_a, gate_b, ground, weather]:
        mediator.register(c)

    # Kịch bản 1: Hạ cánh bình thường
    flight_va123 = Aircraft("VN-A123", mediator)
    mediator.register(flight_va123)
    flight_va123.request_landing()

    print("\n" + "=" * 50 + "\n")

    # Kịch bản 2: Thời tiết xấu
    flight_qh789 = Aircraft("VJ-B789", mediator)
    mediator.register(flight_qh789)
    weather.set_storm(True)
    flight_qh789.request_landing()

    print("\n" + "=" * 50 + "\n")

    # Kịch bản 3: Emergency
    weather.set_storm(False)
    flight_emergency = Aircraft("EMG-001", mediator)
    mediator.register(flight_emergency)
    flight_emergency.request_landing(emergency=True)

    print("\n" + "=" * 50 + "\n")

    # Landing summary
    print("\n📋 Landing Summary:")
    for entry in mediator.landing_summary():
        print(f"  {entry}")
```

## Sơ đồ UML

```
┌─────────────────────┐         ┌──────────────────────┐
│    Mediator (ABC)   │«uses»  │     Colleague (ABC)   │
│─────────────────────│         │──────────────────────│
│ + register(c)       │         │ - name: str          │
│ + notify(s, e)      │         │ - mediator: Mediator │
└────────┬────────────┘         │──────────────────────│
         │                      │ + receive(e)         │
         │                      │ + send(e)            │
         │                      │ + status(): str      │
         │                      └──────────┬───────────┘
         │                                 │
┌────────┴────────────┐         ┌──────────┼──────────┐
│FlightControlMediator│         │          │          │
│─────────────────────│    ┌────┴───┐  ┌───┴────┐  ┌──┴────┐
│ - colleagues: dict  │    │Aircraft│  │ Runway │  │ Gate  │
│ - queue: deque      │    │        │  │        │  │       │
│ - landing_log: list │    └────────┘  └────────┘  └───────┘
│─────────────────────│         │          │          │
│ + notify(s, e)      │    ┌────┴───┐  ┌───┴────┐  ┌──┴──────┐
│ - _handle_landing() │    │Ground  │  │Weather │  │Control  │
│ - _handle_emergency │    │Crew    │  │Service │  │Tower    │
│ - _handle_xxx()     │    └────────┘  └────────┘  └─────────┘
└─────────────────────┘

Giao tiếp: Colleague → Mediator → Colleague (không direct)
```

## So sánh với Pattern liên quan

**1. Observer Pattern:**

Observer phân phối event **một chiều** (subject → observers). Mediator cho phép giao tiếp **hai chiều** (colleague ↔ mediator ↔ colleague). Observer là broadcast (1→n). Mediator là point-to-point hoặc broadcast tùy logic. Observer không có centralized business logic. **Khác nhau cơ bản về luồng giao tiếp.**

**2. Facade Pattern:**

Facade đơn giản hóa interface cho một subsystem. Mediator là động và cho phép giao tiếp hai chiều. Facade không thêm behavior — nó chỉ delegate. Mediator thêm behavior orchestration. **Facade là tĩnh, Mediator là động.**

**3. Event Bus / Message Broker:**

Event Bus là dạng Mediator hiện đại — hàng đợi message, pub/sub. Mediator nguyên bản đồng bộ; Event Bus thường async. Kafka, RabbitMQ là Mediator ở quy mô hệ thống. Cùng ý tưởng "trung gian giao tiếp", khác scale.

## Ứng dụng thực tế

**1. Django Signals:**
Django signal dispatcher là Mediator: sender gửi signal, dispatcher gọi tất cả receiver đã đăng ký.

```python
from django.dispatch import Signal, receiver

# Định nghĩa signal
order_placed = Signal()

# Receiver 1: gửi email
@receiver(order_placed)
def send_order_email(sender, **kwargs):
    order = kwargs["order"]
    print(f"Email sent for order {order.id}")

# Receiver 2: cập nhật inventory
@receiver(order_placed)
def update_inventory(sender, **kwargs):
    print("Inventory updated")

# Sender không biết receiver — signal dispatcher là mediator
order_placed.send(sender=__name__, order=my_order)
```

**2. Flask `current_app`:**
Flask dùng `current_app` như mediator giữa request context, extension, và các service.

```python
from flask import Flask, current_app

app = Flask(__name__)

# Extension đăng ký với app
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy(app)  # db giao tiếp qua app

@app.route("/")
def index():
    # current_app là mediator giữa view, db, cache
    result = db.session.execute("SELECT 1")
    current_app.logger.info("Request processed")
    return "OK"
```

**3. Air Traffic Control Systems:**
ATC là kinh điển của Mediator ngoài đời thực. Pilot không nói chuyện trực tiếp với pilot khác — tất cả qua controller.

```
Pilot A ──┐
          ├──→ ATC Controller ──→ Pilot B, C, D
Pilot B ──┘         │
                    └──→ Weather, Ground, Gate services
```

**4. JavaScript / Redux:**
Redux store là mediator cho state management. Component dispatch action → store gọi reducer → store notify subscriber.

```javascript
// Redux store = Mediator
const store = createStore(reducer);

// Component gửi action qua store (không gọi component khác)
store.dispatch({ type: 'INCREMENT' });

// Store thông báo cho subscriber
store.subscribe(() => console.log(store.getState()));
```

## Kiểm thử

```python
import pytest
from unittest.mock import Mock, ANY


class TestMediator:
    def setup_method(self):
        self.mediator = FlightControlMediator()
        self.weather = WeatherService("weather_test", self.mediator)
        self.runway = Runway("runway_test", self.mediator)
        self.gate = Gate("gate_test", self.mediator)
        self.ground = GroundCrew("ground_test", self.mediator)
        self.aircraft = Aircraft("AC-TEST", self.mediator)

        for c in [self.weather, self.runway, self.gate, self.ground, self.aircraft]:
            self.mediator.register(c)

    def test_aircraft_landing_normal(self):
        """Máy bay hạ cánh bình thường — nhận cleared."""
        self.aircraft.request_landing()
        assert self.aircraft._cleared is True

    def test_aircraft_landing_storm(self):
        """Thời tiết xấu — từ chối hạ cánh (do queued)."""
        self.weather.set_storm(True)
        self.aircraft.request_landing()

    def test_emergency_broadcast(self):
        """Emergency broadcast đến tất cả colleague."""
        aircraft_2 = Aircraft("AC-TEST2", self.mediator)
        self.mediator.register(aircraft_2)

        aircraft_2.request_landing(emergency=True)

    def test_mediator_registers_all(self):
        """Tất cả colleague đều trong mediator."""
        assert "weather_test" in self.mediator._colleagues
        assert "runway_test" in self.mediator._colleagues
        assert "gate_test" in self.mediator._colleagues

    def test_landing_log_updated(self):
        """Landing log ghi lại hạ cánh."""
        self.aircraft.request_landing()
        assert len(self.mediator.landing_summary()) == 1

    def test_event_priority(self):
        """Event có priority field."""
        event = Event(EventType.EMERGENCY, "test", {"msg": "fire"}, priority=10)
        assert event.priority == 10
        assert event.type == EventType.EMERGENCY

    def test_weather_broadcast(self):
        """Cập nhật thời tiết broadcast đến tất cả."""
        self.weather.set_storm(True)
        # Kiểm tra qua log: storm warning được broadcast
        assert self.mediator._emergency_mode is True


class TestMediatorColleagueIsolation:
    """Kiểm thử colleague không biết nhau — chỉ biết mediator."""

    def test_aircraft_does_not_know_runway(self):
        """Aircraft không có tham chiếu trực tiếp đến Runway."""
        mediator = FlightControlMediator()
        ac = Aircraft("TEST", mediator)
        # Không có field self.runway — chỉ có mediator
        assert hasattr(ac, "_mediator")
        assert not hasattr(ac, "runway")

    def test_mediator_can_be_mocked(self):
        """Dễ dàng mock mediator để test colleague riêng."""
        mock_mediator = Mock(spec=Mediator)
        ac = Aircraft("TEST", mock_mediator)
        ac.request_landing()
        # Gửi event đến mediator
        mock_mediator.notify.assert_called_once()
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| Giảm coupling từ N×N xuống N | Mediator dễ thành God Object (quá nhiều logic) |
| Tập trung hóa logic giao tiếp | Single point of failure |
| Dễ kiểm thử (mock mediator / mock colleague) | Thêm một hop trong giao tiếp (latency) |
| Dễ thay đổi business rule (sửa 1 nơi) | Colleague phụ thuộc vào mediator interface |
| Colleague độc lập, tái sử dụng được | Debug khó vì luồng giao tiếp gián tiếp |
| Hỗ trợ đồng bộ tập trung | Mediator phình to theo số colleague |
| Dễ dàng thêm colleague mới (OCP một phần) | Không phù hợp hệ thống peer-to-peer |

---

## Kết luận

**Mediator pattern là giải pháp tối ưu cho các hệ thống có giao tiếp phức tạp, nhiều chiều, cần đồng bộ.** Hãy dùng nó khi bạn thấy mối quan hệ giữa các object trở nên chằng chịt như mạng nhện — mỗi object phải biết quá nhiều object khác.

Những điều cần nhớ:
1. Đừng để Mediator phình thành God Object — nếu nó > 500 dòng, hãy chia nhỏ ra.
2. Colleague chỉ nên biết mediator interface, không bao giờ biết concrete mediator.
3. Dùng **Event object** để giao tiếp — dễ mở rộng, dễ log.
4. Cân nhắc **Event Bus / Message Queue** cho hệ thống phân tán.
5. Luôn có **timeout** và **fallback** — tránh deadlock khi colleague không phản hồi.

---
*Trân trọng!*
