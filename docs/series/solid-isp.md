---
id: solid-isp
title: I — Interface Segregation Principle
sidebar_label: I — Interface Segregation
sidebar_position: 29
---

# I — Interface Segregation Principle

> **"No client should be forced to depend on methods it does not use."** — Robert C. Martin

**Nhiều interface nhỏ, chuyên biệt** tốt hơn **một interface lớn, tổng hợp**. Client không nên bị ép phải implement những method chúng không cần.

## Bài toán: Interface "khổng lồ"

```python
from abc import ABC, abstractmethod

class Worker(ABC):
    @abstractmethod
    def work(self): pass

    @abstractmethod
    def eat(self): pass

    @abstractmethod
    def sleep(self): pass

    @abstractmethod
    def attend_meeting(self): pass

class HumanDeveloper(Worker):
    def work(self): print('👨‍💻 Coding...')
    def eat(self): print('🍜 Ăn trưa...')
    def sleep(self): print('😴 Ngủ...')
    def attend_meeting(self): print('📋 Họp...')

class RobotDeveloper(Worker):
    def work(self): print('🤖 Coding 24/7...')

    def eat(self):
        raise NotImplementedError('Robot không ăn')  # ❌

    def sleep(self):
        raise NotImplementedError('Robot không ngủ')  # ❌

    def attend_meeting(self):
        raise NotImplementedError('Robot không họp')  # ❌
```

**Vấn đề:** `Robot` bị ép implement 3 method vô dụng. Vi phạm LSP (khi gọi `eat()` → crash) và ISP (phụ thuộc vào method không dùng).

## Giải pháp: Tách interface

```python
from abc import ABC, abstractmethod

class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

class Sleepable(ABC):
    @abstractmethod
    def sleep(self): pass

class MeetingParticipant(ABC):
    @abstractmethod
    def attend_meeting(self): pass

class HumanDeveloper(Workable, Eatable, Sleepable, MeetingParticipant):
    def work(self): print('👨‍💻 Coding...')
    def eat(self): print('🍜 Ăn trưa...')
    def sleep(self): print('😴 Ngủ...')
    def attend_meeting(self): print('📋 Họp...')

class RobotDeveloper(Workable):
    def work(self): print('🤖 Coding 24/7...')

# ✅ Không cần implement method vô dụng
robot = RobotDeveloper()
robot.work()  # OK

# Type hints vẫn an toàn
def assign_task(worker: Workable):
    worker.work()

assign_task(HumanDeveloper())
assign_task(RobotDeveloper())  # ✅ Cả hai đều workable
```

## Ví dụ thực tế: Machine

```python
from abc import ABC, abstractmethod

class Printer(ABC):
    @abstractmethod
    def print(self, doc): pass

class Scanner(ABC):
    @abstractmethod
    def scan(self): pass

class Fax(ABC):
    @abstractmethod
    def fax(self, doc): pass

class BasicPrinter(Printer):
    def print(self, doc): print(f'🖨️ In: {doc}')

class MultiFunctionPrinter(Printer, Scanner, Fax):
    def print(self, doc): print(f'🖨️ In: {doc}')
    def scan(self): print('📄 Scan tài liệu...')
    def fax(self, doc): print(f'📠 Gửi fax: {doc}')

class ScannerOnly(Scanner):
    def scan(self): print('📄 Scan...')
```

## Dấu hiệu nhận biết vi phạm ISP

- Interface có method `raise NotImplementedError` hoặc `pass`
- Client implement interface nhưng bỏ trống method
- Interface có tên chung chung (`IProcessor`, `IManager`)
- Một class implement quá nhiều method không liên quan

## Kết luận

ISP khuyến khích **interface nhỏ, rõ ràng, một trách nhiệm**. Giống như SRP nhưng dành cho interface. Dễ nhớ: "Phân nhỏ ra". Nếu interface của bạn có chữ "and" trong tên, hãy tách nó.
