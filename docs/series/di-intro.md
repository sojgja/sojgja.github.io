---
id: di-intro
title: Dependency Injection — Giới thiệu
sidebar_label: 💉 Giới thiệu DI
sidebar_position: 31
---

# Dependency Injection — Giới thiệu

> *"Dependency Injection is where the dependencies are pushed into the object from the outside, rather than the object pulling them in from the inside. It is about inverting the control of the dependencies."* — **Martin Fowler, "Inversion of Control Containers and the Dependency Injection Pattern", 2004**

Dependency Injection (DI) là một kỹ thuật thiết kế phần mềm trong đó các dependency (phụ thuộc) của một object được cung cấp từ bên ngoài thay vì object đó tự tạo ra chúng. Martin Fowler — người đặt ra thuật ngữ này năm 2004 — đã nhấn mạnh rằng DI là một hình thức cụ thể của Inversion of Control (IoC). Nếu IoC là nguyên lý tổng quát "framework gọi code của bạn", thì DI là cách cụ thể để implement IoC trong việc quản lý dependency. DI không phải là một thư viện, framework, hay pattern mới — nó là một kỹ thuật đơn giản với tác động sâu rộng đến kiến trúc phần mềm. Trước khi Fowler viết bài nổi tiếng về DI, các lập trình viên Java đã sử dụng nó dưới dạng "parameter injection" và "factory pattern" từ thập niên 1990. Nhưng chính bài viết của Fowler, cùng với sự ra đời của PicoContainer và Spring Framework, đã đưa DI trở thành kỹ thuật standard trong enterprise software development.

## Lịch sử và nguồn gốc của Dependency Injection

DI không xuất hiện đột ngột. Nó là kết quả của một quá trình tiến hóa trong tư duy thiết kế phần mềm:

- **Thập niên 70-80 (Mainframe era)**: Object tự quản lý mọi thứ — tạo database connection, đọc config file, ghi log. Dependency là trách nhiệm của object. Code đơn giản nhưng khó bảo trì.
- **Thập niên 90 (Three-tier architecture)**: Business logic, presentation, và data access bắt đầu được tách riêng. Tuy nhiên, tầng business vẫn thường khởi tạo trực tiếp tầng data access — dependency vẫn bị hardcode.
- **1996 — Inversion of Control Containers**: Các framework IoC đầu tiên xuất hiện (PicoContainer, Avalon). Khái niệm "container quản lý lifecycle của component" bắt đầu hình thành.
- **2000 — J2EE và EJB2**: EJB 2.0 sử dụng Service Locator pattern — các service được tra cứu từ global registry. Đây là tiền thân của DI nhưng có nhiều hạn chế (phụ thuộc vào container, global state, khó test).
- **2004 — Martin Fowler's Article**: Fowler viết bài "Inversion of Control Containers and the Dependency Injection Pattern", phân tích ưu nhược điểm của Service Locator vs DI và định nghĩa DI thành 3 hình thức: Constructor Injection, Setter Injection, và Interface Injection.
- **2004 — Spring Framework**: Spring 1.0 ra mắt với DI container mạnh mẽ, XML-based configuration. DI bùng nổ trong thế giới Java.
- **2005-2010 — Python DI**: Python cộng đồng bắt đầu áp dụng DI qua các thư viện như Zope Component Architecture, Spring Python.
- **2018 — FastAPI**: Fastapi ra mắt với built-in DI qua `Depends()`, làm cho DI trở nên phổ biến trong Python community.

## Dependency Injection vs Service Locator — So sánh chi tiết

Service Locator là một pattern khác implement IoC, nhưng khác DI ở chỗ dependency được "tra cứu" từ một registry trung tâm (locator) thay vì được "inject" vào object. Hai cách tiếp cận này có sự khác biệt sâu sắc:

| Tiêu chí | Dependency Injection | Service Locator |
|----------|---------------------|-----------------|
| **Cách lấy dependency** | Inject từ bên ngoài | Tra cứu từ registry global |
| **Tính tường minh** | Rõ ràng — khai báo trong constructor | Ẩn — gọi `locator.get()` bên trong method |
| **Dependency graph** | Nhìn vào constructor biết ngay | Phải đọc toàn bộ class mới biết |
| **Testability** | Tuyệt vời — inject mock qua constructor | Khó hơn — phải mock locator, locator có global state |
| **Global state** | Không — không có global registry | Có — locator thường là global/singleton |
| **Compile-time safety** | Có thể kiểm tra type | Không — lỗi chỉ phát hiện runtime |
| **Phụ thuộc vào DI framework** | Thấp — có thể dùng manual DI | Cao — locator thường là framework |
| **Khả năng kiểm soát scope** | Dễ — container quản lý lifecycle | Khó — locator trả về instance, scope khó quản lý |

Trong thực tế, Service Locator có thể chấp nhận được cho các framework / library (nơi bạn không muốn người dùng phải inject thủ công), nhưng DI luôn được ưu tiên cho application code vì tính tường minh và testability.

## Tại sao DI lại quan trọng?

Một số lập trình viên Python có thể nghĩ: "Python đã linh hoạt rồi, cần gì DI?" Câu trả lời là DI không phải là về ngôn ngữ — nó là **về kiến trúc**. DI giúp:

1. **Testability (Khả năng kiểm thử)**: Class nhận dependency từ bên ngoài — bạn có thể inject mock trong test, real implementation trong production. Không cần infrastructure thật để chạy unit test.

2. **Flexibility (Linh hoạt)**: Có thể thay đổi implementation mà không sửa code. Muốn chuyển từ PostgreSQL sang MongoDB? Tạo `MongoRepository` và inject nó — không cần sửa class đang dùng.

3. **Decoupling (Giảm kết nối)**: Class không cần biết cách tạo dependency. Nó chỉ cần biết cách *dùng* dependency (qua abstraction). Giảm knowledge của class xuống mức tối thiểu.

4. **Separation of Concerns**: Khởi tạo object (wiring) tách rời khỏi business logic. Một class chỉ tập trung vào việc gì nó làm, không phải cần gì để làm.

5. **Reusability (Tái sử dụng)**: Vì không bị ràng buộc vào implementation cụ thể, class có thể dùng lại trong nhiều context khác nhau.

6. **Configuration management**: Dễ dàng có config khác nhau cho development, staging, production — inject implementation khác nhau qua config hoặc DI container.

## 3 hình thức Dependency Injection

### 1. Constructor Injection (Ưu tiên số 1)

```python
from __future__ import annotations
from typing import Protocol

class UserRepository(Protocol):
    def save(self, user: User) -> None: ...

class UserService:
    def __init__(self, repo: UserRepository) -> None:  # Injection qua constructor
        self._repo = repo

    def register(self, user: User) -> None:
        self._repo.save(user)
```

Dependency được set một lần, bất biến trong suốt vòng đời của object. Đây là hình thức được khuyến khích nhất vì nó rõ ràng, đảm bảo object luôn có đủ dependency, và immutable (giảm bug).

### 2. Setter Injection

```python
class UserService:
    def __init__(self) -> None:
        self._repo = None

    def set_repository(self, repo: UserRepository) -> None:  # Injection qua setter
        self._repo = repo
```

Cho phép thay đổi dependency sau khi khởi tạo — hữu ích cho dependency không bắt buộc (optional) hoặc cần thay đổi runtime.

### 3. Method Injection (Parameter Injection)

```python
class UserService:
    def register(self, user: User, repo: UserRepository) -> None:  # Injection qua tham số
        repo.save(user)
```

Dependency chỉ cần trong method cụ thể, không cần lưu trữ.

## DI Container — Khi manual DI không đủ

Khi ứng dụng có vài service, manual DI hoàn toàn ổn:

```python
db = Database(config)
cache = RedisCache(config)
repo = UserRepository(db, cache)
service = UserService(repo, email_svc)
```

Khi có 100+ service, manual wiring trở thành nightmare. **DI Container** ra đời để giải quyết vấn đề này: nó tự động phân tích dependency graph và tạo object theo đúng thứ tự, đồng thời quản lý lifecycle (singleton, transient, scoped). DI Container không phải là bắt buộc — bạn có thể sống tốt mà không cần nó ở giai đoạn đầu — nhưng nó trở nên cần thiết khi hệ thống phát triển.

## DI trong các framework thực tế

- **FastAPI**: DI built-in với `Depends()`. Hỗ trợ async, có thể dùng cả function và class làm dependency. Đơn giản nhất trong các framework Python.
- **Django**: Không có DI container chính thức. Dùng manual DI hoặc thư viện `django-injector`. Django's CBV (Class-Based Views) có thể dùng `__init__` injection.
- **Flask**: `Flask-Injector` — tích hợp Flask với injector library.
- **Spring Boot (Java)**: DI container nổi tiếng nhất — `@Autowired`, `@Component`, `@Bean`, hỗ trợ prototype, singleton, request, session scopes.
- **ASP.NET Core**: Built-in DI container mạnh mẽ, hỗ trợ đầy đủ scopes.
- **dependency-injector (Python)**: Thư viện DI chuyên dụng, hỗ trợ auto-wiring, scopes, configuration, và tích hợp với nhiều frameworks.

## Khi nào KHÔNG nên dùng DI?

DI không phải là giải pháp cho mọi vấn đề. Có những trường hợp không nên dùng DI:

1. **Script nhỏ, một lần**: Script Python 50 dòng để xử lý file CSV — DI là over-engineering.
2. **Performance-critical code path**: Constructor injection có overhead nhỏ. Trong hot path (game engine, real-time system), DI có thể chậm.
3. **Prototype / MVP**: Khi chưa biết sản phẩm có tồn tại không, ưu tiên speed over architecture. Có thể refactor về sau.
4. **Data classes / Value objects**: `User(name, email)` — không cần inject gì cả.
5. **Dependency không thay đổi**: `datetime.now()`, `uuid.uuid4()` — không cần abstract hóa và inject.

## Kết luận

DI là một kỹ thuật kiến trúc mạnh mẽ giúp tách creation khỏi usage, tăng testability và flexibility của code. Nó không phải là "điều kiện tiên quyết" cho mọi dự án — hãy áp dụng nó khi lợi ích vượt quá chi phí (khi bạn cần test, khi infrastructure thay đổi, khi hệ thống có > 5-10 services). Năm nguyên tắc vàng khi áp dụng DI:

1. **Constructor Injection first** — luôn ưu tiên inject qua constructor.
2. **Depend on abstractions, not concretions** — kết hợp với DIP.
3. **Manual DI trước, container sau** — không cần DI Container ngay từ đầu.
4. **DI là về kiến trúc, không phải framework** — bạn có thể DI mà không cần thư viện.
5. **Không DI hóa mọi thứ** — chỉ áp dụng cho những dependency có khả năng thay đổi.
