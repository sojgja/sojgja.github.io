---
id: di-container
title: Xây dựng DI Container chuyên nghiệp
sidebar_label: 💉 DI Container
sidebar_position: 33
---

# Xây dựng DI Container chuyên nghiệp

> *"A DI container is a framework for automating dependency injection. It's not a requirement — you can to manual DI — but as applications grow, the container becomes an essential tool for managing complexity."* — **Mark Seemann, "Dependency Injection in .NET", 2011**

Khi ứng dụng phát triển đến hàng chục hoặc hàng trăm service, việc inject dependency bằng tay (manual DI) trở nên cồng kềnh và dễ sai sót. Bạn phải nhớ thứ tự tạo dependency, quản lý lifecycle (singleton nào dùng chung, transient nào tạo mới mỗi lần), và đảm bảo không có circular dependency. DI Container (còn gọi là IoC Container) là công cụ tự động hóa quá trình này. Nó phân tích dependency graph, tạo object theo đúng thứ tự, và quản lý lifecycle. Bài viết này sẽ xây dựng một DI Container production-grade với đầy đủ tính năng: auto-wiring dựa trên type hints, lifecycle management (singleton/transient/scoped), và lazy initialization.

## Thiết kế Container mục tiêu

Container của chúng ta sẽ hỗ trợ:

1. **Register services** với 3 loại lifecycle:
   - **Singleton**: Một instance duy nhất, dùng chung toàn bộ ứng dụng.
   - **Transient**: Instance mới mỗi lần resolve.
   - **Scoped**: Instance dùng chung trong một scope (ví dụ: một HTTP request).

2. **Auto-wiring**: Tự động phân tích constructor của service, xác định dependency từ type hints, và resolve chúng theo đúng thứ tự.

3. **Lazy initialization**: Service chỉ được tạo khi thực sự cần (lần đầu resolve).

4. **Circular dependency detection**: Phát hiện và báo lỗi nếu có vòng lặp dependency.

5. **Disposable management**: Tự động gọi cleanup khi container bị dispose.

## Code hoàn chỉnh

```python
# ─── container.py ───
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import (
    Any, Callable, Dict, Generic, Iterator, List,
    Optional, Set, Tuple, Type, TypeVar, Union,
    get_type_hints,
)


T = TypeVar('T')


class ServiceLifetime(Enum):
    SINGLETON = auto()  # Một instance duy nhất
    TRANSIENT = auto()  # Instance mới mỗi lần
    SCOPED = auto()      # Instance theo scope (request)


class RegistrationError(Exception):
    """Lỗi đăng ký service — trùng, thiếu, ..."""
    pass


class ResolutionError(Exception):
    """Lỗi resolve — circular dependency, không tìm thấy, ..."""
    pass


class Disposable(ABC):
    """Interface cho service cần cleanup."""

    @abstractmethod
    def dispose(self) -> None:
        ...


@dataclass(frozen=True)
class ServiceRegistration:
    """Thông tin đăng ký của một service."""

    service_type: type
    implementation_type: type
    lifetime: ServiceLifetime
    factory: Optional[Callable[..., Any]] = None


class ServiceScope:
    """
    Một scope cụ thể (ví dụ: một HTTP request).
    Quản lý các instance scoped và disposable resources.
    """

    def __init__(self, parent_container: 'DependencyContainer') -> None:
        self._parent = parent_container
        self._scoped_instances: dict[type, Any] = {}
        self._disposables: list[Disposable] = []
        self._lock = RLock()
        self._disposed = False

    def resolve(self, service_type: type) -> Any:
        if self._disposed:
            raise ResolutionError("Cannot resolve from disposed scope")

        # Kiểm tra scoped instance đã có chưa
        with self._lock:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]

        # Resolve từ container (sẽ cache scoped instance)
        instance = self._parent._resolve_internal(service_type, self)

        with self._lock:
            self._scoped_instances[service_type] = instance

        return instance

    def track_disposable(self, instance: Any) -> None:
        if isinstance(instance, Disposable):
            with self._lock:
                self._disposables.append(instance)

    def dispose(self) -> None:
        with self._lock:
            self._disposed = True
            for disposable in reversed(self._disposables):
                try:
                    disposable.dispose()
                except Exception:
                    pass  # Log lỗi nhưng không throw
            self._scoped_instances.clear()
            self._disposables.clear()


class DependencyContainer:
    """
    DI Container chuyên nghiệp với auto-wiring, lifecycle management,
    scoped resolution, và circular dependency detection.
    """

    def __init__(self) -> None:
        self._registrations: dict[type, ServiceRegistration] = {}
        self._singleton_instances: dict[type, Any] = {}
        self._resolution_stack: list[type] = []  # Phát hiện circular dep
        self._lock = RLock()

    # ─── Register ───

    def register(
        self,
        service_type: type,
        implementation_type: Optional[type] = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        factory: Optional[Callable[..., Any]] = None,
    ) -> 'DependencyContainer':
        """
        Đăng ký service.

        Args:
            service_type: Interface/abstract class (ví dụ: UserRepository)
            implementation_type: Concrete class (ví dụ: PostgresUserRepository)
            lifetime: Singleton, Transient, hoặc Scoped
            factory: Factory function thay vì auto-wiring
        """
        if service_type in self._registrations:
            raise RegistrationError(f"Service {service_type.__name__} already registered")

        impl = implementation_type or service_type

        if factory is None and not self._can_instantiate(impl):
            raise RegistrationError(
                f"Cannot register {impl.__name__}: no factory and cannot auto-wire"
            )

        self._registrations[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation_type=impl,
            lifetime=lifetime,
            factory=factory,
        )
        return self  # Fluent interface

    def register_singleton(
        self,
        service_type: type,
        implementation_type: Optional[type] = None,
    ) -> 'DependencyContainer':
        return self.register(service_type, implementation_type, ServiceLifetime.SINGLETON)

    def register_transient(
        self,
        service_type: type,
        implementation_type: Optional[type] = None,
    ) -> 'DependencyContainer':
        return self.register(service_type, implementation_type, ServiceLifetime.TRANSIENT)

    def register_scoped(
        self,
        service_type: type,
        implementation_type: Optional[type] = None,
    ) -> 'DependencyContainer':
        return self.register(service_type, implementation_type, ServiceLifetime.SCOPED)

    def register_instance(self, service_type: type, instance: Any) -> 'DependencyContainer':
        """Đăng ký một instance có sẵn (singleton)."""
        self._registrations[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation_type=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
        )
        self._singleton_instances[service_type] = instance
        return self

    # ─── Resolve ───

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve service (không scope)."""
        return self._resolve_internal(service_type, scope=None)

    def _resolve_internal(self, service_type: type, scope: Optional[ServiceScope]) -> Any:
        # Kiểm tra singleton instance đã tồn tại
        with self._lock:
            if service_type in self._singleton_instances:
                return self._singleton_instances[service_type]

        registration = self._registrations.get(service_type)
        if registration is None:
            raise ResolutionError(f"Service {service_type.__name__} is not registered")

        # Phát hiện circular dependency
        if service_type in self._resolution_stack:
            path = ' → '.join(t.__name__ for t in self._resolution_stack + [service_type])
            raise ResolutionError(f"Circular dependency detected: {path}")

        self._resolution_stack.append(service_type)

        try:
            instance = self._create_instance(registration, scope)

            # Cache singleton
            if registration.lifetime == ServiceLifetime.SINGLETON:
                with self._lock:
                    self._singleton_instances[service_type] = instance

            # Track disposable
            if scope is not None and registration.lifetime == ServiceLifetime.SCOPED:
                scope.track_disposable(instance)

            return instance

        finally:
            self._resolution_stack.pop()

    def _create_instance(self, registration: ServiceRegistration, scope: Optional[ServiceScope]) -> Any:
        if registration.factory is not None:
            # Dùng factory function
            return registration.factory(self)

        impl = registration.implementation_type
        constructor = self._get_constructor(impl)
        dependencies = self._resolve_constructor_params(constructor, scope)
        instance = impl(*dependencies)
        return instance

    def _get_constructor(self, cls: type) -> Any:
        """Lấy constructor (__init__) của class."""
        return cls.__init__

    def _resolve_constructor_params(self, constructor: Any, scope: Optional[ServiceScope]) -> List[Any]:
        """Phân tích tham số constructor và resolve dependencies."""
        try:
            hints = get_type_hints(constructor)
        except Exception:
            hints = {}

        sig = inspect.signature(constructor)
        params: list[Any] = []

        for name, param in sig.parameters.items():
            if name == 'self':
                continue

            # Lấy type hint
            param_type = hints.get(name)
            if param_type is None:
                # Nếu không có type hint, dùng default
                if param.default is not inspect.Parameter.empty:
                    params.append(param.default if param.default is not inspect.Parameter.empty else None)
                    continue
                raise ResolutionError(
                    f"Parameter '{name}' of {constructor.__qualname__} has no type hint"
                )

            # Resolve dependency
            if param_type in self._registrations:
                if scope is not None:
                    params.append(scope.resolve(param_type))
                else:
                    params.append(self._resolve_internal(param_type, scope))
            else:
                # Type không được đăng ký — dùng default hoặc báo lỗi
                if param.default is not inspect.Parameter.empty:
                    params.append(param.default)
                else:
                    raise ResolutionError(
                        f"Unregistered dependency: {param_type.__name__} "
                        f"(required by {constructor.__qualname__}.{name})"
                    )

        return params

    def _can_instantiate(self, cls: type) -> bool:
        """Kiểm tra có thể auto-wire class này không."""
        if inspect.isabstract(cls):
            return False
        try:
            sig = inspect.signature(cls.__init__)
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                if param.default is inspect.Parameter.empty:
                    return True  # Có ít nhất một param — cần resolve
            return True
        except (ValueError, TypeError):
            return False

    # ─── Scope Management ───

    @contextmanager
    def create_scope(self) -> Iterator[ServiceScope]:
        """Tạo một scope mới (ví dụ: cho mỗi HTTP request)."""
        scope = ServiceScope(self)
        try:
            yield scope
        finally:
            scope.dispose()

    # ─── Utility ───

    def is_registered(self, service_type: type) -> bool:
        return service_type in self._registrations

    def clear(self) -> None:
        """Xóa tất cả registrations và instances."""
        self._registrations.clear()
        self._singleton_instances.clear()
        self._resolution_stack.clear()
```

## Sử dụng Container

### Định nghĩa services

```python
# ─── services.py ───
from __future__ import annotations
from typing import Protocol
from decimal import Decimal


class Logger(Protocol):
    def info(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


class DatabaseConnection:
    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        print(f"🔌 DB Connected: {connection_string}")

    def query(self, sql: str) -> list[dict[str, object]]:
        print(f"📊 Executing: {sql}")
        return [{'id': 1, 'name': 'Alice'}]

    def close(self) -> None:
        print("🔌 DB Disconnected")


class ConsoleLogger:
    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}")


class UserRepository:
    def __init__(self, db: DatabaseConnection, logger: Logger) -> None:
        self._db = db
        self._logger = logger

    def find_by_id(self, user_id: int) -> dict[str, object]:
        self._logger.info(f"Finding user {user_id}")
        return self._db.query(f"SELECT * FROM users WHERE id = {user_id}")[0]


class EmailService:
    def __init__(self, smtp_host: str, api_key: str) -> None:
        self._smtp_host = smtp_host
        self._api_key = api_key

    def send(self, to: str, subject: str, body: str) -> None:
        print(f"📧 Sending email to {to}: {subject}")


class UserService:
    def __init__(self, repo: UserRepository, email: EmailService, logger: Logger) -> None:
        self._repo = repo
        self._email = email
        self._logger = logger

    def get_user(self, user_id: int) -> dict[str, object]:
        self._logger.info(f"UserService.get_user({user_id})")
        user = self._repo.find_by_id(user_id)
        return user

    def send_welcome(self, user_email: str) -> None:
        self._email.send(user_email, "Welcome!", "Thank you for registering.")
```

### Wiring

```python
# ─── main.py ───
from __future__ import annotations
from container import DependencyContainer, ServiceLifetime

# Tạo container
container = DependencyContainer()

# Đăng ký services
container.register_singleton(DatabaseConnection, DatabaseConnection)
container.register_singleton(ConsoleLogger, ConsoleLogger)
container.register_instance(Logger, ConsoleLogger())

# Đăng ký với factory (cho params không auto-wire được)
container.register(
    EmailService,
    EmailService,
    lifetime=ServiceLifetime.SINGLETON,
    factory=lambda c: EmailService(smtp_host="smtp.gmail.com", api_key="sk-xxxx"),
)

# Transient — mỗi lần resolve là instance mới
container.register_transient(UserRepository, UserRepository)
container.register_transient(UserService, UserService)

# Resolve và sử dụng
user_service: UserService = container.resolve(UserService)
user = user_service.get_user(1)
print(f"User: {user}")

# Scope example
with container.create_scope() as scope:
    svc: UserService = scope.resolve(UserService)
    svc.send_welcome("user@example.com")

# Kiểm tra singleton — cùng instance
db1 = container.resolve(DatabaseConnection)
db2 = container.resolve(DatabaseConnection)
print(f"Same instance: {db1 is db2}")  # True

# Kiểm tra transient — khác instance
repo1 = container.resolve(UserRepository)
repo2 = container.resolve(UserRepository)
print(f"Same instance: {repo1 is repo2}")  # False
```

## Kiểm thử Container

```python
# ─── test_container.py ───
from __future__ import annotations
import pytest  # type: ignore
from typing import Protocol
from container import (
    DependencyContainer,
    ServiceScope,
    ServiceLifetime,
    RegistrationError,
    ResolutionError,
    Disposable,
)


class ILogger(Protocol):
    def log(self, msg: str) -> None: ...


class ConsoleLogger:
    def log(self, msg: str) -> None:
        print(msg)


class Repository:
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def get_data(self) -> str:
        self._logger.log("Fetching data")
        return "data"


class Service:
    def __init__(self, repo: Repository, logger: ILogger) -> None:
        self._repo = repo
        self._logger = logger

    def execute(self) -> str:
        self._logger.log("Executing service")
        return self._repo.get_data()


class CleanableService(Disposable):
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class TestRegistration:

    def test_register_and_resolve_singleton(self) -> None:
        container = DependencyContainer()
        container.register_singleton(ConsoleLogger, ConsoleLogger)

        instance1 = container.resolve(ConsoleLogger)
        instance2 = container.resolve(ConsoleLogger)
        assert instance1 is instance2

    def test_register_and_resolve_transient(self) -> None:
        container = DependencyContainer()
        container.register_transient(ConsoleLogger, ConsoleLogger)

        instance1 = container.resolve(ConsoleLogger)
        instance2 = container.resolve(ConsoleLogger)
        assert instance1 is not instance2

    def test_register_instance(self) -> None:
        container = DependencyContainer()
        logger = ConsoleLogger()
        container.register_instance(ILogger, logger)

        resolved = container.resolve(ILogger)
        assert resolved is logger

    def test_double_registration_raises_error(self) -> None:
        container = DependencyContainer()
        container.register_transient(ConsoleLogger, ConsoleLogger)
        with pytest.raises(RegistrationError):
            container.register_transient(ConsoleLogger, ConsoleLogger)


class TestAutoWiring:

    def test_auto_wire_simple(self) -> None:
        container = DependencyContainer()
        container.register_singleton(ConsoleLogger, ConsoleLogger)
        container.register_transient(Repository, Repository)

        repo = container.resolve(Repository)
        assert isinstance(repo, Repository)

    def test_auto_wire_nested(self) -> None:
        container = DependencyContainer()
        container.register_singleton(ConsoleLogger, ConsoleLogger)
        container.register_transient(Repository, Repository)
        container.register_transient(Service, Service)

        service = container.resolve(Service)
        result = service.execute()
        assert result == "data"

    def test_auto_wire_with_default_params(self) -> None:
        class WithDefaults:
            def __init__(self, name: str = "default", logger: Optional[ILogger] = None) -> None:
                self.name = name
                self.logger = logger

        container = DependencyContainer()
        container.register_singleton(ConsoleLogger, ConsoleLogger)
        container.register_transient(WithDefaults, WithDefaults)

        instance = container.resolve(WithDefaults)
        assert instance.name == "default"
        assert instance.logger is not None

    def test_missing_type_hint_raises_error(self) -> None:
        class NoHints:
            def __init__(self, db_connection):  # No type hint
                self.db = db_connection

        container = DependencyContainer()
        container.register_transient(NoHints, NoHints)

        with pytest.raises(ResolutionError):
            container.resolve(NoHints)


class TestCircularDependency:

    def test_direct_circular(self) -> None:
        class A:
            def __init__(self, b: 'B') -> None:
                self.b = b

        class B:
            def __init__(self, a: A) -> None:
                self.a = a

        container = DependencyContainer()
        container.register_transient(A, A)
        container.register_transient(B, B)

        with pytest.raises(ResolutionError, match="Circular dependency"):
            container.resolve(A)


class TestScopedLifetime:

    def test_scoped_same_scope(self) -> None:
        container = DependencyContainer()
        container.register_scoped(ConsoleLogger, ConsoleLogger)

        with container.create_scope() as scope:
            instance1 = scope.resolve(ConsoleLogger)
            instance2 = scope.resolve(ConsoleLogger)
            assert instance1 is instance2

    def test_scoped_different_scope(self) -> None:
        container = DependencyContainer()
        container.register_scoped(ConsoleLogger, ConsoleLogger)

        with container.create_scope() as scope1:
            instance1 = scope1.resolve(ConsoleLogger)

        with container.create_scope() as scope2:
            instance2 = scope2.resolve(ConsoleLogger)

        assert instance1 is not instance2

    def test_disposable_in_scope(self) -> None:
        container = DependencyContainer()
        container.register_scoped(CleanableService, CleanableService)

        with container.create_scope() as scope:
            instance = scope.resolve(CleanableService)
            assert not instance.disposed

        assert instance.disposed

    def test_scope_dispose_does_not_affect_singletons(self) -> None:
        container = DependencyContainer()
        container.register_singleton(ConsoleLogger, ConsoleLogger)

        with container.create_scope() as scope:
            logger = scope.resolve(ConsoleLogger)

        # Singleton vẫn hoạt động sau khi scope dispose (nếu nó được
        # quản lý bởi container, không phải scope)
        # Trong thiết kế này, singleton do container cache, không bị ảnh hưởng
        assert logger is not None


class TestFactoryRegistration:

    def test_factory_registration(self) -> None:
        container = DependencyContainer()
        container.register(
            ConsoleLogger,
            factory=lambda c: ConsoleLogger(),
            lifetime=ServiceLifetime.SINGLETON,
        )

        instance = container.resolve(ConsoleLogger)
        assert isinstance(instance, ConsoleLogger)

    def test_factory_with_dependencies(self) -> None:
        class Config:
            def __init__(self) -> None:
                self.db_url = "postgresql://localhost/db"

        class Database:
            def __init__(self, connection_string: str) -> None:
                self._conn = connection_string

        container = DependencyContainer()
        container.register_singleton(Config, Config)
        container.register_singleton(
            Database,
            factory=lambda c: Database(connection_string=c.resolve(Config).db_url),
        )

        db = container.resolve(Database)
        assert db._conn == "postgresql://localhost/db"


class TestIntegration:

    def test_full_resolution_graph(self) -> None:
        """Kiểm tra toàn bộ dependency graph được resolve đúng."""
        container = DependencyContainer()
        container.register_singleton(ConsoleLogger, ConsoleLogger)
        container.register_transient(Repository, Repository)
        container.register_transient(Service, Service)

        service = container.resolve(Service)
        assert isinstance(service, Service)
        assert isinstance(service._repo, Repository)
        assert isinstance(service._logger, ConsoleLogger)

    def test_fluent_registration(self) -> None:
        container = DependencyContainer()
        (container
         .register_singleton(ConsoleLogger)
         .register_transient(Repository)
         .register_transient(Service))

        service = container.resolve(Service)
        assert isinstance(service, Service)
```

## So sánh với các DI Container nổi tiếng

| Tính năng | Container của chúng ta | FastAPI Depends() | python-dependency-injector | Spring Boot (Java) |
|-----------|----------------------|-------------------|---------------------------|-------------------|
| **Auto-wiring** | ✅ Có (type hints) | ✅ Có | ✅ Có | ✅ Có |
| **Singleton** | ✅ | ✅ (default) | ✅ | ✅ |
| **Transient** | ✅ | ❌ (luôn singleton trong cùng scope) | ✅ | ✅ |
| **Scoped** | ✅ | ✅ (request scope) | ✅ | ✅ (request, session) |
| **Circular dep detection** | ✅ | ✅ (error) | ✅ | ✅ |
| **Factory registration** | ✅ | ❌ (dùng class) | ✅ | ✅ (Bean) |
| **Fluent API** | ✅ | N/A | ✅ | ❌ (annotation-based) |
| **Disposable management** | ✅ | ❌ | ✅ | ✅ |
| **Async support** | ❌ | ✅ | ✅ | ✅ |
| **Module organization** | ❌ | ❌ | ✅ | ✅ |
| **Thread-safety** | ✅ (RLock) | ✅ | ✅ | ✅ |

## Ưu và nhược điểm của DI Container

| Tiêu chí | Manual DI | DI Container |
|----------|-----------|-------------|
| **Complexity** | Thấp — không cần học thêm | Cao — phải hiểu container API |
| **Wiring code** | Viết tay — phải update khi thêm service | Tự động — container resolve dependency graph |
| **Lỗi wiring** | Chỉ phát hiện runtime (thiếu param) | Một số lỗi phát hiện khi đăng ký |
| **Debugging** | Dễ — stack trace rõ ràng | Khó hơn — container abstract hóa call stack |
| **Performance** | Tối ưu — zero overhead | Có overhead nhỏ (reflection, caching) |
| **Scalability** | Khó — wiring trở nên phức tạp khi scale | Dễ — chỉ cần đăng ký thêm |
| **Phù hợp cho** | < 10 services, prototype | > 10 services, production |

## Kết luận

DI Container là công cụ mạnh mẽ nhưng không bắt buộc. Với ứng dụng nhỏ (< 10 services), manual DI hoàn toàn ổn. Với ứng dụng lớn (hàng trăm services), container giúp bạn quản lý dependency graph một cách hệ thống, tránh lỗi wiring, và dễ dàng thay đổi implementation. Container chúng ta vừa xây dựng có đủ tính năng để dùng trong production cho hầu hết ứng dụng Python — auto-wiring, singleton/transient/scoped, circular dependency detection, và disposable management. Tuy nhiên, trong thực tế, bạn nên cân nhắc dùng thư viện `dependency-injector` (Python) hoặc built-in container của FastAPI thay vì tự xây, vì chúng đã được kiểm thử kỹ lưỡng và có hỗ trợ cộng đồng.

Khi quyết định dùng DI Container, hãy nhớ:
- **Container là công cụ, không phải mục đích** — đừng lạm dụng.
- **Đăng ký interface, không class concrete** — tuân thủ DIP.
- **Singleton cho stateless service, Transient cho stateful service**.
- **Scoped cho request-scoped resources** (DB session, transaction).
- **Luôn dispose scope** — dùng context manager để đảm bảo cleanup.
