---
id: di-container
title: Xây dựng DI Container đơn giản
sidebar_label: 💉 DI Container
sidebar_position: 33
---

# Xây dựng DI Container đơn giản

Khi ứng dụng có hàng chục service, việc inject thủ công trở nên cồng kềnh:

```python
db = MySQLDatabase()
logger = FileLogger()
email = SendGridEmailService()
user_repo = MySQLUserRepository(db, logger)
order_repo = MySQLOrderRepository(db, logger)
user_service = UserService(user_repo, email, logger)
order_service = OrderService(order_repo, email, logger)
# ...
```

**DI Container** tự động quản lý việc tạo và inject dependency.

## DI Container tự xây

```python
class Container:
    def __init__(self):
        self._services = {}
        self._instances = {}

    def register(self, name, factory, singleton=False):
        self._services[name] = {
            'factory': factory,
            'singleton': singleton,
        }

    def resolve(self, name):
        if name in self._instances:
            return self._instances[name]

        service = self._services.get(name)
        if not service:
            raise ValueError(f'Service {name} not registered')

        instance = service['factory'](self)
        if service['singleton']:
            self._instances[name] = instance
        return instance
```

## Sử dụng

```python
# Định nghĩa services
class Database:
    def query(self, sql):
        return f'Kết quả: {sql}'

class Logger:
    def log(self, msg):
        print(f'[LOG] {msg}')

class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    def get_users(self):
        return self.db.query('SELECT * FROM users')

class UserService:
    def __init__(self, repo: UserRepository, logger: Logger):
        self.repo = repo
        self.logger = logger

    def list_users(self):
        self.logger.log('Fetching users...')
        return self.repo.get_users()
```

```python
# Đăng ký vào container
container = Container()
container.register('db', lambda c: Database(), singleton=True)
container.register('logger', lambda c: Logger(), singleton=True)
container.register('repo', lambda c: UserRepository(c.resolve('db')))
container.register('user_service', lambda c: UserService(
    c.resolve('repo'),
    c.resolve('logger'),
))

# Sử dụng
service = container.resolve('user_service')
print(service.list_users())
# [LOG] Fetching users...
# Kết quả: SELECT * FROM users
```

## Singleton vs Transient

| Loại | Container trả về | Khi nào dùng |
|------|-----------------|--------------|
| **Singleton** | Cùng một instance | Database connection, Logger, Config |
| **Transient** | Instance mới mỗi lần | Repository, Service (nếu không có state) |

## Auto-wiring với type hints (nâng cao)

```python
import inspect

class AutoContainer:
    def __init__(self):
        self._services = {}

    def register(self, cls, singleton=False):
        self._services[cls] = {'singleton': singleton, 'instance': None}
        return cls

    def resolve(self, cls):
        service = self._services.get(cls)
        if not service:
            raise ValueError(f'{cls.__name__} not registered')

        if service['singleton'] and service['instance']:
            return service['instance']

        # Tự động inspect constructor và resolve dependencies
        sig = inspect.signature(cls.__init__)
        deps = []
        for name, param in sig.parameters.items():
            if name == 'self': continue
            if param.annotation is inspect.Parameter.empty:
                raise ValueError(f'{name} has no type hint')
            dep = self.resolve(param.annotation)
            deps.append(dep)

        instance = cls(*deps)
        if service['singleton']:
            service['instance'] = instance
        return instance

# Sử dụng
container = AutoContainer()
container.register(Database, singleton=True)
container.register(Logger, singleton=True)
container.register(UserRepository)
container.register(UserService)

service = container.resolve(UserService)
print(service.list_users())
```

## Khi nào dùng DI Container?

- ✅ Ứng dụng có 10+ service phụ thuộc chéo
- ✅ Cần centralized lifecycle management
- ✅ Cần hot-swap implementation (mock, staging, production)
- ❌ Không dùng cho script nhỏ (< 5 class)

## Thực tế

Các framework Python có DI Container mạnh mẽ:
- **FastAPI** — built-in DI với `Depends()`
- **Django** — không có container chính thức, dùng manual DI
- **Flask** — `Flask-Injector`
- **dependency-injector** — thư viện DI chuyên dụng cho Python
