---
id: di-realworld
title: DI trong ứng dụng thực tế (FastAPI + MySQL)
sidebar_label: 💉 DI thực tế
sidebar_position: 34
---

# DI trong ứng dụng thực tế

Ứng dụng **User Management** với FastAPI, MySQL, Redis Cache, Email Service, Logger — dùng DI để kết nối mọi thứ.

## Cấu trúc project

```
app/
├── container.py       # DI Container
├── main.py           # FastAPI app
├── core/
│   ├── database.py   # Database connection
│   ├── cache.py      # Redis cache
│   └── logger.py     # Logger
├── repositories/
│   └── user_repo.py  # UserRepository
├── services/
│   └── user_service.py # UserService
└── email/
    └── email_service.py # EmailService
```

## Định nghĩa từng module

```python
# core/logger.py
class Logger:
    def info(self, msg): print(f'[INFO] {msg}')
    def error(self, msg): print(f'[ERROR] {msg}')
```

```python
# core/database.py
class Database:
    def __init__(self, config: dict):
        self.config = config
        self.connection = None

    def connect(self):
        print(f'🔌 Kết nối MySQL: {self.config["host"]}:{self.config["port"]}')
        self.connection = {'connected': True}

    def query(self, sql: str, params=None):
        print(f'📊 Query: {sql}')
        return [{'id': 1, 'name': 'Alice'}]
```

```python
# core/cache.py
class Cache:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def get(self, key: str):
        print(f'🔍 Redis GET {key}')
        return None

    def set(self, key: str, value, ttl: int = 300):
        print(f'💾 Redis SET {key} (TTL: {ttl}s)')
```

```python
# repositories/user_repo.py
class UserRepository:
    def __init__(self, db: Database, cache: Cache, logger: Logger):
        self.db = db
        self.cache = cache
        self.logger = logger

    def find_by_id(self, user_id: int):
        # Check cache first
        cached = self.cache.get(f'user:{user_id}')
        if cached:
            return cached

        # Query database
        user = self.db.query('SELECT * FROM users WHERE id = ?', [user_id])
        self.cache.set(f'user:{user_id}', user)
        self.logger.info(f'Fetched user {user_id}')
        return user
```

```python
# email/email_service.py
class EmailService:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send(self, to: str, subject: str, body: str):
        print(f'📧 Gửi email: {subject} → {to}')
        return True
```

```python
# services/user_service.py
class UserService:
    def __init__(
        self,
        repo: UserRepository,
        email: EmailService,
        logger: Logger,
    ):
        self.repo = repo
        self.email = email
        self.logger = logger

    def get_user(self, user_id: int):
        self.logger.info(f'Getting user {user_id}')
        return self.repo.find_by_id(user_id)

    def send_welcome(self, user_email: str):
        self.email.send(user_email, 'Chào mừng!', 'Cảm ơn bạn đã đăng ký.')
```

## DI Container

```python
# container.py
from core.database import Database
from core.cache import Cache
from core.logger import Logger
from email.email_service import EmailService
from repositories.user_repo import UserRepository
from services.user_service import UserService

class Container:
    def __init__(self):
        # Singleton — một instance dùng chung
        self.logger = Logger()

        self.db = Database({'host': 'localhost', 'port': 3306, 'db': 'users'})
        self.db.connect()

        self.cache = Cache('localhost', 6379)

        self.email = EmailService(api_key='sk-123456')

        # Transient — tạo mới mỗi lần (có thể factory)
        self.user_repo = UserRepository(self.db, self.cache, self.logger)
        self.user_service = UserService(self.user_repo, self.email, self.logger)

container = Container()
```

## FastAPI app

```python
# main.py
from fastapi import FastAPI, Depends
from container import container
from services.user_service import UserService

app = FastAPI()

def get_user_service() -> UserService:
    return container.user_service

@app.get('/users/{user_id}')
def get_user(user_id: int, service: UserService = Depends(get_user_service)):
    user = service.get_user(user_id)
    return {'data': user}

@app.post('/users/{user_id}/welcome')
def send_welcome(user_email: str, service: UserService = Depends(get_user_service)):
    service.send_welcome(user_email)
    return {'message': 'Email đã gửi'}
```

## Test với mock

```python
def test_get_user():
    # Mock dependencies
    mock_logger = MagicMock()
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_db = MagicMock()
    mock_db.query.return_value = [{'id': 1, 'name': 'Alice'}]

    repo = UserRepository(mock_db, mock_cache, mock_logger)
    email = MagicMock()
    service = UserService(repo, email, mock_logger)

    result = service.get_user(1)
    assert result[0]['name'] == 'Alice'
    mock_db.query.assert_called_once()
```

## Kết luận

Trong ứng dụng thực tế:
- Dùng **DI** để tách creation khỏi business logic
- Dùng **DI Container** để quản lý dependency tập trung
- Dùng **Singleton** cho connection, logger, cache
- Dùng **Transient** cho service, repository (nếu stateless)
- Dùng **Mock** trong test để không phụ thuộc vào infrastructure
