---
id: di-realworld
title: DI trong ứng dụng thực tế (FastAPI + SQLAlchemy + Redis)
sidebar_label: 💉 DI thực tế
sidebar_position: 34
---

# DI trong ứng dụng thực tế (FastAPI + SQLAlchemy + Redis)

> *"The real power of DI is revealed in real-world applications where infrastructure changes, services grow, and testability becomes critical. Theory is good — practice is better."* — **Mark Seemann**

Bài viết này áp dụng tất cả những gì đã học về SOLID và DI để xây dựng một ứng dụng **User Management** hoàn chỉnh với FastAPI, SQLAlchemy (PostgreSQL), Redis cache, JWT authentication, email service, và background tasks. Mục tiêu: cho thấy DI hoạt động như thế nào trong một ứng dụng production-grade, với đầy đủ các tầng (layers), các dependency được inject qua constructor, và một DI Container đơn giản quản lý toàn bộ.

## Kiến trúc tổng quan

```
user-management/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── container.py               # DI Container
│   ├── config.py                  # Configuration management
│   │
│   ├── domain/                    # Business logic (core)
│   │   ├── __init__.py
│   │   ├── models.py              # Domain models
│   │   ├── interfaces.py          # Abstract interfaces (ports)
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── user_service.py    # UserService — use case
│   │   │   └── auth_service.py    # AuthService — authentication
│   │   └── exceptions.py          # Domain exceptions
│   │
│   ├── infrastructure/            # External implementations (adapters)
│   │   ├── __init__.py
│   │   ├── database.py            # SQLAlchemy engine + session
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   └── user_repository.py # PostgreSQL UserRepository
│   │   ├── cache.py               # Redis cache
│   │   ├── email.py               # Email service (SMTP / SendGrid)
│   │   ├── auth.py                # JWT token service
│   │   └── unit_of_work.py        # Unit of Work pattern
│   │
│   └── api/                       # Presentation layer (FastAPI routes)
│       ├── __init__.py
│       ├── routes/
│       │   ├── __init__.py
│       │   └── user_routes.py     # User API endpoints
│       ├── dependencies.py         # FastAPI Depends() functions
│       └── schemas.py             # Pydantic request/response models
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures (DI)
│   ├── test_user_service.py       # Unit tests
│   └── test_user_routes.py        # Integration tests
│
└── requirements.txt
```

## Domain Layer — Core Business Logic

```python
# ─── app/domain/models.py ───
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class UserRole(str, Enum):
    ADMIN = 'admin'
    USER = 'user'
    MODERATOR = 'moderator'


@dataclass(frozen=True)
class User:
    id: str
    email: str
    name: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


@dataclass(frozen=True)
class CreateUserRequest:
    email: str
    name: str
    password: str
    role: UserRole = UserRole.USER


@dataclass(frozen=True)
class UpdateUserRequest:
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


# ─── app/domain/interfaces.py ───
from __future__ import annotations
from typing import Optional, Protocol
from app.domain.models import User, CreateUserRequest, UpdateUserRequest


class UserRepository(Protocol):
    """Port cho repository — database implementation không quan trọng."""

    def save(self, user: User) -> User: ...
    def find_by_id(self, user_id: str) -> Optional[User]: ...
    def find_by_email(self, email: str) -> Optional[User]: ...
    def update(self, user_id: str, request: UpdateUserRequest) -> Optional[User]: ...
    def delete(self, user_id: str) -> bool: ...
    def list_all(self, skip: int = 0, limit: int = 100) -> list[User]: ...


class CacheService(Protocol):
    """Port cho caching — có thể là Redis, Memcached, hoặc in-memory."""

    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, ttl_seconds: int = 300) -> None: ...
    def invalidate(self, key: str) -> None: ...


class EmailService(Protocol):
    """Port cho email — SMTP, SendGrid, SES, ..."""

    def send(self, to: str, subject: str, body: str) -> bool: ...


class TokenService(Protocol):
    """Port cho JWT token."""

    def create_access_token(self, user_id: str, role: str) -> str: ...
    def verify_token(self, token: str) -> Optional[dict]: ...


class UnitOfWork(Protocol):
    """Port cho transaction management."""

    def __enter__(self) -> 'UnitOfWork': ...
    def __exit__(self, *args: object) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
```

## Domain Services — Use Cases

```python
# ─── app/domain/services/user_service.py ───
from __future__ import annotations
import uuid
from datetime import datetime
from app.domain.models import User, CreateUserRequest, UpdateUserRequest, UserRole
from app.domain.interfaces import UserRepository, CacheService, EmailService, UnitOfWork
from app.domain.exceptions import UserNotFoundError, DuplicateEmailError, InvalidOperationError


class UserService:
    """
    Xử lý các use case liên quan đến User.
    Business logic thuần túy — không biết gì về database, cache, hay framework.
    """

    def __init__(
        self,
        repository: UserRepository,
        cache: CacheService,
        email: EmailService,
        uow: UnitOfWork,
    ) -> None:
        self._repo = repository
        self._cache = cache
        self._email = email
        self._uow = uow

    def create_user(self, request: CreateUserRequest) -> User:
        # Validate business rules
        existing = self._repo.find_by_email(request.email)
        if existing is not None:
            raise DuplicateEmailError(f"Email {request.email} already exists")

        if request.role == UserRole.ADMIN:
            # Chỉ admin mới tạo được admin — logic này thuộc business domain
            raise InvalidOperationError("Cannot create admin user directly")

        user = User(
            id=str(uuid.uuid4()),
            email=request.email,
            name=request.name,
            role=request.role,
            created_at=datetime.now(),
        )

        with self._uow:
            saved = self._repo.save(user)
            self._uow.commit()

        # Gửi email chào mừng (async — có thể dispatch background task)
        self._email.send(
            to=user.email,
            subject="Chào mừng bạn đến với hệ thống!",
            body=f"Xin chào {user.name}, tài khoản của bạn đã được tạo thành công.",
        )

        # Invalidate cache
        self._cache.invalidate("users:all")

        return saved

    def get_user(self, user_id: str) -> User:
        # Check cache trước
        cache_key = f"user:{user_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            import json
            data = json.loads(cached)
            return User(**data)

        # Query database
        user = self._repo.find_by_id(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        # Cache result
        import json
        self._cache.set(cache_key, json.dumps({
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'role': user.role.value,
            'is_active': user.is_active,
        }), ttl_seconds=300)

        return user

    def update_user(self, user_id: str, request: UpdateUserRequest) -> User:
        with self._uow:
            updated = self._repo.update(user_id, request)
            if updated is None:
                raise UserNotFoundError(f"User {user_id} not found")
            self._uow.commit()

        # Invalidate cache
        self._cache.invalidate(f"user:{user_id}")
        self._cache.invalidate("users:all")

        return updated

    def delete_user(self, user_id: str) -> bool:
        with self._uow:
            result = self._repo.delete(user_id)
            if not result:
                raise UserNotFoundError(f"User {user_id} not found")
            self._uow.commit()

        self._cache.invalidate(f"user:{user_id}")
        self._cache.invalidate("users:all")
        return True

    def list_users(self, skip: int = 0, limit: int = 100) -> list[User]:
        cache_key = f"users:list:{skip}:{limit}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            import json
            data = json.loads(cached)
            return [User(**u) for u in data]

        users = self._repo.list_all(skip, limit)

        import json
        self._cache.set(cache_key, json.dumps([
            {'id': u.id, 'email': u.email, 'name': u.name, 'role': u.role.value}
            for u in users
        ]), ttl_seconds=60)

        return users
```

```python
# ─── app/domain/services/auth_service.py ───
from __future__ import annotations
from app.domain.models import User, UserRole
from app.domain.interfaces import UserRepository, TokenService
from app.domain.exceptions import AuthenticationError, UserNotFoundError


class AuthService:
    def __init__(self, repository: UserRepository, token_service: TokenService) -> None:
        self._repo = repository
        self._token_service = token_service

    def authenticate(self, email: str, password: str) -> str:
        user = self._repo.find_by_email(email)
        if user is None:
            raise AuthenticationError("Invalid email or password")

        # Trong thực tế: verify password hash
        # if not verify_password(password, user.password_hash): ...

        token = self._token_service.create_access_token(
            user_id=user.id,
            role=user.role.value,
        )
        return token

    def verify_token(self, token: str) -> dict:
        payload = self._token_service.verify_token(token)
        if payload is None:
            raise AuthenticationError("Invalid or expired token")
        return payload
```

## Infrastructure Layer — Implementations

```python
# ─── app/infrastructure/database.py ───
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class DatabaseSessionManager:
    """Quản lý SQLAlchemy engine và session factory."""

    def __init__(self, connection_string: str, echo: bool = False) -> None:
        self._engine = create_engine(connection_string, echo=echo, pool_pre_ping=True)
        self._session_factory = sessionmaker(bind=self._engine, autocommit=False, autoflush=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        from app.infrastructure.models import Base
        Base.metadata.create_all(self._engine)
```

```python
# ─── app/infrastructure/repositories/user_repository.py ───
from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session

from app.domain.models import User, CreateUserRequest, UpdateUserRequest
from app.domain.interfaces import UserRepository
from app.infrastructure.models import UserModel


class SQLAlchemyUserRepository:
    """Implement UserRepository port với SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, user: User) -> User:
        model = UserModel(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
        )
        self._session.add(model)
        return user

    def find_by_id(self, user_id: str) -> Optional[User]:
        model = self._session.query(UserModel).filter(UserModel.id == user_id).first()
        if model is None:
            return None
        return self._model_to_domain(model)

    def find_by_email(self, email: str) -> Optional[User]:
        model = self._session.query(UserModel).filter(UserModel.email == email).first()
        if model is None:
            return None
        return self._model_to_domain(model)

    def update(self, user_id: str, request: UpdateUserRequest) -> Optional[User]:
        model = self._session.query(UserModel).filter(UserModel.id == user_id).first()
        if model is None:
            return None

        if request.name is not None:
            model.name = request.name
        if request.email is not None:
            model.email = request.email
        if request.role is not None:
            model.role = request.role.value
        if request.is_active is not None:
            model.is_active = request.is_active

        self._session.flush()
        return self._model_to_domain(model)

    def delete(self, user_id: str) -> bool:
        model = self._session.query(UserModel).filter(UserModel.id == user_id).first()
        if model is None:
            return False
        self._session.delete(model)
        self._session.flush()
        return True

    def list_all(self, skip: int = 0, limit: int = 100) -> list[User]:
        models = self._session.query(UserModel).offset(skip).limit(limit).all()
        return [self._model_to_domain(m) for m in models]

    def _model_to_domain(self, model: UserModel) -> User:
        return User(
            id=model.id,
            email=model.email,
            name=model.name,
            role=model.role,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
```

```python
# ─── app/infrastructure/models.py ───
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Enum
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()


class UserModel(Base):
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default='user')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, onupdate=datetime.now)
```

```python
# ─── app/infrastructure/cache.py ───
from __future__ import annotations
from typing import Optional
import json


class RedisCacheService:
    """Implement CacheService với Redis."""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0) -> None:
        self._host = host
        self._port = port
        self._db = db
        self._client: Optional[Any] = None

    def _get_client(self):
        if self._client is None:
            import redis  # type: ignore
            self._client = redis.Redis(host=self._host, port=self._port, db=self._db)
        return self._client

    def get(self, key: str) -> Optional[str]:
        client = self._get_client()
        value = client.get(key)
        if value is not None:
            return value.decode('utf-8')
        return None

    def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        client = self._get_client()
        client.setex(key, ttl_seconds, value)

    def invalidate(self, key: str) -> None:
        client = self._get_client()
        client.delete(key)


class InMemoryCacheService:
    """Implement CacheService trong memory — dùng cho development/test."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> Optional[str]:
        import time
        data = self._store.get(key)
        if data is None:
            return None
        value, expiry = data
        if expiry < time.time():
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        import time
        self._store[key] = (value, time.time() + ttl_seconds)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)
```

```python
# ─── app/infrastructure/email.py ───
from __future__ import annotations


class SmtpEmailService:
    """Implement EmailService với SMTP."""

    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password

    def send(self, to: str, subject: str, body: str) -> bool:
        import smtplib
        from email.mime.text import MIMEText  # type: ignore
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['To'] = to
            msg['From'] = self._username
            with smtplib.SMTP(self._host, self._port) as server:
                server.starttls()
                server.login(self._username, self._password)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Email failed: {e}")
            return False


class ConsoleEmailService:
    """Ghi email ra console — dùng cho development."""

    def send(self, to: str, subject: str, body: str) -> bool:
        print(f"📧 [EMAIL] To: {to}")
        print(f"   Subject: {subject}")
        print(f"   Body: {body[:100]}...")
        return True
```

## DI Container

```python
# ─── app/container.py ───
from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session

from app.config import AppConfig
from app.domain.services.user_service import UserService
from app.domain.services.auth_service import AuthService
from app.domain.interfaces import UserRepository, CacheService, EmailService, TokenService
from app.infrastructure.database import DatabaseSessionManager
from app.infrastructure.repositories.user_repository import SQLAlchemyUserRepository
from app.infrastructure.cache import RedisCacheService, InMemoryCacheService
from app.infrastructure.email import SmtpEmailService, ConsoleEmailService
from app.infrastructure.auth import JwtTokenService
from app.infrastructure.unit_of_work import SQLAlchemyUnitOfWork


class ApplicationContainer:
    """
    DI Container cho ứng dụng User Management.
    Quản lý tất cả service registrations và lifecycle.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._singletons: dict[type, object] = {}
        self._db_manager: Optional[DatabaseSessionManager] = None

    # ─── Singleton Initialization ───

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def db_manager(self) -> DatabaseSessionManager:
        if self._db_manager is None:
            self._db_manager = DatabaseSessionManager(
                connection_string=self._config.database_url,
                echo=self._config.debug,
            )
            if self._config.auto_create_tables:
                self._db_manager.create_tables()
        return self._db_manager

    @property
    def cache(self) -> CacheService:
        key = CacheService
        if key not in self._singletons:
            if self._config.use_redis:
                self._singletons[key] = RedisCacheService(
                    host=self._config.redis_host,
                    port=self._config.redis_port,
                )
            else:
                self._singletons[key] = InMemoryCacheService()
        return self._singletons[key]  # type: ignore

    @property
    def email_service(self) -> EmailService:
        key = EmailService
        if key not in self._singletons:
            if self._config.email_provider == 'smtp':
                self._singletons[key] = SmtpEmailService(
                    host=self._config.smtp_host,
                    port=self._config.smtp_port,
                    username=self._config.smtp_username,
                    password=self._config.smtp_password,
                )
            else:
                self._singletons[key] = ConsoleEmailService()
        return self._singletons[key]  # type: ignore

    @property
    def token_service(self) -> TokenService:
        key = TokenService
        if key not in self._singletons:
            self._singletons[key] = JwtTokenService(
                secret_key=self._config.jwt_secret,
                algorithm=self._config.jwt_algorithm,
                expiration_minutes=self._config.jwt_expiration_minutes,
            )
        return self._singletons[key]  # type: ignore

    # ─── Session-scoped Dependencies ───

    def create_repository(self, session: Session) -> UserRepository:
        return SQLAlchemyUserRepository(session)

    def create_unit_of_work(self, session: Session):
        from app.infrastructure.unit_of_work import SQLAlchemyUnitOfWork
        return SQLAlchemyUnitOfWork(session)

    # ─── Services (Transient — mỗi request instance mới) ───

    def create_user_service(self, session: Session) -> UserService:
        return UserService(
            repository=self.create_repository(session),
            cache=self.cache,
            email=self.email_service,
            uow=self.create_unit_of_work(session),
        )

    def create_auth_service(self, session: Session) -> AuthService:
        return AuthService(
            repository=self.create_repository(session),
            token_service=self.token_service,
        )
```

## FastAPI Application

```python
# ─── app/api/schemas.py ───
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from app.domain.models import UserRole


class CreateUserRequestSchema(BaseModel):
    email: str
    name: str
    password: str
    role: UserRole = UserRole.USER


class UpdateUserRequestSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponseSchema(BaseModel):
    id: str
    email: str
    name: str
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TokenResponseSchema(BaseModel):
    access_token: str
    token_type: str = 'bearer'


class ErrorResponseSchema(BaseModel):
    detail: str
    error_code: str
```

```python
# ─── app/api/dependencies.py ───
from __future__ import annotations
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.container import ApplicationContainer
from app.domain.services.user_service import UserService
from app.domain.services.auth_service import AuthService
from app.domain.exceptions import AuthenticationError


# Global container — được khởi tạo khi app start
_container: ApplicationContainer = None  # type: ignore


def get_container() -> ApplicationContainer:
    return _container


def get_db_session(container: ApplicationContainer = Depends(get_container)) -> Session:
    with container.db_manager.session() as session:
        yield session


def get_user_service(
    session: Session = Depends(get_db_session),
    container: ApplicationContainer = Depends(get_container),
) -> UserService:
    return container.create_user_service(session)


def get_auth_service(
    session: Session = Depends(get_db_session),
    container: ApplicationContainer = Depends(get_container),
) -> AuthService:
    return container.create_auth_service(session)


security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
) -> dict:
    try:
        payload = auth_service.verify_token(credentials.credentials)
        return payload
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
```

```python
# ─── app/api/routes/user_routes.py ───
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import (
    CreateUserRequestSchema, UpdateUserRequestSchema,
    UserResponseSchema, ErrorResponseSchema,
)
from app.api.dependencies import get_user_service, get_current_user
from app.domain.services.user_service import UserService
from app.domain.exceptions import UserNotFoundError, DuplicateEmailError

router = APIRouter(prefix='/users', tags=['users'])


@router.post('/', response_model=UserResponseSchema, status_code=status.HTTP_201_CREATED)
def create_user(
    request: CreateUserRequestSchema,
    service: UserService = Depends(get_user_service),
):
    from app.domain.models import CreateUserRequest as CreateUserReq
    domain_req = CreateUserReq(
        email=request.email,
        name=request.name,
        password=request.password,
        role=request.role,
    )
    try:
        user = service.create_user(domain_req)
        return UserResponseSchema(
            id=user.id, email=user.email, name=user.name,
            role=user.role, is_active=user.is_active,
            created_at=user.created_at, updated_at=user.updated_at,
        )
    except DuplicateEmailError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.get('/{user_id}', response_model=UserResponseSchema)
def get_user(
    user_id: str,
    service: UserService = Depends(get_user_service),
):
    try:
        user = service.get_user(user_id)
        return UserResponseSchema(
            id=user.id, email=user.email, name=user.name,
            role=user.role, is_active=user.is_active,
            created_at=user.created_at, updated_at=user.updated_at,
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.patch('/{user_id}', response_model=UserResponseSchema)
def update_user(
    user_id: str,
    request: UpdateUserRequestSchema,
    service: UserService = Depends(get_user_service),
    current_user: dict = Depends(get_current_user),
):
    from app.domain.models import UpdateUserRequest
    domain_req = UpdateUserRequest(
        name=request.name,
        email=request.email,
        role=request.role,
        is_active=request.is_active,
    )
    try:
        user = service.update_user(user_id, domain_req)
        return UserResponseSchema(
            id=user.id, email=user.email, name=user.name,
            role=user.role, is_active=user.is_active,
            created_at=user.created_at, updated_at=user.updated_at,
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
```

```python
# ─── app/main.py ───
from __future__ import annotations
from fastapi import FastAPI
from app.config import AppConfig
from app.container import ApplicationContainer
from app.api.dependencies import _container

def create_app() -> FastAPI:
    config = AppConfig()
    container = ApplicationContainer(config)

    # Gán global container cho dependencies
    import app.api.dependencies as deps
    deps._container = container

    app = FastAPI(
        title='User Management API',
        version='1.0.0',
        description='Demo application for Dependency Injection',
    )

    from app.api.routes import user_routes
    app.include_router(user_routes.router)

    return app

app = create_app()


@app.on_event('startup')
async def startup():
    # Khởi tạo database tables
    container = _container
    if container.config.auto_create_tables:
        container.db_manager.create_tables()
```

## Testing với DI

```python
# ─── tests/conftest.py ───
from __future__ import annotations
import pytest
from unittest.mock import Mock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.domain.models import User, UserRole
from app.domain.interfaces import UserRepository, CacheService, EmailService, TokenService
from app.domain.services.user_service import UserService
from app.domain.services.auth_service import AuthService
from app.infrastructure.models import Base


@pytest.fixture
def mock_repo() -> Mock:
    repo = Mock(spec=UserRepository)
    return repo


@pytest.fixture
def mock_cache() -> Mock:
    cache = Mock(spec=CacheService)
    cache.get.return_value = None  # Miss cache mặc định
    return cache


@pytest.fixture
def mock_email() -> Mock:
    return Mock(spec=EmailService)


@pytest.fixture
def mock_token_service() -> Mock:
    svc = Mock(spec=TokenService)
    svc.create_access_token.return_value = "test_jwt_token"
    svc.verify_token.return_value = {'sub': 'user-1', 'role': 'user'}
    return svc


@pytest.fixture
def mock_uow() -> Mock:
    uow = MagicMock()
    uow.__enter__.return_value = uow
    uow.__exit__.return_value = None
    return uow


@pytest.fixture
def user_service(mock_repo: Mock, mock_cache: Mock, mock_email: Mock, mock_uow: Mock) -> UserService:
    return UserService(
        repository=mock_repo,
        cache=mock_cache,
        email=mock_email,
        uow=mock_uow,
    )


@pytest.fixture
def auth_service(mock_repo: Mock, mock_token_service: Mock) -> AuthService:
    return AuthService(repository=mock_repo, token_service=mock_token_service)


@pytest.fixture
def sample_user() -> User:
    from datetime import datetime
    return User(
        id='user-1',
        email='user@example.com',
        name='Nguyễn Văn A',
        role=UserRole.USER,
        created_at=datetime.now(),
    )
```

```python
# ─── tests/test_user_service.py ───
from __future__ import annotations
from unittest.mock import Mock, ANY
from datetime import datetime
import pytest  # type: ignore
from app.domain.models import CreateUserRequest, UpdateUserRequest, UserRole
from app.domain.services.user_service import UserService
from app.domain.exceptions import UserNotFoundError, DuplicateEmailError, InvalidOperationError


class TestCreateUser:

    def test_create_user_success(self, user_service: UserService, mock_repo: Mock,
                                  mock_cache: Mock, mock_email: Mock, mock_uow: Mock) -> None:
        mock_repo.find_by_email.return_value = None

        request = CreateUserRequest(
            email='new@example.com',
            name='New User',
            password='secure123',
            role=UserRole.USER,
        )
        user = user_service.create_user(request)

        assert user.email == 'new@example.com'
        assert user.name == 'New User'
        assert user.id is not None
        mock_repo.save.assert_called_once()
        mock_uow.commit.assert_called_once()
        mock_email.send.assert_called_once()
        mock_cache.invalidate.assert_called_once_with('users:all')

    def test_create_user_duplicate_email(self, user_service: UserService, mock_repo: Mock) -> None:
        existing_user = Mock(id='existing', email='dup@example.com')
        mock_repo.find_by_email.return_value = existing_user

        request = CreateUserRequest(
            email='dup@example.com', name='Dup', password='123', role=UserRole.USER,
        )
        with pytest.raises(DuplicateEmailError):
            user_service.create_user(request)
        mock_repo.save.assert_not_called()

    def test_create_admin_user_blocked(self, user_service: UserService, mock_repo: Mock) -> None:
        mock_repo.find_by_email.return_value = None
        request = CreateUserRequest(
            email='admin@test.com', name='Admin', password='123', role=UserRole.ADMIN,
        )
        with pytest.raises(InvalidOperationError):
            user_service.create_user(request)


class TestGetUser:

    def test_get_user_cache_hit(self, user_service: UserService, mock_cache: Mock,
                                  mock_repo: Mock, sample_user: User) -> None:
        import json
        mock_cache.get.return_value = json.dumps({
            'id': sample_user.id, 'email': sample_user.email,
            'name': sample_user.name, 'role': sample_user.role.value,
            'is_active': sample_user.is_active,
        })

        user = user_service.get_user('user-1')
        assert user.id == 'user-1'
        mock_repo.find_by_id.assert_not_called()  # Không cần query DB

    def test_get_user_cache_miss(self, user_service: UserService, mock_cache: Mock,
                                  mock_repo: Mock, sample_user: User) -> None:
        mock_cache.get.return_value = None
        mock_repo.find_by_id.return_value = sample_user

        user = user_service.get_user('user-1')
        assert user.id == 'user-1'
        mock_repo.find_by_id.assert_called_once_with('user-1')
        mock_cache.set.assert_called_once()  # Lưu cache

    def test_get_user_not_found(self, user_service: UserService, mock_repo: Mock) -> None:
        mock_repo.find_by_id.return_value = None

        with pytest.raises(UserNotFoundError):
            user_service.get_user('nonexistent')


class TestUpdateUser:

    def test_update_user_success(self, user_service: UserService, mock_repo: Mock,
                                  mock_uow: Mock, mock_cache: Mock, sample_user: User) -> None:
        mock_repo.update.return_value = sample_user

        request = UpdateUserRequest(name='Updated Name')
        user = user_service.update_user('user-1', request)

        assert user.name == 'Nguyễn Văn A'
        mock_repo.update.assert_called_once_with('user-1', request)
        mock_uow.commit.assert_called_once()
        mock_cache.invalidate.assert_called()

    def test_update_user_not_found(self, user_service: UserService, mock_repo: Mock) -> None:
        mock_repo.update.return_value = None

        with pytest.raises(UserNotFoundError):
            user_service.update_user('nonexistent', UpdateUserRequest(name='X'))


class TestDeleteUser:

    def test_delete_user_success(self, user_service: UserService, mock_repo: Mock,
                                  mock_uow: Mock, mock_cache: Mock) -> None:
        mock_repo.delete.return_value = True

        result = user_service.delete_user('user-1')
        assert result is True
        mock_repo.delete.assert_called_once_with('user-1')
        mock_uow.commit.assert_called_once()
        mock_cache.invalidate.assert_called()

    def test_delete_user_not_found(self, user_service: UserService, mock_repo: Mock) -> None:
        mock_repo.delete.return_value = False

        with pytest.raises(UserNotFoundError):
            user_service.delete_user('nonexistent')
```

## Kết luận

Ứng dụng User Management này minh họa đầy đủ cách DI, DIP, và Clean Architecture hoạt động trong thực tế:

1. **Domain layer** (core) hoàn toàn không biết gì về FastAPI, SQLAlchemy, Redis, hay bất kỳ framework nào. Nó chỉ biết đến interfaces (Protocols) định nghĩa các port.
2. **Infrastructure layer** implement các port đó — SQLAlchemy cho database, Redis cho cache, SMTP cho email, JWT cho token.
3. **DI Container** (ApplicationContainer) kết nối mọi thứ — nó quyết định implementation nào được dùng tùy theo config (dev/prod).
4. **FastAPI** chỉ là delivery mechanism — nó gọi services thông qua `Depends()`.

Lợi ích của kiến trúc này:

- **Testability**: Unit test business logic với mock — không cần database, Redis, email thật. Test chạy trong milliseconds.
- **Flexibility**: Đổi cache từ Redis sang Memcached? Chỉ cần implement `CacheService` interface mới. Đổi database từ PostgreSQL sang MongoDB? Chỉ cần implement `UserRepository` interface mới. Không cần sửa domain code.
- **Rõ ràng**: Mỗi class có một trách nhiệm duy nhất (SRP). Dependency được inject rõ ràng qua constructor.
- **Dễ maintain**: Add tính năng mới = thêm service + interface mới. Không sửa code cũ.

Đây chính là sức mạnh thực sự của Dependency Injection — không phải là một "công nghệ" hay "framework", mà là một **cách tổ chức code** giúp ứng dụng tồn tại và phát triển qua nhiều năm thay đổi.
