---
id: solid-dip
title: D — Dependency Inversion Principle
sidebar_label: D — Dependency Inversion
sidebar_position: 30
---

# D — Dependency Inversion Principle

> **"Depend upon abstractions, not concretions."** — Robert C. Martin

1. Các module cấp cao không nên phụ thuộc vào module cấp thấp. Cả hai nên phụ thuộc vào **abstraction**.
2. Abstraction không nên phụ thuộc vào chi tiết. **Chi tiết phụ thuộc vào abstraction.**

## Bài toán: Module cấp cao phụ thuộc trực tiếp vào module cấp thấp

```python
class MySQLDatabase:
    def save_user(self, user):
        print(f'💾 Lưu {user.name} vào MySQL')

class PostgresDatabase:
    def save_user(self, user):
        print(f'💾 Lưu {user.name} vào PostgreSQL')

class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # ❌ Phụ thuộc vào concrete

    def register(self, user):
        # Validate...
        self.db.save_user(user)
        print(f'📧 Gửi email chào mừng đến {user.email}')
```

**Vấn đề:** `UserService` **phụ thuộc cứng** vào `MySQLDatabase`. Muốn đổi sang Postgres, MongoDB, Redis → phải sửa `UserService`. Không thể test được (phải có MySQL thật).

## Giải pháp: Dependency Inversion

```python
from abc import ABC, abstractmethod

class UserRepository(ABC):
    @abstractmethod
    def save(self, user): pass

class MySQLUserRepository(UserRepository):
    def save(self, user):
        print(f'💾 Lưu {user.name} vào MySQL')

class PostgresUserRepository(UserRepository):
    def save(self, user):
        print(f'💾 Lưu {user.name} vào PostgreSQL')

class MongoUserRepository(UserRepository):
    def save(self, user):
        print(f'🍃 Lưu {user.name} vào MongoDB')

class UserService:
    def __init__(self, repo: UserRepository):  # ✅ Phụ thuộc abstraction
        self.repo = repo

    def register(self, user):
        # Validate...
        self.repo.save(user)
        print(f'📧 Gửi email chào mừng đến {user.email}')

# Sử dụng
service = UserService(MySQLUserRepository())
service.register(user)

# Đổi database không cần sửa UserService
service2 = UserService(MongoUserRepository())
```

## Dependency Injection — Inject dependency từ bên ngoài

```python
class EmailService(ABC):
    @abstractmethod
    def send(self, to, subject, body): pass

class SMTPEmailService(EmailService):
    def send(self, to, subject, body):
        print(f'📧 Gửi email SMTP đến {to}: {subject}')

class SendGridEmailService(EmailService):
    def send(self, to, subject, body):
        print(f'📧 Gửi email qua SendGrid đến {to}: {subject}')

class UserRegistration:
    def __init__(
        self,
        repo: UserRepository,
        email: EmailService,
    ):
        self.repo = repo
        self.email = email

    def register(self, user):
        self.repo.save(user)
        self.email.send(user.email, 'Chào mừng', 'Bạn đã đăng ký thành công!')
```

## Lợi ích

| Trước (vi phạm DIP) | Sau (đúng DIP) |
|---------------------|----------------|
| Phụ thuộc cứng vào MySQL | Phụ thuộc abstraction `UserRepository` |
| Không thể test | Có thể mock: `MockUserRepository` |
| Đổi database phải sửa code | Đổi database qua constructor |
| Vi phạm OCP | Mở cho mở rộng, đóng cho sửa đổi |

## Test với mock

```python
class MockUserRepository(UserRepository):
    def save(self, user):
        self.saved_user = user  # Không cần DB thật

class MockEmailService(EmailService):
    def send(self, to, subject, body):
        self.last_email = (to, subject, body)

def test_registration():
    repo = MockUserRepository()
    email = MockEmailService()
    service = UserRegistration(repo, email)

    user = User(name='Alice', email='alice@example.com')
    service.register(user)

    assert repo.saved_user.name == 'Alice'
    assert email.last_email[0] == 'alice@example.com'
```

## Kết luận

DIP là nguyên lý **quan trọng nhất** trong SOLID khi xây dựng ứng dụng lớn. Nó cùng với DI (Dependency Injection) tạo nên kiến trúc linh hoạt, dễ test, dễ mở rộng. Hãy nhớ: **phụ thuộc vào interface, không phụ thuộc vào implementation**.
