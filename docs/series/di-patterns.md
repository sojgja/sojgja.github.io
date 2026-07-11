---
id: di-patterns
title: 3 hình thức Dependency Injection
sidebar_label: 💉 Hình thức DI
sidebar_position: 32
---

# 3 hình thức Dependency Injection

## 1. Constructor Injection (Khuyên dùng)

Inject dependency qua `__init__`. Dependency được set **một lần** và **bất biến** trong suốt vòng đời.

```python
class UserService:
    def __init__(self, repo: UserRepository, email: EmailService):
        self._repo = repo
        self._email = email

    def register(self, user):
        self._repo.save(user)
        self._email.send(user.email, 'Chào mừng!')

# Inject
service = UserService(MySQLRepository(), SendGridEmail())
```

**Ưu điểm:** Rõ ràng, bất biến, dễ test, thể hiện đầy đủ dependency.
**Nhược điểm:** Constructor dài nếu nhiều dependency.

## 2. Setter Injection

Inject dependency qua setter sau khi khởi tạo object.

```python
class UserService:
    def __init__(self):
        self._repo = None
        self._email = None

    def set_repository(self, repo: UserRepository):
        self._repo = repo

    def set_email_service(self, email: EmailService):
        self._email = email

    def register(self, user):
        self._repo.save(user)
        self._email.send(user.email, 'Chào mừng!')

# Inject
service = UserService()
service.set_repository(MySQLRepository())
service.set_email_service(SendGridEmail())
```

**Ưu điểm:** Linh hoạt, có thể thay đổi dependency sau khi tạo.
**Nhược điểm:** Có thể quên set → lỗi runtime. Không bất biến.

## 3. Method Injection

Inject dependency trực tiếp qua tham số method.

```python
class UserService:
    def register(self, user, repo: UserRepository, email: EmailService):
        repo.save(user)
        email.send(user.email, 'Chào mừng!')

# Inject
service = UserService()
service.register(user, MySQLRepository(), SendGridEmail())
```

**Ưu điểm:** Dependency chỉ cần khi dùng method, không cần lưu state.
**Nhược điểm:** Method signature dài, dependency không được tái sử dụng.

## So sánh

| Tiêu chí | Constructor | Setter | Method |
|----------|-------------|--------|--------|
| Rõ ràng | ✅ Cao | ⚠️ Trung bình | ❌ Thấp |
| Bất biến | ✅ Có | ❌ Không | ✅ Có |
| Dễ test | ✅ | ⚠️ | ✅ |
| Dependency optional | ❌ | ✅ | ✅ |
| Thay đổi runtime | ❌ | ✅ | ✅ |
| Khuyến dùng | **Luôn ưu tiên** | Khi dependency optional | Khi dependency tạm thời |

## Nguyên tắc vàng

> **Luôn ưu tiên Constructor Injection.** Dùng Setter Injection cho dependency không bắt buộc (optional). Dùng Method Injection khi dependency chỉ cần trong một method cụ thể.
