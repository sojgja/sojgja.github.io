---
id: di-intro
title: Dependency Injection — Giới thiệu
sidebar_label: 💉 Giới thiệu DI
sidebar_position: 31
---

# Dependency Injection — Giới thiệu

**Dependency Injection (DI)** là một kỹ thuật thiết kế trong đó một object nhận các dependency của nó từ **bên ngoài** thay vì tự tạo chúng bên trong.

> "Don't call us, we'll call you." — Hollywood Principle

## Tại sao DI quan trọng?

```python
# ❌ Không DI — class tự tạo dependency
class OrderService:
    def __init__(self):
        self.logger = FileLogger()        # Tự tạo
        self.email = SMTPEmailService()   # Tự tạo
        self.db = MySQLDatabase()         # Tự tạo
```

```python
# ✅ Có DI — dependency được inject từ ngoài
class OrderService:
    def __init__(self, logger, email, db):
        self.logger = logger
        self.email = email
        self.db = db
```

## 3 lợi ích chính

| Lợi ích | Không DI | Có DI |
|---------|----------|-------|
| **Testability** | Không thể mock → phải test với file/log/DB thật | Mock dễ dàng: `MockLogger()`, `MockDB()` |
| **Flexibility** | Muốn đổi email service phải sửa code | Đổi qua constructor — không sửa code |
| **Separation of Concerns** | Class vừa khởi tạo vừa dùng | Class chỉ tập trung vào logic chính |

## Các hình thức DI

1. **Constructor Injection** — phổ biến nhất, inject qua `__init__`
2. **Setter Injection** — inject qua setter method
3. **Method Injection** — inject qua tham số method

## DI Container

Khi ứng dụng lớn (hàng trăm class), bạn sẽ cần **DI Container** — công cụ tự động quản lý việc tạo và inject dependency.

Các bài tiếp theo sẽ đi sâu vào từng hình thức DI và xây dựng DI Container đơn giản.
