---
id: singleton
title: Singleton
sidebar_label: 🥇 Singleton
sidebar_position: 2
---

# Singleton

**Singleton** đảm bảo một class chỉ có duy nhất một instance và cung cấp một global point of access đến instance đó.

## Bài toán

Bạn đang xây dựng một ứng dụng cần kết nối database. Mỗi lần cần query, bạn tạo một kết nối mới. Điều này dẫn đến:
- Tốn tài nguyên (mỗi connection chiếm memory + socket)
- Không kiểm soát được số lượng kết nối
- Nguy cơ quá tải database

## Giải pháp

Singleton đảm bảo chỉ có **một** instance DatabaseConnection được tạo ra. Mọi module trong ứng dụng đều dùng chung instance đó.

```python
class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connection = cls._instance._connect()
        return cls._instance

    def _connect(self):
        print('🔌 Kết nối database thành công')
        return {'connected': True}

    def query(self, sql):
        print(f'📊 Query: {sql}')
        return [{'id': 1, 'name': 'Alice'}]

db1 = Database()
db2 = Database()
print(db1 is db2)  # True
```

## Khi nào dùng

- Cần đúng một instance (database connection, logger, config)
- Instance đó phải được truy cập từ nhiều nơi
- Muốn kiểm soát chặt việc khởi tạo

## Thực tế

- Cấu hình ứng dụng (config) toàn cục
- Logger dùng chung trong toàn bộ ứng dụng
- Kết nối database pool
