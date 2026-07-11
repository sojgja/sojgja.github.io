---
id: chain-of-responsibility
title: Chain of Responsibility
sidebar_label: ⛓️ Chain of Responsibility
sidebar_position: 14
---

# Chain of Responsibility

**Chain of Responsibility** cho phép nhiều object có cơ hội xử lý một request, bằng cách tạo thành chuỗi (chain) các handler. Request được chuyển dọc theo chuỗi cho đến khi có handler xử lý.

## Bài toán

Hệ thống xử lý **khiếu nại** khách hàng. Mỗi khiếu nại có mức độ: `Low`, `Medium`, `High`, `Critical`. Nhân viên CS chỉ xử lý `Low`. Trưởng nhóm xử lý `Medium`. Quản lý xử lý `High`. Ban giám đốc xử lý `Critical`. Nếu dùng if-else, code rất dài và mỗi lần thêm mức mới phải sửa.

## Giải pháp

Chain of Responsibility: mỗi handler quyết định xử lý hoặc chuyển tiếp.

```python
from abc import ABC, abstractmethod

class Handler(ABC):
    def __init__(self):
        self._next = None

    def set_next(self, handler):
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, request):
        pass

class CSHandler(Handler):
    def handle(self, request):
        if request['level'] == 'Low':
            return f'👤 CS xử lý: {request["content"]}'
        if self._next:
            return self._next.handle(request)
        return None

class TeamLeadHandler(Handler):
    def handle(self, request):
        if request['level'] == 'Medium':
            return f'👥 Team lead xử lý: {request["content"]}'
        if self._next:
            return self._next.handle(request)
        return None

class ManagerHandler(Handler):
    def handle(self, request):
        if request['level'] == 'High':
            return f'💼 Manager xử lý: {request["content"]}'
        if self._next:
            return self._next.handle(request)
        return None

class DirectorHandler(Handler):
    def handle(self, request):
        if request['level'] == 'Critical':
            return f'👑 Director xử lý: {request["content"]}'
        return '❌ Không ai xử lý được'

# Xây chain
cs = CSHandler()
tl = TeamLeadHandler()
mgr = ManagerHandler()
dir_ = DirectorHandler()
cs.set_next(tl).set_next(mgr).set_next(dir_)

print(cs.handle({'level': 'High', 'content': 'Lỗi thanh toán'}))
# 💼 Manager xử lý: Lỗi thanh toán

print(cs.handle({'level': 'Critical', 'content': 'Hệ thống sập'}))
# 👑 Director xử lý: Hệ thống sập
```

## Khi nào dùng

- Có nhiều handler xử lý request theo thứ tự
- Không biết trước handler nào sẽ xử lý
- Muốn thêm/xóa handler linh hoạt

## Thực tế

- Django middleware (request → auth → session → csrf → view)
- Python logging handlers (DEBUG → INFO → WARNING → ERROR)
- Exception handling (try/except chain)
