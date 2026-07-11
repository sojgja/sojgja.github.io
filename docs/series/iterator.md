---
id: iterator
title: Iterator
sidebar_label: 🔄 Iterator
sidebar_position: 17
---

# Iterator

**Iterator** cung cấp cách truy cập tuần tự các phần tử của một collection mà không cần lộ cấu trúc bên trong.

## Bài toán

Bạn có nhiều loại collection: `list`, `set`, `dict`, `Tree`, `Graph`. Mỗi collection có cách duyệt khác nhau. Nếu client trực tiếp dùng vòng lặp với index (cho list) hoặc `current.next` (cho linked list), code sẽ phụ thuộc vào cấu trúc cụ thể. Thay đổi collection phải sửa khắp nơi.

## Giải pháp

Iterator chuẩn hóa cách duyệt: tất cả collection đều cung cấp iterator với `__next__()` và `__iter__()`.

```python
from abc import ABC, abstractmethod

class Iterator(ABC):
    @abstractmethod
    def has_next(self) -> bool: pass

    @abstractmethod
    def next(self): pass

class ListIterator(Iterator):
    def __init__(self, collection):
        self._collection = collection
        self._index = 0

    def has_next(self) -> bool:
        return self._index < len(self._collection)

    def next(self):
        if self.has_next():
            value = self._collection[self._index]
            self._index += 1
            return value
        raise StopIteration()

class IterableCollection(ABC):
    @abstractmethod
    def create_iterator(self) -> Iterator: pass

class ListCollection(IterableCollection):
    def __init__(self, items=None):
        self.items = items or []

    def create_iterator(self) -> Iterator:
        return ListIterator(self.items)

# Sử dụng — client không biết cấu trúc bên trong
def print_all(collection: IterableCollection):
    it = collection.create_iterator()
    while it.has_next():
        print(it.next())

list_col = ListCollection(['a', 'b', 'c'])
print_all(list_col)
```

## Iterator trong Python

Python đã tích hợp sẵn Iterator Protocol:

```python
arr = [1, 2, 3]
it = iter(arr)
print(next(it))  # 1
print(next(it))  # 2
print(next(it))  # 3

# for...of dùng iterator
for x in arr:
    print(x)

# Custom iterator với generator
def my_range(n):
    i = 0
    while i < n:
        yield i
        i += 1
```

## Khi nào dùng

- Collection có cấu trúc phức tạp, muốn ẩn khỏi client
- Cần hỗ trợ nhiều cách duyệt khác nhau
- Muốn dùng vòng lặp đồng nhất cho mọi collection

## Thực tế

- `for x in list:` — Python iterator protocol
- Generator functions (`yield`)
- Django `QuerySet` iterator
