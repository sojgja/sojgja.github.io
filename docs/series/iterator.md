---
id: iterator
title: Iterator
sidebar_label: 🔄 Iterator
sidebar_position: 17
---

# Iterator

> **Iterator** — *"Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation."* — GoF, 1994

## Bài toán chi tiết

Xây dựng hệ thống phân tích dữ liệu lớn (big data analytics platform) phải xử lý nhiều loại nguồn dữ liệu khác nhau: file log trên disk, database cursor, Kafka stream, Redis sorted set, và tree cấu trúc thư mục. Mỗi nguồn có cơ chế duyệt riêng: file log dùng `readline()`, database cursor dùng `fetchone()`, Kafka dùng `poll()`, Redis dùng `zrange()`. Module phân tích (analytics engine) cần duyệt qua tất cả các nguồn này để tính toán metric — nhưng API của từng nguồn khác nhau hoàn toàn.

Cách tiếp cận ngây thơ là viết một lớp `DataProcessor` với các method riêng cho từng nguồn: `process_file()`, `process_db()`, `process_kafka()`. Nếu thêm nguồn mới (ví dụ: MongoDB cursor, S3 file stream), bạn phải viết thêm method. Code trùng lặp (duplicate) vì logic xử lý record giống nhau — chỉ khác cách duyệt.

Vấn đề thứ hai là lazy evaluation. Dữ liệu có thể lên đến hàng terabyte, không thể load toàn bộ vào memory. Cần cơ chế duyệt từng phần tử một (one by one) mà không cần biết collection có bao nhiêu phần tử. Iterator pattern cung cấp lazy iteration chuẩn.

Vấn đề thứ ba là multiple traversal strategies. Một tree directory cần 3 cách duyệt: pre-order (root → children), post-order (children → root), và level-order (BFS). Nếu hard-code traversal vào class `DirectoryTree`, mỗi lần thêm cách duyệt mới phải sửa class đó — vi phạm OCP.

Cuối cùng, composite collection. Một collection có thể chứa collection con (composite) — ví dụ: thư mục chứa file và thư mục con. Iterator phải có khả năng duyệt đệ quy toàn bộ cây, không chỉ một cấp.

## Giải pháp với Pattern

Iterator pattern tách biệt **cách duyệt** (traversal) khỏi **cấu trúc collection** (aggregate). Collection implement interface `Iterable` (trả về Iterator). Iterator implement interface `Iterator` với `__next__()` và `__iter__()`. Client chỉ gọi `next()` mà không biết collection là list, tree, hay stream.

Cấu trúc pattern:
- **Iterator (ABC)**: `__next__()` → phần tử tiếp theo hoặc `StopIteration`; `__iter__()` → self.
- **ConcreteIterator**: `ListIterator`, `TreeIterator`, `FileLineIterator`, `KafkaStreamIterator`.
- **Aggregate (Iterable)**: `__iter__()` → trả về Iterator mới.
- **ConcreteAggregate**: `ListCollection`, `DirectoryTree`, `LogFile`.

Pattern giải quyết:
- **Uniform API**: `for record in source:` làm việc với mọi nguồn dữ liệu.
- **Lazy evaluation**: Iterator chỉ tính toán/tải phần tử khi cần.
- **Multiple traversals**: Mỗi cách duyệt là một Iterator class riêng, không sửa collection.
- **Composite traversal**: Iterator có thể duyệt đệ quy cây.

## Phân tích thiết kế

**OOP Principles:**
- **Single Responsibility (SRP)**: Collection quản lý dữ liệu; Iterator quản lý traversal. Hai trách nhiệm tách biệt.
- **Open/Closed (OCP)**: Thêm traversal mới = thêm Iterator class mới. Không sửa collection.
- **Dependency Inversion (DIP)**: Client phụ thuộc vào `Iterator` abstraction, không phụ thuộc vào concrete collection.
- **Interface Segregation (ISP)**: Iterator interface nhỏ gọn (chỉ `__next__` và `__iter__`). Client không bị ép implement method không cần.

**Trade-offs:**
- **Memory overhead**: Mỗi duyệt tạo một Iterator object mới. Với collection siêu nhỏ, overhead không đáng kể.
- **Stateful iteration**: Iterator là stateful — không thể chia sẻ giữa nhiều thread mà không đồng bộ. Cần `copy()` hoặc tạo iterator mới cho mỗi thread.
- **Khó implement skip/backward**: Iterator về cơ bản là forward-only. Để hỗ trợ backward, cần bidirectional iterator (`__prev__`).

**Khi không nên dùng:**
- Collection đơn giản (Python list, tuple) — Python đã có iterator tích hợp sẵn.
- Dữ liệu nhỏ, load hết vào memory được — list comprehension đơn giản hơn.
- Cần truy cập ngẫu nhiên (random access) — dùng index (`list[i]`) nhanh hơn.

## Ví dụ code hoàn chỉnh

### Cách làm sai: Xử lý riêng từng loại collection

```python
from __future__ import annotations
from typing import Any, Optional
import os


class DataProcessor:
    """Xử lý dữ liệu từ nhiều nguồn — vi phạm DRY, khó mở rộng."""

    def process_file(self, filepath: str) -> list[dict]:
        result = []
        with open(filepath, "r") as f:
            for line in f:
                record = self._parse_line(line)
                if record:
                    result.append(self._analyze(record))
        return result

    def process_database(self, cursor) -> list[dict]:
        result = []
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            record = self._parse_row(row)
            if record:
                result.append(self._analyze(record))
        return result

    def process_kafka(self, consumer, timeout: float = 1.0) -> list[dict]:
        result = []
        while True:
            msg = consumer.poll(timeout)
            if msg is None:
                break
            record = self._parse_kafka_msg(msg)
            if record:
                result.append(self._analyze(record))
        return result

    # Mỗi lần thêm nguồn mới: phải viết method process_xxx mới
    # Logic _analyze() lặp lại ở mọi method

    def _parse_line(self, line: str) -> Optional[dict]: ...
    def _parse_row(self, row: tuple) -> Optional[dict]: ...
    def _parse_kafka_msg(self, msg) -> Optional[dict]: ...
    def _analyze(self, record: dict) -> dict: ...
```

### Cách làm đúng: Iterator Pattern

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol
import os
import json
import logging
from collections import deque

logger = logging.getLogger(__name__)


# --- Abstract Iterator (sử dụng Python Protocol) ---

class RecordIterator(Iterator[dict]):
    """Iterator trừu tượng cho các record dữ liệu."""
    @abstractmethod
    def __next__(self) -> dict: ...

    @abstractmethod
    def __iter__(self) -> RecordIterator: ...


# --- Concrete Iterators ---

class FileLineIterator(RecordIterator):
    """Iterator đọc file log từng dòng — lazy evaluation."""
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._file = open(filepath, "r", encoding="utf-8")

    def __next__(self) -> dict:
        line = self._file.readline()
        if not line:
            self._file.close()
            raise StopIteration
        record = self._parse_line(line)
        if record is None:
            return self.__next__()  # Skip invalid lines
        return record

    def __iter__(self) -> RecordIterator:
        return self

    def __del__(self) -> None:
        if not self._file.closed:
            self._file.close()

    def _parse_line(self, line: str) -> Optional[dict]:
        try:
            parts = line.strip().split(" | ")
            if len(parts) < 3:
                return None
            return {
                "timestamp": parts[0],
                "level": parts[1],
                "message": " | ".join(parts[2:]),
                "source": self.filepath,
            }
        except (IndexError, ValueError):
            return None


class DirectoryTreeIterator(RecordIterator):
    """Iterator duyệt cây thư mục — multiple traversal strategies."""

    class TraversalMode:
        PRE_ORDER = "pre"
        POST_ORDER = "post"
        LEVEL_ORDER = "level"

    def __init__(
        self,
        root_path: str,
        mode: str = "pre",
        pattern: str = "*",
    ) -> None:
        self.root_path = root_path
        self.mode = mode
        self.pattern = pattern
        self._stack: list[Any] = []
        self._init_traversal()

    def _init_traversal(self) -> None:
        if self.mode == self.TraversalMode.PRE_ORDER:
            self._stack = [(self.root_path, False)]
        elif self.mode == self.TraversalMode.POST_ORDER:
            self._stack = [(self.root_path, False)]
        elif self.mode == self.TraversalMode.LEVEL_ORDER:
            self._queue = deque([self.root_path])
        else:
            raise ValueError(f"Unknown traversal mode: {self.mode}")

    def __next__(self) -> dict:
        if self.mode == self.TraversalMode.PRE_ORDER:
            return self._next_pre_order()
        elif self.mode == self.TraversalMode.POST_ORDER:
            return self._next_post_order()
        elif self.mode == self.TraversalMode.LEVEL_ORDER:
            return self._next_level_order()
        raise StopIteration

    def _next_pre_order(self) -> dict:
        while self._stack:
            path, visited = self._stack.pop()
            info = self._file_info(path)
            if os.path.isdir(path):
                # Push children
                children = []
                try:
                    children = [
                        os.path.join(path, f)
                        for f in os.listdir(path)
                        if not f.startswith(".")
                    ]
                except PermissionError:
                    pass
                for child in reversed(children):
                    self._stack.append((child, False))
            return info
        raise StopIteration

    def _next_post_order(self) -> dict:
        while self._stack:
            path, visited = self._stack.pop()
            if visited:
                return self._file_info(path)
            self._stack.append((path, True))
            if os.path.isdir(path):
                children = []
                try:
                    children = [
                        os.path.join(path, f)
                        for f in os.listdir(path)
                        if not f.startswith(".")
                    ]
                except PermissionError:
                    pass
                for child in reversed(children):
                    self._stack.append((child, False))
        raise StopIteration

    def _next_level_order(self) -> dict:
        if not self._queue:
            raise StopIteration
        path = self._queue.popleft()
        if os.path.isdir(path):
            try:
                for f in os.listdir(path):
                    if not f.startswith("."):
                        self._queue.append(os.path.join(path, f))
            except PermissionError:
                pass
        return self._file_info(path)

    def _file_info(self, path: str) -> dict:
        try:
            stat = os.stat(path)
            return {
                "path": path,
                "name": os.path.basename(path),
                "is_dir": os.path.isdir(path),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        except OSError:
            return {"path": path, "name": os.path.basename(path), "error": True}

    def __iter__(self) -> RecordIterator:
        return self


class DatabaseCursorIterator(RecordIterator):
    """Iterator cho database cursor — lazy fetch."""
    def __init__(self, cursor, table_name: str) -> None:
        self._cursor = cursor
        self.table_name = table_name

    def __next__(self) -> dict:
        row = self._cursor.fetchone()
        if row is None:
            self._cursor.close()
            raise StopIteration
        return {
            "table": self.table_name,
            "row": dict(row) if hasattr(row, "_asdict") else row,
        }

    def __iter__(self) -> RecordIterator:
        return self


class KafkaStreamIterator(RecordIterator):
    """Iterator cho Kafka stream — consume message."""
    def __init__(self, consumer, topic: str, timeout: float = 1.0) -> None:
        self._consumer = consumer
        self.topic = topic
        self.timeout = timeout

    def __next__(self) -> dict:
        msg = self._consumer.poll(self.timeout)
        if msg is None:
            raise StopIteration
        if msg.error():
            raise RuntimeError(f"Kafka error: {msg.error()}")
        try:
            value = json.loads(msg.value().decode("utf-8"))
        except (json.JSONDecodeError, AttributeError):
            value = {"raw": msg.value().decode("utf-8", errors="replace")}
        return {
            "topic": self.topic,
            "partition": msg.partition(),
            "offset": msg.offset(),
            "key": msg.key().decode() if msg.key() else None,
            "value": value,
            "timestamp": msg.timestamp()[1] if msg.timestamp() else 0,
        }

    def __iter__(self) -> RecordIterator:
        return self


# --- Iterable Collections ---

class LogFile:
    """Aggregate: file log (iterable)."""
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def __iter__(self) -> FileLineIterator:
        return FileLineIterator(self.filepath)


class DirectoryTree:
    """Aggregate: cây thư mục với nhiều traversal mode."""
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path

    def pre_order(self) -> DirectoryTreeIterator:
        return DirectoryTreeIterator(self.root_path, mode="pre")

    def post_order(self) -> DirectoryTreeIterator:
        return DirectoryTreeIterator(self.root_path, mode="post")

    def level_order(self) -> DirectoryTreeIterator:
        return DirectoryTreeIterator(self.root_path, mode="level")


# --- Composite Iterator (duyệt nhiều nguồn đồng thời) ---

class CompositeIterator(RecordIterator):
    """Iterator duyệt qua nhiều iterator khác tuần tự."""
    def __init__(self, iterators: list[RecordIterator]) -> None:
        self._iterators = iterators
        self._current_idx = 0

    def __next__(self) -> dict:
        while self._current_idx < len(self._iterators):
            try:
                return next(self._iterators[self._current_idx])
            except StopIteration:
                self._current_idx += 1
        raise StopIteration

    def __iter__(self) -> RecordIterator:
        return self


# --- Client: Analytics Engine ---

class AnalyticsEngine:
    """Xử lý dữ liệu từ bất kỳ nguồn nào — chỉ biết Iterator."""

    def __init__(self, source: RecordIterator) -> None:
        self.source = source

    def run(self) -> list[dict]:
        results = []
        for record in self.source:
            enriched = self._enrich(record)
            results.append(enriched)
        logger.info(f"Processed {len(results)} records")
        return results

    def _enrich(self, record: dict) -> dict:
        record["processed_at"] = __import__("time").time()
        record["length"] = len(str(record))
        return record


# --- Usage ---
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    # Tạo file log tạm
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write("2024-01-01 10:00:00 | INFO | Server started\n")
        f.write("2024-01-01 10:01:00 | WARNING | High memory usage\n")
        f.write("2024-01-01 10:02:00 | ERROR | Connection timeout\n")
        log_path = f.name

    # Dùng Iterator pattern — AnalyticsEngine không biết FileLineIterator
    log_source = LogFile(log_path)
    engine = AnalyticsEngine(iter(log_source))
    results = engine.run()

    for r in results:
        print(f"  [{r['level']}] {r['message']}")

    # Composite iterator: duyệt nhiều file
    print("\n--- Composite Iterator ---")
    log1 = LogFile(log_path)
    log2 = LogFile(log_path)
    composite = CompositeIterator([iter(log1), iter(log2)])
    count = sum(1 for _ in composite)
    print(f"Total records (2 files): {count}")

    os.unlink(log_path)
```

## Sơ đồ UML

```
┌─────────────────────────┐
│      Iterator<T>        │
│     (Protocol/ABC)      │
│─────────────────────────│
│ + __next__(): T         │
│ + __iter__(): Iterator  │
└────────────┬────────────┘
             │ implements
    ┌────────┼──────────┬────────────────┐
    │        │          │                │
┌───┴────┐ ┌─┴────┐  ┌─┴───────┐  ┌─────┴──────┐
│FileLine│ │DB    │  │Directory│  │KafkaStream │
│Iterator│ │Cursor│  │TreeIter │  │Iterator    │
│        │ │Iter  │  │         │  │            │
└───┬────┘ └──────┘  └──┬──────┘  └────────────┘
    │                    │
    │                    ├─ pre_order()
    │                    ├─ post_order()
    │                    └─ level_order()
    │
┌───┴───────────────────────┐
│  CompositeIterator        │
│  - iterators: list[Iter]  │
│  + __next__(): T          │
└───────────────────────────┘

┌──────────────────┐     ┌──────────────────┐
│  Iterable        │     │    Client        │
│  (Aggregate)     │     │ (AnalyticsEngine)│
│──────────────────│     │──────────────────│
│ + __iter__(): It │────>│ + run(): list    │
└──────────────────┘     └──────────────────┘
```

## So sánh với Pattern liên quan

**1. Composite Pattern:**
Composite tạo cấu trúc cây để client xử lý đồng nhất leaf và composite (dùng recursion). Iterator thường được dùng trong Composite để duyệt cây. Composite trả lời "cấu trúc thế nào"; Iterator trả lời "duyệt ra sao". Hai pattern thường đi cùng nhau.

**2. Visitor Pattern:**
Visitor tách operations khỏi object structure. Iterator tách traversal khỏi collection. Cả hai đều giúp thêm behavior mới mà không sửa class cũ. Khác biệt: Visitor thêm operation (xử lý mỗi node), Iterator thêm traversal (cách đi qua các node).

**3. Generator Pattern (Python-specific):**
Generator là Python's built-in Iterator implementation. Dùng `yield` tạo iterator mà không cần viết class `__next__`. Generator function thay thế ConcreteIterator cho các trường hợp đơn giản.

## Ứng dụng thực tế

**1. Python Built-in Iterator Protocol:**
Python tích hợp Iterator trong mọi collection. `for x in list:` gọi `iter(list)` → `__next__()`.

```python
# Python Iterator Protocol
class Range:
    def __init__(self, n: int):
        self.n = n
    def __iter__(self):
        self._i = 0
        return self
    def __next__(self):
        if self._i >= self.n:
            raise StopIteration
        val = self._i
        self._i += 1
        return val

# Generator dạng đơn giản
def fibonacci(limit: int):
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b
```

**2. Django QuerySet Iterator:**
Django QuerySet là iterable. Khi dùng `for obj in MyModel.objects.all()`, nó lazy fetch từ database, không load toàn bộ memory.

```python
from django.db import models

# QuerySet là iterable
for user in User.objects.filter(is_active=True).iterator(chunk_size=1000):
    # iterator() dùng server-side cursor — không OOM với 10M records
    print(user.email)
```

**3. Java Iterable / Iterator Interface:**
Java có `Iterable<T>` (for-each) và `Iterator<T>` core interfaces. Mọi collection (List, Set, Queue) implement Iterable.

```java
// Java Iterator pattern — built-in
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
Iterator<String> it = names.iterator();
while (it.hasNext()) {
    System.out.println(it.next());
}

// Custom iterable
class TreeNode<T> implements Iterable<T> {
    @Override
    public Iterator<T> iterator() {
        return new TreeIterator<>(this);
    }
}
```

**4. C# IEnumerable / IEnumerator:**
.NET dùng IEnumerable (iterable) và IEnumerator (iterator). LINQ mở rộng với lazy evaluation qua `yield return`.

```csharp
// C# Iterator pattern — yield return
IEnumerable<int> GetEven(IEnumerable<int> source) {
    foreach (int num in source) {
        if (num % 2 == 0)
            yield return num;  // Lazy — mỗi lần next()
    }
}
```

## Kiểm thử

```python
import pytest
import tempfile
import os


class TestFileLineIterator:
    def test_reads_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-01 | INFO | Line 1\n")
            f.write("2024-01-01 | WARN | Line 2\n")
            path = f.name

        try:
            it = FileLineIterator(path)
            records = list(it)
            assert len(records) == 2
            assert records[0]["level"] == "INFO"
            assert records[1]["level"] == "WARN"
        finally:
            os.unlink(path)

    def test_skips_invalid_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("INVALID LINE\n")
            f.write("2024-01-01 | INFO | Valid\n")
            path = f.name

        try:
            it = FileLineIterator(path)
            records = list(it)
            assert len(records) == 1
            assert records[0]["level"] == "INFO"
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            path = f.name

        try:
            it = FileLineIterator(path)
            records = list(it)
            assert len(records) == 0
        finally:
            os.unlink(path)

    def test_closes_file_after_iteration(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-01 | INFO | Test\n")
            path = f.name

        it = FileLineIterator(path)
        _ = list(it)
        assert it._file.closed


class TestDirectoryTreeIterator:
    def test_pre_order_traversal(self, tmp_path):
        d1 = tmp_path / "dir1"
        d1.mkdir()
        (d1 / "file1.txt").write_text("a")
        (tmp_path / "file0.txt").write_text("b")

        it = DirectoryTreeIterator(str(tmp_path), mode="pre")
        paths = [r["path"] for r in it]
        # Root trước, sau đó đến children
        assert any(str(tmp_path) in p for p in paths)

    def test_multiple_traversal_modes(self, tmp_path):
        it1 = DirectoryTreeIterator(str(tmp_path), mode="pre")
        it2 = DirectoryTreeIterator(str(tmp_path), mode="post")
        it3 = DirectoryTreeIterator(str(tmp_path), mode="level")

        l1 = list(it1)
        l2 = list(it2)
        l3 = list(it3)

        assert len(l1) == len(l2) == len(l3)  # Cùng số phần tử
        assert l1 != l2  # Thứ tự khác nhau


class TestCompositeIterator:
    def test_chains_iterators(self):
        data1 = [{"id": 1}, {"id": 2}]
        data2 = [{"id": 3}]

        it = CompositeIterator([iter(data1), iter(data2)])
        results = list(it)
        assert len(results) == 3
        assert results[2]["id"] == 3

    def test_empty_iterators(self):
        it = CompositeIterator([])
        results = list(it)
        assert len(results) == 0


class TestAnalyticsEngine:
    def test_engine_works_with_any_iterator(self):
        data = [{"msg": "test"}, {"msg": "test2"}]
        engine = AnalyticsEngine(iter(data))
        results = engine.run()
        assert len(results) == 2
        assert "processed_at" in results[0]

    def test_engine_empty_source(self):
        engine = AnalyticsEngine(iter([]))
        results = engine.run()
        assert len(results) == 0
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| API đồng nhất cho mọi collection | Overhead tạo object Iterator mới mỗi lần duyệt |
| Lazy evaluation — tiết kiệm memory | Iterator stateful — không thread-safe |
| Dễ thêm traversal mới (OCP) | Không support random access |
| Tách bạch traversal khỏi collection | Khó implement skip/rewind |
| Hỗ trợ composite traversal | Cần đóng resource thủ công (file, DB) — có thể dùng context manager |
| Python built-in support (`for x in c`) | Với collection nhỏ, list comprehension đơn giản hơn |

## Kết luận

Iterator pattern là nền tảng của mọi thao tác duyệt dữ liệu trong lập trình hiện đại. Python đã tích hợp pattern này vào core language (iterator protocol, generator, `for` loop). Sử dụng Iterator khi bạn cần:

1. **Che giấu cấu trúc collection** khỏi client.
2. **Lazy evaluation** cho dữ liệu lớn (không load hết vào memory).
3. **Multiple traversal strategies** trên cùng một collection.
4. **Uniform API** cho nhiều nguồn dữ liệu khác nhau.

**Golden rules:**
1. Luôn implement `__iter__` và `__next__` (hoặc dùng generator với `yield`).
2. Đảm bảo Iterator có thể tạo nhiều lần độc lập (mỗi lần `iter(collection)` trả về iterator mới).
3. Dùng **Generator** nếu traversal đơn giản — tránh viết class boilerplate.
4. Quản lý resource (file handle, DB connection) cẩn thận — dùng context manager nếu cần.
5. Cân nhắc **bidirectional iterator** (`__prev__`, `__len__`) nếu cần skip/rewind.
