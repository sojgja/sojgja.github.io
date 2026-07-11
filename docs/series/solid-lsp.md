---
id: solid-lsp
title: L — Liskov Substitution Principle
sidebar_label: L — Liskov Substitution
sidebar_position: 28
---

# L — Liskov Substitution Principle

> *"If for each object o1 of type S there is an object o2 of type T such that for all programs P defined in terms of T, the behavior of P is unchanged when o1 is substituted for o2, then S is a subtype of T."* — **Barbara Liskov, "Data Abstraction and Hierarchy", OOPSLA 1987**

Liskov Substitution Principle (LSP) được Barbara Liskov — một trong những nhà khoa học máy tính tiên phong, người phụ nữ thứ hai nhận giải Turing — đưa ra tại hội nghị OOPSLA năm 1987. Định nghĩa toán học của bà có thể khó hiểu, nhưng bản chất lại đơn giản: nếu bạn có một class cha (base class) và một class con (subclass), thì bất kỳ đoạn code nào làm việc với class cha cũng phải làm việc được với class con mà không gặp vấn đề gì. Nếu bạn có hàm nhận tham số là `Shape` và truyền `Square` vào cũng được, truyền `Circle` vào cũng xong — đó là LSP. Nếu hàm tính diện tích của `Rectangle` mà truyền `Square` vào cho kết quả sai — đó là LSP bị vi phạm. LSP không phải là "kế thừa là xấu", mà là "kế thừa sai thì mới xấu". Kế thừa đúng là một công cụ cực kỳ mạnh mẽ; kế thừa sai là một trong những nguyên nhân hàng đầu dẫn đến thiết kế mong manh (fragile base class problem).

## Bài toán chi tiết: Hệ thống lưu trữ tài liệu doanh nghiệp

Một công ty xây dựng hệ thống quản lý tài liệu (DMS). Họ có interface `DocumentRepository` với các method: `save(doc)`, `get_by_id(id)`, `search(criteria)`, `update(doc)`, `delete(id)`, `bulk_save(docs)`, `export_csv()`, `generate_report()`. Ban đầu họ implement class `LocalFileRepository` lưu file trên ổ cứng — tất cả method đều hoạt động hoàn chỉnh. Sau đó họ implement `CloudDocumentRepository` dùng AWS S3 — cũng implement tất cả method. Tiếp theo, họ implement `ReadOnlyArchiveRepository` cho phép truy xuất tài liệu đã lưu trữ — nhưng archive này chỉ cho phép đọc, không cho phép sửa hay xóa.

Vấn đề bắt đầu từ đây. `ReadOnlyArchiveRepository` implement interface `DocumentRepository` nhưng `update()`, `delete()`, `bulk_save()` đều ném `NotImplementedError`. `export_csv()` trả về dữ liệu sai format vì archive dùng định dạng khác. `generate_report()` không có ý nghĩa với archive. Khi một module gọi `document_repository.update(doc)` và nhận về exception, hệ thống crash. Developer phải thêm hàng loạt `if isinstance(repo, ReadOnlyArchiveRepository)` để tránh gọi các method không được hỗ trợ — code trở nên xấu xí và dễ sai. Đây là dấu hiệu kinh điển của LSP violation: một class con không thể thay thế class cha một cách trong suốt.

Hậu quả lan rộng: các unit test của module gọi `DocumentRepository` phải được viết riêng cho từng loại repository, làm tăng gấp đôi khối lượng test. Khi có bug, developer không biết là bug do repository cụ thể hay do logic chung. Mỗi lần thêm loại repository mới (ví dụ: `S3CompatibleRepository`, `GoogleDriveRepository`), họ phải review toàn bộ codebase để kiểm tra xem repository mới có vi phạm kỳ vọng của interface ở chỗ nào không. Hệ thống trở nên cực kỳ dễ vỡ: chỉ cần một method không được implement đúng trong một subclass có thể gây lỗi dây chuyền. Cuối cùng, đội ngũ phải dành 2 tuần để thiết kế lại toàn bộ hệ thống repository sử dụng interface segregation — tách `DocumentRepository` thành nhiều interface nhỏ hơn.

## Phân tích vấn đề

Root cause của vấn đề là **interface quá rộng và không phản ánh đúng khả năng của tất cả subtypes**. Một số subtype không thể hỗ trợ tất cả operations mà interface định nghĩa — nhưng vì bị ép implement tất cả, chúng phải chọn giữa ném exception, để method rỗng, hoặc implement sai. Cả ba lựa chọn đều vi phạm LSP:

1. **Subtype ném exception không mong đợi**: `raise NotImplementedError("Read-only!")` — client gọi `update()` một cách hợp lệ (theo interface contract) nhưng nhận exception. Hệ thống crash.
2. **Subtype implement rỗng**: `def update(self, doc): pass` — client gọi `update()` tưởng là thành công nhưng thực tế không có gì xảy ra. Dữ liệu mất.
3. **Subtype thay đổi hành vi**: `def update(self, doc): self._log("update attempted")` — thay vì cập nhật, chỉ log. Client nhận về `None` thay vì document đã update. Lỗi logic.

LSP không chỉ về cú pháp (implement đúng signature), mà về **semantics** (hành vi đúng theo contract). Interface `DocumentRepository` có một *contract* ngầm định: `save()` sẽ lưu document, `update()` sẽ cập nhật nó, `delete()` sẽ xóa nó. Nếu subtype vi phạm contract này, nó đã vi phạm LSP.

**Hệ quả của vi phạm LSP**:

- **Fragile Base Class Problem**: Sửa base class có thể gây lỗi ở bất kỳ subclass nào.
- **Lạm dụng isinstance()**: Client code phải kiểm tra type cụ thể để xử lý riêng, phá vỡ polymorphism.
- **Test complexity**: Không thể viết một test chung cho base type — mỗi subtype cần test riêng.
- **Hidden coupling**: Subtype có thể phụ thuộc vào implementation details của base class mà không được document.

## Giải pháp: Interface Segregation + Contract Design

Giải pháp cho LSP có hai phần: (1) thiết kế interface đúng ngay từ đầu (ISP — Interface Segregation giúp LSP), và (2) đảm bảo mỗi subtype tôn trọng contract của base type (Design by Contract).

**Bước 1 — Tách interface**: Thay vì một interface `DocumentRepository` khổng lồ, tách thành nhiều interface nhỏ, mỗi interface chỉ chứa các method có liên quan và có ý nghĩa với tất cả các subtype. Ví dụ:

- `ReadableRepository` — `get_by_id()`, `search()`
- `WritableRepository` — `save()`, `update()`
- `DeletableRepository` — `delete()`
- `ExportableRepository` — `export_csv()`, `generate_report()`

**Bước 2 — Design by Contract**: Mỗi interface nên có preconditions (điều kiện đầu vào), postconditions (điều kiện đầu ra), và invariants (bất biến) rõ ràng. Subtype không được làm chặt hơn preconditions (không yêu cầu điều kiện nhiều hơn) và không được làm lỏng hơn postconditions (không hứa kết quả ít hơn).

**Bước 3 — Composition over Inheritance**: Nếu một class không thể thay thế class cha, hãy dùng composition. Ví dụ: `ReadOnlyArchive` không nên kế thừa `DocumentRepository` — nên implement `ReadableRepository` riêng.

## Ví dụ code hoàn chỉnh

### VIOLATION — Vi phạm LSP

```python
# documents_violation.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    author: str
    created_at: datetime = field(default_factory=datetime.now)


class DocumentRepository(ABC):
    """Interface quá rộng — không phải subtype nào cũng hỗ trợ tất cả method."""

    @abstractmethod
    def save(self, doc: Document) -> str:
        """Lưu document mới, return ID."""
        ...

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Lấy document theo ID."""
        ...

    @abstractmethod
    def update(self, doc: Document) -> Document:
        """Cập nhật document, return document đã cập nhật."""
        ...

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Xóa document, return True nếu thành công."""
        ...

    @abstractmethod
    def search(self, query: str) -> list[Document]:
        """Tìm kiếm document theo query."""
        ...

    @abstractmethod
    def bulk_save(self, docs: list[Document]) -> int:
        """Lưu nhiều document, return số lượng đã lưu."""
        ...

    @abstractmethod
    def generate_report(self) -> str:
        """Tạo báo cáo thống kê."""
        ...


class LocalFileRepository(DocumentRepository):
    """Lưu trên file system — mọi method hoạt động bình thường."""

    def __init__(self, base_path: str) -> None:
        self._base_path = base_path
        self._store: dict[str, Document] = {}

    def save(self, doc: Document) -> str:
        doc_id = f"{doc.doc_id or len(self._store) + 1}"
        self._store[doc_id] = doc
        return doc_id

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        return self._store.get(doc_id)

    def update(self, doc: Document) -> Document:
        self._store[doc.doc_id] = doc
        return doc

    def delete(self, doc_id: str) -> bool:
        return self._store.pop(doc_id, None) is not None

    def search(self, query: str) -> list[Document]:
        return [d for d in self._store.values() if query.lower() in d.title.lower()]

    def bulk_save(self, docs: list[Document]) -> int:
        for doc in docs:
            self.save(doc)
        return len(docs)

    def generate_report(self) -> str:
        return f"Total documents: {len(self._store)}"


class ReadOnlyArchiveRepository(DocumentRepository):
    """
    VIOLATION LSP: subtype này không thể thay thế base type.
    3 method ném exception — client gặp lỗi runtime.
    """

    def __init__(self, archive_path: str) -> None:
        self._archive: dict[str, Document] = {}
        self._load_archive(archive_path)

    def _load_archive(self, path: str) -> None:
        # Giả lập load từ archive file
        self._archive['ARC-001'] = Document('ARC-001', 'Báo cáo 2024',
                                             'Nội dung...', 'admin')
        self._archive['ARC-002'] = Document('ARC-002', 'Hợp đồng cũ',
                                             'Nội dung...', 'legal')

    def save(self, doc: Document) -> str:
        raise NotImplementedError("Archive is read-only! Cannot save.")  # ❌

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        return self._archive.get(doc_id)

    def update(self, doc: Document) -> Document:
        raise NotImplementedError("Archive is read-only! Cannot update.")  # ❌

    def delete(self, doc_id: str) -> bool:
        raise NotImplementedError("Archive is read-only! Cannot delete.")  # ❌

    def search(self, query: str) -> list[Document]:
        return [d for d in self._archive.values()
                if query.lower() in d.title.lower()]

    def bulk_save(self, docs: list[Document]) -> int:
        raise NotImplementedError("Archive is read-only! Cannot bulk save.")  # ❌

    def generate_report(self) -> str:
        return f"Archive: {len(self._archive)} documents"


# Client code — crash khi gặp ReadOnlyArchiveRepository
def process_documents(repo: DocumentRepository, docs: list[Document]) -> None:
    """Hàm này được viết kỳ vọng làm việc với DocumentRepository bất kỳ."""
    # Lưu document
    for doc in docs:
        repo.save(doc)  # ❌ Crash nếu repo là ReadOnlyArchiveRepository

    # Tìm kiếm
    results = repo.search("báo cáo")
    print(f"Found {len(results)} documents")

    # Tạo báo cáo
    report = repo.generate_report()
    print(report)


# Sử dụng
local_repo = LocalFileRepository("/data/docs")
archive_repo = ReadOnlyArchiveRepository("/data/archive")

docs = [
    Document(doc_id='DOC-001', title='Kế hoạch Q1', content='...', author='admin'),
]

process_documents(local_repo, docs)  # ✅ OK
process_documents(archive_repo, docs)  # ❌ NotImplementedError: Archive is read-only!
```

### REFACTORED — Tuân thủ LSP

```python
# ─── domain/document.py ───
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str
    author: str
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime | None = None


# ─── repositories/interfaces.py ───
from __future__ import annotations
from typing import Optional, Protocol


class ReadableRepository(Protocol):
    """Chỉ đọc — phù hợp với archive, cache, view."""

    def get_by_id(self, doc_id: str) -> Optional[Document]: ...

    def search(self, query: str) -> list[Document]: ...


class WritableRepository(Protocol):
    """Chỉ ghi — phù hợp với log, event store append-only."""

    def save(self, doc: Document) -> str: ...

    def bulk_save(self, docs: list[Document]) -> int: ...


class UpdatableRepository(Protocol):
    """Có thể cập nhật và xóa."""

    def update(self, doc: Document) -> Document: ...

    def delete(self, doc_id: str) -> bool: ...


class FullRepository(ReadableRepository, WritableRepository, UpdatableRepository, Protocol):
    """Kết hợp — cho repository đầy đủ chức năng."""
    pass


# ─── repositories/local_file_repo.py ───
from __future__ import annotations
from typing import Optional


class LocalFileRepository:
    """Implement đầy đủ tất cả interface — có thể dùng như FullRepository."""

    def __init__(self, base_path: str) -> None:
        self._base_path = base_path
        self._store: dict[str, Document] = {}

    def save(self, doc: Document) -> str:
        doc_id = doc.doc_id or str(len(self._store) + 1)
        object.__setattr__(doc, 'doc_id', doc_id)
        object.__setattr__(doc, 'version', 1)
        self._store[doc_id] = doc
        return doc_id

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        return self._store.get(doc_id)

    def update(self, doc: Document) -> Document:
        existing = self._store.get(doc.doc_id)
        if existing is None:
            raise ValueError(f"Document {doc.doc_id} not found")
        object.__setattr__(doc, 'version', existing.version + 1)
        object.__setattr__(doc, 'updated_at', datetime.now())
        self._store[doc.doc_id] = doc
        return doc

    def delete(self, doc_id: str) -> bool:
        return self._store.pop(doc_id, None) is not None

    def search(self, query: str) -> list[Document]:
        return [d for d in self._store.values()
                if query.lower() in d.title.lower()]

    def bulk_save(self, docs: list[Document]) -> int:
        for doc in docs:
            self.save(doc)
        return len(docs)


# ─── repositories/read_only_archive.py ───
from __future__ import annotations
from typing import Optional


class ReadOnlyArchiveRepository:
    """
    Chỉ implement ReadableRepository — KHÔNG implement WritableRepository.
    Client chỉ xử lý archive qua ReadableRepository, không crash.
    """

    def __init__(self, archive_path: str) -> None:
        self._archive: dict[str, Document] = {}
        self._load_archive(archive_path)

    def _load_archive(self, path: str) -> None:
        self._archive['ARC-001'] = Document(
            doc_id='ARC-001', title='Báo cáo 2024',
            content='Báo cáo tài chính...', author='admin',
        )
        self._archive['ARC-002'] = Document(
            doc_id='ARC-002', title='Hợp đồng cũ',
            content='Nội dung hợp đồng...', author='legal',
        )

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        return self._archive.get(doc_id)

    def search(self, query: str) -> list[Document]:
        return [d for d in self._archive.values()
                if query.lower() in d.title.lower()]


# ─── services/document_service.py ───
from __future__ import annotations


class DocumentSearchService:
    """Chỉ cần ReadableRepository — có thể dùng với archive hoặc local."""

    def __init__(self, repo: ReadableRepository) -> None:
        self._repo = repo

    def find(self, query: str) -> list[Document]:
        return self._repo.search(query)


class DocumentManagementService:
    """Cần FullRepository — không thể dùng archive."""

    def __init__(self, repo: FullRepository) -> None:
        self._repo = repo

    def create(self, doc: Document) -> str:
        return self._repo.save(doc)

    def update(self, doc: Document) -> Document:
        return self._repo.update(doc)

    def delete(self, doc_id: str) -> bool:
        return self._repo.delete(doc_id)


# ─── main.py ───
from __future__ import annotations

# Với LocalFileRepository — dùng FullRepository
local_repo: FullRepository = LocalFileRepository("/data/docs")
search_svc = DocumentSearchService(local_repo)
mgmt_svc = DocumentManagementService(local_repo)

# Với Archive — chỉ dùng ReadableRepository
archive_repo: ReadableRepository = ReadOnlyArchiveRepository("/data/archive")
archive_search = DocumentSearchService(archive_repo)  # ✅ OK
# DocumentManagementService(archive_repo)  # ❌ Type error — đúng vì không thể update archive

# Client code an toàn
def print_search_results(service: DocumentSearchService, query: str) -> None:
    results = service.find(query)
    print(f"Found {len(results)} documents matching '{query}'")
    for doc in results:
        print(f"  - {doc.doc_id}: {doc.title} by {doc.author}")

print_search_results(search_svc, "báo cáo")
print_search_results(archive_search, "báo cáo")  # ✅ Không crash
```

## Dấu hiệu nhận biết vi phạm LSP

- **Subclass ném NotImplementedError**: Dấu hiệu rõ ràng nhất. Nếu subclass không thể implement method của base class, đó là LSP violation.
- **Subclass để method rỗng (pass)**: Method không làm gì cả — client gọi mà không biết không có tác dụng. Nguy hiểm hơn ném exception vì khó detect.
- **Lạm dụng isinstance()**: Code kiểu `if isinstance(obj, SomeSubclass): do_something_special()` — bạn đang phải check type thủ công vì subtype không thay thế được base type.
- **Override method và thay đổi hành vi cốt lõi**: Subclass override method nhưng thay đổi output format, thay đổi ý nghĩa của tham số, hoặc thêm preconditions mạnh hơn.
- **Subclass phá vỡ invariants của base class**: Ví dụ: base class đảm bảo `area() > 0`, subclass tạo shape có `area() = 0`. Hoặc base class đảm bảo `save()` tạo record trong DB, subclass không tạo.
- **Preconditions mạnh hơn**: Base class yêu cầu `x > 0`, subclass yêu cầu `x > 10` — code gọi với `x = 5` chạy được trên base, nhưng fail trên subclass.
- **Postconditions yếu hơn**: Base class hứa trả về string không rỗng, subclass trả về empty string.
- **Dùng "is-a" kiểu ngôn ngữ tự nhiên thay vì "is-substitutable"**: "A Square is-a Rectangle" — đúng trong toán học, sai trong OOP vì Square không thay thế được Rectangle.

## Kiểm thử

```python
# test_documents.py
from __future__ import annotations
import pytest  # type: ignore
from unittest.mock import MagicMock
from domain.document import Document
from repositories.interfaces import ReadableRepository, WritableRepository, FullRepository
from repositories.local_file_repo import LocalFileRepository
from repositories.read_only_archive import ReadOnlyArchiveRepository
from services.document_service import DocumentSearchService, DocumentManagementService


@pytest.fixture
def sample_doc() -> Document:
    return Document(doc_id='DOC-001', title='Test Doc', content='Content', author='tester')


@pytest.fixture
def local_repo() -> LocalFileRepository:
    return LocalFileRepository("/tmp/test_docs")


@pytest.fixture
def archive_repo() -> ReadOnlyArchiveRepository:
    return ReadOnlyArchiveRepository("/tmp/test_archive")


# ─── LSP Test: Mọi ReadableRepository phải hoạt động nhất quán ───

class ReadableRepositoryContract:
    """Contract test — mọi implementation của ReadableRepository phải pass."""

    def test_get_by_id_returns_document(self, repo: ReadableRepository, doc_id: str) -> None:
        result = repo.get_by_id(doc_id)
        assert result is not None
        assert isinstance(result, Document)
        assert result.doc_id == doc_id

    def test_get_by_id_nonexistent_returns_none(self, repo: ReadableRepository) -> None:
        result = repo.get_by_id("NONEXISTENT")
        assert result is None

    def test_search_returns_list(self, repo: ReadableRepository, query: str) -> None:
        results = repo.search(query)
        assert isinstance(results, list)
        if results:
            assert all(isinstance(d, Document) for d in results)

    def test_search_empty_query_returns_all(self, repo: ReadableRepository) -> None:
        results = repo.search("")
        # Implementation-specific — không assert cứng, chỉ check type
        assert isinstance(results, list)


class TestLocalFileRepositoryLSP:
    """Kiểm tra LocalFileRepository tuân thủ LSP contract."""

    @pytest.fixture
    def repo(self) -> LocalFileRepository:
        return LocalFileRepository("/tmp/test_local")

    def test_full_lifecycle(self, repo: LocalFileRepository, sample_doc: Document) -> None:
        # Create
        doc_id = repo.save(sample_doc)
        assert doc_id is not None

        # Read
        fetched = repo.get_by_id(doc_id)
        assert fetched is not None
        assert fetched.title == "Test Doc"

        # Update
        updated_doc = Document(doc_id=doc_id, title="Updated", content="New", author="tester")
        result = repo.update(updated_doc)
        assert result.version == 2  # version incremented

        # Search
        results = repo.search("Updated")
        assert len(results) == 1

        # Delete
        assert repo.delete(doc_id) is True
        assert repo.get_by_id(doc_id) is None


class TestReadOnlyArchiveLSP:
    """Kiểm tra ReadOnlyArchiveRepository tuân thủ LSP contract (ReadableRepository)."""

    @pytest.fixture
    def repo(self) -> ReadOnlyArchiveRepository:
        return ReadOnlyArchiveRepository("/tmp/test_archive")

    def test_get_existing_document(self, repo: ReadOnlyArchiveRepository) -> None:
        doc = repo.get_by_id('ARC-001')
        assert doc is not None
        assert doc.title == 'Báo cáo 2024'

    def test_get_nonexistent_returns_none(self, repo: ReadOnlyArchiveRepository) -> None:
        assert repo.get_by_id('FAKE') is None

    def test_search_works(self, repo: ReadOnlyArchiveRepository) -> None:
        results = repo.search('hợp đồng')
        assert len(results) == 1
        assert results[0].doc_id == 'ARC-002'

    def test_is_readable_protocol(self, repo: ReadOnlyArchiveRepository) -> None:
        """Đảm bảo ReadOnlyArchiveRepository đúng là ReadableRepository."""
        from typing import cast
        readable: ReadableRepository = cast(ReadableRepository, repo)
        doc = readable.get_by_id('ARC-001')
        assert doc is not None


class TestDocumentSearchService:
    """Service chỉ dùng ReadableRepository — có thể dùng cả local và archive."""

    def test_with_local_repo(self, local_repo: LocalFileRepository) -> None:
        local_repo.save(Document(doc_id='D1', title='Alpha', content='X', author='user'))
        service = DocumentSearchService(local_repo)
        results = service.find('Alpha')
        assert len(results) == 1

    def test_with_archive_repo(self, archive_repo: ReadOnlyArchiveRepository) -> None:
        service = DocumentSearchService(archive_repo)
        results = service.find('Báo cáo')
        assert len(results) == 1


class TestDocumentManagementService:
    """Service cần FullRepository — chỉ dùng được với local_repo."""

    def test_create_and_update(self, local_repo: LocalFileRepository, sample_doc: Document) -> None:
        service = DocumentManagementService(local_repo)
        doc_id = service.create(sample_doc)
        assert doc_id == 'DOC-001'

        updated = Document(doc_id='DOC-001', title='Updated', content='C', author='tester')
        result = service.update(updated)
        assert result.title == 'Updated'


# ─── LSP Property-Based Test ───

class TestLSPSubstitution:
    """Kiểm tra tính substitutability: code viết cho ReadableRepository
    phải chạy được với mọi implementation."""

    @pytest.mark.parametrize('repo_fixture', ['local_repo', 'archive_repo'])
    def test_search_accepts_any_readable(self, request: pytest.FixtureRequest, repo_fixture: str) -> None:
        """Cùng một test code, chạy được với cả local và archive — đó là LSP."""
        repo: ReadableRepository = request.getfixturevalue(repo_fixture)
        if repo_fixture == 'local_repo':
            repo.save(Document(doc_id='L1', title='Temporary', content='C', author='user'))

        service = DocumentSearchService(repo)
        results = service.find('a')
        assert isinstance(results, list)
        # Mọi ReadableRepository đều trả về list[Document] — correct
        for doc in results:
            assert isinstance(doc, Document)
```

## Ứng dụng thực tế

1. **SQLAlchemy — Identity Map và Session**: SQLAlchemy có các implementation khác nhau của `Session` — `Session` (có write), `AsyncSession` (async), `SessionTransaction` (transaction). Mỗi loại đều tuân thủ LSP: code viết cho `Session` hoạt động được với `AsyncSession` ở mức đủ. Vi phạm LSP thường xảy ra khi developer dùng `Session` nhưng thực tế là `ScopedSession` — dẫn đến lỗi thread-safety.

2. **Django — Model Backends**: Django có multiple database backends (PostgreSQL, MySQL, SQLite). Mỗi backend implementation phải tuân thủ LSP với `BaseDatabaseWrapper`. Tuy nhiên, lịch sử đã có nhiều lần vi phạm: ví dụ, MySQL không hỗ trợ savepoint transaction giống PostgreSQL — code dùng savepoint crash trên MySQL. Django giải quyết bằng cách giới hạn features (không phải subtype nào cũng hỗ trợ tất cả) và document rõ ràng.

3. **FastAPI — Response Models**: FastAPI cho phép trả về bất kỳ object nào — Pydantic model, dict, ORM model, raw response. LSP được đảm bảo nếu mọi response model implement cùng interface (có `.dict()` hoặc tương thích với `jsonable_encoder`). Khi developer trả về custom object không implement đúng contract, FastAPI crash khi cố serialize.

4. **Stream API Design**: `InputStream` interface trong Java/Python với `read()` method. `FileInputStream` và `NetworkInputStream` đều implement `read()` nhưng với behavior khác nhau: file stream có `available()` chính xác, network stream thì không. Client code dùng `available()` để quyết định blocking/non-blocking — điều này vi phạm LSP nếu network stream trả về kết quả sai. Giải pháp là tách thành `BlockingStream` và `NonBlockingStream`.

## Liên hệ với Pattern

- **Template Method Pattern**: Đảm bảo LSP bằng cách định nghĩa skeleton fix cứng trong base class, để subclass chỉ override các hook methods với contract rõ ràng. Base class gọi các hook methods và đảm bảo invariants.
- **Strategy Pattern**: Thay thế inheritance bằng composition — không có subclass, chỉ có các strategy độc lập implement cùng interface. Mỗi strategy đảm bảo contract của interface.
- **Null Object Pattern**: Trả về object "rỗng" thay vì `None` — đảm bảo LSP vì code client không cần check null. `NullLogger` có thể thay `RealLogger` trong mọi tình huống.
- **Composite Pattern**: Leaf và Composite đều implement component interface. LSP được đảm bảo nếu cả hai có behavior nhất quán — một leaf không ném exception khi gọi `add_child()`.
- **Interface Segregation (ISP)**: Tiền đề cho LSP. Interface nhỏ → dễ đảm bảo contract → dễ tuân thủ LSP. Interface lớn → ép subtype phải implement method không phù hợp → vi phạm LSP.
- **Design by Contract (DbC)**: Phương pháp của Bertrand Meyer — định nghĩa preconditions, postconditions, invariants cho mỗi method. Subtype không được làm chặt precondition, không được làm lỏng postcondition. DbC là công cụ chính để đảm bảo LSP.

## Ưu và nhược điểm

| Tiêu chí | Trước (vi phạm LSP) | Sau (tuân thủ LSP) |
|----------|---------------------|-------------------|
| **Type safety** | Runtime errors (NotImplementedError) | Compile-time/type-check safety |
| **Polymorphism** | Broken — phải dùng isinstance() | Full — mọi subtype thay thế được base |
| **Test code** | Phải viết test riêng cho từng subtype | Contract test chung cho tất cả |
| **Rủi ro khi thêm subtype** | Cao — có thể làm hỏng code client | Thấp — không ảnh hưởng |
| **Tính đúng đắn** | Không đảm bảo — method có thể no-op | Đảm bảo — mọi method đúng contract |
| **Số interface** | 1 interface khổng lồ | Nhiều interface nhỏ |
| **Hỗ trợ Duck Typing** | Kém — interface phức tạp | Tốt — interface nhỏ, dễ implement |
| **Khả năng reuse** | Thấp — phải implement cả interface | Cao — chỉ implement interface cần |
| **Documentation cần thiết** | Ít — interface tự document kém | Nhiều — cần contract rõ ràng |
| **Chi phí thiết kế** | Thấp ban đầu, cao về sau | Cao ban đầu, thấp về sau |
| **Phù hợp với** | Prototype, hệ thống nhỏ | Production, hệ thống lớn |

## Kết luận

LSP là nguyên lý thường bị hiểu lầm nhất trong SOLID. Nhiều developer nghĩ LSP chỉ là "đảm bảo kế thừa đúng cú pháp", nhưng thực tế LSP đòi hỏi nhiều hơn thế — nó đòi hỏi **behavioral subtyping**: subtype phải tôn trọng contract của base type, không chỉ signature. Hai nguyên tắc vàng để tránh vi phạm LSP: (1) **ưu tiên composition over inheritance** (các pattern Strategy, State, Decorator giúp tránh được hầu hết các lỗi LSP), và (2) **thiết kế interface nhỏ, chuyên biệt** (ISP) — một interface có 1-3 method liên quan chặt chẽ với nhau sẽ dễ đảm bảo LSP hơn nhiều so với interface "đa năng". Khi nghi ngờ, hãy tự hỏi: "Nếu tôi thay class cha bằng class con này, code có còn hoạt động đúng không?" Nếu câu trả lời là "phải thêm một đoạn xử lý đặc biệt", bạn đang vi phạm LSP. Hãy thiết kế để mọi subtype có thể thay thế base type một cách trong suốt, và hệ thống của bạn sẽ bền vững hơn rất nhiều trước sự thay đổi.
