---
id: state
title: State
sidebar_label: ⚡ State
sidebar_position: 21
---

# State

> "Allow an object to alter its behavior when its internal state changes. The object will appear to change its class."
> — **GoF**, *Design Patterns* (1994)

**State** là một behavioral pattern cho phép một đối tượng thay đổi hành vi của nó khi trạng thái nội tại thay đổi. Pattern này đóng gói mỗi trạng thái thành một class riêng, và ủy quyền (delegate) hành vi cho class trạng thái hiện tại. Object context sẽ "đổi class" khi trạng thái thay đổi.

---

## Bài toán chi tiết

Giả sử bạn đang xây dựng **hệ thống quản lý quy trình phê duyệt tài liệu** (Document Approval Workflow) cho một công ty bảo hiểm. Mỗi tài liệu yêu cầu bồi thường (claim document) đi qua nhiều trạng thái:

1. **Draft**: Người dùng tạo mới, có thể chỉnh sửa
2. **Pending Review**: Đã gửi lên quản lý, chờ xem xét
3. **Under Review**: Quản lý đang xem xét, không thể sửa
4. **Approved**: Đã duyệt, chờ thanh toán
5. **Rejected**: Từ chối, có thể sửa lại và gửi lại
6. **Paid**: Đã thanh toán — terminal state
7. **Archived**: Lưu trữ — terminal state

Mỗi trạng thái có các hành vi (method) khác nhau:

| Hành vi | Draft | Pending Review | Under Review | Approved | Rejected | Paid | Archived |
|---------|-------|----------------|-------------|----------|----------|------|----------|
| `edit()` | ✅ |❌ Chờ duyệt |❌ Đang review|❌ Đã duyệt |✅ |❌ Đã thanh toán|❌ Đã lưu trữ|
| `submit()` | ✅ |❌ Đã gửi |❌ Đang review|❌ Đã duyệt|✅ |❌ Đã thanh toán|❌ Đã lưu trữ|
| `approve()` |❌ Chưa gửi| ✅ |❌ Đang review|❌ Đã duyệt|❌ Đã từ chối|❌ Đã thanh toán|❌ Đã lưu trữ|
| `reject()` |❌ Chưa gửi| ✅ |❌ Đang review|❌ Đã duyệt|❌ Đã từ chối|❌ Đã thanh toán|❌ Đã lưu trữ|
| `pay()` |❌ Chưa duyệt|❌ Chưa duyệt|❌ Chưa duyệt|✅ |❌ Đã từ chối|❌ Đã thanh toán|❌ Đã lưu trữ|
| `archive()` |❌ Đang xử lý|❌ Đang xử lý|❌ Đang xử lý|✅ |✅ |✅ |❌ Đã lưu trữ|

Cách tiếp cận năng nề nhất là dùng `if-else` hoặc `match-case`:

```python
class NaiveDocument:
    def edit(self):
        if self.status == "DRAFT" or self.status == "REJECTED":
            print("Đã sửa tài liệu")
        elif self.status == "PENDING_REVIEW":
            raise Exception("Không thể sửa — đang chờ duyệt")
        elif self.status == "UNDER_REVIEW":
            raise Exception("Không thể sửa — đang được review")
        # ... còn nhiều nữa
```

Vấn đề của cách này:

1. **Violates Open/Closed Principle**: Mỗi lần thêm trạng thái mới (ví dụ: `PendingPayment`), bạn phải sửa tất cả method của `Document`.
2. **Code trùng lặp**: Các điều kiện giống nhau lặp lại ở mọi method. Ví dụ, kiểm tra `status == "PAID" or status == "ARCHIVED"` xuất hiện khắp nơi.
3. **Khả năng sai cao**: Dễ quên cập nhật một method khi thêm/xóa state.
4. **Không thể mở rộng**: Nếu muốn thêm hành vi `escalate()` (chuyển lên cấp trên), phải thêm `elif` vào tất cả các state.
5. **Khó kiểm thử**: Phải test tất cả tổ hợp (state × method) trong một class.

---

## Giải pháp với Pattern

State pattern tách mỗi trạng thái thành một class riêng biệt, implement cùng interface:

- **Context** (`Document`): Duy trì reference đến state hiện tại và ủy quyền (delegate) các method cho state đó
- **State interface** (`DocumentState`): Định nghĩa contract mà tất cả concrete state phải implement
- **Concrete States** (`DraftState`, `PendingReviewState`, ...): Implement hành vi cụ thể cho từng trạng thái

Khi trạng thái thay đổi, context thay đổi reference `state` sang một state object khác. Lần gọi method tiếp theo sẽ được chuyển hướng đến state mới.

---

## Phân tích thiết kế

### Nguyên lý OOP được áp dụng

- **Single Responsibility**: Mỗi class state chỉ chịu trách nhiệm cho hành vi của một trạng thái
- **Open/Closed Principle**: Thêm state mới = thêm class mới — không sửa code cũ
- **Strategy Pattern resemblance**: State và Strategy có cấu trúc giống nhau, nhưng khác về intent. State cho phép context tự động chuyển đổi; Strategy do client chọn.
- **Polymorphism**: Hành vi thay đổi dựa trên runtime type của state

### Trade-offs

1. **Class explosion**: Mỗi state là một class mới. Với hệ thống có 10-15 state, số class tăng đáng kể.
2. **Context-state coupling**: Context phải expose dữ liệu cho state (thường qua parameter). Có thể vi phạm encapsulation.
3. **State transition logic phân tán**: Ai quyết định chuyển state? Có thể để state tự quyết định, hoặc context quyết định, hoặc có state machine riêng.

### Khi nào KHÔNG dùng

- Khi chỉ có 2-3 trạng thái đơn giản — dùng enum + if-else dễ hơn
- Khi hành vi không thay đổi theo trạng thái
- Khi state transition là deterministic và đơn giản (linear flow)
- Khi performance là ưu tiên số 1 (state object allocation overhead)

---

## Ví dụ code hoàn chỉnh

### Cách sai: If-else với Enum

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from typing import Optional


class DocumentStatus(Enum):
    DRAFT = auto()
    PENDING_REVIEW = auto()
    UNDER_REVIEW = auto()
    APPROVED = auto()
    REJECTED = auto()
    PAID = auto()
    ARCHIVED = auto()


@dataclass
class NaiveDocument:
    title: str
    content: str
    author: str
    status: DocumentStatus = DocumentStatus.DRAFT
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    approved_amount: Optional[float] = None
    paid_at: Optional[datetime] = None
    version: int = 1

    def edit(self, new_content: str) -> None:
        if self.status == DocumentStatus.DRAFT:
            self.content = new_content
            self.version += 1
            print(f"📝 Đã cập nhật nội dung (v{self.version})")
        elif self.status == DocumentStatus.PENDING_REVIEW:
            raise RuntimeError("Không thể sửa — tài liệu đang chờ duyệt")
        elif self.status == DocumentStatus.UNDER_REVIEW:
            raise RuntimeError("Không thể sửa — tài liệu đang được review")
        elif self.status == DocumentStatus.APPROVED:
            raise RuntimeError("Không thể sửa — tài liệu đã duyệt")
        elif self.status == DocumentStatus.REJECTED:
            self.content = new_content
            self.version += 1
            print("📝 Đã cập nhật nội dung (tài liệu bị từ chối trước đó)")
        elif self.status == DocumentStatus.PAID:
            raise RuntimeError("Không thể sửa — tài liệu đã thanh toán")
        elif self.status == DocumentStatus.ARCHIVED:
            raise RuntimeError("Không thể sửa — tài liệu đã lưu trữ")

    def submit(self) -> None:
        if self.status == DocumentStatus.DRAFT:
            self.status = DocumentStatus.PENDING_REVIEW
            print("📤 Đã gửi tài liệu chờ duyệt")
        elif self.status == DocumentStatus.PENDING_REVIEW:
            raise RuntimeError("Tài liệu đã được gửi")
        elif self.status == DocumentStatus.REJECTED:
            self.status = DocumentStatus.PENDING_REVIEW
            print("📤 Đã gửi lại tài liệu sau khi sửa")
        else:
            raise RuntimeError(f"Không thể gửi ở trạng thái {self.status.name}")

    # ... mỗi method đều có if-else dài như vậy
```

### Cách đúng: State Pattern

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any


# ============================================================
# State Interface
# ============================================================
class DocumentState(ABC):
    """Interface cho tất cả trạng thái của Document"""

    @abstractmethod
    def edit(self, doc: 'Document', new_content: str) -> None:
        pass

    @abstractmethod
    def submit(self, doc: 'Document') -> None:
        pass

    @abstractmethod
    def approve(self, doc: 'Document', reviewer: str) -> None:
        pass

    @abstractmethod
    def reject(self, doc: 'Document', reviewer: str, reason: str) -> None:
        pass

    @abstractmethod
    def pay(self, doc: 'Document', amount: float) -> None:
        pass

    @abstractmethod
    def archive(self, doc: 'Document') -> None:
        pass

    @property
    @abstractmethod
    def status_name(self) -> str:
        """Tên trạng thái để hiển thị"""
        pass

    @property
    @abstractmethod
    def can_edit(self) -> bool:
        pass

    @property
    @abstractmethod
    def can_submit(self) -> bool:
        pass


# ============================================================
# Concrete States
# ============================================================
class DraftState(DocumentState):
    """Trạng thái nháp — có thể edit và submit"""

    @property
    def status_name(self) -> str:
        return "DRAFT"

    @property
    def can_edit(self) -> bool:
        return True

    @property
    def can_submit(self) -> bool:
        return True

    def edit(self, doc: 'Document', new_content: str) -> None:
        doc.content = new_content
        doc.version += 1
        print("📝 [DRAFT] Đã cập nhật nội dung")

    def submit(self, doc: 'Document') -> None:
        doc.state = PendingReviewState()
        print("📤 [DRAFT → PENDING_REVIEW] Tài liệu đã gửi chờ duyệt")

    def approve(self, doc: 'Document', reviewer: str) -> None:
        raise RuntimeError("Tài liệu chưa được gửi — không thể duyệt")

    def reject(self, doc: 'Document', reviewer: str, reason: str) -> None:
        raise RuntimeError("Tài liệu chưa được gửi — không thể từ chối")

    def pay(self, doc: 'Document', amount: float) -> None:
        raise RuntimeError("Tài liệu chưa được duyệt — không thể thanh toán")

    def archive(self, doc: 'Document') -> None:
        doc.state = ArchivedState()
        print("📦 [DRAFT → ARCHIVED] Đã lưu trữ tài liệu nháp")


class PendingReviewState(DocumentState):
    """Trạng thái chờ duyệt — chỉ có thể approve/reject"""

    @property
    def status_name(self) -> str:
        return "PENDING_REVIEW"

    @property
    def can_edit(self) -> bool:
        return False

    @property
    def can_submit(self) -> bool:
        return False

    def edit(self, doc: 'Document', new_content: str) -> None:
        raise RuntimeError("Không thể sửa — tài liệu đang chờ duyệt")

    def submit(self, doc: 'Document') -> None:
        raise RuntimeError("Tài liệu đã được gửi — đang chờ duyệt")

    def approve(self, doc: 'Document', reviewer: str) -> None:
        doc.reviewer = reviewer
        doc.reviewed_at = datetime.now()
        doc.state = UnderReviewState()
        print(f"✅ [PENDING_REVIEW → UNDER_REVIEW] {reviewer} bắt đầu review")

    def reject(self, doc: 'Document', reviewer: str, reason: str) -> None:
        doc.reviewer = reviewer
        doc.reviewed_at = datetime.now()
        doc.rejection_reason = reason
        doc.state = RejectedState()
        print(f"❌ [PENDING_REVIEW → REJECTED] {reviewer} từ chối: {reason}")

    def pay(self, doc: 'Document', amount: float) -> None:
        raise RuntimeError("Tài liệu chưa được duyệt")

    def archive(self, doc: 'Document') -> None:
        raise RuntimeError("Tài liệu đang chờ xử lý — không thể lưu trữ")


class UnderReviewState(DocumentState):
    """Trạng thái đang review"""

    @property
    def status_name(self) -> str:
        return "UNDER_REVIEW"

    @property
    def can_edit(self) -> bool:
        return False

    @property
    def can_submit(self) -> bool:
        return False

    def edit(self, doc: 'Document', new_content: str) -> None:
        raise RuntimeError("Không thể sửa — tài liệu đang được review")

    def submit(self, doc: 'Document') -> None:
        raise RuntimeError("Tài liệu đã được gửi")

    def approve(self, doc: 'Document', reviewer: str) -> None:
        doc.reviewer = reviewer
        doc.reviewed_at = datetime.now()
        doc.state = ApprovedState()
        print(f"👍 [UNDER_REVIEW → APPROVED] {reviewer} đã duyệt")

    def reject(self, doc: 'Document', reviewer: str, reason: str) -> None:
        doc.reviewer = reviewer
        doc.reviewed_at = datetime.now()
        doc.rejection_reason = reason
        doc.state = RejectedState()
        print(f"👎 [UNDER_REVIEW → REJECTED] {reviewer} từ chối: {reason}")

    def pay(self, doc: 'Document', amount: float) -> None:
        raise RuntimeError("Tài liệu chưa được duyệt")

    def archive(self, doc: 'Document') -> None:
        raise RuntimeError("Tài liệu đang được review — không thể lưu trữ")


class ApprovedState(DocumentState):
    """Đã duyệt — chờ thanh toán hoặc lưu trữ"""

    @property
    def status_name(self) -> str:
        return "APPROVED"

    @property
    def can_edit(self) -> bool:
        return False

    @property
    def can_submit(self) -> bool:
        return False

    def edit(self, doc: 'Document', new_content: str) -> None:
        raise RuntimeError("Tài liệu đã duyệt — không thể sửa")

    def submit(self, doc: 'Document') -> None:
        raise RuntimeError("Tài liệu đã duyệt")

    def approve(self, doc: 'Document', reviewer: str) -> None:
        raise RuntimeError("Tài liệu đã được duyệt rồi")

    def reject(self, doc: 'Document', reviewer: str, reason: str) -> None:
        doc.state = RejectedState()
        doc.rejection_reason = reason
        print(f"🔄 [APPROVED → REJECTED] Đã thu hồi phê duyệt: {reason}")

    def pay(self, doc: 'Document', amount: float) -> None:
        doc.approved_amount = amount
        doc.paid_at = datetime.now()
        doc.state = PaidState()
        print(f"💰 [APPROVED → PAID] Đã thanh toán {amount:,.0f} VND")

    def archive(self, doc: 'Document') -> None:
        doc.state = ArchivedState()
        print(f"📦 [APPROVED → ARCHIVED] Đã lưu trữ tài liệu đã duyệt")


class RejectedState(DocumentState):
    """Bị từ chối — có thể sửa và gửi lại"""

    @property
    def status_name(self) -> str:
        return "REJECTED"

    @property
    def can_edit(self) -> bool:
        return True

    @property
    def can_submit(self) -> bool:
        return True

    def edit(self, doc: 'Document', new_content: str) -> None:
        doc.content = new_content
        doc.version += 1
        print(f"📝 [REJECTED] Đã sửa lại tài liệu (v{doc.version})")

    def submit(self, doc: 'Document') -> None:
        doc.state = PendingReviewState()
        print(f"📤 [REJECTED → PENDING_REVIEW] Đã gửi lại sau khi sửa")

    def approve(self, doc: 'Document', reviewer: str) -> None:
        raise RuntimeError("Tài liệu đã bị từ chối — hãy yêu cầu gửi lại")

    def reject(self, doc: 'Document', reviewer: str, reason: str) -> None:
        raise RuntimeError("Tài liệu đã bị từ chối rồi")

    def pay(self, doc: 'Document', amount: float) -> None:
        raise RuntimeError("Tài liệu bị từ chối — không thể thanh toán")

    def archive(self, doc: 'Document') -> None:
        doc.state = ArchivedState()
        print(f"📦 [REJECTED → ARCHIVED] Đã lưu trữ tài liệu bị từ chối")


class PaidState(DocumentState):
    """Đã thanh toán — terminal state"""

    @property
    def status_name(self) -> str:
        return "PAID"

    @property
    def can_edit(self) -> bool:
        return False

    @property
    def can_submit(self) -> bool:
        return False

    def edit(self, doc: 'Document', new_content: str) -> None:
        raise RuntimeError("Tài liệu đã thanh toán — không thể sửa")

    def submit(self, doc: 'Document') -> None:
        raise RuntimeError("Tài liệu đã thanh toán")

    def approve(self, doc: 'Document', reviewer: str) -> None:
        raise RuntimeError("Tài liệu đã thanh toán")

    def reject(self, doc: 'Document', reviewer: str, reason: str) -> None:
        raise RuntimeError("Tài liệu đã thanh toán")

    def pay(self, doc: 'Document', amount: float) -> None:
        raise RuntimeError("Tài liệu đã thanh toán rồi")

    def archive(self, doc: 'Document') -> None:
        doc.state = ArchivedState()
        print(f"📦 [PAID → ARCHIVED] Đã lưu trữ tài liệu đã thanh toán")


class ArchivedState(DocumentState):
    """Đã lưu trữ — terminal state, read-only"""

    @property
    def status_name(self) -> str:
        return "ARCHIVED"

    @property
    def can_edit(self) -> bool:
        return False

    @property
    def can_submit(self) -> bool:
        return False

    def edit(self, doc: 'Document', new_content: str) -> None:
        raise RuntimeError("Tài liệu đã lưu trữ — không thể sửa")

    def submit(self, doc: 'Document') -> None:
        raise RuntimeError("Tài liệu đã lưu trữ")

    def approve(self, doc: 'Document', reviewer: str) -> None:
        raise RuntimeError("Tài liệu đã lưu trữ")

    def reject(self, doc: 'Document', reviewer: str, reason: str) -> None:
        raise RuntimeError("Tài liệu đã lưu trữ")

    def pay(self, doc: 'Document', amount: float) -> None:
        raise RuntimeError("Tài liệu đã lưu trữ")

    def archive(self, doc: 'Document') -> None:
        print("Tài liệu đã được lưu trữ trước đó")


# ============================================================
# Context
# ============================================================
@dataclass
class Document:
    title: str
    content: str
    author: str
    state: DocumentState = field(default_factory=DraftState)
    version: int = 1
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    approved_amount: Optional[float] = None
    paid_at: Optional[datetime] = None

    def edit(self, new_content: str) -> None:
        self.state.edit(self, new_content)

    def submit(self) -> None:
        self.state.submit(self)

    def approve(self, reviewer: str) -> None:
        self.state.approve(self, reviewer)

    def reject(self, reviewer: str, reason: str) -> None:
        self.state.reject(self, reviewer, reason)

    def pay(self, amount: float) -> None:
        self.state.pay(self, amount)

    def archive(self) -> None:
        self.state.archive(self)

    def status(self) -> str:
        return self.state.status_name


# ============================================================
# Usage
# ============================================================
def main() -> None:
    doc = Document(
        title="Yêu cầu bồi thường bảo hiểm",
        content="Chi tiết yêu cầu bồi thường cho vụ tai nạn...",
        author="Nguyen Van A"
    )

    print(f"\n=== Tạo tài liệu: {doc.title} ===")
    print(f"Trạng thái: {doc.status()}")
    print(f"Có thể sửa? {doc.state.can_edit}")
    print(f"Có thể gửi? {doc.state.can_submit}")

    print(f"\n=== Bước 1: Sửa nội dung ===")
    doc.edit("Nội dung cập nhật lần 1")

    print(f"\n=== Bước 2: Gửi duyệt ===")
    doc.submit()

    print(f"\n=== Bước 3: Review ===")
    doc.approve("Tran Thi B")

    print(f"\n=== Bước 4: Thanh toán ===")
    doc.pay(15_000_000)

    print(f"\n=== Bước 5: Lưu trữ ===")
    doc.archive()

    print(f"\n=== Kiểm tra terminal state ===")
    try:
        doc.edit("Không thể sửa")
    except RuntimeError as e:
        print(f"⛔ Lỗi như mong đợi: {e}")

    print(f"\n--- Trạng thái cuối: {doc.status()} ---")
    print(f"Người duyệt: {doc.reviewer}")
    print(f"Số tiền: {doc.approved_amount:,.0f} VND")


if __name__ == "__main__":
    main()
```

---

## Sơ đồ UML

```
┌──────────────────────────────┐
│       Document (Context)     │
├──────────────────────────────┤
│ - state: DocumentState       │
│ - title: str                 │
│ - content: str               │
│ - author: str                │
│ - version: int               │
│ - reviewer: Optional[str]    │
│ - rejection_reason: ...      │
│ - approved_amount: ...       │
├──────────────────────────────┤
│ + edit(content)              │
│ + submit()                   │
│ + approve(reviewer)          │
│ + reject(reviewer, reason)   │
│ + pay(amount)                │
│ + archive()                  │
│ + status(): str              │
└──────┬───────────────────────┘
       │  state ──────┐
       ▼              ▼
┌────────────────────────────────────┐
│  <<interface>>                     │
│  DocumentState                     │
├────────────────────────────────────┤
│ + edit(doc, content)               │
│ + submit(doc)                      │
│ + approve(doc, reviewer)           │
│ + reject(doc, reviewer, reason)    │
│ + pay(doc, amount)                 │
│ + archive(doc)                     │
│ + status_name: str                 │
│ + can_edit: bool                   │
│ + can_submit: bool                 │
└──────────┬─────────────────────────┘
           │
    ┌──────┼──────────┬──────────┬──────────┬──────────┬──────────┐
    ▼      ▼          ▼          ▼          ▼          ▼          ▼
┌──────┐┌────────┐┌─────────┐┌──────────┐┌──────────┐┌──────────┐
│Draft ││Pending ││Under    ││Approved  ││Rejected  ││Paid      │
│State ││Review  ││Review   ││State     ││State     ││State     │
│      ││State   ││State    ││          ││          ││          │
└──────┘└────────┘└─────────┘└──────────┘└──────────┘└──────────┘
```

---

## So sánh với Pattern liên quan

### 1. State vs Strategy

| Tiêu chí | State | Strategy |
|----------|-------|----------|
| Intent | Thay đổi hành vi khi state thay đổi | Chọn thuật toán từ họ các thuật toán |
| Ai chọn implementation? | Context tự động chuyển state | Client chọn strategy |
| State biết về nhau | ✅ State thường biết state kế tiếp | ❌ Strategy độc lập, không biết nhau |
| Số lượng object | Một state active tại một thời điểm | Một strategy được chọn |

**Điểm giống**: Cả hai đều dùng composition, delegate hành vi cho object khác, và có cấu trúc class giống nhau.

**Cách phân biệt**: Hãy tự hỏi — "Object có tự động đổi implementation hay không?" Nếu **có** → State. Nếu **do client chọn** → Strategy.

### 2. State vs Finite State Machine (FSM)

FSM là khái niệm rộng hơn, có thể implement bằng:
- **Bảng chuyển trạng thái** (State Transition Table): Dùng ma trận [state × event] → next state
- **State pattern**: Dùng OOP, mỗi state là một class

| Tiêu chí | State Pattern | State Table (FSM) |
|----------|--------------|-------------------|
| Khi state có hành vi phức tạp | ✅ Tốt | ❌ Khó |
| Dễ đọc với ít state | ❌ Overkill | ✅ Đơn giản |
| Thêm state/hành vi mới | Thêm class | Thêm dòng/cột |
| Xử lý độc lập | ✅ | Dùng callback |

### 3. State vs Command

Command đóng gói một request thành object. State đóng gói toàn bộ hành vi của một trạng thái.

- **Command**: Một lệnh, có undo. Phù hợp hành động đơn lẻ.
- **State**: Một trạng thái, có nhiều hành động. Phù hợp workflow.

**Kết hợp**: Có thể dùng Command Pattern để implement các hành động trong State. Ví dụ, mỗi method trong `DocumentState` trả về một Command object.

---

## Ứng dụng thực tế

### 1. Django FSM (django-fsm)

Thư viện `django-fsm` implement State pattern cho Django models:

```python
from django.db import models
from django_fsm import FSMField, transition

class Document(models.Model):
    state = FSMField(default='draft')

    @transition(field=state, source='draft', target='pending_review')
    def submit(self):
        """Gửi tài liệu"""

    @transition(field=state, source='pending_review', target='approved')
    def approve(self):
        """Duyệt tài liệu"""

    @transition(field=state, source='*', target='archived')
    def archive(self):
        """Lưu trữ — có thể archive từ mọi state"""

# Sử dụng
doc = Document.objects.create()
doc.submit()  # state → pending_review
doc.approve()  # state → approved
doc.archive()  # state → archived
doc.submit()  # raise TransitionNotAllowed — 'approved' không thể submit
```

### 2. UI Component States (React/Vue)

UI components thường có state machine: `loading → ready → error`

```python
from enum import Enum, auto

class UIState(Enum):
    IDLE = auto()
    LOADING = auto()
    READY = auto()
    ERROR = auto()

# Mỗi state có hành vi render khác nhau
# State pattern xuất hiện trong Redux, Vuex,...
```

### 3. Game Character States (Unity/Godot)

Trong game, nhân vật có state machine: `Idle → Running → Jumping → Falling`:

```python
# Mỗi state có update logic riêng
class CharacterState(ABC):
    @abstractmethod
    def handle_input(self, character, event): pass
    @abstractmethod
    def update(self, character, dt): pass

class IdleState(CharacterState):
    def handle_input(self, character, event):
        if event == "PRESS_UP":
            return JumpingState()
        if event == "PRESS_LEFT" or event == "PRESS_RIGHT":
            return RunningState()
        return self

    def update(self, character, dt):
        # Giảm stamina, hồi phục...
        pass
```

### 4. Workflow Engine (Camunda, Temporal)

Các workflow engine như Camunda BPMN, Temporal.io implement State pattern ở mức độ cao hơn:

```python
# Temporal.io Workflow (Go)
# Mỗi workflow state là một trạng thái
func OrderWorkflow(ctx workflow.Context, input OrderInput) error {
    state := NewPendingState()
    for state != nil {
        state = state.Execute(ctx)
    }
    return nil
}
```

---

## Kiểm thử

```python
import unittest
from datetime import datetime


class TestDocumentStatePattern(unittest.TestCase):
    def setUp(self):
        self.doc = Document(
            title="Test Document",
            content="Test content",
            author="Tester"
        )

    def test_initial_state(self):
        """Document khởi tạo ở trạng thái DRAFT"""
        self.assertEqual(self.doc.status(), "DRAFT")
        self.assertTrue(self.doc.state.can_edit)
        self.assertTrue(self.doc.state.can_submit)

    def test_draft_to_archived(self):
        """DRAFT → ARCHIVED là hợp lệ"""
        self.doc.archive()
        self.assertEqual(self.doc.status(), "ARCHIVED")

    def test_full_approval_flow(self):
        """Luồng phê duyệt hoàn chỉnh"""
        self.doc.edit("Updated content")
        self.doc.submit()
        self.assertEqual(self.doc.status(), "PENDING_REVIEW")
        self.doc.approve("Reviewer 1")
        self.assertEqual(self.doc.status(), "UNDER_REVIEW")
        self.doc.approve("Reviewer 2")
        self.assertEqual(self.doc.status(), "APPROVED")
        self.doc.pay(10_000_000)
        self.assertEqual(self.doc.status(), "PAID")

    def test_cannot_edit_after_submit(self):
        """Không thể sửa sau khi đã gửi duyệt"""
        self.doc.submit()
        with self.assertRaises(RuntimeError) as ctx:
            self.doc.edit("Should fail")
        self.assertIn("đang chờ duyệt", str(ctx.exception))

    def test_reject_then_resubmit(self):
        """Bị từ chối — sửa lại — gửi lại"""
        self.doc.submit()
        self.doc.reject("Reviewer", "Sai định dạng")
        self.assertEqual(self.doc.status(), "REJECTED")
        self.assertEqual(self.doc.rejection_reason, "Sai định dạng")
        self.doc.edit("Fixed content")
        self.doc.submit()
        self.assertEqual(self.doc.status(), "PENDING_REVIEW")

    def test_cannot_pay_before_approval(self):
        """Không thể thanh toán trước khi duyệt"""
        with self.assertRaises(RuntimeError):
            self.doc.pay(100_000)

    def test_approved_to_archived(self):
        """APPROVED có thể archive"""
        self.doc.submit()
        self.doc.approve("Reviewer")
        self.doc.approve("Reviewer")
        self.doc.archive()
        self.assertEqual(self.doc.status(), "ARCHIVED")

    def test_reject_after_approval(self):
        """Có thể thu hồi phê duyệt (approve → reject)"""
        self.doc.submit()
        self.doc.approve("R1")
        self.doc.approve("R2")  # UnderReview → Approved
        self.doc.reject("Manager", "Phát hiện sai sót")
        self.assertEqual(self.doc.status(), "REJECTED")

    def test_paid_to_archived(self):
        """PAID → ARCHIVED"""
        self.doc.submit()
        self.doc.approve("R1")
        self.doc.approve("R2")
        self.doc.pay(5_000_000)
        self.doc.archive()
        self.assertEqual(self.doc.status(), "ARCHIVED")

    def test_terminal_states_immutable(self):
        """Terminal state không thể thay đổi"""
        self.doc.archive()
        with self.assertRaises(RuntimeError):
            self.doc.edit("Nope")
        with self.assertRaises(RuntimeError):
            self.doc.submit()

    def test_rejection_reason_stored(self):
        """Lý do từ chối được lưu lại"""
        self.doc.submit()
        self.doc.reject("Reviewer", "Thiếu giấy tờ")
        self.assertEqual(self.doc.rejection_reason, "Thiếu giấy tờ")
        self.assertIsInstance(self.doc.reviewed_at, datetime)


if __name__ == "__main__":
    unittest.main()
```

---

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Open/Closed**: Thêm state = thêm class, không sửa code cũ | **Class explosion**: Mỗi state một class — nhiều class cho hệ thống lớn |
| **Loại bỏ if-else/switch**: Hành vi state được đóng gói | **Độ phức tạp cao hơn**: Nhiều class hơn, khó follow flow hơn |
| **Tổ chức code tốt hơn**: Code state gom vào một chỗ | **Context phải expose dữ liệu**: State cần truy cập context data |
| **Dễ bảo trì**: Sửa logic một state không ảnh hưởng state khác | **Không phù hợp state đơn giản**: Nếu chỉ 2-3 state, if-else đơn giản hơn |
| **Dễ kiểm thử**: Test từng state riêng biệt | **State transition phân tán**: Khó nhìn toàn cảnh transitions |
| **State machine rõ ràng**: Visualize được các transitions | **Tốn bộ nhớ**: Mỗi context giữ reference đến state object |

---

## Kết luận

State pattern là công cụ mạnh mẽ để xử lý các hệ thống có nhiều trạng thái với hành vi phức tạp. Pattern này chuyển đổi những đoạn `if-else` dài vô tận thành các class nhỏ gọn, dễ bảo trì.

### Khi nào áp dụng

- ✅ Object có từ 4-5 trạng thái trở lên, mỗi trạng thái có hành vi riêng
- ✅ Code có nhiều `if status == ...` hoặc `switch(status)` lặp đi lặp lại
- ✅ Các trạng thái có transition phức tạp, không tuyến tính
- ✅ Cần thêm state mới thường xuyên
- ✅ Cần kiểm soát chặt chẽ state transitions (ví dụ: hợp đồng thông minh, workflow pháp lý)

### Golden Rules

1. **State transition ở đâu?** Để state tự quyết định transition (như ví dụ trên) hoặc để context quyết định qua transition table. Chọn một cách và nhất quán.
2. **Terminal states**: Luôn định nghĩa rõ state nào là terminal (không thể thoát). Terminal state giúp tránh bug.
3. **Không dùng State pattern cho linear flow đơn giản**: Nếu state A → B → C → D không rẽ nhánh, dùng enum + method đơn giản hơn.
4. **Kết hợp với Factory Method**: Dùng factory để tạo state object, đặc biệt nếu state cần dependency injection.
5. **Document transitions**: Vẽ state machine diagram trước khi code để thống nhất với team.
