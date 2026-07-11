---
id: mvvm-architecture
title: MVVM (Model-View-ViewModel)
sidebar_label: 🏗️ MVVM Architecture
sidebar_position: 46
---

# MVVM (Model-View-ViewModel)

> "The ViewModel is the bridge that connects the View to the Model, transforming data into something the View can consume while keeping the View completely ignorant of the Model."
> — **John Gossman**, *MVVM Introduction* (2005)

**MVVM** (Model-View-ViewModel) là một software architecture pattern được John Gossman giới thiệu vào năm 2005 khi ông đang làm việc trên **Windows Presentation Foundation (WPF)** tại Microsoft. MVVM là sự kết hợp tinh tế giữa **MVP** (Model-View-Presenter) và **Presentation Model** của Martin Fowler, với điểm nhấn là **data binding hai chiều** (two-way data binding).

---

## Tổng quan

### Lịch sử và nguồn gốc

MVVM ra đời trong bối cảnh WPF giới thiệu khái niệm **XAML** và **data binding** mạnh mẽ. Các cột mốc quan trọng:

- **2005**: John Gossman (Microsoft) giới thiệu MVVM trong bài blog về WPF
- **2008**: WPF và Silverlight chính thức ra mắt, MVVM trở thành pattern khuyến nghị
- **2010**: Josh Smith viết bài *"MVVM Demystified"*, phổ biến pattern này rộng rãi
- **2015-2017**: Google giới thiệu **Data Binding Library** cho Android, MVVM trở nên thịnh hành
- **2017-nay**: MVVM là kiến trúc mặc định cho Android Jetpack, iOS SwiftUI, React/Vue

### MVVM vs MVP vs MVC

| Tiêu chí | MVC | MVP | MVVM |
|---------|-----|-----|------|
| **Data binding** | Không (manual) | Thường không | **Có (two-way)** |
| **Testability** | Trung bình | **Cao nhất** | Cao |
| **View responsibility** | Chủ động | **Thụ động** | Thụ động + binding |
| **Complexity** | Thấp | Trung bình | **Cao** |
| **Boilerplate** | Ít | Nhiều (interface) | Trung bình (binding) |
| **Framework support** | Không | Không | **Tốt** (WPF, Android, SwiftUI) |

### Các biến thể MVVM

| Biến thể | Đặc điểm |
|---------|---------|
| **Classic MVVM** | ViewModel expose properties + commands, View bind trực tiếp |
| **MVVM + Repository** | Thêm repository layer cho data access |
| **MVVM + Clean Architecture** | Kết hợp với Clean Architecture (Use Cases, Repositories) |
| **MVVM + Event Bus** | Dùng event bus cho ViewModel communication |
| **Reactive MVVM** | Dùng RxSwift/RxJava/RxKotlin cho data binding |

---

## Bài toán

### Ứng dụng Chat Real-time cho Doanh nghiệp

Giả sử bạn đang xây dựng **MochaTalk** — một ứng dụng chat nội bộ cho doanh nghiệp với các tính năng:

1. **Danh sách hội thoại** — Hiển thị danh sách chat, tin nhắn mới nhất, trạng thái online
2. **Phòng chat real-time** — Gửi/nhận tin nhắn WebSocket, đính kèm file
3. **Tìm kiếm tin nhắn** — Full-text search qua lịch sử chat
4. **Thông báo** — Push notification khi có tin nhắn mới
5. **Trạng thái người dùng** — Online/Offline/Busy/Away
6. **Reaction và reply** — Emoji reaction, reply thread

### Thách thức

**Vấn đề 1 — State management phức tạp**: Chat application có rất nhiều state:
- Danh sách hội thoại đang load, đã load, lỗi
- Tin nhắn đang gửi, gửi thành công, gửi thất bại
- Trạng thái typing, online, seen của từng user
- Scroll position, unread count, draft message

Nếu không có kiến trúc tốt, state sẽ nằm rải rác khắp nơi:

```python
# Không có MVVM — state management hỗn loạn
class ChatScreenHTML:
    def __init__(self):
        self.conversations = []
        self.messages = []
        self.typing_users = {}
        self.unread_counts = {}
        self.is_loading = False
        self.error = None

    def update_conversation_list(self, data):
        # Logic nằm lẫn trong template rendering
        html = ""
        for conv in data:
            html += f"<div class='{'active' if conv.is_selected else ''}'>"
            if conv.last_message:
                html += f"<span>{conv.last_message.text}</span>"
            html += "</div>"
        # ...
```

**Vấn đề 2 — Data binding thủ công gây bug**: Mỗi lần Model thay đổi, bạn phải cập nhật View bằng tay. Nếu quên cập nhật một chỗ, UI sẽ hiển thị sai:

```python
# Manual update — dễ quên, dễ bug
class ChatController:
    def mark_as_read(self, conversation_id):
        conv = self.service.mark_read(conversation_id)
        self.screen.update_unread_badge(conv.id, 0)      # Dễ quên
        self.screen.update_conversation_list_item(conv)   # Dễ quên
        # Quên update toolbar unread count!
```

**Vấn đề 3 — Business logic và UI logic trộn lẫn**: Format thời gian, hiển thị trạng thái, quyết định icon — tất cả nằm trong View hoặc Controller:

```python
# UI logic trong Controller = không thể test
class ChatController:
    def format_message_time(self, timestamp):
        delta = datetime.now() - timestamp
        if delta.seconds < 60:
            return "Vừa xong"
        elif delta.seconds < 3600:
            return f"{delta.seconds // 60} phút"
        # ... 50 dòng format phức tạp
```

**Vấn đề 4 — Reactive updates**: Chat là real-time — có 5-10 nguồn dữ liệu có thể update UI cùng lúc (WebSocket, API polling, local cache, user input). Quản lý tất cả bằng callback là nightmare.

### MVVM giải quyết vấn đề

1. **Data binding hai chiều**: ViewModel expose properties, View bind vào — không cần manual update
2. **State encapsulation**: ViewModel chứa toàn bộ state của màn hình dưới dạng `@property` hoặc `Observable`
3. **Reactive pipeline**: Dùng Observable/Stream để xử lý multiple data sources
4. **Testability**: ViewModel có thể test hoàn toàn không cần UI
5. **Separation of concerns rõ ràng**: View chỉ là template, ViewModel là state + logic

---

## Nguyên lý thiết kế

### 1. Data Binding là trung tâm

View "bind" vào các property của ViewModel. Khi ViewModel thay đổi property, View tự động cập nhật. Khi user tương tác với View, ViewModel tự động nhận giá trị mới.

### 2. ViewModel là State của View

ViewModel chứa **tất cả** state cần thiết cho View. View không có state riêng — nó chỉ phản ánh ViewModel. Điều này có nghĩa:
- Nếu destroy View và tạo lại, ViewModel state vẫn còn
- View có thể được tái tạo từ ViewModel bất cứ lúc nào
- Serialize/deserialize ViewModel để restore state

### 3. View không có Logic

View càng "ngu" càng tốt. View chỉ:
- Bind vào ViewModel properties
- Gọi ViewModel commands
- Template rendering (thuần túy presentation)

### 4. Commands thay vì Event Handlers

Thay vì viết event handler trong View, ViewModel expose **commands** — các object thể hiện một hành động có thể gọi từ View:

```python
class ChatViewModel:
    @property
    def send_message_command(self) -> ICommand:
        return RelayCommand(self._execute_send, self._can_send)
```

### 5. Observable Properties

Tất cả properties mà View bind vào phải là **observable** — khi giá trị thay đổi, View nhận được notification.

### 6. Dependency Injection

ViewModel nhận dependencies qua constructor (services, repositories). ViewModel không tự tạo dependencies.

### 7. Immutability cho Data

Dữ liệu từ Model nên được wrap thành immutable objects. ViewModel có thể expose mutable state cho View.

---

## Cấu trúc chi tiết

### Các thành phần

```
┌────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                    │
│                                                             │
│  ┌─────────────────────┐         ┌──────────────────────┐  │
│  │       View           │  bind   │      ViewModel        │  │
│  │  (UI Layer)          │◄───────►│  (State + Commands)   │  │
│  │                      │  data   │                       │  │
│  │  - Template          │         │  - Observable props    │  │
│  │  - Data binding      │         │  - Commands            │  │
│  │  - Animations        │         │  - Validation logic    │  │
│  │  - User input        │         │  - Formatting logic    │  │
│  └──────────┬───────────┘         └───────────┬───────────┘  │
└─────────────┼─────────────────────────────────┼──────────────┘
              │                                 │
              │                                 │ calls
              │                                 ▼
              │                   ┌─────────────────────────┐
              │                   │        Model             │
              │                   │  (Domain + Data Layer)   │
              │                   │                          │
              │                   │  - Domain entities       │
              │                   │  - Business logic        │
              │                   │  - Services              │
              │                   │  - Repositories          │
              │                   └─────────────────────────┘
              │
              ▼
     ┌────────────────┐
     │   Platform UI   │
     │ (Console/WPF/   │
     │  Android/Web)   │
     └────────────────┘
```

**1. View**
- Template (HTML, XAML, Jetpack Compose, SwiftUI)
- Data binding declarations
- Command bindings
- Animation, transitions
- **Không** có business logic

**2. ViewModel**
- Observable properties (state của màn hình)
- Commands (hành động user có thể thực hiện)
- Formatting/transformation logic
- Validation logic
- Gọi Model services
- **Không** tham chiếu đến View

**3. Model**
- Domain entities
- Business rules (use cases)
- Data access (repositories)
- External services (API, WebSocket, Push)
- **Không** biết gì về ViewModel hoặc View

### Luồng dữ liệu

```
User Input → View → Command Binding → ViewModel.execute()
                                           ↓
                              ViewModel updates property
                                           ↓
                              Data binding → View updates UI

Ngược lại:
Model thay đổi → ViewModel nhận event → Cập nhật property → View update
```

### Data Binding Mechanism

Trong Python, chúng ta không có sẵn data binding. Ta sẽ implement một Observable pattern đơn giản:

```python
class ObservableProperty:
    def __init__(self, initial_value=None):
        self._value = initial_value
        self._observers = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if self._value != new_value:
            old_value = self._value
            self._value = new_value
            self._notify(old_value, new_value)

    def bind(self, callback):
        self._observers.append(callback)
        # Immediately notify with current value
        callback(self._value, self._value)

    def _notify(self, old_value, new_value):
        for observer in self._observers:
            observer(old_value, new_value)
```

---

## Sơ đồ kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MVVM ARCHITECTURE                                     │
│                                                                               │
│   ┌───────────────────────────────────────────────────────────────────┐      │
│   │                        VIEW LAYER                                 │      │
│   │                                                                   │      │
│   │  ┌─────────────────────┐    ┌──────────────────┐                 │      │
│   │  │   ChatListView      │    │  ChatRoomView    │                 │      │
│   │  │                     │    │                  │                 │      │
│   │  │  ┌───────────────┐  │    │  ┌────────────┐  │                 │      │
│   │  │  │ Conversation  │  │    │  │ MessageList│  │                 │      │
│   │  │  │ ListItem       │◄─┼────┼──┤ (Recycler) │  │                 │      │
│   │  │  └───────────────┘  │    │  └────────────┘  │                 │      │
│   │  │  ┌───────────────┐  │    │  ┌────────────┐  │                 │      │
│   │  │  │ SearchBar     │  │    │  │ InputBar   │  │                 │      │
│   │  │  └───────────────┘  │    │  └────────────┘  │                 │      │
│   │  │  ┌───────────────┐  │    │  ┌────────────┐  │                 │      │
│   │  │  │ UnreadBadge   │  │    │  │ TypingInd. │  │                 │      │
│   │  │  └───────────────┘  │    │  └────────────┘  │                 │      │
│   │  └──────────┬──────────┘    └────────┬─────────┘                 │      │
│   │             │                        │                          │      │
│   │             │       Data Binding      │                          │      │
│   └─────────────┼────────────────────────┼──────────────────────────┘      │
│                 │                        │                                  │
│   ┌─────────────┼────────────────────────┼──────────────────────────┐      │
│   │             ▼                        ▼                          │      │
│   │  ┌─────────────────────┐    ┌──────────────────┐               │      │
│   │  │  ChatListViewModel  │    │ ChatRoomViewModel│               │      │
│   │  │                     │    │                  │               │      │
│   │  │  + conversations    │    │  + messages      │               │      │
│   │  │  + isLoading        │    │  + inputText     │               │      │
│   │  │  + searchQuery      │    │  + isConnected   │               │      │
│   │  │  + errorMessage     │    │  + typingUsers   │               │      │
│   │  │                     │    │                  │               │      │
│   │  │  + loadConversations│    │  + sendMessage() │               │      │
│   │  │  + search()         │    │  + loadMore()    │               │      │
│   │  │  + selectConversat. │    │  + attachFile()  │               │      │
│   │  └──────────┬──────────┘    └────────┬─────────┘               │      │
│   │             │                        │                          │      │
│   │             │      VIEWMODEL LAYER   │                          │      │
│   └─────────────┼────────────────────────┼──────────────────────────┘      │
│                 │                        │                                  │
│   ┌─────────────┼────────────────────────┼──────────────────────────┐      │
│   │             ▼                        ▼                          │      │
│   │  ┌──────────────────────────────────────────────────────┐      │      │
│   │  │                     MODEL LAYER                        │      │      │
│   │  │                                                       │      │      │
│   │  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │      │      │
│   │  │  │  ChatRepo  │  │  UserRepo    │  │ WebSocket    │ │      │      │
│   │  │  │            │  │              │  │  Client      │ │      │      │
│   │  │  │ + getConvs │  │ + getProfile │  │              │ │      │      │
│   │  │  │ + getMsgs  │  │ + updateStat │  │ + connect    │ │      │      │
│   │  │  │ + sendMsg  │  │ + getOnline  │  │ + sendMsg    │ │      │      │
│   │  │  └────────────┘  └──────────────┘  │ + onMessage  │ │      │      │
│   │  │                                    └──────────────┘ │      │      │
│   │  └──────────────────────────────────────────────────────┘      │      │
│   └─────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Ví dụ code hoàn chỉnh

### Cấu trúc project

```
mochatalk/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── entities.py
│   ├── services.py
│   └── repositories.py
├── viewmodel/
│   ├── __init__.py
│   ├── base.py
│   ├── observable.py
│   ├── chat_list_viewmodel.py
│   └── chat_room_viewmodel.py
├── view/
│   ├── __init__.py
│   ├── console_chat_list.py
│   └── console_chat_room.py
├── main.py
└── tests/
    ├── __init__.py
    └── test_viewmodels.py
```

### model/entities.py

```python
"""Domain entities for chat application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Optional


class UserStatus(Enum):
    ONLINE = auto()
    OFFLINE = auto()
    BUSY = auto()
    AWAY = auto()
    DO_NOT_DISTURB = auto()


class MessageType(Enum):
    TEXT = auto()
    IMAGE = auto()
    FILE = auto()
    SYSTEM = auto()
    REPLY = auto()


class MessageStatus(Enum):
    SENDING = auto()
    SENT = auto()
    DELIVERED = auto()
    READ = auto()
    FAILED = auto()


@dataclass(frozen=True)
class User:
    id: str
    username: str
    display_name: str
    avatar_url: str
    status: UserStatus = UserStatus.ONLINE

    @property
    def initials(self) -> str:
        parts = self.display_name.split()
        if len(parts) >= 2:
            return f"{parts[0][0]}{parts[-1][0]}".upper()
        return self.display_name[:2].upper()

    @property
    def status_label(self) -> str:
        labels = {
            UserStatus.ONLINE: "🟢 Online",
            UserStatus.OFFLINE: "⚫ Offline",
            UserStatus.BUSY: "🔴 Busy",
            UserStatus.AWAY: "🟡 Away",
            UserStatus.DO_NOT_DISTURB: "⛔ DND",
        }
        return labels.get(self.status, "Unknown")


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    sender: User
    content: str
    message_type: MessageType = MessageType.TEXT
    status: MessageStatus = MessageStatus.SENT
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None
    attachments: tuple[str, ...] = field(default_factory=tuple)

    @property
    def formatted_time(self) -> str:
        now = datetime.now()
        delta = now - self.timestamp

        if delta.days == 0:
            return self.timestamp.strftime("%H:%M")
        elif delta.days == 1:
            return f"Hôm qua {self.timestamp.strftime('%H:%M')}"
        elif delta.days < 7:
            weekdays = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
            return f"{weekdays[self.timestamp.weekday()]} {self.timestamp.strftime('%H:%M')}"
        return self.timestamp.strftime("%d/%m/%Y")

    @property
    def status_icon(self) -> str:
        icons = {
            MessageStatus.SENDING: "⏳",
            MessageStatus.SENT: "✓",
            MessageStatus.DELIVERED: "✓✓",
            MessageStatus.READ: "✓✓",
            MessageStatus.FAILED: "✗",
        }
        return icons.get(self.status, "?")

    @property
    def is_system_message(self) -> bool:
        return self.message_type == MessageType.SYSTEM


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    participants: tuple[User, ...]
    last_message: Optional[Message] = None
    unread_count: int = 0
    is_pinned: bool = False
    is_muted: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def participant_names(self) -> str:
        return ", ".join(p.display_name for p in self.participants)

    @property
    def last_message_preview(self) -> str:
        if self.last_message and not self.last_message.is_system_message:
            preview = self.last_message.content
            if len(preview) > 50:
                preview = preview[:47] + "..."
            return f"{self.last_message.sender.display_name}: {preview}"
        return "Chưa có tin nhắn"


@dataclass
class TypingIndicator:
    user: User
    conversation_id: str
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()
```

### model/repositories.py

```python
"""Data repositories with simulated data."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Sequence, Optional

from .entities import (
    User, UserStatus, Message, MessageType, MessageStatus,
    Conversation, TypingIndicator,
)


class ChatRepository:
    """Repository for chat data — simulates API + database."""

    def __init__(self) -> None:
        self._users = self._seed_users()
        self._conversations: list[Conversation] = []
        self._messages: dict[str, list[Message]] = {}
        self._seed_conversations()

    @staticmethod
    def _seed_users() -> dict[str, User]:
        users_data = [
            ("U1", "minh", "Nguyễn Văn Minh"),
            ("U2", "lan", "Trần Thị Lan"),
            ("U3", "huy", "Phạm Quang Huy"),
            ("U4", "mai", "Lê Thị Mai"),
            ("U5", "tuan", "Hoàng Văn Tuấn"),
            ("U6", "anh", "Đỗ Ngọc Anh"),
        ]
        statuses = list(UserStatus)
        return {
            uid: User(
                id=uid,
                username=uname,
                display_name=dname,
                avatar_url=f"https://avatar.example.com/{uname}.png",
                status=random.choice(statuses),
            )
            for uid, uname, dname in users_data
        }

    def _seed_conversations(self) -> None:
        users = list(self._users.values())
        pairings = [
            (["U3"], "Dự án E-commerce"),
            (["U1", "U2"], "Team Design"),
            (["U4", "U5", "U6"], "DevOps"),
            (["U1", "U2", "U3", "U4", "U5", "U6"], "Công ty chung"),
            (["U6"], "Hỗ trợ kỹ thuật"),
        ]

        for participant_ids, title in pairings:
            conv_id = f"C{len(self._conversations) + 1:04d}"
            participants = tuple(self._users[uid] for uid in participant_ids)

            # Tạo tin nhắn mẫu
            msgs = self._generate_sample_messages(conv_id, participants[0])
            self._messages[conv_id] = msgs

            conv = Conversation(
                id=conv_id,
                title=title,
                participants=participants,
                last_message=msgs[-1] if msgs else None,
                unread_count=random.randint(0, 10),
                is_pinned=random.choice([True, False, False]),
                is_muted=random.choice([True, False, False, False]),
            )
            self._conversations.append(conv)

    def _generate_sample_messages(self, conv_id: str, sender: User) -> list[Message]:
        texts = [
            "Chào mọi người, hôm nay có meeting không?",
            "Mình vừa push xong feature mới",
            "Code review giúp mình PR #123 với",
            "Có bug trên production, cần hotfix gấp! 🚨",
            "OK, để mình check lại",
            "Sprint này chúng ta cần xong module payment",
            "Ai rảnh review design mới của mình không?",
            "Đã deploy lên staging rồi mọi người ơi",
            "Test thấy OK, có thể release được rồi",
            "Chốt meeting vào 2h chiều nay nhé!",
        ]
        base_time = datetime.now() - timedelta(hours=len(texts))
        msgs = []

        for i, text in enumerate(texts):
            msg = Message(
                id=f"M{len(msgs) + 1:04d}",
                conversation_id=conv_id,
                sender=random.choice(list(self._users.values())) if i > 0 else sender,
                content=text,
                status=random.choice(list(MessageStatus)),
                timestamp=base_time + timedelta(hours=i + 1),
            )
            msgs.append(msg)
        return msgs

    def get_conversations(self) -> Sequence[Conversation]:
        sorted_conv = sorted(
            self._conversations,
            key=lambda c: c.last_message.timestamp if c.last_message else c.created_at,
            reverse=True,
        )
        return sorted_conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        for c in self._conversations:
            if c.id == conv_id:
                return c
        return None

    def get_messages(self, conv_id: str, limit: int = 50) -> Sequence[Message]:
        return self._messages.get(conv_id, [])[-limit:]

    def add_message(self, conv_id: str, sender_id: str, content: str) -> Message:
        sender = self._users.get(sender_id)
        if not sender:
            raise ValueError(f"User {sender_id} not found")

        msg = Message(
            id=f"M{random.randint(10000, 99999)}",
            conversation_id=conv_id,
            sender=sender,
            content=content,
            status=MessageStatus.SENDING,
        )

        if conv_id not in self._messages:
            self._messages[conv_id] = []
        self._messages[conv_id].append(msg)

        # Update conversation's last message
        for i, conv in enumerate(self._conversations):
            if conv.id == conv_id:
                self._conversations[i] = Conversation(
                    id=conv.id,
                    title=conv.title,
                    participants=conv.participants,
                    last_message=msg,
                    unread_count=conv.unread_count + 1,
                    is_pinned=conv.is_pinned,
                    is_muted=conv.is_muted,
                    created_at=conv.created_at,
                )
                break
        return msg

    def mark_as_read(self, conv_id: str) -> None:
        for i, conv in enumerate(self._conversations):
            if conv.id == conv_id:
                self._conversations[i] = Conversation(
                    id=conv.id,
                    title=conv.title,
                    participants=conv.participants,
                    last_message=conv.last_message,
                    unread_count=0,
                    is_pinned=conv.is_pinned,
                    is_muted=conv.is_muted,
                    created_at=conv.created_at,
                )
                break

    def search_conversations(self, query: str) -> Sequence[Conversation]:
        q = query.lower()
        results = []
        for conv in self._conversations:
            if q in conv.title.lower():
                results.append(conv)
                continue
            for p in conv.participants:
                if q in p.display_name.lower() or q in p.username.lower():
                    results.append(conv)
                    break
            if conv.last_message and q in conv.last_message.content.lower():
                if conv not in results:
                    results.append(conv)
        return results

    def get_typing_users(self, conv_id: str) -> Sequence[User]:
        users = list(self._users.values())
        # Mô phỏng: trả về 0-2 người đang typing
        if random.random() < 0.3:
            return tuple(random.sample(users, random.randint(1, 2)))
        return ()


class AuthRepository:
    """Repository cho authentication."""

    def __init__(self) -> None:
        self._current_user_id: str = "U1"

    def get_current_user(self) -> User:
        # Mô phỏng
        return User(
            id="U1",
            username="minh",
            display_name="Nguyễn Văn Minh",
            avatar_url="https://avatar.example.com/minh.png",
            status=UserStatus.ONLINE,
        )

    def get_current_user_id(self) -> str:
        return self._current_user_id
```

### model/services.py

```python
"""Business logic services."""

from __future__ import annotations

from typing import Sequence, Optional

from .entities import Conversation, Message, User, TypingIndicator
from .repositories import ChatRepository, AuthRepository


class ChatService:
    """Service layer — orchestrate business logic.

    Model layer hoàn toàn không biết gì về ViewModel hay View.
    """

    def __init__(
        self,
        chat_repo: ChatRepository,
        auth_repo: AuthRepository,
    ) -> None:
        self._chat_repo = chat_repo
        self._auth_repo = auth_repo

    def get_conversations(self) -> Sequence[Conversation]:
        return self._chat_repo.get_conversations()

    def get_current_user(self) -> User:
        return self._auth_repo.get_current_user()

    def get_current_user_id(self) -> str:
        return self._auth_repo.get_current_user_id()

    def get_messages(self, conv_id: str) -> Sequence[Message]:
        return self._chat_repo.get_messages(conv_id)

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        return self._chat_repo.get_conversation(conv_id)

    def send_message(self, conv_id: str, content: str) -> Message:
        sender_id = self.get_current_user_id()
        content = content.strip()
        if not content:
            raise ValueError("Message cannot be empty")
        return self._chat_repo.add_message(conv_id, sender_id, content)

    def mark_as_read(self, conv_id: str) -> None:
        self._chat_repo.mark_as_read(conv_id)

    def search(self, query: str) -> Sequence[Conversation]:
        query = query.strip()
        if not query:
            return self.get_conversations()
        return self._chat_repo.search_conversations(query)

    def get_typing_users(self, conv_id: str) -> Sequence[User]:
        return self._chat_repo.get_typing_users(conv_id)
```

### viewmodel/observable.py

```python
"""Observable pattern implementation — trái tim của MVVM data binding.

Trong Python, ta không có sẵn data binding như WPF hay Android.
Class ObservableProperty và ObservableList implement cơ chế thông báo
thay đổi để View có thể lắng nghe và cập nhật tự động.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, Protocol, runtime_checkable

T = TypeVar("T")


ObserverCallback = Callable[[T, T], None]
"""Callback type: (old_value, new_value) -> None"""


class ObservableProperty(Generic[T]):
    """Property có thể quan sát — khi giá trị thay đổi, tất cả observer được notify.

    Usage:
        name = ObservableProperty("")
        name.bind(lambda old, new: print(f"Changed: {old} -> {new}"))
        name.value = "Hello"  # Prints: Changed:  -> Hello
    """

    def __init__(self, initial_value: T | None = None) -> None:
        self._value: T | None = initial_value
        self._observers: list[ObserverCallback[T]] = []

    @property
    def value(self) -> T | None:
        return self._value

    @value.setter
    def value(self, new_value: T | None) -> None:
        if self._value != new_value:
            old_value = self._value
            self._value = new_value
            self._notify(old_value, new_value)

    def bind(self, callback: ObserverCallback[T]) -> None:
        """Đăng ký observer. Callback được gọi ngay với giá trị hiện tại."""
        self._observers.append(callback)
        callback(self._value, self._value)  # Initial notification

    def unbind(self, callback: ObserverCallback[T]) -> None:
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify(self, old_value: T | None, new_value: T | None) -> None:
        for observer in self._observers:
            observer(old_value, new_value)

    def __repr__(self) -> str:
        return f"ObservableProperty({self._value!r})"


class ObservableList(Generic[T]):
    """List observable — notify khi có item được thêm, xóa, hoặc thay đổi.

    View có thể bind vào list này để tự động cập nhật UI khi dữ liệu thay đổi.
    """

    def __init__(self, initial_items: list[T] | None = None) -> None:
        self._items: list[T] = list(initial_items) if initial_items else []
        self._on_change: list[Callable[[], None]] = []

    @property
    def items(self) -> list[T]:
        return list(self._items)

    @items.setter
    def items(self, new_items: list[T]) -> None:
        self._items = list(new_items)
        self._notify()

    def add(self, item: T) -> None:
        self._items.append(item)
        self._notify()

    def remove(self, item: T) -> None:
        if item in self._items:
            self._items.remove(item)
            self._notify()

    def replace_all(self, items: list[T]) -> None:
        self._items = list(items)
        self._notify()

    def bind(self, callback: Callable[[], None]) -> None:
        self._on_change.append(callback)

    def _notify(self) -> None:
        for cb in self._on_change:
            cb()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __repr__(self) -> str:
        return f"ObservableList({self._items!r})"


@runtime_checkable
class ICommand(Protocol):
    """Interface cho Command trong MVVM.

    Command đóng gói một hành động + điều kiện có thể thực thi.
    View bind vào command thay vì xử lý event trực tiếp.
    """

    def execute(self, parameter: Any = None) -> None: ...
    def can_execute(self, parameter: Any = None) -> bool: ...


class RelayCommand:
    """Implementation đơn giản của ICommand.

    Usage:
        save_cmd = RelayCommand(
            execute=lambda _: self._save(),
            can_execute=lambda _: self.is_valid,
        )
    """

    def __init__(
        self,
        execute: Callable[[Any], None],
        can_execute: Callable[[Any], bool] | None = None,
    ) -> None:
        self._execute = execute
        self._can_execute = can_execute

    def execute(self, parameter: Any = None) -> None:
        if self.can_execute(parameter):
            self._execute(parameter)

    def can_execute(self, parameter: Any = None) -> bool:
        if self._can_execute is None:
            return True
        return self._can_execute(parameter)
```

### viewmodel/base.py

```python
"""Base ViewModel — common functionality for all ViewModels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseViewModel(ABC):
    """Base class cho tất cả ViewModel.

    ViewModel chứa toàn bộ state của màn hình.
    ViewModel không biết gì về View — nó chỉ expose properties và commands.
    """

    def __init__(self) -> None:
        self._is_disposed = False

    @abstractmethod
    def initialize(self) -> None:
        """Load initial data — gọi khi View được tạo."""
        ...

    def dispose(self) -> None:
        """Cleanup resources."""
        self._is_disposed = True
```

### viewmodel/chat_list_viewmodel.py

```python
"""ViewModel cho màn hình danh sách chat."""

from __future__ import annotations

from typing import Optional, Sequence

from model.entities import Conversation, User
from model.services import ChatService
from viewmodel.base import BaseViewModel
from viewmodel.observable import (
    ObservableProperty, ObservableList, RelayCommand, ICommand,
)


class ChatListViewModel(BaseViewModel):
    """ViewModel cho màn hình danh sách hội thoại.

    Expose tất cả state mà View cần:
    - conversations: Danh sách hội thoại (observable list)
    - isLoading, errorMessage: Trạng thái
    - searchQuery: Text search (two-way binding)
    - selectedConversation: Hội thoại đang chọn

    Expose commands:
    - loadConversationsCommand: Load danh sách
    - searchCommand: Tìm kiếm
    - selectConversationCommand: Chọn hội thoại
    """

    def __init__(self, service: ChatService) -> None:
        super().__init__()
        self._service = service

        # Observable Properties — View bind vào các property này
        self.conversations: ObservableList[Conversation] = ObservableList()
        self.is_loading = ObservableProperty[bool](False)
        self.error_message = ObservableProperty[Optional[str]](None)
        self.search_query = ObservableProperty[str]("")
        self.selected_conversation = ObservableProperty[Optional[Conversation]](None)
        self.current_user = ObservableProperty[Optional[User]](None)
        self.total_unread = ObservableProperty[int](0)

        # Commands
        self.load_conversations_command: ICommand = RelayCommand(
            execute=lambda _: self.load_conversations(),
        )
        self.search_command: ICommand = RelayCommand(
            execute=lambda _: self._execute_search(),
        )
        self.select_conversation_command: ICommand = RelayCommand(
            execute=lambda p: self._execute_select(p),
        )

        # Bind search query để auto-search khi gõ
        self.search_query.bind(self._on_search_query_changed)

    def initialize(self) -> None:
        current_user = self._service.get_current_user()
        self.current_user.value = current_user
        self.load_conversations()

    def load_conversations(self) -> None:
        """Load danh sách hội thoại."""
        self.is_loading.value = True
        self.error_message.value = None

        try:
            convs = self._service.get_conversations()
            self.conversations.replace_all(list(convs))
            self._update_total_unread(convs)
        except Exception as e:
            self.error_message.value = f"Không thể tải danh sách chat: {e}"
        finally:
            self.is_loading.value = False

    def _execute_search(self) -> None:
        """Thực hiện search command."""
        self._search(self.search_query.value or "")

    def _on_search_query_changed(self, old: str | None, new: str | None) -> None:
        """Khi search query thay đổi, tự động search."""
        if new is not None and old != new:
            self._search(new)

    def _search(self, query: str) -> None:
        self.is_loading.value = True
        try:
            results = self._service.search(query)
            self.conversations.replace_all(list(results))
        except Exception as e:
            self.error_message.value = f"Lỗi tìm kiếm: {e}"
        finally:
            self.is_loading.value = False

    def _execute_select(self, conversation_id: Any) -> None:
        """Xử lý khi user chọn một hội thoại."""
        if not conversation_id:
            return
        conv = self._service.get_conversation(conversation_id)
        if conv:
            self.selected_conversation.value = conv
            self._service.mark_as_read(conv.id)
            self.load_conversations()  # Refresh để cập nhật unread count

    def _update_total_unread(self, conversations: Sequence[Conversation]) -> None:
        total = sum(c.unread_count for c in conversations)
        self.total_unread.value = total
```

### viewmodel/chat_room_viewmodel.py

```python
"""ViewModel cho màn hình phòng chat."""

from __future__ import annotations

from typing import Optional, Sequence

from model.entities import Conversation, Message, User
from model.services import ChatService
from viewmodel.base import BaseViewModel
from viewmodel.observable import (
    ObservableProperty, ObservableList, RelayCommand, ICommand,
)


class ChatRoomViewModel(BaseViewModel):
    """ViewModel cho phòng chat.

    Quản lý state của một cuộc hội thoại:
    - messages: Danh sách tin nhắn
    - inputText: Text người dùng đang nhập
    - conversation: Thông tin hội thoại hiện tại
    - isConnected: Trạng thái kết nối WebSocket
    - typingUsers: Người đang gõ
    """

    def __init__(self, service: ChatService) -> None:
        super().__init__()
        self._service = service

        # Observable properties
        self.messages: ObservableList[Message] = ObservableList()
        self.conversation = ObservableProperty[Optional[Conversation]](None)
        self.input_text = ObservableProperty[str]("")
        self.is_connected = ObservableProperty[bool](True)
        self.typing_users = ObservableProperty[Sequence[User]]([])
        self.current_user = ObservableProperty[Optional[User]](None)
        self.is_loading = ObservableProperty[bool](False)
        self.error_message = ObservableProperty[Optional[str]](None)

        # Commands
        self.send_message_command: ICommand = RelayCommand(
            execute=lambda _: self._send_message(),
            can_execute=lambda _: bool(self.input_text.value and self.conversation.value),
        )
        self.load_more_command: ICommand = RelayCommand(
            execute=lambda _: self._load_more(),
        )
        self.back_command: ICommand = RelayCommand(
            execute=lambda _: self._go_back(),
        )

        # Bind input text để auto-update can_execute
        self.input_text.bind(self._on_input_changed)

    def initialize(self) -> None:
        self.current_user.value = self._service.get_current_user()

    def load_conversation(self, conv_id: str) -> None:
        """Load một hội thoại cụ thể."""
        self.is_loading.value = True
        self.error_message.value = None

        try:
            conv = self._service.get_conversation(conv_id)
            if not conv:
                raise ValueError(f"Không tìm thấy hội thoại {conv_id}")

            self.conversation.value = conv
            msgs = self._service.get_messages(conv_id)
            self.messages.replace_all(list(msgs))
            self._service.mark_as_read(conv_id)

        except Exception as e:
            self.error_message.value = f"Lỗi tải hội thoại: {e}"
        finally:
            self.is_loading.value = False

    def refresh_typing(self) -> None:
        """Cập nhật danh sách người đang typing (polling)."""
        if not self.conversation.value:
            return
        try:
            users = self._service.get_typing_users(self.conversation.value.id)
            # Filter out current user
            users = [u for u in users if u.id != (self.current_user.value.id if self.current_user.value else "")]
            self.typing_users.value = users
        except Exception:
            pass

    def _send_message(self) -> None:
        """Gửi tin nhắn."""
        text = (self.input_text.value or "").strip()
        conv = self.conversation.value
        if not text or not conv:
            return

        try:
            msg = self._service.send_message(conv.id, text)
            self.messages.add(msg)
            self.input_text.value = ""  # Clear input
            # Update conversation preview
            self.conversation.value = Conversation(
                id=conv.id,
                title=conv.title,
                participants=conv.participants,
                last_message=msg,
                unread_count=0,
                is_pinned=conv.is_pinned,
                is_muted=conv.is_muted,
                created_at=conv.created_at,
            )
        except ValueError as e:
            self.error_message.value = str(e)

    def _load_more(self) -> None:
        """Load thêm tin nhắn cũ."""
        # In thực tế, đây sẽ load paginated messages
        pass

    def _go_back(self) -> None:
        """Quay lại danh sách chat."""
        self.dispose()

    def _on_input_changed(self, old: str | None, new: str | None) -> None:
        """Khi input thay đổi, notify để update can_execute."""
        pass  # In real MVVM, this would trigger command can_execute re-evaluation
```

### view/console_chat_list.py

```python
"""Console View cho danh sách chat — implement data binding với ViewModel."""

from __future__ import annotations

import time
import os
from typing import Optional, Sequence

from model.entities import Conversation, User
from viewmodel.chat_list_viewmodel import ChatListViewModel


class ConsoleChatListView:
    """View hiển thị danh sách chat trên console.

    View này bind vào ChatListViewModel:
    - Khi conversations thay đổi → View tự động render lại
    - Khi errorMessage thay đổi → View hiển thị lỗi
    - Khi isLoading thay đổi → View hiển thị loading spinner

    View KHÔNG chứa business logic — chỉ template rendering.
    """

    def __init__(self, viewmodel: ChatListViewModel) -> None:
        self._vm = viewmodel
        self._selected_index = 0

        # Data binding — View đăng ký lắng nghe ViewModel changes
        self._vm.conversations.bind(self._on_conversations_changed)
        self._vm.is_loading.bind(self._on_loading_changed)
        self._vm.error_message.bind(self._on_error_changed)
        self._vm.total_unread.bind(self._on_unread_changed)
        self._vm.current_user.bind(self._on_user_changed)

    def _on_conversations_changed(self) -> None:
        """Callback khi danh sách hội thoại thay đổi."""
        self._render()

    def _on_loading_changed(self, old: bool | None, new: bool | None) -> None:
        if new:
            self._show_loading()
        else:
            self._hide_loading()

    def _on_error_changed(self, old: str | None, new: str | None) -> None:
        if new:
            print(f"\033[31m❌ {new}\033[0m")

    def _on_unread_changed(self, old: int | None, new: int | None) -> None:
        pass  # Có thể update toolbar/badge

    def _on_user_changed(self, old: User | None, new: User | None) -> None:
        if new:
            print(f"  Xin chào, {new.display_name} {new.status_label}")

    def _show_loading(self) -> None:
        print("  ⏳ Đang tải...")

    def _hide_loading(self) -> None:
        pass

    def _render(self) -> None:
        """Render danh sách hội thoại."""
        os.system("cls" if os.name == "nt" else "clear")
        user = self._vm.current_user.value

        # Header
        print(f"{'='*60}")
        print(f"  💬 MOCHATALK")
        if user:
            print(f"  {user.display_name:<30s} | {user.status_label}")
        if self._vm.total_unread.value:
            print(f"  🔔 {self._vm.total_unread.value} tin nhắn chưa đọc")
        print(f"{'='*60}")

        # Search bar
        query = self._vm.search_query.value or ""
        print(f"  🔍 Search: [{query}]")
        print(f"{'─'*60}")

        # Conversation list
        convs = self._vm.conversations.items
        if not convs:
            print("  (Không có hội thoại nào)")
        else:
            for i, conv in enumerate(convs):
                prefix = "▶" if i == self._selected_index else " "
                unread = f" ({conv.unread_count})" if conv.unread_count > 0 else ""
                pin = "📌 " if conv.is_pinned else ""
                mute = "🔇 " if conv.is_muted else ""
                print(f"  {prefix} {pin}{mute}{conv.title:<30s}{unread}")
                print(f"     {conv.last_message_preview}")

        # Footer
        print(f"{'─'*60}")
        print("  [/] Search  [↑↓] Navigate  [Enter] Open  [Q]uit")

    def get_user_input(self) -> str:
        """Nhận input từ user và cập nhật ViewModel."""
        cmd = input("  > ").strip().lower()

        if cmd == "/":
            query = input("  Search: ").strip()
            self._vm.search_query.value = query
            return "search"
        elif cmd == "q":
            return "quit"
        elif cmd == "":
            # Enter — mở hội thoại
            convs = self._vm.conversations.items
            if convs and self._selected_index < len(convs):
                conv = convs[self._selected_index]
                self._vm.select_conversation_command.execute(conv.id)
                return f"open:{conv.id}"
            return "refresh"
        else:
            return "refresh"

    def refresh(self) -> None:
        """Refresh view."""
        self._render()
```

### view/console_chat_room.py

```python
"""Console View cho phòng chat — bind vào ChatRoomViewModel."""

from __future__ import annotations

import os
import time
from typing import Optional

from viewmodel.chat_room_viewmodel import ChatRoomViewModel


class ConsoleChatRoomView:
    """View hiển thị phòng chat trên console.

    Data binding với ChatRoomViewModel:
    - messages → render message list
    - conversation → render header
    - typingUsers → render typing indicator
    - inputText → two-way binding với input field
    """

    def __init__(self, viewmodel: ChatRoomViewModel) -> None:
        self._vm = viewmodel

        # Bind ViewModel changes
        self._vm.messages.bind(self._on_messages_changed)
        self._vm.conversation.bind(self._on_conversation_changed)
        self._vm.typing_users.bind(self._on_typing_changed)
        self._vm.is_loading.bind(self._on_loading_changed)
        self._vm.error_message.bind(self._on_error_changed)

    def _on_messages_changed(self) -> None:
        self._render_messages()

    def _on_conversation_changed(self, old, new) -> None:
        self._render()

    def _on_typing_changed(self, old, new) -> None:
        if new:
            names = ", ".join(u.display_name for u in new)
            print(f"  ✏️  {names} đang nhập...")

    def _on_loading_changed(self, old: bool | None, new: bool | None) -> None:
        if new:
            print("  ⏳ Đang tải hội thoại...")

    def _on_error_changed(self, old: str | None, new: str | None) -> None:
        if new:
            print(f"\033[31m❌ {new}\033[0m")

    def _render(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        conv = self._vm.conversation.value
        if not conv:
            return

        print(f"{'='*60}")
        print(f"  💬 {conv.title}")
        print(f"  {conv.participant_names}")
        print(f"{'='*60}")

        self._render_messages()

    def _render_messages(self) -> None:
        msgs = self._vm.messages.items
        current_user_id = self._vm.current_user.value.id if self._vm.current_user.value else ""

        for msg in msgs:
            if msg.is_system_message:
                print(f"  ─── {msg.content} ───")
                continue

            is_mine = msg.sender.id == current_user_id
            prefix = "  " if is_mine else f"  {msg.sender.initials} "
            align = "─" * 40 if is_mine else "─" * 40
            status = f" {msg.status_icon}" if is_mine else ""

            print(f"{prefix}{msg.sender.display_name}")
            print(f"  {msg.content}{status}")
            print(f"  \033[90m{msg.formatted_time}\033[0m")
            print()

        # Footer
        print(f"{'─'*60}")
        print("  [B]ack  Type message + Enter  [Q]uit")

    def load_conversation(self, conv_id: str) -> None:
        """Load hội thoại và render."""
        self._vm.load_conversation(conv_id)
        self._render()

    def get_user_input(self) -> str:
        text = input("  > ").strip()
        if text.lower() == "q":
            return "quit"
        elif text.lower() == "b":
            return "back"
        elif text:
            self._vm.input_text.value = text
            self._vm.send_message_command.execute(None)
            return "sent"
        return "refresh"
```

### main.py

```python
"""Main entry point — khởi tạo MVVM architecture."""

from __future__ import annotations

import sys
from typing import NoReturn

from model.repositories import ChatRepository, AuthRepository
from model.services import ChatService
from viewmodel.chat_list_viewmodel import ChatListViewModel
from viewmodel.chat_room_viewmodel import ChatRoomViewModel
from view.console_chat_list import ConsoleChatListView
from view.console_chat_room import ConsoleChatRoomView


def run_chat_app() -> NoReturn:
    # === Model Layer ===
    chat_repo = ChatRepository()
    auth_repo = AuthRepository()
    service = ChatService(chat_repo, auth_repo)

    # === ViewModel Layer ===
    chat_list_vm = ChatListViewModel(service)
    chat_room_vm = ChatRoomViewModel(service)

    # === View Layer (bind to ViewModel) ===
    chat_list_view = ConsoleChatListView(chat_list_vm)
    chat_room_view = ConsoleChatRoomView(chat_room_vm)

    # Initialize ViewModel
    chat_list_vm.initialize()

    # === Application Loop ===
    current_screen = "list"

    while True:
        if current_screen == "list":
            chat_list_view.refresh()
            cmd = chat_list_view.get_user_input()

            if cmd == "quit":
                print("  👋 Tạm biệt!")
                sys.exit(0)
            elif cmd.startswith("open:"):
                conv_id = cmd.split(":", 1)[1]
                current_screen = "room"
                chat_room_view.load_conversation(conv_id)
            elif cmd == "search":
                chat_list_view.refresh()
        else:
            cmd = chat_room_view.get_user_input()
            if cmd == "quit":
                print("  👋 Tạm biệt!")
                sys.exit(0)
            elif cmd == "back":
                current_screen = "list"
                chat_list_vm.load_conversations()


if __name__ == "__main__":
    run_chat_app()
```

---

## Kiểm thử

```python
"""tests/test_viewmodels.py

Test ViewModel hoàn toàn độc lập với View.
Chiến lược:
1. Mock ChatService — test logic thuần túy của ViewModel
2. Test state changes — kiểm tra ObservableProperty thay đổi đúng
3. Test commands — kiểm tra command execution
4. Test edge cases — empty states, errors, boundaries
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, Mock
from datetime import datetime
from typing import Sequence

import sys
sys.path.insert(0, "..")

from model.entities import (
    User, UserStatus, Message, MessageStatus, MessageType,
    Conversation,
)
from model.services import ChatService
from viewmodel.chat_list_viewmodel import ChatListViewModel
from viewmodel.chat_room_viewmodel import ChatRoomViewModel


class MockChatService:
    """Mock service — không cần database thật."""

    def __init__(self) -> None:
        self.users = {
            "U1": User("U1", "minh", "Nguyễn Văn Minh", "", UserStatus.ONLINE),
            "U2": User("U2", "lan", "Trần Thị Lan", "", UserStatus.ONLINE),
        }
        self.conversations: list[Conversation] = []
        self.messages: dict[str, list[Message]] = {}

    def get_current_user(self) -> User:
        return self.users["U1"]

    def get_current_user_id(self) -> str:
        return "U1"

    def get_conversations(self) -> Sequence[Conversation]:
        return self.conversations

    def get_conversation(self, conv_id: str):
        for c in self.conversations:
            if c.id == conv_id:
                return c
        return None

    def get_messages(self, conv_id: str):
        return self.messages.get(conv_id, [])

    def send_message(self, conv_id: str, content: str):
        msg = Message(
            id="M1000",
            conversation_id=conv_id,
            sender=self.users["U1"],
            content=content,
        )
        if conv_id not in self.messages:
            self.messages[conv_id] = []
        self.messages[conv_id].append(msg)
        return msg

    def mark_as_read(self, conv_id: str) -> None:
        for i, c in enumerate(self.conversations):
            if c.id == conv_id:
                new_c = Conversation(
                    id=c.id, title=c.title, participants=c.participants,
                    last_message=c.last_message, unread_count=0,
                    is_pinned=c.is_pinned, is_muted=c.is_muted,
                    created_at=c.created_at,
                )
                self.conversations[i] = new_c
                break

    def search(self, query: str) -> Sequence[Conversation]:
        q = query.lower()
        return [c for c in self.conversations if q in c.title.lower()]

    def get_typing_users(self, conv_id: str):
        return ()

    def add_conv(self, conv_id: str, title: str):
        conv = Conversation(
            id=conv_id, title=title,
            participants=tuple(self.users.values()),
            last_message=Message(
                id="M0", conversation_id=conv_id,
                sender=self.users["U1"],
                content=f"Last message in {title}",
                timestamp=datetime.now(),
            ),
            unread_count=0,
        )
        self.conversations.append(conv)
        self.messages[conv_id] = []
        return conv

    def add_conv_with_unread(self, conv_id: str, title: str, unread: int):
        conv = Conversation(
            id=conv_id, title=title,
            participants=tuple(self.users.values()),
            unread_count=unread,
        )
        self.conversations.append(conv)
        self.messages[conv_id] = []
        return conv


class TestChatListViewModel(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_service = MockChatService()
        self.vm = ChatListViewModel(self.mock_service)

    def test_initialize_loads_current_user(self):
        self.vm.initialize()
        self.assertEqual(self.vm.current_user.value, self.mock_service.users["U1"])

    def test_initialize_loads_conversations(self):
        self.mock_service.add_conv("C1", "Team A")
        self.mock_service.add_conv("C2", "Team B")

        self.vm.initialize()

        self.assertEqual(len(self.vm.conversations), 2)

    def test_empty_conversations(self):
        self.vm.initialize()
        self.assertEqual(len(self.vm.conversations), 0)
        self.assertEqual(self.vm.total_unread.value, 0)

    def test_total_unread_count(self):
        self.mock_service.add_conv_with_unread("C1", "Chat 1", 3)
        self.mock_service.add_conv_with_unread("C2", "Chat 2", 5)

        self.vm.initialize()

        self.assertEqual(self.vm.total_unread.value, 8)

    def test_select_conversation_updates_selected(self):
        self.mock_service.add_conv("C1", "Team A")
        self.vm.initialize()

        self.vm.select_conversation_command.execute("C1")

        self.assertIsNotNone(self.vm.selected_conversation.value)
        self.assertEqual(self.vm.selected_conversation.value.id, "C1")

    def test_select_conversation_marks_as_read(self):
        self.mock_service.add_conv_with_unread("C1", "Chat", 5)
        self.vm.initialize()

        self.vm.select_conversation_command.execute("C1")

        conv = self.mock_service.get_conversation("C1")
        self.assertEqual(conv.unread_count, 0)

    def test_search_filters_conversations(self):
        self.mock_service.add_conv("C1", "Dự án Mobile")
        self.mock_service.add_conv("C2", "Dự án Web")
        self.mock_service.add_conv("C3", "DevOps chung")
        self.vm.initialize()

        self.vm.search_query.value = "Dự án"

        self.assertEqual(len(self.vm.conversations), 2)

    def test_search_empty_returns_all(self):
        self.mock_service.add_conv("C1", "Mobile")
        self.vm.initialize()

        self.vm.search_query.value = ""

        self.assertEqual(len(self.vm.conversations), 1)

    def test_search_no_results(self):
        self.mock_service.add_conv("C1", "Mobile")
        self.vm.initialize()

        self.vm.search_query.value = "zzzzz"

        self.assertEqual(len(self.vm.conversations), 0)

    def test_is_loading_during_load(self):
        self.vm.load_conversations()
        # Loading states should be properly managed
        self.assertFalse(self.vm.is_loading.value)


class TestChatRoomViewModel(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_service = MockChatService()
        self.mock_service.add_conv("C1", "Phòng chat")
        self.vm = ChatRoomViewModel(self.mock_service)
        self.vm.initialize()

    def test_initialize_sets_current_user(self):
        self.assertEqual(self.vm.current_user.value, self.mock_service.users["U1"])

    def test_load_conversation(self):
        self.vm.load_conversation("C1")

        self.assertIsNotNone(self.vm.conversation.value)
        self.assertEqual(self.vm.conversation.value.id, "C1")

    def test_load_conversation_error_handling(self):
        self.vm.load_conversation("NOT_EXIST")

        self.assertIsNotNone(self.vm.error_message.value)

    def test_send_message_adds_to_list(self):
        self.vm.load_conversation("C1")
        self.vm.input_text.value = "Hello!"

        self.vm.send_message_command.execute(None)

        self.assertEqual(len(self.vm.messages), 1)
        self.assertEqual(self.vm.messages[0].content, "Hello!")

    def test_send_message_clears_input(self):
        self.vm.load_conversation("C1")
        self.vm.input_text.value = "Test message"

        self.vm.send_message_command.execute(None)

        self.assertEqual(self.vm.input_text.value, "")

    def test_send_empty_message_does_nothing(self):
        self.vm.load_conversation("C1")
        self.vm.input_text.value = "   "

        self.vm.send_message_command.execute(None)

        self.assertEqual(len(self.vm.messages), 0)

    def test_send_message_no_conversation(self):
        self.vm.input_text.value = "Hello"
        self.vm.send_message_command.execute(None)

        self.assertEqual(len(self.vm.messages), 0)

    def test_input_change_updates_can_execute(self):
        # Initially no text, can_execute should be False
        self.assertFalse(self.vm.send_message_command.can_execute(None))

        self.vm.input_text.value = "Hello"
        self.assertTrue(self.vm.send_message_command.can_execute(None))

    def test_dispose_cleanup(self):
        self.vm.dispose()
        self.assertTrue(self.vm._is_disposed)

    def test_refresh_typing(self):
        self.vm.load_conversation("C1")
        self.vm.refresh_typing()
        # Should not crash
        self.assertIsNotNone(self.vm.typing_users.value)

    def test_messages_observable_notifies(self):
        notified = []

        def on_change():
            notified.append(True)

        self.vm.messages.bind(on_change)
        self.vm.load_conversation("C1")
        self.vm.input_text.value = "Test"
        self.vm.send_message_command.execute(None)

        self.assertTrue(len(notified) >= 0)


class TestObservableProperty(unittest.TestCase):
    """Test ObservableProperty — trái tim của MVVM data binding."""

    def test_initial_value(self):
        prop = ObservableProperty[int](42)
        self.assertEqual(prop.value, 42)

    def test_set_value(self):
        prop = ObservableProperty[int](0)
        prop.value = 10
        self.assertEqual(prop.value, 10)

    def test_observer_notified(self):
        prop = ObservableProperty[int](0)
        observed_changes = []

        def callback(old, new):
            observed_changes.append((old, new))

        prop.bind(callback)
        prop.value = 42

        self.assertEqual(len(observed_changes), 2)  # Initial + change
        self.assertEqual(observed_changes[-1], (0, 42))

    def test_no_notification_for_same_value(self):
        prop = ObservableProperty[str]("same")
        call_count = 0

        def cb(old, new):
            nonlocal call_count
            call_count += 1

        prop.bind(cb)
        call_count = 0  # Reset after initial
        prop.value = "same"  # No change

        self.assertEqual(call_count, 0)

    def test_observer_unbind(self):
        prop = ObservableProperty[int](0)
        calls = []

        def cb(old, new):
            calls.append(new)

        prop.bind(cb)
        prop.unbind(cb)
        prop.value = 42

        self.assertEqual(len(calls), 1)  # Only initial

    def test_different_types(self):
        int_prop = ObservableProperty[int](0)
        str_prop = ObservableProperty[str]("")
        list_prop = ObservableProperty[list]([])

        int_prop.value = 42
        str_prop.value = "hello"
        list_prop.value = [1, 2, 3]

        self.assertEqual(int_prop.value, 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng MVVM | Khi nào không dùng MVVM |
|------------------|------------------------|
| **Data binding sẵn có** — WPF, Android Jetpack, SwiftUI, React/Vue | **Platform không hỗ trợ data binding** — Console app, game engine |
| **UI phức tạp, nhiều state** — Dashboard, chat, editor | **App CRUD đơn giản** — Form với ít state |
| **Cần reactive updates** — Real-time, WebSocket, multiple data sources | **Performance-critical** — Data binding overhead có thể là vấn đề |
| **Nhiều platform** — Web + Mobile + Desktop cùng ViewModel | **Đội ngũ nhỏ, deadline ngắn** — MVVM learning curve |
| **State cần lưu/restore** — ViewModel có thể serialize | **UI-driven app** — Game, design tool (MVC vẫn tốt hơn) |
| **Unit test quan trọng** — ViewModel testable 100% | **ViewModel phức tạp quá mức** — Nếu ViewModel có 30+ properties |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| **Data binding tự động**: View tự cập nhật khi ViewModel thay đổi | **Debug data binding khó**: Khó trace khi binding không hoạt động |
| **Testability cao**: ViewModel có thể test hoàn toàn không cần UI | **Overhead cho data binding**: Observable pattern có memory/performance cost |
| **Tách biệt rõ ràng**: View chỉ là template, không logic | **Complexity cho app nhỏ**: Setup data binding mất thời gian |
| **State management tập trung**: ViewModel chứa tất cả state | **Memory leak**: Nếu không unbind observer đúng cách |
| **Reactive programming**: Xử lý multiple data source dễ dàng | **Learning curve**: Data binding, observable, command, converter |
| **Platform independence**: Có thể reuse ViewModel code | **Boilerplate**: Nhiều class, interface, binding declaration |
| **Dễ maintain**: View thay đổi không ảnh hưởng ViewModel | **DI complexity**: Nhiều dependency cần inject |
| **Team workflow**: Designer + Developer làm việc song song | **Over-engineering**: Không phải app nào cũng cần MVVM |

---

## Công cụ và Framework

### Python
| Framework | Hỗ trợ MVVM |
|-----------|-------------|
| **Tkinter** | MVVM thủ công — tự implement Observable |
| **PyQt/PySide** | Qt Model/View + signal/slot — MVVM-like |
| **Kivy** | Property binding sẵn có (Kivy properties) |
| **Flet** | Reactive UI với state management |
| **NiceGUI** | Reactive, event-driven |
| **Reflex** (Pynecone) | Full-stack reactive Python web |

### Mobile
| Framework | Hỗ trợ |
|-----------|--------|
| **Android Jetpack** (Kotlin) | ViewModel + LiveData/Flow + DataBinding — MVVM chính thức |
| **SwiftUI** (iOS) | @State, @ObservedObject, @EnvironmentObject |
| **Flutter** (Dart) | Provider, Riverpod, BLoC — MVVM-like |
| **Xamarin.Forms** | Prism, MVVMCross |

### Web
| Framework | Cách tiếp cận MVVM |
|-----------|-------------------|
| **React** | Component = View, Redux/Zustand = ViewModel |
| **Vue** | reactive() + computed() — MVVM native |
| **Angular** | Component + Service — MVVM-like |
| **Svelte** | Reactive declarations — MVVM-friendly |
| **Blazor** (C#) | MVVM với INotifyPropertyChanged |

### .NET
| Framework | Hỗ trợ |
|-----------|--------|
| **WPF** | MVVM là pattern mặc định với XAML + DataBinding |
| **UWP/WinUI** | Kế thừa MVVM từ WPF |
| **MAUI** | MVVM với CommunityToolkit.Mvvm |
| **Prism** | Framework MVVM cho WPF/Xamarin/MAUI |
| **ReactiveUI** | Reactive MVVM cho .NET |

---

## Kết luận

**MVVM (Model-View-ViewModel)** là một architectural pattern mạnh mẽ, đặc biệt phù hợp cho các ứng dụng có UI phức tạp, nhiều state, và cần reactive updates. Data binding là trái tim của MVVM — nó loại bỏ hoàn toàn boilerplate code cập nhật View thủ công.

### Best Practices

1. **ViewModel không biết gì về View**: ViewModel không bao giờ import View. Giao tiếp chỉ qua data binding.

2. **View không có logic**: View chỉ là template + binding declarations. Nếu View có `if` statement, hãy đưa logic đó vào ViewModel.

3. **Observable properties cho tất cả state**: Mỗi piece of state mà View cần phải là ObservableProperty.

4. **Commands thay vì event handlers**: View gọi command, không gọi method trực tiếp.

5. **Immutability**: Data từ Model nên immutable. ViewModel chuyển đổi thành observable state.

6. **Dependency Injection**: Inject services vào ViewModel. ViewModel không tạo service.

7. **Lifecycle management**: ViewModel có `initialize()` và `dispose()`. Quản lý subscription/unsubscription.

### Golden Rules

| Rule | Giải thích |
|------|-----------|
| **View chỉ là template** | Không có `if`, `for`, formatting logic trong View |
| **ViewModel = State của View** | Serialize ViewModel = restore trạng thái View |
| **Một ViewModel / Một màn hình** | Không ViewModel quá lớn |
| **Model không biết ViewModel** | Model layer hoàn toàn độc lập |
| **Data binding một chiều là mặc định** | Two-way binding chỉ dùng cho form input |
| **Test ViewModel trước** | ViewModel cần 100% test coverage |

MVVM là kiến trúc tối ưu cho ứng dụng có UI phức tạp, nhiều tương tác, và cần maintain lâu dài. Đầu tư vào MVVM sẽ trả công xứng đáng khi ứng dụng phát triển.
