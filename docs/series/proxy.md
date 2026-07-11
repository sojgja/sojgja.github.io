---
id: proxy
title: Proxy
sidebar_label: 🥷 Proxy
sidebar_position: 13
---

# Proxy

> "Provide a surrogate or placeholder for another object to control access to it." — Erich Gamma, *Design Patterns: Elements of Reusable Object-Oriented Software*

## Bài toán chi tiết

Một công ty công nghệ giáo dục (edtech) đang xây dựng nền tảng học trực tuyến với thư viện video bài giảng khổng lồ. Mỗi video bài giảng là file HD có dung lượng từ 200 MB đến 2 GB, được lưu trữ trên cloud storage (AWS S3). Nền tảng hiển thị danh sách khóa học với thumbnail, mỗi khóa học có thể chứa 50-100 video. Khi người dùng truy cập trang danh sách, hệ thống phải hiển thị thông tin của tất cả video (tên, thời lượng, kích thước, thumbnail).

Vấn đề xảy ra khi implement theo cách thông thường: mỗi object Video, khi được khởi tạo, sẽ tải toàn bộ metadata và có thể prefetch một phần nội dung để lấy thông tin. Với 100 video trong một trang, điều này đồng nghĩa với việc tải 100 file metadata từ S3 — mỗi request mất 200-500ms, tổng cộng 20-50 giây cho một lần load trang. Kết quả là trang load cực kỳ chậm, người dùng thoát trang trước khi danh sách hiển thị xong.

Giải pháp naive là lazy load — chỉ tải video khi người dùng click vào. Nhưng nếu client gọi trực tiếp constructor của class Video (với logic tải nặng bên trong), thì lazy load không khả thi vì constructor đã kích hoạt việc tải. Cần một đối tượng thay thế (placeholder) có thể khởi tạo nhanh, chỉ tải dữ liệu thật khi cần — đây chính là lúc Proxy Pattern phát huy tác dụng.

Ngoài ra, còn các vấn đề khác: kiểm soát quyền truy cập (chỉ student đã đăng ký mới xem được), caching (video phổ biến nên cache), logging (ai đã xem video nào), remote access (video từ server khác). Tất cả đều có thể giải quyết bằng các biến thể khác nhau của Proxy.

## Giải pháp với Pattern

Proxy Pattern cung cấp một object thay thế (surrogate) có cùng interface với object thật. Proxy kiểm soát truy cập đến object thật, có thể thêm các hành vi bổ sung trước hoặc sau khi ủy quyền (delegate) cho object thật. Quan trọng: Proxy và RealSubject implement cùng một interface, nên client hoàn toàn không biết mình đang dùng proxy hay object thật.

Các biến thể chính của Proxy:
- **Virtual Proxy**: Lazy loading — chỉ tạo RealSubject khi thực sự cần. Phù hợp với video, hình ảnh, tài liệu lớn.
- **Protection Proxy**: Kiểm soát quyền truy cập — kiểm tra authentication/authorization trước khi delegate.
- **Remote Proxy**: Đại diện cho object ở remote location — che giấu chi tiết mạng (RPC, gRPC, REST).
- **Logging Proxy**: Ghi log mỗi lần method được gọi — auditing, monitoring.
- **Caching Proxy**: Lưu kết quả để tái sử dụng — giảm tải cho backend.

## Phân tích thiết kế

Proxy Pattern dựa trên nguyên lý **Single Responsibility Principle**: proxy chịu trách nhiệm kiểm soát truy cập (một việc cụ thể), không trộn lẫn với business logic của RealSubject. Nó cũng thể hiện **Lazy Initialization** và **Principle of Least Privilege** (bảo vệ tài nguyên).

**Proxy vs other patterns:**
- Proxy vs Decorator: Cả hai đều wrap object và giữ nguyên interface. Decorator thêm hành vi, Proxy kiểm soát truy cập. Decorator có thể wrap decorator khác tạo thành chain; Proxy thường chỉ wrap RealSubject. Decorator client biết mình đang dùng decorator (có thể), Proxy client không cần biết.
- Proxy vs Adapter: Adapter thay đổi interface, Proxy giữ nguyên interface. Adapter làm cho class tương thích, Proxy kiểm soát truy cập.

**Khi KHÔNG nên dùng Proxy:**
- Khi object thật khởi tạo nhanh — virtual proxy là over-engineering.
- Khi security có thể xử lý ở tầng kiến trúc khác (middleware, firewall).
- Khi performance overhead của proxy lớn hơn lợi ích.

**Trade-offs:**
- Thêm một lớp gián tiếp — tăng latency nhẹ.
- Tăng độ phức tạp — thêm class, thêm code.
- Cần đồng bộ giữa proxy và real subject interface.
- Nếu proxy có bug, có thể gây lỗi khó debug.

## Ví dụ code hoàn chỉnh

### Cách làm sai: Load trực tiếp (không proxy)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol
import time


class Video:
    """Video được tải ngay khi khởi tạo — chậm, tốn tài nguyên."""

    def __init__(self, video_id: str, title: str, url: str) -> None:
        self.video_id = video_id
        self.title = title
        self.url = url
        self._metadata: dict = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Giả lập tải metadata từ S3 — mất 0.5 giây."""
        time.sleep(0.5)
        self._metadata = {
            "duration": "45:30",
            "size_mb": 512,
            "resolution": "1920x1080",
            "thumbnail": f"https://cdn.example.com/thumbs/{self.video_id}.jpg",
        }
        print(f"[Loaded] {self.title} — metadata fetched")

    def get_info(self) -> dict:
        return {
            "id": self.video_id,
            "title": self.title,
            **self._metadata,
        }

    def play(self) -> str:
        return f"▶️ Playing: {self.title} ({self.url})"


# Client — load tất cả video cùng lúc
videos = [
    Video("v001", "Python Basics", "https://s3.example.com/v001.mp4"),
    Video("v002", "OOP in Python", "https://s3.example.com/v002.mp4"),
    Video("v003", "Design Patterns", "https://s3.example.com/v003.mp4"),
]
# Mất 1.5 giây cho 3 video — với 100 video mất 50 giây!
```

### Cách đúng: Virtual Proxy + Protection Proxy

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import time


# --- Common Interface ---
class Video(ABC):
    """Interface chung cho RealVideo và VideoProxy."""

    @abstractmethod
    def get_info(self) -> dict:
        ...

    @abstractmethod
    def play(self, user: User) -> str:
        ...


# --- Value Objects ---
class UserRole(Enum):
    GUEST = auto()
    STUDENT = auto()
    INSTRUCTOR = auto()
    ADMIN = auto()


@dataclass
class User:
    user_id: str
    name: str
    role: UserRole
    enrolled_courses: set[str] = set()


# --- Real Subject ---
class RealVideo(Video):
    """Video thật — chứa logic tải nặng."""

    def __init__(self, video_id: str, title: str, url: str, course_id: str) -> None:
        self.video_id = video_id
        self.title = title
        self.url = url
        self.course_id = course_id
        self._metadata: dict = {}

    def _ensure_loaded(self) -> None:
        if not self._metadata:
            self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Giả lập tải từ S3."""
        time.sleep(0.3)
        return {
            "duration": "45:30",
            "size_mb": 512,
            "resolution": "1920x1080",
            "thumbnail": f"https://cdn.example.com/thumbs/{self.video_id}.jpg",
            "format": "mp4",
            "codec": "h264",
        }

    def get_info(self) -> dict:
        self._ensure_loaded()
        return {
            "id": self.video_id,
            "title": self.title,
            "course_id": self.course_id,
            **self._metadata,
        }

    def play(self, user: User) -> str:
        self._ensure_loaded()
        return f"▶️ Streaming {self.title} ({self.video_id}) — {self._metadata['resolution']}"


# --- Virtual Proxy (Lazy Loading) ---
class VideoProxy(Video):
    """Virtual Proxy — chỉ tải RealVideo khi thực sự cần."""

    def __init__(self, video_id: str, title: str, url: str, course_id: str) -> None:
        self.video_id = video_id
        self.title = title
        self.url = url
        self.course_id = course_id
        self._real_video: Optional[RealVideo] = None

    def _get_real(self) -> RealVideo:
        if self._real_video is None:
            print(f"[Proxy] Loading RealVideo: {self.title}")
            self._real_video = RealVideo(self.video_id, self.title, self.url, self.course_id)
        return self._real_video

    def get_info(self) -> dict:
        # Ưu tiên trả về thông tin cơ bản mà không load full metadata
        return {
            "id": self.video_id,
            "title": self.title,
            "course_id": self.course_id,
            "duration": "Loading...",
            "thumbnail": f"https://cdn.example.com/thumbs/{self.video_id}.jpg",
        }

    def play(self, user: User) -> str:
        return self._get_real().play(user)


# --- Protection Proxy ---
class ProtectedVideoProxy(Video):
    """Protection Proxy — kiểm tra quyền trước khi delegate."""

    def __init__(self, target: Video, required_role: UserRole = UserRole.STUDENT) -> None:
        self._target = target
        self._required_role = required_role

    def get_info(self) -> dict:
        return self._target.get_info()

    def play(self, user: User) -> str:
        if user.role.value < self._required_role.value and self._required_role == UserRole.INSTRUCTOR:
            return f"⛔ Access denied: {user.name} ({user.role.name}) cannot play this video"

        if user.role == UserRole.STUDENT and hasattr(self._target, 'course_id'):
            if self._target.course_id not in user.enrolled_courses:
                return f"⛔ {user.name} not enrolled in course {self._target.course_id}"

        return self._target.play(user)


# --- Logging Proxy ---
class LoggingVideoProxy(Video):
    """Logging Proxy — ghi log mỗi lần truy cập."""

    def __init__(self, target: Video) -> None:
        self._target = target
        self._access_log: list[dict] = []

    def get_info(self) -> dict:
        return self._target.get_info()

    def play(self, user: User) -> str:
        self._access_log.append({
            "user_id": user.user_id,
            "user_name": user.name,
            "video_id": self._target.video_id,
            "action": "play",
            "timestamp": time.time(),
        })
        print(f"[Log] {user.name} played video {self._target.title}")
        return self._target.play(user)

    def get_access_log(self) -> list[dict]:
        return self._access_log


# --- Caching Proxy ---
class CachingVideoProxy(Video):
    """Caching Proxy — cache kết quả để tránh load lại."""

    def __init__(self, target: Video, ttl_seconds: int = 60) -> None:
        self._target = target
        self._cache: Optional[dict] = None
        self._cache_time: float = 0
        self._ttl = ttl_seconds

    def get_info(self) -> dict:
        now = time.time()
        if self._cache is None or (now - self._cache_time) > self._ttl:
            print("[Cache] Fetching fresh data...")
            self._cache = self._target.get_info()
            self._cache_time = now
        else:
            print("[Cache] Serving from cache")
        return self._cache

    def play(self, user: User) -> str:
        return self._target.play(user)


# --- Usage ---
if __name__ == "__main__":
    # Tạo danh sách video với Virtual Proxy — không load gì cả
    print("=== Creating video catalog (no loading) ===")
    catalog: list[Video] = [
        VideoProxy("v001", "Python Basics", "url", "course_py"),
        VideoProxy("v002", "Advanced Python", "url", "course_py"),
        VideoProxy("v003", "Design Patterns", "url", "course_dp"),
    ]
    for v in catalog:
        print(f"  - {v.title}: {v.get_info()['duration']}")

    # Khi user play — mới thực sự load
    print("\n=== Playing video ===")
    user = User(user_id="u1", name="Alice", role=UserRole.STUDENT, enrolled_courses={"course_py"})
    result = catalog[0].play(user)
    print(result)

    # Protection Proxy
    print("\n=== Access Control ===")
    protected = ProtectedVideoProxy(catalog[1], required_role=UserRole.STUDENT)
    guest = User(user_id="g1", name="Bob", role=UserRole.GUEST)
    print(protected.play(guest))  # Access denied

    # Logging Proxy wrap Virtual Proxy
    print("\n=== Logging + Lazy Loading ===")
    logging_proxy = LoggingVideoProxy(VideoProxy("v004", "Proxy Pattern", "url", "course_dp"))
    student = User("u2", "Charlie", UserRole.STUDENT, {"course_dp"})
    logging_proxy.play(student)
    logging_proxy.play(student)
    print(f"Access log entries: {len(logging_proxy.get_access_log())}")

    # Caching Proxy
    print("\n=== Caching ===")
    real = RealVideo("v005", "Flyweight", "url", "course_dp")
    cached = CachingVideoProxy(real, ttl_seconds=5)
    cached.get_info()  # Fetch
    cached.get_info()  # Cache
    cached.get_info()  # Cache
```

## Sơ đồ UML

```
┌──────────────────────┐
│     «interface»      │
│       Video           │
│──────────────────────│
│+ get_info() → dict   │
│+ play(User) → str    │
└──────────┬───────────┘
           │
     ┌─────┼─────┬──────────────────┬──────────────────┐
     │     │     │                  │                  │
┌────┴──┐ ┌┴────┐│           ┌─────┴──────┐   ┌──────┴──────┐
│Real   │ │Video││           │LoggingProxy│   │CachingProxy │
│Video  │ │Proxy││           │- target    │   │- target     │
│(Real  │ │(Virt││           │- accessLog │   │- cache      │
│Subject)│ │ual) ││           └────────────┘   └─────────────┘
└───────┘ └─────┘│
                  │
          ┌───────┴────────────┐
          │ProtectedVideoProxy │
          │- target            │
          │- requiredRole      │
          │+ play(User)        │
          └────────────────────┘

Kết hợp (nested proxies):
client → LoggingProxy → ProtectedProxy → CachingProxy → VideoProxy → RealVideo
```

## So sánh với Pattern liên quan

**Proxy vs Decorator**: Đây là hai pattern dễ nhầm nhất. Cả hai đều wrap một object và implement cùng interface. Decorator thêm hành vi mới một cách chủ động — nó bổ sung chức năng. Proxy kiểm soát truy cập một cách thụ động — nó quản lý việc sử dụng object thật. Decorator có thể wrap thành chain dài không giới hạn; Proxy thường chỉ có một lớp duy nhất. Decorator client biết (hoặc không biết) mình đang dùng decorator; Proxy client thường không biết.

**Proxy vs Adapter**: Adapter thay đổi interface để tương thích; Proxy giữ nguyên interface. Adapter giải quyết vấn đề không tương thích; Proxy giải quyết vấn đề kiểm soát truy cập, lazy loading, remote access.

**Proxy vs Facade**: Proxy kiểm soát truy cập đến một object duy nhất. Facade cung cấp interface đơn giản hóa cho cả một subsystem. Proxy giữ nguyên interface, Facade tạo interface mới.

## Ứng dụng thực tế

**1. Django QuerySet — Lazy Evaluation**: Django QuerySet là Virtual Proxy kinh điển. Khi bạn filter, exclude, annotate, bạn đang xây dựng một QuerySet proxy — chưa có SQL nào được chạy. Chỉ khi bạn iterate, slice, hoặc gọi `list()`, QuerySet mới thực sự truy vấn database:

```python
# Chưa có query nào — Virtual Proxy
qs = Article.objects.filter(published=True).order_by('-created_at')

# Lúc này mới thực sự query database
for article in qs:
    print(article.title)

# .count() cũng kích hoạt query
total = qs.count()
```

**2. SQLAlchemy — Lazy Loading**: SQLAlchemy dùng proxy pattern cho relationships. Khi bạn truy cập `user.posts`, SQLAlchemy trả về một `AppenderQuery` proxy — chỉ query database khi bạn thực sự truy cập dữ liệu:

```python
from sqlalchemy.orm import lazyload

user = session.query(User).first()
# user.posts là proxy — chưa query
posts = user.posts  # Lúc này mới query: SELECT * FROM posts WHERE user_id = ?
```

**3. Python `@lru_cache` — Caching Proxy**: `functools.lru_cache` là một caching proxy decorator. Nó wrap function gốc, lưu kết quả vào cache dựa trên arguments:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Lần đầu gọi với n=100 — tính toán thật
# Lần thứ hai — trả về từ cache
result = fibonacci(100)
```

**4. Java RMI / Python Pyro — Remote Proxy**: Remote Method Invocation dùng proxy để che giấu chi tiết mạng. Client gọi method trên proxy, proxy serializes arguments, gửi qua network, nhận kết quả, deserializes và trả về:

```python
# Pyro4 — Remote Proxy example
import Pyro4

# Client — gọi method trên proxy (che giấu network)
uri = "PYRO:obj_1234@localhost:9090"
remote_obj = Pyro4.Proxy(uri)  # Remote Proxy
result = remote_obj.calculate(42)  # Thực tế chạy trên server khác
```

## Kiểm thử

```python
import pytest
from unittest.mock import MagicMock, patch
from proxy import (
    Video, RealVideo, VideoProxy, ProtectedVideoProxy,
    LoggingVideoProxy, CachingVideoProxy, User, UserRole,
)


class TestBedVideo:
    def setup_method(self) -> None:
        self.video = RealVideo("v001", "Test", "url", "course_01")

    def test_get_info_triggers_load(self) -> None:
        info = self.video.get_info()
        assert info["id"] == "v001"
        assert info["duration"] is not None

    def test_play_returns_string(self) -> None:
        user = User("u1", "Alice", UserRole.STUDENT)
        result = self.video.play(user)
        assert "Playing" in result or "Streaming" in result


class TestVideoProxy:
    def test_no_real_video_on_creation(self) -> None:
        proxy = VideoProxy("v001", "Test", "url", "course_01")
        assert proxy._real_video is None

    def test_real_video_created_on_play(self) -> None:
        proxy = VideoProxy("v001", "Test", "url", "course_01")
        user = User("u1", "Alice", UserRole.STUDENT, {"course_01"})
        proxy.play(user)
        assert proxy._real_video is not None

    def test_reuses_real_video(self) -> None:
        proxy = VideoProxy("v001", "Test", "url", "course_01")
        user = User("u1", "Alice", UserRole.STUDENT, {"course_01"})
        proxy.play(user)
        real = proxy._real_video
        proxy.play(user)
        assert proxy._real_video is real  # Same instance


class TestProtectedVideoProxy:
    def test_student_can_play_enrolled(self) -> None:
        video = RealVideo("v001", "Test", "url", "course_01")
        proxy = ProtectedVideoProxy(video, UserRole.STUDENT)
        user = User("u1", "Alice", UserRole.STUDENT, {"course_01"})
        result = proxy.play(user)
        assert "⛔" not in result

    def test_guest_denied(self) -> None:
        video = RealVideo("v001", "Test", "url", "course_01")
        proxy = ProtectedVideoProxy(video, UserRole.STUDENT)
        user = User("g1", "Bob", UserRole.GUEST)
        result = proxy.play(user)
        assert "⛔" in result

    def test_unenrolled_student_denied(self) -> None:
        video = RealVideo("v001", "Test", "url", "course_01")
        proxy = ProtectedVideoProxy(video, UserRole.STUDENT)
        user = User("u1", "Alice", UserRole.STUDENT, {"course_02"})
        result = proxy.play(user)
        assert "⛔" in result


class TestLoggingProxy:
    def test_logs_access(self) -> None:
        real = RealVideo("v001", "Test", "url", "course_01")
        proxy = LoggingVideoProxy(real)
        user = User("u1", "Alice", UserRole.STUDENT, {"course_01"})
        proxy.play(user)
        assert len(proxy.get_access_log()) == 1

    def test_logs_multiple_accesses(self) -> None:
        real = RealVideo("v001", "Test", "url", "course_01")
        proxy = LoggingVideoProxy(real)
        user = User("u1", "Alice", UserRole.STUDENT, {"course_01"})
        proxy.play(user)
        proxy.play(user)
        proxy.play(user)
        assert len(proxy.get_access_log()) == 3


class TestCachingProxy:
    def test_cache_hits(self) -> None:
        real = RealVideo("v001", "Test", "url", "course_01")
        proxy = CachingVideoProxy(real, ttl_seconds=60)
        info_first = proxy.get_info()
        info_second = proxy.get_info()
        assert info_first == info_second

    def test_cache_expiry(self) -> None:
        real = RealVideo("v001", "Test", "url", "course_01")
        proxy = CachingVideoProxy(real, ttl_seconds=0)  # Expire immediately
        import time
        time.sleep(0.1)
        info_first = proxy.get_info()
        info_second = proxy.get_info()
        assert info_first == info_second


class TestProxyStack:
    def test_multiple_proxies_compose(self) -> None:
        """Nhiều proxy có thể kết hợp với nhau."""
        video = RealVideo("v001", "Design Patterns", "url", "course_dp")
        proxy: Video = LoggingVideoProxy(
            ProtectedVideoProxy(
                CachingVideoProxy(video)
            )
        )
        user = User("u1", "Alice", UserRole.STUDENT, {"course_dp"})
        result = proxy.play(user)
        assert result is not None
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Kiểm soát truy cập tinh vi — virtual, protection, caching, logging | Thêm độ phức tạp — nhiều class, nhiều layer |
| Lazy loading — tiết kiệm tài nguyên cho object nặng | Tăng latency — mỗi proxy thêm một lần gọi hàm |
| Dễ kết hợp — nhiều proxy có thể stack lên nhau | Debugging khó — stack trace sâu, nhiều lớp wrap |
| Tuân thủ SRP — proxy giữ access control tách biệt | Nếu thiết kế không cẩn thận, proxy trở nên phức tạp |
| Client không cần thay đổi code — cùng interface | Cần đồng bộ interface giữa proxy và real subject |

## Kết luận

Proxy Pattern là giải pháp linh hoạt cho nhiều vấn đề về kiểm soát truy cập: từ lazy loading (virtual proxy), bảo vệ quyền (protection proxy), caching, logging, đến remote access. Proxy cho phép thêm các cross-cutting concerns mà không làm ô nhiễm business logic của class chính.

**Nguyên tắc vàng**: Hãy dùng Proxy khi bạn cần thêm một lớp kiểm soát trước khi truy cập vào object thật — và việc kiểm soát đó không thuộc trách nhiệm của object thật. Hãy nhớ: Proxy và RealSubject phải cùng interface để client không bị ảnh hưởng. Và đừng lạm dụng — nếu chỉ cần lazy loading cho một object, một `if` đơn giản trong method getter cũng đủ, không cần proxy pattern phức tạp.
