---
id: proxy
title: Proxy
sidebar_label: 🥷 Proxy
sidebar_position: 13
---

# Proxy

**Proxy** cung cấp một object thay thế (placeholder) kiểm soát truy cập đến object gốc.

## Bài toán

Ứng dụng xem video: Mỗi video là file HD hàng trăm MB. `Video.load()` tốn thời gian và băng thông. Trang hiển thị danh sách 100 video dạng thumbnail. Nếu load tất cả video cùng lúc, app sẽ chậm và crash.

## Giải pháp

Proxy `VideoProxy` thay thế `RealVideo`. Chỉ load video thật khi người dùng click play.

```python
class RealVideo:
    def __init__(self, url):
        self.url = url
        self._load()

    def _load(self):
        print(f'📥 Đang tải video {self.url}... (HD 500MB)')

    def play(self):
        print(f'▶️ Phát video: {self.url}')

class VideoProxy:
    def __init__(self, url):
        self.url = url
        self._real_video = None

    def play(self):
        if self._real_video is None:
            self._real_video = RealVideo(self.url)
        self._real_video.play()

# Sử dụng
videos = [
    VideoProxy('cat.mp4'),
    VideoProxy('dog.mp4'),
    VideoProxy('bird.mp4'),
]

# Chỉ hiển thị danh sách — chưa load gì
videos[0].play()  # Lúc này mới load
```

## Biến thể Proxy

| Loại | Mục đích |
|------|----------|
| **Virtual** | Lazy loading (ví dụ trên) |
| **Protection** | Kiểm soát quyền truy cập |
| **Remote** | Gọi object từ xa (RPC) |
| **Logging** | Ghi log mỗi lần truy cập |
| **Cache** | Lưu kết quả để tái sử dụng |

## Thực tế

- Django `QuerySet` lazy evaluation
- SQLAlchemy lazy loading (chỉ query khi truy cập)
- `@lru_cache` decorator trong Python
