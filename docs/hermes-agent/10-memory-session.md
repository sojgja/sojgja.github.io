---
id: 10-memory-session
title: "Memory, session & state"
sidebar_label: "Memory, session & state"
sidebar_position: 10
---

# Memory, session & state

Hermes nhớ mọi thứ trong SQLite (có FTS5 để tìm toàn văn), nén context khi đầy, và tạo
checkpoint để bạn rollback được. Trang này giải thích dữ liệu nằm đâu, tìm lại thế nào, và
phục hồi ra sao khi có sự cố.

## 1. Lưu trữ: SQLite + FTS5 {#luu-tru}

Phiên và trạng thái nằm trong một cơ sở dữ liệu SQLite dưới `~/.hermes/`
(`hermes_state.py` + các file `hermes_state_*.py`). Chỉ mục FTS5 cho
phép tìm kiếm toàn văn toàn bộ hội thoại.

| Khối cấu hình | Ý nghĩa |
| --- | --- |
| `database.journal_mode` | `wal` (mặc định) hoặc `delete` cho filesystem không hợp WAL (virtiofs/NFS/SMB) |
| `database.synchronous` | Mức bền: OFF/NORMAL/FULL/EXTRA. Trên macOS là *sàn*, không phải ghim (giá trị dưới FULL bị từ chối) |

Trạng thái là **theo profile**. `hermes backup` sao lưu cả thư mục
Hermes thành zip.

## 2. Bộ nhớ (memory) {#memory}

Toolset `memory` cho agent tự quản bộ nhớ bền qua phiên. Quá trình "memory capture"
diễn ra có nhịp — agent được nhắc persist kiến thức quan trọng. Bạn cũng có thể yêu cầu xem
hoặc tinh chỉnh bộ nhớ trực tiếp trong chat.

- Bật/tắt trong setup (chế độ Blank Slate tắt memory mặc định).
- Bộ nhớ là theo profile — tách công việc/cá nhân bằng profile.

## 3. Session & tìm lại {#session}

```
hermes -c                      # tiếp tục phiên gần nhất
hermes -r latest               # tương đương, phạm vi theo workspace
hermes -r "tên phiên"          # theo title
# trong chat:
/resume [name]                 # duyệt & tiếp tục phiên có tên
/sessions                      # (TUI): mở switcher
/session_search "..."          # (tool) tìm trong hội thoại cũ
```

Phiên được đặt tên giúp tìm lại dễ hơn: `/new my-experiment` tạo phiên mới có
title ngay.

## 4. Nén context {#nen}

Khi context gần đầy, **context engine** nén bớt (mặc định là summarization có
mất mát). Có thể can thiệp thủ công:

| Lệnh | Việc |
| --- | --- |
| `/compress` | Nén ngay: flush memory + tóm tắt |
| `/compress here [N]` | Giữ N lượt gần nhất nguyên văn, tóm tắt phần còn lại |
| `/compress focus <chủ đề>` | Thu hẹp thứ mà bản tóm tắt giữ lại |
| `/context` | Xem phân rã context window và các context file |

Context engine có thể thay bằng plugin (`developer-guide/context-engine-plugin.md`).
Có cả micro-compaction và prompt caching (Anthropic) để tiết kiệm token.

## 5. Checkpoint & rollback {#checkpoint}

Hermes tạo checkpoint filesystem để bạn quay lại trạng thái trước. Kho shadow nằm ở
`~/.hermes/checkpoints/`.

| Lệnh | Việc |
| --- | --- |
| `/rollback` | Liệt kê hoặc khôi phục checkpoint (`/rollback <số>`) |
| `/diff [staged\|all\|session]` | Xem thay đổi git; `session` = diff tích luỹ mọi thứ Hermes đã đổi |
| `hermes checkpoints` | Xem/prune/xoá kho checkpoint (chạy không tham số = tổng quan) |
| `/snapshot create\|restore\|prune` | Snapshot config/state (khác checkpoint filesystem) |

> **Lưu ý.** Khôi phục snapshot database ghi qua SQLite backup API nên tiến trình đang sống (gateway, dashboard) thấy dữ liệu mới an toàn. Nếu đường đó thất bại khi tiến trình khác vẫn giữ DB, lệnh **từ chối** thay vì liều gây hỏng — hãy dừng bên đang giữ rồi thử lại.

## 6. Backup & phục hồi state {#phuc-hoi}

```
hermes backup                 # sao lưu ~/.hermes thành zip
hermes import                 # phục hồi từ zip backup
hermes logs                   # xem/tail/lọc log
hermes dump                   # tóm tắt setup để hỗ trợ
```

Nếu database gặp sự cố, xem `developer-guide/state-db-recovery.md`. Nguyên tắc:
sao lưu trước khi sửa, và không xoá data root để "cài lại cho sạch".

> **Nguồn.** `website/docs/developer-guide/session-storage.md`, `state-db-recovery.md`, `user-guide/checkpoints-and-rollback.md`, `user-guide/features/*`.
