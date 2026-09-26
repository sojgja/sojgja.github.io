---
id: 12-cron-automation
title: "Cron & tự động hoá"
sidebar_label: "Cron & tự động hoá"
sidebar_position: 12
---

# Cron & tự động hoá

Hermes có scheduler cron sẵn trong: báo cáo ngày, backup đêm, audit tuần — mô tả bằng ngôn ngữ
tự nhiên, chạy không cần người. Và khi không cần LLM, có hai đường zero-token.

## 1. Khái niệm then chốt {#khai-niem}

> **Quan trọng.** Cron job chạy trong **phiên agent mới**, không nhớ gì về chat hiện tại. Prompt phải **tự chứa đầy đủ ngữ cảnh** — mọi thứ agent cần phải nằm trong prompt hoặc trong script đi kèm.

Tham số `script` là vũ khí: một script Python chạy *trước* mỗi lần thực thi,
stdout của nó trở thành ngữ cảnh cho agent. Script lo phần cơ học (fetch, diff); agent lo phần
suy luận (thay đổi này có đáng quan tâm không?).

## 2. Quản lý cron {#quan-ly}

```
hermes cron                  # xem & tick scheduler
hermes pause                 # dừng khẩn cấp toàn cục (không có cron mới nổ)
hermes resume                # chạy lại
```

Trong chat, tool `cronjob_manage` biết khi nào nên chọn chế độ `no_agent=True`
và tự viết script cho bạn — bạn chỉ cần mô tả bằng lời.

## 3. Zero-token: script-only & `hermes send` {#script-only}

| Tình huống | Dùng |
| --- | --- |
| Watchdog định kỳ mà script đã tạo ra *đúng* thông điệp (cảnh báo đĩa, heartbeat) | Cron script-only (`no_agent=True`) — cùng scheduler, không LLM |
| Một phát từ script đang chạy (bước CI, post-commit hook, deploy script) | `hermes send` — pipe stdout/file thẳng tới Telegram/Discord/Slack… |

## 4. Mẫu thực tế {#mau}

### Mẫu 1 — Giám sát thay đổi website

Script băm nội dung URL, lưu state; agent chỉ được gọi khi có thay đổi:

```
# ~/.hermes/scripts/watch-site.py
import hashlib, json, os, urllib.request

URL = "https://example.com/pricing"
STATE = os.path.expanduser("~/.hermes/scripts/.watch-site-state.json")
# ... fetch, so sánh hash, in diff nếu khác ...
```

Cron cấu hình `script=watch-site.py`; agent đọc stdout và quyết định có nên báo bạn không.

### Các mẫu khác trong tài liệu gốc

- Báo cáo định kỳ (daily briefing)
- Pipeline nhiều skill
- Audit/kiểm tra bảo mật định kỳ
- Backup đêm

Xem `guides/automation-blueprints.md` và `reference/automation-blueprints-catalog.mdx`.

## 5. Bền vững & an toàn {#durable}

| Chủ đề | Ghi chú |
| --- | --- |
| Bền vững (durable) | Cần việc sống qua đóng phiên/khởi động lại → cron hoặc `terminal(background=True, notify_on_complete=True)`, không dùng delegation cấp cao |
| Dừng khẩn cấp | `hermes pause` chặn cron mới, kanban, gateway; việc đang chạy không bị kill |
| Catch-up | Có cơ chế misfire catch-up; heartbeat/loop trong phiên thì không bền |
| Gỡ lỗi | `guides/cron-troubleshooting.md` |

Phân biệt rõ ba thứ dễ lẫn: `hermes cron` (bền, cô lập) · `/loop`,
`/heartbeat` (trong phiên, trong tiến trình) · `hermes send` (một phát,
không agent).

> **Nguồn.** `website/docs/guides/automate-with-cron.md`, `cron-script-only.md`, `cron-troubleshooting.md`, `guides/pipe-script-output.md`.
