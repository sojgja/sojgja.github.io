---
id: 21-playbook-team-assistant
title: "Playbook · Trợ lý nhóm trên messaging"
sidebar_label: "Playbook · Trợ lý nhóm trên messaging"
sidebar_position: 21
---

# Playbook · Trợ lý nhóm trên messaging

Dựng một bot Telegram/Slack/Discord cho cả team dùng chung: mỗi người có phiên riêng, phân
quyền theo user, chạy trên server với đầy đủ tool, và có task định kỳ đẩy vào channel nhóm.

## 1. Bot này làm được gì {#muc-tieu}

- Mọi thành viên **đã được duyệt** DM để nhờ: review code, research, lệnh shell, debug.
- Chạy trên server/VPS với đầy đủ tool: terminal, sửa file, web search, code execution.
- **Phiên riêng theo user** — mỗi người một ngữ cảnh hội thoại.
- An toàn mặc định: chỉ user được duyệt tương tác, hai cách phân quyền.
- Task định kỳ (standup, health check, nhắc việc) gửi vào channel nhóm.

## 2. Chuẩn bị {#chuan-bi}

- Hermes cài trên server/VPS (không phải laptop — bot phải chạy liên tục).
- Tài khoản Telegram của bạn (chủ bot).
- Provider LLM đã cấu hình trong `~/.hermes/.env`.

> **Chi phí.** VPS $5/tháng là đủ. Hermes nhẹ — tiền thực nằm ở lời gọi API LLM, và chúng diễn ra từ xa.

## 3. Bước 1: tạo bot Telegram {#tao-bot}

1. Mở Telegram, tìm `@BotFather`.
2. Gửi `/newbot` → đặt display name và username (kết thúc bằng `bot`).
3. Copy **bot token** BotFather trả về.
4. Tuỳ chọn: `/setcommands` để tạo menu lệnh:
`new`, `model`, `status`, `help`, `stop`.

> **Bí mật.** Giữ token bí mật — ai có token là điều khiển được bot. Nếu lộ, dùng `/revoke` trong BotFather để tạo token mới.

## 4. Bước 2: cấu hình gateway {#cau-hinh}

**Cách A — wizard (khuyến nghị):**

```
hermes gateway setup      # chọn Telegram, dán token, nhập user ID
```

**Cách B — thủ công** trong `~/.hermes/.env`:

```
TELEGRAM_BOT_TOKEN=<bot-token-từ-BotFather>
TELEGRAM_ALLOWED_USERS=<telegram-user-id-của-bạn>
```

Tìm user ID Telegram (số, không phải @username): nhắn `@userinfobot`.

## 5. Bước 3: chạy gateway {#chay}

```
hermes gateway            # foreground để xem log
hermes gateway install    # cài làm service (chạy nền)
hermes status             # xem nền tảng đã kết nối
```

## 6. Bước 4: phân quyền (admin/user) {#phan-quyen}

Nền tảng có allowlist theo user hỗ trợ hai tầng trong block `extra:` của nền tảng
trong `~/.hermes/config.yaml`:

| Cấu hình | Ý nghĩa |
| --- | --- |
| `allow_admin_from` | User là admin — có *mọi* slash command |
| `user_allowed_commands` | Lệnh user thường được dùng (cộng sàn `/help`, `/whoami`) |
| `group_allow_admin_from` / `group_user_allowed_commands` | Bản tương ứng cho nhóm |

**DM pairing** — thay vì thu thập user ID thủ công:

```
# thành viên DM bot → nhận mã ghép đôi một lần
hermes pairing approve telegram XKGH5N7P
```

> **Tuyệt đối tránh.** Không bao giờ đặt `GATEWAY_ALLOW_ALL_USERS=true` trên bot có quyền terminal. Mặc định của Hermes là *deny*: không cấu hình allowlist thì mọi user bị từ chối — hãy giữ nó tường minh.

## 7. Bước 5: dùng trong nhóm & task định kỳ {#nhom}

```
# trong chat nhóm, đặt home channel cho job định kỳ
/sethome
```

Task định kỳ cho team (ví dụ standup/health check) gửi vào channel:

```
hermes cron create "0 9 * * 1-5" \
  "Post a daily standup prompt to the team channel: ask each member for
   yesterday/today/blockers. Keep it short." \
  --name "team-standup" --deliver telegram
```

## 8. Best practices cho team {#best}

| Chủ đề | Khuyến nghị |
| --- | --- |
| Phân quyền | Allowlist theo numeric ID hoặc pairing; tách admin/user rõ ràng |
| Cách ly | Chạy gateway trên VPS riêng; backend terminal `docker` |
| Home channel | `/sethome` để cron có đích gửi |
| Đặt tên phiên | `/title` để phân biệt các phiên |
| Chế độ hiển thị | Trong messaging để `/verbose` ở mức "new" cho gọn |
| Bí mật | Token trong `.env`, `chmod 600`, không commit |
| Giám sát | Xem `~/.hermes/logs/` cho truy cập bất thường |

> **Nguồn.** `guides/team-telegram-assistant.md`, `reference/slash-commands.md` (phân quyền admin/user), `guides/tips.md` (DM pairing, home channel).
