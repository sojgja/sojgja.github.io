---
id: 24-playbook-trien-khai
title: "Playbook · Triển khai gateway production"
sidebar_label: "Playbook · Triển khai gateway production"
sidebar_position: 24
---

# Playbook · Triển khai gateway production

Đưa agent lên chạy thật 24/7: đóng gói Docker, cách ly thực thi qua SSH, checklist hardening,
worktree cho agent song song, và pattern `BOOT.md` để chạy checklist mỗi lần gateway
khởi động.

## 1. Kiến trúc triển khai an toàn {#kien-truc}

Nguyên tắc: **tách nơi agent nhận lệnh khỏi nơi lệnh được thực thi**. Gateway
(messaging + agent) chạy một nơi; lệnh shell chạy trong container hoặc máy khác.

```
[Telegram/Slack] ──▶ Hermes gateway (VPS) ──▶ terminal backend
                                              ├─ docker  (container)
                                              ├─ ssh     (máy worker)
                                              └─ modal/daytona (serverless)
```

## 2. Docker / Compose {#docker}

```
# từ checkout source
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d
```

| Thành phần | Ý nghĩa |
| --- | --- |
| `HERMES_UID`/`HERMES_GID` | Remap user trong container theo user sở hữu `~/.hermes` → file vẫn đọc/ghi được trên host |
| `~/.hermes:/opt/data` | Dữ liệu bền (config, session, skill) nằm trên host |
| `/init` (s6-overlay) | PID 1: chown, reconcile profile, bật/tắt dashboard trước khi service chạy. **Không bỏ qua** |
| `restart: unless-stopped` | Tự khởi động lại |

> **Bảo mật.** Dashboard mặc định bind `127.0.0.1` và lưu API key. Muốn truy cập ngoài: SSH tunnel hoặc reverse proxy có xác thực — **không** `--insecure --host 0.0.0.0`. Docker **không** hỗ trợ `hermes update`: cập nhật bằng image mới.

## 3. Cách ly thực thi: SSH backend {#cach-ly}

Chạy gateway trên một máy, thực thi lệnh trên máy worker khác. Chi tiết kết nối để trong
`.env` (không phải `config.yaml`) để không bị chia sẻ khi export profile:

```
# ~/.hermes/config.yaml
terminal:
  backend: ssh
```

```
# ~/.hermes/.env
TERMINAL_SSH_HOST=agent-worker.local
TERMINAL_SSH_USER=hermes
TERMINAL_SSH_KEY=~/.ssh/hermes_agent_key
```

Tương tự với `backend: docker` — mọi container chạy với cấu hình hardened: bỏ hết
Linux capability (chỉ add lại tối thiểu), `no-new-privileges`, giới hạn số tiến
trình, tmpfs có giới hạn kích thước.

## 4. Hardening checklist {#hardening}

```
approvals:
  mode: manual
  timeout: 300
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"

security:
  redact_secrets: true

checkpoints:
  enabled: true

terminal:
  backend: docker
  cwd: /path/to/workdir        # không để agent chạy từ thư mục nhạy cảm
  docker_forward_env: []       # không rò secret vào container
```

```
# .env
TELEGRAM_ALLOWED_USERS=123456789
# KHÔNG đặt GATEWAY_ALLOW_ALL_USERS=true
chmod 600 ~/.hermes/.env
```

| # | Việc |
| --- | --- |
| 1 | Allowlist tường minh, không allow-all |
| 2 | Backend container/ssh |
| 3 | Giới hạn CPU/memory/disk |
| 4 | Secret trong `.env`, quyền 600 |
| 5 | Bật DM pairing |
| 6 | Audit `command_allowlist` định kỳ |
| 7 | Đặt `terminal.cwd` |
| 8 | Chạy non-root |
| 9 | Giám sát `~/.hermes/logs/` |
| 10 | Cập nhật thường xuyên |

## 5. Pattern `BOOT.md` {#boot}

Pattern phổ biến từ cộng đồng: đặt một checklist markdown ở `~/.hermes/BOOT.md` và
để agent chạy nó **mỗi lần gateway khởi động** — ví dụ "kiểm tra cron lỗi qua đêm
và ping Discord nếu có", hoặc "tóm tắt 24h log deploy gần nhất".

### Bước 1 — viết checklist

```
# ~/.hermes/BOOT.md

# Startup Checklist
- Kiểm tra ~/.hermes/logs/ cho lỗi cron qua đêm
- Nếu có lỗi, ping Discord #ops với tóm tắt
- Nếu hôm nay là thứ Hai, kiểm tra log deploy tuần
```

### Bước 2 — đăng ký gateway event hook

Tạo thư mục hook trong profile home với `HOOK.yaml` + `handler.py`; hook
chạy lúc gateway khởi động và spawn một phiên agent một-phát để thực thi checklist.

```
# ~/.hermes/hooks/boot-checklist/
#   HOOK.yaml   — khai báo event kích hoạt (gateway startup)
#   handler.py  — spawn agent one-shot đọc BOOT.md và làm theo
```

> **Mở rộng.** Xoá `~/.hermes/BOOT.md` để tắt (hook vẫn nạp nhưng tự bỏ qua khi file không có). Có thể key theo ngày trong tuần (thứ Hai làm việc khác). Nhiều checklist → trỏ hook tới file khác (`STARTUP.md`, `MORNING.md`).

> **Mô hình tin cậy.** Gateway nạp mọi thư mục hook hợp lệ trong tiến trình, với quyền của gateway. Hãy đối xử với `~/.hermes/hooks/` như `config.yaml` — **đọc kỹ handler.py** trước khi đặt vào, và đưa `ls ~/.hermes/hooks/` vào quy trình audit định kỳ.

## 6. Agent song song với git worktree {#worktree}

Chạy nhiều agent trên cùng repo rất dễ giẫm chân. Cách an toàn: mỗi agent một worktree + branch
riêng.

```
cd /path/to/your/repo
hermes -w            # tự tạo worktree tạm dưới .worktrees/ + branch riêng
hermes -w -z "Fix issue #123"
```

Mở nhiều terminal, mỗi cái một `hermes -w` → mỗi tiến trình một worktree độc lập.

- Một worktree mỗi thí nghiệm; đặt tên branch theo thí nghiệm.
- Commit thường xuyên cho mốc lớn; dùng checkpoint/`/rollback` làm lưới an toàn giữa các commit.
- Không chạy Hermes từ repo root khi dùng worktree — ưu tiên thư mục worktree để phạm vi rõ ràng.

## 7. Cập nhật, backup & giám sát {#cap-nhat}

```
hermes backup                 # sao lưu ~/.hermes (zip) trước nâng cấp
hermes update                 # source/bundle install (KHÔNG dùng cho Docker)
hermes migrate                # nâng config qua thay đổi schema
hermes doctor                 # chẩn đoán sau nâng cấp
hermes status / logs / usage  # trạng thái, log, hạn mức
hermes pause / resume         # dừng khẩn cấp toàn cục
hermes security audit         # audit chuỗi cung ứng (OSV.dev)
```

> **Trước khi nâng cấp.** Sao lưu `~/.hermes`; ghi lại thay đổi cục bộ; sau đó chạy `hermes doctor` và một chat thật để xác minh. Đọc `developer-guide/stable-releases.md` khi cần.

> **Nguồn.** `docker-compose.yml`, `user-guide/security.md` (Network Isolation, Gateway Deployment Checklist), `developer-guide/git-worktrees.md`, `user-guide/features/hooks.md` + `llms-full.txt` (BOOT.md pattern), `getting-started/updating.md`.
