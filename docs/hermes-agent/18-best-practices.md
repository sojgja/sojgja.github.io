---
id: 18-best-practices
title: "Best practices cốt lõi (tổng hợp mạnh nhất)"
sidebar_label: "Best practices cốt lõi (tổng hợp mạnh nhất)"
sidebar_position: 18
---

# Best practices cốt lõi (tổng hợp mạnh nhất)

Trang này gom những best practice **đã được kiểm chứng trong tài liệu sản xuất**
của Hermes: checklist triển khai production, tư thế bảo mật, kinh tế prompt cache, kỷ luật
context/memory, và an toàn khi agent sửa file. Đây là trang nên đọc trước mọi playbook khác.

## 1. Checklist triển khai production {#checklist}

Trích nguyên từ tài liệu gateway của Hermes — mười việc bắt buộc trước khi cho chạy thật:

| # | Việc |
| --- | --- |
| 1 | **Đặt allowlist tường minh** — tuyệt đối không dùng `GATEWAY_ALLOW_ALL_USERS=true` trong production |
| 2 | **Dùng backend container** — `terminal.backend: docker` trong `config.yaml` |
| 3 | **Giới hạn tài nguyên** — CPU, memory, disk |
| 4 | **Lưu secret an toàn** — key trong `~/.hermes/.env`, quyền file đúng (`chmod 600`) |
| 5 | **Bật pairing** — dùng mã ghép đôi thay vì hardcode user ID |
| 6 | **Soát allowlist lệnh** — audit `command_allowlist` định kỳ |
| 7 | **Đặt `terminal.cwd`** — không để agent chạy từ thư mục nhạy cảm |
| 8 | **Không chạy bằng root** |
| 9 | **Giám sát log** — `~/.hermes/logs/` để phát hiện truy cập bất thường |
| 10 | **Cập nhật thường xuyên** — `hermes update` cho bản vá bảo mật |

## 2. Tư thế bảo mật {#bao-mat}

### Mặc định đã an toàn (không cần cấu hình)

- **Phê duyệt lệnh:** `approvals.mode: smart` — LLM phụ đánh giá rủi ro; lệnh nguy hiểm bị từ chối, ca mơ hồ hỏi người.
- **Fail-closed:** không trả lời prompt phê duyệt trong timeout (mặc định 300s) → lệnh bị **từ chối**.
- **Blocklist cứng:** `rm -rf /`, fork bomb… bị chặn *bất kể* mode, `--yolo`, hay "allow always".
- **Chặn ghi file nhạy cảm:** `write_file`/`patch` không đụng được `~/.ssh/`, `~/.aws/`, `~/.kube/`, `/etc/sudoers`, `.env`…
- **Redaction secret bật mặc định:** chuỗi giống key/token bị che trước khi vào context & log.
- **Không telemetry:** dữ liệu chỉ đi tới provider bạn cấu hình; lưu cục bộ ở `~/.hermes/`.

### Siết chặt cho máy công việc

```
approvals:
  mode: manual                  # tự xem mọi lệnh bị gắn cờ
  timeout: 300                  # không trả lời = từ chối (fail-closed)
  deny:                         # danh sách "không bao giờ chạy" — sống cả khi /yolo
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"

security:
  redact_secrets: true

checkpoints:
  enabled: true                 # snapshot trước thao tác phá hoại

terminal:
  backend: docker               # hoặc ssh — giữ thực thi ngoài host
  docker_forward_env: []        # allowlist rỗng = không rò secret vào container
```

```
# ~/.hermes/.env — sandbox ghi file
HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

> **Hiểu đúng phạm vi.** Deny rules và file-write guard là **guardrail** chống một agent trung thực nhưng sai — *không* phải sandbox chống tiến trình đối kháng. Muốn cách ly thật, dùng backend terminal cô lập (Docker/Modal). Đó mới là ranh giới được thiết kế cho việc đó.

> **Đánh đổi quan trọng.** Khi chạy backend container (Docker/Singularity/Modal/Daytona), **kiểm tra lệnh nguy hiểm
bị bỏ qua** vì container là ranh giới. Nghĩa là ảnh container của bạn phải được khoá chặt — đừng dùng ảnh chứa secret hay mount thư mục nhạy cảm.

## 3. Kinh tế prompt cache {#chi-phi}

Hầu hết provider cache tiền tố hội thoại (system prompt + history). Giữ system prompt ổn định
(cùng context file, cùng memory) → các lượt sau **cache hit**, rẻ hơn hẳn.

- Cache gắn với *model* và *tài khoản*. Đổi model, fallback tự động, hay xoay
credential pool đều buộc lượt sau đọc lại toàn bộ hội thoại với giá input đầy đủ.
- Đổi model thường xuyên trong phiên dài → nhân chi phí lên. Thường rẻ hơn nếu mở
*phiên mới* trên model kia.
- Dùng `/compress` trước khi chạm giới hạn; `/usage` để xem mức dùng;
`hermes prompt-size` để biết chi phí cố định mỗi tin (chạy offline).
- `execute_code` gộp nhiều bước thành một lượt có chi phí context bằng không;
`delegate_task` giữ dữ liệu trung gian ngoài context chính.

## 4. Context file: AGENTS.md & SOUL.md {#context}

| File | Dùng cho | Ví dụ |
| --- | --- | --- |
| `AGENTS.md` | Chỉ dẫn theo dự án (đọc mỗi phiên ở cwd) | Stack, quy ước code, nơi đặt test, điều cấm |
| `SOUL.md` | Tính cách bền vững toàn cục | "Bạn là kỹ sư backend cấp cao, trả lời ngắn gọn" |
| `.cursorrules` / `.cursor/rules/*.mdc` | Tận dụng quy ước sẵn có | Không cần nhân đôi |
| `CLAUDE.md` | Tương thích | Đọc tự động |

> **Mẹo.** Giữ context file *ngắn và tập trung* — mỗi ký tự bị tính vào token budget của *mọi* tin. Nếu không hiểu vì sao một context file bị bỏ qua, chạy `/context` để xem trạng thái nạp từng file.

## 5. Memory vs Skills {#memory}

|  | Memory | Skills |
| --- | --- | --- |
| Chứa gì | Sự thật: môi trường, sở thích, vị trí dự án | Thủ tục: quy trình nhiều bước, công thức tái dùng |
| Trả lời | "Cái gì" | "Làm thế nào" |
| Giới hạn | ~2.200 ký tự (MEMORY.md), ~1.375 (USER.md) | Không giới hạn cứng; nạp theo nhu cầu |

- Task **5+ bước** và sẽ lặp lại → nhờ agent "lưu thành skill". Lần sau chỉ gõ `/<tên-skill>`.
- Memory quá 80% dung lượng → hợp nhất entry trước khi thêm. Nói "clean up your memory".
- Memory là **snapshot đóng băng**: thay đổi trong phiên không hiện ở system prompt
tới phiên sau (agent ghi đĩa ngay, nhưng cache prompt không bị vô hiệu giữa phiên).

## 6. An toàn khi agent sửa file {#an-toan-file}

- Bật `checkpoints.enabled: true` — snapshot trước `write_file`,
`patch`, và lệnh phá hoại.
- `/rollback diff <N>` *trước* khi khôi phục; `/rollback <N>` để quay lại.
- Dùng `hermes -w` (worktree mode) cho mỗi thí nghiệm: worktree + branch riêng biệt tự động.
- Chạy nhiều agent song song → **mỗi agent một worktree**, tránh giẫm chân.

## 7. Tham chiếu nhanh {#quickref}

### Cú pháp lịch cron

| Biểu thức | Nghĩa |
| --- | --- |
| `every 30m` | Mỗi 30 phút |
| `0 2 * * *` | Hằng ngày 2:00 |
| `0 9 * * 1-5` | Ngày thường, 9:00 |
| `0 */6 * * *` | Mỗi 6 giờ |

### Đích gửi kết quả (`--deliver`)

| Đích | Cờ |
| --- | --- |
| Chính chat tạo job | `origin` (mặc định) |
| File cục bộ | `local` → `~/.hermes/cron/output/` |
| Telegram / Discord / Slack | `telegram` · `discord` · `slack` (home channel hoặc `telegram:CHAT_ID`) |
| SMS | `sms:+15551234567` |
| Bình luận GitHub | `github_comment` |

### Mẫu `[SILENT]`

Nếu phản hồi của cron chứa `[SILENT]`, việc gửi bị chặn. Dùng để tránh spam khi
lần chạy không có gì đáng báo:

```
If nothing noteworthy happened, respond with [SILENT].
```

## 8. Anti-patterns {#anti}

| Anti-pattern | Vì sao sai | Thay bằng |
| --- | --- | --- |
| `GATEWAY_ALLOW_ALL_USERS=true` | Bot có quyền shell mở cho mọi người | Allowlist theo nền tảng hoặc pairing |
| Chọn "always" ở prompt phê duyệt | Vĩnh viễn allowlist mẫu lệnh đó | Chọn "session" tới khi thật quen |
| Đổi model liên tục trong phiên dài | Phá prompt cache, nhân chi phí input | Ổn định model; phiên mới nếu cần model khác |
| Nhồi mọi thứ vào context file | Tốn token mỗi tin | Ngắn gọn; đưa thủ tục vào skill |
| Cron prompt mơ hồ ("làm briefing như thường") | Phiên mới không có ngữ cảnh | Prompt tự chứa đầy đủ |
| Chạy agent trên máy có secret mà không cách ly | Rủi ro rò rỉ/phá hoại | Backend docker/ssh + deny rules |

> **Nguồn.** `guides/tips.md`, `guides/secure-hermes-on-a-work-machine.md`, `user-guide/security.md` (Gateway Deployment Checklist), `guides/automation-blueprints.md` (quick reference), qua `llms-full.txt`.
