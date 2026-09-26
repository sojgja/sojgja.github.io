---
id: 08-tools-toolsets
title: "Tools & toolsets"
sidebar_label: "Tools & toolsets"
sidebar_position: 8
---

# Tools & toolsets

Hermes đóng gói **70+ tool** thành **28 toolset**. Toolset là đơn
vị bạn bật/tắt, gộp theo mục đích. Trang này liệt kê các nhóm quan trọng, preset nền tảng, và
mô hình phê duyệt/permission.

## 1. Tool, toolset, preset {#khai-niem}

- **Tool** — một hàm agent gọi được (ví dụ `read_file`, `terminal`, `web_search`).
- **Toolset** — nhóm tool cùng chủ đề (ví dụ `file` gồm `patch`, `read_file`, `search_files`, `write_file`).
- **Preset nền tảng** — tập hợp toolset mặc định cho một nền tảng (CLI, desktop, messaging, cron…).

## 2. Các nhóm toolset chính {#nhom}

| Toolset | Tool | Việc |
| --- | --- | --- |
| `file` | `read_file`, `write_file`, `patch`, `search_files` | Đọc/ghi/sửa/tìm file |
| `terminal` | `terminal`, `process_manage` | Chạy shell; quản tiến trình nền |
| `search` / web | `web_search`, `web_extract` | Tìm & trích nội dung web |
| `browser` | `browser_navigate`, `browser_click`, `browser_type`, `browser_snapshot`, `browser_exec`, `browser_vault_*`, `browser_vision` | Tự động hoá trình duyệt (5 backend) |
| `delegation` | `delegate_task` | Spawn subagent cô lập chạy song song |
| `code_execution` | `execute_code` | Chạy script Python gọi tool qua RPC |
| `memory` | `memory` | Bộ nhớ bền qua phiên |
| `session_search` | `session_search` | Tìm lại hội thoại cũ |
| `skills` | `skill_manage`, `skill_view`, `skills_list` | Tạo/xem/duyệt skill |
| `cronjob` | `cronjob_manage` | Đặt & quản task định kỳ |
| `vision` | `vision_analyze` | Phân tích ảnh |
| `tts` | `text_to_speech` | Sinh audio |
| `image_gen` / `video_gen` | `image_generate`, `video_generate` | Sinh ảnh/video |
| `todo` | `todo_list` | Danh sách việc trong phiên |
| `clarify` | `clarify` | Hỏi lại user khi cần làm rõ |
| `kanban` | `kanban_*` | Phối hợp đa agent qua bảng task |
| `desktop_ui` | `desktop_preview`, `gui_tour`… | Điều khiển chính app desktop (chỉ desktop) |

`browser_cdp` và `browser_dialog` chỉ đăng ký khi có CDP endpoint lúc
mở phiên. `web_search` cố tình *không* thuộc toolset `browser` —
tắt browser không làm mất tìm kiếm web.

## 3. Preset tổng hợp {#preset}

| Preset | Gồm | Dùng khi |
| --- | --- | --- |
| `coding` | file + terminal + search + web + skills + browser + todo + memory + session_search + clarify + code_execution + delegation + vision | Làm việc code |
| `debugging` | file + terminal + web | Gỡ lỗi, xem tiến trình |
| `safe` | `web_search`, `web_extract`, `image_generate`, `vision_analyze` | Research chỉ-đọc, không ghi file/không terminal |

## 4. Quản lý toolset {#quan-ly}

```
hermes tools                          # xem/bật/tắt toolset theo nền tảng
hermes tools enable kanban --platform telegram
hermes -z "..." --enabled-toolsets file,terminal
hermes-agent --list-tools             # in tool rồi thoát
```

Cấu hình có các khoá `platform_toolsets.<platform>` và
`agent.disabled_toolsets`. Chế độ *Blank Slate* ghi danh sách tường minh để
không gì tự nạp lại kể cả sau `hermes update`.

## 5. Phê duyệt & an toàn {#phe-duyet}

- **Phê duyệt lệnh nguy hiểm:** mặc định Hermes hỏi trước khi chạy lệnh có rủi ro. `--yolo` bỏ qua (không khuyến nghị).
- **Allowlist:** `hermes approvals` khai thác lịch sử phê duyệt để đề xuất allowlist.
- **Hooks:** `hermes hooks` xem/phê duyệt/xoá shell hook khai báo trong `config.yaml`.
- **Egress proxy:** `hermes egress` — firewall chèn credential cho sandbox terminal từ xa (iron-proxy), mặc định tắt.

## 6. Sandbox & backend terminal {#backend}

Tool `terminal` chạy trên một trong 7 backend: local, Docker, SSH, Singularity,
Modal, Daytona, Vercel Sandbox. Đây là ranh giới cách ly chính:

```
# ví dụ: chạy agent trong container, giữ dữ liệu trên host
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d
```

> **Nguyên tắc.** Bật toolset tối thiểu cần thiết. Với tác vụ trên máy nhạy cảm, dùng backend cách ly (Docker/SSH/Modal) và preset `safe` khi chỉ cần đọc.

> **Nguồn.** `website/docs/reference/toolsets-reference.md`, `tools-reference.md`, `user-guide/features/*`.
