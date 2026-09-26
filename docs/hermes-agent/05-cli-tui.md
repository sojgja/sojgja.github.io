---
id: 05-cli-tui
title: "CLI, TUI & profiles"
sidebar_label: "CLI, TUI & profiles"
sidebar_position: 5
---

# CLI, TUI & profiles

Mặt tiền dòng lệnh: cách gọi `hermes`, các tuỳ chọn toàn cục, hai giao diện
(classic CLI và TUI), và cách tách nhiều "bản thể" bằng profile.

## 1. Điểm vào & cú pháp {#entry}

```
hermes [global-options] <command> [subcommand/options]
```

Không có subcommand thì mở phiên tương tác. Có `-z "prompt"` thì chạy một lượt
rồi thoát.

## 2. Tuỳ chọn toàn cục {#global}

| Tuỳ chọn | Ý nghĩa |
| --- | --- |
| `--version`, `-V` | In phiên bản rồi thoát |
| `--profile <name>`, `-p` | Chọn profile cho lần chạy này (đè lên mặc định dính của `hermes profile use`) |
| `--resume <session>`, `-r` | Tiếp tục phiên theo ID/title; `latest` = phiên gần nhất |
| `--continue [name]`, `-c` | Tiếp tục phiên gần nhất, hoặc gần nhất khớp title |
| `--in <dir>` | chdir vào `<dir>` trước khi bắt đầu/tiếp tục |
| `--worktree`, `-w` | Bắt đầu trong git worktree cô lập (chạy agent song song) |
| `--yolo` | Bỏ qua prompt phê duyệt lệnh nguy hiểm |
| `--pass-session-id` | Đưa session ID vào system prompt |
| `--ignore-user-config` | Bỏ `~/.hermes/config.yaml`, dùng mặc định (vẫn nạp `.env`) |
| `--ignore-rules` | Không tự chèn `AGENTS.md`, `SOUL.md`, `.cursorrules`, memory, skill nạp sẵn |
| `--tui` / `--cli` | Ép giao diện TUI / ép REPL cổ điển (xem mục 4) |
| `--dev` | Với `--tui`: chạy trực tiếp source TypeScript qua `tsx` |

> **An toàn.** `--yolo` bỏ hết phê duyệt. Chỉ dùng trong môi trường cách ly (container, sandbox) hoặc khi bạn đã hiểu rõ rủi ro.

## 3. Các nhóm lệnh chính {#commands}

### Trò chuyện & model

| Lệnh | Việc |
| --- | --- |
| `hermes chat` | Chat tương tác hoặc one-shot |
| `hermes model` | Chọn provider & model mặc định (tương tác) |
| `hermes moa` | Cấu hình preset Mixture of Agents |
| `hermes fallback` | Quản provider dự phòng khi model chính lỗi |
| `hermes proxy` | Proxy OpenAI-compatible cục bộ gắn credential OAuth |

### Cấu hình & chẩn đoán

| Lệnh | Việc |
| --- | --- |
| `hermes setup` | Wizard cấu hình |
| `hermes config` | Xem/sửa/migrate/query cấu hình |
| `hermes doctor` | Chẩn đoán config & phụ thuộc |
| `hermes status` / `usage` | Trạng thái agent/auth/nền tảng; hạn mức tài khoản |
| `hermes dump` / `debug` | Tóm tắt setup để hỗ trợ; upload log |
| `hermes logs` | Xem/tail/lọc log agent, gateway, error |
| `hermes security audit` | Audit chuỗi cung ứng (OSV.dev) |

### Vận hành & mở rộng

| Lệnh | Việc |
| --- | --- |
| `hermes gateway` | Chạy/quản gateway nhắn tin |
| `hermes cron` | Xem & tick scheduler cron |
| `hermes pause` / `resume` | Dừng khẩn cấp toàn cục (cron, kanban, gateway) |
| `hermes skills` | Duyệt, cài, publish, audit, cấu hình skill |
| `hermes auth` | Quản credential — add/list/remove/reset/status/logout, OAuth |
| `hermes secrets` | Nguồn secret ngoài (Bitwarden) thay cho `.env` |
| `hermes webhook` | Đăng ký webhook kích hoạt theo sự kiện |
| `hermes send` | Gửi một tin tới nền tảng đã cấu hình — không cần agent loop |
| `hermes backup` / `import` | Sao lưu / phục hồi thư mục Hermes (zip) |
| `hermes checkpoints` | Xem/prune/xoá kho shadow dùng bởi `/rollback` |
| `hermes kanban` / `project` | Bảng cộng tác đa profile; workspace nhiều thư mục |

Danh sách đầy đủ: `hermes --help` và Tài liệu 17.

## 4. TUI vs CLI {#tui}

|  | Classic CLI | TUI |
| --- | --- | --- |
| Bật | `hermes --cli` hoặc mặc định | `hermes --tui` hoặc `HERMES_TUI=1` |
| Điểm mạnh | Nhẹ, prompt_toolkit REPL, ổn định | Sửa nhiều dòng, autocomplete, streaming tool output, đổi theme/skin |
| Session | Một agent chia sẻ giữa các phiên (chặn chuyển giữa lượt) | Nhiều phiên TUI sống song song (`/sessions new`) |

`--tui` luôn thắng `display.interface` trong config.

## 5. Profiles {#profile}

Một profile là một Hermes độc lập: config, phiên, skill, bộ nhớ, và dữ liệu riêng. Hữu ích
khi tách công việc cá nhân và công việc, hoặc chạy nhiều bot khác nhau trên cùng máy.

```
hermes profile            # xem & quản profile
hermes -p work chat       # chạy một lệnh dưới profile "work"
hermes -p work status
```

Trạng thái kanban và project là *theo profile*. Xem
`reference/profile-commands.md`.

## 6. One-shot & runner legacy {#oneshot}

```
hermes -z "liệt kê các file .py lớn nhất trong tools/"   # one-shot
hermes-agent --query "tóm tắt README.md"                  # runner tối giản (có trong bản cài)
hermes-agent --list-tools                                 # in danh sách tool rồi thoát
```

`hermes-agent` chỉ gửi một truy vấn rồi thoát — hợp cho script. Mọi thứ khác
dùng `hermes`.

> **Nguồn.** `website/docs/reference/cli-commands.md`, `profile-commands.md`, `user-guide/cli.md`, `user-guide/tui.md`.
