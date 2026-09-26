---
id: 04-bat-dau-nhanh
title: "Bắt đầu nhanh"
sidebar_label: "Bắt đầu nhanh"
sidebar_position: 4
---

# Bắt đầu nhanh

Từ số 0 tới một cuộc trò chuyện chạy được, và biết chính xác phải làm gì khi có trục trặc.
Nguyên tắc vàng: nếu Hermes chưa trò chuyện được bình thường, **đừng** thêm
tính năng nào khác.

## 1. Đường nhanh nhất {#nhanh}

| Mục tiêu | Làm trước | Làm sau |
| --- | --- | --- |
| Chỉ muốn Hermes chạy trên máy | `hermes setup` | Chạy một chat thật và xác minh nó trả lời |
| Đã biết provider | `hermes model` | Lưu cấu hình, rồi chat |
| Muốn bot / luôn trực | CLI chạy được trước đã | `hermes gateway setup` (Telegram, Discord, Slack…) |
| Model cục bộ / tự host | `hermes model` → custom endpoint | Xác minh endpoint, tên model, context length |
| Fallback đa provider | `hermes model` trước | Thêm routing & fallback sau khi chat cơ bản chạy |

## 2. Chạy setup wizard {#setup}

```
hermes setup
```

Trên bản cài mới, wizard đưa ba chế độ:

| Chế độ | Nội dung |
| --- | --- |
| **Quick Setup (Nous Portal)** | Đăng nhập OAuth, không quản API key; bật model + Tool Gateway (web search, image gen, TTS, cloud browser). Đường nhanh khuyến nghị. |
| **Full Setup** | Tự đi qua từng provider, tool, tuỳ chọn (mang key của bạn). |
| **Blank Slate** | Mọi thứ tắt trừ tối thiểu: provider & model, File Operations, Terminal. Không web, browser, code execution, vision, memory, delegation, cron, skills, plugin, MCP. Bật lại sau bằng `hermes tools`, `hermes skills opt-in --sync`, `hermes setup agent`. |

Nếu dùng Nous Portal, một lệnh là xong:

```
hermes setup --portal     # đăng nhập + đặt Nous làm provider + bật Tool Gateway
```

## 3. Chọn provider {#provider}

```
hermes model
```

Bảng chọn provider phổ biến (đầy đủ ở Tài liệu 07):

| Provider | Thiết lập |
| --- | --- |
| Nous Portal | OAuth qua `hermes model` (gói thuê bao) |
| OpenAI (ChatGPT/Codex) | `hermes model` → device code |
| Anthropic (Claude) | `hermes model` → OAuth Max, hoặc API key |
| OpenRouter | `OPENROUTER_API_KEY` trong `~/.hermes/.env` |
| DeepSeek | `DEEPSEEK_API_KEY` trong `~/.hermes/.env` |
| Google / Gemini | `GOOGLE_API_KEY` (hoặc `GEMINI_API_KEY`) |
| Cục bộ (Ollama/LM Studio) | `hermes model` → chọn endpoint cục bộ |

> **Key ở đâu.** Secret để trong `~/.hermes/.env`; cấu hình không bí mật để trong `~/.hermes/config.yaml`. Chỉ các biến bí mật được ghi tài liệu trong `.env` mới đè lên cấu hình YAML tương ứng.

## 4. Xác minh bằng một cuộc trò chuyện {#chat}

Cách tương tác:

```
hermes                      # mở CLI/TUI tương tác
hermes -z "tóm tắt README.md"   # one-shot, in ra rồi thoát
```

Cách kiểm tra nhanh sau khi đổi cấu hình:

```
hermes status               # agent, auth, nền tảng
hermes doctor               # chẩn đoán config & phụ thuộc
hermes usage                # hạn mức của tài khoản (--json cho script)
```

Hãy yêu cầu một việc thật (ví dụ "đọc package.json và liệt kê script") để chắc rằng cả
*model* lẫn *tool* đều hoạt động, không chỉ model trả lời.

## 5. Mở rộng dần {#tiep}

| Muốn | Đọc / chạy |
| --- | --- |
| Nhắn tin từ điện thoại | Tài liệu 11 · `hermes gateway setup` |
| Task định kỳ | Tài liệu 12 · `hermes cron` |
| Thêm tool ngoài | Tài liệu 13 · MCP |
| Nhúng vào app | Tài liệu 14 · Python library |
| Chạy 24/7 trên server | Tài liệu 16 · Triển khai |

## 6. Khi có sự cố {#loi}

| Triệu chứng | Xử lý nhanh |
| --- | --- |
| Chat không trả lời | `hermes doctor`; kiểm tra model & key trong `hermes status` |
| Sai config sau khi sửa tay | `hermes config` / `hermes migrate` |
| Nghi ngờ provider lỗi | `hermes fallback` thêm provider dự phòng |
| Cần gửi log hỗ trợ | `hermes dump` (bản tóm tắt để dán) hoặc `hermes debug` |
| Xem log | `hermes logs` |

Chi tiết đầy đủ ở Tài liệu 17 (Bảo mật, vận hành & FAQ).

> **Nguồn.** `website/docs/getting-started/quickstart.md`, `updating.md`, `reference/cli-commands.md`.
