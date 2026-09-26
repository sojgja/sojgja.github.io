---
id: 15-plugin-mo-rong
title: "Plugin & mở rộng"
sidebar_label: "Plugin & mở rộng"
sidebar_position: 15
---

# Plugin & mở rộng

Khi skill không đủ, Hermes mở rộng qua plugin và các điểm cắm có hợp đồng công khai: thêm
tool, thêm provider, thêm platform adapter, móc middleware/observer — tất cả không đụng core.

## 1. Các điểm cắm {#diem-cam}

| Điểm cắm | Mở rộng gì |
| --- | --- |
| Skill | Năng lực dạng markdown (Tài liệu 09) — ưu tiên đầu tiên |
| Tool | Hàm agent gọi được, có schema |
| Provider / model | Nguồn inference (kể cả endpoint nội bộ) |
| Platform adapter | Nền tảng nhắn tin mới cho gateway |
| MCP server | Tool từ bên ngoài (Tài liệu 13) |
| Plugin | Gói đóng nhiều mở rộng trên |
| Webhook | Kích hoạt agent theo sự kiện ngoài |

## 2. Plugin & Plugin SDK {#plugin}

Plugin đóng gói và cài đặt như một đơn vị. Có nhiều loại plugin theo từng subsystem:

| Loại plugin | Ví dụ |
| --- | --- |
| Model/provider | `model-provider-plugin.md` |
| Memory | `memory-provider-plugin.md` |
| Web search | `web-search-provider-plugin.md` |
| Image / Video gen | `image-gen-provider-plugin.md`, `video-gen-provider-plugin.md` |
| Terminal environment | `terminal-environment-plugin.md` |
| Context engine | `context-engine-plugin.md` |
| Secret source | `secret-source-plugin.md` |
| Browser provider | `browser-provider-plugin.md` |
| Desktop UI | `desktop-plugin-sdk.md` |
| LLM access cho plugin | `plugin-llm-access.md` |

## 3. Thêm một tool {#tool}

Tool native cần: mô tả schema (tên, tham số, mục đích) + logic thực thi *chính xác mỗi
lần*. Đọc `developer-guide/adding-tools.md` và `tools-runtime.md`.

Quy tắc chọn: nếu diễn đạt được bằng hướng dẫn + tool sẵn có → làm skill; nếu cần logic
chính xác, auth, dữ liệu nhị phân → làm tool.

## 4. Thêm một provider {#provider}

Provider adapter ánh xạ một API về một trong ba API mode của Hermes
(`chat_completions`, `codex_responses`, `anthropic`). Một
provider thường cần: base URL, cách xác thực, danh mục model, và metadata (context length).
Đọc `developer-guide/adding-providers.md` và `provider-runtime.md`.

## 5. Thêm platform adapter {#platform}

Adapter nền tảng cắm vào gateway để nhận/gửi tin và map slash command. Đọc
`developer-guide/adding-platform-adapters.md`. Từ bản mới, nền tảng như
**Microsoft Teams** được thêm như plugin thay vì native.

## 6. Middleware & observer {#hook}

| Cơ chế | Dùng để |
| --- | --- |
| Middleware | Chèn xử lý vào luồng agent (`middleware.md`) |
| Observer hooks | Quan sát sự kiện, ghi log/telemetry (`observer-hooks.md`) |
| Shell hooks | Script khai báo trong `config.yaml`; quản bằng `hermes hooks` |
| Webhook | `hermes webhook` — kích hoạt theo sự kiện ngoài |

## 7. Catalog & phân phối {#catalog}

| Việc | Cách |
| --- | --- |
| Duyệt catalog plugin/skill | Desktop: tool `manage_catalog` (thẻ phê duyệt) |
| Publish skill | `hermes skills` (publish) |
| Ghim phiên bản MCP server | Cấu hình + `hermes security audit` |
| Audit chuỗi cung ứng | `hermes security audit` (OSV.dev) |

> **Thư mục liên quan trong repo.** `plugins/` (đóng gói sẵn), `plugin-catalog/`, `optional-mcps/`, `providers/`, `skills/`, `optional-skills/`.

> **Nguồn.** `website/docs/developer-guide/*-plugin*.md`, `adding-tools.md`, `adding-providers.md`, `adding-platform-adapters.md`, `middleware.md`.
