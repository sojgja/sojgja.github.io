---
id: 25-tich-hop-harness
title: "Tích hợp harness vào Hermes"
sidebar_label: "Tích hợp harness vào Hermes"
sidebar_position: 25
---

# Tích hợp harness vào Hermes

Bạn triển khai cả `soigia-harness` lẫn Hermes. Trang này là bản đồ để đưa
harness vào Hermes đúng cách: sáu mô hình tích hợp, tiêu chí chọn, và những best practice
giữ hai bên không giẫm chân nhau.

## 1. Nguyên tắc nền {#nguyen-tac}

- **Không bao giờ sửa core Hermes.** Mọi tích hợp đi qua điểm cắm công khai
(plugin, MCP, skill, API).
- **Tách tiến trình, giao tiếp bằng JSON.** Harness chạy như tiến trình riêng,
trao đổi qua stdout JSON — tránh xung đột phiên bản Python và vòng đời `AIAgent`.
- **Một bên sở hữu model config.** Đừng để cả hai cùng cấu hình provider cho
một tác vụ — dễ sinh hai nguồn sự thật.
- **Đo chi phí context.** Mỗi tool/MCP thêm schema; kiểm bằng `/context all`.

## 2. Sáu mô hình tích hợp {#ban-do}

| # | Mô hình | Cơ chế Hermes | Độ tách rời |
| --- | --- | --- | --- |
| A | **Chia sẻ skill** | `~/.hermes/skills/` hoặc `hermes skills tap add` — chuẩn agentskills.io dùng chung | Cao (chỉ markdown) |
| B | **Wrapper tool qua plugin** | Plugin `register(ctx)` đăng ký tool gọi CLI harness | Vừa |
| C | **Harness như MCP server** | `mcp_servers` trong `config.yaml` (stdio) | Cao (chuẩn hoá) |
| D | **Harness như sub-agent** | Plugin tool spawn tiến trình harness một-phát; hoặc dùng `delegate_task` nội bộ Hermes cho việc khác | Vừa |
| E | **Cầu model** | `hermes proxy` (OpenAI-compatible) ↔ harness `OpenAICompatibleModel` | Thấp (chia credential) |
| F | **Port tool của harness** | Đăng ký tool Hermes (plugin) tái dùng logic như `read_many`, `search_context`, `run_tests` | Thấp→vừa |

## 3. Chọn mô hình nào {#chon}

| Bạn muốn… | Chọn | Vì sao |
| --- | --- | --- |
| Dùng quy trình điều tra dựa trên bằng chứng trong Hermes | **A** | Skill là markdown, dùng chung chuẩn — gần như miễn phí |
| Expose vài lệnh harness cho agent gọi | **B** | Plugin là bề mặt chính thức cho tool tuỳ biến |
| Cho agent dùng cả bộ tool của harness như tool bản địa | **C** | MCP là lớp adapter đúng cho "tool ngoài" |
| Ủy thác một việc trọn gói cho harness chạy độc lập | **D** | Harness tự lo vòng lặp; Hermes chỉ nhận kết quả |
| Cho harness dùng model/credential của Hermes | **E** | Một nguồn credential, không nhân đôi key |
| Tái dùng đúng logic tool của harness | **F** | Tránh viết lại; nhưng phải port sang schema Hermes |

> **Lộ trình gợi ý.** Bắt đầu **A** (rẻ, an toàn) → thêm **C** (MCP) khi cần tool thật → chỉ dùng **E** nếu muốn thống nhất credential. Đừng làm hết cùng lúc.

## 4. Best practices {#best}

| Chủ đề | Khuyến nghị |
| --- | --- |
| Giao tiếp | Harness in JSON ra stdout; tool Hermes **phải** trả về chuỗi JSON (`json.dumps`) |
| Lỗi | Tool Hermes trả `{"error": "..."}`, **không** raise exception |
| Đặt tên | Namespace rõ, ví dụ `harness__run_tests`, tránh trùng built-in |
| Tiến trình | Gọi harness bằng subprocess với timeout; không import in-process nếu không cần |
| Kiểm thử | `hermes plugins doctor . --ci` trước khi cài; test cả hai phía |
| Context | Đo bằng `/context all`; chỉ bật tool/MCP cần thiết |
| Model config | Một bên sở hữu; bên kia chỉ trỏ tới (ví dụ harness trỏ `base_url` về `hermes proxy`) |
| Bí mật | Không nhúng key vào `mcp.json`/skill; dùng `.env` + redaction |
| Phiên bản | Pin phiên bản MCP server/plugin; audit chuỗi cung ứng (`hermes security audit`) |
| Hợp đồng | Nếu harness và Hermes cùng tạo "bằng chứng", dùng **một** chuẩn bằng chứng chung |

## 5. Pitfalls {#pitfall}

| Pitfall | Hậu quả | Tránh thế nào |
| --- | --- | --- |
| Import harness in-process vào plugin | Xung đột dependency/vòng đời agent | Gọi subprocess, trao đổi JSON |
| Hai bên cùng cấu hình provider cho một việc | Hai nguồn sự thật, khó debug | Chọn một bên sở hữu (thường Hermes) |
| Bọc quá nhiều tool vào MCP | Phình schema, tốn token, agent lú | Chỉ lộ tool cần; lọc theo server |
| Sửa core Hermes để tích hợp | Vỡ khi `hermes update` | Dùng plugin/MCP/skill |
| Dùng import nội bộ Hermes | Vỡ sau đợt tách module (2026-09-14) | Dùng điểm cắm công khai; chạy `hermes plugins compat` |
| Không namespace tool | Đè built-in hoặc plugin khác | Tiền tố rõ; kiểm bằng `plugins doctor` |

> **Nguồn.** `developer-guide/plugins/index.md`, `adding-tools.md`, `programmatic-integration.md`, `user-guide/features/mcp.md`, `guides/python-library.md`, và API của `soigia-harness/adapters/`.
