---
id: 01-tong-quan
title: "Tổng quan — Hermes là gì"
sidebar_label: "Tổng quan — Hermes là gì"
sidebar_position: 1
---

# Tổng quan — Hermes là gì

Hermes Agent là agent AI tự cải thiện do Nous Research xây dựng: một agent
terminal-native có vòng lặp tool calling, bộ nhớ bền qua các phiên, skill tự tạo và tự
cải thiện, cùng một gateway sống trên hơn 21 nền tảng nhắn tin. Trang này chốt lại
"nó là gì, khác gì, và khi nào nên dùng".

## 1. TL;DR {#tldr}

Một tiến trình agent chạy ở đâu cũng được — laptop, VPS $5, cụm GPU, hay serverless
hibernate khi rảnh. Bạn nói chuyện với nó từ CLI, TUI, desktop, hay Telegram/Discord/Slack;
nó dùng bất kỳ model nào (Nous Portal, OpenRouter, OpenAI, Anthropic, Google, hoặc endpoint
OpenAI-compatible của riêng bạn) và đổi model bằng một lệnh `hermes model`.

Điểm khiến Hermes khác phần còn lại là **vòng học khép kín**: sau một task
phức tạp nó tự đúc kết thành skill, skill tự cải thiện trong lúc dùng, bộ nhớ được chưng
cất có nhịp, và nó tìm lại chính các cuộc trò chuyện cũ của mình.

## 2. Tám năng lực cốt lõi {#tinh-nang}

| Năng lực | Nghĩa là gì |
| --- | --- |
| **Giao diện terminal thật** | TUI đầy đủ: sửa nhiều dòng, autocomplete slash-command, lịch sử, ngắt-và-đổi-hướng, streaming tool output. |
| **Sống cùng bạn** | Telegram, Discord, Slack, WhatsApp, Signal, SMS, Matrix… từ một tiến trình gateway; phiên liền mạch qua nền tảng. |
| **Vòng học khép kín** | Bộ nhớ do agent tự quản; tự tạo skill sau task phức tạp; skill tự cải thiện; tìm phiên cũ bằng FTS5 + tóm tắt LLM. |
| **Cron có sẵn** | Lịch chạy bằng ngôn ngữ tự nhiên, giao kết quả về bất kỳ nền tảng nào: báo cáo ngày, backup đêm, audit tuần. |
| **Ủy quyền & song song** | Spawn subagent cô lập cho các nhánh việc song song; viết script Python gọi tool qua RPC để gộp pipeline. |
| **Chạy ở đâu cũng được** | Bảy backend terminal: local, Docker, SSH, Singularity, Modal, Daytona, Vercel Sandbox. Modal/Daytona có persistence serverless. |
| **Sẵn sàng cho nghiên cứu** | Sinh batch trajectory, nén trajectory để train thế hệ model tool-calling tiếp theo. |
| **Mở rộng không đụng core** | Skill (markdown), tool, provider plugin, platform adapter, MCP server, webhook — cắm vào qua hợp đồng công khai. |

## 3. Từ vựng cần nắm {#khai-niem}

| Khái niệm | Định nghĩa ngắn |
| --- | --- |
| **Agent (`AIAgent`)** | Bộ điều phối trung tâm trong `run_agent.py`: dựng prompt, chọn provider, dispatch tool, quản context. |
| **Toolset** | Nhóm tool theo mục đích (file, terminal, browser, web, memory…). 28 toolset đóng gói 70+ tool. |
| **Skill** | Gói hướng dẫn `SKILL.md` + script/reference. Cách ưu tiên để thêm năng lực mà không sửa code. |
| **Gateway** | Tiến trình cầu nối agent với các nền tảng nhắn tin và webhook. |
| **Cron job** | Task chạy theo lịch trong phiên agent mới, prompt phải tự chứa đủ ngữ cảnh. |
| **Delegation / subagent** | Spawn agent con cô lập; chỉ tóm tắt cuối quay về context của bạn. |
| **Profile** | Một "bản thể" Hermes độc lập: config, phiên, skill, bộ nhớ riêng. |
| **MCP** | Lớp adapter để cắm tool từ server ngoài vào Hermes. |

## 4. Hermes vs `soigia-harness` {#so-sanh}

Trong repo này cả hai cùng tồn tại. Hiểu ranh giới sẽ giúp bạn chọn đúng công cụ:

| Tiêu chí | `soigia-harness` | Hermes Agent |
| --- | --- | --- |
| Mục tiêu | Lõi tham chiếu tối giản, dễ đọc | Agent hoàn chỉnh dùng thật |
| Phụ thuộc | Chỉ Python stdlib | Runtime đầy đủ (Python + Node + tool hệ thống) |
| Model | Protocol JSON nhỏ, chạy Ollama cục bộ | 60+ provider, 3 API mode (chat_completions, codex_responses, anthropic) |
| Tool | 5 primitive + registry | 70+ tool / 28 toolset, browser, vision, TTS, image/video gen |
| Đa nền tảng | Không | Gateway 21+ nền tảng nhắn tin |
| Mở rộng | Viết Python vào core | Skill, plugin, MCP, webhook — không đụng core |
| Vận hành | Script Python | Cron, gateway, backup, checkpoint, desktop app |

> **Gợi ý.** Dùng `soigia-harness` để *hiểu* agent hoạt động ra sao (đọc Tài liệu bộ Harness). Dùng Hermes để *làm việc thật* và *tích hợp*.

## 5. Khi nào dùng — khi nào không {#khi-nao}

**Nên dùng khi**

Cần một agent tự trị đa năng: tự động hoá lặp lại, trợ lý nhắn tin luôn-trực, review PR,
research, vận hành hạ tầng; hoặc cần một runtime agent để nhúng vào sản phẩm của bạn.

**Không nên khi**

Chỉ cần một lời gọi LLM đơn lẻ (dùng SDK provider trực tiếp); cần runtime tối giản
không phụ thuộc (dùng `soigia-harness`); hoặc môi trường cấm chạy shell.

## 6. Backend thực thi {#backend}

| Backend | Đặc điểm |
| --- | --- |
| Local | Chạy trên máy bạn — mặc định. |
| Docker | Cô lập container; volume `~/.hermes:/opt/data`. |
| SSH | Terminal trên máy từ xa. |
| Singularity | Cho môi trường HPC (không cần root). |
| Modal | Serverless; môi trường hibernate khi rảnh, đánh thức khi cần. |
| Daytona | Serverless; tương tự Modal, gần như không tốn phí giữa các phiên. |
| Vercel Sandbox | Sandbox cô lập tạm thời. |

## 7. Nên đọc gì tiếp {#tiep}

- **Hiểu hệ thống** → Tài liệu 02 (Kiến trúc).
- **Cài đặt** → Tài liệu 03; **chạy ngay** → Tài liệu 04.
- **Dùng mỗi ngày** → 05 (CLI/TUI) rồi 06 (slash commands).
- **Tích hợp vào hệ của bạn** → 14 (Python/API/subagent).
- **Lên production** → 16 (triển khai) và 17 (bảo mật/vận hành).

> **Nguồn.** Tổng hợp từ `README.md`, `website/docs/index.mdx`, `docs/getting-started/*` và `docs/developer-guide/architecture.md`.
