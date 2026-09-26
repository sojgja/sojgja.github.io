---
id: 02-kien-truc
title: "Kiến trúc hệ thống"
sidebar_label: "Kiến trúc hệ thống"
sidebar_position: 2
---

# Kiến trúc hệ thống

Bản đồ tổng thể phần bên trong Hermes: các điểm vào, bộ điều phối `AIAgent`,
kho phiên SQLite, và các backend tool. Đọc trang này để biết "mã nằm ở đâu" trước khi đào sâu.

## 1. Sơ đồ hệ thống {#so-do}

```
┌─────────────────────────────────────────────────────────────┐
│                      Điểm vào (Entry Points)                 │
│  CLI (cli.py)   Gateway (gateway/run.py)   ACP (acp_adapter) │
│  Batch Runner   API Server                  Python Library    │
└──────────┬──────────────┬───────────────────────┬────────────┘
           ▼              ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   AIAgent (run_agent.py)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ Prompt      │  │ Provider   │  │ Tool       │             │
│  │ Builder     │  │ Resolution │  │ Dispatch   │             │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
│  ┌─────┴──────┐  ┌─────┴──────┐  ┌─────┴──────┐             │
│  │Compression │  │ 3 API mode │  │Tool Registry│            │
│  │& Caching   │  │chat_compl. │  │70+ tools    │            │
│  │            │  │codex_resp. │  │28 toolsets  │            │
│  │            │  │anthropic   │  │             │            │
│  └────────────┘  └────────────┘  └────────────┘             │
└─────────┴─────────────────┴─────────────────┴───────────────┘
           ▼                                    ▼
┌───────────────────┐              ┌──────────────────────┐
│ Session Storage   │              │ Tool Backends         │
│ SQLite + FTS5     │              │ Terminal (7 backend)  │
│ hermes_state.py   │              │ Browser (5 backend)   │
│ gateway/session.py│              │ Web (4 backend)       │
└───────────────────┘              │ MCP (động)            │
                                   │ File, Vision, …       │
                                   └──────────────────────┘
```

## 2. Sáu điểm vào {#entry}

| Điểm vào | File | Dùng cho |
| --- | --- | --- |
| CLI / TUI | `cli.py` | Trò chuyện tương tác, slash command |
| Gateway | `gateway/run.py` | Nhắn tin đa nền tảng (Telegram, Slack…) |
| ACP | `acp_adapter/` | Nhúng vào IDE (VS Code, Zed, JetBrains) |
| Batch runner | `batch_runner.py` | Sinh trajectory cho nghiên cứu/train |
| API server | gateway API | Endpoint OpenAI-compatible |
| Python library | `run_agent.py` | Nhúng `AIAgent` vào app của bạn |

## 3. Bộ điều phối `AIAgent` {#aiagent}

Mọi điểm vào đều hội tụ về `AIAgent` trong `run_agent.py`. Ba khối
chính bên trong:

| Khối | File | Việc |
| --- | --- | --- |
| Prompt builder | `agent/prompt_builder.py` | Lắp system prompt: rules, memory, skills index, tool schema |
| Provider resolution | `runtime_provider.py` | Chọn provider/model, credentials, fallback, routing |
| Tool dispatch | `model_tools.py` | Khám phá tool, thu schema, gọi tool, xử kết quả |

Vòng lặp hội thoại nằm trong `agent/conversation_loop.py` cùng các file
`agent/turn_*.py` — `run_agent.py` chỉ là facade.

## 4. Ba API mode {#api-mode}

Hermes nói chuyện với model qua ba giao thức, chọn tự động theo provider:

| Mode | Dùng cho |
| --- | --- |
| `chat_completions` | Chuẩn OpenAI — phần lớn provider và endpoint tự host |
| `codex_responses` | OpenAI Codex / ChatGPT subscription (Responses API) |
| `anthropic` | Anthropic Messages API (Claude trực tiếp) |

Nhờ lớp này, đổi từ Claude sang DeepSeek hay một endpoint vLLM không cần sửa code — chỉ đổi
cấu hình provider.

## 5. Session storage {#state}

Toàn bộ phiên và trạng thái nằm trong SQLite với FTS5 để tìm kiếm toàn văn:
`hermes_state.py` (facade) + các file `hermes_state_*.py`. Cơ sở dữ
liệu mặc định dùng WAL; tự hạ xuống DELETE khi filesystem không hợp WAL (virtiofs/NFS/SMB).

## 6. Tool backend {#backend}

| Backend | Số lượng | Ví dụ |
| --- | --- | --- |
| Terminal | 7 | local, Docker, SSH, Singularity, Modal, Daytona, Vercel |
| Browser | 5 | Browserbase, Camofox, CDP, browser-use… |
| Web | 4 | Tìm kiếm & extract với nhiều nhà cung cấp |
| MCP | động | Server ngoài cắm thêm tool |
| Khác | — | File, vision, image/video gen, TTS, memory… |

## 7. Cấu trúc thư mục {#thu-muc}

```
hermes-agent/
├── run_agent.py          # AIAgent facade
├── cli.py                # CLI facade (mixin trong hermes_cli/)
├── model_tools.py        # khám phá & dispatch tool
├── toolsets.py           # nhóm tool + preset nền tảng
├── hermes_state.py       # SQLite session/state (+ hermes_state_*.py)
├── hermes_constants.py   # HERMES_HOME, đường dẫn theo profile
├── batch_runner.py       # sinh batch trajectory
├── agent/                # nội bộ agent
│   ├── prompt_builder.py
│   ├── context_engine.py / context_compressor.py
│   ├── prompt_caching.py
│   ├── auxiliary_client.py   # LLM phụ (vision, tóm tắt)
│   └── model_metadata.py
├── gateway/              # cầu nhắn tin
├── hermes_cli/           # lệnh CLI
├── skills/               # skill đóng gói sẵn
├── optional-skills/      # skill tuỳ chọn
├── plugins/              # plugin đóng gói sẵn
├── providers/            # adapter provider
├── tools/                # tool native
└── website/docs/         # tài liệu gốc
```

## 8. Luồng một lượt hội thoại {#luong}

```
1. Người dùng nhập (CLI/gateway/library)
2. AIAgent: dựng/làm mới system prompt + tool schema
3. Provider resolution: chọn model, credentials, fallback
4. Gọi model theo 1 trong 3 API mode
5. Nếu model trả tool call → dispatch qua registry → nối kết quả
6. Lặp bước 4–5 tới khi có câu trả lời cuối (hoặc chạm cap)
7. Lưu phiên vào SQLite; có thể nén context nếu vượt ngưỡng
8. (tuỳ chọn) hậu xử lý: trích memory, đề xuất skill
```

> **Nguồn.** `website/docs/developer-guide/architecture.md`, `agent-loop.md`, `tools-runtime.md`, `session-storage.md`.
