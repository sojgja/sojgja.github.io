---
id: 07-provider-model
title: "Provider & model"
sidebar_label: "Provider & model"
sidebar_position: 7
---

# Provider & model

Hermes không gắn chặt với nhà cung cấp nào. Bạn cần ít nhất một provider, đổi bằng
`hermes model`, cấu hình bằng `~/.hermes/config.yaml` (không bí mật)
và `~/.hermes/.env` (secret).

## 1. Cấu hình nằm ở đâu {#cau-hinh}

| File | Chứa |
| --- | --- |
| `~/.hermes/config.yaml` | Cấu hình không bí mật: provider, model, toolset, theme, giới hạn runtime… |
| `~/.hermes/.env` | Secret: API key, token |

```
hermes model                 # wizard chọn provider/model
hermes auth add <provider>   # thêm credential (hỗ trợ OAuth)
hermes config set <khối>.<khoá> <giá trị>
hermes config                 # xem/sửa/migrate cấu hình
```

Chỉ các *biến bí mật* được ghi tài liệu trong `.env` mới đè lên cấu hình
YAML tương ứng.

## 2. Provider thiết lập nhanh {#nhanh}

| Provider | Cách thiết lập |
| --- | --- |
| Nous Portal | `hermes model` (OAuth, gói thuê bao) — một gói gồm 300+ model + Tool Gateway |
| OpenAI Codex / ChatGPT | `hermes model` → "ChatGPT or Codex Subscription" (device code) |
| GitHub Copilot | `hermes model` (OAuth device code; hoặc `COPILOT_GITHUB_TOKEN`/`GH_TOKEN`/`gh auth token`) |
| Anthropic | `hermes model` → OAuth (Claude Max + credit), hoặc API key Anthropic |
| xAI Grok (SuperGrok) | `hermes model` → OAuth, không cần API key |
| Google Vertex AI | `hermes model` (OAuth2 service-account hoặc ADC) |
| AWS Bedrock | `hermes model` (chuỗi credential AWS qua boto3) |
| Azure AI Foundry | `hermes model` (endpoint + key) |

## 3. Provider theo API key {#khac}

Với các provider còn lại, đặt biến trong `~/.hermes/.env`. Một số provider tiêu biểu:

| Provider | Biến |
| --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` (+ `OPENROUTER_BASE_URL`) |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Google / Gemini | `GOOGLE_API_KEY` hoặc `GEMINI_API_KEY` |
| OpenAI API (trực tiếp) | `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`) |
| xAI (Responses API) | `XAI_API_KEY` |
| Fireworks AI | `FIREWORKS_API_KEY` |
| z.ai / GLM | `GLM_API_KEY` (alias `ZAI_API_KEY`, `Z_AI_API_KEY`) |
| Kimi / Moonshot | `KIMI_API_KEY` (bản Trung Quốc: `KIMI_CN_API_KEY`) |
| MiniMax | `MINIMAX_API_KEY` (`minimax-oauth` dùng OAuth) |
| Qwen / DashScope | `DASHSCOPE_API_KEY` |
| NVIDIA Build | `NVIDIA_API_KEY` |
| Hugging Face | `HF_TOKEN` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` |

Danh sách provider đầy đủ (60+) ở `reference/environment-variables.md` và
`integrations/providers.md`. Nhiều provider có alias tên, ví dụ
`fw` = `fireworks`.

## 4. Model cục bộ / tự host {#cuc-bo}

| Cách | Thiết lập |
| --- | --- |
| Ollama | `hermes model` → chọn Ollama, hoặc Ollama Cloud |
| LM Studio | `hermes model` → "LM Studio" (`LM_BASE_URL` mặc định `http://localhost:1234/v1`) |
| vLLM / SGLang (OpenAI-compatible) | `OPENAI_BASE_URL` + `OPENAI_API_KEY` (key có thể là placeholder) |

> **Kiểm tra.** Sau khi đổi sang endpoint cục bộ, hãy xác minh: endpoint sống, *tên model* đúng, và *context length* khớp — ba lỗi này gây phần lớn sự cố "kết nối được mà không chạy".

## 5. Routing, fallback & MoA {#dinh-tuyen}

| Lệnh | Việc |
| --- | --- |
| `hermes fallback` | Quản provider dự phòng, thử khi model chính lỗi |
| `hermes moa` | Cấu hình preset Mixture of Agents, chọn từ model picker; gọi bằng `/moa` |
| `hermes proxy` | Proxy OpenAI-compatible cục bộ gắn credential OAuth (dùng cho tool khác) |

Chiến lược an toàn: chỉ thêm fallback/routing *sau khi* chat cơ bản đã chạy. Đổi
provider KHÔNG cần sửa code — cả ba API mode (chat_completions, codex_responses, anthropic)
do lớp provider lo.

## 6. Quy ước biến môi trường {#bien}

- Đa số theo mẫu `<PROVIDER>_API_KEY`; một số có `_BASE_URL` để đè endpoint.
- Alias thường gặp: `ZAI_API_KEY` = `GLM_API_KEY`; `GEMINI_API_KEY` = `GOOGLE_API_KEY`.
- Biến `HERMES_*` điều khiển hành vi runtime (ví dụ `HERMES_TUI`, `HERMES_HOME`).
- Biến `COPILOT_GITHUB_TOKEN`/`GH_TOKEN`/`GITHUB_TOKEN` theo thứ tự ưu tiên cho Copilot. PAT cổ điển `ghp_*` **không** được hỗ trợ ở đây.

> **Nguồn.** `website/docs/integrations/providers.md`, `reference/environment-variables.md`, `reference/model-catalog.md`.
