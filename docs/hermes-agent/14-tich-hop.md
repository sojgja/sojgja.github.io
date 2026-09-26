---
id: 14-tich-hop
title: "Tích hợp: Python, API, subagent"
sidebar_label: "Tích hợp: Python, API, subagent"
sidebar_position: 14
---

# Tích hợp: Python, API, subagent

Hermes không chỉ là CLI. Bạn nhúng `AIAgent` vào script Python, gọi qua API server
OpenAI-compatible, nhúng vào IDE qua ACP, hoặc ủy quyền cho subagent chạy song song.

## 1. Dùng như Python library {#python}

Cách đơn giản nhất là `chat()` — đưa tin, nhận chuỗi về:

```
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)
response = agent.chat("Thủ đô nước Pháp là gì?")
print(response)
```

`chat()` tự lo toàn bộ vòng hội thoại (tool call, retry) và chỉ trả về văn bản cuối.
Cần kiểm soát nhiều hơn thì dùng `run_conversation()`:

```
result = agent.run_conversation(
    user_message="Tìm tính năng mới của Python 3.13",
    task_id="my-task-1",
)
print(result["final_response"])
print(len(result["messages"]))
```

> **Bắt buộc.** Luôn đặt `quiet_mode=True` khi nhúng. Không có nó, agent in spinner/tiến trình ra terminal và làm bẩn output ứng dụng của bạn.

Cần chuẩn bị source checkout qua PM (`git clone` + `source ./activate`).
Hermes **không** phát hành wheel cho `pip install hermes-agent`. Các
biến môi trường dùng cho CLI cũng cần thiết khi dùng như library.

## 2. API server (OpenAI-compatible) {#api}

Gateway có API server OpenAI-compatible — tắt mặc định. Muốn mở, đặt
`API_SERVER_HOST` và `API_SERVER_KEY` (key **bắt buộc** để
xác thực). Xem `user-guide/api-server.md` trước khi làm trên host internet.

```
# trong docker-compose.yml
# - API_SERVER_HOST=0.0.0.0
# - API_SERVER_KEY=${API_SERVER_KEY}
```

## 3. ACP & batch runner {#acp}

| Điểm vào | Dùng cho |
| --- | --- |
| ACP adapter (`acp_adapter/`) | Nhúng vào IDE (VS Code, Zed, JetBrains) |
| Batch runner (`batch_runner.py`) | Sinh batch trajectory cho nghiên cứu/train |

## 4. Delegation & subagent {#delegation}

Hermes spawn agent con cô lập: mỗi subagent có hội thoại, terminal và toolset riêng.
**Chỉ bản tóm tắt cuối** quay về context của bạn — tool call trung gian không bao
giờ lọt vào context chính.

**Nên delegate**

- Việc nặng suy luận (debug, review code, tổng hợp research)
- Việc làm ngập context bằng dữ liệu trung gian
- Nhiều nhánh độc lập chạy song song
- Việc cần context tươi, không thiên lệch

**Dùng thứ khác**

- Một tool call đơn → gọi trực tiếp
- Việc cơ học nhiều bước có logic → `execute_code`
- Việc cần tương tác user → subagent không dùng được `clarify`
- Việc phải sống qua khởi động lại → cron hoặc terminal background

Tool: `delegate_task(goal, context)`; batch `delegate_task(tasks=[...])`;
nền `background=true`. Trong chat: `/review` spawn subagent review độc lập.

## 5. `execute_code` & RPC {#rpc}

`execute_code` cho agent viết script Python gọi trực tiếp các tool Hermes qua RPC.
Nhờ vậy, một pipeline nhiều bước gộp thành lượt có *chi phí context bằng không* — dữ
liệu trung gian không đi qua context.

## 6. Kanban đa agent {#kanban}

Toolset `kanban` phối hợp nhiều agent qua bảng task (tạo, liên kết, block, review,
heartbeat). Đây là các tool cho worker do dispatcher spawn (`HERMES_KANBAN_TASK`).
Lưu ý: con của `delegate_task` không phải chủ sở hữu task Kanban — schema của chúng
gỡ toolset này.

```
hermes kanban                 # bảng cộng tác đa profile
hermes project                # workspace nhiều thư mục (anchor cho desktop & kanban)
```

> **Nguồn.** `website/docs/guides/python-library.md`, `guides/delegation-patterns.md`, `developer-guide/programmatic-integration.md`, `user-guide/api-server.md`.
