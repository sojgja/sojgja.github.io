---
id: 06-slash-commands
title: "Slash commands"
sidebar_label: "Slash commands"
sidebar_position: 6
---

# Slash commands

Gõ `/` trong CLI để mở autocomplete. Slash command điều khiển phiên, model,
workflow và tool — và chúng hoạt động ở cả CLI lẫn các nền tảng nhắn tin, sinh từ cùng một
`COMMAND_REGISTRY`.

## 1. Hai mặt & phân quyền {#hai-mat}

- **Interactive CLI slash commands** — do `cli.py` dispatch, autocomplete từ registry.
- **Messaging slash commands** — do `gateway/run.py` dispatch, menu/help sinh từ registry.

Trên nền tảng có allowlist theo user (Telegram, Discord, Slack, Matrix, Mattermost, Signal…)
có thể tách hai tầng: **admin** có mọi lệnh, **user thường** chỉ có
các lệnh bạn liệt kê trong `user_allowed_commands` (cộng sàn luôn cho phép
`/help` và `/whoami`). Cấu hình trong block `extra:` của
nền tảng trong `~/.hermes/config.yaml`.

## 2. Session {#session}

| Lệnh | Việc |
| --- | --- |
| `/new [name]` (alias `/reset`) | Phiên mới; đặt luôn title để tìm lại sau |
| `/clear` | Xoá màn hình + phiên mới |
| `/history`, `/save` | Xem lịch sử; lưu hội thoại |
| `/retry`, `/undo` | Gửi lại tin cuối; bỏ cặp user/assistant cuối |
| `/resume [name]`, `/sessions` | Tiếp tục phiên cũ; duyệt phiên |
| `/branch` (alias `/fork`) | Tách phiên thành bản độc lập để thử hướng khác |
| `/compress [here [N] \| focus]` | Nén context: tóm tắt, giữ N lượt gần nhất nguyên văn |
| `/rollback`, `/diff` | Khôi phục checkpoint filesystem; xem thay đổi git (unstaged/staged/all/session) |
| `/snapshot` (alias `/snap`) | Tạo/phục hồi/prune snapshot config & state |
| `/stop` | Kill mọi tiến trình nền đang chạy |
| `/status` | Thông tin phiên + recap tính cục bộ (không tốn token) |

## 3. Model & context {#model}

| Lệnh | Việc |
| --- | --- |
| `/model` | Đổi model/provider cho phiên |
| `/moa <prompt>` | Chạy một prompt qua preset Mixture of Agents |
| `/context [all]` (alias `/ctx`) | Phân rã context window: system prompt, tool, rules, skill index, MCP, memory, hội thoại |
| `/agents` (alias `/tasks`) | Xem agent & task đang chạy |

> **Mẹo gỡ lỗi.** `/context` liệt kê cả các *context file* (`.hermes.md`, chuỗi `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `SOUL.md`) kèm token ước lượng và trạng thái nạp — câu trả lời cho "vì sao CLAUDE.md của tôi bị bỏ qua?".

## 4. Workflow: goal, loop, queue, steer {#workflow}

| Lệnh | Việc |
| --- | --- |
| `/goal <text>` | Đặt mục tiêu sống qua nhiều lượt; model phụ phán DONE/CONTINUE. Sub: `status`, `pause`, `resume`, `clear` |
| `/subgoal <text>` | Thêm tiêu chí vào goal đang chạy |
| `/loop [interval] <prompt>` | Chạy lặp prompt theo nhịp; `--times N`, `--until <điều kiện>` |
| `/heartbeat every <interval> <prompt>` | Prompt định kỳ quay lại *phiên này* khi rảnh (min 60s) |
| `/queue <prompt>` (alias `/q`) | Xếp prompt cho lượt sau, không ngắt lượt hiện tại |
| `/steer <prompt>` | Chèn ghi chú giữa lượt — tới agent *sau tool call kế tiếp*, không ngắt |
| `/bg <prompt>` | Chạy prompt trong phiên nền riêng |
| `/btw <câu hỏi>` | Hỏi nhanh về hội thoại hiện tại, không ngắt |

Khác biệt: `/goal` bền qua `/resume`; `/loop` và
`/heartbeat` là trong phiên, trong tiến trình. Cần lịch bền vững, cô lập → dùng
`hermes cron` (Tài liệu 12).

## 5. Tự động hoá trong phiên {#tu-dong}

| Lệnh | Việc |
| --- | --- |
| `/refine [focus]` | Chạy ngay vòng tự cải thiện memory/skill (thay vì chờ sau lượt) |
| `/review [instructions]` | Spawn subagent review độc lập cho việc vừa bàn (PR, code, docs) |
| `/webhook` | Quản subscription webhook kích hoạt theo sự kiện |

## 6. Skill như slash command {#skill}

Skill đã cài tự động trở thành slash command động trên cả hai mặt. Nếu tên skill trùng một
built-in (hoặc alias của nó), built-in thắng — skill vẫn gọi được bằng
`/skill <name>`. `/skills list`, `/help skills` và
command palette sẽ đánh dấu skill đó là "slash command không khả dụng — trùng tên built-in".

> **Nguồn.** `website/docs/reference/slash-commands.md`, `hermes_cli/commands.py`.
