---
id: index
title: "Hermes Agent"
sidebar_label: 🏠 Tổng quan
sidebar_position: 0
---

# Hermes Agent

Bộ tài liệu này giúp bạn **dùng**, **tích hợp** và
**triển khai** [Hermes Agent](https://hermes-agent.nousresearch.com/) —
agent tự cải thiện của Nous Research. Nội dung được biên soạn từ chính mã nguồn và tài liệu
gốc nằm trong `www/hermes-agent/website/docs/` (447 file) cộng với tài liệu
công khai trên website.

**Đây là gì?**

Một agent terminal-native: vòng lặp tool calling, bộ nhớ bền, skill tự tạo, gateway
nhắn tin đa nền tảng, cron, subagent và 7 backend terminal. Chạy trên $5 VPS tới cụm GPU.

**Khác soigia-harness?**

`soigia-harness` là lõi tham chiếu tối giản (stdlib). Hermes là hệ hoàn
chỉnh: 70+ tool, 28 toolset, gateway, MCP, plugin, desktop. Đọc Tài liệu 01 để so sánh.

**Cách đọc**

Người mới: 01 → 04. Dùng hằng ngày: 05 → 10. Cắm vào hệ của bạn: 13 → 15.
Lên production: 16 → 17.

## Bảy nhóm tài liệu {#nam-nhom}

- **Bắt đầu** (NHÓM A · 4 TRANG) — Hermes là gì, kiến trúc, cài đặt mọi nền tảng, và con đường nhanh nhất
tới một cuộc trò chuyện chạy được. [Xem →](/docs/hermes-agent/01-tong-quan)

- **Sử dụng hằng ngày** (NHÓM B · 6 TRANG) — CLI & TUI, slash commands, provider/model, tools & toolsets,
skills tự học, bộ nhớ & session. [Xem →](/docs/hermes-agent/05-cli-tui)

- **Kết nối & tự động hoá** (NHÓM C · 3 TRANG) — Gateway 21+ nền tảng nhắn tin, cron & script tự động, MCP để cắm
tool ngoài. [Xem →](/docs/hermes-agent/11-gateway-messaging)

- **Tích hợp & mở rộng** (NHÓM D · 2 TRANG) — Dùng như Python library, API, delegation/subagent; viết plugin,
tool, provider, platform adapter. [Xem →](/docs/hermes-agent/14-tich-hop)

- **Triển khai & vận hành** (NHÓM E · 2 TRANG) — Docker, remote backend, serverless; bảo mật, backup, cập nhật,
xử lý sự cố, FAQ. [Xem →](/docs/hermes-agent/16-trien-khai)

- **Playbook triển khai** (NHÓM F · 7 TRANG) — Best practices mạnh nhất + 6 playbook nhiệm vụ thật: PR review agent,
daily briefing, trợ lý nhóm, giám sát & cảnh báo, nghiên cứu song song,
triển khai production (Docker/SSH, BOOT.md, worktree). [Xem →](/docs/hermes-agent/18-best-practices)

- **Tích hợp harness** (NHÓM G · 3 TRANG) — Sáu mô hình đưa `soigia-harness` vào Hermes; case study thật
(oh-my-hermes, hermes-lcm, portable plugins, skill ecosystem); playbook code: skill,
plugin tool, MCP server, cầu model. [Xem →](/docs/hermes-agent/25-tich-hop-harness)

## Mục lục đầy đủ {#muc-luc}

| # | Trang | Nội dung chính |
| --- | --- | --- |
| 01 | [Tổng quan](/docs/hermes-agent/01-tong-quan) | Khái niệm, tính năng, so sánh với soigia-harness |
| 02 | [Kiến trúc](/docs/hermes-agent/02-kien-truc) | Entry point, AIAgent, state, tool backend, 3 API mode |
| 03 | [Cài đặt](/docs/hermes-agent/03-cai-dat) | Desktop bundle, script, Windows, Termux, Docker, Nix |
| 04 | [Bắt đầu nhanh](/docs/hermes-agent/04-bat-dau-nhanh) | Setup wizard, provider, chat đầu tiên, xác minh |
| 05 | [CLI, TUI & profiles](/docs/hermes-agent/05-cli-tui) | Global options, command families, profile |
| 06 | [Slash commands](/docs/hermes-agent/06-slash-commands) | Session, model, workflow, goal/loop/heartbeat |
| 07 | [Provider & model](/docs/hermes-agent/07-provider-model) | Nous, OpenAI, Anthropic, OpenRouter, local, fallback |
| 08 | [Tools & toolsets](/docs/hermes-agent/08-tools-toolsets) | 70+ tool, 28 toolset, preset, approval & permission |
| 09 | [Skills](/docs/hermes-agent/09-skills) | Chuẩn agentskills.io, cấu trúc, tạo skill, catalog |
| 10 | [Memory & session](/docs/hermes-agent/10-memory-session) | SQLite+FTS5, search, compaction, checkpoint/rollback |
| 11 | [Gateway & messaging](/docs/hermes-agent/11-gateway-messaging) | 21+ nền tảng, setup Telegram/Slack/Discord, bot mode |
| 12 | [Cron & tự động hoá](/docs/hermes-agent/12-cron-automation) | Cron job, script-only, `hermes send`, blueprint |
| 13 | [MCP](/docs/hermes-agent/13-mcp) | Mô hình tư duy, cấu hình, lọc tool, an toàn |
| 14 | [Tích hợp](/docs/hermes-agent/14-tich-hop) | Python library, API server, delegation/subagent, kanban |
| 15 | [Plugin & mở rộng](/docs/hermes-agent/15-plugin-mo-rong) | Plugin SDK, thêm tool/provider/platform, webhook |
| 16 | [Triển khai](/docs/hermes-agent/16-trien-khai) | Docker, SSH/Modal/Daytona, VPS, serverless |
| 17 | [Bảo mật & vận hành](/docs/hermes-agent/17-bao-mat-van-hanh) | Secret redaction, backup, update, troubleshooting, FAQ |
| 18 | [Best practices cốt lõi](/docs/hermes-agent/18-best-practices) | Checklist production, bảo mật, prompt cache, memory/skills, quick ref |
| 19 | [Playbook: PR review agent](/docs/hermes-agent/19-playbook-pr-review) | Skill review, memory quy ước, cron/webhook, deliver github_comment |
| 20 | [Playbook: Daily briefing](/docs/hermes-agent/20-playbook-daily-briefing) | Prompt tự chứa, delegation, persona, [SILENT] |
| 21 | [Playbook: Trợ lý nhóm](/docs/hermes-agent/21-playbook-team-assistant) | Telegram bot, allowlist, pairing, phân quyền admin/user |
| 22 | [Playbook: Giám sát](/docs/hermes-agent/22-playbook-giam-sat) | Watchdog script+agent, zero-token, uptime, alert triage |
| 23 | [Playbook: Nghiên cứu](/docs/hermes-agent/23-playbook-nghien-cuu) | Delegation song song, execute_code, repo scout |
| 24 | [Playbook: Triển khai](/docs/hermes-agent/24-playbook-trien-khai) | Docker, SSH backend, hardening, BOOT.md, worktree |
| 25 | [Tích hợp harness (bản đồ)](/docs/hermes-agent/25-tich-hop-harness) | Sáu mô hình tích hợp, tiêu chí chọn, best practice, pitfall |
| 26 | [Case study thật](/docs/hermes-agent/26-case-study) | oh-my-hermes, hermes-lcm, portable plugins, skill ecosystem |
| 27 | [Playbook: Tích hợp (code)](/docs/hermes-agent/27-playbook-tich-hop-code) | Skill, plugin tool, MCP server, cầu model — code đầy đủ |

> **Nguồn.** Mã nguồn: `www/hermes-agent/` (submodule, ghim tại `d0288be5`). Tài liệu gốc: `www/hermes-agent/website/docs/`. Bản công khai: [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/).
