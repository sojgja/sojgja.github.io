---
id: 26-case-study
title: "Case study thật từ hệ sinh thái Hermes"
sidebar_label: "Case study thật từ hệ sinh thái Hermes"
sidebar_position: 26
---

# Case study thật từ hệ sinh thái Hermes

Bốn dự án thật đã tích hợp vào Hermes theo bốn cách khác nhau — và bài học rút ra cho việc
đưa `soigia-harness` vào. Mỗi case nói rõ: họ cắm vào điểm nào, cách phân phối, và
điều gì đáng học.

## 1. oh-my-hermes (OMH) — lớp vận hành trên skill {#ohmy}

Một "operating layer" phủ lên Hermes: biến một yêu cầu thường thành *capability* rõ ràng,
một bước kế tiếp hữu ích, và **một bản ghi trung thực về việc đã thực sự xảy ra**.
Cam kết cốt lõi của họ rất đáng học:

**never replacing Hermes**

| Khía cạnh | Cách họ làm |
| --- | --- |
| Điểm cắm | Skills (và plugin) — không sửa core |
| Định vị | "Lớp trên", tăng cường chứ không thay thế |
| Giá trị cốt lõi | Ranh giới bằng chứng rõ + ghi chép trung thực |

> **Liên hệ.** "Evidence boundaries" gần như trùng triết lý với skill điều tra dựa trên bằng chứng của bạn — dấu hiệu cho thấy đây là hướng được cộng đồng đánh giá cao.

## 2. hermes-lcm — context engine plugin {#lcm}

Thay bộ nén context một-phát của Hermes bằng một **context engine dựa trên DAG + SQLite**:
"Bounded context, unbounded memory. Nothing is ever lost." Lấy ý tưởng từ bài báo LCM và dự án
lossless-claw cho OpenClaw.

| Khía cạnh | Cách họ làm |
| --- | --- |
| Điểm cắm | Context engine plugin (thay `context_compressor.py`) |
| Vấn đề giải | Nén có mất mát làm mất chi tiết chính xác |
| Giải pháp | Giữ prompt bounded nhưng lưu raw; cho agent tool truy hồi chi tiết |
| Phân phối | Repo riêng, cài vào plugin; có CI, release, migration từ công cụ khác |

Bài học: plugin có thể thay *một subsystem* (nén context) mà không đụng phần còn lại —
đúng tinh thần "pluggable interfaces".

## 3. Portable Agent Plugins v1 {#portable}

Chuẩn đóng gói để chia sẻ agent plugin giữa các runtime. Cấu trúc tối thiểu:

```
my-portable-plugin/
├── plugin.json
├── skills/
│   └── summarize/SKILL.md
└── mcp.json
```

```
hermes plugins install owner/repository --no-enable
hermes plugins list
hermes plugins enable <plugin-name>
```

| Quy tắc | Chi tiết |
| --- | --- |
| Mặc định tắt | Gói portable bị disable sau khi cài tới khi bạn bật |
| Skill | Read-only, namespace dạng `agent-plugin-<slug>-<hash>` |
| MCP stdio | Lệnh truyền dạng token + argument list, **không** qua shell |
| Bí mật | `env` trong `mcp.json` là dữ liệu công khai — **không** để credential |

## 4. Skill ecosystem — phân phối quy mô lớn {#skills}

Cộng đồng vận hành các *directory* skill khổng lồ và cài bằng một lệnh:

```
hermes skills install skills-sh/ZeroPointRepo/youtube-skills/skills/youtube-full
hermes skills tap add <repo>      # thêm nguồn skill riêng
```

| Nguồn | Quy mô / vai trò |
| --- | --- |
| `0xNyk/awesome-hermes-agent` | Directory độc lập: skill, plugin, memory provider, bridge |
| `ZeroPointRepo/awesome-hermes-skills` | 368 skill + 82 built-in + 117 optional catalog |

Bài học: **skill là đơn vị phân phối rẻ nhất**. Nếu quy trình của harness diễn đạt
được bằng markdown + lệnh, hãy chia sẻ nó dạng skill trước khi nghĩ tới tool/MCP.

## 5. Các case khác đáng tham khảo {#khac}

| Dự án | Loại tích hợp |
| --- | --- |
| `linke-ai/hermes-agent-team` | Hệ đa-agent cục bộ trên Hermes (web) |
| `JPeetz/Hermes-Studio` | Web UI/dashboard cho chat, memory, skill, terminal, approvals |
| `nexu-io/open-design`, `liustack/modlens` | Plugin cho "DeepSeek Harness" — vision/design bridge cho model text-only |
| `farion1231/cc-switch` | Tiện ích đa runtime (Claude Code, Codex, OpenCode, OpenClaw, Hermes…) |

## 6. Bài học cho soigia-harness {#bai-hoc}

| Bài học | Áp dụng cho bạn |
| --- | --- |
| Không thay thế Hermes | Định vị harness là *lớp bổ trợ*: quy trình điều tra + tool tái dùng, không tranh vai agent |
| Ranh giới bằng chứng rõ | Giữ `00-rules/` làm chuẩn bằng chứng; chia sẻ nó cho cả hai bên |
| Skill trước, tool sau | Đóng gói quy trình harness thành SKILL.md (agentskills.io) và phát hành qua `skills tap` |
| Thay subsystem được thì thay | Nếu muốn, harness có thể cung cấp context engine hoặc toolset qua plugin, không đụng core |
| Đóng gói portable | Dùng cấu trúc `plugin.json` + `skills/` + `mcp.json` để chia sẻ dễ |
| Repo riêng cho tích hợp | Plugin tích hợp sản phẩm bên thứ ba nên nằm ở repo riêng, cài vào `~/.hermes/plugins/` |

> **Nguồn.** GitHub: `0xNyk/awesome-hermes-agent`, `rlaope/oh-my-hermes`, `stephenschoettler/hermes-lcm`, `ZeroPointRepo/awesome-hermes-skills`, `linke-ai/hermes-agent-team`, `JPeetz/Hermes-Studio`; docs Hermes: `plugins/index.md` (Portable Agent Plugins v1, third-party policy), `user-guide/features/skills.md`.
