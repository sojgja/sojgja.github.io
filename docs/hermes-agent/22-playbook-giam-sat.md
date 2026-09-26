---
id: 22-playbook-giam-sat
title: "Playbook · Giám sát & cảnh báo"
sidebar_label: "Playbook · Giám sát & cảnh báo"
sidebar_position: 22
---

# Playbook · Giám sát & cảnh báo

Watchdog chạy 24/7 nhưng chỉ báo khi có chuyện: theo dõi thay đổi website/giá, uptime, triage
cảnh báo. Điểm mấu chốt là **script lo phần cơ học, agent lo phần suy luận** —
và dùng chế độ zero-token khi không cần LLM.

## 1. Nguyên tắc: script + agent {#nguyen-tac}

Tham số `script` của cron job là vũ khí. Một script Python chạy **trước**
mỗi lần thực thi, stdout của nó thành ngữ cảnh cho agent:

```
script (fetch, diff, tính toán)  ──stdout──▶  agent (đánh giá, quyết định, định dạng)
```

Nhờ vậy agent không tốn token cho việc cơ học, và chỉ được gọi khi thực sự có điều đáng xem.

## 2. Hai đường zero-token {#zero-token}

| Tình huống | Dùng |
| --- | --- |
| Watchdog định kỳ mà script đã tạo ra *đúng* thông điệp (cảnh báo đĩa, heartbeat) | Cron script-only (`no_agent=True`) — cùng scheduler, không LLM |
| Một phát từ script đang chạy (bước CI, post-commit hook, deploy script) | `hermes send` — pipe stdout/file thẳng tới Telegram/Discord/Slack |

```
echo "cảnh báo: đĩa 92%" | hermes send slack -
hermes send telegram "Deploy xong: v1.2.3"
```

## 3. Playbook: giám sát thay đổi website {#website}

Theo dõi một URL, chỉ báo khi nội dung đổi. Script băm nội dung và lưu state; agent đọc
stdout và quyết định thay đổi này có đáng quan tâm không.

```
# ~/.hermes/scripts/watch-site.py
import hashlib, json, os, urllib.request

URL = "https://example.com/pricing"
STATE = os.path.expanduser("~/.hermes/scripts/.watch-site-state.json")

body = urllib.request.urlopen(URL, timeout=20).read()
digest = hashlib.sha256(body).hexdigest()
old = None
if os.path.exists(STATE):
    old = json.load(open(STATE)).get("digest")
json.dump({"digest": digest}, open(STATE, "w"))

if old and old != digest:
    print(f"CHANGED: {URL} (old={old[:8]} new={digest[:8]})")
# ngược lại: in rỗng → không có gì để agent làm
```

Đăng ký cron (dùng script + agent để đánh giá mức quan trọng):

```
hermes cron create "every 30m" \
  "Đọc output của script. Nếu trống, trả lời [SILENT].
   Nếu có CHANGED, hãy tóm tắt thay đổi đáng chú ý và gửi cảnh báo ngắn." \
  --name "watch-pricing" --deliver telegram
```

> **Biến thể zero-token.** Nếu chỉ cần "báo mỗi khi đổi" mà không cần agent suy luận, đặt `no_agent=True` và để script tự in ra đúng thông điệp — không tốn token nào.

## 4. Playbook: uptime & alert triage {#uptime}

| Bài toán | Cách dựng |
| --- | --- |
| Uptime monitor | Cron mỗi 5 phút gọi `curl`/health check; chỉ báo khi fail |
| Alert triage | Cron đọc cảnh báo (log, monitoring API), phân loại & tóm tắt cái nào cần người |
| Deploy verification | Cron sau deploy kiểm tra health, version, log lỗi; báo kết quả |
| Dependency security audit | Cron tuần chạy audit (OSV/SCA), tóm tắt lỗ hổng mới |

```
# ví dụ: alert triage — chỉ báo thứ đáng báo
hermes cron create "*/15 * * * *" \
  "Đọc cảnh báo mới trong 15 phút qua từ monitoring.
   Phân loại: P0 (cần người ngay), P1, nhiễu.
   Nếu KHÔNG có P0/P1, trả lời [SILENT] và không gửi gì.
   Ngược lại, gửi danh sách P0/P1 kèm 1 dòng hành động đề xuất." \
  --name "alert-triage" --deliver telegram
```

## 5. Chống spam bằng `[SILENT]` {#silent}

> **Mẫu quan trọng.** Nếu phản hồi của cron chứa `[SILENT]`, việc gửi bị chặn. Đây là cách chuẩn để watchdog im lặng khi mọi thứ bình thường:

```
If nothing noteworthy happened, respond with [SILENT].
```

Nguyên tắc: **chỉ được thông báo khi agent có điều để nói.**

## 6. Chọn đích gửi kết quả {#deliver}

| Đích | Cờ `--deliver` |
| --- | --- |
| Chính chat tạo job | `origin` (mặc định) |
| File cục bộ (không thông báo) | `local` |
| Telegram / Discord / Slack | `telegram` · `discord` · `slack` |
| SMS | `sms:+15551234567` |
| Topic trong forum Telegram | `telegram:-100123:456` |

Không có messaging? Dùng `deliver: local` trong lúc thử, rồi nối thông báo sau.

> **Nguồn.** `guides/automation-blueprints.md` (Uptime Monitor, Alert Triage, Deploy Verification, Dependency Security Audit, [SILENT], Delivery Targets), `guides/automate-with-cron.md`, `guides/cron-script-only.md`.
