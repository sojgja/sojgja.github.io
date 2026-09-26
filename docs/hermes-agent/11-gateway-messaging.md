---
id: 11-gateway-messaging
title: "Gateway & messaging"
sidebar_label: "Gateway & messaging"
sidebar_position: 11
---

# Gateway & messaging

Một tiến trình gateway duy nhất đưa cùng một agent lên hơn 21 nền tảng nhắn tin — bạn nhắn
từ Telegram trong khi nó chạy trên VM đám mây. Trang này nói cách bật, nối nền tảng, và giữ an toàn.

## 1. Kiến trúc gateway {#kien-truc}

Gateway là một tiến trình chạy nền (`gateway/run.py`) kết nối nhiều adapter nền
tảng vào cùng lõi `AIAgent`. Phiên liền mạch qua nền tảng; slash command và menu
trợ giúp sinh từ cùng `COMMAND_REGISTRY` như CLI.

```
hermes gateway            # chạy / quản gateway
hermes gateway setup      # wizard nối nền tảng
hermes status             # xem trạng thái nền tảng
```

## 2. Nền tảng hỗ trợ {#nen-tang}

Native trong gateway (trích):

Telegram · Discord · Slack · WhatsApp (bridge Baileys) · WhatsApp Cloud (Meta API chính thức) ·
Signal · SMS · Matrix · Mattermost · IRC · Email · Feishu/Lark · DingTalk · WeCom · QQ ·
LINE · Google Chat · BlueBubbles · ntfy · Open WebUI · Buzz · Photon · Yuanbao · Raft

Ngoài ra: **Microsoft Teams** (qua plugin/bot framework), **Teams Meetings**,
A2A, relay connector, và webhook chung.

## 3. Thiết lập một nền tảng {#setup}

Quy trình chung (ví dụ Telegram):

1. Chạy `hermes gateway setup` và chọn nền tảng.
2. Nhập token/credential (BotFather token cho Telegram, app manifest cho Slack…).
3. Cấu hình allowlist user/group trong block `extra:` của nền tảng trong `~/.hermes/config.yaml`.
4. Với cặp đôi cần phê duyệt: dùng `hermes pairing` để chấp thuận/thu hồi mã ghép đôi.

```
# vài trợ thủ riêng
hermes slack                  # sinh app manifest (mọi lệnh thành slash native)
hermes whatsapp               # cấu hình & ghép WhatsApp (Baileys)
hermes whatsapp-cloud         # cấu hình Meta WhatsApp Business Cloud API
```

## 4. Quyền & phê duyệt {#quyen}

| Cơ chế | Mô tả |
| --- | --- |
| `allow_admin_from` | Ai là admin (có mọi slash command) |
| `user_allowed_commands` | Lệnh user thường được dùng (cộng sàn `/help`, `/whoami`) |
| `group_allow_admin_from` / `group_user_allowed_commands` | Bản tương ứng cho nhóm |
| `hermes pairing` | Chấp thuận/thu hồi mã ghép đôi |

Nếu `allow_admin_from` không đặt cho một scope, scope đó ở chế độ tương thích ngược
không hạn chế. Xem tài liệu từng nền tảng để có ví dụ y hệt nhau về cấu trúc.

## 5. Gửi tin từ script: `hermes send` {#send}

Gửi một tin một-chiều tới nền tảng đã cấu hình — **không** chạy agent loop,
**không** tốn token:

```
hermes send telegram "Deploy xong: v1.2.3"
echo "cảnh báo: đĩa 92%" | hermes send slack -
```

Hợp cho shell script, hook CI, daemon giám sát. Xem thêm
`guides/pipe-script-output.md`.

## 6. Bot mode & peer-to-peer {#bot-peer}

| Cơ chế | Dùng cho |
| --- | --- |
| Bot mode | Agent như một bot trên nền tảng; xem `user-guide/bot-mode.md` |
| `hermes peer` | Đăng ký gateway Hermes ở máy khác, DM agent của nó (`hermes peer dm <peer>[/<agent>] "…"`) |
| Webhook | `hermes webhook` — kích hoạt theo sự kiện; ví dụ review PR |
| Multiplexing | Nhiều kết nối nền tảng trong một gateway |

> **An toàn.** Gateway chạy một agent có quyền shell. Luôn đặt allowlist, dùng pairing, và cân nhắc chạy trên backend cách ly. Mã trong `.env` của nền tảng là secret — không chia sẻ.

> **Nguồn.** `website/docs/user-guide/messaging/*`, `user-guide/bot-mode.md`, `guides/team-telegram-assistant.md`, `developer-guide/gateway-internals.md`.
