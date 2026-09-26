---
id: 16-trien-khai
title: "Triển khai (Docker, remote, serverless)"
sidebar_label: "Triển khai (Docker, remote, serverless)"
sidebar_position: 16
---

# Triển khai (Docker, remote, serverless)

Từ laptop tới VPS $5, cụm GPU, hay serverless hibernate khi rảnh. Trang này gom các đường
triển khai và những điểm cần chú ý về quyền, volume và cập nhật.

## 1. Chọn mô hình triển khai {#chon}

| Hoàn cảnh | Chọn |
| --- | --- |
| Máy cá nhân, tương tác | Cài native (Tài liệu 03) |
| Máy chủ riêng, muốn cô lập | Docker Compose |
| Máy khoẻ ở xa, máy bạn chỉ là terminal | Backend SSH |
| Không muốn giữ máy chạy 24/7 | Modal / Daytona (hibernate khi rảnh) |
| Môi trường HPC không root | Singularity |
| Cần sandbox tạm thời | Vercel Sandbox |

## 2. Docker / Docker Compose {#docker}

```
# từ checkout source
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d
```

Ý nghĩa các thành phần trong `docker-compose.yml`:

| Thành phần | Vai trò |
| --- | --- |
| `HERMES_UID`/`HERMES_GID` | Remap user trong container theo user sở hữu `~/.hermes` để file vẫn đọc/ghi được trên host |
| volume `~/.hermes:/opt/data` | Dữ liệu bền nằm trên host |
| `network_mode: host` | Gateway dùng mạng host (đơn giản hoá mở cổng) |
| `restart: unless-stopped` | Tự khởi động lại |
| `/init` (s6-overlay) | PID 1: chạy cont-init.d (chown, reconcile profile, bật/tắt dashboard) rồi mới tới service |

> **An toàn.** Dashboard mặc định bind `127.0.0.1` và lưu API key. Muốn truy cập ngoài: SSH tunnel hoặc reverse proxy có xác thực — **không** dùng `--insecure --host 0.0.0.0`. Nếu override entrypoint, giữ `/init` đầu tiên, nếu không gateway sẽ không chạy đúng.

Mở API server (OpenAI-compatible) cần `API_SERVER_HOST` + `API_SERVER_KEY`.
Google Chat cần mount file service-account JSON vào container rồi trỏ biến tới mount path.

## 3. Backend terminal từ xa {#remote}

Tool `terminal` có thể chạy trên 7 backend. Điều này tách "nơi agent chạy" khỏi
"nơi lệnh thực thi". Với backend từ xa, cân nhắc bật **egress proxy** để chèn
credential thay vì để secret nằm trong sandbox:

```
hermes egress      # trạng thái egress proxy (mặc định tắt)
```

## 4. VPS & luôn-trực {#vps}

- Chạy gateway trong Docker để tự khởi động lại và cô lập.
- Dùng `hermes gateway setup` nối Telegram/Slack… để điều khiển từ xa.
- Đặt `hermes cron` cho báo cáo/backup định kỳ.
- Bật allowlist & pairing cho gateway — agent có quyền shell.
- `hermes backup` định kỳ (hoặc cron) để sao lưu `~/.hermes`.

## 5. Serverless (Modal/Daytona) {#serverless}

Modal và Daytona cung cấp persistence serverless: môi trường của agent *hibernate* khi
rảnh và thức dậy khi có yêu cầu — gần như không tốn phí giữa các phiên. Hợp với tác vụ thưa
(vài lần/ngày) nhưng cần máy khoẻ khi chạy.

## 6. Cập nhật & vòng đời {#cap-nhat}

| Cách cài | Cập nhật |
| --- | --- |
| Source script | `hermes update` |
| Desktop bundle | Cơ chế tự cập nhật của app |
| Docker | **Không** hỗ trợ `hermes update` — chạy image mới |
| Nix | Nix sở hữu runtime & cập nhật |

```
hermes backup            # sao lưu trước khi nâng cấp
hermes import            # phục hồi
hermes migrate           # nâng cấp config qua các thay đổi schema
hermes doctor            # chẩn đoán sau nâng cấp
```

> **Trước khi nâng cấp.** Đọc `developer-guide/stable-releases.md`; sao lưu `~/.hermes`; ghi lại các thay đổi cục bộ. Sau nâng cấp chạy `hermes doctor` và một chat thật để xác minh.

> **Nguồn.** `docker-compose.yml`, `Dockerfile`, `website/docs/user-guide/docker.md`, `getting-started/updating.md`, `developer-guide/stable-releases.md`.
