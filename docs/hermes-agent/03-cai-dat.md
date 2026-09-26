---
id: 03-cai-dat
title: "Cài đặt (mọi nền tảng)"
sidebar_label: "Cài đặt (mọi nền tảng)"
sidebar_position: 3
---

# Cài đặt (mọi nền tảng)

Năm đường cài đặt: gói desktop, script nguồn, Docker, Nix và Termux. Trang này nói rõ
chọn đường nào, chúng làm gì, và kết quả để lại trên máy ở đâu.

## 1. Chọn đường cài đặt {#chon}

| Hoàn cảnh | Dùng |
| --- | --- |
| Muốn cả CLI lẫn app desktop, cài nhanh | Gói desktop (macOS/Windows) |
| Chỉ cần CLI, Linux/macOS/WSL2 | `install.sh` |
| Chỉ cần CLI, Windows native | `install.ps1` |
| Android aarch64 | APT repo Termux |
| Máy chủ, cô lập, tái lặp | Docker |
| NixOS / declarative | Nix flake & module |

## 2. Gói desktop (macOS/Windows) {#desktop}

Tải gói cho nền tảng của bạn từ [website Hermes](https://hermes-agent.nousresearch.com/).

- **Windows:** mở file `.appinstaller` bằng Windows App Installer.
Nó cài bundle MSIX đã ký và ghi lại nguồn cập nhật. MSIX yêu cầu **Windows 11 22H2** trở lên.
- **macOS:** mở DMG, kéo `Hermes.app` vào Applications. Bản ZIP mang
app đã ký dùng cho cơ chế tự cập nhật. **Chỉ Apple Silicon**.

Gói bundle đã chứa sẵn agent, Python, phụ thuộc được hỗ trợ và giao diện đã build — lần đầu
mở không phải build runtime. Truy cập provider và tích hợp tuỳ chọn vẫn có thể cần mạng.

> **Phân biệt.** `Hermes-Setup` là bootstrap installer *khác*: nó tải bản cài từ source rồi build app desktop. Biến thể **Light** là bản remote-only, không kèm runtime cục bộ.

## 3. Script nguồn {#script}

### Linux / macOS / WSL2

```
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows native

```
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Sau khi xong, nạp lại shell:

```
source ~/.bashrc   # hoặc: source ~/.zshrc
```

Đã cài CLI nhưng sau muốn thêm desktop: chạy `hermes desktop`.

### Script làm những gì

1. Clone source, bootstrap `uv`, giao việc chuẩn bị phụ thuộc cho **PM**.
2. PM cung cấp Python, Node.js, npm, ripgrep, FFmpeg đã *ghim phiên bản*.
3. Chọn extra Python `all` (không phải mọi extra tuỳ chọn).
4. Cài browser tool (`agent-browser` + Chromium đã ghim) — mặc định.
5. Tạo launcher, chuẩn bị thư mục dữ liệu; chạy setup & cấu hình gateway nếu tương tác.

| Cờ | Tác dụng |
| --- | --- |
| `--skip-browser` / `-SkipBrowser` | Không cài browser tool (ghi nhớ lựa chọn cho cả `hermes update`) |
| `--non-interactive` / `-NonInteractive` | Bỏ các bước cần nhập liệu |
| `--include-desktop` / `-IncludeDesktop` | Build luôn app desktop từ source |
| `--verbose` / `-Verbose` | In toàn bộ output |
| `--dir` (POSIX) | Chọn thư mục checkout source |

Trên terminal, script in một dòng trạng thái mỗi bước và ghi output của git/uv/build vào
`logs/install.log` trong thư mục dữ liệu Hermes. Muốn gỡ browser tool đã cài:
`hermes pm install agent-browser` để cài lại và xoá lựa chọn.

## 4. Android / Termux {#termux}

Trên thiết bị Android aarch64, dùng APT repo đã ký của Termux:

```
# Cấu hình repo đã ký (xem hướng dẫn Termux để lấy lệnh mới nhất)
pkg install hermes-agent
```

Đây là gói prerelease kèm Python, Node và TUI. Chạy gateway trong một session Termux —
Android có thể kill tiến trình nền.

## 5. Docker {#docker}

```
# từ checkout source
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d
```

Đặt `HERMES_UID`/`HERMES_GID` theo user sở hữu `~/.hermes`
để file tạo trong container vẫn đọc/ghi được trên host. Stage2 hook của s6-overlay remap user
`hermes` nội bộ theo hai biến này.

> **An toàn.** Dashboard mặc định bind `127.0.0.1` và **lưu API key**. Muốn truy cập từ xa thì dùng SSH tunnel hoặc reverse proxy có xác thực — **không** dùng `--insecure --host 0.0.0.0`. Nếu override entrypoint, giữ `/init` là lệnh đầu tiên (nó là PID 1 của s6-overlay).

Docker không hỗ trợ `hermes update`; cập nhật bằng cách chạy image mới.

## 6. Nix {#nix}

Có flake và module NixOS: từ `nix run` nhanh tới module declarative đầy đủ với
chế độ container. Nix sở hữu việc cài và cập nhật runtime. Đây là hỗ trợ Tier 2 (best-effort).

## 7. Cài xong để lại gì ở đâu {#layout}

| Cách cài | Mã nguồn | Điểm vào CLI | Dữ liệu mặc định |
| --- | --- | --- | --- |
| Script POSIX | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` | `~/.hermes/` |
| Script Windows | `%LOCALAPPDATA%\hermes\hermes-agent\` | `%LOCALAPPDATA%\hermes\bin\` | `%LOCALAPPDATA%\hermes\` |
| Desktop bundle | Trong gói app | Launcher đóng gói | Thư mục dữ liệu nền tảng |
| Docker | `/opt/hermes/` | Entrypoint + `hermes` | `/opt/data/` (mount) |
| Termux APT | `$PREFIX/lib/hermes-agent/` | Symlink trong `$PREFIX/bin/` | `~/.hermes/` |

Biến `HERMES_HOME` chọn thư mục dữ liệu người dùng. `--dir` của script
POSIX chọn checkout source độc lập. Windows có `-HermesHome` và `-InstallDir`.

> **Lưu ý.** Không xoá data root để "sửa" bản cài ứng dụng — tool store của PM và các thế hệ Python có vòng đời riêng. Xem `reference/package-management.md`.

## 8. Sau khi cài {#sau}

```
source ~/.bashrc     # hoặc ~/.zshrc
hermes               # bắt đầu trò chuyện
```

Cấu hình từng phần về sau bằng các lệnh chuyên biệt (ví dụ `hermes model`, `hermes gateway setup`).

## 9. Không được hỗ trợ {#khong-ho-tro}

- Termux trên thiết bị không phải aarch64
- Cài qua AUR, `pip`/`uv tool install`, hay `brew`
- macOS Intel x86 (gói desktop) và macOS 32-bit

> **Nguồn.** `website/docs/getting-started/installation.md`, `platform-support.md`, `termux.md`, `nix-setup.md`, `user-guide/docker.md`, `docker-compose.yml`.
