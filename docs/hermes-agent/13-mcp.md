---
id: 13-mcp
title: "MCP — cắm tool ngoài"
sidebar_label: "MCP — cắm tool ngoài"
sidebar_position: 13
---

# MCP — cắm tool ngoài

MCP (Model Context Protocol) là lớp adapter để cắm tool từ server ngoài vào Hermes mà không
sửa core. Nhưng dùng tốt MCP không phải "kết nối mọi thứ" — mà là "kết nối đúng thứ, với bề
mặt nhỏ nhất đủ dùng".

## 1. Khi nào dùng MCP {#khi-nao}

**Nên dùng MCP khi**

- Tool đã tồn tại dưới dạng MCP và bạn không muốn viết tool native
- Muốn Hermes vận hành một hệ thống local/remote qua lớp RPC sạch
- Cần kiểm soát chi tiết phần nào của mỗi server được lộ ra
- Nối tới API nội bộ, database, hệ thống công ty mà không sửa Hermes core

**Không nên khi**

- Tool built-in của Hermes đã giải quyết tốt
- Server lộ bề mặt tool khổng lồ nguy hiểm mà bạn chưa sẵn sàng lọc
- Chỉ cần một tích hợp hẹp — tool native sẽ đơn giản và an toàn hơn

## 2. Mô hình tư duy {#mo-hinh}

```
Hermes (agent)  +  MCP server (nguồn tool)  →  tool khám phá lúc khởi động/reload
                                             →  model dùng như tool thường
Bạn kiểm soát: mỗi server lộ bao nhiêu tool
```

Tool MCP được đăng ký động và xuất hiện cùng các tool khác trong danh sách. Chúng đóng góp
vào chi phí context — xem `/context` để biết phần "MCP" tốn bao nhiêu token.

## 3. Cấu hình server {#cau-hinh}

Khai báo server MCP trong `~/.hermes/config.yaml` (khoá `mcp_servers` /
tương đương — chi tiết ở `reference/mcp-config-reference.md`). Hỗ trợ server
dạng stdio (chạy tiến trình cục bộ) và HTTP/SSE (từ xa), cùng native MCP.

```
# dạng mô tả — xem reference để có schema chính xác
mcp_servers:
  my-server:
    command: "npx"
    args: ["-y", "@some/mcp-server"]
    # hoặc url: "https://..."   cho server HTTP/SSE
```

Trên desktop, tool `manage_connections` cho phép kết nối tài khoản connector và
MCP server cục bộ từ catalog qua thẻ phê duyệt.

## 4. Lọc tool theo server {#loc}

Vì MCP server có thể lộ rất nhiều tool, Hermes cho phép giới hạn bề mặt mỗi server. Nguyên tắc:

- Chỉ bật server bạn thực sự cần cho công việc hiện tại.
- Với server lớn, lọc chỉ giữ tool cần thiết.
- Kiểm tra `/context all` để thấy chi phí schema của từng toolset/MCP.

## 5. An toàn {#an-toan}

> **Cảnh báo.** MCP server là mã bên thứ ba chạy với quyền của bạn và có thể lộ tool ghi/xoá. Hãy: pin phiên bản server, đọc kỹ tool nó cung cấp, chạy trong backend cách ly khi có thể, và `hermes security audit` để kiểm tra chuỗi cung ứng cho các MCP server đã ghim.

| Việc | Lệnh |
| --- | --- |
| Audit chuỗi cung ứng | `hermes security audit` |
| Xem chi phí MCP trong context | `/context all` |
| Kết nối connector/MCP từ catalog (desktop) | tool `manage_connections` |

> **Nguồn.** `website/docs/guides/use-mcp-with-hermes.md`, `reference/mcp-config-reference.md`, `developer-guide/native-mcp.md`.
