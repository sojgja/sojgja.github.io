---
id: opencode-cli
title: Hướng dẫn sử dụng OpenCode CLI Chi Tiết (Tình huống thực tế)
sidebar_label: OpenCode CLI
sidebar_position: 10
---

# Hướng dẫn sử dụng OpenCode CLI Chi Tiết

> OpenCode là AI coding agent mã nguồn mở, miễn phí, giúp bạn viết code ngay trong terminal.
> Bài viết này hướng dẫn **từng chức năng một** kèm **tình huống thực tế** để bạn áp dụng ngay.

---

## Mục lục

1. [Cài đặt OpenCode](#1-cài-đặt-opencode)
2. [Cấu hình Provider (AI Model)](#2-cấu-hình-provider-ai-model)
3. [Khởi tạo dự án lần đầu](#3-khởi-tạo-dự-án-lần-đầu)
4. [TUI Mode - Giao diện terminal chính](#4-tui-mode---giao-diện-terminal-chính)
5. [Non-interactive Mode (Run) - Chạy tự động](#5-non-interactive-mode-run---chạy-tự-động)
6. [Continue Session - Tiếp tục công việc dở dang](#6-continue-session---tiếp-tục-công-việc-dở-dang)
7. [Web Mode - Dùng OpenCode qua trình duyệt](#7-web-mode---dùng-opencode-qua-trình-duyệt)
8. [Serve Mode - Chạy OpenCode dạng server](#8-serve-mode---chạy-opencode-dạng-server)
9. [Quản lý Session](#9-quản-lý-session)
10. [Quản lý Models](#10-quản-lý-models)
11. [Quản lý Agents (AI Agent tùy chỉnh)](#11-quản-lý-agents-ai-agent-tùy-chỉnh)
12. [Quản lý MCP Servers](#12-quản-lý-mcp-servers)
13. [Tích hợp GitHub](#13-tích-hợp-github)
14. [Plugins](#14-plugins)
15. [Stats & Debug - Xem thống kê & gỡ lỗi](#15-stats--debug---xem-thống-kê--gỡ-lỗi)
16. [Nâng cấp & Gỡ cài đặt](#16-nâng-cấp--gỡ-cài-đặt)
17. [Global Flags & Biến môi trường](#17-global-flags--biến-môi-trường)
18. [Cấu hình TUI nâng cao](#18-cấu-hình-tui-nâng-cao)
19. [Mẹo & Thủ thuật thực tế](#19-mẹo--thủ-thuật-thực-tế)

---

## 1. Cài đặt OpenCode

### 1.1. Tình huống: Bạn là developer mới, muốn cài OpenCode nhanh nhất

**Cách nhanh nhất - dùng script một dòng:**

```bash
curl -fsSL https://opencode.ai/install | bash
```

> Script này tự động phát hiện hệ điều hành, tải bản phù hợp và cài đặt.

### 1.2. Tình huống: Bạn dùng npm và muốn quản lý version qua npm

```bash
npm install -g opencode-ai
```

### 1.3. Tình huống: Bạn dùng macOS và thích Homebrew

```bash
brew install anomalyco/tap/opencode
```

### 1.4. Tình huống: Bạn cài trên nhiều môi trường khác nhau

```bash
# Bun
bun install -g opencode-ai

# pnpm
pnpm install -g opencode-ai

# Yarn
yarn global add opencode-ai

# Windows Scoop
scoop install opencode

# Windows Chocolatey
choco install opencode

# Arch Linux
sudo pacman -S opencode

# Docker (không cần cài đặt gì cả)
docker run -it --rm ghcr.io/anomalyco/opencode
```

### 1.5. Tình huống: Bạn muốn kiểm tra đã cài đặt thành công chưa

```bash
opencode --version
# ví dụ output: opencode version 0.1.48

opencode --help
# Hiển thị toàn bộ trợ giúp
```

---

## 2. Cấu hình Provider (AI Model)

### 2.1. Tình huống: Bạn lần đầu chạy OpenCode và cần kết nối AI

Bước 1: Chạy OpenCode
```bash
opencode
```

Bước 2: Trong giao diện TUI, gõ lệnh:
```
/connect
```

Bước 3: Một menu hiện ra, bạn chọn:
- **opencode** (Zen) - Khuyên dùng cho người mới, có model miễn phí
- **Anthropic** - Dùng Claude
- **OpenAI** - Dùng GPT
- **Google** - Dùng Gemini
- và 75+ provider khác

Bước 4: Làm theo hướng dẫn để dán API key.

### 2.2. Tình huống: Bạn muốn dùng CLI để cấu hình (không vào TUI)

```bash
# Đăng nhập provider
opencode auth login

# Chỉ định provider cụ thể
opencode auth login -p anthropic

# Đăng nhập với phương thức cụ thể
opencode auth login -p openai -m api-key
```

### 2.3. Tình huống: Bạn muốn kiểm tra provider nào đang được kết nối

```bash
opencode auth list
# hoặc
opencode auth ls
```

Output ví dụ:
```
anthropic    ✓ Authenticated
openai       ✓ Authenticated
google       ✗ Not authenticated
```

### 2.4. Tình huống: Bạn muốn đăng xuất khỏi provider

```bash
opencode auth logout
# Sau đó chọn provider muốn logout
```

### 2.5. Tình huống: Bạn muốn dùng nhiều provider cùng lúc

Bạn có thể kết nối nhiều provider và chuyển đổi qua lại trong TUI bằng lệnh:
```
/models
```
Sau đó chọn model mong muốn. Trong non-interactive mode:
```bash
opencode run -m anthropic/claude-sonnet-4 "Giải thích code này"
opencode run -m openai/gpt-4o "Viết unit test cho module này"
```

---

## 3. Khởi tạo dự án lần đầu

### 3.1. Tình huống: Bạn vừa clone một dự án mới và muốn OpenCode hiểu dự án

```bash
cd ~/projects/my-app
opencode
```

Trong TUI, gõ:
```
/init
```

OpenCode sẽ:
1. Phân tích toàn bộ cấu trúc thư mục
2. Đọc file package.json, tsconfig, Dockerfile, v.v.
3. Phát hiện framework, ngôn ngữ, thư viện
4. Tạo file `AGENTS.md` với đầy đủ thông tin

Sau đó commit file này:
```bash
git add AGENTS.md
git commit -m "Add AGENTS.md for OpenCode"
```

### 3.2. Tình huống: Bạn muốn tự viết AGENTS.md để hướng dẫn OpenCode

Mở file `AGENTS.md` và viết:

```markdown
# Dự án MyApp

## Công nghệ
- Next.js 14 (App Router)
- TypeScript
- Prisma + PostgreSQL
- Tailwind CSS

## Coding Standards
- Sử dụng functional components, không class components
- Import theo thứ tự: React, thư viện, components, utils, styles
- Đặt tên file: PascalCase cho components, camelCase cho utils
- Mỗi component có file .tsx riêng

## Các lệnh quan trọng
- npm run dev: chạy dev server (port 3000)
- npm run build: build production
- npm run test: chạy Jest
- npm run lint: kiểm tra ESLint
- npm run typecheck: kiểm tra TypeScript

## Kiến trúc
- /app: App Router pages
- /components: Shared components
- /lib: Utilities, helpers
- /prisma: Schema & migrations
- /public: Static assets
```

### 3.3. Tình huống: Dự án đã có AGENTS.md và bạn muốn cập nhật

Chạy lại `/init` trong TUI, OpenCode sẽ hỏi bạn có muốn gộp nội dung mới vào file cũ không.

---

## 4. TUI Mode - Giao diện terminal chính

### 4.1. Tình huống: Bắt đầu một ngày làm việc với OpenCode

```bash
cd ~/projects/my-app
opencode
```

Bạn sẽ thấy:
- Dòng trên cùng: version, model đang dùng, thư mục làm việc
- Vùng chat ở giữa: nơi bạn nhập lệnh
- Thanh trạng thái dưới cùng: chế độ, session info

### 4.2. Tình huống: Hỏi OpenCode về codebase khi chưa hiểu dự án

Bạn vừa vào team mới, cần hiểu dự án nhanh:

```
Dự án này dùng công nghệ gì?

Cấu trúc thư mục như thế nào?

Giải thích luồng đăng nhập trong dự án này

File nào là entry point chính?
```

### 4.3. Tình huống: Tham chiếu file cụ thể khi hỏi

Bạn muốn hỏi về một file cụ thể, gõ `@` để tìm kiếm:

```
Giải thích cách hoạt động của hàm handleAuth trong @src/lib/auth.ts

Tìm giúp tôi lỗi trong @src/api/users.ts

So sánh cách xử lý error ở @src/api/orders.ts và @src/api/products.ts
```

> Mẹo: Mỗi khi bạn gõ `@`, OpenCode sẽ fuzzy-search toàn bộ dự án.
> Bạn cũng có thể gõ `@docs/` để duyệt thư mục docs.

### 4.4. Tình huống: Chạy lệnh Bash trực tiếp từ TUI

Bắt đầu dòng lệnh với `!`:

```
!npm run test
!git status
!ls -la src/
!docker ps
```

Output trả về ngay trong conversation, OpenCode có thể phân tích kết quả.

### 4.5. Tình huống: Lập kế hoạch trước khi code (Plan Mode)

Nhấn **Tab** để chuyển sang Plan Mode (góc dưới phải hiển thị "Plan").

Sau đó nhập:
```
Tôi muốn thêm tính năng:
1. Khi user xóa note, đánh dấu deleted trong database
2. Tạo màn hình Recycle Bin để xem notes đã xóa
3. User có thể khôi phục hoặc xóa vĩnh viễn

Hãy lập kế hoạch chi tiết cho tôi
```

OpenCode sẽ tạo plan gồm:
- Các file cần thay đổi
- Các bước thực hiện
- Các rủi ro tiềm ẩn

Sau khi review plan, nhấn **Tab** lần nữa để về Build Mode và nói:
```
OK, triển khai theo plan đó
```

### 4.6. Tình huống: Kéo thả ảnh thiết kế vào prompt

Bạn có ảnh mockup UI, kéo thả trực tiếp vào terminal:

```
Đây là thiết kế mới cho trang profile. Hãy implement nó.
[Kéo thả ảnh vào đây]
```

OpenCode sẽ đọc ảnh và code theo đúng thiết kế.

### 4.7. Tình huống: Undo/Redo khi OpenCode làm sai

```bash
# OpenCode vừa sửa sai một file, bạn gõ:
/undo
```

Tác dụng: Hoàn tác **toàn bộ** thay đổi của message cuối cùng (cả code và chat).

```bash
# Bạn undo nhầm, muốn lấy lại:
/redo
```

> Lưu ý: Undo/Redo dùng Git internally, nên dự án **cần là Git repository**.

### 4.8. Tình huống: Làm việc đa nhiệm với nhiều Session

```bash
# Tạo session mới
/new

# Chuyển đổi giữa các session
/sessions

# hoặc dùng phím tắt
Ctrl+x l
```

Ví dụ thực tế:
- Session 1: Đang fix bug login
- Session 2: Đang thêm feature mới
- Session 3: Đang review code

Chuyển qua lại mà không mất context.

### 4.9. Tình huống: Nén context khi session quá dài

Sau một lúc chat dài, context đầy, model trả lời chậm:

```
/compact
# hoặc
/summarize
```

OpenCode sẽ tóm tắt toàn bộ conversation, giữ lại những thông tin quan trọng, release bộ nhớ.

### 4.10. Tình huống: Chia sẻ session với đồng nghiệp

Bạn muốn đồng nghiệp xem bạn đã làm gì:

```
/share
```

OpenCode tạo link dạng `https://opencode.ai/s/abc123` và copy vào clipboard.
Gửi link đó cho đồng nghiệp.

Hủy chia sẻ:
```
/unshare
```

### 4.11. Tình huống: Xuất session ra file Markdown

Bạn muốn lưu lại conversation làm tài liệu:

```
/export
```

Hoặc phím tắt:
```
Ctrl+x x
```

File Markdown sẽ mở trong editor mặc định.

### 4.12. Tình huống: Dùng external editor để soạn tin nhắn dài

Khi cần viết prompt dài và phức tạp:

```
/editor
```
hoặc
```
Ctrl+x e
```

Mở editor (VS Code, vim, nano...) để soạn thảo thoải mái.

Cấu hình editor:
```bash
# Linux/macOS
export EDITOR="code --wait"

# Windows CMD
set EDITOR=code --wait

# Windows PowerShell
$env:EDITOR = "code --wait"
```

### 4.13. Tình huống: Xem danh sách themes và đổi theme

```
/themes
# Sau đó chọn theme ưa thích
```

Hoặc dùng phím tắt:
```
Ctrl+x t
```

### 4.14. Tình huống: Bật thinking blocks để xem AI "suy nghĩ"

Đối với model hỗ trợ extended thinking (Claude, DeepSeek...):

```
/thinking
```

Bạn sẽ thấy model "suy nghĩ" từng bước trước khi trả lời. Rất hữu ích để hiểu logic của AI.

Để chuyển đổi model variant (bật/tắt thinking thực sự):
```
Ctrl+t
```

### 4.15. Tình huống: Xem chi tiết tool execution

Khi OpenCode chạy tools, bạn muốn xem chi tiết:

```
/details
```

Hiển thị từng bước: đang đọc file nào, chạy lệnh gì, output ra sao.

---

## 5. Non-interactive Mode (Run) - Chạy tự động

### 5.1. Tình huống: Bạn chỉ muốn hỏi nhanh, không cần chat dài

```bash
opencode run "Giải thích closure trong JavaScript là gì?"
```

Output trả về thẳng terminal rồi thoát.

### 5.2. Tình huống: Code review tự động cho Git hook

.tạo file `.git/hooks/pre-commit`:
```bash
#!/bin/bash
opencode run --auto "Review các file đã staged, tìm lỗi tiềm ẩn"
```

### 5.3. Tình huống: Phân tích log lỗi trong CI/CD

```bash
opencode run -f test-results.xml "Phân tích test failures và đề xuất cách fix"
```

### 5.4. Tình huống: Dùng model khác cho từng task

```bash
# Dùng Claude cho task phức tạp
opencode run -m anthropic/claude-sonnet-4 "Refactor module auth"

# Dùng GPT-4o cho task đơn giản
opencode run -m openai/gpt-4o "Viết documentation cho API"

# Dùng Gemini cho task phân tích
opencode run -m google/gemini-2.0-flash "Phân tích sentiment của comments"
```

### 5.5. Tình huống: Kết hợp với các lệnh Unix khác

```bash
# Pipe nội dung file vào OpenCode
cat error.log | opencode run "Phân tích lỗi và đề xuất giải pháp"

# Kết hợp với git
git diff HEAD~5 | opencode run "Tạo release notes từ các commit này"

# Với curl
curl https://api.example.com/health | opencode run "Phân tích health check response"
```

### 5.6. Tình huống: Tự động approve permissions (cẩn thận!)

```bash
# Chỉ dùng khi bạn tin tưởng task
opencode run --auto "Format code và fix lint errors"
```

> Cảnh báo: `--auto` sẽ tự động approve mọi hành động, kể cả xóa file.

### 5.7. Tình huống: Fork session để thử nghiệm

```bash
# Tiếp tục session trước nhưng fork ra nhánh riêng
opencode run -c --fork "Thử cách tiếp cận khác"
```

### 5.8. Tình huống: Chạy với title cụ thể

```bash
opencode run --title "Fix bug production - 2024-01-15" "Fix lỗi 500 ở /api/users"
```

---

## 6. Continue Session - Tiếp tục công việc dở dang

### 6.1. Tình huống: Hôm qua làm dở, hôm nay muốn tiếp tục

```bash
# Tự động load session gần nhất
opencode -c
# hoặc
opencode --continue
```

### 6.2. Tình huống: Có nhiều session, muốn tiếp tục session cụ thể

```bash
# Liệt kê session để lấy ID
opencode session list

# Tiếp tục session cụ thể
opencode -s session_abc123
```

### 6.3. Tình huống: Tiếp tục và chạy ngay một lệnh

```bash
opencode -c -s session_abc123 -p "Tiếp tục từ chỗ refactor hàm calculateTotal"
```

---

## 7. Web Mode - Dùng OpenCode qua trình duyệt

### 7.1. Tình huống: Bạn muốn dùng OpenCode trên máy tính không cài đặt

Trên máy chính:
```bash
opencode web
```

Mở trình duyệt và truy cập `http://localhost:xxxx` (port random).

### 7.2. Tình huống: Dùng OpenCode trên máy khác trong cùng mạng

Trên máy server:
```bash
opencode web --port 4096 --hostname 0.0.0.0
```

Trên máy client: mở `http://<IP-máy-server>:4096`

### 7.3. Tình huống: Bảo mật web mode bằng mật khẩu

```bash
export OPENCODE_SERVER_PASSWORD=matkhau@123
opencode web
```

Lúc này truy cập web sẽ yêu cầu nhập username/password.

---

## 8. Serve Mode - Chạy OpenCode dạng server

### 8.1. Tình huống: Bạn muốn chạy OpenCode server để nhiều TUI kết nối đến

```bash
# Terminal 1: Start server
opencode serve --port 4096
```

```bash
# Terminal 2: Kết nối TUI vào server
opencode attach http://localhost:4096
```

### 8.2. Tình huống: Kết nối từ xa qua SSH

```bash
# Trên server remote
opencode serve --port 4096 --hostname 0.0.0.0
export OPENCODE_SERVER_PASSWORD=secret

# Trên máy local qua SSH tunnel
ssh -L 4096:localhost:4096 user@server.com

# Kết nối TUI local vào server remote
opencode attach http://localhost:4096
```

### 8.3. Tình huống: Dùng Run mode với server để tăng tốc

Khi attach vào server, MCP servers, LSP đã được load sẵn:

```bash
opencode run --attach http://localhost:4096 "Giải thích code này"
```

Nhanh hơn nhiều so với khởi động từ đầu mỗi lần.

---

## 9. Quản lý Session

### 9.1. Tình huống: Cuối tuần muốn dọn dẹp session cũ

```bash
# Xem 20 session gần nhất
opencode session list -n 20

# Xem dạng JSON (cho script xử lý)
opencode session list --format json
```

### 9.2. Tình huống: Xóa session chứa thông tin nhạy cảm

```bash
opencode session delete session_abc123
```

### 9.3. Tình huống: Export session để backup hoặc chia sẻ

```bash
# Export đầy đủ
opencode export session_abc123

# Export đã ẩn dữ liệu nhạy cảm (API key, secret...)
opencode export session_abc123 --sanitize

# Export toàn bộ session để backup
opencode export session_abc123 > backup.json
```

### 9.4. Tình huống: Import session từ đồng nghiệp

```bash
# Từ file
opencode import session.json

# Từ share URL
opencode import https://opncd.ai/s/abc123
```

---

## 10. Quản lý Models

### 10.1. Tình huống: Bạn muốn xem có những model nào khả dụng

```bash
opencode models
```

Output:
```
anthropic/claude-sonnet-4
anthropic/claude-haiku-3.5
openai/gpt-4o
openai/gpt-4o-mini
google/gemini-2.0-flash
[...]
```

### 10.2. Tình huống: Chỉ xem model của một provider

```bash
opencode models anthropic
opencode models openai
```

### 10.3. Tình huống: Vừa thêm provider mới, cần refresh danh sách

```bash
opencode models --refresh
```

### 10.4. Tình huống: So sánh giá giữa các model

```bash
opencode models --verbose
```

Hiển thị thêm cost per token, context window size, v.v.

---

## 11. Quản lý Agents (AI Agent tùy chỉnh)

### 11.1. Tình huống: Tạo agent chuyên review code

```bash
opencode agent create
```

Trả lời các câu hỏi:
- Path: `.opencode/agent/code-reviewer.md`
- Description: "Chuyên gia review code, kiểm tra security, performance"
- Mode: primary
- Permissions: read, grep, glob
- Model: anthropic/claude-sonnet-4

Hoặc tạo non-interactive:
```bash
opencode agent create \
  --path .opencode/agent/code-reviewer.md \
  --description "Chuyên gia review code" \
  --mode primary \
  --permissions "read,grep,glob" \
  -m anthropic/claude-sonnet-4
```

### 11.2. Tình huống: Tạo agent chuyên viết test

```bash
opencode agent create \
  --path .opencode/agent/test-writer.md \
  --description "Viết unit test và integration test" \
  --mode subagent \
  --permissions "read,edit,bash" \
  -m anthropic/claude-sonnet-4
```

### 11.3. Tình huống: Xem danh sách agents đã tạo

```bash
opencode agent list
```

### 11.4. Tình huống: Dùng agent cụ thể trong session

Trong TUI:
```
Sử dụng agent code-reviewer để review code mới
```

Hoặc từ CLI:
```bash
opencode run --agent code-reviewer "Review file src/auth.ts"
```

---

## 12. Quản lý MCP Servers

### 12.1. Tình huống: Kết nối OpenCode với GitHub qua MCP

```bash
opencode mcp add
```

Chọn loại server, nhập thông tin:
- Name: github
- Command: npx
- Args: -y @modelcontextprotocol/server-github
- Token: <GitHub PAT>

### 12.2. Tình huống: Kết nối với database qua MCP

```bash
opencode mcp add
```

- Name: postgres
- Command: npx
- Args: -y @anthropic-ai/mcp-server-postgres
- Connection string: postgresql://user:pass@localhost:5432/mydb

### 12.3. Tình huống: Kiểm tra MCP server đang chạy

```bash
opencode mcp list
# hoặc
opencode mcp ls
```

Hiển thị trạng thái: running, error, disconnected.

### 12.4. Tình huống: Xác thực OAuth cho MCP server

```bash
opencode mcp auth github
opencode mcp auth list
opencode mcp auth ls
```

Xóa credentials:
```bash
opencode mcp logout github
```

Debug khi gặp lỗi:
```bash
opencode mcp debug github
```

---

## 13. Tích hợp GitHub

### 13.1. Tình huống: Tự động review PR trên GitHub

```bash
# Cài đặt GitHub agent trong repo
opencode github install
```

Lệnh này tạo GitHub Actions workflow trong `.github/workflows/`.

### 13.2. Tình huống: Chạy GitHub agent thủ công

```bash
opencode github run --event pull_request --token ghp_xxxx
```

### 13.3. Tình huống: Checkout PR và chạy OpenCode ngay

```bash
# Checkout PR #42 và chạy OpenCode
opencode pr 42
```

OpenCode sẽ:
1. Fetch PR branch
2. Checkout
3. Mở TUI để bạn làm việc

---

## 14. Plugins

### 14.1. Tình huống: Cài đặt plugin mới

```bash
opencode plugin <tên-plugin>
# hoặc
opencode plug <tên-plugin>
```

### 14.2. Tình huống: Cài đặt plugin global (dùng cho mọi dự án)

```bash
opencode plugin -g <tên-plugin>
```

### 14.3. Tình huống: Cập nhật plugin khi có phiên bản mới

```bash
opencode plugin -f <tên-plugin>
```

---

## 15. Stats & Debug - Xem thống kê & gỡ lỗi

### 15.1. Tình huống: Cuối tháng muốn xem đã dùng OpenCode bao nhiêu

```bash
# Tổng quan
opencode stats

# 30 ngày gần nhất
opencode stats --days 30

# Top 10 tools được dùng nhiều nhất
opencode stats --tools 10

# Model nào dùng nhiều nhất (kèm số lượng)
opencode stats --models 5

# Chỉ xem cho project hiện tại
opencode stats --project "my-app"
```

### 15.2. Tình huống: Gỡ lỗi khi OpenCode hoạt động bất thường

```bash
opencode debug
```

Xem thông tin chi tiết về:
- Cấu hình
- Providers
- MCP servers
- LSP servers
- System info

### 15.3. Tình huống: Truy vấn trực tiếp database

```bash
opencode db "SELECT COUNT(*) FROM sessions"
opencode db "SELECT * FROM sessions ORDER BY created_at DESC LIMIT 5"

# Xem đường dẫn database
opencode db path
# Output: ~/.local/share/opencode/data.db
```

---

## 16. Nâng cấp & Gỡ cài đặt

### 16.1. Tình huống: Muốn cập nhật OpenCode lên bản mới nhất

```bash
opencode upgrade
```

### 16.2. Tình huống: Cần một phiên bản cụ thể cho tương thích

```bash
opencode upgrade v0.1.48
```

### 16.3. Tình huống: Chỉ định cách cài đặt khi upgrade

```bash
# Nếu bạn cài bằng npm
opencode upgrade -m npm

# Nếu bạn cài bằng brew
opencode upgrade -m brew

# Nếu bạn cài bằng script
opencode upgrade -m curl
```

### 16.4. Tình huống: Muốn gỡ hoàn toàn OpenCode

```bash
# Gỡ hoàn toàn
opencode uninstall

# Giữ lại cấu hình (để cài lại sau)
opencode uninstall -c

# Giữ lại session data
opencode uninstall -d

# Xem trước những gì sẽ bị xóa
opencode uninstall --dry-run
```

---

## 17. Global Flags & Biến môi trường

### 17.1. Global Flags

```bash
# Xem help
opencode -h

# Xem version
opencode -v

# In logs ra stderr (hữu ích khi debug)
opencode run "..." --print-logs

# Chỉ định log level
opencode run "..." --log-level DEBUG

# Chạy không có plugins (khi gặp lỗi do plugin)
opencode --pure
```

### 17.2. Biến môi trường quan trọng và tình huống sử dụng

**Tình huống: Bạn ở Trung Quốc, cần dùng models URL khác**
```bash
export OPENCODE_MODELS_URL="https://your-mirror.com/models"
```

**Tình huống: Server production, không muốn auto update**
```bash
export OPENCODE_DISABLE_AUTOUPDATE=true
```

**Tình huống: Chạy OpenCode trên server headless**
```bash
export OPENCODE_SERVER_PASSWORD=matkhau@123
export OPENCODE_SERVER_USERNAME=admin
```

**Tình huống: Tích hợp với hệ thống CI**
```bash
export OPENCODE_PERMISSION='{"allow": ["Bash(npm *)", "Read", "Edit"]}'
```

**Tình huống: Windows, cần chỉ định Git Bash path**
```bash
export OPENCODE_GIT_BASH_PATH="C:\Program Files\Git\bin\bash.exe"
```

**Tình huống: Bật web search cho OpenCode**
```bash
export OPENCODE_ENABLE_EXA=true
```

**Tình huống: Dùng config từ file tùy chỉnh**
```bash
export OPENCODE_CONFIG="/path/to/custom-config.json"
export OPENCODE_CONFIG_DIR="/path/to/config-dir"
export OPENCODE_TUI_CONFIG="/path/to/tui-config.json"
```

**Tình huống: Tắt các tính năng để tăng tốc**
```bash
export OPENCODE_DISABLE_LSP_DOWNLOAD=true    # Không tải LSP
export OPENCODE_DISABLE_AUTOCOMPACT=true     # Tắt auto compact
export OPENCODE_DISABLE_MODELS_FETCH=true    # Không fetch models từ remote
export OPENCODE_DISABLE_MOUSE=true           # Tắt mouse trong TUI
export OPENCODE_DISABLE_TERMINAL_TITLE=true  # Không đổi title terminal
```

---

## 18. Cấu hình TUI nâng cao

### 18.1. File `tui.json` đầy đủ

```json
{
  "$schema": "https://opencode.ai/tui.json",
  "theme": "opencode",
  "leader_timeout": 2000,
  "keybinds": {
    "leader": "ctrl+x",
    "command_list": "ctrl+p"
  },
  "scroll_speed": 3,
  "scroll_acceleration": {
    "enabled": true
  },
  "diff_style": "auto",
  "mouse": true,
  "attention": {
    "enabled": true,
    "notifications": true,
    "sound": true,
    "volume": 0.4,
    "sound_pack": "opencode.default",
    "sounds": {
      "error": "./sounds/error.mp3",
      "done": "./sounds/done.mp3"
    }
  }
}
```

### 18.2. Tình huống: Bạn muốn custom keybinds

```json
{
  "keybinds": {
    "leader": "ctrl+space",
    "command_list": "ctrl+shift+p",
    "new_session": "ctrl+n",
    "toggle_theme": "ctrl+t"
  }
}
```

### 18.3. Tình huống: Bạn muốn dùng theme tối

```json
{
  "theme": "catppuccin-mocha"
}
```

Các theme phổ biến: `opencode`, `catppuccin-latte`, `catppuccin-mocha`, `dracula`, `nord`, `monokai`, `github-dark`, `github-light`.

### 18.4. Tình huống: Bạn muốn nhận thông báo khi task hoàn thành

```json
{
  "attention": {
    "enabled": true,
    "notifications": true,
    "sound": true
  }
}
```

### 18.5. Tình huống: Bạn thích scroll mượt như macOS

```json
{
  "scroll_acceleration": {
    "enabled": true
  }
}
```

### 18.6. Tình huống: Bạn muốn ẩn username trong TUI

Dùng command palette: `Ctrl+p` → search "hide username"

---

## 19. Mẹo & Thủ thuật thực tế

### 19.1. Tình huống: Bug production khẩn cấp

```bash
opencode -c "Tiếp tục fix bug login. Nhớ:
1. Kiểm tra log lỗi ở /var/log/app.log
2. Xem file @src/auth/login.ts
3. Fix và chạy test trước khi push
4. Tạo PR mới"
```

### 19.2. Tình huống: Code review cho team

```bash
opencode run --auto -f src/new-feature.ts "Review code này:
- Kiểm tra security issues
- Kiểm tra performance
- Gợi ý cải thiện code style
- Kiểm tra error handling"
```

### 19.3. Tình huống: Refactor toàn bộ module

Nhấn Tab sang Plan Mode:
```
Tôi muốn refactor module @src/api/users.ts:
- Tách thành nhiều file nhỏ hơn
- Thêm error handling
- Thêm validation
- Viết unit test

Hãy lập plan chi tiết
```

Review plan, về Build Mode:
```
OK, thực hiện theo plan. Nhưng nhớ giữ nguyên API interface.
```

### 19.4. Tình huống: Tạo REST API hoàn chỉnh

```
Tạo REST API cho quản lý sản phẩm:
- GET /api/products (list, phân trang, filter)
- GET /api/products/:id (detail)
- POST /api/products (create)
- PUT /api/products/:id (update)
- DELETE /api/products/:id (soft delete)

Dùng Express + TypeScript + Prisma + PostgreSQL.
Có validation, error handling, và unit test.
```

### 19.5. Tình huống: Phân tích technical debt

```
Phân tích technical debt của dự án:
1. Xem các file có complexity cao
2. Xem các file thiếu error handling
3. Xem các file thiếu type safety
4. Đề xuất các ưu tiên cần refactor
```

### 19.6. Tình huống: Học codebase mới nhanh

Khi bạn join dự án mới:
```
Tôi là dev mới. Hãy:
1. Tóm tắt kiến trúc tổng thể
2. Giải thích luồng dữ liệu chính
3. Chỉ ra các file quan trọng nhất
4. Giải thích conventions của dự án
```

### 19.7. Tình huống: Tự động hóa quy trình CRUD

Tạo script `generate-crud.sh`:
```bash
#!/bin/bash
MODEL=$1
FIELDS=$2

opencode run --auto "Tạo CRUD hoàn chỉnh cho model $MODEL với fields: $FIELDS
- Model Prisma
- API routes (RESTful)
- Validation với Zod
- Unit tests với Jest"
```

### 19.8. Tình huống: Migrate codebase

```
Migrate codebase từ JavaScript sang TypeScript:
1. Đọc file @jsconfig.json để hiểu cấu hình
2. Chuyển đổi từng file trong @src/ sang .ts
3. Thêm type definitions
4. Chạy typecheck để đảm bảo không lỗi
```

### 19.9. Tình huống: Tối ưu performance

```
Phân tích performance của API endpoint /api/products:
1. Xem code xử lý
2. Xem database queries
3. Đề xuất tối ưu (index, caching, N+1 problem)
4. Implement các cải thiện
```

### 19.10. Tình huống: Docker hóa ứng dụng

```
Tạo Docker setup cho dự án:
1. Dockerfile cho production (multi-stage build)
2. Dockerfile cho development (hot reload)
3. docker-compose.yml với database
4. Docker ignore file
```

### 19.11. Tình huống: Script CI/CD hàng ngày

```bash
# Chạy vào mỗi sáng để review code
0 9 * * * cd /project && opencode run --auto "Review code mới từ hôm qua, tìm lỗi"

# Chạy trước khi deploy
opencode run --auto -f CHANGELOG.md "Cập nhật changelog từ git log"
```

### 19.12. Tình huống: Xử lý conflict merge

Khi có merge conflict:
```
Tôi có merge conflict ở file @src/app.ts và @src/utils/helper.ts.
Hãy phân tích conflict và giúp tôi resolve.
```

### 19.13. Tình huống: Tạo documentation tự động

```
Tạo API documentation từ code:
1. Đọc các route handlers trong @src/api/
2. Phân tích input/output types
3. Tạo file API.md với các endpoint, parameters, examples
```

### 19.14. Tình huống: Security audit

```
Security audit cho dự án:
1. Kiểm tra dependencies có lỗ hổng (npm audit)
2. Kiểm tra XSS vulnerabilities
3. Kiểm tra SQL injection
4. Kiểm tra authentication/authorization
5. Báo cáo các vấn đề và cách fix
```

### 19.15. Keyboard Shortcuts tổng hợp

| Phím tắt | Chức năng |
|----------|-----------|
| `Ctrl+x n` | New session |
| `Ctrl+x l` | List sessions |
| `Ctrl+x u` | Undo |
| `Ctrl+x r` | Redo |
| `Ctrl+x c` | Compact session |
| `Ctrl+x e` | Open editor |
| `Ctrl+x x` | Export session |
| `Ctrl+x q` | Exit |
| `Ctrl+x m` | List models |
| `Ctrl+x t` | List themes |
| `Tab` | Switch Plan/Build mode |
| `Ctrl+p` | Command palette |
| `@` | File search |
| `Ctrl+t` | Toggle thinking variant |

### 19.16. Lưu ý quan trọng

1. **Luôn có Git repo** - Undo/Redo cần Git
2. **Commit AGENTS.md** - Giúp AI hiểu dự án
3. **Dùng Plan Mode trước** - Cho feature phức tạp
4. **Tham chiếu file bằng @** - Chính xác hơn
5. **/compact thường xuyên** - Khi context đầy
6. **Bảo mật API key** - Không commit key lên Git
7. **Kiểm tra code trước khi commit** - AI có thể sai
