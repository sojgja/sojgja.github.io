---
id: claude-cli
title: Hướng dẫn sử dụng Claude CLI (Claude Code)
sidebar_label: Claude CLI
sidebar_position: 11
---

# Hướng dẫn sử dụng Claude CLI (Claude Code) Chi Tiết

Claude Code là AI coding agent của Anthropic, giúp bạn đọc codebase, chỉnh sửa file, chạy lệnh và tích hợp với các công cụ phát triển trực tiếp từ terminal. Bài viết này hướng dẫn chi tiết cách cài đặt, cấu hình và sử dụng Claude CLI.

---

## 1. Cài đặt Claude Code

### 1.1. Native Install (khuyên dùng)

**macOS, Linux, WSL:**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**

```powershell
irm https://claude.ai/install.ps1 | iex
```

**Windows CMD:**

```batch
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

### 1.2. Cài đặt qua Homebrew (macOS)

```bash
# Stable channel
brew install --cask claude-code

# Latest channel
brew install --cask claude-code@latest
```

### 1.3. Cài đặt qua WinGet (Windows)

```powershell
winget install Anthropic.ClaudeCode
```

### 1.4. Cài đặt qua Linux package managers

```bash
# Debian/Ubuntu
apt install claude-code

# Fedora/RHEL
dnf install claude-code

# Alpine
apk add claude-code
```

---

## 2. Đăng nhập

Sau khi cài đặt, chạy lệnh `claude` để bắt đầu:

```bash
cd /path/to/project
claude
```

Lần đầu tiên, bạn sẽ được yêu cầu đăng nhập. Làm theo hướng dẫn để xác thực trong trình duyệt.

Các loại tài khoản hỗ trợ:

- **Claude Pro, Max, Team, Enterprise** (khuyên dùng)
- **Claude Console** (API access, thanh toán trả trước)
- **Amazon Bedrock, Google Cloud Agent Platform, Microsoft Foundry**
- **Claude apps gateway** (self-hosted, SSO doanh nghiệp)

Để đăng nhập lại trong session:

```
/login
```

Xem trạng thái xác thực:

```bash
claude auth status
# JSON output
claude auth status --json
```

Đăng xuất:

```bash
claude auth logout
```

Đăng nhập với các tùy chọn đặc biệt:

```bash
# Pre-fill email
claude auth login --email user@example.com

# Force SSO
claude auth login --sso

# Dùng Anthropic Console (API billing)
claude auth login --console
```

---

## 3. Các chế độ sử dụng

### 3.1. Interactive Mode (Mặc định)

```bash
claude
# Chạy với prompt ban đầu
claude "Giải thích dự án này"
```

### 3.2. Non-interactive Mode (Print Mode)

Chạy một query và thoát ngay:

```bash
claude -p "Explain this function"
```

### 3.3. Piped Mode

Xử lý nội dung được pipe vào:

```bash
cat logs.txt | claude -p "Phân tích lỗi trong logs"

# Kết hợp với các lệnh khác
tail -200 app.log | claude -p "Slack cho tôi nếu thấy bất thường"

# Bulk operations
git diff main --name-only | claude -p "Review các file thay đổi"
```

### 3.4. Continue Mode

Tiếp tục session gần nhất:

```bash
claude -c
# Tiếp tục và chạy query
claude -c -p "Check for type errors"
```

### 3.5. Resume Mode

Tiếp tục session theo ID hoặc tên:

```bash
claude -r "auth-refactor" "Finish this PR"
```

### 3.6. Background Mode

Chạy session nền:

```bash
claude --bg "Khắc phục test flaky"
```

### 3.7. Cloud Mode

Tạo session trên claude.ai:

```bash
claude --cloud "Fix the login bug"
```

---

## 4. Các lệnh CLI

| Lệnh | Mô tả |
|------|-------|
| `claude` | Bắt đầu interactive session |
| `claude "query"` | Interactive với prompt ban đầu |
| `claude -p "query"` | Non-interactive, thoát sau khi chạy |
| `claude -c` | Tiếp tục session gần nhất |
| `claude -r <session>` | Resume session theo ID/tên |
| `claude update` | Cập nhật lên phiên bản mới nhất |
| `claude install [version]` | Cài đặt/reinstall bản cụ thể |
| `claude auth login` | Đăng nhập |
| `claude auth logout` | Đăng xuất |
| `claude auth status` | Xem trạng thái xác thực |
| `claude agents` | Mở agent view |
| `claude attach <id>` | Attach vào background session |
| `claude stop <id>` | Dừng background session |
| `claude rm <id>` | Xóa background session |
| `claude respawn <id>` | Khởi động lại background session |
| `claude logs <id>` | Xem output của background session |
| `claude doctor` | Diagnostic cài đặt |
| `claude gateway` | Khởi động gateway server |
| `claude mcp` | Cấu hình MCP servers |
| `claude mcp login <name>` | OAuth login cho MCP server |
| `claude mcp logout <name>` | Xóa OAuth credentials |
| `claude plugin` | Quản lý plugins |
| `claude ultrareview [target]` | Chạy ultrareview non-interactive |
| `claude remote-control` | Bật Remote Control server |
| `claude setup-token` | Tạo OAuth token cho CI/scripts |
| `claude project purge [path]` | Xóa local state của project |
| `claude daemon status` | Xem trạng thái supervisor |
| `claude daemon stop --any` | Dừng supervisor |

---

## 5. CLI Flags quan trọng

| Flag | Mô tả |
|------|-------|
| `-p, --print` | Non-interactive mode |
| `-c, --continue` | Tiếp tục session gần nhất |
| `-r, --resume` | Resume session theo ID |
| `--bg, --background` | Chạy background agent |
| `--model` | Chỉ định model |
| `--agent` | Chỉ định agent |
| `--permission-mode` | Chế độ permission: `edit`, `plan`, `bypassPermissions`, `auto` |
| `--allowedTools` | Tools được phép (không cần prompt) |
| `--disallowedTools` | Tools bị cấm |
| `--dangerously-skip-permissions` | Bỏ qua tất cả permission prompts |
| `--add-dir` | Thêm thư mục làm việc bổ sung |
| `--bare` | Minimal mode (bỏ qua auto-discovery) |
| `--debug` | Debug mode |
| `--debug-file <path>` | Ghi debug logs vào file |
| `--chrome` | Bật Chrome browser integration |
| `--cloud` | Tạo web session trên claude.ai |
| `--advisor <model>` | Bật advisor tool |
| `--append-system-prompt` | Thêm text vào system prompt |
| `--append-system-prompt-file` | Load system prompt từ file |
| `--ax-screen-reader` | Chế độ screen-reader friendly |
| `--betas` | Beta headers cho API requests |
| `--disable-slash-commands` | Tắt skills và commands |
| `--output-format` | Định dạng output |
| `--output-style` | Style output (`full`, `diff`, `compact`) |

---

## 6. Session Commands (bên trong Claude Code)

| Lệnh | Mô tả |
|------|-------|
| `/help` | Hiển thị trợ giúp |
| `/clear` | Xóa lịch sử hội thoại |
| `/exit` hoặc `Ctrl+D` | Thoát Claude Code |
| `/resume` | Tiếp tục hội thoại trước |
| `/login` | Đăng nhập lại |
| `/model` | Chuyển đổi model |
| `/config` | Mở giao diện cài đặt |
| `/doctor` | Kiểm tra và sửa lỗi cài đặt |
| `/memory` | Bật/tắt auto memory |
| `/compact` | Nén context |
| `/undo` | Hoàn tác thay đổi cuối |
| `/redo` | Làm lại thay đổi |
| `/review` | Review thay đổi trước khi commit |
| `/plan` | Tạo plan cho task phức tạp |
| `/schedule` | Lên lịch recurring task |

---

## 7. Permission Modes

Nhấn `Shift+Tab` để chuyển đổi giữa các chế độ:

| Chế độ | Mô tả |
|--------|-------|
| **Edit** (mặc định) | Hỏi permission trước khi chạy lệnh, đọc file, edit file |
| **Plan** | Chỉ đọc, không được phép edit hay chạy lệnh |
| **Auto** | Tự động phân loại hành động, chỉ hỏi với hành động rủi ro |
| **Bypass Permissions** | Bỏ qua tất cả, không hỏi gì cả |

---

## 8. Cấu hình Settings

### 8.1. Các file settings

| Scope | Đường dẫn | Mô tả |
|-------|-----------|-------|
| **User** | `~/.claude/settings.json` | Cá nhân, áp dụng toàn bộ projects |
| **Project** | `.claude/settings.json` | Chia sẻ với team (commit lên git) |
| **Local** | `.claude/settings.local.json` | Cá nhân cho project này (gitignored) |
| **Managed** | (Server/Registry) | Chính sách doanh nghiệp |

### 8.2. Ví dụ settings.json

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test *)",
      "Read(~/.zshrc)"
    ],
    "deny": [
      "Bash(curl *)",
      "Read(./.env)",
      "Read(./secrets/**)"
    ]
  },
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1"
  }
}
```

### 8.3. Các setting phổ biến

| Key | Mô tả |
|-----|-------|
| `permissions.allow` | Danh sách tools được phép tự động |
| `permissions.deny` | Danh sách tools bị cấm |
| `model` | Model mặc định |
| `agent` | Agent mặc định |
| `env` | Biến môi trường cho session |
| `autoCompactEnabled` | Tự động nén context (mặc định: `true`) |
| `autoMemoryEnabled` | Bật auto memory (mặc định: `true`) |
| `verbose` | Chế độ verbose (mặc định: `false`) |
| `gitStatus` | Hiển thị git status (mặc định: `true`) |
| `theme` | Theme hiển thị |
| `autoUpdatesChannel` | Channel cập nhật: `latest` hoặc `stable` |

### 8.4. CLAUDE.md

Thêm file `CLAUDE.md` vào thư mục gốc của dự án để hướng dẫn Claude Code:

```markdown
# Coding Standards
- Sử dụng TypeScript cho tất cả code mới
- Tuân theo ESLint config
- Test coverage tối thiểu 80%

# Build Commands
- npm run dev: chạy development server
- npm run build: build production
- npm run test: chạy tests

# Architecture
- Sử dụng React với Next.js
- API routes trong /pages/api
- Database: PostgreSQL qua Prisma
```

---

## 9. Hooks

Hooks là các shell commands chạy tự động trước/sau các hành động của Claude Code.

Định nghĩa trong `settings.json`:

```json
{
  "hooks": {
    "preToolUse": {
      "Bash": "echo 'About to run: $CLAUDE_TOOL_INPUT'"
    },
    "postToolUse": {
      "edit": "npm run lint --fix"
    },
    "preMessage": "echo 'Starting to process message: $CLAUDE_MESSAGE'",
    "postMessage": "echo 'Done processing message'"
  }
}
```

Các loại hooks:

- `preToolUse` - Chạy trước khi tool được gọi
- `postToolUse` - Chạy sau khi tool hoàn thành
- `preMessage` - Chạy trước khi xử lý message
- `postMessage` - Chạy sau khi xử lý message
- `preCommit` - Chạy trước khi commit
- `postCommit` - Chạy sau khi commit
- `preReview` - Chạy trước khi review

---

## 10. MCP Servers

MCP (Model Context Protocol) cho phép Claude Code kết nối với các công cụ bên ngoài.

### 10.1. Cấu hình MCP trong `.mcp.json`

```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["@anthropic-ai/claude-code-mcp-github"]
    },
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

### 10.2. Quản lý MCP qua CLI

```bash
# Xem trạng thái MCP
claude mcp status

# Thêm MCP server (trong session)
/mcp add
```

---

## 11. Plugins

```bash
# Cài đặt plugin từ marketplace
claude plugin install code-review@claude-plugins-official

# Liệt kê plugins đã cài
claude plugin list

# Xóa plugin
claude plugin remove <name>
```

Danh sách plugin marketplace chính thức: `claude-plugins-official`.

---

## 12. Agents & Sub-agents

### 12.1. Chạy nhiều agents song song

```bash
# Mở agent view
claude agents

# Xem danh sách agents dạng JSON (cho scripting)
claude agents --json

# Dispatch agent với cấu hình cụ thể
claude agents --permission-mode auto --model sonnet
```

### 12.2. Tạo agent từ CLI

```bash
claude --agent my-custom-agent

# Tạo agent động với JSON
claude --agents '{"reviewer":{"description":"Code reviewer","prompt":"You review code for bugs"}}'
```

---

## 13. Background Sessions

```bash
# Chạy session nền
claude --bg "Phân tích test failures"

# Xem danh sách session nền
claude agents --json

# Attach vào session
claude attach 7c5dcf5d

# Xem logs
claude logs 7c5dcf5d

# Dừng session
claude stop 7c5dcf5d

# Xóa session
claude rm 7c5dcf5d

# Khởi động lại session
claude respawn 7c5dcf5d
```

---

## 14. Remote Control

Điều khiển Claude Code từ xa qua claude.ai:

```bash
# Bật Remote Control server
claude remote-control --name "My Project"
```

Sau đó có thể:
- Tiếp tục làm việc từ điện thoại qua Claude iOS app
- Điều khiển từ trình duyệt
- Nhận push notifications khi task hoàn thành

---

## 15. Skills

Skills là các reusable workflows có thể gọi bằng `/tên-skill`:

```markdown
---
name: review-pr
description: Code review cho Pull Request
---
1. Đọc diff của PR
2. Kiểm tra code style
3. Kiểm tra security issues
4. Đề xuất cải thiện
```

Đặt trong `.claude/skills/` hoặc `~/.claude/skills/`.

---

## 16. Ultrareview

```bash
# Chạy non-interactive
claude ultrareview 1234

# JSON output
claude ultrareview 1234 --json

# Tùy chỉnh timeout
claude ultrareview 1234 --timeout 60
```

---

## 17. Scheduled Tasks (Routines)

### 17.1. Cloud Routines (chạy trên Anthropic infrastructure)

```
/schedule "Review pending PRs mỗi sáng 9h"
```

### 17.2. Desktop Scheduled Tasks

```
/schedule "Kiểm tra dependency updates mỗi tuần"
```

### 17.3. Loop trong session

```
/loop "Kiểm tra CI status mỗi 5 phút"
```

---

## 18. Các biến môi trường

| Biến | Mô tả |
|------|-------|
| `CLAUDE_CODE_SIMPLE` | Bật bare mode |
| `CLAUDE_CODE_DISABLE_AUTO_MEMORY` | Tắt auto memory |
| `DISABLE_AUTO_COMPACT` | Tắt auto compact |
| `CLAUDE_CODE_ENABLE_TELEMETRY` | Bật telemetry |
| `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` | TTL cho API key helper |
| `CLAUDE_CODE_DEBUG_LOGS_DIR` | Thư mục lưu debug logs |
| `CLAUDE_AX_SCREEN_READER` | Bật screen-reader mode |
| `CLAUDE_CODE_ENABLE_AWAY_SUMMARY` | Bật session recap |
| `MAX_THINKING_TOKENS` | Giới hạn thinking tokens |

---

## 19. GitHub Actions Integration

```yaml
name: Claude Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Claude Code Review
        uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## 20. GitLab CI/CD Integration

```yaml
claude-code:
  script:
    - claude -p "Review merge request changes for bugs and security issues"
  only:
    - merge_requests
```

---

## 21. Mẹo & Thủ thuật

### 21.1. Prompting hiệu quả

- **Cụ thể**: Thay vì "fix bug", hãy nói "fix lỗi login khi user nhập sai password bị màn hình trắng"
- **Step-by-step**: Chia task phức tạp thành các bước nhỏ
- **Ngữ cảnh**: Cung cấp đủ ngữ cảnh và ví dụ

### 21.2. Sử dụng Plan Mode cho task phức tạp

Trước khi thực hiện thay đổi lớn, chuyển sang Plan Mode để Claude lập kế hoạch trước.

### 21.3. Tận dụng CLAUDE.md

Dùng CLAUDE.md để định nghĩa coding standards, kiến trúc, và các lệnh build/test.

### 21.4. Auto Memory

Claude Code tự động ghi nhớ thông tin về dự án qua các session như lệnh build, debugging insights,...

### 21.5. Dùng bare mode cho scripts

```bash
claude --bare -p "Quick query"  # Nhanh hơn, bỏ qua auto-discovery
```

### 21.6. Keyboard shortcuts

| Phím | Mô tả |
|------|-------|
| `Tab` | Autocomplete |
| `Shift+Tab` | Chuyển permission mode |
| `↑` | Lịch sử lệnh |
| `Ctrl+D` | Thoát |
| `Ctrl+C` | Hủy lệnh hiện tại |

---

## 22. Troubleshooting

### Kiểm tra cài đặt:

```bash
claude doctor
```

### Xóa local state:

```bash
claude project purge ~/work/repo --dry-run  # Preview
claude project purge ~/work/repo -y          # Thực hiện
claude project purge --all                   # Tất cả projects
```

### Debug logs:

```bash
claude --debug api,mcp
claude --debug-file /tmp/claude-debug.log
```

### Cập nhật phiên bản cụ thể:

```bash
claude install 2.1.118
claude install stable
claude install latest
```
