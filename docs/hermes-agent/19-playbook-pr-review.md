---
id: 19-playbook-pr-review
title: "Playbook · GitHub PR review agent"
sidebar_label: "Playbook · GitHub PR review agent"
sidebar_position: 19
---

# Playbook · GitHub PR review agent

**Vấn đề thật:** team mở PR nhanh hơn khả năng review; PR nằm chờ nhiều ngày;
junior merge bug vì không ai kịp xem. **Giải pháp:** một agent theo dõi repo,
review mọi PR về bug/bảo mật/chất lượng, và gửi tóm tắt — bạn chỉ dành thời gian cho PR cần
phán đoán người.

## 1. Kiến trúc & hai lựa chọn trigger {#kien-truc}

```
Cron Timer  ──▶  Hermes Agent  ──▶  GitHub API  ──▶  Review
(mỗi 2 giờ)      + gh CLI            (diff PR)        delivery
                 + skill                               (Telegram/local/
                 + memory                              github_comment)
```

| Cách | Khi nào dùng | Đặc điểm |
| --- | --- | --- |
| **Cron poll** | Không có endpoint công khai, sau NAT/firewall | Đơn giản, trễ tối đa bằng chu kỳ |
| **Webhook** | Có endpoint công khai | Real-time khi PR mở/cập nhật |

## 2. Chuẩn bị {#chuan-bi}

```
# Gateway chạy (để cron thực thi)
hermes gateway install        # hoặc: hermes gateway (foreground)

# GitHub CLI đã đăng nhập
brew install gh               # hoặc: sudo apt install gh
gh auth login
```

Không có messaging? Dùng `--deliver local` → kết quả lưu ở
`~/.hermes/cron/output/`. Tốt để thử trước khi nối thông báo.

## 3. Bước 1–2: chạy tay để xác minh {#thu-cong}

Trong `hermes`, kiểm tra quyền truy cập GitHub:

```
Run: gh pr list --repo myorg/backend-api --state open --limit 3
```

Rồi nhờ review một PR thật:

```
Review this pull request. Read the diff, check for bugs, security issues,
and code quality. Be specific about line numbers and quote problematic code.

Run: gh pr diff 3888 --repo myorg/backend-api
```

> **Nguyên tắc.** Nếu chất lượng review tay chưa tốt, **đừng** tự động hoá. Chất lượng review phụ thuộc vào skill + memory (bước 3–4), không phải vào việc đặt lịch.

## 4. Bước 3: tạo skill review {#skill}

Không có skill, chất lượng review thay đổi giữa các lần chạy. Skill cho hướng dẫn nhất quán,
bền qua phiên và mọi lần cron:

```
mkdir -p ~/.hermes/skills/code-review
```

Tạo `~/.hermes/skills/code-review/SKILL.md`:

```
---
name: code-review
description: Review pull requests for bugs, security issues, and code quality
---

# Code Review Guidelines

## What to Check
1. Bugs — logic errors, off-by-one, null/undefined
2. Security — injection, auth bypass, secrets in code, SSRF
3. Performance — N+1 queries, unbounded loops, memory leaks
4. Style — naming, dead code, missing error handling
5. Tests — có test cho hành vi mới chưa? edge case?

## Output Format
Mỗi phát hiện:
- **File:Line** — vị trí chính xác
- **Severity** — Critical / Warning / Suggestion
- **What's wrong** — một câu
- **Fix** — cách sửa

## Rules
- Cụ thể. Trích code có vấn đề.
- Không bắt lỗi style vặt nếu không ảnh hưởng đọc hiểu.
- Nếu PR tốt, nói tốt. Đừng bịa vấn đề.
- Kết thúc bằng: APPROVE / REQUEST_CHANGES / COMMENT
```

## 5. Bước 4: dạy quy ước của bạn {#memory}

Đây là thứ khiến reviewer thực sự hữu dụng. Dạy bằng memory (bền vĩnh viễn):

```
Remember: backend dùng Python + FastAPI. Mọi endpoint phải có type
annotation và Pydantic model. Không cho raw SQL — chỉ SQLAlchemy ORM.
Test nằm trong tests/ và dùng pytest fixture.

Remember: frontend dùng TypeScript + React. Không dùng type `any`.
Mọi component phải có props interface. Dùng React Query để fetch,
không dùng useEffect cho API call.
```

## 6. Bước 5: cron tự động {#cron}

```
hermes cron create "0 */2 * * *" \
  "Check for new open PRs and review them.

Repos to monitor:
- myorg/backend-api
- myorg/frontend-app

Steps:
1. Run: gh pr list --repo REPO --state open --limit 5 --json number,title,author,createdAt
2. For each PR created or updated in the last 4 hours:
   - Run: gh pr diff NUMBER --repo REPO
   - Review the diff using the code-review guidelines
3. Format output as:

## PR Reviews — today
### [repo] #[number]: [title]
**Author:** [name] | **Verdict:** APPROVE/REQUEST_CHANGES/COMMENT
[findings]

If no new PRs found, say: No new PRs to review." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

Lịch hữu ích:

| Lịch | Khi |
| --- | --- |
| `0 */2 * * *` | Mỗi 2 giờ |
| `0 9,13,17 * * 1-5` | 3 lần/ngày, ngày thường |
| `30m` | Mỗi 30 phút (repo traffic cao) |

```
hermes cron list          # xác minh đã lên lịch
hermes cron run pr-review # chạy ngay không chờ
```

## 7. Biến thể webhook (real-time) {#webhook}

Post review trực tiếp lên PR, không gửi qua chat:

```
hermes webhook subscribe github-pr-review \
  --events "pull_request" \
  --prompt "Review this pull request:
Repository: {repository.full_name}
PR #{pull_request.number}: {pull_request.title}
Author: {pull_request.user.login}
Action: {action}
Diff URL: {pull_request.diff_url}

Fetch the diff with: curl -sL {pull_request.diff_url}
Review for security, performance, code quality, missing tests.
If the PR is a trivial docs/typo change, say so briefly." \
  --skills github-code-review \
  --deliver github_comment
```

Cấu hình route tĩnh trong `config.yaml` (block `platforms.webhook.extra.routes`)
rồi trỏ webhook GitHub tới `http://server:8644/webhooks/github-pr-review`,
Content type `application/json`, event **Pull requests**.

> **Quyền token.** `gh` phải có token scope `repo`. Review sẽ được đăng dưới danh nghĩa tài khoản mà `gh` đang xác thực — cân nhắc dùng bot account riêng.

## 8. Vận hành & xử lý sự cố {#van-hanh}

| Triệu chứng | Xử lý |
| --- | --- |
| `gh: command not found` | Gateway chạy môi trường tối giản — đảm bảo `gh` trong PATH hệ thống rồi restart gateway |
| Review quá chung chung | Thêm skill `code-review`; dạy quy ước qua memory; càng nhiều ngữ cảnh càng tốt |
| Cron không chạy | `hermes gateway status`; `hermes cron list` (job còn bật?) |
| Sợ rate limit | GitHub cho 5.000 request/giờ; mỗi review ~3–5 request → 100 PR/ngày vẫn an toàn |

> **Mở rộng.** Thêm repo = thêm vào prompt. Muốn dashboard tuần: tạo cron `0 9 * * 1` tổng hợp PR đang mở, tuổi PR cũ nhất, PR đã merge, PR stale (>5 ngày), PR chưa gán reviewer. Dùng profile riêng cho reviewer để tách memory/config.

> **Nguồn.** `guides/github-pr-review-agent.md`, `guides/webhook-github-pr-review.md`, `guides/automation-blueprints.md`.
