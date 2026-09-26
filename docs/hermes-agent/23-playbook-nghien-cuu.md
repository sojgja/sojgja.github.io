---
id: 23-playbook-nghien-cuu
title: "Playbook · Nghiên cứu & delegation"
sidebar_label: "Playbook · Nghiên cứu & delegation"
sidebar_position: 23
---

# Playbook · Nghiên cứu & delegation

Khi công việc chia được thành nhiều nhánh độc lập, delegation biến phiên chính từ "nút cổ chai"
thành "ban điều phối": mỗi subagent chạy song song với context riêng, chỉ tóm tắt cuối quay về.

## 1. Delegation hoạt động ra sao {#khai-niem}

`delegate_task` sinh agent con với **context cô lập**, toolset bị giới
hạn, và terminal riêng. Mặc định chạy 3 subagent song song (cấu hình được). Tool call trung
gian *không bao giờ* vào context chính — chỉ bản tóm tắt cuối quay về.

```
phiên chính ──delegate_task──▶ subagent A ─┐
                                subagent B ─┼─▶ chỉ summary cuối về
                                subagent C ─┘
```

## 2. Khi nào delegate — khi nào không {#khi-nao}

**Nên delegate**

- Việc nặng suy luận (debug, review code, tổng hợp research)
- Việc làm ngập context bằng dữ liệu trung gian
- Nhiều nhánh độc lập chạy song song
- Việc cần context tươi, không thiên lệch

**Dùng thứ khác**

- Một tool call đơn → gọi trực tiếp
- Việc cơ học nhiều bước có logic → `execute_code`
- Việc cần tương tác user → subagent không có `clarify`
- Việc phải sống qua restart → cron hoặc terminal background

## 3. Playbook: nghiên cứu song song {#playbook}

```
Research these three topics in parallel:
1. Current state of WebAssembly outside the browser
2. RISC-V server chip adoption in 2025
3. Practical quantum computing applications

For each, return a structured summary with sources.
```

Ba subagent tìm độc lập cùng lúc; phiên chính chỉ nhận 3 tóm tắt gọn — tiết kiệm mạnh token
so với việc tự tìm tuần tự và nhồi mọi trang vào context.

## 4. `execute_code`: gộp pipeline zero-context {#rpc}

Thay vì chạy từng lệnh terminal một, bảo agent viết script làm tất cả trong một lượt. Dữ liệu
trung gian không đi qua context:

```
Write a Python script to rename all .jpeg files to .jpg in ./assets and run it.
```

Với `execute_code`, agent gọi trực tiếp các tool Hermes qua RPC bên trong script —
multi-step pipeline gộp thành một lượt có chi phí context bằng không.

## 5. Blueprint: competitive repo scout {#scout}

Theo dõi repo đối thủ/nguồn tham khảo, tóm tắt thay đổi đáng chú ý định kỳ:

```
hermes cron create "0 9 * * 1" \
  "Scout these repos for notable changes this week: owner/repo-a, owner/repo-b.
   1. List merged PRs and new releases in the last 7 days
   2. Highlight changes in public APIs, performance, security
   3. Summarize what matters for us
   If nothing notable, respond [SILENT]." \
  --name "repo-scout" --deliver telegram
```

Cùng công thức áp cho: paper digest (tóm tắt bài báo mới), AI news digest, docs drift detection
(phát hiện code đổi mà docs không đổi).

## 6. Best practices {#best}

| Chủ đề | Khuyến nghị |
| --- | --- |
| Cô lập | Mỗi subagent có context/toolset riêng — đừng kỳ vọng nó nhớ phiên chính |
| Song song | Mặc định 3 concurrent; tăng nếu tác vụ thật sự độc lập |
| Tương tác | Subagent không dùng được `clarify` — đừng thiết kế việc cần hỏi user |
| Bền vững | Delegation là async nhưng *trong tiến trình*; cần sống qua restart → cron/background |
| Kanban | Con của `delegate_task` không sở hữu task Kanban |
| Chi phí | Delegation giảm token phiên chính — đo bằng `/usage` |

> **Nguồn.** `guides/delegation-patterns.md`, `user-guide/features/delegation.md`, `guides/tips.md` (Delegate for Parallel Work, execute_code), `guides/automation-blueprints.md` (Competitive Repository Scout).
