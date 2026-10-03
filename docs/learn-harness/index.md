---
title: Learn Harness Engineering
description: Khóa học dựa trên dự án về xây dựng môi trường, quản lý trạng thái, cơ chế kiểm chứng và điều khiển giúp AI coding agent hoạt động đáng tin cậy.
---

# Learn Harness Engineering

> Khóa học dựa trên dự án về xây dựng **môi trường, quản lý trạng thái, cơ chế kiểm chứng và điều khiển** giúp các AI coding agent hoạt động đáng tin cậy.

![Learn Harness Engineering](/img/learn-harness/harness-pattern.svg)

Nguồn: [walkinglabs/learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering) (MIT) — bản tiếng Việt được tổng hợp lại tại đây. Các tài liệu tham khảo cốt lõi của khóa học:

- [OpenAI: Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- [Anthropic: Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Anthropic: Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Awesome Harness Engineering](https://github.com/walkinglabs/awesome-harness-engineering)

## Mô hình thì thông minh, Harness giúp nó đáng tin cậy

Có một sự thật khắc nghiệt: **mô hình mạnh nhất thế giới vẫn sẽ thất bại trên các tác vụ kỹ thuật thực tế nếu bạn không xây dựng một môi trường phù hợp xung quanh nó.**

Bạn giao một tác vụ cho Claude hoặc GPT trong kho mã của mình. Nó bắt đầu tốt — đọc tệp, viết code, trông có vẻ hiệu quả. Sau đó có điều gì đó sai: nó bỏ qua một bước, làm hỏng một bài kiểm tra, nói "xong" nhưng thực tế không có gì hoạt động.

Đây không phải là vấn đề của mô hình. Đây là vấn đề của **harness**.

Bằng chứng từ Anthropic: cùng mô hình (Opus 4.5), cùng prompt ("xây dựng trình soạn thảo game retro 2D"). Không có harness: 9 đô la trong 20 phút, kết quả không dùng được. Với harness đầy đủ (planner + generator + evaluator): 200 đô la trong 6 giờ, ra một game chơi được thật. **Mô hình không thay đổi. Harness đã thay đổi.**

## Năm hệ thống con của một Harness

Một harness production gồm 5 hệ thống con, làm việc cùng nhau như một vòng lặp khép kín:

![Năm hệ thống con của harness](/img/learn-harness/harness-subsystems.svg)

| Hệ thống con | Vai trò |
| --- | --- |
| **Instructions (Chỉ dẫn)** | AGENTS.md, quy ước, phạm vi — nói cho agent biết làm gì, thứ tự nào |
| **Tools (Công cụ)** | CLI, script, MCP — tay chân của agent |
| **Environment (Môi trường)** | init.sh, dependency, seed data — nơi agent sống |
| **State (Trạng thái)** | feature list, progress file, lịch sử git — trí nhớ xuyên phiên |
| **Feedback (Phản hồi)** | test, lint, typecheck, review — agent chỉ dừng khi kiểm chứng đạt |

## Lộ trình học tập

![Lộ trình học tập](/img/learn-harness/harness-learning-path.svg)

## Chuỗi bài viết

### 📖 14 bài giảng

| # | Bài giảng |
| --- | --- |
| 01 | [Mô hình mạnh không có nghĩa là thực thi đáng tin cậy](./lectures/lecture-01-why-capable-agents-still-fail/) |
| 02 | [Harness thực sự có nghĩa là gì](./lectures/lecture-02-what-a-harness-actually-is/) |
| 03 | [Vì sao repository phải trở thành hệ thống ghi chép chính](./lectures/lecture-03-why-the-repository-must-become-the-system-of-record/) |
| 04 | [Vì sao một file chỉ dẫn khổng lồ lại thất bại](./lectures/lecture-04-why-one-giant-instruction-file-fails/) |
| 05 | [Vì sao tác vụ dài hạn mất tính liên tục](./lectures/lecture-05-why-long-running-tasks-lose-continuity/) |
| 06 | [Vì sao khởi tạo cần một giai đoạn riêng](./lectures/lecture-06-why-initialization-needs-its-own-phase/) |
| 07 | [Vì sao agent làm quá phạm vi và bỏ dở công việc](./lectures/lecture-07-why-agents-overreach-and-under-finish/) |
| 08 | [Vì sao danh sách tính năng là "nguyên thủy" của harness](./lectures/lecture-08-why-feature-lists-are-harness-primitives/) |
| 09 | [Vì sao agent tuyên bố chiến thắng quá sớm](./lectures/lecture-09-why-agents-declare-victory-too-early/) |
| 10 | [Vì sao kiểm thử end-to-end thay đổi kết quả](./lectures/lecture-10-why-end-to-end-testing-changes-results/) |
| 11 | [Vì sao quan sát được (observability) phải nằm trong harness](./lectures/lecture-11-why-observability-belongs-inside-the-harness/) |
| 12 | [Vì sao mỗi phiên làm việc phải để lại trạng thái sạch](./lectures/lecture-12-why-every-session-must-leave-a-clean-state/) |
| 13 | [Loop Engineering — từ nhắc lệnh thủ công đến vòng lặp tự chủ](./lectures/lecture-13-loop-engineering/) |
| 14 | [Graph Engineering — từ vòng lặp đơn lẻ đến đồ thị](./lectures/lecture-14-graph-engineering/) |

### 🛠️ 8 dự án thực hành

| # | Dự án |
| --- | --- |
| 01 | [Chỉ Prompt vs. Harness tối thiểu](./projects/project-01-baseline-vs-minimal-harness/) |
| 02 | [Không gian làm việc "đọc được" bởi agent](./projects/project-02-agent-readable-workspace/) |
| 03 | [Tính liên tục đa phiên](./projects/project-03-multi-session-continuity/) |
| 04 | [Index hóa tăng dần](./projects/project-04-incremental-indexing/) |
| 05 | [QA có căn cứ xác thực](./projects/project-05-grounded-qa-verification/) |
| 06 | [Quan sát runtime và gỡ lỗi](./projects/project-06-runtime-observability-and-debugging/) |
| 07 | [Xây dựng vòng lặp tự động đầu tiên](./projects/project-07-loop-engineering-first-loop/) |
| 08 | [Vẽ quy trình làm việc thành đồ thị](./projects/project-08-graph-engineering-first-graph/) |

### 🔬 Phân tích thiết kế harness thực tế

- [Cách **Pi** xây dựng harness](./harness-designs/pi/) — kernel tối giản, mở rộng bằng code, context engineering
- [Cách **Claude Code** xây dựng harness](./harness-designs/claude-code/) — bộ nhớ 4 lớp, compaction 5 cấp, hooks, sub-agent
- [Cách **Codex** xây dựng harness](./harness-designs/codex/) — repository là nguồn chân lý, AGENTS.md, worktree isolation
- [Cách **DeepSeek** xây dựng harness](./harness-designs/deepseek/) — "mọi thứ đều là plugin", pipeline sự kiện

### 📚 Tài nguyên dùng ngay

- [Templates](./resources/templates/) — AGENTS.md, feature_list.json, init.sh, session-handoff…
- [Reference](./resources/reference/) — playbook khởi tạo, bản đồ phương pháp, prompt calibration
- [OpenAI Advanced](./resources/openai-advanced/) — repo template & SOPs tiên tiến
- [Skills](./skills/) — kỹ thuật xây harness

## Xem trước khóa học

| Trang chủ | Bài giảng |
| --- | --- |
| ![Trang chủ khóa học](/img/learn-harness/screenshots/en-home.png) | ![Bài giảng](/img/learn-harness/screenshots/en-lecture-01.png) |

## Bắt đầu từ đâu?

1. Đọc [Bài 01](./lectures/lecture-01-why-capable-agents-still-fail/) để hiểu vì sao harness quan trọng hơn mô hình.
2. Làm [Dự án 01](./projects/project-01-baseline-vs-minimal-harness/) để cảm nhận khác biệt bằng chính tay mình.
3. Chép [bộ template tối thiểu](./resources/templates/) (AGENTS.md, feature_list.json, init.sh) vào dự án thật của bạn.
4. Khi đã quen, đọc [Bài 13 Loop Engineering](./lectures/lecture-13-loop-engineering/) và [Bài 14 Graph Engineering](./lectures/lecture-14-graph-engineering/) để nâng cấp lên vòng lặp tự chủ.
