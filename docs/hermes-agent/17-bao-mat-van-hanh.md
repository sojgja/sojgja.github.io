---
id: 17-bao-mat-van-hanh
title: "Bảo mật, vận hành & FAQ"
sidebar_label: "Bảo mật, vận hành & FAQ"
sidebar_position: 17
---

# Bảo mật, vận hành & FAQ

Trang cuối: giữ an toàn cho một agent có quyền shell, vận hành nó mỗi ngày, và trả lời những
câu hỏi hay gặp nhất.

## 1. Mô hình bảo mật {#bao-mat}

Ba lớp chính:

| Lớp | Bảo vệ |
| --- | --- |
| Phê duyệt lệnh | Hỏi trước lệnh nguy hiểm; allowlist dần theo lịch sử |
| Cách ly backend | Chạy tool trên Docker/SSH/Modal/Singularity/Vercel thay vì máy thật |
| Redaction secret | Che chuỗi giống key/token trước khi vào context và log |

## 2. Secret & redaction {#secret}

**Secret redaction bật mặc định**: output của tool (stdout terminal,
`read_file`, nội dung web, tóm tắt subagent…) được quét các chuỗi giống API key,
token, secret trước khi vào context và log.

| Cách lưu secret | Khi nào dùng |
| --- | --- |
| `~/.hermes/.env` | Mặc định — key/token provider và nền tảng |
| `hermes secrets` | Nguồn secret ngoài (Bitwarden Secrets Manager), kéo key lúc khởi động thay vì để trong file |
| `hermes auth` | Quản credential & OAuth tập trung |

> **Nguyên tắc.** Không dán key vào prompt hay file skill. Không commit `.env`. Dashboard lưu API key nên đừng để nó lộ ra mạng. Sau khi nghi ngờ lộ, hãy **rotate** key.

## 3. Phê duyệt & allowlist {#quyen}

```
hermes approvals      # khai thác lịch sử phê duyệt → đề xuất allowlist
hermes hooks          # xem/phê duyệt/xoá shell hook trong config.yaml
hermes pairing        # chấp thuận/thu hồi mã ghép đôi nền tảng
```

`--yolo` bỏ toàn bộ phê duyệt — chỉ dùng trong sandbox cách ly.

## 4. Chạy trên máy công việc {#may-cong-viec}

Có hướng dẫn riêng: `guides/secure-hermes-on-a-work-machine.md`. Tóm tắt:

- Dùng backend cách ly thay vì chạy trực tiếp trên máy có dữ liệu nhạy cảm.
- Giữ allowlist chặt; bật pairing cho mọi nền tảng nhắn tin.
- Không bật API server/dashboard ra mạng không xác thực.
- Sao lưu định kỳ nhưng mã hoá phần chứa secret.

## 5. Vận hành & giám sát {#van-hanh}

| Việc | Lệnh |
| --- | --- |
| Trạng thái tổng | `hermes status` |
| Chẩn đoán config/phụ thuộc | `hermes doctor` |
| Log | `hermes logs` |
| Hạn mức tài khoản | `hermes usage` (`--json` cho script) |
| Kích thước system prompt | `hermes prompt-size` (chạy offline) |
| Giám sát gateway | `developer-guide/gateway-monitoring.md` |
| Audit bảo mật | `hermes security audit` |
| Dừng khẩn cấp | `hermes pause` / `hermes resume` |

## 6. Bảng xử lý sự cố {#su-co}

| Triệu chứng | Nguyên nhân thường gặp | Xử lý |
| --- | --- | --- |
| Chat không trả lời | Provider/key sai, endpoint chết | `hermes doctor`, `hermes status`, `hermes model` |
| "Kết nối được mà không chạy" với model cục bộ | Sai tên model hoặc context length | Xác minh endpoint + tên model + context length |
| Agent bỏ qua `CLAUDE.md`/`AGENTS.md` | Bị shadow, truncate, hay install-tree guard | `/context` xem trạng thái từng context file |
| Context phình nhanh | Toolset/MCP lộ quá nhiều schema | `/context all`, tắt toolset thừa, lọc MCP |
| Gateway không nhận tin | Token/allowlist/pairing | `hermes status`, `hermes pairing`, log gateway |
| Cron không nổ | Đang `pause`, hoặc prompt không tự chứa | `hermes resume`; `guides/cron-troubleshooting.md` |
| DB lỗi/không mở được | WAL không hợp filesystem | Đặt `database.journal_mode: delete`; `state-db-recovery.md` |
| Cần gửi hỗ trợ | — | `hermes dump` + `hermes debug` |

## 7. FAQ {#faq}

| Câu hỏi | Trả lời |
| --- | --- |
| Cài bằng `pip install`/`brew` được không? | Không — đây là cách cài không được hỗ trợ. Dùng script/bundle/Docker/Nix. |
| Chạy model cục bộ miễn phí được không? | Được — Ollama/LM Studio hoặc endpoint OpenAI-compatible (vLLM, SGLang). |
| Đổi model có phải sửa code? | Không — `hermes model`, cả ba API mode do lớp provider lo. |
| Cập nhật Docker thế nào? | Chạy image mới; Docker không hỗ trợ `hermes update`. |
| Hermes có nhớ giữa các phiên không? | Có — SQLite + memory + session search; theo profile. |
| Skill khác tool thế nào? | Skill = hướng dẫn markdown, không sửa core; tool = code cần chính xác/auth. |
| Làm sao để nhiều bot độc lập trên một máy? | Dùng profile (`hermes profile`). |

## 8. Giới hạn & lưu ý {#gioi-han}

- Delegation cấp cao là bất đồng bộ nhưng *trong tiến trình* — không sống qua khởi động lại; cần bền thì dùng cron/terminal background.
- Subagent không dùng được `clarify` (không tương tác user).
- Con của `delegate_task` không sở hữu task Kanban.
- `hermes-agent` (runner legacy) chỉ chạy một truy vấn rồi thoát.

> **Nguồn.** `website/docs/guides/secure-hermes-on-a-work-machine.md`, `guides/troubleshooting-agent-quality.md`, `reference/faq.md`, `user-guide/*`, `developer-guide/state-db-recovery.md`.
