---
id: 09-skills
title: "Skills (tự học)"
sidebar_label: "Skills (tự học)"
sidebar_position: 9
---

# Skills (tự học)

Skill là cách **ưu tiên** để thêm năng lực cho Hermes: dễ tạo hơn tool, không
cần sửa code agent, và chia sẻ được. Đây cũng là trái tim của vòng học khép kín — Hermes tự
đúc kết skill từ kinh nghiệm và tự cải thiện chúng.

## 1. Skill hay Tool? {#skill-hay-tool}

**Làm Skill khi…**

- Năng lực diễn đạt được bằng hướng dẫn + lệnh shell + tool sẵn có
- Bọc một CLI/API mà agent gọi qua `terminal` hoặc `web_extract`
- Không cần tích hợp Python hay quản API key trong agent
- Ví dụ: tìm arXiv, git workflow, quản Docker, xử lý PDF

**Làm Tool khi…**

- Cần tích hợp đầu-cuối với API key, luồng auth, cấu hình nhiều thành phần
- Logic tuỳ biến phải chạy *chính xác* mỗi lần
- Xử lý dữ liệu nhị phân, streaming, sự kiện realtime
- Ví dụ: browser automation, TTS, phân tích ảnh

## 2. Cấu trúc một skill {#cau-truc}

Skill đóng gói sẵn nằm trong `skills/` theo chủ đề; skill tuỳ chọn cùng cấu trúc trong `optional-skills/`:

```
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md            # BẮT BUỘC — hướng dẫn chính
│       └── scripts/            # tuỳ chọn — script hỗ trợ
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

Chuẩn mở: [agentskills.io](https://agentskills.io). Skill bạn viết ở đây dùng
cùng spec với các skill của hệ sinh thái soigia.

## 3. Định dạng SKILL.md {#skill-md}

Một skill tối thiểu là một file markdown với frontmatter mô tả khi nào dùng:

```
---
name: arxiv-search
description: Tìm & tóm tắt bài báo arXiv theo truy vấn.
---

# arXiv Search

## Khi nào dùng
Khi cần tìm bài báo khoa học theo chủ đề.

## Các bước
1. Gọi `scripts/search_arxiv.py "<query>"`
2. Đọc kết quả, tóm tắt 3 bài hàng đầu
...
```

> **Mẹo.** `description` là thứ agent dùng để quyết định nạp skill — viết nó như một dấu hiệu kích hoạt rõ ràng, kèm từ khoá người dùng sẽ gõ.

## 4. Quản lý skill {#quan-ly}

```
hermes skills                 # duyệt, cài, publish, audit, cấu hình
hermes skills list            # (trong chat: /skills list)
hermes skills opt-in --sync   # seed skill vào chế độ cấu hình hiện tại
/skill <name>                # gọi skill bị trùng tên built-in
```

| Việc | Cách |
| --- | --- |
| Xem skill có sẵn | `/skills list` trong chat, hoặc `hermes skills` |
| Cài skill từ hub | `hermes skills` (browse/install) |
| Audit skill | `hermes skills` (audit) |
| Publish skill của bạn | `hermes skills` (publish) |

Tham chiếu catalog: `reference/skills-catalog.md` (skill đóng gói) và
`reference/optional-skills-catalog.md` (tuỳ chọn).

## 5. Vòng học khép kín {#tu-hoc}

Đây là điểm khác biệt cốt lõi của Hermes so với các agent khác:

1. **Tạo skill:** sau một task phức tạp, Hermes tự đúc kết thành skill.
2. **Skill tự cải thiện:** trong lúc dùng, skill được tinh chỉnh.
3. **Nhắc lưu tri thức:** agent tự nhắc mình persist kiến thức.
4. **Tìm lại quá khứ:** search chính các hội thoại cũ (FTS5 + tóm tắt LLM).
5. **Mô hình hoá người dùng:** tích hợp [Honcho](https://github.com/plastic-labs/honcho) dialectic.

Bạn có thể chủ động kích hoạt vòng này trong chat bằng `/refine [focus]` — ví dụ
`/refine lưu quy trình deploy này thành skill`.

## 6. Tạo & chia sẻ skill {#tao}

**Quy trình tối thiểu**

1. Tạo thư mục `<chủ-đề>/<tên-skill>/`
2. Viết `SKILL.md` với frontmatter `name` + `description`
3. Thêm `scripts/` hoặc `references/` nếu cần
4. Thử bằng `hermes chat` rồi gọi skill

**Nguyên tắc chất lượng**

- Một skill giải một việc rõ ràng
- Hướng dẫn theo bước, có lệnh copy-paste được
- Ghi rõ khi nào KHÔNG dùng skill
- Không nhúng secret vào skill

> **Nguồn.** `website/docs/developer-guide/creating-skills.md`, `guides/work-with-skills.md`, `reference/skills-catalog.md`.
