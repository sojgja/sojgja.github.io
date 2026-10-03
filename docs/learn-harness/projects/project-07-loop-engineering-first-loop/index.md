
# Dự án 07. Xây dựng Vòng lặp Tự động Đầu tiên

> Bài giảng liên quan: [Bài 13. Tại sao bạn cần ngừng nhắc lệnh Agent của mình](./../../lectures/lecture-13-loop-engineering/index.md)

## Bạn Sẽ Làm Gì

Đây là dự án chuyển tiếp từ "Harness" sang "Loop." Bạn đã biết cách thiết lập một agent với môi trường, chỉ dẫn, và phản hồi phù hợp — bây giờ bạn sẽ biến thiết lập đó thành một vòng lặp tự chạy.

Bạn sẽ thực hiện ba thí nghiệm tiến triển: đầu tiên biến một nhiệm vụ từ thủ công thành `/goal`, sau đó biến một nhiệm vụ giám sát thành bộ hẹn giờ `/loop`, và cuối cùng xây dựng một vòng lặp maker-checker đầy đủ để trải nghiệm cảm giác khi **bạn bước ra khỏi vòng lặp.**

## Các Tệp Dự án

Đường dẫn repo: [`projects/project-07/`](https://github.com/walkinglabs/learn-harness-engineering/tree/main/projects/project-07)

| Thư mục | Nội dung | Bạn Làm Gì |
|-----------|--------------|-------------|
| [`starter/`](https://github.com/walkinglabs/learn-harness-engineering/tree/main/projects/project-07/starter) | Một dự án cơ sở kiến thức nhỏ với một bộ harness hoàn chỉnh (trạng thái cuối của P06), bao gồm AGENTS.md, feature_list.json, init.sh, session-handoff.md, clean-state-checklist.md. | Biến bộ harness này thành bộ có thể lặp tự động. |
| [`solution/`](https://github.com/walkinglabs/learn-harness-engineering/tree/main/projects/project-07/solution) | Triển khai đầy đủ của ba vòng lặp: goal loop, timer loop, maker-checker loop, cộng với các file trạng thái vòng lặp và các script xác minh. | Tham khảo cho các mẫu thiết kế vòng lặp và quản lý trạng thái. |

## Công cụ Bạn Sẽ Dùng

- Claude Code hoặc Codex
- Git
- Bộ harness hoàn chỉnh của bạn từ P06
- Một trình đa hợp terminal (tmux hoặc screen, để quan sát các vòng lặp chạy lâu)
- Tùy chọn: GitHub Actions hoặc cron (cho các thí nghiệm điều khiển theo sự kiện / theo lịch nâng cao)

## Các Bước

### Chuẩn bị

1. Bắt đầu từ cùng commit nơi bạn đã hoàn thành P06.
2. Tạo ba nhánh: `p07-goal-loop`, `p07-timer-loop`, `p07-maker-checker`.
3. Xác minh harness của bạn hoạt động: chạy init.sh, xác minh file trạng thái, danh sách tính năng, và các tài liệu bàn giao đều đã sẵn sàng.
4. Chọn một **nhiệm vụ mục tiêu** bạn muốn vòng lặp làm việc lặp đi lặp lại. Chọn thứ gì đó cỡ trung bình với các tiêu chí hoàn thành rõ ràng — ví dụ: "thêm unit test cho tất cả các module, đạt 80% độ phủ" hoặc "thêm xác minh đầu vào cho tất cả các endpoint API."

### Thí nghiệm 1: Vòng lặp Mục tiêu — Từ Chạy Thủ công sang Chạy Tự động

Chuyển sang nhánh `p07-goal-loop`.

1. **Viết mô tả mục tiêu**: Biến nhiệm vụ bạn đã chọn thành một file `goal.md` chứa:
   - Mục tiêu rõ ràng ("gì được tính là xong")
   - Phương pháp xác minh ("làm sao để xác nhận nó xong" — chạy test? chạy lint? kiểm tra độ phủ?)
   - Điều kiện dừng ("khi nào nó nên dừng" — lượt tối đa? giới hạn thời gian? giới hạn ngân sách?)
   - Các ràng buộc ("gì không được động" — cấu hình sản xuất, schema cơ sở dữ liệu, v.v.)

2. **Lần chạy thủ công đầu tiên**: Đưa nhiệm vụ cho agent thủ công, bằng chính bạn. Ghi lại nó đã mất bao nhiêu lượt, bạn đã can thiệp bao nhiêu lần, và chất lượng kết quả. Đây là đường cơ sở của bạn.

3. **Chạy với `/goal`**: Sử dụng cùng `goal.md` làm đầu vào và chạy nó ở chế độ `/goal`. Agent tự lặp lại cho đến khi đạt được mục tiêu hoặc điều kiện dừng kích hoạt.

4. **So sánh kết quả**:
   - Sự khác biệt về số lượt
   - Sự khác biệt về số lần can thiệp của bạn
   - Sự khác biệt về chất lượng kết quả (sử dụng cùng tiêu chuẩn xác minh)
   - Sự khác biệt về thời gian bạn đã dành

5. **Lặp lại trên goal.md**: Nếu kết quả kém, sửa lại mô tả mục tiêu và chạy lại. Tiếp tục cho đến khi bạn hài lòng với kết quả, hoặc cho đến khi bạn đã xác nhận giới hạn của những gì một vòng lặp mục tiêu có thể làm trên nhiệm vụ này.

### Thí nghiệm 2: Vòng lặp Bộ hẹn giờ — Biến Giám sát thành Nhịp Tim

Chuyển sang nhánh `p07-timer-loop`.

1. **Chọn một nhiệm vụ giám sát**: Tìm một kiểm tra lặp lại bạn thường làm thủ công. Ví dụ:
   - Chạy bộ test mỗi giờ, sửa lỗi
   - Kiểm tra các bản cập nhật bảo mật phụ thuộc mỗi buổi sáng
   - Kiểm tra các vi phạm phong cách code sau mỗi commit
   - Định kỳ quét các comment TODO để xem cái nào đã cũ

2. **Viết prompt/script giám sát**: Trình bày rõ ràng các bước giám sát — kiểm tra gì, làm gì khi tìm thấy vấn đề, và khi nào gọi con người.

3. **Chạy với `/loop` (hoặc Codex Thread automation)**:
   - Đặt một khoảng thời gian hợp lý (khuyến nghị 10-30 phút — quá ngắn bạn sẽ bị phiền, quá dài bạn sẽ không thấy hiệu ứng)
   - Để nó chạy ít nhất 2 giờ (hoặc đi làm gì đó khác và quay lại sau)

4. **Ghi lại kết quả**:
   - Nó đã tìm thấy bao nhiêu vấn đề?
   - Nó đã tự sửa được bao nhiêu?
   - Bao nhiêu là dương tính giả?
   - Bao nhiêu cái nó làm tồi tệ hơn?
   - Bạn đã dành bao nhiêu thời gian theo dõi kết quả của nó?

5. **Suy ngẫm**: Nhiệm vụ giám sát này có đáng tự động hóa không? So sánh thời gian bạn tiết kiệm so với thời gian bạn dành để theo dõi. Nếu không đáng, bạn đã chọn sai nhiệm vụ, hay vòng lặp được thiết kế kém?

### Thí nghiệm 3: Vòng lặp Maker-Checker — Đưa Bản thân Ra Khỏi Vòng lặp

Chuyển sang nhánh `p07-maker-checker`.

Đây là thí nghiệm quan trọng nhất trong ba thí nghiệm. Bạn sẽ xây dựng một **vòng lặp hoàn chỉnh không cần bạn ở đó:**

1. **Thiết kế cấu trúc vòng lặp**:
   - **Agent Maker**: triển khai, viết code, sửa đổi tệp
   - **Agent Checker**: xác minh, chạy test, thực hiện review code, đạt / thất bại
   - **File trạng thái** (`loop-state.md`): ghi lại vòng hiện tại, gì đã làm, kết quả xác minh, gì tiếp theo
   - **Điều kiện dừng**: N lần đạt liên tiếp, hoặc đạt số vòng tối đa

2. **Viết ba prompt**:
   - Hướng dẫn của Maker (phải làm gì, làm sao để làm, gì không được động)
   - Hướng dẫn của Checker (phải xác minh gì, làm sao để xác minh, gì được tính là đạt, làm sao để đưa ra phản hồi)
   - Logic điều khiển vòng lặp (ai đi trước, bàn giao hoạt động như thế nào, làm sao để bắt đầu vòng tiếp theo)

3. **Chạy ít nhất 5 vòng**:
   - Vòng 1: Maker triển khai → Checker xác minh → Thất bại → Phản hồi cho Maker
   - Vòng 2: Maker sửa đổi dựa trên phản hồi → Checker xác minh → ...
   - ...
   - Cho đến khi đạt liên tiếp, hoặc bạn dừng lại

4. **Ghi lại trạng thái của mỗi vòng**:
   - Số thứ tự vòng
   - Maker đã làm gì
   - Checker đã tìm thấy những vấn đề gì
   - Đạt / thất bại
   - Bạn có can thiệp không? (nếu có, tại sao?)

5. **Tổng kết cuối cùng**:
   - Bạn đã can thiệp bao nhiêu lần? Tại sao?
   - Điều gì sẽ xảy ra nếu bạn không can thiệp?
   - Checker có bỏ lỡ bất kỳ vấn đề nào không?
   - Maker có tiếp tục mắc cùng một lỗi không?
   - Trần chất lượng của vòng lặp này ở đâu? Năng lực của Maker, hay năng lực của Checker?

## Cách Đo lường Kết quả

| Chỉ số | Thí nghiệm 1 (Mục tiêu) | Thí nghiệm 2 (Bộ hẹn giờ) | Thí nghiệm 3 (Maker-Checker) |
|--------|-------------|--------------|----------------------|
| Tỷ lệ hoàn thành nhiệm vụ | Mục tiêu có đạt được không? | Bao nhiêu chu kỳ giám sát đã chạy? | Bao nhiêu vòng cho đến khi đạt? |
| Sự can thiệp của con người | Bạn đã can thiệp bao nhiêu lần? | Bạn đã dành bao nhiêu thời gian theo dõi? | Bạn đã can thiệp bao nhiêu lần? |
| Chất lượng kết quả | Nó so với thủ công như thế nào? | Tỷ lệ dương tính giả? Bỏ lỡ vấn đề? | Checker đã tìm thấy bao nhiêu vấn đề mà bạn sẽ không tìm thấy? |
| Thời gian tiết kiệm | Bạn đã tiết kiệm được bao nhiêu thời gian? | Có đáng để tự động hóa không? | Thời gian dành để thiết kế vòng lặp so với thời gian tiết kiệm |
| Độ tin cậy | Điều kiện dừng có đáng tin không? | Nó có chạy失控 không? | Vòng lặp có thể bị kẹt ở cùng một chỗ không? |

## Nộp Gì

- `goal.md` (mô tả mục tiêu của Thí nghiệm 1, ít nhất hai lần lặp)
- Ghi chú so sánh Thí nghiệm 1: thủ công so với vòng lặp mục tiêu
- Prompt giám sát Thí nghiệm 2 + nhật ký chạy 2 giờ
- Ba prompt của Thí nghiệm 3 (Maker / Checker / Điều khiển vòng lặp)
- `loop-state.md` của Thí nghiệm 3 (ít nhất 5 vòng được ghi lại)
- Tổng kết cuối cùng: những bài học rút ra từ cả ba thí nghiệm, sự hiểu biết của bạn về loop engineering đã thay đổi như thế nào, những thứ gì là ứng viên tốt để vòng lặp hóa và những thứ gì không

## Các Bài giảng Liên quan

- [Bài 13 — Tại sao bạn cần ngừng nhắc lệnh Agent của mình](../../lectures/lecture-13-loop-engineering/index.md)
- [Bài 12 — Tại sao mỗi phiên phải để lại trạng thái sạch](../../lectures/lecture-12-why-every-session-must-leave-a-clean-state/index.md) (mỗi vòng của một vòng lặp cần trạng thái sạch)
- [Bài 11 — Tại sao tính quan sát được thuộc về bên trong Harness](../../lectures/lecture-11-why-observability-belongs-inside-the-harness/index.md) (bạn cần thấy những gì đang xảy ra bên trong vòng lặp)
- [Bài 05 — Tại sao các file trạng thái là xương sống của tính liên tục](../../lectures/lecture-05-why-long-running-tasks-lose-continuity/index.md) (các file trạng thái vòng lặp là một phần mở rộng của các file trạng thái)
