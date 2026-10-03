# Dự án 08. Vẽ Quy trình Làm việc của bạn thành một Đồ thị

> Bài giảng liên quan: [L14. Từ Vòng lặp Đơn lẻ đến Kỹ thuật Đồ thị](./../../lectures/lecture-14-graph-engineering/index.md)

## Bạn sẽ làm gì

Đây là dự án nhảy vọt từ "Loop" sang "Graph". Ở bài giảng trước bạn đã dựng một maker-checker loop — triển khai, xác minh, phản hồi, triển khai lại, mọi quyết định đều xảy ra trong cửa sổ ngữ cảnh của cùng một agent. Nhiệm vụ của bạn trong dự án này là **vẽ ra tường minh cấu trúc giấu bên trong vòng lặp**: nút, cạnh, trạng thái dùng chung, quy tắc định tuyến, viết rõ từng chữ một.

Bạn sẽ làm ba thí nghiệm tiến dần: trước tiên vẽ maker-checker loop của P07 thành một đồ thị tường minh, rồi thêm cho đồ thị một nút fan-out/fan-in song song, cuối cùng thêm một cạnh quay lại có điều kiện và một nút phê duyệt thủ công. Làm xong bạn sẽ tự mình cảm nhận một điều: **đồ thị không phải phát minh mới, nó là thứ vòng lặp tự trở thành khi loop của bạn phức tạp đến một mức độ nào đó.**

## Dùng công cụ gì

- Claude Code hoặc Codex
- Git
- maker-checker loop bạn đã dựng ở P07 (hoặc bất kỳ quy trình làm việc agent nào bạn có thể chạy đi chạy lại)
- Một trình soạn thảo văn bản hoặc công cụ vẽ (vẽ không phải để đẹp, mà để viết rõ cấu trúc; `mermaid` hoặc viết tay `graph.md` đều được)

## Các bước cụ thể

### Công tác chuẩn bị

1. Bắt đầu từ kho mã sau khi hoàn thành P07, hoặc trực tiếp dùng bất kỳ quy trình làm việc agent nào bạn đang chạy.
2. Tạo ba nhánh: `p08-explicit-graph`, `p08-parallel`, `p08-human-in-the-loop`.
3. Chuẩn bị một `state.md` làm file trạng thái dùng chung: yêu cầu, tiến độ, kết quả xác minh đều ghi ở đây. Đây là "bàn làm việc chung" của đồ thị.

### Thí nghiệm một: Vẽ Loop thành đồ thị tường minh

Chuyển sang nhánh `p08-explicit-graph`.

1. **Liệt kê tất cả các nút:** viết mỗi bước trong maker-checker loop của P07 thành một nút. Mỗi nút viết rõ: trách nhiệm của nó, đầu vào của nó, đầu ra của nó, nó là agent hay mã xác định.
2. **Vẽ tất cả các cạnh:** liệt kê từng cạnh giữa các nút. Nhấn mạnh đánh dấu hai cạnh đặc biệt:
   - Cạnh điều kiện: xác minh đạt/thất bại, đi theo hướng nào
   - Cạnh quay lại: thất bại quay về nút nào
3. **Viết trạng thái dùng chung:** liệt kê rõ trạng thái có những trường nào (yêu cầu, code, kết quả test, kết luận đánh giá), ai đọc ai ghi.
4. **Viết quy tắc định tuyến:** dùng ngôn ngữ if-then đơn giản nhất viết quy tắc "bước tiếp theo đi đâu", ví dụ:
   ```
   if xác minh đạt → nút hợp nhất
   if xác minh thất bại → nút triển khai
   if nút triển khai thông tin không đủ → nút nghiên cứu
   ```
5. **Viết thành `graph.md`:** tổng hợp các nội dung trên thành một tài liệu. Dùng mermaid vẽ một đồ thị, kèm bảng nút và quy tắc định tuyến.
6. **Trả lời câu hỏi này:** sau khi vẽ xong, tìm ra ít nhất một **cạnh vốn là ẩn** — đường quyết định trước đây giấu trong ngữ cảnh agent, đến chính bạn cũng không biết nó tồn tại.

### Thí nghiệm hai: Thêm một nút Fan-out / Fan-in song song

Chuyển sang nhánh `p08-parallel`.

1. **Chọn một điểm có thể song song:** tìm một chỗ trong nhiệm vụ có thể tách thành hai phần độc lập. Ví dụ:
   - Tách triển khai thành hai module độc lập, hai agent viết song song
   - Tách xác minh thành hai cuộc đánh giá độc lập: một chạy test và lint, một làm đánh giá code (chỉ dẫn khác nhau, mối quan tâm khác nhau)
   - Tách nghiên cứu thành hai hướng, hai agent mỗi cái tra một đường
2. **Viết quy tắc fan-out:** ghi trong trạng thái dùng chung "nhiệm vụ này được tách thành N tác vụ con song song", mỗi tác vụ con một context độc lập, một nút độc lập.
3. **Viết quy tắc fan-in:** sau khi tất cả tác vụ con hoàn thành, ai hợp nhất kết quả? Tiêu chuẩn hợp nhất là gì (ví dụ: hai cuộc đánh giá đều đạt mới hợp nhất, hay có một đạt là được)?
4. **Dùng worktree để cô lập:** mỗi tác vụ con song song chạy trong một git worktree độc lập, tránh va chạm file về mặt vật lý (ôn lại nguyên thủy Worktree ở bài 13).
5. **Chạy một lần và ghi lại:** ghi thời gian wall-clock trước/sau khi song song, mức tiêu thụ token, chất lượng kết quả. Song song có thực sự nhanh hơn không? Hay chi phí điều phối ăn mất thời gian tiết kiệm được?

### Thí nghiệm ba: Thêm một cạnh quay lại và một nút phê duyệt thủ công

Chuyển sang nhánh `p08-human-in-the-loop`.

Đây là thí nghiệm quan trọng nhất trong ba thí nghiệm. Bạn phải thêm hai loại nút vào đồ thị:

1. **Cạnh quay lại có điều kiện:** thêm cho nút xác minh một đường "đạt một phần" — không phải đánh trả toàn bộ về nút triển khai, mà mang theo phản hồi cụ thể quay về **nút đã tạo ra vấn đề**. Ví dụ: test đều đạt nhưng đánh giá code phát hiện hiểu sai yêu cầu, quay lại nút nghiên cứu chứ không phải nút triển khai. Điều này đòi hỏi trạng thái dùng chung của bạn ghi "vấn đề nằm ở tầng nào".
2. **Nút phê duyệt thủ công (Human-in-the-loop):** thêm một nút người trước nút hợp nhất. Đi đến đây, đồ thị **dừng lại**, chờ bạn ghi "phê duyệt" hoặc "đánh trả" trong `state.md`. Nút phê duyệt có thể có quy tắc hết giờ: sau N giờ không phản hồi, tự động đánh trả hoặc tự động nâng cấp.
3. **Viết định dạng interrupt:** viết yêu cầu phê duyệt rõ ràng như thế nào — chuyện gì đã xảy ra, đã thay đổi gì, vì sao cần con người, hậu quả của phê duyệt/đánh trả mỗi loại là gì.
4. **Chạy ít nhất 2 vòng quy trình đầy đủ:** mỗi vòng đều đi đến nút phê duyệt thủ công, bạn tự phê duyệt hoặc đánh trả một lần. Ghi lại: quyết định phê duyệt của bạn có khớp với phán đoán của nút xác minh không? Nút phê duyệt đã chặn được thứ gì mà nút xác minh không chặn được không?

## Cách đo lường kết quả

| Chỉ số | Thí nghiệm một (đồ thị tường minh) | Thí nghiệm hai (song song) | Thí nghiệm ba (người-máy cộng tác) |
|------|----------------|--------------|------------------|
| Tính nhìn thấy của cấu trúc | Tìm ra được mấy cạnh ẩn? | Trạng thái dùng chung có chịu được các tác vụ con song song không? | Cạnh quay lại có định vị chính xác tầng của vấn đề không? |
| Định vị thất bại | Khi thất bại có thể chỉ thẳng cạnh nào sai không? | Khi tác vụ con song song thất bại, định vị được cái nào? | Khi phê duyệt đánh trả, chỉ ra được vấn đề ở tầng nào không? |
| Chi phí cộng tác | Vẽ đồ thị mất bao lâu? | Thời gian song song tiết kiệm vs chi phí điều phối | Thời gian chờ phê duyệt vs giá trị của vấn đề chặn được |
| Tính quan sát | Mỗi bước xảy ra gì, giờ nhìn thấy được chưa? | Trạng thái của mỗi tác vụ con song song có nhìn thấy không? | Yêu cầu phê duyệt viết đủ rõ không? |
| Độ tin cậy | Mô tả đồ thị và chạy thực tế có khớp không? | Tiêu chuẩn hợp nhất fan-in có đáng tin không? | Quy tắc hết giờ/nâng cấp có thực sự kích hoạt không? |

## Cần nộp gì

- `graph.md` (mô tả đồ thị đầy đủ của thí nghiệm một: đồ thị mermaid + bảng nút + bảng cạnh + trường trạng thái dùng chung + quy tắc định tuyến)
- Danh sách các cạnh ẩn tìm được ở thí nghiệm một (ít nhất một cái)
- Quy tắc fan-out/fan-in của thí nghiệm hai và một bản ghi chạy song song (so sánh thời gian/chi phí/chất lượng)
- Quy tắc cạnh quay lại, định dạng nút phê duyệt và bản ghi 2 vòng người-máy cộng tác của thí nghiệm ba
- Tổng kết cuối: từ loop đến graph, cách làm việc của bạn đã thay đổi gì? Nhiệm vụ nào đáng vẽ đồ thị, nhiệm vụ nào không?

## Bài giảng tương ứng

- [Lecture 14 — Từ Vòng lặp Đơn lẻ đến Kỹ thuật Đồ thị](../../lectures/lecture-14-graph-engineering/index.md)
- [Lecture 13 — Từ nhắc lệnh thủ công đến vòng lặp tự chủ](../../lectures/lecture-13-loop-engineering/index.md) (loop của bạn chính là một nút trong đồ thị; dự án này là trải phẳng cấu trúc bên trong của nút)
- [Lecture 09 — Tại sao agent tuyên bố hoàn thành quá sớm](../../lectures/lecture-09-why-agents-declare-victory-too-early/index.md) (vì sao nút xác minh phải độc lập với nút triển khai, trong đồ thị là vấn đề cấu trúc)
- [Lecture 11 — Tại sao tính quan sát thuộc về bên trong harness](../../lectures/lecture-11-why-observability-belongs-inside-the-harness/index.md) (đồ thị càng phức tạp, càng cần nhìn thấy mỗi nút đang làm gì)