---
id: big-o
title: Time Complexity & Big O
sidebar_label: ⏱️ Big O Notation
sidebar_position: 2
---

# Time Complexity & Big O

Đo lường hiệu suất thuật toán theo kích thước đầu vào.

## Các độ phức tạp thường gặp

| Ký hiệu | Tên | Ví dụ |
|---------|-----|-------|
| O(1) | Hằng số | Truy cập mảng theo index |
| O(log n) | Logarithmic | Binary search |
| O(n) | Tuyến tính | Duyệt mảng |
| O(n log n) | Linearithmic | Merge sort, Quick sort |
| O(n²) | Bậc hai | Bubble sort, nested loop |
| O(2ⁿ) | Exponential | Fibonacci đệ quy |

## Quy tắc

- Bỏ hằng số: O(2n) → O(n)
- Lấy cấp cao nhất: O(n² + n) → O(n²)
- Luôn tính cho worst case
