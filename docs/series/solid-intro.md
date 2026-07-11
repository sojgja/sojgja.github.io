---
id: solid-intro
title: SOLID — 5 nguyên lý thiết kế OOP
sidebar_label: 📐 Giới thiệu SOLID
sidebar_position: 25
---

# SOLID — 5 nguyên lý thiết kế OOP

**SOLID** là 5 nguyên lý thiết kế hướng đối tượng do **Robert C. Martin** (Uncle Bob) giới thiệu, giúp code dễ bảo trì, mở rộng và linh hoạt.

## 5 nguyên lý

| Chữ | Nguyên lý | Ý nghĩa |
|-----|-----------|---------|
| **S** | **Single Responsibility** | Một class chỉ nên có một lý do để thay đổi |
| **O** | **Open/Closed** | Mở cho mở rộng, đóng cho sửa đổi |
| **L** | **Liskov Substitution** | Subclass phải thay thế được base class |
| **I** | **Interface Segregation** | Nhiều interface nhỏ tốt hơn một interface lớn |
| **D** | **Dependency Inversion** | Phụ thuộc vào abstraction, không phụ thuộc vào concrete |

## Tại sao SOLID quan trọng?

- **Dễ bảo trì**: Thay đổi một chỗ không ảnh hưởng toàn bộ
- **Dễ test**: Class nhỏ, rõ trách nhiệm → dễ viết unit test
- **Dễ mở rộng**: Thêm tính năng mới mà không sửa code cũ
- **Giảm coupling**: Các module ít phụ thuộc lẫn nhau

Các bài tiếp theo sẽ đi sâu vào từng nguyên lý với ví dụ thực tế bằng Python.
