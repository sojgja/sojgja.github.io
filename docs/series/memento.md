---
id: memento
title: Memento
sidebar_label: 💾 Memento
sidebar_position: 19
---

# Memento

**Memento** cho phép lưu và khôi phục trạng thái trước đó của object mà không vi phạm encapsulation.

## Bài toán

Trình soạn thảo văn bản có tính năng **Undo**. Mỗi lần gõ chữ, bạn muốn lưu snapshot để có thể quay lại. Nếu lưu toàn bộ state của Editor ra ngoài (public properties), bạn phá vỡ encapsulation. Còn nếu giữ state bên trong, làm sao để external code (History) lưu và khôi phục được?

## Giải pháp

Memento là "snapshot" bất biến của state. Chỉ Editor mới tạo và khôi phục được Memento.

```python
class Memento:
    def __init__(self, state: str):
        self._state = state  # private — chỉ Editor truy cập

    def get_state(self):
        return self._state

class Editor:
    def __init__(self):
        self._content = ''

    def type(self, text: str):
        self._content += text

    def save(self) -> Memento:
        return Memento(self._content)

    def restore(self, memento: Memento):
        self._content = memento.get_state()

    def get_content(self):
        return self._content

class History:
    def __init__(self):
        self._snapshots = []

    def push(self, memento: Memento):
        self._snapshots.append(memento)

    def pop(self) -> Memento:
        return self._snapshots.pop()

# Sử dụng
editor = Editor()
history = History()

editor.type('Hello ')
history.push(editor.save())

editor.type('World!')
history.push(editor.save())

editor.type(' Hôm nay là thứ 2.')

print(editor.get_content())  # Hello World! Hôm nay là thứ 2.

editor.restore(history.pop())
print(editor.get_content())  # Hello World!

editor.restore(history.pop())
print(editor.get_content())  # Hello
```

## Khi nào dùng

- Cần undo/redo
- Cần snapshot để rollback
- Không muốn phá vỡ encapsulation

## Thực tế

- Ctrl+Z trong mọi text editor/IDE
- Git commit (memento của toàn bộ project)
- Database transaction rollback (BEGIN → SAVEPOINT → ROLLBACK)
