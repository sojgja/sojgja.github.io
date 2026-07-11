---
id: composite
title: Composite
sidebar_label: 🌳 Composite
sidebar_position: 9
---

# Composite

**Composite** tổ chức object thành cấu trúc cây (tree) để biểu diễn quan hệ whole-part, cho phép client xử lý individual object và composition một cách đồng nhất.

## Bài toán

Ứng dụng quản lý file: File và Folder. Folder chứa File và Folder con. Bạn muốn tính tổng dung lượng. Cần duyệt cây, nếu gặp File thì lấy `get_size()`, nếu gặp Folder thì cộng dồn các con. Client phải kiểm tra kiểu mỗi lần — code dài dòng, dễ sai.

## Giải pháp

Composite cho File và Folder cùng một interface `get_size()`. Folder gọi `get_size()` trên từng phần tử con.

```python
from abc import ABC, abstractmethod

class FileSystemNode(ABC):
    @abstractmethod
    def get_size(self): pass

class File(FileSystemNode):
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def get_size(self):
        return self.size

class Folder(FileSystemNode):
    def __init__(self, name):
        self.name = name
        self.children = []

    def add(self, child):
        self.children.append(child)

    def get_size(self):
        return sum(child.get_size() for child in self.children)

# Sử dụng
root = Folder('root')
docs = Folder('documents')
docs.add(File('resume.pdf', 200))
docs.add(File('photo.jpg', 500))

src = Folder('src')
src.add(File('main.py', 50))
src.add(File('utils.py', 30))

root.add(docs)
root.add(src)

print(f'Tổng dung lượng: {root.get_size()} KB')  # 780 KB
```

## Khi nào dùng

- Có cấu trúc cây với whole-part
- Muốn client xử lý đồng nhất leaf và composite
- Cần thao tác đệ quy trên cấu trúc

## Thực tế

- UI component tree (tkinter, PyQt)
- HTML DOM tree
- Menu item hierarchy
