---
id: template-method
title: Template Method
sidebar_label: 📋 Template Method
sidebar_position: 23
---

# Template Method

**Template Method** định nghĩa khung (skeleton) của một thuật toán trong method, để subclass cài đặt các bước chi tiết mà không thay đổi cấu trúc thuật toán.

## Bài toán

Ứng dụng **build CI/CD** có quy trình chung: `checkout → build → test → deploy`. Nhưng mỗi loại project khác nhau:
- Python: `pip install` → `pytest` → `deploy to PyPI`
- Node: `npm install` → `npm test` → `deploy to npm`
- Java: `mvn compile` → `mvn test` → `deploy to Maven`

Nếu viết riêng từng pipeline, code sẽ lặp lại cấu trúc giống nhau.

## Giải pháp

Template Method đặt cấu trúc pipeline trong base class, để subclass override từng bước.

```python
from abc import ABC, abstractmethod

class Pipeline(ABC):
    def run(self):
        """Template method — khung cố định"""
        self.checkout()
        self.install_deps()
        self.build()
        self.test()
        self.deploy()

    def checkout(self):
        print('📥 Checkout source code...')

    # Các bước có thể override
    def install_deps(self): pass
    def build(self): pass

    @abstractmethod
    def test(self): pass

    @abstractmethod
    def deploy(self): pass

class PythonPipeline(Pipeline):
    def install_deps(self):
        print('📦 pip install -r requirements.txt')

    def build(self):
        print('🔨 python setup.py build')

    def test(self):
        print('🧪 pytest --cov .')

    def deploy(self):
        print('🚀 twine upload dist/*')

class NodePipeline(Pipeline):
    def install_deps(self):
        print('📦 npm install')

    def test(self):
        print('🧪 npm test')

    def deploy(self):
        print('🚀 npm publish')

# Sử dụng
print('=== Python Pipeline ===')
PythonPipeline().run()

print('\n=== Node Pipeline ===')
NodePipeline().run()
```

## Khi nào dùng

- Các class có thuật toán giống nhau, chỉ khác vài bước
- Muốn tránh duplicate code
- Muốn kiểm soát cấu trúc thuật toán từ base class

## Thực tế

- Django class-based views: `get()`, `post()`, `dispatch()`
- Python `threading.Thread`: `run()` là template method
- ORM `save()` lifecycle: `pre_save → save → post_save`
- Quy trình xử lý request trong framework web
