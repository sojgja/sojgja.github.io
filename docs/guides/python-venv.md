---
id: python-venv
title: Python Virtual Environment
sidebar_label: 🐍 Python Virtual Environment
sidebar_position: 3
---

# Python Virtual Environment

## Tạo môi trường ảo

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

## Quản lý packages

```bash
pip install requests
pip freeze > requirements.txt
pip install -r requirements.txt
pip list
pip uninstall requests
```

## Vì sao cần?

- Cô lập dependencies giữa các dự án
- Tránh xung đột version
- Dễ tái tạo môi trường (requirements.txt)
- Không ảnh hưởng đến hệ thống Python global
