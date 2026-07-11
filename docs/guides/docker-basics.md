---
id: docker-basics
title: Docker Basics
sidebar_label: 🐳 Docker Basics
sidebar_position: 4
---

# Docker Basics

## Khái niệm

- **Image** — template chỉ đọc để tạo container
- **Container** — instance chạy từ image
- **Dockerfile** — script để build image
- **Volume** — lưu dữ liệu persistent
- **Network** — kết nối các container

## Các lệnh cơ bản

```bash
docker build -t my-app .    # Build image
docker run -p 3000:3000 my-app  # Chạy container
docker ps                   # Xem container đang chạy
docker stop <id>            # Dừng container
docker images               # Xem danh sách images
docker compose up           # Chạy multi-container
```

## Dockerfile mẫu

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```
