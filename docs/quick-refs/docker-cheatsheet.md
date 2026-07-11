---
id: docker-cheatsheet
title: Docker Cheatsheet
sidebar_label: 🐳 Docker Cheatsheet
sidebar_position: 3
---

# Docker Cheatsheet

## Container

```bash
docker ps                    # Đang chạy
docker ps -a                 # Tất cả
docker start <id>
docker stop <id>
docker rm <id>
docker logs -f <id>          # Theo dõi log
docker exec -it <id> sh     # Vào container
```

## Image

```bash
docker images
docker rmi <image>
docker build -t <name> .
docker pull <image>
```

## Compose

```bash
docker compose up -d
docker compose down
docker compose logs -f
docker compose restart
```

## Volume / Network

```bash
docker volume ls
docker network ls
docker network connect <net> <container>
```

## Clean up

```bash
docker system prune -a       # Xoá tất cả unused
docker container prune
docker image prune
```
