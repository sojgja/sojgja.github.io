---
id: git-basics
title: Git Basics
sidebar_label: 🐙 Git Basics
sidebar_position: 2
---

# Git Basics

## Cấu hình

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

## Workflow cơ bản

```bash
git init                    # Khởi tạo repo
git add .                   # Stage tất cả
git commit -m "message"     # Commit
git push origin main        # Push lên remote
git pull                    # Pull từ remote
```

## Branching

```bash
git branch feature-x        # Tạo branch mới
git checkout feature-x      # Chuyển branch
git checkout -b feature-x   # Tạo + chuyển
git merge feature-x         # Merge vào branch hiện tại
```

## Những lệnh hữu ích

```bash
git status          # Xem trạng thái
git log --oneline   # Xem lịch sử commit
git diff            # Xem thay đổi
git stash           # Tạm cất thay đổi
git reset HEAD~1    # Undo commit gần nhất
```
