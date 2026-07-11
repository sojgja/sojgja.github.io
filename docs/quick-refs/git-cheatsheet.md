---
id: git-cheatsheet
title: Git Cheatsheet
sidebar_label: 🐙 Git Cheatsheet
sidebar_position: 2
---

# Git Cheatsheet

## Config

```bash
git config --global user.name "name"
git config --global user.email "email"
```

## Branch

```bash
git branch -a              # List all
git checkout -b <name>     # Create + switch
git branch -d <name>       # Delete local
git push origin --delete <name>  # Delete remote
```

## Commit

```bash
git add -p                 # Stage từng phần
git commit --amend         # Sửa commit gần nhất
git reset HEAD~1           # Undo commit (giữ changes)
git reset --hard HEAD~1    # Undo commit (xóa changes)
```

## Stash

```bash
git stash                  # Cất tạm
git stash pop              # Lấy lại
git stash list
```

## Merge / Rebase

```bash
git merge <branch>
git rebase <branch>
```

## Log

```bash
git log --oneline --graph
git log --oneline -5
```
