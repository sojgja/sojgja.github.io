---
id: bash-cheatsheet
title: Bash Cheatsheet
sidebar_label: 💻 Bash Cheatsheet
sidebar_position: 5
---

# Bash Cheatsheet

## File operations

```bash
ls -la                    # List all
cp -r src dst             # Copy recursive
mv src dst                # Move / rename
rm -rf dir                # Delete recursive
find . -name "*.py"       # Find files
grep -r "pattern" .       # Search content
```

## Permissions

```bash
chmod +x script.sh
chown user:group file
```

## Process

```bash
ps aux                    # List processes
kill -9 <pid>             # Force kill
top / htop                # Monitor
```

## Network

```bash
curl -X GET http://...
netstat -tulpn
ping host
```

## Aliases

```bash
alias ll='ls -la'
alias gs='git status'
```
