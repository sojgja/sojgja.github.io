---
id: python-cheatsheet
title: Python Cheatsheet
sidebar_label: 🐍 Python Cheatsheet
sidebar_position: 4
---

# Python Cheatsheet

## List comprehension

```python
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

## Dict

```python
d = {"a": 1, "b": 2}
d.get("c", 0)              # Safe get
{k: v for k, v in d.items() if v > 1}
```

## Lambda

```python
add = lambda a, b: a + b
sorted(list, key=lambda x: x[1])
```

## File I/O

```python
with open("file.txt", "r") as f:
    content = f.read()
```

## Error handling

```python
try:
    risky()
except ValueError as e:
    print(e)
finally:
    cleanup()
```

## Type hints

```python
def greet(name: str) -> str:
    return f"Hello {name}"
```
