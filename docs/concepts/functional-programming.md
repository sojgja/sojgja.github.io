---
id: functional-programming
title: Functional Programming
sidebar_label: λ Functional Programming
sidebar_position: 4
---

# Functional Programming

Lập trình hàm là paradigm tập trung vào **pure functions** và **immutable data**.

## Nguyên lý cốt lõi

### Pure Functions
Cùng đầu vào → cùng đầu ra, không side effect.

```python
# Pure
def add(a, b): return a + b

# Impure (có side effect)
def add_to_total(x):
    global total
    total += x
```

### Immutability
Không thay đổi dữ liệu gốc, tạo bản sao mới.

```python
# Thay đổi trực tiếp (impure)
numbers.append(5)

# Tạo mới (pure)
new_numbers = [*numbers, 5]
```

### Higher-Order Functions
Hàm nhận hàm khác làm tham số hoặc trả về hàm.

```python
map(lambda x: x * 2, [1, 2, 3])
filter(lambda x: x > 0, [-1, 2, -3, 4])
reduce(lambda a, b: a + b, [1, 2, 3, 4])
```

## Lợi ích

- Dễ test (pure function = deterministic)
- Dễ debug (không side effect bất ngờ)
- Dễ parallel processing
- Code ngắn gọn, dễ đọc
