---
id: oop
title: Object-Oriented Programming
sidebar_label: 🔵 OOP Concepts
sidebar_position: 3
---

# Object-Oriented Programming

## 4 Tính chất chính

### 1. Encapsulation (Đóng gói)
Che giấu dữ liệu, chỉ cho phép truy cập qua methods.

```python
class BankAccount:
    def __init__(self):
        self.__balance = 0  # private

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
```

### 2. Inheritance (Kế thừa)
Class con tái sử dụng class cha.

```python
class Animal:
    def speak(self): pass

class Dog(Animal):
    def speak(self):
        return "Woof!"
```

### 3. Polymorphism (Đa hình)
Cùng interface, nhiều implementation.

### 4. Abstraction (Trừu tượng)
Che giấu chi tiết phức tạp, chỉ expose những gì cần thiết.

## SOLID Principles

- **S** - Single Responsibility
- **O** - Open/Closed
- **L** - Liskov Substitution
- **I** - Interface Segregation
- **D** - Dependency Inversion
