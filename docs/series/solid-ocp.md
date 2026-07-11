---
id: solid-ocp
title: O — Open/Closed Principle
sidebar_label: O — Open/Closed
sidebar_position: 27
---

# O — Open/Closed Principle

> **"Software entities should be open for extension, but closed for modification."** — Bertrand Meyer

Class/module nên **mở cho việc mở rộng** (có thể thêm hành vi mới) nhưng **đóng cho việc sửa đổi** (không cần sửa code cũ).

## Bài toán: Tính lương nhân viên

```python
class SalaryCalculator:
    def calculate(self, employee_type, base_salary):
        if employee_type == 'fulltime':
            return base_salary - base_salary * 0.1  # Trừ bảo hiểm
        elif employee_type == 'parttime':
            return base_salary * 0.8  # 80%
        elif employee_type == 'intern':
            return base_salary * 0.5  # 50%
        elif employee_type == 'contractor':
            return base_salary * 0.9  # 90%
        # Mỗi lần thêm loại nhân viên mới → phải sửa method này!
```

**Vấn đề:** Mỗi lần thêm loại nhân viên mới (freelancer, seasonal, ...) bạn phải:
1. Mở file này
2. Thêm `elif`
3. Sửa unit test
4. Rủi ro làm hỏng logic cũ

## Giải pháp: Dùng abstraction (OCP)

```python
from abc import ABC, abstractmethod

class Employee(ABC):
    def __init__(self, name, base_salary):
        self.name = name
        self.base_salary = base_salary

    @abstractmethod
    def calculate_salary(self):
        pass

class FullTimeEmployee(Employee):
    def calculate_salary(self):
        return self.base_salary - self.base_salary * 0.1

class PartTimeEmployee(Employee):
    def calculate_salary(self):
        return self.base_salary * 0.8

class InternEmployee(Employee):
    def calculate_salary(self):
        return self.base_salary * 0.5

class ContractorEmployee(Employee):
    def calculate_salary(self):
        return self.base_salary * 0.9

class SalaryCalculator:
    def calculate(self, employee: Employee):
        return employee.calculate_salary()

# Sử dụng — thêm loại mới KHÔNG cần sửa code cũ
class FreelancerEmployee(Employee):
    def calculate_salary(self):
        return self.base_salary * 0.85

calc = SalaryCalculator()
employees = [
    FullTimeEmployee('Alice', 10_000_000),
    InternEmployee('Bob', 5_000_000),
    FreelancerEmployee('Charlie', 8_000_000),
]

for emp in employees:
    print(f'{emp.name}: {calc.calculate(emp):,.0f} VND')
```

## Mở rộng: Strategy Pattern kết hợp OCP

```python
class DiscountStrategy(ABC):
    @abstractmethod
    def apply(self, total): pass

class NoDiscount(DiscountStrategy):
    def apply(self, total): return total

class PercentageDiscount(DiscountStrategy):
    def __init__(self, percent):
        self.percent = percent

    def apply(self, total): return total * (1 - self.percent / 100)

class FixedDiscount(DiscountStrategy):
    def __init__(self, amount):
        self.amount = amount

    def apply(self, total): return max(0, total - self.amount)

class BlackFridayDiscount(DiscountStrategy):
    def apply(self, total): return total * 0.5  # Giảm 50%

# Thêm discount mới → không sửa code cũ
# class MemberDiscount(DiscountStrategy): ...
```

## Dấu hiệu nhận biết vi phạm OCP

- Dùng `if/elif/else` kiểm tra type hoặc enum để quyết định hành vi
- Mỗi lần thêm tính năng mới phải sửa nhiều file
- Có switch-case lớn trong code
- Dùng `isinstance()` trong logic nghiệp vụ

## Kết luận

OCP là chìa khóa cho code **dễ mở rộng**. Hãy dùng abstraction (`ABC`/interface) và Strategy Pattern. Khi thêm tính năng mới, bạn chỉ cần **thêm file mới** chứ không cần sửa file cũ.
