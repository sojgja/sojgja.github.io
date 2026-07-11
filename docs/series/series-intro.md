---
id: series-intro
title: Design Patterns — 23 mẫu thiết kế
sidebar_label: 📖 Giới thiệu
sidebar_position: 1
---

# Design Patterns — 23 mẫu thiết kế

> "Each pattern describes a problem which occurs over and over again in our environment, and then describes the core of the solution to that problem, in such a way that you can use this solution a million times over, without ever doing it the same way twice."
> — **Christopher Alexander**, *A Pattern Language* (1977)

**Design Patterns** (còn gọi là **Gang of Four patterns** hay **GoF patterns**) là tập hợp 23 mẫu thiết kế phần mềm được đúc kết bởi Erich Gamma, Richard Helm, Ralph Johnson và John Vlissides trong cuốn sách *Design Patterns: Elements of Reusable Object-Oriented Software* (1994). Đây được coi là kinh thánh của lập trình hướng đối tượng.

Mỗi pattern là một **giải pháp tổng quát, đã được kiểm chứng** cho một vấn đề thiết kế thường gặp. Pattern không phải là thư viện hay framework — bạn không thể `pip install observer` — mà là một **khuôn mẫu tư duy** để bạn áp dụng linh hoạt vào code của mình.

---

## Tổng quan 23 mẫu thiết kế

23 pattern được chia làm 3 nhóm dựa trên mục đích sử dụng:

| Nhóm | Mục đích | Số lượng | Các Pattern |
|------|----------|----------|-------------|
| **Creational** (Khởi tạo) | Giải quyết vấn đề tạo đối tượng linh hoạt, tránh `new` cứng nhắc | 5 | Singleton, Factory Method, Abstract Factory, Builder, Prototype |
| **Structural** (Cấu trúc) | Sắp xếp các lớp và đối tượng thành cấu trúc lớn hơn | 7 | Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy |
| **Behavioral** (Hành vi) | Tối ưu tương tác và phân chia trách nhiệm giữa các đối tượng | 11 | Chain of Responsibility, Command, Interpreter, Iterator, Mediator, Memento, Observer, State, Strategy, Template Method, Visitor |

### Creational Patterns (5 mẫu)

| Pattern | Mục đích | Ví dụ thực tế |
|---------|----------|--------------|
| **Singleton** | Đảm bảo một class chỉ có một instance duy nhất | Database connection pool, logging service |
| **Factory Method** | Định nghĩa interface tạo đối tượng, để subclass quyết định class nào được tạo | Django model forms, framework views |
| **Abstract Factory** | Tạo họ các đối tượng liên quan mà không chỉ định class cụ thể | UI toolkit cho nhiều OS, database drivers |
| **Builder** | Tách việc xây dựng đối tượng phức tạp khỏi biểu diễn của nó | Cấu hình HTTP request, build SQL query |
| **Prototype** | Tạo đối tượng mới bằng cách clone đối tượng có sẵn | Cache, undo/redo, game object spawning |

### Structural Patterns (7 mẫu)

| Pattern | Mục đích | Ví dụ thực tế |
|---------|----------|--------------|
| **Adapter** | Chuyển interface của một class thành interface khác mà client mong đợi | Tích hợp third-party API, legacy code |
| **Bridge** | Tách abstraction khỏi implementation để cả hai có thể thay đổi độc lập | Cross-platform GUI, database drivers |
| **Composite** | Tổ chức các đối tượng thành cấu trúc cây để biểu diễn quan hệ whole-part | UI component tree, file system |
| **Decorator** | Gắn thêm trách nhiệm vào đối tượng một cách linh hoạt | Middleware trong web framework, I/O streams |
| **Facade** | Cung cấp interface đơn giản cho một hệ thống phức tạp | REST API wrapper, ORM |
| **Flyweight** | Chia sẻ các đối tượng nhỏ để tiết kiệm bộ nhớ | Text rendering, game particle systems |
| **Proxy** | Cung cấp đối tượng thay thế để kiểm soát truy cập đến đối tượng khác | Lazy loading, access control, logging |

### Behavioral Patterns (11 mẫu)

| Pattern | Mục đích | Ví dụ thực tế |
|---------|----------|--------------|
| **Chain of Responsibility** | Cho nhiều object có cơ hội xử lý request, chuyển tiếp nhau | Middleware pipeline, logging handlers |
| **Command** | Đóng gói request thành đối tượng, cho phép parameterize và queue | Undo/redo, task queue, macro recording |
| **Interpreter** | Định nghĩa ngữ pháp và interpreter cho một ngôn ngữ | Regex, SQL parser, template engine |
| **Iterator** | Truy cập các phần tử của collection mà không lộ cấu trúc bên trong | Database cursor, tree traversal |
| **Mediator** | Định nghĩa đối tượng trung gian để giảm coupling giữa các component | Chat room, GUI dialog, event bus |
| **Memento** | Lưu và khôi phục trạng thái của đối tượng mà không vi phạm encapsulation | Undo/redo, save game, transaction rollback |
| **Observer** | Định nghĩa cơ chế one-to-many dependency để thông báo khi state thay đổi | Event system, pub/sub, reactive programming |
| **State** | Cho phép đối tượng thay đổi hành vi khi trạng thái thay đổi | Workflow engine, UI state machine |
| **Strategy** | Định nghĩa họ thuật toán, đóng gói và hoán đổi cho nhau | Payment methods, sorting algorithms |
| **Template Method** | Định nghĩa khung thuật toán, để subclass cài đặt các bước chi tiết | CI/CD pipeline, data processing |
| **Visitor** | Thêm thao tác mới vào object hierarchy mà không sửa class | AST analysis, code generation, export formats |

---

## Kiến thức nền tảng cần có

Để hiểu sâu 23 pattern này, bạn cần nắm vững:

### 1. Lập trình hướng đối tượng (OOP)

- **Bốn tính chất**: Encapsulation, Inheritance, Polymorphism, Abstraction
- **Abstract class và Interface**: Phân biệt giữa kế thừa implementation và kế thừa contract
- **Composition over Inheritance**: Tại sao ưu tiên composition hơn inheritance
- **SOLID principles**: Đặc biệt là Open/Closed Principle và Dependency Inversion Principle

### 2. Python (phiên bản 3.10+)

Các ví dụ trong series này sử dụng Python 3.10+ với:
- **`abc.ABC`** và **`@abstractmethod`** cho interface
- **`dataclasses`** cho value objects
- **`enum.Enum`** và **`enum.auto`** cho hằng số
- **`typing`** module với type hints (`Protocol`, `Generic`, `Optional`, etc.)
- **`contextlib`**, **`functools`** cho patterns nâng cao

Người dùng thành thạo Java, C#, C++ cũng có thể đọc được dễ dàng vì Python syntax rất clear.

### 3. Kiến thức bổ trợ

- **UML class diagram**: Hiểu các ký hiệu (inheritance, association, dependency)
- **Testing**: Cách viết unit test với `unittest` hoặc `pytest`
- **Design principles**: GRASP, DRY, KISS, YAGNI

---

## Cấu trúc mỗi bài viết

Mỗi pattern trong series được trình bày theo cấu trúc 11 phần nhất quán:

| Phần | Nội dung |
|------|----------|
| **Mở đầu** | Định nghĩa pattern kèm trích dẫn GoF gốc |
| **Bài toán chi tiết** | Tình huống thực tế với context cụ thể, phân tích pain points |
| **Giải pháp với Pattern** | Pattern giải quyết bài toán như thế nào, mapping giữa vấn đề và giải pháp |
| **Phân tích thiết kế** | Nguyên lý OOP, trade-offs, anti-patterns, khi nào KHÔNG dùng |
| **Code hoàn chỉnh** | Code sai → code đúng, type hints, ABC, test scenarios |
| **Sơ đồ UML** | ASCII UML class diagram minh họa mối quan hệ |
| **So sánh với pattern liên quan** | So sánh 2-3 pattern gần giống, cách phân biệt |
| **Ứng dụng thực tế** | Pattern xuất hiện ở đâu trong thư viện/framework nổi tiếng |
| **Kiểm thử** | Unit test cho pattern, mock objects, assertion strategies |
| **Ưu và nhược điểm** | Bảng so sánh ưu/nhược điểm chi tiết |
| **Kết luận** | Khi nào áp dụng, dấu hiệu nhận biết, golden rules |

---

## Lộ trình đề xuất

### Cách 1: Theo nhóm (recommended)

Đọc lần lượt từng nhóm để thấy sự liên quan giữa các pattern:

1. **Creational** (Singleton → Factory Method → Abstract Factory → Builder → Prototype)
2. **Structural** (Adapter → Bridge → Composite → Decorator → Facade → Flyweight → Proxy)
3. **Behavioral** (Template Method → Strategy → State → Observer → Chain → Command → Iterator → Mediator → Memento → Visitor → Interpreter)

### Cách 2: Theo mức độ phổ biến

Bắt đầu với các pattern hay dùng nhất:

1. **Rất phổ biến**: Singleton, Factory Method, Strategy, Observer, Decorator, Adapter
2. **Phổ biến**: Builder, Facade, Proxy, Command, State, Template Method, Iterator
3. **Ít phổ biến hơn**: Abstract Factory, Bridge, Composite, Prototype, Flyweight, Chain of Responsibility, Mediator, Memento, Visitor, Interpreter

### Cách 3: Theo tình huống thực tế

Nếu bạn đang gặp vấn đề cụ thể:
- "Code của tôi có quá nhiều if-else để xử lý trạng thái" → **State**
- "Tôi muốn thêm algorithm mới mà không sửa code cũ" → **Strategy**
- "Tôi cần thông báo cho nhiều component khi dữ liệu thay đổi" → **Observer**
- "Tôi muốn định nghĩa quy trình xử lý chuẩn" → **Template Method**
- "Tôi có object hierarchy ổn định và muốn thêm thao tác mới" → **Visitor**

---

## Lưu ý khi học Design Patterns

1. **Không phải solution cho mọi vấn đề**: Pattern có trade-off riêng. Đừng áp dụng pattern chỉ vì nó "ngầu" — hãy áp dụng khi nó giải quyết được vấn đề thực sự.
2. **Pattern là ngôn ngữ chung**: Biết pattern giúp bạn trao đổi với đồng nghiệp hiệu quả hơn. "Dùng Strategy pattern cho payment methods" nhanh hơn nhiều so với giải thích dài dòng.
3. **Pattern sống trong code, không trong sách**: Đọc xong pattern, hãy viết code ngay. Pattern chỉ trở nên hữu ích khi bạn đã implement nó ít nhất một lần.
4. **Ngôn ngữ ảnh hưởng đến pattern**: Một số pattern dễ dàng hơn trong ngôn ngữ này so với ngôn ngữ khác. Ví dụ, Python có first-class functions nên Strategy đơn giản hơn Java. Hãy hiểu bản chất, không chỉ implementation.
5. **Không học thuộc lòng**: Mục tiêu không phải nhớ tên 23 pattern, mà là phát triển **tư duy thiết kế** — khả năng nhìn ra vấn đề và chọn giải pháp phù hợp.

---

> "Design patterns are not about design. They are about common solutions to common problems."
> — **Erich Gamma**

Hãy bắt đầu hành trình chinh phục 23 mẫu thiết kế phần mềm. Bài đầu tiên là **Template Method** — pattern đơn giản nhất trong nhóm Behavioral, nền tảng cho nhiều pattern khác.
