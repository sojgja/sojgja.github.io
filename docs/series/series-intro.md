---
id: series-intro
title: Design Patterns — 23 mẫu thiết kế
sidebar_label: 📖 Giới thiệu
sidebar_position: 1
---

# Design Patterns — 23 mẫu thiết kế

> *"Mỗi pattern mô tả một vấn đề xuất hiện đi xuất hiện lại trong môi trường của chúng ta, và sau đó mô tả cốt lõi của giải pháp cho vấn đề đó, theo cách bạn có thể dùng giải pháp đó hàng triệu lần mà không bao giờ làm nó theo cùng một cách hai lần."*
> — **Christopher Alexander**, *A Pattern Language* (1977)

Bạn có bao giờ tự hỏi, tại sao có những lập trình viên viết code hàng giờ đồng hồ không biết mệt, trong khi người khác cứ loay hoay sửa bug hết ngày này qua ngày khác? Câu trả lời không nằm ở việc họ thông minh hơn bạn, mà nằm ở thứ đơn giản hơn nhiều: họ biết dùng **design patterns**.

**Design Patterns** (còn gọi là **Gang of Four patterns**) là tập hợp 23 mẫu thiết kế được đúc kết bởi 4 ông lớn: Erich Gamma, Richard Helm, Ralph Johnson và John Vlissides trong cuốn sách *Design Patterns: Elements of Reusable Object-Oriented Software* (1994). Cuốn sách này được coi là kinh thánh của lập trình hướng đối tượng.

Mỗi pattern là một **giải pháp tổng quát, đã được kiểm chứng** cho một vấn đề thiết kế thường gặp. Pattern không phải là thư viện hay framework — bạn không thể `pip install observer` — mà là một **khuôn mẫu tư duy** để bạn linh hoạt áp dụng vào code của mình.

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
| **Bridge** | Tách abstraction khỏi implementation để cả hai độc lập thay đổi | Cross-platform GUI, database drivers |
| **Composite** | Tổ chức đối tượng thành cấu trúc cây whole-part | UI component tree, file system |
| **Decorator** | Gắn thêm trách nhiệm vào đối tượng linh hoạt | Middleware, I/O streams |
| **Facade** | Interface đơn giản cho hệ thống phức tạp | REST API wrapper, ORM |
| **Flyweight** | Chia sẻ đối tượng nhỏ để tiết kiệm bộ nhớ | Text rendering, game particle systems |
| **Proxy** | Đối tượng thay thế kiểm soát truy cập | Lazy loading, access control |

### Behavioral Patterns (11 mẫu)

| Pattern | Mục đích | Ví dụ thực tế |
|---------|----------|--------------|
| **Chain of Responsibility** | Request được chuyển tiếp qua chuỗi xử lý | Middleware pipeline, logging handlers |
| **Command** | Đóng gói request thành đối tượng | Undo/redo, task queue |
| **Interpreter** | Định nghĩa ngữ pháp cho một ngôn ngữ | Regex, SQL parser |
| **Iterator** | Duyệt collection không lộ cấu trúc bên trong | Database cursor, tree traversal |
| **Mediator** | Giảm coupling giữa các component | Chat room, event bus |
| **Memento** | Lưu và khôi phục trạng thái | Undo/redo, save game |
| **Observer** | Thông báo khi state thay đổi | Event system, pub/sub |
| **State** | Hành vi thay đổi theo trạng thái | Workflow engine, state machine |
| **Strategy** | Họ thuật toán có thể hoán đổi | Payment methods, sorting |
| **Template Method** | Khung thuật toán, để subclass cài đặt chi tiết | CI/CD pipeline, data processing |
| **Visitor** | Thêm thao tác mới không sửa class cũ | AST analysis, code generation |

---

## Kiến thức nền tảng cần có

Để hiểu sâu 23 pattern này, bạn cần nắm vững:

### 1. Lập trình hướng đối tượng (OOP)

- **Bốn tính chất**: Encapsulation, Inheritance, Polymorphism, Abstraction
- **Abstract class và Interface**: Kế thừa implementation vs kế thừa contract
- **Composition over Inheritance**: Tại sao ưu tiên composition hơn inheritance
- **SOLID principles**: Đặc biệt là Open/Closed và Dependency Inversion

### 2. Python (phiên bản 3.10+)

Tôi dùng Python cho các ví dụ trong series này:
- **`abc.ABC`** và **`@abstractmethod`** cho interface
- **`dataclasses`** cho value objects
- **`typing`** module với type hints
- **`contextlib`**, **`functools`** cho patterns nâng cao

Người dùng Java, C#, C++ cũng có thể đọc được dễ dàng.

### 3. Kiến thức bổ trợ

- **UML class diagram**: Ký hiệu inheritance, association, dependency
- **Testing**: Unit test với `unittest` hoặc `pytest`
- **Design principles**: GRASP, DRY, KISS, YAGNI

---

## Cấu trúc mỗi bài viết

Mỗi pattern trong series được trình bày theo cấu trúc nhất quán:

| Phần | Nội dung |
|------|----------|
| **Mở đầu** | Pattern là gì, trích dẫn GoF gốc |
| **Bài toán** | Tình huống thực tế, vấn đề cụ thể |
| **Giải pháp** | Pattern giải quyết vấn đề ra sao |
| **Phân tích thiết kế** | Trade-offs, anti-patterns, khi nào KHÔNG dùng |
| **Code** | Cách sai → cách đúng, code hoàn chỉnh |
| **UML** | Sơ đồ minh họa quan hệ |
| **So sánh** | Phân biệt với pattern tương tự |
| **Ứng dụng** | Pattern xuất hiện ở framework/thư viện nào |
| **Kiểm thử** | Unit test cho pattern |
| **Kết luận** | Khi nào áp dụng, golden rules |

---

## Lộ trình học

### Cách 1: Theo nhóm (khuyên dùng)

Đọc từng nhóm để thấy sự liên quan giữa các pattern:

1. **Creational**: Singleton → Factory Method → Abstract Factory → Builder → Prototype
2. **Structural**: Adapter → Bridge → Composite → Decorator → Facade → Flyweight → Proxy
3. **Behavioral**: Template Method → Strategy → State → Observer → Chain → Command → Iterator → Mediator → Memento → Visitor → Interpreter

### Cách 2: Theo mức độ phổ biến

- **Rất phổ biến**: Singleton, Factory Method, Strategy, Observer, Decorator, Adapter
- **Phổ biến**: Builder, Facade, Proxy, Command, State, Template Method, Iterator
- **Ít phổ biến hơn**: Abstract Factory, Bridge, Composite, Prototype, Flyweight, Chain of Responsibility, Mediator, Memento, Visitor, Interpreter

### Cách 3: Theo vấn đề gặp phải

- "Code có quá nhiều if-else xử lý trạng thái" → **State**
- "Muốn thêm algorithm mới không sửa code cũ" → **Strategy**
- "Cần thông báo cho nhiều component khi dữ liệu thay đổi" → **Observer**

---

## Lưu ý khi học Design Patterns

Tôi đã từng mắc sai lầm khi mới học: cố nhồi nhét pattern vào mọi chỗ, bất kể nó có phù hợp hay không. Hãy rút kinh nghiệm từ tôi.

1. **Không phải solution cho mọi vấn đề**: Pattern có trade-off riêng. Đừng áp dụng pattern chỉ vì nó "ngầu". Hãy áp dụng khi nó thực sự giải quyết được vấn đề.
2. **Pattern là ngôn ngữ chung**: "Dùng Strategy pattern cho payment methods" nhanh hơn giải thích dài dòng.
3. **Pattern sống trong code, không trong sách**: Đọc xong, hãy viết code ngay. Pattern chỉ có ích khi bạn đã implement nó ít nhất một lần.
4. **Ngôn ngữ ảnh hưởng đến pattern**: Python có first-class functions nên Strategy đơn giản hơn Java. Hiểu bản chất, không chỉ implementation.
5. **Không học thuộc lòng**: Mục tiêu không phải nhớ tên 23 pattern, mà là phát triển **tư duy thiết kế**.

---

> *"Design patterns không phải là về thiết kế. Chúng là về những giải pháp chung cho những vấn đề chung."*
> — **Erich Gamma**

Hãy bắt đầu hành trình chinh phục 23 mẫu thiết kế phần mềm. Bài đầu tiên là **Template Method** — pattern đơn giản nhất trong nhóm Behavioral, nền tảng cho nhiều pattern khác.

---
*Trân trọng!*
