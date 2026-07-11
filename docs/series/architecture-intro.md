---
id: architecture-intro
title: Giới thiệu về Kiến trúc Phần mềm
sidebar_label: 🏛️ Giới thiệu Kiến trúc
sidebar_position: 35
---

# Giới thiệu về Kiến trúc Phần mềm

> *"Architecture is about the important stuff. Whatever that is."* — **Martin Fowler**

Kiến trúc phần mềm là bức tranh tổng thể của một hệ thống — tập hợp các quyết định thiết kế nền tảng định hình cấu trúc, hành vi, và sự tiến hóa của phần mềm. Nó không đơn thuần là sơ đồ khối hay lựa chọn công nghệ; kiến trúc là xương sống quyết định liệu một hệ thống có thể tồn tại và phát triển qua năm tháng hay sẽ sụp đổ dưới sức nặng của chính nó. Trong series này, chúng ta sẽ khám phá 20 kiến trúc phần mềm quan trọng nhất, từ những mô hình cổ điển đã đồng hành cùng ngành công nghiệp nhiều thập kỷ cho đến những xu hướng kiến trúc hiện đại đang định hình tương lai.

---

## Kiến trúc phần mềm là gì?

Theo định nghĩa của ISO/IEC/IEEE 42010, kiến trúc phần mềm là "cấu trúc cơ bản của một hệ thống, được thể hiện qua các thành phần của nó, mối quan hệ giữa các thành phần với nhau và với môi trường, cùng các nguyên lý chi phối thiết kế và sự tiến hóa của chúng." Nói một cách dễ hiểu hơn, kiến trúc là những quyết định khó thay đổi — chúng giống như bộ xương của một tòa nhà. Bạn có thể thay đổi màu sơn, thay đồ nội thất, nhưng thay đổi cấu trúc chịu lực là một việc hoàn toàn khác.

### Lịch sử hình thành

Kiến trúc phần mềm không phải là một khái niệm mới. Nó đã trải qua nhiều giai đoạn phát triển:

| Thời kỳ | Đặc điểm | Kiến trúc tiêu biểu |
|---------|----------|---------------------|
| 1960s-1970s | Mainframe, monolithic | Batch processing, Centralized |
| 1980s | Client-Server xuất hiện | Two-tier, File sharing |
| 1990s | Internet bùng nổ | Three-tier, Layered, CORBA |
| 2000s | Web 2.0, SOA | SOAP, REST, Service-Oriented |
| 2010s | Cloud, Container | Microservices, Event-Driven, CQRS |
| 2020s | AI-native, Edge, Real-time | Serverless, Data Mesh, Event Sourcing |

Mỗi thập kỷ đều mang đến những thách thức mới và giải pháp kiến trúc tương ứng. Từ những mainframe tập trung đến microservices phân tán, từ synchronous request-response đến event-driven asynchronous, sự tiến hóa này phản ánh nhu cầu ngày càng cao về scalability, maintainability, và độ phức tạp của hệ thống.

### Tại sao kiến trúc quan trọng?

Kiến trúc ảnh hưởng đến mọi khía cạnh của dự án phần mềm: performance (hiệu năng), scalability (khả năng mở rộng), maintainability (bảo trì), testability (khả năng kiểm thử), deployability (triển khai), và security (bảo mật). Một kiến trúc tốt không chỉ giúp hệ thống chạy đúng hôm nay mà còn đảm bảo nó có thể thích nghi với yêu cầu ngày mai. Một nghiên cứu của McKinsey cho thấy các dự án có kiến trúc được thiết kế bài bản có tỷ lệ thành công cao hơn 68% so với các dự án không có, và chi phí bảo trì giảm đến 40% trong vòng đời 5 năm.

### Các khía cạnh của kiến trúc

Kiến trúc phần mềm bao gồm nhiều khía cạnh, mỗi khía cạnh giải quyết một nhóm vấn đề cụ thể:

1. **Structural**: Tổ chức các thành phần (components, modules, services) và mối quan hệ giữa chúng
2. **Behavioral**: Tương tác và giao tiếp giữa các thành phần (synchronous/asynchronous, events, messages)
3. **Non-functional**: Performance, scalability, availability, security, reliability
4. **Deployment**: Cách triển khai, môi trường chạy, infrastructure
5. **Data**: Lưu trữ, truy xuất, consistency, replication
6. **Business**: Alignment với mục tiêu kinh doanh, domain modeling

---

## 20 Kiến trúc trong Series

Series này sẽ giới thiệu **20 kiến trúc phần mềm** quan trọng, từ cổ điển đến hiện đại. Chúng được phân loại theo 5 nhóm:

### Nhóm 1: Kiến trúc Cổ điển (Classical)

| # | Kiến trúc | Tên tiếng Việt | Đặc điểm chính | Khi nào dùng |
|---|-----------|----------------|----------------|--------------|
| 1 | **Layered Architecture (N-Tier)** | Kiến trúc Phân lớp | Phân tách theo layer: presentation, business, persistence, database | Ứng dụng enterprise truyền thống, monolith |
| 2 | **Client-Server Architecture** | Kiến trúc Khách-Chủ | Hai thành phần: client (UI) và server (logic + data) | Web application, mobile backend |
| 3 | **MVC / MVP / MVVM** | Model-View-Controller | Tách UI (View) khỏi logic (Model) qua Controller/Presenter | GUI desktop, mobile, web frontend |
| 4 | **Repository Architecture** | Kho lưu trữ | Abstraction layer giữa domain và data source | Hệ thống cần nhiều data source khác nhau |
| 5 | **Pipe-and-Filter** | Đường ống và Bộ lọc | Xử lý theo chuỗi: output của filter là input của filter kế tiếp | Data processing pipeline, ETL, compiler |

### Nhóm 2: Kiến trúc Service-Based

| # | Kiến trúc | Tên tiếng Việt | Đặc điểm chính | Khi nào dùng |
|---|-----------|----------------|----------------|--------------|
| 6 | **Service-Oriented Architecture (SOA)** | Kiến trúc Hướng Dịch vụ | Dịch vụ lớn (enterprise bus), SOAP/XML | Hệ thống doanh nghiệp lớn, integration |
| 7 | **Microservices Architecture** | Kiến trúc Vi dịch vụ | Dịch vụ nhỏ, độc lập, triển khai riêng | Hệ thống lớn, team nhiều người, cloud-native |
| 8 | **Service Mesh** | Lưới Dịch vụ | Proxy sidecar quản lý giao tiếp giữa services | Microservices ở quy mô lớn (100+ services) |
| 9 | **Serverless Architecture** | Kiến trúc Không Máy chủ | Function-as-a-Service, auto-scaling, pay-per-use | Event-driven, batch job, API backend |
| 10 | **Backend-for-Frontend (BFF)** | Backend riêng cho Frontend | Mỗi frontend có một backend riêng, optimized | Multi-platform (web, mobile, IoT) |

### Nhóm 3: Kiến trúc Event-Driven

| # | Kiến trúc | Tên tiếng Việt | Đặc điểm chính | Khi nào dùng |
|---|-----------|----------------|----------------|--------------|
| 11 | **Event-Driven Architecture (EDA)** | Kiến trúc Hướng Sự kiện | Producer phát event, consumer xử lý bất đồng bộ | Real-time system, IoT, notification |
| 12 | **Event Sourcing** | Lưu trữ Sự kiện | Lưu toàn bộ sự kiện thay vì trạng thái hiện tại | Audit log, financial system, undo |
| 13 | **CQRS (Command Query Responsibility Segregation)** | Phân tách Truy vấn và Ghi | Command (write) khác Query (read), có thể khác DB | Hệ thống read-heavy, phức tạp |
| 14 | **Pub/Sub Architecture** | Publish-Subscribe | Message broker trung gian, decoupling hoàn toàn | Notification system, real-time feed |

### Nhóm 4: Kiến trúc Modular & Domain

| # | Kiến trúc | Tên tiếng Việt | Đặc điểm chính | Khi nào dùng |
|---|-----------|----------------|----------------|--------------|
| 15 | **Hexagonal Architecture** | Kiến trúc Lục giác (Ports & Adapters) | Core business logic độc lập với infrastructure | Domain-driven design, testable system |
| 16 | **Clean Architecture** | Kiến trúc Sạch | Dependency rule: các lớp đồng tâm, outer → inner | Enterprise system, long-lived project |
| 17 | **Onion Architecture** | Kiến trúc Hành tây | Tương tự Clean Architecture, nhấn mạnh domain | DDD, hệ thống phức tạp |
| 18 | **Domain-Driven Design (DDD)** | Thiết kế Hướng Miền | Ubiquitous language, bounded context, aggregate | Hệ thống có domain logic phức tạp |

### Nhóm 5: Kiến trúc Phân tán & Mở rộng

| # | Kiến trúc | Tên tiếng Việt | Đặc điểm chính | Khi nào dùng |
|---|-----------|----------------|----------------|--------------|
| 19 | **Space-Based Architecture** | Kiến trúc Dựa trên Không gian | In-memory data grid, loại bỏ database bottleneck | Hệ thống high-load, real-time |
| 20 | **Data Mesh** | Lưới Dữ liệu | Data decentralized theo domain, data-as-a-product | Data platform lớn, data-driven organization |

---

## Bảng so sánh tổng quan

| Kiến trúc | Coupling | Scalability | Maintainability | Deployment | Use case điển hình |
|-----------|----------|-------------|-----------------|------------|-------------------|
| Layered (N-Tier) | Tight (trong layer) | Vertical | Cao | Đơn giản (1-3 server) | ERP, CRM, CMS |
| Client-Server | Tight | Trung bình | Trung bình | 2 server | Web app cổ điển |
| MVC/MVP/MVVM | Loose (Model-View) | Trung bình | Cao | Frontend + API | UI application |
| Repository | Loose | Trung bình | Cao | Tùy kiến trúc | Data-driven app |
| Pipe-and-Filter | Loose (giữa filter) | Horizontal (parallel) | Cao | Đơn giản | ETL, data pipeline |
| SOA | Tight (qua ESB) | Trung bình | Trung bình | Nhiều server | Enterprise integration |
| **Microservices** | **Loose** (qua API) | **Horizontal** | **Cao (từng service)** | **Phức tạp** | **Cloud-native, scale lớn** |
| Service Mesh | Loose (qua sidecar) | Horizontal | Trung bình | Rất phức tạp | Service quản lý traffic |
| Serverless | Rất Loose | Auto-scaling | Cao (hạn chế) | Đơn giản (deploy code) | Event-driven, API |
| BFF | Trung bình | Trung bình | Cao | Trung bình | Multi-platform app |
| **EDA** | **Rất Loose** | **Horizontal** | **Cao** | **Trung bình** | **Real-time, streaming** |
| Event Sourcing | Loose | Horizontal | Trung bình | Phức tạp | Audit, financial |
| CQRS | Trung bình | Horizontal (scale read/write riêng) | Trung bình | Phức tạp | Read-heavy system |
| Pub/Sub | Rất Loose | Horizontal | Cao | Trung bình | Notification, feed |
| **Hexagonal** | **Rất Loose** | **Trung bình** | **Rất Cao** | **Đơn giản** | **DDD, testable system** |
| Clean | Rất Loose | Trung bình | Rất Cao | Đơn giản | Enterprise system |
| Onion | Rất Loose | Trung bình | Rất Cao | Đơn giản | DDD system |
| DDD | Loose (bounded context) | Horizontal | Cao | Tùy implementation | Domain phức tạp |
| Space-Based | Loose | Rất Cao (in-memory grid) | Trung bình | Phức tạp | High-load, gaming |
| Data Mesh | Loose (domain) | Rất Cao | Cao | Rất phức tạp | Data platform |

---

## Cách chọn kiến trúc phù hợp

Không có kiến trúc nào là "tốt nhất" — chỉ có kiến trúc phù hợp nhất với bối cảnh cụ thể. Dưới đây là quy trình 7 bước để đưa ra quyết định sáng suốt:

### Bước 1: Xác định yêu cầu phi chức năng (NFRs)

Đây là yếu tố quan trọng nhất. Hãy trả lời các câu hỏi sau:

- **Scalability**: Hệ thống cần handle bao nhiêu user? 100? 10,000? 10,000,000?
- **Availability**: Downtime cho phép là bao nhiêu? 99.9% (8.7h/năm) hay 99.999% (5 phút/năm)?
- **Performance**: Latency tối đa? Throughput mong đợi?
- **Security**: Yêu cầu compliance (GDPR, HIPAA, PCI DSS)?
- **Maintainability**: Đội ngũ bao nhiêu người? Skill set ra sao?
- **Time-to-market**: Deadline trong 3 tháng hay 3 năm?
- **Budget**: Chi phí infrastructure, license, nhân sự?

### Bước 2: Đánh giá độ phức tạp domain

- Domain đơn giản (CRUD) → Layered, MVC
- Domain phức tạp, nhiều business rule → DDD, Hexagonal, Clean
- Domain thay đổi liên tục → Event-Driven, CQRS

### Bước 3: Xác định quy mô và cấu trúc team

- 1-5 developers, monolith → Layered (N-Tier)
- 5-20 developers, 1 team → Modular Monolith, Hexagonal
- 20-200 developers, nhiều team → Microservices, BFF
- 200+ developers → Service Mesh, Data Mesh

### Bước 4: Phân tích data characteristics

- Read-heavy → CQRS (tách read/write), cache layer
- Write-heavy → Event Sourcing, messaging queue
- Real-time → Event-Driven, Streaming (Kafka)
- Audit/logging → Event Sourcing
- Strong consistency → Monolith, distributed transaction
- Eventual consistency → Microservices, EDA

### Bước 5: Đánh giá infrastructure và DevOps

- On-premise, limited DevOps → Layered, MVC
- Cloud-native, mature DevOps → Microservices, Serverless
- Kubernetes, CI/CD tự động → Microservices, Service Mesh
- Limited ops → Serverless, Platform-as-a-Service

### Bước 6: Phân tích risk và trade-offs

| Quyết định | Risk | Mitigation |
|-----------|------|------------|
| Chọn Microservices | Distributed complexity, network latency | Start với modular monolith, split sau |
| Chọn Event-Driven | Eventually consistency, debugging khó | Event sourcing + saga pattern |
| Chọn Serverless | Vendor lock-in, cold start | Abstraction layer (adapter pattern) |
| Chọn Hexagonal | Over-engineering nếu domain đơn giản | Chỉ dùng cho domain phức tạp |
| Chọn Layered | Không scale được, maintenance khó | Modular monolith như bước đệm |

### Bước 7: Iterate và học hỏi

Kiến trúc không phải là quyết định một lần. Hãy áp dụng **Evolutionary Architecture** — bắt đầu đơn giản, đo lường, và tiến hóa. Như Martin Fowler đã nói: *"If you can't decide between two architectures, choose the simpler one and refactor when you learn more."*

---

## Lộ trình học

Series này được thiết kế để bạn có thể đọc theo bất kỳ thứ tự nào, nhưng tôi đề xuất lộ trình sau dựa trên mức độ phức tạp và sự phụ thuộc giữa các bài:

### Giai đoạn 1: Nền tảng (Bài 35-39)

Bắt đầu với những kiến trúc nền tảng mà hầu hết developer đều gặp hàng ngày:

```
35. Giới thiệu Kiến trúc ← Bạn đang ở đây
36. Layered Architecture (N-Tier) ← Kiến trúc phổ biến nhất
37. Microservices Architecture ← Xu hướng cloud-native
38. Event-Driven Architecture (EDA) ← Real-time, async
39. Hexagonal Architecture ← DDD, testability
```

### Giai đoạn 2: Service & Message (Bài 40-44)

Sau khi nắm vững các kiến trúc nền tảng, chuyển sang các mô hình service và message:

```
40. Client-Server Architecture
41. MVC / MVP / MVVM
42. Repository Architecture
43. Service-Oriented Architecture (SOA)
44. Pub/Sub Architecture
```

### Giai đoạn 3: Advanced Patterns (Bài 45-49)

Các kiến trúc nâng cao đòi hỏi hiểu biết sâu về distributed systems:

```
45. CQRS (Command Query Responsibility Segregation)
46. Event Sourcing
47. Clean Architecture
48. Onion Architecture
49. Pipe-and-Filter Architecture
```

### Giai đoạn 4: Modern & Scalable (Bài 50-54)

Các kiến trúc hiện đại cho hệ thống lớn:

```
50. Domain-Driven Design (DDD)
51. Serverless Architecture
52. Backend-for-Frontend (BFF)
53. Service Mesh
54. Space-Based Architecture & Data Mesh
```

---

## Kiến thức nền tảng cần có

Để hiểu sâu series này, bạn nên có:

### Bắt buộc (Must-have)
- **OOP**: Class, inheritance, polymorphism, encapsulation
- **Python**: Type hints, dataclasses, ABC, enums, generators
- **Design Patterns**: Đặc biệt là Adapter, Facade, Observer, Strategy (đã có trong series Design Patterns)
- **SOLID Principles**: Đặc biệt là DIP (Dependency Inversion) — nền tảng của Hexagonal và Clean Architecture
- **HTTP/REST**: Request-response cycle, status codes, API design
- **Database**: SQL, ACID, transactions, indexing

### Khuyến khích (Nice-to-have)
- **Distributed Systems**: CAP theorem, consistency models, consensus algorithms
- **Message Queues**: RabbitMQ, Kafka concepts
- **Docker/Kubernetes**: Container orchestration
- **Cloud Services**: AWS, GCP, Azure basics
- **Testing**: Unit test, integration test, test doubles

### Tài liệu tham khảo cốt lõi

Nếu bạn muốn đọc thêm từ các nguồn uy tín:

| Tác giả | Sách | Năm |
|---------|------|-----|
| Martin Fowler | *Patterns of Enterprise Application Architecture* | 2002 |
| Eric Evans | *Domain-Driven Design: Tackling Complexity in the Heart of Software* | 2003 |
| Robert C. Martin | *Clean Architecture: A Craftsman's Guide* | 2017 |
| Sam Newman | *Building Microservices* | 2015, 2021 (2nd) |
| Gregor Hohpe | *Enterprise Integration Patterns* | 2003 |
| Vaughn Vernon | *Implementing Domain-Driven Design* | 2013 |
| Chris Richardson | *Microservices Patterns* | 2018 |
| Neal Ford | *Building Evolutionary Architectures* | 2017 |
| Len Bass | *Software Architecture in Practice* (4th ed.) | 2021 |

---

## Cấu trúc chung của mỗi bài viết

Mỗi kiến trúc trong series này đều tuân theo cấu trúc thống nhất:

1. **Tổng quan**: Định nghĩa, nguồn gốc, tác giả nổi tiếng
2. **Bài toán thực tế**: Vấn đề cụ thể mà kiến trúc giải quyết
3. **Nguyên lý thiết kế**: Core concepts, invariant rules
4. **Cấu trúc chi tiết**: Components, responsibilities
5. **Sơ đồ kiến trúc**: ASCII diagram trực quan
6. **Ví dụ code hoàn chỉnh**: Python 3.10+, production-quality, runnable
7. **So sánh và lựa chọn**: Khi nào dùng / Khi nào không dùng
8. **Ưu điểm và nhược điểm**: Bảng so sánh chi tiết
9. **Công cụ và Framework**: Tools thực tế cho từng kiến trúc
10. **Kiểm thử**: Chiến lược test với pytest
11. **Kết luận**: Best practices, golden rules

---

## Kết luận

Kiến trúc phần mềm là một hành trình, không phải đích đến. 20 kiến trúc trong series này không phải là những công thức cứng nhắc mà là những công cụ tư duy — mỗi kiến trúc là một cách nhìn khác nhau về cách tổ chức phần mềm, với những trade-offs riêng. Người kiến trúc sư giỏi không phải người thuộc lòng tất cả kiến trúc, mà là người biết chọn đúng kiến trúc cho đúng bối cảnh.

Hãy bắt đầu với bài tiếp theo: **Layered Architecture (N-Tier)** — kiến trúc phổ biến nhất và là nền tảng để hiểu các kiến trúc phức tạp hơn.

> *"The first 90% of the code accounts for the first 90% of the development time. The remaining 10% of the code accounts for the other 90% of the development time."* — **Tom Cargill, Bell Labs**
