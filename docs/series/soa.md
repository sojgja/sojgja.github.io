---
id: soa
title: Service-Oriented Architecture (SOA)
sidebar_label: 🏗️ SOA Architecture
sidebar_position: 48
---

# Service-Oriented Architecture (SOA)

> "Services are the building blocks of SOA. Each service is a self-contained, reusable unit of functionality that can be composed to create complex business processes."
> — **Thomas Erl**, *SOA: Principles of Service Design* (2007)

**Service-Oriented Architecture (SOA)** là một phong cách kiến trúc phần mềm trong đó các chức năng của ứng dụng được tổ chức thành các **dịch vụ** (services) độc lập, có tính **tái sử dụng cao**, và giao tiếp với nhau qua mạng bằng các **protocol chuẩn** (SOAP, REST, gRPC). Mỗi service là một đơn vị business logic hoàn chỉnh, có thể được phát triển, triển khai, và bảo trì độc lập.

---

## Tổng quan

### Lịch sử và nguồn gốc

SOA không phải là một khái niệm mới. Nó là sự hội tụ của nhiều xu hướng kiến trúc trước đó:

- **1980s**: CORBA (Common Object Request Broker Architecture) — distributed objects
- **1990s**: DCOM, RMI (Java) — remote method invocation
- **2000**: XML-RPC, SOAP — web services đầu tiên
- **2001**: WSDL, UDDI — service description và discovery
- **2004-2007**: **Thomas Erl** xuất bản loạt sách về SOA, định hình các nguyên lý
- **2005**: Enterprise Service Bus (ESB) ra đời
- **2008**: RESTful services trở nên phổ biến
- **2010-2014**: Microservices bắt đầu thay thế SOA cho nhiều use case
- **2015-nay**: SOA vẫn tồn tại trong enterprise, kết hợp với microservices và API management

### Những người tiên phong

| Tên | Đóng góp |
|-----|---------|
| **Thomas Erl** | Tác giả bộ sách *SOA Principles of Service Design* — cha đẻ của SOA principles |
| **Grady Booch** | Phát triển Unified Modeling Language (UML) cho service design |
| **Roy Fielding** | Tác giả REST — kiến trúc Web services hiện đại |
| **Clemens Szyperski** | Nghiên cứu về component software và service composition |
| **Martin Fowler** | Phân tích SOA, microservices, và sự khác biệt |
| **Sam Newman** | Cầu nối giữa SOA và microservices |

### SOA vs Microservices

Nhiều người nhầm lẫn SOA và Microservices. Thực tế:

| Tiêu chí | SOA | Microservices |
|---------|-----|--------------|
| **Kích thước service** | Lớn (coarse-grained) | Nhỏ (fine-grained) |
| **Giao tiếp** | SOAP, ESB, message queue | REST, gRPC, event-driven |
| **Data storage** | Shared database (thường) | Database per service |
| **Governance** | Centralized (ESB) | Decentralized |
| **Triển khai** | Enterprise service bus | Container (Docker/K8s) |
| **Reuse** | Cao (enterprise-wide) | Trong bounded context |
| **Độ phức tạp** | Rất cao | Trung bình-cao |
| **Phù hợp** | Enterprise lớn, legacy | Cloud-native, startup |

---

## Bài toán

### Hệ thống Ngân hàng Đa kênh (Omni-channel Banking)

Giả sử bạn là kiến trúc sư cho **VietGlobal Bank** — một ngân hàng lớn tại Việt Nam với:

1. **Kênh giao dịch truyền thống**: Quầy giao dịch (teller), ATM, Call center
2. **Kênh số**: Internet Banking, Mobile App, Zalo Mini App
3. **Kênh đối tác**: Ví điện tử (MoMo, ZaloPay), đối tác thanh toán (VNPay)
4. **Hệ thống lõi**: Core Banking System (Silverlake, T24), CRM, LOS
5. **Yêu cầu pháp lý**: NHNN compliance, AML, KYC, Basel II/III, PCI-DSS

### Khó khăn với kiến trúc nguyên khối (Monolith)

Một ngân hàng điển hình ban đầu xây dựng hệ thống dưới dạng **monolith** — mọi logic trong một ứng dụng lớn:

```python
# Monolith Banking — god class
class BankingSystem:
    def transfer_money(self, from_acc, to_acc, amount):
        # 1. Validate accounts
        # 2. Check balance
        # 3. Apply fees
        # 4. Check transaction limits
        # 5. Apply exchange rate (nếu ngoại tệ)
        # 6. Send OTP
        # 7. Verify OTP
        # 8. Execute transfer
        # 9. Update balances
        # 10. Send notification
        # 11. Log audit trail
        # 12. Report to NHNN
        # ... 300 dòng code
```

**Vấn đề 1 — Business functions bị duplicate**:

```python
# Module A: Internet Banking
def check_kyc_status(user_id):
    # Query user database
    # Check if KYC documents uploaded
    # Check if verified
    # Return status

# Module B: Mobile App  
def check_kyc_status(user_id):
    # SAME logic — đã copy-paste
    # Query user database
    # Check if KYC documents uploaded  
    # Check if verified
    # Return status
```

Khi NHNN thay đổi quy định KYC, phải update ở 5 modules khác nhau. Một lần quên update = compliance violation.

**Vấn đề 2 — Không thể scale độc lập**:

```
Monolith: [Web + Mobile + API + Batch + Reporting] — 1 server
- Traffic cao: Transfer service quá tải, kéo theo cả Login chậm
- Batch processing: End-of-day batch chiếm 100% CPU → user không thể transfer
- Không thể scale chỉ Transfer service riêng
```

**Vấn đề 3 — Khó tích hợp với đối tác**:

```python
# Mỗi đối tác cần API riêng — monolithic khó mở
class PartnerIntegration:
    def momo_callback(self, data):
        # Parse Momo format
        pass
    
    def zalopay_callback(self, data):
        # Parse ZaloPay format (khác Momo)
        pass
    
    def vnpay_callback(self, data):
        # Parse VNPay format (khác cả hai)
        pass
```

**Vấn đề 4 — Không có service tái sử dụng**:

```
Module A: tự xây dựng Customer Service
Module B: tự xây dựng Customer Service (vì không biết A đã có)
Module C: tự xây dựng Customer Service
→ 3 implementations của cùng "Customer" concept
→ Không có single source of truth
→ Khi merge dữ liệu, phải transform giữa các format
```

### SOA giải quyết vấn đề

1. **Service reuse**: Mỗi service là single source of truth cho một business capability
2. **Independent scalability**: Scale từng service riêng (Transfer service 10 instances, Login chỉ 2)
3. **Standardized communication**: Tất cả service giao tiếp qua ESB/service mesh với protocol chuẩn
4. **Polyglot development**: Mỗi service có thể viết bằng ngôn ngữ khác nhau (Java cho Core, Python cho AML)
5. **Gradual modernization**: Từng bước tách monolith thành services
6. **Governance và security tập trung**: ESB xử lý auth, logging, transformation
7. **Business agility**: Thêm kênh mới chỉ cần integrate với services hiện có

---

## Nguyên lý thiết kế

### 1. Service Contract (Hợp đồng dịch vụ)

Mỗi service phải có một contract được định nghĩa rõ ràng:
- **Interface**: Các operations service cung cấp (WSDL, OpenAPI, protobuf)
- **Data schema**: Input/output format (XSD, JSON Schema, Avro)
- **Quality of Service**: SLA, latency, availability, throughput
- **Policies**: Authentication, rate limiting, versioning

```yaml
# Service Contract — OpenAPI 3.0
openapi: 3.0.0
info:
  title: Customer Service
  version: 2.1.0
  description: Quản lý thông tin khách hàng — single source of truth
paths:
  /customers/{id}:
    get:
      summary: Lấy thông tin khách hàng
      parameters:
        - name: id
          in: path
          required: true
          schema: { type: string }
      responses:
        '200':
          description: Customer object
```

### 2. Service Loose Coupling (Liên kết lỏng)

Services phải độc lập nhất có thể:
- **Technology coupling**: Service A (Java) không biết Service B (Python) chạy công nghệ gì
- **Time coupling**: Giao tiếp async qua message queue — không cần response ngay lập tức
- **Location coupling**: Service chỉ biết endpoint, không biết physical location
- **Data coupling**: Mỗi service sở hữu dữ liệu của riêng mình

### 3. Service Abstraction (Trừu tượng hóa)

Service ẩn chi tiết implementation:
- Internal logic: Không ai biết service dùng database gì, thuật toán gì
- Internal data: Service không expose raw database schema
- Internal dependencies: Service có thể gọi service khác mà client không biết

```python
# Client chỉ thấy:
customer = customer_service.get_customer("123")

# Client không thấy:
# - customer_service gọi CRM database
# - customer_service gọi KYC service
# - customer_service transform dữ liệu từ 3 nguồn
```

### 4. Service Reusability (Tái sử dụng)

Service phải được thiết kế để tái sử dụng bởi nhiều consumers:
- **Generic interface**: Không thiết kế cho một consumer cụ thể
- **Stateless**: Càng stateless càng dễ reuse
- **Idempotent**: Có thể gọi nhiều lần với cùng kết quả
- **Discoverable**: Service registry để các service khác tìm thấy

### 5. Service Autonomy (Tự chủ)

Service phải tự quản lý:
- **Own database**: Không share database với service khác
- **Own deployment**: Có thể deploy độc lập
- **Own lifecycle**: Phát triển, test, deploy, scale riêng

### 6. Service Statelessness

Service không giữ state giữa các request:
- State lưu trong database, không trong memory
- Session state lưu ở client hoặc external store (Redis)
- Mỗi request là độc lập

### 7. Service Composability (Khả năng kết hợp)

Service có thể kết hợp để tạo business process phức tạp:
- **Choreography**: Service A gọi B, B gọi C (peer-to-peer)
- **Orchestration**: Service Orchestrator điều khiển luồng

### 8. Service Granularity

Service có kích thước phù hợp — không quá nhỏ (nanoservice) và không quá lớn (macroservice):
- Một service tương ứng với một business capability
- Có 3-10 operations per service
- Data model: 5-20 entities per service

---

## Cấu trúc chi tiết

### Các thành phần trong SOA

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ENTERPRISE SOA ARCHITECTURE                       │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  Channel  │  │  Channel │  │  Channel │  │  Channel │  │  Partner │ │
│  │   Teller  │  │  Mobile  │  │ Internet │  │   ATM    │  │   API    │ │
│  └─────┬─────┘  └─────┬────┘  └────┬─────┘  └─────┬────┘  └─────┬────┘ │
│        │              │            │              │              │      │
│  ┌─────┴──────────────┴────────────┴──────────────┴──────────────┴──┐   │
│  │                    ENTERPRISE SERVICE BUS (ESB)                    │   │
│  │  ┌─────────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐ │   │
│  │  │ Auth Gateway│  │  Router  │  │Transform   │  │  Protocol  │ │   │
│  │  │ (SSO, OAuth)│  │  (URI)   │  │(XML↔JSON)  │  │  Adapter   │ │   │
│  │  └─────────────┘  └──────────┘  └────────────┘  └────────────┘ │   │
│  │  ┌─────────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐ │   │
│  │  │ Rate Limiter│  │ Logging  │  │ Monitoring │  │  SLA       │ │   │
│  │  └─────────────┘  └──────────┘  └────────────┘  └────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ BUSINESS      │  │ BUSINESS     │  │ BUSINESS     │  │ BUSINESS   │ │
│  │ SERVICES      │  │ SERVICES     │  │ SERVICES     │  │ SERVICES   │ │
│  │               │  │              │  │              │  │            │ │
│  │ ┌───────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌────────┐ │ │
│  │ │ Customer  │ │  │ │ Account  │ │  │ │ Transfer │ │  │ │ Loan   │ │ │
│  │ │ Service   │ │  │ │ Service  │ │  │ │ Service  │ │  │ │ Service│ │ │
│  │ └───────────┘ │  │ └──────────┘ │  │ └──────────┘ │  │ └────────┘ │ │
│  │ ┌───────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌────────┐ │ │
│  │ │ KYC       │ │  │ │ Payment  │ │  │ │ FX       │ │  │ │ AML    │ │ │
│  │ │ Service   │ │  │ │ Service  │ │  │ │ Service  │ │  │ │ Service│ │ │
│  │ └───────────┘ │  │ └──────────┘ │  │ └──────────┘ │  │ └────────┘ │ │
│  └──────┬────────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                  │                 │                 │        │
│  ┌──────┴──────────────────┴─────────────────┴─────────────────┴──┐     │
│  │                    INFRASTRUCTURE SERVICES                       │     │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐  │     │
│  │  │ Service  │  │  Config  │  │  Identity  │  │ Audit &    │  │     │
│  │  │ Registry │  │  Server  │  │  Provider  │  │ Compliance │  │     │
│  │  └──────────┘  └──────────┘  └────────────┘  └────────────┘  │     │
│  └───────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Các loại Service trong SOA

**1. Business Services (Entity Services)**
- Đại diện cho business entity (Customer, Account, Loan)
- CRUD operations trên entity
- Single source of truth
- Ví dụ: `CustomerService`, `AccountService`

**2. Process Services (Task Services)**
- Orchestrate business processes
- Gọi nhiều business services
- Stateless (process state lưu trong database)
- Ví dụ: `TransferService`, `LoanOriginationService`

**3. Utility Services (Infrastructure Services)**
- Cung cấp chức năng kỹ thuật
- Không chứa business logic
- Ví dụ: `NotificationService`, `AuditService`, `FileService`

**4. Integration Services**
- Kết nối với legacy systems
- Transform data giữa các format
- Ví dụ: `CoreBankingAdapter`, `SWIFTAdapter`

### Enterprise Service Bus (ESB)

ESB là trung tâm của SOA truyền thống:

| Chức năng | Mô tả |
|-----------|-------|
| **Message routing** | Định tuyến request đến đúng service |
| **Protocol transformation** | SOAP ↔ REST ↔ JMS ↔ AMQP |
| **Data transformation** | XML → JSON, XSLT mapping |
| **Load balancing** | Phân phối request giữa multiple instances |
| **Failover** | Tự động chuyển sang service khác khi fail |
| **Security** | Authentication, authorization, encryption |
| **Monitoring** | SLA monitoring, logging, metrics |
| **Throttling** | Rate limiting, circuit breaker |
| **Transaction management** | Distributed transaction coordination |

---

## Sơ đồ kiến trúc

```
VIETGLOBAL BANK — SOA ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════

  CHANNEL LAYER
  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  ┌─────────────┐
  │ Web Portal  │  │ Mobile App  │  │ ATM/CLM   │  │ Partner API │
  │ (Angular)   │  │ (Flutter)   │  │ (C++)      │  │ (Momo... )  │
  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  └──────┬──────┘
         │                │               │                │
         ├────────────────┴───────────────┴────────────────┘
         │                    HTTPS / JSON
         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │               API GATEWAY / ESB (Kong / WSO2)                       │
  │                                                                     │
  │  Auth ──→ Rate Limiter ──→ Router ──→ Transform ──→ Log            │
  │  (OAuth2)  (1000/min)     (URI)       (XML↔JSON)    (Audit)        │
  └─────────────────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
  ┌───────────┐      ┌──────────────┐    ┌──────────────┐
  │ Customer  │      │   Account    │    │  Transfer    │
  │ Service   │      │   Service    │    │  Service     │
  ├───────────┤      ├──────────────┤    ├──────────────┤
  │ REST API  │      │  REST/SOAP   │    │  Async (JMS) │
  │ Python    │      │  Java/Spring │    │  C#/.NET     │
  │ DB: MySQL │      │  DB: Oracle  │    │  DB: SQLSvr  │
  └───────────┘      └──────────────┘    └──────────────┘
         │                    │                  │
         └────────────────────┼──────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                   SERVICE LAYER (Business)                          │
  │                                                                     │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
  │  │   KYC    │ │ Payment  │ │   FX     │ │   AML    │ │   Loan   │ │
  │  │ Service  │ │ Service  │ │ Service  │ │ Service  │ │ Service  │ │
  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
  └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    INFRASTRUCTURE SERVICES                          │
  │                                                                     │
  │  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────────┐  │
  │  │  Service   │ │   Config   │ │  Audit   │ │   Notification   │  │
  │  │  Registry  │ │   Server   │ │  Service │ │   Service        │  │
  │  │ (Consul)   │ │ (Spring)   │ │          │ │ (SMS/Email/Push) │  │
  │  └────────────┘ └────────────┘ └──────────┘ └──────────────────┘  │
  └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    LEGACY / CORE SYSTEMS                            │
  │                                                                     │
  │  ┌────────────────────┐   ┌────────────────────┐                   │
  │  │  Core Banking      │   │   CRM (Salesforce) │                   │
  │  │  (Silverlake T24)  │   │                    │                   │
  │  └────────────────────┘   └────────────────────┘                   │
  └─────────────────────────────────────────────────────────────────────┘

  DATA FLOW:
  ──────────
  Mobile App ──→ API Gateway ──→ Transfer Service ──→ Account Service
                                                      → Customer Service
                                                      → AML Service
                                                      → Notification Service
                                                      → Audit Service
```

---

## Ví dụ code hoàn chỉnh

### Cấu trúc project

```
vietglobal-bank/
├── services/
│   ├── customer_service/
│   │   ├── __init__.py
│   │   ├── api.py              # REST API endpoints
│   │   ├── service.py          # Business logic
│   │   ├── repository.py       # Database access
│   │   ├── models.py           # Domain models
│   │   ├── validators.py       # Business validators
│   │   └── requirements.txt
│   ├── account_service/
│   │   └── ...
│   ├── transfer_service/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── service.py
│   │   ├── orchestrator.py    # Process orchestrator
│   │   ├── validators.py
│   │   └── models.py
│   ├── kyc_service/
│   │   └── ...
│   ├── notification_service/
│   │   └── ...
│   └── audit_service/
│       └── ...
├── common/
│   ├── __init__.py
│   ├── event_bus.py           # Message queue abstraction
│   ├── service_registry.py    # Service discovery
│   ├── tracing.py             # Distributed tracing
│   └── serialization.py       # Common serialization
├── esb/
│   ├── __init__.py
│   ├── gateway.py             # API Gateway / ESB
│   ├── router.py              # Request router
│   └── transformer.py         # Data transformer
├── tests/
│   ├── test_customer_service.py
│   ├── test_transfer_service.py
│   └── integration/
└── docker-compose.yml
```

### services/customer_service/models.py

```python
"""Customer Service — domain models."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum, auto
from typing import Optional
import re


class CustomerType(Enum):
    INDIVIDUAL = auto()
    CORPORATE = auto()
    JOINT = auto()


class CustomerStatus(Enum):
    ACTIVE = auto()
    INACTIVE = auto()
    FROZEN = auto()
    CLOSED = auto()
    PENDING_KYC = auto()


class Gender(Enum):
    MALE = auto()
    FEMALE = auto()
    OTHER = auto()


class IDType(Enum):
    CCCD = auto()  # Căn cước công dân
    CMND = auto()  # Chứng minh nhân dân
    PASSPORT = auto()
    DRIVER_LICENSE = auto()


@dataclass
class Address:
    street: str
    ward: str
    district: str
    city: str
    country: str = "Việt Nam"
    is_primary: bool = True

    def full_address(self) -> str:
        return f"{self.street}, {self.ward}, {self.district}, {self.city}"


@dataclass
class Customer:
    """Customer entity — single source of truth cho thông tin khách hàng."""
    customer_id: str
    customer_type: CustomerType
    status: CustomerStatus

    # Personal info
    full_name: str
    date_of_birth: date
    gender: Gender
    email: str
    phone: str

    # Identity
    id_type: IDType
    id_number: str
    id_issue_date: date
    id_expiry_date: date
    nationality: str = "Việt Nam"

    # Address
    addresses: list[Address] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    risk_rating: str = "low"  # low, medium, high
    kyc_completed: bool = False
    kyc_completed_at: Optional[datetime] = None

    @property
    def age(self) -> int:
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )

    @property
    def is_verified(self) -> bool:
        return self.kyc_completed and self.status == CustomerStatus.ACTIVE

    def to_dict(self) -> dict:
        result = asdict(self)
        result['customer_type'] = self.customer_type.name
        result['status'] = self.status.name
        result['gender'] = self.gender.name
        result['id_type'] = self.id_type.name
        result['date_of_birth'] = self.date_of_birth.isoformat()
        result['id_issue_date'] = self.id_issue_date.isoformat()
        result['id_expiry_date'] = self.id_expiry_date.isoformat()
        return result


class CustomerValidationError(Exception):
    pass


class CustomerValidator:
    """Business validators cho Customer service."""

    @staticmethod
    def validate_email(email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_phone(phone: str) -> bool:
        # Vietnam phone: 10 digits, starts with 03/05/07/08/09
        pattern = r'^(0[35789])[0-9]{8}$'
        return bool(re.match(pattern, phone))

    @staticmethod
    def validate_id_number(id_type: IDType, number: str) -> bool:
        if id_type == IDType.CCCD:
            return len(number) == 12 and number.isdigit()
        elif id_type == IDType.CMND:
            return len(number) in (9, 12) and number.isdigit()
        elif id_type == IDType.PASSPORT:
            return bool(re.match(r'^[A-Z]{1,2}[0-9]{6,8}$', number))
        return True

    @staticmethod
    def validate_customer(customer: Customer) -> list[str]:
        errors: list[str] = []

        if not customer.full_name or len(customer.full_name.strip()) < 2:
            errors.append("Tên khách hàng phải có ít nhất 2 ký tự")

        if not CustomerValidator.validate_email(customer.email):
            errors.append(f"Email không hợp lệ: {customer.email}")

        if not CustomerValidator.validate_phone(customer.phone):
            errors.append(f"Số điện thoại không hợp lệ: {customer.phone}")

        if not CustomerValidator.validate_id_number(customer.id_type, customer.id_number):
            errors.append(f"Số giấy tờ không hợp lệ: {customer.id_number}")

        if customer.date_of_birth > date.today():
            errors.append("Ngày sinh không thể trong tương lai")

        if customer.age < 18:
            errors.append("Khách hàng phải đủ 18 tuổi")

        if customer.id_expiry_date < date.today():
            errors.append("Giấy tờ tùy thân đã hết hạn")

        return errors
```

### services/customer_service/repository.py

```python
"""Customer Service — data repository (database access)."""

from __future__ import annotations

from typing import Optional, Sequence
from datetime import datetime
import threading
import uuid

from .models import (
    Customer, CustomerType, CustomerStatus, Gender, IDType, Address,
)


class CustomerRepository:
    """Repository cho Customer data.

    Mô phỏng database access. Trong thực tế: SQLAlchemy, Django ORM, JDBC.
    Service sở hữu database riêng — không share với service khác.
    """

    def __init__(self) -> None:
        self._customers: dict[str, Customer] = {}
        self._lock = threading.Lock()

    def save(self, customer: Customer) -> Customer:
        with self._lock:
            if not customer.customer_id:
                customer.customer_id = str(uuid.uuid4())
            customer.updated_at = datetime.now()
            self._customers[customer.customer_id] = customer
            return customer

    def get_by_id(self, customer_id: str) -> Optional[Customer]:
        return self._customers.get(customer_id)

    def get_by_email(self, email: str) -> Optional[Customer]:
        for c in self._customers.values():
            if c.email == email:
                return c
        return None

    def get_by_id_number(self, id_number: str) -> Optional[Customer]:
        for c in self._customers.values():
            if c.id_number == id_number:
                return c
        return None

    def get_by_phone(self, phone: str) -> Optional[Customer]:
        for c in self._customers.values():
            if c.phone == phone:
                return c
        return None

    def update_status(self, customer_id: str, status: CustomerStatus) -> Optional[Customer]:
        with self._lock:
            customer = self._customers.get(customer_id)
            if customer:
                customer.status = status
                customer.updated_at = datetime.now()
            return customer

    def search(
        self,
        name_contains: Optional[str] = None,
        status: Optional[CustomerStatus] = None,
        customer_type: Optional[CustomerType] = None,
        limit: int = 50,
    ) -> Sequence[Customer]:
        results = list(self._customers.values())

        if name_contains:
            name_lower = name_contains.lower()
            results = [c for c in results if name_lower in c.full_name.lower()]

        if status:
            results = [c for c in results if c.status == status]

        if customer_type:
            results = [c for c in results if c.customer_type == customer_type]

        return results[:limit]

    def count_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self._customers.values():
            status = c.status.name
            counts[status] = counts.get(status, 0) + 1
        return counts

    def seed_data(self) -> None:
        """Seed sample data for demo."""
        sample = Customer(
            customer_id="CUST001",
            customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.ACTIVE,
            full_name="Nguyễn Văn An",
            date_of_birth=date(1990, 5, 15),
            gender=Gender.MALE,
            email="an.nguyen@email.com",
            phone="0912345678",
            id_type=IDType.CCCD,
            id_number="079090005123",
            id_issue_date=date(2020, 1, 10),
            id_expiry_date=date(2030, 1, 10),
            addresses=[
                Address("123 Nguyễn Huệ", "Bến Nghé", "Quận 1", "TP.HCM"),
            ],
            kyc_completed=True,
        )
        self.save(sample)
```

### services/customer_service/service.py

```python
"""Customer Service — business logic layer.

Service này là single source of truth cho thông tin khách hàng.
Tất cả các service khác (Account, Transfer, KYC) đều gọi qua đây.
"""

from __future__ import annotations

from typing import Optional, Sequence
from datetime import datetime

from .models import (
    Customer, CustomerStatus, CustomerType, CustomerValidationError, CustomerValidator,
)
from .repository import CustomerRepository


class CustomerService:
    """Core business logic cho Customer.

    Service này:
    - Giữ toàn bộ business rules về customer
    - Tương tác với database qua Repository
    - Không biết gì về HTTP/REST — chỉ xử lý domain logic
    """

    def __init__(self, repository: CustomerRepository) -> None:
        self._repo = repository

    def create_customer(self, customer: Customer) -> Customer:
        """Tạo khách hàng mới với validation."""
        # Validate
        errors = CustomerValidator.validate_customer(customer)
        if errors:
            raise CustomerValidationError("\n".join(errors))

        # Check duplicate
        if self._repo.get_by_email(customer.email):
            raise CustomerValidationError(f"Email {customer.email} đã tồn tại")
        if self._repo.get_by_id_number(customer.id_number):
            raise CustomerValidationError(f"Số giấy tờ {customer.id_number} đã tồn tại")

        return self._repo.save(customer)

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        return self._repo.get_by_id(customer_id)

    def get_customer_by_email(self, email: str) -> Optional[Customer]:
        return self._repo.get_by_email(email)

    def update_status(self, customer_id: str, status: CustomerStatus) -> Optional[Customer]:
        """Cập nhật trạng thái khách hàng.

        Gọi bởi KYC service, AML service, hoặc admin.
        """
        customer = self._repo.get_by_id(customer_id)
        if not customer:
            return None

        # Business rules for status transitions
        valid_transitions = {
            CustomerStatus.PENDING_KYC: [CustomerStatus.ACTIVE, CustomerStatus.FROZEN],
            CustomerStatus.ACTIVE: [CustomerStatus.INACTIVE, CustomerStatus.FROZEN],
            CustomerStatus.FROZEN: [CustomerStatus.ACTIVE, CustomerStatus.CLOSED],
            CustomerStatus.INACTIVE: [CustomerStatus.ACTIVE, CustomerStatus.CLOSED],
        }

        allowed = valid_transitions.get(customer.status, [])
        if status not in allowed:
            raise CustomerValidationError(
                f"Không thể chuyển từ {customer.status.name} sang {status.name}"
            )

        return self._repo.update_status(customer_id, status)

    def complete_kyc(self, customer_id: str) -> Customer:
        """Hoàn tất KYC — gọi bởi KYC service."""
        customer = self._repo.get_by_id(customer_id)
        if not customer:
            raise CustomerValidationError(f"Không tìm thấy customer {customer_id}")

        customer.kyc_completed = True
        customer.kyc_completed_at = datetime.now()
        customer.status = CustomerStatus.ACTIVE
        return self._repo.save(customer)

    def search_customers(
        self,
        name: Optional[str] = None,
        status: Optional[CustomerStatus] = None,
        customer_type: Optional[CustomerType] = None,
    ) -> Sequence[Customer]:
        return self._repo.search(name, status, customer_type)

    def get_customer_summary(self, customer_id: str) -> Optional[dict]:
        """Lấy thông tin tóm tắt cho các service khác.

        Các service khác không cần toàn bộ Customer object.
        Service cung cấp DTO (Data Transfer Object) chứa vừa đủ thông tin.
        """
        customer = self._repo.get_by_id(customer_id)
        if not customer:
            return None

        return {
            "customer_id": customer.customer_id,
            "full_name": customer.full_name,
            "email": customer.email,
            "phone": customer.phone,
            "status": customer.status.name,
            "is_verified": customer.is_verified,
            "risk_rating": customer.risk_rating,
        }
```

### services/customer_service/api.py

```python
"""Customer Service — REST API endpoints."""

from __future__ import annotations

import json
from typing import Any
from http.server import BaseHTTPRequestHandler
from datetime import date

from .models import (
    Customer, CustomerStatus, CustomerType, Gender, IDType,
    Address, CustomerValidationError,
)
from .service import CustomerService
from .repository import CustomerRepository


# Singleton — trong thực tế dùng DI container
_repo = CustomerRepository()
_repo.seed_data()
_service = CustomerService(_repo)


class CustomerAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler cho Customer Service.

    Các endpoints:
    GET    /customers/{id}          — Lấy thông tin KH
    POST   /customers               — Tạo KH mới
    GET    /customers/search        — Tìm kiếm KH
    PATCH  /customers/{id}/status   — Cập nhật trạng thái
    POST   /customers/{id}/kyc      — Hoàn tất KYC
    """

    def do_GET(self) -> None:
        if self.path.startswith("/customers/search"):
            self._handle_search()
        elif self.path.startswith("/customers/"):
            customer_id = self.path.split("/")[-1]
            self._handle_get_customer(customer_id)
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/customers":
            self._handle_create_customer()
        elif "/kyc" in self.path:
            customer_id = self.path.split("/")[-2]
            self._handle_complete_kyc(customer_id)
        else:
            self._send_json(404, {"error": "Not found"})

    def do_PATCH(self) -> None:
        if "/status" in self.path:
            customer_id = self.path.split("/")[-2]
            self._handle_update_status(customer_id)
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_get_customer(self, customer_id: str) -> None:
        summary = _service.get_customer_summary(customer_id)
        if summary:
            self._send_json(200, summary)
        else:
            self._send_json(404, {"error": "Customer not found"})

    def _handle_create_customer(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            customer = Customer(
                customer_id=data.get("customer_id", ""),
                customer_type=CustomerType[data.get("customer_type", "INDIVIDUAL")],
                status=CustomerStatus.PENDING_KYC,
                full_name=data["full_name"],
                date_of_birth=date.fromisoformat(data["date_of_birth"]),
                gender=Gender[data.get("gender", "MALE")],
                email=data["email"],
                phone=data["phone"],
                id_type=IDType[data.get("id_type", "CCCD")],
                id_number=data["id_number"],
                id_issue_date=date.fromisoformat(data["id_issue_date"]),
                id_expiry_date=date.fromisoformat(data["id_expiry_date"]),
            )

            created = _service.create_customer(customer)
            self._send_json(201, created.to_dict())

        except (KeyError, ValueError) as e:
            self._send_json(400, {"error": f"Invalid request: {e}"})
        except CustomerValidationError as e:
            self._send_json(422, {"error": str(e)})

    def _handle_search(self) -> None:
        from urllib.parse import urlparse, parse_qs
        params = parse_qs(urlparse(self.path).query)

        name = params.get("name", [None])[0]
        status_str = params.get("status", [None])[0]
        status = CustomerStatus[status_str] if status_str else None

        customers = _service.search_customers(name=name, status=status)
        self._send_json(200, {
            "total": len(customers),
            "customers": [c.to_dict() for c in customers],
        })

    def _handle_update_status(self, customer_id: str) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            new_status = CustomerStatus[data["status"]]
            customer = _service.update_status(customer_id, new_status)

            if customer:
                self._send_json(200, customer.to_dict())
            else:
                self._send_json(404, {"error": "Customer not found"})

        except (KeyError, CustomerValidationError) as e:
            self._send_json(422, {"error": str(e)})

    def _handle_complete_kyc(self, customer_id: str) -> None:
        try:
            customer = _service.complete_kyc(customer_id)
            self._send_json(200, customer.to_dict())
        except CustomerValidationError as e:
            self._send_json(404, {"error": str(e)})

    def _send_json(self, status: int, data: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Service-Name", "customer-service")
        self.send_header("X-Service-Version", "2.1.0")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())
```

### services/transfer_service/orchestrator.py

```python
"""Transfer Service — Process Orchestrator.

Service này orchestrate quy trình chuyển tiền:
1. Validate accounts (gọi Account Service)
2. Check balance (gọi Account Service)
3. Apply fees (tính toán nội bộ)
4. Check AML (gọi AML Service)
5. Execute transfer (gọi Account Service)
6. Send notification (gọi Notification Service)
7. Log audit (gọi Audit Service)

Đây là Process Service — nó không sở hữu dữ liệu,
mà orchestrate các Business Services khác.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional
from datetime import datetime
import uuid


class TransferStatus(Enum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REVERSED = auto()


class TransferType(Enum):
    INTERNAL = auto()       # Cùng ngân hàng
    EXTERNAL = auto()       # Khác ngân hàng (Napas)
    CROSS_BORDER = auto()   # Chuyển tiền quốc tế
    SAME_OWNER = auto()     # Cùng chủ tài khoản


@dataclass
class TransferRequest:
    transaction_id: str = field(default_factory=lambda: f"TXN{uuid.uuid4().hex[:8].upper()}")
    from_account: str
    to_account: str
    amount: Decimal
    currency: str = "VND"
    transfer_type: TransferType = TransferType.INTERNAL
    description: str = ""
    fee_amount: Decimal = Decimal("0")
    fee_payer: str = "sender"  # sender, receiver
    exchange_rate: Decimal = Decimal("1")
    otp_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""


@dataclass
class TransferResult:
    transaction_id: str
    status: TransferStatus
    from_account: str
    to_account: str
    amount: Decimal
    fee_amount: Decimal
    total_debit: Decimal
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    steps_completed: list[str] = field(default_factory=list)


class TransferOrchestrator:
    """Orchestrate quy trình chuyển tiền.

    Orchestrator gọi các service khác qua interface (simulated).
    Trong thực tế: HTTP requests, message queue, gRPC.
    """

    def __init__(
        self,
        account_service: Any,
        aml_service: Any,
        notification_service: Any,
        audit_service: Any,
        fx_service: Any,
    ) -> None:
        self._account_service = account_service
        self._aml_service = aml_service
        self._notification_service = notification_service
        self._audit_service = audit_service
        self._fx_service = fx_service

    def execute_transfer(self, request: TransferRequest) -> TransferResult:
        """Execute transfer với đầy đủ các bước."""
        steps = []
        result = TransferResult(
            transaction_id=request.transaction_id,
            status=TransferStatus.PROCESSING,
            from_account=request.from_account,
            to_account=request.to_account,
            amount=request.amount,
            fee_amount=Decimal("0"),
            total_debit=Decimal("0"),
        )

        try:
            # Step 1: Validate accounts
            from_acct = self._account_service.get_account(request.from_account)
            if not from_acct:
                raise TransferError(f"Tài khoản nguồn {request.from_account} không tồn tại")

            to_acct = self._account_service.get_account(request.to_account)
            if not to_acct:
                raise TransferError(f"Tài khoản đích {request.to_account} không tồn tại")

            if from_acct["status"] != "ACTIVE":
                raise TransferError(f"Tài khoản nguồn đang {from_acct['status']}")
            if to_acct["status"] != "ACTIVE":
                raise TransferError(f"Tài khoản đích đang {to_acct['status']}")

            steps.append("validate_accounts")

            # Step 2: Check balance
            fee = self._calculate_fee(request)
            total_debit = request.amount + (fee if request.fee_payer == "sender" else Decimal("0"))

            if from_acct["balance"] < total_debit:
                raise TransferError(
                    f"Số dư không đủ: {from_acct['balance']} < {total_debit}"
                )

            steps.append("check_balance")

            # Step 3: AML check
            aml_result = self._aml_service.screen_transaction(
                from_account=request.from_account,
                to_account=request.to_account,
                amount=request.amount,
                transfer_type=request.transfer_type,
            )
            if aml_result.get("flagged"):
                raise TransferError(
                    f"Giao dịch bị từ chối bởi AML: {aml_result.get('reason')}"
                )

            steps.append("aml_check")

            # Step 4: Exchange rate (if cross-currency)
            debit_amount = request.amount
            credit_amount = request.amount
            if request.transfer_type == TransferType.CROSS_BORDER:
                rate_info = self._fx_service.get_rate(request.currency, to_acct["currency"])
                credit_amount = request.amount * rate_info["rate"]
                request.exchange_rate = rate_info["rate"]
                steps.append("fx_conversion")

            # Step 5: Execute debit/credit
            self._account_service.debit(
                account_id=request.from_account,
                amount=total_debit,
                reference=request.transaction_id,
                description=f"Transfer to {request.to_account}",
            )
            if fee > 0 and request.fee_payer == "sender":
                fee_amount_db = fee
            else:
                fee_amount_db = Decimal("0")

            self._account_service.credit(
                account_id=request.to_account,
                amount=credit_amount,
                reference=request.transaction_id,
                description=f"Transfer from {request.from_account}",
            )

            steps.append("execute_transfer")

            # Step 6: Update result
            result.status = TransferStatus.COMPLETED
            result.fee_amount = fee
            result.total_debit = total_debit
            result.completed_at = datetime.now()
            result.steps_completed = steps

            # Step 7: Send notification (async)
            self._notification_service.send_notification(
                from_acct.get("customer_id"),
                "transfer_success",
                {
                    "amount": str(request.amount),
                    "to_account": request.to_account,
                    "transaction_id": request.transaction_id,
                },
            )

            # Step 8: Audit log
            self._audit_service.log(
                action="TRANSFER_EXECUTED",
                transaction_id=request.transaction_id,
                details={
                    "from": request.from_account,
                    "to": request.to_account,
                    "amount": str(request.amount),
                    "fee": str(fee),
                    "status": "COMPLETED",
                },
            )

            return result

        except TransferError as e:
            result.status = TransferStatus.FAILED
            result.error_message = str(e)
            result.steps_completed = steps

            # Audit failed attempt
            self._audit_service.log(
                action="TRANSFER_FAILED",
                transaction_id=request.transaction_id,
                details={
                    "error": str(e),
                    "steps_completed": steps,
                },
            )
            return result

    def _calculate_fee(self, request: TransferRequest) -> Decimal:
        if request.transfer_type == TransferType.INTERNAL:
            return Decimal("0")  # Internal transfer free
        elif request.transfer_type == TransferType.EXTERNAL:
            # Fee: 0.05% of amount, min 5,000, max 50,000
            base = request.amount * Decimal("0.0005")
            return max(Decimal("5000"), min(base, Decimal("50000")))
        elif request.transfer_type == TransferType.CROSS_BORDER:
            # Fee: 0.1% of amount, min 50,000, max 500,000
            base = request.amount * Decimal("0.001")
            return max(Decimal("50000"), min(base, Decimal("500000")))
        return Decimal("0")


class TransferError(Exception):
    pass


# Type hint cho Any (service interfaces)
from typing import Any
```

### services/notification_service/service.py

```python
"""Notification Service — gửi thông báo qua nhiều kênh.

Utility service — không chứa business logic, chỉ gửi notification.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


class NotificationChannel(Enum):
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    ZALO = "zalo"
    WEBHOOK = "webhook"


class NotificationPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Notification:
    notification_id: str = ""
    channel: NotificationChannel = NotificationChannel.EMAIL
    recipient: str = ""
    template: str = ""
    params: dict = field(default_factory=dict)
    priority: NotificationPriority = NotificationPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    status: str = "pending"


class NotificationService:
    """Gửi notification qua nhiều channels.

    Service này được gọi bởi tất cả các service khác.
    Nó không biết gì về business logic — chỉ gửi thông báo.
    """

    def __init__(self) -> None:
        self._history: list[Notification] = []

    def send_notification(
        self,
        customer_id: str,
        template: str,
        params: dict,
        channel: NotificationChannel = NotificationChannel.PUSH,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> str:
        """Gửi notification cho khách hàng."""
        notification = Notification(
            channel=channel,
            recipient=customer_id,
            template=template,
            params=params,
            priority=priority,
        )

        # Simulate sending
        notification.status = "sent"
        notification.sent_at = datetime.now()
        self._history.append(notification)

        print(f"[Notification] {channel.value.upper()} → {customer_id}: {template}")
        return notification.notification_id

    def send_otp(self, customer_id: str, phone: str, otp: str) -> str:
        """Gửi OTP qua SMS."""
        return self.send_notification(
            customer_id=customer_id,
            template="otp_sms",
            params={"otp": otp, "phone": phone},
            channel=NotificationChannel.SMS,
            priority=NotificationPriority.URGENT,
        )

    def send_transfer_confirmation(
        self,
        customer_id: str,
        amount: str,
        to_account: str,
        transaction_id: str,
    ) -> str:
        """Gửi xác nhận chuyển tiền."""
        return self.send_notification(
            customer_id=customer_id,
            template="transfer_success",
            params={
                "amount": amount,
                "to_account": to_account,
                "transaction_id": transaction_id,
            },
        )

    def get_notification_history(self, customer_id: str) -> list[Notification]:
        return [n for n in self._history if n.recipient == customer_id]
```

### common/service_registry.py

```python
"""Service Registry — service discovery cho SOA.

Trong thực tế: Consul, Eureka, ZooKeeper, Kubernetes DNS.
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import random


@dataclass
class ServiceInstance:
    """Một instance của service."""
    service_name: str
    instance_id: str
    host: str
    port: int
    protocol: str = "http"
    health_check_url: str = "/health"
    version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True


class ServiceRegistry:
    """Service registry cho SOA.

    Services đăng ký khi khởi động.
    Các service khác lookup để tìm endpoint.
    Hỗ trợ load balancing (random, round-robin).
    """

    def __init__(self) -> None:
        self._services: dict[str, list[ServiceInstance]] = {}

    def register(self, instance: ServiceInstance) -> None:
        """Đăng ký service instance."""
        if instance.service_name not in self._services:
            self._services[instance.service_name] = []
        self._services[instance.service_name].append(instance)

    def unregister(self, service_name: str, instance_id: str) -> None:
        """Hủy đăng ký service instance."""
        instances = self._services.get(service_name, [])
        self._services[service_name] = [
            i for i in instances if i.instance_id != instance_id
        ]

    def discover(self, service_name: str) -> Optional[ServiceInstance]:
        """Tìm một instance healthy của service."""
        instances = self._services.get(service_name, [])
        healthy = [i for i in instances if i.is_healthy]
        if not healthy:
            return None
        return random.choice(healthy)

    def discover_all(self, service_name: str) -> list[ServiceInstance]:
        """Tìm tất cả instances của service."""
        return self._services.get(service_name, [])

    def health_check(self, service_name: str, instance_id: str) -> bool:
        """Health check — cập nhật heartbeat."""
        for inst in self._services.get(service_name, []):
            if inst.instance_id == instance_id:
                inst.last_heartbeat = datetime.now()
                inst.is_healthy = True
                return True
        return False

    def mark_unhealthy(self, service_name: str, instance_id: str) -> None:
        """Mark instance as unhealthy."""
        for inst in self._services.get(service_name, []):
            if inst.instance_id == instance_id:
                inst.is_healthy = False
                break

    def get_all_services(self) -> dict[str, int]:
        """Liệt kê tất cả services và số instances."""
        return {name: len(instances) for name, instances in self._services.items()}


# Global registry — singleton
global_registry = ServiceRegistry()
```

### esb/gateway.py

```python
"""ESB Gateway — API Gateway cho SOA.

Gateway xử lý:
- Authentication (JWT validation)
- Rate limiting
- Request routing
- Protocol transformation (XML↔JSON)
- Logging và auditing
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading


@dataclass
class GatewayRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class GatewayResponse:
    status_code: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0


class RateLimiter:
    """Simple rate limiter — sliding window."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            # Clean old entries
            self._requests[client_id] = [
                t for t in self._requests[client_id]
                if now - t < self._window
            ]
            if len(self._requests[client_id]) >= self._max_requests:
                return False
            self._requests[client_id].append(now)
            return True


class Gateway:
    """ESB Gateway — routing, rate limiting, transformation."""

    def __init__(self) -> None:
        self._routes: dict[str, dict] = {}
        self._rate_limiter = RateLimiter(max_requests=1000)
        self._request_count = 0
        self._lock = threading.Lock()

    def add_route(
        self,
        path_pattern: str,
        target_service: str,
        target_path: str,
        method: str = "GET",
        auth_required: bool = True,
        rate_limit: int = 100,
    ) -> None:
        """Thêm route cho service."""
        self._routes[f"{method}:{path_pattern}"] = {
            "target_service": target_service,
            "target_path": target_path,
            "auth_required": auth_required,
            "rate_limit": rate_limit,
        }

    def handle_request(
        self,
        request: GatewayRequest,
        service_discovery: Callable[[str], Optional[str]],
    ) -> GatewayResponse:
        """Xử lý request qua gateway."""
        start = time.time()
        route_key = f"{request.method}:{request.path}"

        # Find route
        route = self._routes.get(route_key)
        if not route:
            return GatewayResponse(
                status_code=404,
                body={"error": f"No route for {request.method} {request.path}"},
            )

        # Rate limiting
        client_id = request.headers.get("X-Client-ID", "anonymous")
        if not self._rate_limiter.is_allowed(client_id):
            return GatewayResponse(status_code=429, body={"error": "Rate limit exceeded"})

        # Auth
        if route["auth_required"]:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return GatewayResponse(status_code=401, body={"error": "Unauthorized"})
            # In production: validate JWT token

        # Service discovery
        target_url = service_discovery(route["target_service"])
        if not target_url:
            return GatewayResponse(
                status_code=503,
                body={"error": f"Service {route['target_service']} unavailable"},
            )

        # Transform request (if needed)
        transformed_body = self._transform_request(
            request.body,
            source_format=request.headers.get("Content-Type", "application/json"),
        )

        duration = (time.time() - start) * 1000

        with self._lock:
            self._request_count += 1

        return GatewayResponse(
            status_code=200,
            body={
                "service": route["target_service"],
                "path": route["target_path"],
                "method": request.method,
                "transformed": transformed_body is not None,
                "duration_ms": round(duration, 2),
            },
            headers={
                "X-Gateway": "vietglobal-esb/1.0",
                "X-Request-Count": str(self._request_count),
            },
            duration_ms=duration,
        )

    def _transform_request(
        self,
        body: Any,
        source_format: str,
        target_format: str = "application/json",
    ) -> Any:
        """Transform request body giữa các format."""
        if source_format == target_format:
            return body
        # XML → JSON transformation
        # Trong thực tế: XSLT, JAXB
        return body

    def get_stats(self) -> dict:
        return {
            "total_requests": self._request_count,
            "routes": len(self._routes),
        }
```

### tests/test_customer_service.py

```python
"""Tests for Customer Service."""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

import sys
sys.path.insert(0, "../..")

from services.customer_service.models import (
    Customer, CustomerStatus, CustomerType, Gender, IDType,
    Address, CustomerValidator, CustomerValidationError,
)
from services.customer_service.service import CustomerService
from services.customer_service.repository import CustomerRepository


class TestCustomerModel(unittest.TestCase):
    def test_customer_creation(self):
        customer = Customer(
            customer_id="TEST001",
            customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.ACTIVE,
            full_name="Trần Thị B",
            date_of_birth=date(1995, 8, 20),
            gender=Gender.FEMALE,
            email="b.tran@email.com",
            phone="0987654321",
            id_type=IDType.CCCD,
            id_number="079095001234",
            id_issue_date=date(2021, 3, 15),
            id_expiry_date=date(2031, 3, 15),
        )
        self.assertEqual(customer.full_name, "Trần Thị B")
        self.assertEqual(customer.age, 30)  # In 2026

    def test_is_verified(self):
        customer = Customer(
            customer_id="T2", customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.ACTIVE, full_name="Test",
            date_of_birth=date(1990, 1, 1), gender=Gender.MALE,
            email="test@test.com", phone="0912345678",
            id_type=IDType.CCCD, id_number="123456789012",
            id_issue_date=date(2020, 1, 1), id_expiry_date=date(2030, 1, 1),
            kyc_completed=True,
        )
        self.assertTrue(customer.is_verified)

    def test_customer_not_verified(self):
        customer = Customer(
            customer_id="T3", customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.PENDING_KYC, full_name="Test",
            date_of_birth=date(1990, 1, 1), gender=Gender.MALE,
            email="test@test.com", phone="0912345678",
            id_type=IDType.CCCD, id_number="123456789012",
            id_issue_date=date(2020, 1, 1), id_expiry_date=date(2030, 1, 1),
            kyc_completed=False,
        )
        self.assertFalse(customer.is_verified)

    def test_address_full_address(self):
        addr = Address("1 Lê Lợi", "Bến Nghé", "Q.1", "TP.HCM")
        self.assertIn("1 Lê Lợi", addr.full_address())
        self.assertIn("TP.HCM", addr.full_address())


class TestCustomerValidator(unittest.TestCase):
    def test_valid_email(self):
        self.assertTrue(CustomerValidator.validate_email("test@example.com"))

    def test_invalid_email(self):
        self.assertFalse(CustomerValidator.validate_email("not-an-email"))

    def test_valid_phone(self):
        self.assertTrue(CustomerValidator.validate_phone("0912345678"))

    def test_invalid_phone(self):
        self.assertFalse(CustomerValidator.validate_phone("12345"))

    def test_cccd_valid(self):
        self.assertTrue(CustomerValidator.validate_id_number(IDType.CCCD, "079090005123"))

    def test_passport_valid(self):
        self.assertTrue(CustomerValidator.validate_id_number(IDType.PASSPORT, "B1234567"))

    def test_full_customer_validation_valid(self):
        customer = Customer(
            customer_id="", customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.PENDING_KYC, full_name="Nguyễn Văn A",
            date_of_birth=date(1990, 1, 1), gender=Gender.MALE,
            email="a@example.com", phone="0912345678",
            id_type=IDType.CCCD, id_number="079090005123",
            id_issue_date=date(2020, 1, 1), id_expiry_date=date(2030, 1, 1),
        )
        errors = CustomerValidator.validate_customer(customer)
        self.assertEqual(len(errors), 0)

    def test_full_customer_validation_invalid(self):
        customer = Customer(
            customer_id="", customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.PENDING_KYC, full_name="A",
            date_of_birth=date(2010, 1, 1), gender=Gender.MALE,
            email="invalid-email", phone="123",
            id_type=IDType.CCCD, id_number="abc",
            id_issue_date=date(2020, 1, 1), id_expiry_date=date(2020, 6, 1),
        )
        errors = CustomerValidator.validate_customer(customer)
        self.assertGreater(len(errors), 0)


class TestCustomerService(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = CustomerRepository()
        self.service = CustomerService(self.repo)

    def test_create_customer_success(self):
        customer = Customer(
            customer_id="", customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.PENDING_KYC, full_name="Lê Văn C",
            date_of_birth=date(1988, 3, 15), gender=Gender.MALE,
            email="c.le@email.com", phone="0911111111",
            id_type=IDType.CCCD, id_number="079088001234",
            id_issue_date=date(2020, 1, 1), id_expiry_date=date(2030, 1, 1),
        )
        created = self.service.create_customer(customer)
        self.assertIsNotNone(created.customer_id)
        self.assertEqual(created.status, CustomerStatus.PENDING_KYC)

    def test_create_duplicate_email(self):
        customer1 = Customer(
            customer_id="", customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.PENDING_KYC, full_name="Dup Email",
            date_of_birth=date(1990, 1, 1), gender=Gender.MALE,
            email="duplicate@email.com", phone="0922222222",
            id_type=IDType.CCCD, id_number="079077001234",
            id_issue_date=date(2020, 1, 1), id_expiry_date=date(2030, 1, 1),
        )
        self.service.create_customer(customer1)

        customer2 = Customer(
            customer_id="", customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.PENDING_KYC, full_name="Dup Again",
            date_of_birth=date(1990, 1, 1), gender=Gender.MALE,
            email="duplicate@email.com", phone="0933333333",
            id_type=IDType.CCCD, id_number="079066001234",
            id_issue_date=date(2020, 1, 1), id_expiry_date=date(2030, 1, 1),
        )
        with self.assertRaises(CustomerValidationError):
            self.service.create_customer(customer2)

    def test_get_customer_summary(self):
        self.repo.seed_data()
        summary = self.service.get_customer_summary("CUST001")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["full_name"], "Nguyễn Văn An")
        self.assertTrue(summary["is_verified"])

    def test_get_nonexistent_customer(self):
        result = self.service.get_customer("NONEXISTENT")
        self.assertIsNone(result)

    def test_update_status_valid(self):
        self.repo.seed_data()
        updated = self.service.update_status("CUST001", CustomerStatus.INACTIVE)
        self.assertEqual(updated.status, CustomerStatus.INACTIVE)

    def test_update_status_invalid_transition(self):
        self.repo.seed_data()
        with self.assertRaises(CustomerValidationError):
            # CLOSED → ACTIVE is not in the allowed transitions for INACTIVE
            # Actually from ACTIVE, we can go to INACTIVE or FROZEN
            # Let's test: can't go from ACTIVE directly to CLOSED
            self.service.update_status("CUST001", CustomerStatus.CLOSED)

    def test_complete_kyc(self):
        customer = Customer(
            customer_id="", customer_type=CustomerType.INDIVIDUAL,
            status=CustomerStatus.PENDING_KYC, full_name="KYC Test",
            date_of_birth=date(1990, 1, 1), gender=Gender.MALE,
            email="kyc@test.com", phone="0944444444",
            id_type=IDType.CCCD, id_number="079055001234",
            id_issue_date=date(2020, 1, 1), id_expiry_date=date(2030, 1, 1),
        )
        created = self.service.create_customer(customer)
        completed = self.service.complete_kyc(created.customer_id)
        self.assertTrue(completed.kyc_completed)
        self.assertEqual(completed.status, CustomerStatus.ACTIVE)

    def test_search_by_name(self):
        self.repo.seed_data()
        results = self.service.search_customers(name="An")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("Nguyễn Văn An", [c.full_name for c in results])


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng SOA | Khi nào không dùng SOA |
|-----------------|----------------------|
| **Enterprise lớn** — 10+ business units, 50+ applications | **Startup nhỏ** — 1-2 products, < 10 developers |
| **Legacy system modernization** — Từng bước migrate monolith | **Greenfield project nhỏ** — Monolith đơn giản hơn |
| **Cần reuse service** — Nhiều ứng dụng dùng chung Customer Service | **Không có governance** — SOA yêu cầu strong governance |
| **Multi-channel** — Web, Mobile, ATM, Partner API | **Performance-critical** — ESB overhead có thể là vấn đề |
| **Complex business processes** — Loan origination, Trade finance | **Tight deadline** — SOA implementation tốn thời gian |
| **Compliance yêu cầu audit** — NHNN, Basel, AML, KYC | **Team phân tán** — Cần coordination cho service contracts |
| **Polyglot environment** — Java, Python, C#, COBOL cùng tồn tại | **Không có ESB/Service Mesh** — Thiếu infrastructure layer |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| **Tái sử dụng service**: Giảm duplicate code, single source of truth | **ESB bottleneck**: ESB có thể trở thành single point of failure |
| **Independent scalability**: Mỗi service scale riêng | **Complexity cực cao**: Nhiều service, ESB, governance |
| **Polyglot**: Mỗi service dùng ngôn ngữ phù hợp | **Latency overhead**: Gọi qua ESB → network → transformation |
| **Gradual modernization**: Migrate từng phần | **Testing complexity**: Integration test 20+ services |
| **Governance tập trung**: Security, monitoring, SLA | **Vendor lock-in**: ESB thường là proprietary |
| **Business agility**: Thêm channel mới dễ dàng | **Team coordination**: Service contract negotiation |
| **Distributed ownership**: Mỗi team sở hữu service riêng | **Data distribution**: Distributed transaction khó |
| **Standardized communication**: SOAP/REST/gRPC chuẩn | **Versioning nightmare**: Nhiều consumers khó thay đổi API |
| **Compliance và audit**: Centralized logging | **Cost**: ESB license, infrastructure, team training |
| **Loose coupling**: Service thay đổi không ảnh hưởng nhau | **Over-engineering**: Không phải app nào cũng cần SOA |

---

## Công cụ và Framework

### ESB / Integration
| Công cụ | Mô tả |
|---------|-------|
| **WSO2 EI** | Open-source ESB, hỗ trợ SOAP/REST/GraphQL |
| **MuleSoft Anypoint** | Commercial ESB, API management, connectors |
| **IBM App Connect** | Enterprise ESB cho legacy integration |
| **Apache Camel** | Open-source integration framework |
| **Spring Integration** | Java-based EIP (Enterprise Integration Patterns) |
| **Oracle Service Bus** | ESB cho Oracle ecosystem |
| **Azure Logic Apps / Service Bus** | Cloud ESB của Microsoft |
| **Kong / Tyk** | API Gateway (không phải ESB đầy đủ) |

### Service Registry & Discovery
| Công cụ | Mô tả |
|---------|-------|
| **HashiCorp Consul** | Service discovery + health check + KV store |
| **Netflix Eureka** | Service registry (Spring Cloud ecosystem) |
| **Apache ZooKeeper** | Distributed coordination và service registry |
| **etcd** | Key-value store cho service discovery |
| **Kubernetes DNS** | Service discovery mặc định trong K8s |

### Message Queue
| Công cụ | Mô tả |
|---------|-------|
| **Apache Kafka** | Event streaming, message broker |
| **RabbitMQ** | AMQP message broker, phổ biến cho SOA |
| **ActiveMQ** | JMS-compatible message broker |
| **IBM MQ** | Enterprise message queue cho legacy |
| **AWS SQS / SNS** | Cloud message queue |

### SOA Governance
| Công cụ | Mô tả |
|---------|-------|
| **WSO2 Registry** | Service registry + governance |
| **IBM WebSphere Service Registry** | Enterprise SOA governance |
| **ServiceNow** | IT service management + SOA governance |
| **Apigee** (Google) | API management + analytics |

### Testing
| Công cụ | Mô tả |
|---------|-------|
| **SoapUI** | Testing SOAP/REST services |
| **Postman** | API testing và documentation |
| **Pact** | Consumer-driven contract testing |
| **WireMock** | Mock HTTP services cho integration test |
| **Karate** | API test automation framework |

---

## Kiểm thử

### Chiến lược kiểm thử SOA

SOA testing có 4 levels:

**1. Unit Tests — Từng service riêng lẻ**
```bash
python -m pytest services/customer_service/tests/ -v
python -m pytest services/transfer_service/tests/ -v
```

**2. Contract Tests — Service interface**
```python
# Consumer-Driven Contract Test (Pact-like)
class TestCustomerServiceContract(unittest.TestCase):
    def test_get_customer_returns_expected_format(self):
        # Test rằng service trả về đúng contract
        service = CustomerService(CustomerRepository())
        customer = service.get_customer_summary("CUST001")
        self.assertIn("customer_id", customer)
        self.assertIn("full_name", customer)
        self.assertIn("is_verified", customer)
```

**3. Integration Tests — Service composition**
```python
# Test Transfer Orchestrator với mock services
class TestTransferOrchestrator(unittest.TestCase):
    def test_transfer_success(self):
        mock_account = MockAccountService()
        mock_aml = MockAMLService()
        orchestrator = TransferOrchestrator(mock_account, mock_aml, ...)
        result = orchestrator.execute_transfer(request)
        self.assertEqual(result.status, TransferStatus.COMPLETED)
```

**4. End-to-End Tests — Full flow**
```bash
# Deploy tất cả services lên môi trường test
docker-compose up -d

# Chạy E2E tests qua API Gateway
python e2e/test_transfer_flow.py
python e2e/test_customer_kyc_flow.py
```

Xem `tests/test_customer_service.py` cho unit test examples chi tiết.

---

## Kết luận

**Service-Oriented Architecture (SOA)** là một kiến trúc enterprise mạnh mẽ đã được kiểm chứng qua hàng nghìn hệ thống ngân hàng, bảo hiểm, viễn thông, và chính phủ trên thế giới. Dù Microservices đang là xu hướng, SOA vẫn là lựa chọn tối ưu cho các hệ thống enterprise phức tạp cần governance, compliance, và reuse.

### Best Practices

1. **Design service contract first**: Interface trước, implementation sau. Contract là cam kết giữa service provider và consumer.

2. **Avoid shared database**: Mỗi service sở hữu database riêng. Gọi service khác qua API, không qua DB.

3. **Async communication mặc định**: Dùng message queue (Kafka, RabbitMQ) thay vì synchronous HTTP khi có thể.

4. **Handle failure gracefully**: Circuit breaker, retry, dead letter queue, timeout.

5. **Versioning strategy**: API versioning (URL, header, contract). Hỗ trợ multiple versions.

6. **Centralized monitoring**: Distributed tracing (OpenTelemetry, Jaeger), centralized logging.

7. **Service governance**: Service registry, SLA monitoring, rate limiting, access control.

8. **Gradual adoption**: Không chuyển đổi tất cả sang SOA cùng lúc. Từng bước tách monolith.

### Golden Rules

| Rule | Giải thích |
|------|-----------|
| **Service = Business Capability** | Mỗi service tương ứng với một business capability |
| **Contract first** | Viết interface trước code |
| **Share nothing** | Không share database, không share schema |
| **Design for failure** | Mọi service có thể fail bất cứ lúc nào |
| **Stateless khi có thể** | State lưu trong database, không trong service |
| **Idempotent operations** | Gọi N lần cho cùng kết quả |
| **SLA is part of contract** | Latency, availability, throughput là một phần của contract |
| **Governance is not optional** | Không có governance = distributed chaos |

SOA không phải là giải pháp rẻ hay đơn giản. Nó yêu cầu đầu tư lớn về infrastructure, governance, và team capability. Nhưng đối với các enterprise lớn với hàng trăm ứng dụng và yêu cầu compliance nghiêm ngặt, SOA là kiến trúc đã được kiểm chứng để đạt được business agility và operational excellence.
