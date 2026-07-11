---
id: serverless
title: Serverless Architecture
sidebar_label: 🏗️ Serverless Architecture
sidebar_position: 47
---

# Serverless Architecture

> "Serverless doesn't mean there are no servers; it means you don't have to think about them. Focus on your code, not your infrastructure."
> — **Amazon Web Services**, *AWS Lambda Whitepaper* (2014)

**Serverless Architecture** là một mô hình phát triển và triển khai ứng dụng trong đó nhà phát triển chỉ tập trung vào viết code, còn toàn bộ hạ tầng (server, networking, scaling) được cloud provider quản lý hoàn toàn. Thuật ngữ "Serverless" không có nghĩa là không có server, mà là bạn **không phải quản lý server**.

---

## Tổng quan

### Lịch sử và nguồn gốc

Serverless architecture phát triển qua nhiều giai đoạn:

- **2006**: Amazon ra mắt EC2 — khởi đầu của cloud computing
- **2008**: Google App Engine — Platform as a Service (PaaS) đầu tiên
- **2014**: **AWS Lambda** ra mắt — Function-as-a-Service (FaaS) đầu tiên
- **2016**: Azure Functions, Google Cloud Functions
- **2017**: AWS Step Functions cho orchestration
- **2018-2019**: Serverless Framework, AWS SAM, CloudFormation
- **2020-nay**: Serverless trở thành mainstream, edge computing (Cloudflare Workers, Lambda@Edge)

### Những người tiên phong

| Tổ chức | Đóng góp |
|---------|---------|
| **Amazon/AWS** | AWS Lambda (FaaS), API Gateway, DynamoDB, S3, Step Functions |
| **Microsoft/Azure** | Azure Functions, Logic Apps, Cosmos DB |
| **Google Cloud** | Cloud Functions, Cloud Run, Firebase |
| **Cloudflare** | Cloudflare Workers, KV, Durable Objects |
| **Netlify/Vercel** | Jamstack, Edge Functions |
| **Serverless, Inc.** | Serverless Framework (Terraform-compatible) |
| **OpenFaaS / Knative** | Open-source self-hosted serverless |

### Các thành phần cốt lõi

Serverless architecture thường bao gồm:

1. **FaaS** (Function-as-a-Service): AWS Lambda, Azure Functions, Google Cloud Functions
2. **BaaS** (Backend-as-a-Service): Auth0, Firebase, Supabase
3. **Managed databases**: DynamoDB, Aurora Serverless, Cosmos DB
4. **API Gateway**: REST/graphQL endpoint cho functions
5. **Event sources**: SQS, SNS, EventBridge, Kinesis
6. **Storage**: S3, Cloud Storage
7. **Orchestration**: Step Functions, Durable Functions

---

## Bài toán

### Nền tảng Xử lý Ảnh và Video cho Social Media

Giả sử bạn đang xây dựng **PixPro** — một nền tảng chỉnh sửa và chia sẻ ảnh/video trực tuyến với các tính năng:

1. **Upload ảnh/video** — Người dùng upload file từ web/mobile
2. **Xử lý hậu kỳ** — Resize, compress, filter, watermark, format conversion
3. **Tạo thumbnail** — Tự động tạo thumbnail cho gallery
4. **Chia sẻ** — Tạo link chia sẻ, phân quyền xem
5. **AI enhancement** — Tự động cải thiện chất lượng ảnh (denoise, upscale)
6. **CDN delivery** — Phân phối nội dung qua CDN toàn cầu
7. **Analytics** — Thống kê lượt xem, dung lượng, top ảnh

### Khó khăn với kiến trúc truyền thống

**Vấn đề 1 — Traffic không đồng đều**:
- Ngày thường: 1,000 uploads/giờ
- Giờ cao điểm (tối, cuối tuần): 50,000 uploads/giờ
- Sự kiện đặc biệt (Black Friday, concert): 1,000,000 uploads/giờ
- Nếu dùng server dedicated: phải provision cho peak → lãng phí ~70% capacity

```
Server utilization:
  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (25% — mostly idle)
  ████████████████████████████████████████████  (peak — 100%)
  → Lãng phí 75% chi phí infrastructure
```

**Vấn đề 2 — Xử lý batch phức tạp**:
Upload ảnh cần một pipeline:
```
Upload → Validate → Resize → Compress → Watermark → 
Denoise → Generate Thumbnail → Save to CDN → 
Send Notification → Update Analytics
```

Nếu mỗi bước là một service riêng (trên dedicated server):
- Phải quản lý 10+ services
- Mỗi service cần scaling riêng
- Orchestration khó khăn (message queue, retry, error handling)
- Chi phí vận hành cao

**Vấn đề 3 — Chi phí cho idle time**:
- Hầu hết services chỉ hoạt động khi có request
- Server vẫn tốn tiền dù không làm gì
- Database connection pool vẫn chạy 24/7
- Load balancer vẫn tính phí dù không có traffic

**Vấn đề 4 — Scaling nhanh**:
- Khi ảnh viral đột ngột (ví dụ: bài đăng triệu view):
  - Phải scale compute resources trong vài phút
  - Auto-scaling group mất 5-15 phút để warm up
  - Database connection pool bị quá tải
  - CDN chưa kịp cache → server chết

**Vấn đề 5 — Operations overhead**:
- Phải quản lý OS patches, security updates
- Monitoring, logging, alerting infrastructure
- Backup, disaster recovery
- Capacity planning

### Serverless giải quyết vấn đề

1. **Auto-scaling tức thì**: Lambda scale từ 0 → hàng nghìn concurrent executions trong milliseconds
2. **Pay-per-use**: Chỉ trả tiền khi function chạy (ms + memory)
3. **Không quản lý server**: OS, runtime, security patches được cloud provider xử lý
4. **Event-driven**: Pipeline xử lý ảnh tự động theo sự kiện (S3 upload → Lambda trigger)
5. **Built-in high availability**: AWS/Azure tự động replicate functions
6. **Giảm operational cost**: Không cần DevOps team cho infrastructure

---

## Nguyên lý thiết kế

### 1. Function là đơn vị triển khai cơ bản

Mỗi function là một đơn vị code độc lập, single-purpose, stateless. Function:
- Nhận event → xử lý → trả về kết quả (hoặc gọi function khác)
- Không giữ state giữa các lần gọi
- Có timeout (15 phút với Lambda, 10 phút với Cloud Functions)
- Có memory/CPU allocation riêng

### 2. Event-driven Architecture

Serverless hoạt động theo sự kiện:
```
S3 Upload → Lambda Resize → DynamoDB Update → SNS Notification → ...
```

Mỗi function được trigger bởi một event từ:
- HTTP request (API Gateway)
- File upload (S3/GCS)
- Database change (DynamoDB Streams)
- Message queue (SQS)
- Timer (CloudWatch Events)
- IoT device (IoT Core)

### 3. Stateless Design

Functions không giữ state giữa các lần gọi:
- State lưu trong external storage (DynamoDB, Redis, S3)
- Context được khởi tạo lại mỗi lần gọi (cold start)
- Singleton connection pool cần được reuse
- Không dùng in-memory cache (trừ khi cache layer riêng)

### 4. Single Responsibility Principle

Mỗi function chỉ làm đúng MỘT việc:
```python
# BAD — function làm quá nhiều
def handler(event, context):
    image = download_image(event['url'])
    resized = resize_image(image, 800, 600)
    compressed = compress_jpeg(resized, 85)
    saved = save_to_s3(compressed, event['key'])
    thumbnail = create_thumbnail(image)
    save_thumbnail(thumbnail)
    update_database(event['key'], status='processed')
    send_notification(event['user_id'])
    return {'status': 'ok'}

# GOOD — mỗi function một việc
# Function 1: resize_and_compress
# Function 2: generate_thumbnail
# Function 3: update_database
# Function 4: send_notification
```

### 5. Infrastructure as Code (IaC)

Toàn bộ infrastructure được định nghĩa trong code:
```yaml
# serverless.yml example
functions:
  processImage:
    handler: handler.process_image
    events:
      - s3: my-bucket
      - events: s3:ObjectCreated:*
```

### 6. Graceful Degradation và Retry

Serverless functions cần handle:
- **Idempotency**: Cùng event xử lý nhiều lần cho cùng kết quả
- **Retry**: Lambda tự động retry 3 lần khi fail
- **Dead Letter Queue**: Event không xử lý được → DLQ để debug
- **Timeout**: Xử lý timeout graceful (cleanup, partial state save)

### 7. Security by Design

- Least privilege IAM roles
- Environment variables cho secrets (dùng KMS)
- VPC cho database access (nhưng hạn chế cold start)
- Request validation ở API Gateway

---

## Cấu trúc chi tiết

### Các thành phần trong Serverless Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                 │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐        │
│  │  Web    │  │ Mobile  │  │ 3rd Party│  │ IoT/Devices  │        │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └──────┬───────┘        │
│       │            │            │               │                 │
└───────┼────────────┼────────────┼───────────────┼─────────────────┘
        │            │            │               │
        ▼            ▼            ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    API GATEWAY LAYER                                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │           Amazon API Gateway / Cloud Endpoints              │    │
│  │                                                             │    │
│  │  /api/v1/images  │  /api/v1/users  │  /api/v1/analytics    │    │
│  │  Auth (Cognito)  │  Rate Limiting  │  Request Validation   │    │
│  │  API Keys        │  CORS           │  WAF (Web ACL)        │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FUNCTIONS LAYER (FaaS)                          │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │     │
│  │  │ Image    │  │Thumbnail │  │ Watermark│  │ AI      │  │     │
│  │  │ Processor│  │ Generator│  │  Engine  │  │ Upscaler│  │     │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘  │     │
│  │  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐  ┌────┴────┐  │     │
│  │  │ Notify   │  │Analytics │  │ CDN      │  │ DB      │  │     │
│  │  │ Service  │  │ Reporter │  │ Invalid  │  │ Updater │  │     │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │     │
│  └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EVENT & MESSAGING LAYER                          │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  S3 Bucket    │  SQS Queue   │  SNS Topic  │ EventBridge │      │
│  │  (Storage)    │  (Buffer)    │ (Fan-out)  │ (Router)    │      │
│  └───────────────┴──────────────┴─────────────┴─────────────┘      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────┐ │
│  │  DynamoDB    │  │ ElastiCache  │  │ S3 Glacier │  │ Aurora  │ │
│  │  (NoSQL)     │  │  (Redis)     │  │ (Archive)  │  │ (SQL)   │ │
│  └──────────────┘  └──────────────┘  └────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Từng lớp chi tiết

**1. API Gateway Layer**
- Entry point duy nhất cho REST/graphQL requests
- Xác thực (Cognito, API Keys, JWT)
- Rate limiting, throttling
- Request/response transformation
- CORS, WAF protection
- Custom domain + SSL termination

**2. Functions Layer (FaaS)**
- Mỗi function là một Lambda/Cloud Function
- Có thể có runtime khác nhau (Python, Node, Go, Java)
- Function chaining qua Step Functions
- Layers cho shared dependencies
- Reserved concurrency để tránh throttling

**3. Event & Messaging Layer**
- **S3**: File storage + event source
- **SQS**: Buffering và decoupling
- **SNS**: Fan-out notifications
- **EventBridge**: Event routing với rules
- **Step Functions**: Workflow orchestration

**4. Data Layer**
- **DynamoDB**: NoSQL, serverless, auto-scaling
- **S3**: Object storage, lifecycle policies
- **ElastiCache**: Redis caching layer
- **Aurora Serverless**: SQL database on-demand

---

## Sơ đồ kiến trúc

```
PIXPRO — SERVERLESS ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

  User ─→ [CDN / CloudFront]
                     │
                     ▼
              [API Gateway]
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
  [Auth Lambda]  [Upload Lambda] [Analytics Lambda]
                      │
                      ▼
                 [S3 Bucket]
                 raw-images/
                      │
              (S3 Event Trigger)
                      │
                      ▼
            ┌─────────────────┐
            │  Orchestrator   │
            │  Step Functions │
            └───┬──────┬──────┘
                │      │
        ┌───────┘      └───────┐
        ▼                      ▼
  [Resize Lambda]      [Watermark Lambda]
        │                      │
        ▼                      ▼
  [Compress Lambda]     [AI Enhance Lambda]
        │                      │
        └───────┬──────────────┘
                ▼
          [Thumbnail Lambda]
                │
                ▼
       [Processed S3 Bucket]
                │
        ┌───────┴───────┐
        ▼               ▼
  [CDN Invalidation] [DynamoDB Update]
        │               │
        ▼               ▼
  [SNS Notification] [Analytics Lambda]

Event Flow (real-time):
──────────────────────
  1. User uploads via API Gateway → Upload Lambda
  2. Upload Lambda saves to S3 raw-images/
  3. S3 event triggers Step Functions workflow
  4. Step Functions orchestrates parallel/serial processing
  5. Each step is a Lambda function
  6. Processed images saved to S3 processed/
  7. Metadata saved to DynamoDB
  8. CDN invalidated for fresh content
  9. Notification sent via SNS
  10. Analytics updated
```

---

## Ví dụ code hoàn chỉnh

### Cấu trúc project

```
pixpro-serverless/
├── infra/
│   └── serverless.yml          # Serverless Framework config
├── functions/
│   ├── __init__.py
│   ├── auth.py                 # Authentication handler
│   ├── upload.py               # Upload handler
│   ├── process_image.py        # Image processing pipeline
│   ├── thumbnail.py            # Thumbnail generation
│   ├── watermark.py            # Watermark engine
│   ├── compress.py             # Compression engine
│   ├── ai_upscale.py           # AI upscaling
│   ├── notifications.py        # Push notifications
│   └── analytics.py            # Analytics processor
├── lib/
│   ├── __init__.py
│   ├── s3_utils.py             # S3 helper functions
│   ├── db_utils.py             # DynamoDB helpers
│   ├── image_utils.py          # Image processing utilities
│   └── api_utils.py            # API response formatting
├── models/
│   ├── __init__.py
│   ├── image.py                # Image model
│   ├── user.py                 # User model
│   └── events.py               # Event schemas
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_process_image.py
│   └── test_handlers.py
├── requirements.txt
└── Dockerfile                   # For containerized Lambda
```

### models/image.py

```python
"""Image domain models — shared across all functions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Optional
from decimal import Decimal
import uuid


class ImageFormat(Enum):
    JPEG = auto()
    PNG = auto()
    WEBP = auto()
    AVIF = auto()
    GIF = auto()
    SVG = auto()


class ProcessingStatus(Enum):
    PENDING = auto()
    QUEUED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()


class ImageCategory(Enum):
    PROFILE = auto()
    GALLERY = auto()
    POST = auto()
    MESSAGE = auto()
    DOCUMENT = auto()


@dataclass
class ImageRecord:
    """Image metadata stored in DynamoDB."""
    image_id: str
    user_id: str
    category: ImageCategory

    # File info
    original_filename: str
    original_format: ImageFormat
    original_size: int  # bytes
    original_width: int
    original_height: int

    # Processing
    status: ProcessingStatus = ProcessingStatus.PENDING
    processed_size: Optional[int] = None
    processed_format: Optional[ImageFormat] = None
    thumbnail_key: Optional[str] = None
    watermark_key: Optional[str] = None
    hd_key: Optional[str] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None

    # Analytics
    view_count: int = 0
    download_count: int = 0
    processing_time_ms: Optional[int] = None

    @classmethod
    def new(
        cls,
        user_id: str,
        filename: str,
        fmt: ImageFormat,
        size: int,
        width: int,
        height: int,
        category: ImageCategory = ImageCategory.GALLERY,
    ) -> ImageRecord:
        return cls(
            image_id=str(uuid.uuid4()),
            user_id=user_id,
            category=category,
            original_filename=filename,
            original_format=fmt,
            original_size=size,
            original_width=width,
            original_height=height,
        )

    @property
    def s3_key_original(self) -> str:
        return f"raw/{self.user_id}/{self.image_id}/{self.original_filename}"

    @property
    def s3_key_processed(self) -> str:
        return f"processed/{self.user_id}/{self.image_id}/optimized"

    @property
    def s3_key_thumbnail(self) -> str:
        return f"thumbnails/{self.user_id}/{self.image_id}/thumb_256"

    @property
    def cdn_url(self) -> str:
        return f"https://cdn.pixpro.example/{self.s3_key_processed}"

    def to_dict(self) -> dict:
        result = asdict(self)
        result['original_format'] = self.original_format.name
        result['category'] = self.category.name
        result['status'] = self.status.name
        if self.processed_format:
            result['processed_format'] = self.processed_format.name
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ImageRecord:
        data = dict(data)
        data['original_format'] = ImageFormat[data['original_format']]
        data['category'] = ImageCategory[data['category']]
        data['status'] = ProcessingStatus[data['status']]
        if data.get('processed_format'):
            data['processed_format'] = ImageFormat[data['processed_format']]
        return cls(**data)


@dataclass
class ProcessingRequest:
    """Event payload sent between Lambda functions."""
    image_id: str
    user_id: str
    bucket: str
    s3_key: str
    output_bucket: str
    options: dict = field(default_factory=dict)

    @classmethod
    def from_s3_event(cls, event: dict) -> list[ProcessingRequest]:
        """Parse S3 event notification."""
        requests = []
        for record in event.get('Records', []):
            if record.get('eventSource') != 'aws:s3':
                continue
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            # Extract user_id and image_id from key: raw/{user_id}/{image_id}/{filename}
            parts = key.split('/')
            if len(parts) >= 3:
                requests.append(cls(
                    image_id=parts[2],
                    user_id=parts[1],
                    bucket=bucket,
                    s3_key=key,
                    output_bucket=f"{bucket}-processed",
                ))
        return requests


@dataclass
class ProcessingResult:
    """Result from each processing step."""
    step: str
    success: bool
    output_key: Optional[str] = None
    output_size: Optional[int] = None
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)
```

### lib/image_utils.py

```python
"""Image processing utilities — core business logic."""

from __future__ import annotations

from io import BytesIO
from typing import Tuple, Optional
from PIL import Image as PILImage, ImageFilter, ImageEnhance
import struct
from pathlib import Path

from models.image import ImageFormat


class ImageProcessor:
    """Xử lý ảnh — resize, compress, watermark, AI enhancement.

    Đây là core business logic, độc lập với infrastructure (S3, Lambda).
    """

    SUPPORTED_FORMATS = {
        ImageFormat.JPEG: "JPEG",
        ImageFormat.PNG: "PNG",
        ImageFormat.WEBP: "WEBP",
        ImageFormat.AVIF: "AVIF" if hasattr(PILImage, 'AVIF') else "WEBP",  # Fallback
    }

    QUALITY_RANGES = {
        "max": 100,
        "high": 92,
        "medium": 80,
        "low": 60,
        "minimal": 40,
    }

    THUMBNAIL_SIZES = {
        "tiny": (64, 64),
        "small": (128, 128),
        "medium": (256, 256),
        "large": (512, 512),
        "og": (1200, 630),  # Open Graph
    }

    @staticmethod
    def resize(
        image_bytes: bytes,
        target_width: int,
        target_height: int,
        maintain_aspect: bool = True,
        fit: str = "cover",  # cover, contain, fill
    ) -> tuple[bytes, int, int]:
        """Resize image to target dimensions."""
        img = PILImage.open(BytesIO(image_bytes))
        original_width, original_height = img.size

        if maintain_aspect:
            if fit == "cover":
                # Crop to fill
                scale = max(target_width / original_width, target_height / original_height)
                new_w = int(original_width * scale)
                new_h = int(original_height * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
                # Center crop
                left = (new_w - target_width) // 2
                top = (new_h - target_height) // 2
                img = img.crop((left, top, left + target_width, top + target_height))
            else:  # contain
                scale = min(target_width / original_width, target_height / original_height)
                new_w = int(original_width * scale)
                new_h = int(original_height * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
        else:
            img = img.resize((target_width, target_height), PILImage.LANCZOS)

        output = BytesIO()
        img.save(output, format=img.format or "JPEG", quality=92)
        return output.getvalue(), img.width, img.height

    @staticmethod
    def compress(
        image_bytes: bytes,
        quality: str = "high",
        target_format: ImageFormat = ImageFormat.JPEG,
    ) -> tuple[bytes, int, ImageFormat]:
        """Nén ảnh với chất lượng chỉ định."""
        quality_val = ImageProcessor.QUALITY_RANGES.get(quality, 80)
        img = PILImage.open(BytesIO(image_bytes))
        fmt = ImageProcessor.SUPPORTED_FORMATS.get(target_format, "JPEG")

        output = BytesIO()
        if fmt == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(
            output,
            format=fmt,
            quality=quality_val,
            optimize=True,
            progressive=True,
        )
        return output.getvalue(), quality_val, target_format

    @staticmethod
    def create_thumbnail(
        image_bytes: bytes,
        size_name: str = "medium",
    ) -> tuple[bytes, int, int]:
        """Tạo thumbnail cho ảnh."""
        size = ImageProcessor.THUMBNAIL_SIZES.get(size_name, (256, 256))
        return ImageProcessor.resize(image_bytes, size[0], size[1])

    @staticmethod
    def apply_watermark(
        image_bytes: bytes,
        watermark_text: str = "PixPro",
        opacity: float = 0.3,
        position: str = "bottom_right",
    ) -> bytes:
        """Apply watermark text lên ảnh."""
        from PIL import ImageDraw, ImageFont

        img = PILImage.open(BytesIO(image_bytes))
        # Convert to RGBA for transparency
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Create watermark layer
        watermark = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        # Use default font (scaled)
        font_size = max(min(img.width, img.height) // 20, 12)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position
        padding = int(font_size * 0.8)
        positions = {
            "top_left": (padding, padding),
            "top_right": (img.width - text_width - padding, padding),
            "bottom_left": (padding, img.height - text_height - padding),
            "bottom_right": (img.width - text_width - padding, img.height - text_height - padding),
            "center": ((img.width - text_width) // 2, (img.height - text_height) // 2),
        }
        pos = positions.get(position, positions["bottom_right"])

        # Draw text with opacity
        fill_color = (255, 255, 255, int(255 * opacity))
        draw.text(pos, watermark_text, fill=fill_color, font=font)

        # Composite
        result = PILImage.alpha_composite(img, watermark)
        output = BytesIO()
        result.save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def ai_enhance(
        image_bytes: bytes,
        denoise_strength: float = 0.5,
        sharpen: bool = True,
        contrast: float = 1.2,
    ) -> bytes:
        """Cải thiện chất lượng ảnh (mô phỏng AI enhancement).

        Trong thực tế, đây sẽ gọi API đến model AI (Replicate, HuggingFace, SageMaker).
        """
        img = PILImage.open(BytesIO(image_bytes))

        # Denoise
        if denoise_strength > 0:
            img = img.filter(ImageFilter.MedianFilter(size=3))
            if denoise_strength > 0.7:
                img = img.filter(ImageFilter.BLUR)
                img = img.filter(ImageFilter.SMOOTH_MORE)

        # Sharpen
        if sharpen:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)

        # Contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)

        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)

        output = BytesIO()
        img.save(output, format=img.format or "JPEG", quality=95)
        return output.getvalue()

    @staticmethod
    def get_image_info(image_bytes: bytes) -> dict:
        """Lấy thông tin cơ bản của ảnh."""
        img = PILImage.open(BytesIO(image_bytes))
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_bytes": len(image_bytes),
        }


class ValidationError(Exception):
    pass


class ImageValidator:
    """Validate image files trước khi xử lý."""

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_FORMATS = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/avif"}
    MIN_DIMENSION = 32
    MAX_DIMENSION = 16384  # 16K

    @classmethod
    def validate(cls, image_bytes: bytes, content_type: str) -> None:
        if len(image_bytes) > cls.MAX_FILE_SIZE:
            raise ValidationError(
                f"File quá lớn: {len(image_bytes)} bytes (max: {cls.MAX_FILE_SIZE})"
            )

        if content_type not in cls.ALLOWED_FORMATS:
            raise ValidationError(
                f"Định dạng không hỗ trợ: {content_type}. "
                f"Hỗ trợ: {', '.join(cls.ALLOWED_FORMATS)}"
            )

        info = ImageProcessor.get_image_info(image_bytes)
        if info["width"] < cls.MIN_DIMENSION or info["height"] < cls.MIN_DIMENSION:
            raise ValidationError(
                f"Ảnh quá nhỏ: {info['width']}x{info['height']} "
                f"(min: {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION})"
            )

        if info["width"] > cls.MAX_DIMENSION or info["height"] > cls.MAX_DIMENSION:
            raise ValidationError(
                f"Ảnh quá lớn: {info['width']}x{info['height']} "
                f"(max: {cls.MAX_DIMENSION}x{cls.MAX_DIMENSION})"
            )
```

### lib/s3_utils.py

```python
"""S3 utility functions for Lambda handlers."""

from __future__ import annotations

from typing import Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import struct

s3_config = Config(
    max_pool_connections=50,
    retries={"max_attempts": 3, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=30,
)

s3_client = boto3.client("s3", config=s3_config)
s3_resource = boto3.resource("s3", config=s3_config)


class S3Manager:
    """Manage S3 operations — upload, download, copy, delete.

    Singleton pattern: s3_client được reuse across invocations.
    """

    DEFAULT_EXPIRATION = 3600  # 1 hour presigned URL
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

    @staticmethod
    def download_bytes(bucket: str, key: str) -> bytes:
        """Download file from S3 as bytes."""
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            raise RuntimeError(f"S3 download failed: {e}") from e

    @staticmethod
    def upload_bytes(
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "image/jpeg",
        metadata: Optional[dict] = None,
    ) -> None:
        """Upload bytes to S3."""
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata or {},
                StorageClass="INTELLIGENT_TIERING",
            )
        except ClientError as e:
            raise RuntimeError(f"S3 upload failed: {e}") from e

    @staticmethod
    def copy_object(
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> None:
        """Copy object between S3 locations."""
        try:
            copy_source = {"Bucket": source_bucket, "Key": source_key}
            s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key,
                StorageClass="INTELLIGENT_TIERING",
            )
        except ClientError as e:
            raise RuntimeError(f"S3 copy failed: {e}") from e

    @staticmethod
    def delete_object(bucket: str, key: str) -> None:
        """Delete object from S3."""
        try:
            s3_client.delete_object(Bucket=bucket, Key=key)
        except ClientError as e:
            raise RuntimeError(f"S3 delete failed: {e}") from e

    @staticmethod
    def generate_presigned_url(
        bucket: str,
        key: str,
        expiration: int = DEFAULT_EXPIRATION,
    ) -> str:
        """Generate presigned URL for temporary access."""
        try:
            return s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expiration,
            )
        except ClientError as e:
            raise RuntimeError(f"Presigned URL generation failed: {e}") from e

    @staticmethod
    def object_exists(bucket: str, key: str) -> bool:
        """Check if object exists in S3."""
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
```

### lib/db_utils.py

```python
"""DynamoDB utility functions."""

from __future__ import annotations

from typing import Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from decimal import Decimal
import json

from models.image import ImageRecord, ProcessingStatus

dynamodb = boto3.resource("dynamodb")
dynamodb_client = boto3.client("dynamodb")

# Table names — set via environment variables
IMAGES_TABLE = "pixpro-images"
USERS_TABLE = "pixpro-users"
ANALYTICS_TABLE = "pixpro-analytics"


class ImageRepository:
    """DynamoDB repository for ImageRecord."""

    @staticmethod
    def save(record: ImageRecord) -> None:
        """Save or update image record."""
        table = dynamodb.Table(IMAGES_TABLE)
        try:
            item = record.to_dict()
            item['original_size'] = Decimal(str(item['original_size']))
            if item.get('processed_size'):
                item['processed_size'] = Decimal(str(item['processed_size']))
            if item.get('view_count'):
                item['view_count'] = Decimal(str(item['view_count']))
            if item.get('download_count'):
                item['download_count'] = Decimal(str(item['download_count']))
            if item.get('processing_time_ms'):
                item['processing_time_ms'] = Decimal(str(item['processing_time_ms']))

            table.put_item(Item=item)
        except ClientError as e:
            raise RuntimeError(f"DynamoDB save failed: {e}") from e

    @staticmethod
    def get(image_id: str) -> Optional[ImageRecord]:
        """Get image record by ID."""
        table = dynamodb.Table(IMAGES_TABLE)
        try:
            response = table.get_item(Key={"image_id": image_id})
            item = response.get("Item")
            if item:
                return ImageRecord.from_dict(item)
            return None
        except ClientError as e:
            raise RuntimeError(f"DynamoDB get failed: {e}") from e

    @staticmethod
    def update_status(
        image_id: str,
        status: ProcessingStatus,
        **extra_fields,
    ) -> None:
        """Update processing status and optional fields."""
        table = dynamodb.Table(IMAGES_TABLE)
        try:
            update_expr = "SET #status = :status, updated_at = :updated"
            expr_attrs = {
                "#status": "status",
            }
            attr_values = {
                ":status": status.name,
                ":updated": __import__('datetime').datetime.utcnow().isoformat(),
            }

            for key, value in extra_fields.items():
                update_expr += f", {key} = :{key}"
                if isinstance(value, (int, float)):
                    attr_values[f":{key}"] = Decimal(str(value))
                else:
                    attr_values[f":{key}"] = value

            table.update_item(
                Key={"image_id": image_id},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_attrs,
                ExpressionAttributeValues=attr_values,
            )
        except ClientError as e:
            raise RuntimeError(f"DynamoDB update failed: {e}") from e

    @staticmethod
    def increment_counter(image_id: str, field: str, amount: int = 1) -> None:
        """Increment a counter field (view_count, download_count)."""
        table = dynamodb.Table(IMAGES_TABLE)
        try:
            table.update_item(
                Key={"image_id": image_id},
                UpdateExpression=f"ADD {field} :inc",
                ExpressionAttributeValues={
                    ":inc": amount,
                },
            )
        except ClientError as e:
            raise RuntimeError(f"DynamoDB increment failed: {e}") from e

    @staticmethod
    def query_by_user(
        user_id: str,
        limit: int = 50,
        last_key: Optional[dict] = None,
    ) -> tuple[list[ImageRecord], Optional[dict]]:
        """Query images by user with pagination."""
        table = dynamodb.Table(IMAGES_TABLE)
        try:
            params = {
                "IndexName": "user_id-index",
                "KeyConditionExpression": "#uid = :uid",
                "ExpressionAttributeNames": {"#uid": "user_id"},
                "ExpressionAttributeValues": {":uid": user_id},
                "Limit": limit,
                "ScanIndexForward": False,  # Most recent first
            }
            if last_key:
                params["ExclusiveStartKey"] = last_key

            response = table.query(**params)
            items = [ImageRecord.from_dict(item) for item in response.get("Items", [])]
            last = response.get("LastEvaluatedKey")
            return items, last
        except ClientError as e:
            raise RuntimeError(f"DynamoDB query failed: {e}") from e
```

### functions/upload.py

```python
"""Upload Lambda — handle file upload, validation, initial processing."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any
from datetime import datetime

from models.image import (
    ImageRecord, ImageFormat, ImageCategory, ProcessingRequest,
)
from lib.image_utils import ImageValidator, ValidationError
from lib.s3_utils import S3Manager
from lib.db_utils import ImageRepository


# Environment variables
UPLOAD_BUCKET = os.environ.get("UPLOAD_BUCKET", "pixpro-uploads")
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET", "pixpro-processed")
IMAGE_TABLE = os.environ.get("IMAGE_TABLE", "pixpro-images")


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handle image upload request.

    Expected event:
    {
        "user_id": "user123",
        "filename": "vacation.jpg",
        "content_type": "image/jpeg",
        "body": "<base64_encoded_bytes>"
    }
    """
    try:
        # Parse request
        body = json.loads(event.get("body", "{}"))
        user_id = body.get("user_id")
        filename = body.get("filename", "untitled")
        content_type = body.get("content_type", "image/jpeg")
        image_b64 = body.get("body")

        if not all([user_id, image_b64]):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required fields"}),
            }

        # Decode image
        import base64
        image_bytes = base64.b64decode(image_b64)

        # Validate
        try:
            ImageValidator.validate(image_bytes, content_type)
        except ValidationError as e:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": str(e)}),
            }

        # Get image info
        from lib.image_utils import ImageProcessor
        info = ImageProcessor.get_image_info(image_bytes)
        fmt = ImageFormat.JPEG
        for name, ext in [("JPEG", ImageFormat.JPEG), ("PNG", ImageFormat.PNG),
                          ("WEBP", ImageFormat.WEBP)]:
            if info["format"] == name:
                fmt = ImageFormat[name]

        # Create record
        record = ImageRecord.new(
            user_id=user_id,
            filename=filename,
            fmt=fmt,
            size=len(image_bytes),
            width=info["width"],
            height=info["height"],
            category=ImageCategory.GALLERY,
        )

        # Upload raw to S3
        S3Manager.upload_bytes(
            bucket=UPLOAD_BUCKET,
            key=record.s3_key_original,
            data=image_bytes,
            content_type=content_type,
            metadata={
                "user_id": user_id,
                "image_id": record.image_id,
                "uploaded_at": datetime.utcnow().isoformat(),
            },
        )

        # Save record to DynamoDB
        ImageRepository.save(record)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "image_id": record.image_id,
                "url": record.cdn_url,
                "message": "Upload thành công, đang xử lý...",
            }),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal error: {str(e)}"}),
        }
```

### functions/process_image.py

```python
"""Process Image Lambda — trigger by S3 event, orchestrate processing."""

from __future__ import annotations

import os
import json
import time
from typing import Any

from models.image import (
    ImageRecord, ProcessingRequest, ProcessingResult,
    ProcessingStatus, ImageFormat,
)
from lib.s3_utils import S3Manager
from lib.db_utils import ImageRepository
from lib.image_utils import ImageProcessor

PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET", "pixpro-processed")


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main processing handler — triggered by S3 upload event.

    Pipeline:
    1. Download raw image from S3
    2. Resize to multiple sizes
    3. Compress
    4. Generate thumbnail
    5. Apply watermark
    6. AI enhance
    7. Upload processed versions
    8. Update DynamoDB
    9. Return results
    """
    start_time = time.monotonic()

    # Parse S3 event
    requests = ProcessingRequest.from_s3_event(event)
    if not requests:
        return {"statusCode": 400, "body": "Invalid S3 event"}

    req = requests[0]
    results = []

    try:
        # Update status
        ImageRepository.update_status(req.image_id, ProcessingStatus.PROCESSING)

        # 1. Download
        raw_bytes = S3Manager.download_bytes(req.bucket, req.s3_key)

        # 2. Get info
        info = ImageProcessor.get_image_info(raw_bytes)

        # 3. Compress
        compressed_data, quality, fmt = ImageProcessor.compress(
            raw_bytes, quality="high", target_format=ImageFormat.WEBP
        )
        compressed_key = f"{req.s3_key}/compressed.webp"
        S3Manager.upload_bytes(
            PROCESSED_BUCKET, compressed_key, compressed_data,
            content_type="image/webp",
        )
        results.append(ProcessingResult(
            step="compress",
            success=True,
            output_key=compressed_key,
            output_size=len(compressed_data),
        ))

        # 4. Thumbnail
        thumb_data, tw, th = ImageProcessor.create_thumbnail(
            raw_bytes, size_name="medium"
        )
        thumb_key = f"thumbnails/{req.user_id}/{req.image_id}/thumb_256.jpg"
        S3Manager.upload_bytes(
            PROCESSED_BUCKET, thumb_key, thumb_data,
            content_type="image/jpeg",
        )
        results.append(ProcessingResult(
            step="thumbnail",
            success=True,
            output_key=thumb_key,
            output_size=len(thumb_data),
            output_width=tw,
            output_height=th,
        ))

        # 5. Watermark
        watermarked = ImageProcessor.apply_watermark(raw_bytes)
        watermark_key = f"{req.s3_key}/watermarked.png"
        S3Manager.upload_bytes(
            PROCESSED_BUCKET, watermark_key, watermarked,
            content_type="image/png",
        )
        results.append(ProcessingResult(
            step="watermark",
            success=True,
            output_key=watermark_key,
            output_size=len(watermarked),
        ))

        # 6. AI Enhance (mô phỏng)
        enhanced = ImageProcessor.ai_enhance(raw_bytes)
        hd_key = f"{req.s3_key}/hd_enhanced.jpg"
        S3Manager.upload_bytes(
            PROCESSED_BUCKET, hd_key, enhanced,
            content_type="image/jpeg",
        )
        results.append(ProcessingResult(
            step="ai_enhance",
            success=True,
            output_key=hd_key,
            output_size=len(enhanced),
        ))

        # 7. Update DynamoDB
        total_time = int((time.monotonic() - start_time) * 1000)
        ImageRepository.update_status(
            req.image_id,
            ProcessingStatus.COMPLETED,
            processed_size=len(compressed_data),
            processed_format=fmt.name,
            thumbnail_key=thumb_key,
            watermark_key=watermark_key,
            hd_key=hd_key,
            processing_time_ms=total_time,
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "image_id": req.image_id,
                "status": "completed",
                "processing_time_ms": total_time,
                "results": [
                    {"step": r.step, "success": r.success, "size": r.output_size}
                    for r in results
                ],
            }),
        }

    except Exception as e:
        ImageRepository.update_status(req.image_id, ProcessingStatus.FAILED)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Processing failed: {str(e)}",
                "image_id": req.image_id,
            }),
        }
```

### functions/notifications.py

```python
"""Notification Lambda — send push notifications after processing."""

from __future__ import annotations

import json
import os
from typing import Any

from models.image import ImageRecord
from lib.db_utils import ImageRepository


SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "arn:aws:sns:...")
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "https://sqs...")


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Send notification after image processing is complete.

    Supports:
    - SNS push notification to user devices
    - Email notification
    - WebSocket push (if connected)
    """
    for record in event.get("Records", []):
        try:
            body = json.loads(record.get("body", "{}"))
            image_id = body.get("image_id") or body.get("imageId")

            if not image_id:
                continue

            image = ImageRepository.get(image_id)
            if not image:
                continue

            # Build notification payload
            notification = {
                "type": "image_processed",
                "image_id": image_id,
                "user_id": image.user_id,
                "title": "Ảnh của bạn đã sẵn sàng!",
                "body": f"'{image.original_filename}' đã được xử lý thành công.",
                "data": {
                    "url": image.cdn_url,
                    "thumbnail_url": f"https://cdn.pixpro.example/{image.thumbnail_key}" if image.thumbnail_key else None,
                    "category": image.category.name,
                },
            }

            # In production: send to SNS/SQS, push notification, WebSocket
            _send_to_sns(notification)
            _send_to_websocket(notification)

        except Exception as e:
            print(f"Error sending notification: {e}")
            continue

    return {"statusCode": 200, "body": "Notifications sent"}


def _send_to_sns(notification: dict) -> None:
    """Send notification to SNS topic."""
    import boto3
    client = boto3.client("sns")
    try:
        client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(notification, ensure_ascii=False),
            MessageStructure="string",
        )
    except Exception:
        pass  # Logged, not critical


def _send_to_websocket(notification: dict) -> None:
    """Send notification via WebSocket (API Gateway WebSocket)."""
    pass  # In production: ApiGatewayManagementApi client
```

### functions/analytics.py

```python
"""Analytics Lambda — process image events for analytics."""

from __future__ import annotations

import json
import os
from typing import Any
from datetime import datetime

import boto3

ANALYTICS_TABLE = os.environ.get("ANALYTICS_TABLE", "pixpro-analytics")
KINESIS_STREAM = os.environ.get("KINESIS_STREAM", "pixpro-analytics-stream")


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Process analytics events from image lifecycle.

    Events:
    - image:uploaded
    - image:processed
    - image:viewed
    - image:downloaded
    - image:deleted
    """
    for record in event.get("Records", []):
        try:
            body = json.loads(record.get("body", "{}"))
            event_type = body.get("type", "unknown")

            if event_type == "image:uploaded":
                _process_upload_event(body)
            elif event_type == "image:processed":
                _process_processed_event(body)
            elif event_type == "image:viewed":
                _process_view_event(body)

        except Exception as e:
            print(f"Analytics error: {e}")
            continue

    return {"statusCode": 200, "body": "Analytics processed"}


def _process_upload_event(body: dict) -> None:
    """Process upload event for analytics."""
    _write_to_kinesis({
        "event": "upload",
        "user_id": body.get("user_id"),
        "image_id": body.get("image_id"),
        "timestamp": datetime.utcnow().isoformat(),
        "file_size": body.get("file_size"),
        "category": body.get("category"),
    })


def _process_processed_event(body: dict) -> None:
    """Process completion event for analytics."""
    _write_to_kinesis({
        "event": "processed",
        "image_id": body.get("image_id"),
        "processing_time_ms": body.get("processing_time_ms"),
        "compression_ratio": body.get("compression_ratio"),
        "timestamp": datetime.utcnow().isoformat(),
    })


def _process_view_event(body: dict) -> None:
    """Process view event for analytics."""
    _write_to_kinesis({
        "event": "view",
        "image_id": body.get("image_id"),
        "user_id": body.get("user_id"),
        "viewer_id": body.get("viewer_id"),
        "timestamp": datetime.utcnow().isoformat(),
    })


def _write_to_kinesis(data: dict) -> None:
    """Write event to Kinesis stream."""
    if not KINESIS_STREAM:
        return
    client = boto3.client("kinesis")
    try:
        client.put_record(
            StreamName=KINESIS_STREAM,
            Data=json.dumps(data),
            PartitionKey=data.get("user_id", "default"),
        )
    except Exception:
        pass
```

### tests/test_process_image.py

```python
"""Tests for image processing pipeline."""

from __future__ import annotations

import unittest
from io import BytesIO
from PIL import Image as PILImage
import json

import sys
sys.path.insert(0, "..")

from models.image import (
    ImageRecord, ImageFormat, ProcessingStatus, ImageCategory,
    ProcessingRequest, ProcessingResult,
)
from lib.image_utils import ImageProcessor, ImageValidator, ValidationError


class TestImageProcessor(unittest.TestCase):
    """Test ImageProcessor — core business logic, no infrastructure needed."""

    @classmethod
    def setUpClass(cls) -> None:
        # Create a test image
        cls.test_image = PILImage.new("RGB", (1920, 1080), color="blue")
        cls.test_bytes = BytesIO()
        cls.test_image.save(cls.test_bytes, format="JPEG", quality=95)
        cls.test_bytes = cls.test_bytes.getvalue()

    def test_resize_maintains_aspect_ratio(self):
        result, w, h = ImageProcessor.resize(self.test_bytes, 800, 600)
        self.assertGreater(len(result), 0)
        self.assertEqual(w, 800)
        self.assertEqual(h, 600)

    def test_resize_contain(self):
        result, w, h = ImageProcessor.resize(
            self.test_bytes, 800, 600, fit="contain"
        )
        # 1920/1080 = 16:9, with contain, height should be 600
        self.assertLessEqual(w, 800)
        self.assertEqual(h, 600)

    def test_compress_reduces_size(self):
        result, quality, fmt = ImageProcessor.compress(
            self.test_bytes, quality="low"
        )
        self.assertLess(len(result), len(self.test_bytes))
        self.assertEqual(fmt, ImageFormat.JPEG)

    def test_compress_webp(self):
        result, quality, fmt = ImageProcessor.compress(
            self.test_bytes, quality="high", target_format=ImageFormat.WEBP
        )
        self.assertGreater(len(result), 0)
        self.assertEqual(fmt, ImageFormat.WEBP)

    def test_thumbnail_generation(self):
        result, w, h = ImageProcessor.create_thumbnail(
            self.test_bytes, size_name="medium"
        )
        self.assertEqual(w, 256)
        self.assertEqual(h, 256)

    def test_thumbnail_small(self):
        result, w, h = ImageProcessor.create_thumbnail(
            self.test_bytes, size_name="small"
        )
        self.assertEqual(w, 128)
        self.assertEqual(h, 128)

    def test_watermark_adds_text(self):
        result = ImageProcessor.apply_watermark(
            self.test_bytes, watermark_text="PixPro Test"
        )
        self.assertGreater(len(result), 0)
        # Watermark is PNG (can have transparency)
        img = PILImage.open(BytesIO(result))
        self.assertEqual(img.format, "PNG")

    def test_ai_enhance(self):
        result = ImageProcessor.ai_enhance(self.test_bytes)
        self.assertGreater(len(result), 0)

    def test_get_image_info(self):
        info = ImageProcessor.get_image_info(self.test_bytes)
        self.assertEqual(info["width"], 1920)
        self.assertEqual(info["height"], 1080)
        self.assertEqual(info["format"], "JPEG")

    def test_compress_with_different_qualities(self):
        sizes = {}
        for name in ["max", "high", "medium", "low", "minimal"]:
            result, _, _ = ImageProcessor.compress(self.test_bytes, quality=name)
            sizes[name] = len(result)

        # Higher quality should produce larger files
        self.assertGreater(sizes["max"], sizes["low"])

    def test_resize_tiny_image(self):
        tiny = PILImage.new("RGB", (50, 50), color="red")
        tiny_bytes = BytesIO()
        tiny.save(tiny_bytes, format="JPEG")
        result, w, h = ImageProcessor.resize(tiny_bytes.getvalue(), 25, 25)
        self.assertEqual(w, 25)
        self.assertEqual(h, 25)


class TestImageValidator(unittest.TestCase):
    """Test validation rules."""

    @classmethod
    def setUpClass(cls) -> None:
        img = PILImage.new("RGB", (100, 100), color="green")
        cls.valid_bytes = BytesIO()
        img.save(cls.valid_bytes, format="JPEG")
        cls.valid_bytes = cls.valid_bytes.getvalue()

    def test_valid_image(self):
        ImageValidator.validate(self.valid_bytes, "image/jpeg")

    def test_invalid_content_type(self):
        with self.assertRaises(ValidationError):
            ImageValidator.validate(self.valid_bytes, "application/pdf")

    def test_small_dimensions(self):
        tiny = PILImage.new("RGB", (10, 10), color="red")
        tiny_bytes = BytesIO()
        tiny.save(tiny_bytes, format="JPEG")
        with self.assertRaises(ValidationError):
            ImageValidator.validate(tiny_bytes.getvalue(), "image/jpeg")


class TestImageRecord(unittest.TestCase):
    """Test ImageRecord model."""

    def test_new_record(self):
        record = ImageRecord.new(
            user_id="user1",
            filename="test.jpg",
            fmt=ImageFormat.JPEG,
            size=1024,
            width=800,
            height=600,
        )
        self.assertEqual(record.user_id, "user1")
        self.assertEqual(record.status, ProcessingStatus.PENDING)
        self.assertIsNotNone(record.image_id)

    def test_s3_keys(self):
        record = ImageRecord.new("u1", "photo.jpg", ImageFormat.JPEG, 100, 400, 300)
        self.assertIn("raw", record.s3_key_original)
        self.assertIn("processed", record.s3_key_processed)
        self.assertIn("thumbnails", record.s3_key_thumbnail)

    def test_to_dict_roundtrip(self):
        record = ImageRecord.new("u1", "test.jpg", ImageFormat.JPEG, 1024, 1920, 1080)
        data = record.to_dict()
        restored = ImageRecord.from_dict(data)
        self.assertEqual(record.image_id, restored.image_id)
        self.assertEqual(record.original_width, restored.original_width)

    def test_cdn_url(self):
        record = ImageRecord.new("u1", "test.jpg", ImageFormat.JPEG, 100, 400, 300)
        self.assertTrue(record.cdn_url.startswith("https://cdn."))
        self.assertIn("processed", record.cdn_url)


class TestProcessingRequest(unittest.TestCase):
    """Test ProcessingRequest parsing from S3 events."""

    def test_from_s3_event(self):
        event = {
            "Records": [
                {
                    "eventSource": "aws:s3",
                    "s3": {
                        "bucket": {"name": "my-bucket"},
                        "object": {"key": "raw/user123/img001/photo.jpg"},
                    },
                }
            ]
        }
        requests = ProcessingRequest.from_s3_event(event)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].image_id, "img001")
        self.assertEqual(requests[0].user_id, "user123")
        self.assertEqual(requests[0].bucket, "my-bucket")

    def test_ignore_non_s3_events(self):
        event = {
            "Records": [
                {"eventSource": "aws:sqs", "body": "test"}
            ]
        }
        requests = ProcessingRequest.from_s3_event(event)
        self.assertEqual(len(requests), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

### tests/test_handlers.py

```python
"""Tests for Lambda handlers with mocked infrastructure."""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os

import sys
sys.path.insert(0, "..")


class TestUploadHandler(unittest.TestCase):
    """Test upload Lambda handler."""

    @patch("functions.upload.S3Manager.upload_bytes")
    @patch("functions.upload.ImageRepository.save")
    def test_upload_success(self, mock_save, mock_upload):
        from functions.upload import lambda_handler

        # Create a small valid JPEG
        from PIL import Image
        from io import BytesIO
        import base64

        img = Image.new("RGB", (100, 100), color="blue")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        event = {
            "body": json.dumps({
                "user_id": "user1",
                "filename": "test.jpg",
                "content_type": "image/jpeg",
                "body": b64,
            })
        }

        result = lambda_handler(event, None)
        body = json.loads(result["body"])

        self.assertEqual(result["statusCode"], 200)
        self.assertIn("image_id", body)

    def test_upload_missing_fields(self):
        from functions.upload import lambda_handler

        event = {"body": json.dumps({"user_id": "user1"})}
        result = lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 400)

    def test_upload_invalid_file(self):
        from functions.upload import lambda_handler
        import base64

        event = {
            "body": json.dumps({
                "user_id": "user1",
                "filename": "test.txt",
                "content_type": "text/plain",
                "body": base64.b64encode(b"not an image").decode(),
            })
        }

        result = lambda_handler(event, None)
        self.assertEqual(result["statusCode"], 400)


class TestProcessImageHandler(unittest.TestCase):
    """Test process image Lambda handler."""

    @patch("functions.process_image.S3Manager.download_bytes")
    @patch("functions.process_image.S3Manager.upload_bytes")
    @patch("functions.process_image.ImageRepository.update_status")
    def test_process_success(self, mock_status, mock_upload, mock_download):
        from functions.process_image import lambda_handler
        from PIL import Image
        from io import BytesIO

        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        mock_download.return_value = buf.getvalue()

        event = {
            "Records": [
                {
                    "eventSource": "aws:s3",
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "raw/user1/img001/photo.jpg"},
                    },
                }
            ]
        }

        result = lambda_handler(event, None)
        body = json.loads(result["body"])

        self.assertEqual(result["statusCode"], 200)
        self.assertIn("processing_time_ms", body)
        self.assertEqual(body["status"], "completed")

    def test_invalid_s3_event(self):
        from functions.process_image import lambda_handler

        result = lambda_handler({"Records": []}, None)
        self.assertEqual(result["statusCode"], 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

### infra/serverless.yml

```yaml
# serverless.yml — Serverless Framework configuration
service: pixpro-image-processor

provider:
  name: aws
  runtime: python3.11
  region: ap-southeast-1
  stage: ${opt:stage, 'dev'}
  memorySize: 512
  timeout: 30
  logRetentionInDays: 14

  iam:
    role:
      statements:
        - Effect: Allow
          Action:
            - s3:GetObject
            - s3:PutObject
          Resource: "arn:aws:s3:::pixpro-*/*"
        - Effect: Allow
          Action:
            - dynamodb:GetItem
            - dynamodb:PutItem
            - dynamodb:UpdateItem
            - dynamodb:Query
          Resource: "arn:aws:dynamodb:*:*:table/pixpro-*"
        - Effect: Allow
          Action:
            - sns:Publish
          Resource: "arn:aws:sns:*:*:pixpro-*"

  environment:
    UPLOAD_BUCKET: pixpro-uploads-${self:provider.stage}
    PROCESSED_BUCKET: pixpro-processed-${self:provider.stage}
    IMAGE_TABLE: pixpro-images-${self:provider.stage}
    ANALYTICS_TABLE: pixpro-analytics-${self:provider.stage}

functions:
  upload:
    handler: functions/upload.lambda_handler
    events:
      - http:
          path: /api/v1/images/upload
          method: post
          cors: true
          authorizer: aws_iam

  processImage:
    handler: functions/process_image.lambda_handler
    events:
      - s3:
          bucket: pixpro-uploads-${self:provider.stage}
          event: s3:ObjectCreated:*
          rules:
            - prefix: raw/

  notifications:
    handler: functions/notifications.lambda_handler
    events:
      - sqs:
          arn:
            Fn::GetAtt:
              - ProcessQueue
              - Arn

  analytics:
    handler: functions/analytics.lambda_handler
    events:
      - sqs:
          arn:
            Fn::GetAtt:
              - AnalyticsQueue
              - Arn

  auth:
    handler: functions/auth.lambda_handler
    events:
      - http:
          path: /api/v1/auth/{proxy+}
          method: any
          cors: true

resources:
  Resources:
    ProcessedBucket:
      Type: AWS::S3::Bucket
      Properties:
        BucketName: pixpro-processed-${self:provider.stage}
        LifecycleConfiguration:
          Rules:
            - Id: GlacierTransition
              Status: Enabled
              Transitions:
                - Days: 90
                  StorageClass: GLACIER

    ImagesTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: pixpro-images-${self:provider.stage}
        BillingMode: PAY_PER_REQUEST
        AttributeDefinitions:
          - AttributeName: image_id
            AttributeType: S
          - AttributeName: user_id
            AttributeType: S
        KeySchema:
          - AttributeName: image_id
            KeyType: HASH
        GlobalSecondaryIndexes:
          - IndexName: user_id-index
            KeySchema:
              - AttributeName: user_id
                KeyType: HASH
            Projection:
              ProjectionType: ALL

    ProcessQueue:
      Type: AWS::SQS::Queue
      Properties:
        QueueName: pixpro-process-${self:provider.stage}
        VisibilityTimeout: 60
        RedrivePolicy:
          deadLetterTargetArn: !GetAtt ProcessDLQ.Arn
          maxReceiveCount: 3

    ProcessDLQ:
      Type: AWS::SQS::Queue
      Properties:
        QueueName: pixpro-process-dlq-${self:provider.stage}

    AnalyticsQueue:
      Type: AWS::SQS::Queue
      Properties:
        QueueName: pixpro-analytics-${self:provider.stage}
```

---

## Khi nào dùng / Khi nào không

| Khi nào dùng Serverless | Khi nào không dùng Serverless |
|------------------------|-------------------------------|
| **Event-driven workloads** — Xử lý file, notification, stream | **Long-running processes** — > 15 phút (cần containerized) |
| **Traffic thất thường** — Peak 100x so với baseline | **Predictable, steady traffic** — Reserved instances rẻ hơn |
| **Microservices nhỏ** — Single-purpose functions | **Monolith migration** — Không thể split dễ dàng |
| **Rapid prototyping** — Time-to-market quan trọng | **Low latency requirement** — Cold start có thể > 1s |
| **Startup / SMB** — Không có DevOps team | **Complex stateful apps** — WebSocket gaming, real-time sync |
| **Batch processing** — Ảnh, video, data transform | **High compute GPU/ML** — Instance-based vẫn nhanh hơn |
| **IoT data pipeline** — Many devices, small data | **Legacy enterprise** — Compliance, on-premise requirement |
| **API backend** — REST/GraphQL với auto-scaling | **Tightly coupled systems** — Serverless khó debug distributed |

---

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| **Zero server management**: Không patching, không OS updates | **Cold start latency**: Function first invocation chậm (100ms-5s) |
| **Auto-scaling hoàn hảo**: Scale từ 0 đến hàng nghìn concurrent | **Timeout limit**: AWS Lambda 15 phút, không cho long-running |
| **Pay-per-use**: Chỉ trả cho compute time thực tế | **Debugging khó**: Distributed tracing phức tạp |
| **Built-in HA và fault tolerance**: AWS replicate tự động | **Vendor lock-in**: Khó migrate giữa các cloud provider |
| **Giảm operational cost**: Không cần infrastructure team | **Cost unpredictable**: Traffic spike → cost spike bất ngờ |
| **Event-driven tự nhiên**: S3, SQS, DynamoDB Streams triggers | **State management khó**: Function phải stateless |
| **Security built-in**: IAM roles, VPC, KMS | **Resource limits**: 512MB-10GB memory, 100MB deployment package |
| **Rapid deployment**: Serverless Framework, SAM, Terraform | **Testing complexity**: Cần mock AWS services |
| **Per-function scaling**: Mỗi function scale độc lập | **Concurrency limits**: Mặc định 1000 concurrent Lambda (có thể tăng) |
| **Eco-friendly**: Chỉ dùng resource khi thực sự cần | **No local filesystem**: /tmp chỉ 512MB |

---

## Công cụ và Framework

### Cloud Providers
| Provider | FaaS | Database | Storage |
|----------|------|----------|---------|
| **AWS** | Lambda, Fargate | DynamoDB, Aurora Serverless | S3, EFS |
| **Azure** | Functions, Container Apps | Cosmos DB, SQL Serverless | Blob Storage |
| **GCP** | Cloud Functions, Cloud Run | Firestore, BigQuery | Cloud Storage |
| **Cloudflare** | Workers, Pages Functions | D1, KV, Durable Objects | R2 |
| **Alibaba** | Function Compute | Table Store, OceanBase | OSS |

### Frameworks & Tools
| Tool | Mô tả |
|------|-------|
| **Serverless Framework** | IaC + deployment cho multi-cloud |
| **AWS SAM** | AWS-native serverless template |
| **Terraform** | Infrastructure as Code (multi-cloud) |
| **Pulumi** | IaC với Python/TypeScript/Go |
| **Chalice** | Python serverless microframework (AWS) |
| **Zappa** | Django/Flask → serverless deployment |
| **Mangum** | ASGI adapter cho Lambda (FastAPI) |
| **LocalStack** | AWS services emulator cho local dev |
| **AWS X-Ray** | Distributed tracing for serverless |
| **Dashbird / Lumigo** | Serverless monitoring và debugging |

### Python Libraries
| Library | Công dụng |
|---------|-----------|
| **boto3** | AWS SDK cho Python |
| **Pillow** | Xử lý ảnh |
| **Pydantic** | Data validation (model layer) |
| **AWS Lambda Powertools** | Logging, tracing, middleware cho Lambda |
| **moto** | Mock AWS services cho testing |
| **s3fs** | S3 filesystem interface |
| **jsonlines** | JSON streaming (analytics) |

---

## Kiểm thử

### Chiến lược kiểm thử Serverless

```
1. Unit Tests (nhanh, local)
   ├── Test pure business logic (ImageProcessor, validators)
   ├── Test models (entities, serialization)
   └── Test handlers với mocked AWS

2. Integration Tests (cần AWS hoặc LocalStack)
   ├── Test S3 triggers → Lambda
   ├── Test DynamoDB read/write
   └── Test Step Functions workflow

3. End-to-End Tests (AWS account thật)
   ├── Upload file → verify processed output
   ├── Test error scenarios (invalid file, timeout)
   └── Performance test with concurrent uploads
```

Xem test files ở phần code mẫu (`tests/test_process_image.py`, `tests/test_handlers.py`) cho unit test examples.

```bash
# Chạy unit tests
python -m pytest tests/test_process_image.py -v --cov=lib --cov=models

# Chạy integration test với LocalStack
# Cần cài LocalStack: pip install localstack
localstack start -d
python -m pytest tests/test_integration.py -v

# Deploy và test thật
serverless deploy --stage dev
serverless invoke --function upload --path test_events/upload.json
```

---

## Kết luận

**Serverless Architecture** là một bước tiến lớn trong cách chúng ta xây dựng và vận hành ứng dụng. Nó giải phóng developers khỏi gánh nặng infrastructure, cho phép tập trung hoàn toàn vào business logic.

### Best Practices

1. **Design for failure**: Serverless functions fail — hãy thiết kế idempotent handlers, retry logic, DLQ.

2. **Minimize cold starts**: Dùng provisioned concurrency, giữ deployment package nhỏ, dùng các runtime warm (Python/Node/Go).

3. **Stateless là chìa khóa**: Không lưu state trong function. Dùng DynamoDB, S3, Redis cho state.

4. **Single responsibility**: Mỗi function làm một việc. Nếu function có nhiều hơn 3-4 bước, dùng Step Functions.

5. **Infrastructure as Code**: Mọi resource (S3, DynamoDB, SQS) phải được định nghĩa trong code. Không dùng AWS Console.

6. **Monitor và trace**: Dùng X-Ray, CloudWatch Logs, và third-party tools (Dashbird, Lumigo) để debug.

7. **Cost optimization**: Set memory phù hợp (memory = compute power). Dùng reserved concurrency để tránh cost spike.

8. **Security first**: IAM roles least privilege, environment variables cho secrets, VPC cho database.

### Golden Rules

| Rule | Giải thích |
|------|-----------|
| **Function = one action** | Mỗi Lambda chỉ làm đúng một việc |
| **Event-driven > Request-driven** | Dùng event để trigger functions, không gọi trực tiếp |
| **State belongs to database** | Function không lưu state local |
| **Fail fast, fail gracefully** | Validate input đầu tiên, dùng DLQ cho failures |
| **Measure cold start** | Theo dõi P50/P95/P99 cold start latency |
| **Set memory = compute need** | Memory cao = CPU cao, nhưng cost cũng cao |

Serverless không phải là giải pháp cho mọi bài toán, nhưng nó là công cụ cực kỳ mạnh mẽ cho event-driven, batch processing, và API workloads. Khi được áp dụng đúng, serverless giúp giảm 70-90% chi phí vận hành so với traditional infrastructure.
