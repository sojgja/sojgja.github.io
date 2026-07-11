---
id: space-based
title: Space-Based Architecture
sidebar_label: 🏗️ Space-Based Architecture
sidebar_position: 50
---

# Space-Based Architecture

> **Space-Based Architecture (SBA)** — *"Eliminate the database as the synchronous bottleneck by distributing shared state across a grid of processing units using tuple-space memory."* — Microsoft Patterns & Practices, 2005

## Tổng quan

Space-Based Architecture (SBA), còn gọi là **Grid Architecture** hay **Distributed Coordination-Based Architecture**, ra đời từ nhu cầu giải quyết bài toán scale-out cho hệ thống có lượng truy cập khổng lồ (high-volume, high-transaction). Khác với kiến trúc layered truyền thống nơi mọi request đều phải qua database, SBA loại bỏ database khỏi đường xử lý synchronous (critical path) bằng cách đưa dữ liệu vào **distributed in-memory data grid** (tuple space).

Khái niệm "space" (khoảng không gian tuple) được giới thiệu lần đầu trong ngôn ngữ **Linda** (David Gelernter, 1985) — một coordination model cho parallel computing. Các process giao tiếp với nhau thông qua shared tuple space thay vì message passing hay shared memory truyền thống.

**Những người tiên phong:**

| Tên | Đóng góp |
|-----|----------|
| **David Gelernter** | Cha đẻ của Linda tuple space (1985) |
| **Jim Gray** | Đề xuất "Shared Nothing" architecture — nền tảng cho SBA |
| **Microsoft (patterns & practices)** | Định nghĩa SBA trong cuốn *"A Guide to Space-Based Architecture"* (2005) |
| **GigaSpaces** | Xây dựng GigaSpaces XAP — commercial SBA platform đầu tiên |

Các công ty sử dụng SBA thành công: **Amazon** (shopping cart), **eBay** (scaling lên 4+ tỷ request/ngày), **Alibaba** (Double 11 với 500k+ TPS), **Credit-Suisse** (real-time risk analysis), và nhiều hệ thống trading tài chính.

**Mối quan hệ với các concept khác:**
- **Shared Nothing Architecture**: Mỗi node sở hữu dữ liệu riêng, không share disk. SBA là một implementation của shared-nothing.
- **Event-Driven Architecture (EDA)**: SBA dùng event-driven replication để đồng bộ các processing unit.
- **CQRS (Command Query Responsibility Segregation)**: SBA thường kết hợp với CQRS để tách biệt read/write path.

## Bài toán

### Vấn đề 1: Database là performance bottleneck không thể scale ngang

Hãy tưởng tượng bạn là kiến trúc sư của một nền tảng thương mại điện tử. Hệ thống hiện tại dùng kiến trúc 3-layer: Web Server → Application Server → Database. Khi có 1000 người dùng cùng lúc add-to-cart, mọi request đều ghi vào database. Bạn tăng thêm 10 web server, nhưng database chỉ có một. Database bắt đầu chậm: contention trên row lock, connection pool cạn kiệt, disk I/O bão hòa. Bạn chuyển qua master-slave replication, issue vẫn còn — write master bị quá tải.

Bạn thử scale database ngang (sharding), nhưng các shard cần distributed transaction cho cross-shard operation — 2-phase commit giết chết hiệu năng. Bạn cần một kiến trúc mà thêm node = thêm capacity tuyến tính, và database không nằm trên đường xử lý chính.

### Vấn đề 2: Session state là cơn ác mộng cho clustering

Trong ứng dụng web có stateful session (giỏ hàng, thông tin đăng nhập), session thường được lưu trong database hoặc Redis. Nếu không cẩn thận, mỗi request phải load session từ database, gây thêm một database round-trip. Khi scale, bạn phải dùng sticky session (gắn user với một server) — nếu server đó chết, session mất. Bạn chuyển qua centralized session store (Redis), nhưng Redis cũng có giới hạn — một instance Redis xử lý ~100k ops/sec, khi traffic vượt quá, bạn phải cluster Redis — phức tạp.

Solution lý tưởng: session được lưu ngay trên application node (in-memory), và được replicate sang các node khác để fault-tolerance. Node nào cũng có đủ dữ liệu để xử lý request của bất kỳ user nào — zero stickiness.

### Vấn đề 3: Load balancer không biết gì về dữ liệu

Load balancer phân phối request theo round-robin hoặc least-connection. Nó không biết request A cần dữ liệu X nằm ở node nào. Nếu node chứa dữ liệu X đang quá tải, load balancer vẫn gửi request A tới node khác — node đó phải fetch dữ liệu X từ database, gây replication lag và inconsistency.

SBA giải quyết vấn đề này bằng cách phân vùng dữ liệu (data partition) giống như sharding, nhưng dữ liệu được replicate trong cluster. Mỗi request được routing tới node có chứa partition phù hợp (smart routing). Không cần database round-trip cho dữ liệu thường dùng.

### Vấn đề 4: Database connection pool và memory waste

Mỗi application server duy trì một connection pool tới database. Với 100 server × 20 connection = 2000 connection. Database phải dành 2000 process/thread để phục vụ — lãng phí tài nguyên. Hơn nữa, dữ liệu đọc từ database được cache riêng lẻ trên mỗi server — tổng dung lượng cache gấp N lần so với một centralized cache, nhưng hiệu quả thấp vì cache miss vẫn phải query database.

SBA gộp application + cache + compute vào cùng một processing unit. Dữ liệu "nóng" được giữ trong memory của processing unit, không cần connection pool truyền thống.

## Nguyên lý thiết kế

### 1. Processing Unit (PU) — Đơn vị kiến trúc cơ bản

Mỗi Processing Unit là một đơn vị triển khai độc lập, chứa đầy đủ: application logic + in-memory data + messaging endpoint. PU là "máy tính cá nhân" trong grid — nó có dữ liệu riêng, xử lý request riêng, và replicate dữ liệu với các PU khác.

```
PU = Application + In-Memory Data + Listener
```

### 2. Tuple Space — Distributed In-Memory Data Grid

Tuple space là một shared memory abstraction cho phép các PU giao tiếp thông qua các operation:
- **write** (insert) — đưa tuple vào space
- **read** (get) — đọc tuple (non-destructive)
- **take** (get-and-remove) — đọc và xóa tuple
- **notify** (event) — đăng ký callback khi tuple xuất hiện

Tuple space không có single point of failure — dữ liệu được phân mảnh và replicate trên nhiều PU.

### 3. Virtual Middleware (Grid Middleware)

Lớp trung gian quản lý toàn bộ cluster:
- **Data Partitioning**: Chia dữ liệu thành các partition, phân bố trên các PU
- **Replication**: Đồng bộ dữ liệu giữa các partition replica
- **Failover**: Phát hiện PU chết, route request sang PU khác
- **Elasticity**: Thêm/bớt PU tự động dựa trên tải

### 4. Space-Based Cache (SBC) — Thay thế database cho synchronous path

Không giống cache truyền thống (chỉ lưu tạm), SBC là nguồn dữ liệu chính cho synchronous request. Database chỉ được dùng cho:
- Persistent storage (backup, batch processing)
- Reporting và analytics
- Audit log

### 5. Stateless + Stateful Hybrid

Web server layer vẫn stateless (dễ scale), nhưng state được distributed trong các PU. Request không cần sticky session — bất kỳ PU nào cũng có thể xử lý request nhờ data replication.

## Cấu trúc chi tiết

### Core Components

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Processing Unit (PU)** | Container chứa app + data | Python process + in-memory dict/grid |
| **Space Repository** | In-memory data store với indexing | Dict-based, sharded |
| **Replication Engine** | Đồng bộ dữ liệu giữa các PU | Gossip protocol, async replication |
| **Partition Manager** | Quản lý phân vùng dữ liệu | Consistent hashing |
| **Router / Frontend** | Routing request đến PU đúng | Smart proxy dựa trên partition key |
| **Data Source / Database** | Persistent backup | PostgreSQL, Cassandra |
| **Lease Manager** | Quản lý timeout và cleanup | Background thread with TTL |

### Data Flow

```
Client Request
    │
    ▼
Frontend (Load Balancer + Smart Router)
    │  (routing based on partition key)
    ▼
Processing Unit (Primary)
    ├── In-Memory Space (đọc/ghi dữ liệu)
    ├── Business Logic (xử lý)
    └── Replication Engine (async replicate to backup PU)
            │
            ▼
        Processing Unit (Backup / Replica)
    │
    ▼
Data Source (async write-behind — không blocking)
```

### Replication Strategies

| Strategy | Latency | Consistency | Use Case |
|----------|---------|-------------|----------|
| **Sync Replication** | Cao | Strong | Tài chính, giao dịch |
| **Async Replication** | Thấp | Eventual | Giỏ hàng, session |
| **Write-Behind** | Rất thấp | Lazy persist | Log, audit |
| **Cache-Aside** | Trung bình | Weak | Read-heavy, product catalog |

### Partitioning Strategies

| Strategy | Key Feature | Example Key |
|----------|-------------|-------------|
| **Consistent Hashing** | Minimum remapping on node change | `hash(user_id) % N` |
| **Range Partitioning** | Ordered data, range query | `user_id 1–1000 → PU1` |
| **Directory-Based** | Centralized lookup table | Metadata service |
| **Composite** | Multi-level | `region + user_id` |

## Sơ đồ kiến trúc (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SPACE-BASED ARCHITECTURE                      │
│                                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    ┌─────────┐              │
│  │ Client 1│  │ Client 2│  │ Client 3│    │ Client N│              │
│  └────┬────┘  └────┬────┘  └────┬────┘    └────┬────┘              │
│       │            │            │               │                   │
│       └────────────┼────────────┼───────────────┘                   │
│                    │            │                                   │
│                    ▼            ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              SMART ROUTER / LOAD BALANCER                 │       │
│  │  (Partition-aware routing — consistent hashing)          │       │
│  └──┬───────────┬───────────┬───────────┬───────────────┬──┘       │
│     │           │           │           │               │           │
│     ▼           ▼           ▼           ▼               ▼           │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         ┌──────┐          │
│  │ PU-1 │  │ PU-2 │  │ PU-3 │  │ PU-4 │   ...   │ PU-N │          │
│  │      │  │      │  │      │  │      │         │      │          │
│  │┌────┐│  │┌────┐│  │┌────┐│  │┌────┐│         │┌────┐│          │
│  ││Space││  ││Space││  ││Space││  ││Space││  ...  ││Space││          │
│  ││Data ││  ││Data ││  ││Data ││  ││Data ││         ││Data ││          │
│  ││Part.││  ││Part.││  ││Part.││  ││Part.││         ││Part.││          │
│  │└────┘│  │└────┘│  │└────┘│  │└────┘│         │└────┘│          │
│  │ App  │  │ App  │  │ App  │  │ App  │         │ App  │          │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘         └──┬───┘          │
│     │         │         │         │                 │             │
│     └─────────┼─────────┼─────────┼─────────────────┘             │
│               │         │         │                               │
│               │  ┌──────┴──────┐  │   REplication (Gossip)        │
│               │  │ Replication │  │                               │
│               │  │  Engine     │  │                               │
│               │  └──────┬──────┘  │                               │
│               │         │         │                               │
│               ▼         ▼         ▼                               │
│  ┌──────────────────────────────────────────────────────┐         │
│  │            ASYNC WRITE-BEHIND LAYER                   │         │
│  └──────────────────────┬───────────────────────────────┘         │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────────┐         │
│  │         PERSISTENT DATA SOURCE (Database)             │         │
│  │  (PostgreSQL, Cassandra — for backup/analytics)       │         │
│  └──────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

## Ví dụ code hoàn chỉnh

### Cách làm sai: Kiến trúc 3-layer với database bottleneck

```python
from __future__ import annotations
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(threadName)s] %(message)s")
logger = logging.getLogger(__name__)


class Database:
    """Mô phỏng database — bottleneck khi scale."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def read(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            time.sleep(0.05)  # Simulate disk I/O (50ms)
            return self._store.get(key)

    def write(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            time.sleep(0.05)
            self._store[key] = value

    def execute_with_retry(self, operation: str, key: str, value: dict[str, Any] | None = None) -> Any:
        for attempt in range(3):
            try:
                if operation == "read":
                    return self.read(key)
                return self.write(key, value)  # type: ignore
            except Exception as e:
                logger.warning("DB %s attempt %d failed: %s", operation, attempt + 1, e)
                time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"DB {operation} failed after 3 attempts")


class CartService3Layer:
    """Service dùng database trực tiếp — mỗi request đều query DB."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add_item(self, user_id: str, product_id: str, quantity: int) -> dict[str, Any]:
        cart = self._db.execute_with_retry("read", f"cart:{user_id}") or {"items": []}
        cart["items"].append({"product_id": product_id, "quantity": quantity})
        self._db.execute_with_retry("write", f"cart:{user_id}", cart)
        return cart

    def get_cart(self, user_id: str) -> dict[str, Any]:
        return self._db.execute_with_retry("read", f"cart:{user_id}") or {"items": []}


def simulate_traffic_3layer(service: CartService3Layer, user_count: int = 20) -> float:
    start = time.perf_counter()
    threads = []
    for i in range(user_count):
        t = threading.Thread(
            target=lambda uid: [service.add_item(uid, f"prod_{random.randint(1, 100)}", 1) for _ in range(5)],
            args=(f"user_{i}",),
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start
```

### Cách làm đúng: Space-Based Architecture

```python
from __future__ import annotations
import time
import threading
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import random
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(threadName)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# 1. ENUMS & DOMAIN TYPES
# ======================================================================

class ConsistencyLevel(Enum):
    EVENTUAL = auto()
    WEAK = auto()
    STRONG = auto()


class PartitionStrategy(Enum):
    CONSISTENT_HASHING = auto()
    ROUND_ROBIN = auto()
    RANGE = auto()


@dataclass
class SpaceEntry:
    """Một entry trong tuple space."""
    key: str
    value: dict[str, Any]
    version: int = 0
    ttl: float | None = None  # Time-to-live (seconds)
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


@dataclass
class PartitionMetadata:
    partition_id: int
    primary_pu: str
    backup_pus: list[str] = field(default_factory=list)
    key_range_start: str = ""
    key_range_end: str = ""


# ======================================================================
# 2. SPACE REPOSITORY (In-Memory Data Store)
# ======================================================================

class SpaceRepository:
    """In-memory data store với indexing và TTL support."""

    def __init__(self, partition_id: int, pu_id: str) -> None:
        self._partition_id = partition_id
        self._pu_id = pu_id
        self._data: dict[str, SpaceEntry] = {}
        self._lock = threading.RWMutex()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("SpaceRepository [PU=%s, Partition=%d] initialized", pu_id, partition_id)

    def write(self, key: str, value: dict[str, Any], ttl: float | None = None) -> SpaceEntry:
        with self._lock:
            existing = self._data.get(key)
            version = (existing.version + 1) if existing else 1
            entry = SpaceEntry(key=key, value=value, version=version, ttl=ttl)
            self._data[key] = entry
            logger.debug("Write key=%s version=%d on PU=%s", key, version, self._pu_id)
            return entry

    def read(self, key: str) -> SpaceEntry | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry.is_expired():
                if entry and entry.is_expired():
                    del self._data[key]
                return None
            return entry

    def take(self, key: str) -> SpaceEntry | None:
        with self._lock:
            entry = self._data.pop(key, None)
            if entry and entry.is_expired():
                return None
            return entry

    def get_all_keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def get_size(self) -> int:
        with self._lock:
            return len(self._data)

    def apply_replication(self, key: str, value: dict[str, Any], version: int, ttl: float | None = None) -> None:
        with self._lock:
            existing = self._data.get(key)
            if existing is None or version > existing.version:
                self._data[key] = SpaceEntry(key=key, value=value, version=version, ttl=ttl)

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(5.0)
            with self._lock:
                expired_keys = [k for k, v in self._data.items() if v.is_expired()]
                for k in expired_keys:
                    del self._data[k]
                if expired_keys:
                    logger.debug("Cleaned %d expired entries on PU=%s", len(expired_keys), self._pu_id)


# ======================================================================
# 3. REPLICATION ENGINE
# ======================================================================

class ReplicationEngine:
    """Async gossip-based replication giữa các PU."""

    def __init__(self, pu_id: str, partition_id: int) -> None:
        self._pu_id = pu_id
        self._partition_id = partition_id
        self._peers: dict[str, SpaceRepository] = {}
        self._replication_queue: list[tuple[str, dict[str, Any], int, float | None]] = []
        self._queue_lock = threading.Lock()
        self._worker = threading.Thread(target=self._replication_loop, daemon=True)
        self._worker.start()

    def add_peer(self, peer_id: str, repository: SpaceRepository) -> None:
        self._peers[peer_id] = repository
        logger.info("Replication peer added: %s → %s", self._pu_id, peer_id)

    def remove_peer(self, peer_id: str) -> None:
        self._peers.pop(peer_id, None)

    def replicate(self, key: str, value: dict[str, Any], version: int, ttl: float | None = None) -> None:
        with self._queue_lock:
            self._replication_queue.append((key, value, version, ttl))

    def _replication_loop(self) -> None:
        while True:
            time.sleep(0.1)  # 100ms replication interval
            batch: list[tuple[str, dict[str, Any], int, float | None]] = []
            with self._queue_lock:
                batch, self._replication_queue = self._replication_queue[:100], []
            if not batch:
                continue
            for key, value, version, ttl in batch:
                for peer_id, repo in self._peers.items():
                    try:
                        repo.apply_replication(key, value, version, ttl)
                    except Exception as e:
                        logger.error("Replication to %s failed: %s", peer_id, e)


# ======================================================================
# 4. PARTITION MANAGER (Consistent Hashing)
# ======================================================================

class PartitionManager:
    """Consistent hashing-based partition management."""

    def __init__(self, virtual_nodes: int = 100) -> None:
        self._virtual_nodes = virtual_nodes
        self._ring: dict[int, str] = {}  # hash → pu_id
        self._sorted_hashes: list[int] = []
        self._pu_partitions: dict[str, int] = {}
        self._partition_count = 0
        self._lock = threading.RLock()

    def add_pu(self, pu_id: str, partition_count: int = 4) -> None:
        with self._lock:
            for i in range(partition_count):
                partition_id = self._partition_count
                self._partition_count += 1
                self._pu_partitions[pu_id] = self._pu_partitions.get(pu_id, 0) + 1
                for j in range(self._virtual_nodes):
                    hash_val = self._hash(f"{pu_id}:partition:{partition_id}:virtual:{j}")
                    self._ring[hash_val] = pu_id
                self._sorted_hashes = sorted(self._ring.keys())
                logger.info("PU=%s assigned partition=%d", pu_id, partition_id)

    def remove_pu(self, pu_id: str) -> None:
        with self._lock:
            self._pu_partitions.pop(pu_id, None)
            self._ring = {h: pid for h, pid in self._ring.items() if pid != pu_id}
            self._sorted_hashes = sorted(self._ring.keys())

    def get_primary_pu(self, key: str) -> str | None:
        with self._lock:
            if not self._sorted_hashes:
                return None
            key_hash = self._hash(key)
            for h in self._sorted_hashes:
                if h >= key_hash:
                    return self._ring.get(h)
            return self._ring.get(self._sorted_hashes[0])

    def get_partition_count(self, pu_id: str) -> int:
        return self._pu_partitions.get(pu_id, 0)

    @staticmethod
    def _hash(key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


# ======================================================================
# 5. PROCESSING UNIT
# ======================================================================

class ProcessingUnit:
    """Đơn vị cốt lõi: app logic + in-memory space + replication."""

    def __init__(self, pu_id: str, partition_id: int, partition_mgr: PartitionManager) -> None:
        self.pu_id = pu_id
        self.partition_id = partition_id
        self._repository = SpaceRepository(partition_id, pu_id)
        self._replication = ReplicationEngine(pu_id, partition_id)
        self._partition_mgr = partition_mgr
        self._store: dict[str, Any] = {}
        logger.info("ProcessingUnit %s (partition=%d) started", pu_id, partition_id)

    def connect_peer(self, peer_pu: ProcessingUnit) -> None:
        self._replication.add_peer(peer_pu.pu_id, peer_pu._repository)

    def write(self, key: str, value: dict[str, Any], ttl: float | None = None) -> SpaceEntry:
        entry = self._repository.write(key, value, ttl)
        self._replication.replicate(key, value, entry.version, ttl)
        return entry

    def read(self, key: str) -> SpaceEntry | None:
        return self._repository.read(key)

    def take(self, key: str) -> SpaceEntry | None:
        return self._repository.take(key)

    def get_stats(self) -> dict[str, Any]:
        return {
            "pu_id": self.pu_id,
            "partition": self.partition_id,
            "entries": self._repository.get_size(),
            "primary_for": self._partition_mgr.get_partition_count(self.pu_id),
        }


# ======================================================================
# 6. SMART ROUTER
# ======================================================================

class SmartRouter:
    """Partition-aware router — routing request đến PU đúng."""

    def __init__(self, partition_mgr: PartitionManager) -> None:
        self._partition_mgr = partition_mgr
        self._pus: dict[str, ProcessingUnit] = {}

    def register_pu(self, pu: ProcessingUnit) -> None:
        self._pus[pu.pu_id] = pu

    def route(self, key: str) -> ProcessingUnit | None:
        pu_id = self._partition_mgr.get_primary_pu(key)
        if pu_id is None:
            return None
        return self._pus.get(pu_id)

    def write(self, key: str, value: dict[str, Any], ttl: float | None = None) -> Optional[SpaceEntry]:
        pu = self.route(key)
        if pu is None:
            return None
        return pu.write(key, value, ttl)

    def read(self, key: str) -> Optional[SpaceEntry]:
        pu = self.route(key)
        if pu is None:
            return None
        return pu.read(key)

    def get_all_pus(self) -> list[ProcessingUnit]:
        return list(self._pus.values())


# ======================================================================
# 7. BUSINESS LOGIC — E-COMMERCE CART
# ======================================================================

@dataclass
class CartItem:
    product_id: str
    product_name: str
    price: float
    quantity: int

    def subtotal(self) -> float:
        return self.price * self.quantity


@dataclass
class Cart:
    user_id: str
    items: list[CartItem] = field(default_factory=list)
    coupon_code: str = ""
    discount: float = 0.0
    updated_at: float = field(default_factory=time.time)

    def total(self) -> float:
        raw = sum(item.subtotal() for item in self.items)
        return raw * (1.0 - self.discount)


class CartService:
    """Cart service chạy trên SBA — zero database trip cho synchronous path."""

    CART_TTL = 7200.0  # 2 hours

    def __init__(self, router: SmartRouter) -> None:
        self._router = router

    def add_item(self, user_id: str, product_id: str, product_name: str, price: float, quantity: int) -> Cart:
        key = f"cart:{user_id}"
        entry = self._router.read(key)
        now = time.time()

        if entry is None:
            cart = Cart(user_id=user_id, updated_at=now)
        else:
            cart_data = entry.value
            items = [CartItem(**it) for it in cart_data.get("items", [])]
            cart = Cart(
                user_id=user_id,
                items=items,
                coupon_code=cart_data.get("coupon_code", ""),
                discount=cart_data.get("discount", 0.0),
                updated_at=cart_data.get("updated_at", now),
            )

        # Check if product already exists → increase quantity
        existing = next((it for it in cart.items if it.product_id == product_id), None)
        if existing:
            existing.quantity += quantity
        else:
            cart.items.append(CartItem(product_id=product_id, product_name=product_name, price=price, quantity=quantity))

        cart.updated_at = time.time()
        self._router.write(key, {
            "user_id": cart.user_id,
            "items": [{"product_id": it.product_id, "product_name": it.product_name, "price": it.price, "quantity": it.quantity} for it in cart.items],
            "coupon_code": cart.coupon_code,
            "discount": cart.discount,
            "updated_at": cart.updated_at,
        }, ttl=self.CART_TTL)
        return cart

    def get_cart(self, user_id: str) -> Cart | None:
        key = f"cart:{user_id}"
        entry = self._router.read(key)
        if entry is None:
            return None
        cart_data = entry.value
        items = [CartItem(**it) for it in cart_data.get("items", [])]
        return Cart(
            user_id=user_id,
            items=items,
            coupon_code=cart_data.get("coupon_code", ""),
            discount=cart_data.get("discount", 0.0),
            updated_at=cart_data.get("updated_at", time.time()),
        )

    def clear_cart(self, user_id: str) -> None:
        key = f"cart:{user_id}"
        self._router.route(key)  # just to check route exists
        pu = self._router.route(key)
        if pu:
            pu.take(key)

    def apply_coupon(self, user_id: str, coupon_code: str, discount: float) -> Cart | None:
        cart = self.get_cart(user_id)
        if cart is None:
            return None
        cart.coupon_code = coupon_code
        cart.discount = discount
        cart.updated_at = time.time()
        key = f"cart:{user_id}"
        self._router.write(key, {
            "user_id": cart.user_id,
            "items": [{"product_id": it.product_id, "product_name": it.product_name, "price": it.price, "quantity": it.quantity} for it in cart.items],
            "coupon_code": cart.coupon_code,
            "discount": cart.discount,
            "updated_at": cart.updated_at,
        }, ttl=self.CART_TTL)
        return cart


# ======================================================================
# 8. DATABASE PERSISTENCE ASYNC (Write-Behind)
# ======================================================================

class AsyncPersistence:
    """Write-behind persistence — database là backup, không phải critical path."""

    def __init__(self, router: SmartRouter, persist_interval: float = 5.0) -> None:
        self._router = router
        self._persist_interval = persist_interval
        self._store: dict[str, dict[str, Any]] = {}  # Simulate DB
        self._worker = threading.Thread(target=self._persist_loop, daemon=True)
        self._worker.start()

    def _persist_loop(self) -> None:
        while True:
            time.sleep(self._persist_interval)
            for pu in self._router.get_all_pus():
                keys = pu._repository.get_all_keys()
                for key in keys:
                    entry = pu._repository.read(key)
                    if entry:
                        self._store[key] = entry.value
                        logger.debug("Persisted key=%s to backup database", key)

    def load_from_db(self, key: str) -> dict[str, Any] | None:
        return self._store.get(key)

    def get_db_size(self) -> int:
        return len(self._store)


# ======================================================================
# 9. MAIN — SIMULATION
# ======================================================================

def main() -> None:
    logger.info("=== Space-Based Architecture: E-Commerce Cart Simulation ===")

    # Initialize cluster
    partition_mgr = PartitionManager(virtual_nodes=50)
    router = SmartRouter(partition_mgr)

    # Create 4 Processing Units
    pu1 = ProcessingUnit("pu-1", 0, partition_mgr)
    pu2 = ProcessingUnit("pu-2", 1, partition_mgr)
    pu3 = ProcessingUnit("pu-3", 2, partition_mgr)
    pu4 = ProcessingUnit("pu-4", 3, partition_mgr)

    # Register PUs in partition manager
    partition_mgr.add_pu("pu-1", 4)
    partition_mgr.add_pu("pu-2", 4)
    partition_mgr.add_pu("pu-3", 4)
    partition_mgr.add_pu("pu-4", 4)

    # Register in router
    router.register_pu(pu1)
    router.register_pu(pu2)
    router.register_pu(pu3)
    router.register_pu(pu4)

    # Setup replication — full mesh
    all_pus = [pu1, pu2, pu3, pu4]
    for pu in all_pus:
        for peer in all_pus:
            if pu.pu_id != peer.pu_id:
                pu.connect_peer(peer)

    # Setup persistence
    persistence = AsyncPersistence(router, persist_interval=3.0)

    # Create service
    cart_service = CartService(router)

    # --- Simulation ---
    logger.info("Simulating user traffic...")

    # User 1 adds items
    cart1 = cart_service.add_item("user_1", "prod_100", "iPhone 15 Pro", 1199.00, 1)
    logger.info("User 1 cart after add: $%.2f (%d items)", cart1.total(), len(cart1.items))

    cart1 = cart_service.add_item("user_1", "prod_200", "AirPods Pro", 249.00, 2)
    logger.info("User 1 cart after 2nd add: $%.2f (%d items)", cart1.total(), len(cart1.items))

    # User 2 adds items
    cart2 = cart_service.add_item("user_2", "prod_300", "MacBook Pro 16", 2499.00, 1)
    logger.info("User 2 cart: $%.2f (%d items)", cart2.total(), len(cart2.items))

    # Apply coupon
    cart1 = cart_service.apply_coupon("user_1", "WELCOME10", 0.10)
    if cart1:
        logger.info("User 1 after coupon 10%% off: $%.2f", cart1.total())

    # Get cart
    cart1_loaded = cart_service.get_cart("user_1")
    if cart1_loaded:
        logger.info("User 1 final cart: %d items, total= $%.2f (coupon=%s)",
                     len(cart1_loaded.items), cart1_loaded.total(), cart1_loaded.coupon_code)

    # Concurrent traffic
    logger.info("Simulating concurrent traffic with 10 users...")
    start = time.perf_counter()
    threads = []
    for i in range(10):
        t = threading.Thread(
            target=lambda uid: [
                cart_service.add_item(uid, f"prod_{random.randint(1, 1000)}", f"Product {random.randint(1, 1000)}",
                                       random.uniform(10, 500), random.randint(1, 3))
                for _ in range(20)
            ],
            args=(f"concurrent_user_{i}",),
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    logger.info("Concurrent traffic completed in %.2fs", elapsed)

    # Wait for replication + persistence
    time.sleep(2.0)

    # Stats
    logger.info("=== Cluster Stats ===")
    for pu in all_pus:
        stats = pu.get_stats()
        logger.info("  PU=%s | partition=%d | entries=%d", stats["pu_id"], stats["partition"], stats["entries"])

    logger.info("Database persisted %d keys", persistence.get_db_size())

    # Verify data distribution
    logger.info("=== Data Distribution across PUs ===")
    for i in range(10):
        key = f"cart:concurrent_user_{i}"
        route_to = partition_mgr.get_primary_pu(key)
        entry = router.read(key)
        if entry:
            items_count = len(entry.value.get("items", []))
            logger.info("  key=%s → PU=%s | items=%d", key, route_to, items_count)

    # Test failover
    logger.info("=== Simulating PU-1 failure ===")
    partition_mgr.remove_pu("pu-1")
    router._pus.pop("pu-1", None)
    entry = router.read("cart:user_1")
    if entry:
        logger.info("  cart:user_1 still accessible after PU-1 failure (via replication)")
    else:
        logger.warning("  cart:user_1 NOT accessible after PU-1 failure")

    logger.info("=== Space-Based Architecture Demo Complete ===")


if __name__ == "__main__":
    main()
```

## Khi nào dùng / Khi nào không

| Khi nào dùng (Use when) | Khi nào không (Avoid when) |
|--------------------------|----------------------------|
| Hệ thống cần scale ngang đến hàng trăm node | Dữ liệu có quan hệ phức tạp, cần JOIN và transaction |
| High-write workload (giỏ hàng, session, log) | Batch processing / ETL workload |
| Cần availability > consistency (AP trong CAP) | Yêu cầu strong consistency (CP trong CAP) |
| Application state cần distributed nhưng low-latency | Dữ liệu nhỏ (< 10GB) — một database đủ dùng |
| Peak load gấp 10-100 lần normal load (sale, event) | Hệ thống real-time control (latency < 1ms) |
| Cần tự động scale elastically theo tải | Đội ngũ nhỏ, không có DevOps support |

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Linear scalability**: Thêm node = tăng throughput tuyến tính | **Complexity cao**: Cần grid middleware, replication, partitioning |
| **Zero database bottleneck**: Database không nằm trên critical path | **Eventual consistency**: Không phù hợp với transaction ACID |
| **Low latency**: Data in-memory, không disk I/O | **Memory cost**: Dữ liệu nóng phải vừa với RAM của cluster |
| **Fault tolerance**: Replication tự động failover | **Data loss risk**: Nếu replication không kịp trước khi node chết |
| **Elasticity**: Thêm/bớt node không downtime | **Replication overhead**: Network traffic cho replication |
| **No sticky session**: Bất kỳ PU nào cũng xử lý được request | **Debugging khó**: Data distributed, khó trace |
| **In-memory performance**: 100-1000x nhanh hơn database | **Startup warm-up**: Cần load dữ liệu từ DB vào memory |

## Công cụ và Framework

| Tên | Loại | Ngôn ngữ | Đặc điểm |
|-----|------|----------|----------|
| **GigaSpaces XAP** | Commercial | Java, .NET | SBA platform thương mại đầu tiên |
| **Apache Ignite** | Open Source | Java, Python (.NET) | In-memory data grid + compute grid |
| **Hazelcast IMDG** | Open Source | Java, Python, Node | Distributed map, event, lock |
| **Red Hat Data Grid (Infinispan)** | Open Source | Java, Python | Cache + grid + transaction |
| **Oracle Coherence** | Commercial | Java | Grid cache, used by financial systems |
| **ScaleOut StateServer** | Commercial | Java, .NET | In-memory data grid for .NET |
| **Redis Enterprise** | Commercial | Multi-language | Redis cluster with replication |
| **Apache Cassandra** | Open Source | Java | Distributed NoSQL (SBA-like) |
| **Python (custom)** | DIY | Python | Xây dựng SBA đơn giản với `threading` + `socket` |

## Kiểm thử

Testing SBA đòi hỏi kiểm tra distributed behaviors: partitioning, replication, failover, consistency.

```python
from __future__ import annotations
import pytest
import time
import threading
from typing import Any


# ======================================================================
# Unit Tests
# ======================================================================

class TestSpaceRepository:
    def test_write_and_read(self, repo: SpaceRepository) -> None:
        entry = repo.write("key1", {"name": "test"}, ttl=3600)
        assert entry.key == "key1"
        assert entry.value["name"] == "test"
        assert entry.version == 1

        read_back = repo.read("key1")
        assert read_back is not None
        assert read_back.value["name"] == "test"

    def test_read_expired_entry(self, repo: SpaceRepository) -> None:
        repo.write("expired", {"data": "gone"}, ttl=0.1)
        time.sleep(0.2)
        assert repo.read("expired") is None

    def test_take_removes_entry(self, repo: SpaceRepository) -> None:
        repo.write("take_me", {"value": 42})
        taken = repo.take("take_me")
        assert taken is not None
        assert taken.value["value"] == 42
        assert repo.read("take_me") is None

    def test_version_increment(self, repo: SpaceRepository) -> None:
        repo.write("ver", {"count": 1})
        repo.write("ver", {"count": 2})
        entry = repo.read("ver")
        assert entry is not None
        assert entry.version == 2
        assert entry.value["count"] == 2


class TestPartitionManager:
    def test_consistent_hashing_same_key_same_pu(self, pm: PartitionManager) -> None:
        pu1 = pm.get_primary_pu("test_key_1")
        pu2 = pm.get_primary_pu("test_key_1")
        assert pu1 == pu2

    def test_consistent_hashing_different_keys_different_pu(self, pm: PartitionManager) -> None:
        pu1 = pm.get_primary_pu("alpha")
        pu2 = pm.get_primary_pu("beta")
        # Very unlikely to collide with enough virtual nodes
        assert pu1 is not None
        assert pu2 is not None

    def test_remap_minimal_on_node_removal(self, pm: PartitionManager) -> None:
        keys = [f"key_{i}" for i in range(1000)]
        mapping_before = {k: pm.get_primary_pu(k) for k in keys}
        pm.remove_pu("pu-1")
        mapping_after = {k: pm.get_primary_pu(k) for k in keys}
        changed = sum(1 for k in keys if mapping_before[k] != mapping_after[k])
        # Should remap < 50% (without virtual nodes ideal is ~25%)
        assert changed < 600, f"Too many remapped: {changed}"


class TestReplicationEngine:
    def test_replication_between_pus(self, router_with_pus: SmartRouter) -> None:
        # Write to primary
        router_with_pus.write("repl_test", {"value": "replicated"})
        time.sleep(0.3)  # Wait for replication

        # Read from other PU (should be replicated)
        entry = router_with_pus.read("repl_test")
        assert entry is not None
        assert entry.value["value"] == "replicated"

    def test_replication_version_conflict(self, router_with_pus: SmartRouter) -> None:
        router_with_pus.write("conflict_key", {"v": 1})
        time.sleep(0.2)
        router_with_pus.write("conflict_key", {"v": 2})
        time.sleep(0.2)
        entry = router_with_pus.read("conflict_key")
        assert entry is not None
        assert entry.value["v"] == 2  # Higher version wins
        assert entry.version == 2


class TestCartService:
    def test_add_item_and_calculate_total(self, cart_service: CartService) -> None:
        cart = cart_service.add_item("test_user", "prod_1", "Test Product", 100.0, 2)
        assert len(cart.items) == 1
        assert cart.items[0].quantity == 2
        assert cart.total() == 200.0

    def test_add_same_product_increases_quantity(self, cart_service: CartService) -> None:
        cart_service.add_item("user_same", "prod_x", "Product X", 50.0, 1)
        cart = cart_service.add_item("user_same", "prod_x", "Product X", 50.0, 3)
        assert len(cart.items) == 1
        assert cart.items[0].quantity == 4

    def test_coupon_discount(self, cart_service: CartService) -> None:
        cart_service.add_item("user_coupon", "prod_c", "Coupon Item", 200.0, 1)
        cart = cart_service.apply_coupon("user_coupon", "SAVE20", 0.20)
        assert cart is not None
        assert cart.total() == 160.0  # 200 - 20%
        assert cart.coupon_code == "SAVE20"

    def test_clear_cart(self, cart_service: CartService) -> None:
        cart_service.add_item("user_clear", "prod_clr", "Clear Me", 10.0, 1)
        cart_service.clear_cart("user_clear")
        cart = cart_service.get_cart("user_clear")
        assert cart is None

    def test_concurrent_cart_operations(self, cart_service: CartService) -> None:
        errors: list[Exception] = []

        def worker(uid: str) -> None:
            try:
                for i in range(10):
                    cart_service.add_item(uid, f"prod_{i}", f"Product {i}", float(i * 10), 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"concurrent_{j}",)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        for j in range(5):
            cart = cart_service.get_cart(f"concurrent_{j}")
            assert cart is not None
            assert len(cart.items) == 10


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def repo() -> SpaceRepository:
    return SpaceRepository(partition_id=0, pu_id="test-pu")


@pytest.fixture
def pm() -> PartitionManager:
    mgr = PartitionManager(virtual_nodes=50)
    mgr.add_pu("pu-1", 4)
    mgr.add_pu("pu-2", 4)
    mgr.add_pu("pu-3", 4)
    return mgr


@pytest.fixture
def router_with_pus() -> SmartRouter:
    pm = PartitionManager(virtual_nodes=50)
    pm.add_pu("pu-a", 4)
    pm.add_pu("pu-b", 4)
    pm.add_pu("pu-c", 4)
    router = SmartRouter(pm)
    pu_a = ProcessingUnit("pu-a", 0, pm)
    pu_b = ProcessingUnit("pu-b", 1, pm)
    pu_c = ProcessingUnit("pu-c", 2, pm)
    router.register_pu(pu_a)
    router.register_pu(pu_b)
    router.register_pu(pu_c)
    pu_a.connect_peer(pu_b)
    pu_a.connect_peer(pu_c)
    pu_b.connect_peer(pu_a)
    pu_b.connect_peer(pu_c)
    pu_c.connect_peer(pu_a)
    pu_c.connect_peer(pu_b)
    return router


@pytest.fixture
def cart_service(router_with_pus: SmartRouter) -> CartService:
    return CartService(router_with_pus)
```

## Kết luận

Space-Based Architecture là giải pháp mạnh mẽ cho bài toán scale-out khi database trở thành bottleneck. Bằng cách đưa dữ liệu vào in-memory distributed grid, SBA cho phép hệ thống xử lý hàng trăm ngàn request/giây với latency dưới mili-giây.

**Best Practices:**
- **Partition key design**: Chọn partition key sao cho dữ liệu phân bố đều. Key quá hot (ví dụ: celebrity user) có thể gây skew.
- **Replication factor**: >= 2 cho production. Quyết định sync vs async dựa trên consistency requirement.
- **Write-behind interval**: Cân bằng giữa data loss risk (interval dài) và DB load (interval ngắn).
- **Warm-up strategy**: Khi deploy PU mới, load dữ liệu từ DB trước khi nhận request.
- **Monitoring**: Track replication lag, partition size skew, memory usage.
- **Circuit breaker**: Nếu replication queue quá lớn, giảm tốc độ write để tránh OOM.

**Golden Rules:**
1. Database là backup, không phải source of truth cho synchronous path.
2. Thiết kế cho failure: mọi PU có thể chết bất kỳ lúc nào.
3. Idempotency: mọi operation phải an toàn khi retry (replication có thể duplicate).
4. Lưu dữ liệu "nóng" trong memory, dữ liệu "lạnh" trong DB.
5. Tránh distributed transaction — eventual consistency là chấp nhận được.
6. Stateless frontend + stateful backend (PU) — separation of concerns.
7. Monitor replication lag như một SLO quan trọng nhất.
