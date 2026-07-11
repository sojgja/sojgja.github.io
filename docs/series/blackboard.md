---
id: blackboard
title: Blackboard Architecture
sidebar_label: 🏗️ Blackboard Architecture
sidebar_position: 51
---

# Blackboard Architecture

> **Blackboard Architecture** — *"Multiple specialized subsystems (knowledge sources) cooperate to solve a complex problem by working together on a shared structured memory (blackboard), coordinated by a control component that decides which knowledge source to activate next."* — H. Penny Nii, 1986

## Tổng quan

Blackboard Architecture là một kiến trúc phần mềm lấy cảm hứng từ cách giải quyết vấn đề của con người: nhiều chuyên gia (knowledge sources) cùng nhìn vào một bảng đen (blackboard) và đóng góp giải pháp từng phần. Không có chuyên gia nào có đủ tri thức để giải toàn bộ bài toán — mỗi người chỉ giỏi một lĩnh vực hẹp.

Kiến trúc này được phát triển ban đầu trong lĩnh vực **trí tuệ nhân tạo (AI)** vào những năm 1970-1980. Hệ thống Blackboard đầu tiên là **HEARSAY-II** (Carnegie Mellon University, 1975) — một hệ thống nhận dạng giọng nói. Sau đó là **HASP/SIAP** (Stanford, 1978) — phân tích tín hiệu sonar. Các hệ thống này thành công vì chúng kết hợp được nhiều nguồn tri thức khác nhau (ngữ âm, từ vựng, ngữ pháp, ngữ nghĩa) để đạt độ chính xác cao hơn bất kỳ nguồn đơn lẻ nào.

**Những người tiên phong:**

| Tên | Đóng góp |
|-----|----------|
| **H. Penny Nii** | Định nghĩa chuẩn hóa Blackboard Architecture (1986) |
| **Lee Erman, Victor Lesser** | HEARSAY-II — ứng dụng đầu tiên |
| **Barbara Hayes-Roth** | BB1 — blackboard control architecture |
| **Robert Engelmore** | HASP/SIAP — signal understanding |
| **Alan Bond, Les Gasser** | Mở rộng blackboard cho multi-agent systems |

Ngày nay, Blackboard Architecture không chỉ dùng trong AI thuần túy. Nó được áp dụng rộng rãi trong **computer vision**, **robotics** (ROS — Robot Operating System), **phân tích tài chính** (trading signals từ nhiều indicator), **cybersecurity** (SIEM — kết hợp log từ nhiều nguồn), và **medical diagnosis**.

## Bài toán

### Vấn đề 1: Không có thuật toán duy nhất giải được bài toán

Xây dựng hệ thống phát hiện tấn công mạng (Intrusion Detection System — IDS) cho một công ty lớn. Hệ thống phải phân tích hàng triệu log events mỗi giây để phát hiện tấn công. Có nhiều loại tấn công khác nhau — SQL injection, DDoS, brute force, zero-day — và mỗi loại cần phương pháp phát hiện riêng. Không một thuật toán hay mô hình duy nhất nào có thể phát hiện tất cả các loại tấn công với độ chính xác cao.

Một vài rule engine chỉ bắt được các pattern đã biết (signature-based). Machine learning model có thể phát hiện anomaly nhưng tỷ lệ false positive cao. Nếu viết tất cả vào một monolithic codebase, việc thêm phương pháp mới là cực kỳ khó — mỗi lần thêm detection technique phải sửa core logic.

### Vấn đề 2: Các chuyên gia (module) cần hợp tác, không cạnh tranh

Hệ thống phân tích tấn công mạng có ba module: (1) **SignatureDetector** — so khớp với database các pattern tấn công đã biết, (2) **AnomalyDetector** — phát hiện bất thường dựa trên baseline, (3) **CorrelationEngine** — kết hợp các event lẻ tẻ thành attack chain. Các module này không thể chạy độc lập — SignatureDetector phát hiện SQL injection attempt, AnomalyDetector xác nhận traffic bất thường, CorrelationEngine kết luận đây là một coordinated attack. Kết quả của module này là input cho module kia.

Với kiến trúc pipeline (module A → B → C), thứ tự xử lý cố định, không linh hoạt. Với message queue (pub/sub), mỗi module chỉ thấy event của mình, không có cái nhìn tổng thể.

### Vấn đề 3: Dữ liệu chưa đầy đủ, cần incremental reasoning

Trong thực tế, dữ liệu log đến không đồng thời. Một event cảnh báo "login failed" có thể vô hại nếu chỉ có một lần. Nhưng nếu có 1000 lần từ cùng IP trong 1 giây → brute force attack. Hệ thống cần khả năng ghi nhận incremental evidence — từng phần thông tin nhỏ được thêm vào, các module đánh giá và đưa ra hypothesis tạm thời. Khi có thêm evidence, hypothesis được củng cố hoặc loại bỏ.

### Vấn đề 4: Tích hợp legacy systems và third-party

Hệ thống cần tích hợp với các công cụ bảo mật có sẵn: Snort (signature-based IDS), Zeek (network analysis), VirusTotal, SIEM platform. Mỗi công cụ có API, format dữ liệu riêng. Blackboard architecture giải quyết bằng cách wrapper mỗi công cụ thành một Knowledge Source (KS), KS chỉ giao tiếp với blackboard, không giao tiếp với nhau.

## Nguyên lý thiết kế

Blackboard Architecture có ba thành phần chính, hoạt động theo cơ chế giống như các nhà khoa học cùng giải một bài toán trên bảng đen:

### 1. Blackboard (Bảng đen)

Là cấu trúc dữ liệu chia sẻ, được tổ chức phân cấp thành các **level** (mức abstraction). Ở mức thấp là dữ liệu thô (raw log events). Ở mức trung là các pattern, hypothesis. Ở mức cao là kết luận, quyết định.

```
Level 3: Conclusions / Decisions
Level 2: Hypothesis / Patterns
Level 1: Aggregated Data
Level 0: Raw Data / Facts
```

Blackboard cho phép:
- **write** — KS ghi kết quả
- **read** — KS đọc dữ liệu cần
- **subscribe** — KS đăng ký nhận thông báo khi có thay đổi

### 2. Knowledge Sources (KS) — Các chuyên gia

Mỗi KS là một module chuyên biệt, có khả năng giải quyết một phần bài toán. KS có dạng:
- **Trigger condition** — điều kiện để KS "quan tâm" đến dữ liệu trên blackboard
- **Action** — khi trigger, KS đọc input từ blackboard, xử lý, ghi kết quả trở lại

KS không giao tiếp trực tiếp với nhau — mọi tương tác đều qua blackboard. Điều này đảm bảo loose coupling.

Đặc điểm của KS:
- **Self-contained**: KS chứa đầy đủ tri thức cho một lĩnh vực
- **Opportunistic**: KS hoạt động khi có cơ hội đóng góp
- **Independent**: KS có thể thêm, xóa, sửa mà không ảnh hưởng KS khác

### 3. Control Component (Điều phối viên)

Control component quyết định **KS nào được kích hoạt tiếp theo** dựa trên:
- **Focus of attention** — vùng nào của blackboard cần được xử lý
- **Priority** — KS nào có khả năng đóng góp nhiều nhất
- **Conflict resolution** — nhiều KS cùng muốn kích hoạt, chọn ai?

Chiến lược control phổ biến:
- **Goal-driven** (top-down): Bắt đầu từ mục tiêu, tìm KS có thể thỏa mãn
- **Data-driven** (bottom-up): Bắt đầu từ dữ liệu thô, kích hoạt KS phù hợp
- **Mixed-initiative**: Kết hợp cả hai

### 4. Cơ chế hoạt động

```
Loop:
  1. Update Blackboard (KS ghi kết quả mới)
  2. Trigger Evaluation (kiểm tra KS nào đủ điều kiện)
  3. Scheduling (chọn KS tiếp theo)
  4. Execution (chạy KS được chọn)
```

Đây là vòng lặp **opportunistic reasoning** — KS được kích hoạt khi có cơ hội, không theo thứ tự định trước.

## Cấu trúc chi tiết

### Core Components

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Blackboard** | Shared structured memory | Hierarchical dict with event system |
| **KnowledgeSource (KS)** | Specialized reasoning module | Abstract class with trigger() + execute() |
| **BlackboardController** | Scheduling and conflict resolution | Priority queue + scheduling strategy |
| **BlackboardEvent** | Notification mechanism | Observer pattern |
| **Solution** | Kết quả cuối cùng | Accumulated on blackboard |
| **Hypothesis** | Giả thuyết tạm thời | Evidence-based, confidence score |

### Blackboard Levels (Cyber Security Example)

| Level | Content | Example |
|-------|---------|---------|
| **Level 0 — Raw Events** | Log entries, network packets | `src=192.168.1.5 dst=10.0.0.1 port=443` |
| **Level 1 — Aggregated** | Thống kê, baseline | `IP 192.168.1.5: 150 failed logins in 5s` |
| **Level 2 — Hypothesis** | Nghi vấn tấn công | `Brute force attack: 85% confidence` |
| **Level 3 — Conclusion** | Kết luận + action | `Block IP 192.168.1.5 for 3600s` |

### Data Flow

```
Raw Log Event
    │
    ▼
Blackboard Level 0 (Raw Events)
    │
    ▼
[Trigger] SignatureDetector KS ────► ghi "SQL Injection Attempt" ──► Level 1
                                      │
                                      ▼
[Trigger] AnomalyDetector KS ────────► ghi "Traffic spike 300%" ───► Level 1
                                      │
                                      ▼
[Trigger] CorrelationEngine KS ──────► ghi "Attack Chain: SQLi → Data Exfil" ──► Level 2
                                      │
                                      ▼
[Trigger] ResponsePlanner KS ────────► ghi "Action: Block IP, Alert SOC" ──► Level 3
```

## Sơ đồ kiến trúc (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        BLACKBOARD ARCHITECTURE                            │
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                        BLACKBOARD                               │    │
│   │  ┌────────────────────────────────────────────────────────────┐ │    │
│   │  │  Level 3: Conclusions & Actions                            │ │    │
│   │  │  [Block IP] [Alert SOC] [Scale Resources]                 │ │    │
│   │  └────────────────────────────────────────────────────────────┘ │    │
│   │  ┌────────────────────────────────────────────────────────────┐ │    │
│   │  │  Level 2: Hypothesis (Confidence)                         │ │    │
│   │  │  [Brute Force: 0.85] [SQL Injection: 0.92] [DDoS: 0.70]  │ │    │
│   │  └────────────────────────────────────────────────────────────┘ │    │
│   │  ┌────────────────────────────────────────────────────────────┐ │    │
│   │  │  Level 1: Aggregated Data                                  │ │    │
│   │  │  [Failed Logins: 150] [Bytes Out: 500MB] [Anomaly: 3σ]    │ │    │
│   │  └────────────────────────────────────────────────────────────┘ │    │
│   │  ┌────────────────────────────────────────────────────────────┐ │    │
│   │  │  Level 0: Raw Events                                       │ │    │
│   │  │  [Event: login_fail] [Event: http_500] [Packet: tcp/443]  │ │    │
│   │  └────────────────────────────────────────────────────────────┘ │    │
│   └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│   │ Signature KS   │  │ Anomaly KS     │  │ Correlation KS │            │
│   │ ────────────   │  │ ────────────   │  │ ────────────   │            │
│   │ Trigger: watch │  │ Trigger: watch │  │ Trigger: watch │            │
│   │ Level 0 events │  │ Level 1 stats  │  │ Level 1-2 hypo │            │
│   │ Knows: SQLi,   │  │ Knows: ML,     │  │ Knows: attack  │            │
│   │ XSS, RCE       │  │ baseline, std  │  │ chain, kill    │            │
│   │                │  │ deviation      │  │ chain          │            │
│   └────────────────┘  └────────────────┘  └────────────────┘            │
│                                                                           │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│   │ Response KS    │  │ Enrichment KS  │  │ ThreatIntel KS │            │
│   │ ────────────   │  │ ────────────   │  │ ────────────   │            │
│   │ Trigger: watch │  │ Trigger: watch │  │ Trigger: watch │            │
│   │ Level 2 hypo   │  │ Level 0 IPs    │  │ Level 1-2 IOCs │            │
│   │ Knows: mitiga- │  │ Knows: geo IP, │  │ Knows: virus   │            │
│   │ tion, block    │  │ Whois, DNS     │  │ total, MISP    │            │
│   └────────────────┘  └────────────────┘  └────────────────┘            │
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                    CONTROL COMPONENT                            │    │
│   │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │    │
│   │  │ Trigger      │  │ Scheduler    │  │ Focus Manager      │   │    │
│   │  │ Evaluator    │  │ (Priority Q) │  │ (Which level to    │   │    │
│   │  │              │  │              │  │  focus on)         │   │    │
│   │  └──────────────┘  └──────────────┘  └────────────────────┘   │    │
│   └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Ví dụ code hoàn chỉnh

### Cách làm sai: Pipeline cứng nhắc

```python
from __future__ import annotations
import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RigidPipelineIDS:
    """Pipeline cố định: module A → B → C. Thêm module mới phải sửa pipeline."""

    def __init__(self) -> None:
        self._results: dict[str, Any] = {}

    def run(self, raw_events: list[dict[str, Any]]) -> None:
        step1 = self._signature_detect(raw_events)
        step2 = self._anomaly_detect(step1)
        step3 = self._correlate(step2)
        final = self._respond(step3)
        logger.info("Final decision: %s", final)

    def _signature_detect(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [e for e in events if "SQL" in str(e)]  # Simplified

    def _anomaly_detect(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [e for e in events if e.get("count", 0) > 100]

    def _correlate(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return events  # Dummy

    def _respond(self, events: list[dict[str, Any]]) -> str:
        return "Block IP" if events else "OK"
```

### Cách làm đúng: Blackboard Architecture

```python
from __future__ import annotations
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections import defaultdict
from heapq import heappush, heappop
import json
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# 1. ENUMS & DOMAIN TYPES
# ======================================================================

class BlackboardLevel(Enum):
    RAW_EVENTS = 0
    AGGREGATED = 1
    HYPOTHESIS = 2
    CONCLUSION = 3


class KnowledgeSourcePriority(Enum):
    LOW = 10
    NORMAL = 50
    HIGH = 90
    CRITICAL = 100


@dataclass
class BlackboardEntry:
    """Một entry trên blackboard — đơn vị dữ liệu cơ bản."""
    id: str
    level: BlackboardLevel
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source_ks: str = ""
    confidence: float = 0.0

    def matches(self, pattern: dict[str, Any]) -> bool:
        """Kiểm tra entry có khớp với pattern không."""
        for key, value in pattern.items():
            if key not in self.data:
                return False
            if isinstance(value, (int, float, str)) and self.data[key] != value:
                return False
        return True


@dataclass
class Hypothesis:
    """Giả thuyết được xây dựng từ evidence."""
    id: str
    description: str
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, confirmed, rejected

    def add_evidence(self, entry_id: str, weight: float = 0.1) -> None:
        self.evidence.append(entry_id)
        self.confidence = min(1.0, self.confidence + weight * (1.0 - self.confidence))

    def reject(self) -> None:
        self.status = "rejected"
        self.confidence = 0.0


# ======================================================================
# 2. KNOWLEDGE SOURCE (KS) — ABSTRACT
# ======================================================================

class KnowledgeSource(ABC):
    """Abstract base cho mọi knowledge source."""

    def __init__(self, ks_id: str, priority: KnowledgeSourcePriority = KnowledgeSourcePriority.NORMAL) -> None:
        self.ks_id = ks_id
        self.priority = priority
        logger.info("KnowledgeSource %s initialized (priority=%s)", ks_id, priority.name)

    @abstractmethod
    def trigger_condition(self, blackboard: Blackboard) -> bool:
        """Kiểm tra xem KS này có muốn kích hoạt không dựa trên trạng thái blackboard."""
        ...

    @abstractmethod
    def execute(self, blackboard: Blackboard) -> list[BlackboardEntry]:
        """Thực thi KS: đọc từ blackboard, xử lý, ghi kết quả trở lại."""
        ...


# ======================================================================
# 3. BLACKBOARD — CORE
# ======================================================================

class Blackboard:
    """Shared structured memory — trung tâm của architecture."""

    def __init__(self) -> None:
        self._entries: dict[str, BlackboardEntry] = {}
        self._hypotheses: dict[str, Hypothesis] = {}
        self._lock = threading.RLock()
        self._listeners: dict[BlackboardLevel, list[Callable[[BlackboardEntry], None]]] = {
            level: [] for level in BlackboardLevel
        }
        self._change_counter = 0
        logger.info("Blackboard initialized")

    def write(self, entry: BlackboardEntry) -> None:
        with self._lock:
            self._entries[entry.id] = entry
            self._change_counter += 1
            logger.debug("Write: level=%s id=%s source=%s", entry.level.name, entry.id, entry.source_ks)
            # Notify listeners
            for listener in self._listeners.get(entry.level, []):
                try:
                    listener(entry)
                except Exception as e:
                    logger.error("Listener error: %s", e)

    def read(self, entry_id: str) -> BlackboardEntry | None:
        with self._lock:
            return self._entries.get(entry_id)

    def query(self, level: BlackboardLevel | None = None, pattern: dict[str, Any] | None = None) -> list[BlackboardEntry]:
        """Query entries với filter theo level và pattern."""
        with self._lock:
            results = list(self._entries.values())
            if level is not None:
                results = [e for e in results if e.level == level]
            if pattern:
                results = [e for e in results if e.matches(pattern)]
            return sorted(results, key=lambda e: e.timestamp, reverse=True)

    def remove(self, entry_id: str) -> None:
        with self._lock:
            self._entries.pop(entry_id, None)

    def clear_level(self, level: BlackboardLevel) -> None:
        with self._lock:
            self._entries = {k: v for k, v in self._entries.items() if v.level != level}

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        with self._lock:
            self._hypotheses[hypothesis.id] = hypothesis
            logger.info("New hypothesis: %s (confidence=%.2f)", hypothesis.description, hypothesis.confidence)

    def get_hypothesis(self, hyp_id: str) -> Hypothesis | None:
        with self._lock:
            return self._hypotheses.get(hyp_id)

    def get_all_hypotheses(self) -> list[Hypothesis]:
        with self._lock:
            return list(self._hypotheses.values())

    def subscribe(self, level: BlackboardLevel, callback: Callable[[BlackboardEntry], None]) -> None:
        with self._lock:
            self._listeners[level].append(callback)

    def get_change_count(self) -> int:
        with self._lock:
            return self._change_counter

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hypotheses": len(self._hypotheses),
                "per_level": {level.name: sum(1 for e in self._entries.values() if e.level == level)
                              for level in BlackboardLevel},
                "changes": self._change_counter,
            }


# ======================================================================
# 4. CONTROL COMPONENT
# ======================================================================

@dataclass(order=True)
class ScheduledKS:
    priority: int
    ks_id: str
    ks: KnowledgeSource = field(compare=False)


class BlackboardController:
    """Điều phối viên — quyết định KS nào chạy tiếp theo."""

    def __init__(self, blackboard: Blackboard, max_iterations: int = 100) -> None:
        self._blackboard = blackboard
        self._max_iterations = max_iterations
        self._knowledge_sources: dict[str, KnowledgeSource] = {}
        self._is_running = False

    def register_ks(self, ks: KnowledgeSource) -> None:
        self._knowledge_sources[ks.ks_id] = ks
        logger.info("Registered KS: %s (priority=%s)", ks.ks_id, ks.priority.name)

    def unregister_ks(self, ks_id: str) -> None:
        self._knowledge_sources.pop(ks_id, None)
        logger.info("Unregistered KS: %s", ks_id)

    def run(self) -> None:
        """Vòng lặp opportunistic reasoning chính."""
        self._is_running = True
        iteration = 0
        last_change_count = 0

        logger.info("BlackboardController started (max_iterations=%d)", self._max_iterations)

        while self._is_running and iteration < self._max_iterations:
            iteration += 1
            triggered: list[tuple[int, str, KnowledgeSource]] = []

            # Phase 1: Evaluate triggers
            for ks_id, ks in self._knowledge_sources.items():
                try:
                    if ks.trigger_condition(self._blackboard):
                        triggered.append((ks.priority.value, ks_id, ks))
                except Exception as e:
                    logger.error("Trigger evaluation failed for %s: %s", ks_id, e)

            if not triggered:
                # Check if any new info was added
                if self._blackboard.get_change_count() == last_change_count:
                    logger.info("No triggered KS and no changes — stopping")
                    break
                last_change_count = self._blackboard.get_change_count()
                continue

            # Phase 2: Schedule — pick highest priority
            triggered.sort(key=lambda x: x[0], reverse=True)
            _, selected_id, selected_ks = triggered[0]
            logger.info("Iteration %d: Selected KS=%s (priority=%d)", iteration, selected_id, triggered[0][0])

            # Phase 3: Execute
            try:
                new_entries = selected_ks.execute(self._blackboard)
                for entry in new_entries:
                    self._blackboard.write(entry)
                logger.info("KS=%s produced %d entries", selected_id, len(new_entries))
            except Exception as e:
                logger.error("KS execution failed for %s: %s", selected_id, e)

            time.sleep(0.01)  # Prevent busy-loop

        logger.info("BlackboardController finished after %d iterations", iteration)

    def stop(self) -> None:
        self._is_running = False


# ======================================================================
# 5. CONCRETE KNOWLEDGE SOURCES — CYBER SECURITY IDS
# ======================================================================

class SignatureDetectorKS(KnowledgeSource):
    """Phát hiện tấn công dựa trên signature database."""

    def __init__(self) -> None:
        super().__init__("signature_detector", KnowledgeSourcePriority.HIGH)
        self._signatures: dict[str, str] = {
            "SQL_INJECTION": r"('|--|union|select.*from|or\s+1=1)",
            "XSS": r"(<script|alert\(|onerror=|onload=)",
            "RCE": r"(exec\(|system\(|passthru\(|`.*`)",
            "PATH_TRAVERSAL": r"(\.\.\/|\/etc\/passwd)",
        }
        self._processed: set[str] = set()

    def trigger_condition(self, blackboard: Blackboard) -> bool:
        entries = blackboard.query(level=BlackboardLevel.RAW_EVENTS)
        # Trigger if there's an unprocessed raw event
        return any(e.id not in self._processed for e in entries)

    def execute(self, blackboard: Blackboard) -> list[BlackboardEntry]:
        new_entries: list[BlackboardEntry] = []
        raw_events = blackboard.query(level=BlackboardLevel.RAW_EVENTS)

        for event in raw_events:
            if event.id in self._processed:
                continue
            self._processed.add(event.id)
            payload = str(event.data.get("payload", ""))
            src_ip = event.data.get("src_ip", "unknown")

            for attack_type, pattern in self._signatures.items():
                import re
                if re.search(pattern, payload, re.IGNORECASE):
                    entry = BlackboardEntry(
                        id=f"sig_{event.id}_{attack_type}",
                        level=BlackboardLevel.AGGREGATED,
                        data={
                            "event_type": "signature_match",
                            "attack_type": attack_type,
                            "src_ip": src_ip,
                            "original_event_id": event.id,
                            "severity": "high",
                            "description": f"Detected {attack_type} from {src_ip}",
                        },
                        source_ks=self.ks_id,
                        confidence=0.95,
                    )
                    new_entries.append(entry)
                    logger.info("Signature match: %s from %s (event=%s)", attack_type, src_ip, event.id)

        return new_entries


class AnomalyDetectorKS(KnowledgeSource):
    """Phát hiện bất thường dựa trên baseline statistical analysis."""

    def __init__(self, threshold_std: float = 2.0) -> None:
        super().__init__("anomaly_detector", KnowledgeSourcePriority.NORMAL)
        self._threshold_std = threshold_std
        self._baselines: dict[str, dict[str, float]] = {}
        self._processed_events: set[str] = set()

    def trigger_condition(self, blackboard: Blackboard) -> bool:
        entries = blackboard.query(level=BlackboardLevel.RAW_EVENTS)
        return any(e.id not in self._processed_events for e in entries)

    def execute(self, blackboard: Blackboard) -> list[BlackboardEntry]:
        new_entries: list[BlackboardEntry] = []
        raw_events = blackboard.query(level=BlackboardLevel.RAW_EVENTS)

        # Aggregate per src_ip
        ip_counts: dict[str, int] = defaultdict(int)
        ip_ports: dict[str, set[int]] = defaultdict(set)
        ip_bytes: dict[str, int] = defaultdict(int)

        for event in raw_events:
            if event.id not in self._processed_events:
                self._processed_events.add(event.id)
                src_ip = event.data.get("src_ip", "unknown")
                ip_counts[src_ip] += 1
                ip_ports[src_ip].add(event.data.get("dst_port", 0))
                ip_bytes[src_ip] += event.data.get("bytes", 0)

        for ip, count in ip_counts.items():
            if ip not in self._baselines:
                self._baselines[ip] = {"mean": count, "std": 0, "sample_count": 1}
                continue

            baseline = self._baselines[ip]
            # Update baseline (exponential moving average)
            n = baseline["sample_count"]
            old_mean = baseline["mean"]
            baseline["mean"] = old_mean + (count - old_mean) / (n + 1)
            baseline["std"] = baseline.get("std", 0) + abs(count - baseline["mean"]) / (n + 1)
            baseline["sample_count"] = n + 1

            if baseline["std"] > 0 and abs(count - baseline["mean"]) > self._threshold_std * baseline["std"]:
                entry = BlackboardEntry(
                    id=f"anomaly_{ip}_{time.time_ns()}",
                    level=BlackboardLevel.AGGREGATED,
                    data={
                        "event_type": "anomaly",
                        "src_ip": ip,
                        "observed_count": count,
                        "expected_mean": round(baseline["mean"], 2),
                        "std_dev": round(baseline["std"], 2),
                        "severity": "medium",
                        "description": f"Anomalous traffic from {ip}: {count} events (mean={baseline['mean']:.1f})",
                    },
                    source_ks=self.ks_id,
                    confidence=min(0.9, abs(count - baseline["mean"]) / (baseline["std"] + 1) * 0.2),
                )
                new_entries.append(entry)
                logger.info("Anomaly detected: %s count=%d mean=%.1f std=%.1f", ip, count, baseline["mean"], baseline["std"])

        return new_entries


class CorrelationEngineKS(KnowledgeSource):
    """Kết hợp các event lẻ thành attack chain hypothesis."""

    def __init__(self) -> None:
        super().__init__("correlation_engine", KnowledgeSourcePriority.CRITICAL)

    def trigger_condition(self, blackboard: Blackboard) -> bool:
        # Check for uncorrelated aggregate events
        agg = blackboard.query(level=BlackboardLevel.AGGREGATED)
        existing_hyp = blackboard.get_all_hypotheses()
        processed_ids = set()
        for h in existing_hyp:
            processed_ids.update(h.evidence)
        return any(e.id not in processed_ids for e in agg)

    def execute(self, blackboard: Blackboard) -> list[BlackboardEntry]:
        new_entries: list[BlackboardEntry] = []
        agg_events = blackboard.query(level=BlackboardLevel.AGGREGATED)
        existing_hyp = blackboard.get_all_hypotheses()

        processed_ids = set()
        for h in existing_hyp:
            processed_ids.update(h.evidence)

        # Group events by src_ip
        by_ip: dict[str, list[BlackboardEntry]] = defaultdict(list)
        for event in agg_events:
            ip = event.data.get("src_ip", "")
            if ip and event.id not in processed_ids:
                by_ip[ip].append(event)

        for ip, events in by_ip.items():
            if len(events) < 2:
                continue

            attack_types = [e.data.get("attack_type", e.data.get("event_type", "")) for e in events]
            confidence = sum(e.confidence for e in events) / len(events) * min(1.0, len(events) / 3.0)

            # Determine if this is a coordinated attack
            has_signature = any("signature_match" in str(e.data) for e in events)
            has_anomaly = any("anomaly" in str(e.data) for e in events)

            hyp_id = f"attack_chain_{ip}_{time.time_ns()}"
            if has_signature and has_anomaly:
                hypothesis = Hypothesis(
                    id=hyp_id,
                    description=f"Coordinated multi-vector attack from {ip}",
                    confidence=min(confidence + 0.2, 1.0),
                    status="confirmed" if confidence > 0.7 else "pending",
                )
                for e in events:
                    hypothesis.add_evidence(e.id, 0.15)
                blackboard.add_hypothesis(hypothesis)

                entry = BlackboardEntry(
                    id=f"corr_{hyp_id}",
                    level=BlackboardLevel.HYPOTHESIS,
                    data={
                        "event_type": "coordinated_attack",
                        "src_ip": ip,
                        "confidence": hypothesis.confidence,
                        "attack_types": attack_types,
                        "hypothesis_id": hyp_id,
                        "description": hypothesis.description,
                    },
                    source_ks=self.ks_id,
                    confidence=hypothesis.confidence,
                )
                new_entries.append(entry)
                logger.warning("COORDINATED ATTACK from %s types=%s confidence=%.2f",
                               ip, attack_types, hypothesis.confidence)

            elif has_signature:
                hypothesis = Hypothesis(
                    id=hyp_id,
                    description=f"Signature-based attack from {ip} (multiple signatures)",
                    confidence=min(confidence, 0.8),
                    status="pending",
                )
                for e in events:
                    hypothesis.add_evidence(e.id, 0.1)
                blackboard.add_hypothesis(hypothesis)

                entry = BlackboardEntry(
                    id=f"corr_{hyp_id}",
                    level=BlackboardLevel.HYPOTHESIS,
                    data={
                        "event_type": "signature_attack",
                        "src_ip": ip,
                        "confidence": hypothesis.confidence,
                        "attack_types": attack_types,
                        "hypothesis_id": hyp_id,
                        "description": hypothesis.description,
                    },
                    source_ks=self.ks_id,
                    confidence=hypothesis.confidence,
                )
                new_entries.append(entry)
                logger.info("Signature attack detected from %s: %s", ip, attack_types)

        return new_entries


class ThreatIntelEnrichmentKS(KnowledgeSource):
    """Enrich IP với threat intelligence data (geo, reputation)."""

    def __init__(self) -> None:
        super().__init__("threat_intel_enrich", KnowledgeSourcePriority.LOW)
        self._enriched_ips: set[str] = set()

    def trigger_condition(self, blackboard: Blackboard) -> bool:
        agg = blackboard.query(level=BlackboardLevel.AGGREGATED)
        return any(e.data.get("src_ip", "") not in self._enriched_ips for e in agg)

    def execute(self, blackboard: Blackboard) -> list[BlackboardEntry]:
        new_entries: list[BlackboardEntry] = []
        agg_events = blackboard.query(level=BlackboardLevel.AGGREGATED)

        for event in agg_events:
            ip = event.data.get("src_ip", "")
            if ip and ip not in self._enriched_ips:
                self._enriched_ips.add(ip)
                # Simulate geo IP lookup
                countries = ["VN", "US", "RU", "CN", "KR"]
                entry = BlackboardEntry(
                    id=f"intel_{ip}_{time.time_ns()}",
                    level=BlackboardLevel.AGGREGATED,
                    data={
                        "event_type": "ip_enrichment",
                        "src_ip": ip,
                        "country": random.choice(countries),
                        "is_proxy": random.random() > 0.7,
                        "reputation_score": random.randint(1, 100),
                        "description": f"Enriched {ip}: country={random.choice(countries)}",
                    },
                    source_ks=self.ks_id,
                    confidence=0.8,
                )
                new_entries.append(entry)

        return new_entries


class ResponsePlannerKS(KnowledgeSource):
    """Lên kế hoạch phản ứng dựa trên hypothesis đã xác nhận."""

    def __init__(self) -> None:
        super().__init__("response_planner", KnowledgeSourcePriority.CRITICAL)

    def trigger_condition(self, blackboard: Blackboard) -> bool:
        hyp = blackboard.get_all_hypotheses()
        return any(h.status == "confirmed" for h in hyp)

    def execute(self, blackboard: Blackboard) -> list[BlackboardEntry]:
        new_entries: list[BlackboardEntry] = []
        confirmed_hyp = [h for h in blackboard.get_all_hypotheses() if h.status == "confirmed"]

        for hypothesis in confirmed_hyp:
            # Skip if response already planned
            existing = blackboard.query(
                level=BlackboardLevel.CONCLUSION,
                pattern={"hypothesis_id": hypothesis.id},
            )
            if existing:
                continue

            # Determine response
            if "multi-vector" in hypothesis.description:
                response = "BLOCK_IP_AND_ALERT_SOC"
                block_duration = 86400  # 24h
            elif "signature" in hypothesis.description:
                response = "RATE_LIMIT_AND_MONITOR"
                block_duration = 3600  # 1h
            else:
                response = "LOG_ONLY"
                block_duration = 0

            entry = BlackboardEntry(
                id=f"response_{hypothesis.id}",
                level=BlackboardLevel.CONCLUSION,
                data={
                    "event_type": "response_plan",
                    "hypothesis_id": hypothesis.id,
                    "response": response,
                    "block_duration_seconds": block_duration,
                    "target_ip": hypothesis.description.split()[-1],
                    "severity": "critical" if response == "BLOCK_IP_AND_ALERT_SOC" else "high",
                    "description": f"Response for {hypothesis.description}: {response}",
                },
                source_ks=self.ks_id,
                confidence=0.99,
            )
            new_entries.append(entry)
            logger.critical("Response planned: %s → %s", hypothesis.id, response)

        return new_entries


# ======================================================================
# 6. MAIN — SIMULATION
# ======================================================================

class LogSimulator:
    """Mô phỏng log events từ mạng."""

    def __init__(self, blackboard: Blackboard) -> None:
        self._blackboard = blackboard
        self._event_counter = 0
        self._attackers = ["192.168.1.100", "10.0.0.50", "172.16.0.200"]
        self._normal_ips = ["192.168.1.{}".format(i) for i in range(10, 50)]

    def generate_normal_traffic(self, count: int = 20) -> None:
        for _ in range(count):
            self._event_counter += 1
            ip = random.choice(self._normal_ips)
            entry = BlackboardEntry(
                id=f"raw_{self._event_counter}",
                level=BlackboardLevel.RAW_EVENTS,
                data={
                    "src_ip": ip,
                    "dst_ip": "10.0.0.1",
                    "dst_port": random.choice([80, 443, 22]),
                    "protocol": random.choice(["TCP", "UDP"]),
                    "bytes": random.randint(64, 1500),
                    "payload": f"GET /index.html HTTP/1.1 Host: example.com",
                    "timestamp": time.time(),
                },
                source_ks="simulator",
            )
            self._blackboard.write(entry)

    def generate_sql_injection(self, count: int = 3) -> None:
        for i in range(count):
            self._event_counter += 1
            ip = random.choice(self._attackers)
            payloads = [
                "GET /search?q=1' OR '1'='1 HTTP/1.1",
                "POST /login username=admin'--&password=x",
                "GET /products?id=1 UNION SELECT * FROM users",
            ]
            entry = BlackboardEntry(
                id=f"raw_{self._event_counter}",
                level=BlackboardLevel.RAW_EVENTS,
                data={
                    "src_ip": ip,
                    "dst_ip": "10.0.0.1",
                    "dst_port": 80,
                    "protocol": "TCP",
                    "bytes": random.randint(200, 800),
                    "payload": random.choice(payloads),
                    "timestamp": time.time(),
                },
                source_ks="simulator",
            )
            self._blackboard.write(entry)

    def generate_xss_attack(self, count: int = 2) -> None:
        for i in range(count):
            self._event_counter += 1
            ip = random.choice(self._attackers)
            payloads = [
                "POST /comment content=<script>alert('xss')</script>",
                "GET /profile?name=<img src=x onerror=alert(1)>",
            ]
            entry = BlackboardEntry(
                id=f"raw_{self._event_counter}",
                level=BlackboardLevel.RAW_EVENTS,
                data={
                    "src_ip": ip,
                    "dst_ip": "10.0.0.1",
                    "dst_port": 80,
                    "protocol": "TCP",
                    "bytes": random.randint(200, 800),
                    "payload": random.choice(payloads),
                    "timestamp": time.time(),
                },
                source_ks="simulator",
            )
            self._blackboard.write(entry)

    def generate_brute_force(self, count: int = 50) -> None:
        ip = random.choice(self._attackers)
        for i in range(count):
            self._event_counter += 1
            entry = BlackboardEntry(
                id=f"raw_{self._event_counter}",
                level=BlackboardLevel.RAW_EVENTS,
                data={
                    "src_ip": ip,
                    "dst_ip": "10.0.0.1",
                    "dst_port": 22,
                    "protocol": "TCP",
                    "bytes": random.randint(100, 300),
                    "payload": f"SSH login attempt user=admin password=pass{i}",
                    "timestamp": time.time(),
                },
                source_ks="simulator",
            )
            self._blackboard.write(entry)


def main() -> None:
    logger.info("=== Blackboard Architecture: Cyber Security IDS ===")

    # Initialize blackboard
    blackboard = Blackboard()

    # Initialize controller
    controller = BlackboardController(blackboard, max_iterations=200)

    # Register knowledge sources
    controller.register_ks(SignatureDetectorKS())
    controller.register_ks(AnomalyDetectorKS())
    controller.register_ks(CorrelationEngineKS())
    controller.register_ks(ThreatIntelEnrichmentKS())
    controller.register_ks(ResponsePlannerKS())

    # Generate traffic
    simulator = LogSimulator(blackboard)

    # Phase 1: Normal traffic
    logger.info("Phase 1: Generating normal traffic...")
    simulator.generate_normal_traffic(15)

    # Phase 2: Simulate attacks
    logger.info("Phase 2: Simulating attacks...")
    simulator.generate_sql_injection(3)
    simulator.generate_xss_attack(2)
    simulator.generate_brute_force(50)

    # Phase 3: More normal traffic
    logger.info("Phase 3: More normal traffic...")
    simulator.generate_normal_traffic(10)

    # Run controller
    logger.info("Starting Blackboard controller...")
    controller_thread = threading.Thread(target=controller.run)
    controller_thread.start()
    controller_thread.join(timeout=10)

    # Results
    logger.info("=== Results ===")
    stats = blackboard.get_stats()
    logger.info("Blackboard stats: %s", stats)

    hypotheses = blackboard.get_all_hypotheses()
    logger.info("Hypotheses generated: %d", len(hypotheses))
    for h in hypotheses:
        logger.info("  %s | confidence=%.2f | status=%s | evidence=%d",
                     h.description, h.confidence, h.status, len(h.evidence))

    conclusions = blackboard.query(level=BlackboardLevel.CONCLUSION)
    logger.info("Conclusions: %d", len(conclusions))
    for c in conclusions:
        logger.info("  %s | %s", c.data.get("response", ""), c.data.get("description", ""))

    logger.info("=== Blackboard Architecture Demo Complete ===")


if __name__ == "__main__":
    main()
```

## Khi nào dùng / Khi nào không

| Khi nào dùng | Khi nào không |
|--------------|---------------|
| Bài toán không có thuật toán duy nhất | Bài toán có solution deterministic, biết trước |
| Cần kết hợp nhiều nguồn tri thức khác nhau | Chỉ có một nguồn tri thức duy nhất |
| Dữ liệu đến không đồng thời, incremental | Dữ liệu đầy đủ ngay từ đầu |
| Cần tích hợp nhiều legacy/third-party systems | Có thể dùng pipeline đơn giản |
| Giải pháp cần thay đổi linh hoạt (thêm/bớt KS) | Hiệu năng real-time cực cao (microseconds) |
| Hệ thống expert system / decision support | Hệ thống CRUD đơn giản |

## Ưu điểm / Nhược điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| **Modular cực cao**: Mỗi KS độc lập, dễ thêm/bớt/sửa | **Không đảm bảo kết thúc**: Vòng lặp opportunistic có thể infinite |
| **Tận dụng tri thức đa dạng**: Kết hợp nhiều phương pháp | **Performance overhead**: Trigger evaluation + scheduling tốn CPU |
| **Incremental reasoning**: Xử lý từng phần dữ liệu khi có | **Debugging cực kỳ khó**: Behavior không deterministic, KS tương tác gián tiếp |
| **Tích hợp legacy dễ dàng**: Wrapper thành KS | **Không có global view**: Mỗi KS chỉ thấy phần của blackboard |
| **Open/Closed**: Thêm KS mới không ảnh hưởng KS khác | **Hard to test**: Cần integration test với blackboard đầy đủ |
| **Tự nhiên, giống cách người giải quyết vấn đề** | **Scalability của blackboard**: Một blackboard có thể bottleneck |

## Công cụ và Framework

| Tên | Loại | Ngôn ngữ | Ghi chú |
|-----|------|----------|---------|
| **ROS (Robot Operating System)** | Open Source | C++, Python | Blackboard qua topic + service |
| **Google's MapReduce / Beam** | Framework | Java, Python | Dataflow pipeline (gần với blackboard) |
| **Apache Storm** | Stream processing | Java | Real-time processing topology |
| **Drools** | Rule engine | Java | Rule-based KS implementation |
| **CLIPS / JESS** | Expert system | C, Java | Production rule system |
| **Prolog** | Logic programming | Prolog | Built-in blackboard via dynamic DB |
| **Custom Python** | DIY | Python | Dễ implement với threading + Observer |

## Kiểm thử

Testing Blackboard Architecture tập trung vào: (1) từng KS riêng lẻ, (2) interaction giữa KS và blackboard, (3) controller scheduling.

```python
from __future__ import annotations
import pytest
import time
import threading
from typing import Any


class TestBlackboard:
    def test_write_and_query(self, blackboard: Blackboard) -> None:
        entry = BlackboardEntry(id="e1", level=BlackboardLevel.RAW_EVENTS, data={"key": "val"})
        blackboard.write(entry)
        results = blackboard.query(level=BlackboardLevel.RAW_EVENTS)
        assert len(results) == 1
        assert results[0].id == "e1"

    def test_query_with_pattern(self, blackboard: Blackboard) -> None:
        blackboard.write(BlackboardEntry(id="e1", level=BlackboardLevel.AGGREGATED, data={"type": "attack", "ip": "1.2.3.4"}))
        blackboard.write(BlackboardEntry(id="e2", level=BlackboardLevel.AGGREGATED, data={"type": "normal", "ip": "5.6.7.8"}))
        results = blackboard.query(level=BlackboardLevel.AGGREGATED, pattern={"type": "attack"})
        assert len(results) == 1
        assert results[0].id == "e1"

    def test_hypothesis_management(self, blackboard: Blackboard) -> None:
        hyp = Hypothesis(id="h1", description="Test hypothesis")
        hyp.add_evidence("e1", 0.5)
        blackboard.add_hypothesis(hyp)
        loaded = blackboard.get_hypothesis("h1")
        assert loaded is not None
        assert loaded.confidence == 0.5
        assert loaded.status == "pending"

    def test_subscribe_notification(self, blackboard: Blackboard) -> None:
        received: list[BlackboardEntry] = []

        def callback(entry: BlackboardEntry) -> None:
            received.append(entry)

        blackboard.subscribe(BlackboardLevel.CONCLUSION, callback)
        entry = BlackboardEntry(id="conc1", level=BlackboardLevel.CONCLUSION, data={"result": "done"})
        blackboard.write(entry)
        assert len(received) == 1
        assert received[0].id == "conc1"


class TestKnowledgeSources:
    def test_signature_detector_trigger(self, blackboard_with_events: Blackboard) -> None:
        detector = SignatureDetectorKS()
        assert detector.trigger_condition(blackboard_with_events) is True

    def test_signature_detector_execute(self, blackboard_with_events: Blackboard) -> None:
        detector = SignatureDetectorKS()
        entries = detector.execute(blackboard_with_events)
        assert len(entries) > 0
        for e in entries:
            assert e.level == BlackboardLevel.AGGREGATED
            assert e.source_ks == "signature_detector"

    def test_signature_detector_sql_injection(self, blackboard: Blackboard) -> None:
        blackboard.write(BlackboardEntry(
            id="sqli_test", level=BlackboardLevel.RAW_EVENTS,
            data={"src_ip": "1.2.3.4", "payload": "1' OR '1'='1"}
        ))
        detector = SignatureDetectorKS()
        entries = detector.execute(blackboard)
        assert len(entries) == 1
        assert entries[0].data["attack_type"] == "SQL_INJECTION"

    def test_anomaly_detector(self, blackboard: Blackboard) -> None:
        # Write baseline events
        for i in range(5):
            blackboard.write(BlackboardEntry(
                id=f"norm_{i}", level=BlackboardLevel.RAW_EVENTS,
                data={"src_ip": "10.0.0.1", "dst_port": 80, "bytes": 500}
            ))
        detector = AnomalyDetectorKS(threshold_std=0.5)
        # Should not trigger anomaly for low count
        entries_1 = detector.execute(blackboard)
        # Now add many events from same IP
        for i in range(100):
            blackboard.write(BlackboardEntry(
                id=f"burst_{i}", level=BlackboardLevel.RAW_EVENTS,
                data={"src_ip": "10.0.0.1", "dst_port": 80, "bytes": 500}
            ))
        entries_2 = detector.execute(blackboard)
        assert len(entries_2) > 0  # Should detect anomaly

    def test_response_planner(self, blackboard: Blackboard) -> None:
        hyp = Hypothesis(id="h_resp", description="Confirmed attack from test", status="confirmed", confidence=0.95)
        blackboard.add_hypothesis(hyp)
        planner = ResponsePlannerKS()
        assert planner.trigger_condition(blackboard) is True
        entries = planner.execute(blackboard)
        assert len(entries) >= 1
        assert "BLOCK_IP" in str(entries[0].data.get("response", ""))


class TestBlackboardController:
    def test_controller_runs_ks(self, blackboard_with_events: Blackboard) -> None:
        controller = BlackboardController(blackboard_with_events, max_iterations=10)
        detector = SignatureDetectorKS()
        controller.register_ks(detector)
        controller.run()
        results = blackboard_with_events.query(level=BlackboardLevel.AGGREGATED)
        assert len(results) > 0

    def test_controller_multiple_ks(self, blackboard_with_events: Blackboard) -> None:
        controller = BlackboardController(blackboard_with_events, max_iterations=30)
        controller.register_ks(SignatureDetectorKS())
        controller.register_ks(AnomalyDetectorKS())
        controller.register_ks(ResponsePlannerKS())
        controller.run()
        hyp = blackboard_with_events.get_all_hypotheses()
        conclusions = blackboard_with_events.query(level=BlackboardLevel.CONCLUSION)
        # Should have at least some output
        assert len(blackboard_with_events.query(level=BlackboardLevel.AGGREGATED)) > 0


class TestEndToEnd:
    def test_full_ids_pipeline(self) -> None:
        bb = Blackboard()
        controller = BlackboardController(bb, max_iterations=100)
        controller.register_ks(SignatureDetectorKS())
        controller.register_ks(AnomalyDetectorKS())
        controller.register_ks(CorrelationEngineKS())
        controller.register_ks(ResponsePlannerKS())

        # Simulate attack
        bb.write(BlackboardEntry(
            id="attack_1", level=BlackboardLevel.RAW_EVENTS,
            data={"src_ip": "5.5.5.5", "payload": "1 UNION SELECT * FROM admin"}
        ))
        for i in range(30):
            bb.write(BlackboardEntry(
                id=f"brute_{i}", level=BlackboardLevel.RAW_EVENTS,
                data={"src_ip": "5.5.5.5", "payload": f"login attempt {i}"}
            ))

        controller.run()

        conclusions = bb.query(level=BlackboardLevel.CONCLUSION)
        assert len(conclusions) > 0
        # Should have a blocking response
        assert any("BLOCK" in str(c.data) for c in conclusions)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def blackboard() -> Blackboard:
    return Blackboard()


@pytest.fixture
def blackboard_with_events(blackboard: Blackboard) -> Blackboard:
    bb = blackboard
    bb.write(BlackboardEntry(
        id="raw_1", level=BlackboardLevel.RAW_EVENTS,
        data={"src_ip": "192.168.1.1", "payload": "GET /search?q=1' OR '1'='1"}
    ))
    bb.write(BlackboardEntry(
        id="raw_2", level=BlackboardLevel.RAW_EVENTS,
        data={"src_ip": "192.168.1.1", "payload": "GET /index.html"}
    ))
    bb.write(BlackboardEntry(
        id="raw_3", level=BlackboardLevel.RAW_EVENTS,
        data={"src_ip": "10.0.0.2", "payload": "<script>alert(1)</script>"}
    ))
    return bb
```

## Kết luận

Blackboard Architecture là một kiến trúc mạnh mẽ cho các bài toán phức tạp, không có thuật toán duy nhất, cần kết hợp nhiều nguồn tri thức. Nó đặc biệt hiệu quả trong các lĩnh vực AI, cybersecurity, robotics, và medical diagnosis — nơi các chuyên gia khác nhau cần hợp tác để giải quyết vấn đề.

**Best Practices:**
- **Thiết kế KS granularity vừa phải**: KS quá nhỏ → quản lý phức tạp; KS quá lớn → mất tính linh hoạt
- **Blackboard level design**: Chọn số level phù hợp (3-5 là lý tưởng). Quá ít → thiếu abstraction; quá nhiều → phức tạp
- **Control strategy**: Data-driven cho bài toán analysis (dữ liệu thô → kết luận). Goal-driven cho bài toán planning
- **Confidence propagation**: Thiết lập cơ chế kết hợp confidence từ nhiều KS (trung bình, trọng số, fuzzy logic)
- **Termination guarantee**: Luôn đặt max iteration/timeout để tránh infinite loop
- **Event-driven blackboard updates**: Dùng observer pattern để tránh polling

**Golden Rules:**
1. KS không giao tiếp trực tiếp — mọi thứ qua blackboard.
2. Blackboard là single source of truth.
3. Control component có thể thay thế được (pluggable scheduler).
4. Confidence score là mandatory — mọi kết luận đều có độ tin cậy kèm theo.
5. Thiết kế cho incremental processing — không giả định dữ liệu đầy đủ.
6. Log đầy đủ KS activation sequence để debug.
7. Ưu tiên eventual correctness hơn real-time perfection.
