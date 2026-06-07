---
id: architecture
title: System Architecture Patterns
sidebar_label: Architecture
sidebar_position: 1
description: Scalable system architecture patterns — event-driven, CQRS, microservices, event sourcing for trading platforms and high-throughput systems.
keywords: [system-design, architecture, event-driven, cqrs, microservices, event-sourcing, scalability]
---

# System Architecture

Architecture patterns for building scalable, fault-tolerant trading systems.

## Event-Driven Architecture

```python
import asyncio
from collections import defaultdict

class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: callable):
        self._handlers[event_type].append(handler)

    async def publish(self, event_type: str, data: dict):
        tasks = [h(data) for h in self._handlers[event_type]]
        await asyncio.gather(*tasks)

# Usage
bus = EventBus()
bus.subscribe('order.filled', update_position)
bus.subscribe('order.filled', notify_user)
bus.subscribe('order.filled', log_to_db)
```

## CQRS Pattern

```python
# Command side (write)
class CreateOrderCommand:
    def __init__(self, symbol: str, side: str, qty: float, price: float):
        self.data = {'symbol': symbol, 'side': side, 'qty': qty, 'price': price}

class OrderCommandHandler:
    def handle(self, cmd: CreateOrderCommand):
        # Validate, write to DB, emit event
        order = Order.objects.create(**cmd.data)
        bus.publish('order.created', {'id': order.id})

# Query side (read)
class OrderReadModel:
    def get_open_orders(self, user_id: int) -> list:
        # Optimized read from materialized view or cache
        return cache.get(f'orders:open:{user_id}')
```

## Circuit Breaker

```python
from functools import wraps
import time

def circuit_breaker(max_failures=3, reset_timeout=60):
    def decorator(func):
        state = {'failures': 0, 'last_failure': 0}
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if state['failures'] >= max_failures:
                if time.time() - state['last_failure'] < reset_timeout:
                    raise Exception('Circuit breaker OPEN')
                state['failures'] = 0
            try:
                result = await func(*args, **kwargs)
                state['failures'] = 0
                return result
            except Exception:
                state['failures'] += 1
                state['last_failure'] = time.time()
                raise
        return wrapper
    return decorator
```

## Scaling Strategy

| Component | Pattern | Reason |
|-----------|---------|--------|
| Order API | Horizontal (stateless) | Scale with load balancer |
| Matching Engine | Vertical (single instance) | In-memory state, lock-free |
| Market Data | Pub/Sub (Redis/Kafka) | Fan-out to multiple consumers |
| Database | Read replicas | Separate read/write paths |
