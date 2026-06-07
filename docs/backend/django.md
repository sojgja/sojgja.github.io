---
id: django
title: Django Backend Guide
sidebar_label: Django
sidebar_position: 1
description: Production-grade Django patterns — ORM optimization, middleware, async views, Celery task queues, and PostgreSQL integration.
keywords: [django, backend, python, orm, celery, postgresql, middleware, async]
---

# Django Backend

Production Django patterns for high-performance APIs and data processing.

## Architecture

```python
# project/settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'trading_db',
        'USER': 'app',
        'PASSWORD': os.environ['DB_PASSWORD'],
        'HOST': 'postgres',
        'PORT': 5432,
        'CONN_MAX_AGE': 600,
    }
}

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://redis:6379/1',
    }
}
```

## ORM Optimization

```python
# Use select_related for FK, prefetch_related for M2M
orders = (
    Order.objects
    .select_related('user', 'symbol')
    .prefetch_related('fills')
    .only('id', 'price', 'quantity', 'status')
)

# Bulk operations
Order.objects.bulk_create([
    Order(symbol='BTCUSDT', price=p) for p in prices
], batch_size=1000)
```

## Celery Task Queue

```python
# tasks.py
from celery import shared_task

@shared_task(bind=True, max_retries=3)
def process_order(self, order_id: str):
    try:
        order = Order.objects.get(id=order_id)
        engine.execute(order)
    except Exception as exc:
        self.retry(exc=exc, countdown=5)
```

## Async Views (Django 4.1+)

```python
from django.http import JsonResponse

async def market_data(request, symbol: str):
    data = await redis_client.get(f"ticker:{symbol}")
    return JsonResponse(json.loads(data))
```
