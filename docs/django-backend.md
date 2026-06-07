---
sidebar_position: 3
---

# Django Backend

The backend API built with Django REST Framework provides data persistence, authentication, and business logic.

## Tech Stack

- **Framework:** Django 5.x + Django REST Framework
- **Database:** PostgreSQL 16
- **Cache:** Redis
- **Task Queue:** Celery
- **Monitoring:** Prometheus + Grafana

## API Endpoints

### Trading API

```python
# Example: Place an order
POST /api/v1/orders/
{
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "quantity": 0.01,
    "price": 50000.00
}
```

### Market Data API

```python
# Example: Get OHLCV candles
GET /api/v1/market/candles/?symbol=BTCUSDT&interval=1h&limit=100
```

## Database Schema

```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    type VARCHAR(10) NOT NULL,
    quantity DECIMAL(18,8),
    price DECIMAL(18,8),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Deployment

```yaml
# docker-compose.prod.yml
services:
  backend:
    build: ./backend
    environment:
      - DJANGO_SETTINGS_MODULE=config.production
    depends_on:
      - postgres
      - redis
```
