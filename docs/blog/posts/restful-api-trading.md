---
date: 2025-01-25
authors:
  - soigia
categories: [API Development, Backend Development]
title: Xây dựng RESTful API cho Trading - FastAPI và Best Practices
description: >
  Hướng dẫn xây dựng RESTful API cho trading systems với FastAPI: authentication, rate limiting, error handling và documentation.
---

# Xây dựng RESTful API cho Trading

RESTful APIs là cách standard để expose trading functionality. Trong bài viết này, chúng ta sẽ xây dựng trading API với FastAPI.

<!-- more -->

## API Endpoints Design

```
GET  /api/v1/tickers/{symbol}        # Get ticker
GET  /api/v1/ohlcv/{symbol}          # Get OHLCV data
GET  /api/v1/portfolio               # Get portfolio
POST /api/v1/orders                   # Place order
GET  /api/v1/orders/{order_id}       # Get order status
```

## Bước 1: FastAPI Setup

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import jwt

app = FastAPI(title="Trading API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models
class TickerResponse(BaseModel):
    symbol: str
    exchange: str
    price: float
    volume_24h: Optional[float]
    change_24h: Optional[float]
    timestamp: datetime

class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str = 'MARKET'  # 'MARKET' or 'LIMIT'
    price: Optional[float] = None

class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: str
    quantity: float
    status: str
    timestamp: datetime

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Endpoints
@app.get("/api/v1/tickers/{symbol}", response_model=TickerResponse)
async def get_ticker(symbol: str, exchange: str = "binance"):
    """Get ticker for symbol"""
    # Implementation
    pass

@app.get("/api/v1/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    exchange: str = "binance",
    interval: str = "1h",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100
):
    """Get OHLCV data"""
    # Implementation
    pass

@app.post("/api/v1/orders", response_model=OrderResponse)
async def place_order(
    order: OrderRequest,
    user: dict = Depends(verify_token)
):
    """Place trading order"""
    # Implementation
    pass

@app.get("/api/v1/portfolio")
async def get_portfolio(user: dict = Depends(verify_token)):
    """Get user portfolio"""
    # Implementation
    pass
```

## Bước 2: Rate Limiting

```python
# api/rate_limit.py
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/v1/tickers/{symbol}")
@limiter.limit("10/minute")
async def get_ticker(request: Request, symbol: str):
    # Implementation
    pass
```

## Best Practices

1. **Authentication**: Use JWT tokens
2. **Rate Limiting**: Prevent abuse
3. **Error Handling**: Consistent error responses
4. **Validation**: Validate all inputs
5. **Documentation**: Auto-generated docs
6. **Logging**: Log all requests
7. **Monitoring**: Monitor API health

## Kết luận

RESTful APIs enable:
- Standardized interfaces
- Easy integration
- Scalable architecture
- Better developer experience

Build robust APIs for your trading systems!
