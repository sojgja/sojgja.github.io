---
date: 2025-01-25
authors:
  - soigia
categories: [API Development, Real-time Systems]
title: WebSocket API cho Trading - Real-time Data Streaming
description: >
  Xây dựng WebSocket API để stream real-time trading data: connection management, message handling và scaling strategies.
---

# WebSocket API cho Trading

![WebSocket Real-time Communication](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&h=600&fit=crop)

Trong thế giới của algorithmic trading và real-time market data processing, tốc độ và độ trễ (latency) là những yếu tố quyết định thành công. Trong khi REST API truyền thống hoạt động theo mô hình request-response với overhead lớn và độ trễ cao, WebSocket API cung cấp một kênh giao tiếp real-time, bidirectional, và hiệu quả hơn rất nhiều. Với WebSocket, chúng ta có thể thiết lập một kết nối persistent giữa client và server, cho phép server push dữ liệu đến client ngay lập tức khi có thay đổi, mà không cần client phải liên tục polling. Điều này đặc biệt quan trọng trong trading, nơi mà mỗi mili giây đều có giá trị và việc nhận được thông tin giá mới nhất có thể quyết định thành bại của một giao dịch.

Trong bài viết chi tiết này, chúng ta sẽ cùng nhau xây dựng một WebSocket API hoàn chỉnh cho trading systems, sử dụng FastAPI - một framework Python hiện đại, nhanh chóng, và được thiết kế đặc biệt cho việc xây dựng APIs. Chúng ta sẽ học cách thiết lập WebSocket connections, quản lý nhiều clients đồng thời, xử lý các message types khác nhau, implement authentication và authorization, và quan trọng nhất là xây dựng một hệ thống có khả năng scale để xử lý hàng nghìn connections cùng lúc. Chúng ta cũng sẽ thảo luận về các patterns như pub/sub để broadcast market data đến nhiều clients, connection pooling, heartbeat mechanisms để detect và xử lý dead connections, và error handling robust.

Bài viết này sẽ hướng dẫn bạn từng bước một, từ việc setup FastAPI với WebSocket support, implement connection manager để track và quản lý các active connections, xây dựng message handlers cho các loại message khác nhau (subscribe, unsubscribe, trade execution, etc.), đến việc tích hợp với trading bot và market data providers. Chúng ta cũng sẽ học cách test WebSocket APIs, monitor performance, và implement các security best practices. Cuối cùng, bạn sẽ có trong tay một WebSocket API mạnh mẽ, có thể stream real-time trading data và execute trades với độ trễ cực thấp.

<!-- more -->

## WebSocket vs REST

- **REST**: Request-response, stateless
- **WebSocket**: Persistent connection, bidirectional, real-time

## Implementation

```python
# api/websocket_api.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio

app = FastAPI()

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, List[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = []
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str, symbol: str = None):
        for connection in self.active_connections:
            if symbol is None or symbol in self.subscriptions.get(connection, []):
                try:
                    await connection.send_text(message)
                except:
                    self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/ticker")
async def websocket_ticker(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('action') == 'subscribe':
                symbol = message.get('symbol')
                manager.subscriptions[websocket].append(symbol)
                await manager.send_personal_message(
                    json.dumps({'status': 'subscribed', 'symbol': symbol}),
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

## Kết luận

WebSocket APIs enable real-time trading data streaming!
