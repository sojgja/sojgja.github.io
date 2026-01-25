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

WebSocket APIs cung cấp real-time, bidirectional communication cho trading systems. Trong bài viết này, chúng ta sẽ xây dựng WebSocket API với FastAPI và WebSockets.

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
