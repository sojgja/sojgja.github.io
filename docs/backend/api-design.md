---
id: api-design
title: REST API Design Patterns
sidebar_label: API Design
sidebar_position: 2
description: REST API design best practices — versioning, pagination, rate limiting, JWT auth, OpenAPI schema generation for Django DRF.
keywords: [api, rest, drf, django, pagination, jwt, authentication, openapi]
---

# REST API Design

Standard patterns for building maintainable, performant REST APIs.

## Versioning

```python
# urls.py
from django.urls import path, include

urlpatterns = [
    path('api/v1/', include('api.v1.urls')),
    path('api/v2/', include('api.v2.urls')),
]
```

## Pagination

```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.CursorPagination',
    'PAGE_SIZE': 100,
}

# Custom pagination for trading data
class CursorPagination(CursorPagination):
    ordering = '-created_at'
    page_size_query_param = 'limit'
    max_page_size = 1000
```

## JWT Authentication

```python
from rest_framework_simplejwt.views import TokenObtainPairView

# Login → returns access + refresh tokens
# POST /api/v1/auth/login/
class LoginView(TokenObtainPairView):
    serializer_class = CustomTokenSerializer

# Access header: Authorization: Bearer <access_token>
```

## Rate Limiting

```python
from django_ratelimit.decorators import ratelimit

@ratelimit(key='user', rate='100/m', method='POST')
def place_order(request):
    # Trading endpoints: 100 req/min per user
    ...
```
