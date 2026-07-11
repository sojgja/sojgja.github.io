---
id: solid-isp
title: I — Interface Segregation Principle
sidebar_label: I — Interface Segregation
sidebar_position: 29
---

# I — Interface Segregation Principle

> *"No client should be forced to depend on methods it does not use."* — **Robert C. Martin, *Agile Software Development, Principles, Patterns, and Practices*, 2002**

Interface Segregation Principle (ISP) là nguyên lý thứ tư trong SOLID, được Robert C. Martin phát biểu như một hệ quả trực tiếp của SRP áp dụng cho interface design. Nếu SRP nói "một class chỉ có một lý do để thay đổi", ISP nói "một interface chỉ nên phục vụ một client duy nhất" — hay nói cách khác, interface tổng quát phục vụ nhiều client khác nhau là thiết kế tồi. ISP khuyến khích tạo ra nhiều interface nhỏ, chuyên biệt thay vì một interface lớn, tổng hợp. Điều này nghe có vẻ đơn giản, nhưng việc xác định đâu là ranh giới giữa "interface vừa đủ" và "interface quá nhỏ" đòi hỏi kinh nghiệm và sự hiểu biết về business domain. Một interface bị "fat" (quá béo) buộc client phải implement những method không cần thiết, dẫn đến code trùng lặp, exception không mong muốn, và vi phạm cả LSP.

## Bài toán chi tiết: Hệ thống notification đa kênh

Một công ty công nghệ tài chính (FinTech) xây dựng hệ thống notification để gửi thông báo đến khách hàng qua nhiều kênh: email, SMS, push notification mobile, WebSocket real-time, Slack cho nội bộ, và webhook cho đối tác. Họ thiết kế một interface `NotificationService` với đầy đủ methods:

```python
class NotificationService(ABC):
    @abstractmethod def send_email(self, to, subject, body): ...
    @abstractmethod def send_sms(self, phone, message): ...
    @abstractmethod def send_push(self, device_token, payload): ...
    @abstractmethod def send_websocket(self, connection_id, data): ...
    @abstractmethod def send_slack(self, channel, message): ...
    @abstractmethod def send_webhook(self, url, payload): ...
    @abstractmethod def send_bulk(self, recipients, template): ...
    @abstractmethod def get_delivery_status(self, notification_id): ...
    @abstractmethod def cancel_notification(self, notification_id): ...
    @abstractmethod def schedule_notification(self, when, message): ...
```

Ban đầu chỉ có email và SMS. Team implement `EmailNotificationService` và `SMSNotificationService` — cả hai đều phải implement 10 method, dù chỉ cần 2-3 method thực sự. `send_push()`, `send_websocket()`, `send_slack()`, `send_webhook()` — tất cả đều để trống hoặc ném `NotImplementedError`. Sau 6 tháng, họ thêm push notification. `PushNotificationService` implement interface này — và lại phải implement các method không liên quan. `send_email()` trong push service được implement tạm bợ, gây nhầm lẫn cho các developer khác.

Vấn đề trở nên nghiêm trọng khi một junior developer gọi `push_service.send_email()` — tưởng rằng nó gửi email, nhưng thực ra nó chỉ log ra console. Bug lọt ra production, khách hàng không nhận được email xác nhận giao dịch quan trọng. Sau sự cố, team quyết định thêm type check: `if isinstance(service, PushNotificationService): ...` — code trở nên xấu và vi phạm OCP. Mỗi lần thêm channel mới, họ phải mở tất cả các service hiện có để thêm method mới vào. Interface này trở thành "god interface" — một anti-pattern kinh điển làm tê liệt khả năng phát triển của hệ thống. Cuối cùng, họ mất 4 tuần để refactor toàn bộ, tách thành các interface nhỏ hơn.

## Phân tích vấn đề

Root cause là interface `NotificationService` phục vụ quá nhiều client khác nhau: email client, SMS client, push client, Slack client — mỗi client chỉ cần một subset nhỏ của interface. Khi interface lớn, mọi thay đổi (dù nhỏ) đều ảnh hưởng đến tất cả implementations. Cụ thể:

1. **Interface Pollution**: Interface chứa method không liên quan đến domain của một số implementations. Ví dụ: `send_slack()` không liên quan gì đến email notification.
2. **Forced Implementation**: Subclass buộc phải implement method không cần thiết — dẫn đến code chết (dead code), no-op, hoặc exception.
3. **Increased Coupling**: Một thay đổi trong interface (ví dụ: thêm tham số `attachments` vào `send_email()`) buộc tất cả subclass (kể cả SMS, Push, Slack) phải cập nhật.
4. **Semantic Confusion**: Method có tên giống nhau nhưng ý nghĩa khác nhau tùy implementation. `send_email()` trong push service không thực sự gửi email — gây hiểu lầm.
5. **Violation of LSP**: Subclass ném `NotImplementedError` — vi phạm LSP vì không thay thế được base class.
6. **Interface versioning nightmare**: Khi interface phát triển, tất cả implementations phải được cập nhật đồng bộ — khó khăn trong hệ thống lớn.

**Code smells** của vi phạm ISP: interface có tên chứa chữ "and" (ví dụ: `EmailAndSmsService`), method `raise NotImplementedError` trong subclass, class implement interface có method `pass` nhiều, interface có hơn 5-6 method không liên quan chặt chẽ, client phải biết về các method không dùng đến.

## Giải pháp: Role-based Interface Segregation

Giải pháp là áp dụng **Role Interface Pattern** — mỗi interface đại diện cho một role (vai trò) mà client cần. Thay vì một interface notification "đa năng", tách thành các interface chuyên biệt:

1. **`EmailSender`** — chỉ gửi email
2. **`SmsSender`** — chỉ gửi SMS
3. **`PushNotificationSender`** — chỉ gửi push notification
4. **`WebSocketSender`** — chỉ gửi WebSocket message
5. **`SlackNotifier`** — chỉ gửi Slack message
6. **`WebhookSender`** — chỉ gửi webhook
7. **`BulkNotifier`** — gửi bulk notification
8. **`NotificationTracker`** — theo dõi trạng thái, hủy, lên lịch

Mỗi class chỉ implement những interface phù hợp với nó. `EmailNotificationService` implement `EmailSender` và `NotificationTracker`. `PushNotificationService` implement `PushNotificationSender`. Không ai bị ép phải implement method không dùng. Khi thêm channel mới (Zalo, Telegram, Viber), chỉ cần tạo class mới implement interface tương ứng.

## Ví dụ code hoàn chỉnh

### VIOLATION — Vi phạm ISP

```python
# notification_violation.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class NotificationRequest:
    recipient: str
    subject: str | None = None
    body: str = ''
    template_id: str | None = None
    attachments: list[str] = field(default_factory=list)
    priority: int = 0
    scheduled_at: datetime | None = None


@dataclass
class NotificationResult:
    success: bool
    notification_id: str | None = None
    error: str | None = None
    retry_count: int = 0


class NotificationService(ABC):
    """
    FAT INTERFACE: 10 method, không phải client nào cũng cần tất cả.
    Bất kỳ thay đổi nào cũng ảnh hưởng đến tất cả implementations.
    """

    @abstractmethod
    def send_email(self, request: NotificationRequest) -> NotificationResult:
        ...

    @abstractmethod
    def send_sms(self, request: NotificationRequest) -> NotificationResult:
        ...

    @abstractmethod
    def send_push(self, token: str, payload: dict[str, Any]) -> NotificationResult:
        ...

    @abstractmethod
    def send_websocket(self, connection_id: str, data: Any) -> None:
        ...

    @abstractmethod
    def send_slack(self, channel: str, message: str) -> NotificationResult:
        ...

    @abstractmethod
    def send_webhook(self, url: str, payload: dict[str, Any]) -> NotificationResult:
        ...

    @abstractmethod
    def send_bulk(self, recipients: list[str], template_id: str) -> list[NotificationResult]:
        ...

    @abstractmethod
    def get_status(self, notification_id: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def cancel(self, notification_id: str) -> bool:
        ...

    @abstractmethod
    def schedule(self, request: NotificationRequest, when: datetime) -> str:
        ...


class EmailNotificationService(NotificationService):
    """
    VIOLATION: Chỉ cần send_email, get_status, cancel, schedule.
    Phải implement 6 method khác vô dụng.
    """

    def send_email(self, request: NotificationRequest) -> NotificationResult:
        print(f"📧 Gửi email đến {request.recipient}: {request.subject}")
        return NotificationResult(success=True, notification_id=f"email_{id(self)}")

    def send_sms(self, request: NotificationRequest) -> NotificationResult:
        raise NotImplementedError("Email service không hỗ trợ SMS!")  # ❌

    def send_push(self, token: str, payload: dict[str, Any]) -> NotificationResult:
        raise NotImplementedError("Email service không hỗ trợ Push!")  # ❌

    def send_websocket(self, connection_id: str, data: Any) -> None:
        raise NotImplementedError("Email service không hỗ trợ WebSocket!")  # ❌

    def send_slack(self, channel: str, message: str) -> NotificationResult:
        raise NotImplementedError("Email service không hỗ trợ Slack!")  # ❌

    def send_webhook(self, url: str, payload: dict[str, Any]) -> NotificationResult:
        raise NotImplementedError("Email service không hỗ trợ Webhook!")  # ❌

    def send_bulk(self, recipients: list[str], template_id: str) -> list[NotificationResult]:
        results: list[NotificationResult] = []
        for r in recipients:
            req = NotificationRequest(recipient=r, template_id=template_id)
            results.append(self.send_email(req))
        return results

    def get_status(self, notification_id: str) -> dict[str, Any]:
        return {'id': notification_id, 'status': 'sent', 'channel': 'email'}

    def cancel(self, notification_id: str) -> bool:
        return True

    def schedule(self, request: NotificationRequest, when: datetime) -> str:
        print(f"⏰ Lên lịch gửi email {request.recipient} lúc {when}")
        return f"scheduled_{id(self)}"


class PushNotificationService(NotificationService):
    """Cũng phải implement 10 method dù chỉ cần send_push."""

    def send_email(self, request: NotificationRequest) -> NotificationResult:
        # ⚠️ Gây hiểu lầm: method tên send_email nhưng không gửi email
        print(f"Push service không thể gửi email!")
        return NotificationResult(success=False, error="Not supported")

    def send_sms(self, request: NotificationRequest) -> NotificationResult:
        raise NotImplementedError("Push service không hỗ trợ SMS!")

    def send_push(self, token: str, payload: dict[str, Any]) -> NotificationResult:
        print(f"📱 Gửi push đến {token}")
        return NotificationResult(success=True, notification_id=f"push_{id(self)}")

    # ... 7 method còn lại đều NotImplementedError
    def send_websocket(self, connection_id: str, data: Any) -> None: ...
    def send_slack(self, channel: str, message: str) -> NotificationResult: ...
    def send_webhook(self, url: str, payload: dict[str, Any]) -> NotificationResult: ...
    def send_bulk(self, recipients: list[str], template_id: str) -> list[NotificationResult]: ...
    def get_status(self, notification_id: str) -> dict[str, Any]: ...
    def cancel(self, notification_id: str) -> bool: ...
    def schedule(self, request: NotificationRequest, when: datetime) -> str: ...


# Client code — dễ gọi sai
def send_important_notification(service: NotificationService, user_email: str, message: str) -> None:
    """Ai đó gọi send_email trên push service → không gửi được email thật!"""
    req = NotificationRequest(recipient=user_email, subject="Quan trọng", body=message)
    result = service.send_email(req)  # ❌ Crash nếu push service, hoặc false success
    if not result.success:
        print(f"Failed: {result.error}")
```

### REFACTORED — Tuân thủ ISP

```python
# ─── interfaces/notification_interfaces.py ───
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class EmailRequest:
    to: str
    subject: str
    body: str
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SmsRequest:
    phone: str
    message: str
    sender_name: str = ""


@dataclass(frozen=True)
class PushRequest:
    device_token: str
    title: str
    body: str
    data: dict[str, Any] = field(default_factory=dict)
    badge: int = 0


@dataclass(frozen=True)
class NotificationResult:
    success: bool
    notification_id: str | None = None
    error: str | None = None


# ─── Interfaces nhỏ, chuyên biệt ───

class EmailSender(Protocol):
    """Chỉ gửi email — role nhỏ nhất."""

    def send_email(self, request: EmailRequest) -> NotificationResult: ...


class SmsSender(Protocol):
    """Chỉ gửi SMS."""

    def send_sms(self, request: SmsRequest) -> NotificationResult: ...


class PushSender(Protocol):
    """Chỉ gửi push notification."""

    def send_push(self, request: PushRequest) -> NotificationResult: ...


class NotificationTracker(Protocol):
    """Theo dõi trạng thái notification."""

    def get_status(self, notification_id: str) -> dict[str, Any]: ...

    def cancel(self, notification_id: str) -> bool: ...


class NotificationScheduler(Protocol):
    """Lên lịch gửi notification."""

    def schedule(self, request: Any, when: datetime) -> str: ...


class BulkNotifier(Protocol):
    """Gửi bulk notification."""

    def send_bulk(self, recipients: list[str], template_id: str) -> list[NotificationResult]: ...


# ─── implementations/email_service.py ───
from __future__ import annotations


class SmtpEmailService:
    """Chỉ implement các interface cần thiết: EmailSender + NotificationTracker + NotificationScheduler."""

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._username = username
        self._password = password

    def send_email(self, request: EmailRequest) -> NotificationResult:
        import smtplib
        from email.mime.text import MIMEText  # type: ignore
        try:
            msg = MIMEText(request.body)
            msg['Subject'] = request.subject
            msg['To'] = request.to
            with smtplib.SMTP(self._host, self._port) as server:
                server.login(self._username, self._password)
                server.send_message(msg)
            return NotificationResult(success=True, notification_id=f"email_{request.to}")
        except Exception as e:
            return NotificationResult(success=False, error=str(e))

    def get_status(self, notification_id: str) -> dict[str, Any]:
        return {'id': notification_id, 'channel': 'email', 'status': 'sent'}

    def cancel(self, notification_id: str) -> bool:
        return True

    def schedule(self, request: EmailRequest, when: datetime) -> str:
        # Giả lập lưu vào job queue
        job_id = f"schedule_{id(self)}_{when.timestamp()}"
        print(f"⏰ Scheduled email to {request.to} at {when}, job={job_id}")
        return job_id


# ─── implementations/sms_service.py ───
from __future__ import annotations


class TwilioSmsService:
    """Chỉ implement SmsSender và NotificationTracker."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str) -> None:
        self._sid = account_sid
        self._token = auth_token
        self._from = from_number

    def send_sms(self, request: SmsRequest) -> NotificationResult:
        print(f"📱 Gửi SMS từ {self._from} đến {request.phone}: {request.message[:50]}...")
        return NotificationResult(success=True, notification_id=f"sms_{request.phone}")

    def get_status(self, notification_id: str) -> dict[str, Any]:
        return {'id': notification_id, 'channel': 'sms', 'status': 'sent'}

    def cancel(self, notification_id: str) -> bool:
        return False  # SMS không thể hủy sau khi gửi


# ─── implementations/push_service.py ───
from __future__ import annotations


class FirebasePushService:
    """Chỉ implement PushSender."""

    def __init__(self, server_key: str) -> None:
        self._server_key = server_key

    def send_push(self, request: PushRequest) -> NotificationResult:
        print(f"📲 Gửi push đến {request.device_token[:20]}...: {request.title}")
        return NotificationResult(success=True, notification_id=f"push_{request.device_token}")


# ─── implementations/multi_channel_service.py ───
from __future__ import annotations


class MultiChannelNotificationService:
    """Service tổng hợp — implement nhiều interface cùng lúc."""

    def __init__(self, email_svc: EmailSender, sms_svc: SmsSender, push_svc: PushSender) -> None:
        self._email = email_svc
        self._sms = sms_svc
        self._push = push_svc

    def send_notification(self, channel: str, request: Any) -> NotificationResult:
        if channel == 'email' and isinstance(request, EmailRequest):
            return self._email.send_email(request)
        elif channel == 'sms' and isinstance(request, SmsRequest):
            return self._sms.send_sms(request)
        elif channel == 'push' and isinstance(request, PushRequest):
            return self._push.send_push(request)
        raise ValueError(f"Unsupported channel: {channel}")


# ─── main.py ───
from __future__ import annotations

email_svc = SmtpEmailService("smtp.gmail.com", 587, "user@gmail.com", "app_password")
sms_svc = TwilioSmsService("ACxxxx", "token", "+84123456789")
push_svc = FirebasePushService("server_key_xxxx")

# Client chỉ gọi đúng interface — không thể gọi nhầm
def send_welcome_email(sender: EmailSender, user_email: str, name: str) -> None:
    request = EmailRequest(
        to=user_email,
        subject="Chào mừng bạn đến với hệ thống!",
        body=f"Xin chào {name}, cảm ơn bạn đã đăng ký.",
    )
    result = sender.send_email(request)
    if not result.success:
        print(f"Gửi email thất bại: {result.error}")

send_welcome_email(email_svc, "user@example.com", "Nguyễn Văn A")
# send_welcome_email(push_svc, ...)  # ❌ Type error — PushSender không phải EmailSender
```

## Dấu hiệu nhận biết vi phạm ISP

- **Interface có tên chung chung**: `IManager`, `IProcessor`, `Service`, `Handler` — thường là dấu hiệu của fat interface.
- **Client implement interface nhưng bỏ trống method**: `def method(self): pass` hoặc `raise NotImplementedError`.
- **Method trong interface không được sử dụng bởi tất cả clients**: Một số client gọi method A nhưng không gọi method B, trong khi interface gộp cả A và B.
- **Interface có quá nhiều method**: Hơn 5-6 method thường là dấu hiệu cần xem xét tách interface.
- **Thay đổi interface ảnh hưởng đến quá nhiều class**: Khi thêm một method vào interface, bạn phải update 10+ implementations.
- **Client code import interface nhưng chỉ dùng 2-3 method**: Client phụ thuộc vào những thứ chúng không cần.
- **Interface có method với tham số không dùng đến**: Method `send(to, cc, bcc, attachments, priority, tags)` — nhiều client chỉ cần `to`.
- **Xuất hiện "God Interface" hoặc "Utility Interface"**: Interface cố gắng làm mọi thứ.

## Kiểm thử

```python
# test_notification_interfaces.py
from __future__ import annotations
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import pytest  # type: ignore
from interfaces.notification_interfaces import (
    EmailRequest, SmsRequest, PushRequest, NotificationResult,
    EmailSender, SmsSender, PushSender, NotificationTracker, NotificationScheduler,
)
from implementations.email_service import SmtpEmailService
from implementations.sms_service import TwilioSmsService
from implementations.push_service import FirebasePushService
from implementations.multi_channel_service import MultiChannelNotificationService


class TestEmailService:

    @pytest.fixture
    def email_service(self) -> SmtpEmailService:
        return SmtpEmailService("smtp.test.com", 587, "test@test.com", "pass")

    def test_send_email_success(self, email_service: SmtpEmailService) -> None:
        request = EmailRequest(to="a@test.com", subject="Test", body="Hello")
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            result = email_service.send_email(request)
        assert result.success is True
        assert result.notification_id is not None

    def test_send_email_smtp_failure(self, email_service: SmtpEmailService) -> None:
        request = EmailRequest(to="a@test.com", subject="Test", body="Hello")
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            mock_server.send_message.side_effect = ConnectionError("SMTP timeout")
            result = email_service.send_email(request)
        assert result.success is False
        assert result.error is not None

    def test_schedule_email(self, email_service: SmtpEmailService) -> None:
        future = datetime.now() + timedelta(hours=1)
        request = EmailRequest(to="b@test.com", subject="Later", body="Scheduled")
        job_id = email_service.schedule(request, future)
        assert job_id.startswith("schedule_")

    def test_cancel_notification(self, email_service: SmtpEmailService) -> None:
        assert email_service.cancel("any_id") is True

    def test_implements_email_sender(self, email_service: SmtpEmailService) -> None:
        """Kiểm tra interface compatibility."""
        from typing import cast
        sender: EmailSender = cast(EmailSender, email_service)
        assert sender is not None


class TestSmsService:

    @pytest.fixture
    def sms_service(self) -> TwilioSmsService:
        return TwilioSmsService("AC_test", "token_test", "+84111111111")

    def test_send_sms_success(self, sms_service: TwilioSmsService) -> None:
        request = SmsRequest(phone="+84988888888", message="Test SMS")
        result = sms_service.send_sms(request)
        assert result.success is True
        assert "sms_" in (result.notification_id or "")

    def test_sms_cancel_always_false(self, sms_service: TwilioSmsService) -> None:
        """SMS thường không hủy được sau khi gửi."""
        assert sms_service.cancel("any") is False

    def test_sms_tracker(self, sms_service: TwilioSmsService) -> None:
        status = sms_service.get_status("sms_test")
        assert status['channel'] == 'sms'


class TestPushService:

    @pytest.fixture
    def push_service(self) -> FirebasePushService:
        return FirebasePushService("test_key")

    def test_send_push(self, push_service: FirebasePushService) -> None:
        request = PushRequest(
            device_token="device_abc_123",
            title="Alert",
            body="Push message",
            data={'order_id': 'ORD-001'},
        )
        result = push_service.send_push(request)
        assert result.success is True


class TestMultiChannelService:

    def test_multi_channel_routing(self) -> None:
        email_mock = Mock(spec=EmailSender)
        sms_mock = Mock(spec=SmsSender)
        push_mock = Mock(spec=PushSender)

        email_mock.send_email.return_value = NotificationResult(success=True, notification_id="e1")
        sms_mock.send_sms.return_value = NotificationResult(success=True, notification_id="s1")
        push_mock.send_push.return_value = NotificationResult(success=True, notification_id="p1")

        service = MultiChannelNotificationService(email_mock, sms_mock, push_mock)

        # Test gửi email
        email_req = EmailRequest(to="a@t.com", subject="S", body="B")
        result = service.send_notification('email', email_req)
        assert result.success is True
        email_mock.send_email.assert_called_once_with(email_req)

        # Test gửi SMS
        sms_req = SmsRequest(phone="+84", message="Hi")
        result = service.send_notification('sms', sms_req)
        assert result.success is True
        sms_mock.send_sms.assert_called_once_with(sms_req)

        # Test invalid channel
        with pytest.raises(ValueError):
            service.send_notification('telegram', {})


class TestInterfaceSegregation:
    """Kiểm tra mỗi class chỉ implement interface nó cần."""

    def test_push_service_not_email_sender(self) -> None:
        """Push service không implement EmailSender — không thể gọi send_email."""
        push = FirebasePushService("key")
        # push.send_email(...)  # ❌ Type error — không có method send_email
        # Chỉ có các method cần thiết:
        request = PushRequest(device_token="t", title="T", body="B")
        result = push.send_push(request)
        assert result.success is True

    def test_sms_service_not_bulk_notifier(self) -> None:
        sms = TwilioSmsService("sid", "token", "+84")
        # sms.send_bulk(...)  # ❌ Không có method send_bulk
        # Chỉ có send_sms, get_status, cancel
        assert hasattr(sms, 'send_sms')
        assert hasattr(sms, 'get_status')
        assert hasattr(sms, 'cancel')

    def test_email_service_full_featured(self) -> None:
        """Email service implement nhiều interface nhất — nhưng vẫn chỉ
        những interface nó cần."""
        email = SmtpEmailService("host", 587, "u", "p")
        assert hasattr(email, 'send_email')
        assert hasattr(email, 'get_status')
        assert hasattr(email, 'cancel')
        assert hasattr(email, 'schedule')
        assert not hasattr(email, 'send_push')  # Không bị ép implement
        assert not hasattr(email, 'send_sms')
```

## Ứng dụng thực tế

1. **Django REST Framework — Serializers và Views**: DRF có nhiều loại serializers (`Serializer`, `ModelSerializer`, `ListSerializer`). Mỗi loại implement một tập method khác nhau — `create()`, `update()`, `to_representation()`, `to_internal_value()`. Không ai bắt `ReadOnlyModelSerializer` phải implement `create()` và `update()`. Đây là ISP đúng: interface `BaseSerializer` chỉ có method chung, các method cụ thể được tách riêng.

2. **FastAPI — Dependency Injection**: FastAPI phân biệt rõ các loại dependencies: `Depends()` cho services, `Query()` cho query params, `Path()` cho path params, `Body()` cho request body, `Header()` cho headers, `Cookie()` cho cookies. Mỗi loại chỉ expose những method cần thiết. Không có một "giải pháp chung cho mọi tham số" — điều này tuân thủ ISP triệt để.

3. **SQLAlchemy — Core vs ORM**: SQLAlchemy tách riêng core (connection, engine, SQL expression) và ORM (session, model, query). Người dùng core không cần biết về session, người dùng ORM không cần biết về connection pooling chi tiết. `Engine` và `Session` là hai interface hoàn toàn riêng biệt — không ai bị ép phải implement cả hai.

4. **Python Standard Library — Collection ABCs**: `collections.abc` định nghĩa các abstract classes nhỏ: `Iterable`, `Collection`, `Sequence`, `MutableSequence`, `Set`, `MutableSet`. `tuple` implement `Sequence` (chỉ đọc), `list` implement `MutableSequence` (có cả ghi). `tuple` không bị ép implement `append()` — ISP được tuân thủ triệt để.

5. **AWS SDK (boto3) — Paginators, Waiters, Resources**: Boto3 không có một interface "AWS service" khổng lồ. Mỗi service (S3, EC2, DynamoDB) có các clients, paginators, waiters riêng — mỗi loại chỉ có method liên quan đến role của nó. `S3.Client.list_objects()` và `S3.Paginator.paginate()` là hai interface khác nhau cho cùng data.

## Liên hệ với Pattern

- **Role Interface Pattern**: Interface được thiết kế dựa trên role (vai trò) của client, không dựa trên class implement nó. Một class có thể implement nhiều role interfaces khác nhau.
- **Adapter Pattern**: Cho phép convert interface của một class thành interface khác mà client mong đợi. ISP giúp mỗi adapter chỉ cần chuyển đổi một interface nhỏ, chuyên biệt.
- **Decorator Pattern**: Mỗi decorator thêm một behavior cụ thể — nếu interface quá lớn, decorator phải implement tất cả method dù chỉ thay đổi một behavior.
- **Facade Pattern**: Cung cấp interface đơn giản cho subsystem phức tạp. ISP giúp facade chỉ expose những method cần thiết, không làm client bối rối.
- **Proxy Pattern**: Proxy kiểm soát truy cập đến real subject. ISP giúp proxy nhẹ hơn vì chỉ cần implement interface nhỏ.
- **Command Pattern**: Encapsulate request thành object. ISP giúp mỗi command có interface đơn giản (`execute()`, `undo()`), không phải implement business methods không liên quan.

## Ưu và nhược điểm

| Tiêu chí | Trước (vi phạm ISP) | Sau (tuân thủ ISP) |
|----------|---------------------|-------------------|
| **Số interface** | 1 interface "fat" | 5-6 interface nhỏ, chuyên biệt |
| **Số method/interface** | 10+ method | 1-3 method |
| **NotImplementedError** | Xuất hiện ở nhiều subclass | Không còn — class không implement interface không phù hợp |
| **Client coupling** | Client phụ thuộc vào method không dùng | Client chỉ phụ thuộc vào method cần thiết |
| **Ảnh hưởng khi thay đổi** | Thay đổi interface ảnh hưởng tất cả | Thay đổi interface nhỏ chỉ ảnh hưởng một số class |
| **Tính linh hoạt** | Thấp — khó thêm channel mới | Cao — thêm channel = thêm class + interface mới |
| **Dễ hiểu** | Thấp — interface quá nhiều method | Cao — mỗi interface một vai trò rõ ràng |
| **Số lượng class** | N (implementations) | N + M (implementations + interfaces) |
| **Dependency management** | Phức tạp — một dependency to | Dễ — dependency nhỏ theo role |
| **Rủi ro gọi nhầm method** | Cao — send_email trên push service | Thấp — type system ngăn gọi sai |
| **Phù hợp với** | Hệ thống nhỏ, ít thay đổi | Hệ thống lớn, nhiều biến thể |

## Kết luận

ISP đơn giản nhưng sâu sắc: "không ai nên bị ép buộc phụ thuộc vào thứ họ không dùng." Mỗi interface nên phục vụ một role cụ thể, một use case cụ thể. Nếu bạn viết interface có chữ "and" trong tên (`EmailAndSmsService`), hãy tách nó ra. Nếu bạn thấy client implement interface và ném `NotImplementedError`, đó là lúc cần tách interface. Nguyên tắc thực hành: **Interface Discovery** — đừng thiết kế interface ngay từ đầu, hãy để interface "hiện ra" từ nhu cầu của client. Viết client code trước, xem client cần gì, rồi tách interface ra dựa trên nhu cầu đó. Điều này đảm bảo interface thực sự phục vụ client, không phải là sản phẩm của trí tưởng tượng về "sẽ cần sau này". Kết hợp ISP với Dependency Inversion Principle (DIP), bạn sẽ có một hệ thống với các dependency nhỏ, rõ ràng, dễ test, dễ thay thế — và quan trọng nhất: mỗi class chỉ phụ thuộc vào đúng những gì nó cần, không hơn, không kém.
