---
id: solid-ocp
title: O — Open/Closed Principle
sidebar_label: O — Open/Closed
sidebar_position: 27
---

# O — Open/Closed Principle

> *"Software entities (classes, modules, functions, etc.) should be open for extension, but closed for modification."* — **Bertrand Meyer, *Object-Oriented Software Construction*, 1988**

Open/Closed Principle (OCP) là nguyên lý đầu tiên được Bertrand Meyer giới thiệu trong cuốn sách kinh điển năm 1988, trước khi SOLID ra đời đến hơn một thập kỷ. Meyer định nghĩa: một thực thể phần mềm nên "mở cho việc mở rộng" (open for extension) — nghĩa là ta có thể thêm hành vi mới cho nó, nhưng "đóng cho việc sửa đổi" (closed for modification) — nghĩa là không cần phải sửa mã nguồn của nó. Đạt được điều này tưởng chừng mâu thuẫn: làm sao có thể thay đổi hành vi của một module mà không chạm vào code của nó? Câu trả lời nằm ở sức mạnh của **abstraction** và **polymorphism**: module cha định nghĩa các extension points (interface, abstract class), và module con cung cấp implementation cụ thể. Khi cần thêm hành vi mới, ta chỉ cần viết module con mới — không cần mở module cha ra để sửa.

## Bài toán chi tiết: Hệ thống tính phí vận chuyển thương mại điện tử

Một startup giao hàng xây dựng hệ thống tính phí vận chuyển. Ban đầu, họ chỉ hỗ trợ giao hàng nội thành Hà Nội với công thức đơn giản: `phí = khoảng cách × 5,000 VND/kg`. Tính năng này được implement trong class `ShippingCalculator` với một method `calculate(distance, weight)` trả về phí. Code chạy tốt, test pass, mọi thứ hoàn hảo. Sau 2 tháng, startup mở rộng ra các tỉnh, và mỗi tỉnh có cách tính phí khác nhau. Developer thêm tham số `city: str` và dùng `if/elif/else` để phân nhánh. Thêm 3 tháng nữa, có thêm dịch vụ giao hàng hỏa tốc (phí gấp đôi), giao hàng tiết kiệm (phí rẻ hơn 30%), giao hàng COD (phụ phí 1%), giao hàng quốc tế (thuế nhập khẩu + bảo hiểm). Mỗi lần thêm một loại hình vận chuyển mới, developer lại mở file `ShippingCalculator` và thêm một `elif`.

Sau 1 năm, `ShippingCalculator` trở thành một con quái vật với hơn 3000 dòng code, 40+ biến thể `if/elif`, và 15 tham số constructor. Bug xuất hiện thường xuyên: một lần sửa phí giao hàng nội thành vô tình làm thay đổi thuế nhập khẩu của giao hàng quốc tế vì cả hai dùng chung biến `base_rate` không được reset đúng cách. Một lần khác, thêm hình thức giao hàng "siêu tốc" cho TP. Hồ Chí Minh làm hỏng logic giao hàng tiết kiệm ở Đà Nẵng. Đội ngũ phát triển bắt đầu sợ chạm vào class này. Mỗi pull request liên quan đến tính phí vận chuyển đều phải qua 3 vòng review và mất 2 ngày để test. Đây là một case study kinh điển về vi phạm OCP: một module được thiết kế kém, buộc developer phải sửa đổi thay vì mở rộng.

Hậu quả nghiêm trọng hơn khi đội ngũ kỹ thuật muốn chạy A/B testing cho các công thức tính phí khác nhau. Với thiết kế hiện tại, họ không thể có hai implementation của `ShippingCalculator` cùng tồn tại trong một request — vì logic tính phí bị gắn cứng trong code. Họ cũng không thể unit test một cách độc lập từng phương thức vận chuyển, vì tất cả đều nằm trong một method duy nhất. Khi có bug trong logic giao hàng hỏa tốc, họ phải chạy toàn bộ test suite của `ShippingCalculator` (bao gồm test cho các phương thức khác), làm tăng thời gian feedback. Cuối cùng, startup phải dừng lại 3 tuần để viết lại toàn bộ module từ đầu, áp dụng OCP với Strategy Pattern.

## Phân tích vấn đề

Root cause của vấn đề là việc sử dụng **conditional branching dựa trên type codes** (string/enum) để quyết định hành vi. Cấu trúc `if/elif/else` là dấu hiệu kinh điển của code không tuân thủ OCP vì nó buộc bạn phải sửa module hiện tại mỗi khi thêm behavior mới. Cụ thể:

1. **Vi phạm Open/Closed**: `ShippingCalculator` không "closed for modification" — mỗi lần thêm phương thức vận chuyển mới, bạn phải mở nó ra và thêm code.
2. **Vi phạm Single Responsibility**: Một class vừa quản lý tất cả các công thức tính phí — khi sửa một công thức có thể ảnh hưởng đến các công thức khác vì dùng chung state hoặc do lỗi copy-paste.
3. **Tight coupling giữa business rules**: Các quy tắc tính phí vận chuyển không có ranh giới rõ ràng. Thay đổi một quy tắc có thể ảnh hưởng đến các quy tắc khác thông qua shared mutable state.
4. **Khó kiểm thử**: Để test một nhánh cụ thể, bạn phải setup tất cả các tham số điều khiển — và có thể vô tình chạy vào nhánh khác.
5. **Không mở rộng được bằng plugin**: Bạn không thể implement một module tính phí mới dưới dạng plugin — bạn chỉ có thể sửa code.

**Code smells** của vi phạm OCP: `if type == ...`, `if isinstance(...)`, switch-case dài; method có tham số string type để phân nhánh; class có tiền tố "Base" hoặc hậu tố "Type"; một method duy nhất chứa nhiều algorithm không liên quan; khó thêm behavior mới mà không ảnh hưởng đến behavior cũ.

## Giải pháp: Abstraction và Strategy Pattern

Giải pháp cốt lõi là thay thế conditional branching bằng **polymorphism**. Định nghĩa một interface (hoặc abstract class) chung cho tất cả các phương thức tính phí, và mỗi implementation là một class riêng. Module gọi (context) chỉ biết đến interface — không cần biết implementation cụ thể nào đang được dùng. Khi cần thêm phương thức vận chuyển mới, ta chỉ cần tạo class mới implements interface đó, mà không cần sửa bất kỳ code nào trong context hay trong các implementation khác.

Có hai cách tiếp cận phổ biến để tuân thủ OCP:
1. **Template Method Pattern**: Định nghĩa skeleton của thuật toán trong abstract class, các class con override các bước cụ thể.
2. **Strategy Pattern**: Định nghĩa family of algorithms, đóng gói mỗi algorithm thành một class riêng, và làm cho chúng interchangeable.

Trong ví dụ này, Strategy Pattern là lựa chọn tối ưu vì mỗi cách tính phí là một algorithm độc lập, không chia sẻ chung cấu trúc.

## Ví dụ code hoàn chỉnh

### VIOLATION — Vi phạm OCP

```python
# shipping_violation.py
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


class ShippingMethod(str, Enum):
    STANDARD = 'standard'
    EXPRESS = 'express'
    SAVER = 'saver'
    INTERNATIONAL = 'international'
    COD = 'cod'
    SAME_DAY = 'same_day'


@dataclass
class Parcel:
    weight_kg: Decimal
    distance_km: Decimal
    declared_value: Decimal
    is_fragile: bool = False
    requires_insurance: bool = False


class ShippingCalculator:
    """
    VIOLATION: Mỗi lần thêm phương thức vận chuyển mới, phải sửa class này.
    Hiện tại đã có 6 phương thức, nếu thêm 'drone_delivery' phải thêm elif nữa.
    """

    def __init__(self, base_rate: Decimal = Decimal('5000')) -> None:
        self._base_rate = base_rate
        self._service_fees: dict[str, Decimal] = {
            'express': Decimal('2.0'),
            'same_day': Decimal('3.0'),
            'international': Decimal('1.5'),
            'saver': Decimal('0.7'),
            'cod': Decimal('0.01'),  # 1% phí thu hộ
            'standard': Decimal('1.0'),
        }

    def calculate(self, parcel: Parcel, method: ShippingMethod) -> dict[str, Any]:
        weight = parcel.weight_kg
        distance = parcel.distance_km

        # Phí cơ bản = khoảng cách × trọng lượng × base_rate
        if method == ShippingMethod.STANDARD:
            fee = distance * weight * self._base_rate * self._service_fees['standard']
            return {'fee': fee, 'description': 'Giao hàng tiêu chuẩn (3-5 ngày)'}

        elif method == ShippingMethod.EXPRESS:
            fee = distance * weight * self._base_rate * self._service_fees['express']
            fee += Decimal('20000')  # Phụ phí hỏa tốc
            return {'fee': fee, 'description': 'Giao hàng hỏa tốc (1-2 ngày)'}

        elif method == ShippingMethod.SAVER:
            fee = distance * weight * self._base_rate * self._service_fees['saver']
            fee = max(fee, Decimal('15000'))  # Phí tối thiểu
            return {'fee': fee, 'description': 'Giao hàng tiết kiệm (5-7 ngày)'}

        elif method == ShippingMethod.INTERNATIONAL:
            duty = parcel.declared_value * Decimal('0.15')  # 15% thuế nhập khẩu
            insurance = Decimal('0')
            if parcel.requires_insurance:
                insurance = parcel.declared_value * Decimal('0.02')
            fee = distance * weight * self._base_rate * self._service_fees['international']
            fee += duty + insurance
            return {'fee': fee, 'description': 'Vận chuyển quốc tế', 'duty': duty, 'insurance': insurance}

        elif method == ShippingMethod.COD:
            fee = distance * weight * self._base_rate * self._service_fees['standard']
            cod_fee = parcel.declared_value * self._service_fees['cod']  # 1% phí thu hộ
            fee += cod_fee
            return {'fee': fee, 'description': 'Giao hàng COD', 'cod_fee': cod_fee}

        elif method == ShippingMethod.SAME_DAY:
            fee = distance * weight * self._base_rate * self._service_fees['same_day']
            fee += Decimal('30000')  # Phụ phí giao trong ngày
            if parcel.is_fragile:
                fee += Decimal('10000')  # Phụ phí hàng dễ vỡ
            return {'fee': fee, 'description': 'Giao hàng trong ngày'}

        else:
            raise ValueError(f'Unknown shipping method: {method}')
```

### REFACTORED — Tuân thủ OCP

```python
# ─── domain/parcel.py ───
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Parcel:
    weight_kg: Decimal
    distance_km: Decimal
    declared_value: Decimal = Decimal('0')
    is_fragile: bool = False
    requires_insurance: bool = False


# ─── shipping/shipping_fee.py ───
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol


class ShippingResult:
    def __init__(self, fee: Decimal, description: str, **extra: object) -> None:
        self.fee = fee
        self.description = description
        self.extra = extra

    def to_dict(self) -> dict[str, object]:
        return {
            'fee': self.fee,
            'description': self.description,
            **self.extra,
        }


class ShippingFeeStrategy(Protocol):
    """Interface cho tất cả các strategy tính phí vận chuyển."""

    def calculate(self, parcel: Parcel) -> ShippingResult:
        ...

    @property
    def method_name(self) -> str:
        ...


# ─── shipping/strategies.py ───
from __future__ import annotations
from decimal import Decimal

BASE_RATE = Decimal('5000')


class StandardShipping:
    method_name = 'standard'

    def calculate(self, parcel: Parcel) -> ShippingResult:
        fee = parcel.distance_km * parcel.weight_kg * BASE_RATE
        return ShippingResult(fee=fee, description='Giao hàng tiêu chuẩn (3-5 ngày)')


class ExpressShipping:
    method_name = 'express'

    def calculate(self, parcel: Parcel) -> ShippingResult:
        fee = parcel.distance_km * parcel.weight_kg * BASE_RATE * Decimal('2.0')
        fee += Decimal('20000')
        return ShippingResult(fee=fee, description='Giao hàng hỏa tốc (1-2 ngày)')


class SaverShipping:
    method_name = 'saver'

    def calculate(self, parcel: Parcel) -> ShippingResult:
        fee = parcel.distance_km * parcel.weight_kg * BASE_RATE * Decimal('0.7')
        fee = max(fee, Decimal('15000'))
        return ShippingResult(fee=fee, description='Giao hàng tiết kiệm (5-7 ngày)')


class InternationalShipping:
    method_name = 'international'

    DEFAULT_DUTY_RATE = Decimal('0.15')
    DEFAULT_INSURANCE_RATE = Decimal('0.02')

    def __init__(self, duty_rate: Decimal = DEFAULT_DUTY_RATE,
                 insurance_rate: Decimal = DEFAULT_INSURANCE_RATE) -> None:
        self._duty_rate = duty_rate
        self._insurance_rate = insurance_rate

    def calculate(self, parcel: Parcel) -> ShippingResult:
        duty = parcel.declared_value * self._duty_rate
        insurance = Decimal('0')
        if parcel.requires_insurance:
            insurance = parcel.declared_value * self._insurance_rate
        fee = parcel.distance_km * parcel.weight_kg * BASE_RATE * Decimal('1.5')
        fee += duty + insurance
        return ShippingResult(
            fee=fee, description='Vận chuyển quốc tế',
            duty=duty, insurance=insurance,
        )


class CODShipping:
    method_name = 'cod'

    def __init__(self, cod_rate: Decimal = Decimal('0.01')) -> None:
        self._cod_rate = cod_rate

    def calculate(self, parcel: Parcel) -> ShippingResult:
        standard_fee = parcel.distance_km * parcel.weight_kg * BASE_RATE
        cod_fee = parcel.declared_value * self._cod_rate
        fee = standard_fee + cod_fee
        return ShippingResult(fee=fee, description='Giao hàng COD', cod_fee=cod_fee)


class SameDayShipping:
    method_name = 'same_day'

    FRAGILE_SURCHARGE = Decimal('10000')
    EXPRESS_SURCHARGE = Decimal('30000')

    def calculate(self, parcel: Parcel) -> ShippingResult:
        fee = parcel.distance_km * parcel.weight_kg * BASE_RATE * Decimal('3.0')
        fee += self.EXPRESS_SURCHARGE
        if parcel.is_fragile:
            fee += self.FRAGILE_SURCHARGE
        return ShippingResult(fee=fee, description='Giao hàng trong ngày')


# ─── shipping/calculator.py ───
from __future__ import annotations
from typing import Optional


class ShippingCalculator:
    """
    OCP-compliant: MỞ cho mở rộng (thêm strategy), ĐÓNG cho sửa đổi (strategy cũ không đổi).
    """

    def __init__(self) -> None:
        self._strategies: dict[str, ShippingFeeStrategy] = {}

    def register(self, strategy: ShippingFeeStrategy) -> None:
        self._strategies[strategy.method_name] = strategy

    def calculate(self, parcel: Parcel, method: str) -> ShippingResult:
        strategy = self._strategies.get(method)
        if strategy is None:
            raise ValueError(f'Unknown shipping method: {method}')
        return strategy.calculate(parcel)


# ─── main.py ───
from __future__ import annotations

calculator = ShippingCalculator()
calculator.register(StandardShipping())
calculator.register(ExpressShipping())
calculator.register(SaverShipping())
calculator.register(InternationalShipping())
calculator.register(CODShipping())
calculator.register(SameDayShipping())

parcel = Parcel(weight_kg=Decimal('2.5'), distance_km=Decimal('15'),
                declared_value=Decimal('2000000'), is_fragile=True)

result = calculator.calculate(parcel, 'same_day')
print(result.to_dict())  # {'fee': ..., 'description': 'Giao hàng trong ngày', ...}


# ─── Thêm phương thức mới KHÔNG cần sửa code cũ ───
class DroneShipping:
    method_name = 'drone'

    def calculate(self, parcel: Parcel) -> ShippingResult:
        fee = parcel.distance_km * parcel.weight_kg * BASE_RATE * Decimal('5.0')
        fee += Decimal('50000')
        return ShippingResult(fee=fee, description='Giao hàng bằng drone (30 phút)')

calculator.register(DroneShipping())
result = calculator.calculate(parcel, 'drone')
print(result.to_dict())
```

## Dấu hiệu nhận biết vi phạm OCP

- **Conditional branching trên type codes**: Bất kỳ `if`, `elif`, `match/case` nào kiểm tra giá trị của một enum/string để quyết định logic xử lý. Đặc biệt là khi các nhánh này ngày càng nhiều theo thời gian.
- **Switch-case hay if-else chain dài**: Một method duy nhất có 5+ nhánh `elif`. Đây là dấu hiệu rõ ràng nhất.
- **Sử dụng isinstance() trong logic nghiệp vụ**: Khi bạn thấy code kiểu `if isinstance(obj, SomeClass)`, đó là dấu hiệu bạn nên dùng polymorphism thay vì type checking.
- **Class có quá nhiều dependency để phục vụ các use case khác nhau**: Một service được inject quá nhiều dependency khác nhau vì nó cố gắng làm quá nhiều việc.
- **Mỗi lần thêm tính năng mới phải sửa nhiều file**: Nếu bạn phải sửa 3-4 file mỗi khi thêm một business rule mới, rất có thể bạn đang vi phạm OCP.
- **Khó tạo unit test cho một behavior cụ thể**: Nếu bạn phải setup 5-6 biến điều kiện để test một nhánh duy nhất, method đó đang có vấn đề.

## Kiểm thử

```python
# test_shipping.py
from __future__ import annotations
from decimal import Decimal
import pytest  # type: ignore
from shipping.shipping_fee import ShippingFeeStrategy, ShippingResult
from shipping.strategies import (
    StandardShipping, ExpressShipping, SaverShipping,
    InternationalShipping, CODShipping, SameDayShipping, DroneShipping,
)
from shipping.calculator import ShippingCalculator
from domain.parcel import Parcel


@pytest.fixture
def standard_parcel() -> Parcel:
    return Parcel(weight_kg=Decimal('2.5'), distance_km=Decimal('10'))


@pytest.fixture
def fragile_parcel() -> Parcel:
    return Parcel(weight_kg=Decimal('1.0'), distance_km=Decimal('5'),
                  declared_value=Decimal('5000000'), is_fragile=True)


@pytest.fixture
def international_parcel() -> Parcel:
    return Parcel(weight_kg=Decimal('5.0'), distance_km=Decimal('10000'),
                  declared_value=Decimal('150000000'), requires_insurance=True)


class TestStandardShipping:
    def test_basic_fee(self, standard_parcel: Parcel) -> None:
        strategy = StandardShipping()
        result = strategy.calculate(standard_parcel)
        # 10 * 2.5 * 5000 = 125000
        assert result.fee == Decimal('125000')

    def test_zero_weight(self) -> None:
        parcel = Parcel(weight_kg=Decimal('0'), distance_km=Decimal('10'))
        strategy = StandardShipping()
        result = strategy.calculate(parcel)
        assert result.fee == Decimal('0')


class TestSaverShipping:
    def test_minimum_fee_applied(self) -> None:
        parcel = Parcel(weight_kg=Decimal('0.5'), distance_km=Decimal('2'))
        strategy = SaverShipping()
        result = strategy.calculate(parcel)
        # 2 * 0.5 * 5000 * 0.7 = 3500, but minimum is 15000
        assert result.fee == Decimal('15000')

    def test_normal_fee(self, standard_parcel: Parcel) -> None:
        strategy = SaverShipping()
        result = strategy.calculate(standard_parcel)
        # 10 * 2.5 * 5000 * 0.7 = 87500
        assert result.fee == Decimal('87500')


class TestInternationalShipping:
    def test_with_insurance(self, international_parcel: Parcel) -> None:
        strategy = InternationalShipping()
        result = strategy.calculate(international_parcel)
        # distance * weight * 5000 * 1.5 + duty(150M * 15%) + insurance(150M * 2%)
        expected_fee = Decimal('10000') * Decimal('5') * Decimal('5000') * Decimal('1.5')
        expected_fee += Decimal('150000000') * Decimal('0.15')
        expected_fee += Decimal('150000000') * Decimal('0.02')
        assert result.fee == expected_fee
        assert result.extra['duty'] == Decimal('22500000')
        assert result.extra['insurance'] == Decimal('3000000')

    def test_without_insurance(self) -> None:
        parcel = Parcel(weight_kg=Decimal('1'), distance_km=Decimal('1000'),
                        declared_value=Decimal('10000000'), requires_insurance=False)
        strategy = InternationalShipping()
        result = strategy.calculate(parcel)
        assert result.extra['insurance'] == Decimal('0')


class TestSameDayShipping:
    def test_fragile_surcharge(self, fragile_parcel: Parcel) -> None:
        strategy = SameDayShipping()
        result = strategy.calculate(fragile_parcel)
        # (5 * 1 * 5000 * 3.0) + 30000 + 10000 = 115000
        assert result.fee == Decimal('115000')

    def test_non_fragile(self, standard_parcel: Parcel) -> None:
        strategy = SameDayShipping()
        result = strategy.calculate(standard_parcel)
        # (10 * 2.5 * 5000 * 3.0) + 30000 = 405000
        assert result.fee == Decimal('405000')


class TestCalculator:
    def test_register_and_calculate(self, standard_parcel: Parcel) -> None:
        calculator = ShippingCalculator()
        calculator.register(StandardShipping())
        calculator.register(ExpressShipping())

        result = calculator.calculate(standard_parcel, 'standard')
        assert result.description == 'Giao hàng tiêu chuẩn (3-5 ngày)'

    def test_unknown_method(self, standard_parcel: Parcel) -> None:
        calculator = ShippingCalculator()
        with pytest.raises(ValueError, match='Unknown shipping method: teleport'):
            calculator.calculate(standard_parcel, 'teleport')

    def test_drone_shipping_new_strategy(self, standard_parcel: Parcel) -> None:
        """Kiểm tra tính mở rộng: thêm strategy mới không ảnh hưởng strategy cũ."""
        calculator = ShippingCalculator()
        calculator.register(StandardShipping())
        calculator.register(DroneShipping())

        result = calculator.calculate(standard_parcel, 'drone')
        # 10 * 2.5 * 5000 * 5.0 + 50000 = 675000
        assert result.fee == Decimal('675000')

        # Kiểm tra strategy cũ vẫn hoạt động
        result2 = calculator.calculate(standard_parcel, 'standard')
        assert result2.fee == Decimal('125000')
```

## Ứng dụng thực tế

1. **Django REST Framework — Permissions và Authentication**: DRF định nghĩa các base classes `BasePermission` và `BaseAuthentication`. Mỗi permission/authentication class là một strategy riêng. Khi cần thêm logic xác thực mới (ví dụ: JWT + OAuth2 + API Key), bạn chỉ cần subclass và override các method cần thiết. Framework sẽ gọi đúng class thông qua settings, không cần sửa DRF core.

2. **FastAPI — Dependency Injection với Depends()**: FastAPI cho phép định nghĩa các dependency function và kết hợp chúng linh hoạt. Mỗi `Depends()` là một extension point. Bạn có thể thêm authentication, database session, permission checking bằng cách tạo dependency mới — không cần sửa FastAPI core. Đây là ứng dụng tinh tế của OCP trong thư viện.

3. **SQLAlchemy — Dialect System**: SQLAlchemy hỗ trợ nhiều database backend (PostgreSQL, MySQL, SQLite, Oracle) thông qua hệ thống dialect plugins. Mỗi dialect là một module riêng implement cùng một interface. Để thêm hỗ trợ cho CockroachDB, bạn chỉ cần tạo dialect mới mà không cần sửa SQLAlchemy core. Đây là OCP ở cấp độ kiến trúc.

4. **Payment Gateway Integration**: Một hệ thống payment có thể tích hợp VNPay, MoMo, ZaloPay, Stripe, PayPal. Mỗi gateway là một strategy implement interface chung `PaymentGateway.process(amount)`. Khi có gateway mới, chỉ cần thêm file Python mới — không cần sửa bất kỳ code xử lý payment nào đã tồn tại.

5. **Plugin Architecture trong sản phẩm thực tế**: Một công ty SaaS Việt Nam đã thiết kế toàn bộ module báo cáo dạng plugin, mỗi loại báo cáo là một class implement interface `Report.generate(data)`. Họ có hơn 50 loại báo cáo. Khi khách hàng yêu cầu báo cáo tùy chỉnh, họ chỉ cần tạo file plugin mới, deploy mà không cần restart server. OCP giúp họ tăng gấp đôi năng suất phát triển tính năng tùy chỉnh.

## Liên hệ với Pattern

- **Strategy Pattern**: Đây là pattern trực tiếp nhất để implement OCP. Family of algorithms được đóng gói trong các strategy class riêng biệt, interchangeable thông qua interface chung.
- **Template Method Pattern**: Phù hợp khi các algorithm chia sẻ chung một cấu trúc (skeleton) nhưng khác nhau ở một số bước. Abstract class định nghĩa skeleton, subclass override các bước cụ thể.
- **Factory Method / Abstract Factory**: Dùng để tạo các đối tượng tuân thủ OCP. Factory quyết định concrete class nào được instantiate dựa trên context.
- **Decorator Pattern**: Mở rộng behavior của object mà không sửa class gốc. Mỗi decorator là một extension point.
- **Observer Pattern**: Cho phép thêm behavior mới (observers) mà không sửa subject. Khi subject thay đổi, tất cả observers được thông báo.
- **Plugin / Extension Point Architecture**: OCP ở cấp kiến trúc. Hệ thống định nghĩa các extension points (hooks, interfaces), và modules bên ngoài implement chúng.

## Ưu và nhược điểm

| Tiêu chí | Trước (vi phạm OCP) | Sau (tuân thủ OCP) |
|----------|---------------------|-------------------|
| **Cách thêm tính năng mới** | Sửa file cũ, thêm elif | Tạo file mới, không sửa file cũ |
| **Rủi ro khi thêm tính năng** | Cao — có thể làm hỏng logic cũ | Thấp — không ảnh hưởng code cũ |
| **Testability** | Khó — phải test cả method khổng lồ | Dễ — test từng strategy riêng biệt |
| **Code complexity** | Cao — một method 3000 dòng | Thấp — mỗi strategy 10-30 dòng |
| **Thời gian code review** | Lâu — phải hiểu toàn bộ method | Nhanh — mỗi strategy là một unit nhỏ |
| **Số lượng file** | 1 file chính + 0 file phụ | 1 interface + N strategy files |
| **Khả năng runtime flexibility** | Không — logic gắn cứng | Có — có thể switch strategy linh hoạt |
| **Plugability** | Không thể plugin hóa | Dễ dàng — mỗi strategy là một plugin |
| **Hỗ trợ A/B testing** | Không thể | Dễ dàng — thay strategy trong runtime |
| **Số lượng class tăng** | 0 (không đổi) | Nhiều (mỗi behavior một class mới) |
| **Chi phí setup ban đầu** | Thấp (if/elif nhanh hơn) | Cao hơn (phải thiết kế interface) |
| **Khi nào phù hợp** | Prototype, script một lần | Hệ thống production, cần bảo trì lâu dài |

## Kết luận

OCP là nguyên lý biến phần mềm từ một khối đá nguyên khối cứng nhắc thành một hệ thống linh hoạt, có thể mở rộng như những viên gạch Lego. Chìa khóa để đạt được OCP là **abstraction** — xác định đúng các "variation points" (điểm có thể thay đổi) và encapsulate chúng sau một interface. Một câu hỏi heuristic hữu ích khi thiết kế: "Nếu tuần sau có yêu cầu thêm một biến thể mới, tôi có cần sửa code hiện tại không?" Nếu câu trả lời là "có", hãy refactor ngay trước khi bạn có biến thể thứ hai. Nguyên tắc "Three Strikes And Refactor" — khi bạn có 3 biến thể tương tự nhau, đã đến lúc abstract hóa chúng thành interface + strategy. Trong thực tế, không cần OCP cho mọi thứ — hãy dành nó cho những phần có khả năng thay đổi cao. Nhưng một khi đã áp dụng, OCP sẽ trả công xứng đáng qua từng sprint, khi bạn thêm tính năng mới mà không cần động tay đến code đã chạy ổn định suốt nhiều tháng.
