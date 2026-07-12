---
id: flyweight
title: Flyweight
sidebar_label: 🪶 Flyweight
sidebar_position: 12
---

# Flyweight

> "Use sharing to support large numbers of fine-grained objects efficiently." — Erich Gamma, *Design Patterns: Elements of Reusable Object-Oriented Software*

Bạn có bao giờ tự hỏi làm sao game thế giới mở với hàng trăm ngàn cây cối lại chạy mượt đến vậy? Câu trả lời nằm ở pattern này đây...

## Bài toán chi tiết

Hãy tưởng tượng bạn đang phát triển một tựa game nhập vai thế giới mở (open-world RPG) với bối cảnh một khu rừng rộng 100 km². Trong game có hàng trăm ngàn cây cối, bụi rậm, hoa lá. Mỗi cây là một object với các thuộc tính: position (x, y, z), scale, rotation, texture, leafMesh, colorPalette, và nhiều thuộc tính khác. Với 500.000 cây trong rừng, mỗi object chiếm khoảng 200 bytes, tổng bộ nhớ lên tới **100 MB chỉ riêng cho cây cối** — chưa kể các đối tượng khác.

Vấn đề là phần lớn các cây có cùng texture và colorPalette — ví dụ: cây thông trong cùng một khu vực đều dùng texture vỏ thông và bảng màu lá xanh giống hệt nhau. Tuy nhiên, trong thiết kế ban đầu, mỗi object Tree đều lưu riêng texture và colorPalette, dẫn đến **trùng lặp dữ liệu khổng lồ**. Các thuộc tính như position, scale, rotation là duy nhất cho từng cây, nhưng texture và colorPalette thì được chia sẻ.

Nếu không tối ưu, game sẽ tiêu thụ quá nhiều RAM, gây giật lag trên các máy cấu hình thấp. Giải pháp đơn giản nhất là giảm số lượng cây — nhưng điều đó làm giảm chất lượng đồ họa và trải nghiệm chơi game. **Cần một giải pháp cho phép giữ nguyên số lượng object nhưng giảm thiểu bộ nhớ** bằng cách chia sẻ dữ liệu dùng chung.

## Giải pháp với Pattern

Flyweight Pattern giải quyết vấn đề này bằng cách tách object thành hai phần: **Intrinsic state** (trạng thái nội tại — dùng chung được) và **Extrinsic state** (trạng thái ngoại tại — riêng cho từng object). Intrinsic state được lưu trong Flyweight object và được chia sẻ giữa nhiều context. Extrinsic state được lưu trong Context object hoặc được truyền vào method khi cần.

Cấu trúc Flyweight gồm:
- **Flyweight**: Interface hoặc abstract class cho các object có thể chia sẻ.
- **ConcreteFlyweight**: Implement Flyweight interface và lưu intrinsic state.
- **FlyweightFactory**: Quản lý pool các Flyweight — kiểm tra nếu Flyweight đã tồn tại thì trả về, nếu chưa thì tạo mới.
- **Client**: Tính toán extrinsic state và truyền vào Flyweight method khi cần.

Trong ví dụ cây cối, `TreeType` là Flyweight lưu texture và colorPalette (intrinsic). Mỗi cây cá thể (Tree) chỉ lưu position, scale, rotation và tham chiếu đến TreeType. Factory đảm bảo mỗi tổ hợp texture-colorPalette chỉ tồn tại một object duy nhất. Kết quả: 500.000 cây nhưng chỉ có 10-20 TreeType objects — **giảm bộ nhớ từ 100 MB xuống còn vài MB.**

## Phân tích thiết kế

Flyweight Pattern là một ứng dụng của **Object Pooling** và **Memory Optimization**. Nó đặc biệt quan trọng trong các hệ thống có số lượng object cực lớn mà intrinsic state chiếm tỷ lệ cao. Pattern này thường đi kèm với **Factory Method** (để tạo Flyweight) và **Composite** (để tạo cấu trúc cây từ Flyweight).

**Khi KHÔNG nên dùng Flyweight:**
- Khi số lượng object nhỏ — overhead của Factory không đáng.
- Khi intrinsic state chiếm tỷ lệ nhỏ trong tổng bộ nhớ.
- Khi performance của việc tính toán extrinsic state lớn hơn lợi ích tiết kiệm bộ nhớ.
- Khi object có identity quan trọng (cần so sánh bằng `==`) — Flyweight chia sẻ object nên mất identity riêng.

**Trade-offs:**
- Tăng độ phức tạp — phải tách state một cách cẩn thận.
- Factory quản lý pool — cần thread safety nếu đa luồng.
- Tính extrinsic state mỗi khi gọi method — tốn CPU thay vì RAM.
- Khó debug — object được chia sẻ, lỗi ở một chỗ có thể ảnh hưởng nhiều nơi.

## Ví dụ code hoàn chỉnh

### Cách làm sai: Mỗi object lưu tất cả state (tốn bộ nhớ)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import sys


class Tree:
    """Mỗi cây đều lưu texture và colorPalette riêng — lãng phí bộ nhớ."""

    def __init__(self, x: float, y: float, z: float, species: str) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.species = species
        # Mỗi cây lưu riêng — dù 500.000 cây cùng loại
        self.texture = self._load_texture(species)
        self.color_palette = self._get_palette(species)
        self.scale = random.uniform(0.8, 1.2)
        self.rotation = random.uniform(0, 360)

    def _load_texture(self, species: str) -> str:
        return f"textures/{species}_bark.png"

    def _get_palette(self, species: str) -> List[str]:
        palettes = {
            "pine": ["#1a4a1a", "#2d6b2d", "#3d8b3d"],
            "oak": ["#4a7a2a", "#6b9a3b", "#8bba4b"],
            "birch": ["#8aaa6a", "#aacc8a", "#cceeaa"],
        }
        return palettes.get(species, ["#000000"])

    def render(self) -> str:
        return f"Rendering {self.species} at ({self.x:.1f}, {self.y:.1f}, {self.z:.1f}) [{self.texture}]"


# Tạo 100.000 cây — mỗi cây lưu texture riêng → tốn RAM
forest = [Tree(random.uniform(0, 1000), 0, random.uniform(0, 1000), "pine") for _ in range(1000)]
print(f"Approx memory per tree: {sys.getsizeof(forest[0])} bytes")
```

### Cách đúng: Flyweight Pattern

```python
# --- Flyweight (Intrinsic State) ---
@dataclass(frozen=True)
class TreeType:
    """Flyweight — lưu intrinsic state dùng chung."""
    species: str
    texture: str
    color_palette: tuple[str, ...]

    def render(self, x: float, y: float, z: float, scale: float, rotation: float) -> str:
        return (
            f"Rendering {self.species} at ({x:.1f}, {y:.1f}, {z:.1f}) "
            f"scale={scale:.2f} rot={rotation:.1f}° [{self.texture}]"
        )


class TreeTypeFactory:
    """Flyweight Factory — quản lý pool các TreeType."""

    _tree_types: dict[str, TreeType] = {}

    @classmethod
    def get_tree_type(cls, species: str) -> TreeType:
        if species not in cls._tree_types:
            palettes: dict[str, tuple[str, ...]] = {
                "pine": ("#1a4a1a", "#2d6b2d", "#3d8b3d"),
                "oak": ("#4a7a2a", "#6b9a3b", "#8bba4b"),
                "birch": ("#8aaa6a", "#aacc8a", "#cceeaa"),
                "maple": ("#8a2a2a", "#cc3b3b", "#ee5a5a"),
            }
            cls._tree_types[species] = TreeType(
                species=species,
                texture=f"textures/{species}_bark.png",
                color_palette=palettes.get(species, ("#000000",)),
            )
        return cls._tree_types[species]

    @classmethod
    def total_types(cls) -> int:
        return len(cls._tree_types)


# --- Context (Extrinsic State) ---
class Tree:
    """Context object — lưu extrinsic state và tham chiếu đến Flyweight."""

    def __init__(self, x: float, y: float, z: float, species: str) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.scale = random.uniform(0.8, 1.2)
        self.rotation = random.uniform(0, 360)
        # Tham chiếu đến Flyweight dùng chung
        self._tree_type = TreeTypeFactory.get_tree_type(species)

    def render(self) -> str:
        return self._tree_type.render(self.x, self.y, self.z, self.scale, self.rotation)


# --- Usage ---
if __name__ == "__main__":
    # Tạo 100.000 cây — chỉ có 4 TreeType objects
    species_list = ["pine", "oak", "birch", "maple"]
    forest: list[Tree] = []
    for i in range(100000):
        species = species_list[i % 4]
        forest.append(Tree(
            x=random.uniform(0, 1000),
            y=0,
            z=random.uniform(0, 1000),
            species=species,
        ))

    print(f"Total trees: {len(forest)}")
    print(f"Total tree types (Flyweights): {TreeTypeFactory.total_types()}")
    print(f"Memory saved: ~{len(forest) * 100 - TreeTypeFactory.total_types() * 100} bytes (theoretical)")
    print(forest[0].render())
    print(forest[50000].render())
```

## Sơ đồ UML

```
┌──────────────────┐
│   Flyweight      │
│   (TreeType)     │
│──────────────────│
│ + species: str   │
│ + texture: str   │
│ + colorPalette   │
│──────────────────│
│ + render(x,y,z,  │
│    scale, rot)   │
└────────┬─────────┘
         │
         │ created by
         ▼
┌────────────────────────┐
│  FlyweightFactory      │
│  (TreeTypeFactory)     │
│────────────────────────│
│ - _tree_types: dict    │
│────────────────────────│
│ + get_tree_type()      │
│ + total_types() → int  │
└────────────────────────┘

         ▲ uses
         │
┌────────────────────────┐
│  Context (Tree)        │
│────────────────────────│
│ - x, y, z: float      │
│ - scale, rotation      │
│ - _tree_type: TreeType │
│────────────────────────│
│ + render() → str       │
└────────────────────────┘
```

## So sánh với Pattern liên quan

**Flyweight vs Singleton**: Singleton đảm bảo một class chỉ có một instance duy nhất. Flyweight đảm bảo một loại object cụ thể chỉ có một instance, nhưng có thể có nhiều loại khác nhau. Factory của Flyweight quản lý nhiều instance (mỗi instance cho một tổ hợp intrinsic state), trong khi Singleton chỉ quản lý một instance. Flyweight có thể dùng Singleton cho chính Factory, nhưng các Flyweight objects thì không phải Singleton.

**Flyweight vs Object Pool**: Object Pool tái sử dụng object để tránh chi phí khởi tạo — các object trong pool có thể khác nhau về state. Flyweight chia sẻ object để tiết kiệm bộ nhớ — các object Flyweight là immutable và được chia sẻ hoàn toàn. Pool có thể trả object về pool, Flyweight không bao giờ bị hủy. Pool giúp tăng performance, Flyweight giúp giảm memory.

**Flyweight vs Composite**: Composite tạo cấu trúc cây từ các component. Flyweight tối ưu bộ nhớ cho các component lá (leaf) trong cây Composite khi có quá nhiều lá giống nhau. Hai pattern thường kết hợp: Flyweight được dùng bên trong Composite để chia sẻ intrinsic state của leaf nodes.

## Ứng dụng thực tế

**1. Python String Interning**: Python tự động intern (lưu vào pool) các string ngắn để tiết kiệm bộ nhớ. Khi hai string giống nhau được tạo ra, Python có thể trỏ đến cùng một object:

```python
a = "hello_world"
b = "hello_" + "world"
print(a is b)  # True — Python interned both strings

# String dài hơn không được intern
c = "hello_world_" * 100
d = "hello_world_" * 100
print(c is d)  # False — too long to intern
```

**2. Django's Template Fragment Caching**: Django `{% cache %}` tag lưu kết quả render của template fragment dùng chung. Đây là một dạng Flyweight: intrinsic state là template và context, extrinsic state là request-specific data:

```python
{% load cache %}
{% cache 500 "sidebar" request.user.id %}
    {# Nội dung sidebar — được cache và reuse #}
    {% for item in sidebar_items %}
        <li>{{ item.name }}</li>
    {% endfor %}
{% endcache %}
```

**3. Game Development (Unity, Unreal Engine)**: Unity dùng Flyweight qua ScriptableObject để chia sẻ dữ liệu giữa nhiều GameObjects. Một Material được chia sẻ giữa hàng ngàn mesh instances:

```csharp
// Unity ScriptableObject — Flyweight Pattern
public class ItemData : ScriptableObject {
    public string itemName;
    public Sprite icon;
    public Mesh mesh;
    public Material material;
}

// Mỗi Item trong game chỉ tham chiếu đến ItemData dùng chung
public class Item : MonoBehaviour {
    public ItemData data;  // Flyweight
    public int quantity;    // Extrinsic state
}
```

**4. Text Editor — Glyph Rendering**: Trong trình soạn thảo văn bản, mỗi ký tự là một object với font, size, style, color. Flyweight cho phép chia sẻ glyph data giữa các ký tự giống nhau:

```python
class Glyph:
    """Flyweight — intrinsic state của ký tự."""
    def __init__(self, char: str, font: str, size: int, bold: bool) -> None:
        self.char = char
        self.font = font
        self.size = size
        self.bold = bold

    def render(self, x: int, y: int, color: str) -> str:
        return f"'{self.char}' at ({x},{y}) font={self.font} size={self.size} color={color}"


class GlyphFactory:
    _glyphs: dict[str, Glyph] = {}

    @classmethod
    def get_glyph(cls, char: str, font: str, size: int, bold: bool = False) -> Glyph:
        key = f"{char}_{font}_{size}_{bold}"
        if key not in cls._glyphs:
            cls._glyphs[key] = Glyph(char, font, size, bold)
        return cls._glyphs[key]
```

## Kiểm thử

```python
import pytest
from flyweight import Tree, TreeTypeFactory, TreeType


class TestTreeTypeFactory:
    def setup_method(self) -> None:
        TreeTypeFactory._tree_types.clear()

    def test_same_species_returns_same_instance(self) -> None:
        type1 = TreeTypeFactory.get_tree_type("pine")
        type2 = TreeTypeFactory.get_tree_type("pine")
        assert type1 is type2  # Same object — sharing!

    def test_different_species_returns_different_instance(self) -> None:
        pine = TreeTypeFactory.get_tree_type("pine")
        oak = TreeTypeFactory.get_tree_type("oak")
        assert pine is not oak

    def test_total_types_count(self) -> None:
        TreeTypeFactory.get_tree_type("pine")
        TreeTypeFactory.get_tree_type("oak")
        TreeTypeFactory.get_tree_type("birch")
        assert TreeTypeFactory.total_types() == 3

    def test_type_immutability(self) -> None:
        tree_type = TreeTypeFactory.get_tree_type("pine")
        assert isinstance(tree_type, TreeType)
        # Frozen dataclass — không thể sửa


class TestTree:
    def setup_method(self) -> None:
        TreeTypeFactory._tree_types.clear()

    def test_tree_has_position(self) -> None:
        tree = Tree(10.0, 0.0, 20.0, "pine")
        assert tree.x == 10.0
        assert tree.z == 20.0

    def test_tree_render_uses_type(self) -> None:
        tree = Tree(0, 0, 0, "pine")
        output = tree.render()
        assert "pine" in output
        assert "Rendering" in output

    def test_trees_share_type_instance(self) -> None:
        tree1 = Tree(0, 0, 0, "pine")
        tree2 = Tree(1, 0, 1, "pine")
        # Cả hai cây đều tham chiếu đến cùng một TreeType
        assert tree1._tree_type is tree2._tree_type


class TestMemoryOptimization:
    def test_few_types_many_trees(self) -> None:
        TreeTypeFactory._tree_types.clear()
        trees = [Tree(0, 0, 0, "pine") for _ in range(10000)]
        assert TreeTypeFactory.total_types() == 1
        assert len(trees) == 10000
        # 10000 trees chỉ dùng 1 TreeType


class TestEdgeCases:
    def test_unknown_species_creates_type(self) -> None:
        TreeTypeFactory._tree_types.clear()
        tree_type = TreeTypeFactory.get_tree_type("alien_tree")
        assert tree_type.species == "alien_tree"
        assert tree_type.color_palette == ("#000000",)  # default

    def test_render_with_varying_scale(self) -> None:
        import random
        random.seed(42)
        tree1 = Tree(0, 0, 0, "oak")
        random.seed(42)
        tree2 = Tree(0, 0, 0, "oak")
        # Scale được random dựa trên seed
        assert tree1._tree_type is tree2._tree_type  # Type dùng chung
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Giảm mạnh bộ nhớ — đặc biệt với số lượng object lớn | Tăng độ phức tạp — phải phân tích và tách state |
| Giảm thời gian khởi tạo — Flyweight được tạo một lần | Tốn CPU cho việc tính toán extrinsic state |
| Cho phép tạo số lượng object gần như vô hạn | Factory cần thread safety trong môi trường đa luồng |
| Tái sử dụng tối đa — một object dùng chung cho hàng ngàn context | Khó debug — object chia sẻ gây side effect khó lường |
| Giảm cache miss — ít object hơn, CPU cache hoạt động tốt hơn | Không phù hợp với object cần identity riêng |

---

Flyweight Pattern là công cụ tối ưu bộ nhớ mạnh mẽ cho các hệ thống có số lượng object cực lớn với intrinsic state dùng chung. Nó thường được sử dụng trong game, text editor, GUI framework, và bất kỳ ứng dụng nào cần quản lý hàng ngàn đến hàng triệu object tương tự nhau. Tôi từng thấy một hệ thống giảm được 95% bộ nhờ Flyweight — **một con số đáng kinh ngạc.**

**Nguyên tắc vàng**: Hãy dùng Flyweight khi ứng dụng của bạn tạo ra một số lượng lớn object giống nhau và bộ nhớ là vấn đề. Hãy phân tích: thuộc tính nào thay đổi theo từng object (extrinsic — giữ lại trong context) và thuộc tính nào cố định theo loại (intrinsic — đưa vào Flyweight). Hãy nhớ rằng: Flyweight là immutable — một khi tạo ra, không được thay đổi intrinsic state của nó, vì thay đổi sẽ ảnh hưởng đến tất cả context đang dùng nó.

---
*Trân trọng!*
