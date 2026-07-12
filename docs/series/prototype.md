---
id: prototype
title: Prototype
sidebar_label: 🧬 Prototype
sidebar_position: 6
---

# Prototype

> *"Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype."* — Gang of Four, *Design Patterns: Elements of Reusable Object-Oriented Software*, 1994.

Có bao giờ bạn cần tạo 100 object giống hệt nhau, và mỗi lần tạo là một lần load dữ liệu từ disk, parse config, khởi tạo kết nối... chậm như rùa bò? Tôi biết cảm giác đó — nó như kiểu mỗi lần muốn uống cà phê, bạn lại phải đi trồng cây cà phê vậy.

**Prototype** thuộc nhóm **Creational Patterns**, tạo object mới bằng cách **clone** một object hiện có (prototype) thay vì gọi constructor. Điểm mạnh cốt lõi của Prototype nằm ở chỗ nó cho phép tạo object mà **không cần biết class cụ thể** — chỉ cần biết interface của prototype. Điều này đặc biệt hữu ích khi việc khởi tạo object từ constructor quá tốn kém hoặc phức tạp.

## Bài toán chi tiết

Bạn đang phát triển một **hệ thống game engine** cho một studio game lớn. Game của bạn có hàng trăm loại quái vật (enemies) khác nhau: Goblin, Orc, Dragon, Skeleton, Slime, v.v. Mỗi loại quái vật có:
- **Thuộc tính cơ bản**: HP, MP, attack, defense, speed.
- **Kỹ năng**: skills, spells, special abilities.
- **Trang bị**: weapon, armor, accessories.
- **Trạng thái**: buffs, debuffs, effects.
- **Hoạt ảnh**: animation frames, textures, particle effects.
- **AI behavior**: behavior tree, state machine, pathfinding config.

Mỗi quái vật khi được spawn (xuất hiện) trong game cần có toàn bộ các thuộc tính này. Nếu khởi tạo từ constructor, mỗi lần spawn phải:
1. Load texture từ disk (I/O tốn kém).
2. Parse JSON config cho stats và skills.
3. Khởi tạo behavior tree.
4. Load animation frames.
5. Thiết lập equipment mặc định.

Quá trình này mất 50-200ms mỗi lần spawn. Với game, khi người chơi vào dungeon, hệ thống có thể spawn 50-100 quái vật cùng lúc — nghĩa là 5-20 giây loading. **Không thể chấp nhận được.**

Cách tiếp cận ngây thơ: dùng constructor với tham số mặc định:

```python
orc = Enemy(
    name="Orc Warrior",
    hp=200, mp=50, attack=35, defense=25, speed=10,
    skills=["Bash", "Roar"],
    weapon="Axe", armor="Plate",
    texture_path="enemies/orc/warrior.png",
    behavior_tree="aggressive_melee.json",
    # ... 20+ tham số nữa
)
```

Vấn đề:
1. **Performance**: Load texture + parse config mỗi lần spawn — I/O bottleneck.
2. **Complexity**: Constructor với 20+ tham số, khó đọc, khó maintain.
3. **Duplication**: Nếu spawn 20 con Orc giống nhau, phải truyền lặp lại 20 lần.
4. **Variation**: Spawn Orc "đột biến" (mạnh hơn 20%) — phải copy manual từng thuộc tính.

## Giải pháp với Pattern

Prototype giải quyết bằng cách: tạo **một** instance "master" cho mỗi loại (prototype), sau đó clone nó mỗi khi cần spawn:

1. **Prototype interface**: Định nghĩa method `clone()`.
2. **Concrete Prototypes**: Mỗi loại enemy implement `clone()`.
3. **Prototype Registry**: Dictionary lưu các prototype master — lấy ra và clone.

Với Prototype:
- **Performance**: Clone trong memory (microseconds) thay vì load từ disk (milliseconds) — **nhanh hơn 1000x**.
- **Simplicity**: Clone giữ nguyên toàn bộ state — không cần truyền tham số.
- **Variation**: Clone rồi modify — spawn "Elite Orc" bằng cách clone Orc thường rồi tăng stats.
- **Dynamic types**: Có thể clone mà không biết class cụ thể — chỉ biết interface.

## Phân tích thiết kế

**OOP Principles áp dụng:**

- **Creational flexibility**: Tạo object mà không phụ thuộc vào class cụ thể.
- **Polymorphism**: Prototype Registry lưu trữ object bằng interface, không phải class.
- **Composition over inheritance**: Clone giữ nguyên toàn bộ composition tree.

**Shallow vs Deep Copy:**

- **Shallow copy** (`copy.copy`): Chỉ copy object gốc, nested objects vẫn là reference. Nhanh, rẻ, nhưng nguy hiểm — thay đổi nested object ảnh hưởng đến prototype gốc.
- **Deep copy** (`copy.deepcopy`): Copy toàn bộ object graph. An toàn, nhưng chậm hơn và không handle circular references tốt.

**Khi nào cần implement `clone()` thủ công thay vì dùng `copy.deepcopy`?**
- Khi object chứa resources không copy được (file handle, database connection, GPU texture).
- Khi cần custom clone logic (ví dụ: reset state, sinh ID mới).
- Khi object graph quá lớn, deep copy quá chậm.

**Khi nào KHÔNG nên dùng Prototype:**

- Khi object đơn giản, constructor không tốn kém.
- Khi object chứa circular references phức tạp.
- Khi không cần tạo nhiều biến thể từ một object gốc.
- Khi memory là vấn đề (giữ prototype master tốn memory).

## Ví dụ code hoàn chỉnh

### Cách làm sai (Constructor Hell)

```python
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Skill:
    name: str
    damage: int
    cooldown: float
    mana_cost: int


@dataclass
class AnimationData:
    idle_frames: list[str]
    attack_frames: list[str]
    hit_frames: list[str]
    death_frames: list[str]


class Enemy:
    """Cách sai: constructor tốn kém, khó dùng, chậm."""

    def __init__(
        self,
        name: str,
        enemy_type: str,
        hp: int,
        mp: int,
        attack: int,
        defense: int,
        speed: int,
        skills: list[Skill],
        weapon: str,
        armor: str,
        texture_path: str,
        animation: AnimationData,
        behavior_tree: str,
        team: str = "monster",
        level: int = 1,
        is_elite: bool = False,
    ) -> None:
        self.name = name
        self.enemy_type = enemy_type
        self.hp = hp
        self.max_hp = hp
        self.mp = mp
        self.max_mp = mp
        self.attack = attack
        self.defense = defense
        self.speed = speed
        self.skills = skills
        self.weapon = weapon
        self.armor = armor
        self.texture_path = texture_path
        self.animation = animation
        self.behavior_tree = behavior_tree
        self.team = team
        self.level = level
        self.is_elite = is_elite
        self.id = id(self)
        print(f"[SLOW] Enemy {name}: loading textures, parsing JSON...")
        time.sleep(0.05)  # Giả lập load tốn kém

    def __repr__(self) -> str:
        return f"Enemy(id={self.id}, name='{self.name}', type='{self.enemy_type}', level={self.level})"


# Tạo prototype mẫu
idle_frames = [f"orc_idle_{i}.png" for i in range(10)]
attack_frames = [f"orc_attack_{i}.png" for i in range(8)]
hit_frames = [f"orc_hit_{i}.png" for i in range(4)]
death_frames = [f"orc_death_{i}.png" for i in range(6)]

start = time.time()
orc_proto = Enemy(
    name="Orc Warrior",
    enemy_type="orc",
    hp=200, mp=50, attack=35, defense=25, speed=10,
    skills=[Skill("Bash", 40, 3.0, 10), Skill("Roar", 0, 8.0, 20)],
    weapon="Battle Axe", armor="Plate Armor",
    texture_path="enemies/orc/warrior.png",
    animation=AnimationData(idle_frames, attack_frames, hit_frames, death_frames),
    behavior_tree="aggressive_melee.json",
)
print(f"Time to create prototype: {(time.time() - start) * 1000:.1f}ms")

# Spawn lần 2 — lại tốn nguyên thời gian
start = time.time()
orc2 = Enemy(
    name="Orc Warrior",
    enemy_type="orc",
    hp=200, mp=50, attack=35, defense=25, speed=10,
    skills=[Skill("Bash", 40, 3.0, 10), Skill("Roar", 0, 8.0, 20)],
    weapon="Battle Axe", armor="Plate Armor",
    texture_path="enemies/orc/warrior.png",
    animation=AnimationData(idle_frames, attack_frames, hit_frames, death_frames),
    behavior_tree="aggressive_melee.json",
)
print(f"Time to spawn second enemy (no cache): {(time.time() - start) * 1000:.1f}ms")
```

### Refactored với Prototype Pattern

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Self
import copy
import time
import uuid


# ============== DOMAIN MODELS ==============

@dataclass(frozen=True)
class Skill:
    name: str
    damage: int
    cooldown: float
    mana_cost: int
    description: str = ""


@dataclass(frozen=True)
class AnimationData:
    idle_frames: list[str]
    attack_frames: list[str]
    hit_frames: list[str]
    death_frames: list[str]
    scale: float = 1.0

    def __deepcopy__(self, memo: dict) -> "AnimationData":
        return AnimationData(
            idle_frames=copy.deepcopy(self.idle_frames, memo),
            attack_frames=copy.deepcopy(self.attack_frames, memo),
            hit_frames=copy.deepcopy(self.hit_frames, memo),
            death_frames=copy.deepcopy(self.death_frames, memo),
            scale=self.scale,
        )


# ============== PROTOTYPE INTERFACE ==============

class Prototype(ABC):
    """Interface cho tất cả các object có thể clone."""

    @abstractmethod
    def clone(self, **kwargs) -> Self: ...

    @abstractmethod
    def reset(self) -> None: ...


# ============== CONCRETE PROTOTYPE ==============

class Enemy(Prototype):
    """Concrete Prototype — quái vật trong game."""

    def __init__(
        self,
        name: str = "",
        enemy_type: str = "",
        hp: int = 100,
        mp: int = 50,
        attack: int = 10,
        defense: int = 5,
        speed: int = 10,
        skills: Optional[list[Skill]] = None,
        weapon: str = "fist",
        armor: str = "cloth",
        texture_path: str = "",
        animation: Optional[AnimationData] = None,
        behavior_tree: str = "default.json",
        team: str = "monster",
        level: int = 1,
        is_elite: bool = False,
        entity_id: Optional[str] = None,
    ) -> None:
        self.entity_id = entity_id or str(uuid.uuid4())[:8]
        self.name = name
        self.enemy_type = enemy_type
        self.hp = hp
        self.max_hp = hp
        self.mp = mp
        self.max_mp = mp
        self.attack = attack
        self.defense = defense
        self.speed = speed
        self.skills = skills or []
        self.weapon = weapon
        self.armor = armor
        self.texture_path = texture_path
        self.animation = animation
        self.behavior_tree = behavior_tree
        self.team = team
        self.level = level
        self.is_elite = is_elite
        self.buffs: list[dict] = []
        self.debuffs: list[dict] = []
        self.current_target: Optional[str] = None
        self._initialized = True

    def clone(self, **overrides: Any) -> "Enemy":
        """Clone enemy với khả năng override thuộc tính."""
        cloned = copy.deepcopy(self)
        # Reset state cụ thể cho instance mới
        cloned.entity_id = str(uuid.uuid4())[:8]
        cloned.hp = cloned.max_hp
        cloned.mp = cloned.max_mp
        cloned.buffs = []
        cloned.debuffs = []
        cloned.current_target = None
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(cloned, key):
                setattr(cloned, key, value)
        return cloned

    def reset(self) -> None:
        """Reset về trạng thái ban đầu."""
        self.hp = self.max_hp
        self.mp = self.max_mp
        self.buffs = []
        self.debuffs = []
        self.current_target = None

    def take_damage(self, amount: int) -> int:
        """Nhận sát thương, trả về damage thực tế."""
        actual = max(0, amount - self.defense)
        self.hp = max(0, self.hp - actual)
        return actual

    def is_alive(self) -> bool:
        return self.hp > 0

    def scale_to_level(self, level: int) -> "Enemy":
        """Tạo bản sao với level khác (scale stats)."""
        scale_factor = 1 + (level - self.level) * 0.15
        return self.clone(
            level=level,
            hp=int(self.max_hp * scale_factor),
            max_hp=int(self.max_hp * scale_factor),
            attack=int(self.attack * scale_factor),
            defense=int(self.defense * scale_factor),
            speed=int(self.speed * scale_factor),
        )

    def to_elite(self) -> "Enemy":
        """Tạo bản sao Elite (mạnh hơn, có prefix)."""
        elite = self.clone(
            name=f"Elite {self.name}",
            is_elite=True,
            hp=int(self.max_hp * 3),
            max_hp=int(self.max_hp * 3),
            attack=int(self.attack * 1.5),
            defense=int(self.defense * 2),
            speed=int(self.speed * 1.2),
        )
        # Thêm skill đặc biệt cho elite
        elite.skills.append(Skill("Elite Strike", 80, 5.0, 30))
        return elite

    def __repr__(self) -> str:
        elite_tag = " [ELITE]" if self.is_elite else ""
        return f"Enemy(id={self.entity_id}, name='{self.name}', type='{self.enemy_type}', level={self.level}{elite_tag})"


# ============== PROTOTYPE REGISTRY ==============

class PrototypeRegistry:
    """Registry lưu trữ prototype master và clone khi cần."""

    def __init__(self) -> None:
        self._prototypes: dict[str, Prototype] = {}

    def register(self, key: str, prototype: Prototype) -> None:
        """Đăng ký prototype."""
        self._prototypes[key] = prototype

    def unregister(self, key: str) -> None:
        """Hủy đăng ký prototype."""
        self._prototypes.pop(key, None)

    def spawn(self, key: str, **overrides: Any) -> Prototype:
        """Clone prototype từ registry."""
        proto = self._prototypes.get(key)
        if proto is None:
            raise KeyError(f"Prototype '{key}' không tồn tại")
        return proto.clone(**overrides)

    def spawn_batch(self, key: str, count: int, **overrides: Any) -> list[Prototype]:
        """Spawn nhiều bản sao cùng lúc."""
        return [self.spawn(key, **overrides) for _ in range(count)]

    def list_prototypes(self) -> list[str]:
        return list(self._prototypes.keys())


# ============== GAME ENGINE USAGE ==============

class EnemySpawner:
    """Spawner — dùng Prototype pattern để spawn enemy hiệu quả."""

    def __init__(self, registry: PrototypeRegistry) -> None:
        self.registry = registry
        self._spawn_count = 0

    def spawn_goblin(self, level: int = 1) -> Enemy:
        self._spawn_count += 1
        return self.registry.spawn("goblin", level=level)

    def spawn_orc(self, level: int = 1, elite: bool = False) -> Enemy:
        self._spawn_count += 1
        enemy = self.registry.spawn("orc", level=level)
        if elite:
            enemy_elite = enemy.to_elite()
            enemy_elite.level = level
            return enemy_elite
        return enemy

    def spawn_dragon(self, level: int = 10) -> Enemy:
        self._spawn_count += 1
        return self.registry.spawn("dragon", level=level)

    def spawn_wave(self, enemies: list[tuple[str, int]]) -> list[Enemy]:
        """Spawn một wave với nhiều loại enemy."""
        result = []
        for enemy_type, count in enemies:
            for _ in range(count):
                enemy = self.spawn_goblin() if enemy_type == "goblin" else \
                        self.spawn_orc() if enemy_type == "orc" else \
                        self.spawn_dragon()
                result.append(enemy)
        self._spawn_count += len(result)
        return result

    @property
    def total_spawned(self) -> int:
        return self._spawn_count


# ========== SỬ DỤNG THỰC TẾ ==========

if __name__ == "__main__":
    # --- Khởi tạo animation data (tốn kém — chỉ làm 1 lần) ---
    print("=== BUILDING PROTOTYPES ===")
    start = time.time()

    goblin_anim = AnimationData(
        idle_frames=[f"goblin_idle_{i}.png" for i in range(8)],
        attack_frames=[f"goblin_attack_{i}.png" for i in range(6)],
        hit_frames=[f"goblin_hit_{i}.png" for i in range(3)],
        death_frames=[f"goblin_death_{i}.png" for i in range(5)],
    )
    goblin_proto = Enemy(
        name="Goblin Scout",
        enemy_type="goblin",
        hp=80, mp=20, attack=15, defense=8, speed=14,
        skills=[Skill("Stab", 20, 2.0, 5)],
        weapon="Dagger", armor="Leather",
        texture_path="enemies/goblin/scout.png",
        animation=goblin_anim,
        behavior_tree="coward_melee.json",
        level=1,
    )

    orc_anim = AnimationData(
        idle_frames=[f"orc_idle_{i}.png" for i in range(10)],
        attack_frames=[f"orc_attack_{i}.png" for i in range(8)],
        hit_frames=[f"orc_hit_{i}.png" for i in range(4)],
        death_frames=[f"orc_death_{i}.png" for i in range(6)],
    )
    orc_proto = Enemy(
        name="Orc Warrior",
        enemy_type="orc",
        hp=200, mp=50, attack=35, defense=25, speed=10,
        skills=[Skill("Bash", 40, 3.0, 10), Skill("Roar", 0, 8.0, 20)],
        weapon="Battle Axe", armor="Plate Armor",
        texture_path="enemies/orc/warrior.png",
        animation=orc_anim,
        behavior_tree="aggressive_melee.json",
        level=3,
    )

    dragon_anim = AnimationData(
        idle_frames=[f"dragon_idle_{i}.png" for i in range(15)],
        attack_frames=[f"dragon_attack_{i}.png" for i in range(12)],
        hit_frames=[f"dragon_hit_{i}.png" for i in range(6)],
        death_frames=[f"dragon_death_{i}.png" for i in range(10)],
    )
    dragon_proto = Enemy(
        name="Fire Dragon",
        enemy_type="dragon",
        hp=2000, mp=500, attack=120, defense=80, speed=5,
        skills=[
            Skill("Fire Breath", 150, 6.0, 50),
            Skill("Tail Swipe", 80, 3.0, 20),
            Skill("Fear Roar", 0, 10.0, 30),
        ],
        weapon="Claws", armor="Dragon Scales",
        texture_path="enemies/dragon/fire.png",
        animation=dragon_anim,
        behavior_tree="boss_ai.json",
        level=15,
        is_elite=True,
    )

    # Đăng ký prototype
    registry = PrototypeRegistry()
    registry.register("goblin", goblin_proto)
    registry.register("orc", orc_proto)
    registry.register("dragon", dragon_proto)

    build_time = (time.time() - start) * 1000
    print(f"Build prototypes time: {build_time:.1f}ms")
    print(f"Prototypes registered: {registry.list_prototypes()}")

    # --- SPAWN ENEMIES NHANH HON 1000x ---
    print("\n=== SPAWNING ENEMIES ===")
    spawner = EnemySpawner(registry)

    start = time.time()
    for i in range(50):
        goblin = spawner.spawn_goblin(level=1)
        goblin.take_damage(5)  # Mỗi con nhận damage khác nhau
    spawn_time = (time.time() - start) * 1000
    print(f"Spawn 50 goblins: {spawn_time:.1f}ms (avg: {spawn_time/50:.3f}ms each)")

    start = time.time()
    for i in range(20):
        orc = spawner.spawn_orc(level=3)
    spawn_time = (time.time() - start) * 1000
    print(f"Spawn 20 orcs: {spawn_time:.1f}ms (avg: {spawn_time/20:.3f}ms each)")

    # Spawn Elite Orc
    start = time.time()
    elite_orc = spawner.spawn_orc(level=5, elite=True)
    spawn_time = (time.time() - start) * 1000
    print(f"Spawn 1 elite orc: {spawn_time:.3f}ms")
    print(f"  {elite_orc}")
    print(f"  HP: {elite_orc.hp}/{elite_orc.max_hp}, Attack: {elite_orc.attack}")
    print(f"  Skills: {[s.name for s in elite_orc.skills]}")

    # Spawn Dragon (boss)
    dragon = spawner.spawn_dragon(level=15)
    print(f"\nDragon: {dragon}")

    # Clone và modify — spawn "Goblin King"
    goblin_king = goblin_proto.clone(
        name="Goblin King",
        hp=500, max_hp=500,
        attack=40,
        is_elite=True,
        team="boss",
    )
    print(f"\nCustom: {goblin_king}")
    print(f"  HP: {goblin_king.hp}/{goblin_king.max_hp}")

    # Scale to level
    high_level_orc = orc_proto.scale_to_level(20)
    print(f"\nScaled Orc (level 20): {high_level_orc}")
    print(f"  HP: {high_level_orc.hp}, Attack: {high_level_orc.attack}")

    print(f"\nTotal spawned: {spawner.total_spawned}")
    print(f"Prototypes preserved: {id(goblin_proto)} vs spawned: {id(spawner.spawn_goblin())}")
```

## So do UML

```
+-----------------------------------------------+
|              «interface»                       |
|               Prototype                        |
+-----------------------------------------------+
| + clone(**kwargs) -> Self                     |
| + reset()                                      |
+-----------------------------------------------+
          ^
          |
+---------+--------------------+
|           Enemy              |
+-------------------------------+
| - entity_id: str             |
| - name: str                  |
| - enemy_type: str            |
| - hp, max_hp: int            |
| - mp, max_mp: int            |
| - attack, defense: int       |
| - speed: int                 |
| - skills: list[Skill]        |
| - weapon, armor: str         |
| - texture_path: str          |
| - animation: AnimationData   |
| - behavior_tree: str         |
| - buffs, debuffs: list       |
+-------------------------------+
| + clone(**kwargs) -> Enemy   |
| + reset()                    |
| + take_damage(amount) -> int |
| + is_alive() -> bool         |
| + scale_to_level(lvl) -> En |
| + to_elite() -> Enemy        |
+-------------------------------+
          ^
          |  co the co nhieu ConcretePrototype
          |
+---------+---------+---------+
|  Goblin  |   Orc   |  Dragon |  ...
+---------+---------+---------+

+-----------------------------------------------+
|            PrototypeRegistry                   |
+-----------------------------------------------+
| - _prototypes: dict[str, Prototype]           |
+-----------------------------------------------+
| + register(key, prototype)                    |
| + unregister(key)                             |
| + spawn(key, **kwargs) -> Prototype           |
| + spawn_batch(key, count) -> list[Prototype]  |
| + list_prototypes() -> list[str]              |
+-----------------------------------------------+

+-----------------------------------------------+
|              EnemySpawner (Client)             |
+-----------------------------------------------+
| - registry: PrototypeRegistry                 |
| - _spawn_count: int                           |
+-----------------------------------------------+
| + spawn_goblin(level) -> Enemy                |
| + spawn_orc(level, elite) -> Enemy            |
| + spawn_dragon(level) -> Enemy                |
| + spawn_wave(list) -> list[Enemy]             |
+-----------------------------------------------+
```

## So sanh voi Pattern lien quan

| Pattern | Diem giong | Diem khac biet chinh |
|---------|-----------|---------------------|
| **Factory Method** | Deu tao object ma khong can biet class cu the | Factory Method tao object *tu dau* (goi constructor). Prototype tao object *tu co san* (clone). Prototype nhanh hon nhung ton bo nho cho prototype master. |
| **Abstract Factory** | Deu co the tao nhieu loai object | Abstract Factory tao object qua factory interface. Prototype tao object qua clone. Ket hop: Abstract Factory co the chua Prototype Registry ben trong. |
| **Builder** | Deu tao object phuc tap | Builder xay dung object *tung buoc*. Prototype clone nguyen khoi. Builder cho phep tuy chinh tung phan. Prototype nhanh hon nhung it linh hoat hon. |

## Ung dung thuc te

### 1. Python `copy` module — copy standard

```python
import copy

# Shallow copy
original = {"nested": [1, 2, 3], "value": 42}
shallow = copy.copy(original)
shallow["nested"].append(4)
print(original["nested"])  # [1, 2, 3, 4] — bi anh huong!

# Deep copy
deep = copy.deepcopy(original)
deep["nested"].append(5)
print(original["nested"])  # [1, 2, 3, 4] — khong bi anh huong
```

### 2. Django Model — clone instance

```python
from django.db import models

class EnemyTemplate(models.Model):
    """Prototype trong Django — ke thua template de tao instance."""
    name = models.CharField(max_length=100)
    hp = models.IntegerField(default=100)
    attack = models.IntegerField(default=10)
    defense = models.IntegerField(default=5)
    skills = models.JSONField(default=list)
    texture_path = models.CharField(max_length=255)
    behavior_tree = models.CharField(max_length=255)

    def spawn(self, level: int = 1) -> dict:
        """Clone template de tao enemy instance."""
        scale = 1 + (level - 1) * 0.15
        return {
            "name": self.name,
            "hp": int(self.hp * scale),
            "attack": int(self.attack * scale),
            "defense": int(self.defense * scale),
            "skills": self.skills,
            "texture": self.texture_path,
        }

# Su dung
goblin_template = EnemyTemplate.objects.get(name="Goblin Scout")
for i in range(10):
    enemy_data = goblin_template.spawn(level=i + 1)
    # Tao enemy tu data
```

### 3. Unity Game Engine — ScriptableObject

Unity dung ScriptableObject nhu Prototype:

```csharp
// C# — Unity ScriptableObject lam Prototype
[CreateAssetMenu(fileName = "EnemyData", menuName = "Enemy/Prototype")]
public class EnemyPrototype : ScriptableObject
{
    public string enemyName;
    public int hp;
    public int attack;
    public int defense;
    public GameObject prefab;
    public RuntimeAnimatorController animator;
    public List<SkillData> skills;
}

// Su dung
public class EnemySpawner : MonoBehaviour
{
    public EnemyPrototype orcPrototype;

    void SpawnOrc()
    {
        // Clone prototype de tao instance
        GameObject orc = Instantiate(orcPrototype.prefab);
        EnemyStats stats = orc.GetComponent<EnemyStats>();
        stats.hp = orcPrototype.hp;
        stats.attack = orcPrototype.attack;
        // ...
    }
}
```

### 4. Redis Lua Script — cached compilation

```python
import redis
import hashlib

class ScriptCache:
    """Prototype pattern cho Redis Lua scripts — compile 1 lan, clone nhieu lan."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._scripts: dict[str, str] = {}  # key -> sha

    def register(self, name: str, script: str) -> None:
        """Dang ky Lua script (compile 1 lan)."""
        sha = self.redis.script_load(script)
        self._scripts[name] = sha
        print(f"Script '{name}' compiled -> SHA: {sha}")

    def execute(self, name: str, keys: list[str] = None,
                args: list[str] = None) -> Any:
        """Execute script da compile (clone execution)."""
        sha = self._scripts.get(name)
        if sha is None:
            raise KeyError(f"Script '{name}' chua duoc dang ky")
        return self.redis.evalsha(sha, len(keys or []), *(keys or []), *(args or []))
```

## Kiem thu

```python
import pytest
from unittest.mock import MagicMock, patch
import copy


class TestEnemyPrototype:
    def test_clone_creates_new_object(self):
        proto = Enemy(name="Test", enemy_type="test", hp=100, attack=10)
        cloned = proto.clone()
        assert cloned is not proto  # Khac object
        assert cloned.entity_id != proto.entity_id  # Khac ID

    def test_clone_preserves_attributes(self):
        skills = [Skill("Fire", 50, 3.0, 20)]
        anim = AnimationData(["idle1"], ["atk1"], ["hit1"], ["death1"])
        proto = Enemy(
            name="Orc", enemy_type="orc", hp=200, attack=35,
            skills=skills, animation=anim,
        )
        cloned = proto.clone()
        assert cloned.name == "Orc"
        assert cloned.hp == 200
        assert cloned.attack == 35
        assert len(cloned.skills) == 1

    def test_clone_deep_copies_nested_objects(self):
        anim = AnimationData(["idle1"], ["atk1"], ["hit1"], ["death1"])
        proto = Enemy(name="Goblin", enemy_type="goblin", animation=anim)
        cloned = proto.clone()
        # Deep copy — thay doi cloned khong anh huong proto
        cloned.animation.idle_frames.append("idle2")
        assert len(proto.animation.idle_frames) == 1
        assert len(cloned.animation.idle_frames) == 2

    def test_clone_resets_state(self):
        proto = Enemy(name="Test", hp=100, attack=10)
        proto.take_damage(30)
        assert proto.hp == 70

        cloned = proto.clone()
        assert cloned.hp == 100  # Reset ve max

    def test_clone_with_overrides(self):
        proto = Enemy(name="Orc", enemy_type="orc", hp=200, attack=35)
        cloned = proto.clone(name="Elite Orc", hp=600, attack=50)
        assert cloned.name == "Elite Orc"
        assert cloned.hp == 600
        assert cloned.attack == 50

    def test_scale_to_level(self):
        proto = Enemy(name="Orc", level=1, hp=200, attack=35)
        scaled = proto.scale_to_level(10)
        assert scaled.level == 10
        assert scaled.hp > proto.hp
        assert scaled.attack > proto.attack
        # Scale factor: 1 + (10-1) * 0.15 = 2.35
        assert scaled.hp == int(200 * 2.35)

    def test_to_elite(self):
        proto = Enemy(name="Orc", hp=200, attack=35, defense=25)
        elite = proto.to_elite()
        assert elite.is_elite is True
        assert "Elite" in elite.name
        assert elite.hp == 600  # *3
        assert elite.attack == int(35 * 1.5)
        # Elite co them skill
        assert any(s.name == "Elite Strike" for s in elite.skills)

    def test_take_damage(self):
        enemy = Enemy(name="Test", hp=100, defense=10)
        actual = enemy.take_damage(30)
        assert actual == 20  # 30 - 10 defense
        assert enemy.hp == 80

    def test_take_damage_min_zero(self):
        enemy = Enemy(name="Test", hp=100, defense=50)
        actual = enemy.take_damage(20)
        assert actual == 0  # 20 - 50 = 0
        assert enemy.hp == 100  # Khong mat mau

    def test_prototype_isolation(self):
        proto = Enemy(name="Goblin", hp=100)
        clones = [proto.clone() for _ in range(5)]
        clones[0].take_damage(50)
        assert clones[0].hp == 50
        for c in clones[1:]:
            assert c.hp == 100  # Cac clone khac khong bi anh huong


class TestPrototypeRegistry:
    def test_register_and_spawn(self):
        registry = PrototypeRegistry()
        proto = Enemy(name="Goblin", hp=100)
        registry.register("goblin", proto)

        spawned = registry.spawn("goblin")
        assert spawned is not proto
        assert spawned.name == "Goblin"

    def test_spawn_unknown_key_raises_error(self):
        registry = PrototypeRegistry()
        with pytest.raises(KeyError):
            registry.spawn("unknown")

    def test_spawn_batch(self):
        registry = PrototypeRegistry()
        registry.register("orc", Enemy(name="Orc", hp=200))

        batch = registry.spawn_batch("orc", 10)
        assert len(batch) == 10
        assert all(o.name == "Orc" for o in batch)
        assert all(o is not batch[0] for o in batch[1:])  # All different objects


class TestEnemySpawner:
    @pytest.fixture
    def spawner(self):
        registry = PrototypeRegistry()
        registry.register("goblin", Enemy(name="Goblin", enemy_type="goblin", hp=80, attack=15))
        registry.register("orc", Enemy(name="Orc", enemy_type="orc", hp=200, attack=35))
        return EnemySpawner(registry)

    def test_spawn_goblin(self, spawner):
        goblin = spawner.spawn_goblin()
        assert goblin.name == "Goblin"
        assert goblin.enemy_type == "goblin"

    def test_spawn_orc_elite(self, spawner):
        orc = spawner.spawn_orc(elite=True)
        assert orc.is_elite

    def test_spawn_count(self, spawner):
        initial = spawner.total_spawned
        spawner.spawn_goblin()
        spawner.spawn_orc()
        assert spawner.total_spawned == initial + 2

    def test_spawn_wave(self, spawner):
        wave = spawner.spawn_wave([("goblin", 3), ("orc", 2)])
        assert len(wave) == 5
        assert sum(1 for e in wave if e.enemy_type == "goblin") == 3
        assert sum(1 for e in wave if e.enemy_type == "orc") == 2

## Uu va nhuoc diem

| Uu diem | Nhuoc diem |
|---------|-----------|
| **Performance**: Clone nhanh hon nhieu so voi tao tu constructor (dac biet khi I/O hoac tinh toan nang) | **Deep copy cost**: copy.deepcopy() co the cham neu object graph lon hoac co circular references |
| **Giam coupling**: Client khong can biet class cu the, chi can biet interface Prototype | **Memory**: Phai giu prototype master trong memory — ton RAM |
| **Dynamic configuration**: Co the spawn object voi cau hinh dong (clone + modify) | **Custom clone logic**: Can tu implement clone() cho nhieu class — de quen hoac sai |
| **Variety**: De dang tao nhieu bien the tu mot prototype goc (elite, scaled, customized) | **Shallow copy risk**: Neu dung shallow copy, clone co the gay side effects tren prototype |
| **Registry pattern**: Tap trung quan ly prototype o mot cho, de dang them/xoa loai object | **Resource management**: Object chua file handle, network connection... can logic clone dac biet |
| **Prototype inheritance**: Clone tu prototype co the ke thua ca state va behavior | **Initialization**: Van can tao prototype master lan dau — ton thoi gian va tai nguyen |
| **Anti-telescoping**: Khong can constructor do so tham so | **Debugging**: Kho debug vi object duoc tao tu clone, khong ro nguon goc |
| **Performance tuning**: Co the dung Object Pool + Prototype de toi uu spawn rate | **Thread-safety**: Clone trong moi truong da luong can co che dong bo |

---

## Ket luan

Prototype la pattern ly tuong khi **chi phi tao object tu constructor la qua dat** va ban can tao nhieu object co cau truc tuong tu nhau. **Golden rule**: Neu ban thay minh dang goi constructor voi 10+ tham so giong nhau nhieu lan, hoac neu ban can spawn nhieu object trong vong lap voi performance cao — do la luc Prototype toa sang.

Tôi nhớ có lần đọc được câu nói của Jeff Bezos: *"We are stubborn on vision, but flexible on details."* Prototype cũng vậy — cứng nhắc về cấu trúc (clone từ master), linh hoạt về chi tiết (override sau clone).

Cac truong hop dac biet phu hop voi Prototype:
1. **Game development**: Spawn enemy, bullet, particle, item — hang ngan object trong 1 frame.
2. **Document/Object templates**: Invoice template, contract template, email template.
3. **Configuration template**: Microservice config, deployment config, environment config.
4. **Performance-critical systems**: Object pooling, cache, flyweight ket hop.

Hay nho:
- **Dung `copy.deepcopy` cho object phuc tap** (nested objects, lists, dicts).
- **Dung `copy.copy` cho object don gian** (flat attributes, immutable children).
- **Implement `__deepcopy__` custom** khi can kiem soat clone logic.
- **Resett ID/state** sau khi clone — clone khong nen copy state dac thu (vi tri, target, buff).
- **Dung Prototype Registry** de tap trung quan ly — de dang them/sua/xoa prototype.

Prototype thuong duoc ket hop voi **Factory Method** (factory su dung prototype ben trong) va **Object Pool** (pool clone thay vi tao moi). Trong kien truc phan mem hien dai, Prototype la cong cu dam bao **performance va flexibility** khi tao object.

---

*Trân trọng!*
