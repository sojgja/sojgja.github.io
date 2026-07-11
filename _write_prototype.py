import os

path = r'F:\git\sojgja.github.io\docs\series\prototype.md'

content = r'''---
id: prototype
title: Prototype
sidebar_label: \U0001f9ec Prototype
sidebar_position: 6
---

# Prototype

> *"Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype."* — Gang of Four, *Design Patterns: Elements of Reusable Object-Oriented Software*, 1994.

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

Quá trình này mất 50-200ms mỗi lần spawn. Với game, khi người chơi vào dungeon, hệ thống có thể spawn 50-100 quái vật cùng lúc — nghĩa là 5-20 giây loading. Không thể chấp nhận được.

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
- **Performance**: Clone trong memory (microseconds) thay vì load từ disk (milliseconds) — nhanh hơn 1000x.
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
'''

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Prototype part 1 written: {os.path.getsize(path)} bytes')
