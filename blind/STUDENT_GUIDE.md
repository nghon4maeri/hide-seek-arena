# Blind Adversary — Student Guide (Lab 2)

Chào mừng đến với Lab 2: **Blind Adversary**. Lab này giới thiệu **Partial Observability** — agent chỉ nhìn thấy một phần bản đồ và phải ra quyết định dưới thông tin không đầy đủ.

---

## Mục lục

1. [Quick Start](#quick-start)
2. [Hiểu về Game](#hiểu-về-game)
3. [Tạo Agent Cho Blind Mode](#tạo-agent-cho-blind-mode)
4. [Cài đặt Thuật toán](#cài-đặt-thuật-toán)
5. [Test Agent](#test-agent)
6. [Debugging Tips](#debugging-tips)
7. [Common Errors](#common-errors)
8. [Advanced Strategies](#advanced-strategies)

---

## Quick Start

### 1. Tạo Submission Folder

```bash
cd blind/submissions
mkdir <your_student_id>
```

### 2. Copy Template

```bash
cp TEMPLATE_agent.py <your_student_id>/agent.py
```

### 3. Chỉnh sửa Agent

Mở `submissions/<your_student_id>/agent.py` và cài đặt thuật toán của bạn.

### 4. Test

```bash
cd blind/src
python arena.py --seek <your_id> --hide example_student \
    --pacman-obs-radius 5 --ghost-obs-radius 5
```

---

## Hiểu về Game

### Partial Observability (Quan sát Một phần)

Đây là điểm khác biệt **quan trọng nhất** so với Lab 1:

- Agent chỉ nhìn thấy vùng **hình chữ thập** (cross-shaped) quanh vị trí hiện tại
- Tầm nhìn giới hạn trong **5 ô** theo 4 hướng: Up, Down, Left, Right
- Tia nhìn bị **chặn bởi tường** — toàn bộ các ô phía sau tường đều ẩn
- Tường luôn luôn nhìn thấy được (kiến trúc map cố định)

### Ba loại Cell trong `map_state`

| Giá trị | Ý nghĩa | Có thể đi vào? |
|---------|---------|----------------|
| `0` | Đường trống — **đang thấy** | Có |
| `1` | Tường — **luôn thấy** | Không |
| `-1` | **UNSEEN** — chưa biết có phải đường hay không | **Không được assume** |

### `enemy_position` có thể là `None`

Đây là trường hợp quan trọng phải xử lý. Khi đối thủ nằm ngoài tầm nhìn, `enemy_position = None`.

```python
def step(self, map_state, my_position, enemy_position, step_number):
    if enemy_position is not None:
        # Đối thủ đang thấy → pursue/evade
        pass
    else:
        # Đối thủ ẩn → search/explore/predict
        pass
```

### Điều kiện Thắng

- **Pacman thắng**: Manhattan distance `< 2` (bắt được Ghost)
- **Ghost thắng**: Sống sót qua `max_steps` (mặc định 200)

---

## Tạo Agent Cho Blind Mode

### Cấu trúc Code Bắt buộc

```python
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """Blind Seeker."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.last_seen_enemy = None      # Track last known enemy position
        self.memory_map = None           # Accumulated map knowledge

    def step(self, map_state, my_position, enemy_position, step_number):
        my_position = tuple(my_position)

        # 1. Update memory map
        self._update_memory(map_state)

        # 2. Update enemy tracking
        if enemy_position is not None:
            self.last_seen_enemy = enemy_position

        # 3. Decide action
        # ... your algorithm here ...

        return Move.STAY  # or (Move.UP, 2)

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible = (map_state != -1)
        self.memory_map[visible] = map_state[visible]


class GhostAgent(BaseGhostAgent):
    """Blind Hider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_seen_enemy = None
        self.memory_map = None

    def step(self, map_state, my_position, enemy_position, step_number):
        my_position = tuple(my_position)
        self._update_memory(map_state)

        if enemy_position is not None:
            self.last_seen_enemy = enemy_position

        # ... your algorithm here ...

        return Move.STAY  # ONLY Move, not tuple

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible = (map_state != -1)
        self.memory_map[visible] = map_state[visible]
```

### Helper Functions

```python
def _is_valid_position(self, pos, map_state):
    """Check if position is valid (not wall, in bounds)."""
    row, col = pos
    h, w = map_state.shape
    if row < 0 or row >= h or col < 0 or col >= w:
        return False
    # Treat -1 (unseen) as potentially valid, only block walls (1)
    return map_state[row, col] != 1

def _apply_move(self, pos, move):
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)

def _get_neighbors(self, pos, map_state):
    neighbors = []
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        nxt = self._apply_move(pos, move)
        if self._is_valid_position(nxt, map_state):
            neighbors.append((nxt, move))
    return neighbors

def _manhattan_distance(self, pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

---

## Cài đặt Thuật toán

### 1. Mental Map (Bắt buộc cho Blind Mode)

Agent phải tự xây dựng "bản đồ tinh thần" — tích lũy thông tin từ các observation qua các step:

```python
def _update_memory(self, map_state):
    if self.memory_map is None:
        self.memory_map = np.full_like(map_state, -1, dtype=int)
    visible_mask = (map_state != -1)
    self.memory_map[visible_mask] = map_state[visible_mask]
```

Sau đó, tất cả thuật toán tìm đường (A*, BFS) **phải chạy trên `self.memory_map`**, không phải `map_state` thô.

### 2. A* Search với Memory Map

```python
def astar(self, start, goal):
    """A* on self.memory_map. Treats -1 cells as traversable (optimistic)."""
    ms = self.memory_map
    h, w = ms.shape
    # ... standard A* but ms[r,c] != 1 instead of == 0
```

### 3. Pacman Strategies (Blind Seeker)

| Chiến lược | Mô tả |
|------------|-------|
| **Greedy Chase** | Đuổi theo vị trí cuối cùng thấy được của Ghost |
| **A\* to Last Known** | Tìm đường ngắn nhất đến vị trí cuối cùng |
| **Frontier Exploration** | Khi không thấy Ghost: di chuyển về phía frontier (ranh giới known/unknown) |
| **Belief State Prediction** | Dự đoán phân phối xác suất vị trí Ghost |
| **Interception** | Dự đoán hướng di chuyển của Ghost, cắt đường |

### 4. Ghost Strategies (Blind Hider)

| Chiến lược | Mô tả |
|------------|-------|
| **Maximize Distance** | Đi xa nhất khỏi vị trí cuối cùng thấy Pacman |
| **A\* to Farthest Cell** | Tìm ô xa nhất Ghost đến được trước Pacman |
| **Junction Preference** | Ưu tiên ở gần ngã rẽ để có nhiều lối thoát |
| **Dead-End Avoidance** | Tránh đi vào ngõ cụt (chỉ 1 lối ra) |
| **Minimax Evasion** | Mô hình hóa bước đi của Pacman để chọn nước an toàn nhất |

### 5. Recommended Algorithms from PDF

Theo tài liệu Blind-2526-3.pdf, các thuật toán được gợi ý:
- **Minimax** + **Alpha-Beta Pruning**: Adversarial search thích nghi cho partial observability
- **Monte Carlo Tree Search**: Mô phỏng nhiều kịch bản với tính ngẫu nhiên
- **Expectiminimax**: Kết hợp xác suất với adversarial search
- **Belief State / Particle Filter**: Ước lượng phân phối vị trí đối thủ

---

## Test Agent

### Basic Testing

```bash
cd blind/src

# Test Pacman của bạn vs Ghost mẫu
python arena.py --seek <your_id> --hide example_student \
    --pacman-obs-radius 5 --ghost-obs-radius 5

# Test Ghost của bạn vs Pacman mẫu
python arena.py --seek example_student --hide <your_id> \
    --pacman-obs-radius 5 --ghost-obs-radius 5

# Test cả hai
python arena.py --seek <your_id> --hide <your_id> \
    --pacman-obs-radius 5 --ghost-obs-radius 5
```

### Advanced Options

```bash
cd blind/src

# Không hiển thị (nhanh hơn)
python arena.py --seek <your_id> --hide example_student \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --no-viz

# Chỉnh observation radius (test độ khó khác nhau)
python arena.py --seek <your_id> --hide example_student \
    --pacman-obs-radius 3 --ghost-obs-radius 7

# Stochastic start + full steps
python arena.py --seek <your_id> --hide example_student \
    --pacman-obs-radius 5 --ghost-obs-radius 5 \
    --start-mode stochastic --max-steps 200 --no-viz

# Không fog-of-war (test fallback với Perfect Information)
python arena.py --seek <your_id> --hide example_student \
    --pacman-obs-radius 0 --ghost-obs-radius 0
```

### Benchmark

```bash
# Từ repo root
python blind/scripts/benchmark_agents.py --seek <your_id> --hide example_student --games 20
```

---

## Debugging Tips

### 1. In Memory Map

```python
def step(self, map_state, my_position, enemy_position, step_number):
    self._update_memory(map_state)
    if step_number % 10 == 0:
        known = (self.memory_map != -1).sum()
        total = self.memory_map.size
        print(f"Step {step_number}: known {known}/{total} cells ({100*known/total:.0f}%)")
```

### 2. In Observation

```python
def step(self, map_state, my_position, enemy_position, step_number):
    visible_count = (map_state != -1).sum()
    print(f"Step {step_number}: visible={visible_count} enemy={'seen' if enemy_position else 'hidden'}")
```

### 3. Xem Chậm để Debug

```bash
cd blind/src
python arena.py --seek <your_id> --hide example_student \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --delay 1.0
```

---

## Common Errors

### Error: Agent crashes với TypeError khi enemy_position = None

```python
# SAI
distance = manhattan(my_position, enemy_position)  # enemy_position có thể là None!

# ĐÚNG
if enemy_position is not None:
    distance = manhattan(my_position, enemy_position)
```

### Error: Agent đi vào tường do assume cell -1 là đường

```python
# SAI: assume mọi cell != 1 đều đi được
return map_state[row, col] != 1

# ĐÚNG: chỉ đi vào cell đã biết là đường (0)
return map_state[row, col] == 0
```

Lưu ý: khi chạy A* trên bản đồ tích lũy (memory map), có thể treat `-1` là optimistic (có thể đi) để tìm đường qua vùng chưa khám phá.

### Error: Ghost trả về tuple thay vì Move

```python
# SAI — GhostAgent không được trả tuple
return (Move.UP, 2)  # → AgentLoadError

# ĐÚNG
return Move.UP
```

---

## Quick Reference

### Input Parameters

| Tham số | Mô tả |
|---------|-------|
| `map_state` | `numpy.ndarray` (21×21): `0`=đường thấy, `1`=tường, `-1`=unseen |
| `my_position` | `(row, col)` tuple |
| `enemy_position` | `(row, col)` tuple hoặc `None` |
| `step_number` | `int`, bắt đầu từ 1 |

### Return Values

| Agent | Hợp lệ |
|-------|--------|
| PacmanAgent | `Move.UP` hoặc `(Move.UP, 2)` |
| GhostAgent | Chỉ `Move.UP` |

### CLI Flags cho Blind Mode

| Flag | Mặc định | Mô tả |
|------|---------|-------|
| `--pacman-obs-radius` | `5` | Tầm nhìn Pacman (0 = full) |
| `--ghost-obs-radius` | `5` | Tầm nhìn Ghost (0 = full) |
| `--max-steps` | `200` | Số bước tối đa |
| `--pacman-speed` | `2` | Tốc độ Pacman |
| `--capture-distance` | `2` | Khoảng cách bắt |

---

## Good Luck!

Hãy nhớ:
- **Xử lý `enemy_position = None`** — đây là case quan trọng nhất
- **Xây dựng memory map** — tích lũy thông tin qua các step
- **Test với nhiều observation radius** — agent phải hoạt động ở mọi mức
- **Start simple, improve incrementally**
