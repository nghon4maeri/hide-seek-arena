# Pacman vs Ghost Arena — Lab 1: Hide and Seek Arena

Môn học: **CSC14003 — Nhập môn Trí tuệ Nhân tạo**

---

## 1. Tổng quan Dự án

Đây là không gian làm việc chính (`pacman/`) của bài tập lớn **Hide and Seek Arena** — một trò chơi đối kháng hai người trên bản đồ lưới Pacman cổ điển. Hai agent AI tranh đấu trong môi trường **thông tin hoàn chỉnh (Perfect Information)** với mục tiêu đối lập: một bên săn đuổi, một bên lẩn trốn.

### 1.1. Vai trò Agent

| Vai trò | Agent | Nhiệm vụ | Điều kiện thắng |
|---------|-------|----------|-----------------|
| **Seeker** (Người đi săn) | **PacmanAgent** | Đuổi bắt Ghost | Khoảng cách Manhattan `< 2` |
| **Hider** (Người đi trốn) | **GhostAgent** | Sinh tồn đến hết trận | Sống sót qua `max_steps` (200 bước) |

### 1.2. Tiêu chí Chấm điểm

| Hạng mục | Điểm |
|----------|------|
| Hoàn thiện cài đặt giải thuật | 3 |
| Xếp hạng trong lần nộp đầu tiên | Tối đa 3 |
| Xếp hạng trong lần nộp tối ưu | Tối đa 4 |

**Tie-break**: Team có chênh lệch giữa số bước trung bình Pacman và Ghost thấp hơn sẽ
xếp hạng cao hơn (xem mục 4.8).

### 1.3. Đặc tả Kỹ thuật

| Thành phần | Mô tả |
|------------|-------|
| **Bản đồ** | Lưới 21×21. `0` = đường trống, `1` = tường. Map cố định dạng Pacman cổ điển |
| **Chế độ quan sát** | Fog-of-war **TẮT**. Hai agent luôn nhìn thấy vị trí của nhau (Perfect Information) |
| **Cơ chế di chuyển** | **Đồng thời (Simultaneous)** — cả hai nhận state, quyết định, rồi cùng cập nhật vị trí trong 1 step. Không ai thấy nước đi của đối phương trước khi chọn |

### 1.4. Khác biệt Tốc độ

| | Pacman (Seeker) | Ghost (Hider) |
|---|---|---|
| **Số ô / step** | 1 ô, hoặc nhiều ô thẳng (`pacman_speed`, mặc định = 2) | **1 ô duy nhất** |
| **Hành động trả về** | `Move` hoặc `(Move, steps)` | **Chỉ** `Move` |
| **Ràng buộc** | `1 ≤ steps ≤ pacman_speed`; chỉ đi thẳng, dừng nếu gặp tường | Không được rẽ góc chữ L trong 1 lượt. Trả về tuple hoặc string sẽ bị xử **thua ngay lập tức** |

---

## 2. Cấu trúc Thư mục

```text
pacman/
├── src/                          # Framework lõi — NGHIÊM CẤM CHỈNH SỬA
│   ├── arena.py                  # Điều phối trận đấu, vòng lặp game
│   ├── environment.py            # Bản đồ, luật di chuyển, điều kiện thắng, observation
│   ├── agent_interface.py        # Base class: PacmanAgent, GhostAgent
│   ├── agent_loader.py           # Dynamic import + validate agent
│   └── visualizer.py             # Hiển thị trận đấu trên terminal
│
├── submissions/                  # Khu vực code của nhóm
│   ├── 24127561/agent.py         # Kỹ sư Pacman (Seeker)
│   ├── 24127192/agent.py         # Kỹ sư Ghost (Hider)
│   ├── 24127457/agent.py         # Sandbox của Leader (tích hợp & review)
│   ├── team_submission/agent.py  # Bản merge cuối cùng — DÙNG ĐỂ NỘP BÀI
│   ├── example_student/          # Agent mẫu cơ bản của giảng viên
│   ├── simple_agent/             # Agent random đơn giản (baseline)
│   ├── TEMPLATE_agent.py         # Template khởi tạo cho sinh viên
│   ├── broken_agent/             # Agent test lỗi runtime
│   ├── slow_agent/               # Agent test timeout
│   └── exit_test/                # Agent test sys.exit()
│
├── scripts/                      # Công cụ đánh giá & đóng gói
│   ├── benchmark_agents.py       # Chạy nhiều trận đấu, thống kê kết quả
│   ├── benchmark_full.py         # Benchmark tự động N games stochastic + stats
│   ├── run_smoke_test.py         # Smoke test nhanh (5 step, no-viz)
│   └── export_submission.py      # Đóng gói bài nộp
│
├── tests/                        # Bộ test kiểm tra runtime
│   ├── test_submission_interface.py  # Kiểm tra interface hợp lệ
│   ├── test_workspace_structure.py   # Kiểm tra cấu trúc thư mục
│   └── test_runtime_smoke.py         # Kiểm tra chạy không crash
│
├── docs/                         # Tài liệu nội bộ nhóm
└── README.md                     # File này
```

### 2.1. Quy tắc Phân quyền

| Khu vực | Ai được sửa | Ghi chú |
|---------|------------|---------|
| `src/` | **Không ai** | Framework của giảng viên |
| `submissions/24127561/` | Kỹ sư Pacman | Phát triển thuật toán săn đuổi |
| `submissions/24127192/` | Kỹ sư Ghost | Phát triển thuật toán lẩn trốn |
| `submissions/24127457/` | Leader | Sandbox tích hợp, test chéo |
| `submissions/team_submission/` | **Leader** (sau review) | Bản merge cuối cùng để nộp |
| `scripts/`, `tests/`, `docs/` | Cả nhóm | Cập nhật khi cần |

---

## 3. Định nghĩa Hàm `step()`

Đây là hàm **duy nhất** mà Framework gọi mỗi step. Cả hai agent phải implement đúng signature này.

### 3.1. Signature

```python
def step(self, map_state, my_position, enemy_position, step_number):
```

### 3.2. Mô tả Tham số

| Tham số | Kiểu dữ liệu | Mô tả |
|---------|-------------|-------|
| `map_state` | `numpy.ndarray` (2D, shape: `21×21`) | Bản đồ toàn cục. `0` = đường trống, `1` = tường |
| `my_position` | `tuple` — `(row: int, col: int)` | Vị trí hiện tại của agent trong hệ tọa độ tuyệt đối |
| `enemy_position` | `tuple` — `(row: int, col: int)` | Vị trí đối thủ. **Luôn có giá trị** (Perfect Information) |
| `step_number` | `int` | Step hiện tại của game, **bắt đầu từ 1** |

> **Ghi chú:** Không có object `state` hay `percept` nào được gói lại. Framework truyền 4 tham số riêng biệt trực tiếp vào `step()`.

### 3.3. Giá trị Trả về

**PacmanAgent** — có 2 lựa chọn:

```python
# Cách 1: Trả về Move enum (tương đương steps = 1)
return Move.UP
return Move.DOWN
return Move.LEFT
return Move.RIGHT
return Move.STAY

# Cách 2: Trả về tuple (Move, steps) để di chuyển nhiều ô thẳng
return (Move.UP, 2)       # Đi lên 2 ô
return (Move.RIGHT, 3)    # Đi phải 3 ô (nếu pacman_speed >= 3)

# Ràng buộc: 1 <= steps <= self.pacman_speed
# Pacman đi thẳng từng ô, dừng nếu gặp tường
```

**GhostAgent** — **CHỈ** được trả về `Move` enum:

```python
return Move.UP
return Move.DOWN
return Move.LEFT
return Move.RIGHT
return Move.STAY

return (Move.UP, 2)    # Tuple → AgentLoadError
return "UP"             # String → AgentLoadError
return (-1, 0)          # Tọa độ → AgentLoadError
return None             # None → AgentLoadError
```

### 3.4. Code Mẫu Tối thiểu

```python
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """Pacman Seeker — thuật toán săn đuổi."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_seen_enemy = None

    def step(self, map_state, my_position, enemy_position, step_number):
        # enemy_position luôn có giá trị (Perfect Information)
        my_position = tuple(my_position)
        enemy_position = tuple(enemy_position)

        # → Cài đặt thuật toán tìm đường (A*, BFS, Minimax...) tại đây

        return Move.STAY  # hoặc (Move.UP, 2)


class GhostAgent(BaseGhostAgent):
    """Ghost Hider — thuật toán lẩn trốn."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, map_state, my_position, enemy_position, step_number):
        my_position = tuple(my_position)
        enemy_position = tuple(enemy_position)

        # → Cài đặt thuật toán sinh tồn (BFS, Minimax, Flood Fill...) tại đây

        return Move.STAY  # CHỈ return Move enum, không return tuple
```

---

## 4. Hướng dẫn Chạy & Đánh giá

Có 2 cách chạy, chọn 1 trong 2:

| Cách | Mô tả |
|------|-------|
| **A: Chạy từ repo root** | Cần thêm `--submissions-dir pacman/submissions` cho mọi lệnh `arena.py` |
| **B: `cd pacman/src` trước** | Không cần flag `--submissions-dir` (tự động dùng `../submissions`) |

> Hướng dẫn bên dưới dùng **cách B** — ngắn gọn và ít lỗi hơn. Tất cả lệnh `benchmark_agents.py` và test script vẫn chạy từ repo root bình thường.

---

### 4.0. Chạy có Trực quan (Visualization)

Xem agent chạy trực tiếp trên terminal — hữu ích để debug và hiểu hành vi agent.

```bash
cd pacman/src

# === Chạy mặc định (delay 0.1s giữa các bước) ===
python arena.py --seek 24127561 --hide 24127192

# === Xem chậm để debug (delay 0.5s) ===
python arena.py --seek 24127561 --hide 24127192 --delay 0.5

# === Xem rất chậm (delay 1.0s) ===
python arena.py --seek 24127561 --hide 24127192 --delay 1.0

# === Pacman 24127457 vs Ghost example_student ===
python arena.py --seek 24127457 --hide example_student --delay 0.3

# === Pacman example_student vs Ghost 24127457 ===
python arena.py --seek example_student --hide 24127457 --delay 0.3

# === Self-play 24127457 ===
python arena.py --seek 24127457 --hide 24127457 --delay 0.3
```

| Flag | Mặc định | Ý nghĩa |
|------|----------|---------|
| `--delay` | `0.1` | Giây chờ giữa mỗi step (tăng để xem chậm) |
| `--no-viz` | tắt | Tắt hiển thị, chỉ in kết quả cuối |
| `--max-steps` | `200` | Giới hạn số bước tối đa |

> **Mẹo:** Luôn chạy thử với `--delay 0.3` hoặc `--delay 0.5` khi mới code để quan sát
> hành vi agent. Khi đã OK thì thêm `--no-viz` để benchmark nhanh.

---

### 4.1. Đấu giữa Pacman (24127561) và Ghost (24127192)

```bash
# === Benchmark 10 trận (deterministic, chạy từ repo root) ===
python pacman/scripts/benchmark_agents.py --seek 24127561 --hide 24127192 --games 10 --max-steps 200

# === Chạy 1 trận không hiển thị, vị trí cố định ===
cd pacman/src
python arena.py --seek 24127561 --hide 24127192 --no-viz --max-steps 200

# === Chạy 1 trận không hiển thị, vị trí ngẫu nhiên ===
cd pacman/src
python arena.py --seek 24127561 --hide 24127192 --no-viz --start-mode stochastic --max-steps 200
```

> **Lưu ý về `benchmark_agents.py`:**
> - Script hiện chỉ hỗ trợ chế độ deterministic (không có flag `--start-mode`).
> - Output chỉ báo cáo `return_code` của từng trận (0 = chạy thành công), chưa tự động
>   thống kê thắng/thua hay số bước trung bình. Cần đọc output arena để lấy kết quả.
> - Để benchmark với stochastic start, dùng vòng lặp `arena.py` thủ công (xem 4.7).


### 4.2. Test Pacman (24127561) với Ghost mẫu của giảng viên

```bash
# Benchmark 5 trận — Pacman nhóm vs Ghost mẫu
python pacman/scripts/benchmark_agents.py --seek 24127561 --hide example_student --games 5 --max-steps 200

# Chạy 1 trận không hiển thị
cd pacman/src
python arena.py --seek 24127561 --hide example_student --no-viz --max-steps 200
```

### 4.3. Test Ghost (24127192) với Pacman mẫu của giảng viên

```bash
# Benchmark 5 trận — Pacman mẫu vs Ghost nhóm
python pacman/scripts/benchmark_agents.py --seek example_student --hide 24127192 --games 5 --max-steps 200

# Chạy 1 trận không hiển thị
cd pacman/src
python arena.py --seek example_student --hide 24127192 --no-viz --max-steps 200
```

### 4.4. Test bản merge cuối cùng (`team_submission`)

```bash
cd pacman/src

# Team submission làm Pacman vs Ghost mẫu
python arena.py --seek team_submission --hide example_student --no-viz --max-steps 200

# Team submission làm Ghost vs Pacman mẫu
python arena.py --seek example_student --hide team_submission --no-viz --max-steps 200

# Team submission đấu với chính nó (Pacman và Ghost đều là team)
python arena.py --seek team_submission --hide team_submission --no-viz --max-steps 200
```

### 4.5. Smoke Test & Unit Test

```bash
# Smoke test nhanh (chạy từ repo root)
python pacman/scripts/run_smoke_test.py

# Chạy toàn bộ test suite
python -m pytest pacman/tests

# Chạy test cụ thể
python -m pytest pacman/tests/test_submission_interface.py -v
```

### 4.6. Các Tùy chọn Nâng cao

```bash
cd pacman/src

# Điều chỉnh tốc độ Pacman (mặc định: 2)
python arena.py --seek 24127561 --hide 24127192 --pacman-speed 3

# Điều chỉnh ngưỡng bắt (mặc định: 2, tức là Manhattan < 2)
python arena.py --seek 24127561 --hide 24127192 --capture-distance 3

# Giảm thời gian trực quan hóa (xem chậm để debug)
python arena.py --seek 24127561 --hide 24127192 --delay 0.5

# Chế độ khởi đầu ngẫu nhiên (công bằng hơn deterministic start)
python arena.py --seek 24127561 --hide 24127192 --start-mode stochastic --max-steps 200 --no-viz

# Giới hạn thời gian mỗi step (mặc định: 1.0s, chỉ hoạt động trên Linux/macOS)
python arena.py --seek 24127561 --hide 24127192 --step-timeout 1.0
```

> **Lưu ý:** `--step-timeout` dùng `SIGALRM` nên **không hoạt động trên Windows**. Trên Windows, warning sẽ hiện ra và timeout bị vô hiệu hóa.

### 4.7. Benchmark Stochastic Thủ công

Do `benchmark_agents.py` chưa hỗ trợ `--start-mode stochastic`, để chạy benchmark với
vị trí khởi đầu ngẫu nhiên, dùng vòng lặp `arena.py`:

**Windows (PowerShell):**
```powershell
cd pacman/src
$wins = 0; $steps = 0; 1..20 | ForEach-Object {
    $out = python arena.py --seek 24127561 --hide 24127192 --no-viz --start-mode stochastic --max-steps 200 2>&1
    if ($out -match "WINNER: 24127192") { $wins++ }
    if ($out -match "Total Steps: (\d+)") { $steps += [int]$Matches[1] }
    Write-Host "game $_ done"
}; Write-Host "Ghost wins: $wins/20, total steps: $steps"
```

**Linux/macOS (Bash):**
```bash
cd pacman/src
wins=0; steps=0; for i in $(seq 1 20); do
  out=$(python arena.py --seek 24127561 --hide 24127192 --no-viz --start-mode stochastic --max-steps 200 2>&1)
  echo "$out" | grep -q "WINNER: 24127192" && wins=$((wins + 1))
  s=$(echo "$out" | grep -oP 'Total Steps:\s*\K\d+')
  steps=$((steps + s))
  echo "game $i done"
done; echo "Ghost wins: $wins/20, total steps: $steps"
```

### 4.8. Cách tính Tỷ lệ Thắng & Tie-Break

Theo quy định của BTC:

| Vai trò | Công thức tỷ lệ thắng |
|---------|------------------------|
| **Ghost (Hider)** | `winrate_hide = Số trận Ghost thắng / Tổng số trận làm Ghost` |
| **Pacman (Seeker)** | `winrate_seek = Số trận Pacman thắng / Tổng số trận làm Pacman` |

**Tie-break:** Nếu 2 team có tỷ lệ thắng bằng nhau, team nào có **chênh lệch giữa số bước
trung bình của Pacman và Ghost thấp hơn** sẽ xếp hạng cao hơn.

```
diff = avg_steps_pacman - avg_steps_ghost   # Càng thấp càng tốt
```

Điều này có nghĩa:
- Pacman cần bắt Ghost **càng nhanh càng tốt** (giảm avg_steps_pacman)
- Ghost cần sống **càng lâu càng tốt** (tăng avg_steps_ghost)
- Cả hai chỉ số đều quan trọng như nhau cho thứ hạng cuối cùng

### 4.9. Benchmark Tự động 100 Games (Stochastic)

Script `benchmark_full.py` tự động chạy N games (mặc định 100) với vị trí spawn ngẫu
nhiên, parse kết quả và báo cáo thống kê chi tiết:

```bash
# Chạy từ repo root
python pacman/scripts/benchmark_full.py --seek 24127561 --hide 24127457 --games 100

# Output JSON để parse tự động
python pacman/scripts/benchmark_full.py --seek 24127457 --hide 24127192 --games 100 --json

# Tùy chỉnh max steps và các tham số khác
python pacman/scripts/benchmark_full.py \
    --seek 24127561 --hide 24127457 \
    --games 200 --max-steps 300 \
    --capture-distance 2 --pacman-speed 2

# So sánh 2 Ghost với cùng 1 Pacman
python pacman/scripts/benchmark_full.py --seek 24127561 --hide 24127457 --games 100
python pacman/scripts/benchmark_full.py --seek 24127561 --hide 24127192 --games 100
```

**Output mẫu:**

```text
============================================================
  RESULTS
============================================================

  Total games  : 100
  Completed    : 100
  Errors       : 0

  24127561 (Seeker) wins : 72 (72.0%)
  24127457 (Hider) wins  : 28 (28.0%)

  Seeker avg capture steps : 14.2  (min=5, max=52)
  Hider  avg survival steps: 200.0  (min=200, max=200)
  Overall avg steps/game   : 66.1

  Tie-break diff (seek_avg - hide_avg) : -185.8  (lower = better)

  Total time: 45s  (0.5s per game)
```

| Flag | Mặc định | Ý nghĩa |
|------|----------|---------|
| `--seek` | *(bắt buộc)* | ID của Pacman (Seeker) |
| `--hide` | *(bắt buộc)* | ID của Ghost (Hider) |
| `--games` | `100` | Số game cần chạy |
| `--max-steps` | `200` | Giới hạn bước mỗi game |
| `--capture-distance` | `2` | Ngưỡng Manhattan để Pacman bắt Ghost |
| `--pacman-speed` | `2` | Tốc độ Pacman |
| `--json` | tắt | In thêm JSON summary để dùng với script tự động |

---

## 5. Ràng buộc Kỹ thuật Khi Nộp Bài

### 5.1. Giới hạn Hệ thống

| Ràng buộc | Giá trị | Ghi chú |
|-----------|--------|---------|
| **Thời gian / step** | Tối đa **1.0 giây** | Vượt quá → agent bị xử thua (AgentTimeoutError) |
| **Bộ nhớ (RAM)** | Tối đa **128 MB** | Tiêu chuẩn Google Colab CPU-only |
| **Thư viện hợp lệ** | `numpy`, `pandas`, `scipy`, `gurobi` | Không dùng ML library (PyTorch, TF, sklearn...) |

### 5.2. Yêu cầu File Nộp

- **Chỉ nộp** nội dung trong `submissions/team_submission/`
- File `agent.py` phải định nghĩa đúng 2 class: `PacmanAgent` và `GhostAgent`
- Cả 2 class phải kế thừa từ `BasePacmanAgent` và `BaseGhostAgent` (trong `agent_interface.py`)
- Hàm `step()` phải tuân thủ đúng signature 4 tham số
- Không được import hoặc sửa các file trong `src/`

### 5.3. Quy trình Làm việc Nhóm

```text
1. 24127561 phát triển Pacman trong submissions/24127561/agent.py
2. 24127192 phát triển Ghost  trong submissions/24127192/agent.py
3. Leader (24127457) test chéo, review interface
4. Leader merge code đã duyệt vào submissions/team_submission/agent.py
5. Cả nhóm chạy smoke test + benchmark để xác nhận
6. Leader export và nộp team_submission
```

### 5.4. Đóng gói Bài Nộp

```bash
python pacman/scripts/export_submission.py team_submission --force
```

---

## 6. Phụ lục: Các Thuật toán Gợi ý

| Thuật toán | Ứng dụng cho Pacman (Seeker) | Ứng dụng cho Ghost (Hider) |
|------------|------------------------------|----------------------------|
| **BFS** | Tìm đường ngắn nhất đến Ghost | Tính khoảng cách an toàn, phân tích vùng |
| **A\*** | Đuổi bắt tối ưu với heuristic | Lập kế hoạch đường thoát |
| **Flood Fill** | Phân tích áp lực bẫy | Ước lượng vùng an toàn, tránh ngõ cụt |
| **Minimax** | Dự đoán phản ứng của Ghost | Dự đoán bước đuổi của Pacman |
| **Alpha-Beta Pruning** | Cắt tỉa nhánh Minimax | Cắt tỉa nhánh Minimax |
| **Adversarial Search** | Mô hình trò chơi đối kháng | Mô hình trò chơi đối kháng |
