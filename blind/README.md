# Blind Adversary Arena — Lab 2: Partial Observability

Môn học: **CSC14003 — Nhập môn Trí tuệ Nhân tạo**

---

## 1. Tổng quan

Đây là không gian làm việc chính (`blind/`) của bài tập lớn **Blind Adversary** — Lab 2 của Hide and Seek Arena. Trong Lab này, chúng ta giới thiệu **quan sát một phần (Partial Observability)**: agent không có quyền truy cập toàn bộ bản đồ mà chỉ nhìn thấy một vùng giới hạn hình chữ thập (cross-shaped field of view).

### 1.1. Khác biệt chính so với Lab 1

| | Lab 1 (HideSeek) | Lab 2 (Blind Adversary) |
|---|---|---|
| **Thông tin** | Perfect Information (toàn bộ bản đồ) | Partial Observability |
| **Tầm nhìn** | Toàn bộ 21×21 | Cross-shaped, 5 ô / 4 hướng, bị chặn bởi tường |
| **`map_state`** | `0`=đường, `1`=tường | `0`=đường thấy được, `1`=tường, **`-1`=chưa biết** |
| **`enemy_position`** | Luôn có giá trị | Có thể là **`None`** (khi không thấy đối thủ) |
| **Chiến lược** | Tìm đường tối ưu | Suy luận dưới thông tin không đầy đủ |

### 1.2. Mô hình Tầm nhìn (Cross-Shaped Vision)

Agent nhìn thấy tối đa **5 ô** ra mỗi hướng Up/Down/Left/Right. Tia nhìn bị **chặn hoàn toàn** khi gặp tường.

```
Ví dụ: Ghost tại G, # = tường, . = thấy được, ? = không thấy

? ? ? . ? ? ?
? ? ? . ? ? ?
. . . G . . .
? ? ? . ? ? ?
? ? ? # ? ? ?    ← tường chặn tia nhìn xuống
? ? ? ? ? ? ?
```

### 1.3. Vai trò Agent

| Vai trò | Agent | Nhiệm vụ | Điều kiện thắng |
|---------|-------|----------|-----------------|
| **Seeker** | **PacmanAgent** | Đuổi bắt Ghost | Manhattan distance `< 2` |
| **Hider** | **GhostAgent** | Sinh tồn đến hết trận | Sống sót qua `max_steps` (200 bước) |

### 1.4. Tiêu chí Chấm điểm

| Hạng mục | Điểm |
|----------|------|
| Hoàn thiện cài đặt giải thuật | 3 |
| Xếp hạng trong lần nộp đầu tiên | Tối đa 3 |
| Xếp hạng trong lần nộp tối ưu | Tối đa 4 |

---

## 2. Cấu trúc Thư mục

```text
blind/
├── src/                          # Framework lõi — KHÔNG CHỈNH SỬA
│   ├── arena.py                  # Điều phối trận đấu
│   ├── environment.py            # Bản đồ, luật, observation (cross-shaped)
│   ├── agent_interface.py        # Base class: PacmanAgent, GhostAgent
│   ├── agent_loader.py           # Dynamic import + validate agent
│   └── visualizer.py             # Hiển thị trận đấu trên terminal
│
├── submissions/                  # Khu vực code của nhóm
│   ├── example_student/agent.py  # Agent mẫu cho Blind mode
│   ├── TEMPLATE_agent.py         # Template khởi tạo
│   └── team_submission/agent.py  # Bản merge cuối cùng — DÙNG ĐỂ NỘP BÀI
│
├── scripts/                      # Công cụ đánh giá & đóng gói
│   ├── benchmark_agents.py       # Chạy nhiều trận với fog-of-war mặc định
│   ├── run_smoke_test.py         # Smoke test nhanh
│   └── export_submission.py      # Đóng gói bài nộp
│
├── docs/                         # Tài liệu nội bộ nhóm
├── README.md                     # File này
├── STUDENT_GUIDE.md              # Hướng dẫn chi tiết cho sinh viên
├── run_game.sh                   # Quick run với fog-of-war mặc định
└── requirements.txt              # Dependencies
```

---

## 3. Định nghĩa Hàm `step()`

### 3.1. Signature

```python
def step(self, map_state, my_position, enemy_position, step_number):
```

### 3.2. Mô tả Tham số

| Tham số | Kiểu dữ liệu | Mô tả |
|---------|-------------|-------|
| `map_state` | `numpy.ndarray` (2D, 21×21) | **0**=đường thấy được, **1**=tường, **-1**=chưa biết |
| `my_position` | `tuple` — `(row, col)` | Vị trí hiện tại của agent |
| `enemy_position` | `tuple` hoặc **`None`** | Vị trí đối thủ nếu thấy, `None` nếu ngoài tầm nhìn |
| `step_number` | `int` | Step hiện tại, bắt đầu từ 1 |

### 3.3. Giá trị Trả về

**PacmanAgent:**
```python
return Move.UP              # Di chuyển 1 ô
return (Move.RIGHT, 2)      # Đi thẳng 2 ô (1 <= steps <= pacman_speed)
```

**GhostAgent:**
```python
return Move.DOWN            # CHỈ return Move, không được return tuple
```

---

## 4. Hướng dẫn Chạy

### 4.1. Smoke Test

```bash
# Chạy từ repo root
python blind/scripts/run_smoke_test.py

# Hoặc tự chạy
cd blind/src
python arena.py --seek team_submission --hide example_student \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --no-viz --max-steps 20
```

### 4.2. Benchmark

```bash
# Benchmark 10 trận (Blind mode mặc định)
python blind/scripts/benchmark_agents.py --seek example_student --hide example_student --games 10

# Tùy chỉnh observation radius
python blind/scripts/benchmark_agents.py --seek team_submission --hide example_student \
    --pacman-obs 3 --ghost-obs 7 --games 5
```

### 4.3. Chạy 1 trận có hiển thị

```bash
cd blind/src
python arena.py --seek example_student --hide example_student \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --delay 0.3
```

### 4.4. Export Bài Nộp

```bash
python blind/scripts/export_submission.py team_submission --force
```

---

## 5. Thuật toán Gợi ý cho Partial Observability

| Thuật toán | Ứng dụng |
|------------|----------|
| **Minimax + Alpha-Beta** | Dự đoán đối thủ dưới thông tin không đầy đủ |
| **Monte Carlo Tree Search** | Mô phỏng ngẫu nhiên cho observation không chắc chắn |
| **Expectiminimax** | Kết hợp xác suất với adversarial search |
| **Belief State Tracking** | Duy trì phân phối xác suất vị trí đối thủ |
| **Frontier-based Exploration** | Khám phá vùng chưa biết để tìm đối thủ |
| **A\* with Memory** | Tìm đường trên bản đồ tích lũy (mental map) |

---

## 6. Ràng buộc Kỹ thuật

| Ràng buộc | Giá trị |
|-----------|--------|
| **Thời gian / step** | Tối đa 1.0 giây |
| **Bộ nhớ (RAM)** | Tối đa 128 MB |
| **Thư viện hợp lệ** | `numpy`, `pandas`, `scipy`, `gurobi`, `pytorch`, `scikit-learn` |
| **Python** | 3.11 |

---

## 7. Cách tính Tỷ lệ Thắng & Tie-Break

| Vai trò | Công thức |
|---------|-----------|
| **Ghost (Hider)** | `winrate_hide = Hide wins / Hide games` |
| **Pacman (Seeker)** | `winrate_seek = Seek wins / Seek games` |

**Tie-break:** Team có chênh lệch giữa số bước trung bình Pacman và Ghost thấp hơn sẽ xếp hạng cao hơn.

```
diff = avg_steps_pacman - avg_steps_ghost   # Càng thấp càng tốt
```
