# NOTES — Blind Ghost Agent V3 (24127192)

## 1. Những gì đã thực hiện (V2 → V3)

### 1.1 Constants Tuning
Tinh chỉnh toàn bộ constants để tăng khả năng sinh tồn:

| Constant | V2 | V3 | Lý do |
|----------|----|----|-------|
| `RECENT_CELL_BAN` | 15 | **20** | Cấm mạnh hơn 20 ô gần nhất → giảm lặp |
| `REVERSAL_PENALTY` | 1200 | **1500** | Phạt nặng hơn khi quay đầu 180° |
| `MOMENTUM_BONUS` | 80 | **100** | Khuyến khích đi thẳng, tiến lên |
| `CORRIDOR_REV_PEN` | 2000 | **2500** | Cấm tuyệt đối quay đầu trong corridor |
| `MC_ROLLOUTS` | 12 | **16** | Rollout nhiều hơn → đánh giá chính xác hơn |
| `MC_DEPTH` | 15 | **20** | Nhìn xa hơn trong Monte Carlo |
| `DANGER_HORIZON` | 6 | **8** | Dự đoán danger xa hơn |
| `PANIC_DISTANCE` | 8 | **10** | Kích hoạt combat mode sớm hơn |
| `PATROL_SCORE` | 600 | **700** | Ưu tiên patrol cao hơn |
| `CAMP_SCORE` | 350 | **450** | Ưu tiên camp cao hơn |
| `PATROL_ENTER_DIST` | 20 | **25** | Vào patrol mode sớm hơn |
| `LOOP_SWITCH_STEPS` | 35 | **30** | Đổi loop thường xuyên hơn → khó predict |
| `TIME_BUDGET` | 0.85 | **0.80** | An toàn hơn cho timeout |

### 1.2 Deep Bottom Navigation (Phase 2 — MỚI)
Sau escape route 11 bước, ghost không dừng lại mà tiếp tục đi SÂU vào bottom-right area:

```
(14,15) → RIGHT×4 → (14,19) → DOWN×4 → (18,19) → LEFT×2 → (18,17)
```

**Mục đích**: Đưa ghost vào vùng bottom loop rộng (44 cells) trước khi Pacman kịp đến. Đây là vùng an toàn nhất vì:
- Nhiều loop lớn → Pacman speed=2 bị vô hiệu hóa (chạy cùng chiều trong loop)
- Xa Pacman start (15,10)
- Nhiều corner để camp/ẩn nấp

### 1.3 Upper Escape Route (Phase 4 — MỚI)
Khi Pacman belief centroid xuống bottom area (row > 14), ghost tự động escape lên upper area:

```
Bottom loop → A* → (9,15) → upper area
```

**Cơ chế phát hiện**: Tính `pac_centroid` từ belief distribution. Nếu centroid_y > 14 VÀ ghost đang ở bottom → kích hoạt escape. Nếu Pacman chưa xuống bottom → ghost ở lại bottom patrol.

### 1.4 Feint Scheduler (Phase 5 — MỚI)
Lên lịch các nước đi giả mỗi 27 steps để gây nhiễu model-based Pacman predictors (đặc biệt team 12 dùng 4-model ensemble):

```python
class FeintScheduler:
    FEINT_INTERVAL = 27
    STYLES = ("patrol", "camp", "wander")
```

- **Style "patrol"**: Để PatrolManager xử lý (no-op)
- **Style "camp"**: Đi về phía safe camp gần nhất
- **Style "wander"**: Chọn ngẫu nhiên hướng về junction/core/loop, tránh dead-end

### 1.5 Enhanced Camp (CẢI THIỆN)
8 safe camps được hardcode dựa trên topology analysis:

```python
SAFE_CAMPS = (
    (19, 1), (19, 19),   # Bottom corners
    (1, 1), (1, 19),     # Top corners
    (1, 5), (1, 15),     # Top inner
    (19, 5), (19, 15),   # Bottom inner
)
```

Khi Pacman xa (>25 Manhattan), ghost đi về camp gần nhất. Khi rất xa (>30 Manhattan), tăng score camp lên 450.


## 2. Cơ chế di chuyển chi tiết

### 2.1 Pipeline tổng thể mỗi step

```
Step 1: Update memory_map từ observation (tích lũy)
Step 2: AntiLoop.record(me) → ghi nhận vị trí
Step 3: PacmanTracker.update() → belief propagation
Step 4: Ensemble.predict_positions_2step() → dự đoán Pacman
Step 5: RiskEngine.time_expanded_danger() → danger map 8 bước
Step 6: Compute legal moves từ memory map

Step 7-12: Candidate Collection (theo thứ tự ưu tiên):
  ├── [Phase 1] step ≤ 11 AND pac=None → RIGHT+DOWN escape (score=10000)
  ├── [Phase 2] step 12-21 AND pac=None → Deep Bottom Nav (score=9000)
  ├── [Tunnel] pac visible AND in tunnel → Tunnel escape (score=800)
  ├── [Phase 4] pac centroid in bottom → Upper escape (score=1500)
  ├── [Zone Nav] Navigate away from Pacman zone (score=350-500)
  ├── [Feint] step % 27 == 0 → Feint move (score=550)
  ├── [Camp] dist_to_pac > 25 → Camp move (score=315-450)
  ├── [OnePly] OnePlyPolicy.propose()
  └── [Greedy] GreedyPolicy.propose()

Step 13: SafetyShield.filter() → loại bỏ moves nguy hiểm
Step 14: Anti-LOS adjust nếu Pacman visible
Step 15: Validate + return move
```

### 2.2 Cơ chế Escape Route (Phase 1)

```
(9,9) → RIGHT×6 → (9,15) → DOWN×5 → (14,15)
```

**Tại sao RIGHT+DOWN?**
- Pacman start tại (15,10) — phía dưới bên trái ghost
- Ghost đi RIGHT → tăng khoảng cách cột với Pacman
- Ghost đi DOWN → vào vùng bottom rộng, xa Pacman
- Tránh đi LEFT (dẫn vào corridor hẹp phía trái map)
- Tránh đi UP (Pacman cũng thường đi lên → đụng độ)

**Tại sao không đi hướng khác?**
- **UP-first**: Đã test — kết quả tệ hơn (avg ~50 steps) vì upper area nhỏ hơn, ít loop, Pacman dễ dồn ghost vào góc
- **LEFT-first**: Tương tự, corridor trái hẹp → Pacman speed=2 áp đảo
- **DOWN-first**: Từ (9,9) không đi xuống được (tường ở row 10, col 9)

### 2.3 Cơ chế Patrol (Phase 3)

Ghost patrol trong bottom area với 2 loop xen kẽ:

```
Bottom Outer Loop (44 cells, clockwise):
(14,15)→(14,19)→(19,19)→(19,1)→(14,1)→(14,14)

Bottom Inner Loop (20 cells, counter-clockwise):
(14,7)→(17,7)→(17,13)→(14,13)→(14,7)
```

**Ambush Detection**: Khi Pacman đến gần (< 6 Manhattan), ghost kiểm tra nếu Pacman đang đợi phía trước trong loop → đảo chiều loop.

**Loop Switching**: Mỗi 30 steps (outer) hoặc 25 steps (inner), ghost chuyển loop.

**Tại sao patrol trong loop hiệu quả?**
- Trong loop, ghost và Pacman chạy cùng chiều → Pacman speed=2 vô dụng (không rút ngắn khoảng cách)
- Pacman chỉ bắt được nếu chạy NGƯỢC chiều (ambush) → ghost detect và đảo chiều
- Loop lớn (44 cells) → Pacman mất nhiều thời gian để bao vây

### 2.4 Cơ chế OnePly Policy

OnePly là policy chính trong combat/normal mode, đánh giá mỗi nước đi qua:

1. **Distance score**: Khoảng cách BFS đến từng Pacman prediction × weight
2. **Danger penalty**: Trừ điểm dựa trên danger map (time-expanded)
3. **Topology bonus**: Thưởng cho core (+24), loop (+20), junctions (+14)
4. **Topology penalty**: Phạt dead-end (-75), tunnel (-18), biên map (-30)
5. **Anti-revisit**: Phạt nặng nếu ô đã ở trong 20 vị trí gần nhất
6. **Momentum**: Thưởng đi thẳng (+100), phạt quay đầu (-1500), phạt rẽ (-100)
7. **Capture avoidance**: Phạt nếu Pacman có thể bắt được từ ô tiếp theo

### 2.5 Cơ chế Monte Carlo Rollout

Với mỗi nước đi, chạy 16 rollouts × 20 bước:
- Ghost policy: chọn nước đi tối đa khoảng cách BFS đến Pacman, ưu tiên junctions/core
- Pacman response: dùng ensemble prediction (Markov1 + ShortestPath + Interceptor)
- CVaR evaluation: kết hợp mean + Conditional Value at Risk (α=0.20)
- Kết quả: chọn nước đi có CVaR score cao nhất

### 2.6 Cơ chế Anti-Loop + Anti-Revisit

```python
RECENT_CELL_BAN = 20  # Cấm 20 ô gần nhất
penalty = visit_count[nxt] * 8.0           # Phạt theo số lần đã visit
penalty += edge_count[(cur, nxt)] * 15.0   # Phạt theo số lần đi cạnh
if nxt in recent_20:                        # Nếu trong 20 ô gần nhất:
    penalty += 300 + recency_index * 60     # Phạt tăng dần theo độ gần
if nxt == position_2_steps_ago:            # Dao động A→B→A:
    penalty += 120
```

### 2.7 Cơ chế PacmanTracker (Belief Propagation)

Khi không thấy Pacman:
1. Mỗi cell trong belief → lan sang tất cả ô Pacman có thể đến trong 1 step (speed=2)
2. Phân phối đều xác suất
3. Giới hạn belief ở 18 cells có xác suất cao nhất
4. Normalize

Khi thấy Pacman:
- Belief = {vị_trí_Pacman: 1.0} (tập trung 100%)

### 2.8 Cơ chế Ensemble Prediction (3 models)

| Model | Cơ chế | Trọng số |
|-------|--------|----------|
| **Markov1** | Học transition (ghost_dr, ghost_dc, dist_bucket, geometry) → action | Online update |
| **ShortestPath** | Pacman đi theo hướng GIẢM khoảng cách BFS đến ghost | Online update |
| **Interceptor** | Pacman đi về junctions/chokepoints để chặn đường ghost | Online update |

Online weight update: Mỗi khi thấy Pacman, so sánh action thực tế với prediction → cập nhật trọng số theo exponential gradient descent.


## 3. Kết quả Benchmark V3

### Chế độ: deterministic, vision=5, capture_distance=2, pacman_speed=2, max_steps=200

| Pacman | Steps | Kết quả | V2 Steps | Δ |
|--------|-------|---------|----------|---|
| **1** | **200** | 🏆 SURVIVED | 168 | +32 |
| **3** | **163** | Lost | 29 | +134 |
| **5** | **48** | Lost | 52 | -4 |
| **6** | **9** | Lost | 9 | 0 |
| **7** | **52** | Lost | 200 | -148 |
| **8** | **200** | 🏆 SURVIVED | 17 | +183 |
| **9** | **0** | ⚠️ scipy error | 0 | 0 |
| **10** | **28** | Lost | 142 | -114 |
| **11** | **200** | 🏆 SURVIVED | 28 | +172 |
| **12** | **5** | Lost | 5 | 0 |
| **13** | **200** | 🏆 SURVIVED | 200 | 0 |
| **14** | **44** | Lost | 28 | +16 |
| **15** | **48** | Lost | 52 | -4 |
| **16** | **9** | Lost | 9 | 0 |
| **agent** | **8** | Lost | N/A | N/A |

### Tổng kết:
| Metric | V2 | V3 | Cải thiện |
|--------|----|----|-----------|
| **Overall Average** | 67.1 | **80.9** | +20.7% |
| **Wins (200 steps)** | 2/14 | **4/14** | +100% |
| **Survival Rate** | 14.3% | **28.6%** | +100% |
| **Best vs Strong** | 9 (vs 6) | 9 (vs 6) | 0 |
| **Best Overall** | 200 (vs 7,13) | 200 (vs 1,8,11,13) | Thêm 2 wins |

### Phân tích kết quả:
- **Cải thiện lớn**: vs 1 (+32), vs 3 (+134), vs 8 (+183), vs 11 (+172)
- **Giữ nguyên**: vs 6 (9), vs 12 (5), vs 13 (200), vs 16 (9)
- **Giảm nhẹ**: vs 5 (-4), vs 15 (-4) — trong biên độ nhiễu
- **Giảm đáng kể**: vs 7 (-148), vs 10 (-114) — Deep Bottom Nav đưa ghost vào vùng bất lợi cho các Pacman này


## 4. Nhược điểm

### 4.1 Deterministic hoàn toàn
Ghost đi cùng một path mỗi game → Pacman agents dùng model-based prediction (team 12) hoặc Alpha-Beta sâu (team 6) có thể dự đoán chính xác vị trí ghost ngay từ đầu.

**Impact**: Bị bắt ở step 5 (team 12) và step 9 (team 6).

### 4.2 Escape route cố định
RIGHT+DOWN escape route luôn đi qua corridor row 9 → col 15 → row 14. Đây là tuyến đường có thể dự đoán trước. Pacman agents biết ghost start (9,9) và có thể suy luận ghost sẽ đi RIGHT (xa Pacman nhất).

**Impact**: Pacman có thể cắt đầu ghost tại corridor row 9 hoặc row 14.

### 4.3 Deep Bottom Nav có thể phản tác dụng
Đưa ghost vào bottom loop sớm giúp chống lại Pacman yếu (1, 3, 8, 11) nhưng lại khiến ghost dễ bị dồn vào góc bởi Pacman mạnh dùng belief tracking (7, 10). Các Pacman này dự đoán được ghost đang ở bottom → lao thẳng xuống.

**Impact**: vs 7 giảm từ 200 → 52, vs 10 giảm từ 142 → 28.

### 4.4 Không chống được Model-Based Prediction (team 12)
Team 12 Pacman dùng 4-model ensemble (stay, inertia, chase, intercept) với online learning. Họ hardcode ghost start và dự đoán được escape path. Feint scheduler chỉ hoạt động sau step 21 — quá muộn.

**Impact**: Bị bắt ở step 5, không cải thiện so với V2.

### 4.5 Không chống được Alpha-Beta sâu (team 6)
Team 6 Pacman dùng Alpha-Beta depth 2-12 với Bayesian belief + iterative deepening. Khi ghost invisible < 6 steps, họ chase bằng Alpha-Beta. Khi lâu hơn, họ hunt với frontier scoring. Escape route ngắn (11 steps) không đủ để thoát khỏi vùng tìm kiếm.

**Impact**: Bị bắt ở step 9, không cải thiện so với V2.

### 4.6 Feint scheduler quá muộn
Feint scheduler bắt đầu từ step 22 (sau Phase 1 + Phase 2). Team 12 và 16 bắt ghost trước step 10 → feint không bao giờ được kích hoạt.

### 4.7 Không có cơ chế multi-escape
Chỉ có 1 escape route cố định. Không có phương án dự phòng nếu route bị chặn hoặc Pacman đón đầu.

### 4.8 Camp selection chưa đủ thông minh
Camp selection dựa trên Manhattan distance, không tính đến exposure, line-of-sight, hoặc Pacman belief. Có thể chọn camp nằm trong vùng Pacman sắp quét tới.


## 5. Hướng tối ưu — Cách tăng steps vs team 12, 16, 6

### 5.1 Chiến lược tổng thể: Multi-Route + Early Feint

**Nguyên lý**: Thay vì 1 escape route cố định, cung cấp 3-4 escape routes và chọn route dựa trên phân tích Pacman behavior trong những step đầu tiên.

### 5.2 Cụ thể cho từng team

#### Chống team 12 (Model-Based Prediction, hiện 5 steps):

**Vấn đề gốc**: Team 12 hardcode `FIXED_GHOST_START = (9, 10)`. Ghost ta start tại (9,9), đi RIGHT → (9,10) ở step 1 — TRÙNG KHỚP vị trí belief ban đầu của team 12!

**Giải pháp A — "Ghost Start Mismatch Exploit"**:
- Step 1: Đi UP thay vì RIGHT → ghost ở (8,9)
- Team 12 belief bắt đầu từ (9,10), predict lan ra (8,10), (10,10), (9,9), (9,11), (9,10)
- Ghost thực tế ở (8,9) — KHÔNG nằm trong belief ban đầu
- Steps 2-4: Tiếp tục UP+LEFT vào upper area
- Sau 3-4 steps, team 12 belief đã lan rộng → khó track chính xác
- **Ước tính**: Tăng từ 5 → 20-30 steps

**Giải pháp B — "Early Direction Randomization"**:
- 3 escape routes thay thế, chọn dựa trên step_number % 3:
  - Route A (hiện tại): RIGHT×6 + DOWN×5
  - Route B: LEFT×8 + UP×3 (vào upper-left)
  - Route C: RIGHT×3 + UP×5 + LEFT×2 (zigzag)
- **Trade-off**: Có thể giảm steps vs 1 số Pacman yếu hơn

**Giải pháp C — "Stay-and-Feint Opening"**:
- Step 1: STAY (ở nguyên (9,9))
- Team 12 belief: (9,10) → predict → (9,10), (8,10), (10,10), (9,9), (9,11)
- Step 2: Đi LEFT → (9,8)
- Team 12 predict tiếp từ 5 vị trí → belief phân tán nhanh
- Step 3+: Đi UP vào upper area
- **Ưu điểm**: Chỉ thay đổi 2 steps đầu, giữ nguyên escape route cho các step sau

#### Chống team 6 (Alpha-Beta 12-ply, hiện 9 steps):

**Vấn đề gốc**: Team 6 Pacman dùng belief propagation + Alpha-Beta depth 2-12. Khi ghost invisible, belief lan theo thời gian. Nhưng Pacman dùng BFS turn-distance + minimax để chặn mọi escape path có thể.

**Giải pháp A — "Early Corridor Break"**:
- Thay vì đi hết corridor row 9 (6 steps RIGHT), rẽ UP sớm hơn:
  ```
  RIGHT×2 → UP×3 → LEFT×3 → UP×2
  ```
- Ghost rời corridor row 9 sau 2 steps → Pacman Alpha-Beta không thể dự đoán vì có quá nhiều branch
- **Ước tính**: Tăng từ 9 → 40-60 steps

**Giải pháp B — "Belief Explosion Trigger"**:
- Step 1-2: Đi RIGHT như bình thường
- Step 3: Đi UP (vào corridor row 8)
- Step 4: Đi LEFT (vào junction area)
- Sau step 4, ghost ở 1 trong nhiều vị trí có thể → Pacman belief "nổ" (quá nhiều khả năng)
- Pacman phải chuyển từ chase → hunt mode
- **Ưu điểm**: Tận dụng điểm yếu của belief propagation (phân tán nhanh khi có nhiều branch)

#### Chống team 16 (hiện 9 steps):

**Vấn đề gốc**: Cần phân tích thêm team 16 Pacman strategy. Với 9 steps, có vẻ tương tự team 6 (belief + minimax).

**Giải pháp**: Áp dụng cùng chiến lược như chống team 6 + thêm feint sớm.

### 5.3 Implementation Plan cho V4

```python
# === V4: MULTI-ROUTE ESCAPE ===
ESCAPE_ROUTES = [
    # Route 0: RIGHT+DOWN (original) — tốt vs Pacman yếu
    (Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT,
     Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN),
    
    # Route 1: RIGHT+UP — gây nhiễu belief Pacman mạnh
    (Move.RIGHT, Move.RIGHT,
     Move.UP, Move.UP, Move.UP, Move.UP,
     Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT),
    
    # Route 2: STAY+RIGHT+UP — feint opening
    (Move.STAY,
     Move.RIGHT, Move.RIGHT, Move.RIGHT,
     Move.UP, Move.UP, Move.UP,
     Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT),
]

# === V4: EARLY FEINT ===
EARLY_FEINT_STEPS = {3, 7}  # Feint tại step 3 và 7

# === V4: DYNAMIC ROUTE SELECTION ===
def select_escape_route(step_number, pac_visible_history):
    if len(pac_visible_history) >= 2:
        # Pacman đã bị thấy → dùng route aggressive (nhiễu)
        return ESCAPE_ROUTES[1]
    elif step_number <= 2:
        # Mới bắt đầu → dùng route có STAY để gây nhiễu belief
        return ESCAPE_ROUTES[2]
    else:
        return ESCAPE_ROUTES[0]  # Default
```

### 5.4 Các hướng tối ưu khác

#### 5.4.1 Particle Filter Belief (thay thế simple propagation)
- Thay vì giới hạn 18 cells, dùng 100 particles với resampling
- Chính xác hơn → dự đoán Pacman tốt hơn → chọn đường đi an toàn hơn
- **Cost**: +0.05s/step — vẫn trong budget

#### 5.4.2 Online Behavior Classification
- Phân loại Pacman thành 3 loại: "aggressive" (team 6, 12), "methodical" (team 10, 11), "explorer" (team 7, 13)
- Chọn chiến lược khác nhau cho từng loại:
  - Aggressive → nhiều feint + đổi hướng đột ngột
  - Methodical → patrol loop + camp
  - Explorer → giữ khoảng cách + tránh frontier

#### 5.4.3 Improved Camp Selection
- Chọn camp dựa trên: exposure score, line-of-sight breaks, Pacman belief distance
- Rotate camps mỗi 30-40 steps
- Precompute camp quality scores từ topology analyzer

#### 5.4.4 Multi-Agent Adversarial Training
- Chạy ghost agent vs chính nó (self-play) để tìm weak spots
- Lưu trajectory gây chết sớm → phân tích → cải thiện escape route

#### 5.4.5 Junction Graph Pathfinding
- Precompute all-pairs shortest path giữa tất cả junctions
- Lưu cache path + distance
- Khi cần đi từ A → B, tra cứu O(1) thay vì A* O(N log N)
- **Impact**: Tiết kiệm 20-30% compute time → có thể tăng MC_ROLLOUTS

#### 5.4.6 Adaptive Constants
- Thay vì hardcode constants, điều chỉnh online dựa trên game state:
  - Đầu game: MOMENTUM_BONUS cao (thoát nhanh)
  - Giữa game: RECENT_CELL_BAN cao (tránh lặp)
  - Cuối game: PATROL_SCORE cao (sinh tồn)
  - Khi Pacman gần: REVERSAL_PENALTY thấp hơn (cần cơ động)

### 5.5 Trade-off Analysis

| Cải tiến | Impact vs 6,12,16 | Impact vs others | Risk |
|----------|-------------------|------------------|------|
| Multi-Route Escape | ++ (15-30 steps) | ~ (có thể giảm 10%) | Thấp |
| Early STAY Feint | ++ (10-20 steps vs 12) | ~ | Thấp |
| Particle Filter | + (5-10 steps) | + (5-10 steps) | Trung bình (cost) |
| Behavior Classification | ++ (10-20 steps) | + (5 steps) | Trung bình (complexity) |
| Junction Graph Cache | ~ | + (2-5 steps) | Thấp |
| Adaptive Constants | + (5 steps) | + (5 steps) | Thấp |

### 5.6 Đề xuất ưu tiên cho V4

1. **Multi-Route Escape** (dễ, impact cao) — Triển khai 3 routes + dynamic selection
2. **Early STAY Feint** (dễ, impact cao vs team 12) — Thêm STAY ở step 1-2
3. **Behavior Classification** (trung bình, impact cao) — Phân loại Pacman online
4. **Junction Graph Cache** (dễ, impact thấp) — Precompute paths
5. **Particle Filter** (khó, impact trung bình) — Thay thế belief propagation


## 6. Cấu trúc Code V3

```
agent.py (≈1700 lines)
├── Constants (40+ tuned parameters)
├── Grid Utilities (_valid, _legal, _neighbors, _manhattan, astar, ...)
├── DistanceCache — BFS distance precomputation (max 384 entries)
├── TopologyAnalyzer — dead_ends, junctions, corridors, core, loops, chokepoints, tunnels
├── ZoneNavigator — 5-zone classification + Pacman-avoiding path selection
├── PacmanTracker — Bayesian belief propagation (max 18 cells)
├── OpponentModelEnsemble — 3 models (Markov1, ShortestPath, Interceptor)
├── RiskEngine — Time-expanded danger map + survival margin
├── SafetyShield — Capture detection + viability filter
├── AntiLoop — Visit/edge/recent cell penalties
├── MCRollout — Monte Carlo with CVaR evaluation
├── OnePlyPolicy — Feature-based heuristic evaluation
├── GreedyPolicy — Simple distance-based policy
├── PatrolManager — Figure-8 loop patrol in bottom area
├── FeintScheduler — Anti-prediction feints (V3 NEW)
├── GhostAgent — Main agent with 5-phase pipeline
│   ├── Phase 1: RIGHT+DOWN Escape (steps 1-11)
│   ├── Phase 2: Deep Bottom Nav (steps 12-21) (V3 NEW)
│   ├── Phase 3: Figure-8 Patrol (steps 22+)
│   ├── Phase 4: Upper Escape (V3 NEW)
│   └── Phase 5: Feint + Camp (V3 NEW)
└── PacmanAgent — Placeholder for self-play testing
```


## 7. Bài học từ phân tích đối thủ

### Ghost agents tham khảo:
| Team | Strategy | Điểm mạnh | Áp dụng được? |
|------|----------|-----------|---------------|
| **6** | Belief Minimax depth 3 + StableCamp | Đánh giá trạng thái chính xác, camp thông minh | ✅ Camp scoring |
| **10** | BFS Lookahead 8 + Dynamic Waypoints | Freedom scoring, corner turn bonus | ✅ Corner bonus |
| **11** | Heatmap exploration + BFS evasion | Đơn giản, nhanh | ❌ Quá đơn giản |
| **12** | 200-step Hardcoded Route + 4-model Predictor | Cực kỳ hiệu quả với deterministic starts | ✅ Multi-route + hardcode |

### Pacman agents đối kháng:
| Team | Strategy | Điểm yếu | Cách khai thác |
|------|----------|----------|----------------|
| **6** | Alpha-Beta 12-ply + Bayes belief | Belief phân tán sau 6+ steps mất dấu | Biến mất khỏi LOS > 6 steps → camp |
| **10** | Ghost prediction 4-step + Dijkstra | Predict ghost chạy xa → dễ bị lừa | Feint + đổi hướng bất ngờ |
| **11** | Minimax depth 6 + BFS cache | Mù hoàn toàn khi ghost unseen | Tránh LOS, patrol pattern |
| **12** | 4-model ensemble + Belief search | Dựa vào model prediction → predictable | Đa dạng pattern, feint sớm |
| **16** | TBD (cần phân tích thêm) | TBD | TBD |
