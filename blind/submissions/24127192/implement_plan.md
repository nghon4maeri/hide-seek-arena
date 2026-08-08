# Implement Plan — Blind Ghost Agent V3 (Deterministic Mode)

## 📊 Phân tích đối thủ (14 Pacman agents)

### Các pattern Pacman phổ biến:
| Pattern | Teams | Cách đối phó |
|---------|-------|-------------|
| **Belief + Minimax AlphaBeta** | 6, 11, 12 | Đi theo vùng ngược lại, break LOS, camp corner |
| **BFS Pursuit + Visit Decay** | 1, 3, 5, 7, 8, 9, 10 | Chạy loop, patrol → Pacman chase theo loop vô ích |
| **Predictive Interception** | 10, 12 | Feint/đổi hướng bất ngờ, corridor evasion |
| **Frontier Exploration** | 10, 11, 13, 14, 15, 16 | Tránh vùng fog/unexplored, ở lại vùng đã clear |
| **Model-based Ensemble** | 12 | Đa dạng pattern → không để bị predict |

### Điểm yếu của từng Pacman mạnh:
| Team | Điểm yếu | Khai thác |
|------|---------|-----------|
| **6** | Belief tản mạn sau 6+ steps mất dấu → hunt thụ động | Khuất tầm nhìn > 6 steps → camp corner xa |
| **10** | Predict ghost chạy xa Pacman → có thể bị lừa | Đổi hướng bất ngờ (feint) |
| **11** | Minimax depth=6, mù khi ghost unseen | Tránh LOS, dùng patrol pattern |
| **12** | Heavy reliance on model prediction → predictable | Đa dạng pattern, feint scheduler |

---

## 🎯 V3 Strategy: 5-Phase Escape + Smart Patrol + Feint

### Phase 1: RIGHT+DOWN Escape (steps 1-11) — GIỮ NGUYÊN
```
(9,9) → RIGHT×6 → (9,15) → DOWN×5 → (14,15)
```
- Đi NGƯỢC hướng Pacman start (15,10): Pacman thường đi LEFT/UP đầu game
- Đến bottom-right area — rộng, nhiều loop, xa Pacman start

### Phase 2: Deep Bottom Navigation (steps 12-25) — **MỚI**
Sau escape, Pacman thường đã vào khu vực center/upper. Ghost tiếp tục đi SÂU xuống bottom:
```
(14,15) → RIGHT×4 → (14,19) → DOWN×4 → (18,19) → LEFT vào bottom loop
```
- Mục tiêu: vào bottom loop area trước khi Pacman đến
- Nếu Pacman detected: skip straight to patrol

### Phase 3: Figure-8 Loop Patrol (steps 25+) — **CẢI THIỆN**
- Chạy patrol pattern trong bottom area:
  - **Outer loop**: (14,15)→(14,19)→(19,19)→(19,1)→(14,1)→(14,7) ~44 cells
  - **Inner loop**: (14,7)→(17,7)→(17,13)→(14,13)→(14,7) ~20 cells
- Xen kẽ outer/inner mỗi 35 steps
- **Ambush detection**: Nếu Pacman gần (< 6 Manhattan) → reverse direction
- **Camp mode**: Nếu Pacman xa (> 25 Manhattan) → camp tại safe corner

### Phase 4: Zone Evasion (khi Pacman đến bottom) — **MỚI**
- Nếu Pacman centroid y > 14 (Pacman đã xuống bottom):
  - Escape lên upper qua col 15: (14,15) → UP×5 → (9,15) → LEFT vào upper loop
  - Chạy upper loop pattern
- Nếu Pacman centroid y < 7 (Pacman ở upper):
  - Ở lại bottom, tiếp tục patrol

### Phase 5: Anti-Prediction Feints — **MỚI**
- **Feint scheduler**: Mỗi 25-30 steps, thực hiện 1 nước đi giả:
  - Đi 1 bước theo hướng A, rồi quay lại hướng ngược
  - Làm nhiễu model-based predictors (team 12)
- **Pattern variation**: Cycle 3 styles:
  - Style A (patrol): chạy loop
  - Style B (camp): ẩn corner
  - Style C (wander): đi tự do trong bottom area

---

## 🏗️ Kiến trúc V3

### Giữ lại từ V2:
✅ Hardcoded map + KNOWN_LAYOUT_STR + PACMAN_START + GHOST_START  
✅ TopologyAnalyzer (junctions, dead_ends, corridors, core, loops, chokepoints, tunnels)  
✅ DistanceCache (BFS distance precomputation)  
✅ RIGHT+DOWN escape route (Phase 1)  
✅ PacmanTracker (belief state propagation)  
✅ 3-model ensemble (Markov1, ShortestPath, Interceptor) với online weight update  
✅ RiskEngine (time-expanded danger, survival margin)  
✅ SafetyShield (capture detection filter)  
✅ ZoneNavigator với Pacman-avoiding path  
✅ AntiLoop (RECENT_CELL_BAN=15, edge penalty, reversal penalty)  
✅ Forward momentum (MOMENTUM_BONUS, REVERSAL_PENALTY, TURN_PENALTY_90)  
✅ OnePlyPolicy + GreedyPolicy  
✅ MCRollout với CVaR evaluation  
✅ PatrolManager với bottom outer/inner loops + upper loop  

### Cải thiện:
| Component | V2 | V3 | Lý do |
|-----------|----|----|------|
| **ESCAPE_LEN** | 11 | 11 (giữ nguyên) | Escape route hiệu quả |
| **RECENT_CELL_BAN** | 15 | 20 | Mạnh hơn, giảm revisit |
| **REVERSAL_PENALTY** | 1200 | 1500 | Tránh quay đầu mạnh hơn |
| **MOMENTUM_BONUS** | 80 | 100 | Khuyến khích tiến thẳng |
| **CORRIDOR_REV_PEN** | 2000 | 2500 | Đặc biệt mạnh trong corridor |
| **PATROL_SCORE** | 600 | 700 | Ưu tiên patrol cao hơn |
| **CAMP_SCORE** | 350 | 450 | Ưu tiên camp cao hơn |
| **LOOP_SWITCH_STEPS** | 35 | 30 | Đổi loop thường xuyên hơn |
| **PATROL_ENTER_DIST** | 20 | 25 | Vào patrol mode sớm hơn |
| **MC_ROLLOUTS** | 12 | 16 | Rollout nhiều hơn → chính xác hơn |
| **MC_DEPTH** | 15 | 20 | Nhìn xa hơn |
| **DANGER_HORIZON** | 6 | 8 | Dự đoán danger xa hơn |
| **PANIC_DISTANCE** | 8 | 10 | Kích hoạt combat sớm hơn |
| **TIME_BUDGET** | 0.85 | 0.80 | An toàn hơn cho timeout |

### Thêm mới:
| Component | Mô tả |
|-----------|-------|
| **FeintScheduler** | Lên lịch feint mỗi 25-30 steps, xen kẽ 3 pattern styles |
| **DeepBottomNav** | Phase 2: navigation từ (14,15) → bottom loop entry |
| **UpperEscapeRoute** | Phase 4: escape từ bottom → upper khi Pacman xuống bottom |
| **EnhancedCamp** | Camp tại safe corners (corners có degree=2, exposure thấp, xa Pacman) |

---

## 🔄 Pipeline Mỗi Step (V3)

```
1. Update memory_map từ observation
2. AntiLoop.record(me)
3. PacmanTracker → belief (propagate nếu không thấy)
4. Ensemble → pac_preds (2-step prediction)
5. RiskEngine → danger_t0 (time-expanded)
6. Compute legal moves

7. [Phase 1] IF step ≤ ESCAPE_LEN AND pac=None:
     → OPTIMAL_ESCAPE[step-1] (score=10000)
     
8. [Phase 2] IF ESCAPE_LEN < step ≤ 25 AND pac=None:
     → DeepBottomNav: navigate to bottom loop entry (18,19) or (19,15)
     
9. [Tunnel] IF pac visible AND me in tunnel:
     → Tunnel escape

10. [Phase 4] IF Pacman centroid in bottom area (y > 14):
     → Upper escape route: navigate (14,15)→(9,15)→upper loop

11. [Phase 3/5] IF step > 25 OR pac visible:
     → FeintScheduler.check(): nếu đến lúc → feint move (score=550)
     → ZoneNavigator: navigate away from Pacman zone
     → PatrolManager: patrol move (score=700)
     → Nếu Pacman xa (dist > 25) → camp move (score=450)
     → Nếu Pacman rất xa (dist > 30) → enhanced camp (score=500)

12. OnePly + Greedy policies → candidates
13. SafetyShield.filter() → lọc an toàn
14. Anti-LOS adjustment nếu Pacman visible
15. Validate + return move
```

---

## 📐 Hardcoded Cache Mở Rộng

### Deep Bottom Navigation Route:
```python
DEEP_BOTTOM_ROUTE: Tuple[Move, ...] = (
    # Từ (14,15) → (14,19) → (18,19) → vào bottom loop
    Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT,  # → (14,19)
    Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN,       # → (18,19)
    Move.LEFT,                                         # → (18,18) vào loop
)
```

### Upper Escape Route (khi Pacman xuống bottom):
```python
UPPER_ESCAPE_ROUTE: Tuple[Move, ...] = (
    # Từ bottom loop → (14,15) → (9,15) → upper
    Move.UP, Move.UP, Move.UP, Move.UP, Move.UP,  # → (9,15) nếu start từ (14,15)
    Move.LEFT,                                      # → (9,14)
)
```

### Safe Camps (precomputed corners):
```python
SAFE_CAMPS: Tuple[Pos, ...] = (
    (19, 1),   # Bottom-left corner
    (19, 19),  # Bottom-right corner  
    (1, 1),    # Top-left corner
    (1, 19),   # Top-right corner
    (14, 7),   # Inner loop corner
    (14, 13),  # Inner loop corner
)
```

### Junction Distances (precomputed):
- Precompute all-pairs shortest path between junctions
- Lưu trong dict: `{(j1, j2): [Move, ...]}`
- Dùng để chọn đường đi nhanh nhất giữa các vùng

---

## 🎲 Feint Scheduler

```python
class FeintScheduler:
    FEINT_INTERVAL = 27  # steps between feints
    STYLES = ["patrol", "camp", "wander"]
    
    def check(self, step_number, ghost, pac_est, legal) -> Optional[Move]:
        if step_number % self.FEINT_INTERVAL != 0:
            return None
        style = self.STYLES[(step_number // self.FEINT_INTERVAL) % 3]
        if style == "patrol":
            return None  # patrol manager handles this
        elif style == "camp":
            return self._camp_feint(ghost, legal)
        elif style == "wander":
            return self._wander_feint(ghost, legal, pac_est)
```

---

## 📊 Constants V3

```python
# Escape
ESCAPE_LEN = 11
DEEP_BOTTOM_END = 25

# Anti-Loop
RECENT_CELL_BAN = 20
REVERSAL_PENALTY = 1500
MOMENTUM_BONUS = 100
CORRIDOR_REV_PEN = 2500

# Patrol
PATROL_SCORE = 700
CAMP_SCORE = 450
PATROL_ENTER_DIST = 25
LOOP_SWITCH_STEPS = 30
LOOP_SWITCH_STEPS_UPPER = 25

# MC
MC_ROLLOUTS = 16
MC_DEPTH = 20
DANGER_HORIZON = 8
PANIC_DISTANCE = 10

# Feint
FEINT_INTERVAL = 27
FEINT_SCORE = 550

# Time
TIME_BUDGET = 0.80
```

---

## 📅 Implementation Sequence

### Bước 1: Cập nhật constants + AntiLoop
- Tăng RECENT_CELL_BAN → 20
- Tăng REVERSAL_PENALTY → 1500, MOMENTUM_BONUS → 100
- Tăng CORRIDOR_REV_PEN → 2500
- Tăng MC_ROLLOUTS → 16, MC_DEPTH → 20

### Bước 2: Thêm Deep Bottom Navigation (Phase 2)
- Hardcode DEEP_BOTTOM_ROUTE
- Tích hợp vào GhostAgent.step(): steps 12-25

### Bước 3: Thêm Upper Escape Route (Phase 4)
- Hardcode UPPER_ESCAPE_ROUTE  
- Detect Pacman centroid in bottom → trigger escape
- Navigate từ bottom → upper area

### Bước 4: Enhanced Camp
- Hardcode SAFE_CAMPS
- Camp selection dựa trên distance to Pacman, exposure, degree
- Khi Pacman xa (> 25 Manhattan) → camp mode

### Bước 5: Feint Scheduler
- Implement FeintScheduler class
- Tích hợp feint moves vào candidate pool

### Bước 6: Tích hợp + Test
- Tích hợp tất cả phases vào GhostAgent.step()
- Benchmark vs 14 opponents
- Điều chỉnh constants

### Bước 7: Benchmark Script
- Cập nhật benchmark.py để chạy 1 game/opponent (deterministic)
- Output: steps per game + average

---

## 🎯 Mục Tiêu

| Metric | V2 | V3 Target |
|--------|----|-----------| 
| Overall Avg Steps | ~67 | **80+** |
| Wins (200 steps) | ~2 opponents | **5-6 opponents** |
| vs Team 6 | ~9 steps | **30+ steps** |
| vs Team 12 | ~5 steps | **25+ steps** |
| vs Team 11 | ~40 steps | **60+ steps** |
| Survival Rate | ~14% | **30%+** |

---

## ⚠️ Ràng buộc
- CHỈ sửa file trong `submissions/24127192/`
- CHỈ dùng deterministic mode (--start-mode deterministic)
- Vision=5, capture_distance=2, pacman_speed=2
- Phản hồi < 0.9s mỗi step
- Các thay đổi phải qua contact.md để được duyệt
