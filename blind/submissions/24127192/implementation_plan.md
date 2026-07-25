# Implementation Plan — Blind Multi-Layer Ghost Agent (24127192)

## 🎯 Mục tiêu

Xây dựng Blind Ghost Agent với kiến trúc **multi-layer** dựa trên `non_blind_agent.py`, điều chỉnh cho:
- **Blind Mode**: vision = 5 (cross-shaped partial observability)
- **Fixed Starts**: Pacman tại (15, 10), Ghost tại (9, 9) trên hardcoded layout
- **Response time < 0.9s** mỗi step

---

## 🏗️ Kiến trúc Multi-Layer (8 Layers)

Dựa trên non_blind_agent.py, điều chỉnh cho Blind Mode:

### Layer 0: Hardcoded Map + Precomputed Topology
- **Mục đích**: Thay vì fingerprint map từ observation, hardcode trực tiếp layout đã biết
- **Cài đặt**:
  - `KNOWN_LAYOUT`: hardcode string list → numpy array 21×21
  - Precompute một lần trong `__init__`:
    - `TopologyAnalyzer`: dead_ends, junctions, corridors, core, loops, chokepoints, tunnels
    - `DistanceCache`: BFS precompute cho tất cả key cells (junctions + chokepoints)
    - `OfflineTable`: Lookup table cho hướng di chuyển tối ưu
- **Khác biệt với non_blind**: Không cần MapRepository (fingerprint), dùng map cố định

### Layer 1: Memory Map + Observation Accumulation
- **Mục đích**: Tích lũy thông tin quan sát qua các step (Mental Map)
- **Cài đặt**:
  - `memory_map`: numpy array 21×21, khởi tạo toàn -1 (unseen)
  - Mỗi step: cập nhật các ô visible (≠ -1) từ observation vào memory_map
  - Các thuật toán A*, BFS, Flood Fill đều chạy trên memory_map
  - Tận dụng hardcoded map để "fill" các vùng chưa thấy nếu map khớp

### Layer 2: Pacman Belief State Tracking
- **Mục đích**: Dự đoán phân phối xác suất vị trí Pacman khi không thấy
- **Cài đặt**:
  - `PacmanTracker`: duy trì `belief: Dict[Pos, float]`
  - Khi thấy Pacman: belief = {vị_trí: 1.0}
  - Khi không thấy: propagate belief qua `_pacman_reach()` với speed=2
  - Prune về top 18 cells (BELIEF_MAX_CELLS)
  - `best_estimate`: vị trí có xác suất cao nhất
  - **Khác biệt với non_blind**: Khởi tạo belief ban đầu tại (15, 10) là vị trí fixed start của Pacman

### Layer 3: Opponent Model Ensemble (6 Models)
- **Mục đích**: Dự đoán nước đi tiếp theo của Pacman
- **Cài đặt** (port từ non_blind, chạy trên memory_map):
  - `MarkovOrder1`: Dựa trên relative position (dr, dc, distance bucket, geometry)
  - `MarkovOrder2`: Dựa trên action trước + relative position
  - `ShortestPathModel`: Pacman đi theo đường ngắn nhất đến Ghost
  - `InterceptionModel`: Pacman ưu tiên junctions, chokepoints
  - `RandomLegalModel`: Fallback random
  - `AdversarialModel`: Minimax 1-ply
  - Cập nhật weights thích nghi theo exponential loss
  - **Khác biệt với non_blind**: Models chạy trên memory_map, xử lý unseen cells

### Layer 4: Risk Engine
- **Mục đích**: Đánh giá mức độ nguy hiểm theo thời gian
- **Cài đặt** (port từ non_blind):
  - `time_expanded_danger()`: Tính danger map qua DANGER_HORIZON=3 steps
  - `survival_margin()`: Khoảng cách an toàn giữa Ghost và Pacman
  - `is_viable()`: Kiểm tra vị trí có khả năng sống sót
  - **Khác biệt với non_blind**: Chạy trên memory_map, xử lý belief state thay vì vị trí chính xác

### Layer 5: Monte Carlo Rollout (MC V3)
- **Mục đích**: Mô phỏng nhiều kịch bản để đánh giá chất lượng nước đi
- **Cài đặt** (port từ non_blind):
  - `MC_ROLLOUTS = 10`, `MC_DEPTH = 15`
  - Ghost policy: ưu tiên distance + exits + topology (core, loop, junction)
  - Pacman response: dựa trên ensemble prediction
  - CVaR scoring (bottom 20% returns)
  - **Khác biệt với non_blind**: Chạy simulation trên memory_map (có thể có unseen cells)

### Layer 6: Alpha-Beta Search
- **Mục đích**: Tìm kiếm chiến thuật minimax khi Pacman ở gần
- **Cài đặt** (port từ non_blind):
  - `AB_MAX_DEPTH = 7`, iterative deepening
  - Transposition table (max 40K entries)
  - Ghost moves ordered by distance to Pacman
  - Pacman moves dựa trên ensemble prediction
  - **Khác biệt với non_blind**: Chạy trên memory_map

### Layer 7: 5-Layer Policy Portfolio
- **Mục đích**: Đa dạng chiến lược, chọn policy phù hợp theo tình huống
- **Cài đặt** (port từ non_blind):
  - `HybridPolicy` (Layer 0): MC + Markov + OfflineTable (cho close combat)
  - `TablePolicy` (Layer 1): Offline table lookup + safety check
  - `AlphaBetaPolicy` (Layer 2): Iterative deepening alpha-beta
  - `OnePlyPolicy` (Layer 3): One-step evaluation toàn diện
  - `GreedyPolicy` (Layer 4): Distance + danger + escape capacity (fallback)
- **Khác biệt với non_blind**: Tất cả policies chạy trên memory_map

### Layer 8: Safety Shield + Budget Controller
- **Mục đích**: Lọc nước đi an toàn + điều chỉnh chiến lược theo time budget
- **Cài đặt** (port từ non_blind):
  - `SafetyShield` 4 cấp:
    1. Legal move check
    2. Immediate capture check
    3. Time-expanded danger < FATAL_DANGER
    4. Local viability check
  - `BudgetController` 5 modes:
    - `emergency`: danger ≥ FATAL_DANGER*0.8 → greedy + oneply
    - `tunnel_escape`: trong tunnel có Pacman gần → oneply + greedy
    - `combat`: Pacman ≤ PANIC_DISTANCE → hybrid + ab + oneply + greedy
    - `cheap`: Pacman > 15 → table + greedy (tiết kiệm thời gian)
    - `normal`: table + oneply + ab + greedy

### Bổ sung cho Blind Mode:

### Layer 9: Exploration Strategy (khi không thấy Pacman)
- Khi Pacman ẩn quá lâu (> 3 steps): chuyển sang chế độ khám phá
- BFS flood fill để tìm vùng rộng nhất có thể đến
- Ưu tiên khám phá frontier (ranh giới known/unknown)
- Kết hợp Monte Carlo dự đoán để ưu tiên hướng ngược với Pacman dự đoán

### Layer 10: Anti Line-of-Sight + Corridor Escape
- Khi thấy Pacman và cùng hàng/cột: ưu tiên rẽ vuông góc
- Khi trong tunnel: ưu tiên thoát ra junction gần nhất, hướng ngược Pacman

---

## 📊 Pipeline Xử Lý Mỗi Step

```
step(map_state, my_position, enemy_position, step_number):
  t0 = time.time()
  
  1. Cập nhật memory_map từ observation
  2. PacmanTracker.update(enemy_position) → belief state
  3. Nếu thấy Pacman:
       - Ensemble.update_if_observed()
       - StyleClassifier.update()
  4. Ensemble.predict_positions_2step() → pac_preds
  5. RiskEngine.time_expanded_danger() → danger map
  6. RiskEngine.survival_margin() → margin
  7. BudgetController.select_mode() → mode
  8. PolicyPortfolio.select_policies(mode) → active policies
  9. for each policy:
       proposal = policy.propose(ctx)
  10. SafetyShield.filter(proposals) → safe proposals
  11. Arbitrator.arbitrate(safe proposals) → best move
  12. Validate + return
```

---

## ⚡ Tối Ưu Cho Blind Mode (vision=5, fixed starts)

| Kỹ thuật | Mô tả |
|----------|-------|
| **Precompute topology** | Hardcoded map → precompute trong `__init__`, dùng lại toàn bộ game |
| **Distance cache** | Precompute BFS cho tất cả junctions + chokepoints |
| **Memory map** | Tích lũy observation, các thuật toán chạy trên memory_map |
| **Fixed start knowledge** | Biết Pacman bắt đầu tại (15,10) → khởi tạo belief tại đó |
| **TIME_BUDGET = 0.85s** | Đảm bảo dưới 0.9s |
| **Early exit** | Nếu time sắp hết → dùng GreedyPolicy |
| **Cache validation** | Chỉ recompute khi memory_map thay đổi đáng kể |

---

## 📁 Cấu Trúc File agent.py

```
blind/submissions/24127192/agent.py
├── Constants & Type Aliases
├── Grid Utilities (_shape, _cell, _apply, _valid, _legal, _manhattan, _exits, _pacman_reach)
├── DistanceCache (BFS with cache)
├── TopologyAnalyzer (dead_ends, junctions, corridors, core, loops, chokepoints, tunnels)
├── PacmanTracker (belief state)
├── Opponent Models (MarkovOrder1, MarkovOrder2, ShortestPath, Interception, Random, Adversarial)
├── OpponentModelEnsemble
├── RiskEngine (time_expanded_danger, survival_margin, is_viable)
├── SafetyShield (4-level filter)
├── MCRolloutV3 (CVaR scoring)
├── AlphaBetaSearch (TT + iterative deepening)
├── AntiLoop
├── PacmanStyleClassifier
├── OfflineTable
├── Policies (HybridPolicy, TablePolicy, AlphaBetaPolicy, OnePlyPolicy, GreedyPolicy)
├── PolicyPortfolio
├── BudgetController
├── GhostAgent (main class)
└── PacmanAgent (placeholder)
```

---

## 🔄 Điểm Khác Biệt Chính so với non_blind_agent.py

| Thành phần | non_blind_agent.py | Blind Agent |
|-----------|-------------------|-------------|
| **Map** | Perfect information (toàn bộ map) | Memory map tích lũy từ observation |
| **MapRepository** | Fingerprint từ map_state | Hardcoded KNOWN_LAYOUT, precompute sẵn |
| **PacmanTracker** | Khởi tạo belief rỗng | Khởi tạo belief tại (15,10) |
| **Enemy position** | Luôn có giá trị | Có thể None → dùng belief estimate |
| **Observation** | Toàn bộ 21×21 | Cross-shaped, 5 ô / hướng |
| **A*/BFS/FloodFill** | Trên map đầy đủ | Trên memory_map (có unseen cells) |
| **Tunnel detection** | Từ map đầy đủ | Từ topology precomputed (hardcoded) |
| **Exploration** | Không cần | Cần khi Pacman ẩn lâu |

---

## 📅 Pha Triển Khai

### Phase 1: Foundation
- [ ] Hardcoded KNOWN_LAYOUT + precompute topology
- [ ] Memory map + observation accumulation
- [ ] Grid utilities trên memory_map

### Phase 2: Belief & Prediction
- [ ] PacmanTracker với fixed start knowledge
- [ ] 6 Opponent Models port + adapt cho memory_map
- [ ] OpponentModelEnsemble

### Phase 3: Risk & Safety
- [ ] RiskEngine (danger, survival_margin, viability)
- [ ] SafetyShield 4-level

### Phase 4: Search & Simulation
- [ ] DistanceCache precompute
- [ ] MCRolloutV3 port
- [ ] AlphaBetaSearch port

### Phase 5: Policies & Portfolio
- [ ] 5 Policies (Hybrid, Table, AlphaBeta, OnePly, Greedy)
- [ ] PolicyPortfolio + Arbitrator
- [ ] BudgetController 5 modes

### Phase 6: Blind-Specific
- [ ] Exploration strategy
- [ ] Anti Line-of-Sight evasion
- [ ] Corridor escape nâng cao
- [ ] AntiLoop + history tracking

### Phase 7: Integration & Optimization
- [ ] Pipeline tích hợp
- [ ] Performance tuning (< 0.9s)
- [ ] Test với blind mode (vision=5, fixed starts)
