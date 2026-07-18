# Ghi chú Ghost Agent V3 — Risk-aware Multi-policy Strategy

**Student:** 24127192  
**Phiên bản:** V3.0  
**File:** `submissions/24127192/agent.py` (~1789 dòng)

---

## 1. Concept (Ý tưởng thiết kế)

### 1.1. Vấn đề cần giải quyết

Ghost (Hider) cần sống sót tối đa 200 bước trước sự truy đuổi của Pacman (Seeker). Pacman có tốc độ 2 (di chuyển 2 ô mỗi lượt trên đường thẳng), trong khi Ghost chỉ di chuyển 1 ô. Ghost bị bắt khi khoảng cách Manhattan < 2.

### 1.2. Ý tưởng chính

Thay vì dùng 1 thuật toán cố định (greedy, minimax, hay MCTS), V3 kết hợp **nhiều policy song song** thông qua kiến trúc **Policy Portfolio + Safety Shield**:

- **Nhiều policy cùng đề xuất** nước đi → mỗi policy tạo ra 1 `Proposal` chứa giá trị, rủi ro, độ tin cậy.
- **Safety Shield 4 lớp** lọc bỏ action nguy hiểm trước khi chọn.
- **Arbitrator** chấm điểm tổng hợp và chọn action tốt nhất.
- **Budget Controller** phân bổ thời gian tính toán theo mức độ nguy hiểm.

### 1.3. Các thuật toán/chiến lược cốt lõi

| Thuật toán | Mục đích |
|---|---|
| MapRepository (fingerprint cache) | Cache bản đồ ngay lần đầu, tránh tính lại |
| TopologyAnalyzer (Tarjan) | Phát hiện dead-end, junction, chokepoint, tunnel, loop, core |
| PacmanTracker (Belief State) | Phân phối xác suất vị trí Pacman khi mất dấu |
| OpponentModelEnsemble (6 models) | Dự đoán Pacman từ nhiều góc nhìn, tự điều chỉnh trọng số |
| TimeExpandedDanger | Danger map theo thời gian (multi-horizon) |
| SurvivalMargin | Đánh giá ghost có kịp thoát trước Pacman không |
| LocalViability | Kiểm tra vị trí có sống được K bước không |
| Monte Carlo V3 (CVaR) | Rollout 8 lần, depth 15, đánh giá bằng CVaR_20 |
| Alpha-Beta + TT | Tìm kiếm sâu với transposition table |
| Safety Shield | Lọc 4 mức: legal → capture → danger → viability |
| Policy Portfolio + Arbitrator | 5 policy đề xuất, Arbitrator chọn tối ưu |
| BudgetController | 5 chế độ: cheap / normal / combat / tunnel_escape / emergency |
| PacmanStyleClassifier | Nhận diện style Pacman online |
| AntiLoop | Chống lặp vòng bằng visit count + edge count + cycle detection |

---

## 2. Luồng hoạt động (Pipeline mỗi step)

Mỗi lượt `GhostAgent.step()` thực hiện 12 bước tuần tự:

```
┌─────────────────────────────────────────────────────┐
│ 1. MapRepository.get(map_state)                      │
│    → Cache/lấy (MapCache, Topology, DistCache, OT)   │
│                                                       │
│ 2. Ghi nhận vị trí ghost (ghost_hist + AntiLoop)      │
│                                                       │
│ 3. PacmanTracker.update(enemy_position)               │
│    → Cập nhật belief distribution                     │
│                                                       │
│ 4. OpponentModelEnsemble.update_if_observed()         │
│    → Cập nhật Markov O1/O2 + trọng số ensemble       │
│    → PacmanStyleClassifier.update()                   │
│                                                       │
│ 5. Ensemble.predict_positions_2step()                 │
│    → Dự đoán vị trí Pacman trong 2 bước tới          │
│                                                       │
│ 6. RiskEngine                                         │
│    → TimeExpandedDanger (horizon=3)                   │
│    → SurvivalMargin (margin = pac_arrival - ghost)    │
│                                                       │
│ 7. BudgetController.select_mode()                     │
│    → cheap / normal / combat / tunnel_escape / emergency│
│                                                       │
│ 8. PolicyPortfolio.select_policies(mode, style)       │
│    → Chọn tập policies phù hợp với tình huống        │
│                                                       │
│ 9. Thu thập Proposals từ các policy đã chọn           │
│    → HybridPolicy (MC + table + flee)                 │
│    → TablePolicy (offline lookup + safety)            │
│    → AlphaBetaPolicy (ID α-β + TT)                   │
│    → OnePlyPolicy (đánh giá 1 bước)                  │
│    → GreedyPolicy (distance + danger + escape)        │
│    → Tunnel escape override                           │
│                                                       │
│ 10. SafetyShield.filter(proposals)                    │
│     Level 1: legal (valid cell + legal move)          │
│     Level 2: immediate capture (Pacman reach next)    │
│     Level 3: time-expanded danger < FATAL_DANGER      │
│     Level 4: local viability (sống được K bước)       │
│                                                       │
│ 11. Arbitrator.choose_best(safe_proposals)            │
│     score = value - 0.35*risk + 0.15*confidence       │
│             - 0.05*cost                               │
│                                                       │
│ 12. Validate / STAY                                   │
│     → Kiểm tra action hợp lệ, fallback = STAY        │
└─────────────────────────────────────────────────────┘
```

### 2.1. Chi tiết các thành phần

#### MapRepository (dòng 229–299)
- Nhận diện map bằng **fingerprint** (hash bytes).
- Lần đầu: xây `_MapCache` (valid_cells, adjacency), `TopologyAnalyzer`, `DistanceCache` (BFS lazy + precompute junctions/chokepoints), `OfflineTable`.
- Class-level `_store` để chia sẻ cache giữa các instance.

#### TopologyAnalyzer (dòng 302–470)
- **Dead-end propagation:** Từ các cell degree ≤ 1, lan rộng theo kiểu "peeling" để tìm toàn bộ nhánh ngõ cụt + ghi `trap_depth`.
- **2-core:** Loại lặp cell degree ≤ 1 cho đến khi ổn định → core an toàn.
- **Largest loop:** DFS tìm cycles trong core, chọn cycle dài nhất.
- **Chokepoints:** Thuật toán Tarjan iterative tìm articulation points.
- **Tunnels:** Từ junction, đi theo corridor cho đến junction kế tiếp.
- **Escape capacity:** Đếm số exit không dẫn vào dead-end cho mỗi cell.

#### PacmanTracker (dòng 474–527)
- Khi Pacman **visible**: belief = `{pac_pos: 1.0}`.
- Khi Pacman **invisible**: lan truyền belief qua `_pacman_reach()`, prune giữ top 18 cells.
- Thuộc tính `best_estimate` trả cell xác suất cao nhất.

#### OpponentModelEnsemble (dòng 686–800)
6 sub-models chạy song song:

| Model | Logic |
|---|---|
| MarkovOrder1 | State = (relative_bucket, distance_bucket, geometry) → action count |
| MarkovOrder2 | State = (prev_action, relative_bucket) → action count, yêu cầu ≥ 3 mẫu |
| ShortestPathModel | BFS weight: 1/distance tới ghost |
| InterceptionModel | Ưu tiên move tới junction/chokepoint gần ghost |
| RandomLegalModel | Phân phối đều các move hợp lệ |
| AdversarialModel | Minimax 1-ply: Pacman chọn move tệ nhất cho ghost escape |

**Weight update:** Sau mỗi bước Pacman quan sát được:
```
error = -log(predicted_prob)
weight *= exp(-0.12 * error)
normalize(weights)
```

**2-step prediction:** `predict_positions_2step()` kết hợp ensemble distribution bước 1, rồi mở rộng bước 2 qua ensemble lần nữa. Thêm greedy fallback (Pacman tiến gần ghost = xác suất 0.45).

#### RiskEngine (dòng 802–895)
- **TimeExpandedDanger:** Tạo `danger[t][cell]` cho t = 0..3. Mỗi horizon: tính danger từ belief, lan truyền distribution, áp dụng tunnel penalty.
- **SurvivalMargin:** Tìm exit tốt nhất (core/loop/junction), tính `margin = pac_arrival_time - ghost_arrival_time`. Pac arrival tính theo speed 2.
- **LocalViability:** Cell viable nếu: escape_capacity ≥ 3, hoặc nằm trong core/loop, hoặc survival_margin ≥ 1, hoặc xa Pacman ≥ 5 + có ≥ 2 exit.

#### SafetyShield (dòng 943–1000)
4 mức lọc tuần tự:
1. **Legal:** Cell hợp lệ + move nằm trong legal moves.
2. **Capture:** Pacman (với speed 2) có thể bắt ghost ở vị trí mới ngay lượt sau? (belief ≥ 15%).
3. **Danger:** `danger_t0[nxt] < FATAL_DANGER (120.0)`.
4. **Viability:** Cell phải viable theo RiskEngine.

Nếu không cell nào qua cả 4 mức → **relaxed fallback**: sắp xếp theo khoảng cách xa nhất từ Pacman.

#### MCRolloutV3 (dòng 1002–1113)
- 8 rollouts mỗi move, depth 15.
- Ghost policy: ưu tiên core/loop/junction, phạt dead-end/tunnel, kịch bản đa dạng (`scenario % 4`).
- Pacman response: dùng ensemble prediction, chọn action theo scenario (`scenario % 3`).
- **CVaR scoring:** `score = 0.40 × mean + 0.60 × CVaR_20` (trung bình 20% rollout tệ nhất).
  → Ưu tiên action có worst-case tốt, không chỉ mean tốt.

#### AlphaBetaSearch (dòng 1115–1212)
- Iterative deepening, max depth 7.
- **Transposition table** (40K entries) tránh tính lại state.
- **Move ordering:** Ghost moves sắp theo distance xa Pacman giảm dần. Pacman moves sắp theo ensemble prediction + distance.
- Pacman phản ứng: lấy top 3 ensemble predictions + pacman_reach, giới hạn 5 options.
- Anti-oscillation: bỏ qua move vào ghost_hist[-4:].
- Time check: dừng nếu > 72% budget.

#### Policies (dòng 1338–1528)
5 policy tạo `Proposal`:

| Policy | Khi nào | Logic | Confidence |
|---|---|---|---|
| HybridPolicy | Pac visible, dist ≤ 8 | MC CVaR + table value + flee distance - danger - loop_pen | high |
| TablePolicy | Position có trong offline table | Offline lookup + danger check + dead-end check | medium |
| AlphaBetaPolicy | Pac visible, đủ budget | Iterative deepening α-β | high |
| OnePlyPolicy | Luôn | Đánh giá 1 bước: distance × Markov + topology + danger + anti-loop | medium |
| GreedyPolicy | Luôn | min_distance × 10 - danger + escape_capacity - dead-end penalty | low |

#### BudgetController (dòng 1309–1336)
5 chế độ dựa trên tình huống:

| Mode | Điều kiện | Policies chạy |
|---|---|---|
| emergency | danger ≥ 80% FATAL | Greedy + OnePly |
| tunnel_escape | Trong tunnel, Pac gần entrance ≤ 4 | OnePly + Greedy |
| combat | Pac visible, dist ≤ 8 | Hybrid + AB + OnePly + Greedy |
| cheap | Pac xa > 15 | Table + Greedy |
| normal | Mặc định | Table + OnePly + AB + Greedy |

#### PacmanStyleClassifier (dòng 1246–1286)
Online tracking:
- **distance_reduction_rate:** % moves làm giảm distance tới ghost.
- **chokepoint_preference:** % moves đi vào chokepoint/junction.
- Phân loại: `SHORTEST_PATH_CHASER` (dr > 0.8), `INTERCEPTOR` (choke > 0.55), `RANDOM_EXPLORER` (dr < 0.35), `GREEDY_CHASER` (default).
- Style ảnh hưởng thứ tự policy trong combat mode.

#### AntiLoop (dòng 1214–1244)
- `visit_count[cell]`: số lần ghost đã ghé.
- `edge_count[(prev, cur)]`: số lần đi cạnh này.
- **Cycle detection:** Nếu move quay đầu (nxt == recent[-2]) → penalty +20. Nếu pattern lặp 3 lần → penalty +30.
- Khi Pac gần và ghost đang trong loop region → giảm penalty (loop là chiến thuật sống tốt).

#### Diagnostics (dòng 1288–1307)
- `policy_failures`: đếm số lần mỗi policy throw exception.
- `policy_usage`: đếm số lần mỗi policy được chọn.
- `safety_rejections`: số lần Safety Shield loại bớt proposals.

---

## 3. Những cải tiến đã thực hiện (V1 → V3)

### 3.1. So sánh V1 (old.py) → V3 (agent.py)

| Tính năng | V1 (old.py, 351 dòng) | V3 (agent.py, 1789 dòng) |
|---|---|---|
| Map cache | Không có, tính BFS mỗi bước | MapRepository + fingerprint hash, precompute junctions |
| Topology | `Topo` đơn giản (chỉ dead-end branches) | TopologyAnalyzer: dead-ends, junctions, corridors, core, loops, chokepoints (Tarjan), tunnels, trap_depth, escape_capacity |
| Pacman tracking | `last_pp` (vị trí cuối) | BeliefState phân phối xác suất, propagation khi invisible |
| Dự đoán Pacman | `USLStar` + `_predict_astar_seeker` (A* hardcode) | OpponentModelEnsemble 6 models + adaptive weight update |
| Danger map | Không có | TimeExpandedDanger multi-horizon (t=0..3) |
| Tìm kiếm | Iterative Deepening DFS + `lru_cache` A* | Alpha-Beta với transposition table (40K) + move ordering |
| Monte Carlo | Không có | MCRolloutV3: 8 rollouts, depth 15, CVaR_20 scoring |
| Decision | Strict fallback (1 layer trả, bỏ qua các layer sau) | PolicyPortfolio: 5 policies song song → Arbitrator chọn tốt nhất |
| Safety | Không có (chỉ kiểm tra distance < 2) | SafetyShield 4 mức (legal → capture → danger → viability) |
| Anti-loop | Không có | visit_count + edge_count + cycle detection |
| Adaptation | Phụ thuộc A* hardcode | BudgetController 5 modes + PacmanStyleClassifier |
| Diagnostics | Không có | Policy failures, usage, safety rejections |

### 3.2. Cải tiến chi tiết

1. **Map caching triệt để:** Lần đầu xây map mất ~20ms, các lần sau O(1) lookup qua fingerprint.
2. **Topology phong phú hơn:** V1 chỉ biết dead-end branches. V3 biết chokepoints (Tarjan), tunnels, loops, core, trap_depth, escape_capacity → ghost biết vùng nào an toàn, vùng nào bẫy.
3. **Belief state thay vì last_known_position:** Khi Pacman biến mất, V3 lan truyền xác suất thay vì giữ vị trí cũ đã lỗi thời.
4. **Ensemble thay vì hardcode A*:** V1 hardcode logic A* của đối thủ cụ thể. V3 dùng 6 models tổng quát, tự học trọng số → thích ứng với MỌI loại Pacman.
5. **Safety Shield ngăn action nguy hiểm:** V1 không có cơ chế chặn → ghost dễ bị dồn vào dead-end. V3 lọc 4 mức nghiêm ngặt.
6. **CVaR thay vì mean scoring:** V1 chỉ dùng evaluation function. V3 dùng CVaR_20 → ưu tiên action có worst-case tốt, tránh action 90% tốt nhưng 10% chết.
7. **Anti-loop:** V1 ghost hay rung qua lại 2 ô. V3 phạt visit_count + edge_count + cycle detection.
8. **Budget Controller:** V1 hay timeout (> 0.95s). V3 phân bổ budget theo tình huống, mode emergency chỉ chạy Greedy+OnePly.

---

## 4. Kết quả Benchmark

### Benchmark: Ghost V3 vs tất cả Pacman trong `pacman/sum` (deterministic, no-viz, pacman_speed=2)

| Pacman Agent | Kết quả | Steps |
|---|---|---|
| 02 | **ghost_wins** ✅ | 200 |
| 3 | **ghost_wins** ✅ | 200 |
| 5 | **ghost_wins** ✅ | 200 |
| 6 | **ghost_wins** ✅ | 200 |
| 7 | **ghost_wins** ✅ | 200 |
| 8 | pacman_wins ❌ | 41 |
| 9 | **ghost_wins** ✅ | 1 (agent lỗi syntax) |
| 10 | **ghost_wins** ✅ | 200 |
| 11 | **ghost_wins** ✅ | 200 |
| 12 | pacman_wins ❌ | 149 |
| 13 | **ghost_wins** ✅ | 200 |
| 15 | **ghost_wins** ✅ | 200 |
| 16 | **ghost_wins** ✅ | 200 |
| Max Verstappen | pacman_wins ❌ | 118 |

- **Win rate:** 11/14 (78.6%)
- **Average steps:** 164.93
- **Ghost thua 3 ván:** agents 8, 12, Max Verstappen

---

## 5. Hạn chế và Cải thiện trong tương lai

### 5.1. Hạn chế hiện tại

1. **Thua trước Pacman mạnh (agents 8, 12, Max Verstappen):** Các Pacman sử dụng chiến thuật chặn đường (interception) hoặc A* tốc độ 2 kết hợp ngoại suy vận tốc có thể dồn ghost vào góc trước khi Safety Shield kịp phản ứng.
2. **Ensemble chậm hội tụ:** Cần ~8-10 bước quan sát trước khi trọng số ensemble ổn định. Trong 8 bước đầu, dự đoán Pacman còn sai lệch.
3. **LocalViability quá đơn giản:** Hiện chỉ kiểm tra escape_capacity + survival_margin. Chưa thực sự chạy minimax K-step để xác minh ghost có thoát được hay không.
4. **Chưa có Quiescence Search:** Alpha-Beta có thể bị horizon effect — đánh giá state "an toàn" nhưng thực chất Pacman bắt ở bước tiếp.
5. **MC Rollout chưa adaptive:** Luôn chạy 8 rollouts. Khi danger cao nên tăng lên 16-32, khi Pacman xa nên giảm xuống 4.
6. **Tunnel escape đơn giản:** Hiện chỉ chạy BFS tới exit xa Pacman. Chưa xét trường hợp Pacman chặn cả 2 đầu tunnel.
7. **Không có Zobrist Hashing:** Transposition table dùng tuple key, chậm hơn Zobrist hash.
8. **Chưa có Influence Map:** Chưa phát hiện trước hướng bị bao vây nếu 2+ Pacman vây bắt.

### 5.2. Hướng cải thiện tương lai

1. **Nâng cấp LocalViability thành full Viability Kernel:** Chạy backward induction K=5 bước để xác minh chính xác ghost có thoát được hay không trước khi chấp nhận action.
2. **Quiescence Search cho Alpha-Beta:** Mở rộng search khi state ở biên horizon có capture ngay (capture extension).
3. **Adaptive MC Rollouts:** Tăng/giảm số rollouts theo urgency:
   - emergency: 2 rollouts (tiết kiệm budget)
   - combat: 16-24 rollouts (cần quyết định chính xác)
   - cheap: 0 rollouts (chỉ dùng table/greedy)
4. **Zobrist Hashing:** Thay tuple key bằng XOR hash → transposition table nhanh hơn.
5. **Tunnel Solver nâng cao:** Khi ghost ở trong tunnel, tính chính xác xem ghost có thoát kịp không dựa trên Pacman distance tới cả 2 entrance.
6. **Multi-agent support:** Nếu arena mở rộng 2 Pacman vây 1 Ghost, cần Influence Map để phát hiện hướng bị kẹp.
7. **MCTS (Monte Carlo Tree Search):** Thay MC Rollout bằng MCTS với UCB1 selection → rollout sâu hơn, tận dụng tree structure.
8. **Offline Training:** Chạy self-play offline để pre-train Markov model và offline table tốt hơn, thay vì học online từ đầu mỗi trận.
9. **Reinforcement Learning:** Dùng PPO/DQN để train ghost policy offline, dùng như model thứ 7 trong ensemble.
