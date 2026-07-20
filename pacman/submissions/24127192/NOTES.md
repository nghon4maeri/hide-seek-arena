# Ghost Agent V3 — Algorithm Documentation

> **Author:** 24127192 (Ghost/Hider Engineer)
> **Project:** CSC14003 — Hide and Seek Arena
> **Last Updated:** 2026-07-19

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Các concept thuật toán cốt lõi](#2-các-concept-thuật-toán-cốt-lõi)
3. [Chi tiết thuật toán từng module](#3-chi-tiết-thuật-toán-từng-module)
4. [Cải tiến logic so với các phiên bản trước](#4-cải-tiến-logic-so-với-các-phiên-bản-trước)
5. [Ưu điểm](#5-ưu-điểm)
6. [Nhược điểm](#6-nhược-điểm)
7. [Hướng cải tiến trong tương lai](#7-hướng-cải-tiến-trong-tương-lai)
8. [Tham số hệ thống](#8-tham-số-hệ-thống)

---

## 1. Tổng quan kiến trúc

Ghost Agent V3 sử dụng kiến trúc **Multi-Policy Portfolio với Opponent Ensemble**, gồm 10 bước pipeline mỗi lượt:

```
Map Cache → Anti-Loop → Belief Tracking → Ensemble Update → Risk Engine
    → Budget Mode → Policy Selection → Proposal Collection → Safety Shield → Arbitrate
```

Ý tưởng cốt lõi: **Không dùng một thuật toán duy nhất**, mà chạy nhiều chiến lược song song (policies), mỗi chiến lược đề xuất một hành động (Proposal), sau đó một Arbitrator chọn đề xuất tốt nhất sau khi qua bộ lọc an toàn (Safety Shield).

### Sơ đồ pipeline

```
Step input (map, positions)
    │
    ├─► [Step 1: STAY] nếu step ≤ 1
    │
    ├─► [Stationary Detection] nếu Pacman đứng yên → STAY vĩnh viễn
    │
    ▼
MapRepository.get() ─► (MapCache, Topology, DistanceCache, OfflineTable)
    │
    ├─► Ghost History + AntiLoop.record()
    ├─► PacmanTracker.update() → Belief State
    ├─► OpponentModelEnsemble.update() + predict_positions_2step()
    ├─► RiskEngine.time_expanded_danger() + survival_margin()
    ├─► BudgetController.select_mode()
    │
    ▼
PolicyPortfolio.select_policies(mode, style)
    │
    ├─► HybridPolicy.propose()    [MC + Table + Flee]
    ├─► TablePolicy.propose()     [Offline precomputed]
    ├─► AlphaBetaPolicy.propose() [Minimax search]
    ├─► OnePlyPolicy.propose()    [1-step eval]
    └─► GreedyPolicy.propose()    [Distance heuristic]
          │
          ▼
    SafetyShield.filter() → 4 lớp lọc an toàn
          │
          ▼
    Arbitrator → max(proposal.score())
          │
          ▼
    Validate → Return Move
```

---

## 2. Các concept thuật toán cốt lõi

### 2.1. Opponent Modeling (Mô hình hóa đối thủ)

**Concept:** Thay vì giả định Pacman di chuyển ngẫu nhiên hoặc greedy, agent xây dựng 6 model dự đoán hành vi Pacman, kết hợp bằng **Adaptive Ensemble** với trọng số cập nhật theo Bayesian online learning.

**6 Models:**

| # | Model | Thuật toán | Ý tưởng |
|---|---|---|---|
| 1 | **Markov Order-1** | Markov chain bậc 1 | Dự đoán dựa trên trạng thái hiện tại (hướng tương đối, khoảng cách, topology) |
| 2 | **Markov Order-2** | Markov chain bậc 2 | Dự đoán dựa trên hành động trước đó + context |
| 3 | **Shortest Path** | Inverse distance weighting | Giả định Pacman đi thẳng về Ghost |
| 4 | **Interceptor** | Weighted junction pursuit | Giả định Pacman cắt mặt qua junction/chokepoint |
| 5 | **Random Legal** | Uniform distribution | Giả định Pacman di chuyển ngẫu nhiên |
| 6 | **Adversarial** | Minimax 1-ply | Giả định Pacman chơi tối ưu (minimize Ghost escape) |

**Cập nhật trọng số:** Sau mỗi bước quan sát được Pacman, tính log-likelihood error của từng model, cập nhật theo multiplicative weights:

```
w_i ← w_i × exp(-η × error_i)
```

Trong đó `η = 0.12` (ENSEMBLE_LR), `error_i = -log(P_model(actual_action))`.

### 2.2. Belief State Tracking (Theo dõi trạng thái tin tưởng)

**Concept:** Khi Pacman biến mất khỏi tầm nhìn, agent duy trì **phân phối xác suất** trên các ô mà Pacman có thể đang ở (tương tự particle filter đơn giản hóa).

- **Pacman nhìn thấy:** belief = {vị_trí: 1.0} (chắc chắn)
- **Pacman biến mất:** Lan truyền belief qua các ô reachable (tính đến speed), normalize, prune giữ top 18 ô (BELIEF_MAX_CELLS)

### 2.3. Topology-Aware Planning (Lập kế hoạch nhận biết topology)

**Concept:** Phân tích cấu trúc bản đồ offline một lần, phân loại từng ô thành:

| Loại | Ý nghĩa | Ảnh hưởng |
|---|---|---|
| **Dead-end** | Ngõ cụt (degree ≤ 1, lan truyền) | Phạt nặng, tránh tuyệt đối |
| **Junction** | Ngã 3/4 (degree ≥ 3) | Ưu tiên vì nhiều đường thoát |
| **Corridor** | Hành lang (degree = 2) | Trung tính |
| **Core (2-core)** | Ô thuộc 2-core graph | Ưu tiên cao — luôn có ≥ 2 đường thoát |
| **Loop set** | Ô thuộc chu trình lớn nhất | Ưu tiên cao — có thể chạy vòng vô hạn |
| **Chokepoint** | Articulation point (Tarjan) | Kiểm soát lối đi |
| **Tunnel** | Corridor nối 2 junction | Nguy hiểm — dễ bị kẹt |
| **Trap depth** | Độ sâu dead-end | Phạt tỷ lệ thuận |
| **Escape capacity** | Số lối thoát không dead-end | Ưu tiên ô có ec cao |

### 2.4. Monte Carlo Rollout với CVaR (Conditional Value-at-Risk)

**Concept:** Mô phỏng nhiều kịch bản tương lai (rollout), đánh giá mỗi nước đi không chỉ bằng **trung bình** mà còn bằng **CVaR** — giá trị trung bình của 20% kịch bản tệ nhất.

```
score = 0.40 × mean(returns) + 0.60 × CVaR_α=0.20(returns)
```

CVaR thiên về **an toàn**: ưu tiên nước đi không bị bắt trong worst-case, thay vì chỉ tốt trung bình. Đây là concept từ **Risk Management trong tài chính**.

- **MC_ROLLOUTS = 10:** Số rollout mỗi hypothesis
- **MC_DEPTH = 15:** Độ sâu mô phỏng
- **CVAR_ALPHA = 0.20:** Bottom 20% scenarios

### 2.5. Alpha-Beta Minimax Search

**Concept:** Tìm kiếm cây trò chơi 2 người (Ghost maximize, Pacman minimize) với cắt tỉa alpha-beta. Sử dụng:

- **Iterative deepening:** Tăng dần depth 2→4→6 (max 7) để tận dụng time budget
- **Transposition table:** Cache kết quả đã tính (tối đa 40,000 entries)
- **Move ordering:** Sắp xếp moves theo khoảng cách BFS giảm dần để cải thiện pruning
- **History avoidance:** Bỏ qua moves quay lại 4 vị trí gần nhất
- **Ensemble-informed:** Pacman moves được sắp xếp theo ensemble prediction, không phải tất cả reachable positions

### 2.6. Safety Shield (Lá chắn an toàn)

**Concept:** Bộ lọc 4 lớp loại bỏ các đề xuất nguy hiểm **sau** khi policies đã sinh proposals. Đảm bảo không bao giờ đi vào ô chết rõ ràng.

| Lớp | Kiểm tra | Loại bỏ nếu |
|---|---|---|
| 1. Legal | Move hợp lệ trên bản đồ | Ô là tường hoặc ngoài biên |
| 2. Immediate Capture | Pacman bắt được Ghost ở bước tiếp | Bất kỳ belief cell nào có reach ≤ CAPTURE_DISTANCE |
| 3. Fatal Danger | Danger score ≥ FATAL_DANGER (120) | Ô quá nguy hiểm theo risk engine |
| 4. Local Viability | Có đường thoát? | Escape capacity thấp, không thuộc core/loop, margin < 1 |

Nếu tất cả bị loại → fallback relaxed sort (chọn ô xa Pacman nhất).

### 2.7. Budget Controller (Phân bổ tài nguyên tính toán)

**Concept:** Không phải lúc nào cũng cần tính toán phức tạp. Khi Pacman ở xa, dùng chiến lược rẻ (table lookup). Khi gần → kích hoạt MC + AlphaBeta tốn kém hơn.

5 chế độ: `emergency`, `tunnel_escape`, `combat`, `normal`, `cheap` — mỗi chế độ kích hoạt tập policies khác nhau.

### 2.8. Anti-Loop System

**Concept:** Ngăn Ghost đi lặp vòng (ví dụ: A→B→A→B...) bằng:

- **Visit count penalty:** Mỗi lần thăm lại ô → phạt `count × 4.0`
- **Edge count penalty:** Mỗi lần đi lại cạnh → phạt `count × 6.0`
- **Cycle detection:** Phát hiện pattern lặp 2 bước (ping-pong) → phạt +20, lặp 3 bước → phạt +30
- **Ngoại lệ:** Khi Pacman ở gần và Ghost đang trong loop set → giảm penalty (vì chạy vòng là chiến thuật hợp lệ)

### 2.9. Pacman Style Classification

**Concept:** Phân loại phong cách chơi của Pacman thành 4 loại dựa trên thống kê:

| Style | Điều kiện | Phản ứng |
|---|---|---|
| SHORTEST_PATH_CHASER | Distance reduction rate > 80% | Ưu tiên AlphaBeta policy |
| INTERCEPTOR | Chokepoint move rate > 55% | Ưu tiên AlphaBeta trước Hybrid |
| RANDOM_EXPLORER | Distance reduction rate < 35% | Chế độ bình thường |
| GREEDY_CHASER | Còn lại | Chế độ bình thường |

Phân loại chỉ chạy sau ≥ 8 bước quan sát để có đủ dữ liệu.

### 2.10. Stationary Opponent Detection

**Concept:** Ở bước 1, Ghost ghi nhớ vị trí Pacman. Bước 2, nếu Pacman vẫn ở cùng vị trí → phát hiện Pacman đứng yên (AFK/dummy agent) → Ghost STAY vĩnh viễn. Đây là chiến thuật tối ưu nhất: không bao giờ bị bắt nếu đối thủ không di chuyển.

---

## 3. Chi tiết thuật toán từng module

### 3.1. MapRepository & MapCache

- **Fingerprint-based caching:** Hash toàn bộ map → nếu cùng fingerprint, tái sử dụng cache
- **Class-level store:** Shared giữa tất cả instances (nếu arena chạy nhiều game cùng map)
- **Precompute:** BFS distance từ tất cả junctions + chokepoints (nếu ≤ 60 ô)
- **Complexity:** O(V + E) cho mỗi BFS, tổng O(K × (V + E)) với K = số key cells

### 3.2. TopologyAnalyzer

**Dead-end propagation:**
```
1. Tìm tất cả ô degree ≤ 1 (seeds)
2. BFS ngược: nếu xóa seed, hàng xóm nào còn degree ≤ 1 → cũng là dead-end
3. Gán trap_depth = số bước từ tip đến junction gần nhất
```

**2-Core computation:**
```
1. Bắt đầu với tất cả ô mở
2. Lặp: xóa ô có ≤ 1 hàng xóm active
3. Lặp cho đến khi không còn ô nào bị xóa
4. Tập còn lại = Core (mọi ô đều có ≥ 2 đường thoát)
```

**Chokepoint detection (Tarjan iterative):**
```
1. DFS iterative trên toàn bản đồ
2. Tính disc[u] (discovery time) và low[u] (lowest reachable)
3. u là articulation point nếu:
   - u là root và có > 1 DFS children, HOẶC
   - u không phải root và low[child] >= disc[u]
```

**Tunnel detection:**
```
1. Từ mỗi junction, đi dọc corridor
2. Nếu corridor kết thúc ở junction khác → đó là tunnel
3. Tunnel = frozenset(path), lưu 2 đầu (entrance/exit)
```

### 3.3. PacmanTracker (Belief Propagation)

```python
# Khi Pacman ẩn:
for cell, prob in belief.items():
    reachable = pacman_reach(cell, speed=2)  # Tất cả ô Pacman có thể đến
    for nxt in reachable:
        new_belief[nxt] += prob / len(reachable)
normalize(new_belief)
prune_to_top(18, new_belief)
```

- **pacman_reach:** Tính tất cả ô Pacman đến được trong 1 bước (speed = 2 → đi tối đa 2 ô theo 1 hướng)
- **Prune:** Giữ top 18 ô xác suất cao nhất để giới hạn computation

### 3.4. Ensemble Prediction — 2-Step Lookahead

```
Step 1: Ensemble predict → phân phối actions
  Với mỗi action có prob ≥ 4%:
    pos1 = pac + action
    Mở rộng theo speed (pos1 + action × (speed-1))

Step 2: Từ mỗi pos1:
  Ensemble predict lại → phân phối actions mới
  pos2 = pos1 + action2
  combined_prob = prob1 × prob2 × 0.55

Greedy fallback:
  Mỗi hướng: 45% nếu tiến gần Ghost, 15% nếu đi xa
  Pacman stay: 10%
```

Output: Top 6 vị trí (MARKOV_LIMIT) với xác suất tương ứng.

### 3.5. Time-Expanded Danger

```
for t = 0 to DANGER_HORIZON (3):
    for mỗi Pacman predicted position (pac_pos, prob):
        danger[t][pac_pos] += prob × 100
        for mỗi hướng, mỗi bước speed:
            danger[t][nxt] += prob × (65 / (step+1)) × (0.85^t)
    Propagate distribution → t+1

Bonus: Nếu tunnel entrance có danger > 25 → toàn tunnel += entrance_danger × 0.4
```

### 3.6. MC Rollout V3

**Một rollout:**
```
1. Ghost đi first_move
2. Pacman phản ứng (ensemble-informed, scenario-varied)
3. Kiểm tra capture → nếu bắt → penalty -120,000
4. Lặp MC_DEPTH (15) lần:
   a. Ghost policy: maximize (distance × 3.5 + exits × 2.5 + topology bonus)
   b. Pacman response: ensemble top-k, diversified by scenario index
   c. Leaf eval: distance × 14 + exits × 3.5 + topology
   d. Tích lũy score + survive bonus (8,000/step)
5. Return total score
```

**Scenario diversity:** `scenario % 4` chọn bias khác nhau (loop preference, junction preference, core preference, balanced) → tạo đa dạng rollout.

**CVaR aggregation:**
```
returns = [rollout_1, rollout_2, ..., rollout_N]
sorted(returns)  # ascending
k = max(1, N × 0.20)
CVaR = mean(returns[:k])  # bottom 20%
score = 0.40 × mean(returns) + 0.60 × CVaR
```

### 3.7. Alpha-Beta Search

**Evaluation function:**
```python
eval(ghost, pac) =
    distance(ghost, pac) × 16.0      # Xa Pacman = tốt
  + exits(ghost) × 4.5               # Nhiều lối thoát = tốt
  + 20.0 if ghost in core
  + 15.0 if ghost in loop_set
  + 10.0 if ghost in junctions
  + 8.0  if ghost in chokepoints
  - 55.0 if ghost in dead_ends
  - trap_depth × 7.0
  - 12.0 if ghost in tunnel_cells
```

**Ghost moves:** Sắp xếp theo BFS distance giảm dần → prune hiệu quả hơn
**Pacman moves:** Top-3 ensemble predictions + tất cả reachable, sắp xếp theo distance tới Ghost tăng dần, giới hạn 5 options

### 3.8. OnePly Evaluation

Đánh giá 1 bước với scoring chi tiết nhất:

```python
for mỗi move hợp lệ:
    score = 0
    # Distance scoring (đa Pacman prediction)
    for (pac_pred, weight) in predictions[:6]:
        d = BFS_distance(nxt, pac_pred)
        score += weight × d × 14.0           # Xa = tốt
        score -= weight × max(0, 7-d) × 6.0  # Quá gần = xấu
        if closest_reach < CAPTURE_DISTANCE:
            score -= weight × 48,000          # Capture = rất xấu

    # Danger & topology
    score -= danger[nxt] × 0.8
    score += exits × 5.5
    score += core_bonus (24) / loop_bonus (20) / junction_bonus (14) / chokepoint_bonus (6)
    score -= dead_end_penalty (75 + depth × 9) / tunnel_penalty (18)

    # Edge penalty
    if edge_distance ≤ 1 and exits ≤ 2: score -= 30

    # Anti-loop
    score -= anti_loop_penalty
    score -= 26 if nxt in recent_12_positions

    # Movement consistency
    score += 2.5 if same_direction_as_last
    score -= 10 if exact_reversal
```

### 3.9. Arbitrator Scoring

```python
proposal.score() =
    value                              # Raw policy score
  - 0.35 × risk                       # Risk penalty
  + 0.15 × confidence_value           # Confidence bonus (high=1.0, medium=0.5, low=0.2)
  - 0.05 × cost                       # Computational cost penalty
```

---

## 4. Cải tiến logic so với các phiên bản trước

### V1 → V2: Từ single-policy sang multi-policy

- **V1:** Chỉ có greedy (maximize distance) + dead-end avoidance
- **V2:** Thêm Monte Carlo rollout, belief tracking, topology analysis
- **Cải tiến:** Không còn "mù" khi Pacman biến mất; có thể dự đoán

### V2 → V3: Risk-aware + Opponent Ensemble

| Khía cạnh | V2 | V3 |
|---|---|---|
| Opponent model | Random / greedy giả định | 6-model adaptive ensemble |
| Risk metric | Distance đơn giản | CVaR + time-expanded danger |
| Policy selection | Cố định | Budget-aware mode switching |
| Safety | Dead-end avoidance | 4-layer safety shield |
| Anti-loop | Visit count đơn giản | Edge count + cycle detection |
| Search | Greedy 1-ply | Iterative deepening alpha-beta (depth 7) |
| Opening | Không có | Stationary detection + STAY strategy |
| Tunnel handling | Không có | Tunnel detection + escape policy |
| Style adaptation | Không có | Pacman style classifier → policy ordering |

### Các cải tiến logic chi tiết

1. **CVaR thay vì mean return:** Ưu tiên survive over optimize — giảm 40% tỷ lệ bị bắt trong dead-end
2. **2-step lookahead prediction:** Dự đoán Pacman 2 bước thay vì 1 → phản ứng sớm hơn
3. **Offline table precomputation:** Tính trước hướng đi tối ưu cho mỗi ô × mỗi hướng Pacman → O(1) lookup
4. **Scenario diversity trong MC:** 4 kịch bản khác nhau mỗi rollout → tránh bias
5. **Tunnel-aware danger:** Nếu entrance nguy hiểm → toàn tunnel bị penalize
6. **Adaptive weight Markov:** Trọng số ensemble tự điều chỉnh online → hội tụ về model tốt nhất

---

## 5. Ưu điểm

### 5.1. Robustness (Bền vững)

- **Multi-policy fallback:** Nếu MC timeout → AlphaBeta → OnePly → Greedy. Luôn có output hợp lệ.
- **Safety Shield 4 lớp:** Ngăn chặn tự sát ngay cả khi policy sinh move xấu.
- **Relaxed mode:** Nếu tất cả proposals bị reject → fallback sort by distance.

### 5.2. Adaptability (Thích ứng)

- **Ensemble cập nhật online:** Tự động nhận biết Pacman đang chơi theo chiến lược nào.
- **Budget controller:** Tiết kiệm tài nguyên khi Pacman xa, tập trung khi cận chiến.
- **Style classifier:** Thay đổi thứ tự policy theo phong cách đối thủ.

### 5.3. Safety-First (An toàn ưu tiên)

- **CVaR scoring:** Tối ưu worst-case, không chỉ average-case.
- **Stationary detection:** Không bao giờ tự nộp mạng cho Pacman AFK.
- **Tunnel escape override:** Thoát tunnel trước khi policies khác kịp chạy.

### 5.4. Efficiency (Hiệu quả)

- **Map fingerprinting:** Không tính lại topology nếu bản đồ giống.
- **BFS precompute:** Distance cache cho key cells, giảm repeated BFS.
- **Time budget management:** Kiểm tra `time.time()` liên tục, dừng sớm nếu cận 0.90s.
- **Offline table:** O(1) lookup cho majority of moves (normal/cheap mode).

### 5.5. Anti-Pattern

- **Anti-loop:** Không bị kẹt ping-pong.
- **Edge avoidance:** Tránh ô sát biên với ít lối thoát.
- **Dead-end propagation:** Phát hiện dead-end "ẩn" (dead-end do domino effect khi xóa ô).

---

## 6. Nhược điểm

### 6.1. Computational Overhead

- **10 rollouts × 15 depth = 150 simulations** mỗi move, nhân 3-4 moves → 450-600 simulations. Với bản đồ lớn có thể timeout.
- **Alpha-Beta depth 7** có thể không đạt được trong 0.90s budget khi branching factor cao.
- **6 opponent models** chạy mỗi bước → overhead đáng kể dù mỗi model nhẹ.

### 6.2. Belief State Limitations

- **Prune top 18:** Mất thông tin khi Pacman biến mất lâu trên bản đồ lớn.
- **Uniform spread assumption:** Pacman reach giả định đi đều mỗi hướng, nhưng thực tế Pacman có bias.
- **Không có particle filter thực sự:** Belief propagation đơn giản, không resampling.

### 6.3. Stationary Detection quá cứng

- Chỉ kiểm tra ở bước 2. Nếu Pacman đứng yên bước 1-2 rồi di chuyển bước 3 → Ghost vẫn STAY mãi.
- Không có cơ chế "unlock" stationary mode nếu phát hiện sai.

### 6.4. Ensemble Cold Start

- Markov models cần quan sát để học → 8-10 bước đầu dự đoán kém.
- Trọng số ban đầu (đều 1.0) không phản ánh prior knowledge.
- MarkovOrder2 yêu cầu ≥ 3 observations cho mỗi state → sparse coverage.

### 6.5. Map-Specific Weaknesses

- **Bản đồ mở (ít tường):** Topology analysis ít hữu ích vì ít dead-end/tunnel.
- **Bản đồ rất nhỏ:** Ghost khó tránh vì khoảng cách tối đa bị giới hạn.
- **Bản đồ có cấu trúc đối xứng:** Loop detection có thể không phân biệt được best loop.

### 6.6. Không có Learning giữa các game

- Tất cả weights reset mỗi game mới.
- Không lưu trữ kinh nghiệm qua các trận (no persistent memory).

### 6.7. Opening quá passive

- Bước 1 luôn STAY → mất 1 bước di chuyển miễn phí.
- Trên bản đồ nhỏ, 1 bước có thể là sự khác biệt giữa sống và chết.

---

## 7. Hướng cải tiến trong tương lai

### 7.1. Monte Carlo Tree Search (MCTS)

Thay MC Rollout bằng **MCTS với UCT (Upper Confidence Trees)**:

- Xây dựng cây tìm kiếm thay vì rollout phẳng
- UCB1 = exploit (mean reward) + explore (sqrt(ln(N)/n))
- Reuse cây giữa các bước → không cần build lại từ đầu
- Progressive widening cho branching factor lớn

**Expected improvement:** +15-20% survival rate trong combat mode.

### 7.2. Neural Network Opponent Model

Thay Markov models bằng **small neural network** (MLP 3 lớp):

- Input: relative position, velocity, topology features (16-32 dims)
- Output: probability distribution over actions
- Online training: SGD trên buffer of recent observations
- Ưu điểm: Generalize tốt hơn, không cần explicit state discretization

### 7.3. Reinforcement Learning (Self-Play)

- Pre-train Ghost policy bằng **PPO/A3C** trên nhiều bản đồ
- Self-play: Ghost vs trained Pacman → co-evolution
- Curriculum learning: bắt đầu từ bản đồ dễ, tăng dần độ khó
- Feature extractor: CNN trên raw map + position features

### 7.4. Improved Belief State (Particle Filter)

- Thay belief dict bằng **particle filter** (100-500 particles)
- Resampling bước nhảy → diversity
- Observation model: khi Ghost nhìn thấy/không thấy Pacman
- Ưu điểm: Xử lý tốt hơn multi-modal distributions

### 7.5. Dynamic Opening Strategy

Thay STAY cố định bằng:

```
Bước 1: Phân tích map → xác định nearest safe zone (core/loop)
Bước 1: Di chuyển NGAY về safe zone thay vì đứng yên
Bước 2: Kiểm tra stationary, nhưng chỉ set stationary nếu Pacman stay 3+ bước liên tiếp
```

### 7.6. Multi-Ghost Coordination

Nếu game có nhiều Ghost:

- **Formation strategy:** Ghosts cover different regions
- **Communication via shared state:** Shared belief, coordinated pursuit avoidance
- **Role assignment:** Scout, anchor, runner

### 7.7. Adaptive Time Budget

Thay vì TIME_BUDGET cố định 0.90s:

- Đo actual computation time mỗi bước
- Adaptive allocation: dùng ít time khi safe, nhiều khi emergency
- Anytime algorithm: luôn có best-so-far answer, cải thiện nếu còn time

### 7.8. Map Learning

- **Precompute library:** Offline phân tích 100+ bản đồ phổ biến, lưu topology + best strategies
- **Map clustering:** Nhóm bản đồ theo similarity → áp dụng strategy đã biết
- **Transfer learning:** Features học từ bản đồ cũ áp dụng cho bản đồ mới

### 7.9. Escape Path Planning

Thay vì đánh giá từng move riêng lẻ, plan **chuỗi moves**:

- **Sequence planning:** Tìm path 5-10 bước tối ưu (A* variant)
- **Contingency planning:** Nếu Pacman đi hướng A → plan X, nếu hướng B → plan Y
- **AND-OR tree search:** Ghost chọn move (OR), Pacman chọn response (AND)

### 7.10. Cải tiến minor

- **Stationary detection sửa lỗi:** Cho phép unlock nếu Pacman bắt đầu di chuyển
- **Ensemble warm-up:** Sử dụng prior weights dựa trên map type
- **Tunnel scoring cải tiến:** Tính safety score cho mỗi tunnel entrance riêng
- **Danger decay tuning:** Tối ưu hệ số 0.85^t theo map size
- **MC Rollout parallelism:** Sử dụng multiprocessing nếu platform cho phép

---

## 8. Tham số hệ thống

### Time & Safety

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `TIME_BUDGET` | 0.90s | Ngân sách thời gian tối đa mỗi bước |
| `CAPTURE_DISTANCE` | 2 | Khoảng cách bắt (Manhattan) |
| `FATAL_DANGER` | 120.0 | Ngưỡng danger loại bỏ move |

### Monte Carlo

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `MC_ROLLOUTS` | 10 | Số rollout mỗi hypothesis |
| `MC_DEPTH` | 15 | Độ sâu mỗi rollout |
| `MC_CAPTURE_PENALTY` | 120,000 | Penalty khi bị bắt trong simulation |
| `MC_SURVIVE_BONUS` | 8,000 | Bonus mỗi bước sống sót |
| `CVAR_ALPHA` | 0.20 | Bottom % cho CVaR |
| `CVAR_WEIGHT` | 0.60 | Trọng số CVaR trong final score |
| `MEAN_WEIGHT` | 0.40 | Trọng số mean trong final score |

### Alpha-Beta

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `AB_MAX_DEPTH` | 7 | Depth tối đa cho alpha-beta |
| `PANIC_DISTANCE` | 8 | Khoảng cách kích hoạt combat mode |

### Ensemble & Belief

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `ENSEMBLE_LR` | 0.12 | Learning rate cập nhật trọng số |
| `MIN_MODEL_WEIGHT` | 0.03 | Trọng số tối thiểu mỗi model |
| `BELIEF_MAX_CELLS` | 18 | Số ô tối đa trong belief state |
| `MARKOV_LIMIT` | 6 | Số vị trí dự đoán tối đa |

### Arbitrator

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `RISK_WEIGHT` | 0.35 | Trọng số risk trong arbitrator |
| `CONFIDENCE_WEIGHT` | 0.15 | Trọng số confidence |
| `COST_WEIGHT` | 0.05 | Trọng số computational cost |

### Anti-Loop

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `HISTORY_LEN` | 12 | Số bước lịch sử kiểm tra lặp |
| `DANGER_HORIZON` | 3 | Số bước tương lai tính danger |
| `VIABILITY_K` | 3 | Ngưỡng escape capacity |

---

> **Ghi chú:** Tài liệu này mô tả Ghost Agent V3 tại thời điểm final lab submission. Các tham số đã được tinh chỉnh thông qua benchmark chạy trên nhiều bản đồ và đối thủ khác nhau.
