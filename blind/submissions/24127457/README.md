# 24127457 — Blind Adversary (Lab 2)

## Tổng quan

Agent cho bài toán **Blind Adversary** — Lab 2 môn CSC14003 (Nhập môn Trí tuệ Nhân tạo). Trong môi trường này, cả Pacman (Seeker) và Ghost (Hider) đều bị giới hạn tầm nhìn hình chữ thập (cross-shaped FOV, bán kính 5 ô), tạo thành bài toán POMDP (Partially Observable Markov Decision Process).

Kiến trúc V6 kết hợp **Heuristic chiến thuật** (từ các đội Top-tier Bomberland) với **Deep RL** (PPO + RecurrentActorCritic) để đạt hiệu suất cao trong điều kiện thông tin không đầy đủ.

### Tổng quan kiến trúc

```mermaid
graph TB
    subgraph "Input: Partial Observability"
        FOV[Cross-shaped FOV<br/>radius=5]
        MAP[map_state 21×21<br/>0=seen, 1=wall, -1=unseen]
    end
    
    subgraph "Modular Components"
        TOPO[topology_analyzer.py<br/>TopologyAnalyzer]
        STATE[state_trackers.py<br/>BeliefState + Heatmap]
        TACT[tactical_engines.py<br/>Intent + Trap + Mode]
        NET[network_architect.py<br/>6-channel CNN+LSTM]
    end
    
    subgraph "Runtime Agents"
        PAC[PacmanAgent<br/>3-Tier Lexicographic]
        GHO[GhostAgent<br/>Dynamic Mode Selection]
    end
    
    FOV --> MAP
    MAP --> TOPO
    MAP --> STATE
    STATE --> TACT
    TOPO --> TACT
    TACT --> PAC
    TACT --> GHO
    NET --> PAC
    NET --> GHO
    
    PAC -->|Move| ENV[Environment]
    GHO -->|Move| ENV
```

---

## Kiến trúc Modular

```
24127457/
├── topology_analyzer.py   # Phân tích hình học bản đồ tĩnh (junction, dead-end, corridor, loop)
├── state_trackers.py      # SpatialHeatmap + BeliefStateTracker (theo dõi fog & xác suất enemy)
├── tactical_engines.py    # IntentTracker + TrapEvaluator + DynamicModeSelector
├── network_architect.py   # RecurrentActorCritic (6-channel CNN + LSTM)
├── agent.py               # Runtime entrypoint: PacmanAgent + GhostAgent
├── train_rl.py            # 3-Phase PPO training pipeline
└── train_pipeline.ipynb   # Notebook huấn luyện trực quan
```

---

## PacmanAgent — 3-Tier Lexicographic Architecture

Lấy cảm hứng từ "Fortress-Farmer" (Top 2 Global Bomberland), Pacman sử dụng pipeline quyết định phân tầng nghiêm ngặt:

### Tier 1: Safety & Trap Filter
- **Dead-end escape**: Nếu Pacman đang ở ô có ≤1 lối thoát → buộc rời ngay
- **Corridor trap avoidance**: Lọc bỏ các nước đi dẫn vào hành lang mà Ghost có thể thoát
- **Anti-stuck**: Phát hiện oscillation (lặp vị trí ≥4 lần) → reset LSTM + random move
- Nếu Tier 1 kích hoạt → **OVERRIDE** toàn bộ tier dưới

### Tier 2: Dynamic Interception & Intent Prediction
- **Phase A — Intent Projection**: Nếu Ghost di chuyển ổn định ≥2 bước cùng hướng → dự đoán junction đích đến (depth 5) → tính bottleneck node để chặn đầu
- **Phase B — Direct Chase**: A* trực tiếp đến Ghost nhưng validate đường đi không qua dead-end corridor
- **Phase C — Belief-Guided Search**: Nếu mất dấu ≤15 bước → dùng BeliefStateTracker dự đoán vị trí Ghost → A* đến cell có xác suất cao nhất

### Tier 3: Strategic Exploration & DRQN Fallback
- **Topological Heatmap**: Tìm ô có fog-age cao nhất × topology weight (junction=4x, core=3x, corridor=0.5x)
- **Frontier Search**: Tìm frontier cell (biết/không biết) gần nhất có trọng số topology cao
- **DRQN Fallback**: Nếu heuristic không quyết định được → gọi RecurrentActorCritic (6-channel CNN + LSTM)

### PacmanAgent Decision Flow

```mermaid
flowchart TD
    START[step: map_state, my_pos, enemy_pos] --> MEM[_update_memory<br/>accumulate map]
    MEM --> TOPO[_ensure_topo<br/>analyze junctions/corridors]
    TOPO --> BELIEF[_ensure_belief<br/>update enemy probability]
    BELIEF --> T1{Tier 1:<br/>Safety Filter}
    
    T1 -->|Dead-end ≤1 exit| ESC[Force escape<br/>to junction]
    T1 -->|Trap detected| SAFE[Choose safe move<br/>avoid corridors]
    T1 -->|Anti-stuck oscillation| RESET[Reset LSTM<br/>random move]
    T1 -->|Pass| T2{Tier 2:<br/>Interception}
    
    T2 -->|Enemy visible<br/>streak ≥ 2| INTENT[Project intent<br/>predict junction]
    INTENT -->|Bottleneck found| INTERCEPT[A* to intercept<br/>block escape]
    T2 -->|Enemy visible| CHASE[A* direct chase<br/>validate path]
    T2 -->|Enemy lost ≤15 steps| BELIEF_SEARCH[Belief-guided<br/>search]
    T2 -->|Cannot decide| T3{Tier 3:<br/>Exploration}
    
    T3 --> HEAT[Topological Heatmap<br/>fog-age × topology]
    T3 --> FRONTIER[Frontier Search<br/>known→unknown boundary]
    T3 --> RL[DRQN Fallback<br/>6-channel CNN+LSTM]
    
    ESC --> ACTION[Return Move]
    SAFE --> ACTION
    RESET --> ACTION
    INTERCEPT --> ACTION
    CHASE --> ACTION
    BELIEF_SEARCH --> ACTION
    HEAT --> ACTION
    FRONTIER --> ACTION
    RL --> ACTION
```

---

## GhostAgent — Dynamic Mode Selection Architecture

Lấy cảm hứng từ "samnu" HCMUS (Multi-Layer Predictive Core), Ghost sử dụng bộ lọc an toàn tương lai + 4 mode di chuyển linh hoạt:

### Tier 1: Provable Survival Gate
- **Dead-end escape**: Nếu Ghost đang ở dead-end/corridor → buộc thoát ra junction
- **Escape margin check**: Mô phỏng reachable space của Ghost vs Pacman sau 6 bước → nếu ratio < 0.5 → báo động đỏ, chỉ chọn nước đi mở rộng không gian
- Nếu Tier 1 kích hoạt → **OVERRIDE** toàn bộ tier dưới

### Tier 2: Dynamic Mode Dispatch
Mode được chọn dựa trên `distance_to_enemy`, `game_progress`, `topology_safety`:

| Mode | Điều kiện | Chiến lược |
|------|-----------|------------|
| **PANIC** | distance < 4 | 3-ply Minimax (Ghost→Pacman→Ghost) với trap detection |
| **EVASION** | 4 ≤ distance < 8 | Strategic Flee + Future Simulation (2-step Pacman lookahead) |
| **FORTRESS** | distance ≥ 8 hoặc game > 75% | Maximize distance to threat_center, ưu tiên loops/junctions |
| **EXPLORATION** | enemy not visible | Belief-guided safe explore, BFS 6-step, topology-weighted scoring |

### Tier 3: DRQN Fallback
- Nếu heuristic không quyết định được → gọi RecurrentActorCritic (5-action)

### GhostAgent Decision Flow

```mermaid
flowchart TD
    START[step: map_state, my_pos, enemy_pos] --> MEM[_update_memory<br/>accumulate map]
    MEM --> TOPO[_ensure_topo<br/>analyze from STATIC_FULL_MAP]
    TOPO --> STUCK{Anti-stuck<br/>check}
    
    STUCK -->|Oscillation detected| RESET[Reset LSTM<br/>random move]
    STUCK -->|Pass| T1{Tier 1:<br/>Survival Gate}
    
    T1 -->|In dead-end/corridor| ESC[Force escape<br/>to junction]
    T1 -->|Escape margin < 0.5| OPEN[Choose move<br/>to open space]
    T1 -->|Safe| T2{Tier 2:<br/>Mode Dispatch}
    
    T2 --> MODE{DynamicModeSelector<br/>based on distance & game}
    
    MODE -->|dist < 4| PANIC[PANIC MODE<br/>3-ply Minimax<br/>Ghost→Pacman→Ghost]
    MODE -->|4 ≤ dist < 8| EVASION[EVASION MODE<br/>Strategic Flee<br/>+ Future Simulation]
    MODE -->|dist ≥ 8<br/>or game > 75%| FORTRESS[FORTRESS MODE<br/>Maximize distance<br/>to threat_center<br/>patrol loops]
    MODE -->|enemy not visible| EXPLORATION[EXPLORATION MODE<br/>Belief-guided<br/>safe explore<br/>BFS 6-step]
    
    PANIC --> CHECK{Move valid?<br/>dist ≥ 2}
    EVASION --> CHECK
    FORTRESS --> CHECK
    EXPLORATION --> CHECK
    
    CHECK -->|Yes| ACTION[Return Move]
    CHECK -->|No| T3{Tier 3:<br/>DRQN Fallback}
    
    T3 --> RL[RecurrentActorCritic<br/>5-action CNN+LSTM]
    RL --> ACTION
    
    ESC --> ACTION
    OPEN --> ACTION
    RESET --> ACTION
```

### Mode Selection Logic

```mermaid
graph LR
    subgraph "Input Signals"
        D[distance_to_enemy]
        G[game_progress]
        T[topology_safety]
    end
    
    subgraph "Mode Selector"
        SEL{DynamicModeSelector<br/>hysteresis=3}
    end
    
    subgraph "Output Modes"
        P[PANIC<br/>Minimax 3-ply]
        E[EVASION<br/>Strategic Flee]
        F[FORTRESS<br/>Max Distance]
        X[EXPLORATION<br/>Belief-guided]
    end
    
    D --> SEL
    G --> SEL
    T --> SEL
    
    SEL -->|dist < 4| P
    SEL -->|4 ≤ dist < 8| E
    SEL -->|dist ≥ 8<br/>or game > 75%| F
    SEL -->|enemy None| X
```

---

## 6-Channel CNN Architecture

Mạng neural nhận 6 kênh đầu vào để "nhìn thấy" trực tiếp hình học bản đồ:

```
Channel 0: Wall Map          — 1.0 tại ô tường, 0.0 tại ô trống
Channel 1: Seen-Empty Map    — 1.0 tại ô trống đang thấy, 0.0 tại fog
Channel 2: Fog Map           — 1.0 tại ô chưa khám phá (unseen)
Channel 3: Enemy Position    — 1.0 tại vị trí enemy (nếu thấy), 0.0 nếu mất dấu
Channel 4: Belief Map        — Phân phối xác suất Bayesian về vị trí enemy
Channel 5: Topology Map      — Trọng số hình học (junction=1.0, core=0.7, corridor=0.3, dead-end=0.0)
```

**Position Vector (7-dim)**:
```
[my_r/H, my_c/W, enemy_r/H, enemy_c/W, visible_flag, threat_level, game_progress]
```

Kiến trúc mạng:
```
obs (6×21×21) → Conv2D(6→16, k3, s2) → Conv2D(16→32, k3, s2) → FC(1152→128)
pos (7)       → FC(7→32)
                                         concat → FC(160→128) → LSTM(128) → Actor(128→N_act)
                                                                           → Critic(128→1)
```

### CNN Architecture Diagram

```mermaid
flowchart LR
    subgraph "Input Channels (6×21×21)"
        C0[Ch0: Wall Map]
        C1[Ch1: Seen-Empty]
        C2[Ch2: Fog Map]
        C3[Ch3: Enemy Position]
        C4[Ch4: Belief Map]
        C5[Ch5: Topology Map]
    end
    
    subgraph "CNN Feature Extractor"
        CONV1[Conv2D<br/>6→16, k3, s2<br/>21×21 → 11×11]
        CONV2[Conv2D<br/>16→32, k3, s2<br/>11×11 → 6×6]
        FC1[FC 1152→128]
        CONV1 --> CONV2 --> FC1
    end
    
    subgraph "Position Vector (7-dim)"
        POS[my_r, my_c<br/>enemy_r, enemy_c<br/>visible, threat, progress]
        FC2[FC 7→32]
        POS --> FC2
    end
    
    subgraph "Temporal Processing"
        CONCAT[Concat 128+32=160]
        FC3[FC 160→128]
        LSTM[LSTM 128]
        FC1 --> CONCAT
        FC2 --> CONCAT
        CONCAT --> FC3 --> LSTM
    end
    
    subgraph "Output Heads"
        ACTOR[Actor Head<br/>128→N_actions]
        CRITIC[Critic Head<br/>128→1]
        LSTM --> ACTOR
        LSTM --> CRITIC
    end
    
    C0 & C1 & C2 & C3 & C4 & C5 --> CONV1
```

---

## Training Pipeline — 3-Phase Curriculum Learning

```mermaid
flowchart TD
    subgraph "Phase 1: Train Pacman (1000 episodes)"
        P1_START[Load pacman_model_bc.pth<br/>warm-start]
        P1_TRAIN[Train Pacman DRQN<br/>vs Frozen Ghost V6]
        P1_REWARD[Reward Shaping:<br/>+200 capture, +10 dead-end<br/>+5 corridor, +3 proximity]
        P1_SAVE[Save pacman_model.pth]
        P1_START --> P1_TRAIN --> P1_REWARD --> P1_SAVE
    end
    
    subgraph "Phase 2: Train Ghost (1000 episodes)"
        P2_START[Load ghost_model.pth<br/>if exists]
        P2_TRAIN[Train Ghost DRQN<br/>vs Frozen Pacman V6]
        P2_REWARD[Reward Shaping:<br/>-200 captured, +5 loop<br/>+3 junction, -10 dead-end]
        P2_SAVE[Save ghost_model.pth]
        P2_START --> P2_TRAIN --> P2_REWARD --> P2_SAVE
    end
    
    subgraph "Phase 3: Joint Self-Play (2000 episodes)"
        P3_START[Load both models<br/>from Phase 1 & 2]
        P3_TRAIN[Train Both Agents<br/>co-evolution]
        P3_ENTROPY[Entropy Decay:<br/>0.15 × 0.9992^ep]
        P3_LR[LR = 1/3 of Phase 1&2<br/>fine-tuning]
        P3_SAVE[Save final models]
        P3_START --> P3_TRAIN --> P3_ENTROPY --> P3_LR --> P3_SAVE
    end
    
    P1_SAVE --> P2_START
    P2_SAVE --> P3_START
    
    subgraph "Output"
        FINAL[pacman_model.pth<br/>ghost_model.pth]
    end
    
    P3_SAVE --> FINAL
```

### Phase 1: Train Pacman DRQN vs Frozen Ghost Heuristic (1000 episodes)
- **Mục tiêu**: Huấn luyện Pacman DRQN để đuổi bắt Ghost V6 heuristic (đóng băng)
- **Warm-start**: `pacman_model_bc.pth` (Behavioral Cloning từ expert data)
- **Reward shaping**:
  - +200 nếu bắt được Ghost
  - -1.5 penalty mỗi bước
  - +1.0 × (prev_dist - cur_dist) — thưởng khi tiến gần Ghost
  - +10.0 nếu Ghost ở dead-end, +5.0 nếu Ghost ở corridor, +3.0 × (4 - ghost_exits) — thưởng khi dồn ép
  - +5.0 × (4 - dist) nếu dist < 4 — thưởng proximity

### Phase 2: Train Ghost DRQN vs Frozen Pacman Heuristic (1000 episodes)
- **Mục tiêu**: Huấn luyện Ghost DRQN để né tránh Pacman V6 heuristic (đóng băng)
- **Reward shaping**:
  - -200 nếu bị bắt
  - +1.0 - 1.0/dist + (cur_dist - prev_dist) × 0.5 — thưởng khi duy trì/khoảng cách
  - +3.0 nếu ở junction, +5.0 nếu ở loop, +2.0 nếu ở core — thưởng topology an toàn
  - -10.0 nếu ở dead-end, -3.0 nếu ở corridor — phạt topology nguy hiểm
  - +2.0 nếu exits ≥ 3 — thưởng không gian thoát

### Phase 3: Joint Self-Play Co-Evolution (2000 episodes)
- **Mục tiêu**: Cả Pacman và Ghost cùng train đối kháng (self-play)
- **Entropy decay**: `0.15 × 0.9992^ep` (giảm dần để khám phá → khai thác)
- **Learning rate**: Giảm còn 1/3 so với Phase 1 & 2 (fine-tuning)
- **Reward**: Kết hợp reward shaping từ Phase 1 & 2

---

## Cách chạy

### 1. Smoke test (kiểm tra code hoạt động)
```bash
cd blind/src
python arena.py --seek 24127457 --hide 24127457 \
    --pacman-obs-radius 5 --ghost-obs-radius 5 --max-steps 200 --no-viz
```

### 2. Huấn luyện model weights
```bash
cd blind/submissions/24127457
python train_rl.py --device cpu          # CPU
python train_rl.py --device cuda         # GPU
```

Hoặc dùng notebook:
```bash
jupyter notebook train_pipeline.ipynb
```

### 3. Benchmark sau khi train
```bash
cd blind
python scripts/benchmark_full.py --seek 24127457 --hide 24127457 --games 50
```

---

## Ước tính thời gian huấn luyện

| Cấu hình | Phase 1 (1000 ep) | Phase 2 (1000 ep) | Phase 3 (2000 ep) | **Tổng** |
|----------|-------------------|-------------------|-------------------|----------|
| CPU (i7/Ryzen 7) | ~12 phút | ~12 phút | ~25 phút | **~50 phút** |
| RTX 3060/4060 | ~3 phút | ~3 phút | ~6 phút | **~12 phút** |

---

## Ràng buộc kỹ thuật

- **Thời gian/step**: ≤ 1.0 giây (STEP_TIME_LIMIT = 0.90s với time bailout)
- **Bộ nhớ**: ≤ 128 MB (CNN ~800KB + belief/topo/heatmap < 50KB → tổng < 50MB)
- **Thư viện**: numpy, torch (PyTorch)

---

## Tham khảo

- **Fortress-Farmer** (Top 2 Global Bomberland): 3-Tier Lexicographic Architecture
- **samnu HCMUS** (Bomberland): Multi-Layer Predictive Core, Enemy Intent Tracking
- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)
- **GAE**: Generalized Advantage Estimation (Schulman et al., 2016)
