# Ghost Agent V3 — SharedGhostBrain + Prediction + Risk-aware Multi-policy Strategy

Xây dựng kế hoạch triển khai `agent2.py`/`agent_v3.py` cho bài Hide-Seek dựa trên `agent.py` hiện tại và `implementation_plan.md` cũ. Mục tiêu là tạo một ghost agent có khả năng:

- Cache bản đồ ngay lần chạy đầu.
- Học online hành vi Pacman bằng Markov model.
- Dự đoán Pacman theo 2 bước/lượt.
- Tạo danger map theo thời gian.
- Tránh tunnel, dead-end, chokepoint và trap region.
- Kết hợp nhiều policy như Hybrid, Offline Table, Alpha-Beta, One-ply, Greedy, Monte Carlo.
- Dùng Safety Shield để chặn các action nguy hiểm trước khi trả về.
- Có thể tái sử dụng các module này để nâng cấp nhiều ghost agent khác.

---

## Proposed Changes

### [NEW] `submissions/24127192/agent2.py`

Tạo file agent mới dựa trên `agent.py` gốc, nhưng tổ chức lại thành kiến trúc module hóa. File vẫn chỉ thao tác trong thư mục `24127192`, không sửa các file framework khác.

Kiến trúc mới:

```text
Observation
    ↓
MapRepository / MapCache
    ↓
TopologyAnalyzer
    ↓
PacmanTracker / BeliefState
    ↓
OpponentModelEnsemble
    ↓
RiskEngine / DangerMap
    ↓
PolicyPortfolio
    ↓
SafetyShield
    ↓
Validate Action
```

---

## 1. Map Cache & Distance Cache

### Mục tiêu

Cache bản đồ ngay lần đầu chạy để giảm chi phí BFS lặp lại trong mỗi lượt.

### Thành phần

- `MapFingerprint`: nhận diện map hiện tại bằng hash.
- `MapCache`: lưu valid cells, walls, adjacency list.
- `DistanceCache`: BFS lazy cache + precompute cho junctions/chokepoints.
- `LandmarkDistance`: heuristic khoảng cách cho map lớn.

### Dữ liệu lưu

```python
MapCache = {
    "fingerprint": map_hash,
    "width": W,
    "height": H,
    "walls": set(),
    "valid_cells": set(),
    "neighbors": dict[cell, list[cell]],
    "distance_cache": DistanceCache,
    "topology": Topology
}
```

### Mã giả

```python
function GET_MAP_CACHE(map_state):
    fingerprint = HASH_MAP(map_state)

    if fingerprint in GLOBAL_MAP_REPOSITORY:
        return GLOBAL_MAP_REPOSITORY[fingerprint]

    valid_cells = FIND_WALKABLE_CELLS(map_state)
    neighbors = BUILD_ADJACENCY(valid_cells, map_state)

    distance_cache = DistanceCache(neighbors)
    topology = TopologyAnalyzer(valid_cells, neighbors).analyze()

    map_cache = MapCache(
        fingerprint,
        valid_cells,
        neighbors,
        distance_cache,
        topology
    )

    GLOBAL_MAP_REPOSITORY[fingerprint] = map_cache
    return map_cache
```

### Counter

- Tránh tính BFS lại liên tục.
- Tránh dùng nhầm cache giữa các map khác nhau.
- Hỗ trợ tất cả agent cần khoảng cách nhanh.

---

## 2. Topology Analysis

### Mục tiêu

Phân tích cấu trúc bản đồ để ghost không chỉ chạy xa Pacman mà còn biết vùng nào là bẫy.

### Phát hiện

- Dead-end.
- Junction.
- Chokepoint / articulation point.
- Corridor.
- Tunnel.
- Loop region.
- Core safe region.
- Trap depth.
- Escape capacity.
- Biconnected components.

### Mã giả

```python
function ANALYZE_TOPOLOGY(valid_cells, neighbors):
    dead_ends = {cell | degree(cell) == 1}
    junctions = {cell | degree(cell) >= 3}
    corridors = {cell | degree(cell) == 2}

    chokepoints = FIND_ARTICULATION_POINTS(valid_cells, neighbors)
    tunnels = DETECT_TUNNELS(corridors, junctions, neighbors)
    loop_set = DETECT_CYCLES(valid_cells, neighbors)
    core = FIND_SAFE_CORE(loop_set, junctions, chokepoints)

    trap_depth = COMPUTE_TRAP_DEPTH(dead_ends, core, neighbors)
    escape_capacity = COMPUTE_ESCAPE_CAPACITY(valid_cells, neighbors, chokepoints)
    biconnected = FIND_BICONNECTED_COMPONENTS(valid_cells, neighbors)

    return Topology(
        dead_ends,
        junctions,
        corridors,
        chokepoints,
        tunnels,
        loop_set,
        core,
        trap_depth,
        escape_capacity,
        biconnected
    )
```

### Counter

- Ghost bị dồn vào ngõ cụt.
- Ghost chạy vào tunnel khi Pacman gần entrance.
- Pacman không đuổi trực tiếp mà chặn chokepoint.
- Greedy distance chọn nhầm vùng có vẻ xa nhưng không có lối thoát.

---

## 3. Pacman Tracker & Belief State

### Mục tiêu

Thay `last_known_enemy_pos` bằng phân phối xác suất Pacman có thể đang ở đâu khi không nhìn thấy.

### Dữ liệu

```python
PacmanTracker = {
    "last_seen": cell | None,
    "history": list[cell],
    "belief": dict[cell, probability]
}
```

### Mã giả

```python
function UPDATE_BELIEF(enemy_position, visibility_info, map_cache, opponent_models):
    if enemy_position is not None:
        belief = {tuple(enemy_position): 1.0}
        last_seen = tuple(enemy_position)
        history.append(tuple(enemy_position))
        return belief

    new_belief = empty_distribution()

    for old_cell, old_prob in belief.items():
        reachable = GET_REACHABLE_CELLS(old_cell, steps=2, map_cache)

        for next_cell in reachable:
            transition_prob = opponent_models.transition_probability(
                old_cell,
                next_cell
            )
            new_belief[next_cell] += old_prob * transition_prob

    new_belief = APPLY_VISIBILITY_FILTER(new_belief, visibility_info)
    belief = NORMALIZE(new_belief)
    return belief
```

### Counter

- Pacman mất khỏi tầm nhìn rồi đổi hướng.
- Ghost dùng vị trí cũ đã lỗi thời.
- Pacman lợi dụng fog-of-war.

---

## 4. Online Markov Model

### Mục tiêu

Học online pattern di chuyển của Pacman trong trận.

### State đề xuất

```python
state = (
    relative_pos_bucket,
    pacman_geometry_bucket,
    ghost_direction_bucket,
    incoming_direction,
    local_topology_type
)
```

### Action

```python
action = (dr, dc)
```

### Dữ liệu

```python
transition_count[state][action] += 1
transition_recent[state][action] += recency_weight
```

### Mã giả cập nhật

```python
function UPDATE_MARKOV(pacman_history, ghost_position, map_cache):
    if len(pacman_history) < 3:
        return

    p0 = pacman_history[-3]
    p1 = pacman_history[-2]
    p2 = pacman_history[-1]

    state = ENCODE_STATE(p0, p1, ghost_position, map_cache)
    action = (p2.row - p1.row, p2.col - p1.col)

    transition_count[state][action] += 1
    transition_recent[state][action] += 1
```

### Mã giả dự đoán 1 bước

```python
function PREDICT_MARKOV_1_STEP(pacman_prev, pacman_cur, ghost_pos, map_cache):
    state = ENCODE_STATE(pacman_prev, pacman_cur, ghost_pos, map_cache)

    if state not in transition_count:
        return UNIFORM_LEGAL_DISTRIBUTION(pacman_cur, map_cache)

    counts = transition_count[state]
    probs = NORMALIZE_WITH_SMOOTHING(counts)
    return probs
```

### Mã giả dự đoán 2 bước

```python
function PREDICT_MARKOV_2_STEPS(pacman_prev, pacman_cur, ghost_pos, map_cache):
    dist1 = PREDICT_MARKOV_1_STEP(pacman_prev, pacman_cur, ghost_pos, map_cache)
    final_dist = empty_distribution()

    for action1, prob1 in dist1.items():
        pos1 = APPLY_ACTION(pacman_cur, action1)
        dist2 = PREDICT_MARKOV_1_STEP(pacman_cur, pos1, ghost_pos, map_cache)

        for action2, prob2 in dist2.items():
            pos2 = APPLY_ACTION(pos1, action2)
            final_dist[pos2] += prob1 * prob2

    return NORMALIZE(final_dist)
```

### Markov Confidence

```python
function MARKOV_CONFIDENCE(state):
    n = SUM(transition_count[state])
    sample_factor = n / (n + K)

    sorted_probs = SORT_DESC(NORMALIZE(transition_count[state]))
    probability_margin = sorted_probs[0] - sorted_probs[1]

    recency_factor = COMPUTE_RECENCY_FACTOR(state)

    return sample_factor * probability_margin * recency_factor
```

### Counter

- Pacman có thói quen rẽ theo pattern.
- Pacman thường đi shortest path.
- Pacman hay quay đầu hoặc chặn đường.
- Tuy nhiên không quá tin khi dữ liệu ít.

---

## 5. Opponent Model Ensemble

### Mục tiêu

Không phụ thuộc vào một Markov Model duy nhất. Kết hợp nhiều predictor và tự điều chỉnh trọng số theo độ chính xác.

### Các model

```text
1. MarkovOrder1
2. MarkovOrder2
3. ShortestPathChaserModel
4. InterceptionModel
5. RandomLegalModel
6. AdversarialModel
```

### Dữ liệu

```python
model_weights = {
    "markov1": 1.0,
    "markov2": 1.0,
    "shortest_path": 1.0,
    "interceptor": 1.0,
    "random": 0.5,
    "adversarial": 0.8
}
```

### Mã giả dự đoán

```python
function ENSEMBLE_PREDICT(observation, belief, map_cache):
    final_distribution = empty_distribution()

    for model in models:
        dist = model.predict(observation, belief, map_cache)
        weight = model_weights[model.name]

        for cell, prob in dist.items():
            final_distribution[cell] += weight * prob

    return NORMALIZE(final_distribution)
```

### Mã giả cập nhật trọng số

```python
function UPDATE_MODEL_WEIGHTS(actual_pacman_move):
    for model in models:
        predicted_prob = model.last_prediction.get(actual_pacman_move, epsilon)
        prediction_error = -log(predicted_prob)

        model_weights[model.name] *= exp(-learning_rate * prediction_error)

    model_weights = NORMALIZE_WEIGHTS(model_weights)
```

### Counter

- Pacman đổi chiến thuật giữa trận.
- Markov bị lừa bằng pattern giả.
- Pacman random.
- Pacman chơi kiểu interceptor thay vì greedy chase.

---

## 6. Time-expanded Danger Map

### Mục tiêu

Tạo danger map theo thời gian thay vì danger map tĩnh.

Pacman đi 2 bước/lượt, nên sau `t` lượt Pacman có thể tới các cell có graph distance `<= 2t`.

### Dữ liệu

```python
danger[t][cell] = risk_score
```

### Mã giả

```python
function BUILD_TIME_EXPANDED_DANGER(belief, prediction, map_cache, horizon):
    danger = [empty_map() for t in range(horizon + 1)]

    current_distribution = MERGE_BELIEF_AND_PREDICTION(belief, prediction)

    for t in range(0, horizon + 1):
        for pac_cell, prob in current_distribution.items():
            reachable = GET_REACHABLE_CELLS(
                pac_cell,
                steps=2,
                map_cache=map_cache
            )

            for cell, dist in reachable.items():
                topology_factor = TOPOLOGY_RISK_FACTOR(cell, map_cache.topology)
                time_decay = GAMMA ** t
                distance_decay = 1.0 / (1.0 + dist)

                danger[t][cell] += prob * topology_factor * time_decay * distance_decay

        current_distribution = PROPAGATE_PACMAN_DISTRIBUTION(
            current_distribution,
            map_cache
        )

    return danger
```

### Counter

- Ghost đi vào cell hiện tại an toàn nhưng sẽ nguy hiểm sau 1-2 lượt.
- Pacman nhanh hơn ghost.
- Pacman chiếm lối thoát thay vì đuổi thẳng.

---

## 7. Survival Margin

### Mục tiêu

Đánh giá ghost có kịp thoát khỏi vùng nguy hiểm trước Pacman hay không.

### Công thức

```text
margin(exit) = pacman_arrival_time(exit) - ghost_arrival_time(exit)
```

Với Pacman speed = 2, Ghost speed = 1:

```text
pacman_arrival_time = ceil(distance(pacman, exit) / 2)
ghost_arrival_time  = distance(ghost, exit)
```

### Mã giả

```python
function COMPUTE_SURVIVAL_MARGIN(ghost_pos, pacman_distribution, map_cache):
    exits = FIND_CANDIDATE_EXITS(ghost_pos, map_cache.topology)
    best_margin = -INF

    for exit_cell in exits:
        ghost_time = DIST(ghost_pos, exit_cell, map_cache)

        worst_pacman_time = INF
        for pac_cell, prob in pacman_distribution.items():
            pac_time = CEIL(DIST(pac_cell, exit_cell, map_cache) / 2)
            worst_pacman_time = MIN(worst_pacman_time, pac_time)

        margin = worst_pacman_time - ghost_time
        best_margin = MAX(best_margin, margin)

    return best_margin
```

### Counter

- Chạy xa nhưng vào trap.
- Pacman chặn exit.
- Tunnel có vẻ an toàn nhưng Pacman tới cửa trước.

---

## 8. Viability Kernel Local

### Mục tiêu

Kiểm tra một state có còn sống được trong K lượt không.

### Mã giả

```python
function LOCAL_VIABILITY(ghost_pos, pacman_distribution, map_cache, K):
    viable[0] = ALL_NON_CAPTURE_LOCAL_STATES()

    for t in range(1, K + 1):
        for state in LOCAL_STATES_AROUND(ghost_pos, pacman_distribution):
            ghost_actions = LEGAL_GHOST_ACTIONS(state.ghost)

            viable_action_exists = False

            for g_action in ghost_actions:
                ghost_next = APPLY_ACTION(state.ghost, g_action)

                safe_against_all = True
                for pac_next in DANGEROUS_PACMAN_RESPONSES(state.pacman):
                    next_state = (ghost_next, pac_next)

                    if next_state not in viable[t - 1]:
                        safe_against_all = False
                        break

                if safe_against_all:
                    viable_action_exists = True
                    break

            viable[t][state] = viable_action_exists

    return viable[K]
```

### Counter

- Action hiện tại tốt nhưng chắc chắn chết sau vài lượt.
- Pacman chơi gần tối ưu.
- False-safe cells.

---

## 9. Policy Portfolio thay cho strict fallback

### Mục tiêu

Không để một layer trả action quá sớm rồi bỏ qua các policy khác tốt hơn.

Thay vì:

```text
Layer 0 fail → Layer 1 fail → Layer 2 fail → ...
```

Dùng:

```text
Nhiều policy cùng đề xuất action → Safety Shield → Arbitrator chọn action tốt nhất
```

### Proposal format

```python
Proposal = {
    "action": Move,
    "estimated_value": float,
    "estimated_risk": float,
    "confidence": float,
    "source_policy": str,
    "cost": float,
    "reason": str
}
```

### Mã giả

```python
function COLLECT_POLICY_PROPOSALS(context):
    proposals = []

    for policy in selected_policies:
        try:
            proposal = policy.propose(context)
            if proposal is not None:
                proposals.append(proposal)
        except Exception as e:
            diagnostics.record_failure(policy.name, e)

    return proposals
```

### Arbitration

```python
function CHOOSE_PROPOSAL(proposals, risk_context):
    safe_proposals = SAFETY_SHIELD(proposals, risk_context)

    if safe_proposals is empty:
        return ROBUST_GREEDY_FALLBACK(risk_context)

    best = None
    best_score = -INF

    for p in safe_proposals:
        score = (
            p.estimated_value
            - RISK_WEIGHT * p.estimated_risk
            + CONFIDENCE_WEIGHT * p.confidence
            - COST_WEIGHT * p.cost
        )

        if score > best_score:
            best_score = score
            best = p

    return best.action
```

### Counter

- Layer 0 luôn trả action nhưng không phải tốt nhất.
- Offline table lỗi thời.
- Alpha-beta có kết quả tốt nhưng bị bỏ qua.
- Markov confidence thấp nhưng vẫn bị dùng quá mạnh.

---

## 10. Multi-layer Decision System

Vẫn giữ khái niệm fallback layer để đảm bảo an toàn khi policy fail/timeout, nhưng layer được triển khai trong `PolicyPortfolio`.

### Layer 0 — Hybrid Policy + Markov + MC Rollout

#### Dùng khi

- Pacman visible.
- Distance <= 8.
- Danger score cao.

#### Thành phần

- Table value floor.
- Markov prediction.
- Time-expanded danger.
- Monte Carlo rollout depth 15, rollout count 8.
- Flee tie-break.

#### Counter

- Pacman đuổi gần.
- Pacman đi 2 bước/lượt.
- Cần quyết định mạnh ngay.
- Khoảng cách hiện tại gây đánh lừa.

#### Mã giả

```python
function HYBRID_POLICY_PROPOSE(context):
    best_action = None
    best_score = -INF

    for action in LEGAL_ACTIONS(context.ghost_pos):
        ghost_next = APPLY_ACTION(context.ghost_pos, action)

        table_value = LOOKUP_TABLE_VALUE(ghost_next, context)
        markov_risk = DANGER_AT(ghost_next, context.danger)
        mc_value = MC_ROLLOUT_VALUE(action, context)
        flee_score = DISTANCE_FROM_DANGER(ghost_next, context)
        survival_margin = SURVIVAL_MARGIN_SCORE(ghost_next, context)

        score = (
            0.25 * table_value
            + 0.25 * mc_value
            + 0.25 * survival_margin
            + 0.15 * flee_score
            - 0.35 * markov_risk
        )

        if score > best_score:
            best_score = score
            best_action = action

    return Proposal(best_action, best_score, markov_risk, confidence="high", source="hybrid")
```

---

### Layer 1 — Offline Table + Markov Safety Check

#### Dùng khi

- State có trong bảng.
- Table confidence cao.
- Markov danger không quá cao.

#### Counter

- State quen thuộc.
- Tránh tính search lại từ đầu.
- Tận dụng kinh nghiệm offline.

#### Mã giả

```python
function TABLE_POLICY_PROPOSE(context):
    action = OFFLINE_TABLE_LOOKUP(context.state_key)

    if action is None:
        return None

    ghost_next = APPLY_ACTION(context.ghost_pos, action)

    if SAFETY_RISK(ghost_next, context) >= HIGH_DANGER:
        return None

    value = TABLE_VALUE(context.state_key, action)
    risk = SAFETY_RISK(ghost_next, context)

    return Proposal(action, value, risk, confidence="medium", source="table")
```

---

### Layer 2 — Iterative Deepening Alpha-Beta + Markov Branch Ordering

#### Dùng khi

- Pacman visible hoặc belief tập trung.
- Distance 3-15.
- Có đủ budget.

#### Counter

- Pacman chơi tối ưu.
- Pacman chặn đường.
- Ghost bị dồn vào trap sau vài bước.

#### Nâng cấp

- Transposition table.
- Move ordering.
- Quiescence search.
- Markov dùng để order branch, không xóa nhánh nguy hiểm xác suất thấp.

#### Mã giả

```python
function ALPHA_BETA_POLICY_PROPOSE(context):
    best_action = None
    best_value = -INF

    for depth in range(1, MAX_DEPTH + 1):
        if BUDGET_EXCEEDED():
            break

        action, value = ALPHA_BETA_ROOT(context, depth)

        if value > best_value:
            best_value = value
            best_action = action

    if best_action is None:
        return None

    risk = SAFETY_RISK(APPLY_ACTION(context.ghost_pos, best_action), context)
    return Proposal(best_action, best_value, risk, confidence="high", source="alpha_beta")
```

---

### Layer 3 — One-ply Search + Markov Evaluation

#### Dùng khi

- Thiếu thời gian.
- Search sâu timeout.
- Cần nước đi hợp lý ngay.

#### Counter

- Không đủ budget nhưng không muốn random/greedy thô.

#### Mã giả

```python
function ONE_PLY_POLICY_PROPOSE(context):
    best_action = None
    best_score = -INF

    for action in LEGAL_ACTIONS(context.ghost_pos):
        ghost_next = APPLY_ACTION(context.ghost_pos, action)
        score = EVALUATE_STATE_WITH_RISK(ghost_next, context)

        if score > best_score:
            best_score = score
            best_action = action

    return Proposal(best_action, best_score, SAFETY_RISK(ghost_next, context), "medium", "one_ply")
```

---

### Layer 4 — Greedy Escape + Markov Danger

#### Dùng khi

- Tất cả policy trên fail/timeout.
- Pacman không thấy rõ.
- Cần một action nhanh, chắc chắn hợp lệ.

#### Counter

- Lỗi toàn bộ hệ thống cao cấp.
- Vẫn phải chạy xa vùng nguy hiểm.

#### Mã giả

```python
function GREEDY_POLICY_PROPOSE(context):
    best_action = None
    best_score = -INF

    for action in LEGAL_ACTIONS(context.ghost_pos):
        ghost_next = APPLY_ACTION(context.ghost_pos, action)

        distance_score = DISTANCE_FROM_PACMAN_BELIEF(ghost_next, context)
        danger_score = DANGER_AT(ghost_next, context.danger)
        trap_penalty = TRAP_RISK(ghost_next, context.map_cache.topology)
        escape_score = ESCAPE_CAPACITY(ghost_next, context)
        loop_penalty = RECENT_POSITION_PENALTY(ghost_next, context)

        score = (
            4 * distance_score
            + 6 * escape_score
            - 8 * trap_penalty
            - 10 * danger_score
            - 2 * loop_penalty
        )

        if score > best_score:
            best_score = score
            best_action = action

    return Proposal(best_action, best_score, danger_score, "low", "greedy")
```

---

### Layer 5 — Validate + STAY

#### Counter

- Action `None`.
- Action sai kiểu.
- Action đi xuyên tường.
- Crash protection.

#### Mã giả

```python
function VALIDATE_OR_STAY(action, ghost_pos, map_cache):
    if action is None:
        return Move.STAY

    if action not in ALL_MOVES:
        return Move.STAY

    next_pos = APPLY_ACTION(ghost_pos, action)

    if next_pos not in map_cache.valid_cells:
        return Move.STAY

    return action
```

---

## 11. Safety Shield

### Mục tiêu

Mọi action từ mọi policy đều phải đi qua Safety Shield.

### Mức an toàn

```text
Level 1: legal safety
Level 2: immediate capture safety
Level 3: time-expanded danger safety
Level 4: local viability safety
```

### Mã giả

```python
function SAFETY_SHIELD(proposals, context):
    legal = []

    for proposal in proposals:
        action = proposal.action
        ghost_next = APPLY_ACTION(context.ghost_pos, action)

        if ghost_next not in context.map_cache.valid_cells:
            continue

        if CAN_PACMAN_CAPTURE_NEXT_TURN(ghost_next, context):
            continue

        if TIME_EXPANDED_DANGER(ghost_next, context) >= FATAL_DANGER:
            continue

        if LOCAL_VIABILITY_FAILS(ghost_next, context):
            continue

        legal.append(proposal)

    if legal:
        return legal

    return RELAXED_SAFETY(proposals, context)
```

### Relaxed fallback

```python
function RELAXED_SAFETY(proposals, context):
    return SORT_BY_MAXIMUM_CAPTURE_TIME(proposals, context)
```

### Counter

- Greedy chạy xa nhưng vào ngõ cụt.
- Table action lỗi thời.
- Monte Carlo action trung bình tốt nhưng có nhánh chết ngay.
- Alpha-beta bị horizon effect.

---

## 12. Risk-sensitive Monte Carlo

### Mục tiêu

Không chọn action chỉ vì điểm trung bình rollout cao. Ưu tiên action có rủi ro xấu nhất thấp.

### Score

```python
score = 0.4 * mean_return + 0.6 * CVaR_20
```

Trong đó `CVaR_20` là trung bình 20% rollout tệ nhất.

### Mã giả

```python
function MC_ROLLOUT_VALUE(action, context):
    returns = []
    rollout_count = ADAPTIVE_ROLLOUT_COUNT(context)

    for i in range(rollout_count):
        result = SIMULATE_ROLLOUT(action, context, depth=15)
        returns.append(result.survival_score)

    mean_return = MEAN(returns)
    cvar_20 = MEAN(WORST_PERCENTILE(returns, 20))

    return 0.4 * mean_return + 0.6 * cvar_20
```

### Counter

- Action 90% tốt nhưng 10% chết ngay.
- Pacman response hiếm nhưng nguy hiểm.
- Rollout trung bình đánh lừa agent.

---

## 13. Search Optimizations

### Nâng cấp Alpha-Beta

- Transposition table.
- Move ordering.
- Markov branch ordering.
- Quiescence search.
- Adaptive depth.

### Mã giả transposition table

```python
function ALPHA_BETA(state, depth, alpha, beta):
    key = (state.ghost_pos, state.pacman_belief_signature, depth)

    if key in TRANSPOSITION_TABLE:
        return TRANSPOSITION_TABLE[key]

    if depth == 0 or TERMINAL(state):
        value = EVALUATE_STATE_WITH_RISK(state)
        TRANSPOSITION_TABLE[key] = value
        return value

    ordered_moves = ORDER_MOVES_BY_SAFETY_AND_MARKOV(state)

    value = SEARCH_CHILDREN(ordered_moves, alpha, beta)
    TRANSPOSITION_TABLE[key] = value
    return value
```

### Counter

- Search lặp trạng thái.
- Timeout.
- Horizon effect.
- Nhánh Markov xác suất thấp nhưng nguy hiểm bị bỏ qua.

---

## 14. Anti-loop & Route Diversity

### Mục tiêu

Tránh ghost lặp lại vô ích, nhưng vẫn cho phép chạy vòng nếu loop đó thật sự an toàn.

### Dữ liệu

```python
recent_positions = last 8 cells
visit_count[cell] += 1
edge_visit_count[(prev_cell, cur_cell)] += 1
cycle_signature = detect_recent_cycle(recent_positions)
```

### Mã giả

```python
function LOOP_PENALTY(next_cell, context):
    if context.pacman_near and LOOP_REGION_IS_SAFE(next_cell, context):
        return small_penalty

    return (
        L1 * visit_count[next_cell]
        + L2 * edge_visit_count[(context.ghost_pos, next_cell)]
        + L3 * CYCLE_SIGNATURE_PENALTY(context.recent_positions)
    )
```

### Counter

- Ghost rung qua lại hai ô.
- Ghost chạy pattern dễ học.
- Không phá chiến thuật chạy vòng khi đó là lựa chọn sống tốt.

---

## 15. Pacman Style Classifier

### Mục tiêu

Nhận diện kiểu seeker để chọn policy phù hợp.

### Style

```text
GREEDY_CHASER
SHORTEST_PATH_CHASER
INTERCEPTOR
RANDOM_EXPLORER
PATTERNED_AGENT
ADVERSARIAL_SEARCHER
```

### Feature

```text
- Tỷ lệ move làm giảm distance tới ghost.
- Entropy action.
- Tỷ lệ quay đầu.
- Tỷ lệ chọn chokepoint.
- Accuracy của từng predictor.
- Tần suất intercept thay vì chase trực tiếp.
```

### Mã giả

```python
function CLASSIFY_PACMAN_STYLE(history, predictions, map_cache):
    features = EXTRACT_STYLE_FEATURES(history, predictions, map_cache)

    if features.distance_reduction_rate > 0.8:
        return SHORTEST_PATH_CHASER

    if features.chokepoint_preference > 0.6:
        return INTERCEPTOR

    if features.action_entropy > HIGH_ENTROPY:
        return RANDOM_EXPLORER

    if features.markov_accuracy > 0.7:
        return PATTERNED_AGENT

    return UNKNOWN
```

### Policy selection

```python
function SELECT_POLICIES(style, urgency):
    if style == GREEDY_CHASER:
        return [Hybrid, GreedyLoopEscape, OnePly]

    if style == INTERCEPTOR:
        return [AlphaBeta, SurvivalMarginPolicy, TunnelEscape]

    if style == RANDOM_EXPLORER:
        return [Greedy, SafetyShield, Table]

    if style == PATTERNED_AGENT:
        return [MarkovHybrid, MC, OnePly]

    return [Hybrid, AlphaBeta, OnePly, Greedy]
```

### Counter

- Một chiến thuật dùng cho mọi seeker.
- Pacman đổi phong cách.
- Pacman không đuổi trực tiếp mà chặn đường.

---

## 16. Adaptive Budget Controller

### Mục tiêu

Phân phối thời gian tính toán theo độ nguy hiểm.

### Urgency

```python
urgency = f(
    distance_to_pacman_belief,
    danger_probability,
    trap_depth,
    branching_factor,
    visibility
)
```

### Mã giả

```python
function SELECT_BUDGET_MODE(context):
    if context.danger_score >= CRITICAL:
        return "emergency"

    if context.pacman_visible and context.distance <= 8:
        return "combat"

    if context.in_tunnel:
        return "tunnel_escape"

    if context.pacman_far:
        return "cheap"

    return "normal"
```

### Policy theo mode

```text
cheap:
    update model + greedy safe + table

normal:
    table + one-ply + shallow alpha-beta

combat:
    hybrid + alpha-beta + MC

tunnel_escape:
    tunnel solver + survival margin + safety shield

emergency:
    local viability + robust greedy + validate
```

### Counter

- Tốn budget khi Pacman xa.
- Không đủ thời gian khi Pacman gần.
- Agent timeout.

---

## 17. Diagnostics

### Mục tiêu

Không nuốt lỗi âm thầm bằng `except Exception: action = None` mà không biết layer nào fail.

### Dữ liệu

```python
diagnostics = {
    "layer_failures": defaultdict(int),
    "timeouts": defaultdict(int),
    "table_misses": 0,
    "markov_accuracy": RunningAverage(),
    "model_weights": dict(),
    "last_selected_policy": str,
    "last_safety_rejection_reason": str
}
```

### Mã giả

```python
function RECORD_POLICY_FAILURE(policy_name, exception):
    diagnostics.layer_failures[policy_name] += 1
    diagnostics.last_error_type = TYPE(exception)

function RECORD_SELECTED_POLICY(policy_name):
    diagnostics.last_selected_policy = policy_name

function RECORD_SAFETY_REJECTION(action, reason):
    diagnostics.safety_rejections[reason] += 1
```

### Counter

- Không biết module nào không hoạt động.
- Không đo được Markov có tốt không.
- Không biết action bị Safety Shield loại vì lý do gì.

---

## 18. Main `GhostAgent.step()` Flow

### Mã giả tổng thể

```python
class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.map_repository = MapRepository()
        self.tracker = PacmanTracker()
        self.opponent_models = OpponentModelEnsemble()
        self.risk_engine = RiskEngine()
        self.policy_portfolio = PolicyPortfolio()
        self.safety_shield = SafetyShield()
        self.budget_controller = BudgetController()
        self.diagnostics = Diagnostics()

        self.recent_positions = []
        self.visit_count = defaultdict(int)
        self.edge_visit_count = defaultdict(int)

    def step(self, map_state, my_position, enemy_position, step_number):
        ghost_pos = tuple(my_position)

        # 1. Map cache
        map_cache = self.map_repository.get(map_state)

        # 2. Ghost memory
        self.update_ghost_history(ghost_pos)

        # 3. Pacman tracker / belief
        belief = self.tracker.update(
            enemy_position,
            ghost_pos,
            map_cache,
            self.opponent_models
        )

        # 4. Update Markov / opponent models
        self.opponent_models.update_if_observed(
            self.tracker.history,
            ghost_pos,
            map_cache
        )

        # 5. Ensemble prediction
        prediction = self.opponent_models.predict(
            ghost_pos,
            belief,
            map_cache
        )

        # 6. Risk context
        risk_context = self.risk_engine.build(
            ghost_pos=ghost_pos,
            belief=belief,
            prediction=prediction,
            map_cache=map_cache,
            recent_positions=self.recent_positions,
            visit_count=self.visit_count,
            edge_visit_count=self.edge_visit_count
        )

        # 7. Budget mode
        budget_mode = self.budget_controller.select_mode(risk_context)

        # 8. Select policies
        selected_policies = self.policy_portfolio.select(
            budget_mode,
            risk_context,
            self.opponent_models.style
        )

        # 9. Collect proposals
        proposals = []
        for policy in selected_policies:
            try:
                proposal = policy.propose(risk_context)
                if proposal is not None:
                    proposals.append(proposal)
            except Exception as e:
                self.diagnostics.record_failure(policy.name, e)

        # 10. Safety filter
        safe_proposals = self.safety_shield.filter(proposals, risk_context)

        # 11. Arbitration
        action = self.policy_portfolio.choose_best(
            safe_proposals,
            risk_context
        )

        # 12. Final validate
        action = VALIDATE_OR_STAY(action, ghost_pos, map_cache)

        return action
```

---

## 19. Nâng cấp áp dụng cho các ghost agent khác

### 19.1 Greedy Ghost

Nâng cấp:

```text
Greedy distance
+ danger map
+ survival margin
+ trap penalty
+ safety shield
```

Score:

```python
score = (
    4 * distance_from_belief
    + 6 * survival_margin
    + 3 * escape_capacity
    - 8 * trap_depth
    - 10 * danger_probability
)
```

---

### 19.2 Blind Ghost

Nâng cấp:

```text
last_known_position
→ belief distribution
→ Markov propagation
→ visibility filtering
```

---

### 19.3 Offline Table Ghost

Nâng cấp:

```text
table_action
+ Markov safety check
+ online residual correction
+ safety shield
```

Mã giả:

```python
final_value(action) = table_value(action) + online_residual(action)
```

---

### 19.4 Alpha-Beta Ghost

Nâng cấp:

```text
alpha-beta
+ transposition table
+ move ordering
+ quiescence search
+ Markov branch ordering
+ belief-state search
```

---

### 19.5 Monte Carlo Ghost

Nâng cấp:

```text
MC rollout
+ Markov-guided Pacman policy
+ topology-aware ghost policy
+ CVaR scoring
+ adaptive rollout count
```

---

### 19.6 Random/Fallback Ghost

Nâng cấp:

```python
safe_actions = SAFETY_SHIELD(legal_actions)
return WEIGHTED_RANDOM(safe_actions, weight=escape_margin + route_diversity)
```

---

## 20. Implementation Phases

### Phase 1 — Core safety, ít rủi ro

```text
1. Map fingerprint cache
2. DistanceCache lazy BFS
3. TopologyAnalyzer cơ bản
4. Time-expanded danger map
5. Safety Shield
6. Greedy + One-ply dùng danger scoring
7. Diagnostics tối thiểu
```

### Phase 2 — Online learning

```text
8. PacmanTracker belief state
9. MarkovModel order 1/order 2
10. Markov confidence
11. OpponentModelEnsemble
12. Model weight update
13. Pacman style classifier
```

### Phase 3 — Strong policies

```text
14. Hybrid Policy + Markov + MC
15. Risk-sensitive Monte Carlo
16. Alpha-Beta với transposition table
17. Move ordering + quiescence search
18. Tunnel escape solver
19. Survival margin nâng cao
```

### Phase 4 — Robustness

```text
20. Local Viability Kernel
21. Policy Portfolio arbitration
22. Adaptive Budget Controller
23. Full Diagnostics
24. A/B test với agent.py cũ
```

---

## 21. Verification Plan

### Manual Verification

Chạy agent cũ:

```bash
cd /Users/mac/Documents/AI/Hide-Seek/pacman/src
python arena.py --seek example_student --hide 24127192 --submissions-dir ../submissions --start-mode deterministic
```

Nếu agent loader mặc định đọc `agent.py`, có thể test bằng cách:

```bash
cp ../submissions/24127192/agent.py ../submissions/24127192/agent_old.py
cp ../submissions/24127192/agent2.py ../submissions/24127192/agent.py
python arena.py --seek example_student --hide 24127192 --submissions-dir ../submissions --start-mode deterministic
```

Sau khi test xong, restore:

```bash
mv ../submissions/24127192/agent_old.py ../submissions/24127192/agent.py
```

### A/B Test

So sánh:

```text
agent.py gốc vs agent2.py
```

Theo tiêu chí:

```text
- Số step sống sót trung bình.
- Tỷ lệ sống trên 100 step.
- Số lần bị bắt trong tunnel.
- Số lần đi vào dead-end khi Pacman gần.
- Markov prediction accuracy.
- Policy nào được chọn nhiều nhất.
- Số lần Safety Shield reject action.
- Số lần Alpha-Beta timeout.
```

### Stress Test

```text
1. Pacman greedy chase.
2. Pacman shortest path.
3. Pacman random.
4. Pacman interceptor.
5. Pacman adversarial.
6. Map nhiều tunnel.
7. Map nhiều loop.
8. Map có chokepoint hẹp.
9. Pacman visible liên tục.
10. Pacman thường xuyên mất dấu.
```

---

## 22. Expected Benefits

```text
Map Cache:
    Giảm chi phí tính toán lặp lại.

Belief State:
    Không còn phụ thuộc vào last_known_enemy_pos lỗi thời.

Markov + Ensemble:
    Dự đoán tốt hơn khi Pacman có pattern hoặc đổi chiến thuật.

Time-expanded Danger:
    Né vùng Pacman có thể tới trong tương lai, đặc biệt vì Pacman đi 2 bước/lượt.

Survival Margin:
    Tránh chạy vào tunnel hoặc exit bị Pacman chặn.

Safety Shield:
    Chặn action nguy hiểm từ mọi policy.

Policy Portfolio:
    Không bị phụ thuộc tuyệt đối vào một layer.

Risk-sensitive MC:
    Tránh action có xác suất chết nhỏ nhưng nghiêm trọng.

Diagnostics:
    Dễ debug, đo lường và cải tiến agent.
```

---

## 23. Final Target

Agent cuối nên đạt được các tính chất:

```text
1. Không crash.
2. Không đi xuyên tường.
3. Không quá tin vào Markov khi dữ liệu ít.
4. Không vào tunnel/dead-end khi Pacman gần.
5. Có thể sống tốt trước greedy Pacman.
6. Có khả năng chống Pacman interceptor.
7. Có fallback khi search timeout.
8. Có diagnostics để debug.
9. Có thể tái sử dụng module cho nhiều ghost agent khác.
10. Tối ưu mục tiêu sống >= 100 steps.
```

---

## 24. Notes for Implementation

- Không sửa file framework ngoài thư mục `24127192` nếu ràng buộc bài yêu cầu.
- Nên giữ `agent.py` cũ làm baseline.
- Tạo `agent2.py` trước, test độc lập.
- Khi cần nộp, đổi tên `agent2.py` thành `agent.py` hoặc import lại trong `agent.py`.
- Ưu tiên implement Phase 1 trước khi thêm các module học phức tạp.
- Mọi policy đều phải đi qua `SafetyShield` và `validate_or_stay`.
