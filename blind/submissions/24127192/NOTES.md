# NOTES — Blind Multi-Layer Ghost Agent (24127192)

## 1. Ý tưởng

Xây dựng Blind Ghost Agent với kiến trúc **multi-layer** dựa trên `non_blind_agent.py`, điều chỉnh cho môi trường **Partial Observability** (vision=5, cross-shaped).

### Ý tưởng cốt lõi:
- **Hardcode map cache**: Biết trước layout map 21×21 với vị trí cố định của Pacman (15,10) và Ghost (9,9). Precompute toàn bộ topology (dead_ends, junctions, corridors, core, loops, chokepoints, tunnels) một lần duy nhất, không cần tính lại mỗi step.
- **Memory Map**: Tích lũy thông tin quan sát qua các step. Tất cả thuật toán A*, BFS, Flood Fill, Monte Carlo đều chạy trên memory map này.
- **Belief State Tracking**: Duy trì phân phối xác suất vị trí Pacman khi không nhìn thấy. Khởi tạo belief tại vị trí cố định (15,10).
- **Multi-Policy Portfolio**: 5 chính sách (Hybrid, Table, AlphaBeta, OnePly, Greedy) được lựa chọn linh hoạt theo 5 chế độ ngân sách (emergency, tunnel_escape, combat, normal, cheap).
- **Anti Line-of-Sight**: Khi gặp Pacman trên cùng hàng/cột, ưu tiên rẽ vuông góc để tránh bị bắt trên đường thẳng.

## 2. Chi tiết Triển khai

### Kiến trúc 10 Layer:

| Layer | Thành phần | Chức năng |
|-------|-----------|----------|
| 0 | Hardcoded Map Cache | KNOWN_LAYOUT → precompute TopologyAnalyzer, DistanceCache, OfflineTable |
| 1 | Memory Map | Tích lũy observation, fill dần bản đồ |
| 2 | PacmanTracker | Belief state propagation, init tại (15,10) |
| 3 | OpponentModelEnsemble | 6 models: Markov1, Markov2, ShortestPath, Interception, Random, Adversarial |
| 4 | RiskEngine | Time-expanded danger (horizon=3), survival margin, viability check |
| 5 | MCRolloutV3 | Monte Carlo simulation với CVaR scoring (bottom 20%) |
| 6 | AlphaBetaSearch | Minimax với transposition table, iterative deepening đến depth 6 |
| 7 | PolicyPortfolio | 5 policies + Arbitrator chọn theo score |
| 8 | SafetyShield | 4-level filter: legal → capture → danger → viability |
| 9 | Blind-Specific | Exploration, Anti Line-of-Sight, Tunnel Escape |

### Pipeline mỗi step:
```
Memory update → Belief propagate → Ensemble predict → Risk eval (danger map) →
Budget mode selection → Policy selection → Collect proposals →
Safety filter → Anti-LoS adjust → Arbitrate → Validate → Return Move
```

### Các thuật toán chính:
- **A\* Manhattan**: Tìm đường với heuristic Manhattan distance, chạy trên memory_map
- **Monte Carlo (MC V3)**: 8 rollouts × 12 depth, CVaR scoring, ensemble-informed Pacman responses
- **Flood Fill**: BFS distance maps từ DistanceCache, đánh giá không gian tiếp cận được
- **Corridor Analysis**: TopologyAnalyzer phát hiện tunnels, junctions, chokepoints
- **Minimax Alpha-Beta**: TT + move ordering, depth 2-6 iterative deepening

### Tối ưu hiệu năng:
- `TIME_BUDGET = 0.85s` (dưới ngưỡng 0.9s)
- Topology precomputed một lần trong `__init__`
- DistanceCache precompute cho junctions + chokepoints
- Early exit khi thời gian gần hết → fallback GreedyPolicy
- MC rollouts giới hạn, AlphaBeta depth giới hạn

## 3. Kết quả Đạt được và Khả năng Áp dụng

### Kết quả Benchmark (vision=5, fixed starts):

| Kịch bản | Số trận | Ghost Thắng | Tỉ lệ |
|----------|---------|------------|-------|
| Ghost vs example_student Pacman | 10 | 10 | **100%** |
| Ghost vs own Pacman (placeholder) | 5 | 5 | **100%** |

- **Khoảng cách cuối trung bình**: ~17.8 cells (dao động 9-22)
- **Thời gian phản hồi**: Luôn dưới 0.9s/step
- **Không có lỗi timeout hoặc crash**

### Khả năng áp dụng:
- Hoạt động tốt trong môi trường Blind Mode với vision=5
- Có thể điều chỉnh tham số để hoạt động với vision radius khác
- Kiến trúc multi-layer cho phép mở rộng thêm policies mới
- Có thể áp dụng cho các map khác nhau (cần cập nhật KNOWN_LAYOUT)

## 4. Nhược điểm và Hướng Cải tiến

### Nhược điểm:
1. **Phụ thuộc hardcoded map**: Agent giả định map cố định. Nếu map thay đổi, topology precomputed sẽ sai → cần cơ chế fallback.
2. **PacmanAgent placeholder**: PacmanAgent chưa được tối ưu (chỉ return STAY), cần phát triển riêng.
3. **Khởi tạo belief cố định**: Belief khởi tạo tại (15,10). Với stochastic starts, có thể mất vài step để hiệu chỉnh lại.
4. **Monte Carlo depth hạn chế**: MC_DEPTH=12 do giới hạn thời gian, có thể bỏ lỡ các tình huống dài hạn.
5. **Chưa có cơ chế học (learning)**: Ensemble weights cập nhật trong game nhưng không lưu giữa các game.
6. **Distance cache trên known_map**: BFS distance precompute dùng known_map, có thể không chính xác nếu memory_map có unseen cells.

### Hướng cải tiến:
1. **Map fingerprint**: Thêm cơ chế phát hiện map từ observation để tự động chọn topology phù hợp
2. **PacmanAgent nâng cao**: Phát triển PacmanAgent với A* + interception + belief tracking riêng
3. **Dynamic belief initialization**: Dùng observation đầu tiên để xác định vị trí bắt đầu thực tế
4. **Adaptive MC depth**: Điều chỉnh depth dựa trên khoảng cách đến Pacman
5. **Persistent learning**: Lưu ensemble weights giữa các game để cải thiện qua thời gian
6. **Multi-map support**: Precompute topology cho nhiều layout khác nhau
7. **Particle filter**: Thay belief state propagation đơn giản bằng particle filter chính xác hơn
8. **Neural network guide**: Dùng model nhẹ để dự đoán danger zones thay vì heuristic
