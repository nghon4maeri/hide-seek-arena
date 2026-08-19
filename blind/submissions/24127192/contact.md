# Contact — Request for Approval (V3)

## Trạng thái: CHỜ DUYỆT

Kính gửi người review,

Sau khi phân tích toàn bộ:
- **14 pacman agents**: nhóm 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, agent
- **14 ghost agents**: nhóm 6 (Belief Minimax), 10 (BFS Lookahead 8), 11 (Heatmap Evasion), 12 (200-step Hardcoded Route + Model Prediction)
- **Source code**: src/arena.py, src/environment.py, src/agent_interface.py, src/agent_loader.py
- **ML Docs**: blind/docs/RL_Exercises.pdf
- **Logic hiện có**: 24127192/agent.py (V2, 1525 lines)

### Phân tích đối thủ chính:
| Team | Pacman Strategy | Ghost Strategy |
|------|----------------|----------------|
| **6** | Bayes Belief + AlphaBeta depth 2-12 + Move Ordering | Belief Minimax depth 3 + StableCamp |
| **10** | Ghost prediction 4-step + Dijkstra turns + Visit decay | BFS Lookahead 8 + Dynamic Waypoints |
| **11** | Minimax AlphaBeta depth 6 + BFS distance cache | Heatmap exploration + BFS evasion |
| **12** | 4-model Ghost predictor + Belief search + Camp clearing | **200-step Hardcoded Route** + 4-model Pacman predictor |

### Đề xuất thay đổi V3 (đã chi tiết trong implement_plan.md):

**A. Constants Tuning**: RECENT_CELL_BAN 15→20, REVERSAL_PENALTY 1200→1500, MOMENTUM_BONUS 80→100, CORRIDOR_REV_PEN 2000→2500, MC_ROLLOUTS 12→16, MC_DEPTH 15→20

**B. Deep Bottom Navigation (Phase 2)**: Sau RIGHT+DOWN escape, tiếp tục đi sâu vào bottom loop qua (14,19)→(18,19)

**C. Upper Escape Route (Phase 4)**: Khi Pacman centroid y > 14, escape lên upper qua col 15

**D. Enhanced Camp**: Hardcode 6 safe camps, camp khi Pacman xa (>25 Manhattan)

**E. Feint Scheduler**: Mỗi 27 steps → 1 feint move, cycle 3 styles (patrol/camp/wander)

**F. Benchmark Script**: 24127192 ghost vs 14 pacman agents, 1 game/opponent, deterministic mode

### Scope thay đổi: CHỈ trong `submissions/24127192/`

### Files sẽ sửa/tạo:
- `agent.py` (MODIFY - thêm Phase 2/4/5, cập nhật constants)
- `benchmark.py` (MODIFY - cập nhật 1 game/opponent)
- `implement_plan.md` (đã cập nhật - chi tiết V3)
- `NOTES.md` (UPDATE sau khi test)

**Để phê duyệt**: Vui lòng trả lời "APPROVED" hoặc comment cụ thể.
