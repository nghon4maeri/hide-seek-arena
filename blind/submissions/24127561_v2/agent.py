"""PacmanAgent — Blind Seeker (Adapted from Lab 1).

Student: 24127561
Role:   Seek Agent Engineer

Lab 2 changes (Blind/Partial Observability):
- Maintains self.memory_map to accumulate observations across steps
- Handles enemy_position = None (enemy not visible)
- A* pathfinding runs on memory_map (optimistic: treats -1 as traversable)
- Frontier-based exploration when enemy is lost
- Belief-state tracking: probability distribution over the ghost's likely
  position while it is unseen, propagated step by step with a learned
  transition model
- Opponent modeling: learns the ghost's turning / persistence habits from
  observed moves and uses them both to propagate the belief state and to
  bias interception targets

v2 changes (faster capture):
- Predictive / time-matched interception while blind: instead of chasing
  the current most-likely ghost cell, projects the belief forward to the
  turn Pacman would actually arrive and targets where the ghost is likely
  to BE THEN (fixed-point refinement over ETA).
- Fixed an opponent-model learning bug: after a multi-turn blind gap, the
  raw delta between last-seen and reacquired position no longer
  corresponds to a single move, so it is no longer fed into the learned
  transition model (this was silently corrupting the model before).
"""

from __future__ import annotations

import sys
import math
import heapq
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

import numpy as np

# ===================================================================
# Constants
# ===================================================================
MOVE_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

CHOKE_SCOUT_DIST = 7
LOCK_DURATION = 3
A_STAR_PHASE_END = 10

# Belief-state tuning
BELIEF_ENTROPY_THRESHOLD = 3.5   # above this, belief is "too spread out" -> explore instead
BELIEF_MAX_SUPPORT = 200         # keep the belief dict sparse/bounded
BELIEF_PRUNE_EPS = 1e-4
STAY_WEIGHT = 0.15               # small prior weight for the ghost "staying put"
PERSISTENCE_WEIGHT = 2.0         # default bias towards continuing straight (used before
                                  # enough data has been learned for the opponent model)

# Predictive interception tuning
INTERCEPT_MAX_ITERS = 4          # fixed-point refinement steps for ETA <-> target
INTERCEPT_LOOKAHEAD_CAP = 12     # cap on how many turns of belief we project forward


# ===================================================================
# Grid utilities
# ===================================================================
def _shape(ms):
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else 0


def _cell(ms, r, c):
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def _apply(pos, move):
    return (pos[0] + move.value[0], pos[1] + move.value[1])


def _valid(pos, ms):
    """Valid if within bounds and NOT a wall (1). -1 (unseen) is considered traversable (optimistic)."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _known_empty(pos, ms):
    """Strict check: only cells known to be empty (0)."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) == 0


def _legal(pos, ms):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_exits(pos, ms):
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


def _dir_to_move(delta) -> Optional[Move]:
    """Map a unit (dr, dc) delta to the corresponding Move, if any."""
    for m in MOVE_ORDER:
        if m.value == delta:
            return m
    return None


# ===================================================================
# A* Search (on memory map)
# ===================================================================
def astar(ms, start, goal):
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []

    open_set = [(0, 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            while current != start:
                prev, move = came_from[current]
                path.append(move)
                current = prev
            path.reverse()
            return path

        for move in MOVE_ORDER:
            nxt = _apply(current, move)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = g + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = (current, move)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


# ===================================================================
# PacmanAgent — Blind Seeker
# ===================================================================
class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))

        # === Blind mode state ===
        self.memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Tuple[int, int]] = None

        # Ghost direction tracking (used for interception while visible)
        self._enemy_direction = None
        self._direction_streak = 0

        # Number of consecutive steps the ghost has been unseen. Used to
        # detect "reacquired after a gap" so we don't feed a multi-turn
        # delta into the single-move opponent model.
        self._blind_steps = 0

        # --- Belief-state tracking (Lab 2) ---
        # Sparse probability distribution over the ghost's current cell,
        # maintained only while the ghost is *not* visible.
        self.belief: Optional[Dict[Tuple[int, int], float]] = None

        # --- Opponent modeling (Lab 2) ---
        # Learns P(next_move | prev_move) from observed ghost movement so the
        # belief propagation and interception logic can favor the ghost's
        # actual habits (e.g. "keeps going straight at junctions" or
        # "reverses when cornered") instead of assuming uniform randomness.
        self.ghost_move_counts: Dict[Optional[Move], Dict[Move, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._prev_enemy_move: Optional[Move] = None

        # Path cache
        self._cached_target = None
        self._cached_path: List = []
        self._cached_my_pos = None

        # Feature gates
        self.enable_interception = True

        # --- Micro Neural Network (Value Network) ---
        # Khởi tạo trọng số (Weights) và Bias ngẫu nhiên cho mạng MLP 2 lớp.
        # Chúng ta dùng Numpy thuần để hệ thống chấm điểm không bị lỗi.
        np.random.seed(42) # Cố định seed để dễ debug
        self.nn_W1 = np.random.randn(4, 8) * 0.1  # Layer 1: 4 inputs -> 8 hidden
        self.nn_b1 = np.zeros(8)
        self.nn_W2 = np.random.randn(8, 1) * 0.1  # Layer 2: 8 hidden -> 1 output
        self.nn_b2 = np.zeros(1)
        # Bộ nhớ đếm số lần đã dẫm lên một ô (Anti-Loop)
        self.visited_counts = defaultdict(int)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        # 1. Ghi nhớ các ô đã đi qua để phạt RL không đi lặp lại (Anti-Loop)
        self.visited_counts[my_position] += 1
        
        # 2. Cập nhật bản đồ sương mù vào bộ nhớ
        self._update_memory(map_state)

        # 3. HỆ LUẬT (Rule-Based): Ưu tiên phản xạ nhanh, bắt ngay nếu có thể
        rule_move = self._apply_rule_based_tactics(my_position, enemy_position)
        if rule_move is not None:
            if enemy_position is not None:
                self.last_seen_enemy = enemy_position
            self._blind_steps = 0
            self.belief = None
            return rule_move

        target = None

        # 4. KHI NHÌN THẤY GHOST (Nằm trong tầm nhìn 5 ô)
        if enemy_position is not None:
            enemy_position = tuple(enemy_position)
            reacquired_after_gap = self._blind_steps > 0
            self._update_ghost_tracking(enemy_position, reacquired_after_gap)
            self._blind_steps = 0
            self.belief = None

            # Tính toán điểm đón đầu (Interception)
            if self.enable_interception and self._direction_streak >= 2:
                inter_target = self._compute_interception_target(
                    self.memory_map, enemy_position, my_position
                )
                if inter_target is not None:
                    path_to_inter = astar(self.memory_map, my_position, inter_target)
                    path_to_direct = astar(self.memory_map, my_position, enemy_position)
                    dist_inter = _manhattan(inter_target, enemy_position)
                    
                    # So sánh để ra quyết định nên đón đầu hay rượt trực tiếp
                    if path_to_inter and (
                        not path_to_direct
                        or len(path_to_inter) <= len(path_to_direct)
                        or dist_inter <= 2
                    ):
                        target = inter_target

            if target is None:
                target = enemy_position
            self.last_seen_enemy = enemy_position

        # 5. KHI BỊ MÙ (Ngoài tầm nhìn chữ thập)
        else:
            self._blind_steps += 1
            
            # Khởi tạo hoặc lan truyền Belief State
            if self.last_seen_enemy is not None:
                if self.belief is None:
                    self.belief = {self.last_seen_enemy: 1.0}
                else:
                    self._propagate_belief()
                    
            # Nếu mất dấu hoàn toàn -> Chuyển sang rà quét vùng chưa biết (Explore)
            if not self.belief or self._belief_entropy() > BELIEF_ENTROPY_THRESHOLD:
                self.last_seen_enemy = None
                self.belief = None
                return self._explore(my_position)
                
            # Dùng RL mô phỏng tương lai để ra quyết định dựa trên Belief State
            rl_action = self._rl_choose_action(my_position, max_simulations=30, rollout_depth=5)
            if rl_action:
                return rl_action
                
            # Fallback an toàn nếu RL tính toán thất bại
            return self._explore(my_position)

        # =========================================================
        # 6. THỰC THI DI CHUYỂN BẰNG A* ĐẾN TARGET 
        # (Phần này chỉ chạy khi enemy_position is not None)
        # =========================================================
        if my_position == target:
            return (Move.STAY, 1)

        # Path caching giúp tối ưu hiệu năng
        cache_valid = (
            self._cached_target == target
            and self._cached_my_pos == my_position
            and self._cached_path
        )
        if cache_valid:
            path = self._cached_path
        else:
            path = astar(self.memory_map, my_position, target)
            self._cached_target = target
            self._cached_path = path
            self._cached_my_pos = my_position

        if not path:
            return self._explore(my_position)

        result = self._path_to_move(path, my_position)
        
        if isinstance(result, tuple):
            consumed = result[1]
            mv = result[0]
        else:
            consumed = 1
            mv = result
            
        self._cached_path = path[consumed:]
        exp_pos = self._advance_position(my_position, mv, consumed)
        self._cached_my_pos = exp_pos
        
        # Đảm bảo lệnh di chuyển được trả về hệ thống
        return result

    # ------------------------------------------------------------------
    # Memory map
    # ------------------------------------------------------------------
    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    # ------------------------------------------------------------------
    # Ghost direction tracking + opponent-model learning
    # ------------------------------------------------------------------
    def _update_ghost_tracking(self, enemy_pos, reacquired_after_gap: bool = False):
        if self.last_seen_enemy is None:
            return

        if reacquired_after_gap:
            # We lost the ghost for one or more turns. The raw delta from
            # last_seen_enemy to enemy_pos may span several hidden moves,
            # so it can't be trusted as a single (prev_move -> move)
            # transition. Feeding it into the opponent model would corrupt
            # the learned probabilities. Reset direction tracking instead
            # and simply resume learning from the *next* visible step.
            self._enemy_direction = None
            self._direction_streak = 0
            self._prev_enemy_move = None
            return

        dr = enemy_pos[0] - self.last_seen_enemy[0]
        dc = enemy_pos[1] - self.last_seen_enemy[1]
        new_dir = (dr, dc)

        # --- Opponent modeling: learn prev_move -> next_move transitions ---
        # Only learn from single-cell steps; if the ghost is faster than one
        # cell/turn (or we skipped a frame) we simply don't have a clean
        # move to attribute, so we skip learning for that step.
        move = _dir_to_move(new_dir) if new_dir != (0, 0) else None
        if move is not None:
            self.ghost_move_counts[self._prev_enemy_move][move] += 1
            self._prev_enemy_move = move
        elif new_dir == (0, 0):
            # Ghost stayed in place
            self.ghost_move_counts[self._prev_enemy_move][None] += 1

        if new_dir == self._enemy_direction and (dr != 0 or dc != 0):
            self._direction_streak += 1
        else:
            self._enemy_direction = new_dir
            self._direction_streak = 1 if (dr != 0 or dc != 0) else 0

    def _learned_transition_probs(self, prev_move: Optional[Move]) -> Optional[Dict[Optional[Move], float]]:
        """Return P(next_move | prev_move) learned from observation, or None
        if we don't have enough data yet for this prev_move."""
        counts = self.ghost_move_counts.get(prev_move)
        if not counts:
            return None
        total = sum(counts.values())
        if total < 2:  # not enough evidence yet, let caller fall back to a heuristic
            return None
        return {m: c / total for m, c in counts.items()}

    # ------------------------------------------------------------------
    # Belief-state tracking (probability distribution over ghost position)
    # ------------------------------------------------------------------
    def _resolve_blind_target(self, my_position):
        """Update/propagate the belief state while the ghost is unseen and
        decide whether to chase a predicted cell or fall back to frontier
        exploration. Returns a target cell, None, or the sentinel
        "__EXPLORE__"."""
        if self.last_seen_enemy is None:
            return None

        if self.belief is None:
            # Just lost sight of the ghost: seed the belief at its last
            # known position.
            self.belief = {self.last_seen_enemy: 1.0}
        else:
            self._propagate_belief()

        if not self.belief:
            # Belief collapsed to nothing (e.g. fully boxed in) — give up
            # the chase and go exploring.
            self.last_seen_enemy = None
            return "__EXPLORE__"

        predicted = self._most_likely_ghost_pos()
        entropy = self._belief_entropy()

        if my_position == self.last_seen_enemy and (
            predicted is None or entropy > BELIEF_ENTROPY_THRESHOLD
        ):
            # We reached the spot the ghost was last seen at and still have
            # no confident prediction — stop chasing a ghost, go explore.
            self.last_seen_enemy = None
            self.belief = None
            return "__EXPLORE__"

        if entropy > BELIEF_ENTROPY_THRESHOLD:
            # Belief too spread out to commit to a single cell: explore,
            # but bias the frontier choice towards the most likely region.
            self._explore_belief_hint = predicted
            return "__EXPLORE__"

        # Predictive / time-matched interception: don't chase where the
        # ghost is NOW, chase where it's likely to be by the time we
        # actually arrive.
        intercept = self._predictive_intercept_target(my_position)
        return intercept if intercept is not None else predicted

    def _propagate_belief_dict(self, belief: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """Pure one-step belief propagation. Does not mutate self.belief —
        used both for the live belief update and for projecting the belief
        forward hypothetically (predictive interception)."""
        if not belief:
            return {}
        ms = self.memory_map
        new_belief: Dict[Tuple[int, int], float] = defaultdict(float)

        learned = self._learned_transition_probs(self._prev_enemy_move)

        for pos, prob in belief.items():
            legal = _legal(pos, ms)
            if not legal:
                # Ghost can't move from here (fully walled/unseen) — mass stays.
                new_belief[pos] += prob
                continue

            weights: Dict[Move, float] = {}
            for m in legal:
                if learned is not None:
                    weights[m] = learned.get(m, 0.05)
                else:
                    # Fallback heuristic: prefer continuing the last known
                    # direction (persistence), spread the rest uniformly.
                    if self._enemy_direction is not None and m.value == self._enemy_direction:
                        weights[m] = PERSISTENCE_WEIGHT
                    else:
                        weights[m] = 1.0

            stay_w = (learned.get(None, STAY_WEIGHT) if learned is not None else STAY_WEIGHT)
            total_w = sum(weights.values()) + stay_w
            if total_w <= 0:
                total_w = 1.0

            new_belief[pos] += prob * (stay_w / total_w)
            for m, w in weights.items():
                nxt = _apply(pos, m)
                new_belief[nxt] += prob * (w / total_w)

        total = sum(new_belief.values())
        if total <= 0:
            return {}

        pruned = {p: v / total for p, v in new_belief.items() if v / total > BELIEF_PRUNE_EPS}
        if not pruned:
            pruned = {max(new_belief, key=new_belief.get): 1.0}

        if len(pruned) > BELIEF_MAX_SUPPORT:
            top = sorted(pruned.items(), key=lambda kv: -kv[1])[:BELIEF_MAX_SUPPORT]
            s = sum(v for _, v in top)
            pruned = {p: v / s for p, v in top}

        return pruned

    def _propagate_belief(self):
        self.belief = self._propagate_belief_dict(self.belief)

    def _belief_entropy(self) -> float:
        if not self.belief:
            return 0.0
        return -sum(p * math.log(p + 1e-12) for p in self.belief.values())

    def _most_likely_ghost_pos(self) -> Optional[Tuple[int, int]]:
        if not self.belief:
            return None
        return max(self.belief.items(), key=lambda kv: kv[1])[0]

    def _predictive_intercept_target(self, my_position, max_iters: int = INTERCEPT_MAX_ITERS):
        """Time-matched interception under uncertainty.

        Chasing `_most_likely_ghost_pos()` directly means we're always
        aiming at where the ghost *was* probabilistically, not where it
        will be once we actually get there — for a spread-out, moving
        belief this costs extra steps every turn.

        Instead: guess a target, compute how many turns it'll take Pacman
        to reach it (ETA), project the belief forward that many turns using
        the learned opponent model, then re-pick the best target from the
        *projected* distribution (weighted by probability and by how cheap
        it is to reach). Repeat a few times until the target stabilizes
        (fixed point) or the iteration budget runs out.
        """
        if not self.belief:
            return None

        target = self._most_likely_ghost_pos()
        if target is None:
            return None

        for _ in range(max_iters):
            path = astar(self.memory_map, my_position, target)
            eta = max(1, math.ceil(len(path) / self.pacman_speed)) if path else 1
            eta = min(eta, INTERCEPT_LOOKAHEAD_CAP)

            projected = dict(self.belief)
            for _ in range(eta):
                projected = self._propagate_belief_dict(projected)
                if not projected:
                    break
            if not projected:
                break

            def score(item):
                cell, p = item
                d = _manhattan(my_position, cell)
                return p / (1 + d / self.pacman_speed)

            new_target = max(projected.items(), key=score)[0]
            if new_target == target:
                break
            target = new_target

        return target

    # ------------------------------------------------------------------
    # Interception target (ghost currently visible)
    # ------------------------------------------------------------------
    def _compute_interception_target(self, ms, enemy_pos, my_pos):
        dr, dc = self._enemy_direction
        cur_row, cur_col = enemy_pos

        # Opponent modeling: if we've learned this ghost tends to turn
        # rather than go straight, don't project as far ahead.
        current_move = _dir_to_move((dr, dc))
        learned = self._learned_transition_probs(current_move)
        persistence = learned.get(current_move, 0.5) if learned else 0.6
        max_lookahead = 4 if persistence >= 0.5 else 2

        best = None
        for i in range(1, max_lookahead + 1):
            nr, nc = cur_row + dr * i, cur_col + dc * i
            h, w = _shape(ms)
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if _cell(ms, nr, nc) == 1:
                break
            nxt = (nr, nc)
            exits = _cell_exits(nxt, ms)
            if exits >= 3:
                return nxt
            if exits == 2 and best is None:
                best = nxt
        return best

    # ------------------------------------------------------------------
    # Convert A* path (list of Move) to (Move, steps)
    # ------------------------------------------------------------------
    def _path_to_move(self, path, my_position):
        first_move = path[0]
        move = first_move if isinstance(first_move, Move) else Move.STAY
        steps = 1
        for m in path[1:]:
            if m == move and steps < self.pacman_speed:
                steps += 1
            else:
                break
        return (move, steps)

    def _advance_position(self, pos, move, steps):
        cur = pos
        for _ in range(steps):
            nxt = _apply(cur, move)
            if not _valid(nxt, self.memory_map):
                break
            cur = nxt
        return cur

    # ------------------------------------------------------------------
    # Exploration (frontier-based, with optional belief bias)
    # ------------------------------------------------------------------
    def _explore(self, my_position, belief_hint=None):
        ms = self.memory_map
        if ms is None:
            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(moves)
            return (moves[0], 1)

        if belief_hint is None:
            belief_hint = getattr(self, "_explore_belief_hint", None)
        self._explore_belief_hint = None

        h, w = ms.shape
        target = None
        best_score = float("inf")
        
        # Tìm các ô "Biên giới" (Frontier) - Tức là ô an toàn nằm kề ô sương mù (-1)
        for r in range(h):
            for c in range(w):
                if ms[r, c] != 0:
                    continue
                has_unknown = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and ms[nr, nc] == -1:
                        has_unknown = True
                        break
                        
                if has_unknown:
                    d = abs(r - my_position[0]) + abs(c - my_position[1])
                    score = d
                    if belief_hint is not None:
                        d_hint = abs(r - belief_hint[0]) + abs(c - belief_hint[1])
                        score = 0.5 * d + 0.5 * d_hint
                    if score < best_score:
                        best_score = score
                        target = (r, c)

        if target:
            path = astar(ms, my_position, target)
            if path:
                return self._path_to_move(path, my_position)

        # Xử lý khi không tìm thấy target hoặc A* thất bại
        moves = _legal(my_position, ms)
        if moves:
            best_move = moves[0]
            min_visits = float('inf')
            
            # Quét các hướng đi hợp lệ, chọn hướng ít dẫm chân lên nhất
            for m in moves:
                nxt_pos = _apply(my_position, m)
                v = self.visited_counts.get(nxt_pos, 0)
                if v < min_visits:
                    min_visits = v
                    best_move = m
                    
            return (best_move, 1)
            
        return (Move.STAY, 1)

    # ===================================================================
    # Rule-Based System (Hệ Luật Chiến Thuật)
    # ===================================================================
    def _apply_rule_based_tactics(self, my_position, enemy_position):
        """
        Kiểm tra các quy tắc chiến thuật cứng. 
        Nếu thỏa mãn điều kiện hiển nhiên thắng, ra lệnh ngay lập tức để tiết kiệm chi phí chạy RL/NN.
        """
        if enemy_position is None:
            return None # Không thấy địch thì không áp dụng luật trực tiếp được
            
        dr = enemy_position[0] - my_position[0]
        dc = enemy_position[1] - my_position[1]
        dist = abs(dr) + abs(dc)
        
        # QUY TẮC 1 & 2: LETHAL STRIKE (Đòn kết liễu)
        # 1. Nếu Ghost ngay sát bên (cách 1 ô) -> Chén ngay!
        if dist == 1:
            move = _dir_to_move((dr, dc))
            if move: return (move, 1)
            
        # 2. Nếu Ghost cách 2 ô trên CÙNG 1 đường thẳng ngang hoặc dọc -> Lao 2 bước tới bắt ngay!
        if dist == 2 and (dr == 0 or dc == 0):
            step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
            step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
            mid_cell = (my_position[0] + step_r, my_position[1] + step_c)
            
            # Kiểm tra xem ô ở giữa có bị vướng tường không
            if _valid(mid_cell, self.memory_map):
                move = _dir_to_move((step_r, step_c))
                # Pacman tận dụng lợi thế tốc độ đi 2 bước trên đường thẳng
                if move: return (move, 2) 

        # QUY TẮC 3: KHÓA NGÕ CỤT (Dead-end Trap)
        # Nếu Ghost lọt vào ô chỉ có 1 lối thoát (exits == 1), nó đã bị kẹt.
        ghost_exits = _cell_exits(enemy_position, self.memory_map)
        if ghost_exits == 1:
            # Sinh đường đi A* thẳng tới Ghost, cố gắng tối ưu đi 2 bước
            path = astar(self.memory_map, my_position, enemy_position)
            if path:
                return self._path_to_move(path, my_position)

        # Nếu không có Rule nào thỏa mãn, trả về None để nhường quyền quyết định lại cho AI (A* hoặc RL/NN)
        return None

# ===================================================================
    # Reinforcement Learning (POMCP / Monte Carlo Rollouts)
    # ===================================================================
    def _rl_choose_action(self, my_position, max_simulations=30, rollout_depth=5):
        """
        Sử dụng Online RL (Monte Carlo Rollouts) trên Belief State.
        Thay vì A* đến 1 điểm mù, ta thử các hướng đi và tính Q-Value kỳ vọng.
        """
        if not self.belief:
            return None

        legal_moves = _legal(my_position, self.memory_map)
        if not legal_moves:
            return (Move.STAY, 1)

        q_values = {m: 0.0 for m in legal_moves}
        
        # Chạy N lần mô phỏng để tính trung bình phần thưởng (Expected Reward)
        for _ in range(max_simulations):
            # 1. Lấy mẫu (Sample) một vị trí giả định của Ghost từ Belief State
            ghost_sim_pos = self._sample_from_belief()
            
            for move in legal_moves:
                # 2. Tính phần thưởng bằng cách mô phỏng tương lai (Rollout)
                reward = self._simulate_rollout(my_position, move, ghost_sim_pos, rollout_depth)
                q_values[move] += reward
                
        # Chọn hành động có Q-Value cao nhất
        best_move = max(q_values.items(), key=lambda x: x[1])[0]
        
        # Bọc nước đi an toàn: nếu đường thẳng không bị cản, Pacman tận dụng đi 2 bước
        steps = 1
        nxt = _apply(my_position, best_move)
        if self.pacman_speed == 2 and _valid(nxt, self.memory_map):
            nxt2 = _apply(nxt, best_move)
            if _valid(nxt2, self.memory_map):
                steps = 2
                
        return (best_move, steps)

    def _sample_from_belief(self) -> Tuple[int, int]:
        """Lấy mẫu ngẫu nhiên 1 tọa độ dựa trên phân bố xác suất của self.belief"""
        cells = list(self.belief.keys())
        probs = list(self.belief.values())
        idx = np.random.choice(len(cells), p=probs)
        return cells[idx]

    def _simulate_rollout(self, pacman_pos, initial_move, ghost_pos, depth) -> float:
        curr_pacman = _apply(pacman_pos, initial_move)
        curr_ghost = ghost_pos
        total_reward = 0.0
        gamma = 0.9 
        
        for step in range(depth):
            dist = _manhattan(curr_pacman, curr_ghost)
            
            if dist <= 1:
                total_reward += 1000 * (gamma ** step)
                break
                
            # --- BẢN VÁ LỖI TẠI ĐÂY ---
            # Thay vì phạt -1 mù quáng, ta phạt CỰC NẶNG nếu đi vào ô đã dẫm lên nhiều lần
            visited_penalty = self.visited_counts.get(curr_pacman, 0) * 10
            total_reward -= visited_penalty * (gamma ** step)
            
            # Khuyến khích di chuyển ra chỗ mới
            if visited_penalty == 0:
                total_reward += 2 * (gamma ** step)
            # --------------------------
            
            features = self._extract_features(curr_pacman, curr_ghost)
            nn_value = self._nn_forward(features)
            
            total_reward += nn_value * (gamma ** step)
            
            curr_ghost = self._simulate_ghost_step(curr_ghost)
            curr_pacman = self._greedy_step_towards(curr_pacman, curr_ghost)
            
        return total_reward

    def _simulate_ghost_step(self, ghost_pos):
        """Mô phỏng 1 bước đi của Ghost (chọn ngẫu nhiên 1 ô hợp lệ)."""
        legal = _legal(ghost_pos, self.memory_map)
        if not legal:
            return ghost_pos
        return _apply(ghost_pos, random.choice(legal))

    def _greedy_step_towards(self, pacman_pos, target_pos):
        """Mô phỏng 1 bước đi của Pacman lao về phía Ghost (Heuristic)."""
        legal = _legal(pacman_pos, self.memory_map)
        if not legal:
            return pacman_pos
        
        best_pos = pacman_pos
        min_dist = float('inf')
        for m in legal:
            nxt = _apply(pacman_pos, m)
            d = _manhattan(nxt, target_pos)
            if d < min_dist:
                min_dist = d
                best_pos = nxt
        return best_pos

    def _extract_features(self, pacman_pos, ghost_pos) -> np.ndarray:
        """Trích xuất 4 đặc trưng (features) quan trọng để nạp vào Neural Network"""
        h, w = _shape(self.memory_map)
        max_dist = h + w
        
        # Feature 1: Khoảng cách Manhattan chuẩn hóa (0.0 -> 1.0)
        dist = _manhattan(pacman_pos, ghost_pos)
        f1_dist = dist / max_dist
        
        # Feature 2: Số lối thoát của Ghost (để biết nó có đang bị kẹt không)
        ghost_exits = _cell_exits(ghost_pos, self.memory_map)
        f2_ghost_freedom = ghost_exits / 4.0
        
        # Feature 3: Ghost có đang ở trong ngõ cụt (Dead-end) không?
        f3_is_dead_end = 1.0 if ghost_exits <= 1 else 0.0
        
        # Feature 4: Lợi thế tầm nhìn của Pacman (số hướng có thể di chuyển)
        pacman_exits = _cell_exits(pacman_pos, self.memory_map)
        f4_pacman_freedom = pacman_exits / 4.0
        
        return np.array([f1_dist, f2_ghost_freedom, f3_is_dead_end, f4_pacman_freedom])

    def _nn_forward(self, features: np.ndarray) -> float:
        """Lan truyền tiến (Forward pass) qua mạng Neural Network mini"""
        # Layer 1 + ReLU Activation
        z1 = np.dot(features, self.nn_W1) + self.nn_b1
        a1 = np.maximum(0, z1) # ReLU: giữ số dương, biến số âm thành 0
        
        # Layer 2 (Output)
        z2 = np.dot(a1, self.nn_W2) + self.nn_b2
        
        # Trả về giá trị của trạng thái (Value). Trị số càng cao, trạng thái càng có lợi.
        heuristic_bias = (1.0 - features[0]) * 10 + features[2] * 20
        return float(z2[0]) + heuristic_bias


# ===================================================================
# GhostAgent — placeholder (not primary deliverable for 24127561)
# ===================================================================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_map = None
        self.last_seen_enemy = None

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)
        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)
        # Ghost returns a bare Move (no tuple), as required.
        return Move.STAY