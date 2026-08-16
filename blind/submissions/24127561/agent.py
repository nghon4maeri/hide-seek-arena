"""PacmanAgent — PPO Blind Seeker (Lab 2 / POMDP).

Student: 24127561
Role:    Seek Agent Engineer

Overview
========
Primary:  CNN + GRU Actor-Critic policy (PPO-trained).
          Loads weights from weights.pth next to this file.
Fallback: Full A*-based blind seeker (same logic as pre-RL agent) used
          whenever PyTorch is unavailable, weights are missing, or
          inference would exceed the per-step time budget.

Observation built per step
--------------------------
Channel 0 — normalised map   : wall=1.0, empty=0.0, fog/unseen=0.5
Channel 1 — self position    : 1.0 at Pacman's cell
Channel 2 — enemy belief map : Gaussian-spread heatmap centred on last
            known / most-likely ghost location; decays over unseen turns

Interface conformance (blind/src/agent_interface.py)
----------------------------------------------------
  step(map_state, my_position, enemy_position, step_number)
      → Move  |  (Move, steps)    1 ≤ steps ≤ pacman_speed
"""

from __future__ import annotations

import heapq
import math
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Path setup — keep working from both blind/src/ and repo root
# ---------------------------------------------------------------------------
_SUBMISSION_DIR = Path(__file__).resolve().parent
SRC_PATH = _SUBMISSION_DIR.parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

# ---------------------------------------------------------------------------
# Optional PyTorch (graceful fallback if missing)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOVE_ORDER     = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
ACTION_TO_IDX  = {Move.UP: 0, Move.DOWN: 1, Move.LEFT: 2, Move.RIGHT: 3, Move.STAY: 4}
IDX_TO_MOVE    = {0: Move.UP, 1: Move.DOWN, 2: Move.LEFT, 3: Move.RIGHT, 4: Move.STAY}

WEIGHTS_PATH   = _SUBMISSION_DIR / "weights.pth"

# Inference timing guard (leave headroom below 0.85 s arena limit)
INFERENCE_BUDGET_S  = 0.65
TOTAL_STEP_BUDGET_S = 0.80

# Belief-state tuning (A* fallback path)
BELIEF_ENTROPY_THRESHOLD = 3.5
BELIEF_MAX_SUPPORT       = 200
BELIEF_PRUNE_EPS         = 1e-4
STAY_WEIGHT              = 0.15
PERSISTENCE_WEIGHT       = 2.0
INTERCEPT_MAX_ITERS      = 4
INTERCEPT_LOOKAHEAD_CAP  = 12

# Enemy belief heatmap parameters (RL path)
BELIEF_SIGMA_INIT  = 0.5     # initial spread when ghost first hidden (cells)
BELIEF_SIGMA_GROW  = 0.3     # additional sigma per unseen turn
BELIEF_SIGMA_MAX   = 5.0


# ===========================================================================
# Grid utilities
# ===========================================================================

def _shape(ms) -> Tuple[int, int]:
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), (len(ms[0]) if ms else 0)


def _cell(ms, r: int, c: int) -> int:
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def _apply(pos: Tuple[int, int], move: Move) -> Tuple[int, int]:
    return (pos[0] + move.value[0], pos[1] + move.value[1])


def _valid(pos: Tuple[int, int], ms) -> bool:
    """Valid = in bounds and NOT a wall. -1 (unseen) is treated as passable."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _legal(pos: Tuple[int, int], ms) -> List[Move]:
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _dir_to_move(delta: Tuple[int, int]) -> Optional[Move]:
    for m in MOVE_ORDER:
        if m.value == delta:
            return m
    return None


def _cell_exits(pos: Tuple[int, int], ms) -> int:
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


# ===========================================================================
# A* search (memory-map aware: treats -1 as passable)
# ===========================================================================

def _astar(ms, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Move]:
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []
    open_set: List[Tuple] = [(0, 0, start)]
    came_from: Dict = {}
    g_score: Dict = {start: 0}
    closed: set = set()
    while open_set:
        f, g, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            path: List[Move] = []
            while cur != start:
                prev, mv = came_from[cur]
                path.append(mv)
                cur = prev
            path.reverse()
            return path
        for mv in MOVE_ORDER:
            nxt = _apply(cur, mv)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = g + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = (cur, mv)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


# ===========================================================================
# Observation builder for the RL model
# ===========================================================================

def build_observation(
    memory_map: np.ndarray,
    my_pos: Tuple[int, int],
    enemy_pos: Optional[Tuple[int, int]],
    steps_unseen: int,
    last_seen_enemy: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (3, H, W) image tensor + (5,) position vector for the RL model.

    Channel 0: normalised map  — wall=1.0, empty=0.0, fog/unseen=0.5
    Channel 1: self mask       — 1.0 at Pacman's cell, else 0.0
    Channel 2: enemy belief    — heatmap of ghost location probability
    """
    H, W = memory_map.shape

    # --- Channel 0: normalised map ---
    ch0 = np.where(memory_map == 1, 1.0,
           np.where(memory_map == -1, 0.5, 0.0)).astype(np.float32)

    # --- Channel 1: self mask ---
    ch1 = np.zeros((H, W), dtype=np.float32)
    r, c = my_pos
    if 0 <= r < H and 0 <= c < W:
        ch1[r, c] = 1.0

    # --- Channel 2: enemy belief / heatmap ---
    ch2 = np.zeros((H, W), dtype=np.float32)
    centre = enemy_pos if enemy_pos is not None else last_seen_enemy
    if centre is not None:
        # Gaussian spread — sigma grows the longer the ghost is unseen
        sigma = BELIEF_SIGMA_INIT + BELIEF_SIGMA_GROW * min(steps_unseen, 20)
        sigma = min(sigma, BELIEF_SIGMA_MAX)
        cr, cc = centre
        for row in range(H):
            for col in range(W):
                if memory_map[row, col] == 1:
                    continue  # wall — ghost can't be here
                d2 = (row - cr) ** 2 + (col - cc) ** 2
                ch2[row, col] = math.exp(-d2 / (2 * sigma ** 2))
        total = ch2.sum()
        if total > 0:
            ch2 /= total

    img = np.stack([ch0, ch1, ch2], axis=0)  # (3, H, W)

    # --- Position vector ---
    if enemy_pos is not None:
        er, ec = float(enemy_pos[0]) / H, float(enemy_pos[1]) / W
    elif last_seen_enemy is not None:
        er, ec = float(last_seen_enemy[0]) / H, float(last_seen_enemy[1]) / W
    else:
        er, ec = -1.0, -1.0

    pos = np.array([
        float(my_pos[0]) / H,
        float(my_pos[1]) / W,
        er,
        ec,
        float(min(steps_unseen, 200)) / 200.0,
    ], dtype=np.float32)

    return img, pos


# ===========================================================================
# PacmanAgent — PPO with A* fallback
# ===========================================================================

class PacmanAgent(BasePacmanAgent):
    """Blind Pacman Seeker powered by PPO (CNN + GRU) with A* fallback.

    On each call to step():
      1. Update accumulated memory map.
      2. Update belief state (ghost direction / learning).
      3. If RL model is available → build observation → run inference →
         apply action mask for legal moves → pick action.
      4. Fallback to A* + belief propagation if RL fails or times out.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed: int = max(1, int(kwargs.get("pacman_speed", 2)))

        # ---------------------------------------------------------------
        # Shared state (used by both RL and A* paths)
        # ---------------------------------------------------------------
        self.memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Tuple[int, int]] = None
        self._steps_unseen: int = 0

        # Ghost direction tracking (for A* interception + RL belief build)
        self._enemy_direction: Optional[Tuple[int, int]] = None
        self._direction_streak: int = 0
        self._blind_steps: int = 0

        # Opponent-model learning (A* fallback)
        self.ghost_move_counts: Dict = defaultdict(lambda: defaultdict(int))
        self._prev_enemy_move: Optional[Move] = None

        # Belief-state (A* fallback)
        self.belief: Optional[Dict[Tuple[int, int], float]] = None
        self._explore_belief_hint: Optional[Tuple[int, int]] = None

        # Path cache (A* fallback)
        self._cached_target: Optional[Tuple[int, int]] = None
        self._cached_path: List[Move] = []
        self._cached_my_pos: Optional[Tuple[int, int]] = None

        # ---------------------------------------------------------------
        # RL model
        # ---------------------------------------------------------------
        self._model = None
        self._gru_hidden = None
        self._rl_available = False
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        if not _TORCH_AVAILABLE:
            return
        if not WEIGHTS_PATH.exists():
            return
        try:
            # Import model lazily to avoid top-level torch dependency errors
            # when torch isn't installed at all
            from model import ActorCriticGRU  # type: ignore[import]
            ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=False)

            # Support two checkpoint formats:
            #  1. Raw state-dict
            #  2. Dict with keys 'model_state', 'map_h', 'map_w'
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                map_h = ckpt.get("map_h", 21)
                map_w = ckpt.get("map_w", 21)
                state = ckpt["model_state"]
            else:
                map_h, map_w = 21, 21
                state = ckpt

            model = ActorCriticGRU(map_h=map_h, map_w=map_w)
            model.load_state_dict(state)
            model.eval()
            self._model = model
            self._rl_available = True
        except Exception:
            # Any loading error → silently fall back to A*
            self._model = None
            self._rl_available = False

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number: int):
        t0 = time.monotonic()

        my_position = tuple(my_position)

        # Update memory and belief state
        self._update_memory(map_state)

        if enemy_position is not None:
            enemy_position = tuple(enemy_position)
            reacquired = self._blind_steps > 0
            self._update_ghost_tracking(enemy_position, reacquired)
            self._blind_steps = 0
            self._steps_unseen = 0
            self.belief = None
        else:
            self._blind_steps += 1
            self._steps_unseen += 1

        # Reset GRU hidden state at the very first step of each episode
        if step_number <= 1:
            self._gru_hidden = None

        # ---------------------------------------------------------------
        # RL path
        # ---------------------------------------------------------------
        if self._rl_available and self._model is not None:
            try:
                action = self._rl_step(
                    my_position, enemy_position, t0
                )
                if action is not None:
                    return action
            except Exception:
                pass  # fall through to A* on any RL error

        # ---------------------------------------------------------------
        # A* fallback
        # ---------------------------------------------------------------
        return self._astar_step(my_position, enemy_position, map_state)

    # ------------------------------------------------------------------
    # RL inference path
    # ------------------------------------------------------------------
    def _rl_step(
        self,
        my_position: Tuple[int, int],
        enemy_position: Optional[Tuple[int, int]],
        t0: float,
    ) -> Optional[object]:
        """Build obs, run GRU, apply action mask, return (Move, steps).
        Returns None if budget exceeded or model fails.
        """
        if time.monotonic() - t0 > INFERENCE_BUDGET_S:
            return None

        H, W = self.memory_map.shape

        img, pos = build_observation(
            self.memory_map,
            my_position,
            enemy_position,
            self._steps_unseen,
            self.last_seen_enemy,
        )

        img_t = torch.from_numpy(img).unsqueeze(0)   # (1, 3, H, W)
        pos_t = torch.from_numpy(pos).unsqueeze(0)   # (1, 5)

        # Build action mask: STAY always allowed; moves only if valid
        mask = torch.zeros(1, 5, dtype=torch.bool)
        legal = _legal(my_position, self.memory_map)
        for mv in legal:
            mask[0, ACTION_TO_IDX[mv]] = True
        mask[0, ACTION_TO_IDX[Move.STAY]] = True

        if not mask.any():
            return None  # shouldn't happen but be safe

        with torch.no_grad():
            action_idx, _, _, self._gru_hidden = self._model.get_action(
                img_t, pos_t, self._gru_hidden,
                deterministic=True,
                action_mask=mask,
            )

        if time.monotonic() - t0 > TOTAL_STEP_BUDGET_S:
            return None

        chosen_move = IDX_TO_MOVE[action_idx]

        # If the model chose STAY but legal moves exist, honour it
        if chosen_move == Move.STAY:
            return (Move.STAY, 1)

        # Pack straight-line multi-step if pacman_speed > 1
        steps = self._count_straight_steps(my_position, chosen_move)
        return (chosen_move, steps)

    def _count_straight_steps(
        self, pos: Tuple[int, int], move: Move
    ) -> int:
        """Count how many consecutive steps in direction `move` are valid
        (up to pacman_speed)."""
        cur = pos
        steps = 0
        for _ in range(self.pacman_speed):
            nxt = _apply(cur, move)
            if not _valid(nxt, self.memory_map):
                break
            steps += 1
            cur = nxt
        return max(1, steps)

    # ------------------------------------------------------------------
    # A* fallback path
    # ------------------------------------------------------------------
    def _astar_step(
        self,
        my_position: Tuple[int, int],
        enemy_position: Optional[Tuple[int, int]],
        map_state,
    ):
        target = None

        if enemy_position is not None:
            # Ghost visible: try interception, else direct chase
            if self._direction_streak >= 2:
                inter = self._compute_interception_target(
                    self.memory_map, enemy_position, my_position
                )
                if inter is not None:
                    path_i = _astar(self.memory_map, my_position, inter)
                    path_d = _astar(self.memory_map, my_position, enemy_position)
                    dist_to_ghost = _manhattan(inter, enemy_position)
                    if path_i and (
                        not path_d
                        or len(path_i) <= len(path_d)
                        or dist_to_ghost <= 2
                    ):
                        target = inter
            if target is None:
                target = enemy_position
            self.last_seen_enemy = enemy_position
        else:
            # Ghost invisible: belief-state prediction
            target = self._resolve_blind_target(my_position, map_state)
            if target is None or target == "__EXPLORE__":
                return self._explore(my_position)

        if my_position == target:
            return (Move.STAY, 1)

        # Path cache
        cache_valid = (
            self._cached_target == target
            and self._cached_my_pos == my_position
            and self._cached_path
        )
        if cache_valid:
            path = self._cached_path
        else:
            path = _astar(self.memory_map, my_position, target)
            self._cached_target = target
            self._cached_path = list(path)
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
        self._cached_my_pos = self._advance_position(my_position, mv, consumed)
        return result

    # ------------------------------------------------------------------
    # Memory map management
    # ------------------------------------------------------------------
    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = map_state != -1
        self.memory_map[visible_mask] = map_state[visible_mask]

    # ------------------------------------------------------------------
    # Ghost direction tracking and opponent model
    # ------------------------------------------------------------------
    def _update_ghost_tracking(
        self, enemy_pos: Tuple[int, int], reacquired_after_gap: bool = False
    ):
        if self.last_seen_enemy is None:
            self.last_seen_enemy = enemy_pos
            return

        if reacquired_after_gap:
            self._enemy_direction = None
            self._direction_streak = 0
            self._prev_enemy_move = None
            self.last_seen_enemy = enemy_pos
            return

        dr = enemy_pos[0] - self.last_seen_enemy[0]
        dc = enemy_pos[1] - self.last_seen_enemy[1]
        new_dir = (dr, dc)

        move = _dir_to_move(new_dir) if new_dir != (0, 0) else None
        if move is not None:
            self.ghost_move_counts[self._prev_enemy_move][move] += 1
            self._prev_enemy_move = move
        elif new_dir == (0, 0):
            self.ghost_move_counts[self._prev_enemy_move][None] += 1

        if new_dir == self._enemy_direction and (dr != 0 or dc != 0):
            self._direction_streak += 1
        else:
            self._enemy_direction = new_dir
            self._direction_streak = 1 if (dr != 0 or dc != 0) else 0

        self.last_seen_enemy = enemy_pos

    def _learned_transition_probs(
        self, prev_move: Optional[Move]
    ) -> Optional[Dict]:
        counts = self.ghost_move_counts.get(prev_move)
        if not counts:
            return None
        total = sum(counts.values())
        if total < 2:
            return None
        return {m: c / total for m, c in counts.items()}

    # ------------------------------------------------------------------
    # Belief state (A* fallback)
    # ------------------------------------------------------------------
    def _resolve_blind_target(self, my_position: Tuple[int, int], map_state):
        if self.belief is None:
            if self.last_seen_enemy is not None:
                self.belief = {self.last_seen_enemy: 1.0}
            else:
                self.belief = {}
                if self.memory_map is not None:
                    h, w = self.memory_map.shape
                    empty_top_cells = []
                    for r in range(h):
                        for c in range(w):
                            if r <= 8 and self.memory_map[r, c] != 1:
                                empty_top_cells.append((r, c))
                    if empty_top_cells:
                        prob = 1.0 / len(empty_top_cells)
                        for cell in empty_top_cells:
                            self.belief[cell] = prob
        else:
            self._propagate_belief(map_state)

        if not self.belief:
            self.last_seen_enemy = None
            return "__EXPLORE__"

        predicted = self._most_likely_ghost_pos()
        entropy = self._belief_entropy()

        if my_position == self.last_seen_enemy and (
            predicted is None or entropy > BELIEF_ENTROPY_THRESHOLD
        ):
            self.last_seen_enemy = None
            self.belief = None
            return "__EXPLORE__"

        if entropy > BELIEF_ENTROPY_THRESHOLD:
            self._explore_belief_hint = predicted
            return "__EXPLORE__"

        intercept = self._predictive_intercept_target(my_position)
        return intercept if intercept is not None else predicted

    def _propagate_belief_dict(
        self, belief: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], float]:
        if not belief:
            return {}
        ms = self.memory_map
        new_belief: Dict = defaultdict(float)
        learned = self._learned_transition_probs(self._prev_enemy_move)

        for pos, prob in belief.items():
            legal = _legal(pos, ms)
            if not legal:
                new_belief[pos] += prob
                continue
            weights: Dict = {}
            for m in legal:
                if learned is not None:
                    weights[m] = learned.get(m, 0.05)
                else:
                    if self._enemy_direction is not None and m.value == self._enemy_direction:
                        weights[m] = PERSISTENCE_WEIGHT
                    else:
                        weights[m] = 1.0
            stay_w = learned.get(None, STAY_WEIGHT) if learned else STAY_WEIGHT
            total_w = sum(weights.values()) + stay_w
            if total_w <= 0:
                total_w = 1.0
            new_belief[pos] += prob * (stay_w / total_w)
            for m, w in weights.items():
                new_belief[_apply(pos, m)] += prob * (w / total_w)

        total = sum(new_belief.values())
        if total <= 0:
            return {}
        pruned = {p: v / total for p, v in new_belief.items()
                  if v / total > BELIEF_PRUNE_EPS}
        if not pruned:
            pruned = {max(new_belief, key=new_belief.get): 1.0}
        if len(pruned) > BELIEF_MAX_SUPPORT:
            top = sorted(pruned.items(), key=lambda kv: -kv[1])[:BELIEF_MAX_SUPPORT]
            s = sum(v for _, v in top)
            pruned = {p: v / s for p, v in top}
        return pruned

    def _propagate_belief(self, map_state):
        self.belief = self._propagate_belief_dict(self.belief)
        if self.belief is not None and map_state is not None:
            h, w = map_state.shape
            pruned = {}
            for pos, prob in self.belief.items():
                r, c = pos
                if 0 <= r < h and 0 <= c < w and map_state[r, c] != -1:
                    continue
                pruned[pos] = prob
            total = sum(pruned.values())
            if total > 0:
                self.belief = {p: v / total for p, v in pruned.items()}
            else:
                self.belief = {}

    def _belief_entropy(self) -> float:
        if not self.belief:
            return 0.0
        return -sum(p * math.log(p + 1e-12) for p in self.belief.values())

    def _most_likely_ghost_pos(self) -> Optional[Tuple[int, int]]:
        if not self.belief:
            return None
        return max(self.belief.items(), key=lambda kv: kv[1])[0]

    def _predictive_intercept_target(
        self, my_position: Tuple[int, int], max_iters: int = INTERCEPT_MAX_ITERS
    ) -> Optional[Tuple[int, int]]:
        if not self.belief:
            return None
        target = self._most_likely_ghost_pos()
        if target is None:
            return None
        for _ in range(max_iters):
            path = _astar(self.memory_map, my_position, target)
            eta = max(1, math.ceil(len(path) / self.pacman_speed)) if path else 1
            eta = min(eta, INTERCEPT_LOOKAHEAD_CAP)
            projected = dict(self.belief)
            for _ in range(eta):
                projected = self._propagate_belief_dict(projected)
                if not projected:
                    break
            if not projected:
                break

            def _score(item):
                cell, p = item
                d = _manhattan(my_position, cell)
                return p / (1 + d / self.pacman_speed)

            new_target = max(projected.items(), key=_score)[0]
            if new_target == target:
                break
            target = new_target
        return target

    # ------------------------------------------------------------------
    # Visible ghost interception (A* path)
    # ------------------------------------------------------------------
    def _compute_interception_target(
        self,
        ms,
        enemy_pos: Tuple[int, int],
        my_pos: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        if self._enemy_direction is None:
            return None
        dr, dc = self._enemy_direction
        cur_r, cur_c = enemy_pos
        current_move = _dir_to_move((dr, dc))
        learned = self._learned_transition_probs(current_move)
        persistence = (
            learned.get(current_move, 0.5) if learned and current_move else 0.6
        )
        max_lookahead = 4 if persistence >= 0.5 else 2
        best = None
        for i in range(1, max_lookahead + 1):
            nr, nc = cur_r + dr * i, cur_c + dc * i
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
    # A* path → (Move, steps) tuple
    # ------------------------------------------------------------------
    def _path_to_move(
        self, path: List[Move], my_position: Tuple[int, int]
    ):
        first_move = path[0] if path else Move.STAY
        move = first_move if isinstance(first_move, Move) else Move.STAY
        steps = 1
        for m in path[1:]:
            if m == move and steps < self.pacman_speed:
                steps += 1
            else:
                break
        return (move, steps)

    def _advance_position(
        self, pos: Tuple[int, int], move: Move, steps: int
    ) -> Tuple[int, int]:
        cur = pos
        for _ in range(steps):
            nxt = _apply(cur, move)
            if not _valid(nxt, self.memory_map):
                break
            cur = nxt
        return cur

    # ------------------------------------------------------------------
    # Frontier exploration (A* fallback when belief is lost)
    # ------------------------------------------------------------------
    def _explore(
        self,
        my_position: Tuple[int, int],
        belief_hint: Optional[Tuple[int, int]] = None,
    ):
        ms = self.memory_map
        if ms is None:
            moves = list(MOVE_ORDER)
            random.shuffle(moves)
            return (moves[0], 1)

        if belief_hint is None:
            belief_hint = getattr(self, "_explore_belief_hint", None)
        self._explore_belief_hint = None

        h, w = ms.shape
        target = None
        best_score = float("inf")
        for r in range(h):
            for c in range(w):
                if ms[r, c] != 0:
                    continue
                has_unknown = any(
                    0 <= r + dr < h and 0 <= c + dc < w and ms[r + dr, c + dc] == -1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if not has_unknown:
                    continue
                d = abs(r - my_position[0]) + abs(c - my_position[1])
                score = d + 4.0 * max(0, r - 8)
                if belief_hint is not None:
                    d_hint = abs(r - belief_hint[0]) + abs(c - belief_hint[1])
                    score += 0.5 * d_hint
                if score < best_score:
                    best_score = score
                    target = (r, c)

        if target is None and belief_hint is not None:
            target = belief_hint

        if target is not None:
            path = _astar(ms, my_position, target)
            if path:
                return self._path_to_move(path, my_position)

        moves = _legal(my_position, ms)
        if moves:
            return (random.choice(moves), 1)
        return (Move.STAY, 1)


# ===========================================================================
# GhostAgent — placeholder (24127561 is the Pacman engineer)
# ===========================================================================

class GhostAgent(BaseGhostAgent):
    """Ghost placeholder.  Not the primary deliverable for student 24127561.
    Uses a simple distance-maximising greedy strategy as a sanity baseline.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Tuple[int, int]] = None

    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = map_state != -1
        self.memory_map[visible_mask] = map_state[visible_mask]

    def step(self, map_state, my_position, enemy_position, step_number: int):
        self._update_memory(map_state)
        my_position = tuple(my_position)
        if enemy_position is not None:
            self.last_seen_enemy = tuple(enemy_position)

        ms = self.memory_map if self.memory_map is not None else map_state
        candidates = _legal(my_position, ms)
        if not candidates:
            return Move.STAY

        threat = self.last_seen_enemy
        if threat is None:
            # No info — pick move with most mobility
            return max(
                candidates,
                key=lambda m: _cell_exits(_apply(my_position, m), ms),
            )

        # Greedy: maximise BFS distance from threat
        from collections import deque as _deque

        def _bfs_dist(start):
            if not _valid(start, ms):
                return {}
            dist: Dict = {start: 0}
            q = _deque([start])
            while q:
                cur = q.popleft()
                for mv in MOVE_ORDER:
                    nxt = _apply(cur, mv)
                    if nxt not in dist and _valid(nxt, ms):
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            return dist

        threat_dist = _bfs_dist(threat)

        def _score(mv: Move) -> float:
            nxt = _apply(my_position, mv)
            distance = threat_dist.get(nxt, _manhattan(nxt, threat) + 20)
            mobility = _cell_exits(nxt, ms)
            return 10 * distance + mobility

        return max(candidates, key=_score)