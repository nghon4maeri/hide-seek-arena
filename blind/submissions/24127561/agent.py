"""PacmanAgent — Blind Seeker (Adapted from Lab 1).

Student: 24127561
Role:   Seek Agent Engineer

Lab 2 changes (Blind/Partial Observability):
- Maintains self.memory_map to accumulate observations across steps
- Handles enemy_position = None (enemy not visible)
- A* pathfinding runs on memory_map (optimistic: treats -1 as traversable)
- Frontier-based exploration when enemy is lost
"""

from __future__ import annotations

import sys
import heapq
import random
from pathlib import Path
from typing import List, Optional, Tuple

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

        # Ghost direction tracking
        self._enemy_direction = None
        self._direction_streak = 0

        # Path cache
        self._cached_target = None
        self._cached_path: List = []
        self._cached_my_pos = None

        # Feature gates
        self.enable_interception = True

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        # Update accumulated memory map
        self._update_memory(map_state)

        target = None

        if enemy_position is not None:
            enemy_position = tuple(enemy_position)
            self._update_ghost_tracking(enemy_position)

            # Interception planning
            if self.enable_interception and self._direction_streak >= 2:
                inter_target = self._compute_interception_target(
                    self.memory_map, enemy_position, my_position
                )
                if inter_target is not None:
                    path_to_inter = astar(self.memory_map, my_position, inter_target)
                    path_to_direct = astar(self.memory_map, my_position, enemy_position)
                    dist_inter_to_ghost = (
                        abs(inter_target[0] - enemy_position[0])
                        + abs(inter_target[1] - enemy_position[1])
                    )
                    if path_to_inter and (
                        not path_to_direct
                        or len(path_to_inter) <= len(path_to_direct)
                        or dist_inter_to_ghost <= 2
                    ):
                        target = inter_target

            if target is None:
                target = enemy_position
            self.last_seen_enemy = enemy_position
        else:
            # Enemy not visible — use last known position or explore
            if self.last_seen_enemy is None:
                return self._explore(my_position)
            if my_position == self.last_seen_enemy:
                self.last_seen_enemy = None
                return self._explore(my_position)
            target = self.last_seen_enemy

        if my_position == target:
            return (Move.STAY, 1)

        # Path caching
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
    # Ghost direction tracking
    # ------------------------------------------------------------------
    def _update_ghost_tracking(self, enemy_pos):
        if self.last_seen_enemy is None:
            return
        dr = enemy_pos[0] - self.last_seen_enemy[0]
        dc = enemy_pos[1] - self.last_seen_enemy[1]
        new_dir = (dr, dc)
        if new_dir == self._enemy_direction and (dr != 0 or dc != 0):
            self._direction_streak += 1
        else:
            self._enemy_direction = new_dir
            self._direction_streak = 1 if (dr != 0 or dc != 0) else 0

    # ------------------------------------------------------------------
    # Interception target
    # ------------------------------------------------------------------
    def _compute_interception_target(self, ms, enemy_pos, my_pos):
        dr, dc = self._enemy_direction
        cur_row, cur_col = enemy_pos
        best = None
        for i in range(1, 5):
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
    # Exploration (frontier-based)
    # ------------------------------------------------------------------
    def _explore(self, my_position):
        ms = self.memory_map
        if ms is None:
            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(moves)
            return (moves[0], 1)

        # Find nearest frontier cell
        h, w = ms.shape
        target = None
        best_dist = float("inf")
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
                    if d < best_dist:
                        best_dist = d
                        target = (r, c)

        if target:
            path = astar(ms, my_position, target)
            if path:
                return self._path_to_move(path, my_position)

        moves = _legal(my_position, ms)
        if moves:
            return (random.choice(moves), 1)
        return (Move.STAY, 1)


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
        return Move.STAY
