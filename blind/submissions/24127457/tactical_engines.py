"""tactical_engines.py — Advanced tactical decision-making components.

Provides:
  IntentTracker       — predicts enemy target junction from velocity history
  TrapEvaluator       — simulates future reachable space to detect traps
  DynamicModeSelector — context-aware mode switching for Ghost agent
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from topology_analyzer import (
    MOVE_ORDER, DIRS, TopologyAnalyzer,
    _shape, _cell, _valid, _apply, _manhattan, _cell_exits,
    bfs_dist, astar,
)


# ===================================================================
# Intent Tracker — predicts enemy target from movement history
# ===================================================================
class IntentTracker:
    """Tracks enemy movement history and projects intended target junction.

    Uses velocity vector + streak to predict which junction the enemy
    is heading toward, enabling interception planning.
    """

    def __init__(self, max_history: int = 8):
        self.history: List[Tuple] = []
        self.velocity: Optional[Tuple] = None
        self.velocity_streak: int = 0
        self.max_history = max_history

    def update(self, current_pos: Tuple) -> None:
        if self.history and current_pos != self.history[-1]:
            dr = current_pos[0] - self.history[-1][0]
            dc = current_pos[1] - self.history[-1][1]
            new_vel = (dr, dc)
            if new_vel == self.velocity and (dr != 0 or dc != 0):
                self.velocity_streak += 1
            else:
                self.velocity = new_vel
                self.velocity_streak = 1 if (dr != 0 or dc != 0) else 0
        self.history.append(current_pos)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def project_target(self, topo: TopologyAnalyzer, ms,
                       depth: int = 5) -> Optional[Tuple]:
        """Project enemy's intended junction/core along velocity vector."""
        if self.velocity_streak < 2 or self.velocity is None or not self.history:
            return None
        dr, dc = self.velocity
        cur = self.history[-1]
        H, W = _shape(ms)
        target = None
        for i in range(1, depth + 1):
            nr, nc = cur[0] + dr * i, cur[1] + dc * i
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if _cell(ms, nr, nc) == 1:
                break
            cell = (nr, nc)
            if cell in topo.junctions:
                target = cell
            elif cell in topo.core and target is None:
                target = cell
        return target

    def predict_bottleneck(self, topo: TopologyAnalyzer, ms,
                           enemy_pos: Tuple, depth: int = 5) -> Optional[Tuple]:
        """Find best junction to intercept enemy along its projected path."""
        projected = self.project_target(topo, ms, depth)
        if projected is not None:
            return projected
        if self.velocity is None:
            return None
        dr, dc = self.velocity
        best, best_score = None, float("-inf")
        for junc in topo.junctions:
            dot = dr * (junc[0] - enemy_pos[0]) + dc * (junc[1] - enemy_pos[1])
            if dot > 0:
                d = _manhattan(junc, enemy_pos)
                if d <= depth:
                    exits = _cell_exits(junc, ms)
                    score = dot / (d + 1) + exits * 10
                    if score > best_score:
                        best_score = score
                        best = junc
        return best


# ===================================================================
# Trap Evaluator — future reachable space simulation
# ===================================================================
class TrapEvaluator:
    """Simulates future positions to detect dead-end traps.

    Includes result caching to stay within 1.0s/step time limit.
    """

    def __init__(self):
        self._trap_cache: Dict[Tuple, bool] = {}

    def reset_cache(self):
        self._trap_cache.clear()

    def is_dead_end_trap(self, cell: Tuple, topo: TopologyAnalyzer,
                         ms, lookahead: int = 6) -> bool:
        """Check if cell leads only to dead-ends within lookahead steps."""
        cache_key = (cell, lookahead)
        if cache_key in self._trap_cache:
            return self._trap_cache[cache_key]

        if cell in topo.dead_ends:
            self._trap_cache[cache_key] = True
            return True
        visited = {cell}
        queue = deque([(cell, 0)])
        found_open = False
        while queue:
            cur, dist = queue.popleft()
            if dist >= lookahead:
                found_open = True
                break
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if nxt in visited or not _valid(nxt, ms):
                    continue
                visited.add(nxt)
                if nxt in topo.junctions or (nxt in topo.core and nxt not in topo.corridor_cells):
                    found_open = True
                    break
                queue.append((nxt, dist + 1))
            if found_open:
                break
        result = not found_open
        self._trap_cache[cache_key] = result
        return result

    def escape_margin(self, my_pos: Tuple, enemy_pos: Tuple,
                      ms, pacman_speed: int = 2, sim_depth: int = 6) -> float:
        """Ratio: Ghost reachable space / Pacman reachable space. < 1.0 = at risk."""
        gd = bfs_dist(ms, my_pos, max_dist=sim_depth)
        pd = bfs_dist(ms, enemy_pos, max_dist=sim_depth * pacman_speed)
        ghost_reachable = len(gd)
        pacman_reachable = len(pd)
        if ghost_reachable == 0:
            return 0.0
        return ghost_reachable / max(1, pacman_reachable)

    def corridor_depth(self, cell: Tuple, topo: TopologyAnalyzer, ms) -> int:
        """How many steps until reaching an open area (junction/core)?"""
        if cell in topo.junctions or cell in topo.core:
            return 0
        visited = {cell}
        queue = deque([(cell, 0)])
        while queue:
            cur, dist = queue.popleft()
            if cur in topo.junctions or cur in topo.core:
                return dist
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if nxt in visited or not _valid(nxt, ms):
                    continue
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return 99


# ===================================================================
# Dynamic Mode Selector for Ghost
# ===================================================================
class DynamicModeSelector:
    """Manages Ghost mode switching based on game state.

    Modes:
      panic       — enemy distance < 4 (3-ply minimax)
      evasion     — enemy distance 4-8 (strategic flee)
      fortress    — enemy distance >= 8 or game > 75% (maximize distance)
      exploration — enemy not visible (belief-guided safe explore)
    """

    MODE_PANIC       = "panic"
    MODE_EVASION     = "evasion"
    MODE_FORTRESS    = "fortress"
    MODE_EXPLORATION = "exploration"

    ALL_MODES = [MODE_PANIC, MODE_EVASION, MODE_FORTRESS, MODE_EXPLORATION]

    def __init__(self, hysteresis: int = 3):
        self.current_mode: str = self.MODE_EXPLORATION
        self.mode_duration: Dict[str, int] = {m: 0 for m in self.ALL_MODES}
        self.mode_history: List[str] = []
        self.hysteresis = hysteresis

    def select(self, enemy_visible: bool, distance: Optional[int],
               step_number: int, max_steps: int,
               topology_safety: float) -> str:
        game_progress = step_number / max_steps

        if not enemy_visible:
            new_mode = self.MODE_EXPLORATION
        elif distance is not None and distance < 4:
            new_mode = self.MODE_PANIC
        elif distance is not None and distance < 8:
            new_mode = self.MODE_EVASION
        elif game_progress > 0.75 or (distance is not None and distance >= 8):
            new_mode = self.MODE_FORTRESS
        else:
            new_mode = self.MODE_EXPLORATION

        if new_mode != self.current_mode:
            if self.mode_duration[self.current_mode] < self.hysteresis:
                return self.current_mode
            self.current_mode = new_mode
            self.mode_duration = {m: 0 for m in self.ALL_MODES}

        self.mode_duration[self.current_mode] += 1
        self.mode_history.append(self.current_mode)
        return self.current_mode
