"""GhostAgent — Blind Hider (Adapted from Lab 1).

Student: 24127192
Role:   Hide Agent Engineer

Lab 2 changes (Blind/Partial Observability):
- Maintains self.memory_map to accumulate observations across steps
- Handles enemy_position = None (Pacman not visible)
- BFS distance maps computed on accumulated memory
- Exploration uses known-empty cells preference
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

import numpy as np

# ===================================================================
# Constants
# ===================================================================
MOVE_ORDER: Tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

TIME_BUDGET = 0.90
A_STAR_THRESHOLD = 4
MINIMAX_DEPTH = 8
HISTORY_BAN = 6


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
    """Valid if within bounds and not wall (1). -1 (unseen) = optimistic traversable."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _legal(pos, ms):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_exits(pos, ms):
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


# ===================================================================
# BFS distance map
# ===================================================================
class BFS:
    def __init__(self, maxsize=64):
        self._cache: Dict[Tuple, Dict[Tuple, int]] = {}
        self._maxsize = maxsize

    def dist(self, ms, start):
        if start in self._cache:
            d = self._cache.pop(start)
            self._cache[start] = d
            return d
        d = self._compute(ms, start)
        self._cache[start] = d
        if len(self._cache) > self._maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return d

    @staticmethod
    def _compute(ms, start):
        if not _valid(start, ms):
            return {}
        d = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if nxt not in d and _valid(nxt, ms):
                    d[nxt] = d[cur] + 1
                    q.append(nxt)
        return d


# ===================================================================
# A* pathfinding
# ===================================================================
def astar(ms, start, goal):
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []

    import heapq
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


def pacman_reach_2(pos, ms):
    reach = {pos}
    for m1 in MOVE_ORDER:
        p1 = _apply(pos, m1)
        if not _valid(p1, ms):
            continue
        reach.add(p1)
        for m2 in MOVE_ORDER:
            p2 = _apply(p1, m2)
            if _valid(p2, ms):
                reach.add(p2)
    return list(reach)


# ===================================================================
# GhostAgent — Blind Hider
# ===================================================================
class GhostAgent(BaseGhostAgent):
    """Blind Ghost: A*-Strategic + Minimax-Tactical, adapted for partial observability."""

    _topology_cache: Dict[int, Dict] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._enemy: Optional[Tuple[int, int]] = None

        # === Blind mode state ===
        self.memory_map: Optional[np.ndarray] = None

        self._bfs = BFS()
        self._history: List[Tuple[int, int]] = []
        self._step_count: int = 0
        self._flee_target: Optional[Tuple[int, int]] = None
        self._dead_ends: Set[Tuple] = set()
        self._initialized: bool = False
        self._last_move: Optional[Move] = None
        self._topo: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()
        me = tuple(int(v) for v in my_position)
        self._step_count = step_number

        # Update memory map
        self._update_memory(map_state)

        # One-time map analysis
        if not self._initialized:
            self._analyze_map(self.memory_map)
            self._initialized = True

        # Track enemy
        if enemy_position is not None:
            self._enemy = tuple(int(v) for v in enemy_position)

        # Track history
        if not self._history or self._history[-1] != me:
            self._history.append(me)

        # Legal moves (on memory map)
        candidates = _legal(me, self.memory_map)
        if not candidates:
            return Move.STAY

        pac = self._enemy
        if pac is None:
            return self._explore(me, candidates)

        # Compute distances on memory map
        pd = self._bfs.dist(self.memory_map, pac)
        gd = self._bfs.dist(self.memory_map, me)
        bfs_d = pd.get(me, _manhattan(me, pac))

        # Filter oscillation
        safe_candidates = self._filter_oscillation(me, candidates)

        chosen_move = None

        # Strategic: A* to farthest safe cell
        if bfs_d > A_STAR_THRESHOLD:
            chosen_move = self._strategic_move(me, pac, pd, gd, safe_candidates)
            if chosen_move is None:
                chosen_move = self._floodfill_move(me, pac, pd, safe_candidates)

        # Tactical: minimax evasion
        if chosen_move is None:
            chosen_move = self._tactical_move(me, pac, pd, gd, safe_candidates, t0)

        # Rerank with floodfill safety
        result = self._rerank_with_floodfill(me, pac, pd, candidates, chosen_move)
        if result is not None:
            self._last_move = result
        return result or Move.STAY

    # ------------------------------------------------------------------
    # Memory map
    # ------------------------------------------------------------------
    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    # ------------------------------------------------------------------
    # Strategic: find farthest safe cell and A* to it
    # ------------------------------------------------------------------
    def _strategic_move(self, me, pac, pd, gd, candidates):
        best_cell = None
        best_dist = 0
        for cell, g_val in gd.items():
            if g_val < 3 or g_val > 18:
                continue
            p_val = pd.get(cell, 999)
            p_turns = (p_val + 1) // 2
            if g_val >= p_turns:
                continue
            if _cell_exits(cell, self.memory_map) <= 1:
                continue
            if p_val > best_dist:
                best_dist = p_val
                best_cell = cell
        if best_cell is None:
            return None
        self._flee_target = best_cell
        path = astar(self.memory_map, me, best_cell)
        if not path or path[0] not in candidates:
            return None
        nxt = _apply(me, path[0])
        if _manhattan(nxt, pac) < 2:
            return None
        return path[0]

    # ------------------------------------------------------------------
    # Floodfill safety scoring
    # ------------------------------------------------------------------
    def _score_position(self, gpos, pac, pd):
        score = 0.0
        bfs_dist = pd.get(gpos, _manhattan(gpos, pac))
        score += 10.0 * bfs_dist

        gd = self._bfs.dist(self.memory_map, gpos)
        safe_count = 0
        for _cell_key, g_val in gd.items():
            if g_val > 15:
                continue
            p_val = pd.get(_cell_key, 999)
            if g_val < (p_val + 1) // 2:
                safe_count += 1
        score += 0.5 * safe_count

        exits = _cell_exits(gpos, self.memory_map)
        if exits >= 3:
            score += 30.0
        elif exits <= 1:
            score -= 50.0
        elif exits == 2 and bfs_dist <= 6:
            score -= 15.0

        if gpos in self._dead_ends:
            score -= 80.0
        return score

    def _floodfill_move(self, me, pac, pd, candidates):
        if not candidates:
            return None
        best_move = None
        best_score = float("-inf")
        for m in candidates:
            nxt = _apply(me, m)
            if _manhattan(nxt, pac) < 2:
                continue
            s = self._score_position(nxt, pac, pd)
            if s > best_score:
                best_score = s
                best_move = m
        return best_move

    def _rerank_with_floodfill(self, me, pac, pd, candidates, chosen_move):
        if not candidates or chosen_move is None:
            return chosen_move
        chosen_nxt = _apply(me, chosen_move)
        chosen_score = self._score_position(chosen_nxt, pac, pd)
        best_move = chosen_move
        best_score = chosen_score
        for m in candidates:
            nxt = _apply(me, m)
            if _manhattan(nxt, pac) < 2:
                continue
            s = self._score_position(nxt, pac, pd)
            if s > best_score:
                best_score = s
                best_move = m
        if best_move == chosen_move:
            return chosen_move
        chosen_exits = _cell_exits(chosen_nxt, self.memory_map)
        chosen_bfs = pd.get(chosen_nxt, _manhattan(chosen_nxt, pac))
        is_dangerous = (chosen_exits <= 1 or chosen_bfs <= 3)
        if best_score > chosen_score + 30.0 or is_dangerous:
            return best_move
        return chosen_move

    # ------------------------------------------------------------------
    # Tactical: minimax alpha-beta
    # ------------------------------------------------------------------
    def _tactical_move(self, me, pac, pd, gd, candidates, t0):
        if not candidates:
            return Move.STAY
        depth = MINIMAX_DEPTH
        alpha, beta = float("-inf"), float("inf")
        best_move = candidates[0]
        best_val = float("-inf")

        cand_next = {m: _apply(me, m) for m in candidates}
        ordered = sorted(candidates,
                         key=lambda m: pd.get(cand_next[m], _manhattan(cand_next[m], pac) + 50),
                         reverse=True)

        for m in ordered:
            ng = cand_next[m]
            if _manhattan(ng, pac) < 2:
                val = -100000.0
            else:
                val = self._minimax(ng, pac, depth - 1, False, alpha, beta, pd, t0)
            if val is None:
                break
            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, val)
        return best_move

    def _minimax(self, gpos, ppos, depth, is_ghost, alpha, beta, pd_orig, t0):
        ms = self.memory_map
        if time.time() - t0 > TIME_BUDGET:
            return None
        md = _manhattan(gpos, ppos)
        if md < 2:
            return -100000.0 + depth * 100
        if depth <= 0:
            bfs_d = pd_orig.get(gpos, md + 50)
            if _cell_exits(gpos, ms) <= 1:
                bfs_d -= 50
            if gpos in self._history[-6:]:
                bfs_d -= 20 * list(self._history[-6:]).count(gpos)
            return bfs_d * 10.0

        if is_ghost:
            value = float("-inf")
            moves = _legal(gpos, ms)
            moves.sort(key=lambda m: pd_orig.get(
                _apply(gpos, m), _manhattan(_apply(gpos, m), ppos) + 50), reverse=True)
            for m in moves:
                ng = _apply(gpos, m)
                if _manhattan(ng, ppos) < 2:
                    v = -100000.0
                else:
                    v = self._minimax(ng, ppos, depth - 1, False, alpha, beta, pd_orig, t0)
                if v is None:
                    return None
                value = max(value, v)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float("inf")
            for np_ in pacman_reach_2(ppos, ms):
                v = self._minimax(gpos, np_, depth - 1, True, alpha, beta, pd_orig, t0)
                if v is None:
                    return None
                value = min(value, v)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # ------------------------------------------------------------------
    # Anti-oscillation
    # ------------------------------------------------------------------
    def _filter_oscillation(self, me, candidates):
        if len(self._history) < 2:
            return candidates
        recent = self._history[-HISTORY_BAN:] if len(self._history) >= HISTORY_BAN else self._history
        recent_set = set(recent)
        filtered = [m for m in candidates if _apply(me, m) not in recent_set]
        return filtered if filtered else candidates

    # ------------------------------------------------------------------
    # Map analysis
    # ------------------------------------------------------------------
    def _analyze_map(self, ms):
        h, w = _shape(ms)
        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1:
                    continue
                if _cell_exits((r, c), ms) <= 1:
                    self._dead_ends.add((r, c))

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------
    def _explore(self, pos, candidates):
        ms = self.memory_map
        best = candidates[0]
        best_flood = 0
        for m in candidates:
            nxt = _apply(pos, m)
            seen = {nxt}
            q = deque([nxt])
            while q:
                cur = q.popleft()
                for mv in MOVE_ORDER:
                    neighbor = _apply(cur, mv)
                    if neighbor not in seen and _valid(neighbor, ms):
                        seen.add(neighbor)
                        q.append(neighbor)
                        if len(seen) > 50:
                            break
                if len(seen) > 50:
                    break
            if len(seen) > best_flood:
                best_flood = len(seen)
                best = m
        return best


# ===================================================================
# PacmanAgent — placeholder (not primary deliverable for 24127192)
# ===================================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
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
        return (Move.STAY, 1)
