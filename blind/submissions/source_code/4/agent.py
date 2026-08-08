"""
agent.py — Combined Blind Ghost Agent (24127192) + Pacman Agent (submissions/agent)

Ghost: Blind Multi-Layer Ghost Agent with Policy Portfolio (from 24127192)
Pacman: Optimized Pacman with Interception + Speed Packing (from submissions/agent)

Supporting modules inlined from submissions/agent/:
  - pathfinding.py (bfs_dist, astar_path, etc.)
  - belief_state.py (BeliefState)
  - topology.py (PacmanTopologyAnalyzer, STATIC_FULL_MAP)
"""

from __future__ import annotations

import math
import random
import sys
import time
import heapq
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

# ===================================================================
# Constants
# ===================================================================
Pos = Tuple[int, int]
Action = Tuple[int, int]

MOVE_ORDER: Tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVE = {(-1, 0): Move.UP, (1, 0): Move.DOWN, (0, -1): Move.LEFT, (0, 1): Move.RIGHT, (0, 0): Move.STAY}

TIME_BUDGET        = 0.85
CAPTURE_DISTANCE   = 2
INF                = 10 ** 9
INF_DIST           = 10 ** 9
FATAL_DANGER       = 120.0

# MC V3
MC_ROLLOUTS        = 12
MC_DEPTH           = 15
MC_CAPTURE_PENALTY = 120_000.0
MC_SURVIVE_BONUS   = 8_000.0
CVAR_ALPHA         = 0.20
CVAR_WEIGHT        = 0.60
MEAN_WEIGHT        = 0.40

# Alpha-Beta
AB_MAX_DEPTH       = 8
PANIC_DISTANCE     = 8

# Ensemble
ENSEMBLE_LR        = 0.2
MIN_MODEL_WEIGHT   = 0.05

# Arbitrator
RISK_WEIGHT        = 0.85
CONFIDENCE_WEIGHT  = 0.15
COST_WEIGHT        = 0.05

# History & Anti-loop
HISTORY_LEN        = 20
BELIEF_MAX_CELLS   = 18
DANGER_HORIZON     = 6
MARKOV_LIMIT       = 6
RECENT_CELL_BAN    = 30

# Early Branch Priority
EARLY_BRANCH_STEPS = 40

# Zone Navigation
ZONE_SWITCH_INTERVAL = 20
ZONE_NAMES = ("top", "center", "bottom", "left", "right")

# Directional Momentum
MOMENTUM_BONUS         = 45.0
REVERSAL_PENALTY       = 500.0
TURN_PENALTY_90        = 100.0

# Hardcoded Known Layout
KNOWN_LAYOUT_STR = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.###.#.###.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........G.........#",
    "#####.#.#####.#.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.....P.....#...#",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################",
]

# Static full map for PacmanAgent (submissions/agent topology)
_STATIC_LAYOUT = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.###.#.###.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#####.#.#####.#.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.........#...#..",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################",
]

STATIC_FULL_MAP = np.array(
    [[1 if c == '#' else 0 for c in row] for row in _STATIC_LAYOUT],
    dtype=int,
)
STATIC_FULL_MAP[9, 10] = 0
STATIC_FULL_MAP[15, 10] = 0

PACMAN_START: Pos = (15, 10)
GHOST_START: Pos = (9, 9)

CONF_MAP = {"high": 1.0, "medium": 0.5, "low": 0.2}


# ===================================================================
# Grid Utilities
# ===================================================================
def _shape(ms) -> Tuple[int, int]:
    if hasattr(ms, "shape"):
        return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else 0


def _cell(ms, r: int, c: int) -> int:
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])


def _apply(pos: Pos, move) -> Pos:
    if isinstance(move, Move):
        return (pos[0] + move.value[0], pos[1] + move.value[1])
    return (pos[0] + move[0], pos[1] + move[1])


def _valid(pos: Pos, ms) -> bool:
    """Valid if within bounds and not wall (1). -1 (unseen) = optimistic traversable."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _legal(pos: Pos, ms) -> List[Move]:
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _legal_deltas(pos: Pos, ms) -> List[Tuple[int, int]]:
    """Like _legal but returns raw delta tuples for PacmanAgent compatibility."""
    return [d for d in DIRS if _valid(_apply(pos, d), ms)]


def _neighbors(pos: Pos, ms) -> List[Pos]:
    return [_apply(pos, m) for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _exits(pos: Pos, ms) -> int:
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


# Alias for PacmanAgent compatibility
_cell_exits = _exits


def _move_key(move: Move) -> int:
    return {Move.UP: 0, Move.LEFT: 1, Move.RIGHT: 2, Move.DOWN: 3,
            Move.STAY: 4}.get(move, 9)


def _pacman_reach(pac: Pos, ms, speed: int = 2) -> List[Pos]:
    """All positions Pacman can reach in one turn (speed steps in straight line)."""
    reach: Set[Pos] = {pac}
    for m in MOVE_ORDER:
        cur = pac
        for _ in range(max(1, speed)):
            nxt = _apply(cur, m)
            if not _valid(nxt, ms):
                break
            reach.add(nxt)
            cur = nxt
    return sorted(reach)


def _bucket(v: int) -> int:
    if v <= -6:   return -4
    if v <= -3:   return -3
    if v < 0:     return -1
    if v == 0:    return 0
    if v < 3:     return 1
    if v < 6:     return 3
    return 4


# ===================================================================
# A* Pathfinding (returns Move list — for GhostAgent)
# ===================================================================
def astar(ms, start: Pos, goal: Pos) -> List[Move]:
    """A* on memory_map. Treats -1 (unseen) as traversable (optimistic).
    Returns list of Move enums."""
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []

    open_set = [(0, 0, start)]
    came_from: Dict[Pos, Tuple[Pos, Move]] = {}
    g_score: Dict[Pos, int] = {start: 0}
    closed: Set[Pos] = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path: List[Move] = []
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
# Pathfinding Extras (for PacmanAgent from submissions/agent)
# ===================================================================
def bfs_dist(ms, start: Pos, max_dist: int = 40) -> Dict[Pos, int]:
    """BFS from start, returns dict of {cell: distance}."""
    if not _valid(start, ms):
        return {start: 0}
    d: Dict[Pos, int] = {start: 0}
    q: deque[Pos] = deque([start])
    while q:
        cur = q.popleft()
        if d[cur] >= max_dist:
            continue
        for delta in DIRS:
            nxt = _apply(cur, delta)
            if nxt not in d and _valid(nxt, ms):
                d[nxt] = d[cur] + 1
                q.append(nxt)
    return d


def astar_path(ms, start: Pos, goal: Pos) -> List[Pos]:
    """A* search returning list of positions along path.
    Used by PacmanAgent (from submissions/agent)."""
    if goal is None or not _valid(start, ms) or not _valid(goal, ms):
        return []
    if start == goal:
        return []
    open_set = [(0, 0, start)]
    came_from: Dict[Pos, Pos] = {}
    g_score: Dict[Pos, int] = {start: 0}
    closed: Set[Pos] = set()
    while open_set:
        _, g_val, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path: List[Pos] = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        for dr, dc in DIRS:
            nxt = (current[0] + dr, current[1] + dc)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = g_val + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = current
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


# ===================================================================
# BeliefState (from submissions/agent/belief_state.py)
# ===================================================================
class BeliefState:
    """Maintains probability distribution over enemy location.

    When enemy is seen: belief collapses to point mass.
    When enemy is lost: nullify visible cells, propagate via BFS.
    Also maintains danger_time[r,c] = estimated steps for enemy to reach (r,c).
    """

    def __init__(self, H: int = 21, W: int = 21):
        self.H, self.W = H, W
        self.belief = np.ones((H, W), dtype=np.float64) / (H * W)
        self.danger_time = np.full((H, W), 999, dtype=np.int32)
        self.last_seen_pos: Optional[Tuple] = None
        self.steps_since_seen: int = 0

    def update(self, my_pos: Tuple, enemy_pos: Optional[Tuple],
               memory_map: np.ndarray, enemy_speed: int = 2):
        """Update belief state based on observation."""
        if enemy_pos is not None:
            self.belief.fill(0.0)
            er, ec = enemy_pos
            if 0 <= er < self.H and 0 <= ec < self.W:
                self.belief[er, ec] = 1.0
            self.last_seen_pos = enemy_pos
            self.steps_since_seen = 0
            self._update_danger_time(enemy_pos, memory_map, enemy_speed)
            return

        self.steps_since_seen += 1
        self._nullify_visible(memory_map)
        self._propagate(memory_map, enemy_speed)
        total = self.belief.sum()
        if total > 0:
            self.belief /= total
        self._update_danger_time_from_belief(memory_map, enemy_speed)

    def _nullify_visible(self, memory_map):
        """Set belief to 0 for all cells we can currently see (empty = no enemy)."""
        for r in range(self.H):
            for c in range(self.W):
                if _cell(memory_map, r, c) == 0:
                    self.belief[r, c] = 0.0

    def _propagate(self, memory_map, speed):
        """Propagate belief via BFS expansion (Markov chain)."""
        new_belief = np.zeros_like(self.belief)
        for r in range(self.H):
            for c in range(self.W):
                prob = self.belief[r, c]
                if prob <= 0:
                    continue
                reachable = self._bfs_reachable((r, c), memory_map, speed)
                denom = max(1, len(reachable))
                for cell in reachable:
                    new_belief[cell[0], cell[1]] += prob / denom
        self.belief = new_belief

    def _bfs_reachable(self, start, memory_map, max_dist):
        """BFS to find all cells reachable within max_dist steps."""
        reachable = [start]
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            cur, dist = queue.popleft()
            if dist >= max_dist:
                continue
            for dr, dc in DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in visited:
                    continue
                if not _valid(nxt, memory_map):
                    continue
                visited.add(nxt)
                reachable.append(nxt)
                queue.append((nxt, dist + 1))
        return reachable

    def _update_danger_time(self, enemy_pos, memory_map, speed):
        """Update danger_time: BFS from enemy position."""
        self.danger_time.fill(999)
        dist_map = bfs_dist(memory_map, enemy_pos, max_dist=40)
        for cell, dist in dist_map.items():
            r, c = cell
            self.danger_time[r, c] = max(1, dist // speed)

    def _update_danger_time_from_belief(self, memory_map, speed):
        """Update danger_time from belief distribution."""
        self.danger_time.fill(999)
        high_prob = np.argwhere(self.belief > 0.05)
        for idx in high_prob:
            r, c = idx[0], idx[1]
            dist_map = bfs_dist(memory_map, (r, c), max_dist=20)
            for cell, dist in dist_map.items():
                cr, cc = cell
                time_est = max(1, dist // speed)
                self.danger_time[cr, cc] = min(self.danger_time[cr, cc], time_est)

    def threat_center(self) -> Tuple[int, int]:
        """Weighted centroid of belief distribution."""
        total = self.belief.sum()
        if total == 0:
            return (self.H // 2, self.W // 2)
        r_center = float(np.sum(np.arange(self.H)[:, None] * self.belief) / total)
        c_center = float(np.sum(np.arange(self.W) * self.belief.sum(axis=0)) / total)
        return (int(r_center), int(c_center))

    def highest_prob_cells(self, top_k: int = 5) -> List[Tuple[Tuple, float]]:
        """Return top-k cells with highest probability."""
        flat = self.belief.ravel()
        if flat.sum() == 0:
            return []
        indices = np.argsort(flat)[::-1][:top_k]
        cells = []
        for idx in indices:
            r, c = divmod(idx, self.W)
            if self.belief[r, c] > 0:
                cells.append(((r, c), float(self.belief[r, c])))
        return cells

    def prob_at(self, pos: Tuple) -> float:
        """Get probability at a specific position."""
        r, c = pos
        if 0 <= r < self.H and 0 <= c < self.W:
            return float(self.belief[r, c])
        return 0.0

    def danger_at(self, pos: Tuple) -> int:
        """Get estimated time for enemy to reach position."""
        r, c = pos
        if 0 <= r < self.H and 0 <= c < self.W:
            return int(self.danger_time[r, c])
        return 999


# ===================================================================
# PacmanTopologyAnalyzer (from submissions/agent/topology.py)
# Renamed to avoid conflict with GhostAgent's TopologyAnalyzer
# ===================================================================
class PacmanTopologyAnalyzer:
    """Full topology analysis with core, loops, dead-end depth.
    Used by the PacmanAgent from submissions/agent/."""

    def __init__(self):
        self.dead_ends: Set[Tuple] = set()
        self.junctions: Set[Tuple] = set()
        self.corridor_cells: Set[Tuple] = set()
        self.core: Set[Tuple] = set()
        self.loops: Set[Tuple] = set()
        self.dead_end_depth: Dict[Tuple, int] = {}
        self.junction_dist: Dict[Tuple, int] = {}
        self.loop_dist: Dict[Tuple, int] = {}
        self.degree: Dict[Tuple, int] = {}
        self._weight_cache: Dict[Tuple, float] = {}
        self.ready = False
        self._H = 21
        self._W = 21

    def analyze(self, ms):
        if self.ready:
            return
        H, W = _shape(ms)
        self._H, self._W = H, W
        all_free = {(r, c) for r in range(H) for c in range(W) if _cell(ms, r, c) != 1}

        # --- 1. Degree classification ---
        for p in all_free:
            deg = _exits(p, ms)
            self.degree[p] = deg
            if deg >= 3:
                self.junctions.add(p)
            elif deg <= 1:
                self.dead_ends.add(p)

        # --- 2. Corridor tracing from dead-ends ---
        dead_seeds = set(self.dead_ends)
        for seed in dead_seeds:
            cur, prev = seed, None
            while True:
                self.corridor_cells.add(cur)
                if cur in self.junctions:
                    break
                nxts = [_apply(cur, m) for m in MOVE_ORDER
                        if _valid(_apply(cur, m), ms) and _apply(cur, m) != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt in self.corridor_cells:
                    break
                prev, cur = cur, nxt

        # --- 3. Dead-end depth ---
        for de in self.dead_ends:
            cur = de
            depth = 0
            visited = {de}
            while cur not in self.junctions:
                found_next = False
                for dr, dc in DIRS:
                    nxt = (cur[0] + dr, cur[1] + dc)
                    if nxt not in visited and _valid(nxt, ms):
                        visited.add(nxt)
                        cur = nxt
                        depth += 1
                        found_next = True
                        break
                if not found_next or depth > 20:
                    break
            self.dead_end_depth[de] = min(depth, 10)

        # --- 4. Core computation (iterative leaf-trimming) ---
        active = set(all_free)
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for cell in active:
                na = sum(1 for m in MOVE_ORDER
                         if _valid(_apply(cell, m), ms) and _apply(cell, m) in active)
                if na <= 1:
                    to_remove.add(cell)
            if to_remove:
                active -= to_remove
                changed = True
        self.core = active

        # --- 5. Loop detection (DFS back-edge, largest cycle) ---
        self.loops = set()
        if self.core:
            cycles = []
            vis = set()
            parent = {}
            for start in list(self.core):
                if start in vis:
                    continue
                stack = [(start, None, iter(
                    _apply(start, m) for m in MOVE_ORDER
                    if _valid(_apply(start, m), ms) and _apply(start, m) in self.core
                ))]
                vis.add(start)
                parent[start] = None
                while stack:
                    u, p, it = stack[-1]
                    try:
                        v = next(it)
                    except StopIteration:
                        stack.pop()
                        continue
                    if v not in self.core:
                        continue
                    if v not in vis:
                        vis.add(v)
                        parent[v] = u
                        stack.append((v, u, iter(
                            _apply(v, m) for m in MOVE_ORDER
                            if _valid(_apply(v, m), ms) and _apply(v, m) in self.core
                        )))
                    elif v != p:
                        cycle = [v]
                        cur = u
                        while cur is not None and cur != v:
                            cycle.append(cur)
                            cur = parent.get(cur)
                        if cur == v and len(cycle) >= 4:
                            cycles.append(cycle)
            if cycles:
                self.loops = set(max(cycles, key=len))

        # --- 6. Junction-distance BFS ---
        self.junction_dist = {p: 10**9 for p in all_free}
        q = deque()
        starts = self.junctions if self.junctions else self.core
        for p in starts:
            self.junction_dist[p] = 0
            q.append(p)
        while q:
            cur = q.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if _valid(nxt, ms) and self.junction_dist.get(nxt, 10**9) > self.junction_dist[cur] + 1:
                    self.junction_dist[nxt] = self.junction_dist[cur] + 1
                    q.append(nxt)

        # --- 7. Loop-distance BFS ---
        self.loop_dist = {p: 10**9 for p in all_free}
        lq = deque()
        starts = self.loops if self.loops else (self.core or self.junctions)
        for p in starts:
            self.loop_dist[p] = 0
            lq.append(p)
        while lq:
            cur = lq.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if _valid(nxt, ms) and self.loop_dist.get(nxt, 10**9) > self.loop_dist[cur] + 1:
                    self.loop_dist[nxt] = self.loop_dist[cur] + 1
                    lq.append(nxt)

        self._weight_cache.clear()
        self.ready = True

    def weight(self, cell, ms):
        """Topological weight for scoring."""
        if cell in self._weight_cache:
            return self._weight_cache[cell]
        w = 1.0
        if cell in self.junctions:
            w = 4.0
        elif cell in self.core and cell not in self.corridor_cells:
            ex = self.degree.get(cell, 2)
            w = 3.0 if ex >= 3 else 2.0 if ex == 2 else 1.0
        elif cell in self.corridor_cells:
            w = 0.5
        elif cell in self.dead_ends:
            w = 0.1
        self._weight_cache[cell] = w
        return w


# ===================================================================
# Proposal
# ===================================================================
class Proposal:
    """Single action proposal from a policy."""
    __slots__ = ("action", "value", "risk", "confidence", "source", "cost", "reason")

    def __init__(self, action: Move, value: float, risk: float,
                 confidence: str = "medium", source: str = "",
                 cost: float = 0.0, reason: str = ""):
        self.action = action
        self.value = value
        self.risk = risk
        self.confidence = confidence
        self.source = source
        self.cost = cost
        self.reason = reason

    def score(self) -> float:
        cv = CONF_MAP.get(self.confidence, 0.3)
        return (self.value
                - RISK_WEIGHT * self.risk
                + CONFIDENCE_WEIGHT * cv
                - COST_WEIGHT * self.cost)


# ===================================================================
# Distance Cache (BFS)
# ===================================================================
class DistanceCache:
    def __init__(self, maxsize: int = 256) -> None:
        self._cache: Dict[Pos, Dict[Pos, int]] = {}
        self._maxsize = maxsize

    def dist(self, ms, start: Pos) -> Dict[Pos, int]:
        if start in self._cache:
            return self._cache[start]
        d = self._bfs(ms, start)
        if len(self._cache) >= self._maxsize:
            del self._cache[next(iter(self._cache))]
        self._cache[start] = d
        return d

    def precompute(self, ms, cells: Set[Pos]) -> None:
        for c in cells:
            if c not in self._cache and _valid(c, ms):
                self._cache[c] = self._bfs(ms, c)

    @staticmethod
    def _bfs(ms, start: Pos) -> Dict[Pos, int]:
        if not _valid(start, ms):
            return {}
        d: Dict[Pos, int] = {start: 0}
        q: deque[Pos] = deque([start])
        while q:
            cur = q.popleft()
            for m in MOVE_ORDER:
                nxt = _apply(cur, m)
                if nxt not in d and _valid(nxt, ms):
                    d[nxt] = d[cur] + 1
                    q.append(nxt)
        return d


# ===================================================================
# Topology Analyzer (for GhostAgent — 24127192)
# ===================================================================
class TopologyAnalyzer:
    def __init__(self) -> None:
        self.dead_ends:       Set[Pos] = set()
        self.junctions:       Set[Pos] = set()
        self.corridors:       Set[Pos] = set()
        self.core:            Set[Pos] = set()
        self.loop_set:        Set[Pos] = set()
        self.trap_depth:      Dict[Pos, int] = {}
        self.escape_capacity: Dict[Pos, int] = {}
        self.chokepoints:     Set[Pos] = set()
        self.tunnels:         List[Tuple[frozenset, Pos, Pos]] = []
        self.tunnel_cells:    Set[Pos] = set()
        self._open:           Set[Pos] = set()
        self._built = False

    def build(self, ms) -> None:
        if self._built:
            return
        h, w = _shape(ms)
        self._open = {(r, c) for r in range(h) for c in range(w) if _cell(ms, r, c) != 1}
        seeds: Set[Pos] = set()
        for p in self._open:
            deg = _exits(p, ms)
            if deg <= 1:    seeds.add(p)
            elif deg >= 3:  self.junctions.add(p)
            else:           self.corridors.add(p)
        self._propagate_dead_ends(ms, seeds)
        self.corridors -= self.dead_ends
        self._compute_core(ms)
        self._find_loops(ms)
        self._find_chokepoints()
        self._find_tunnels(ms)
        self._compute_escape_capacity(ms)
        self._built = True

    def _propagate_dead_ends(self, ms, seeds: Set[Pos]) -> None:
        self.trap_depth = {p: 0 for p in self._open}
        trimmed: Set[Pos] = set()
        degree = {p: _exits(p, ms) for p in self._open}
        q: deque[Pos] = deque(seeds)
        for p in seeds:
            trimmed.add(p); self.dead_ends.add(p); self.trap_depth[p] = 1
        while q:
            u = q.popleft()
            for m in MOVE_ORDER:
                v = _apply(u, m)
                if v not in self._open or v in trimmed:
                    continue
                degree[v] -= 1
                if degree[v] <= 1:
                    trimmed.add(v); self.dead_ends.add(v)
                    self.trap_depth[v] = self.trap_depth[u] + 1
                    q.append(v)

    def _compute_core(self, ms) -> None:
        active = set(self._open)
        changed = True
        while changed:
            changed = False
            rm = {p for p in active
                  if sum(1 for m in MOVE_ORDER if _apply(p, m) in active) <= 1}
            if rm:
                active -= rm; changed = True
        self.core = active

    def _find_loops(self, ms) -> None:
        if not self.core:
            return
        cycles: List[List[Pos]] = []
        vis: Set[Pos] = set()
        parent: Dict[Pos, Optional[Pos]] = {}
        for start in list(self.core):
            if start in vis:
                continue
            nbs = [n for m in MOVE_ORDER if (n := _apply(start, m)) in self.core]
            stack = [(start, None, iter(nbs))]
            vis.add(start); parent[start] = None
            while stack:
                u, p, it = stack[-1]
                try:
                    v = next(it)
                except StopIteration:
                    stack.pop(); continue
                if v not in vis:
                    vis.add(v); parent[v] = u
                    v_nbs = [n for m in MOVE_ORDER if (n := _apply(v, m)) in self.core]
                    stack.append((v, u, iter(v_nbs)))
                elif v != p:
                    cyc: List[Pos] = [v]
                    cur = u
                    while cur is not None and cur != v:
                        cyc.append(cur); cur = parent.get(cur)
                    if cur == v and len(cyc) >= 4:
                        cycles.append(cyc)
        self.loop_set = set(max(cycles, key=len)) if cycles else set()

    def _find_chokepoints(self) -> None:
        if not self._open:
            return
        disc: Dict[Pos, int] = {}; low: Dict[Pos, int] = {}
        parent: Dict[Pos, Optional[Pos]] = {}
        cc: Dict[Pos, int] = defaultdict(int)
        ap: Set[Pos] = set(); timer = [0]
        for sn in self._open:
            if sn in disc:
                continue
            parent[sn] = None; disc[sn] = low[sn] = timer[0]; timer[0] += 1
            nbrs = [_apply(sn, m) for m in MOVE_ORDER if _apply(sn, m) in self._open]
            stack = [(sn, iter(nbrs))]
            while stack:
                u, it = stack[-1]
                try:
                    v = next(it)
                except StopIteration:
                    stack.pop()
                    if stack:
                        p = stack[-1][0]
                        low[p] = min(low[p], low[u])
                        if parent[p] is None and cc[p] > 1: ap.add(p)
                        if parent[p] is not None and low[u] >= disc[p]: ap.add(p)
                    continue
                if v not in disc:
                    cc[u] += 1; parent[v] = u
                    disc[v] = low[v] = timer[0]; timer[0] += 1
                    v_nbrs = [_apply(v, m) for m in MOVE_ORDER if _apply(v, m) in self._open]
                    stack.append((v, iter(v_nbrs)))
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
        self.chokepoints = ap

    def _find_tunnels(self, ms) -> None:
        visited: Set[Pos] = set()
        for junc in self.junctions:
            for m in MOVE_ORDER:
                nxt = _apply(junc, m)
                if nxt not in self.corridors or nxt in visited:
                    continue
                path = [nxt]; visited.add(nxt)
                cur, prev = nxt, junc
                while True:
                    reached = found = False
                    for m2 in MOVE_ORDER:
                        cand = _apply(cur, m2)
                        if cand == prev:
                            continue
                        if cand in self.junctions:
                            self.tunnels.append((frozenset(path), junc, cand))
                            self.tunnel_cells |= set(path)
                            reached = True; break
                        if cand in self.corridors and cand not in visited:
                            visited.add(cand); path.append(cand)
                            prev, cur = cur, cand
                            found = True; break
                    if reached or not found:
                        break

    def _compute_escape_capacity(self, ms) -> None:
        for pos in self._open:
            safe = sum(1 for m in MOVE_ORDER
                       if _valid(_apply(pos, m), ms)
                       and _apply(pos, m) not in self.dead_ends)
            self.escape_capacity[pos] = safe


# ===================================================================
# Zone Navigator
# ===================================================================
class ZoneNavigator:
    """Divides the map into 5 overlapping zones (top, center, bottom,
    left, right) and provides waypoint navigation to switch zones
    frequently, always steering Ghost away from predicted Pacman zone.
    """

    def __init__(self, ms, topo: TopologyAnalyzer) -> None:
        h, w = _shape(ms)
        r3 = h // 3
        c3 = w // 3
        self._row_top    = r3
        self._row_bot    = h - r3
        self._col_left   = c3
        self._col_right  = w - c3

        self._zone_cells: Dict[str, Set[Pos]] = {
            "top": set(), "center": set(), "bottom": set(),
            "left": set(), "right": set(),
        }
        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1:
                    continue
                pos = (r, c)
                if r < self._row_top:
                    self._zone_cells["top"].add(pos)
                elif r >= self._row_bot:
                    self._zone_cells["bottom"].add(pos)
                else:
                    self._zone_cells["center"].add(pos)
                if c < self._col_left:
                    self._zone_cells["left"].add(pos)
                elif c >= self._col_right:
                    self._zone_cells["right"].add(pos)

        self._waypoints: Dict[str, Pos] = {}
        for zname, cells in self._zone_cells.items():
            juncs = [p for p in cells if p in topo.junctions]
            if juncs:
                if cells:
                    cr = sum(p[0] for p in cells) / len(cells)
                    cc = sum(p[1] for p in cells) / len(cells)
                    juncs.sort(key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))
                self._waypoints[zname] = juncs[0]
            elif cells:
                cr = sum(p[0] for p in cells) / len(cells)
                cc = sum(p[1] for p in cells) / len(cells)
                best = min(cells, key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))
                self._waypoints[zname] = best

        self._zone_cycle = ["top", "right", "bottom", "left", "center"]
        self._current_target_zone: Optional[str] = None
        self._steps_in_zone: int = 0
        self._last_zone: Optional[str] = None

        self._opposite: Dict[str, str] = {
            "top": "bottom", "bottom": "top",
            "left": "right", "right": "left",
            "center": "top",
        }

    def classify(self, pos: Pos) -> str:
        r, c = pos
        if r < self._row_top:
            return "top"
        if r >= self._row_bot:
            return "bottom"
        if c < self._col_left:
            return "left"
        if c >= self._col_right:
            return "right"
        return "center"

    def predict_pacman_zone(
        self, belief: Dict[Pos, float], pac_preds: List[Tuple[Pos, float]]
    ) -> str:
        zone_prob: Dict[str, float] = {z: 0.0 for z in ZONE_NAMES}
        for pos, w in belief.items():
            zone_prob[self.classify(pos)] += w
        for pos, w in pac_preds[:6]:
            zone_prob[self.classify(pos)] += w * 0.5
        if not any(zone_prob.values()):
            return self.classify(PACMAN_START)
        return max(zone_prob, key=lambda z: zone_prob[z])

    def select_target_zone(
        self, ghost: Pos, belief: Dict[Pos, float],
        pac_preds: List[Tuple[Pos, float]], step_number: int,
    ) -> str:
        my_zone = self.classify(ghost)
        pac_zone = self.predict_pacman_zone(belief, pac_preds)
        self._steps_in_zone += 1

        need_switch = (
            self._steps_in_zone >= ZONE_SWITCH_INTERVAL
            or self._current_target_zone is None
            or my_zone == pac_zone
        )

        if need_switch:
            opp = self._opposite.get(pac_zone, "top")
            if opp == my_zone:
                candidates = [z for z in self._zone_cycle
                              if z != pac_zone and z != my_zone]
                if candidates:
                    if self._last_zone in candidates:
                        candidates.remove(self._last_zone)
                    opp = candidates[0] if candidates else self._zone_cycle[0]
            self._last_zone = self._current_target_zone
            self._current_target_zone = opp
            self._steps_in_zone = 0

        return self._current_target_zone

    def get_waypoint(self, target_zone: str) -> Optional[Pos]:
        return self._waypoints.get(target_zone)

    def navigate_toward_zone(
        self, ghost: Pos, target_zone: str, legal: List[Move],
        ms, topo: TopologyAnalyzer, dc: DistanceCache,
        danger_t0: Dict[Pos, float],
        pac_preds: List[Tuple[Pos, float]],
    ) -> Optional[Move]:
        waypoint = self.get_waypoint(target_zone)
        if waypoint is None:
            return None
        if ghost == waypoint:
            return None
        path = astar(ms, ghost, waypoint)
        if not path:
            return None
        first_move = path[0]
        if first_move not in legal:
            return None
        nxt = _apply(ghost, first_move)
        if nxt in topo.dead_ends:
            return None
        if danger_t0.get(nxt, 0.0) >= FATAL_DANGER * 0.7:
            return None
        for pp, pw in pac_preds[:4]:
            if pw > 0.2 and _manhattan(nxt, pp) < CAPTURE_DISTANCE + 1:
                return None
        return first_move


# ===================================================================
# Offline Table
# ===================================================================
class OfflineTable:
    def __init__(self) -> None:
        self._table: Dict[Tuple[Pos, Tuple[int, int]], Move] = {}
        self._built = False

    def build(self, ms, topo: TopologyAnalyzer) -> None:
        if self._built:
            return
        key_cells = topo.core | topo.junctions | topo.loop_set
        for pos in key_cells:
            legal = _legal(pos, ms)
            if not legal:
                continue
            for qr in (-1, 0, 1):
                for qc in (-1, 0, 1):
                    if qr == 0 and qc == 0:
                        continue
                    best_m: Optional[Move] = None
                    best_s = float("-inf")
                    for m in legal:
                        nxt = _apply(pos, m)
                        sc = -(nxt[0] - pos[0]) * qr * 12.0 - (nxt[1] - pos[1]) * qc * 12.0
                        if nxt in topo.core:       sc += 18.0
                        if nxt in topo.loop_set:   sc += 14.0
                        if nxt in topo.junctions:  sc += 10.0
                        if nxt in topo.dead_ends:
                            sc -= 60.0 + topo.trap_depth.get(nxt, 1) * 6.0
                        sc += _exits(nxt, ms) * 3.5
                        if nxt in topo.tunnel_cells: sc -= 15.0
                        if sc > best_s:
                            best_s = sc; best_m = m
                    if best_m is not None:
                        self._table[(pos, (qr, qc))] = best_m
        self._built = True

    def lookup(self, ghost: Pos, pac: Pos) -> Optional[Move]:
        dr = pac[0] - ghost[0]; dc = pac[1] - ghost[1]
        qr = (1 if dr > 0 else (-1 if dr < 0 else 0))
        qc = (1 if dc > 0 else (-1 if dc < 0 else 0))
        return self._table.get((ghost, (qr, qc)))


# ===================================================================
# Pacman Tracker (Belief State for GhostAgent)
# ===================================================================
class PacmanTracker:
    """Maintains probability distribution over Pacman's position."""

    def __init__(self) -> None:
        self.belief: Dict[Pos, float] = {}
        self.last_seen: Optional[Pos] = None
        self.history: deque[Pos] = deque(maxlen=50)
        self.steps_invisible: int = 0
        self.belief = {PACMAN_START: 1.0}
        self.last_seen = PACMAN_START

    def update(self, enemy_pos, ms, speed: int = 2) -> Dict[Pos, float]:
        if enemy_pos is not None:
            pac = (int(enemy_pos[0]), int(enemy_pos[1]))
            self.belief = {pac: 1.0}
            self.last_seen = pac
            self.history.append(pac)
            self.steps_invisible = 0
            return self.belief

        self.steps_invisible += 1
        if not self.belief:
            return self.belief

        new_b: Dict[Pos, float] = {}
        for cell, prob in self.belief.items():
            reachable = _pacman_reach(cell, ms, speed)
            n = len(reachable)
            if n == 0:
                continue
            share = prob / n
            for nxt in reachable:
                new_b[nxt] = new_b.get(nxt, 0.0) + share

        total = sum(new_b.values())
        if total > 0:
            for k in new_b:
                new_b[k] /= total

        if len(new_b) > BELIEF_MAX_CELLS:
            ranked = sorted(new_b.items(), key=lambda x: -x[1])[:BELIEF_MAX_CELLS]
            total2 = sum(v for _, v in ranked)
            new_b = {k: v / total2 for k, v in ranked} if total2 > 0 else {}

        self.belief = new_b
        return self.belief

    @property
    def best_estimate(self) -> Optional[Pos]:
        if not self.belief:
            return self.last_seen
        return max(self.belief, key=lambda k: self.belief[k])


# ===================================================================
# Opponent Models
# ===================================================================
class _BaseModel:
    name: str = "base"
    def predict(self, pac: Pos, ghost: Pos, ms, topo, dc, **kw) -> Dict[Action, float]:
        return {}
    def update(self, state, action):
        pass


class MarkovOrder1(_BaseModel):
    name = "markov1"

    def __init__(self) -> None:
        self._trans: Dict[Tuple, Counter] = defaultdict(Counter)
        self._glob: Counter = Counter()

    def _state(self, ghost: Pos, pac: Pos, ms) -> Tuple:
        dr = _bucket(ghost[0] - pac[0]); dc = _bucket(ghost[1] - pac[1])
        db = min(9, _manhattan(ghost, pac) // 2)
        mvs = _legal(pac, ms); deg = len(mvs)
        if deg <= 1:    geo = 0
        elif deg >= 3:  geo = 3
        else:
            m1, m2 = mvs[0], mvs[1]
            geo = 1 if (m1.value[0] + m2.value[0] == 0 and m1.value[1] + m2.value[1] == 0) else 2
        return (dr, dc, db, geo)

    def observe(self, ghost: Pos, pac_prev: Pos, pac_cur: Pos, ms) -> None:
        state = self._state(ghost, pac_prev, ms)
        action = (pac_cur[0] - pac_prev[0], pac_cur[1] - pac_prev[1])
        self._trans[state][action] += 1
        self._glob[action] += 1

    def predict(self, pac, ghost, ms, topo=None, dc=None, **kw):
        state = self._state(ghost, pac, ms)
        counts = self._trans.get(state, self._glob)
        if not counts:
            legal = _legal(pac, ms)
            n = max(1, len(legal))
            return {(m.value[0], m.value[1]): 1.0 / n for m in legal}
        total = sum(counts.values())
        return {a: c / total for a, c in counts.items()}


class MarkovOrder2(_BaseModel):
    name = "markov2"

    def __init__(self) -> None:
        self._trans: Dict[Tuple, Counter] = defaultdict(Counter)

    def observe(self, ghost: Pos, pac_prev2: Pos, pac_prev: Pos, pac_cur: Pos, ms) -> None:
        prev_act = (pac_prev[0] - pac_prev2[0], pac_prev[1] - pac_prev2[1])
        dr = _bucket(ghost[0] - pac_prev[0]); dc = _bucket(ghost[1] - pac_prev[1])
        state = (prev_act, dr, dc)
        action = (pac_cur[0] - pac_prev[0], pac_cur[1] - pac_prev[1])
        self._trans[state][action] += 1

    def predict(self, pac, ghost, ms, topo=None, dc=None, **kw):
        prev_act = kw.get("prev_action", (0, 0))
        dr = _bucket(ghost[0] - pac[0]); dc_ = _bucket(ghost[1] - pac[1])
        state = (prev_act, dr, dc_)
        counts = self._trans.get(state)
        if not counts or sum(counts.values()) < 3:
            return {}
        total = sum(counts.values())
        return {a: c / total for a, c in counts.items()}


class ShortestPathModel(_BaseModel):
    name = "shortest_path"

    def predict(self, pac, ghost, ms, topo=None, dc=None, **kw):
        if dc is None:
            return {}
        gd = dc.dist(ms, ghost)
        legal = _legal(pac, ms)
        if not legal:
            return {}
        probs: Dict[Action, float] = {}
        total = 0.0
        for m in legal:
            nxt = _apply(pac, m)
            d = gd.get(nxt, _manhattan(nxt, ghost))
            w = max(0.01, 1.0 / max(1, d))
            act = (m.value[0], m.value[1])
            probs[act] = w
            total += w
        if total > 0:
            for k in probs:
                probs[k] /= total
        return probs


class InterceptionModel(_BaseModel):
    name = "interceptor"

    def predict(self, pac, ghost, ms, topo=None, dc=None, **kw):
        if dc is None or topo is None:
            return {}
        gd = dc.dist(ms, ghost)
        legal = _legal(pac, ms)
        if not legal:
            return {}
        probs: Dict[Action, float] = {}
        total = 0.0
        for m in legal:
            nxt = _apply(pac, m)
            act = (m.value[0], m.value[1])
            d = gd.get(nxt, _manhattan(nxt, ghost))
            score = max(0.01, 1.0 / max(1, d))
            if nxt in topo.junctions:   score *= 2.0
            if nxt in topo.chokepoints: score *= 2.5
            probs[act] = score
            total += score
        if total > 0:
            for k in probs:
                probs[k] /= total
        return probs


class RandomLegalModel(_BaseModel):
    name = "random"

    def predict(self, pac, ghost, ms, topo=None, dc=None, **kw):
        legal = _legal(pac, ms)
        n = max(1, len(legal))
        return {(m.value[0], m.value[1]): 1.0 / n for m in legal}


class AdversarialModel(_BaseModel):
    """Minimax 1-ply: Pacman chooses move that minimises Ghost's best escape."""
    name = "adversarial"

    def predict(self, pac, ghost, ms, topo=None, dc=None, **kw):
        pac_legal = _legal(pac, ms)
        ghost_legal = _legal(ghost, ms)
        if not pac_legal:
            return {}
        scores: Dict[Action, float] = {}
        for pm in pac_legal:
            pac_nxt = _apply(pac, pm)
            best_ghost_d = 0.0
            for gm in ghost_legal:
                g_nxt = _apply(ghost, gm)
                best_ghost_d = max(best_ghost_d, _manhattan(g_nxt, pac_nxt))
            scores[(pm.value[0], pm.value[1])] = max(0.01, 1.0 / max(1, best_ghost_d))
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total
        return scores


# ===================================================================
# Opponent Model Ensemble
# ===================================================================
class OpponentModelEnsemble:
    """Combines 6 opponent models with adaptive weight updates."""

    def __init__(self) -> None:
        self.markov1 = MarkovOrder1()
        self.markov2 = MarkovOrder2()
        self.sp = ShortestPathModel()
        self.intercept = InterceptionModel()
        self.rand = RandomLegalModel()
        self.adv = AdversarialModel()
        self._models = [self.markov1, self.markov2, self.sp,
                        self.intercept, self.rand, self.adv]
        self.weights: Dict[str, float] = {
            "markov1": 1.0, "markov2": 1.0, "shortest_path": 1.0,
            "interceptor": 1.0, "random": 0.5, "adversarial": 0.8,
        }
        self._last_predictions: Dict[str, Dict[Action, float]] = {}
        self.style: str = "UNKNOWN"

    def update_if_observed(self, tracker: PacmanTracker, ghost: Pos, ms) -> None:
        hist = tracker.history
        if len(hist) < 2:
            return
        pac_cur = hist[-1]; pac_prev = hist[-2]
        actual = (pac_cur[0] - pac_prev[0], pac_cur[1] - pac_prev[1])
        prev_ghost = ghost
        self.markov1.observe(prev_ghost, pac_prev, pac_cur, ms)
        if len(hist) >= 3:
            self.markov2.observe(prev_ghost, hist[-3], pac_prev, pac_cur, ms)
        if self._last_predictions:
            for mdl in self._models:
                pred = self._last_predictions.get(mdl.name, {})
                prob = pred.get(actual, 0.01)
                error = -math.log(max(prob, 0.001))
                self.weights[mdl.name] *= math.exp(-ENSEMBLE_LR * error)
            self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            n = len(self.weights)
            self.weights = {k: 1.0 / n for k in self.weights}
            return
        for k in self.weights:
            self.weights[k] = max(MIN_MODEL_WEIGHT, self.weights[k] / total)
        total2 = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total2

    def predict(self, pac: Pos, ghost: Pos, ms, topo, dc,
                prev_action: Action = (0, 0)) -> Dict[Action, float]:
        combined: Dict[Action, float] = {}
        self._last_predictions.clear()
        for mdl in self._models:
            dist = mdl.predict(pac, ghost, ms, topo=topo, dc=dc,
                               prev_action=prev_action)
            self._last_predictions[mdl.name] = dist
            w = self.weights.get(mdl.name, 0.1)
            for act, prob in dist.items():
                combined[act] = combined.get(act, 0.0) + w * prob
        total = sum(combined.values())
        if total > 0:
            for k in combined:
                combined[k] /= total
        return combined

    def predict_positions_2step(self, ghost: Pos, pac: Pos, ms, topo, dc,
                                 speed: int = 2,
                                 prev_action: Action = (0, 0)
                                 ) -> List[Tuple[Pos, float]]:
        dist1 = self.predict(pac, ghost, ms, topo, dc, prev_action)
        positions: Dict[Pos, float] = {}
        step1: List[Tuple[Pos, float, Action]] = []
        for act, prob in dist1.items():
            if prob < 0.04:
                continue
            pos1 = (pac[0] + act[0], pac[1] + act[1])
            if _valid(pos1, ms):
                positions[pos1] = max(positions.get(pos1, 0.0), prob)
                step1.append((pos1, prob, act))
                cur = pos1
                for _ in range(1, speed):
                    ext = (cur[0] + act[0], cur[1] + act[1])
                    if _valid(ext, ms):
                        positions[ext] = max(positions.get(ext, 0.0), prob * 0.6)
                        cur = ext
                    else:
                        break
        for pos1, prob1, act1 in step1:
            if prob1 < 0.07:
                continue
            d2 = self.predict(pos1, ghost, ms, topo, dc, act1)
            for act2, prob2 in d2.items():
                pos2 = (pos1[0] + act2[0], pos1[1] + act2[1])
                if _valid(pos2, ms):
                    c = prob1 * prob2 * 0.55
                    if c >= 0.02:
                        positions[pos2] = max(positions.get(pos2, 0.0), c)
        for m in MOVE_ORDER:
            nxt = _apply(pac, m)
            if _valid(nxt, ms):
                closer = _manhattan(nxt, ghost) < _manhattan(pac, ghost)
                ch = 0.45 if closer else 0.15
                positions[nxt] = max(positions.get(nxt, 0.0), ch)
        positions[pac] = max(positions.get(pac, 0.0), 0.10)
        ranked = sorted(positions.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:MARKOV_LIMIT]


# ===================================================================
# Risk Engine
# ===================================================================
class RiskEngine:
    def __init__(self, topo: TopologyAnalyzer, dc: DistanceCache) -> None:
        self._topo = topo
        self._dc = dc

    def time_expanded_danger(
        self, pac_preds: List[Tuple[Pos, float]], ms, speed: int = 2,
        horizon: int = DANGER_HORIZON,
    ) -> List[Dict[Pos, float]]:
        danger: List[Dict[Pos, float]] = [{} for _ in range(horizon + 1)]
        current_dist = {p: w for p, w in pac_preds}
        for t in range(horizon + 1):
            for pac_pos, prob in current_dist.items():
                danger[t][pac_pos] = danger[t].get(pac_pos, 0.0) + prob * 100.0
                for m in MOVE_ORDER:
                    cur = pac_pos
                    for s in range(speed):
                        nxt = _apply(cur, m)
                        if not _valid(nxt, ms):
                            break
                        d_score = prob * (65.0 / (s + 1)) * (0.85 ** t)
                        danger[t][nxt] = danger[t].get(nxt, 0.0) + d_score
                        cur = nxt
            new_dist: Dict[Pos, float] = {}
            for cell, prob in current_dist.items():
                reachable = _pacman_reach(cell, ms, speed)
                n = max(1, len(reachable))
                for r in reachable:
                    new_dist[r] = new_dist.get(r, 0.0) + prob / n
            if len(new_dist) > 30:
                ranked = sorted(new_dist.items(), key=lambda x: -x[1])[:30]
                total = sum(v for _, v in ranked)
                new_dist = {k: v / total for k, v in ranked} if total > 0 else {}
            current_dist = new_dist
        for tc, ent1, ent2 in self._topo.tunnels:
            ed = max(danger[0].get(ent1, 0.0), danger[0].get(ent2, 0.0))
            if ed > 25.0:
                for c in tc:
                    danger[0][c] = danger[0].get(c, 0.0) + ed * 0.4
        return danger

    def survival_margin(self, ghost: Pos, belief: Dict[Pos, float],
                         ms, speed: int = 2) -> float:
        exits = list(self._topo.core | self._topo.loop_set | self._topo.junctions)
        if not exits:
            exits = [p for p in self._topo._open if _exits(p, ms) >= 3]
        if not exits or not belief:
            return 10.0
        gd = self._dc.dist(ms, ghost)
        best_margin = -INF
        for ex in exits[:20]:
            ghost_time = gd.get(ex, INF)
            if ghost_time == INF:
                continue
            worst_pac = INF
            for pac_cell, prob in belief.items():
                if prob < 0.05:
                    continue
                pd = self._dc.dist(ms, pac_cell)
                pac_time = math.ceil(pd.get(ex, INF) / max(1, speed))
                worst_pac = min(worst_pac, pac_time)
            margin = worst_pac - ghost_time
            best_margin = max(best_margin, margin)
        return best_margin

    def is_viable(self, pos: Pos, belief: Dict[Pos, float], ms,
                   speed: int = 2) -> bool:
        ec = self._topo.escape_capacity.get(pos, 0)
        if ec >= 3:
            return True
        if pos in self._topo.core or pos in self._topo.loop_set:
            return True
        if not belief:
            return True
        best_pac = min(belief, key=lambda p: _manhattan(p, pos))
        margin = self.survival_margin(pos, belief, ms, speed)
        if margin >= 1:
            return True
        if _manhattan(pos, best_pac) >= 5 and ec >= 2:
            return True
        return False


# ===================================================================
# Safety Shield (4-level)
# ===================================================================
class SafetyShield:
    def __init__(self, topo: TopologyAnalyzer, dc: DistanceCache,
                 risk: RiskEngine) -> None:
        self._topo = topo
        self._dc = dc
        self._risk = risk

    def filter(self, proposals: List[Proposal], ghost: Pos,
               belief: Dict[Pos, float], danger_t0: Dict[Pos, float],
               ms, speed: int = 2) -> List[Proposal]:
        safe: List[Proposal] = []
        for p in proposals:
            nxt = _apply(ghost, p.action)
            if not _valid(nxt, ms) or p.action not in _legal(ghost, ms):
                continue
            if self._can_capture_next(nxt, belief, ms, speed):
                continue
            if danger_t0.get(nxt, 0.0) >= FATAL_DANGER:
                continue
            if not self._risk.is_viable(nxt, belief, ms, speed):
                continue
            safe.append(p)
        if safe:
            return safe
        return self._relaxed(proposals, ghost, belief, ms)

    def _can_capture_next(self, nxt: Pos, belief: Dict[Pos, float],
                           ms, speed: int) -> bool:
        for pac_pos, prob in belief.items():
            if prob < 0.15:
                continue
            reach = _pacman_reach(pac_pos, ms, speed)
            for r in reach:
                if _manhattan(nxt, r) < CAPTURE_DISTANCE:
                    return True
        return False

    def _relaxed(self, proposals: List[Proposal], ghost: Pos,
                  belief: Dict[Pos, float], ms) -> List[Proposal]:
        if not proposals:
            return []
        def sort_key(p):
            nxt = _apply(ghost, p.action)
            if not _valid(nxt, ms):
                return -INF
            min_d = min((_manhattan(nxt, pc) for pc in belief), default=INF)
            return min_d
        return sorted(proposals, key=sort_key, reverse=True)


# ===================================================================
# Monte Carlo Rollout V3 (CVaR scoring)
# ===================================================================
class MCRolloutV3:
    def __init__(self, topo: TopologyAnalyzer, dc: DistanceCache,
                 ensemble: OpponentModelEnsemble) -> None:
        self._topo = topo; self._dc = dc; self._ens = ensemble

    def _ghost_policy(self, ghost, pac, ms, prev, scenario):
        moves = _legal(ghost, ms)
        if not moves:
            return ghost, Move.STAY
        pd = self._dc.dist(ms, pac)
        best_s = float("-inf"); best = (ghost, Move.STAY)
        for m in moves:
            nxt = _apply(ghost, m)
            d = pd.get(nxt, _manhattan(nxt, pac))
            sc = d * 3.5 + _exits(nxt, ms) * 2.5
            if nxt in self._topo.core:       sc += 10.0
            if nxt in self._topo.loop_set:   sc += 9.0
            if nxt in self._topo.junctions:  sc += 7.0
            if nxt in self._topo.dead_ends:
                sc -= 45.0 + self._topo.trap_depth.get(nxt, 1) * 5.0
            if nxt in self._topo.tunnel_cells: sc -= 8.0
            if prev is not None and nxt == prev: sc -= 80.0
            if prev is not None:
                dr = nxt[0] - ghost[0]; dc = nxt[1] - ghost[1]
                prev_dr = ghost[0] - prev[0]; prev_dc = ghost[1] - prev[1]
                if dr + prev_dr == 0 and dc + prev_dc == 0:
                    sc -= 50.0
            if _manhattan(nxt, pac) < CAPTURE_DISTANCE: sc -= MC_CAPTURE_PENALTY
            if scenario % 4 == 1 and nxt in self._topo.loop_set: sc += 7.0
            elif scenario % 4 == 2 and nxt in self._topo.junctions: sc += 6.0
            elif scenario % 4 == 3 and nxt in self._topo.core: sc += 5.0
            if sc > best_s:
                best_s = sc; best = (nxt, m)
        return best

    def _pac_response(self, pac, ghost, ms, scenario, speed=2):
        pred = self._ens.predict(pac, ghost, ms, self._topo, self._dc)
        if pred:
            ranked = sorted(pred.items(), key=lambda x: -x[1])
            idx = min(scenario % 3, len(ranked) - 1)
            act, _ = ranked[idx]
            pos = (pac[0] + act[0], pac[1] + act[1])
            if _valid(pos, ms):
                cur = pos
                for _ in range(1, speed):
                    ext = (cur[0] + act[0], cur[1] + act[1])
                    if _valid(ext, ms): cur = ext
                    else: break
                return cur
        gd = self._dc.dist(ms, ghost)
        reach = _pacman_reach(pac, ms, speed)
        reach.sort(key=lambda p: gd.get(p, INF))
        branch = min(scenario % 3, len(reach) - 1)
        return reach[branch] if reach else pac

    def _leaf(self, ghost, pac, ms):
        pd = self._dc.dist(ms, pac)
        d = pd.get(ghost, _manhattan(ghost, pac))
        sc = d * 14.0 + _exits(ghost, ms) * 3.5
        if ghost in self._topo.core:       sc += 14.0
        if ghost in self._topo.loop_set:   sc += 11.0
        if ghost in self._topo.junctions:  sc += 8.0
        if ghost in self._topo.dead_ends:
            sc -= 40.0 + self._topo.trap_depth.get(ghost, 1) * 6.0
        if ghost in self._topo.tunnel_cells: sc -= 10.0
        return sc

    def rollout(self, g0, first_move, pac0, ms, sn, scenario, speed=2, t0=0.0):
        ghost = _apply(g0, first_move)
        if not _valid(ghost, ms):
            return -MC_CAPTURE_PENALTY
        pac = self._pac_response(pac0, ghost, ms, scenario, speed)
        if _manhattan(ghost, pac) < CAPTURE_DISTANCE:
            return -MC_CAPTURE_PENALTY
        total = 0.0; survived = 0; prev = g0
        for depth in range(MC_DEPTH):
            if time.time() - t0 > TIME_BUDGET * 0.80:
                break
            ng, _ = self._ghost_policy(ghost, pac, ms, prev, scenario + depth)
            np_ = self._pac_response(pac, ng, ms, scenario + depth, speed)
            if _manhattan(ng, np_) < CAPTURE_DISTANCE:
                total -= MC_CAPTURE_PENALTY - depth * 6000.0; break
            total += self._leaf(ng, np_, ms) + depth * 200.0
            survived += 1; prev = ghost; ghost = ng; pac = np_
        return total + survived * MC_SURVIVE_BONUS

    def evaluate_move_cvar(self, ghost, move, pac_preds, ms, sn, t0,
                            speed=2) -> Optional[float]:
        nxt = _apply(ghost, move)
        if not _valid(nxt, ms):
            return None
        returns: List[float] = []
        hyps = pac_preds[:5]
        rp = max(1, MC_ROLLOUTS // max(1, len(hyps)))
        for hi, (pac, hw) in enumerate(hyps):
            if time.time() - t0 > TIME_BUDGET * 0.75:
                break
            for s in range(rp):
                if time.time() - t0 > TIME_BUDGET * 0.75:
                    break
                sid = sn * 13 + hi * 5 + s
                sc = self.rollout(ghost, move, pac, ms, sn, sid, speed, t0)
                returns.append(sc * hw)
        if not returns:
            return None
        returns.sort()
        k = max(1, int(len(returns) * CVAR_ALPHA))
        cvar = sum(returns[:k]) / k
        mean_r = sum(returns) / len(returns)
        return MEAN_WEIGHT * mean_r + CVAR_WEIGHT * cvar


# ===================================================================
# Alpha-Beta Search (TT + move ordering)
# ===================================================================
class AlphaBetaSearch:
    def __init__(self, topo: TopologyAnalyzer, dc: DistanceCache,
                 ensemble: OpponentModelEnsemble) -> None:
        self._topo = topo; self._dc = dc; self._ens = ensemble
        self._tt: Dict[Tuple, Tuple[float, int, str]] = {}
        self._tt_max = 40_000

    def _eval(self, g, p, ms):
        if _manhattan(g, p) < CAPTURE_DISTANCE:
            return -100_000.0
        pd = self._dc.dist(ms, p)
        d = pd.get(g, _manhattan(g, p))
        sc = d * 16.0 + _exits(g, ms) * 4.5
        if g in self._topo.core:       sc += 20.0
        if g in self._topo.loop_set:   sc += 15.0
        if g in self._topo.junctions:  sc += 10.0
        if g in self._topo.dead_ends:
            sc -= 55.0 + self._topo.trap_depth.get(g, 1) * 7.0
        if g in self._topo.tunnel_cells: sc -= 12.0
        if g in self._topo.chokepoints: sc += 8.0
        return sc

    def search(self, ghost, pac, ms, max_depth, t0, speed=2,
               history=None) -> Tuple[float, Optional[Move]]:
        history = history or []
        self._tt.clear()

        def ab(g, p, depth, alpha, beta, is_ghost):
            if time.time() - t0 > TIME_BUDGET * 0.70:
                return self._eval(g, p, ms)
            if _manhattan(g, p) < CAPTURE_DISTANCE:
                return -100_000.0 + depth * 2000.0
            if depth <= 0:
                return self._eval(g, p, ms)
            key = (g, p, depth, is_ghost)
            hit = self._tt.get(key)
            if hit is not None and hit[1] >= depth:
                return hit[0]
            if is_ghost:
                best = float("-inf")
                moves = _legal(g, ms)
                pd = self._dc.dist(ms, p)
                moves.sort(key=lambda m: (-pd.get(_apply(g, m), INF), _move_key(m)))
                tried = False
                for m in moves:
                    ng = _apply(g, m)
                    if ng in history[-4:]:
                        continue
                    tried = True
                    v = ab(ng, p, depth - 1, alpha, beta, False)
                    best = max(best, v); alpha = max(alpha, best)
                    if beta <= alpha: break
                if not tried:
                    for m in moves:
                        v = ab(_apply(g, m), p, depth - 1, alpha, beta, False)
                        best = max(best, v); alpha = max(alpha, best)
                        if beta <= alpha: break
            else:
                best = float("inf")
                pred = self._ens.predict(p, g, ms, self._topo, self._dc)
                pac_opts: List[Pos] = []
                for act, _ in sorted(pred.items(), key=lambda x: -x[1])[:3]:
                    pos = (p[0] + act[0], p[1] + act[1])
                    if _valid(pos, ms):
                        pac_opts.append(pos)
                for pos in _pacman_reach(p, ms, speed):
                    if pos not in pac_opts:
                        pac_opts.append(pos)
                gd = self._dc.dist(ms, g)
                pac_opts.sort(key=lambda pos: (gd.get(pos, INF), pos))
                for np_ in pac_opts[:5]:
                    v = ab(g, np_, depth - 1, alpha, beta, True)
                    best = min(best, v); beta = min(beta, best)
                    if beta <= alpha: break
            if len(self._tt) < self._tt_max:
                self._tt[key] = (best, depth, "exact")
            return best

        best_move = None; best_score = float("-inf")
        moves = _legal(ghost, ms)
        if not moves:
            return float("-inf"), None
        pd = self._dc.dist(ms, pac)
        moves.sort(key=lambda m: (-pd.get(_apply(ghost, m), INF), _move_key(m)))
        for m in moves:
            ng = _apply(ghost, m)
            if _manhattan(ng, pac) < CAPTURE_DISTANCE:
                continue
            v = ab(ng, pac, max_depth - 1, float("-inf"), float("inf"), False)
            if v > best_score:
                best_score = v; best_move = m
        return best_score, best_move


# ===================================================================
# Anti-Loop
# ===================================================================
class AntiLoop:
    def __init__(self) -> None:
        self.visit_count: Dict[Pos, int] = defaultdict(int)
        self.edge_count: Dict[Tuple[Pos, Pos], int] = defaultdict(int)
        self.recent: deque[Pos] = deque(maxlen=12)

    def record(self, pos: Pos) -> None:
        self.visit_count[pos] += 1
        if self.recent:
            self.edge_count[(self.recent[-1], pos)] += 1
        self.recent.append(pos)

    def penalty(self, nxt: Pos, cur: Pos, pac_near: bool,
                topo: TopologyAnalyzer) -> float:
        if pac_near and nxt in topo.loop_set:
            return self.visit_count[nxt] * 1.5
        pen = self.visit_count[nxt] * 6.0
        pen += self.edge_count[(cur, nxt)] * 12.0

        if self.recent:
            recent_list = list(self.recent)
            ban_window = recent_list[-RECENT_CELL_BAN:]
            if nxt in ban_window:
                recency_index = list(reversed(ban_window)).index(nxt)
                pen += 200.0 + recency_index * 40.0

        if len(self.recent) >= 4:
            r = list(self.recent)
            if nxt == r[-2] and cur == r[-1]:
                pen += 80.0
            if len(r) >= 6 and r[-6:-3] == r[-3:]:
                pen += 120.0
            if len(r) >= 4 and nxt == r[-3] and cur == r[-2] and r[-1] == r[-3]:
                pen += 150.0
        return pen


# ===================================================================
# Pacman Style Classifier
# ===================================================================
class PacmanStyleClassifier:
    def __init__(self) -> None:
        self._dr_sum = 0.0
        self._dr_count = 0
        self._reversals = 0
        self._choke_moves = 0
        self._total_moves = 0
        self.style = "UNKNOWN"

    def update(self, pac_prev: Pos, pac_cur: Pos, ghost: Pos,
               topo: TopologyAnalyzer) -> None:
        self._total_moves += 1
        d_prev = _manhattan(pac_prev, ghost)
        d_cur = _manhattan(pac_cur, ghost)
        self._dr_sum += (1.0 if d_cur < d_prev else 0.0)
        self._dr_count += 1
        if pac_cur in topo.chokepoints or pac_cur in topo.junctions:
            self._choke_moves += 1

    def record_reversal(self) -> None:
        self._reversals += 1

    def classify(self) -> str:
        if self._total_moves < 8:
            self.style = "UNKNOWN"; return self.style
        dr_rate = self._dr_sum / max(1, self._dr_count)
        choke_rate = self._choke_moves / max(1, self._total_moves)
        if dr_rate > 0.80:
            self.style = "SHORTEST_PATH_CHASER"
        elif choke_rate > 0.55:
            self.style = "INTERCEPTOR"
        elif dr_rate < 0.35:
            self.style = "RANDOM_EXPLORER"
        else:
            self.style = "GREEDY_CHASER"
        return self.style


# ===================================================================
# Policies (5 policies → Proposals)
# ===================================================================
class _PolicyBase:
    name: str = "base"
    def propose(self, ctx) -> Optional[Proposal]:
        return None


class HybridPolicy(_PolicyBase):
    """Layer 0: MC + Markov + table. For close combat."""
    name = "hybrid"

    def __init__(self, mc: MCRolloutV3, ot: OfflineTable) -> None:
        self._mc = mc; self._ot = ot

    def propose(self, ctx) -> Optional[Proposal]:
        if ctx["pac"] is None:
            return None
        if time.time() - ctx["t0"] > TIME_BUDGET * 0.55:
            return None
        me = ctx["me"]; pac = ctx["pac"]; ms = ctx["ms"]
        legal = ctx["legal"]; preds = ctx["pac_preds"]
        sn = ctx["step"]; t0 = ctx["t0"]; speed = ctx["speed"]
        danger = ctx["danger_t0"]; topo = ctx["topo"]
        anti = ctx["anti"]

        candidates = [m for m in legal if _apply(me, m) not in topo.dead_ends] or legal
        best_m = None; best_sc = float("-inf")
        for m in candidates:
            if time.time() - t0 > TIME_BUDGET * 0.68:
                break
            nxt = _apply(me, m)
            mc_sc = self._mc.evaluate_move_cvar(me, m, preds, ms, sn, t0, speed)
            if mc_sc is None:
                continue
            table_v = 0.0
            tm = self._ot.lookup(nxt, pac)
            if tm is not None:
                table_v = 15.0
            flee = _manhattan(nxt, pac) * 2.5
            danger_v = danger.get(nxt, 0.0)
            loop_pen = anti.penalty(nxt, me, True, topo)
            combined = (0.35 * mc_sc + 0.20 * table_v + 0.20 * flee
                        - 0.30 * danger_v - 0.08 * loop_pen)
            if combined > best_sc:
                best_sc = combined; best_m = m
        if best_m is None:
            return None
        nxt = _apply(me, best_m)
        return Proposal(best_m, best_sc, danger.get(nxt, 0.0),
                        "high", self.name, cost=0.8, reason="mc+table+flee")


class TablePolicy(_PolicyBase):
    """Layer 1: Offline table + safety check."""
    name = "table"

    def __init__(self, ot: OfflineTable) -> None:
        self._ot = ot

    def propose(self, ctx) -> Optional[Proposal]:
        if ctx["pac"] is None:
            return None
        me = ctx["me"]; pac = ctx["pac"]; ms = ctx["ms"]
        legal = ctx["legal"]; danger = ctx["danger_t0"]; topo = ctx["topo"]
        m = self._ot.lookup(me, pac)
        if m is None or m not in legal:
            return None
        nxt = _apply(me, m)
        d = danger.get(nxt, 0.0)
        if d > 80.0 or nxt in topo.dead_ends:
            return None
        if _manhattan(nxt, pac) < CAPTURE_DISTANCE:
            return None
        value = _manhattan(nxt, pac) * 3.0 + _exits(nxt, ms) * 2.0
        return Proposal(m, value, d, "medium", self.name, cost=0.05)


class AlphaBetaPolicy(_PolicyBase):
    """Layer 2: Iterative deepening alpha-beta."""
    name = "alpha_beta"

    def __init__(self, ab: AlphaBetaSearch) -> None:
        self._ab = ab

    def propose(self, ctx) -> Optional[Proposal]:
        if ctx["pac"] is None:
            return None
        if time.time() - ctx["t0"] > TIME_BUDGET * 0.40:
            return None
        me = ctx["me"]; pac = ctx["pac"]; ms = ctx["ms"]
        t0 = ctx["t0"]; speed = ctx["speed"]
        history = list(ctx["ghost_hist"]); danger = ctx["danger_t0"]
        best_overall = None
        for depth in range(2, AB_MAX_DEPTH + 1, 2):
            if time.time() - t0 > TIME_BUDGET * 0.60:
                break
            sc, mv = self._ab.search(me, pac, ms, depth, t0, speed, history)
            if mv is not None:
                best_overall = (sc, mv)
            if sc is not None and sc < -50_000:
                break
        if best_overall is None:
            return None
        score, move = best_overall
        nxt = _apply(me, move)
        return Proposal(move, score, danger.get(nxt, 0.0),
                        "high", self.name, cost=0.5)


class OnePlyPolicy(_PolicyBase):
    """Layer 3: Quick one-step evaluation with directional momentum."""
    name = "one_ply"

    def propose(self, ctx) -> Optional[Proposal]:
        me = ctx["me"]; ms = ctx["ms"]; legal = ctx["legal"]
        danger = ctx["danger_t0"]; topo = ctx["topo"]
        preds = ctx["pac_preds"]; dc = ctx["dc"]
        anti = ctx["anti"]; ghost_hist = ctx["ghost_hist"]
        last_move = ctx["last_move"]
        current_dir = ctx.get("current_dir")
        dir_streak = ctx.get("dir_streak", 0)
        pac_near = ctx["pac"] is not None and _manhattan(ctx["pac"], me) <= 10

        best_m = None; best_sc = float("-inf")
        recent = set(list(ghost_hist)[-HISTORY_LEN:])
        for m in legal:
            nxt = _apply(me, m)
            sc = 0.0
            for pp, pw in preds[:6]:
                pd = dc.dist(ms, pp)
                d = pd.get(nxt, _manhattan(nxt, pp))
                sc += pw * d * 14.0
                sc -= pw * max(0, 7 - d) * 6.0
                reach = _pacman_reach(pp, ms, ctx["speed"])
                gap = min((_manhattan(nxt, r) for r in reach), default=INF)
                if gap < CAPTURE_DISTANCE:
                    sc -= pw * MC_CAPTURE_PENALTY * 0.4
                elif gap <= 2:
                    sc -= pw * 500.0
            sc -= danger.get(nxt, 0.0) * 0.8
            sc += _exits(nxt, ms) * 5.5
            if nxt in topo.core:       sc += 24.0
            if nxt in topo.loop_set:   sc += 20.0
            if nxt in topo.junctions:  sc += 14.0
            if nxt in topo.chokepoints: sc += 6.0
            if nxt in topo.dead_ends:
                sc -= 75.0 + topo.trap_depth.get(nxt, 1) * 9.0
            if nxt in topo.tunnel_cells: sc -= 18.0
            h, w = _shape(ms)
            ed = min(nxt[0], nxt[1], h - 1 - nxt[0], w - 1 - nxt[1])
            if ed <= 1 and _exits(nxt, ms) <= 2: sc -= 30.0
            sc -= anti.penalty(nxt, me, pac_near, topo)
            if nxt in recent: sc -= 26.0

            # Directional Momentum
            if last_move is not None and m == last_move:
                sc += MOMENTUM_BONUS
            if (last_move is not None
                    and m.value[0] + last_move.value[0] == 0
                    and m.value[1] + last_move.value[1] == 0):
                sc -= REVERSAL_PENALTY
            if (last_move is not None
                    and m != last_move
                    and not (m.value[0] + last_move.value[0] == 0
                             and m.value[1] + last_move.value[1] == 0)):
                sc -= TURN_PENALTY_90
            if current_dir is not None and m == current_dir:
                sc += min(dir_streak, 10) * 8.0

            if sc > best_sc:
                best_sc = sc; best_m = m
        if best_m is None:
            return None
        nxt = _apply(me, best_m)
        return Proposal(best_m, best_sc, danger.get(nxt, 0.0),
                        "medium", self.name, cost=0.02)


class GreedyPolicy(_PolicyBase):
    """Layer 4: Distance + danger + escape capacity + directional momentum (fallback)."""
    name = "greedy"

    def propose(self, ctx) -> Optional[Proposal]:
        me = ctx["me"]; ms = ctx["ms"]; legal = ctx["legal"]
        preds = ctx["pac_preds"]; danger = ctx["danger_t0"]; topo = ctx["topo"]
        anti = ctx["anti"]; last_move = ctx["last_move"]
        current_dir = ctx.get("current_dir"); dir_streak = ctx.get("dir_streak", 0)
        best_m = legal[0]; best_sc = float("-inf")
        for m in legal:
            nxt = _apply(me, m)
            min_d = min((_manhattan(nxt, p) for p, _ in preds), default=INF)
            sc = min_d * 10.0 - danger.get(nxt, 0.0)
            ec = topo.escape_capacity.get(nxt, 0)
            sc += ec * 6.0
            if nxt in topo.dead_ends:   sc -= 100.0
            if nxt in topo.tunnel_cells: sc -= 20.0
            sc -= anti.penalty(nxt, me, False, topo) * 0.5

            if last_move is not None and m == last_move:
                sc += MOMENTUM_BONUS * 0.5
            if (last_move is not None
                    and m.value[0] + last_move.value[0] == 0
                    and m.value[1] + last_move.value[1] == 0):
                sc -= REVERSAL_PENALTY * 0.5
            if (last_move is not None
                    and m != last_move
                    and not (m.value[0] + last_move.value[0] == 0
                             and m.value[1] + last_move.value[1] == 0)):
                sc -= TURN_PENALTY_90 * 0.5

            if sc > best_sc:
                best_sc = sc; best_m = m
        nxt = _apply(me, best_m)
        return Proposal(best_m, best_sc, danger.get(nxt, 0.0),
                        "low", self.name, cost=0.01)


# ===================================================================
# Policy Portfolio + Arbitrator
# ===================================================================
class PolicyPortfolio:
    def __init__(self, hybrid: HybridPolicy, table: TablePolicy,
                 ab_pol: AlphaBetaPolicy, oneply: OnePlyPolicy,
                 greedy: GreedyPolicy) -> None:
        self._hybrid = hybrid; self._table = table
        self._ab = ab_pol; self._oneply = oneply; self._greedy = greedy

    def select_policies(self, mode: str, style: str) -> List[_PolicyBase]:
        if mode == "emergency":
            return [self._greedy, self._oneply]
        if mode == "cheap":
            return [self._table, self._greedy]
        if mode == "tunnel_escape":
            return [self._oneply, self._greedy]
        if mode == "combat":
            if style == "INTERCEPTOR":
                return [self._ab, self._hybrid, self._oneply, self._greedy]
            return [self._hybrid, self._ab, self._oneply, self._greedy]
        return [self._table, self._oneply, self._ab, self._greedy]

    @staticmethod
    def arbitrate(proposals: List[Proposal]) -> Optional[Move]:
        if not proposals:
            return None
        best = max(proposals, key=lambda p: p.score())
        return best.action


# ===================================================================
# Budget Controller
# ===================================================================
class BudgetController:
    @staticmethod
    def select_mode(ghost: Pos, belief: Dict[Pos, float],
                    danger_t0: Dict[Pos, float],
                    topo: TopologyAnalyzer, dc: DistanceCache,
                    ms) -> str:
        d0 = danger_t0.get(ghost, 0.0)
        if d0 >= FATAL_DANGER * 0.8:
            return "emergency"
        if ghost in topo.tunnel_cells:
            for tc, e1, e2 in topo.tunnels:
                if ghost in tc:
                    for pac_pos in belief:
                        if _manhattan(pac_pos, e1) <= 4 or _manhattan(pac_pos, e2) <= 4:
                            return "tunnel_escape"
        if belief:
            best_pac = min(belief, key=lambda p: _manhattan(p, ghost))
            dist = _manhattan(best_pac, ghost)
            if dist <= PANIC_DISTANCE:
                return "combat"
            if dist > 15:
                return "cheap"
        return "normal"


# ===================================================================
# Diagnostics
# ===================================================================
class Diagnostics:
    def __init__(self) -> None:
        self.policy_failures: Dict[str, int] = defaultdict(int)
        self.policy_usage: Dict[str, int] = defaultdict(int)
        self.safety_rejections: int = 0
        self.timeouts: int = 0

    def record_failure(self, name: str) -> None:
        self.policy_failures[name] += 1

    def record_usage(self, name: str) -> None:
        self.policy_usage[name] += 1

    def record_rejection(self) -> None:
        self.safety_rejections += 1


# ===================================================================
# MAIN GHOST AGENT — Blind Multi-Layer (from 24127192)
# ===================================================================
class GhostAgent(BaseGhostAgent):
    """Blind Multi-Layer Ghost Agent with Policy Portfolio.

    Pipeline per step:
      1. Update memory_map from observation
      2. PacmanTracker belief update
      3. OpponentModelEnsemble update + prediction
      4. RiskEngine: time-expanded danger + survival margin
      5. BudgetController mode selection
      6. PolicyPortfolio: select policies → collect proposals
      7. SafetyShield filter
      8. Arbitrator → best action
      9. Anti Line-of-Sight check
     10. Validate / STAY
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._speed: int = max(1, int(kwargs.get("pacman_speed", 2)))

        # === Blind mode: Hardcoded map ===
        self._known_map = self._build_known_map()
        self._topo = TopologyAnalyzer()
        self._topo.build(self._known_map)
        self._dc = DistanceCache(maxsize=256)
        key_cells = self._topo.junctions | self._topo.chokepoints
        if len(key_cells) <= 60:
            self._dc.precompute(self._known_map, key_cells)
        self._ot = OfflineTable()
        self._ot.build(self._known_map, self._topo)

        # === Blind mode: Memory map ===
        self.memory_map: Optional[np.ndarray] = None

        # === Tracking ===
        self._tracker = PacmanTracker()
        self._ensemble = OpponentModelEnsemble()
        self._anti = AntiLoop()
        self._style_clf = PacmanStyleClassifier()
        self._diag = Diagnostics()
        self._ghost_hist: deque[Pos] = deque(maxlen=30)
        self._last_move: Optional[Move] = None
        self._last_pac: Optional[Pos] = None
        self._prev_pac_action: Action = (0, 0)
        self._current_dir: Optional[Move] = None
        self._dir_streak: int = 0

        # === Zone Navigator ===
        self._zone_nav = ZoneNavigator(self._known_map, self._topo)

        # === Lazy-init components (need topology) ===
        self._risk: Optional[RiskEngine] = None
        self._shield: Optional[SafetyShield] = None
        self._mc: Optional[MCRolloutV3] = None
        self._ab: Optional[AlphaBetaSearch] = None
        self._portfolio: Optional[PolicyPortfolio] = None
        self._lazy_init()

    def _build_known_map(self) -> np.ndarray:
        h, w = len(KNOWN_LAYOUT_STR), len(KNOWN_LAYOUT_STR[0])
        arr = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                arr[r, c] = 1 if KNOWN_LAYOUT_STR[r][c] == '#' else 0
        return arr

    def _lazy_init(self) -> None:
        if self._risk is not None:
            return
        self._risk = RiskEngine(self._topo, self._dc)
        self._shield = SafetyShield(self._topo, self._dc, self._risk)
        self._mc = MCRolloutV3(self._topo, self._dc, self._ensemble)
        self._ab = AlphaBetaSearch(self._topo, self._dc, self._ensemble)
        hp = HybridPolicy(self._mc, self._ot)
        tp = TablePolicy(self._ot)
        abp = AlphaBetaPolicy(self._ab)
        opp = OnePlyPolicy()
        gp = GreedyPolicy()
        self._portfolio = PolicyPortfolio(hp, tp, abp, opp, gp)

    # ------------------------------------------------------------------
    # Memory Map
    # ------------------------------------------------------------------
    def _update_memory(self, map_state: np.ndarray) -> None:
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position,
             step_number: int) -> Move:
        t0 = time.time()
        ms_obs = np.asarray(map_state, dtype=int)
        me: Pos = (int(my_position[0]), int(my_position[1]))
        visible_pac: Optional[Pos] = None
        if enemy_position is not None:
            visible_pac = (int(enemy_position[0]), int(enemy_position[1]))

        # 1. Update memory map
        self._update_memory(ms_obs)
        mem = self.memory_map if self.memory_map is not None else self._known_map

        # 2. Ghost history + anti-loop
        self._ghost_hist.append(me)
        self._anti.record(me)

        # 3. Pacman tracker / belief
        belief = self._tracker.update(enemy_position, mem, self._speed)
        pac: Optional[Pos] = None
        if enemy_position is not None:
            pac = (int(enemy_position[0]), int(enemy_position[1]))

        # 4. Ensemble update
        if pac is not None and self._last_pac is not None:
            self._prev_pac_action = (pac[0] - self._last_pac[0],
                                     pac[1] - self._last_pac[1])
            self._style_clf.update(self._last_pac, pac, me, self._topo)
        self._ensemble.update_if_observed(self._tracker, me, mem)
        self._ensemble.style = self._style_clf.classify()

        if pac is not None:
            self._last_pac = pac

        # 5. Ensemble prediction
        pac_est = self._tracker.best_estimate
        if pac_est is not None:
            pac_preds = self._ensemble.predict_positions_2step(
                me, pac_est, mem, self._topo, self._dc, self._speed,
                self._prev_pac_action)
        else:
            default = PACMAN_START if _valid(PACMAN_START, mem) else me
            pac_preds = [(default, 1.0)]

        # 6. Risk engine
        danger_layers = self._risk.time_expanded_danger(pac_preds, mem, self._speed)
        danger_t0 = danger_layers[0] if danger_layers else {}
        margin = self._risk.survival_margin(me, belief, mem, self._speed)

        # 7. Legal moves (on memory map)
        legal = _legal(me, mem)
        if not legal:
            return Move.STAY

        # Tunnel escape override
        tunnel_move = self._tunnel_escape(me, pac, legal, mem)

        # 7b. Early branching priority
        early_branch_move = None
        if step_number <= EARLY_BRANCH_STEPS:
            early_branch_move = self._early_branch_priority(
                me, legal, pac_preds, belief, danger_t0, mem, step_number)

        # 7c. Zone navigation
        target_zone = self._zone_nav.select_target_zone(
            me, belief, pac_preds, step_number)
        zone_move = self._zone_nav.navigate_toward_zone(
            me, target_zone, legal, mem, self._topo, self._dc,
            danger_t0, pac_preds)

        # 8. Budget mode
        mode = BudgetController.select_mode(me, belief, danger_t0,
                                             self._topo, self._dc, mem)

        # 9. Select policies + collect proposals
        style = self._ensemble.style
        selected = self._portfolio.select_policies(mode, style)

        ctx = {
            "me": me, "pac": pac, "ms": mem, "legal": legal,
            "pac_preds": pac_preds, "danger_t0": danger_t0,
            "topo": self._topo, "dc": self._dc, "speed": self._speed,
            "step": step_number, "t0": t0, "ghost_hist": self._ghost_hist,
            "anti": self._anti, "last_move": self._last_move,
            "margin": margin, "belief": belief,
            "current_dir": self._current_dir,
            "dir_streak": self._dir_streak,
        }

        proposals: List[Proposal] = []
        # Early branch proposal
        if early_branch_move is not None:
            nxt = _apply(me, early_branch_move)
            proposals.append(Proposal(early_branch_move, 600.0,
                                      danger_t0.get(nxt, 0.0),
                                      "high", "early_branch", cost=0.02,
                                      reason="junction branch away from pacman"))
        # Zone navigation proposal
        if zone_move is not None:
            nxt = _apply(me, zone_move)
            my_zone = self._zone_nav.classify(me)
            pac_zone = self._zone_nav.predict_pacman_zone(belief, pac_preds)
            zone_value = 450.0 if my_zone == pac_zone else 350.0
            proposals.append(Proposal(zone_move, zone_value,
                                      danger_t0.get(nxt, 0.0),
                                      "medium", "zone_nav", cost=0.03,
                                      reason=f"navigate to {target_zone} zone"))
        if tunnel_move is not None:
            nxt = _apply(me, tunnel_move)
            proposals.append(Proposal(tunnel_move, 500.0,
                                      danger_t0.get(nxt, 0.0),
                                      "high", "tunnel_escape", cost=0.01))

        for policy in selected:
            if time.time() - t0 > TIME_BUDGET * 0.82:
                break
            try:
                prop = policy.propose(ctx)
                if prop is not None:
                    proposals.append(prop)
            except Exception:
                self._diag.record_failure(policy.name)

        # 10. Safety shield
        safe_proposals = self._shield.filter(
            proposals, me, belief, danger_t0, mem, self._speed)
        if len(safe_proposals) < len(proposals):
            self._diag.record_rejection()

        # 11. Anti Line-of-Sight
        move = PolicyPortfolio.arbitrate(safe_proposals)
        if move is not None and pac is not None:
            move = self._anti_los_adjust(me, pac, move, legal, mem)
        if move is not None:
            chosen = next((p for p in safe_proposals if p.action == move), None)
            if chosen:
                self._diag.record_usage(chosen.source)

        # 12. Validate
        move = self._validate(me, move, legal, mem)

        # 13. Track directional momentum
        if move != Move.STAY:
            if move == self._current_dir:
                self._dir_streak += 1
            else:
                self._current_dir = move
                self._dir_streak = 1

        self._last_move = move
        return move

    # ------------------------------------------------------------------
    # Tunnel Escape
    # ------------------------------------------------------------------
    def _tunnel_escape(self, me: Pos, pac: Optional[Pos],
                        legal: List[Move], ms) -> Optional[Move]:
        if pac is None or me not in self._topo.tunnel_cells:
            return None
        for tc, e1, e2 in self._topo.tunnels:
            if me not in tc:
                continue
            d1 = _manhattan(pac, e1); d2 = _manhattan(pac, e2)
            if min(d1, d2) > 6:
                return None
            target = e2 if d1 < d2 else e1
            td = self._dc.dist(ms, target)
            best_m = None; best_d = INF
            for m in legal:
                nxt = _apply(me, m)
                d = td.get(nxt, INF)
                if d < best_d:
                    best_d = d; best_m = m
            return best_m
        return None

    # ------------------------------------------------------------------
    # Early Branch Priority
    # ------------------------------------------------------------------
    def _early_branch_priority(
        self, me: Pos, legal: List[Move],
        pac_preds: List[Tuple[Pos, float]],
        belief: Dict[Pos, float],
        danger_t0: Dict[Pos, float],
        ms, step_number: int,
    ) -> Optional[Move]:
        topo = self._topo
        n_exits = len(legal)
        if n_exits < 3:
            return None

        best_move: Optional[Move] = None
        best_score = float("-inf")

        if belief:
            cx = sum(p[0] * w for p, w in belief.items()) / max(1e-9, sum(belief.values()))
            cy = sum(p[1] * w for p, w in belief.items()) / max(1e-9, sum(belief.values()))
            pac_centroid: Pos = (int(round(cx)), int(round(cy)))
        else:
            pac_centroid = PACMAN_START

        for m in legal:
            nxt = _apply(me, m)
            if not _valid(nxt, ms):
                continue

            sc = 0.0

            for pp, pw in pac_preds[:8]:
                d = _manhattan(nxt, pp)
                sc += pw * d * 12.0

            sc += _manhattan(nxt, pac_centroid) * 5.0

            if nxt in topo.core:       sc += 25.0
            if nxt in topo.loop_set:   sc += 20.0
            if nxt in topo.junctions:  sc += 15.0
            if nxt in topo.chokepoints: sc += 5.0

            ec = topo.escape_capacity.get(nxt, 0)
            sc += ec * 8.0

            if nxt in topo.dead_ends:
                sc -= 80.0 + topo.trap_depth.get(nxt, 1) * 10.0
            if nxt in topo.tunnel_cells:
                sc -= 20.0

            sc -= danger_t0.get(nxt, 0.0) * 0.6

            pac_near = any(_manhattan(pp, me) <= 8 for pp, _ in pac_preds[:4])
            sc -= self._anti.penalty(nxt, me, pac_near, topo) * 0.7

            if self._last_move is not None and m == self._last_move:
                sc += MOMENTUM_BONUS * 0.8
            if (self._last_move is not None
                    and m.value[0] + self._last_move.value[0] == 0
                    and m.value[1] + self._last_move.value[1] == 0):
                sc -= REVERSAL_PENALTY * 0.8
            if (self._last_move is not None
                    and m != self._last_move
                    and not (m.value[0] + self._last_move.value[0] == 0
                             and m.value[1] + self._last_move.value[1] == 0)):
                sc -= TURN_PENALTY_90
            if self._current_dir is not None and m == self._current_dir:
                sc += min(self._dir_streak, 10) * 6.0

            if (self._last_move is not None and m != self._last_move
                    and not (m.value[0] + self._last_move.value[0] == 0
                             and m.value[1] + self._last_move.value[1] == 0)):
                sc += 8.0

            if sc > best_score:
                best_score = sc
                best_move = m

        return best_move

    # ------------------------------------------------------------------
    # Anti Line-of-Sight
    # ------------------------------------------------------------------
    def _anti_los_adjust(self, me: Pos, pac: Pos, chosen: Move,
                          legal: List[Move], ms) -> Move:
        if me[0] == pac[0]:
            perp = [m for m in legal if m in (Move.UP, Move.DOWN)]
            if perp:
                best = max(perp, key=lambda m: _exits(_apply(me, m), ms))
                return best
        if me[1] == pac[1]:
            perp = [m for m in legal if m in (Move.LEFT, Move.RIGHT)]
            if perp:
                best = max(perp, key=lambda m: _exits(_apply(me, m), ms))
                return best
        return chosen

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(me: Pos, move: Optional[Move], legal: List[Move],
                   ms) -> Move:
        if move is None:
            return legal[0] if legal else Move.STAY
        nxt = _apply(me, move)
        if _valid(nxt, ms) and move in legal:
            return move
        return legal[0] if legal else Move.STAY


# ===================================================================
# MAIN PACMAN AGENT — Optimized with Interception + Speed Packing
# (from submissions/agent/)
# ===================================================================
class PacmanAgent(BasePacmanAgent):
    """Optimized Pacman Agent with Interception + Speed Packing.

    Uses PacmanTopologyAnalyzer and BeliefState for tracking.
    Supports speed packing for multi-step moves.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.memory_map = None
        self.belief = BeliefState(21, 21)
        self.topo = PacmanTopologyAnalyzer()
        self.last_seen = None
        self._last_seen_step = 0
        self._enemy_dir = None
        self._dir_streak = 0
        self._visited = {}
        self._step_t0 = 0.0

    def _update_memory(self, ms):
        if self.memory_map is None:
            self.memory_map = np.full_like(ms, -1, dtype=int)
        v = (ms != -1)
        self.memory_map[v] = ms[v]

    def _ensure_topo(self):
        if not self.topo.ready and self.memory_map is not None:
            k = (self.memory_map != -1)
            if k.sum() > self.memory_map.size * 0.3:
                t = self.memory_map.copy()
                t[t == -1] = 0
                self.topo.analyze(t)

    def _time_ok(self):
        return (time.time() - self._step_t0) < TIME_BUDGET

    def _pack_speed(self, path, my_pos):
        """Pack consecutive same-direction steps up to self.speed."""
        if not path:
            return (Move.STAY, 1)
        first = path[0]
        delta = (first[0] - my_pos[0], first[1] - my_pos[1])
        move = MOVE.get(delta, Move.STAY)
        steps = 1
        cur = first
        for nxt in path[1:]:
            nd = (nxt[0] - cur[0], nxt[1] - cur[1])
            if nd == delta and steps < self.speed:
                steps += 1
                cur = nxt
            else:
                break
        return (move, steps)

    def _track_enemy(self, enemy_pos):
        if self.last_seen is None:
            return
        dr, dc = enemy_pos[0] - self.last_seen[0], enemy_pos[1] - self.last_seen[1]
        nd = (dr, dc)
        if nd == self._enemy_dir and (dr != 0 or dc != 0):
            self._dir_streak += 1
        else:
            self._enemy_dir = nd
            self._dir_streak = 1 if (dr != 0 or dc != 0) else 0

    def _intercept(self, ms, enemy_pos):
        """Find junction to intercept Ghost along its movement direction."""
        if self._dir_streak < 2 or self._enemy_dir is None:
            return None
        dr, dc = self._enemy_dir
        er, ec = enemy_pos
        H, W = 21, 21
        best, best_s = None, float("-inf")
        for i in range(2, 6):
            nr, nc = er + dr * i, ec + dc * i
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if _cell(ms, nr, nc) == 1:
                break
            nxt = (nr, nc)
            exits = _exits(nxt, ms)
            sc = exits * 200
            if self.topo.ready and nxt in self.topo.junctions:
                sc += 500
            if sc > best_s:
                best_s, best = sc, nxt
        return best

    def _explore(self, me):
        H, W = self.memory_map.shape
        best, best_s = None, float("-inf")
        for r in range(H):
            for c in range(W):
                if self.memory_map[r, c] != 0:
                    continue
                has_fog = any(
                    0 <= r + dr < H and 0 <= c + dc < W
                    and self.memory_map[r + dr, c + dc] == -1
                    for dr, dc in DIRS
                )
                if not has_fog:
                    continue
                d = _manhattan((r, c), me)
                prob = self.belief.prob_at((r, c))
                sc = prob * 500 - d
                if self.topo.ready:
                    sc += self.topo.weight((r, c), self.memory_map) * 20
                if sc > best_s:
                    best_s, best = sc, (r, c)
        if best:
            path = astar_path(self.memory_map, me, best)
            if path:
                return self._pack_speed(path, me)
        moves = _legal_deltas(me, self.memory_map)
        if moves:
            d = min(moves, key=lambda d: self._visited.get(_apply(me, d), 0))
            return MOVE.get(d, Move.STAY)
        return Move.STAY

    def step(self, map_state, my_position, enemy_position, step_number):
        self._step_t0 = time.time()
        self._update_memory(map_state)
        me = tuple(my_position)
        self._ensure_topo()
        self._visited[me] = self._visited.get(me, 0) + 1

        enemy = None
        if enemy_position is not None:
            enemy = tuple(int(v) for v in enemy_position)
            self._track_enemy(enemy)
            self.last_seen = enemy
            self._last_seen_step = step_number
        self.belief.update(me, enemy, self.memory_map, enemy_speed=2)

        if self._visited.get(me, 0) >= 5:
            self._visited.clear()
            m = _legal_deltas(me, self.memory_map)
            if m:
                return MOVE.get(random.choice(m), Move.STAY)

        if self.topo.ready and me in self.topo.dead_ends:
            m = _legal_deltas(me, self.memory_map)
            if m:
                return MOVE.get(
                    max(m, key=lambda d: self.topo.weight(_apply(me, d), self.memory_map)),
                    Move.STAY,
                )

        if enemy is not None:
            intercept = self._intercept(self.memory_map, enemy)
            if intercept:
                path = astar_path(self.memory_map, me, intercept)
                if path:
                    return self._pack_speed(path, me)
            path = astar_path(self.memory_map, me, enemy)
            if path:
                return self._pack_speed(path, me)

        since = step_number - self._last_seen_step if self._last_seen_step > 0 else 999
        if since <= 15 and self.last_seen is not None:
            if me == self.last_seen:
                self.last_seen = None
            else:
                path = astar_path(self.memory_map, me, self.last_seen)
                if path:
                    return self._pack_speed(path, me)

        return self._explore(me)
