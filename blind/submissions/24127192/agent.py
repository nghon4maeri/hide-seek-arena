"""Blind Multi-Layer Ghost Agent — 24127192.

V2: Simplified from V1 10-layer. Key improvements:
  - RIGHT+DOWN optimal escape route (11 steps, dominates early game)
  - Enhanced forward momentum & anti-revisit
  - Simplified 3-model ensemble (was 6)
  - Zone nav with Pacman-avoiding path selection

Tập trung: fixed starts, vision=5, response < 0.9s.
"""

from __future__ import annotations

import math
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

TIME_BUDGET        = 0.80
CAPTURE_DISTANCE   = 2
INF                = 10 ** 9
FATAL_DANGER       = 120.0

# MC V3 (tuned up for V3)
MC_ROLLOUTS        = 16
MC_DEPTH           = 20
MC_CAPTURE_PENALTY = 120_000.0
MC_SURVIVE_BONUS   = 8_000.0
CVAR_ALPHA         = 0.20
CVAR_WEIGHT        = 0.60
MEAN_WEIGHT        = 0.40

# Alpha-Beta
AB_MAX_DEPTH       = 6
PANIC_DISTANCE     = 10

# Ensemble
ENSEMBLE_LR        = 0.15
MIN_MODEL_WEIGHT   = 0.05

# Arbitrator
RISK_WEIGHT        = 0.85
CONFIDENCE_WEIGHT  = 0.15
COST_WEIGHT        = 0.05

# History & Anti-loop
HISTORY_LEN        = 20
BELIEF_MAX_CELLS   = 18
DANGER_HORIZON     = 8
MARKOV_LIMIT       = 6
RECENT_CELL_BAN    = 20    # Stronger ban on recent cells (V3: 15→20)

# Directional Momentum (V3: stronger)
MOMENTUM_BONUS     = 100   # Bonus for continuing same direction (V3: 80→100)
REVERSAL_PENALTY   = 1500  # Heavy penalty for 180° turn (V3: 1200→1500)
TURN_PENALTY_90    = 100   # Penalty for 90° turns
CORRIDOR_REV_PEN   = 2500  # Extra penalty in corridors (V3: 2000→2500)

# Zone Navigation
ZONE_SWITCH_INTERVAL = 16
ZONE_NAMES = ("top", "center", "bottom", "left", "right")

# === OPTIMAL ESCAPE ROUTE: RIGHT + DOWN through col 15 ===
OPTIMAL_ESCAPE: Tuple[Move, ...] = (
    Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT,
    Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN,
)
ESCAPE_LEN = len(OPTIMAL_ESCAPE)  # 11

# === V3: DEEP BOTTOM NAVIGATION (Phase 2, steps 12-21) ===
# Từ (14,15) → (14,19) → (18,19) → vào bottom loop
DEEP_BOTTOM_ROUTE: Tuple[Move, ...] = (
    Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT,  # → (14,19)
    Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN,       # → (18,19)
    Move.LEFT, Move.LEFT,                              # → (18,17) vào loop area
)
DEEP_BOTTOM_LEN = len(DEEP_BOTTOM_ROUTE)  # 10

# === V3: UPPER ESCAPE ROUTE (Phase 4, khi Pacman xuống bottom) ===
# Từ bottom → (14,15) → (9,15) → upper area
UPPER_ESCAPE_ROUTE: Tuple[Move, ...] = (
    Move.UP, Move.UP, Move.UP, Move.UP, Move.UP,  # → (9,15) từ (14,15)
    Move.LEFT,                                      # → (9,14)
)
UPPER_ESCAPE_LEN = len(UPPER_ESCAPE_ROUTE)  # 6

# === V3: SAFE CAMPS (precomputed corners with degree=2, low exposure) ===
SAFE_CAMPS: Tuple[Pos, ...] = (
    (19, 1),   # Bottom-left corner — rất an toàn
    (19, 19),  # Bottom-right corner
    (1, 1),    # Top-left corner
    (1, 19),   # Top-right corner
    (1, 5),    # Top-left inner
    (1, 15),   # Top-right inner
    (19, 5),   # Bottom-left inner
    (19, 15),  # Bottom-right inner
)

# === PATROL: Figure-8 loops in bottom area (precomputed from hardcoded map) ===
# Bottom outer loop (~44 cells, clockwise) — rows 14-19, cols 1-19
BOTTOM_OUTER: Tuple[Pos, ...] = tuple([
    (14,15), (14,16), (14,17), (14,18), (14,19),
    (15,19), (16,19), (17,19), (18,19), (19,19),
    (19,18), (19,17), (19,16), (19,15), (19,14),
    (19,13), (19,12), (19,11), (19,10), (19,9),
    (19,8), (19,7), (19,6), (19,5), (19,4),
    (19,3), (19,2), (19,1),
    (18,1), (17,1), (16,1), (15,1), (14,1),
    (14,2), (14,3), (14,4), (14,5),
    (14,6), (14,7),
    (14,8), (14,9), (14,10), (14,11), (14,12), (14,13), (14,14),
])
# Bottom inner loop (~20 cells, counter-clockwise)
BOTTOM_INNER: Tuple[Pos, ...] = tuple([
    (14,7), (15,7), (16,7), (17,7),
    (17,8), (17,9), (17,10), (17,11), (17,12), (17,13),
    (16,13), (15,13), (14,13),
    (14,12), (14,11), (14,10), (14,9), (14,8),
])
# Upper loop (~40 cells, rows 1-7, cols 1-19) — fallback area
UPPER_LOOP: Tuple[Pos, ...] = tuple([
    (4,5), (3,5), (2,5), (1,5),
    (1,6), (1,7), (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14), (1,15),
    (2,15), (3,15), (4,15), (5,15),
    (5,14), (5,13), (5,12), (5,11),
    (5,10), (5,9), (5,8), (5,7),
    (4,7), (3,7),
])

PATROL_SCORE       = 700    # Patrol move priority (V3: 600→700)
CAMP_SCORE         = 450    # Camp move priority (V3: 350→450)
PATROL_ENTER_DIST  = 25     # Enter patrol when Pacman is farther than this (V3: 20→25)
LOOP_SWITCH_STEPS  = 30     # Switch between outer/inner loop (V3: 35→30)
LOOP_SWITCH_STEPS_UPPER = 25

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

def _apply(pos: Pos, move: Move) -> Pos:
    return (pos[0] + move.value[0], pos[1] + move.value[1])

def _valid(pos: Pos, ms) -> bool:
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1

def _legal(pos: Pos, ms) -> List[Move]:
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]

def _neighbors(pos: Pos, ms) -> List[Pos]:
    return [_apply(pos, m) for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]

def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _exits(pos: Pos, ms) -> int:
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))

def _move_key(move: Move) -> int:
    return {Move.UP: 0, Move.LEFT: 1, Move.RIGHT: 2, Move.DOWN: 3,
            Move.STAY: 4}.get(move, 9)

def _pacman_reach(pac: Pos, ms, speed: int = 2) -> List[Pos]:
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
# A* Pathfinding
# ===================================================================
def astar(ms, start: Pos, goal: Pos) -> List[Move]:
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
# Distance Cache
# ===================================================================
class DistanceCache:
    def __init__(self, maxsize: int = 384) -> None:
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

    def precompute(self, ms, cells) -> None:
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
# Topology Analyzer
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
        self._compute_core()
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

    def _compute_core(self) -> None:
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
    def __init__(self, ms, topo: TopologyAnalyzer) -> None:
        h, w = _shape(ms)
        r3 = h // 3; c3 = w // 3
        self._row_top = r3; self._row_bot = h - r3
        self._col_left = c3; self._col_right = w - c3

        self._zone_cells: Dict[str, Set[Pos]] = {
            "top": set(), "center": set(), "bottom": set(),
            "left": set(), "right": set(),
        }
        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1:
                    continue
                pos = (r, c)
                if r < self._row_top:      self._zone_cells["top"].add(pos)
                elif r >= self._row_bot:   self._zone_cells["bottom"].add(pos)
                else:                       self._zone_cells["center"].add(pos)
                if c < self._col_left:     self._zone_cells["left"].add(pos)
                elif c >= self._col_right: self._zone_cells["right"].add(pos)

        self._waypoints: Dict[str, Pos] = {}
        for zname, cells in self._zone_cells.items():
            juncs = [p for p in cells if p in topo.junctions]
            if juncs:
                cr = sum(p[0] for p in cells) / max(1, len(cells))
                cc = sum(p[1] for p in cells) / max(1, len(cells))
                juncs.sort(key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))
                self._waypoints[zname] = juncs[0]
            elif cells:
                cr = sum(p[0] for p in cells) / len(cells)
                cc = sum(p[1] for p in cells) / len(cells)
                self._waypoints[zname] = min(cells, key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))

        self._zone_cycle = ["top", "right", "bottom", "left", "center"]
        self._current_target_zone: Optional[str] = None
        self._steps_in_zone: int = 0
        self._last_zone: Optional[str] = None
        self._opposite: Dict[str, str] = {
            "top": "bottom", "bottom": "top",
            "left": "right", "right": "left", "center": "top",
        }

    def classify(self, pos: Pos) -> str:
        r, c = pos
        if r < self._row_top:    return "top"
        if r >= self._row_bot:   return "bottom"
        if c < self._col_left:   return "left"
        if c >= self._col_right: return "right"
        return "center"

    def predict_pacman_zone(self, belief, pac_preds) -> str:
        zone_prob: Dict[str, float] = {z: 0.0 for z in ZONE_NAMES}
        for pos, w in belief.items():
            zone_prob[self.classify(pos)] += w
        for pos, w in pac_preds[:6]:
            zone_prob[self.classify(pos)] += w * 0.5
        if not any(zone_prob.values()):
            return self.classify(PACMAN_START)
        return max(zone_prob, key=lambda z: zone_prob[z])

    def select_target_zone(self, ghost, belief, pac_preds, step_number, dist_to_pac=20) -> str:
        my_zone = self.classify(ghost)
        pac_zone = self.predict_pacman_zone(belief, pac_preds)
        self._steps_in_zone += 1
        interval = 12 if dist_to_pac < 15 else 25
        need_switch = (self._steps_in_zone >= interval
                       or self._current_target_zone is None
                       or my_zone == pac_zone)
        if need_switch:
            opp = self._opposite.get(pac_zone, "top")
            if opp == my_zone:
                candidates = [z for z in self._zone_cycle if z != pac_zone and z != my_zone]
                if self._last_zone in candidates and len(candidates) > 1:
                    candidates.remove(self._last_zone)
                opp = candidates[0] if candidates else self._zone_cycle[0]
            self._last_zone = self._current_target_zone
            self._current_target_zone = opp
            self._steps_in_zone = 0
        return self._current_target_zone

    def get_waypoint(self, target_zone: str) -> Optional[Pos]:
        return self._waypoints.get(target_zone)

    def navigate_toward_zone(self, ghost, target_zone, legal, ms, topo, dc,
                              danger_t0, pac_preds, pac_centroid=None) -> Optional[Move]:
        waypoint = self.get_waypoint(target_zone)
        if waypoint is None or ghost == waypoint:
            return None
        path = astar(ms, ghost, waypoint)
        if not path or path[0] not in legal:
            return None
        first_move = path[0]
        nxt = _apply(ghost, first_move)
        if nxt in topo.dead_ends or danger_t0.get(nxt, 0.0) >= FATAL_DANGER * 0.7:
            return None
        for pp, pw in pac_preds[:4]:
            if pw > 0.2 and _manhattan(nxt, pp) < CAPTURE_DISTANCE + 1:
                return None
        # Prefer path AWAY from Pacman when multiple options exist
        if pac_centroid is not None and len(legal) >= 2:
            alt_moves = [m for m in legal if m != first_move]
            best_move, best_dist = first_move, _manhattan(nxt, pac_centroid)
            for alt in alt_moves:
                alt_nxt = _apply(ghost, alt)
                if _manhattan(alt_nxt, waypoint) <= _manhattan(nxt, waypoint) + 3:
                    pac_d = _manhattan(alt_nxt, pac_centroid)
                    if pac_d > best_dist + 1 and alt_nxt not in topo.dead_ends:
                        if danger_t0.get(alt_nxt, 0.0) < FATAL_DANGER * 0.8:
                            safe = all(pw <= 0.2 or _manhattan(alt_nxt, pp) >= CAPTURE_DISTANCE + 2
                                       for pp, pw in pac_preds[:4])
                            if safe:
                                best_move, best_dist = alt, pac_d
            return best_move
        return first_move


# ===================================================================
# Pacman Tracker
# ===================================================================
class PacmanTracker:
    def __init__(self) -> None:
        self.belief: Dict[Pos, float] = {PACMAN_START: 1.0}
        self.last_seen: Optional[Pos] = PACMAN_START
        self.history: deque[Pos] = deque(maxlen=50)
        self.steps_invisible: int = 0

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
            n = max(1, len(reachable))
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
# Opponent Models (simplified 3-model ensemble)
# ===================================================================
class _MarkovOrder1:
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
    def predict(self, pac, ghost, ms, **kw):
        state = self._state(ghost, pac, ms)
        counts = self._trans.get(state, self._glob)
        if not counts:
            legal = _legal(pac, ms)
            n = max(1, len(legal))
            return {(m.value[0], m.value[1]): 1.0 / n for m in legal}
        total = sum(counts.values())
        return {a: c / total for a, c in counts.items()}

class _ShortestPathModel:
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
            probs[(m.value[0], m.value[1])] = w
            total += w
        if total > 0:
            for k in probs:
                probs[k] /= total
        return probs

class _InterceptionModel:
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

class OpponentModelEnsemble:
    def __init__(self) -> None:
        self.markov = _MarkovOrder1()
        self.sp = _ShortestPathModel()
        self.intercept = _InterceptionModel()
        self._models = [self.markov, self.sp, self.intercept]
        self.weights = {"markov1": 1.0, "shortest_path": 1.0, "interceptor": 1.0}
        self._last_predictions: Dict[str, Dict[Action, float]] = {}

    def update_if_observed(self, tracker: PacmanTracker, ghost: Pos, ms) -> None:
        hist = tracker.history
        if len(hist) < 2:
            return
        pac_cur = hist[-1]; pac_prev = hist[-2]
        actual = (pac_cur[0] - pac_prev[0], pac_cur[1] - pac_prev[1])
        self.markov.observe(ghost, pac_prev, pac_cur, ms)
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
            self.weights = {k: 1.0 / len(self.weights) for k in self.weights}
            return
        for k in self.weights:
            self.weights[k] = max(MIN_MODEL_WEIGHT, self.weights[k] / total)
        total2 = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total2

    def predict(self, pac, ghost, ms, topo, dc) -> Dict[Action, float]:
        combined: Dict[Action, float] = {}
        self._last_predictions.clear()
        for mdl in self._models:
            dist = mdl.predict(pac, ghost, ms, topo=topo, dc=dc)
            self._last_predictions[mdl.name] = dist
            w = self.weights.get(mdl.name, 0.1)
            for act, prob in dist.items():
                combined[act] = combined.get(act, 0.0) + w * prob
        total = sum(combined.values())
        if total > 0:
            for k in combined:
                combined[k] /= total
        return combined

    def predict_positions_2step(self, ghost, pac, ms, topo, dc, speed=2,
                                 prev_action=(0, 0)) -> List[Tuple[Pos, float]]:
        dist1 = self.predict(pac, ghost, ms, topo, dc)
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
            d2 = self.predict(pos1, ghost, ms, topo, dc)
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
                positions[nxt] = max(positions.get(nxt, 0.0), 0.45 if closer else 0.15)
        positions[pac] = max(positions.get(pac, 0.0), 0.10)
        ranked = sorted(positions.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:MARKOV_LIMIT]


# ===================================================================
# Risk Engine
# ===================================================================
class RiskEngine:
    def __init__(self, topo: TopologyAnalyzer, dc: DistanceCache) -> None:
        self._topo = topo; self._dc = dc

    def time_expanded_danger(self, pac_preds, ms, speed=2, horizon=DANGER_HORIZON):
        danger: List[Dict[Pos, float]] = [{} for _ in range(horizon + 1)]
        current_dist = {p: w for p, w in pac_preds}
        for t in range(horizon + 1):
            for pac_pos, prob in current_dist.items():
                danger[t][pac_pos] = danger[t].get(pac_pos, 0.0) + prob * 100.0
                for m in MOVE_ORDER:
                    cur = pac_pos
                    for s in range(speed):
                        nxt = _apply(cur, m)
                        if not _valid(nxt, ms): break
                        danger[t][nxt] = danger[t].get(nxt, 0.0) + prob * (65.0 / (s + 1)) * (0.85 ** t)
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

    def survival_margin(self, ghost, belief, ms, speed=2) -> float:
        exits = list(self._topo.core | self._topo.loop_set | self._topo.junctions)
        if not exits:
            exits = [p for p in self._topo._open if _exits(p, ms) >= 3]
        if not exits or not belief:
            return 10.0
        gd = self._dc.dist(ms, ghost)
        best_margin = -INF
        for ex in exits[:20]:
            ghost_time = gd.get(ex, INF)
            if ghost_time == INF: continue
            worst_pac = min((math.ceil(self._dc.dist(ms, pc).get(ex, INF) / max(1, speed))
                             for pc, pb in belief.items() if pb >= 0.05), default=INF)
            best_margin = max(best_margin, worst_pac - ghost_time)
        return best_margin

    def is_viable(self, pos, belief, ms, speed=2) -> bool:
        ec = self._topo.escape_capacity.get(pos, 0)
        if ec >= 3 or pos in self._topo.core or pos in self._topo.loop_set:
            return True
        if not belief:
            return True
        best_pac = min(belief, key=lambda p: _manhattan(p, pos))
        return _manhattan(pos, best_pac) >= 5 and ec >= 2


# ===================================================================
# Safety Shield
# ===================================================================
class SafetyShield:
    def __init__(self, topo, dc, risk) -> None:
        self._topo = topo; self._dc = dc; self._risk = risk

    def filter(self, candidates, ghost, belief, danger_t0, ms, speed=2):
        safe = []
        for move, nxt, sc in candidates:
            if not _valid(nxt, ms) or move not in _legal(ghost, ms):
                continue
            if self._can_capture_next(nxt, belief, ms, speed):
                continue
            if danger_t0.get(nxt, 0.0) >= FATAL_DANGER:
                continue
            if not self._risk.is_viable(nxt, belief, ms, speed):
                continue
            safe.append((move, nxt, sc))
        if not safe:
            relaxed = []
            for move, nxt, sc in candidates:
                if not _valid(nxt, ms) or move not in _legal(ghost, ms):
                    continue
                min_d = min((_manhattan(nxt, pc) for pc in belief), default=INF)
                relaxed.append((move, nxt, min_d))
            relaxed.sort(key=lambda x: x[2], reverse=True)
            return relaxed
        return safe

    def _can_capture_next(self, nxt, belief, ms, speed) -> bool:
        for pac_pos, prob in belief.items():
            if prob < 0.15: continue
            for r in _pacman_reach(pac_pos, ms, speed):
                if _manhattan(nxt, r) < CAPTURE_DISTANCE:
                    return True
        return False


# ===================================================================
# Anti-Loop (enhanced)
# ===================================================================
class AntiLoop:
    def __init__(self) -> None:
        self.visit_count: Dict[Pos, int] = defaultdict(int)
        self.edge_count: Dict[Tuple[Pos, Pos], int] = defaultdict(int)
        self.recent: deque[Pos] = deque(maxlen=20)

    def record(self, pos: Pos) -> None:
        self.visit_count[pos] += 1
        if self.recent:
            self.edge_count[(self.recent[-1], pos)] += 1
        self.recent.append(pos)

    def penalty(self, nxt: Pos, cur: Pos, pac_near: bool, topo) -> float:
        pen = self.visit_count[nxt] * 8.0
        pen += self.edge_count[(cur, nxt)] * 15.0
        if self.recent:
            recent_list = list(self.recent)
            ban_window = recent_list[-RECENT_CELL_BAN:]
            if nxt in ban_window:
                recency_index = list(reversed(ban_window)).index(nxt)
                pen += 300.0 + recency_index * 60.0
        if len(self.recent) >= 4:
            r = list(self.recent)
            if nxt == r[-2] and cur == r[-1]:
                pen += 120.0
            if len(r) >= 4 and nxt == r[-3] and cur == r[-2] and r[-1] == r[-3]:
                pen += 200.0
            if len(r) >= 6 and r[-6:-3] == r[-3:]:
                pen += 150.0
        return pen


# ===================================================================
# Monte Carlo Rollout
# ===================================================================
class MCRollout:
    def __init__(self, topo, dc, ensemble) -> None:
        self._topo = topo; self._dc = dc; self._ens = ensemble

    def _ghost_policy(self, ghost, pac, ms, prev):
        moves = _legal(ghost, ms)
        if not moves: return ghost, Move.STAY
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
                dr, dc = nxt[0] - ghost[0], nxt[1] - ghost[1]
                pdr, pdc = ghost[0] - prev[0], ghost[1] - prev[1]
                if dr + pdr == 0 and dc + pdc == 0: sc -= 50.0
            if _manhattan(nxt, pac) < CAPTURE_DISTANCE: sc -= MC_CAPTURE_PENALTY
            if sc > best_s: best_s, best = sc, (nxt, m)
        return best

    def _pac_response(self, pac, ghost, ms, scenario, speed=2):
        pred = self._ens.predict(pac, ghost, ms, self._topo, self._dc)
        if pred:
            ranked = sorted(pred.items(), key=lambda x: -x[1])
            act, _ = ranked[min(scenario % 3, len(ranked) - 1)]
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
        return reach[min(scenario % 3, len(reach) - 1)] if reach else pac

    def _leaf(self, ghost, pac, ms):
        pd = self._dc.dist(ms, pac)
        d = pd.get(ghost, _manhattan(ghost, pac))
        sc = d * 14.0 + _exits(ghost, ms) * 3.5
        if ghost in self._topo.core:       sc += 14.0
        if ghost in self._topo.loop_set:   sc += 11.0
        if ghost in self._topo.junctions:  sc += 8.0
        if ghost in self._topo.dead_ends:  sc -= 40.0 + self._topo.trap_depth.get(ghost, 1) * 6.0
        if ghost in self._topo.tunnel_cells: sc -= 10.0
        return sc

    def rollout(self, g0, first_move, pac0, ms, sn, scenario, speed=2, t0=0.0):
        ghost = _apply(g0, first_move)
        if not _valid(ghost, ms): return -MC_CAPTURE_PENALTY
        pac = self._pac_response(pac0, ghost, ms, scenario, speed)
        if _manhattan(ghost, pac) < CAPTURE_DISTANCE: return -MC_CAPTURE_PENALTY
        total = 0.0; survived = 0; prev = g0
        for depth in range(MC_DEPTH):
            if time.time() - t0 > TIME_BUDGET * 0.80: break
            ng, _ = self._ghost_policy(ghost, pac, ms, prev)
            np_ = self._pac_response(pac, ng, ms, scenario + depth, speed)
            if _manhattan(ng, np_) < CAPTURE_DISTANCE:
                total -= MC_CAPTURE_PENALTY - depth * 6000.0; break
            total += self._leaf(ng, np_, ms) + depth * 200.0
            survived += 1; prev = ghost; ghost = ng; pac = np_
        return total + survived * MC_SURVIVE_BONUS

    def evaluate_move_cvar(self, ghost, move, pac_preds, ms, sn, t0, speed=2):
        nxt = _apply(ghost, move)
        if not _valid(nxt, ms): return None
        returns: List[float] = []
        hyps = pac_preds[:5]
        rp = max(1, MC_ROLLOUTS // max(1, len(hyps)))
        for hi, (pac, hw) in enumerate(hyps):
            if time.time() - t0 > TIME_BUDGET * 0.75: break
            for s in range(rp):
                if time.time() - t0 > TIME_BUDGET * 0.75: break
                sid = sn * 13 + hi * 5 + s
                sc = self.rollout(ghost, move, pac, ms, sn, sid, speed, t0)
                returns.append(sc * hw)
        if not returns: return None
        returns.sort()
        k = max(1, int(len(returns) * CVAR_ALPHA))
        return MEAN_WEIGHT * (sum(returns) / len(returns)) + CVAR_WEIGHT * (sum(returns[:k]) / k)


# ===================================================================
# Policies (MC, OnePly, Greedy)
# ===================================================================
class OnePlyPolicy:
    name = "one_ply"
    def propose(self, ctx):
        me = ctx["me"]; ms = ctx["ms"]; legal = ctx["legal"]
        danger = ctx["danger_t0"]; topo = ctx["topo"]; dc = ctx["dc"]
        preds = ctx["pac_preds"]; anti = ctx["anti"]
        ghost_hist = ctx["ghost_hist"]; last_move = ctx["last_move"]
        current_dir = ctx.get("current_dir"); dir_streak = ctx.get("dir_streak", 0)

        best_m = None; best_sc = float("-inf")
        recent = set(list(ghost_hist)[-RECENT_CELL_BAN:])
        for m in legal:
            nxt = _apply(me, m)
            sc = 0.0
            for pp, pw in preds[:6]:
                pd = dc.dist(ms, pp)
                d = pd.get(nxt, _manhattan(nxt, pp))
                sc += pw * d * 14.0
                sc -= pw * max(0, 7 - d) * 6.0
                gap = min((_manhattan(nxt, r) for r in _pacman_reach(pp, ms, ctx["speed"])), default=INF)
                if gap < CAPTURE_DISTANCE: sc -= pw * MC_CAPTURE_PENALTY * 0.4
                elif gap <= 2: sc -= pw * 500.0
            sc -= danger.get(nxt, 0.0) * 0.8
            sc += _exits(nxt, ms) * 5.5
            if nxt in topo.core:       sc += 24.0
            if nxt in topo.loop_set:   sc += 20.0
            if nxt in topo.junctions:  sc += 14.0
            if nxt in topo.chokepoints: sc += 6.0
            if nxt in topo.dead_ends:  sc -= 75.0 + topo.trap_depth.get(nxt, 1) * 9.0
            if nxt in topo.tunnel_cells: sc -= 18.0
            h, w = _shape(ms)
            ed = min(nxt[0], nxt[1], h - 1 - nxt[0], w - 1 - nxt[1])
            if ed <= 1 and _exits(nxt, ms) <= 2: sc -= 30.0
            pac_near = ctx["pac"] is not None and _manhattan(ctx["pac"], me) <= 10
            sc -= anti.penalty(nxt, me, pac_near, topo)
            if nxt in recent: sc -= 26.0

            # Directional momentum
            if last_move is not None and m == last_move:
                sc += MOMENTUM_BONUS
            if last_move is not None and m.value[0] + last_move.value[0] == 0 and m.value[1] + last_move.value[1] == 0:
                sc -= REVERSAL_PENALTY + (CORRIDOR_REV_PEN if len(legal) == 2 else 0)
            if last_move is not None and m != last_move and not (m.value[0] + last_move.value[0] == 0 and m.value[1] + last_move.value[1] == 0):
                sc -= TURN_PENALTY_90
            if current_dir is not None and m == current_dir:
                sc += min(dir_streak, 10) * 12.0

            if sc > best_sc: best_sc, best_m = sc, m
        if best_m is None: return None
        return (best_m, _apply(me, best_m), best_sc)


class GreedyPolicy:
    name = "greedy"
    def propose(self, ctx):
        me = ctx["me"]; ms = ctx["ms"]; legal = ctx["legal"]
        preds = ctx["pac_preds"]; danger = ctx["danger_t0"]; topo = ctx["topo"]
        anti = ctx["anti"]; last_move = ctx["last_move"]
        best_m = legal[0]; best_sc = float("-inf")
        for m in legal:
            nxt = _apply(me, m)
            min_d = min((_manhattan(nxt, p) for p, _ in preds), default=INF)
            sc = min_d * 10.0 - danger.get(nxt, 0.0)
            ec = topo.escape_capacity.get(nxt, 0); sc += ec * 6.0
            if nxt in topo.dead_ends:   sc -= 100.0
            if nxt in topo.tunnel_cells: sc -= 20.0
            sc -= anti.penalty(nxt, me, False, topo) * 0.5
            if last_move is not None and m == last_move: sc += MOMENTUM_BONUS * 0.5
            if last_move is not None and m.value[0] + last_move.value[0] == 0 and m.value[1] + last_move.value[1] == 0:
                sc -= REVERSAL_PENALTY * 0.5
            if sc > best_sc: best_sc, best_m = sc, m
        return (best_m, _apply(me, best_m), best_sc)


# ===================================================================
# Proposal (for MC policy)
# ===================================================================
class Proposal:
    __slots__ = ("action", "value", "risk", "confidence", "source", "cost", "reason")
    def __init__(self, action, value, risk, confidence="medium", source="", cost=0.0, reason=""):
        self.action = action; self.value = value; self.risk = risk
        self.confidence = confidence; self.source = source; self.cost = cost; self.reason = reason
    def score(self):
        cv = {"high": 1.0, "medium": 0.5, "low": 0.2}.get(self.confidence, 0.3)
        return self.value - RISK_WEIGHT * self.risk + CONFIDENCE_WEIGHT * cv - COST_WEIGHT * self.cost


# ===================================================================
# Patrol Manager (V3 — figure-8 loop patrol)
# ===================================================================
class PatrolManager:
    """Manages figure-8 patrol in bottom area to negate Pacman speed=2."""

    def __init__(self) -> None:
        self._loops = [BOTTOM_OUTER, BOTTOM_INNER]
        self._current_loop_idx = 0       # 0=outer, 1=inner
        self._pos_in_loop = 0
        self._direction = 1               # 1=forward, -1=reverse
        self._active = False
        self._loop_entry_step = 0
        self._steps_on_loop = 0

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, ghost: Pos, step_number: int) -> None:
        """Enter patrol mode at the closest point on any loop."""
        best_loop_idx = 0
        best_pos = 0
        best_dist = INF
        for li, loop in enumerate(self._loops):
            for i, p in enumerate(loop):
                d = _manhattan(ghost, p)
                if d < best_dist:
                    best_dist = d
                    best_loop_idx = li
                    best_pos = i
        self._current_loop_idx = best_loop_idx
        self._pos_in_loop = best_pos
        self._direction = 1
        self._active = True
        self._loop_entry_step = step_number
        self._steps_on_loop = 0

    def deactivate(self) -> None:
        self._active = False

    def get_patrol_move(self, ghost: Pos, legal: List[Move], ms,
                         pac_est: Pos, step_number: int) -> Optional[Move]:
        if not self._active:
            return None

        loop = self._loops[self._current_loop_idx]
        loop_len = len(loop)
        self._steps_on_loop = step_number - self._loop_entry_step

        # Switch between outer/inner loops periodically
        switch_interval = LOOP_SWITCH_STEPS if self._current_loop_idx == 0 else LOOP_SWITCH_STEPS_UPPER
        if self._steps_on_loop >= switch_interval:
            self._current_loop_idx = 1 - self._current_loop_idx  # Toggle 0↔1
            new_loop = self._loops[self._current_loop_idx]
            # Find closest entry on new loop
            best_i, best_d = 0, INF
            for i, p in enumerate(new_loop):
                d = _manhattan(ghost, p)
                if d < best_d:
                    best_d, best_i = d, p
            self._pos_in_loop = best_i
            self._loop_entry_step = step_number
            self._steps_on_loop = 0
            loop = new_loop
            loop_len = len(loop)

        # If ghost is on the loop, follow it
        if ghost in loop:
            self._pos_in_loop = loop.index(ghost)

            # Ambush detection: if Pacman is ahead in our direction, reverse
            dist_to_pac = _manhattan(ghost, pac_est)
            if dist_to_pac < 6:
                ahead = (self._pos_in_loop + self._direction * 4) % loop_len
                if _manhattan(loop[ahead], pac_est) < 3:
                    self._direction *= -1  # Reverse!

            # Get next cell on loop
            next_idx = (self._pos_in_loop + self._direction) % loop_len
            next_cell = loop[next_idx]

            if _manhattan(ghost, next_cell) == 1:
                for m in legal:
                    if _apply(ghost, m) == next_cell:
                        return m

        # Navigate to closest loop cell
        closest = loop[0]
        closest_d = INF
        for p in loop:
            d = _manhattan(ghost, p)
            if d < closest_d:
                closest_d = d
                closest = p
        if ghost == closest:
            # Pick any neighbor on the loop
            for m in legal:
                nxt = _apply(ghost, m)
                if nxt in loop:
                    return m
        path = astar(ms, ghost, closest)
        if path and path[0] in legal:
            return path[0]
        return None


# ===================================================================
# Feint Scheduler (V3 — anti-prediction)
# ===================================================================
class FeintScheduler:
    """Schedule feint moves every N steps to confuse model-based Pacman predictors.

    Cycles through 3 styles:
      - patrol: let PatrolManager handle (no-op here)
      - camp:   head toward nearest safe camp
      - wander: pick a random junction direction away from Pacman
    """

    FEINT_INTERVAL = 27
    STYLES = ("patrol", "camp", "wander")

    def __init__(self) -> None:
        self._counter = 0

    def check(self, step_number: int, ghost: Pos, pac_est: Optional[Pos],
              legal: List[Move], ms, topo: TopologyAnalyzer,
              dc: DistanceCache) -> Optional[Move]:
        """Return a feint move if it's a feint step, else None."""
        if step_number % self.FEINT_INTERVAL != 0:
            return None

        style = self.STYLES[(step_number // self.FEINT_INTERVAL) % 3]
        self._counter += 1

        if style == "patrol":
            # PatrolManager handles patrol — no feint needed
            return None

        if style == "camp":
            return self._camp_feint(ghost, legal, ms, topo, dc)

        if style == "wander":
            return self._wander_feint(ghost, pac_est, legal, ms, topo, dc)

        return None

    def _camp_feint(self, ghost: Pos, legal: List[Move], ms,
                    topo: TopologyAnalyzer, dc: DistanceCache) -> Optional[Move]:
        """Head toward the nearest safe camp."""
        best_camp, best_dist = None, INF
        for camp in SAFE_CAMPS:
            if not _valid(camp, ms):
                continue
            d = _manhattan(ghost, camp)
            if d < best_dist:
                best_dist, best_camp = d, camp

        if best_camp is None or ghost == best_camp:
            return None

        path = astar(ms, ghost, best_camp)
        if path and path[0] in legal:
            nxt = _apply(ghost, path[0])
            if nxt not in topo.dead_ends:
                return path[0]
        return None

    def _wander_feint(self, ghost: Pos, pac_est: Optional[Pos],
                      legal: List[Move], ms, topo: TopologyAnalyzer,
                      dc: DistanceCache) -> Optional[Move]:
        """Pick a junction direction away from Pacman, or random safe direction."""
        # Prefer moves toward junctions
        candidates: List[Tuple[Move, float]] = []
        for m in legal:
            nxt = _apply(ghost, m)
            sc = 0.0
            if nxt in topo.junctions:
                sc += 3.0
            if nxt in topo.core:
                sc += 2.0
            if nxt in topo.loop_set:
                sc += 2.5
            ec = topo.escape_capacity.get(nxt, 0)
            sc += ec * 1.5
            if nxt in topo.dead_ends:
                sc -= 10.0
            if pac_est is not None:
                sc += _manhattan(nxt, pac_est) * 0.5
            candidates.append((m, sc))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[1])
        # Pick from top 3 randomly (deterministic: based on step)
        idx = (self._counter * 7 + ghost[0] * 3 + ghost[1] * 11) % min(3, len(candidates))
        return candidates[idx][0]


# ===================================================================
# Main Ghost Agent
# ===================================================================
class GhostAgent(BaseGhostAgent):
    """Blind Multi-Layer Ghost Agent V3 — 5-Phase Escape + Patrol + Feint."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._speed: int = max(1, int(kwargs.get("pacman_speed", 2)))

        # Hardcoded map + topology
        self._known_map = self._build_known_map()
        self._topo = TopologyAnalyzer()
        self._topo.build(self._known_map)
        self._dc = DistanceCache(maxsize=384)
        key_cells = self._topo.junctions | self._topo.chokepoints
        if len(key_cells) <= 80:
            self._dc.precompute(self._known_map, key_cells)

        # Belief + prediction
        self.memory_map: Optional[np.ndarray] = None
        self._tracker = PacmanTracker()
        self._ensemble = OpponentModelEnsemble()

        # Risk + safety
        self._risk = RiskEngine(self._topo, self._dc)
        self._shield = SafetyShield(self._topo, self._dc, self._risk)

        # Zone navigation
        self._zone_nav = ZoneNavigator(self._known_map, self._topo)

        # Movement
        self._anti = AntiLoop()
        self._mc = MCRollout(self._topo, self._dc, self._ensemble)
        self._one_ply = OnePlyPolicy()
        self._greedy = GreedyPolicy()
        self._feint = FeintScheduler()  # V3: anti-prediction feints

        # State
        self._ghost_hist: deque[Pos] = deque(maxlen=30)
        self._last_move: Optional[Move] = None
        self._last_pac: Optional[Pos] = None
        self._prev_pac_action: Action = (0, 0)
        self._current_dir: Optional[Move] = None
        self._dir_streak: int = 0
        self._deep_bottom_done = False   # V3: track Phase 2 completion
        self._upper_escape_active = False  # V3: track Phase 4 status

    def _build_known_map(self) -> np.ndarray:
        h, w = len(KNOWN_LAYOUT_STR), len(KNOWN_LAYOUT_STR[0])
        arr = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                arr[r, c] = 1 if KNOWN_LAYOUT_STR[r][c] == '#' else 0
        return arr

    def _update_memory(self, map_state: np.ndarray) -> None:
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    def _select_mode(self, ghost, belief, danger_t0) -> str:
        d0 = danger_t0.get(ghost, 0.0)
        if d0 >= FATAL_DANGER * 0.8: return "emergency"
        if belief:
            best_pac = min(belief, key=lambda p: _manhattan(p, ghost))
            if _manhattan(best_pac, ghost) <= PANIC_DISTANCE: return "combat"
        return "normal"

    def _tunnel_escape(self, me, pac, legal, ms) -> Optional[Move]:
        if pac is None or me not in self._topo.tunnel_cells:
            return None
        for tc, e1, e2 in self._topo.tunnels:
            if me not in tc: continue
            if min(_manhattan(pac, e1), _manhattan(pac, e2)) > 6: return None
            target = e2 if _manhattan(pac, e1) < _manhattan(pac, e2) else e1
            td = self._dc.dist(ms, target)
            best_m, best_d = None, INF
            for m in legal:
                d = td.get(_apply(me, m), INF)
                if d < best_d: best_d, best_m = d, m
            return best_m
        return None

    def _anti_los_adjust(self, me, pac, chosen, legal, ms) -> Move:
        if me[0] == pac[0]:
            perp = [m for m in legal if m in (Move.UP, Move.DOWN)]
            if perp: return max(perp, key=lambda m: _exits(_apply(me, m), ms))
        if me[1] == pac[1]:
            perp = [m for m in legal if m in (Move.LEFT, Move.RIGHT)]
            if perp: return max(perp, key=lambda m: _exits(_apply(me, m), ms))
        return chosen

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number: int) -> Move:
        t0 = time.time()
        ms_obs = np.asarray(map_state, dtype=int)
        me: Pos = (int(my_position[0]), int(my_position[1]))
        visible_pac: Optional[Pos] = None
        if enemy_position is not None:
            visible_pac = (int(enemy_position[0]), int(enemy_position[1]))

        self._update_memory(ms_obs)
        mem = self.memory_map if self.memory_map is not None else self._known_map
        self._ghost_hist.append(me)
        self._anti.record(me)

        belief = self._tracker.update(enemy_position, mem, self._speed)

        pac: Optional[Pos] = None
        if visible_pac is not None:
            pac = visible_pac
            if self._last_pac is not None:
                self._prev_pac_action = (pac[0] - self._last_pac[0], pac[1] - self._last_pac[1])
            self._last_pac = pac
        self._ensemble.update_if_observed(self._tracker, me, mem)

        pac_est = self._tracker.best_estimate
        pac_preds = self._ensemble.predict_positions_2step(
            me, pac_est or PACMAN_START, mem, self._topo, self._dc, self._speed, self._prev_pac_action
        ) if pac_est is not None else [(PACMAN_START, 1.0)]

        danger_layers = self._risk.time_expanded_danger(pac_preds, mem, self._speed)
        danger_t0 = danger_layers[0] if danger_layers else {}

        legal = _legal(me, mem)
        if not legal:
            return Move.STAY

        mode = self._select_mode(me, belief, danger_t0)

        # Compute Pacman centroid
        if belief:
            cx = sum(p[0] * w for p, w in belief.items()) / max(1e-9, sum(belief.values()))
            cy = sum(p[1] * w for p, w in belief.items()) / max(1e-9, sum(belief.values()))
            pac_centroid = (int(round(cx)), int(round(cy)))
        else:
            pac_centroid = PACMAN_START

        dist_to_pac = _manhattan(me, pac_est) if pac_est else 20

        # Collect candidates: (move, next_position, score)
        candidates: List[Tuple[Move, Pos, float]] = []

        # === V3 PHASE 1: OPTIMAL ESCAPE ROUTE (steps 1-11) ===
        escape_active = step_number <= ESCAPE_LEN and pac is None
        if escape_active:
            planned = OPTIMAL_ESCAPE[step_number - 1]
            if planned in legal:
                nxt = _apply(me, planned)
                if nxt not in self._topo.dead_ends:
                    candidates.append((planned, nxt, 10000.0))

        # === V3 PHASE 2: DEEP BOTTOM NAVIGATION (steps ESCAPE_LEN+1 to ESCAPE_LEN+DEEP_BOTTOM_LEN) ===
        deep_bottom_step = step_number - ESCAPE_LEN
        deep_bottom_active = (not escape_active
                              and 1 <= deep_bottom_step <= DEEP_BOTTOM_LEN
                              and pac is None
                              and not self._deep_bottom_done)
        if deep_bottom_active:
            planned = DEEP_BOTTOM_ROUTE[deep_bottom_step - 1]
            if planned in legal:
                nxt = _apply(me, planned)
                if nxt not in self._topo.dead_ends:
                    candidates.append((planned, nxt, 9000.0))
            if deep_bottom_step == DEEP_BOTTOM_LEN:
                self._deep_bottom_done = True

        # === Tunnel escape ===
        tunnel_move = self._tunnel_escape(me, pac, legal, mem)
        if tunnel_move is not None:
            candidates.append((tunnel_move, _apply(me, tunnel_move), 800.0))

        # === V3 PHASE 4: UPPER ESCAPE (khi Pacman centroid xuống bottom) ===
        pac_centroid_y = pac_centroid[0]
        pac_in_bottom = pac_centroid_y > 14
        ghost_in_bottom = me[0] >= 14

        if pac_in_bottom and ghost_in_bottom and pac is None and step_number > ESCAPE_LEN:
            # Pacman đã xuống bottom → escape lên upper
            self._upper_escape_active = True
            # Find path to upper area via col 15
            upper_entry = (9, 15)
            path_up = astar(mem, me, upper_entry)
            if path_up and path_up[0] in legal:
                nxt = _apply(me, path_up[0])
                if nxt not in self._topo.dead_ends:
                    candidates.append((path_up[0], nxt, 1500.0))
        else:
            self._upper_escape_active = False

        # === Zone navigation (skip during escape/deep bottom routes) ===
        if (step_number > ESCAPE_LEN + DEEP_BOTTOM_LEN) or pac is not None:
            target_zone = self._zone_nav.select_target_zone(me, belief, pac_preds, step_number, dist_to_pac)
            zone_move = self._zone_nav.navigate_toward_zone(
                me, target_zone, legal, mem, self._topo, self._dc, danger_t0, pac_preds, pac_centroid)
            if zone_move is not None:
                my_zone = self._zone_nav.classify(me)
                pac_zone = self._zone_nav.predict_pacman_zone(belief, pac_preds)
                zone_val = 500.0 if my_zone == pac_zone else 350.0
                candidates.append((zone_move, _apply(me, zone_move), zone_val))

        # === V3: FEINT SCHEDULER (anti-prediction) ===
        if step_number > ESCAPE_LEN + DEEP_BOTTOM_LEN:
            feint_move = self._feint.check(step_number, me, pac_est, legal, mem, self._topo, self._dc)
            if feint_move is not None:
                nxt = _apply(me, feint_move)
                if nxt not in self._topo.dead_ends:
                    candidates.append((feint_move, nxt, 550.0))

        # === V3: ENHANCED CAMP (khi Pacman rất xa) ===
        if dist_to_pac > 25 and step_number > ESCAPE_LEN + DEEP_BOTTOM_LEN:
            # Find nearest safe camp
            best_camp, best_cd = None, INF
            for camp in SAFE_CAMPS:
                if not _valid(camp, mem):
                    continue
                cd = _manhattan(me, camp)
                # Prefer camps far from Pacman
                pac_dist = _manhattan(camp, pac_centroid)
                camp_score = cd - pac_dist * 0.5
                if camp_score < best_cd:
                    best_cd, best_camp = camp_score, camp

            if best_camp is not None and me != best_camp:
                camp_path = astar(mem, me, best_camp)
                if camp_path and camp_path[0] in legal:
                    nxt = _apply(me, camp_path[0])
                    if nxt not in self._topo.dead_ends:
                        camp_val = CAMP_SCORE if dist_to_pac > 30 else CAMP_SCORE * 0.7
                        candidates.append((camp_path[0], nxt, camp_val))

        # === Policy proposals ===
        if step_number <= ESCAPE_LEN + DEEP_BOTTOM_LEN and pac is None:
            pols = []
        elif mode == "emergency":
            pols = [self._greedy]
        elif mode == "combat":
            pols = [self._one_ply, self._greedy]
        else:
            pols = [self._one_ply, self._greedy]

        ctx = {
            "me": me, "pac": pac, "ms": mem, "legal": legal,
            "pac_preds": pac_preds, "danger_t0": danger_t0,
            "topo": self._topo, "dc": self._dc, "speed": self._speed,
            "step": step_number, "t0": t0, "ghost_hist": self._ghost_hist,
            "anti": self._anti, "last_move": self._last_move, "belief": belief,
            "current_dir": self._current_dir, "dir_streak": self._dir_streak,
        }

        for policy in pols:
            if time.time() - t0 > TIME_BUDGET * 0.82:
                break
            try:
                result = policy.propose(ctx)
                if result is not None:
                    candidates.append(result)
            except Exception:
                pass

        if not candidates:
            return legal[0] if legal else Move.STAY

        # Safety filter
        safe = self._shield.filter(candidates, me, belief, danger_t0, mem, self._speed)

        if not safe:
            return legal[0] if legal else Move.STAY

        # Best by score
        best = max(safe, key=lambda x: x[2])
        move = best[0]

        # Anti Line-of-Sight
        if pac is not None:
            move = self._anti_los_adjust(me, pac, move, legal, mem)

        # Validate
        nxt = _apply(me, move)
        if not _valid(nxt, mem) or move not in legal:
            move = legal[0] if legal else Move.STAY

        # Track momentum
        if move != Move.STAY:
            if move == self._current_dir:
                self._dir_streak += 1
            else:
                self._current_dir = move
                self._dir_streak = 1

        self._last_move = move
        return move


# ===================================================================
# Pacman Agent (placeholder)
# ===================================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.speed: int = max(1, int(kwargs.get("pacman_speed", 2)))
        self._known_map = None
        self.memory_map: Optional[np.ndarray] = None
        self._tracker = PacmanTracker()
        self._last_seen: Optional[Pos] = None
        self._last_seen_step: int = 0
        self._visited: Dict[Pos, int] = {}

    def _build_known_map(self) -> np.ndarray:
        h, w = len(KNOWN_LAYOUT_STR), len(KNOWN_LAYOUT_STR[0])
        arr = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                arr[r, c] = 1 if KNOWN_LAYOUT_STR[r][c] == '#' else 0
        return arr

    def _update_memory(self, map_state: np.ndarray) -> None:
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self.memory_map[visible_mask] = map_state[visible_mask]

    def step(self, map_state, my_position, enemy_position, step_number: int):
        if self._known_map is None:
            self._known_map = self._build_known_map()
        self._update_memory(map_state)
        me: Pos = (int(my_position[0]), int(my_position[1]))
        self._visited[me] = self._visited.get(me, 0) + 1
        mem = self.memory_map if self.memory_map is not None else self._known_map
        enemy: Optional[Pos] = None
        if enemy_position is not None:
            enemy = (int(enemy_position[0]), int(enemy_position[1]))
            self._last_seen = enemy
            self._last_seen_step = step_number
        self._tracker.update(enemy_position, mem, 2)

        if self._visited.get(me, 0) >= 5:
            self._visited.clear()
            m = _legal(me, mem)
            if m: return m[0]

        if enemy is not None:
            path = astar(mem, me, enemy)
            if path:
                mv = path[0]
                steps = 1
                cur = _apply(me, mv)
                if len(path) > 1 and path[1] == mv and steps < self.speed:
                    steps = 2
                return (mv, steps) if steps > 1 else mv

        since = step_number - self._last_seen_step if self._last_seen_step > 0 else 999
        if since <= 15 and self._last_seen is not None and me != self._last_seen:
            path = astar(mem, me, self._last_seen)
            if path:
                mv = path[0]
                return (mv, 2) if self.speed >= 2 and len(path) > 1 and path[1] == mv else (mv, 1)

        moves = _legal(me, mem)
        if moves:
            d = min(moves, key=lambda m: self._visited.get(_apply(me, m), 0))
            return (d, 1)
        return Move.STAY
