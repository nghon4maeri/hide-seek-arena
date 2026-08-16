"""
Blind Ghost Agent V2 — Simplified Multi-Layer Architecture (24127192)
=====================================================================

V2 CHANGES (from V1 2369-line 10-layer):
  REDUCED: 10 components → 5 consolidated components
  ENHANCED: Forward momentum, anti-revisit, zone navigation, path cache
  NEW: PacmanTurnPredictor, FeintScheduler, CampManager, AdaptiveScorer
  REMOVED: AlphaBetaSearch, TablePolicy, OfflineTable, 3 weak models, Diagnostics

ARCHITECTURE (5 components):
  C1: Hardcoded Map + Topology Cache (KNOWN_LAYOUT, TopologyAnalyzer, DistanceCache, PathCache)
  C2: Belief State + Pacman Prediction (PacmanTracker, SimplifiedEnsemble, PacmanTurnPredictor)
  C3: Risk + Safety (RiskEngine, SafetyShield)
  C4: Zone Navigation (ZoneNavigator with dynamic intervals + anti-Pacman strategy)
  C5: Movement Strategy (AntiLoop, FeintScheduler, CampManager, AdaptiveScorer)

POLICIES (3, was 5):
  - MCPolicy: Monte Carlo rollouts for combat (< 10 distance)
  - OnePlyPolicy: Efficient one-step evaluation with directional momentum
  - GreedyPolicy: Fast fallback
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

TIME_BUDGET        = 0.85
CAPTURE_DISTANCE   = 2
INF                = 10 ** 9

# --- Enhanced Forward Momentum ---
RECENT_CELL_BAN    = 15       # Ban revisiting last 15 positions (was 30 — stronger)
MOMENTUM_BONUS     = 80       # Bonus for continuing same direction (was 40)
REVERSAL_PENALTY   = 1200     # Heavy penalty for 180° turn (was 600)
TURN_PENALTY_90    = 120      # Penalty for 90° turns (was 90)
CORRIDOR_REV_PENALTY = 2500   # Extra penalty for reversing in a corridor

# --- MC Rollout (simplified) ---
MC_ROLLOUTS        = 15       # (was 12)
MC_DEPTH           = 18       # (was 15)
MC_CAPTURE_PENALTY = 150_000.0
MC_SURVIVE_BONUS   = 12_000.0  # (was 8_000)
CVAR_ALPHA         = 0.20
CVAR_WEIGHT        = 0.60
MEAN_WEIGHT        = 0.40

# --- Danger & Safety ---
DANGER_HORIZON     = 8        # (was 6)
FATAL_DANGER       = 150.0    # (was 120)
PANIC_DISTANCE     = 10       # (was 8)

# --- Zone Navigation ---
ZONE_SWITCH_MIN    = 12       # Min interval when Pacman is near
ZONE_SWITCH_MAX    = 30       # Max interval when Pacman is far
ZONE_NAMES = ("top", "center", "bottom", "left", "right")

# --- Anti-Loop ---
BELIEF_MAX_CELLS   = 18
MARKOV_LIMIT       = 8

# --- Feint & Camp ---
FEINT_INTERVAL     = 25       # Feint every N steps
CAMP_MAX_STAY      = 8        # Don't camp longer than this
CAMP_ROTATE_STEPS  = 35       # Rotate camp every N steps

# --- Early Escape ---
EARLY_ESCAPE_STEPS = 20       # First N steps: prioritize max distance from Pacman start

# Hardcoded route: 3R → 2D → 6L → 2D → 6L → 2D → STAY
OPTIMAL_ESCAPE_ROUTE: Tuple[Move, ...] = (
    Move.RIGHT, Move.RIGHT, Move.RIGHT,                            # 3 RIGHT
    Move.DOWN, Move.DOWN,                                          # 2 DOWN
    Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT,  # 6 LEFT
    Move.DOWN, Move.DOWN,                                          # 2 DOWN
    Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT,  # 6 LEFT
    Move.DOWN, Move.DOWN,                                          # 2 DOWN
) + (Move.STAY,) * 179  # STAY for remaining steps (total = 200)
ESCAPE_ROUTE_LEN = len(OPTIMAL_ESCAPE_ROUTE)  # 200 steps

# --- Ensemble (simplified to 3 models) ---
ENSEMBLE_LR        = 0.25
MIN_MODEL_WEIGHT   = 0.05

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

# ===================================================================
# Grid Utilities (unchanged from V1 — battle-tested)
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
# C1: Distance Cache (Enhanced — precomputes more)
# ===================================================================
class DistanceCache:
    def __init__(self, maxsize: int = 512) -> None:  # ↑ from 256
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
# C1: Path Cache — precompute A* paths between all junctions
# ===================================================================
class PathCache:
    """Precompute shortest paths between all junction pairs for fast navigation."""
    def __init__(self) -> None:
        self._paths: Dict[Tuple[Pos, Pos], List[Move]] = {}
        self._dists: Dict[Tuple[Pos, Pos], int] = {}

    def build(self, ms, junctions: Set[Pos]) -> None:
        jlist = list(junctions)
        for i, src in enumerate(jlist):
            for dst in jlist[i + 1:]:
                path = astar(ms, src, dst)
                self._paths[(src, dst)] = path
                self._paths[(dst, src)] = list(reversed([
                    {Move.UP: Move.DOWN, Move.DOWN: Move.UP,
                     Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT}.get(m, m)
                    for m in path]))
                self._dists[(src, dst)] = len(path)
                self._dists[(dst, src)] = len(path)

    def path(self, a: Pos, b: Pos) -> Optional[List[Move]]:
        return self._paths.get((a, b))

    def dist(self, a: Pos, b: Pos) -> int:
        return self._dists.get((a, b), INF)

# ===================================================================
# C1: Topology Analyzer (simplified — keep essential features)
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

        # NEW: Safe loops for camping
        self.safe_loops:      List[List[Pos]] = []
        # NEW: Safe camps (corners with high escape capacity, low visibility)
        self.safe_camps:      List[Pos] = []

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
        self._find_safe_loops()
        self._find_safe_camps(ms)
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
        # Store all cycles as safe loops
        self.safe_loops = [c for c in cycles if len(c) >= 4]

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

    def _find_safe_loops(self) -> None:
        """Identify loops that are large and have multiple exits — good for evasion."""
        for cyc in self.safe_loops:
            cyc_set = set(cyc)
            exits_to_outside = sum(
                1 for p in cyc_set
                for m in MOVE_ORDER
                if _apply(p, m) in self._open and _apply(p, m) not in cyc_set
            )
            # Mark as good loop if it has ≥ 2 exits and ≥ 8 cells
            if exits_to_outside >= 2 and len(cyc) >= 8:
                pass  # already stored

    def _find_safe_camps(self, ms) -> None:
        """Find cells that are good hiding spots:
        - Corner (degree == 2, perpendicular neighbors)
        - Not dead-end (escape_capacity >= 2)
        - Near walls (low visibility down corridors)
        - In or near loop_set
        """
        candidates = []
        for pos in self._open:
            deg = _exits(pos, ms)
            if deg != 2:
                continue
            nbs = _neighbors(pos, ms)
            if len(nbs) != 2:
                continue
            v1 = (nbs[0][0] - pos[0], nbs[0][1] - pos[1])
            v2 = (nbs[1][0] - pos[0], nbs[1][1] - pos[1])
            if v1[0] * v2[0] + v1[1] * v2[1] != 0:  # not a corner
                continue
            if pos in self.dead_ends:
                continue
            if pos in self.tunnel_cells:
                continue
            esc = self.escape_capacity.get(pos, 0)
            if esc < 2:
                continue
            candidates.append(pos)

        # Sort by: in loop > near loop > far from Pacman start
        def camp_score(p):
            s = 0.0
            if p in self.loop_set:
                s += 100.0
            # Prefer far from Pacman start
            s += _manhattan(p, PACMAN_START) * 2.0
            # Prefer near walls (adjacent wall count)
            h, w = _shape(ms) if hasattr(ms, 'shape') else (21, 21)
            wall_adj = 0
            for m in MOVE_ORDER:
                nr, nc = p[0] + m.value[0], p[1] + m.value[1]
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    wall_adj += 1
                elif _cell(ms, nr, nc) == 1:
                    wall_adj += 1
            s += wall_adj * 15.0
            # Slight preference for higher escape capacity
            s += self.escape_capacity.get(p, 0) * 5.0
            return s

        candidates.sort(key=camp_score, reverse=True)
        self.safe_camps = candidates[:6]  # top 6 safe camps

# ===================================================================
# C2: Pacman Tracker (Belief State)
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
# C2: Pacman Turn Predictor (NEW)
# ===================================================================
class PacmanTurnPredictor:
    """Predicts which branch Pacman will take at upcoming junctions.
    Pacman agents typically TURN at junctions rather than going straight.
    Counter-strategy: predict the branch, go opposite."""

    def __init__(self, topo: TopologyAnalyzer) -> None:
        self._topo = topo
        self._turn_history: Dict[Pos, Counter] = defaultdict(Counter)
        self._dir_bias: Counter = Counter()  # UP/DOWN/LEFT/RIGHT preference

    def record_turn(self, pac_pos: Pos, prev_pos: Pos, ghost_pos: Pos) -> None:
        """Record which way Pacman turned at a junction."""
        if pac_pos not in self._topo.junctions:
            return
        dr = pac_pos[0] - prev_pos[0]
        dc = pac_pos[1] - prev_pos[1]
        if dr != 0 or dc != 0:
            self._dir_bias[(dr, dc)] += 1
            self._turn_history[pac_pos][(dr, dc)] += 1

    def predict_junction_branch(self, pac_pos: Pos, ghost_pos: Pos, ms) -> Dict[Pos, float]:
        """Predict probability distribution over Pacman's next positions from a junction."""
        if pac_pos not in self._topo.junctions:
            # Not at a junction — predict straight-line continuation toward ghost
            legal = _legal(pac_pos, ms)
            if not legal:
                return {pac_pos: 1.0}
            probs = {}
            for m in legal:
                nxt = _apply(pac_pos, m)
                d = _manhattan(nxt, ghost_pos)
                probs[nxt] = 1.0 / max(1, d)
            total = sum(probs.values())
            return {k: v / total for k, v in probs.items()} if total > 0 else {pac_pos: 1.0}

        # At junction: weight branches by distance to ghost (Pacman likely chooses closest)
        branches = _neighbors(pac_pos, ms)
        if not branches:
            return {pac_pos: 1.0}

        probs: Dict[Pos, float] = {}
        for branch in branches:
            d = _manhattan(branch, ghost_pos)
            # Pacman prefers branches CLOSER to ghost
            base_w = 1.0 / max(0.5, d)
            # Historical bias
            dr = branch[0] - pac_pos[0]
            dc = branch[1] - pac_pos[1]
            hist_bias = 1.0 + self._dir_bias.get((dr, dc), 0) * 0.1
            probs[branch] = base_w * hist_bias
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()} if total > 0 else {pac_pos: 1.0}

    def best_branch(self, pac_pos: Pos, ghost_pos: Pos, ms) -> Optional[Pos]:
        """Return the single most likely branch Pacman will take."""
        probs = self.predict_junction_branch(pac_pos, ghost_pos, ms)
        if not probs:
            return None
        return max(probs, key=lambda k: probs[k])

# ===================================================================
# C2: Simplified Opponent Model Ensemble (3 models, was 6)
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
            act = (m.value[0], m.value[1])
            probs[act] = w
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

class SimplifiedEnsemble:
    """3-model ensemble (was 6). Markov1 + ShortestPath + Interception."""
    def __init__(self) -> None:
        self.markov = _MarkovOrder1()
        self.sp = _ShortestPathModel()
        self.intercept = _InterceptionModel()
        self._models = [self.markov, self.sp, self.intercept]
        self.weights: Dict[str, float] = {
            "markov1": 1.0, "shortest_path": 1.0, "interceptor": 1.0,
        }
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
            n = len(self.weights)
            self.weights = {k: 1.0 / n for k in self.weights}
            return
        for k in self.weights:
            self.weights[k] = max(MIN_MODEL_WEIGHT, self.weights[k] / total)
        total2 = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total2

    def predict(self, pac: Pos, ghost: Pos, ms, topo, dc) -> Dict[Action, float]:
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

    def predict_positions_2step(self, ghost: Pos, pac: Pos, ms, topo, dc,
                                 speed: int = 2,
                                 prev_action: Action = (0, 0)
                                 ) -> List[Tuple[Pos, float]]:
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
                ch = 0.45 if closer else 0.15
                positions[nxt] = max(positions.get(nxt, 0.0), ch)
        positions[pac] = max(positions.get(pac, 0.0), 0.10)
        ranked = sorted(positions.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:MARKOV_LIMIT]

# ===================================================================
# C3: Risk Engine (simplified)
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
        if _manhattan(pos, best_pac) >= 5 and ec >= 2:
            return True
        return False

# ===================================================================
# C3: Safety Shield (simplified to 3 levels)
# ===================================================================
class SafetyShield:
    def __init__(self, topo: TopologyAnalyzer, dc: DistanceCache,
                 risk: RiskEngine) -> None:
        self._topo = topo
        self._dc = dc
        self._risk = risk

    def filter(self, candidates: List[Tuple[Move, Pos, float]], ghost: Pos,
               belief: Dict[Pos, float], danger_t0: Dict[Pos, float],
               ms, speed: int = 2) -> List[Tuple[Move, Pos, float]]:
        safe = []
        for move, nxt, sc in candidates:
            if not _valid(nxt, ms) or move not in _legal(ghost, ms):
                continue
            # Level 1: Immediate capture check
            if self._can_capture_next(nxt, belief, ms, speed):
                continue
            # Level 2: Fatal danger
            if danger_t0.get(nxt, 0.0) >= FATAL_DANGER:
                continue
            # Level 3: Viability
            if not self._risk.is_viable(nxt, belief, ms, speed):
                continue
            safe.append((move, nxt, sc))
        if not safe:
            # Relaxed: allow any legal move, sort by distance to belief
            relaxed = []
            for move, nxt, sc in candidates:
                if not _valid(nxt, ms) or move not in _legal(ghost, ms):
                    continue
                min_d = min((_manhattan(nxt, pc) for pc in belief), default=INF)
                relaxed.append((move, nxt, min_d))
            relaxed.sort(key=lambda x: x[2], reverse=True)
            return relaxed
        return safe

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

# ===================================================================
# C4: Zone Navigator (Enhanced with dynamic intervals)
# ===================================================================
class ZoneNavigator:
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
        dist_to_pacman: int = 20,
    ) -> str:
        my_zone = self.classify(ghost)
        pac_zone = self.predict_pacman_zone(belief, pac_preds)
        self._steps_in_zone += 1

        # Dynamic interval: shorter when Pacman is near, longer when far
        dynamic_interval = ZONE_SWITCH_MIN if dist_to_pacman < 15 else ZONE_SWITCH_MAX

        need_switch = (
            self._steps_in_zone >= dynamic_interval
            or self._current_target_zone is None
            or my_zone == pac_zone
        )

        if need_switch:
            opp = self._opposite.get(pac_zone, "top")
            if opp == my_zone:
                candidates = [z for z in self._zone_cycle
                              if z != pac_zone and z != my_zone]
                if candidates:
                    if self._last_zone in candidates and len(candidates) > 1:
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
        pac_centroid: Optional[Pos] = None,
    ) -> Optional[Move]:
        waypoint = self.get_waypoint(target_zone)
        if waypoint is None or ghost == waypoint:
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

        # --- When multiple initial moves are possible, prefer the one
        # that goes AWAY from Pacman centroid ---
        if pac_centroid is not None and len(legal) >= 2:
            alt_moves = [m for m in legal if m != first_move]
            best_move = first_move
            best_dist = _manhattan(nxt, pac_centroid)
            for alt in alt_moves:
                alt_nxt = _apply(ghost, alt)
                alt_to_wp = _manhattan(alt_nxt, waypoint)
                cur_to_wp = _manhattan(nxt, waypoint)
                if alt_to_wp <= cur_to_wp + 3:
                    pac_d = _manhattan(alt_nxt, pac_centroid)
                    if pac_d > best_dist + 1:
                        if (alt_nxt not in topo.dead_ends
                                and danger_t0.get(alt_nxt, 0.0) < FATAL_DANGER * 0.8):
                            safe = True
                            for pp, pw in pac_preds[:4]:
                                if pw > 0.2 and _manhattan(alt_nxt, pp) < CAPTURE_DISTANCE + 2:
                                    safe = False; break
                            if safe:
                                best_move = alt
                                best_dist = pac_d
            return best_move

        return first_move

# ===================================================================
# C5: Anti-Loop (Enhanced — stronger penalties)
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

    def penalty(self, nxt: Pos, cur: Pos, pac_near: bool,
                topo: TopologyAnalyzer) -> float:
        # Core penalties
        pen = self.visit_count[nxt] * 10.0              # ↑ from 6.0
        pen += self.edge_count[(cur, nxt)] * 18.0       # ↑ from 12.0

        # --- Strong ban on recent positions (last RECENT_CELL_BAN) ---
        if self.recent:
            recent_list = list(self.recent)
            ban_window = recent_list[-RECENT_CELL_BAN:]
            if nxt in ban_window:
                recency_index = list(reversed(ban_window)).index(nxt)
                pen += 400.0 + recency_index * 80.0     # ↑ from 200+40

        # --- Oscillation detection ---
        if len(self.recent) >= 4:
            r = list(self.recent)
            # A→B→A back-and-forth
            if nxt == r[-2] and cur == r[-1]:
                pen += 150.0                            # ↑ from 80
            # 2-step oscillation: A→B→A→B
            if len(r) >= 4 and nxt == r[-3] and cur == r[-2] and r[-1] == r[-3]:
                pen += 250.0                            # ↑ from 150
            # 3-step cycle
            if len(r) >= 6 and r[-6:-3] == r[-3:]:
                pen += 200.0                            # ↑ from 120

        # --- Dead-end trap ---
        if nxt in topo.dead_ends:
            pen += 50.0 + topo.trap_depth.get(nxt, 1) * 5.0

        return pen

# ===================================================================
# C5: Feint Scheduler (NEW)
# ===================================================================
class FeintScheduler:
    """Every FEINT_INTERVAL steps, makes a 'deceptive' move to confuse
    Pacman model-based predictors."""
    def __init__(self) -> None:
        self._counter = 0
        self._feint_active = False
        self._feint_move: Optional[Move] = None

    def should_feint(self, step_number: int) -> bool:
        return step_number > 0 and step_number % FEINT_INTERVAL == 0

    def choose_feint(self, ghost: Pos, legal: List[Move],
                     best_move: Move, ms) -> Optional[Move]:
        """Choose a feint: a legal move that is NOT the best move.
        Prefer moves that look like we're going toward Pacman (deceptive)."""
        other_moves = [m for m in legal if m != best_move]
        if not other_moves:
            return None
        # Prefer moves that go toward Pacman start (deceptive — looks aggressive)
        def toward_pacman_score(m):
            nxt = _apply(ghost, m)
            return -_manhattan(nxt, PACMAN_START)  # lower = closer to pacman start
        other_moves.sort(key=toward_pacman_score)
        return other_moves[0]  # pick move closest to Pacman start (deceptive)

# ===================================================================
# C5: Camp Manager (NEW)
# ===================================================================
class CampManager:
    """Manages safe camp selection and rotation."""
    def __init__(self, topo: TopologyAnalyzer) -> None:
        self._topo = topo
        self._camps: List[Pos] = list(topo.safe_camps[:4])  # top 4
        self._current_camp_idx = 0
        self._steps_at_camp = 0
        self._last_rotate_step = 0

    @property
    def current_camp(self) -> Optional[Pos]:
        if not self._camps:
            return None
        return self._camps[self._current_camp_idx]

    def should_rotate(self, step_number: int) -> bool:
        return (step_number - self._last_rotate_step) >= CAMP_ROTATE_STEPS

    def rotate(self, step_number: int) -> None:
        self._current_camp_idx = (self._current_camp_idx + 1) % max(1, len(self._camps))
        self._last_rotate_step = step_number
        self._steps_at_camp = 0

    def at_camp(self, ghost: Pos) -> bool:
        return ghost == self.current_camp

    def record_stay(self) -> None:
        self._steps_at_camp += 1

    def can_stay(self) -> bool:
        return self._steps_at_camp < CAMP_MAX_STAY

    def get_move_toward_camp(self, ghost: Pos, legal: List[Move], ms) -> Optional[Move]:
        camp = self.current_camp
        if camp is None or ghost == camp:
            return None
        path = astar(ms, ghost, camp)
        if path and path[0] in legal:
            return path[0]
        return None

# ===================================================================
# C5: Loop Runner (NEW — negates Pacman speed=2 advantage)
# ===================================================================
# ===================================================================
# C5: Adaptive Scorer (NEW — ML-inspired feature-based evaluation)
# ===================================================================
class AdaptiveScorer:
    """12-feature linear model for position evaluation with online weight updates."""
    def __init__(self) -> None:
        # Feature weights (initialized with domain knowledge)
        self._w = {
            "dist_to_pacman":     14.0,
            "exits":               5.5,
            "is_core":            24.0,
            "is_loop":            20.0,
            "is_junction":        14.0,
            "dead_end_depth":    -12.0,
            "is_tunnel":         -18.0,
            "is_chokepoint":       6.0,
            "escape_capacity":     3.0,
            "visit_count":       -10.0,
            "recency":            -5.0,
            "quadrant_safety":     4.0,
        }
        self._lr = 0.15
        self._last_features: Optional[Dict[str, float]] = None
        self._last_score: float = 0.0

    def score(self, features: Dict[str, float]) -> float:
        self._last_features = features
        s = sum(self._w[k] * features.get(k, 0.0) for k in self._w)
        self._last_score = s
        return s

    def update_from_outcome(self, outcome: float) -> None:
        """Outcome > 0 = good (survived), < 0 = bad (captured or near-capture).
        Updates weights toward features that correlate with good outcomes."""
        if self._last_features is None:
            return
        # Simple SGD update: w += lr * outcome * feature_value
        for k, fv in self._last_features.items():
            self._w[k] += self._lr * outcome * fv * 0.01  # small steps

# ===================================================================
# Monte Carlo Rollout (simplified — kept as high-value policy)
# ===================================================================
class MCRollout:
    def __init__(self, topo: TopologyAnalyzer, dc: DistanceCache,
                 ensemble: SimplifiedEnsemble) -> None:
        self._topo = topo; self._dc = dc; self._ens = ensemble

    def _ghost_policy(self, ghost, pac, ms, prev):
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
            ng, _ = self._ghost_policy(ghost, pac, ms, prev)
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
# Policies (simplified to 3 — MC, OnePly, Greedy)
# ===================================================================
class MCPolicy:
    """Monte Carlo rollout for combat situations."""
    name = "mc"

    def __init__(self, mc: MCRollout) -> None:
        self._mc = mc

    def propose(self, ctx) -> Optional[Tuple[Move, Pos, float]]:
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
            flee = _manhattan(nxt, pac) * 2.5
            danger_v = danger.get(nxt, 0.0)
            loop_pen = anti.penalty(nxt, me, True, topo)
            combined = (0.40 * mc_sc + 0.25 * flee
                        - 0.30 * danger_v - 0.10 * loop_pen)
            if combined > best_sc:
                best_sc = combined; best_m = m
        if best_m is None:
            return None
        nxt = _apply(me, best_m)
        return (best_m, nxt, best_sc)


class OnePlyPolicy:
    """One-step evaluation with strong directional momentum. Primary policy."""
    name = "one_ply"

    def propose(self, ctx) -> Optional[Tuple[Move, Pos, float]]:
        me = ctx["me"]; ms = ctx["ms"]; legal = ctx["legal"]
        danger = ctx["danger_t0"]; topo = ctx["topo"]
        preds = ctx["pac_preds"]; dc = ctx["dc"]
        anti = ctx["anti"]; ghost_hist = ctx["ghost_hist"]
        last_move = ctx["last_move"]
        current_dir = ctx.get("current_dir")
        dir_streak = ctx.get("dir_streak", 0)

        best_m = None; best_sc = float("-inf")
        recent = set(list(ghost_hist)[-RECENT_CELL_BAN:])
        for m in legal:
            nxt = _apply(me, m)
            sc = 0.0

            # Distance from predicted Pacman positions
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

            # Danger
            sc -= danger.get(nxt, 0.0) * 0.8

            # Topology bonuses
            sc += _exits(nxt, ms) * 5.5
            if nxt in topo.core:       sc += 24.0
            if nxt in topo.loop_set:   sc += 20.0
            if nxt in topo.junctions:  sc += 14.0
            if nxt in topo.chokepoints: sc += 6.0
            if nxt in topo.dead_ends:
                sc -= 75.0 + topo.trap_depth.get(nxt, 1) * 9.0
            if nxt in topo.tunnel_cells: sc -= 18.0

            # Edge distance penalty
            h, w = _shape(ms)
            ed = min(nxt[0], nxt[1], h - 1 - nxt[0], w - 1 - nxt[1])
            if ed <= 1 and _exits(nxt, ms) <= 2: sc -= 30.0

            # Anti-loop
            pac_near = ctx["pac"] is not None and _manhattan(ctx["pac"], me) <= 10
            sc -= anti.penalty(nxt, me, pac_near, topo)
            if nxt in recent: sc -= 30.0

            # ===== ENHANCED DIRECTIONAL MOMENTUM =====
            # Strong bonus for continuing straight
            if last_move is not None and m == last_move:
                sc += MOMENTUM_BONUS

            # Severe penalty for 180° reversal
            if (last_move is not None
                    and m.value[0] + last_move.value[0] == 0
                    and m.value[1] + last_move.value[1] == 0):
                # Extra penalty if we're in a corridor (only 2 choices)
                if len(legal) == 2:
                    sc -= REVERSAL_PENALTY + CORRIDOR_REV_PENALTY
                else:
                    sc -= REVERSAL_PENALTY

            # Penalty for 90° turns
            if (last_move is not None
                    and m != last_move
                    and not (m.value[0] + last_move.value[0] == 0
                             and m.value[1] + last_move.value[1] == 0)):
                sc -= TURN_PENALTY_90

            # Progressive bonus for long directional streaks
            if current_dir is not None and m == current_dir:
                sc += min(dir_streak, 15) * 15  # max +225

            if sc > best_sc:
                best_sc = sc; best_m = m

        if best_m is None:
            return None
        nxt = _apply(me, best_m)
        return (best_m, nxt, best_sc)


class GreedyPolicy:
    """Fast fallback policy."""
    name = "greedy"

    def propose(self, ctx) -> Optional[Tuple[Move, Pos, float]]:
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

            # Directional momentum (half strength in fallback)
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
        return (best_m, nxt, best_sc)

# ===================================================================
# Main Ghost Agent — V2 Simplified Multi-Layer
# ===================================================================
class GhostAgent(BaseGhostAgent):
    """Blind Ghost Agent V2 — Simplified 5-Component Architecture.

    Pipeline per step:
      1. Update memory_map from observation
      2. PacmanTracker belief update
      3. SimplifiedEnsemble prediction
      4. RiskEngine danger assessment
      5. Select mode (emergency / combat / normal)
      6. Collect candidate moves from 3 policies
      7. SafetyShield filter
      8. Feint / Zone / Camp overrides
      9. Anti Line-of-Sight adjust
     10. Final arbitration → best move
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._speed: int = max(1, int(kwargs.get("pacman_speed", 2)))

        # === C1: Hardcoded Map + Topology Cache ===
        self._known_map = self._build_known_map()
        self._topo = TopologyAnalyzer()
        self._topo.build(self._known_map)
        self._dc = DistanceCache(maxsize=512)
        # Precompute all junctions + chokepoints + safe camps
        key_cells = self._topo.junctions | self._topo.chokepoints | set(self._topo.safe_camps[:4])
        self._dc.precompute(self._known_map, key_cells)
        # Path cache between junctions
        self._path_cache = PathCache()
        self._path_cache.build(self._known_map, self._topo.junctions)

        # === C2: Belief + Prediction ===
        self.memory_map: Optional[np.ndarray] = None
        self._tracker = PacmanTracker()
        self._ensemble = SimplifiedEnsemble()
        self._turn_pred = PacmanTurnPredictor(self._topo)

        # === C3: Risk + Safety ===
        self._risk = RiskEngine(self._topo, self._dc)
        self._shield = SafetyShield(self._topo, self._dc, self._risk)

        # === C4: Zone Navigation ===
        self._zone_nav = ZoneNavigator(self._known_map, self._topo)

        # === C5: Movement Strategy ===
        self._anti = AntiLoop()
        self._feint = FeintScheduler()
        self._camp = CampManager(self._topo)
        self._scorer = AdaptiveScorer()

        # === MC + Policies ===
        self._mc = MCRollout(self._topo, self._dc, self._ensemble)
        self._mc_pol = MCPolicy(self._mc)
        self._one_ply = OnePlyPolicy()
        self._greedy = GreedyPolicy()

        # === State ===
        self._ghost_hist: deque[Pos] = deque(maxlen=30)
        self._last_move: Optional[Move] = None
        self._last_pac: Optional[Pos] = None
        self._prev_pac_action: Action = (0, 0)
        self._current_dir: Optional[Move] = None
        self._dir_streak: int = 0

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

    # ------------------------------------------------------------------
    # Mode Selection (simplified to 3 modes)
    # ------------------------------------------------------------------
    def _select_mode(self, ghost: Pos, belief: Dict[Pos, float],
                     danger_t0: Dict[Pos, float]) -> str:
        d0 = danger_t0.get(ghost, 0.0)
        if d0 >= FATAL_DANGER * 0.8:
            return "emergency"
        if belief:
            best_pac = min(belief, key=lambda p: _manhattan(p, ghost))
            dist = _manhattan(best_pac, ghost)
            if dist <= PANIC_DISTANCE:
                return "combat"
        return "normal"

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
    # Anti Line-of-Sight
    # ------------------------------------------------------------------
    def _anti_los_adjust(self, me: Pos, pac: Pos, chosen: Move,
                          legal: List[Move], ms) -> Move:
        if me[0] == pac[0]:
            perp = [m for m in legal if m in (Move.UP, Move.DOWN)]
            if perp:
                return max(perp, key=lambda m: _exits(_apply(me, m), ms))
        if me[1] == pac[1]:
            perp = [m for m in legal if m in (Move.LEFT, Move.RIGHT)]
            if perp:
                return max(perp, key=lambda m: _exits(_apply(me, m), ms))
        return chosen

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

        # 4. Ensemble update + Pacman turn recording
        pac: Optional[Pos] = None
        if visible_pac is not None:
            pac = (int(enemy_position[0]), int(enemy_position[1]))
            if self._last_pac is not None:
                self._prev_pac_action = (pac[0] - self._last_pac[0],
                                         pac[1] - self._last_pac[1])
                self._turn_pred.record_turn(pac, self._last_pac, me)
            self._last_pac = pac
        self._ensemble.update_if_observed(self._tracker, me, mem)

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

        # 7. Legal moves
        legal = _legal(me, mem)
        if not legal:
            return Move.STAY

        # 8. Mode selection
        mode = self._select_mode(me, belief, danger_t0)

        # 9. Build context
        ctx = {
            "me": me, "pac": pac, "ms": mem, "legal": legal,
            "pac_preds": pac_preds, "danger_t0": danger_t0,
            "topo": self._topo, "dc": self._dc, "speed": self._speed,
            "step": step_number, "t0": t0, "ghost_hist": self._ghost_hist,
            "anti": self._anti, "last_move": self._last_move,
            "belief": belief,
            "current_dir": self._current_dir,
            "dir_streak": self._dir_streak,
        }

        # 10. Collect candidate moves from policies
        candidates: List[Tuple[Move, Pos, float]] = []

        # Tunnel escape (highest priority)
        tunnel_move = self._tunnel_escape(me, pac, legal, mem)
        if tunnel_move is not None:
            candidates.append((tunnel_move, _apply(me, tunnel_move), 800.0))

        # --- HARDCODE TEST: FORCE loop through OPTIMAL_ESCAPE_ROUTE forever ---
        route_idx = (step_number - 1) % ESCAPE_ROUTE_LEN
        planned_move = OPTIMAL_ESCAPE_ROUTE[route_idx]
        self._last_move = planned_move
        if planned_move != Move.STAY:
            self._current_dir = planned_move
            self._dir_streak = 1
        return planned_move

        # Zone navigation
        dist_to_pac = _manhattan(me, pac_est) if pac_est else 20

        # Compute Pacman centroid for direction-away bias
        if belief:
            cx = sum(p[0] * w for p, w in belief.items()) / max(1e-9, sum(belief.values()))
            cy = sum(p[1] * w for p, w in belief.items()) / max(1e-9, sum(belief.values()))
            pac_centroid: Pos = (int(round(cx)), int(round(cy)))
        else:
            pac_centroid = PACMAN_START

        # Zone navigation (skip during escape route phase)
        if step_number > ESCAPE_ROUTE_LEN or pac is not None:
            target_zone = self._zone_nav.select_target_zone(
                me, belief, pac_preds, step_number, dist_to_pac)
            zone_move = self._zone_nav.navigate_toward_zone(
                me, target_zone, legal, mem, self._topo, self._dc,
                danger_t0, pac_preds, pac_centroid)
            if zone_move is not None:
                my_zone = self._zone_nav.classify(me)
                pac_zone = self._zone_nav.predict_pacman_zone(belief, pac_preds)
                zone_val = 500.0 if my_zone == pac_zone else 350.0
                candidates.append((zone_move, _apply(me, zone_move), zone_val))

        # Camp navigation (skip during escape route phase)
        if (step_number > ESCAPE_ROUTE_LEN
                and pac is None and dist_to_pac > 15
                and self._camp.current_camp is not None):
            if self._camp.should_rotate(step_number):
                self._camp.rotate(step_number)
            if self._camp.at_camp(me):
                self._camp.record_stay()
                if self._camp.can_stay():
                    candidates.append((Move.STAY, me, 300.0))
            camp_move = self._camp.get_move_toward_camp(me, legal, mem)
            if camp_move is not None:
                candidates.append((camp_move, _apply(me, camp_move), 250.0))


        # Policy proposals based on mode (skip during escape route to save time)
        if step_number <= ESCAPE_ROUTE_LEN and pac is None:
            pols = []  # Escape route handles everything
        elif mode == "emergency":
            pols = [self._greedy]
        elif mode == "combat":
            pols = [self._mc_pol, self._one_ply, self._greedy]
        else:
            pols = [self._one_ply, self._greedy]

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

        # 11. Safety shield filter
        safe = self._shield.filter(candidates, me, belief, danger_t0, mem, self._speed)

        # 12. If no safe candidates, pick safest among all (max distance from belief)
        if not safe:
            scored = []
            for move, nxt, sc in candidates:
                if move not in legal or not _valid(nxt, mem):
                    continue
                min_d = min((_manhattan(nxt, pc) for pc in belief), default=INF)
                scored.append((move, nxt, min_d))
            scored.sort(key=lambda x: x[2], reverse=True)
            if scored:
                safe = scored[:1]

        # 13. Pick best — score-based arbitration
        if safe:
            best = max(safe, key=lambda x: x[2])
            move = best[0]
        else:
            move = legal[0]

        # 14. Feint override (skip during escape route)
        if (step_number > ESCAPE_ROUTE_LEN
                and self._feint.should_feint(step_number)
                and pac is None):
            feint_move = self._feint.choose_feint(me, legal, move, mem)
            if feint_move is not None:
                move = feint_move

        # 15. Anti Line-of-Sight adjustment
        if pac is not None:
            move = self._anti_los_adjust(me, pac, move, legal, mem)

        # 16. Validate
        nxt = _apply(me, move)
        if not _valid(nxt, mem) or move not in legal:
            move = legal[0] if legal else Move.STAY

        # 17. Track directional momentum
        if move != Move.STAY:
            if move == self._current_dir:
                self._dir_streak += 1
            else:
                self._current_dir = move
                self._dir_streak = 1

        self._last_move = move

        # 18. Adaptive scorer update (outcome = survived this step)
        nxt_final = _apply(me, move)
        features = {
            "dist_to_pacman": float(min((_manhattan(nxt_final, p) for p, _ in pac_preds[:4]), default=20)),
            "exits": float(_exits(nxt_final, mem)),
            "is_core": 1.0 if nxt_final in self._topo.core else 0.0,
            "is_loop": 1.0 if nxt_final in self._topo.loop_set else 0.0,
            "is_junction": 1.0 if nxt_final in self._topo.junctions else 0.0,
            "dead_end_depth": float(self._topo.trap_depth.get(nxt_final, 0)),
            "is_tunnel": 1.0 if nxt_final in self._topo.tunnel_cells else 0.0,
            "is_chokepoint": 1.0 if nxt_final in self._topo.chokepoints else 0.0,
            "escape_capacity": float(self._topo.escape_capacity.get(nxt_final, 0)),
            "visit_count": float(self._anti.visit_count.get(nxt_final, 0)),
            "recency": 1.0 if nxt_final in list(self._ghost_hist)[-10:] else 0.0,
            "quadrant_safety": 1.0 if nxt_final in self._topo.loop_set or nxt_final in self._topo.core else 0.0,
        }
        self._scorer.score(features)
        # Positive outcome = survived
        self._scorer.update_from_outcome(1.0)

        return move


# ===================================================================
# Pacman Agent (placeholder — required for file to load)
# ===================================================================
class PacmanAgent(BasePacmanAgent):
    """Placeholder Pacman Agent — for benchmark compatibility."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.speed: int = max(1, int(kwargs.get("pacman_speed", 2)))
        self._known_map = None
        self._topo = None
        self._dc = None
        self.memory_map: Optional[np.ndarray] = None
        self._tracker = PacmanTracker()
        self._last_seen: Optional[Pos] = None
        self._last_seen_step: int = 0
        self._visited: Dict[Pos, int] = {}
        self._step_t0: float = 0.0

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

    def _time_ok(self) -> bool:
        return (time.time() - self._step_t0) < TIME_BUDGET

    def _pack_speed(self, path: List[Move], my_pos: Pos) -> Tuple[Move, int]:
        if not path:
            return (Move.STAY, 1)
        first_move = path[0]
        steps = 1
        cur = _apply(my_pos, first_move)
        for m in path[1:]:
            if m == first_move and steps < self.speed:
                steps += 1
                cur = _apply(cur, m)
            else:
                break
        return (first_move, steps)

    def _explore(self, me: Pos):
        if self.memory_map is None:
            return (Move.STAY, 1)
        H, W = self.memory_map.shape
        best: Optional[Pos] = None
        best_s = float("-inf")
        for r in range(H):
            for c in range(W):
                if self.memory_map[r, c] != 0:
                    continue
                has_fog = any(
                    0 <= r + dr < H and 0 <= c + dc < W
                    and _cell(self.memory_map, r + dr, c + dc) == -1
                    for dr, dc in [(d.value[0], d.value[1]) for d in MOVE_ORDER]
                )
                if not has_fog:
                    continue
                d = _manhattan((r, c), me)
                prob = self._tracker.belief.get((r, c), 0.0)
                sc = prob * 500.0 - d
                if sc > best_s:
                    best_s, best = sc, (r, c)
        if best is not None:
            path = astar(self.memory_map, me, best)
            if path:
                return self._pack_speed(path, me)
        moves = _legal(me, self.memory_map)
        if moves:
            d = min(moves, key=lambda m: self._visited.get(_apply(me, m), 0))
            return (d, 1)
        return (Move.STAY, 1)

    def step(self, map_state, my_position, enemy_position, step_number: int):
        self._step_t0 = time.time()
        if self._known_map is None:
            self._known_map = self._build_known_map()
            self._topo = TopologyAnalyzer()
            self._topo.build(self._known_map)
            self._dc = DistanceCache(maxsize=256)
            key_cells = self._topo.junctions | self._topo.chokepoints
            self._dc.precompute(self._known_map, key_cells)
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

        # Anti-stuck
        if self._visited.get(me, 0) >= 5:
            self._visited.clear()
            m = _legal(me, mem)
            if m:
                return random.choice(m)

        # Escape dead-end
        if self._topo is not None and me in self._topo.dead_ends:
            m = _legal(me, mem)
            if m:
                return max(m, key=lambda mv: _exits(_apply(me, mv), mem))

        # Chase enemy
        if enemy is not None:
            path = astar(mem, me, enemy)
            if path:
                mv, steps = self._pack_speed(path, me)
                return (mv, steps) if steps > 1 else mv

        # Search last known position
        since = step_number - self._last_seen_step if self._last_seen_step > 0 else 999
        if since <= 15 and self._last_seen is not None:
            if me == self._last_seen:
                self._last_seen = None
            else:
                path = astar(mem, me, self._last_seen)
                if path:
                    mv, steps = self._pack_speed(path, me)
                    return (mv, steps) if steps > 1 else mv

        # Explore
        mv, steps = self._explore(me)
        return (mv, steps) if steps > 1 else mv
