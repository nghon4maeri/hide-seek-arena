"""GhostAgent — Blind Hider (Lab 2: Partial Observability).

Student: 24127192
Role:    Hide Agent Engineer
Version: V1.0

Core algorithms:
- Belief-State Monte Carlo Planning (POMCP-lite): Ghost maintains a
  probability distribution over Pacman's likely positions and runs MC
  rollouts through the belief state.
- Pursuit-Evasion Search: Paranoid/worst-case adversarial model for
  Pacman — assume Pacman plays optimally against Ghost.
- Paranoid Search (alpha-beta with min-over-Pacman-branches):
  Used for deep tactical look-ahead when Pacman is visible.
- Information Set MCTS rollouts: Ghost enumerates plausible Pacman
  positions weighted by belief state and simulates deterministic scenarios.
- USL* Online Learning: Tracks Pacman action frequencies in
  abstract states to refine belief updates.
- Memory Map: accumulates cross-step observations (0 / 1 / -1).
- Topology analysis: precomputes dead-ends, junctions, loops, core
  on the known map to bias all heuristics.
"""

from __future__ import annotations

import heapq
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

# ===================================================================
# Type aliases
# ===================================================================
Pos = Tuple[int, int]

# ===================================================================
# Constants
# ===================================================================
MOVE_ORDER: Tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

TIME_BUDGET       = 0.82   # hard step budget (s), below arena's 1s timeout
CAPTURE_DISTANCE  = 2      # Manhattan < 2 → caught
INF               = 10**9

# Belief-state parameters
BELIEF_PARTICLES  = 28     # max weighted hypotheses retained
DEFAULT_PACMAN_START = (15, 10)
VISION_RADIUS     = 5

# Monte Carlo
MC_ROLLOUTS       = 10     # deterministic scenario rollouts per move
MC_HORIZON        = 10     # steps per rollout
MC_CAPTURE_PENALTY= 120_000.0
MC_SURVIVE_BONUS  = 8_000.0

# Paranoid / Alpha-Beta
AB_MAX_DEPTH      = 7
PANIC_DISTANCE    = 8      # BFS ≤ this → switch to paranoid search

# Anti-oscillation
HISTORY_LEN       = 8

# USL* abstract state
USL_LIMIT         = 5

# ===================================================================
# Grid utilities (work on numpy memory maps)
# ===================================================================

def _shape(ms) -> Tuple[int, int]:
    return int(ms.shape[0]), int(ms.shape[1])


def _cell(ms, r: int, c: int) -> int:
    return int(ms[r, c])


def _apply(pos: Pos, move: Move) -> Pos:
    return (pos[0] + move.value[0], pos[1] + move.value[1])


def _valid(pos: Pos, ms) -> bool:
    """Valid: in-bounds and not a confirmed wall (1).
    Unseen cells (-1) are treated as traversable (optimistic)."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1


def _valid_known(pos: Pos, ms) -> bool:
    """Valid: in-bounds and confirmed passable (0)."""
    r, c = pos
    h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) == 0


def _legal(pos: Pos, ms) -> List[Move]:
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _exits(pos: Pos, ms) -> int:
    return sum(1 for m in MOVE_ORDER if _valid(_apply(pos, m), ms))


def _move_key(move: Move) -> int:
    order = {
        Move.UP: 0,
        Move.LEFT: 1,
        Move.RIGHT: 2,
        Move.DOWN: 3,
        Move.STAY: 4,
    }
    return order.get(move, 9)


def _pos_key(pos: Pos) -> Tuple[int, int]:
    return (pos[0], pos[1])


def _normalize(weights: Dict[Pos, float], limit: int = BELIEF_PARTICLES) -> List[Tuple[Pos, float]]:
    cleaned = [(pos, max(0.0, float(weight))) for pos, weight in weights.items() if weight > 0]
    if not cleaned:
        return []
    cleaned.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    cleaned = cleaned[:limit]
    total = sum(weight for _, weight in cleaned)
    if total <= 0:
        equal = 1.0 / len(cleaned)
        return [(pos, equal) for pos, _ in cleaned]
    return [(pos, weight / total) for pos, weight in cleaned]


def _visible_cross_cells(pos: Pos, ms, radius: int = VISION_RADIUS) -> Set[Pos]:
    visible: Set[Pos] = {pos}
    h, w = _shape(ms)
    for move in MOVE_ORDER:
        dr, dc = move.value
        for dist in range(1, radius + 1):
            nr, nc = pos[0] + dr * dist, pos[1] + dc * dist
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if _cell(ms, nr, nc) == 1:
                break
            visible.add((nr, nc))
    return visible


def _in_cross_los(observer: Pos, target: Pos, ms, radius: int = VISION_RADIUS) -> bool:
    if observer == target:
        return True
    same_row = observer[0] == target[0]
    same_col = observer[1] == target[1]
    if not same_row and not same_col:
        return False
    dist = _manhattan(observer, target)
    if dist > radius:
        return False
    dr = 0 if same_row else (1 if target[0] > observer[0] else -1)
    dc = 0 if same_col else (1 if target[1] > observer[1] else -1)
    cur = observer
    for _ in range(dist):
        cur = (cur[0] + dr, cur[1] + dc)
        if _cell(ms, cur[0], cur[1]) == 1:
            return False
    return True


def _pacman_reach(pac: Pos, ms, speed: int = 2) -> List[Pos]:
    reach: Set[Pos] = {pac}
    for move in MOVE_ORDER:
        cur = pac
        for _ in range(max(1, speed)):
            nxt = _apply(cur, move)
            if not _valid(nxt, ms):
                break
            reach.add(nxt)
            cur = nxt
    return sorted(reach, key=_pos_key)


# ===================================================================
# BFS distance map (with LRU cache invalidation per memory state)
# ===================================================================

class BFSCache:
    """BFS distance-map cache keyed by (start, map-hash)."""

    def __init__(self, maxsize: int = 128) -> None:
        self._cache: Dict[Tuple, Dict[Pos, int]] = {}
        self._maxsize = maxsize
        self._map_hash: int = 0

    def invalidate(self) -> None:
        self._cache.clear()

    def dist(self, ms, start: Pos) -> Dict[Pos, int]:
        key = (start, self._map_hash)
        if key in self._cache:
            d = self._cache.pop(key)
            self._cache[key] = d
            return d
        d = self._compute(ms, start)
        self._cache[key] = d
        if len(self._cache) > self._maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return d

    @staticmethod
    def _compute(ms, start: Pos) -> Dict[Pos, int]:
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
# A* pathfinding
# ===================================================================

def astar(ms, start: Pos, goal: Pos) -> List[Move]:
    if not _valid(start, ms) or not _valid(goal, ms) or start == goal:
        return []
    open_set: List[Tuple[int, int, Pos]] = []
    heapq.heappush(open_set, (_manhattan(start, goal), 0, start))
    came_from: Dict[Pos, Tuple[Pos, Move]] = {}
    g: Dict[Pos, int] = {start: 0}
    closed: Set[Pos] = set()
    while open_set:
        _, gc, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            path: List[Move] = []
            while cur != start:
                cur, move = came_from[cur]
                path.append(move)
            path.reverse()
            return path
        for move in MOVE_ORDER:
            nxt = _apply(cur, move)
            if not _valid(nxt, ms) or nxt in closed:
                continue
            ng = gc + 1
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came_from[nxt] = (cur, move)
                heapq.heappush(open_set, (ng + _manhattan(nxt, goal), ng, nxt))
    return []


# ===================================================================
# Topology analysis
# ===================================================================

class Topology:
    """Precompute structural features from the known portion of the map."""

    def __init__(self) -> None:
        self.dead_ends:  Set[Pos] = set()
        self.junctions:  Set[Pos] = set()
        self.corridors:  Set[Pos] = set()
        self.core:       Set[Pos] = set()
        self.loop_set:   Set[Pos] = set()
        self.danger_depth: Dict[Pos, int] = {}
        self._open: Set[Pos] = set()

    def build(self, ms) -> None:
        h, w = _shape(ms)
        # Walls are globally visible in the arena; unknown non-wall cells are
        # usable for deterministic topology planning even before direct visit.
        self._open = {(r, c) for r in range(h) for c in range(w) if _cell(ms, r, c) != 1}
        self.dead_ends.clear()
        self.junctions.clear()
        self.corridors.clear()
        seeds: Set[Pos] = set()
        for p in self._open:
            deg = _exits(p, ms)
            if deg <= 1:
                seeds.add(p)
            elif deg >= 3:
                self.junctions.add(p)
            else:
                self.corridors.add(p)
        self._propagate_dead_ends(ms, seeds)
        self._compute_core(ms)
        self._find_loops(ms)

    def _propagate_dead_ends(self, ms, seeds: Set[Pos]) -> None:
        self.danger_depth = {p: 0 for p in self._open}
        trimmed: Set[Pos] = set()
        degree = {p: _exits(p, ms) for p in self._open}
        q: deque[Pos] = deque(seeds)
        for p in seeds:
            trimmed.add(p)
            self.dead_ends.add(p)
            self.danger_depth[p] = 1
        while q:
            u = q.popleft()
            for m in MOVE_ORDER:
                v = _apply(u, m)
                if v not in self._open or v in trimmed:
                    continue
                degree[v] -= 1
                if degree[v] <= 1:
                    trimmed.add(v)
                    self.dead_ends.add(v)
                    self.danger_depth[v] = self.danger_depth[u] + 1
                    q.append(v)

    def _compute_core(self, ms) -> None:
        active = set(self._open)
        changed = True
        while changed:
            changed = False
            to_rm: Set[Pos] = set()
            for p in active:
                if sum(1 for m in MOVE_ORDER if _apply(p, m) in active) <= 1:
                    to_rm.add(p)
            if to_rm:
                active -= to_rm
                changed = True
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
            stack = [(start, None, iter([n for m in MOVE_ORDER if (n := _apply(start, m)) in self.core]))]
            vis.add(start)
            parent[start] = None
            while stack:
                u, p, it = stack[-1]
                try:
                    v = next(it)
                except StopIteration:
                    stack.pop()
                    continue
                if v not in vis:
                    vis.add(v)
                    parent[v] = u
                    stack.append((v, u, iter([n for m in MOVE_ORDER if (n := _apply(v, m)) in self.core])))
                elif v != p:
                    cycle: List[Pos] = [v]
                    cur = u
                    while cur is not None and cur != v:
                        cycle.append(cur)
                        cur = parent.get(cur)
                    if cur == v and len(cycle) >= 4:
                        cycles.append(cycle)
        self.loop_set = set(max(cycles, key=len)) if cycles else set()


# ===================================================================
# USL* Online Learner (Pacman behaviour modelling)
# ===================================================================

def _bucket(v: int) -> int:
    if v <= -6: return -4
    if v <= -3: return -3
    if v <   0: return -1
    if v ==  0: return  0
    if v <   3: return  1
    if v <   6: return  3
    return 4


class USLStar:
    """Lightweight online learner: tracks Pacman action counts per abstract state."""

    def __init__(self) -> None:
        self._counts: Dict[Tuple, Counter] = defaultdict(Counter)
        self._global: Counter = Counter()

    def abstract_state(self, ghost: Pos, pac: Pos, ms) -> Tuple:
        dr = _bucket(ghost[0] - pac[0])
        dc = _bucket(ghost[1] - pac[1])
        dist_b = min(9, _manhattan(ghost, pac) // 2)
        pac_moves = _legal(pac, ms)
        deg = len(pac_moves)
        if deg <= 1:
            geo = 0
        elif deg >= 3:
            geo = 3
        else:
            m1, m2 = pac_moves[0], pac_moves[1]
            opposite = (m1.value[0] + m2.value[0] == 0 and m1.value[1] + m2.value[1] == 0)
            geo = 1 if opposite else 2
        return (dr, dc, dist_b, geo)

    def observe(self, state: Tuple, action: Tuple[int, int], ms: np.ndarray, from_pos: Pos) -> None:
        to_pos = (from_pos[0] + action[0], from_pos[1] + action[1])
        if _valid(to_pos, ms):
            self._counts[state][action] += 1
            self._global[action] += 1

    def predict_next_positions(self, ghost: Pos, pac: Pos, ms, limit: int = USL_LIMIT) -> List[Tuple[Pos, float]]:
        state = self.abstract_state(ghost, pac, ms)
        candidates: Dict[Pos, float] = {}

        counts = self._counts.get(state)
        if counts:
            total = sum(counts.values())
            ranked_actions = sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            for action, cnt in ranked_actions[:limit]:
                pred = (pac[0] + action[0], pac[1] + action[1])
                if _valid(pred, ms):
                    w = 1.5 + 3.5 * (cnt / total) + min(1.2, total / 8.0)
                    candidates[pred] = max(candidates.get(pred, 0.0), w)

        # Greedy fallback — assume Pacman moves toward ghost
        for m in MOVE_ORDER:
            cur = pac
            for step_len in (1, 2):
                cand = _apply(cur, m)
                if not _valid(cand, ms):
                    break
                w = 1.4 if _manhattan(cand, ghost) < _manhattan(pac, ghost) else 0.55
                if step_len == 2:
                    w += 0.35
                candidates[cand] = max(candidates.get(cand, 0.0), w)
                cur = cand

        # Stay
        if pac not in candidates:
            candidates[pac] = 0.4

        ranked = sorted(candidates.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:limit]
        return ranked  # List[(pos, weight)]


# ===================================================================
# Belief State (deterministic information set over Pacman positions)
# ===================================================================

class BeliefState:
    """Deterministic weighted information set over plausible Pacman positions."""

    def __init__(self, limit: int = BELIEF_PARTICLES) -> None:
        self._limit = limit
        self._belief: Dict[Pos, float] = {}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, pac: Pos, ms) -> None:
        if _valid(pac, ms):
            self._belief = {pac: 1.0}
            self._initialized = True

    def initialize_unknown(self, ghost: Pos, ms) -> None:
        """Prior for deterministic starts, with a generic bottom-map fallback."""
        priors: Dict[Pos, float] = {}

        if _valid(DEFAULT_PACMAN_START, ms):
            priors[DEFAULT_PACMAN_START] = 8.0
            for pos in _pacman_reach(DEFAULT_PACMAN_START, ms, 2):
                priors[pos] = max(priors.get(pos, 0.0), 2.5)

        h, w = _shape(ms)
        center_col = w // 2
        for r in range(max(0, int(h * 0.58)), h):
            for c in range(1, w - 1):
                pos = (r, c)
                if not _valid(pos, ms):
                    continue
                distance_bias = max(0, 8 - abs(c - center_col))
                priors[pos] = max(priors.get(pos, 0.0), 0.25 + distance_bias * 0.08)

        visible = _visible_cross_cells(ghost, ms)
        for pos in list(priors):
            if pos in visible:
                priors[pos] *= 0.05

        ranked = _normalize(priors, self._limit)
        self._belief = dict(ranked)
        self._initialized = bool(self._belief)

    def update_visible(self, pac: Pos) -> None:
        """When Pacman is visible, collapse belief to the exact position."""
        self._belief = {pac: 1.0}
        self._initialized = True

    def predict(
        self,
        usl: USLStar,
        ghost: Pos,
        ms,
        pac_speed: int = 2,
        forbidden_visible: Optional[Set[Pos]] = None,
    ) -> None:
        """Propagate belief deterministically through the Pacman action set."""
        if not self._initialized:
            self.initialize_unknown(ghost, ms)
        if not self._belief:
            return

        next_weights: Dict[Pos, float] = {}
        forbidden_visible = forbidden_visible or set()

        for pac, base_weight in self.top_hypotheses(self._limit):
            preds: Dict[Pos, float] = {pac: 0.35}

            for pos, learned_weight in usl.predict_next_positions(ghost, pac, ms, limit=USL_LIMIT):
                preds[pos] = max(preds.get(pos, 0.0), learned_weight)

            for move in MOVE_ORDER:
                cur = pac
                for step_len in range(1, max(1, pac_speed) + 1):
                    nxt = _apply(cur, move)
                    if not _valid(nxt, ms):
                        break
                    chase_gain = _manhattan(pac, ghost) - _manhattan(nxt, ghost)
                    weight = 0.85 + step_len * 0.25 + max(0, chase_gain) * 0.55
                    preds[nxt] = max(preds.get(nxt, 0.0), weight)
                    cur = nxt

            for pos, trans_weight in preds.items():
                if pos in forbidden_visible:
                    continue
                if not _valid(pos, ms):
                    continue
                los_bonus = 1.15 if _in_cross_los(pos, ghost, ms, VISION_RADIUS) else 0.9
                next_weights[pos] = next_weights.get(pos, 0.0) + base_weight * trans_weight * los_bonus

        ranked = _normalize(next_weights, self._limit)
        if ranked:
            self._belief = dict(ranked)
        else:
            self.initialize_unknown(ghost, ms)

    def top_hypotheses(self, k: int = 5) -> List[Tuple[Pos, float]]:
        if not self._belief:
            return []
        return sorted(
            self._belief.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )[:k]

    def weighted_centroid(self) -> Optional[Pos]:
        hyps = self.top_hypotheses(10)
        if not hyps:
            return None
        wr = sum(pos[0] * weight for pos, weight in hyps)
        wc = sum(pos[1] * weight for pos, weight in hyps)
        return (int(round(wr)), int(round(wc)))


# ===================================================================
# Monte Carlo Rollout Engine (POMCP-lite)
# ===================================================================

class MCRollout:
    """Deterministic information-set Monte Carlo planning for Ghost evasion."""

    def __init__(self, topo: Topology, bfs: BFSCache) -> None:
        self._topo = topo
        self._bfs  = bfs

    def _ghost_policy(self, ghost: Pos, pac: Pos, ms, prev: Optional[Pos], scenario: int) -> Tuple[Pos, Move]:
        moves = _legal(ghost, ms)
        if not moves:
            return ghost, Move.STAY
        pd = self._bfs.dist(ms, pac)
        scored: List[Tuple[float, int, Move, Pos]] = []
        for m in moves:
            nxt = _apply(ghost, m)
            d = pd.get(nxt, _manhattan(nxt, pac))
            score = d * 3.2 + _exits(nxt, ms) * 2.0
            if nxt in self._topo.core:
                score += 9.0
            if nxt in self._topo.loop_set:
                score += 8.0
            if nxt in self._topo.junctions:
                score += 6.0
            if nxt in self._topo.dead_ends:
                score -= 42.0 + self._topo.danger_depth.get(nxt, 1) * 4.0
            if _in_cross_los(pac, nxt, ms, VISION_RADIUS):
                score -= 7.0
            else:
                score += 5.0
            if prev is not None and nxt == prev:
                score -= 12.0
            if _manhattan(nxt, pac) < CAPTURE_DISTANCE:
                score -= MC_CAPTURE_PENALTY

            if scenario % 4 == 1 and nxt in self._topo.loop_set:
                score += 6.0
            elif scenario % 4 == 2 and nxt in self._topo.junctions:
                score += 5.0
            elif scenario % 4 == 3 and not _in_cross_los(pac, nxt, ms, VISION_RADIUS):
                score += 7.0

            scored.append((score, -_move_key(m), m, nxt))

        scored.sort(key=lambda item: (-item[0], -item[1], item[3][0], item[3][1]))
        _, _, move, nxt = scored[0]
        return nxt, move

    def _pacman_responses(self, pac: Pos, ghost: Pos, ms, pac_speed: int = 2) -> List[Pos]:
        gd = self._bfs.dist(ms, ghost)
        scored: List[Tuple[float, Pos]] = []
        for pos in _pacman_reach(pac, ms, pac_speed):
            d = gd.get(pos, _manhattan(pos, ghost))
            score = -d * 12.0
            if _in_cross_los(pos, ghost, ms, VISION_RADIUS):
                score += 5.0
            if _manhattan(pos, ghost) < CAPTURE_DISTANCE:
                score += 1000.0
            score -= _manhattan(pos, pac) * 0.2
            scored.append((score, pos))
        scored.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        return [pos for _, pos in scored]

    def _pacman_response(self, pac: Pos, ghost: Pos, ms, scenario: int, pac_speed: int = 2) -> Pos:
        responses = self._pacman_responses(pac, ghost, ms, pac_speed)
        if not responses:
            return pac
        branch_count = min(3, len(responses))
        return responses[min(scenario % branch_count, branch_count - 1)]

    def _leaf_score(self, ghost: Pos, pac: Pos, ms) -> float:
        pd = self._bfs.dist(ms, pac)
        d = pd.get(ghost, _manhattan(ghost, pac))
        score = d * 14.0
        score += _exits(ghost, ms) * 3.0
        if ghost in self._topo.core:
            score += 12.0
        if ghost in self._topo.loop_set:
            score += 10.0
        if ghost in self._topo.junctions:
            score += 7.0
        if _in_cross_los(pac, ghost, ms, VISION_RADIUS):
            score -= 10.0
        else:
            score += 5.0
        if ghost in self._topo.dead_ends:
            score -= 36.0 + self._topo.danger_depth.get(ghost, 1) * 5.0
        return score

    def rollout(
        self,
        ghost_start: Pos,
        first_move: Move,
        pac_start: Pos,
        ms,
        step_number: int,
        scenario: int,
        pac_speed: int = 2,
        t0: float = 0.0,
    ) -> float:
        ghost = _apply(ghost_start, first_move)
        if not _valid(ghost, ms):
            return -MC_CAPTURE_PENALTY

        pac = self._pacman_response(pac_start, ghost, ms, scenario, pac_speed)
        if _manhattan(ghost, pac) < CAPTURE_DISTANCE:
            return -MC_CAPTURE_PENALTY

        total_score = 0.0
        steps = 0
        prev_ghost = ghost_start

        for depth in range(MC_HORIZON):
            if time.time() - t0 > TIME_BUDGET * 0.92:
                break
            new_ghost, _ = self._ghost_policy(ghost, pac, ms, prev_ghost, scenario + depth)
            new_pac = self._pacman_response(pac, new_ghost, ms, scenario + depth, pac_speed)
            if _manhattan(new_ghost, new_pac) < CAPTURE_DISTANCE:
                total_score -= MC_CAPTURE_PENALTY - depth * 8000.0
                break
            total_score += self._leaf_score(new_ghost, new_pac, ms) + depth * 180.0
            steps += 1
            prev_ghost = ghost
            ghost = new_ghost
            pac = new_pac

        return total_score + steps * MC_SURVIVE_BONUS

    def evaluate_move(
        self,
        ghost: Pos,
        move: Move,
        pac_hypotheses: List[Tuple[Pos, float]],
        ms,
        step_number: int,
        t0: float,
        pac_speed: int = 2,
    ) -> Optional[float]:
        nxt = _apply(ghost, move)
        if not _valid(nxt, ms):
            return None
        total = 0.0
        total_weight = 0.0
        scenarios = min(MC_ROLLOUTS, 4 if len(pac_hypotheses) > 4 else 6)
        hypotheses = pac_hypotheses[:6]
        for hyp_index, (pac, hyp_weight) in enumerate(hypotheses):
            if time.time() - t0 > TIME_BUDGET * 0.88:
                break
            for scenario in range(scenarios):
                if time.time() - t0 > TIME_BUDGET * 0.88:
                    break
                scenario_id = step_number * 13 + hyp_index * 5 + scenario
                scenario_weight = hyp_weight / (1.0 + scenario * 0.35)
                score = self.rollout(ghost, move, pac, ms, step_number, scenario_id, pac_speed, t0)
                total += score * scenario_weight
                total_weight += scenario_weight
        return total / total_weight if total_weight > 0 else None


# ===================================================================
# Paranoid Alpha-Beta Search (visible Pacman, worst-case)
# ===================================================================

class ParanoidSearch:
    """Minimax with alpha-beta: Ghost maximises, Pacman minimises.
    Works on memory_map. Used when Pacman is visible and close."""

    def __init__(self, topo: Topology, bfs: BFSCache) -> None:
        self._topo = topo
        self._bfs  = bfs

    def _evaluate(self, ghost: Pos, pac: Pos, ms) -> float:
        if _manhattan(ghost, pac) < CAPTURE_DISTANCE:
            return -100_000.0
        pd = self._bfs.dist(ms, pac)
        d = pd.get(ghost, _manhattan(ghost, pac))
        score = d * 15.0 + _exits(ghost, ms) * 4.0
        if ghost in self._topo.core:       score += 18.0
        if ghost in self._topo.loop_set:   score += 14.0
        if ghost in self._topo.junctions:  score += 8.0
        if ghost in self._topo.dead_ends:
            score -= 50.0 + self._topo.danger_depth.get(ghost, 1) * 6.0
        return score

    def _pac_reach(self, pac: Pos, ms, speed: int = 2) -> List[Pos]:
        return _pacman_reach(pac, ms, speed)

    def search(
        self,
        ghost: Pos,
        pac: Pos,
        ms,
        max_depth: int,
        t0: float,
        pac_speed: int = 2,
        history: Optional[List[Pos]] = None,
    ) -> Tuple[float, Optional[Move]]:
        history = history or []
        memo: Dict[Tuple, float] = {}

        def ab(g: Pos, p: Pos, depth: int, alpha: float, beta: float, is_ghost: bool) -> float:
            if time.time() - t0 > TIME_BUDGET * 0.86:
                return self._evaluate(g, p, ms)
            if _manhattan(g, p) < CAPTURE_DISTANCE:
                return -100_000.0 + depth * 2000.0
            if depth <= 0:
                return self._evaluate(g, p, ms)
            key = (g, p, depth, is_ghost)
            if key in memo:
                return memo[key]

            if is_ghost:
                best = float("-inf")
                moves = _legal(g, ms)
                moves.sort(key=lambda m: (-self._bfs.dist(ms, p).get(_apply(g, m), INF), _move_key(m)))
                tried = False
                for m in moves:
                    ng = _apply(g, m)
                    if ng in history[-4:]:
                        continue
                    tried = True
                    v = ab(ng, p, depth - 1, alpha, beta, False)
                    if v > best:
                        best = v
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
                if not tried:
                    for m in moves:
                        ng = _apply(g, m)
                        v = ab(ng, p, depth - 1, alpha, beta, False)
                        if v > best:
                            best = v
                        alpha = max(alpha, best)
                        if beta <= alpha:
                            break
                memo[key] = best
                return best
            else:
                best = float("inf")
                pac_options = self._pac_reach(p, ms, pac_speed)
                pac_options.sort(
                    key=lambda pos: (
                        self._bfs.dist(ms, g).get(pos, _manhattan(pos, g)),
                        pos[0],
                        pos[1],
                    )
                )
                for np_ in pac_options:
                    v = ab(g, np_, depth - 1, alpha, beta, True)
                    if v < best:
                        best = v
                    beta = min(beta, best)
                    if beta <= alpha:
                        break
                memo[key] = best
                return best

        best_move: Optional[Move] = None
        best_score = float("-inf")
        moves = _legal(ghost, ms)
        if not moves:
            return float("-inf"), None
        pd = self._bfs.dist(ms, pac)
        moves.sort(key=lambda m: (-pd.get(_apply(ghost, m), INF), _move_key(m)))
        for m in moves:
            ng = _apply(ghost, m)
            if _manhattan(ng, pac) < CAPTURE_DISTANCE:
                continue
            v = ab(ng, pac, max_depth - 1, float("-inf"), float("inf"), False)
            if v > best_score:
                best_score = v
                best_move = m
        return best_score, best_move


# ===================================================================
# Main Ghost Agent
# ===================================================================

class GhostAgent(BaseGhostAgent):
    """
    Blind Ghost Agent — Partial Observability Lab 2.

    Decision pipeline:
    1. Update memory map & topology.
    2. Update belief state (collapse if visible, propagate if hidden).
    3. Feed USL* observations for online learning.
    4. If Pacman visible & BFS close → Paranoid Search (Alpha-Beta).
    5. Else → POMCP-lite (MC rollouts over weighted information sets).
    6. Anti-oscillation filter on final move.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pacman_speed: int = max(2, int(kwargs.get("pacman_speed", 2)))

        # Memory & belief
        self.memory_map: Optional[np.ndarray] = None
        self._map_hash: int = 0
        self._topo = Topology()
        self._topo_built = False

        # Engines
        self._bfs   = BFSCache(maxsize=200)
        self._usl   = USLStar()
        self._mc    = MCRollout(self._topo, self._bfs)
        self._ab    = ParanoidSearch(self._topo, self._bfs)
        self._belief = BeliefState(BELIEF_PARTICLES)

        # Tracking
        self._last_pac: Optional[Pos]  = None
        self._ghost_hist: deque[Pos]   = deque(maxlen=24)
        self._pac_hist:   deque[Pos]   = deque(maxlen=40)
        self._last_move: Optional[Move] = None
        self._steps_since_seen: int    = 0
        self._initial_dist: Optional[int] = None
        self._initial_far: bool = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number: int) -> Move:
        t0 = time.time()
        map_state = np.asarray(map_state, dtype=int)
        me: Pos = tuple(int(v) for v in my_position)

        # 1. Update memory map
        self._update_memory(map_state)
        ms = self.memory_map

        # 2. Build topology on first call (or when map grows)
        self._maybe_rebuild_topology(ms)

        # 3. Legal moves
        legal = _legal(me, ms)
        if not legal:
            return Move.STAY

        # 4. Enemy tracking & belief update
        if enemy_position is not None:
            pac: Pos = tuple(int(v) for v in enemy_position)
            self._steps_since_seen = 0

            # USL* observation
            if self._last_pac is not None:
                action = (pac[0] - self._last_pac[0], pac[1] - self._last_pac[1])
                state  = self._usl.abstract_state(me, self._last_pac, ms)
                self._usl.observe(state, action, ms, self._last_pac)

            self._belief.update_visible(pac)
            self._pac_hist.append(pac)
            self._last_pac = pac

            if self._initial_dist is None:
                self._initial_dist = _manhattan(me, pac)
                self._initial_far  = self._initial_dist >= 10
        else:
            self._steps_since_seen += 1
            if not self._belief.initialized:
                self._belief.initialize_unknown(me, ms)
            visible_now = _visible_cross_cells(me, ms, VISION_RADIUS)
            self._belief.predict(
                self._usl,
                me,
                ms,
                self._pacman_speed,
                forbidden_visible=visible_now,
            )
            pac = None

        # 5. Track ghost history & initial distance update
        self._ghost_hist.append(me)

        # 6. Get Pacman hypotheses from belief state
        pac_hypotheses = self._belief.top_hypotheses(k=6)
        if not pac_hypotheses:
            self._belief.initialize_unknown(me, ms)
            pac_hypotheses = self._belief.top_hypotheses(k=6)
        if not pac_hypotheses:
            pac_hypotheses = [(DEFAULT_PACMAN_START if _valid(DEFAULT_PACMAN_START, ms) else me, 1.0)]

        # 8. No enemy visible → exploration + belief-guided evasion
        if pac is None:
            move = self._blind_evasion(me, pac_hypotheses, legal, step_number, t0)
            self._last_move = move
            return move

        # 9. BFS distance from Pacman
        pd = self._bfs.dist(ms, pac)
        bfs_d = pd.get(me, _manhattan(me, pac))

        # 10. Panic: immediately adjacent → best escape ignoring time
        if bfs_d < CAPTURE_DISTANCE:
            move = self._panic_escape(me, pac, pd, legal)
            self._last_move = move
            return move

        # 11. Paranoid search when Pacman is close and visible
        if bfs_d <= PANIC_DISTANCE:
            _, ab_move = self._ab.search(
                me, pac, ms,
                max_depth=AB_MAX_DEPTH,
                t0=t0,
                pac_speed=self._pacman_speed,
                history=list(self._ghost_hist),
            )
            if ab_move is not None:
                recent = set(list(self._ghost_hist)[-6:])
                direct_move = self._best_scored_move(me, legal, [(pac, 1.0)], ms, recent)
                ab_score = self._position_safety_score(_apply(me, ab_move), [(pac, 1.0)], ms, recent)
                direct_score = self._position_safety_score(_apply(me, direct_move), [(pac, 1.0)], ms, recent)
                if direct_score > ab_score + 10.0:
                    ab_move = direct_move
                # Verify MC agrees (don't use AB if MC finds clearly better)
                ab_move = self._mc_sanity_check(me, pac, pac_hypotheses, ab_move, legal, ms, t0)
                self._last_move = ab_move
                return ab_move

        # 12. Strategic: MC rollouts over belief state
        move = self._mc_strategic(me, pac_hypotheses, legal, ms, step_number, t0)
        self._last_move = move
        return move

    # ------------------------------------------------------------------
    # Memory map
    # ------------------------------------------------------------------
    def _update_memory(self, map_state) -> None:
        map_state = np.asarray(map_state, dtype=int)
        if self.memory_map is None:
            self.memory_map = np.full_like(map_state, -1, dtype=int)
        visible = (map_state != -1)
        # Count newly revealed cells
        newly_revealed = int(np.sum(visible & (self.memory_map == -1)))
        self.memory_map[visible] = map_state[visible]
        # Rebuild topology only when significant new info is gained
        if newly_revealed > 2:
            self._map_hash += 1
            self._bfs._map_hash = self._map_hash
            self._bfs.invalidate()
            self._topo_built = False

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def _maybe_rebuild_topology(self, ms) -> None:
        if not self._topo_built:
            self._topo.build(ms)
            self._mc  = MCRollout(self._topo, self._bfs)
            self._ab  = ParanoidSearch(self._topo, self._bfs)
            self._topo_built = True

    # ------------------------------------------------------------------
    # Shared safety evaluation
    # ------------------------------------------------------------------
    def _position_safety_score(
        self,
        pos: Pos,
        pac_hypotheses: List[Tuple[Pos, float]],
        ms,
        recent: Optional[Set[Pos]] = None,
    ) -> float:
        score = 0.0
        recent = recent or set()

        for pac_pos, weight in pac_hypotheses[:6]:
            pd = self._bfs.dist(ms, pac_pos)
            d = pd.get(pos, _manhattan(pos, pac_pos))
            score += weight * d * 13.5
            score -= weight * max(0, 7 - d) * 5.5

            pac_reach = _pacman_reach(pac_pos, ms, self._pacman_speed)
            reach_gap = min((_manhattan(pos, p) for p in pac_reach), default=INF)
            if reach_gap < CAPTURE_DISTANCE:
                score -= weight * MC_CAPTURE_PENALTY * 0.42
            elif reach_gap <= 2:
                score -= weight * 600.0

            if _in_cross_los(pac_pos, pos, ms, VISION_RADIUS):
                score -= weight * 20.0
            else:
                score += weight * 7.0

        exits = _exits(pos, ms)
        score += exits * 5.0
        h, w = _shape(ms)
        edge_depth = min(pos[0], pos[1], h - 1 - pos[0], w - 1 - pos[1])
        closest_hyp = min(
            (self._bfs.dist(ms, pac_pos).get(pos, _manhattan(pos, pac_pos)) for pac_pos, _ in pac_hypotheses[:6]),
            default=INF,
        )
        if edge_depth <= 1 and exits <= 2:
            score -= 28.0
            if closest_hyp <= 6:
                score -= 45.0
        if closest_hyp <= 5 and exits <= 2:
            score -= 38.0
        if pos in self._topo.core:
            score += 22.0
        if pos in self._topo.loop_set:
            score += 18.0
        if pos in self._topo.junctions:
            score += 12.0
        if pos in self._topo.dead_ends:
            score -= 72.0 + self._topo.danger_depth.get(pos, 1) * 8.0
        if pos in recent:
            score -= 24.0
            hist = list(self._ghost_hist)
            if len(hist) >= 2 and pos == hist[-2]:
                score -= 14.0
        return score

    def _best_scored_move(
        self,
        me: Pos,
        legal: List[Move],
        pac_hypotheses: List[Tuple[Pos, float]],
        ms,
        recent: Optional[Set[Pos]] = None,
    ) -> Move:
        best_move = legal[0] if legal else Move.STAY
        best_score = float("-inf")
        for move in legal:
            nxt = _apply(me, move)
            score = self._position_safety_score(nxt, pac_hypotheses, ms, recent)
            if self._last_move is not None and move == self._last_move:
                score += 2.0
            if (
                self._last_move is not None
                and move.value[0] + self._last_move.value[0] == 0
                and move.value[1] + self._last_move.value[1] == 0
            ):
                score -= 8.0
            key = (score, -_move_key(move), -nxt[0], -nxt[1])
            best_key = (best_score, -_move_key(best_move), -_apply(me, best_move)[0], -_apply(me, best_move)[1])
            if key > best_key:
                best_score = score
                best_move = move
        return best_move

    def _should_hold_stealth(
        self,
        me: Pos,
        pac_hypotheses: List[Tuple[Pos, float]],
        ms,
        step_number: int,
    ) -> bool:
        if step_number > 100:
            return False
        if me in self._topo.dead_ends or _exits(me, ms) < 2:
            return False
        min_dist = INF
        los_risk = 0.0
        reach_risk = 0.0
        for pac_pos, weight in pac_hypotheses[:6]:
            d = self._bfs.dist(ms, pac_pos).get(me, _manhattan(me, pac_pos))
            min_dist = min(min_dist, d)
            if _in_cross_los(pac_pos, me, ms, VISION_RADIUS):
                los_risk += weight
            pac_reach = _pacman_reach(pac_pos, ms, self._pacman_speed)
            if any(_manhattan(me, pos) < CAPTURE_DISTANCE for pos in pac_reach):
                reach_risk += weight
        return min_dist >= 14 and los_risk < 0.18 and reach_risk == 0.0

    def _usl_start_move(self, me: Pos, legal: List[Move], step_number: int) -> Optional[Move]:
        """Deterministic opening for the classic start, then hand off to search."""
        if step_number > 5:
            return None
        opening = {
            (9, 10): Move.RIGHT,
            (9, 11): Move.RIGHT,
            (9, 12): Move.RIGHT,
            (9, 13): Move.DOWN,
            (10, 13): Move.DOWN,
        }
        move = opening.get(me)
        if move in legal:
            return move
        return None

    # ------------------------------------------------------------------
    # Panic: immediate danger
    # ------------------------------------------------------------------
    def _panic_escape(self, me: Pos, pac: Pos, pd: Dict[Pos, int], legal: List[Move]) -> Move:
        _ = pd
        recent = set(list(self._ghost_hist)[-4:])
        return self._best_scored_move(me, legal, [(pac, 1.0)], self.memory_map, recent)

    # ------------------------------------------------------------------
    # MC sanity check: compare AB move vs MC best
    # ------------------------------------------------------------------
    def _mc_sanity_check(
        self,
        me: Pos,
        pac: Pos,
        pac_hypotheses: List[Tuple[Pos, float]],
        ab_move: Move,
        legal: List[Move],
        ms,
        t0: float,
    ) -> Move:
        if time.time() - t0 > TIME_BUDGET * 0.78:
            return ab_move
        mc_move = self._mc_strategic(me, pac_hypotheses, legal, ms, 0, t0, fast=True)
        if mc_move is None or mc_move == ab_move:
            return ab_move
        ab_pos = _apply(me, ab_move)
        mc_pos = _apply(me, mc_move)
        recent = set(list(self._ghost_hist)[-6:])
        ab_score = self._position_safety_score(ab_pos, pac_hypotheses, ms, recent)
        mc_score = self._position_safety_score(mc_pos, pac_hypotheses, ms, recent)
        if mc_score > ab_score + 16.0 and mc_pos not in self._topo.dead_ends:
            return mc_move
        return ab_move

    # ------------------------------------------------------------------
    # Blind evasion: Pacman not visible, use belief state
    # ------------------------------------------------------------------
    def _blind_evasion(
        self,
        me: Pos,
        pac_hypotheses: List[Tuple[Pos, float]],
        legal: List[Move],
        step_number: int,
        t0: float,
    ) -> Move:
        ms = self.memory_map
        if not legal:
            return Move.STAY

        opening_move = self._usl_start_move(me, legal, step_number)
        if opening_move is not None:
            return opening_move

        if self._should_hold_stealth(me, pac_hypotheses, ms, step_number):
            return Move.STAY

        recent = set(list(self._ghost_hist)[-6:])
        best_move = self._best_scored_move(me, legal, pac_hypotheses, ms, recent)

        # During the first 100 deterministic steps, survival outranks map coverage.
        if step_number > 100 and self._steps_since_seen >= 24:
            frontier_move = self._frontier_explore(me, legal, ms, pac_hypotheses)
            frontier_pos = _apply(me, frontier_move)
            best_pos = _apply(me, best_move)
            frontier_score = self._position_safety_score(frontier_pos, pac_hypotheses, ms, recent)
            best_score = self._position_safety_score(best_pos, pac_hypotheses, ms, recent)
            if frontier_score >= best_score - 12.0:
                best_move = frontier_move

        if time.time() - t0 < TIME_BUDGET * 0.55:
            mc_m = self._mc_strategic(me, pac_hypotheses, legal, ms, 0, t0, fast=True)
            if mc_m is not None and mc_m != best_move:
                mc_pos = _apply(me, mc_m)
                cur_pos = _apply(me, best_move)
                mc_score = self._position_safety_score(mc_pos, pac_hypotheses, ms, recent)
                cur_score = self._position_safety_score(cur_pos, pac_hypotheses, ms, recent)
                if mc_score > cur_score + 8.0:
                    best_move = mc_m

        return best_move

    # ------------------------------------------------------------------
    # Frontier exploration (when blind for many steps)
    # ------------------------------------------------------------------
    def _frontier_explore(
        self,
        me: Pos,
        legal: List[Move],
        ms,
        pac_hypotheses: List[Tuple[Pos, float]],
    ) -> Move:
        """Move toward cells at the boundary between known and unknown."""
        h, w = _shape(ms)
        frontier_target: Optional[Pos] = None
        best_score = float("-inf")
        dist_from_me = self._bfs.dist(ms, me)
        recent = set(list(self._ghost_hist)[-6:])

        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) != 0:
                    continue
                pos = (r, c)
                for m in MOVE_ORDER:
                    nb = _apply(pos, m)
                    nr, nc = nb
                    if 0 <= nr < h and 0 <= nc < w and _cell(ms, nr, nc) == -1:
                        d = dist_from_me.get(pos, _manhattan(pos, me))
                        safety = self._position_safety_score(pos, pac_hypotheses, ms, recent)
                        score = safety - d * 3.0
                        if score > best_score:
                            best_score = score
                            frontier_target = pos
                        break

        if frontier_target is None:
            return self._best_scored_move(me, legal, pac_hypotheses, ms, recent)
        path = astar(ms, me, frontier_target)
        if path and path[0] in legal:
            return path[0]
        return self._best_scored_move(me, legal, pac_hypotheses, ms, recent)

    # ------------------------------------------------------------------
    # MC strategic: evaluate candidates via POMCP-lite
    # ------------------------------------------------------------------
    def _mc_strategic(
        self,
        me: Pos,
        pac_hypotheses: List[Tuple[Pos, float]],
        legal: List[Move],
        ms,
        step_number: int,
        t0: float,
        fast: bool = False,
    ) -> Optional[Move]:
        if not legal:
            return None

        recent = set(list(self._ghost_hist)[-HISTORY_LEN:])
        # Pre-filter obviously bad moves
        candidates = [m for m in legal if _apply(me, m) not in self._topo.dead_ends] or legal

        # Greedy pre-sort: prefer moves away from top Pacman hypothesis
        top_pac = pac_hypotheses[0][0] if pac_hypotheses else me
        pd_top = self._bfs.dist(ms, top_pac)
        candidates.sort(key=lambda m: (-pd_top.get(_apply(me, m), 0), _move_key(m)))

        if fast:
            candidates = candidates[:3]

        best_move: Optional[Move] = None
        best_combined = float("-inf")

        for m in candidates:
            if time.time() - t0 > TIME_BUDGET * 0.84:
                break
            nxt = _apply(me, m)

            # MC rollout score
            mc_score = self._mc.evaluate_move(me, m, pac_hypotheses, ms, step_number, t0, self._pacman_speed)
            if mc_score is None:
                continue

            det_score = self._position_safety_score(nxt, pac_hypotheses, ms, recent)

            combined = mc_score * 0.80 + det_score * 0.20
            if (
                combined > best_combined
                or (
                    combined == best_combined
                    and best_move is not None
                    and _move_key(m) < _move_key(best_move)
                )
            ):
                best_combined = combined
                best_move = m

        return best_move


# ===================================================================
# PacmanAgent — placeholder (primary deliverable is GhostAgent)
# ===================================================================

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self._memory_map: Optional[np.ndarray] = None
        self.last_seen_enemy: Optional[Pos] = None

    def _update_memory(self, map_state) -> None:
        map_state = np.asarray(map_state, dtype=int)
        if self._memory_map is None:
            self._memory_map = np.full_like(map_state, -1, dtype=int)
        visible_mask = (map_state != -1)
        self._memory_map[visible_mask] = map_state[visible_mask]

    def step(self, map_state, my_position, enemy_position, step_number):
        map_state = np.asarray(map_state, dtype=int)
        self._update_memory(map_state)
        if enemy_position is not None:
            self.last_seen_enemy = tuple(int(v) for v in enemy_position)
        return (Move.STAY, 1)
