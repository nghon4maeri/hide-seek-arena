"""Ghost/Hider Agent sandbox for student 24127192.

Role: Ghost/Hider Engineer
Version: V3.5
Focus: survive as long as possible by combining:
- US-L* opponent learning with confidence-weighted predictions
- exact map knowledge and exact visible Pacman position
- greedy best-first move ordering
- A* / BFS shortest-path opponent modelling with Manhattan heuristic
- iterative deepening maximin search under multiple Pacman hypotheses
"""

from __future__ import annotations

import sys
import time
import heapq
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

MOVE_ORDER: Tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
TIME_BUDGET = 0.95
INF_DIST = 10**9
CAPTURE_DISTANCE = 2
OPENING_LEARN_STEPS = 20

Pos = Tuple[int, int]
Action = Tuple[int, int]


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


def _ghost_legal(pos: Pos, ms) -> List[Move]:
    return _legal(pos, ms)


def _neighbors(pos: Pos, ms) -> List[Pos]:
    return [_apply(pos, m) for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bucket(v: int) -> int:
    if v <= -6:
        return -4
    if v <= -3:
        return -3
    if v < 0:
        return -1
    if v == 0:
        return 0
    if v < 3:
        return 1
    if v < 6:
        return 3
    return 4


class Topo:
    """Map topology features used by the ghost evaluator.

    deb: cells that belong to dead-end branches or narrow cul-de-sacs.
    junctions: cells with at least three exits.
    corridor: cells with exactly two exits.
    """

    def __init__(self) -> None:
        self.deb: Set[Pos] = set()
        self.junctions: Set[Pos] = set()
        self.corridor: Set[Pos] = set()
        self.ok = False

    def init(self, ms) -> None:
        if self.ok:
            return
        h, w = _shape(ms)
        dead_seeds: Set[Pos] = set()

        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1:
                    continue
                p = (r, c)
                deg = len(_neighbors(p, ms))
                if deg <= 1:
                    dead_seeds.add(p)
                elif deg >= 3:
                    self.junctions.add(p)
                else:
                    self.corridor.add(p)

        for seed in dead_seeds:
            cur = seed
            prev: Optional[Pos] = None
            seen = {cur}
            while True:
                self.deb.add(cur)
                if cur in self.junctions:
                    break
                nxts = [x for x in _neighbors(cur, ms) if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt in seen:
                    break
                prev, cur = cur, nxt
                seen.add(cur)

        self.ok = True


class USLStar:
    """Small online learner for Pacman movement.

    V3 memorised the last action per abstract state. V4 keeps action counts,
    confidence, and exposes ranked candidate next positions. This makes the
    ghost robust: a highly repeated Pacman pattern can dominate, while unseen
    states still fall back to exact search models.
    """

    def __init__(self) -> None:
        self._counts: Dict[Tuple[int, ...], Counter[Action]] = defaultdict(Counter)
        self._global: Counter[Action] = Counter()

    def get_state(self, ms, ghost_pos: Pos, pac_pos: Pos, last_ghost_pos: Optional[Pos]) -> Tuple[int, ...]:
        dx = _bucket(ghost_pos[0] - pac_pos[0])
        dy = _bucket(ghost_pos[1] - pac_pos[1])
        dist_bucket = min(9, _manhattan(ghost_pos, pac_pos) // 2)

        hdr = ghost_pos[0] - last_ghost_pos[0] if last_ghost_pos else 0
        hdc = ghost_pos[1] - last_ghost_pos[1] if last_ghost_pos else 0
        hdr = max(-1, min(1, hdr))
        hdc = max(-1, min(1, hdc))

        pac_moves = _legal(pac_pos, ms)
        deg = len(pac_moves)
        if deg <= 1:
            geo = 0
        elif deg >= 3:
            geo = 3
        else:
            m1, m2 = pac_moves[0], pac_moves[1]
            opposite = m1.value[0] + m2.value[0] == 0 and m1.value[1] + m2.value[1] == 0
            geo = 1 if opposite else 2

        walls = sum((1 << i) for i, m in enumerate(MOVE_ORDER) if not _valid(_apply(pac_pos, m), ms))
        return (dx, dy, dist_bucket, hdr, hdc, geo, walls)

    def observe(self, state: Tuple[int, ...], action: Action, ms, from_pos: Pos) -> None:
        to_pos = (from_pos[0] + action[0], from_pos[1] + action[1])
        if not _valid(to_pos, ms):
            return
        self._counts[state][action] += 1
        self._global[action] += 1

    def ranked_predictions(
        self,
        ms,
        pac_pos: Pos,
        ghost_pos: Pos,
        last_ghost_pos: Optional[Pos],
        fallback_positions: Iterable[Pos],
        limit: int = 5,
    ) -> List[Tuple[Pos, float, str]]:
        state = self.get_state(ms, ghost_pos, pac_pos, last_ghost_pos)
        candidates: Dict[Pos, Tuple[float, str]] = {}

        counts = self._counts.get(state)
        if counts:
            total = sum(counts.values())
            for action, cnt in counts.most_common(limit):
                pred = (pac_pos[0] + action[0], pac_pos[1] + action[1])
                if _valid(pred, ms):
                    # confidence grows with repeated evidence but never becomes absolute
                    weight = 1.6 + 4.0 * (cnt / total) + min(1.5, total / 10.0)
                    candidates[pred] = (max(candidates.get(pred, (0.0, ""))[0], weight), "usl")

        for idx, pred in enumerate(fallback_positions):
            if _valid(pred, ms):
                weight = 2.2 - idx * 0.18
                old = candidates.get(pred)
                if old is None or weight > old[0]:
                    candidates[pred] = (weight, "model")

        if not candidates and self._global:
            total = sum(self._global.values())
            for action, cnt in self._global.most_common(limit):
                pred = (pac_pos[0] + action[0], pac_pos[1] + action[1])
                if _valid(pred, ms):
                    candidates[pred] = (1.0 + cnt / total, "global")

        if not candidates:
            candidates[pac_pos] = (1.0, "stay")

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:limit]
        return [(pos, weight, tag) for pos, (weight, tag) in ranked]


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tp = Topo()
        self._core: Set[Pos] = set()
        self._loop_set: Set[Pos] = set()
        self._usl = USLStar()
        self._pacman_speed = max(2, int(kwargs.get("pacman_speed", 2)))
        self._ghost_hist: deque[Pos] = deque(maxlen=80)
        self._pac_hist: deque[Pos] = deque(maxlen=80)
        self._last_pp: Optional[Pos] = None
        self._static_map: Tuple[Tuple[int, ...], ...] = tuple()
        self._h = 0
        self._w = 0
        self._open_cells: Set[Pos] = set()
        self._degree: Dict[Pos, int] = {}
        self._danger_depth: Dict[Pos, int] = {}
        self._junction_dist: Dict[Pos, int] = {}
        self._loop_dist: Dict[Pos, int] = {}
        self._center: Pos = (0, 0)
        self._last_pac_action: Action = (0, 0)
        self._last_manhattan: Optional[int] = None
        self._initial_distance: Optional[int] = None
        self._initial_far = False

    def _ensure_static(self, ms) -> None:
        if self._static_map:
            return
        self._static_map = tuple(tuple(int(_cell(ms, r, c)) for c in range(_shape(ms)[1])) for r in range(_shape(ms)[0]))
        self._h, self._w = _shape(self._static_map)
        self._open_cells = {(r, c) for r in range(self._h) for c in range(self._w) if self._static_map[r][c] != 1}
        self._degree = {p: len(_neighbors(p, self._static_map)) for p in self._open_cells}
        self._center = (self._h // 2, self._w // 2)
        self._tp.init(self._static_map)
        self._compute_core()
        self._dfs_find_loops()
        self._compute_dead_depth()
        self._compute_junction_dist()
        self._compute_loop_dist()
        self._bfs_dist.cache_clear()
        self._astar_path.cache_clear()
        self._capture_eta.cache_clear()
        self._safe_area.cache_clear()

    def _compute_core(self) -> None:
        active = set(self._open_cells)
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for cell in active:
                deg = sum(1 for nxt in _neighbors(cell, self._static_map) if nxt in active)
                if deg <= 1:
                    to_remove.add(cell)
            if to_remove:
                active -= to_remove
                changed = True
        self._core = active

    def _dfs_find_loops(self) -> None:
        self._loop_set = set()
        if not self._core:
            return
        cycles: List[List[Pos]] = []
        vis: Set[Pos] = set()
        parent: Dict[Pos, Optional[Pos]] = {}

        for start in list(self._core):
            if start in vis:
                continue
            stack = [(start, None, iter(_neighbors(start, self._static_map)))]
            vis.add(start)
            parent[start] = None
            while stack:
                u, p, it = stack[-1]
                try:
                    v = next(it)
                except StopIteration:
                    stack.pop()
                    continue
                if v not in self._core:
                    continue
                if v not in vis:
                    vis.add(v)
                    parent[v] = u
                    stack.append((v, u, iter(_neighbors(v, self._static_map))))
                elif v != p:
                    cycle = [v]
                    cur = u
                    while cur is not None and cur != v:
                        cycle.append(cur)
                        cur = parent.get(cur)
                    if cur == v and len(cycle) >= 4:
                        cycles.append(cycle)

        if cycles:
            self._loop_set = set(max(cycles, key=len))

    def _compute_dead_depth(self) -> None:
        self._danger_depth = {p: 0 for p in self._open_cells}
        q: deque[Pos] = deque()
        trimmed = set()
        degree = dict(self._degree)
        for p, deg in degree.items():
            if deg <= 1:
                q.append(p)
                trimmed.add(p)
                self._danger_depth[p] = 1

        while q:
            u = q.popleft()
            for v in _neighbors(u, self._static_map):
                if v in trimmed:
                    continue
                degree[v] -= 1
                if degree[v] <= 1:
                    trimmed.add(v)
                    self._danger_depth[v] = self._danger_depth[u] + 1
                    q.append(v)

    def _compute_junction_dist(self) -> None:
        self._junction_dist = {p: INF_DIST for p in self._open_cells}
        q: deque[Pos] = deque()
        starts = set(self._tp.junctions) or set(self._core)
        for p in starts:
            if p in self._junction_dist:
                self._junction_dist[p] = 0
                q.append(p)

        while q:
            cur = q.popleft()
            for nxt in _neighbors(cur, self._static_map):
                if self._junction_dist.get(nxt, INF_DIST) > self._junction_dist[cur] + 1:
                    self._junction_dist[nxt] = self._junction_dist[cur] + 1
                    q.append(nxt)

    def _compute_loop_dist(self) -> None:
        self._loop_dist = {p: INF_DIST for p in self._open_cells}
        q: deque[Pos] = deque()
        starts = set(self._loop_set) or set(self._core) or set(self._tp.junctions)
        for p in starts:
            if p in self._loop_dist:
                self._loop_dist[p] = 0
                q.append(p)

        while q:
            cur = q.popleft()
            for nxt in _neighbors(cur, self._static_map):
                if self._loop_dist.get(nxt, INF_DIST) > self._loop_dist[cur] + 1:
                    self._loop_dist[nxt] = self._loop_dist[cur] + 1
                    q.append(nxt)

    @lru_cache(maxsize=4096)
    def _bfs_dist(self, start: Pos) -> Dict[Pos, int]:
        d = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in _neighbors(cur, self._static_map):
                if nxt not in d:
                    d[nxt] = d[cur] + 1
                    q.append(nxt)
        return d

    def _dist(self, a: Pos, b: Pos) -> int:
        return self._bfs_dist(a).get(b, INF_DIST)

    @lru_cache(maxsize=100000)
    def _astar_path(self, start: Pos, target: Pos) -> Tuple[Pos, ...]:
        if start == target:
            return ()
        open_set: List[Tuple[int, int, Pos]] = []
        heapq.heappush(open_set, (_manhattan(start, target), 0, start))
        came_from: Dict[Pos, Pos] = {}
        g_score = {start: 0}
        closed: Set[Pos] = set()
        push_id = 1

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            if current == target:
                path: List[Pos] = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return tuple(path)
            closed.add(current)

            # A* with Manhattan heuristic; order neighbours greedily toward target.
            nxts = _neighbors(current, self._static_map)
            nxts.sort(key=lambda p: _manhattan(p, target))
            for neighbor in nxts:
                tentative = g_score[current] + 1
                if tentative < g_score.get(neighbor, INF_DIST):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score = tentative + _manhattan(neighbor, target)
                    heapq.heappush(open_set, (f_score, push_id, neighbor))
                    push_id += 1
        return ()

    def _astar_next(self, start: Pos, target: Pos) -> Pos:
        path = self._astar_path(start, target)
        return path[0] if path else start

    def _follow_path_with_speed(self, start: Pos, path: Tuple[Pos, ...]) -> Pos:
        if not path:
            return start
        first = path[0]
        dr = first[0] - start[0]
        dc = first[1] - start[1]
        pos = start
        used = 0
        for nxt in path:
            if used >= self._pacman_speed:
                break
            if (nxt[0] - pos[0], nxt[1] - pos[1]) != (dr, dc):
                break
            pos = nxt
            used += 1
        return pos

    def _astar_speed_next(self, start: Pos, target: Pos) -> Pos:
        return self._follow_path_with_speed(start, self._astar_path(start, target))

    def _greedy_best_first_next(self, start: Pos, target: Pos) -> Pos:
        moves = _legal(start, self._static_map)
        if not moves:
            return start
        return min((_apply(start, m) for m in moves), key=lambda p: (_manhattan(p, target), self._dist(p, target)))

    def _straight_progress(self, start: Pos, first: Pos, target: Pos) -> Pos:
        dr = first[0] - start[0]
        dc = first[1] - start[1]
        pos = first
        last_d = self._dist(pos, target)
        for _ in range(1, self._pacman_speed):
            nxt = (pos[0] + dr, pos[1] + dc)
            if not _valid(nxt, self._static_map):
                break
            nxt_d = self._dist(nxt, target)
            if nxt_d >= last_d:
                break
            pos = nxt
            last_d = nxt_d
        return pos

    def _greedy_speed_next(self, start: Pos, target: Pos) -> Pos:
        first = self._greedy_best_first_next(start, target)
        return self._straight_progress(start, first, target) if first != start else start

    def _exact_bfs_chaser_next(self, pac: Pos, ghost: Pos) -> Pos:
        moves = _legal(pac, self._static_map)
        if not moves:
            return pac
        ghost_dist = self._bfs_dist(ghost)
        return min((_apply(pac, m) for m in moves), key=lambda p: (ghost_dist.get(p, INF_DIST), _manhattan(p, ghost)))

    def _exact_bfs_speed_next(self, pac: Pos, ghost: Pos) -> Pos:
        first = self._exact_bfs_chaser_next(pac, ghost)
        return self._straight_progress(pac, first, ghost) if first != pac else pac

    @lru_cache(maxsize=100000)
    def _capture_eta(self, start: Pos, target: Pos) -> int:
        if _manhattan(start, target) < CAPTURE_DISTANCE:
            return 0
        path = self._astar_path(start, target)
        if not path:
            return INF_DIST

        pos = start
        idx = 0
        turns = 0
        while idx < len(path):
            first = path[idx]
            dr = first[0] - pos[0]
            dc = first[1] - pos[1]
            used = 0
            while idx < len(path) and used < self._pacman_speed:
                nxt = path[idx]
                if (nxt[0] - pos[0], nxt[1] - pos[1]) != (dr, dc):
                    break
                pos = nxt
                idx += 1
                used += 1
                if _manhattan(pos, target) < CAPTURE_DISTANCE:
                    return turns + 1
            turns += 1
        return turns

    @lru_cache(maxsize=20000)
    def _safe_area(self, ghost: Pos, pac_positions: Tuple[Pos, ...]) -> float:
        total = 0.0
        q: deque[Pos] = deque([ghost])
        seen = {ghost: 0}
        while q:
            cur = q.popleft()
            gd = seen[cur]
            if gd > 14:
                continue
            pac_eta = min((self._capture_eta(p, cur) for p in pac_positions), default=INF_DIST)
            margin = pac_eta - gd
            if margin > 0:
                total += 1.0 + min(3, margin) * 0.35 + self._degree.get(cur, 0) * 0.08
                for nxt in _neighbors(cur, self._static_map):
                    if nxt not in seen:
                        seen[nxt] = gd + 1
                        q.append(nxt)
        return total

    def _intercept_target(self, ghost: Pos, last_ghost: Optional[Pos]) -> Pos:
        if last_ghost is None:
            return ghost
        dr = ghost[0] - last_ghost[0]
        dc = ghost[1] - last_ghost[1]
        target = (ghost[0] + dr, ghost[1] + dc)
        return target if _valid(target, self._static_map) else ghost

    def _pacman_model_positions(self, pac: Pos, ghost: Pos, last_ghost: Optional[Pos]) -> List[Pos]:
        intercept = self._intercept_target(ghost, last_ghost)
        models = [
            self._astar_speed_next(pac, intercept),
            self._astar_speed_next(pac, ghost),
            self._exact_bfs_speed_next(pac, intercept),
            self._exact_bfs_speed_next(pac, ghost),
            self._greedy_speed_next(pac, intercept),
            self._greedy_speed_next(pac, ghost),
            pac,
        ]
        out: List[Pos] = []
        seen: Set[Pos] = set()
        for p in models:
            if p not in seen and _valid(p, self._static_map):
                seen.add(p)
                out.append(p)
        return out

    def _ranked_pac_predictions(self, pac: Pos, ghost: Pos, last_ghost: Optional[Pos]) -> List[Tuple[Pos, float, str]]:
        fallbacks = self._pacman_model_positions(pac, ghost, last_ghost)
        return self._usl.ranked_predictions(self._static_map, pac, ghost, last_ghost, fallbacks, limit=5)

    def _allow_stay(self, ghost: Pos, pac: Pos, step_number: int) -> bool:
        far_enough = _manhattan(ghost, pac) >= max(self._h, self._w) * 0.5
        return step_number > OPENING_LEARN_STEPS or far_enough or (self._initial_far and step_number <= OPENING_LEARN_STEPS)

    def _phase_moves(self, ghost: Pos, pac: Pos, step_number: int) -> List[Move]:
        moves = _legal(ghost, self._static_map)
        if self._allow_stay(ghost, pac, step_number):
            moves = moves + [Move.STAY]
        return moves

    def _move_safely_changes_distance(self, ghost: Pos, pac: Pos, move: Move) -> bool:
        nxt = _apply(ghost, move)
        return _manhattan(nxt, pac) != _manhattan(ghost, pac)

    def _opening_spread_move(
        self,
        ghost: Pos,
        pac: Pos,
        pac_predictions: List[Tuple[Pos, float, str]],
        last_ghost: Optional[Pos],
        step_number: int,
    ) -> Move:
        moves = self._phase_moves(ghost, pac, step_number)
        if not moves:
            return Move.STAY

        current_real = _manhattan(ghost, pac)
        pac_dist = self._bfs_dist(pac)

        def score(move: Move) -> float:
            nxt = _apply(ghost, move)
            worst_pred_dist = min(self._dist(nxt, p) for p, _, _ in pac_predictions)
            worst_eta = min(self._capture_eta(p, nxt) for p, _, _ in pac_predictions)
            real_gain = _manhattan(nxt, pac) - current_real
            bfs_gain = pac_dist.get(nxt, INF_DIST) - pac_dist.get(ghost, INF_DIST)
            s = self._cell_safety_score(nxt, pac_predictions)
            s += worst_pred_dist * 1450.0 + min(worst_eta, 8) * 1900.0
            s += real_gain * 2400.0 + bfs_gain * 1800.0
            s += self._safe_area(nxt, tuple(p for p, _, _ in pac_predictions[:4])) * 110.0
            s += self._degree.get(nxt, 0) * 360.0
            s += max(0, 4 - self._junction_dist.get(nxt, INF_DIST)) * 520.0
            if nxt in self._core:
                s += 1700.0
            if nxt in self._tp.deb:
                s -= 9000.0 + self._danger_depth.get(nxt, 1) * 900.0
            if last_ghost is not None and nxt == last_ghost:
                s -= 5200.0
            if len(self._ghost_hist) >= 2 and nxt == self._ghost_hist[-2]:
                s -= 3200.0
            if move == Move.STAY:
                s -= 5000.0
            return s

        return max(moves, key=score)

    def _far_reading_move(
        self,
        ghost: Pos,
        pac: Pos,
        pac_predictions: List[Tuple[Pos, float, str]],
        last_ghost: Optional[Pos],
        step_number: int,
    ) -> Move:
        moves = self._phase_moves(ghost, pac, step_number)
        if not moves:
            return Move.STAY

        current_dist = _manhattan(ghost, pac)
        pac_key = tuple(p for p, _, _ in pac_predictions[:4])

        def score(move: Move) -> float:
            nxt = _apply(ghost, move)
            worst_eta = min(self._capture_eta(p, nxt) for p, _, _ in pac_predictions)
            worst_dist = min(self._dist(nxt, p) for p, _, _ in pac_predictions)
            s = self._cell_safety_score(nxt, pac_predictions)
            s += worst_dist * 950.0 + min(worst_eta, 8) * 1500.0
            s += max(0, 4 - self._junction_dist.get(nxt, INF_DIST)) * 1850.0
            s += max(0, 8 - self._loop_dist.get(nxt, INF_DIST)) * 360.0
            s += self._safe_area(nxt, pac_key) * 80.0
            if nxt in self._tp.junctions:
                s += 2400.0
            if nxt in self._core:
                s += 1200.0
            if nxt in self._tp.deb:
                s -= 8500.0
            if move == Move.STAY:
                if current_dist >= max(self._h, self._w) * 0.5 and worst_eta >= 5:
                    s += 2600.0
                else:
                    s -= 9000.0
            if last_ghost is not None and nxt == last_ghost and nxt not in self._tp.junctions:
                s -= 1800.0
            return s

        return max(moves, key=score)

    def _loop_lure_move(
        self,
        ghost: Pos,
        pac: Pos,
        pac_predictions: List[Tuple[Pos, float, str]],
        last_ghost: Optional[Pos],
        step_number: int,
        committed: bool,
    ) -> Move:
        moves = self._phase_moves(ghost, pac, step_number)
        if not moves:
            return Move.STAY

        current_manhattan = _manhattan(ghost, pac)
        current_loop_dist = self._loop_dist.get(ghost, INF_DIST)
        pac_dr, pac_dc = self._last_pac_action
        opposite = (0, 0)
        if abs(pac_dr) + abs(pac_dc) > 0:
            opposite = (-max(-1, min(1, pac_dr)), -max(-1, min(1, pac_dc)))

        pac_key = tuple(p for p, _, _ in pac_predictions[:4])

        def score(move: Move) -> float:
            nxt = _apply(ghost, move)
            next_loop_dist = self._loop_dist.get(nxt, INF_DIST)
            delta_loop = current_loop_dist - next_loop_dist
            next_manhattan = _manhattan(nxt, pac)
            delta_m = next_manhattan - current_manhattan
            worst_eta = min(self._capture_eta(p, nxt) for p, _, _ in pac_predictions)
            worst_dist = min(self._dist(nxt, p) for p, _, _ in pac_predictions)

            s = self._cell_safety_score(nxt, pac_predictions)
            s += self._anti_velocity_score(ghost, nxt, last_ghost, pac_predictions)
            s += worst_dist * 850.0 + min(worst_eta, 8) * 1350.0
            s += delta_loop * (2600.0 if not committed else 1300.0)
            s += max(0, 5 - next_loop_dist) * (900.0 if not committed else 1350.0)
            s += abs(delta_m) * 1500.0
            if delta_m > 0:
                s += 850.0
            if move.value == opposite:
                s += 1400.0
            if nxt in self._loop_set:
                s += 3600.0 if committed else 1900.0
            elif committed:
                s -= next_loop_dist * 550.0
            if nxt in self._tp.junctions:
                s += 1500.0
            if nxt in self._tp.deb:
                s -= 10000.0
            if move == Move.STAY:
                if self._allow_stay(ghost, pac, step_number) and worst_eta >= 5 and current_loop_dist <= 2:
                    s += 1000.0
                else:
                    s -= 10000.0
            if last_ghost is not None and nxt == last_ghost and not committed:
                s -= 1300.0
            s += self._safe_area(nxt, pac_key) * 75.0
            return s

        return max(moves, key=score)

    def _counter_loop_move(
        self,
        ghost: Pos,
        pac: Pos,
        pac_predictions: List[Tuple[Pos, float, str]],
        last_ghost: Optional[Pos],
        step_number: int,
    ) -> Move:
        moves = self._phase_moves(ghost, pac, step_number)
        if not moves:
            return Move.STAY

        current_manhattan = _manhattan(ghost, pac)
        pac_dr, pac_dc = self._last_pac_action
        preferred_opposite: Optional[Action] = None
        if abs(pac_dr) + abs(pac_dc) > 0:
            preferred_opposite = (-max(-1, min(1, pac_dr)), -max(-1, min(1, pac_dc)))

        last_trend = 0
        if self._last_manhattan is not None:
            if current_manhattan > self._last_manhattan:
                last_trend = 1
            elif current_manhattan < self._last_manhattan:
                last_trend = -1

        pac_key = tuple(p for p, _, _ in pac_predictions[:4])

        def score(move: Move) -> float:
            nxt = _apply(ghost, move)
            next_manhattan = _manhattan(nxt, pac)
            delta = next_manhattan - current_manhattan
            worst_pred_dist = min(self._dist(nxt, p) for p, _, _ in pac_predictions)
            worst_eta = min(self._capture_eta(p, nxt) for p, _, _ in pac_predictions)
            s = self._cell_safety_score(nxt, pac_predictions)
            s += self._anti_velocity_score(ghost, nxt, last_ghost, pac_predictions)
            s += worst_pred_dist * 900.0 + min(worst_eta, 8) * 1400.0
            s += abs(delta) * 1800.0
            if delta > 0:
                s += 1000.0
            if last_trend and delta and (delta > 0) != (last_trend > 0):
                s += 1500.0
            if preferred_opposite is not None and move.value == preferred_opposite:
                s += 2100.0
            if self._junction_dist.get(nxt, INF_DIST) <= 1:
                s += 2300.0
            elif self._junction_dist.get(nxt, INF_DIST) <= 3:
                s += 850.0
            if nxt in self._core:
                s += 1600.0
            if nxt in self._loop_set:
                s += 1100.0
            if nxt in self._tp.deb:
                s -= 8500.0 + self._danger_depth.get(nxt, 1) * 800.0
            if move == Move.STAY:
                if self._allow_stay(ghost, pac, step_number) and current_manhattan >= max(self._h, self._w) * 0.5:
                    s += 1250.0
                else:
                    s -= 9000.0
                if worst_eta <= 3:
                    s -= 12000.0
            if last_ghost is not None and nxt == last_ghost and self._junction_dist.get(ghost, INF_DIST) > 1:
                s -= 1600.0
            if not self._move_safely_changes_distance(ghost, pac, move) and move != Move.STAY:
                s -= 900.0
            s += self._safe_area(nxt, pac_key) * 95.0
            return s

        return max(moves, key=score)

    def _survival_oracle_move(
        self,
        ghost: Pos,
        pac: Pos,
        pac_predictions: List[Tuple[Pos, float, str]],
        last_ghost: Optional[Pos],
        step_number: int,
        t0: float,
    ) -> Optional[Move]:
        horizon = 9 if step_number <= 8 else 7
        memo: Dict[Tuple[Pos, Pos, Optional[Pos], int], float] = {}

        def leaf_score(g: Pos, preds: List[Tuple[Pos, float, str]]) -> float:
            worst_d = min(self._dist(g, p) for p, _, _ in preds)
            worst_eta = min(self._capture_eta(p, g) for p, _, _ in preds)
            return self._cell_safety_score(g, preds) + worst_d * 1500.0 + min(worst_eta, 8) * 2200.0

        def rec(g: Pos, p: Pos, lg: Optional[Pos], preds: List[Tuple[Pos, float, str]], depth: int) -> float:
            if time.time() - t0 > TIME_BUDGET * 0.55:
                return leaf_score(g, preds)
            if depth >= horizon:
                return leaf_score(g, preds)
            key = (g, p, lg, depth)
            if key in memo:
                return memo[key]

            best = float("-inf")
            moves = self._phase_moves(g, p, step_number + depth)
            if not moves:
                return -100000.0 + depth

            for move in moves:
                ng = _apply(g, move)
                immediate = leaf_score(ng, preds)
                immediate += self._anti_velocity_score(g, ng, lg, preds)
                if move == Move.STAY and not self._allow_stay(g, p, step_number + depth):
                    continue
                if lg is not None and ng == lg and depth < 4:
                    immediate -= 2200.0

                worst_child = float("inf")
                for pp, weight, _ in preds[:3]:
                    if _manhattan(ng, pp) < CAPTURE_DISTANCE:
                        branch = -120000.0 + depth * 2000.0 - weight * 500.0
                    else:
                        next_preds = self._ranked_pac_predictions(pp, ng, g)
                        branch = rec(ng, pp, g, next_preds, depth + 1) - weight * 180.0
                    worst_child = min(worst_child, branch)
                score = immediate * 0.42 + worst_child * 0.58
                if score > best:
                    best = score

            memo[key] = best
            return best

        best_move: Optional[Move] = None
        best_score = float("-inf")
        for move in self._phase_moves(ghost, pac, step_number):
            ng = _apply(ghost, move)
            if move == Move.STAY and not self._allow_stay(ghost, pac, step_number):
                continue
            if last_ghost is not None and ng == last_ghost:
                continue
            worst = float("inf")
            for pp, weight, _ in pac_predictions[:3]:
                if _manhattan(ng, pp) < CAPTURE_DISTANCE:
                    branch = -120000.0 - weight * 500.0
                else:
                    next_preds = self._ranked_pac_predictions(pp, ng, ghost)
                    branch = rec(ng, pp, ghost, next_preds, 1) - weight * 180.0
                worst = min(worst, branch)
            spread = _manhattan(ng, pac) - _manhattan(ghost, pac)
            score = worst + spread * 2500.0 + self._degree.get(ng, 0) * 500.0
            if ng in self._core:
                score += 1800.0
            if ng in self._tp.deb:
                score -= 9000.0
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _pacman_influence(self, ghost: Pos) -> float:
        penalty = 0.0
        for age, pac in enumerate(reversed(self._pac_hist)):
            if age >= 12:
                break
            d = self._dist(ghost, pac)
            if d < 8:
                penalty += (8 - d) * (0.68 ** age) * 115.0
        return penalty

    def _anti_velocity_score(
        self,
        ghost: Pos,
        nxt: Pos,
        last_ghost: Optional[Pos],
        pac_predictions: List[Tuple[Pos, float, str]],
    ) -> float:
        if last_ghost is None:
            return 0.0
        dr = ghost[0] - last_ghost[0]
        dc = ghost[1] - last_ghost[1]
        expected = (ghost[0] + dr, ghost[1] + dc)
        close_eta = min((self._capture_eta(p, ghost) for p, _, _ in pac_predictions), default=INF_DIST)

        score = 0.0
        if _valid(expected, self._static_map):
            if nxt != expected:
                score += 620.0
                if ghost in self._tp.junctions or nxt in self._tp.junctions:
                    score += 760.0
                if close_eta <= 3:
                    score += 680.0
            elif close_eta <= 4:
                score -= 980.0

        if nxt == last_ghost and close_eta > 3:
            score -= 540.0
        return score

    def _cell_safety_score(self, ghost: Pos, pac_predictions: List[Tuple[Pos, float, str]]) -> float:
        weighted = 0.0
        worst = INF_DIST
        worst_eta = INF_DIST
        for pac, weight, _ in pac_predictions:
            d = self._dist(ghost, pac)
            eta = self._capture_eta(pac, ghost)
            worst = min(worst, d)
            worst_eta = min(worst_eta, eta)
            if d <= 0:
                weighted -= 100000.0 * weight
            elif d == 1:
                weighted -= 22000.0 * weight
            elif d == 2:
                weighted -= 4500.0 * weight
            else:
                weighted += min(d, 18) * 360.0 * weight

            if eta <= 0:
                weighted -= 120000.0 * weight
            elif eta == 1:
                weighted -= 36000.0 * weight
            elif eta == 2:
                weighted -= 9000.0 * weight
            else:
                weighted += min(eta, 10) * 760.0 * weight

        score = weighted
        score += min(worst, 18) * 900.0
        score += min(worst_eta, 10) * 1200.0
        score += self._degree.get(ghost, 0) * 260.0
        score += len(_neighbors(ghost, self._static_map)) * 120.0

        pac_key = tuple(p for p, _, _ in pac_predictions[:4])
        area = self._safe_area(ghost, pac_key)
        score += min(area, 80.0) * 125.0
        if area < 10:
            score -= (10 - area) * 760.0

        if ghost in self._core:
            score += 2400.0
        if ghost in self._loop_set:
            score += 1600.0
        if ghost in self._tp.junctions:
            score += 650.0
        if ghost in self._tp.deb:
            score -= 6500.0 + 420.0 * self._danger_depth.get(ghost, 1)

        if ghost in self._ghost_hist:
            score -= 220.0
        if len(self._ghost_hist) >= 2 and ghost == self._ghost_hist[-2]:
            score -= 900.0

        score -= self._pacman_influence(ghost)
        # Slight preference for central, high-mobility map areas when no immediate danger exists.
        score -= 12.0 * _manhattan(ghost, self._center)
        return score

    def _move_order(
        self,
        ghost: Pos,
        pac_predictions: List[Tuple[Pos, float, str]],
        last_ghost: Optional[Pos],
        pac: Optional[Pos] = None,
        step_number: int = 0,
    ) -> List[Move]:
        moves = self._phase_moves(ghost, pac, step_number) if pac is not None else _ghost_legal(ghost, self._static_map)
        if not moves:
            return []

        def key(move: Move) -> Tuple[float, int]:
            nxt = _apply(ghost, move)
            greedy_clearance = min(self._dist(nxt, pac) for pac, _, _ in pac_predictions)
            tactical = self._anti_velocity_score(ghost, nxt, last_ghost, pac_predictions)
            return (self._cell_safety_score(nxt, pac_predictions) + tactical, greedy_clearance)

        # Greedy best-first ordering: best-looking escapes are searched first,
        # allowing iterative deepening to keep a strong answer under the time limit.
        moves.sort(key=key, reverse=True)
        return moves

    def _search(
        self,
        ghost: Pos,
        pac_predictions: List[Tuple[Pos, float, str]],
        last_ghost: Optional[Pos],
        depth: int,
        max_depth: int,
        alpha: float,
        t0: float,
        pac: Optional[Pos] = None,
        step_number: int = 0,
    ) -> Tuple[Optional[float], Optional[Move]]:
        if time.time() - t0 > TIME_BUDGET * 0.88:
            return None, None
        if depth >= max_depth:
            return self._cell_safety_score(ghost, pac_predictions), None

        best_score = float("-inf")
        best_move: Optional[Move] = None
        ordered_moves = self._move_order(ghost, pac_predictions, last_ghost, pac, step_number)
        if not ordered_moves:
            return -100000.0, None

        for move in ordered_moves:
            ng = _apply(ghost, move)
            immediate = self._cell_safety_score(ng, pac_predictions)
            immediate += self._anti_velocity_score(ghost, ng, last_ghost, pac_predictions)
            if immediate < -60000:
                score = immediate + depth * 500.0
            else:
                worst_child = float("inf")
                for pac, weight, _ in pac_predictions:
                    next_pacs = self._ranked_pac_predictions(pac, ng, ghost)
                    child, _ = self._search(ng, next_pacs, ghost, depth + 1, max_depth, alpha, t0, pac, step_number + depth + 1)
                    if child is None:
                        return None, None
                    # Maximin: assume the most dangerous plausible Pacman branch,
                    # but still respect US-L* confidence through the weight.
                    branch = child - 140.0 * weight
                    worst_child = min(worst_child, branch)
                    if worst_child <= alpha:
                        break
                score = 0.58 * immediate + 0.42 * worst_child

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        return best_score, best_move

    def _panic_move(
        self,
        ghost: Pos,
        pac: Pos,
        pac_predictions: List[Tuple[Pos, float, str]],
        last_ghost: Optional[Pos],
    ) -> Move:
        moves = _ghost_legal(ghost, self._static_map)
        if not moves:
            return Move.STAY
        pac_dist = self._bfs_dist(pac)
        return max(
            moves,
            key=lambda m: (
                self._cell_safety_score(_apply(ghost, m), pac_predictions)
                + self._anti_velocity_score(ghost, _apply(ghost, m), last_ghost, pac_predictions),
                pac_dist.get(_apply(ghost, m), INF_DIST),
                _apply(ghost, m) in self._core,
                _apply(ghost, m) in self._loop_set,
                -int(_apply(ghost, m) in self._tp.deb),
                self._degree.get(_apply(ghost, m), 0),
            ),
        )

    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()
        self._ensure_static(map_state)
        me = tuple(my_position)
        last_ghost = self._ghost_hist[-1] if self._ghost_hist else None

        legal = _ghost_legal(me, self._static_map)
        if not legal:
            return Move.STAY

        if enemy_position is None:
            self._ghost_hist.append(me)
            # Without vision, stay in the safest known high-mobility region.
            return max(legal, key=lambda m: self._degree.get(_apply(me, m), 0))

        pac = tuple(enemy_position)

        if self._last_pp is not None:
            observed_action = (pac[0] - self._last_pp[0], pac[1] - self._last_pp[1])
            self._last_pac_action = observed_action
            observed_speed = abs(observed_action[0]) + abs(observed_action[1])
            if observed_speed > self._pacman_speed:
                self._pacman_speed = min(4, observed_speed)
                self._capture_eta.cache_clear()
                self._safe_area.cache_clear()
            previous_ghost = self._ghost_hist[-1] if self._ghost_hist else me
            before_previous_ghost = self._ghost_hist[-2] if len(self._ghost_hist) >= 2 else None
            state = self._usl.get_state(self._static_map, previous_ghost, self._last_pp, before_previous_ghost)
            self._usl.observe(state, observed_action, self._static_map, self._last_pp)

        self._last_pp = pac
        self._pac_hist.append(pac)
        if self._initial_distance is None:
            self._initial_distance = _manhattan(me, pac)
            self._initial_far = self._initial_distance >= max(self._h, self._w) * 0.5

        pac_predictions = self._ranked_pac_predictions(pac, me, last_ghost)
        min_now = min(self._dist(me, p) for p, _, _ in pac_predictions)
        if min_now <= 1:
            move = self._panic_move(me, pac, pac_predictions, last_ghost)
            self._ghost_hist.append(me)
            self._last_manhattan = _manhattan(me, pac)
            return move

        if step_number <= OPENING_LEARN_STEPS:
            if self._initial_far:
                move = self._far_reading_move(me, pac, pac_predictions, last_ghost, step_number)
            else:
                move = self._survival_oracle_move(me, pac, pac_predictions, last_ghost, step_number, t0)
                if move is None:
                    move = self._opening_spread_move(me, pac, pac_predictions, last_ghost, step_number)
            self._ghost_hist.append(me)
            self._last_manhattan = _manhattan(me, pac)
            return move

        if step_number <= OPENING_LEARN_STEPS + 10:
            move = self._loop_lure_move(me, pac, pac_predictions, last_ghost, step_number, committed=False)
            self._ghost_hist.append(me)
            self._last_manhattan = _manhattan(me, pac)
            return move

        if self._loop_dist.get(me, INF_DIST) <= 3 or self._initial_far:
            move = self._loop_lure_move(me, pac, pac_predictions, last_ghost, step_number, committed=True)
            move_pos = _apply(me, move)
            move_eta = min(self._capture_eta(p, move_pos) for p, _, _ in pac_predictions)
            if move_eta >= 2:
                self._ghost_hist.append(me)
                self._last_manhattan = _manhattan(me, pac)
                return move

        if self._initial_far and step_number > OPENING_LEARN_STEPS:
            move = self._counter_loop_move(me, pac, pac_predictions, last_ghost, step_number)
            self._ghost_hist.append(me)
            self._last_manhattan = _manhattan(me, pac)
            return move

        phase_move = self._counter_loop_move(me, pac, pac_predictions, last_ghost, step_number)
        phase_pos = _apply(me, phase_move)
        phase_eta = min(self._capture_eta(p, phase_pos) for p, _, _ in pac_predictions)

        best_move = self._move_order(me, pac_predictions, last_ghost, pac, step_number)[0]
        best_score = float("-inf")

        # Iterative deepening search: keep a valid best move at every completed depth.
        # Deeper levels are used only while the budget allows it.
        for depth in range(2, 18):
            score, move = self._search(me, pac_predictions, last_ghost, 0, depth, float("-inf"), t0, pac, step_number)
            if score is None:
                break
            if move is not None:
                best_score = score
                best_move = move
            if best_score < -70000:
                break

        best_pos = _apply(me, best_move)
        best_eta = min(self._capture_eta(p, best_pos) for p, _, _ in pac_predictions)
        if phase_eta >= best_eta and step_number > OPENING_LEARN_STEPS:
            best_move = phase_move

        self._ghost_hist.append(me)
        self._last_manhattan = _manhattan(me, pac)
        return best_move


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, map_state, my_position, enemy_position, step_number):
        me = tuple(my_position)
        ms = map_state
        cands = _legal(me, ms)
        if not cands:
            return Move.STAY
        if enemy_position is None:
            return cands[0]

        pac = tuple(enemy_position)

        def bfs(start: Pos) -> Dict[Pos, int]:
            d = {start: 0}
            q = deque([start])
            while q:
                cur = q.popleft()
                for m in MOVE_ORDER:
                    nxt = _apply(cur, m)
                    if _valid(nxt, ms) and nxt not in d:
                        d[nxt] = d[cur] + 1
                        q.append(nxt)
            return d

        enemy_d = bfs(pac)
        best_m = cands[0]
        best_d = float("inf")
        for m in cands:
            nxt = _apply(me, m)
            d = enemy_d.get(nxt, float("inf"))
            if d < best_d:
                best_d = d
                best_m = m
        return best_m
