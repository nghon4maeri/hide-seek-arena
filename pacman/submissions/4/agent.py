"""24127457 — Merged team submission.

GhostAgent ported from 24127192 (all strategy phases, US-L*, capture_eta, safe_area).
PacmanAgent ported from 24127561 (A*, streak interception, speed-2 packing).
"""

import sys, time, heapq, random
from pathlib import Path
from collections import Counter, defaultdict, deque
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move
from search import astar_path

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MOVE_ORDER = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
INF_DIST = 10**9
CAPTURE_DISTANCE = 2
Pos = Tuple[int, int]

def _shape(ms):
    if hasattr(ms, "shape"): return int(ms.shape[0]), int(ms.shape[1])
    return len(ms), len(ms[0]) if ms else 0

def _cell(ms, r, c):
    return int(ms[r, c]) if hasattr(ms, "shape") else int(ms[r][c])

def _valid(pos, ms):
    r, c = pos; h, w = _shape(ms)
    return 0 <= r < h and 0 <= c < w and _cell(ms, r, c) != 1

def _apply(pos, move):
    return (pos[0] + move.value[0], pos[1] + move.value[1])

def _legal(pos, ms):
    return [m for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]

def _neighbors(pos, ms):
    return [_apply(pos, m) for m in MOVE_ORDER if _valid(_apply(pos, m), ms)]

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _bucket(v):
    if v <= -6: return -4
    if v <= -3: return -3
    if v < 0: return -1
    if v == 0: return 0
    if v < 3: return 1
    if v < 6: return 3
    return 4

# ---------------------------------------------------------------------------
# Topology (from 24127192)
# ---------------------------------------------------------------------------

class Topo:
    def __init__(self):
        self.deb: Set[Pos] = set()
        self.junctions: Set[Pos] = set()
        self.corridor: Set[Pos] = set()
        self.ok = False

    def init(self, ms):
        if self.ok: return
        h, w = _shape(ms)
        dead_seeds: Set[Pos] = set()
        for r in range(h):
            for c in range(w):
                if _cell(ms, r, c) == 1: continue
                p = (r, c)
                deg = len(_neighbors(p, ms))
                if deg <= 1: dead_seeds.add(p)
                elif deg >= 3: self.junctions.add(p)
                else: self.corridor.add(p)
        for seed in dead_seeds:
            cur = seed; prev = None; seen = {cur}
            while True:
                self.deb.add(cur)
                if cur in self.junctions: break
                nxts = [x for x in _neighbors(cur, ms) if x != prev]
                if not nxts: break
                nxt = nxts[0]
                if nxt in seen: break
                prev, cur = cur, nxt; seen.add(cur)
        self.ok = True

# ---------------------------------------------------------------------------
# US-L* online learner (from 24127192)
# ---------------------------------------------------------------------------

class USLStar:
    def __init__(self):
        self._counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self._global: Counter = Counter()

    def get_state(self, ms, ghost_pos, pac_pos, last_ghost_pos=None):
        dx = _bucket(ghost_pos[0] - pac_pos[0])
        dy = _bucket(ghost_pos[1] - pac_pos[1])
        dist_bucket = min(9, _manhattan(ghost_pos, pac_pos) // 2)
        hdr = ghost_pos[0] - last_ghost_pos[0] if last_ghost_pos else 0
        hdc = ghost_pos[1] - last_ghost_pos[1] if last_ghost_pos else 0
        hdr = max(-1, min(1, hdr)); hdc = max(-1, min(1, hdc))
        pac_moves = _legal(pac_pos, ms)
        deg = len(pac_moves)
        if deg <= 1: geo = 0
        elif deg >= 3: geo = 3
        else:
            m1, m2 = pac_moves[0], pac_moves[1]
            opposite = m1.value[0] + m2.value[0] == 0 and m1.value[1] + m2.value[1] == 0
            geo = 1 if opposite else 2
        walls = sum((1 << i) for i, m in enumerate(MOVE_ORDER) if not _valid(_apply(pac_pos, m), ms))
        return (dx, dy, dist_bucket, hdr, hdc, geo, walls)

    def observe(self, state, action, ms, from_pos):
        to_pos = (from_pos[0] + action[0], from_pos[1] + action[1])
        if not _valid(to_pos, ms): return
        self._counts[state][action] += 1
        self._global[action] += 1

    def ranked_predictions(self, ms, pac_pos, ghost_pos, last_ghost_pos, fallback_positions, limit=5):
        state = self.get_state(ms, ghost_pos, pac_pos, last_ghost_pos)
        candidates = {}
        counts = self._counts.get(state)
        if counts:
            total = sum(counts.values())
            for action, cnt in counts.most_common(limit):
                pred = (pac_pos[0] + action[0], pac_pos[1] + action[1])
                if _valid(pred, ms):
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


# ---------------------------------------------------------------------------
# GhostAgent (ported from 24127192)
# ---------------------------------------------------------------------------

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tp = Topo()
        self._core: Set[Pos] = set()
        self._loop_set: Set[Pos] = set()
        self._usl = USLStar()
        self._pacman_speed = max(2, int(kwargs.get("pacman_speed", 2)))
        self._ghost_hist: deque = deque(maxlen=80)
        self._pac_hist: deque = deque(maxlen=80)
        self._last_pp = None
        self._static_map = None
        self._open_cells: Set[Pos] = set()
        self._degree: Dict[Pos, int] = {}
        self._danger_depth: Dict[Pos, int] = {}
        self._junction_dist: Dict[Pos, int] = {}
        self._loop_dist: Dict[Pos, int] = {}
        self._center = (0, 0)
        self._last_pac_action = (0, 0)
        self._last_manhattan = None
        self._initial_distance = None
        self._initial_far = False
        self._h = self._w = 0

    def _ensure_static(self, ms):
        if self._static_map is not None: return
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

    def _compute_core(self):
        active = set(self._open_cells)
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for cell in active:
                deg = sum(1 for nxt in _neighbors(cell, self._static_map) if nxt in active)
                if deg <= 1: to_remove.add(cell)
            if to_remove:
                active -= to_remove
                changed = True
        self._core = active

    def _dfs_find_loops(self):
        self._loop_set = set()
        if not self._core: return
        cycles = []
        vis = set()
        parent = {}
        for start in list(self._core):
            if start in vis: continue
            stack = [(start, None, iter(_neighbors(start, self._static_map)))]
            vis.add(start); parent[start] = None
            while stack:
                u, p, it = stack[-1]
                try: v = next(it)
                except StopIteration: stack.pop(); continue
                if v not in self._core: continue
                if v not in vis:
                    vis.add(v); parent[v] = u
                    stack.append((v, u, iter(_neighbors(v, self._static_map))))
                elif v != p:
                    cycle = [v]; cur = u
                    while cur is not None and cur != v:
                        cycle.append(cur); cur = parent.get(cur)
                    if cur == v and len(cycle) >= 4:
                        cycles.append(cycle)
        if cycles: self._loop_set = set(max(cycles, key=len))

    def _compute_dead_depth(self):
        self._danger_depth = {p: 0 for p in self._open_cells}
        q = deque()
        trimmed = set()
        degree = dict(self._degree)
        for p, deg in degree.items():
            if deg <= 1:
                q.append(p); trimmed.add(p); self._danger_depth[p] = 1
        while q:
            u = q.popleft()
            for v in _neighbors(u, self._static_map):
                if v in trimmed: continue
                degree[v] -= 1
                if degree[v] <= 1:
                    trimmed.add(v)
                    self._danger_depth[v] = self._danger_depth[u] + 1
                    q.append(v)

    def _compute_junction_dist(self):
        self._junction_dist = {p: INF_DIST for p in self._open_cells}
        q = deque()
        starts = set(self._tp.junctions) or set(self._core)
        for p in starts:
            self._junction_dist[p] = 0; q.append(p)
        while q:
            cur = q.popleft()
            for nxt in _neighbors(cur, self._static_map):
                if self._junction_dist.get(nxt, INF_DIST) > self._junction_dist[cur] + 1:
                    self._junction_dist[nxt] = self._junction_dist[cur] + 1; q.append(nxt)

    def _compute_loop_dist(self):
        self._loop_dist = {p: INF_DIST for p in self._open_cells}
        q = deque()
        starts = set(self._loop_set) or set(self._core) or set(self._tp.junctions)
        for p in starts:
            self._loop_dist[p] = 0; q.append(p)
        while q:
            cur = q.popleft()
            for nxt in _neighbors(cur, self._static_map):
                if self._loop_dist.get(nxt, INF_DIST) > self._loop_dist[cur] + 1:
                    self._loop_dist[nxt] = self._loop_dist[cur] + 1; q.append(nxt)

    @lru_cache(maxsize=4096)
    def _bfs_dist(self, start):
        d = {start: 0}; q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in _neighbors(cur, self._static_map):
                if nxt not in d: d[nxt] = d[cur] + 1; q.append(nxt)
        return d

    def _dist(self, a, b):
        return self._bfs_dist(a).get(b, INF_DIST)

    @lru_cache(maxsize=4096)
    def _astar_path(self, start, target):
        if start == target: return ()
        open_set = [(_manhattan(start, target), 0, start)]
        came_from = {}; g_score = {start: 0}; closed = set(); push_id = 1
        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current in closed: continue
            closed.add(current)
            if current == target:
                path = []
                while current != start:
                    path.append(current); current = came_from[current]
                path.reverse(); return tuple(path)
            nxts = _neighbors(current, self._static_map)
            nxts.sort(key=lambda p: _manhattan(p, target))
            for neighbor in nxts:
                tentative = g_score[current] + 1
                if tentative < g_score.get(neighbor, INF_DIST):
                    came_from[neighbor] = current; g_score[neighbor] = tentative
                    heapq.heappush(open_set, (tentative + _manhattan(neighbor, target), push_id, neighbor))
                    push_id += 1
        return ()

    def _astar_next(self, start, target):
        path = self._astar_path(start, target)
        return path[0] if path else start

    def _follow_path_with_speed(self, start, path):
        if not path: return start
        dr = path[0][0] - start[0]; dc = path[0][1] - start[1]
        pos = start; used = 0
        for nxt in path:
            if used >= self._pacman_speed: break
            if (nxt[0] - pos[0], nxt[1] - pos[1]) != (dr, dc): break
            pos = nxt; used += 1
        return pos

    def _astar_speed_next(self, start, target):
        return self._follow_path_with_speed(start, self._astar_path(start, target))

    def _greedy_best_first_next(self, start, target):
        moves = _legal(start, self._static_map)
        if not moves: return start
        return min((_apply(start, m) for m in moves),
                   key=lambda p: (_manhattan(p, target), self._dist(p, target)))

    def _straight_progress(self, start, first, target):
        dr = first[0] - start[0]; dc = first[1] - start[1]
        pos = first; last_d = self._dist(pos, target)
        for _ in range(1, self._pacman_speed):
            nxt = (pos[0] + dr, pos[1] + dc)
            if not _valid(nxt, self._static_map): break
            nxt_d = self._dist(nxt, target)
            if nxt_d >= last_d: break
            pos = nxt; last_d = nxt_d
        return pos

    def _greedy_speed_next(self, start, target):
        first = self._greedy_best_first_next(start, target)
        return self._straight_progress(start, first, target) if first != start else start

    def _exact_bfs_chaser_next(self, pac, ghost):
        moves = _legal(pac, self._static_map)
        if not moves: return pac
        ghost_dist = self._bfs_dist(ghost)
        return min((_apply(pac, m) for m in moves),
                   key=lambda p: (ghost_dist.get(p, INF_DIST), _manhattan(p, ghost)))

    def _exact_bfs_speed_next(self, pac, ghost):
        first = self._exact_bfs_chaser_next(pac, ghost)
        return self._straight_progress(pac, first, ghost) if first != pac else pac

    @lru_cache(maxsize=4096)
    def _capture_eta(self, start, target):
        if _manhattan(start, target) < CAPTURE_DISTANCE: return 0
        path = self._astar_path(start, target)
        if not path: return INF_DIST
        pos = start; idx = 0; turns = 0
        while idx < len(path):
            dr = path[idx][0] - pos[0]; dc = path[idx][1] - pos[1]
            used = 0
            while idx < len(path) and used < self._pacman_speed:
                nxt = path[idx]
                if (nxt[0] - pos[0], nxt[1] - pos[1]) != (dr, dc): break
                pos = nxt; idx += 1; used += 1
                if _manhattan(pos, target) < CAPTURE_DISTANCE: return turns + 1
            turns += 1
        return turns

    @lru_cache(maxsize=2048)
    def _safe_area(self, ghost, pac_positions):
        total = 0.0
        q = deque([ghost]); seen = {ghost: 0}
        while q:
            cur = q.popleft()
            gd = seen[cur]
            if gd > 14: continue
            pac_eta = min((self._capture_eta(p, cur) for p in pac_positions), default=INF_DIST)
            margin = pac_eta - gd
            if margin > 0:
                total += 1.0 + min(3, margin) * 0.35 + self._degree.get(cur, 0) * 0.08
                for nxt in _neighbors(cur, self._static_map):
                    if nxt not in seen:
                        seen[nxt] = gd + 1; q.append(nxt)
        return total

    def _intercept_target(self, ghost, last_ghost):
        if last_ghost is None: return ghost
        dr = ghost[0] - last_ghost[0]; dc = ghost[1] - last_ghost[1]
        target = (ghost[0] + dr, ghost[1] + dc)
        return target if _valid(target, self._static_map) else ghost

    def _pacman_model_positions(self, pac, ghost, last_ghost):
        intercept = self._intercept_target(ghost, last_ghost)
        models = [
            self._astar_speed_next(pac, intercept), self._astar_speed_next(pac, ghost),
            self._exact_bfs_speed_next(pac, intercept), self._exact_bfs_speed_next(pac, ghost),
            self._greedy_speed_next(pac, intercept), self._greedy_speed_next(pac, ghost),
            pac,
        ]
        seen = set(); out = []
        for p in models:
            if p not in seen and _valid(p, self._static_map):
                seen.add(p); out.append(p)
        return out

    def _ranked_pac_predictions(self, pac, ghost, last_ghost):
        fallbacks = self._pacman_model_positions(pac, ghost, last_ghost)
        return self._usl.ranked_predictions(self._static_map, pac, ghost, last_ghost, fallbacks, limit=5)

    def _allow_stay(self, ghost, pac, step_number):
        far_enough = _manhattan(ghost, pac) >= max(self._h, self._w) * 0.5
        return step_number > 20 or far_enough or (self._initial_far and step_number <= 20)

    def _phase_moves(self, ghost, pac, step_number):
        moves = _legal(ghost, self._static_map)
        if self._allow_stay(ghost, pac, step_number):
            moves = moves + [Move.STAY]
        return moves

    def _move_safely_changes_distance(self, ghost, pac, move):
        nxt = _apply(ghost, move)
        return _manhattan(nxt, pac) != _manhattan(ghost, pac)

    def _pacman_influence(self, ghost):
        penalty = 0.0
        for age, pac in enumerate(reversed(self._pac_hist)):
            if age >= 12: break
            d = self._dist(ghost, pac)
            if d < 8:
                penalty += (8 - d) * (0.68 ** age) * 115.0
        return penalty

    def _anti_velocity_score(self, ghost, nxt, last_ghost, pac_predictions):
        if last_ghost is None: return 0.0
        dr = ghost[0] - last_ghost[0]; dc = ghost[1] - last_ghost[1]
        expected = (ghost[0] + dr, ghost[1] + dc)
        close_eta = min((self._capture_eta(p, ghost) for p, _, _ in pac_predictions), default=INF_DIST)
        score = 0.0
        if _valid(expected, self._static_map):
            if nxt != expected:
                score += 620.0
                if ghost in self._tp.junctions or nxt in self._tp.junctions: score += 760.0
                if close_eta <= 3: score += 680.0
            elif close_eta <= 4: score -= 980.0
        if nxt == last_ghost and close_eta > 3: score -= 540.0
        return score

    def _cell_safety_score(self, ghost, pac_predictions):
        weighted = 0.0; worst = INF_DIST; worst_eta = INF_DIST
        for pac, weight, _ in pac_predictions:
            d = self._dist(ghost, pac)
            eta = self._capture_eta(pac, ghost)
            worst = min(worst, d); worst_eta = min(worst_eta, eta)
            if d <= 0: weighted -= 100000.0 * weight
            elif d == 1: weighted -= 22000.0 * weight
            elif d == 2: weighted -= 4500.0 * weight
            else: weighted += min(d, 18) * 360.0 * weight
            if eta <= 0: weighted -= 120000.0 * weight
            elif eta == 1: weighted -= 36000.0 * weight
            elif eta == 2: weighted -= 9000.0 * weight
            else: weighted += min(eta, 10) * 760.0 * weight
        score = weighted
        score += min(worst, 18) * 900.0 + min(worst_eta, 10) * 1200.0
        score += self._degree.get(ghost, 0) * 260.0 + len(_neighbors(ghost, self._static_map)) * 120.0
        pac_key = tuple(p for p, _, _ in pac_predictions[:4])
        area = self._safe_area(ghost, pac_key)
        score += min(area, 80.0) * 125.0
        if area < 10: score -= (10 - area) * 760.0
        if ghost in self._core: score += 2400.0
        if ghost in self._loop_set: score += 1600.0
        if ghost in self._tp.junctions: score += 650.0
        if ghost in self._tp.deb: score -= 6500.0 + 420.0 * self._danger_depth.get(ghost, 1)
        if ghost in self._ghost_hist: score -= 220.0
        if len(self._ghost_hist) >= 2 and ghost == self._ghost_hist[-2]: score -= 900.0
        score -= self._pacman_influence(ghost)
        score -= 12.0 * _manhattan(ghost, self._center)
        return score

    def _move_order(self, ghost, pac_predictions, last_ghost, pac=None, step_number=0):
        moves = self._phase_moves(ghost, pac, step_number) if pac is not None else _legal(ghost, self._static_map)
        if not moves: return []
        def key(move):
            nxt = _apply(ghost, move)
            greedy = min(self._dist(nxt, pac) for pac, _, _ in pac_predictions)
            tactical = self._anti_velocity_score(ghost, nxt, last_ghost, pac_predictions)
            return (self._cell_safety_score(nxt, pac_predictions) + tactical, greedy)
        moves.sort(key=key, reverse=True)
        return moves

    def _opening_spread_move(self, ghost, pac, pac_predictions, last_ghost, step_number):
        moves = self._phase_moves(ghost, pac, step_number)
        if not moves: return Move.STAY
        cur_dist = _manhattan(ghost, pac)
        pac_dist_map = self._bfs_dist(pac)
        def score(move):
            nxt = _apply(ghost, move)
            wd = min(self._dist(nxt, p) for p, _, _ in pac_predictions)
            we = min(self._capture_eta(p, nxt) for p, _, _ in pac_predictions)
            s = self._cell_safety_score(nxt, pac_predictions)
            s += wd * 1450.0 + min(we, 8) * 1900.0
            s += (_manhattan(nxt, pac) - cur_dist) * 2400.0
            s += (pac_dist_map.get(nxt, INF_DIST) - pac_dist_map.get(ghost, INF_DIST)) * 1800.0
            s += self._safe_area(nxt, tuple(p for p, _, _ in pac_predictions[:4])) * 110.0
            s += self._degree.get(nxt, 0) * 360.0
            s += max(0, 4 - self._junction_dist.get(nxt, INF_DIST)) * 520.0
            if nxt in self._core: s += 1700.0
            if nxt in self._tp.deb: s -= 9000.0 + self._danger_depth.get(nxt, 1) * 900.0
            if last_ghost is not None and nxt == last_ghost: s -= 5200.0
            if len(self._ghost_hist) >= 2 and nxt == self._ghost_hist[-2]: s -= 3200.0
            if move == Move.STAY: s -= 5000.0
            return s
        return max(moves, key=score)

    def _far_reading_move(self, ghost, pac, pac_predictions, last_ghost, step_number):
        moves = self._phase_moves(ghost, pac, step_number)
        if not moves: return Move.STAY
        cur_dist = _manhattan(ghost, pac)
        pac_key = tuple(p for p, _, _ in pac_predictions[:4])
        def score(move):
            nxt = _apply(ghost, move)
            we = min(self._capture_eta(p, nxt) for p, _, _ in pac_predictions)
            wd = min(self._dist(nxt, p) for p, _, _ in pac_predictions)
            s = self._cell_safety_score(nxt, pac_predictions)
            s += wd * 950.0 + min(we, 8) * 1500.0
            s += max(0, 4 - self._junction_dist.get(nxt, INF_DIST)) * 1850.0
            s += max(0, 8 - self._loop_dist.get(nxt, INF_DIST)) * 360.0
            s += self._safe_area(nxt, pac_key) * 80.0
            if nxt in self._tp.junctions: s += 2400.0
            if nxt in self._core: s += 1200.0
            if nxt in self._tp.deb: s -= 8500.0
            if move == Move.STAY:
                if cur_dist >= max(self._h, self._w) * 0.5 and we >= 5: s += 2600.0
                else: s -= 9000.0
            if last_ghost is not None and nxt == last_ghost and nxt not in self._tp.junctions: s -= 1800.0
            return s
        return max(moves, key=score)

    def _loop_lure_move(self, ghost, pac, pac_predictions, last_ghost, step_number, committed):
        moves = self._phase_moves(ghost, pac, step_number)
        if not moves: return Move.STAY
        cur_md = _manhattan(ghost, pac)
        cur_ld = self._loop_dist.get(ghost, INF_DIST)
        pac_dr, pac_dc = self._last_pac_action
        opposite = (0, 0)
        if abs(pac_dr) + abs(pac_dc) > 0:
            opposite = (-max(-1, min(1, pac_dr)), -max(-1, min(1, pac_dc)))
        pac_key = tuple(p for p, _, _ in pac_predictions[:4])
        def score(move):
            nxt = _apply(ghost, move)
            nld = self._loop_dist.get(nxt, INF_DIST)
            d_ld = cur_ld - nld
            nmd = _manhattan(nxt, pac)
            d_md = nmd - cur_md
            we = min(self._capture_eta(p, nxt) for p, _, _ in pac_predictions)
            wd = min(self._dist(nxt, p) for p, _, _ in pac_predictions)
            s = self._cell_safety_score(nxt, pac_predictions)
            s += self._anti_velocity_score(ghost, nxt, last_ghost, pac_predictions)
            s += wd * 850.0 + min(we, 8) * 1350.0
            s += d_ld * (2600.0 if not committed else 1300.0)
            s += max(0, 5 - nld) * (900.0 if not committed else 1350.0)
            s += abs(d_md) * 1500.0
            if d_md > 0: s += 850.0
            if move.value == opposite: s += 1400.0
            if nxt in self._loop_set: s += 3600.0 if committed else 1900.0
            elif committed: s -= nld * 550.0
            if nxt in self._tp.junctions: s += 1500.0
            if nxt in self._tp.deb: s -= 10000.0
            if move == Move.STAY:
                if self._allow_stay(ghost, pac, step_number) and we >= 5 and cur_ld <= 2: s += 1000.0
                else: s -= 10000.0
            if last_ghost is not None and nxt == last_ghost and not committed: s -= 1300.0
            s += self._safe_area(nxt, pac_key) * 75.0
            return s
        return max(moves, key=score)

    def _counter_loop_move(self, ghost, pac, pac_predictions, last_ghost, step_number):
        moves = self._phase_moves(ghost, pac, step_number)
        if not moves: return Move.STAY
        cur_md = _manhattan(ghost, pac)
        pac_dr, pac_dc = self._last_pac_action
        preferred_opposite = None
        if abs(pac_dr) + abs(pac_dc) > 0:
            preferred_opposite = (-max(-1, min(1, pac_dr)), -max(-1, min(1, pac_dc)))
        last_trend = 0
        if self._last_manhattan is not None:
            if cur_md > self._last_manhattan: last_trend = 1
            elif cur_md < self._last_manhattan: last_trend = -1
        pac_key = tuple(p for p, _, _ in pac_predictions[:4])
        def score(move):
            nxt = _apply(ghost, move)
            n_md = _manhattan(nxt, pac); delta_md = n_md - cur_md
            wd = min(self._dist(nxt, p) for p, _, _ in pac_predictions)
            we = min(self._capture_eta(p, nxt) for p, _, _ in pac_predictions)
            s = self._cell_safety_score(nxt, pac_predictions)
            s += self._anti_velocity_score(ghost, nxt, last_ghost, pac_predictions)
            s += wd * 900.0 + min(we, 8) * 1400.0
            s += abs(delta_md) * 1800.0
            if delta_md > 0: s += 1000.0
            if last_trend and delta_md and (delta_md > 0) != (last_trend > 0): s += 1500.0
            if preferred_opposite is not None and move.value == preferred_opposite: s += 2100.0
            if self._junction_dist.get(nxt, INF_DIST) <= 1: s += 2300.0
            elif self._junction_dist.get(nxt, INF_DIST) <= 3: s += 850.0
            if nxt in self._core: s += 1600.0
            if nxt in self._loop_set: s += 1100.0
            if nxt in self._tp.deb: s -= 8500.0 + self._danger_depth.get(nxt, 1) * 800.0
            if move == Move.STAY:
                if self._allow_stay(ghost, pac, step_number) and cur_md >= max(self._h, self._w) * 0.5: s += 1250.0
                else: s -= 9000.0
                if we <= 3: s -= 12000.0
            if last_ghost is not None and nxt == last_ghost and self._junction_dist.get(ghost, INF_DIST) > 1: s -= 1600.0
            if not self._move_safely_changes_distance(ghost, pac, move) and move != Move.STAY: s -= 900.0
            s += self._safe_area(nxt, pac_key) * 95.0
            return s
        return max(moves, key=score)

    def _survival_oracle_move(self, ghost, pac, pac_predictions, last_ghost, step_number, t0):
        horizon = 9 if step_number <= 8 else 7
        memo = {}
        def leaf_score(g, preds):
            wd = min(self._dist(g, p) for p, _, _ in preds)
            we = min(self._capture_eta(p, g) for p, _, _ in preds)
            return self._cell_safety_score(g, preds) + wd * 1500.0 + min(we, 8) * 2200.0
        def rec(g, p, lg, preds, depth):
            if time.time() - t0 > 0.95 * 0.55: return leaf_score(g, preds)
            if depth >= horizon: return leaf_score(g, preds)
            key = (g, p, lg, depth)
            if key in memo: return memo[key]
            best = float("-inf")
            for move in self._phase_moves(g, p, step_number + depth):
                ng = _apply(g, move)
                imm = leaf_score(ng, preds) + self._anti_velocity_score(g, ng, lg, preds)
                if move == Move.STAY and not self._allow_stay(g, p, step_number + depth): continue
                if lg is not None and ng == lg and depth < 4: imm -= 2200.0
                worst_child = float("inf")
                for pp, weight, _ in preds[:3]:
                    if _manhattan(ng, pp) < CAPTURE_DISTANCE:
                        branch = -120000.0 + depth * 2000.0 - weight * 500.0
                    else:
                        next_preds = self._ranked_pac_predictions(pp, ng, g)
                        branch = rec(ng, pp, g, next_preds, depth + 1) - weight * 180.0
                    worst_child = min(worst_child, branch)
                score = imm * 0.42 + worst_child * 0.58
                if score > best: best = score
            memo[key] = best
            return best
        best_move = None; best_score = float("-inf")
        for move in self._phase_moves(ghost, pac, step_number):
            ng = _apply(ghost, move)
            if move == Move.STAY and not self._allow_stay(ghost, pac, step_number): continue
            if last_ghost is not None and ng == last_ghost: continue
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
            if ng in self._core: score += 1800.0
            if ng in self._tp.deb: score -= 9000.0
            if score > best_score: best_score, best_move = score, move
        return best_move

    def _panic_move(self, ghost, pac, pac_predictions, last_ghost):
        moves = _legal(ghost, self._static_map)
        if not moves: return Move.STAY
        pac_dist = self._bfs_dist(pac)
        return max(moves, key=lambda m: (
            self._cell_safety_score(_apply(ghost, m), pac_predictions)
            + self._anti_velocity_score(ghost, _apply(ghost, m), last_ghost, pac_predictions),
            pac_dist.get(_apply(ghost, m), INF_DIST),
            _apply(ghost, m) in self._core,
            _apply(ghost, m) in self._loop_set,
            -int(_apply(ghost, m) in self._tp.deb),
            self._degree.get(_apply(ghost, m), 0),
        ))

    def _search(self, ghost, pac_predictions, last_ghost, depth, max_depth, alpha, t0, pac=None, step_number=0):
        if time.time() - t0 > 0.95 * 0.88: return None, None
        if depth >= max_depth:
            return self._cell_safety_score(ghost, pac_predictions), None
        best_score = float("-inf"); best_move = None
        ordered_moves = self._move_order(ghost, pac_predictions, last_ghost, pac, step_number)
        if not ordered_moves: return -100000.0, None
        for move in ordered_moves:
            ng = _apply(ghost, move)
            imm = self._cell_safety_score(ng, pac_predictions) + self._anti_velocity_score(ghost, ng, last_ghost, pac_predictions)
            if imm < -60000:
                score = imm + depth * 500.0
            else:
                worst_child = float("inf")
                for pp, weight, _ in pac_predictions:
                    next_pacs = self._ranked_pac_predictions(pp, ng, ghost)
                    child, _ = self._search(ng, next_pacs, ghost, depth + 1, max_depth, alpha, t0, pac, step_number + depth + 1)
                    if child is None: return None, None
                    branch = child - 140.0 * weight
                    worst_child = min(worst_child, branch)
                    if worst_child <= alpha: break
                score = 0.58 * imm + 0.42 * worst_child
            if score > best_score: best_score, best_move = score, move
            alpha = max(alpha, best_score)
        return best_score, best_move

    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.time()
        self._ensure_static(map_state)
        me = tuple(my_position)
        last_ghost = self._ghost_hist[-1] if self._ghost_hist else None
        legal_moves = _legal(me, self._static_map)
        if not legal_moves: return Move.STAY
        if enemy_position is None:
            self._ghost_hist.append(me)
            return max(legal_moves, key=lambda m: self._degree.get(_apply(me, m), 0))
        pac = tuple(enemy_position)
        if self._last_pp is not None:
            observed_action = (pac[0] - self._last_pp[0], pac[1] - self._last_pp[1])
            self._last_pac_action = observed_action
            obs_speed = abs(observed_action[0]) + abs(observed_action[1])
            if obs_speed > self._pacman_speed:
                self._pacman_speed = min(4, obs_speed)
                self._capture_eta.cache_clear()
                self._safe_area.cache_clear()
            prev_ghost = self._ghost_hist[-1] if self._ghost_hist else me
            prev2 = self._ghost_hist[-2] if len(self._ghost_hist) >= 2 else None
            st = self._usl.get_state(self._static_map, prev_ghost, self._last_pp, prev2)
            self._usl.observe(st, observed_action, self._static_map, self._last_pp)
        self._last_pp = pac
        self._pac_hist.append(pac)
        if self._initial_distance is None:
            self._initial_distance = _manhattan(me, pac)
            self._initial_far = self._initial_distance >= max(self._h, self._w) * 0.5
        pac_predictions = self._ranked_pac_predictions(pac, me, last_ghost)
        min_now = min(self._dist(me, p) for p, _, _ in pac_predictions)
        if min_now <= 1:
            move = self._panic_move(me, pac, pac_predictions, last_ghost)
            self._ghost_hist.append(me); self._last_manhattan = _manhattan(me, pac)
            return move
        if step_number <= 20:
            if self._initial_far:
                move = self._far_reading_move(me, pac, pac_predictions, last_ghost, step_number)
            else:
                move = self._survival_oracle_move(me, pac, pac_predictions, last_ghost, step_number, t0)
                if move is None:
                    move = self._opening_spread_move(me, pac, pac_predictions, last_ghost, step_number)
            self._ghost_hist.append(me); self._last_manhattan = _manhattan(me, pac)
            return move
        if step_number <= 30:
            move = self._loop_lure_move(me, pac, pac_predictions, last_ghost, step_number, False)
            self._ghost_hist.append(me); self._last_manhattan = _manhattan(me, pac)
            return move
        if self._loop_dist.get(me, INF_DIST) <= 3 or self._initial_far:
            move = self._loop_lure_move(me, pac, pac_predictions, last_ghost, step_number, True)
            move_pos = _apply(me, move)
            move_eta = min(self._capture_eta(p, move_pos) for p, _, _ in pac_predictions)
            if move_eta >= 2:
                self._ghost_hist.append(me); self._last_manhattan = _manhattan(me, pac)
                return move
        if self._initial_far and step_number > 20:
            move = self._counter_loop_move(me, pac, pac_predictions, last_ghost, step_number)
            self._ghost_hist.append(me); self._last_manhattan = _manhattan(me, pac)
            return move
        phase_move = self._counter_loop_move(me, pac, pac_predictions, last_ghost, step_number)
        phase_pos = _apply(me, phase_move)
        phase_eta = min(self._capture_eta(p, phase_pos) for p, _, _ in pac_predictions)
        best_move = self._move_order(me, pac_predictions, last_ghost, pac, step_number)[0]
        best_score = float("-inf")
        for depth in range(2, 18):
            if time.time() - t0 > 0.80: break
            score, move = self._search(me, pac_predictions, last_ghost, 0, depth, float("-inf"), t0, pac, step_number)
            if score is None: break
            if move is not None:
                best_score, best_move = score, move
            if best_score < -70000: break
        best_pos = _apply(me, best_move)
        best_eta = min(self._capture_eta(p, best_pos) for p, _, _ in pac_predictions)
        if phase_eta >= best_eta and step_number > 20:
            best_move = phase_move
        self._ghost_hist.append(me)
        self._last_manhattan = _manhattan(me, pac)
        return best_move


# ---------------------------------------------------------------------------
# PacmanAgent (ported from 24127561)
# ---------------------------------------------------------------------------

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._last_enemy = None
        self._enemy_direction = None
        self._direction_streak = 0
        self._cached_target = None
        self._cached_path = []
        self._cached_my_pos = None

    def _update_ghost_tracking(self, enemy_pos):
        if self._last_enemy is None: return
        dr = enemy_pos[0] - self._last_enemy[0]
        dc = enemy_pos[1] - self._last_enemy[1]
        new_dir = (dr, dc)
        if new_dir == self._enemy_direction and (dr != 0 or dc != 0):
            self._direction_streak += 1
        else:
            self._enemy_direction = new_dir
            self._direction_streak = 1 if (dr != 0 or dc != 0) else 0

    def _compute_interception_target(self, ms, enemy_pos, my_pos):
        dr, dc = self._enemy_direction
        h, w = _shape(ms)
        cur_row, cur_col = enemy_pos
        best = None
        for i in range(1, 5):
            nr, nc = cur_row + dr * i, cur_col + dc * i
            if not (0 <= nr < h and 0 <= nc < w): break
            if _cell(ms, nr, nc) != 0: break
            nxt = (nr, nc)
            exits = sum(1 for move in MOVE_ORDER if _valid(_apply(nxt, move), ms))
            if exits >= 3: return nxt
            if exits == 2 and best is None: best = nxt
        return best

    def _astar(self, ms, start, goal):
        return astar_path(ms, start, goal)

    def _path_to_move(self, path, my_position):
        first = path[0]
        dr = first[0] - my_position[0]; dc = first[1] - my_position[1]
        if dr == -1: move = Move.UP
        elif dr == 1: move = Move.DOWN
        elif dc == -1: move = Move.LEFT
        elif dc == 1: move = Move.RIGHT
        else: return (Move.STAY, 1)
        steps = 1; cur = first
        for nxt in path[1:]:
            ndr = nxt[0] - cur[0]; ndc = nxt[1] - cur[1]
            if ndr == dr and ndc == dc and steps < self.pacman_speed:
                steps += 1; cur = nxt
            else: break
        return (move, steps)

    def _explore(self, me, ms):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        h, w = _shape(ms)
        for move in moves:
            dr, dc = move.value
            steps = 0; r, c = me
            for _ in range(self.pacman_speed):
                nr = r + dr; nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w): break
                if _cell(ms, nr, nc) != 0: break
                steps += 1; r, c = nr, nc
            if steps > 0: return (move, steps)
        return (Move.STAY, 1)

    def step(self, map_state, my_position, enemy_position, step_number):
        target = None
        if enemy_position is not None:
            self._update_ghost_tracking(enemy_position)
            if self._direction_streak >= 2:
                inter_target = self._compute_interception_target(map_state, enemy_position, my_position)
                if inter_target is not None:
                    path_to_inter = self._astar(map_state, my_position, inter_target)
                    path_to_direct = self._astar(map_state, my_position, enemy_position)
                    dist_ig = abs(inter_target[0] - enemy_position[0]) + abs(inter_target[1] - enemy_position[1])
                    if path_to_inter and (not path_to_direct or len(path_to_inter) <= len(path_to_direct) or dist_ig <= 2):
                        target = inter_target
            if target is None: target = enemy_position
            self._last_enemy = enemy_position
        else:
            if self._last_enemy is None: return self._explore(my_position, map_state)
            if my_position == self._last_enemy:
                self._last_enemy = None
                return self._explore(my_position, map_state)
            target = self._last_enemy
        if my_position == target: return (Move.STAY, 1)
        path = None
        cache_valid = self._cached_target == target and self._cached_my_pos == my_position and self._cached_path
        if cache_valid:
            path = self._cached_path
        else:
            path = self._astar(map_state, my_position, target)
            self._cached_target = target; self._cached_path = path; self._cached_my_pos = my_position
        if not path: return self._explore(my_position, map_state)
        result = self._path_to_move(path, my_position)
        if isinstance(result, tuple): consumed = result[1]; mv = result[0]
        else: consumed = 1; mv = result
        self._cached_path = path[consumed:]
        exp_pos = my_position
        for _ in range(consumed):
            nxt = (exp_pos[0] + mv.value[0], exp_pos[1] + mv.value[1])
            h, w = _shape(map_state)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and _cell(map_state, nxt[0], nxt[1]) == 0:
                exp_pos = nxt
            else: break
        self._cached_my_pos = exp_pos
        return result


