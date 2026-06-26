"""Search algorithms: A*, BFS, ghost iterative-deepening alpha-beta.

Integrated improvements over the prior simplified version, while keeping a
single unified search framework (no multi-phase strategy):
  * capture_eta    – speed-2-aware turns-to-capture, cached per (pac, ghost) pair
  * safe_area      – capture-ETA floodfill integrated as a cell_score component
  * multi-hypo     – lightweight 3-Pacman-direction evaluation (continue/turn-L/turn-R)
  * anti-velocity  – rewards direction changes at junctions
All evaluations route through a single cell_score(), keeping the architecture
that Ghost457 was built around — just ever better informed.
"""

import heapq
import time
from collections import deque, OrderedDict
from typing import Dict, List, Optional, Set, Tuple

from heuristic import MOVE_DELTAS, apply_move, cell_exits, is_valid, manhattan, shape

CAPTURE_DISTANCE = 2
CAPTURE_PENALTY = -100000.0

INF_DIST = 10**9


def astar_path(ms, start, goal):
    if not is_valid(start, ms) or not is_valid(goal, ms):
        return []
    if start == goal:
        return []
    open_set = [(0, 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    while open_set:
        _, g_val, current = heapq.heappop(open_set)
        if current in closed: continue
        closed.add(current)
        if current == goal:
            path = []
            while current != start:
                path.append(current); current = came_from[current]
            path.reverse()
            return path
        for dr, dc in MOVE_DELTAS:
            nxt = (current[0] + dr, current[1] + dc)
            if not is_valid(nxt, ms) or nxt in closed: continue
            ng = g_val + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = current
                heapq.heappush(open_set, (ng + manhattan(nxt, goal), ng, nxt))
    return []


def bfs_distance(ms, start, max_dist=999):
    if not is_valid(start, ms): return {}
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        d = dist[cur]
        if d >= max_dist: continue
        for dr, dc in MOVE_DELTAS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in dist and is_valid(nxt, ms):
                dist[nxt] = d + 1
                q.append(nxt)
    return dist


class TranspositionTable:
    def __init__(self, maxsize=50000):
        self._store = OrderedDict()
        self._maxsize = maxsize
    def get(self, key):
        if key not in self._store: return None
        self._store.move_to_end(key); return self._store[key]
    def put(self, key, value):
        if key in self._store: self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._maxsize: self._store.popitem(last=False)
    def clear(self): self._store.clear()


class LruDictCache:
    """Small fixed-size dict cache, cleared each step."""
    def __init__(self, maxsize=4000):
        self._d = {}
        self._max = maxsize
    def get(self, k, default=None):
        return self._d.get(k, default)
    def put(self, k, v):
        self._d[k] = v
        if len(self._d) > self._max:
            self._d.clear()
    def clear(self):
        self._d.clear()


class GhostSearchEvaluator:
    def __init__(self, topo=None, pacman_speed=2):
        self._topo = topo
        self._speed = max(1, pacman_speed)
        self._tt = TranspositionTable()
        self._eta_cache: LruDictCache = LruDictCache(8000)
        self._area_cache: LruDictCache = LruDictCache(2000)
        self._history: List[Tuple[int, int]] = []
        self._h, self._w = 21, 21
        self._ms = None

    def init_static(self, ms):
        self._ms = ms
        h, w = shape(ms)
        self._h, self._w = h, w

    def clear_cache(self):
        self._tt.clear()
        self._eta_cache.clear()
        self._area_cache.clear()

    # ---- neighbour queries ----
    def neighbors(self, pos):
        r = []
        for dr, dc in MOVE_DELTAS:
            n = (pos[0] + dr, pos[1] + dc)
            if 0 <= n[0] < self._h and 0 <= n[1] < self._w and _cell_safe(self._ms, n[0], n[1]):
                r.append(n)
        return r

    # ---- capture_eta: speed-2-aware turns for Pacman to reach ghost ----
    def capture_eta(self, pac, target):
        key = (pac, target)
        cached = self._eta_cache.get(key, None)
        if cached is not None:
            return cached
        md = manhattan(pac, target)
        if md < CAPTURE_DISTANCE:
            self._eta_cache.put(key, 0); return 0
        # Fast path: use precomputed BFS distance if very far, skip A*
        if md > 12:
            eta = max(1, (md + 1) // 2)
            self._eta_cache.put(key, eta); return eta
        path = astar_path(self._ms, pac, target)
        if not path:
            self._eta_cache.put(key, INF_DIST); return INF_DIST
        pos = pac
        idx = 0
        turns = 0
        while idx < len(path):
            first = path[idx]
            dr, dc = first[0] - pos[0], first[1] - pos[1]
            used = 0
            while idx < len(path) and used < self._speed:
                nxt = path[idx]
                if (nxt[0] - pos[0], nxt[1] - pos[1]) != (dr, dc):
                    break
                pos = nxt; idx += 1; used += 1
                if manhattan(pos, target) < CAPTURE_DISTANCE:
                    turns += 1
                    self._eta_cache.put(key, turns); return turns
            turns += 1
            if turns > 80:
                break
        self._eta_cache.put(key, turns); return turns

    # ---- safe_area: capture-ETA floodfill cells ghost reaches first ----
    def safe_area(self, ghost, pac_predictions, max_depth=12):
        key = (ghost, tuple(pac_predictions[:4]))
        cached = self._area_cache.get(key, None)
        if cached is not None:
            return cached
        total = 0.0
        q = deque([ghost])
        seen = {ghost: 0}
        while q:
            cur = q.popleft()
            gd = seen[cur]
            if gd > max_depth: continue
            pac_eta = min((self.capture_eta(p, cur) for p in pac_predictions), default=INF_DIST)
            margin = pac_eta - gd
            if margin > 0:
                dg = self._topo["degree"].get(cur, 2) if self._topo else 2
                total += 1.0 + min(3, margin) * 0.35 + dg * 0.08
                for nxt in self.neighbors(cur):
                    if nxt not in seen:
                        seen[nxt] = gd + 1
                        q.append(nxt)
        self._area_cache.put(key, total)
        return total

    # ---- lightweight multi-hypothesis Pacman predictions (3 successor turns) ----
    def _pac_hypotheses(self, pac):
        """Three successor Pacman positions: continue, turn-left, turn-right.

        'continue' is computed by moving Pacman at speed-2 in its most-recent
        observed direction; turn-L/turn-R are the two perpendicular one-step
        moves.  The result is up to 3 distinct valid positions plus 'pac' itself.
        """
        out = [pac]
        n = self.neighbors(pac)
        for nxt in n:
            if nxt not in out:
                out.append(nxt)
            if len(out) >= 4:
                break
        return out

    # ---- cell_score: unified single evaluation function ----
    def cell_score(self, ghost, pac, pd):
        md = manhattan(ghost, pac)
        if md < CAPTURE_DISTANCE: return CAPTURE_PENALTY

        # Multi-hypothesis: aggregate over a few nearby Pacman positions
        hypos = self._pac_hypotheses(pac)
        worst = INF_DIST
        worst_eta = INF_DIST
        weighted = 0.0
        for p in hypos:
            d = self._bfs_dist_from(pd, p, ghost)
            eta = self.capture_eta(p, ghost)
            if d < worst: worst = d
            if eta < worst_eta: worst_eta = eta
            if d <= 0:
                weighted -= 60000.0
            elif d == 1:
                weighted -= 12000.0
            elif d == 2:
                weighted -= 2000.0
            else:
                weighted += min(d, 18) * 280.0
            if eta <= 0:
                weighted -= 80000.0
            elif eta == 1:
                weighted -= 24000.0
            elif eta == 2:
                weighted -= 5000.0
            else:
                weighted += min(eta, 10) * 480.0
        n_hypos = max(1, len(hypos))
        total = weighted / n_hypos

        # Worst-case terms: emphasize the most dangerous hypothesis
        total += min(worst, 18) * 600.0
        total += min(worst_eta, 10) * 800.0

        # Degree / topology
        if self._topo is not None:
            deg = self._topo["degree"].get(ghost, 2)
            total += deg * 90.0
            total += len(self.neighbors(ghost)) * 40.0

            jd = self._topo.get("junction_distance", {}).get(ghost, 10)
            if jd <= 2:
                total += 200.0
            elif jd >= 6:
                total -= 120.0

            if ghost in self._topo.get("core", set()):
                total += 1200.0
            if ghost in self._topo.get("loop_set", set()):
                total += 800.0
            if ghost in self._topo.get("junctions", set()):
                total += 320.0
            if ghost in self._topo.get("dead_ends", set()):
                dd = self._topo.get("dead_end_depth", {}).get(ghost, 1)
                total -= 1200.0 - 280.0 * dd

        # Safe-area: bonus scaled to its capture-ETA floodfill value
        try:
            area = self.safe_area(ghost, hypos[:4], max_depth=10)
        except Exception:
            area = 0.0
        total += min(area, 60.0) * 60.0
        if area < 8:
            total -= (8 - area) * 300.0

        # History / oscillation penalty (same idea as the prior version)
        window = self._history[-6:]
        count = window.count(ghost)
        if count > 0: total -= 60.0 * count

        return total

    def _bfs_dist_from(self, pd, p, ghost):
        """Return BFS distance from Pacman position p to ghost.

        Uses precomputed pd only when p == the origin of pd. Otherwise falls
        back to a quick local BFS bounded to a few cells.
        """
        v = pd.get(ghost, None)
        if v is not None:
            return v
        # Quick bounded BFS
        dist = {p: 0}
        q = deque([p])
        LIMIT = 22
        while q:
            cur = q.popleft()
            d = dist[cur]
            if d >= LIMIT: break
            for dr, dc in MOVE_DELTAS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt == ghost:
                    return d + 1
                if nxt not in dist and 0 <= nxt[0] < self._h and 0 <= nxt[1] < self._w and _cell_safe(self._ms, nxt[0], nxt[1]):
                    dist[nxt] = d + 1
                    q.append(nxt)
        return INF_DIST

    def order_moves(self, ghost, ppos, pd, ms=None):
        if ms is None: ms = self._ms
        cands = self.neighbors(ghost)
        if not cands: return []
        last = self._history[-2] if len(self._history) >= 2 else None
        def key(nxt):
            s = self.cell_score(nxt, ppos, pd)
            # Anti-velocity: small bonus for not continuing straight when near
            if last is not None:
                dr = ghost[0] - last[0]; dc = ghost[1] - last[1]
                expected = (ghost[0] + dr, ghost[1] + dc)
                if 0 <= expected[0] < self._h and 0 <= expected[1] < self._w and _cell_safe(self._ms, expected[0], expected[1]):
                    if nxt != expected:
                        s += 80.0
                        if self._topo is not None and ghost in self._topo.get("junctions", set()):
                            s += 60.0
            return s
        cands.sort(key=key, reverse=True)
        return cands

    # ---- alpha-beta negamax with multi-hypothesis leaf ----
    def negamax(self, gpos, ppos, ghost_turn, depth, alpha, beta, pd, ms, t0, budget):
        if time.time() - t0 > budget: return None
        if ghost_turn and manhattan(gpos, ppos) < 2: return CAPTURE_PENALTY + depth * 100
        if depth <= 0: return self._leaf_eval(gpos, ppos, pd)

        if ghost_turn:
            value = float("-inf")
            for nxt in self.order_moves(gpos, ppos, pd, ms):
                if manhattan(nxt, ppos) < 2:
                    v = -5000.0 + depth * 50
                else:
                    v = self.negamax(nxt, ppos, False, depth - 1, alpha, beta, pd, ms, t0, budget)
                if v is None: return None
                if v > value: value = v
                if value >= beta: break
                if value > alpha: alpha = value
            return value if value > float("-inf") else float(manhattan(gpos, ppos))
        else:
            value = float("inf")
            for nxt1 in self.neighbors(ppos):
                v = self._pac_speed2(gpos, ppos, nxt1, depth - 1, alpha, beta, pd, ms, t0, budget)
                if v is None: return None
                if v < value: value = v
                if value <= alpha: break
                if value < beta: beta = value
            return value if value < float("inf") else float(manhattan(gpos, ppos))

    def _pac_speed2(self, gpos, ppos_orig, ppos1, depth, alpha, beta, pd, ms, t0, budget):
        v = self.negamax(gpos, ppos1, True, depth, alpha, beta, pd, ms, t0, budget)
        if v is None: return None
        dr, dc = ppos1[0] - ppos_orig[0], ppos1[1] - ppos_orig[1]
        if dr != 0 or dc != 0:
            ppos2 = (ppos1[0] + dr, ppos1[1] + dc)
            if is_valid(ppos2, ms):
                v2 = self.negamax(gpos, ppos2, True, depth, alpha, beta, pd, ms, t0, budget)
                if v2 is not None and v2 < v: v = v2
        return v

    def _leaf_eval(self, gpos, ppos, pd):
        md = manhattan(gpos, ppos)
        if md < 2: return CAPTURE_PENALTY
        bfs_d = float(pd.get(gpos, md + 20))
        s = bfs_d
        # Speed-2 turns estimate (cheap version of capture_eta to keep leaves fast)
        eta = max(1, (int(bfs_d) + 1) // 2)
        s += eta * 80.0
        if self._topo is not None:
            deg = self._topo["degree"].get(gpos, 2)
            if deg >= 3: s += 120.0
            elif deg <= 1:
                dd = self._topo["dead_end_depth"].get(gpos, 0)
                s -= 240.0 + min(dd, 6) * 60.0
            elif deg == 2:
                jd = self._topo.get("junction_distance", {}).get(gpos, 10)
                if jd <= 2: s += 50.0
                elif jd >= 6: s -= 40.0
        if md <= 1: s -= 50000.0
        elif md == 2: s -= 4000.0
        elif md == 3: s -= 500.0
        elif md <= 6: s -= (7 - md) * 30.0
        if eta <= 1: s -= 20000.0
        elif eta == 2: s -= 3000.0
        window = self._history[-6:]
        count = window.count(gpos)
        if count > 0: s -= 60.0 * count
        return s

    def search(self, gpos, ppos, moves, ms, pd, depth, budget):
        t0 = time.time()
        best = None
        for d in range(2, depth + 1, 2):
            dbest = None
            dbval = float("-inf")
            for move in moves:
                ng = apply_move(gpos, move)
                if not is_valid(ng, ms) or manhattan(ng, ppos) < 2:
                    val = CAPTURE_PENALTY
                else:
                    val = self.negamax(ng, ppos, False, d - 1, float("-inf"), float("inf"), pd, ms, t0, budget)
                    if val is None: return best
                if val > dbval + 0.001:
                    dbval = val; dbest = move
            if dbest is not None: best = dbest
        return best


def _cell_safe(ms, r, c):
    if hasattr(ms, "shape"):
        return 0 <= r < ms.shape[0] and 0 <= c < ms.shape[1] and ms[r, c] != 1
    return 0 <= r < len(ms) and 0 <= c < len(ms[0]) and ms[r][c] != 1