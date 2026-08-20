"""core.py — Grid utilities, precomputed maze distances, capture-eta.

All map geometry is derived from the very first observation: wall cells are
always visible in this game, so the full static layout is known at step 1
without hardcoding.  `DistTable` precomputes all-pairs maze distances once,
which turns every evaluation function into an O(1) lookup.
"""

from collections import deque

import numpy as np

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CAPTURE_DISTANCE = 2
INF = 10 ** 9

MOVE_DELTA = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3, (0, 0): 4}


def in_bounds(r, c, h, w):
    return 0 <= r < h and 0 <= c < w


def cross_visible(p, c, radius=5):
    """True if p sees c along a straight cross-ray within radius."""
    pr, pc = p
    cr, cc = c
    if pr == cr:
        return abs(pc - cc) <= radius
    if pc == cc:
        return abs(pr - cr) <= radius
    return False


def visible_cells_of(grid, p, radius=5):
    """Cells revealed by standing at p: cross-rays that stop at walls."""
    pr, pc = p
    out = {p}
    for dr, dc in DIRS:
        for d in range(1, radius + 1):
            nr, nc = pr + dr * d, pc + dc * d
            if not in_bounds(nr, nc, grid.h, grid.w):
                break
            if grid.map[nr, nc] == 1:
                break
            out.add((nr, nc))
    return out


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Grid:
    """Static map + fast distance lookups."""

    def __init__(self, static_map):
        self.map = np.asarray(static_map, dtype=np.int8)
        self.h, self.w = self.map.shape
        self.n = self.h * self.w
        self.free = [(r, c) for r in range(self.h) for c in range(self.w)
                     if self.map[r, c] != 1]
        self.cell_id = np.full((self.h, self.w), -1, dtype=np.int32)
        self.id_cell = {}
        for i, (r, c) in enumerate(self.free):
            self.cell_id[r, c] = i
            self.id_cell[i] = (r, c)
        self.dist = self._all_pairs_bfs()
        self.degree = self._degrees()

    def _all_pairs_bfs(self):
        m = len(self.free)
        dist = np.full((m, m), INF, dtype=np.int32)
        for i in range(m):
            dist[i, i] = 0
        adj = [[] for _ in range(m)]
        for i in range(m):
            r, c = self.id_cell[i]
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc, self.h, self.w) and self.map[nr, nc] != 1:
                    adj[i].append(self.cell_id[nr, nc])
        for src in range(m):
            q = deque([src])
            d = dist[src]
            while q:
                u = q.popleft()
                nd = d[u] + 1
                for v in adj[u]:
                    if d[v] == INF:
                        d[v] = nd
                        q.append(v)
        return dist

    def _degrees(self):
        deg = np.zeros(self.n, dtype=np.int8)
        for i in range(len(self.free)):
            r, c = self.id_cell[i]
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc, self.h, self.w) and self.map[nr, nc] != 1:
                    deg[i] += 1
        return deg

    # ---- helpers ---------------------------------------------------------
    def id_of(self, pos):
        return int(self.cell_id[pos[0], pos[1]])

    def walkable(self, pos):
        r, c = pos
        return in_bounds(r, c, self.h, self.w) and self.map[r, c] != 1

    def dist_between(self, a, b):
        ia, ib = self.id_of(a), self.id_of(b)
        if ia < 0 or ib < 0:
            return INF
        return int(self.dist[ia, ib])

    def neighbors(self, pos):
        r, c = pos
        out = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, self.h, self.w) and self.map[nr, nc] != 1:
                out.append((nr, nc))
        return out

    def exits(self, pos):
        return len(self.neighbors(pos))


class PacmanModel:
    """Speed-2 movement rules of the Seeker (straight-line advance)."""

    def __init__(self, speed=2):
        self.speed = speed
        self._eta_cache = {}

    def legal_actions(self, grid, pos):
        """Return list of (delta, steps) pacman can perform from pos."""
        acts = []
        for dr, dc in DIRS:
            p1 = (pos[0] + dr, pos[1] + dc)
            if not grid.walkable(p1):
                continue
            acts.append(((dr, dc), 1))
            p2 = (p1[0] + dr, p1[1] + dc)
            if self.speed >= 2 and grid.walkable(p2):
                acts.append(((dr, dc), 2))
        return acts

    def apply(self, grid, pos, delta, steps):
        cur = pos
        for _ in range(steps):
            nxt = (cur[0] + delta[0], cur[1] + delta[1])
            if not grid.walkable(nxt):
                break
            cur = nxt
        return cur

    def reach_set(self, grid, pos):
        """All cells reachable in one turn (used by the Ghost's belief)."""
        out = {pos}
        for delta, steps in self.legal_actions(grid, pos):
            out.add(self.apply(grid, pos, delta, steps))
        return out

    def eta(self, grid, pac, target):
        """Greedy estimate of turns for Pacman to reach capture range."""
        key = (pac, target)
        cached = self._eta_cache.get(key)
        if cached is not None:
            return cached
        if grid.dist_between(pac, target) < CAPTURE_DISTANCE:
            self._eta_cache[key] = 0
            return 0
        pos = pac
        turns = 0
        limit = 200
        while turns < limit:
            d = grid.dist_between(pos, target)
            if d < CAPTURE_DISTANCE:
                break
            best, best_d = None, d
            for dr, dc in DIRS:
                nxt = (pos[0] + dr, pos[1] + dc)
                if not grid.walkable(nxt):
                    continue
                nd = grid.dist_between(nxt, target)
                if nd < best_d:
                    best_d, best = nd, (dr, dc)
            if best is None:
                turns = INF
                break
            pos = self.apply(grid, pos, best, self.speed)
            turns += 1
        self._eta_cache[key] = turns
        if len(self._eta_cache) > 100000:
            self._eta_cache.clear()
        return turns


def pack_path(path, my_pos, speed):
    """Pack leading same-direction steps of a path into (delta, steps)."""
    if not path:
        return ((0, 0), 0)
    first = path[0]
    delta = (first[0] - my_pos[0], first[1] - my_pos[1])
    steps = 1
    cur = first
    for nxt in path[1:]:
        nd = (nxt[0] - cur[0], nxt[1] - cur[1])
        if nd == delta and steps < speed:
            steps += 1
            cur = nxt
        else:
            break
    return delta, steps


def astar_path(grid, start, goal):
    """A* on the static grid returning the list of cells (excl. start)."""
    if grid.id_of(start) < 0 or grid.id_of(goal) < 0:
        return []
    if start == goal:
        return []
    import heapq
    open_set = [(0, 0, start)]
    came = {}
    g_score = {start: 0}
    closed = set()
    while open_set:
        _, gv, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            path = []
            while cur != start:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        for nxt in grid.neighbors(cur):
            if nxt in closed:
                continue
            ng = gv + 1
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_set, (ng + grid.dist_between(nxt, goal), ng, nxt))
    return []


def safe_area_count(grid, pac_model, ghost, pac, topo=None, max_depth=12):
    """Floodfill: weighted count of cells the Ghost reaches before Pacman."""
    if grid.id_of(ghost) < 0:
        return 0.0
    total = 0.0
    seen = {ghost: 0}
    q = deque([ghost])
    while q:
        cur = q.popleft()
        gd = seen[cur]
        if gd > max_depth:
            continue
        pe = pac_model.eta(grid, pac, cur)
        margin = pe - gd
        if margin > 0:
            if topo is not None:
                w = topo.weight_of(cur)
            else:
                w = 1.0
            total += (1.0 + min(3, margin) * 0.35) * w
            for nxt in grid.neighbors(cur):
                if nxt not in seen:
                    seen[nxt] = gd + 1
                    q.append(nxt)
    return total
