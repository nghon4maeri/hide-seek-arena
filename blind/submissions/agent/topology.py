"""topology.py — Static map topology analysis (ported from Lab 1 Top-1).

Features:
  - junctions / dead_ends / dead_end_depth / corridor_cells
  - core computation (iterative leaf-trimming)
  - loop detection (DFS back-edge, largest cycle in core)
  - junction_distance / loop_distance BFS maps
  - topological weight cache
"""

from collections import deque
from typing import Dict, Set, Tuple

import numpy as np

from pathfinding import DIRS, _shape, _cell, _valid, _apply, _cell_exits


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


class TopologyAnalyzer:
    """Full topology analysis with core, loops, dead-end depth."""

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
            deg = _cell_exits(p, ms)
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
                nxts = [_apply(cur, m) for m in DIRS
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
                na = sum(1 for m in DIRS
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
                    _apply(start, m) for m in DIRS
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
                            _apply(v, m) for m in DIRS
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
            for m in DIRS:
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
            for m in DIRS:
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
